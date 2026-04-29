"""
step5_train_gru_v2.py - GRU Training Script v2 (Colab-optimised)
=================================================================
Architecture vs v1:
  - Both encoders bidirectional  (after_enc was unidirectional in v1)
  - Attention pooling over all GRU output steps instead of last hidden state
  - H=128 (same as v1) -- prevents overfitting on 1540 samples

Training improvements vs v1:
  - Smoothness loss: penalises path curvature -> smoother gap trajectories
  - Noise injection during training: Gaussian noise on before/after sequences
  - Linear warmup (5 epochs) + CosineAnnealingLR instead of ReduceLROnPlateau
  - BATCH_SIZE 32->64, EPOCHS 60->80, EARLY_STOP 15->20
  - num_workers=2 + pin_memory

Colab setup:
  1. Upload to MyDrive/aero_project/step4/:
       sequences_train.npz, sequences_val.npz, sequences_test.npz,
       normalization_stats.json
     Upload this file to MyDrive/aero_project/

  2. In Colab:
       from google.colab import drive; drive.mount('/content/drive')
       %cd /content/drive/MyDrive/aero_project
       !python step5_train_gru_v2.py

  3. Results in MyDrive/aero_project/step5_v2/:
       best_model_v2.pt, training_history.json, test_summary.json

  4. Update notebooks:
       from step5_train_gru_v2 import TrajectoryGRU
       MODEL_PATH = Path(r'..../step5_v2/best_model_v2.pt')
"""

from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_DIR   = DRIVE_ROOT / "step4" / "dataset"
OUTPUT_DIR = DRIVE_ROOT / "step5_v2"

# Local run:
# DATA_DIR   = Path("../artifacts/step4_ml_dataset/dataset")
# OUTPUT_DIR = Path("../artifacts/step5_gru_v2")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZE   = 128    # same as v1 -- prevents overfitting on 1540 samples
NUM_LAYERS    = 2
DROPOUT       = 0.25
BATCH_SIZE    = 64
EPOCHS        = 80
EARLY_STOP    = 20
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
SEED          = 42
WARMUP_EPOCHS = 5

# Smoothness penalty: mean 2nd-difference magnitude (metres).
# Discourages noisy/wiggly paths. ~0.5-3% of haversine loss in practice.
SMOOTH_WEIGHT = 0.5

# Noise injection std in normalised feature space (train only).
NOISE_STD     = 0.02

BEFORE_STEPS   = 64
AFTER_STEPS    = 32
N_SEQ_FEATURES = 6
EARTH_R        = 6_371_000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_m_np(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_loss(pred_lat, pred_lon, true_lat, true_lon, mask):
    lat_m = (pred_lat + true_lat) / 2.0
    dlat  = (pred_lat - true_lat) * (math.pi / 180.0) * EARTH_R
    dlon  = (pred_lon - true_lon) * (math.pi / 180.0) * EARTH_R \
            * torch.cos(lat_m * math.pi / 180.0)
    dist  = torch.sqrt(dlat**2 + dlon**2 + 1e-8)
    return (dist * mask).sum() / mask.sum().clamp(min=1.0)


def smoothness_loss(pred_lat, pred_lon, mask):
    """Mean 2nd-difference magnitude (metres) -- penalises path curvature."""
    if pred_lat.shape[1] < 3:
        return pred_lat.sum() * 0.0
    cos_lat  = torch.cos(pred_lat.detach().mean() * (math.pi / 180.0))
    lat_m    = pred_lat * 111_111.0
    lon_m    = pred_lon * 111_111.0 * cos_lat
    d2_lat   = lat_m[:, 2:] - 2 * lat_m[:, 1:-1] + lat_m[:, :-2]
    d2_lon   = lon_m[:, 2:] - 2 * lon_m[:, 1:-1] + lon_m[:, :-2]
    mag      = torch.sqrt(d2_lat**2 + d2_lon**2 + 1e-4)
    mid_mask = mask[:, 1:-1]
    return (mag * mid_mask).sum() / mid_mask.sum().clamp(min=1.0)


def gc_interpolate_batch(lat0, lon0, lat1, lon1, tau):
    lat0r = np.radians(lat0[:,None]); lon0r = np.radians(lon0[:,None])
    lat1r = np.radians(lat1[:,None]); lon1r = np.radians(lon1[:,None])
    x0=np.cos(lat0r)*np.cos(lon0r); y0=np.cos(lat0r)*np.sin(lon0r); z0=np.sin(lat0r)
    x1=np.cos(lat1r)*np.cos(lon1r); y1=np.cos(lat1r)*np.sin(lon1r); z1=np.sin(lat1r)
    dot   = np.clip(x0*x1+y0*y1+z0*z1, -1, 1)
    omega = np.arccos(dot)
    sin_o = np.sin(omega)
    safe  = (sin_o > 1e-10).astype(float)
    sos   = np.where(sin_o > 1e-10, sin_o, 1.0)
    w0 = np.sin((1-tau)*omega)/sos*safe + (1-safe)
    w1 = np.sin(tau*omega)/sos*safe
    xp,yp,zp = w0*x0+w1*x1, w0*y0+w1*y1, w0*z0+w1*z1
    n  = np.sqrt(xp**2+yp**2+zp**2).clip(min=1e-10)
    xp,yp,zp = xp/n, yp/n, zp/n
    return np.degrees(np.arcsin(np.clip(zp,-1,1))), np.degrees(np.arctan2(yp,xp))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.before_seq   = d["before_seq"].astype(np.float32)
        self.before_mask  = d["before_mask"].astype(np.float32)
        self.after_seq    = d["after_seq"].astype(np.float32)
        self.after_mask   = d["after_mask"].astype(np.float32)
        self.adsc_targets = d["adsc_targets"].astype(np.float32)
        self.adsc_tau     = d["adsc_tau"].astype(np.float32)
        self.adsc_mask    = d["adsc_mask"].astype(np.float32)
        self.gap_dur_sec  = d["gap_dur_sec"].astype(np.float32)
        bla = d["before_anchor_lat"].astype(np.float32)
        blo = d["before_anchor_lon"].astype(np.float32)
        ala = d["after_anchor_lat"].astype(np.float32)
        alo = d["after_anchor_lon"].astype(np.float32)
        bl_lat, bl_lon = gc_interpolate_batch(bla, blo, ala, alo, self.adsc_tau)
        self.baseline_lat = bl_lat.astype(np.float32)
        self.baseline_lon = bl_lon.astype(np.float32)
        self.gap_norm = (self.gap_dur_sec / 6000.0).astype(np.float32)
        print(f"  {path.name}: {len(self)} samples")

    def __len__(self): return len(self.before_seq)

    def __getitem__(self, i):
        return dict(
            before_seq   = self.before_seq[i],
            before_mask  = self.before_mask[i],
            after_seq    = self.after_seq[i],
            after_mask   = self.after_mask[i],
            adsc_tau     = self.adsc_tau[i],
            adsc_mask    = self.adsc_mask[i],
            true_lat     = self.adsc_targets[i,:,0],
            true_lon     = self.adsc_targets[i,:,1],
            baseline_lat = self.baseline_lat[i],
            baseline_lon = self.baseline_lon[i],
            gap_norm     = self.gap_norm[i],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TrajectoryGRU(nn.Module):
    """
    v2: both encoders bidirectional + attention pooling, H=128.
    H=128 (same as v1) avoids overfitting on the 1540-sample dataset.
    Attention lets each decoder query focus on the relevant part of the
    before/after track rather than relying on a single compressed state.
    """
    def __init__(self, D=N_SEQ_FEATURES, H=HIDDEN_SIZE, L=NUM_LAYERS, p=DROPOUT):
        super().__init__()
        self.before_enc  = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                   dropout=p if L > 1 else 0.0)
        self.after_enc   = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                   dropout=p if L > 1 else 0.0)
        self.before_attn = nn.Linear(2*H, 1, bias=False)
        self.after_attn  = nn.Linear(2*H, 1, bias=False)
        C = 2*H + 2*H + 1    # 513 with H=128
        self.decoder = nn.Sequential(
            nn.Linear(C + 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(p),
            nn.Linear(256,   128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(p),
            nn.Linear(128,    64), nn.GELU(),
            nn.Linear(64,      2),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param)
            elif "bias" in name:      nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def _encode(self, enc, seq, mask):
        lengths       = mask.sum(1).long().clamp(min=1).cpu()
        packed        = nn.utils.rnn.pack_padded_sequence(seq, lengths,
                          batch_first=True, enforce_sorted=False)
        out_packed, _ = enc(packed)
        out, _        = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return out

    def _attend(self, attn_layer, out, mask):
        T       = out.size(1)
        scores  = attn_layer(out).squeeze(-1)
        m       = mask[:, :T]
        scores  = scores.masked_fill(m == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return (weights.unsqueeze(-1) * out).sum(1)

    def forward(self, b):
        bef_out = self._encode(self.before_enc, b["before_seq"], b["before_mask"])
        aft_out = self._encode(self.after_enc,  b["after_seq"],  b["after_mask"])
        bef_ctx = self._attend(self.before_attn, bef_out, b["before_mask"])
        aft_ctx = self._attend(self.after_attn,  aft_out, b["after_mask"])
        ctx    = torch.cat([bef_ctx, aft_ctx, b["gap_norm"].unsqueeze(-1)], dim=-1)
        B, K   = b["adsc_tau"].shape
        ctx_e  = ctx.unsqueeze(1).expand(-1, K, -1)
        dec_in = torch.cat([ctx_e,
                             b["adsc_tau"].unsqueeze(-1),
                             b["baseline_lat"].unsqueeze(-1),
                             b["baseline_lon"].unsqueeze(-1)], dim=-1)
        res = self.decoder(dec_in)
        return b["baseline_lat"] + res[:,:,0], b["baseline_lon"] + res[:,:,1]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def to_dev(batch, dev):
    return {k: v.to(dev) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def run_epoch(model, loader, optimizer, dev, train: bool):
    model.train(train)
    tot_loss = tot_err = n = 0
    with torch.set_grad_enabled(train):
        for b in loader:
            b = to_dev(b, dev)
            if train and NOISE_STD > 0:
                b["before_seq"] = b["before_seq"] + \
                    torch.randn_like(b["before_seq"]) * NOISE_STD * b["before_mask"].unsqueeze(-1)
                b["after_seq"]  = b["after_seq"]  + \
                    torch.randn_like(b["after_seq"])  * NOISE_STD * b["after_mask"].unsqueeze(-1)
            pl, po  = model(b)
            hloss   = haversine_loss(pl, po, b["true_lat"], b["true_lon"], b["adsc_mask"])
            sloss   = smoothness_loss(pl, po, b["adsc_mask"])
            loss    = hloss + SMOOTH_WEIGHT * sloss
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            with torch.no_grad():
                errs = haversine_m_np(
                    pl.detach().cpu().numpy(), po.detach().cpu().numpy(),
                    b["true_lat"].cpu().numpy(), b["true_lon"].cpu().numpy())
                vm = b["adsc_mask"].cpu().numpy() > 0
                if vm.sum() > 0:
                    tot_err += errs[vm].mean()
            tot_loss += hloss.item()
            n += 1
    return tot_loss / max(n, 1), tot_err / max(n, 1)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading data...")
    tr = TrajectoryDataset(DATA_DIR / "sequences_train.npz")
    va = TrajectoryDataset(DATA_DIR / "sequences_val.npz")
    te = TrajectoryDataset(DATA_DIR / "sequences_test.npz")

    pin = dev.type == "cuda"
    trl = DataLoader(tr, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)
    val = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
    tel = DataLoader(te, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)

    model    = TrajectoryGRU().to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Smooth weight   : {SMOOTH_WEIGHT}  |  Noise std: {NOISE_STD}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup    = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    print(f"\n{'Epoch':>6}  {'TrLoss':>10}  {'TrErr':>10}  {'VaLoss':>10}  {'VaErr':>10}  {'LR':>10}")
    print("-" * 68)

    history  = []
    best_val = float("inf"); best_ep = 0; no_imp = 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, te_ = run_epoch(model, trl, optimizer, dev, train=True)
        vl, ve  = run_epoch(model, val, optimizer, dev, train=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{ep:>6}  {tl:>10.1f}  {te_/1000:>9.2f}km  "
              f"{vl:>10.1f}  {ve/1000:>9.2f}km  {lr_now:>10.2e}  ({time.time()-t0:.0f}s)")
        history.append({"epoch": ep, "train_loss": tl, "train_err_m": te_,
                         "val_loss": vl, "val_err_m": ve, "lr": lr_now})
        if vl < best_val:
            best_val = vl; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_v2.pt")
            print(f"         saved best (epoch {ep})")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                print(f"Early stop at epoch {ep}, best epoch={best_ep}"); break

    json.dump([{k: float(v) if hasattr(v, "item") else v for k, v in h.items()}
               for h in history],
              open(OUTPUT_DIR / "training_history.json", "w"), indent=2)

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model_v2.pt", map_location=dev))
    model.eval()
    PLs, PLo, TLs, TLo, BLs, BLo, Ms = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in tel:
            bd = to_dev(b, dev)
            pl, po = model(bd)
            PLs.append(pl.cpu().numpy());  PLo.append(po.cpu().numpy())
            TLs.append(b["true_lat"].numpy()); TLo.append(b["true_lon"].numpy())
            BLs.append(b["baseline_lat"].numpy()); BLo.append(b["baseline_lon"].numpy())
            Ms.append(b["adsc_mask"].numpy())
    PL,PO = np.concatenate(PLs), np.concatenate(PLo)
    TL,TO = np.concatenate(TLs), np.concatenate(TLo)
    BL,BO = np.concatenate(BLs), np.concatenate(BLo)
    M     = np.concatenate(Ms)
    ge = haversine_m_np(PL,PO,TL,TO)
    be = haversine_m_np(BL,BO,TL,TO)
    gru_pf = np.array([ge[i][M[i]>0].mean() for i in range(len(PL)) if (M[i]>0).sum()>0])
    bl_pf  = np.array([be[i][M[i]>0].mean() for i in range(len(BL)) if (M[i]>0).sum()>0])
    imp_mean   = (1 - gru_pf.mean()     / bl_pf.mean())     * 100
    imp_median = (1 - np.median(gru_pf) / np.median(bl_pf)) * 100
    print("\n" + "="*55)
    print("TEST RESULTS")
    print("="*55)
    print(f"  GRU  -- mean: {gru_pf.mean()/1e3:.2f}km  median: {np.median(gru_pf)/1e3:.2f}km")
    print(f"  Base -- mean: {bl_pf.mean()/1e3:.2f}km  median: {np.median(bl_pf)/1e3:.2f}km")
    print(f"  Improvement: {imp_mean:.1f}% (mean)  {imp_median:.1f}% (median)")
    print("="*55)
    np.savez_compressed(OUTPUT_DIR / "test_predictions.npz",
        pred_lat=PL, pred_lon=PO, true_lat=TL, true_lon=TO,
        baseline_lat=BL, baseline_lon=BO, mask=M,
        gru_errors_m=ge, baseline_errors_m=be)
    summary = {
        "best_epoch": best_ep, "best_val_loss": float(best_val),
        "test_flights": len(gru_pf),
        "gru_mean_error_km":        float(gru_pf.mean()/1e3),
        "gru_median_error_km":      float(np.median(gru_pf)/1e3),
        "gru_p90_error_km":         float(np.percentile(gru_pf,90)/1e3),
        "baseline_mean_error_km":   float(bl_pf.mean()/1e3),
        "baseline_median_error_km": float(np.median(bl_pf)/1e3),
        "improvement_mean_pct":     float(imp_mean),
        "improvement_median_pct":   float(imp_median),
        "model_params":             n_params,
        "hyperparameters": dict(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            dropout=DROPOUT, batch_size=BATCH_SIZE, lr=LR,
            smooth_weight=SMOOTH_WEIGHT, noise_std=NOISE_STD,
            bidirectional_after=True, attention_pooling=True),
    }
    json.dump(summary, open(OUTPUT_DIR / "test_summary.json", "w"), indent=2)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()