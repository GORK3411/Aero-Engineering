"""
step5_train_gru_v3.py - GRU Training Script v3
===============================================
Key improvements over v2:
  1. Before-sequence flip: fixes packing bug (most-recent ADS-B was silently dropped)
  2. CrossAttentionDecoder: each tau-step independently attends to full encoder outputs
  3. Hard endpoint anchoring: output = baseline + tanh(raw)*SCALE * sin(pi*tau)
     Guarantees output == baseline (== anchor) at tau=0 and tau=1
  4. Coordinate delta scaling: tanh(raw) * RESIDUAL_SCALE bounds max deviation
  5. Enhanced smoothness: 2nd-order curvature + 3rd-order jerk penalty
  6. Feature clipping: removes normalised vel/alt outlier spikes before encoding
  7. Trajectory reversal augmentation: doubles effective training samples
  8. Correct evaluation: per-flight mean/median restricted to active adsc_mask positions

Architecture (~2M parameters):
  BiGRU (H=128, L=2, bidir) x2 -> d_enc=256
  CrossAttentionDecoder: 4 heads, FF=512, two cross-attention layers (bef + aft)
  Output: baseline_lat/lon + tanh(res)*RESIDUAL_SCALE * sin(pi*tau)

Target: mean test error < 80 km (v2 baseline was 121.7 km, Kalman 88.0 km)

Colab setup:
  1. Upload to MyDrive/step4/dataset/:
       sequences_train.npz, sequences_val.npz, sequences_test.npz
  2. Upload this file to MyDrive/
  3. In Colab:
       from google.colab import drive; drive.mount('/content/drive')
       %cd /content/drive/MyDrive
       !python step5_train_gru_v3.py
  4. Results in MyDrive/step5_v3/:
       best_model_v3.pt, training_history.json, test_summary.json
"""

from __future__ import annotations
import json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths  (Colab)
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATA_DIR   = DRIVE_ROOT / "step4" / "dataset"
OUTPUT_DIR = DRIVE_ROOT / "step5_v3"

# Local override:
# DATA_DIR   = Path("../artifacts/step4_ml_dataset/dataset")
# OUTPUT_DIR = Path("../artifacts/step5_gru_v3")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZE    = 128
NUM_LAYERS     = 2
DROPOUT        = 0.2
BATCH_SIZE     = 64
EPOCHS         = 120
EARLY_STOP     = 30
LR             = 8e-4
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
SEED           = 42
WARMUP_EPOCHS  = 5

CA_HEADS       = 4      # cross-attention heads
CA_FF_DIM      = 512    # cross-attention feed-forward dimension

SMOOTH_WEIGHT  = 0.15   # weight on 2nd-order curvature loss
JERK_WEIGHT    = 0.05   # weight on 3rd-order jerk loss
NOISE_STD      = 0.01   # Gaussian noise on normalised features (train only)
AUG_PROB       = 0.5    # probability of trajectory reversal per sample

# Max residual: tanh output scaled to this many degrees from baseline.
# At tau=0.5 the model can deviate at most RESIDUAL_SCALE deg from gc-baseline.
RESIDUAL_SCALE = 4.0

BEFORE_STEPS   = 64
AFTER_STEPS    = 32
N_SEQ_FEATURES = 6
EARTH_R        = 6_371_000.0

# Clipping bounds for each normalised feature (removes velocity/alt outliers).
# Order matches step4 dataset: [dlat_norm, dlon_norm, dalt_norm, sin_brg, cos_brg, vel_norm]
FEAT_CLIP_LO = [-11.0, -6.0, -8.0, -1.01, -1.01, -11.0]
FEAT_CLIP_HI = [  5.0,  9.0,  5.0,  1.01,  1.01,   5.0]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_m_np(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
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


def smoothness_jerk_loss(pred_lat, pred_lon, mask):
    """2nd-order curvature + 3rd-order jerk penalty (metres).

    Curvature penalises wiggly paths; jerk penalises sharp direction changes
    at the gap entry/exit, producing tangent-continuous transitions.
    """
    if pred_lat.shape[1] < 3:
        return pred_lat.new_zeros(1).squeeze()

    cos_lat = torch.cos(pred_lat.detach().mean() * (math.pi / 180.0))
    lat_m   = pred_lat * 111_111.0
    lon_m   = pred_lon * 111_111.0 * cos_lat

    # 2nd order: d^2 position / dt^2
    d2_lat = lat_m[:, 2:] - 2 * lat_m[:, 1:-1] + lat_m[:, :-2]
    d2_lon = lon_m[:, 2:] - 2 * lon_m[:, 1:-1] + lon_m[:, :-2]
    curv   = torch.sqrt(d2_lat**2 + d2_lon**2 + 1e-4)
    m2     = mask[:, 1:-1]
    smooth = (curv * m2).sum() / m2.sum().clamp(min=1.0)

    if pred_lat.shape[1] < 4:
        return smooth

    # 3rd order: d^3 position / dt^3 (jerk)
    d3_lat = lat_m[:, 3:] - 3*lat_m[:, 2:-1] + 3*lat_m[:, 1:-2] - lat_m[:, :-3]
    d3_lon = lon_m[:, 3:] - 3*lon_m[:, 2:-1] + 3*lon_m[:, 1:-2] - lon_m[:, :-3]
    jerk   = torch.sqrt(d3_lat**2 + d3_lon**2 + 1e-4)
    m3     = mask[:, 1:-2]
    jerk_l = (jerk * m3).sum() / m3.sum().clamp(min=1.0)

    return smooth + (JERK_WEIGHT / max(SMOOTH_WEIGHT, 1e-9)) * jerk_l


def gc_interpolate_batch(lat0, lon0, lat1, lon1, tau):
    """Vectorised great-circle interpolation. tau in [0,1], shape [B] or [B,K]."""
    lat0r = np.radians(lat0[:, None]); lon0r = np.radians(lon0[:, None])
    lat1r = np.radians(lat1[:, None]); lon1r = np.radians(lon1[:, None])
    x0 = np.cos(lat0r)*np.cos(lon0r); y0 = np.cos(lat0r)*np.sin(lon0r); z0 = np.sin(lat0r)
    x1 = np.cos(lat1r)*np.cos(lon1r); y1 = np.cos(lat1r)*np.sin(lon1r); z1 = np.sin(lat1r)
    dot   = np.clip(x0*x1 + y0*y1 + z0*z1, -1, 1)
    omega = np.arccos(dot)
    sin_o = np.sin(omega)
    safe  = (sin_o > 1e-10).astype(float)
    sos   = np.where(sin_o > 1e-10, sin_o, 1.0)
    w0 = np.sin((1-tau)*omega)/sos*safe + (1-safe)*(1-tau)
    w1 = np.sin(tau*omega)/sos*safe     + (1-safe)*tau
    xp, yp, zp = w0*x0+w1*x1, w0*y0+w1*y1, w0*z0+w1*z1
    n = np.sqrt(xp**2+yp**2+zp**2).clip(min=1e-10)
    xp, yp, zp = xp/n, yp/n, zp/n
    return (np.degrees(np.arcsin(np.clip(zp, -1, 1))),
            np.degrees(np.arctan2(yp, xp)))


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
        self.bef_anc_lat  = d["before_anchor_lat"].astype(np.float32)
        self.bef_anc_lon  = d["before_anchor_lon"].astype(np.float32)
        self.aft_anc_lat  = d["after_anchor_lat"].astype(np.float32)
        self.aft_anc_lon  = d["after_anchor_lon"].astype(np.float32)
        bl_lat, bl_lon = gc_interpolate_batch(
            self.bef_anc_lat, self.bef_anc_lon,
            self.aft_anc_lat, self.aft_anc_lon,
            self.adsc_tau)
        self.baseline_lat = bl_lat.astype(np.float32)
        self.baseline_lon = bl_lon.astype(np.float32)
        self.gap_norm = (self.gap_dur_sec / 6000.0).astype(np.float32)
        print(f"  {path.name}: {len(self)} samples")

    def __len__(self):
        return len(self.before_seq)

    def __getitem__(self, i):
        return dict(
            before_seq   = self.before_seq[i],
            before_mask  = self.before_mask[i],
            after_seq    = self.after_seq[i],
            after_mask   = self.after_mask[i],
            adsc_tau     = self.adsc_tau[i],
            adsc_mask    = self.adsc_mask[i],
            true_lat     = self.adsc_targets[i, :, 0],
            true_lon     = self.adsc_targets[i, :, 1],
            baseline_lat = self.baseline_lat[i],
            baseline_lon = self.baseline_lon[i],
            gap_norm     = self.gap_norm[i],
            bef_anc_lat  = self.bef_anc_lat[i],
            bef_anc_lon  = self.bef_anc_lon[i],
            aft_anc_lat  = self.aft_anc_lat[i],
            aft_anc_lon  = self.aft_anc_lon[i],
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrossAttentionDecoder(nn.Module):
    """
    For each output tau position, independently query the full before and after
    encoder sequences via multi-head cross-attention.  This gives each position
    its own context rather than sharing a global pooled vector.
    """
    def __init__(self, d_enc, n_heads=CA_HEADS, ff_dim=CA_FF_DIM, dropout=DROPOUT):
        super().__init__()
        # Query projection: [tau, bl_lat/90, bl_lon/180] -> d_enc
        self.q_proj = nn.Sequential(
            nn.Linear(3, d_enc), nn.LayerNorm(d_enc), nn.GELU()
        )
        # Global context conditioning: [gap_norm, bef_lat/90, bef_lon/180, aft_lat/90, aft_lon/180]
        self.ctx_proj = nn.Sequential(
            nn.Linear(5, d_enc), nn.LayerNorm(d_enc), nn.GELU()
        )
        self.ca_bef = nn.MultiheadAttention(d_enc, n_heads, dropout=dropout, batch_first=True)
        self.ca_aft = nn.MultiheadAttention(d_enc, n_heads, dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(d_enc)
        self.norm2  = nn.LayerNorm(d_enc)
        self.norm3  = nn.LayerNorm(d_enc)
        self.ff     = nn.Sequential(
            nn.Linear(d_enc, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_enc),
        )
        self.out_proj = nn.Linear(d_enc, 2)

    def _safe_kpm(self, mask, T):
        """Key padding mask for MultiheadAttention: True = ignore."""
        kpm = ~mask[:, :T].bool()
        if kpm.all():
            kpm = torch.zeros_like(kpm)
        return kpm

    def forward(self, tau_feats, bef_out, bef_mask_f, aft_out, aft_mask, ctx_vec):
        """
        tau_feats : [B, K, 3]   tau, bl_lat/90, bl_lon/180
        bef_out   : [B, T_b, d_enc]
        bef_mask_f: [B, T_b]    1=valid (flipped before)
        aft_out   : [B, T_a, d_enc]
        aft_mask  : [B, T_a]    1=valid
        ctx_vec   : [B, d_enc]
        """
        q = self.q_proj(tau_feats) + ctx_vec.unsqueeze(1)  # [B, K, d_enc]

        T_b = bef_out.size(1)
        q1, _ = self.ca_bef(q, bef_out, bef_out,
                             key_padding_mask=self._safe_kpm(bef_mask_f, T_b))
        q = self.norm1(q + q1)

        T_a = aft_out.size(1)
        q2, _ = self.ca_aft(q, aft_out, aft_out,
                             key_padding_mask=self._safe_kpm(aft_mask, T_a))
        q = self.norm2(q + q2)
        q = self.norm3(q + self.ff(q))
        return self.out_proj(q)   # [B, K, 2]


class TrajectoryGRU(nn.Module):
    """
    v3 architecture summary:
      - BiGRU encoders for before (flipped) and after tracks
      - CrossAttentionDecoder for per-position queries
      - output = baseline + tanh(decoder_raw) * RESIDUAL_SCALE * sin(pi*tau)
        -> zero residual guaranteed at tau=0 and tau=1
    """
    def __init__(self, D=N_SEQ_FEATURES, H=HIDDEN_SIZE, L=NUM_LAYERS, p=DROPOUT):
        super().__init__()
        d_enc = 2 * H  # 256 with H=128
        self.before_enc = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                  dropout=p if L > 1 else 0.0)
        self.after_enc  = nn.GRU(D, H, L, batch_first=True, bidirectional=True,
                                  dropout=p if L > 1 else 0.0)
        self.ctx_proj   = nn.Sequential(
            nn.Linear(5, d_enc), nn.LayerNorm(d_enc), nn.GELU()
        )
        self.decoder    = CrossAttentionDecoder(d_enc)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(param)
            elif "weight_hh" in name: nn.init.orthogonal_(param)
            elif "bias" in name:      nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def _clip_feats(self, seq):
        lo = seq.new_tensor(FEAT_CLIP_LO)
        hi = seq.new_tensor(FEAT_CLIP_HI)
        return seq.clamp(min=lo, max=hi)

    def _encode(self, enc, seq, mask):
        lengths  = mask.sum(1).long().clamp(min=1).cpu()
        packed   = nn.utils.rnn.pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False)
        out_p, _ = enc(packed)
        out, _   = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        return out

    def _encode_before(self, seq, mask):
        # step4 layout: [zeros(offset), data(n_valid)]
        # Flip so data is first -> pack_padded_sequence captures most-recent ADS-B
        seq_f  = torch.flip(seq,  dims=[1])
        mask_f = torch.flip(mask, dims=[1])
        out    = self._encode(self.before_enc, seq_f, mask_f)
        return out, mask_f

    def forward(self, b):
        bef_seq = self._clip_feats(b["before_seq"])
        aft_seq = self._clip_feats(b["after_seq"])

        bef_out, mask_f = self._encode_before(bef_seq, b["before_mask"])
        aft_out         = self._encode(self.after_enc, aft_seq, b["after_mask"])

        ctx_in  = torch.stack([
            b["gap_norm"],
            b["bef_anc_lat"] / 90.0,
            b["bef_anc_lon"] / 180.0,
            b["aft_anc_lat"] / 90.0,
            b["aft_anc_lon"] / 180.0,
        ], dim=-1)
        ctx_vec = self.ctx_proj(ctx_in)  # [B, d_enc]

        tau_feats = torch.stack([
            b["adsc_tau"],
            b["baseline_lat"] / 90.0,
            b["baseline_lon"] / 180.0,
        ], dim=-1)  # [B, K, 3]

        raw = self.decoder(tau_feats, bef_out, mask_f, aft_out, b["after_mask"], ctx_vec)
        # [B, K, 2]

        # Coordinate scaling: tanh bounds the raw deviation in degree-space
        res = torch.tanh(raw) * RESIDUAL_SCALE  # [B, K, 2]

        # Hard endpoint anchoring: residual -> 0 as tau -> 0 or 1
        pin = torch.sin(math.pi * b["adsc_tau"]).unsqueeze(-1)  # [B, K, 1]
        res = res * pin

        pred_lat = b["baseline_lat"] + res[:, :, 0]
        pred_lon = b["baseline_lon"] + res[:, :, 1]
        return pred_lat, pred_lon


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def reverse_batch(b, aug_prob=AUG_PROB):
    """Flip a random subset of batch samples: swap before/after, reverse tau."""
    B = b["adsc_tau"].size(0)
    flip_mask = torch.rand(B) < aug_prob
    for i in range(B):
        if not flip_mask[i]:
            continue

        old_bef      = b["before_seq"][i].clone()
        old_bef_mask = b["before_mask"][i].clone()
        old_aft      = b["after_seq"][i].clone()
        old_aft_mask = b["after_mask"][i].clone()

        # New before = time-reversed after, right-aligned (step4 before layout)
        new_bef      = torch.zeros_like(b["before_seq"][i])
        new_bef_mask = torch.zeros_like(b["before_mask"][i])
        n_aft        = int(old_aft_mask.sum().item())
        n_aft        = min(n_aft, BEFORE_STEPS)
        aft_data     = old_aft[:n_aft].flip(0)
        new_bef[BEFORE_STEPS - n_aft:] = aft_data
        new_bef_mask[BEFORE_STEPS - n_aft:] = 1.0

        # New after = time-reversed before (most-recent n steps, left-aligned)
        new_aft      = torch.zeros_like(b["after_seq"][i])
        new_aft_mask = torch.zeros_like(b["after_mask"][i])
        n_bef        = int(old_bef_mask.sum().item())
        n_take       = min(n_bef, AFTER_STEPS)
        bef_data     = old_bef[BEFORE_STEPS - n_bef: BEFORE_STEPS - n_bef + n_take].flip(0)
        new_aft[:n_take]      = bef_data
        new_aft_mask[:n_take] = 1.0

        b["before_seq"][i]   = new_bef
        b["before_mask"][i]  = new_bef_mask
        b["after_seq"][i]    = new_aft
        b["after_mask"][i]   = new_aft_mask

        b["adsc_tau"][i]     = 1.0 - b["adsc_tau"][i]
        b["true_lat"][i]     = b["true_lat"][i].flip(0)
        b["true_lon"][i]     = b["true_lon"][i].flip(0)
        b["adsc_mask"][i]    = b["adsc_mask"][i].flip(0)
        b["baseline_lat"][i] = b["baseline_lat"][i].flip(0)
        b["baseline_lon"][i] = b["baseline_lon"][i].flip(0)

        bla = b["bef_anc_lat"][i].clone(); blo = b["bef_anc_lon"][i].clone()
        b["bef_anc_lat"][i] = b["aft_anc_lat"][i]
        b["bef_anc_lon"][i] = b["aft_anc_lon"][i]
        b["aft_anc_lat"][i] = bla
        b["aft_anc_lon"][i] = blo

    return b


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
            if train:
                if NOISE_STD > 0:
                    bm = b["before_mask"].unsqueeze(-1)
                    am = b["after_mask"].unsqueeze(-1)
                    b["before_seq"] = b["before_seq"] + torch.randn_like(b["before_seq"]) * NOISE_STD * bm
                    b["after_seq"]  = b["after_seq"]  + torch.randn_like(b["after_seq"])  * NOISE_STD * am
                b = reverse_batch(b)

            pl, po = model(b)
            hloss  = haversine_loss(pl, po, b["true_lat"], b["true_lon"], b["adsc_mask"])
            sloss  = smoothness_jerk_loss(pl, po, b["adsc_mask"])
            loss   = hloss + SMOOTH_WEIGHT * sloss

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\nLoading datasets...")
    tr = TrajectoryDataset(DATA_DIR / "sequences_train.npz")
    va = TrajectoryDataset(DATA_DIR / "sequences_val.npz")
    te = TrajectoryDataset(DATA_DIR / "sequences_test.npz")

    pin = dev.type == "cuda"
    trl = DataLoader(tr, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)
    val = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
    tel = DataLoader(te, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)

    model    = TrajectoryGRU().to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters  : {n_params:,}")
    print(f"RESIDUAL_SCALE    : {RESIDUAL_SCALE} deg")
    print(f"SMOOTH_WEIGHT     : {SMOOTH_WEIGHT}  |  JERK_WEIGHT: {JERK_WEIGHT}")
    print(f"AUG_PROB          : {AUG_PROB}  |  NOISE_STD: {NOISE_STD}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup    = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    print(f"\n{'Epoch':>6}  {'TrLoss':>9}  {'TrErr':>9}  "
          f"{'VaLoss':>9}  {'VaErr':>9}  {'LR':>9}")
    print("-" * 65)

    history  = []
    best_val = float("inf"); best_ep = 0; no_imp = 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, te_ = run_epoch(model, trl, optimizer, dev, train=True)
        vl, ve  = run_epoch(model, val, optimizer, dev, train=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{ep:>6}  {tl:>9.1f}  {te_/1000:>8.2f}km  "
              f"{vl:>9.1f}  {ve/1000:>8.2f}km  {lr_now:>9.2e}  ({time.time()-t0:.0f}s)")
        history.append({"epoch": ep,
                         "train_loss": tl,   "train_err_m": float(te_),
                         "val_loss":   vl,   "val_err_m":   float(ve),
                         "lr":         lr_now})
        if vl < best_val:
            best_val = vl; best_ep = ep; no_imp = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model_v3.pt")
            print(f"         saved best (epoch {ep})")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP:
                print(f"Early stop at epoch {ep}, best epoch={best_ep}")
                break

    json.dump(
        [{k: float(v) if hasattr(v, "item") else v for k, v in h.items()} for h in history],
        open(OUTPUT_DIR / "training_history.json", "w"), indent=2)

    # ------------------------------------------------------------------
    # Test evaluation: per-flight mean/median restricted to adsc_mask > 0
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model_v3.pt", map_location=dev))
    model.eval()

    PLs, PLo, TLs, TLo, BLs, BLo, Ms = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in tel:
            bd = to_dev(b, dev)
            pl, po = model(bd)
            PLs.append(pl.cpu().numpy()); PLo.append(po.cpu().numpy())
            TLs.append(b["true_lat"].numpy()); TLo.append(b["true_lon"].numpy())
            BLs.append(b["baseline_lat"].numpy()); BLo.append(b["baseline_lon"].numpy())
            Ms.append(b["adsc_mask"].numpy())

    PL = np.concatenate(PLs); PO = np.concatenate(PLo)
    TL = np.concatenate(TLs); TO = np.concatenate(TLo)
    BL = np.concatenate(BLs); BO = np.concatenate(BLo)
    M  = np.concatenate(Ms)

    # Haversine error at every tau position
    ge = haversine_m_np(PL, PO, TL, TO)   # [N, K]
    be = haversine_m_np(BL, BO, TL, TO)   # [N, K]

    # Per-flight mean restricted to active ADS-C waypoints (adsc_mask > 0)
    gru_pf = []
    bl_pf  = []
    for i in range(len(PL)):
        vm = M[i] > 0
        if vm.sum() > 0:
            gru_pf.append(float(ge[i][vm].mean()))
            bl_pf.append(float(be[i][vm].mean()))
    gru_pf = np.array(gru_pf)
    bl_pf  = np.array(bl_pf)

    imp_mean   = (1 - gru_pf.mean()     / bl_pf.mean())     * 100
    imp_median = (1 - np.median(gru_pf) / np.median(bl_pf)) * 100

    print("\n" + "="*65)
    print("TEST RESULTS")
    print("="*65)
    print(f"  Flights evaluated : {len(gru_pf)}")
    print(f"  GRU  -- mean: {gru_pf.mean()/1e3:.2f} km  "
          f"median: {np.median(gru_pf)/1e3:.2f} km  "
          f"p90: {np.percentile(gru_pf, 90)/1e3:.2f} km")
    print(f"  Base -- mean: {bl_pf.mean()/1e3:.2f} km  "
          f"median: {np.median(bl_pf)/1e3:.2f} km")
    print(f"  Improvement: {imp_mean:.1f}% (mean)   {imp_median:.1f}% (median)")
    print("="*65)

    np.savez_compressed(OUTPUT_DIR / "test_predictions.npz",
        pred_lat=PL, pred_lon=PO, true_lat=TL, true_lon=TO,
        baseline_lat=BL, baseline_lon=BO, mask=M,
        gru_errors_m=ge, baseline_errors_m=be)

    summary = {
        "best_epoch":               best_ep,
        "best_val_loss":            float(best_val),
        "test_flights":             len(gru_pf),
        "gru_mean_error_km":        float(gru_pf.mean()/1e3),
        "gru_median_error_km":      float(np.median(gru_pf)/1e3),
        "gru_p90_error_km":         float(np.percentile(gru_pf, 90)/1e3),
        "baseline_mean_error_km":   float(bl_pf.mean()/1e3),
        "baseline_median_error_km": float(np.median(bl_pf)/1e3),
        "improvement_mean_pct":     float(imp_mean),
        "improvement_median_pct":   float(imp_median),
        "model_params":             n_params,
        "hyperparameters": dict(
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
            batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY,
            smooth_weight=SMOOTH_WEIGHT, jerk_weight=JERK_WEIGHT,
            noise_std=NOISE_STD, aug_prob=AUG_PROB,
            residual_scale=RESIDUAL_SCALE,
            ca_heads=CA_HEADS, ca_ff_dim=CA_FF_DIM,
        ),
    }
    json.dump(summary, open(OUTPUT_DIR / "test_summary.json", "w"), indent=2)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
