"""
step4_build_ml_dataset_aeroml3.py
==================================
Builds the step-4 ML dataset from AeroML3's cleaned_data_final directory.

Input:  noel_part/cleaned_data_final/{source_run}/{flight_name}/
           adsb_before.parquet, adsb_after.parquet, adsc.parquet
Output: artifacts/step4_ml_dataset/
           catalog/flight_splits.parquet   - split assignments per flight
           catalog/step4_summary.json
           catalog/step4_issues.parquet
           dataset/sequences_train.npz
           dataset/sequences_val.npz
           dataset/sequences_test.npz
           dataset/normalization_stats.json

Compatible with step5_kalman_aeroml3.py.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sequence parameters
# ---------------------------------------------------------------------------
BEFORE_STEPS     = 64
AFTER_STEPS      = 32
RESAMPLE_SECONDS = 60
MAX_ADSC_WP      = 64    # waypoints stored per flight (covers more of gap)
N_SEQ_FEATURES   = 6     # lat_norm, lon_norm, vel_norm, hdg_sin, hdg_cos, alt_norm

LAT_MEAN,  LAT_STD  = 53.0,    8.0
LON_MEAN,  LON_STD  = -30.0,  25.0
VEL_MEAN,  VEL_STD  = 240.0,  30.0
ALT_MEAN,  ALT_STD  = 10500.0, 1000.0


@dataclass(frozen=True)
class Step4Config:
    clean_root:   Path  = Path("../noel_part/cleaned_data_final")
    output_root:  Path  = Path("../artifacts/step4_ml_dataset")
    min_adsc_pts: int   = 3
    max_flights:  int | None = None
    train_frac:   float = 0.70
    val_frac:     float = 0.15
    test_frac:    float = 0.15
    seed:         int   = 42
    verbose:      bool  = True
    progress_every: int = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["timestamp"] = _to_ts(w["timestamp"])
    w["latitude"]  = pd.to_numeric(w["latitude"],  errors="coerce")
    w["longitude"] = pd.to_numeric(w["longitude"], errors="coerce")
    for c in ["velocity_mps", "heading_deg", "geoaltitude_m", "altitude"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    return (w.dropna(subset=["timestamp", "latitude", "longitude"])
             .sort_values("timestamp")
             .reset_index(drop=True))


def _safe_f(val) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _norm(v, mean, std):
    if not math.isfinite(v):
        return 0.0
    return (v - mean) / std


def _resample(df: pd.DataFrame, sec: int) -> pd.DataFrame:
    if len(df) < 2:
        return df
    df = _clean(df)
    grid = pd.date_range(
        start=df["timestamp"].min().floor(f"{sec}s"),
        end=df["timestamp"].max().ceil(f"{sec}s"),
        freq=f"{sec}s",
    )
    if len(grid) == 0:
        return df
    cols = [c for c in ["latitude", "longitude", "velocity_mps",
                         "heading_deg", "geoaltitude_m"] if c in df.columns]
    resampled = (
        df.set_index("timestamp")[cols]
        .sort_index()
        .reindex(grid)
        .interpolate(method="time", limit_direction="forward", limit_area="inside")
    )
    return _clean(resampled.reset_index().rename(columns={"index": "timestamp"}))


def _track_to_array(df: pd.DataFrame, n_steps: int, from_end: bool):
    """Convert track to (n_steps, N_SEQ_FEATURES) array. Returns (array, mask)."""
    out  = np.zeros((n_steps, N_SEQ_FEATURES), dtype=np.float32)
    mask = np.zeros(n_steps, dtype=np.float32)
    if df.empty:
        return out, mask
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.tail(n_steps) if from_end else df.head(n_steps)
    df = df.reset_index(drop=True)
    offset = (n_steps - len(df)) if from_end else 0
    for i, row in df.iterrows():
        dest = offset + i if from_end else i
        if dest >= n_steps:
            break
        lat = _safe_f(row.get("latitude"))
        lon = _safe_f(row.get("longitude"))
        vel = _safe_f(row.get("velocity_mps"))
        hdg = _safe_f(row.get("heading_deg"))
        # use geoaltitude_m if available, else altitude
        alt = _safe_f(row.get("geoaltitude_m") if "geoaltitude_m" in row.index
                      else row.get("altitude"))
        out[dest, 0] = _norm(lat, LAT_MEAN, LAT_STD)
        out[dest, 1] = _norm(lon, LON_MEAN, LON_STD)
        out[dest, 2] = _norm(vel, VEL_MEAN, VEL_STD)
        out[dest, 3] = math.sin(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest, 4] = math.cos(math.radians(hdg)) if math.isfinite(hdg) else 0.0
        out[dest, 5] = _norm(alt, ALT_MEAN, ALT_STD)
        mask[dest] = 1.0
    return out, mask


def _safe_dt(ts, t0) -> float:
    try:
        return float((pd.Timestamp(ts) - pd.Timestamp(t0)).total_seconds())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Split by ICAO24 (no data leakage)
# ---------------------------------------------------------------------------

def _assign_splits(catalog: pd.DataFrame, train_frac, val_frac, test_frac, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unique_ac = sorted(catalog["icao24"].dropna().astype(str).unique())
    unique_ac = list(rng.permutation(unique_ac))
    n = len(unique_ac)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val   = min(n_val, n - n_train)
    ac_split = {}
    for i, ac in enumerate(unique_ac):
        if i < n_train:
            ac_split[ac] = "train"
        elif i < n_train + n_val:
            ac_split[ac] = "val"
        else:
            ac_split[ac] = "test"
    out = catalog.copy()
    out["split"] = out["icao24"].astype(str).map(ac_split).fillna("train")
    return out


# ---------------------------------------------------------------------------
# Build one sequence sample
# ---------------------------------------------------------------------------

def _build_sample(row: pd.Series, bef: pd.DataFrame, adsc: pd.DataFrame, aft: pd.DataFrame) -> dict:
    b_anc = bef.sort_values("timestamp").iloc[-1]
    a_anc = aft.sort_values("timestamp").iloc[0]
    t0 = pd.Timestamp(b_anc["timestamp"])
    t1 = pd.Timestamp(a_anc["timestamp"])
    dur = _safe_dt(t1, t0)
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError(f"bad gap duration: {dur:.1f}s")

    bef_r = _resample(bef, RESAMPLE_SECONDS)
    aft_r = _resample(aft, RESAMPLE_SECONDS)
    bef_seq, bef_mask = _track_to_array(bef_r, BEFORE_STEPS, from_end=True)
    aft_seq, aft_mask = _track_to_array(aft_r, AFTER_STEPS,  from_end=False)

    # ADS-C — sample evenly across the gap (up to MAX_ADSC_WP)
    adsc_s = adsc.sort_values("timestamp").reset_index(drop=True)
    # keep only points inside the gap
    adsc_in = adsc_s[
        (adsc_s["timestamp"] >= t0) & (adsc_s["timestamp"] <= t1)
    ].reset_index(drop=True)

    if len(adsc_in) == 0:
        # try relaxed window (ADS-C might start just after t0)
        adsc_in = adsc_s.head(MAX_ADSC_WP).reset_index(drop=True)

    if len(adsc_in) > MAX_ADSC_WP:
        # evenly sample MAX_ADSC_WP points
        idx = np.linspace(0, len(adsc_in) - 1, MAX_ADSC_WP, dtype=int)
        adsc_in = adsc_in.iloc[idx].reset_index(drop=True)

    n_wp = len(adsc_in)
    adsc_targets = np.zeros((MAX_ADSC_WP, 2), dtype=np.float32)
    adsc_tau     = np.zeros(MAX_ADSC_WP,      dtype=np.float32)
    adsc_mask    = np.zeros(MAX_ADSC_WP,      dtype=np.float32)

    for i in range(n_wp):
        wp = adsc_in.iloc[i]
        lat = _safe_f(wp["latitude"])
        lon = _safe_f(wp["longitude"])
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue
        adsc_targets[i, 0] = np.float32(lat)
        adsc_targets[i, 1] = np.float32(lon)
        elapsed = _safe_dt(wp["timestamp"], t0)
        if math.isfinite(elapsed):
            adsc_tau[i] = float(np.clip(elapsed / dur, 0, 1))
        adsc_mask[i] = 1.0

    if adsc_mask.sum() == 0:
        raise ValueError("no valid ADS-C waypoints inside gap")

    return {
        "segment_id":        str(row["segment_id"]),
        "icao24":            str(row["icao24"]),
        "split":             str(row["split"]),
        "before_seq":        bef_seq,
        "before_mask":       bef_mask,
        "after_seq":         aft_seq,
        "after_mask":        aft_mask,
        "adsc_targets":      adsc_targets,
        "adsc_tau":          adsc_tau,
        "adsc_mask":         adsc_mask,
        "gap_dur_sec":       np.float32(dur),
        "n_adsc_wp":         np.int32(n_wp),
        "before_anchor_lat": np.float32(b_anc["latitude"]),
        "before_anchor_lon": np.float32(b_anc["longitude"]),
        "after_anchor_lat":  np.float32(a_anc["latitude"]),
        "after_anchor_lon":  np.float32(a_anc["longitude"]),
    }


def _save_npz(samples: list[dict], path: Path) -> None:
    if not samples:
        return
    np.savez_compressed(
        path,
        segment_ids       = np.array([s["segment_id"] for s in samples], dtype=object),
        icao24            = np.array([s["icao24"]      for s in samples], dtype=object),
        before_seq        = np.stack([s["before_seq"]  for s in samples]),
        before_mask       = np.stack([s["before_mask"] for s in samples]),
        after_seq         = np.stack([s["after_seq"]   for s in samples]),
        after_mask        = np.stack([s["after_mask"]  for s in samples]),
        adsc_targets      = np.stack([s["adsc_targets"]for s in samples]),
        adsc_tau          = np.stack([s["adsc_tau"]    for s in samples]),
        adsc_mask         = np.stack([s["adsc_mask"]   for s in samples]),
        gap_dur_sec       = np.array([s["gap_dur_sec"] for s in samples], dtype=np.float32),
        n_adsc_wp         = np.array([s["n_adsc_wp"]   for s in samples], dtype=np.int32),
        before_anchor_lat = np.array([s["before_anchor_lat"] for s in samples], dtype=np.float32),
        before_anchor_lon = np.array([s["before_anchor_lon"] for s in samples], dtype=np.float32),
        after_anchor_lat  = np.array([s["after_anchor_lat"]  for s in samples], dtype=np.float32),
        after_anchor_lon  = np.array([s["after_anchor_lon"]  for s in samples], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_step4(cfg: Step4Config) -> dict[str, Any]:
    log = (lambda m: print(m, flush=True)) if cfg.verbose else (lambda m: None)

    # 1. Scan flights
    log(f"Scanning {cfg.clean_root} ...")
    rows = []
    for src_dir in sorted(cfg.clean_root.iterdir()):
        if not src_dir.is_dir():
            continue
        for flt_dir in sorted(src_dir.iterdir()):
            if not flt_dir.is_dir():
                continue
            if not all((flt_dir / f).exists() for f in
                       ["adsb_before.parquet", "adsb_after.parquet", "adsc.parquet"]):
                continue
            parts = flt_dir.name.split("_")
            icao24 = parts[1] if len(parts) >= 2 else "unknown"
            rows.append({
                "segment_id":  f"{src_dir.name}/{flt_dir.name}",
                "source_run":  src_dir.name,
                "flight_name": flt_dir.name,
                "icao24":      icao24,
            })

    catalog = pd.DataFrame(rows)
    log(f"Found {len(catalog)} flights across {catalog['source_run'].nunique()} runs")

    if cfg.max_flights is not None:
        catalog = catalog.head(int(cfg.max_flights)).reset_index(drop=True)

    # 2. Split by ICAO24
    catalog = _assign_splits(catalog, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.seed)
    cnts = catalog["split"].value_counts()
    log(f"Split: train={cnts.get('train',0)}  val={cnts.get('val',0)}  test={cnts.get('test',0)}")

    # 3. Setup output
    if cfg.output_root.exists():
        shutil.rmtree(cfg.output_root)
    cat_root = cfg.output_root / "catalog"
    dat_root = cfg.output_root / "dataset"
    cat_root.mkdir(parents=True, exist_ok=True)
    dat_root.mkdir(parents=True, exist_ok=True)

    # 4. Process each flight
    seq_samples: dict[str, list] = {"train": [], "val": [], "test": []}
    issues = []
    total = len(catalog)

    for i, row in catalog.iterrows():
        if cfg.verbose and (i == 0 or (i + 1) % cfg.progress_every == 0 or i + 1 == total):
            log(f"  [{i+1}/{total}] Processing {row['segment_id']}")

        flt_dir = cfg.clean_root / row["source_run"] / row["flight_name"]
        try:
            bef  = _clean(pd.read_parquet(flt_dir / "adsb_before.parquet"))
            aft  = _clean(pd.read_parquet(flt_dir / "adsb_after.parquet"))
            adsc = _clean(pd.read_parquet(flt_dir / "adsc.parquet"))

            if len(adsc) < cfg.min_adsc_pts:
                issues.append({"segment_id": row["segment_id"],
                                "issue_type": "too_few_adsc",
                                "issue_detail": f"{len(adsc)} < {cfg.min_adsc_pts}"})
                continue

            sample = _build_sample(row, bef, adsc, aft)
            seq_samples[row["split"]].append(sample)

        except Exception as exc:
            issues.append({"segment_id": row["segment_id"],
                           "issue_type": "build_failure",
                           "issue_detail": str(exc)})

    # 5. Save NPZ files
    for split in ["train", "val", "test"]:
        n = len(seq_samples[split])
        if n > 0:
            path = dat_root / f"sequences_{split}.npz"
            log(f"Saving {split}: {n} flights -> {path.name}")
            _save_npz(seq_samples[split], path)

    # 6. Save catalog + stats
    catalog.to_parquet(cat_root / "flight_splits.parquet", index=False)
    catalog.to_csv(cat_root / "flight_splits.csv", index=False)

    pd.DataFrame(issues if issues else
                 [{"segment_id": "", "issue_type": "", "issue_detail": ""}]
                 ).to_parquet(cat_root / "step4_issues.parquet", index=False)

    norm_stats = {
        "lat":  {"mean": LAT_MEAN,  "std": LAT_STD},
        "lon":  {"mean": LON_MEAN,  "std": LON_STD},
        "vel":  {"mean": VEL_MEAN,  "std": VEL_STD},
        "alt":  {"mean": ALT_MEAN,  "std": ALT_STD},
        "before_steps":   BEFORE_STEPS,
        "after_steps":    AFTER_STEPS,
        "max_adsc_wp":    MAX_ADSC_WP,
        "n_seq_features": N_SEQ_FEATURES,
        "resample_sec":   RESAMPLE_SECONDS,
    }
    (dat_root / "normalization_stats.json").write_text(json.dumps(norm_stats, indent=2))

    summary = {
        "total_flights_found":  int(len(catalog)),
        "flights_failed":       int(len(issues)),
        "split_method":         "by_icao24",
        "split_flights": {s: int((catalog["split"] == s).sum())
                          for s in ["train", "val", "test"]},
        "sequence_flights": {s: len(seq_samples[s]) for s in ["train", "val", "test"]},
        "max_adsc_wp": MAX_ADSC_WP,
        "before_steps": BEFORE_STEPS,
        "after_steps": AFTER_STEPS,
    }
    (cat_root / "step4_summary.json").write_text(json.dumps(summary, indent=2))
    log("Step 4 done.")
    log(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clean-root",   default="../noel_part/cleaned_data_final")
    p.add_argument("--output-root",  default="../artifacts/step4_ml_dataset")
    p.add_argument("--max-flights",  type=int, default=None)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--quiet",        action="store_true")
    args = p.parse_args()
    run_step4(Step4Config(
        clean_root=Path(args.clean_root),
        output_root=Path(args.output_root),
        max_flights=args.max_flights,
        seed=args.seed,
        verbose=not args.quiet,
    ))
