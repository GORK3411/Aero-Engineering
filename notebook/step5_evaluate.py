"""
step5_evaluate.py — Evaluate all reconstruction methods against ADS-C ground truth
====================================================================================
Compares baseline, Kalman and BiGRU on held-out ADS-C waypoints.
Used by 05_evaluate.ipynb.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Geod
from scipy.interpolate import interp1d
from tqdm import tqdm

geod = Geod(ellps="WGS84")


def _ts_to_float(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.apply(lambda x: x.timestamp()).values
    return pd.to_numeric(series, errors="coerce").values.astype(float)


def _interp_at(recon_df: pd.DataFrame,
               lat_col: str, lon_col: str,
               query_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts = _ts_to_float(recon_df["timestamp"])
    la = recon_df[lat_col].values.astype(float)
    lo = recon_df[lon_col].values.astype(float)
    finite = np.isfinite(la) & np.isfinite(lo)
    if finite.sum() < 2:
        return np.full(len(query_ts), np.nan), np.full(len(query_ts), np.nan)
    f_la = interp1d(ts[finite], la[finite], bounds_error=False, fill_value=np.nan)
    f_lo = interp1d(ts[finite], lo[finite], bounds_error=False, fill_value=np.nan)
    return f_la(query_ts), f_lo(query_ts)


def _mean_error_m(pred_la, pred_lo, true_la, true_lo) -> float:
    vm = (np.isfinite(pred_la) & np.isfinite(pred_lo)
          & np.isfinite(true_la) & np.isfinite(true_lo))
    if vm.sum() == 0:
        return np.nan
    _, _, d = geod.inv(pred_lo[vm], pred_la[vm], true_lo[vm], true_la[vm])
    return float(np.mean(np.abs(d)))


def evaluate_all(
    cleaned_dir:  Path,
    recon_dir:    Path,
    max_flights:  int = 100,
) -> pd.DataFrame:
    """
    For each reconstructed flight, load the ADS-C ground truth and
    compute geodesic error for baseline, Kalman and BiGRU.
    """
    records = []
    count   = 0

    for step_dir in sorted(recon_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        for flight_dir in sorted(step_dir.iterdir()):
            if not flight_dir.is_dir() or count >= max_flights:
                continue

            flight_name = flight_dir.name
            clean_flight = cleaned_dir / step_dir.name / flight_name

            # Required files
            adsc_path    = clean_flight / "adsc.parquet"
            bef_path     = clean_flight / "adsb_before.parquet"
            aft_path     = clean_flight / "adsb_after.parquet"
            bigru_path   = flight_dir   / "full_reconstruction.parquet"
            base_path    = flight_dir   / "baseline_full_reconstruction.parquet"
            kalman_path  = flight_dir   / "kalman_full_reconstruction.parquet"

            if not (adsc_path.exists() and bef_path.exists()
                    and aft_path.exists() and bigru_path.exists()
                    and base_path.exists() and kalman_path.exists()):
                continue

            try:
                adsc   = pd.read_parquet(adsc_path)
                bef    = pd.read_parquet(bef_path)
                aft    = pd.read_parquet(aft_path)
                bigru  = pd.read_parquet(bigru_path)
                base   = pd.read_parquet(base_path)
                kalman = pd.read_parquet(kalman_path)
            except Exception:
                continue

            # Normalize timestamps
            for df in [adsc, bef, aft, bigru, base, kalman]:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

            bef  = bef.sort_values("timestamp").reset_index(drop=True)
            aft  = aft.sort_values("timestamp").reset_index(drop=True)
            adsc = adsc.sort_values("timestamp").reset_index(drop=True)

            t_start  = bef["timestamp"].iloc[-1]
            t_end    = aft["timestamp"].iloc[0]
            gap_min  = (t_end - t_start).total_seconds() / 60
            if gap_min < 5:
                continue

            # ADS-C waypoints strictly inside the gap
            adsc_gap = adsc[
                (adsc["timestamp"] > t_start) & (adsc["timestamp"] < t_end)
            ].reset_index(drop=True)
            if len(adsc_gap) == 0:
                continue

            adsc_ts  = _ts_to_float(adsc_gap["timestamp"])
            true_la  = adsc_gap["latitude"].values.astype(float)
            true_lo  = adsc_gap["longitude"].values.astype(float)

            # Only use interpolated rows for reconstruction eval
            def _gap_rows(df):
                if "interpolated" in df.columns:
                    return df[df["interpolated"] == True].reset_index(drop=True)
                return df

            base_gap   = _gap_rows(base)
            kalman_gap = _gap_rows(kalman)
            bigru_gap  = _gap_rows(bigru)

            bla, blo = _interp_at(base_gap,   "latitude", "longitude", adsc_ts)
            kla, klo = _interp_at(kalman_gap, "latitude", "longitude", adsc_ts)
            gla, glo = _interp_at(bigru_gap,  "latitude", "longitude", adsc_ts)

            records.append({
                "flight":          flight_name,
                "step":            step_dir.name,
                "gap_minutes":     round(gap_min, 1),
                "adsc_waypoints":  len(adsc_gap),
                "baseline_mean_m": _mean_error_m(bla, blo, true_la, true_lo),
                "kalman_mean_m":   _mean_error_m(kla, klo, true_la, true_lo),
                "bigru_mean_m":    _mean_error_m(gla, glo, true_la, true_lo),
            })
            count += 1

    return pd.DataFrame(records)


def print_summary(eval_df: pd.DataFrame) -> None:
    active = [m for m in ["baseline", "kalman", "bigru"]
              if f"{m}_mean_m" in eval_df.columns
              and eval_df[f"{m}_mean_m"].notna().any()]

    print(f"\n{'Method':<12}  {'Mean (km)':>10}  {'Median (km)':>12}  {'Flights':>8}")
    print("-" * 48)
    for m in active:
        col = eval_df[f"{m}_mean_m"].dropna()
        print(f"  {m:<10}  {col.mean()/1000:>10.1f}  "
              f"{col.median()/1000:>12.1f}  {len(col):>6}")

    if "baseline_mean_m" in eval_df and "kalman_mean_m" in eval_df:
        imp_k = (1 - eval_df["kalman_mean_m"].mean()
                 / eval_df["baseline_mean_m"].mean()) * 100
        print(f"\n  Kalman improvement : {imp_k:.1f}%")
    if "baseline_mean_m" in eval_df and "bigru_mean_m" in eval_df:
        imp_g = (1 - eval_df["bigru_mean_m"].mean()
                 / eval_df["baseline_mean_m"].mean()) * 100
        print(f"  BiGRU improvement  : {imp_g:.1f}%")

    print(f"\n  Flights evaluated  : {len(eval_df)}")
    print(f"  Avg gap duration   : {eval_df['gap_minutes'].mean():.1f} min")
    print(f"  Avg ADS-C waypts   : {eval_df['adsc_waypoints'].mean():.1f}")


if __name__ == "__main__":
    cleaned = Path("noel_part/cleaned_data_final")
    recon   = Path("noel_part/final_reconstructions")
    df = evaluate_all(cleaned, recon, max_flights=100)
    print_summary(df)
    df.to_csv("outputs/evaluation_results.csv", index=False)
    print("\nSaved → outputs/evaluation_results.csv")
