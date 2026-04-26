"""
step3_baseline.py — Great-circle baseline reconstruction
=========================================================
Implements reconstruct_gap_baseline() from reconstruction.py.
Used by 03_baseline.ipynb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Geod

geod = Geod(ellps="WGS84")


def great_circle_interp(lat1, lon1, lat2, lon2, n_points):
    """Return n_points evenly spaced along the great-circle arc."""
    if n_points == 0:
        return np.array([]), np.array([])
    pts = geod.npts(lon1, lat1, lon2, lat2, n_points)
    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    return lats, lons


def reconstruct_gap_baseline(before_df: pd.DataFrame,
                              after_df:  pd.DataFrame,
                              dt: float = 15.0) -> pd.DataFrame:
    """
    Great-circle interpolation between last ADS-B before and first ADS-B after.
    Forward and backward arcs are blended linearly.
    """
    last_time  = before_df["timestamp"].iloc[-1]
    first_time = after_df["timestamp"].iloc[0]
    n_steps = max(1, int(round(
        (first_time - last_time).total_seconds() / dt
    )))

    before_c = before_df.drop_duplicates(
        subset=["latitude", "longitude"], keep="last"
    ).reset_index(drop=True)
    after_c = after_df.drop_duplicates(
        subset=["latitude", "longitude"], keep="first"
    ).reset_index(drop=True)

    # Forward: interpolate from last before → first after
    fwd_lats, fwd_lons = great_circle_interp(
        before_c["latitude"].iloc[-1], before_c["longitude"].iloc[-1],
        after_c["latitude"].iloc[0],   after_c["longitude"].iloc[0],
        n_steps
    )
    fwd_alts = np.linspace(
        before_c["altitude"].iloc[-1], after_c["altitude"].iloc[0], n_steps
    )
    fwd = np.stack([fwd_lats, fwd_lons, fwd_alts], axis=1)

    # Backward: same arc reversed
    bwd_lats, bwd_lons = great_circle_interp(
        after_c["latitude"].iloc[0],   after_c["longitude"].iloc[0],
        before_c["latitude"].iloc[-1], before_c["longitude"].iloc[-1],
        n_steps
    )
    bwd = np.stack([bwd_lats[::-1], bwd_lons[::-1], fwd_alts], axis=1)

    alpha   = np.linspace(1.0, 0.0, n_steps).reshape(-1, 1)
    blended = alpha * fwd + (1 - alpha) * bwd

    timestamps = [
        last_time + pd.Timedelta(seconds=dt * (i + 1))
        for i in range(n_steps)
    ]
    result = pd.DataFrame(blended, columns=["latitude", "longitude", "altitude"])
    result["timestamp"]    = timestamps
    result["interpolated"] = True
    return result


def evaluate_baseline(flights_dir: Path, max_flights: int = 100) -> pd.DataFrame:
    """
    Evaluate baseline against real ADS-C ground truth.
    Returns DataFrame with per-flight errors.
    """
    records = []
    for step_dir in sorted(flights_dir.iterdir()):
        if not step_dir.is_dir():
            continue
        for flight_dir in sorted(step_dir.iterdir()):
            if not flight_dir.is_dir():
                continue
            if len(records) >= max_flights:
                break
            try:
                bef  = pd.read_parquet(flight_dir / "adsb_before.parquet")
                aft  = pd.read_parquet(flight_dir / "adsb_after.parquet")
                adsc = pd.read_parquet(flight_dir / "adsc.parquet")
            except Exception:
                continue

            for df in [bef, aft, adsc]:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

            bef  = bef.sort_values("timestamp").reset_index(drop=True)
            aft  = aft.sort_values("timestamp").reset_index(drop=True)
            adsc = adsc.sort_values("timestamp").reset_index(drop=True)

            t_start = bef["timestamp"].iloc[-1]
            t_end   = aft["timestamp"].iloc[0]
            gap_min = (t_end - t_start).total_seconds() / 60
            if gap_min < 5:
                continue

            try:
                recon = reconstruct_gap_baseline(bef, aft)
            except Exception:
                continue

            # Evaluate at ADS-C waypoints
            adsc_gap = adsc[
                (adsc["timestamp"] > t_start) & (adsc["timestamp"] < t_end)
            ]
            if len(adsc_gap) == 0:
                continue

            from scipy.interpolate import interp1d
            recon_ts = recon["timestamp"].apply(lambda x: x.timestamp()).values
            adsc_ts  = adsc_gap["timestamp"].apply(lambda x: x.timestamp()).values

            if len(recon_ts) < 2:
                continue

            f_la = interp1d(recon_ts, recon["latitude"].values,
                            bounds_error=False, fill_value=np.nan)
            f_lo = interp1d(recon_ts, recon["longitude"].values,
                            bounds_error=False, fill_value=np.nan)

            pred_la = f_la(adsc_ts)
            pred_lo = f_lo(adsc_ts)
            true_la = adsc_gap["latitude"].values
            true_lo = adsc_gap["longitude"].values

            vm = (np.isfinite(pred_la) & np.isfinite(pred_lo)
                  & np.isfinite(true_la) & np.isfinite(true_lo))
            if vm.sum() == 0:
                continue

            _, _, dists = geod.inv(pred_lo[vm], pred_la[vm],
                                   true_lo[vm],  true_la[vm])
            records.append({
                "flight":          flight_dir.name,
                "gap_minutes":     round(gap_min, 1),
                "adsc_waypoints":  int(vm.sum()),
                "baseline_mean_m": float(np.mean(np.abs(dists))),
                "baseline_max_m":  float(np.max(np.abs(dists))),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        step, flight = sys.argv[1], sys.argv[2]
        bef  = pd.read_parquet(f"noel_part/cleaned_data_final/{step}/{flight}/adsb_before.parquet")
        aft  = pd.read_parquet(f"noel_part/cleaned_data_final/{step}/{flight}/adsb_after.parquet")
        result = reconstruct_gap_baseline(bef, aft)
        print(f"Baseline reconstruction: {len(result)} steps")
        print(result[["timestamp","latitude","longitude","altitude"]].head())
