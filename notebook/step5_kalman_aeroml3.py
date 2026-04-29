"""
step5_kalman_aeroml3.py
========================
Anchored Kalman smoother for AeroML3.

Key fix over previous version:
  - Initializes AT the before anchor (0,0) with velocity from ADS-B heading
  - Uses ONLY the after anchor as a terminal measurement (avoids P-degeneracy from before context)
  - Large P0 lets the RTS backward pass actually correct the trajectory
  - Heading-based velocity gives the actual NAT-track direction, not just great-circle

Input:  artifacts/step4_ml_dataset/dataset/sequences_{train,val,test}.npz
        noel_part/cleaned_data_final/{source_run}/{flight_name}/adsb_before.parquet
        noel_part/cleaned_data_final/{source_run}/{flight_name}/adsb_after.parquet
Output: artifacts/step5_kalman/
           test_summary.json
           test_predictions.npz
           per_flight_metrics_test.csv
           val_tuning_results.csv
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EARTH_RADIUS_M       = 6_371_000.0
MAX_ADSC_WP          = 64
BEFORE_CONTEXT_STEPS = 64
AFTER_CONTEXT_STEPS  = 32
DEFAULT_RESAMPLE_SEC = 60


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KalmanConfig:
    clean_root:           Path
    step4_dataset_root:   Path
    output_root:          Path
    resample_seconds:     int   = DEFAULT_RESAMPLE_SEC
    tuning_grid:          str   = "compact"
    max_flights_per_split: int | None = None
    verbose:              bool  = True
    clean_existing_output: bool = True


@dataclass(frozen=True)
class KalmanParams:
    measurement_std_m:     float   # terminal anchor uncertainty
    accel_std_along_mps2:  float   # along-track process noise
    accel_std_cross_mps2:  float   # cross-track process noise


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_lon(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1 = math.radians(float(lat1)); p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return EARTH_RADIUS_M * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1 = math.radians(float(lat1)); p2 = math.radians(float(lat2))
    dl = math.radians(float(lon2) - float(lon1))
    y  = math.sin(dl) * math.cos(p2)
    x  = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _xyz(lat, lon) -> np.ndarray:
    la, lo = math.radians(float(lat)), math.radians(float(lon))
    return np.array([math.cos(la)*math.cos(lo),
                     math.cos(la)*math.sin(lo),
                     math.sin(la)], dtype=float)


def _latlon(xyz) -> tuple[float, float]:
    u = xyz / np.linalg.norm(xyz)
    lat = math.degrees(math.asin(float(np.clip(u[2], -1.0, 1.0))))
    lon = math.degrees(math.atan2(float(u[1]), float(u[0])))
    return lat, _wrap_lon(lon)


def gc_interp(slat, slon, elat, elon, tau) -> tuple[float, float]:
    tau = float(np.clip(tau, 0.0, 1.0))
    p0, p1 = _xyz(slat, slon), _xyz(elat, elon)
    omega = math.acos(float(np.clip(np.dot(p0, p1), -1.0, 1.0)))
    if abs(omega) < 1e-12:
        return float(slat), _wrap_lon(float(slon))
    s = math.sin(omega)
    return _latlon(math.sin((1-tau)*omega)/s * p0 + math.sin(tau*omega)/s * p1)


def destination_point(lat, lon, brg, dist_m) -> tuple[float, float]:
    ad = float(dist_m) / EARTH_RADIUS_M
    th = math.radians(float(brg))
    p1 = math.radians(float(lat))
    l1 = math.radians(float(lon))
    sp2 = math.sin(p1)*math.cos(ad) + math.cos(p1)*math.sin(ad)*math.cos(th)
    p2  = math.asin(float(np.clip(sp2, -1.0, 1.0)))
    y   = math.sin(th)*math.sin(ad)*math.cos(p1)
    x   = math.cos(ad) - math.sin(p1)*math.sin(p2)
    l2  = l1 + math.atan2(y, x)
    return math.degrees(p2), _wrap_lon(math.degrees(l2))


def path_state_to_latlon(along_m, cross_m, slat, slon, elat, elon) -> tuple[float, float]:
    total = max(haversine_m(slat, slon, elat, elon), 1.0)
    tau   = float(np.clip(along_m / total, 0.0, 1.0))
    base_lat, base_lon = gc_interp(slat, slon, elat, elon, tau)
    # course at this point along the great circle
    eps = min(0.01, 10_000.0 / max(total, 1.0))
    t0c = float(np.clip(tau - eps, 0.0, 1.0))
    t1c = float(np.clip(tau + eps, 0.0, 1.0))
    if t1c <= t0c:
        t0c = max(0.0, tau - 1e-6); t1c = min(1.0, tau + 1e-6)
    la0, lo0 = gc_interp(slat, slon, elat, elon, t0c)
    la1, lo1 = gc_interp(slat, slon, elat, elon, t1c)
    course = bearing_deg(la0, lo0, la1, lo1)
    offset_brg = course + 90.0 if cross_m >= 0.0 else course - 90.0
    return destination_point(base_lat, base_lon, offset_brg, abs(float(cross_m)))


# ---------------------------------------------------------------------------
# Entry velocity from ADS-B heading
# ---------------------------------------------------------------------------

def _get_entry_velocity(bef: pd.DataFrame, slat, slon, elat, elon) -> tuple[float, float]:
    """
    Estimate along/cross-track velocity at gap start from ADS-B data.
    Uses velocity_mps + heading_deg columns directly (avoids stale last-position
    repeats that the ADS-B receiver often emits after losing signal).
    Falls back to position-difference if those columns are absent.
    """
    df = bef.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["timestamp","latitude","longitude"]).sort_values("timestamp")
    if df.empty:
        return 240.0, 0.0

    path_brg = bearing_deg(slat, slon, elat, elon)

    # Preferred: use on-board velocity_mps + heading_deg (most accurate)
    if "velocity_mps" in df.columns and "heading_deg" in df.columns:
        moving = df[(pd.to_numeric(df["velocity_mps"], errors="coerce").fillna(0) > 100) &
                    pd.to_numeric(df["heading_deg"], errors="coerce").notna()].copy()
        if moving.empty:
            moving = df[(pd.to_numeric(df["velocity_mps"], errors="coerce").fillna(0) > 50) &
                        pd.to_numeric(df["heading_deg"], errors="coerce").notna()].copy()
        if not moving.empty:
            row   = moving.iloc[-1]
            speed = float(np.clip(pd.to_numeric(row["velocity_mps"]), 100.0, 350.0))
            hdg   = float(pd.to_numeric(row["heading_deg"]))
            diff  = float(((hdg - path_brg + 180.0) % 360.0) - 180.0)
            v_along = speed * math.cos(math.radians(diff))
            v_cross = speed * math.sin(math.radians(diff))
            return float(v_along), float(v_cross)

    # Fallback: position differences between last 2 distinct points
    distinct = df.drop_duplicates(subset=["latitude","longitude"]).tail(2)
    if len(distinct) < 2:
        return 240.0, 0.0
    lat1 = float(distinct.iloc[0]["latitude"]); lon1 = float(distinct.iloc[0]["longitude"])
    lat2 = float(distinct.iloc[1]["latitude"]); lon2 = float(distinct.iloc[1]["longitude"])
    dt   = float((pd.Timestamp(distinct.iloc[1]["timestamp"]) -
                  pd.Timestamp(distinct.iloc[0]["timestamp"])).total_seconds())
    if dt < 1.0:
        return 240.0, 0.0
    speed = float(np.clip(haversine_m(lat1, lon1, lat2, lon2) / dt, 100.0, 350.0))
    hdg   = bearing_deg(lat1, lon1, lat2, lon2)
    diff  = float(((hdg - path_brg + 180.0) % 360.0) - 180.0)
    return float(speed * math.cos(math.radians(diff))), float(speed * math.sin(math.radians(diff)))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    w["latitude"]  = pd.to_numeric(w["latitude"],  errors="coerce")
    w["longitude"] = pd.to_numeric(w["longitude"], errors="coerce")
    return (w.dropna(subset=["timestamp","latitude","longitude"])
             .sort_values("timestamp")
             .drop_duplicates(subset=["timestamp"], keep="last")
             .reset_index(drop=True))


# ---------------------------------------------------------------------------
# PreparedFlight
# ---------------------------------------------------------------------------

@dataclass
class PreparedFlight:
    split:             str
    segment_id:        str
    t0:                pd.Timestamp
    t1:                pd.Timestamp
    start_lat:         float
    start_lon:         float
    end_lat:           float
    end_lon:           float
    v0_along:          float         # along-track velocity at gap start (m/s)
    v0_cross:          float         # cross-track velocity at gap start (m/s)
    target_times:      list
    valid_indices:     np.ndarray
    adsc_tau_full:     np.ndarray
    adsc_mask_full:    np.ndarray
    true_lat_full:     np.ndarray
    true_lon_full:     np.ndarray
    baseline_lat_full: np.ndarray
    baseline_lon_full: np.ndarray


def _baseline_wps(slat, slon, elat, elon, tau) -> tuple[np.ndarray, np.ndarray]:
    pred_lat = np.zeros_like(tau, dtype=np.float32)
    pred_lon = np.zeros_like(tau, dtype=np.float32)
    for i, t in enumerate(tau):
        la, lo = gc_interp(slat, slon, elat, elon, float(t))
        pred_lat[i] = np.float32(la)
        pred_lon[i] = np.float32(lo)
    return pred_lat, pred_lon


def _prepare_flight(idx: int, dataset: dict, config: KalmanConfig) -> PreparedFlight | None:
    segment_id = str(dataset["segment_ids"][idx])
    flight_path = config.clean_root / segment_id
    before_p = flight_path / "adsb_before.parquet"
    after_p  = flight_path / "adsb_after.parquet"

    if not before_p.exists() or not after_p.exists():
        return None

    bef = _coerce(pd.read_parquet(before_p))
    aft = _coerce(pd.read_parquet(after_p))
    if bef.empty or aft.empty:
        return None

    t0 = pd.Timestamp(bef.iloc[-1]["timestamp"])
    t1 = pd.Timestamp(aft.iloc[0]["timestamp"])
    if t1 <= t0:
        return None

    slat = float(dataset["before_anchor_lat"][idx])
    slon = float(dataset["before_anchor_lon"][idx])
    elat = float(dataset["after_anchor_lat"][idx])
    elon = float(dataset["after_anchor_lon"][idx])

    v0_along, v0_cross = _get_entry_velocity(bef, slat, slon, elat, elon)

    tau_full  = dataset["adsc_tau"][idx].astype(np.float32)
    mask_full = dataset["adsc_mask"][idx].astype(np.float32)
    valid_idx = np.where(mask_full > 0)[0]
    gap_sec   = float(dataset["gap_dur_sec"][idx])

    target_times = [t0 + pd.Timedelta(seconds=float(tau_full[j] * gap_sec))
                    for j in valid_idx]

    bl_lat, bl_lon = _baseline_wps(slat, slon, elat, elon, tau_full)

    return PreparedFlight(
        split=str(dataset.get("split", ["train"])),
        segment_id=segment_id,
        t0=t0, t1=t1,
        start_lat=slat, start_lon=slon,
        end_lat=elat, end_lon=elon,
        v0_along=v0_along, v0_cross=v0_cross,
        target_times=target_times,
        valid_indices=valid_idx.astype(int),
        adsc_tau_full=tau_full,
        adsc_mask_full=mask_full,
        true_lat_full=dataset["adsc_targets"][idx, :, 0].astype(np.float32),
        true_lon_full=dataset["adsc_targets"][idx, :, 1].astype(np.float32),
        baseline_lat_full=bl_lat,
        baseline_lon_full=bl_lon,
    )


# ---------------------------------------------------------------------------
# Single-flight reconstruction API (used by step 7 visualization)
# ---------------------------------------------------------------------------

#: Default Kalman params — must match the tuned values from step5 evaluation
DEFAULT_KALMAN_PARAMS = KalmanParams(
    measurement_std_m    = 200_000.0,
    accel_std_along_mps2 = 0.005,
    accel_std_cross_mps2 = 4.0,
)


def reconstruct_single_kalman(
    before_df: pd.DataFrame,
    after_df:  pd.DataFrame,
    dt:        float = 15.0,
    params:    KalmanParams | None = None,
) -> pd.DataFrame:
    """
    Reconstruct a single ADS-B gap using the anchored Kalman smoother.

    Parameters
    ----------
    before_df : ADS-B dataframe before the gap (needs timestamp, latitude,
                longitude; altitude optional)
    after_df  : ADS-B dataframe after the gap
    dt        : output time step in seconds (default 15 s, matching step 7)
    params    : KalmanParams; defaults to validation-tuned values

    Returns
    -------
    DataFrame with columns: timestamp, latitude, longitude, altitude,
    interpolated  — same format as reconstruct_gap_kalman in reconstruction.py
    """
    if params is None:
        params = DEFAULT_KALMAN_PARAMS

    empty = pd.DataFrame(columns=["timestamp", "latitude", "longitude",
                                   "altitude", "interpolated"])

    bef = _coerce(before_df[
        [c for c in ["timestamp","latitude","longitude","altitude"]
         if c in before_df.columns]].copy())
    aft = _coerce(after_df[
        [c for c in ["timestamp","latitude","longitude","altitude"]
         if c in after_df.columns]].copy())

    if bef.empty or aft.empty:
        return empty

    t0   = pd.Timestamp(bef.iloc[-1]["timestamp"])
    t1   = pd.Timestamp(aft.iloc[0]["timestamp"])
    if t1 <= t0:
        return empty

    slat = float(bef.iloc[-1]["latitude"]);   slon = float(bef.iloc[-1]["longitude"])
    elat = float(aft.iloc[0]["latitude"]);    elon = float(aft.iloc[0]["longitude"])
    salt = float(bef.iloc[-1]["altitude"]) if "altitude" in bef.columns else 10_000.0
    ealt = float(aft.iloc[0]["altitude"])  if "altitude" in aft.columns else 10_000.0

    v0_along, v0_cross = _get_entry_velocity(before_df, slat, slon, elat, elon)

    # Build output timestamps: include the anchor endpoints plus uniform interior steps.
    # Including t0 and t1 ensures the Kalman line visually connects to the ADS-B tracks.
    total_sec = (t1 - t0).total_seconds()
    n_steps   = max(1, int(round(total_sec / dt)))
    interior  = [t0 + pd.Timedelta(seconds=dt * (i + 1)) for i in range(n_steps - 1)]
    interior  = [t for t in interior if t0 < t < t1]
    out_times = sorted(set([t0] + interior + [t1]))

    flight = PreparedFlight(
        split="single", segment_id="single",
        t0=t0, t1=t1,
        start_lat=slat, start_lon=slon,
        end_lat=elat,   end_lon=elon,
        v0_along=v0_along, v0_cross=v0_cross,
        target_times=out_times,
        valid_indices=np.arange(len(out_times)),
        adsc_tau_full=np.zeros(len(out_times), dtype=np.float32),
        adsc_mask_full=np.ones(len(out_times),  dtype=np.float32),
        true_lat_full=np.zeros(len(out_times),  dtype=np.float32),
        true_lon_full=np.zeros(len(out_times),  dtype=np.float32),
        baseline_lat_full=np.zeros(len(out_times), dtype=np.float32),
        baseline_lon_full=np.zeros(len(out_times), dtype=np.float32),
    )

    pred_lat, pred_lon = kalman_smooth(flight, params)

    taus = np.array([(t - t0).total_seconds() / total_sec for t in out_times])
    alts = salt + taus * (ealt - salt)

    return pd.DataFrame({
        "timestamp":    out_times,
        "latitude":     pred_lat.astype(float),
        "longitude":    pred_lon.astype(float),
        "altitude":     alts,
        "interpolated": True,
    })


def prepare_split(config: KalmanConfig, split: str) -> list[PreparedFlight]:
    path = config.step4_dataset_root / f"sequences_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing step4 dataset: {path}")
    dataset = dict(np.load(path, allow_pickle=True))
    n = len(dataset["segment_ids"])
    if config.max_flights_per_split is not None:
        n = min(n, int(config.max_flights_per_split))

    flights = []
    _log(f"Preparing {split}: {n} flights", config.verbose)
    for i in range(n):
        try:
            f = _prepare_flight(i, dataset, config)
            if f is not None:
                flights.append(f)
        except Exception as exc:
            _log(f"  skip {dataset['segment_ids'][i]}: {exc}", config.verbose)
        if config.verbose and (i + 1) % 100 == 0:
            _log(f"  {i+1}/{n} prepared", config.verbose)
    _log(f"  usable: {len(flights)}", config.verbose)
    return flights


# ---------------------------------------------------------------------------
# Anchored Kalman smoother (RTS)
# ---------------------------------------------------------------------------

def _F(dt: float) -> np.ndarray:
    dt = max(dt, 0.0)
    return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)


def _Q(dt: float, qa: float, qc: float) -> np.ndarray:
    # Zero-order hold (ZOH) continuous-time CWNA covariance — dt-independent result
    # Uses dt^3/3 (not dt^4/4) so the smoother behaves consistently regardless of step size.
    dt2, dt3 = dt*dt, dt*dt*dt
    return np.array([
        [qa*dt3/3, 0, qa*dt2/2, 0],
        [0, qc*dt3/3, 0, qc*dt2/2],
        [qa*dt2/2, 0, qa*dt,   0],
        [0, qc*dt2/2, 0, qc*dt  ],
    ], dtype=float)


def kalman_smooth(flight: PreparedFlight, params: KalmanParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Anchored RTS smoother.

    State: [along_m, cross_m, v_along_mps, v_cross_mps]

    Initialization: gap start (0, 0) with velocity from ADS-B heading.
    Single terminal measurement: after anchor at (total_dist, 0).
    Large P0 velocity uncertainty → backward pass can correct trajectory.
    """
    H  = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    R  = np.diag([params.measurement_std_m**2,
                  params.measurement_std_m**2]).astype(float)
    I4 = np.eye(4, dtype=float)

    total_dist_m = haversine_m(flight.start_lat, flight.start_lon,
                                flight.end_lat,   flight.end_lon)

    # Build timeline: t0, all target times, t1  (all unique, sorted)
    t0, t1 = flight.t0, flight.t1
    ts_dict: dict[int, pd.Timestamp] = {t0.value: t0, t1.value: t1}
    for ts in flight.target_times:
        tval = pd.Timestamp(ts).value
        ts_dict[tval] = pd.Timestamp(ts)
    timeline = [ts_dict[k] for k in sorted(ts_dict)]
    n = len(timeline)

    # Initial state at before anchor
    x0 = np.array([0.0, 0.0, flight.v0_along, flight.v0_cross], dtype=float)

    # P0: tight on position (we know the start exactly), loose on velocity
    # Large velocity P0 is essential: lets the backward smoother adjust the
    # initial heading to satisfy the terminal constraint.
    P0 = np.diag([1_000.0**2,   # pos along:  1 km  (essentially exact)
                  1_000.0**2,   # pos cross:  1 km
                  400.0**2,     # vel along: 400 m/s  (very uncertain)
                  100.0**2,     # vel cross: 100 m/s  (uncertain)
                 ]).astype(float)

    tgt_map = {pd.Timestamp(ts).value: i for i, ts in enumerate(flight.target_times)}
    t1_ns   = t1.value

    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4))
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))

    # Forward pass — only measurement is the terminal anchor at t1
    xp[0] = x0; Pp[0] = P0
    xf[0] = x0; Pf[0] = P0          # no measurement at t0

    for i in range(1, n):
        dt = float((timeline[i] - timeline[i-1]).total_seconds())
        F  = _F(dt)
        Q  = _Q(dt, params.accel_std_along_mps2, params.accel_std_cross_mps2)
        xp[i] = F @ xf[i-1]
        Pp[i] = F @ Pf[i-1] @ F.T + Q

        if timeline[i].value == t1_ns:
            z   = np.array([total_dist_m, 0.0], dtype=float)
            inn = z - H @ xp[i]
            S   = H @ Pp[i] @ H.T + R
            K   = Pp[i] @ H.T @ np.linalg.inv(S)
            xf[i] = xp[i] + K @ inn
            Pf[i] = (I4 - K @ H) @ Pp[i]
        else:
            xf[i] = xp[i]; Pf[i] = Pp[i]

    # RTS backward smoother
    xs = xf.copy(); Ps = Pf.copy()
    for i in range(n - 2, -1, -1):
        dt = float((timeline[i+1] - timeline[i]).total_seconds())
        F  = _F(dt)
        try:
            G = Pf[i] @ F.T @ np.linalg.inv(Pp[i+1])
        except np.linalg.LinAlgError:
            continue
        xs[i] = xf[i] + G @ (xs[i+1] - xp[i+1])
        Ps[i] = Pf[i] + G @ (Ps[i+1] - Pp[i+1]) @ G.T

    # Hard-enforce boundary positions: the trajectory must start exactly at the
    # before anchor and end exactly at the after anchor regardless of smoother drift.
    # Only position components are clamped; velocities retain the smoother estimate.
    xs[0,  0] = 0.0
    xs[0,  1] = 0.0
    xs[-1, 0] = total_dist_m
    xs[-1, 1] = 0.0

    # Convert smoothed states at target times → lat/lon
    ts_to_si = {pd.Timestamp(t).value: j for j, t in enumerate(timeline)}
    pred_lat  = np.zeros(len(flight.target_times), dtype=np.float32)
    pred_lon  = np.zeros(len(flight.target_times), dtype=np.float32)

    for ts in flight.target_times:
        si = ts_to_si[pd.Timestamp(ts).value]
        la, lo = path_state_to_latlon(
            xs[si, 0], xs[si, 1],
            flight.start_lat, flight.start_lon,
            flight.end_lat,   flight.end_lon,
        )
        oi = tgt_map[pd.Timestamp(ts).value]
        pred_lat[oi] = np.float32(la)
        pred_lon[oi] = np.float32(lo)

    return pred_lat, pred_lon


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    flights: list[PreparedFlight],
    params: KalmanParams,
    produce_predictions: bool = False,
    verbose: bool = False,
    label: str = "split",
) -> tuple[dict[str, Any], dict[str, Any] | None]:

    per_rows = []
    k_all, b_all = [], []
    extras_lists = {k: [] for k in
                    ["pred_lat","pred_lon","true_lat","true_lon",
                     "baseline_lat","baseline_lon","mask","tau","seg"]}

    for fi, flight in enumerate(flights, 1):
        try:
            pl_valid, plo_valid = kalman_smooth(flight, params)
        except Exception as exc:
            _log(f"  {flight.segment_id}: kalman error: {exc}", verbose)
            continue

        pl_full  = np.zeros(MAX_ADSC_WP, dtype=np.float32)
        plo_full = np.zeros(MAX_ADSC_WP, dtype=np.float32)
        pl_full[flight.valid_indices]  = pl_valid
        plo_full[flight.valid_indices] = plo_valid

        valid  = flight.adsc_mask_full > 0
        k_errs = np.full(MAX_ADSC_WP, np.nan, dtype=np.float32)
        b_errs = np.full(MAX_ADSC_WP, np.nan, dtype=np.float32)
        for j in np.where(valid)[0]:
            k_errs[j] = haversine_m(pl_full[j],  plo_full[j],
                                    flight.true_lat_full[j], flight.true_lon_full[j])
            b_errs[j] = haversine_m(flight.baseline_lat_full[j], flight.baseline_lon_full[j],
                                    flight.true_lat_full[j], flight.true_lon_full[j])

        k = k_errs[valid]; b = b_errs[valid]
        k_all.append(k.astype(np.float64)); b_all.append(b.astype(np.float64))
        per_rows.append({
            "split": flight.split,
            "segment_id": flight.segment_id,
            "n_waypoints": int(valid.sum()),
            "kalman_mean_error_km":   float(np.mean(k)/1000),
            "kalman_median_error_km": float(np.median(k)/1000),
            "baseline_mean_error_km":   float(np.mean(b)/1000),
            "baseline_median_error_km": float(np.median(b)/1000),
            "improvement_mean_pct":   float((1-np.mean(k)/max(np.mean(b),1e-9))*100),
            "improvement_median_pct": float((1-np.median(k)/max(np.median(b),1e-9))*100),
        })
        if produce_predictions:
            extras_lists["pred_lat"].append(pl_full)
            extras_lists["pred_lon"].append(plo_full)
            extras_lists["true_lat"].append(flight.true_lat_full)
            extras_lists["true_lon"].append(flight.true_lon_full)
            extras_lists["baseline_lat"].append(flight.baseline_lat_full)
            extras_lists["baseline_lon"].append(flight.baseline_lon_full)
            extras_lists["mask"].append(flight.adsc_mask_full)
            extras_lists["tau"].append(flight.adsc_tau_full)
            extras_lists["seg"].append(flight.segment_id)

        if verbose and fi % 50 == 0:
            _log(f"  {label}: {fi}/{len(flights)} evaluated", verbose)

    kf = np.concatenate(k_all) if k_all else np.array([], dtype=np.float64)
    bf = np.concatenate(b_all) if b_all else np.array([], dtype=np.float64)

    summary = {
        "split": label,
        "n_flights": len(per_rows),
        "kalman_mean_error_km":     float(np.mean(kf)/1000)       if len(kf) else 0.0,
        "kalman_median_error_km":   float(np.median(kf)/1000)     if len(kf) else 0.0,
        "kalman_p90_error_km":      float(np.percentile(kf,90)/1000) if len(kf) else 0.0,
        "baseline_mean_error_km":   float(np.mean(bf)/1000)       if len(bf) else 0.0,
        "baseline_median_error_km": float(np.median(bf)/1000)     if len(bf) else 0.0,
        "baseline_p90_error_km":    float(np.percentile(bf,90)/1000) if len(bf) else 0.0,
        "improvement_mean_pct":     float((1-np.mean(kf)/max(np.mean(bf),1e-9))*100)   if len(kf) else 0.0,
        "improvement_median_pct":   float((1-np.median(kf)/max(np.median(bf),1e-9))*100) if len(kf) else 0.0,
    }

    extras = None
    if produce_predictions and extras_lists["seg"]:
        extras = {
            "segment_ids":   np.array(extras_lists["seg"], dtype=object),
            "pred_lat":      np.stack(extras_lists["pred_lat"]).astype(np.float32),
            "pred_lon":      np.stack(extras_lists["pred_lon"]).astype(np.float32),
            "true_lat":      np.stack(extras_lists["true_lat"]).astype(np.float32),
            "true_lon":      np.stack(extras_lists["true_lon"]).astype(np.float32),
            "baseline_lat":  np.stack(extras_lists["baseline_lat"]).astype(np.float32),
            "baseline_lon":  np.stack(extras_lists["baseline_lon"]).astype(np.float32),
            "mask":          np.stack(extras_lists["mask"]).astype(np.float32),
            "adsc_tau":      np.stack(extras_lists["tau"]).astype(np.float32),
            "per_flight_df": pd.DataFrame(per_rows),
        }
    return summary, extras


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def _grid(mode: str) -> list[KalmanParams]:
    if mode == "compact":
        ms = [200_000.0, 500_000.0, 1_000_000.0]
        qa = [0.0005, 0.001, 0.005]
        qc = [0.5, 1.0, 2.0, 4.0]
    elif mode == "deep":
        ms = [100_000.0, 200_000.0, 500_000.0, 1_000_000.0, 2_000_000.0]
        qa = [0.0002, 0.0005, 0.001, 0.005, 0.01]
        qc = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    else:
        raise ValueError(f"Unknown grid: {mode}")
    return [KalmanParams(m, a, c) for m, a, c in itertools.product(ms, qa, qc)]


def tune(val_flights: list[PreparedFlight], grid_mode: str, verbose: bool
         ) -> tuple[KalmanParams, pd.DataFrame]:
    grid = _grid(grid_mode)
    _log(f"Tuning on validation set: {len(grid)} candidates", verbose)
    rows, best_params, best_score = [], None, float("inf")
    for i, params in enumerate(grid, 1):
        summary, _ = evaluate_split(val_flights, params, produce_predictions=False,
                                     verbose=False)
        score = summary["kalman_mean_error_km"]
        rows.append({"candidate": i,
                     "measurement_std_m": params.measurement_std_m,
                     "accel_std_along_mps2": params.accel_std_along_mps2,
                     "accel_std_cross_mps2": params.accel_std_cross_mps2,
                     **summary})
        if score < best_score:
            best_score = score; best_params = params
            _log(f"  [{i}/{len(grid)}] new best={best_score:.3f} km  "
                 f"(std={params.measurement_std_m}, qa={params.accel_std_along_mps2}, "
                 f"qc={params.accel_std_cross_mps2})", verbose)
    df = pd.DataFrame(rows).sort_values("kalman_mean_error_km").reset_index(drop=True)
    if best_params is None:
        raise ValueError("Tuning produced no result")
    return best_params, df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _log(msg: str, enabled: bool) -> None:
    if enabled:
        print(msg, flush=True)


def run_step5_kalman(config: KalmanConfig) -> dict[str, Any]:
    if config.clean_existing_output and config.output_root.exists():
        shutil.rmtree(config.output_root)
    config.output_root.mkdir(parents=True, exist_ok=True)

    train_fl = prepare_split(config, "train")
    val_fl   = prepare_split(config, "val")
    test_fl  = prepare_split(config, "test")

    if not val_fl:
        raise ValueError("No usable validation flights")
    if not test_fl:
        raise ValueError("No usable test flights")

    best_params, tuning_df = tune(val_fl, config.tuning_grid, config.verbose)
    tuning_df.to_csv(config.output_root / "val_tuning_results.csv", index=False)

    _log("Running final validation...", config.verbose)
    val_summary, _ = evaluate_split(val_fl, best_params, produce_predictions=False,
                                     verbose=config.verbose, label="validation final")

    _log("Running final test evaluation...", config.verbose)
    test_summary, test_out = evaluate_split(test_fl, best_params, produce_predictions=True,
                                             verbose=config.verbose, label="test final")
    assert test_out is not None

    per_df = test_out.pop("per_flight_df")
    per_df.to_csv(config.output_root / "per_flight_metrics_test.csv", index=False)

    np.savez_compressed(config.output_root / "test_predictions.npz", **test_out)

    summary = {
        "config": {
            "clean_root": str(config.clean_root),
            "step4_dataset_root": str(config.step4_dataset_root),
            "output_root": str(config.output_root),
            "resample_seconds": config.resample_seconds,
            "tuning_grid": config.tuning_grid,
        },
        "prepared_counts": {"train": len(train_fl), "val": len(val_fl), "test": len(test_fl)},
        "selected_params": asdict(best_params),
        "validation_summary": val_summary,
        "test_summary": test_summary,
    }
    with open(config.output_root / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _log(json.dumps(summary, indent=2), config.verbose)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clean-root",         default="../noel_part/cleaned_data_final")
    p.add_argument("--step4-dataset-root", default="../artifacts/step4_ml_dataset/dataset")
    p.add_argument("--output-root",        default="../artifacts/step5_kalman")
    p.add_argument("--tuning-grid",        choices=["compact","deep"], default="compact")
    p.add_argument("--max-flights",        type=int, default=None)
    p.add_argument("--quiet",              action="store_true")
    args = p.parse_args()
    run_step5_kalman(KalmanConfig(
        clean_root=Path(args.clean_root),
        step4_dataset_root=Path(args.step4_dataset_root),
        output_root=Path(args.output_root),
        tuning_grid=args.tuning_grid,
        max_flights_per_split=args.max_flights,
        verbose=not args.quiet,
    ))
