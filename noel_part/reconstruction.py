import os
import numpy as np
import pandas as pd
import torch
import joblib
import argparse
from torch import nn

FEATURE_COLS = ["latitude", "longitude", "altitude",
                "vx", "vy", "acceleration",
                "turn_rate"]

TARGET_COLS  = ["latitude", "longitude", "altitude"]
                #"vx", "vy"]

SEQUENCE_LEN = 10

INPUT_FOLDER = "cleaned_data_final"

initialize_scalers_called = False

SCALER_PATH = "scalers"
alt_scaler = joblib.load(os.path.join(SCALER_PATH, "alt_scaler.pkl"))
vel_scaler = joblib.load(os.path.join(SCALER_PATH, "vel_scaler.pkl"))
other_scaler = joblib.load(os.path.join(SCALER_PATH, "other_scaler.pkl"))
target_scaler = joblib.load(os.path.join(SCALER_PATH, "target_scaler.pkl"))

class TrajectoryBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=0.3 if num_layers > 1 else 0, bidirectional=True)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.gru(x)                      # (B, 10, H*2)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, 10, 1)
        context = (weights * out).sum(dim=1)       # (B, H*2)
        return self.fc(self.dropout(context))

def scale_data(X, is_target=False):
    X = X.copy()
    original_shape = X.shape
    X = X.reshape(-1, X.shape[-1])

    if is_target:
        X = target_scaler.transform(X)
    else:
        X[:, 2:3] = alt_scaler.transform(X[:, 2:3])
        X[:, 3:5] = vel_scaler.transform(X[:, 3:5])
        X[:, 5:7] = other_scaler.transform(X[:, 5:7])

    return X.reshape(original_shape)

def inverse_scale_targets(Y):
    Y = Y.copy()
    original_shape = Y.shape
    Y = Y.reshape(-1, Y.shape[-1])
    Y = target_scaler.inverse_transform(Y)
    return Y.reshape(original_shape)

def great_circle_extrapolate(lat1, lon1, lat2, lon2):
    """Extrapolate one step forward from two points using great-circle projection."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Bearing from point 1 to point 2
    dlon = lon2 - lon1
    x = np.cos(lat2) * np.sin(dlon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    
    # Angular distance between point 1 and point 2
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    angular_dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Project from point 2 with same bearing and distance
    lat3 = np.arcsin(np.sin(lat2) * np.cos(angular_dist) +
                     np.cos(lat2) * np.sin(angular_dist) * np.cos(bearing))
    lon3 = lon2 + np.arctan2(np.sin(bearing) * np.sin(angular_dist) * np.cos(lat2),
                              np.cos(angular_dist) - np.sin(lat2) * np.sin(lat3))
    
    return np.degrees(lat3), np.degrees(lon3)


def create_windows_relative(flight_df, sequence_len, feature_cols, target_cols):
    X, y, gc = [], [], []

    flight_df = flight_df.sort_values("timestamp").reset_index(drop=True)
    flight_df = flight_df.dropna()

    if len(flight_df) < sequence_len + 1:
        return np.array(X), np.array(y), np.array(gc)

    features = flight_df[feature_cols].values
    targets  = flight_df[target_cols].values
    
    # Raw lat/lon/alt for great-circle computation
    raw_lat = flight_df["latitude"].values
    raw_lon = flight_df["longitude"].values
    raw_alt = flight_df["altitude"].values

    for i in range(len(flight_df) - sequence_len):
        window = features[i : i + sequence_len].copy()
        target = targets[i + sequence_len].copy()

        # Store the last point as reference
        ref_lat = window[-1, 0]
        ref_lon = window[-1, 1]
        ref_alt = window[-1, 2]

        # Make lat/lon/alt relative to last point in window
        window[:, 0] -= ref_lat
        window[:, 1] -= ref_lon
        window[:, 2] -= ref_alt

        target[0] -= ref_lat
        target[1] -= ref_lon
        target[2] -= ref_alt

        # Great-circle extrapolation from last two raw points
        gc_lat, gc_lon = great_circle_extrapolate(
            raw_lat[i + sequence_len - 2], raw_lon[i + sequence_len - 2],
            raw_lat[i + sequence_len - 1], raw_lon[i + sequence_len - 1]
        )
        gc_alt = raw_alt[i + sequence_len - 1]

        # gc_guess also relative to the same reference point
        gc_guess = [gc_lat - ref_lat, gc_lon - ref_lon, gc_alt - ref_alt]

        X.append(window)
        y.append(target)
        gc.append(gc_guess)

 

    return np.array(X), np.array(y), np.array(gc)

def reconstruct_gap(model, before_df, after_df, feature_cols, target_cols,
                    sequence_len, device, dt=15.0):
    """
    Reconstruct missing trajectory between before_df and after_df.
    Returns a DataFrame with columns: timestamp, latitude, longitude, altitude, interpolated
    """
    model.eval()

    def haversine_dist(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * 6371000 * np.arcsin(np.sqrt(a))

    def bearing(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        return np.degrees(np.arctan2(x, y)) % 360

    def recompute_derived(context, dt=15.0):
        n = len(context)
        if n < 2:
            return context
        lat1, lon1 = context[-2, 0], context[-2, 1]
        lat2, lon2 = context[-1, 0], context[-1, 1]
        heading = bearing(lat1, lon1, lat2, lon2)
        velocity = haversine_dist(lat1, lon1, lat2, lon2) / dt
        heading_rad = np.deg2rad(heading)
        context[-1, 3] = velocity * np.sin(heading_rad)
        context[-1, 4] = velocity * np.cos(heading_rad)
        if n >= 3:
            prev_lat, prev_lon = context[-3, 0], context[-3, 1]
            prev_heading = bearing(prev_lat, prev_lon, lat1, lon1)
            prev_velocity = haversine_dist(prev_lat, prev_lon, lat1, lon1) / dt
            context[-1, 5] = (velocity - prev_velocity) / dt
            context[-1, 6] = (heading - prev_heading) / dt
        else:
            context[-1, 5] = 0.0
            context[-1, 6] = 0.0
        return context

    def prepare_context(df, from_end=True):
        if from_end:
            ctx = df[feature_cols].values[-sequence_len:].copy()
        else:
            ctx = df[feature_cols].values[:sequence_len].copy()

        # Pad if not enough rows
        if len(ctx) < sequence_len:
            pad_size = sequence_len - len(ctx)
            if from_end:
                pad = np.repeat(ctx[:1], pad_size, axis=0)
                ctx = np.vstack([pad, ctx])
            else:
                pad = np.repeat(ctx[-1:], pad_size, axis=0)
                ctx = np.vstack([ctx, pad])

        # Fill NaNs forward then backward
        for col in range(ctx.shape[1]):
            if np.isnan(ctx[:, col]).all():
                ctx[:, col] = 0.0
                continue
            for i in range(1, len(ctx)):
                if np.isnan(ctx[i, col]):
                    ctx[i, col] = ctx[i-1, col]
            for i in range(len(ctx)-2, -1, -1):
                if np.isnan(ctx[i, col]):
                    ctx[i, col] = ctx[i+1, col]
        return ctx

    def predict_steps(context, n_steps):
        preds = []
        ctx = context.copy()
        for _ in range(n_steps):
            ref_lat, ref_lon, ref_alt = ctx[-1, 0], ctx[-1, 1], ctx[-1, 2]

            gc_lat, gc_lon = great_circle_extrapolate(
                ctx[-2, 0], ctx[-2, 1], ctx[-1, 0], ctx[-1, 1]
            )
            gc_rel = np.array([gc_lat - ref_lat, gc_lon - ref_lon, 0.0])

            window = ctx.copy()
            window[:, 0] -= ref_lat
            window[:, 1] -= ref_lon
            window[:, 2] -= ref_alt

            window_scaled = scale_data(window.reshape(1, sequence_len, -1), is_target=False)
            input_tensor = torch.FloatTensor(window_scaled).to(device)

            with torch.no_grad():
                pred_scaled = model(input_tensor).cpu().numpy()

            pred_residual = inverse_scale_targets(pred_scaled)[0]

            pred_abs = np.array([
                gc_rel[0] + pred_residual[0] + ref_lat,
                gc_rel[1] + pred_residual[1] + ref_lon,
                gc_rel[2] + pred_residual[2] + ref_alt
            ])
            preds.append(pred_abs)

            new_row = np.zeros(len(feature_cols))
            new_row[0], new_row[1], new_row[2] = pred_abs
            ctx = np.vstack([ctx[1:], new_row])
            ctx = recompute_derived(ctx, dt)
        return np.array(preds)

    # ── Determine number of steps from timestamps ────────────────────────────
    last_time = before_df["timestamp"].iloc[-1]
    first_time = after_df["timestamp"].iloc[0]
    gap_seconds = (first_time - last_time).total_seconds()
    n_steps = max(1, int(round(gap_seconds / dt)))

    # ── Deduplicate context edges ────────────────────────────────────────────
    before_clean = before_df.drop_duplicates(subset=["latitude", "longitude"], keep="last").reset_index(drop=True)
    after_clean = after_df.drop_duplicates(subset=["latitude", "longitude"], keep="first").reset_index(drop=True)

    # ── Forward and backward predictions ─────────────────────────────────────
    ctx_fwd = prepare_context(before_clean, from_end=True)
    forward_preds = predict_steps(ctx_fwd, n_steps)


    # ── Backward prediction (only if after_df has enough points) ─────────
    if len(after_clean) >= sequence_len:
        ctx_bwd = prepare_context(after_clean, from_end=False)
        ctx_bwd = ctx_bwd[::-1].copy()
        backward_preds = predict_steps(ctx_bwd, n_steps)
        backward_preds = backward_preds[::-1]

        alpha = np.linspace(1.0, 0.0, n_steps).reshape(-1, 1)
        blended = alpha * forward_preds + (1 - alpha) * backward_preds
    else:
        blended = forward_preds
  

    # ── Build output DataFrame ───────────────────────────────────────────────
    timestamps = [last_time + pd.Timedelta(seconds=dt * (i + 1)) for i in range(n_steps)]
    result = pd.DataFrame(blended, columns=["latitude", "longitude", "altitude"])
    result["timestamp"] = timestamps
    result["interpolated"] = True

    return result



def reconstruct_gap_baseline(before_df, after_df, dt=15.0):
    """
    Great-circle interpolation between last known before point
    and first known after point. This is the standard baseline.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    last_time  = before_df["timestamp"].iloc[-1]
    first_time = after_df["timestamp"].iloc[0]
    n_steps = max(1, int(round((first_time - last_time).total_seconds() / dt)))

    before_c = before_df.drop_duplicates(
        subset=["latitude","longitude"], keep="last").reset_index(drop=True)
    after_c  = after_df.drop_duplicates(
        subset=["latitude","longitude"], keep="first").reset_index(drop=True)

    lat0 = before_c["latitude"].iloc[-1];  lon0 = before_c["longitude"].iloc[-1]
    alt0 = before_c["altitude"].iloc[-1]
    lat1 = after_c["latitude"].iloc[0];    lon1 = after_c["longitude"].iloc[0]
    alt1 = after_c["altitude"].iloc[0]

    # n_steps interior points along the great-circle arc
    pts  = geod.npts(lon0, lat0, lon1, lat1, n_steps)
    lats = np.array([p[1] for p in pts])
    lons = np.array([p[0] for p in pts])
    alts = np.linspace(alt0, alt1, n_steps)

    timestamps = [last_time + pd.Timedelta(seconds=dt*(i+1)) for i in range(n_steps)]
    result = pd.DataFrame({"latitude": lats, "longitude": lons, "altitude": alts})
    result["timestamp"]    = timestamps
    result["interpolated"] = True
    return result

def reconstruct_gap_kalman(before_df, after_df, dt=15.0):
    """
    Constant-velocity Kalman filter — forward-only pass.
    Warmed up on the last 15 ADS-B points before the gap.
    No forward-backward blending (which causes looping on long gaps).
    Retimed to constant speed so positions match real aircraft timing.
    """
    from pyproj import Geod
    from scipy.interpolate import interp1d as _i1d
    geod = Geod(ellps="WGS84")

    last_time  = before_df["timestamp"].iloc[-1]
    first_time = after_df["timestamp"].iloc[0]
    n_steps = max(1, int(round((first_time - last_time).total_seconds() / dt)))

    before_c = before_df.drop_duplicates(
        subset=["latitude", "longitude"], keep="last").reset_index(drop=True)

    # ── Constant KF matrices ──────────────────────────────────────────────────
    F = np.eye(6)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt

    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    Q = np.diag([5e-9, 5e-9, 0.5, 5e-11, 5e-11, 5e-3])
    R = np.diag([8e-8, 8e-8, 50.0])

    def _kf_update(x, P, z):
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.solve(S, np.eye(3)).T
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P
        return x, P

    # ── Warm up on last 15 ADS-B points before the gap ───────────────────────
    pts = before_c.tail(min(15, len(before_c))).reset_index(drop=True)
    x = np.array([
        pts["latitude"].iloc[0], pts["longitude"].iloc[0],
        pts["altitude"].iloc[0], 0., 0., 0.
    ], dtype=float)
    P = np.diag([1e-7, 1e-7, 1000., 1e-10, 1e-10, 1.0])

    prev_t = None
    for _, row in pts.iterrows():
        if prev_t is not None:
            dt_s = max(abs((row["timestamp"] - prev_t).total_seconds()), 1.0)
            Fs = np.eye(6); Fs[0, 3] = dt_s; Fs[1, 4] = dt_s; Fs[2, 5] = dt_s
            x = Fs @ x; P = Fs @ P @ Fs.T + Q
        z = np.array([row["latitude"], row["longitude"], row["altitude"]])
        x, P = _kf_update(x, P, z)
        prev_t = row["timestamp"]

    # ── Forward predict across the gap ────────────────────────────────────────
    # ── Override velocity to point directly toward after endpoint ─────────────
    # This prevents drift caused by the warmup velocity being slightly off-bearing
    total_sec = (first_time - last_time).total_seconds()
    lat0 = float(before_c["latitude"].iloc[-1])
    lon0 = float(before_c["longitude"].iloc[-1])
    lat1 = float(after_df["latitude"].iloc[0])
    lon1 = float(after_df["longitude"].iloc[0])

    dlat = (lat1 - lat0) / total_sec
    dlon = (lon1 - lon0) / total_sec
    dalt = (float(after_df["altitude"].iloc[0]) - float(before_c["altitude"].iloc[-1])) / total_sec

    x[3] = dlat
    x[4] = dlon
    x[5] = dalt

    # ── Forward predict across the gap ────────────────────────────────────────
    preds = []
    for _ in range(n_steps):
        x = F @ x; P = F @ P @ F.T + Q
        preds.append(x[:3].copy())

    blended = np.array(preds)

    # ── Retime to constant speed ──────────────────────────────────────────────
    # Kalman velocity may drift so we retime the path to constant speed
    # so positions at each timestamp are physically meaningful
    cum_dist = np.zeros(n_steps)
    for i in range(1, n_steps):
        _, _, d = geod.inv(float(blended[i-1, 1]), float(blended[i-1, 0]),
                           float(blended[i,   1]), float(blended[i,   0]))
        cum_dist[i] = cum_dist[i-1] + abs(d)

    total_dist = cum_dist[-1]
    total_sec  = (first_time - last_time).total_seconds()

    if total_dist > 0:
        old_fracs  = cum_dist / total_dist
        time_fracs = np.array([dt*(i+1)/total_sec for i in range(n_steps)])
        time_fracs = np.clip(time_fracs, 0.0, 1.0)

        f_la  = _i1d(old_fracs, blended[:, 0],
                     bounds_error=False, fill_value=(blended[0,0], blended[-1,0]))
        f_lo  = _i1d(old_fracs, blended[:, 1],
                     bounds_error=False, fill_value=(blended[0,1], blended[-1,1]))
        f_alt = _i1d(old_fracs, blended[:, 2],
                     bounds_error=False, fill_value=(blended[0,2], blended[-1,2]))

        new_lats = f_la(time_fracs)
        new_lons = f_lo(time_fracs)
        new_alts = f_alt(time_fracs)
    else:
        new_lats = blended[:, 0]
        new_lons = blended[:, 1]
        new_alts = blended[:, 2]

    timestamps = [last_time + pd.Timedelta(seconds=dt*(i+1)) for i in range(n_steps)]
    result = pd.DataFrame({
        "latitude":     new_lats,
        "longitude":    new_lons,
        "altitude":     new_alts,
        "timestamp":    timestamps,
        "interpolated": True,
    })
    return result

def compute_features(df, dt=15.0):
    """
    Takes a DataFrame with latitude, longitude, altitude, timestamp, interpolated
    and computes vx, vy, acceleration, turn_rate.
    Returns DataFrame with all FEATURE_COLS.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    lat = df["latitude"].values
    lon = df["longitude"].values
    alt = df["altitude"].values

    n = len(df)
    vx = np.zeros(n)
    vy = np.zeros(n)
    acceleration = np.zeros(n)
    turn_rate = np.zeros(n)

    for i in range(1, n):
        lat1, lon1 = np.radians(lat[i-1]), np.radians(lon[i-1])
        lat2, lon2 = np.radians(lat[i]), np.radians(lon[i])

        # Haversine distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        dist = 2 * 6371000 * np.arcsin(np.sqrt(a))

        # Bearing
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        heading = np.degrees(np.arctan2(x, y)) % 360

        # Time delta
        t_delta = (df["timestamp"].iloc[i] - df["timestamp"].iloc[i-1]).total_seconds()
        if t_delta == 0:
            t_delta = dt

        velocity = dist / t_delta
        heading_rad = np.radians(heading)
        vx[i] = velocity * np.sin(heading_rad)
        vy[i] = velocity * np.cos(heading_rad)

        if i >= 2:
            prev_dist = 2 * 6371000 * np.arcsin(np.sqrt(
                np.sin((lat1 - np.radians(lat[i-2]))/2)**2 +
                np.cos(np.radians(lat[i-2]))*np.cos(lat1)*np.sin((lon1 - np.radians(lon[i-2]))/2)**2
            ))
            prev_t = (df["timestamp"].iloc[i-1] - df["timestamp"].iloc[i-2]).total_seconds()
            if prev_t == 0:
                prev_t = dt
            prev_velocity = prev_dist / prev_t

            prev_dlon = lon1 - np.radians(lon[i-2])
            prev_x = np.sin(prev_dlon) * np.cos(lat1)
            prev_y = np.cos(np.radians(lat[i-2]))*np.sin(lat1) - np.sin(np.radians(lat[i-2]))*np.cos(lat1)*np.cos(prev_dlon)
            prev_heading = np.degrees(np.arctan2(prev_x, prev_y)) % 360

            acceleration[i] = (velocity - prev_velocity) / t_delta
            turn_rate[i] = (heading - prev_heading) / t_delta

    # First row copies from second
    vx[0] = vx[1] if n > 1 else 0
    vy[0] = vy[1] if n > 1 else 0

    df = df.copy()
    df["vx"] = vx
    df["vy"] = vy
    df["acceleration"] = acceleration
    df["turn_rate"] = turn_rate

    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("day", type=str)
    parser.add_argument("flight_name", type=str)
    args = parser.parse_args()

    day = args.day
    flight_name = args.flight_name


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryBiGRU(
        input_size=len(FEATURE_COLS),   
        hidden_size=128,
        num_layers=2,
        output_size=len(TARGET_COLS),  
        dropout=0.3
    ).to(device)
    model.load_state_dict(torch.load("models\\BiGRU.pth", map_location=device))
    model.eval()
    
    df_before = pd.read_parquet(rf"{INPUT_FOLDER}\{day}\{flight_name}\adsb_before.parquet")
    df_ads_c = pd.read_parquet(rf"{INPUT_FOLDER}\{day}\{flight_name}\adsc.parquet")
    df_after = pd.read_parquet(rf"{INPUT_FOLDER}\{day}\{flight_name}\adsb_after.parquet")
    # Gap 1: before → ads_c
    gap1 = reconstruct_gap(model, df_before, df_ads_c, FEATURE_COLS, TARGET_COLS, SEQUENCE_LEN, device)
    gap1 = compute_features(gap1)


    # Gap 2: ads_c → after
    gap2 = reconstruct_gap(model, df_ads_c, df_after, FEATURE_COLS, TARGET_COLS, SEQUENCE_LEN, device)
    gap2 = compute_features(gap2)

    # Merge
    df_before["interpolated"] = False
    df_ads_c["interpolated"] = False
    df_after["interpolated"] = False

    cols = ["timestamp", "latitude", "longitude", "altitude",
            "vx", "vy", "acceleration", "turn_rate", "interpolated"]

    # Gap 3 before → after is not needed since we have ads_c in the middle, but if we wanted to do it:
    gap3 = reconstruct_gap(model, df_before, df_after, FEATURE_COLS, TARGET_COLS, SEQUENCE_LEN, device)
    gap3 = compute_features(gap3)
    full = pd.concat([
        df_before[cols],
        gap1[cols],
        df_ads_c[cols],
        gap2[cols],
        df_after[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    ads_b_ads_b = pd.concat([
        df_before[cols],
        gap3[cols],
        df_after[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    ads_b_ads_c = pd.concat([
        df_before[cols],
        gap1[cols],
        df_ads_c[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    ads_c_ads_b = pd.concat([
        df_ads_c[cols],
        gap2[cols],
        df_after[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    output_folder = "final_reconstructions"
    os.makedirs(output_folder, exist_ok=True)
    cur_ouptput_folder = os.path.join(output_folder, day, flight_name)
    os.makedirs(cur_ouptput_folder, exist_ok=True)

    # ── BiGRU outputs ────────────────────────────────────────────────────────
    ads_b_ads_c.to_parquet(f"{cur_ouptput_folder}\\adsb_before-adsc.parquet")
    ads_c_ads_b.to_parquet(f"{cur_ouptput_folder}\\adsc-adsb_after.parquet")
    ads_b_ads_b.to_parquet(f"{cur_ouptput_folder}\\adsb_before-adsb_after.parquet")
    full.to_parquet(f"{cur_ouptput_folder}\\full_reconstruction.parquet")

    # ── Baseline (great-circle dead-reckoning) ───────────────────────────────
    bl_gap1 = compute_features(reconstruct_gap_baseline(df_before, df_ads_c))
    bl_gap2 = compute_features(reconstruct_gap_baseline(df_ads_c,  df_after))
    baseline_full = pd.concat([
        df_before[cols], bl_gap1[cols], df_ads_c[cols], bl_gap2[cols], df_after[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    baseline_full.to_parquet(f"{cur_ouptput_folder}\\baseline_full_reconstruction.parquet")

    # ── Kalman filter ────────────────────────────────────────────────────────
    kf_gap1 = compute_features(reconstruct_gap_kalman(df_before, df_ads_c))
    kf_gap2 = compute_features(reconstruct_gap_kalman(df_ads_c,  df_after))
    kalman_full = pd.concat([
        df_before[cols], kf_gap1[cols], df_ads_c[cols], kf_gap2[cols], df_after[cols]
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    kalman_full.to_parquet(f"{cur_ouptput_folder}\\kalman_full_reconstruction.parquet")

    print("Reconstruction complete and saved to:", cur_ouptput_folder)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print(fr"you probably did not run the cleaning. run etl.ipynb first to clean the data and then run this reconstruction script with python .\reconstruction.py step flight_name ")