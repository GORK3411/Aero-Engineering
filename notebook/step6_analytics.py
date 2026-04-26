"""
step6_analytics.py — Route distance, deviation and CO2 emissions analytics
===========================================================================
Backing module for 06_analytics.ipynb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Geod

geod = Geod(ellps="WGS84")
CO2_KG_PER_KM = 19.0  # ICAO widebody: ~6 kg fuel/km x 3.16 CO2 factor

def path_length_km(df, lat_col="latitude", lon_col="longitude"):
    lats = df[lat_col].values; lons = df[lon_col].values
    finite = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[finite]; lons = lons[finite]
    if len(lats) < 2: return 0.0
    _, _, dists = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return float(np.nansum(np.abs(dists))) / 1000

def max_deviation_km(recon_df, start_lat, start_lon, end_lat, end_lon):
    """Maximum lateral deviation from the great-circle reference path (km)."""
    if len(recon_df) == 0: return 0.0
    brg_ref = np.radians(
        (np.degrees(np.arctan2(
            np.sin(np.radians(end_lon-start_lon))*np.cos(np.radians(end_lat)),
            np.cos(np.radians(start_lat))*np.sin(np.radians(end_lat)) -
            np.sin(np.radians(start_lat))*np.cos(np.radians(end_lat))*
            np.cos(np.radians(end_lon-start_lon))
        ))+360)%360)
    R = 6_371_000.0; deviations = []
    for _, row in recon_df.iterrows():
        la1,lo1=np.radians(start_lat),np.radians(start_lon)
        la2,lo2=np.radians(row["latitude"]),np.radians(row["longitude"])
        dlat,dlon=la2-la1,lo2-lo1
        a=np.sin(dlat/2)**2+np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
        d13=2*np.arcsin(np.sqrt(np.clip(a,0,1)))
        brg13=np.arctan2(np.sin(dlon)*np.cos(la2),
                         np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(dlon))
        xte=np.arcsin(np.clip(np.sin(d13)*np.sin(brg13-brg_ref),-1,1))*R
        deviations.append(abs(xte))
    return float(np.max(deviations))/1000 if deviations else 0.0

def print_analytics_summary(df, methods=None):
    if methods is None: methods=["baseline","kalman","bigru"]
    print(f"Flights analysed : {len(df)}")
    print(f"Avg gap duration : {df['gap_minutes'].mean():.1f} min")
    print(f"Avg gap distance : {df['gc_gap_km'].mean():.0f} km")
    print("\n── Route Distance ──")
    raw_mean=df["raw_adsb_dist_km"].mean()
    print(f"  {'Raw ADS-B':<15} {raw_mean:.0f} km  (missing oceanic gap)")
    for m in methods:
        col=f"{m}_dist_km"
        if col in df:
            val=df[col].dropna().mean()
            print(f"  {m.capitalize():<15} {val:.0f} km  (+{val-raw_mean:.0f} km vs raw)")
    if "adsc_true_dist_km" in df and df["adsc_true_dist_km"].notna().sum()>0:
        val=df["adsc_true_dist_km"].dropna().mean()
        print(f"  {'ADS-C truth':<15} {val:.0f} km")
    print("\n── CO2 Proxy ──")
    print(f"  {'Raw ADS-B':<15} {df['raw_co2_kg'].mean():.0f} kg")
    for m in methods:
        col=f"{m}_co2_kg"
        if col in df:
            val=df[col].dropna().mean()
            raw=df["raw_co2_kg"].mean()
            print(f"  {m.capitalize():<15} {val:.0f} kg  (+{val-raw:.0f} kg vs raw)")