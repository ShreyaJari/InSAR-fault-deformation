"""
Feature engineering for the aftershock forecaster (v2).

Reframed as binary classification following DeVries et al. (2018, Nature):
predict whether a grid cell experiences >=1 aftershock in a given time
window, rather than raw counts (too sparse at fine grid resolution for
this catalog size).

Usage:
    python3 scripts/build_features.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

DATA_DIR = Path("data")
IN_CSV = DATA_DIR / "kahramanmaras_aftershocks_30d.csv"
OUT_CSV = DATA_DIR / "aftershock_grid_features.csv"

MAINSHOCK_LON = 37.032
MAINSHOCK_LAT = 37.166

# Coarser grid: ~0.1 deg =~ 11km at this latitude
GRID_RES = 0.1

# Study region bounds (matches the catalog bbox from fetch_aftershocks.py)
MIN_LON, MAX_LON = 35.5, 39.5
MIN_LAT, MAX_LAT = 35.5, 39.0

TIME_BINS = [0, 1, 3, 7, 14, 30]
TIME_LABELS = ["0-1d", "1-3d", "3-7d", "7-14d", "14-30d"]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def build_features():
    df = pd.read_csv(IN_CSV, parse_dates=["time"])

    df["dist_from_mainshock_km"] = haversine_km(
        df["longitude"], df["latitude"], MAINSHOCK_LON, MAINSHOCK_LAT
    )
    df["days_since_mainshock"] = df["hours_since_mainshock"] / 24
    df["time_bin"] = pd.cut(
        df["days_since_mainshock"], bins=TIME_BINS, labels=TIME_LABELS,
        include_lowest=True
    )
    df["grid_lon"] = (df["longitude"] / GRID_RES).round() * GRID_RES
    df["grid_lat"] = (df["latitude"] / GRID_RES).round() * GRID_RES

    # Positive examples: cells that DID have >=1 event in that time bin
    positives = (
        df.groupby(["grid_lon", "grid_lat", "time_bin"], observed=True)
        .agg(
            event_count=("usgs_id", "count"),
            max_magnitude=("magnitude", "max"),
        )
        .reset_index()
    )
    positives["had_aftershock"] = 1

    # Build the FULL grid x time-bin space (including cells with NO events)
    # so the model sees negative examples too - this is what makes binary
    # classification learnable.
    lon_vals = np.arange(MIN_LON, MAX_LON, GRID_RES).round(2)
    lat_vals = np.arange(MIN_LAT, MAX_LAT, GRID_RES).round(2)
    full_grid = pd.DataFrame(
        list(product(lon_vals, lat_vals, TIME_LABELS)),
        columns=["grid_lon", "grid_lat", "time_bin"]
    )

    merged = full_grid.merge(
        positives, on=["grid_lon", "grid_lat", "time_bin"], how="left"
    )
    merged["had_aftershock"] = merged["had_aftershock"].fillna(0).astype(int)
    merged["event_count"] = merged["event_count"].fillna(0).astype(int)
    merged["max_magnitude"] = merged["max_magnitude"].fillna(0)

    # Distance from mainshock (cell center) - a feature every cell has,
    # regardless of whether it had an aftershock.
    merged["dist_from_mainshock_km"] = haversine_km(
        merged["grid_lon"], merged["grid_lat"], MAINSHOCK_LON, MAINSHOCK_LAT
    )

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(merged)} grid-cell x time-bin rows to {OUT_CSV}")
    print(f"Positive rate: {merged['had_aftershock'].mean():.1%}")
    print(f"({merged['had_aftershock'].sum()} positive / "
          f"{len(merged) - merged['had_aftershock'].sum()} negative)")
    return merged


if __name__ == "__main__":
    build_features()