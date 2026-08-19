"""
Fetch the aftershock catalog for the 2023 Kahramanmaras earthquake
sequence from the USGS FDSNWS event API.

Mainshock: Mw 7.8, 2023-02-06 01:17:35 UTC, near Pazarcik, Turkey.
Window: mainshock to +30 days.
Region: bounding box around the East Anatolian Fault rupture zone.

Usage:
    python3 scripts/fetch_aftershocks.py
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "kahramanmaras_aftershocks_30d.csv"

MAINSHOCK_TIME = datetime(2023, 2, 6, 1, 17, 35)
WINDOW_DAYS = 30

# Bounding box loosely covering the East Anatolian + Cardak Fault rupture
# (~345km + ~175km), with margin. lon/lat.
MIN_LON, MAX_LON = 35.5, 39.5
MIN_LAT, MAX_LAT = 35.5, 39.0

MIN_MAGNITUDE = 2.5  # catalog completeness threshold - adjust if needed

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def fetch_catalog():
    start = MAINSHOCK_TIME.strftime("%Y-%m-%dT%H:%M:%S")
    end = (MAINSHOCK_TIME + timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minlatitude": MIN_LAT,
        "maxlatitude": MAX_LAT,
        "minlongitude": MIN_LON,
        "maxlongitude": MAX_LON,
        "minmagnitude": MIN_MAGNITUDE,
        "orderby": "time",
    }

    print(f"Querying USGS: {start} to {end}, bbox ({MIN_LON},{MIN_LAT})-({MAX_LON},{MAX_LAT})")
    resp = requests.get(USGS_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    features = data["features"]
    print(f"Retrieved {len(features)} events")

    rows = []
    for f in features:
        props = f["properties"]
        lon, lat, depth = f["geometry"]["coordinates"]
        rows.append({
            "time": datetime.utcfromtimestamp(props["time"] / 1000),
            "magnitude": props["mag"],
            "mag_type": props["magType"],
            "longitude": lon,
            "latitude": lat,
            "depth_km": depth,
            "place": props["place"],
            "usgs_id": f["id"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)

    # Time since mainshock, in hours - a key feature for Day 2 modeling
    df["hours_since_mainshock"] = (
        df["time"] - MAINSHOCK_TIME
    ).dt.total_seconds() / 3600

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} events to {OUT_CSV}")
    print(f"Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
    return df


if __name__ == "__main__":
    fetch_catalog()