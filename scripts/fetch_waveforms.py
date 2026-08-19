"""
Fetch seismic waveform data near the 2023 Kahramanmaras mainshock via
ObsPy's IRIS FDSN client, for the Day 3 denoising autoencoder.

Searches for the nearest available broadband stations within a radius
of the epicenter, then pulls a waveform segment bracketing the
mainshock origin time for each.

Usage:
    python3 scripts/fetch_waveforms.py
"""
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from pathlib import Path

OUT_DIR = Path("data/waveforms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAINSHOCK_TIME = UTCDateTime("2023-02-06T01:17:35")
MAINSHOCK_LAT = 37.166
MAINSHOCK_LON = 37.032

SEARCH_RADIUS_DEG = 8.0  # ~890km - widened further; many nearby stations
                          # return HTTP 204 (metadata exists, no archived
                          # data for this window), so cast a wider net
MAX_CANDIDATES = 20      # try more candidates than we need, since a lot
                          # will fail with "no data available"
TARGET_STATIONS = 5      # stop once we have this many successful pulls

# Window: 2 min before origin time to 8 min after - captures P/S arrival
# and coda for a range of distances within the search radius.
WINDOW_BEFORE = 120
WINDOW_AFTER = 480

CLIENT = Client("EARTHSCOPE")  # IRIS was renamed EarthScope; same service


def find_stations():
    print(f"Searching for broadband stations within {SEARCH_RADIUS_DEG} deg "
          f"of ({MAINSHOCK_LAT}, {MAINSHOCK_LON})...")
    inventory = CLIENT.get_stations(
        latitude=MAINSHOCK_LAT,
        longitude=MAINSHOCK_LON,
        maxradius=SEARCH_RADIUS_DEG,
        channel="BHZ",  # broadband, vertical component - simplest starting point
        level="station",
        starttime=MAINSHOCK_TIME - 86400,
        endtime=MAINSHOCK_TIME + 86400,
    )
    stations = []
    for net in inventory:
        for sta in net:
            stations.append((net.code, sta.code, sta.latitude, sta.longitude))
    print(f"Found {len(stations)} candidate stations")
    return stations[:MAX_CANDIDATES]


def fetch_waveforms(stations):
    saved = []
    for net_code, sta_code, lat, lon in stations:
        if len(saved) >= TARGET_STATIONS:
            print(f"Reached target of {TARGET_STATIONS} stations, stopping.")
            break
        try:
            st = CLIENT.get_waveforms(
                network=net_code,
                station=sta_code,
                location="*",
                channel="BHZ",
                starttime=MAINSHOCK_TIME - WINDOW_BEFORE,
                endtime=MAINSHOCK_TIME + WINDOW_AFTER,
            )
            if len(st) == 0:
                print(f"  {net_code}.{sta_code}: no data returned, skipping")
                continue
            out_path = OUT_DIR / f"{net_code}_{sta_code}_BHZ.mseed"
            st.write(str(out_path), format="MSEED")
            print(f"  {net_code}.{sta_code}: saved {len(st)} trace(s) to {out_path}")
            saved.append(out_path)
        except Exception as e:
            print(f"  {net_code}.{sta_code}: FAILED ({e})")
    return saved


if __name__ == "__main__":
    stations = find_stations()
    if not stations:
        print("No stations found - try increasing SEARCH_RADIUS_DEG")
    else:
        saved = fetch_waveforms(stations)
        print(f"\nSaved {len(saved)} waveform file(s) to {OUT_DIR}")