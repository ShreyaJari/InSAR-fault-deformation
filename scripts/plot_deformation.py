
"""
Extract and plot InSAR deformation results for the 2023 Kahramanmaras
earthquake sequence, from LiCSBAS output (cum_filt.h5).

Usage:
    python3 scripts/plot_deformation.py
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

TS_DIR = Path("LiCSBAS/TS_GEOCml10mask")
CUM_FILE = TS_DIR / "cum_filt.h5"
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Points of interest: (x, y) pixel coords, label
# (from manual GUI inspection - update if you re-derive these)
POINTS = [
    (138, 237, "East block (brown zone)"),
    (165, 209, "East block, far from fault"),
    (145, 271, "West block (blue zone)"),
    (162, 307, "West block, far from fault"),
]

EARTHQUAKE_DATE = datetime(2023, 2, 6)


def load_cum(path):
    with h5py.File(path, "r") as f:
        cum = f["cum"][:]              # (n_im, length, width), mm
        vel = f["vel"][:]              # (length, width), mm/yr
        imdates = [d.decode() if isinstance(d, bytes) else str(d)
                   for d in f["imdates"][:]]
    dates = [datetime.strptime(d, "%Y%m%d") for d in imdates]
    return cum, vel, dates


def plot_velocity_map(vel, out_path):
    fig, ax = plt.subplots(figsize=(7, 8))
    vmax = np.nanpercentile(np.abs(vel), 98)
    im = ax.imshow(vel, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("LOS Velocity (mm/yr) - 116A_05207_252525\n"
                  "Kahramanmaras Earthquake Sequence, Jan-May 2023")
    fig.colorbar(im, ax=ax, label="mm/yr")
    for x, y, label in POINTS:
        ax.plot(x, y, "k+", markersize=10, markeredgewidth=2)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_time_series(cum, dates, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for x, y, label in POINTS:
        series = cum[:, y, x]
        ax.plot(dates, series, marker="o", label=f"{label} ({x},{y})")
    ax.axvline(EARTHQUAKE_DATE, color="red", linestyle="--",
               label="Mw 7.8 mainshock (6 Feb 2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative displacement (mm)")
    ax.set_title("LOS Displacement Time Series Across the Rupture")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    cum, vel, dates = load_cum(CUM_FILE)
    plot_velocity_map(vel, OUT_DIR / "velocity_map.png")
    plot_time_series(cum, dates, OUT_DIR / "time_series_comparison.png")