"""Re-plot the 95% DACP forecasts from saved CSVs.

Use this AFTER dacp_arima.py has finished, to iterate on plot styling without
re-running the multi-hour experiment. Reads the per-station DACP CSVs
(results/dacp/DACP_<station>_95%.csv) and writes new PNGs to plots/dacp/.

Window rule (applies to every panel):
    Each panel shows a WINDOW_SPAN-day window centred on the largest observed
    peak in that station's true_value series. Windows differ across stations
    because peak timing varies. This is the rule referenced in the paper
    caption -- do not override individual stations without updating the
    caption to match.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

RESULTS_DIR = "results/dacp"
PLOT_DIR = "plots/dacp"

STATIONS = [
    "Swift_Creek",
    "Bunchgrass_Meadow",
    "Touchet",
    "Paradise",
    "Easy_Pass",
]

# Full window span in days. The window is centred on each station's largest
# observed peak. Change once here to apply uniformly; do not vary per station.
WINDOW_SPAN = 200

# Hand override per station. None = auto-pick by the rule above. Only set a
# (start, end) tuple if you have a documented reason -- and then update the
# paper caption so it no longer claims a uniform rule.
PLOT_WINDOWS = {
    "Swift_Creek":       None,
    "Bunchgrass_Meadow": None,
    "Touchet":           None,
    "Paradise":          None,
    "Easy_Pass":         None,
}

# Distinct truth / forecast colours so the lines never visually merge, and a
# muted warm interval band that sits behind the data.
COLOR_TRUTH    = "#1A1A1A"   # near-black, solid, on top
COLOR_FORECAST = "#0072B2"   # blue, dashed, behind truth
COLOR_BAND     = "#E8A33D"   # warm orange, low-alpha


def _set_style():
    rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "figure.dpi": 150,
    })


def _auto_window(df, span=WINDOW_SPAN):
    """`span`-day window centred on the largest true_value peak, clipped to
    the available step range so we don't fall off the ends of the series."""
    peak_step = int(df.loc[df["true_value"].idxmax(), "step"])
    half = span // 2
    return (max(int(df["step"].min()), peak_step - half),
            min(int(df["step"].max()), peak_step + half))


def plot_one(stem, override):
    csv_path = os.path.join(RESULTS_DIR, f"DACP_{stem}_95%.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP {stem}: no CSV at {csv_path}")
        return

    df_full = pd.read_csv(csv_path)
    window = override if override is not None else _auto_window(df_full)
    start, end = window
    df = df_full[(df_full["step"] >= start) & (df_full["step"] <= end)]
    if df.empty:
        print(f"  WARN {stem}: window {window} captured no rows")
        return

    station = stem.replace("_", " ")
    out_path = os.path.join(
        PLOT_DIR, f"forecast_{stem}_95_window_{start}_{end}.png")

    _set_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    # Band first (behind), then forecast (dashed, behind truth), then truth
    # (solid, on top). This ordering keeps both lines visible even on stations
    # where the forecast nearly equals the observation.
    ax.fill_between(
        df["step"], df["lower_bound"], df["upper_bound"],
        color=COLOR_BAND, alpha=0.25, linewidth=0,
        label="95% interval",
    )
    ax.plot(df["step"], df["prediction"],
            color=COLOR_FORECAST, lw=0.9, ls=(0, (4, 2)),
            label="Forecast (ARIMA)")
    ax.plot(df["step"], df["true_value"],
            color=COLOR_TRUTH, lw=1.0, label="Observed")

    # Y-axis scaled to the OBSERVED signal -- a freshet band can balloon well
    # past the truth and would otherwise squash the baseflow detail. Extra
    # headroom is added on top so the legend sits above the data rather than
    # overlapping it. The band may visually clip at extreme peaks; that's
    # honest and noted in the caption.
    truth = df["true_value"].values
    upper = df["upper_bound"].values
    obs_lo, obs_hi = float(np.min(truth)), float(np.max(truth))
    span_y = obs_hi - obs_lo if obs_hi > obs_lo else 1.0
    y_lo = max(0.0, obs_lo - 0.08 * span_y)
    band_hi = float(np.max(upper))
    y_hi = min(band_hi, obs_hi + 0.35 * span_y) + 0.05 * span_y
    # extra headroom for the legend (~20% of the visible range)
    y_hi += 0.20 * (y_hi - y_lo)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Day")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title(station, fontsize=9, pad=4)
    ax.legend(fontsize=6.5, loc="upper right", ncol=3,
              handlelength=1.4, columnspacing=1.0)
    ax.margins(x=0.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    peak_step = int(df_full.loc[df_full["true_value"].idxmax(), "step"])
    print(f"  {station}: peak at step {peak_step}, "
          f"window ({start}, {end}), {len(df)} rows -> {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("Re-plotting DACP 95% forecasts from saved CSVs ...\n")
    for stem in STATIONS:
        plot_one(stem, PLOT_WINDOWS.get(stem))
    print("\nDone.")


if __name__ == "__main__":
    main()