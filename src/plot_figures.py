"""
Overlay comparison plots for NAPS 2026 paper.

Two figures (Bunchgrass_Meadow, Touchet), each showing all three conformal
methods on the same axes at the 95% target. Window is centred on the largest
observed peak in the DACP series for that station.

Methods compared:
  - DACP: dual adaptive (k and t adapt online)
  - Weighted conformal: exponentially-weighted quantile, no adaptation
  - Split conformal: unweighted quantile from rolling calibration block
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"D:\desk top\NAPS\results\dacp")
OUT_DIR  = Path(r"D:\desk top\NAPS\results\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["DACP", "weighted_conformal", "split_conformal"]

METHOD_LABELS = {
    "DACP":               "DACP",
    "weighted_conformal": "Weighted Conformal",
    "split_conformal":    "Split Conformal",
}

# Band fill colors — distinct, low alpha so they layer visibly
BAND_COLORS = {
    "DACP":               "#2166ac",   # blue
    "weighted_conformal": "#E8A33D",   # warm orange
    "split_conformal":    "#4dac26",   # green
}
BAND_ALPHA = {
    "DACP":               0.30,
    "weighted_conformal": 0.22,
    "split_conformal":    0.18,
}
# Edge lines for the interval bounds (same hue, slightly darker)
EDGE_COLORS = {
    "DACP":               "#1a4f8a",
    "weighted_conformal": "#b87820",
    "split_conformal":    "#2e7a18",
}

COLOR_TRUTH    = "#1A1A1A"
COLOR_FORECAST = "#0072B2"

WINDOW_SPAN = 200


def set_style():
    rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.linewidth":    0.6,
        "axes.edgecolor":    "#444444",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon":    False,
        "figure.dpi":        150,
    })


def load_csv(method, station, level):
    fname = DATA_DIR / f"{method}_{station}_{level}%.csv"
    if not fname.exists():
        print(f"  WARNING: missing {fname.name}")
        return None
    return pd.read_csv(fname)


def empirical_coverage(df):
    return ((df["true_value"] >= df["lower_bound"]) &
            (df["true_value"] <= df["upper_bound"])).mean()


def auto_window(df, span=WINDOW_SPAN):
    peak_step = int(df.loc[df["true_value"].idxmax(), "step"])
    half = span // 2
    return (max(int(df["step"].min()), peak_step - half),
            min(int(df["step"].max()), peak_step + half))


def plot_overlay(station, level=95):
    print(f"Building overlay plot: {station} @ {level}%...")
    set_style()

    # Anchor window on DACP peak
    df_anchor = load_csv("DACP", station, level)
    if df_anchor is None:
        print(f"  SKIP {station}: DACP CSV missing")
        return
    start, end = auto_window(df_anchor)

    fig, ax = plt.subplots(figsize=(7, 3.8))

    # Draw bands back-to-front: split conformal (typically widest) first,
    # weighted next, DACP on top so its band is fully visible.
    draw_order = ["split_conformal", "weighted_conformal", "DACP"]

    for method in draw_order:
        df_full = load_csv(method, station, level)
        if df_full is None:
            continue
        df = df_full[(df_full["step"] >= start) & (df_full["step"] <= end)]
        cov = empirical_coverage(df_full)

        ax.fill_between(
            df["step"], df["lower_bound"], df["upper_bound"],
            color=BAND_COLORS[method], alpha=BAND_ALPHA[method],
            linewidth=0,
            label=f"{METHOD_LABELS[method]}  (cov={cov:.3f})"
        )
        # Thin edge lines so interval boundaries are visible when bands overlap
        ax.plot(df["step"], df["upper_bound"],
                color=EDGE_COLORS[method], lw=0.6, alpha=0.6)
        ax.plot(df["step"], df["lower_bound"],
                color=EDGE_COLORS[method], lw=0.6, alpha=0.6)

    # ARIMA forecast (from DACP file — same base model for all methods)
    df_dacp = load_csv("DACP", station, level)
    df_dacp = df_dacp[(df_dacp["step"] >= start) & (df_dacp["step"] <= end)]
    ax.plot(df_dacp["step"], df_dacp["prediction"],
            color=COLOR_FORECAST, lw=0.9, ls=(0, (4, 2)),
            label="Forecast (ARIMA)", zorder=4)

    # Observed on top
    ax.plot(df_dacp["step"], df_dacp["true_value"],
            color=COLOR_TRUTH, lw=1.0, label="Observed", zorder=5)

    # Y-axis: scale to observed + headroom across all three methods'
    # upper bounds (not just split conformal, since weighted can sometimes
    # be wider than split at low targets)
    truth  = df_dacp["true_value"].values
    obs_lo, obs_hi = float(np.min(truth)), float(np.max(truth))
    span_y = obs_hi - obs_lo if obs_hi > obs_lo else 1.0
    y_lo   = max(0.0, obs_lo - 0.08 * span_y)

    # Find the widest upper bound across all three methods in the window
    all_uppers = []
    for method in METHODS:
        df_m = load_csv(method, station, level)
        if df_m is None:
            continue
        df_m_w = df_m[(df_m["step"] >= start) & (df_m["step"] <= end)]
        all_uppers.append(float(df_m_w["upper_bound"].max()))
    band_hi = max(all_uppers) if all_uppers else obs_hi

    y_hi  = min(band_hi, obs_hi + 0.40 * span_y) + 0.05 * span_y
    y_hi += 0.22 * (y_hi - y_lo)   # headroom for legend
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Discharge (cfs)", fontsize=9)
    station_label = station.replace("_", " ")
    ax.set_title(
        f"{station_label} — {level}% prediction intervals: all three methods",
        fontsize=10, pad=5
    )
    ax.margins(x=0.01)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1,
              handlelength=1.6, borderaxespad=0.4)

    fig.tight_layout(pad=0.5)

    safe = station.lower()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT_DIR / f"fig_overlay_{safe}_{level}pct.{ext}",
                    dpi=200, bbox_inches="tight")
    print(f"  Saved: fig_overlay_{safe}_{level}pct  (window {start}–{end})")
    plt.close(fig)


if __name__ == "__main__":
    plot_overlay("Bunchgrass_Meadow", level=95)
    plot_overlay("Touchet",           level=95)
    print(f"\nAll figures saved to: {OUT_DIR}")