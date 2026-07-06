"""Two-panel summary figure: calibration error and interval width vs target.

Reads results/dacp/summary_all_methods.csv and produces a paper-quality figure
showing how each method's calibration error (left) and average interval width
(right) scale with the nominal coverage target. Across the 5 stations the
panels show the mean as a bold line; the shaded band shows +/- 1 std (for
calibration error, left panel) or the min-max range (for width, right panel
-- std is not meaningful when widths span two orders of magnitude across
stations).

This is the headline summary figure -- the two main claims (DACP is
better-calibrated AND narrower than split_conformal) in one figure.

Run AFTER dacp_arima.py finishes. Runs in a few seconds.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

RESULTS_DIR = "results/dacp"
PLOT_DIR = "plots/dacp"

METHODS_ORDER = ["DACP", "single_aci", "split_conformal"]
METHOD_COLORS = {
    "DACP": "#0072B2",
    "single_aci": "#D55E00",
    "split_conformal": "#009E73",
}
METHOD_LABELS = {
    "DACP": "DACP",
    "single_aci": "single ACI",
    "split_conformal": "split conformal",
}

# Line style + marker per method. single_aci is dashed-triangle so it visually
# reads as an ablation/contrast even in greyscale prints.
METHOD_STYLES = {
    "DACP":            {"linestyle": "-",  "marker": "o"},
    "single_aci":      {"linestyle": "--", "marker": "^"},
    "split_conformal": {"linestyle": "-",  "marker": "o"},
}

TARGET_NUMERIC = {"70%": 70, "80%": 80, "90%": 90, "95%": 95}


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


def _agg(df, method, target, col, fn):
    sub = df[(df["method"] == method) & (df["target_num"] == target)][col]
    if sub.empty:
        return np.nan
    return fn(sub)


def _plot_calibration_error(ax, df, targets):
    for m in METHODS_ORDER:
        means = np.array([_agg(df, m, t, "CalibrationError", np.mean)
                          for t in targets])
        stds = np.array([_agg(df, m, t, "CalibrationError", np.std)
                         for t in targets])
        color = METHOD_COLORS[m]
        style = METHOD_STYLES[m]
        ax.plot(targets, means, ms=4, lw=1.4, color=color,
                linestyle=style["linestyle"], marker=style["marker"],
                label=METHOD_LABELS[m], zorder=3)
        ax.fill_between(targets, np.maximum(means - stds, 0), means + stds,
                        color=color, alpha=0.15, linewidth=0, zorder=2)

    ax.set_xlabel("Nominal target (%)")
    ax.set_ylabel("Calibration error")
    ax.set_xticks(targets)
    ax.set_ylim(bottom=0)
    ax.set_title("(a) Calibration error", fontsize=10, pad=4)
    ax.legend(loc="upper right", fontsize=8, handlelength=1.5)


def _plot_width(ax, df, targets):
    stations = sorted(df["station"].unique())

    for m in METHODS_ORDER:
        color = METHOD_COLORS[m]
        style = METHOD_STYLES[m]

        # Spaghetti: one thin per-station line for each method. Same colour
        # and linestyle as the bold mean so the reader can trace the method.
        for st in stations:
            vals = []
            for t in targets:
                sel = df[(df["method"] == m) & (df["target_num"] == t)
                         & (df["station"] == st)]["AvgWidth"]
                vals.append(float(sel.iloc[0]) if not sel.empty else np.nan)
            ax.plot(targets, vals, lw=0.6, color=color, alpha=0.35,
                    linestyle=style["linestyle"], zorder=2)

        # Bold mean line on top, with marker + legend entry.
        means = np.array([_agg(df, m, t, "AvgWidth", np.mean) for t in targets])
        ax.plot(targets, means, ms=4, lw=1.6, color=color,
                linestyle=style["linestyle"], marker=style["marker"],
                label=METHOD_LABELS[m], zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("Nominal target (%)")
    ax.set_ylabel("Average interval width (log scale)")
    ax.set_xticks(targets)
    ax.set_title("(b) Interval width", fontsize=10, pad=4)
    ax.legend(loc="upper left", fontsize=8, handlelength=1.5)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    _set_style()

    csv_path = os.path.join(RESULTS_DIR, "summary_all_methods.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}. Run dacp_arima.py first.")
        return

    df = pd.read_csv(csv_path)
    df = df[df["method"].isin(METHODS_ORDER)].copy()
    df["target_num"] = df["target"].map(TARGET_NUMERIC)
    targets = sorted(t for t in df["target_num"].dropna().unique())

    n_stations = df["station"].nunique()
    print(f"Aggregating across {n_stations} station(s), {len(targets)} target(s).")

    fig, (ax_cal, ax_w) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    _plot_calibration_error(ax_cal, df, targets)
    _plot_width(ax_w, df, targets)
    fig.tight_layout(pad=0.6)

    png_path = os.path.join(PLOT_DIR, "summary_calibration_width.png")
    pdf_path = os.path.join(PLOT_DIR, "summary_calibration_width.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Summary figure -> {png_path}")
    print(f"Summary figure -> {pdf_path}")


if __name__ == "__main__":
    main()
