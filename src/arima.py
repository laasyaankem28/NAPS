"""ARIMA order JUSTIFICATION for the NAPS 2026 inflow stations.

The forecasting experiment (dacp_arima.py) refits ARIMA on short rolling
windows. Order selection on the full series therefore does not transfer
reliably to the estimation regime actually used, and higher-order models
are numerically unstable on short windows. We therefore adopt a fixed,
parsimonious ARIMA(1,0,1) for all stations.

This script does NOT select the order -- it FIXES it at (1,0,1) and
produces the evidence that justifies that choice, following the
Box-Jenkins methodology (identification -> selection -> diagnostic check):

  1. IDENTIFICATION  -- ACF/PACF plots per station (visual support for a
                        low-order specification).
  2. SELECTION       -- AIC compared among a small set of LOW-ORDER
                        candidates (not a blind wide grid search), per
                        station, plus the full-series ADF stationarity
                        result -- reported as full-series diagnostics.
  3. DIAGNOSTIC CHECK -- Ljung-Box test on the ARIMA(1,0,1) residuals,
                        per station, confirming residual adequacy.

Outputs:
  results/arima/tuned_orders.json      -- (1,0,1) for every station
  results/arima/order_justification.csv-- ADF, candidate AIC, Ljung-Box
  plots/arima/acf_pacf_<station>.pdf   -- identification plots
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

DATA_DIR = "A2_Experiment data"
RESULTS_DIR = "results/arima"
PLOT_DIR = "plots/arima"

# The order is FIXED. This is the deployed specification for all stations.
FIXED_ORDER = (1, 0, 1)

# Small set of LOW-ORDER candidates for the AIC comparison (Box-Jenkins
# Stage 2). These are parsimonious orders appropriate to short-window
# estimation -- NOT a wide unconstrained grid. q reported with d applied.
LOW_ORDER_CANDIDATES = [(0, 1), (1, 0), (1, 1)]   # (p, q); d added per station

# Ljung-Box is evaluated at this many lags.
LB_LAGS = 10

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]


def _set_style():
    rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


# ----------------------------------------------------------------------
# STEP 1 -- IDENTIFICATION: ACF / PACF plots
# ----------------------------------------------------------------------
def acf_pacf_figure(series, station, out_path, n_lags=30):
    """ACF and PACF of the log-transformed series. Visual support for a
    low-order ARIMA: a sharp PACF cutoff after lag 1 supports AR(1); an
    ACF that tails off supports including an MA term."""
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    plot_acf(series, lags=n_lags, ax=axes[0], color="#0072B2",
             vlines_kwargs={"colors": "#0072B2"})
    plot_pacf(series, lags=n_lags, ax=axes[1], color="#D55E00",
              method="ywm", vlines_kwargs={"colors": "#D55E00"})
    axes[0].set_title(f"{station} -- ACF", fontsize=9)
    axes[1].set_title(f"{station} -- PACF", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Lag")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  identification figure -> {out_path}")


# ----------------------------------------------------------------------
# STEP 2 -- SELECTION: ADF + AIC among low-order candidates
# ----------------------------------------------------------------------
def choose_d(series, alpha=0.05):
    """ADF stationarity test on the (already log-transformed) series.
    Reported as a full-series diagnostic."""
    p_value = adfuller(series, autolag="AIC")[1]
    return (0 if p_value < alpha else 1), float(p_value)


def candidate_aic(series, d):
    """Fit each low-order candidate; return {order: aic}. The fixed order
    (1,0,1) is included so its AIC can be compared against the others."""
    orders = [(p, d, q) for p, q in LOW_ORDER_CANDIDATES]
    fixed_with_d = (FIXED_ORDER[0], d, FIXED_ORDER[2])
    if fixed_with_d not in orders:
        orders.append(fixed_with_d)
    aics = {}
    for order in orders:
        try:
            aics[order] = float(ARIMA(series, order=order).fit().aic)
        except Exception:
            aics[order] = float("inf")
    return aics, fixed_with_d


# ----------------------------------------------------------------------
# STEP 3 -- DIAGNOSTIC CHECK: Ljung-Box on the (1,0,1) residuals
# ----------------------------------------------------------------------
def ljung_box_check(series, order, lags=LB_LAGS):
    """Fit ARIMA(order) on the full series, run Ljung-Box on the residuals.
    A high p-value => residuals are not significantly autocorrelated =>
    the order adequately captures the linear structure."""
    try:
        fit = ARIMA(series, order=order).fit()
        lb = acorr_ljungbox(fit.resid, lags=[lags], return_df=True)
        return float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return float("nan"), float("nan")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    tuned_orders = {}
    rows = []

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        print(f"\n=== {station} ===")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = np.log1p(df["DISCHRG"].values)

        # the deployed order is fixed
        tuned_orders[station] = list(FIXED_ORDER)

        # STEP 1 -- identification plots
        acf_pacf_figure(series, station, os.path.join(
            PLOT_DIR, f"acf_pacf_{station.replace(' ', '_')}.pdf"))

        # STEP 2 -- ADF + low-order candidate AIC
        d, adf_p = choose_d(series)
        aics, fixed_with_d = candidate_aic(series, d)
        best_order = min(aics, key=aics.get)
        fixed_aic = aics[fixed_with_d]
        print(f"  ADF p={adf_p:.4f}  -> d={d} "
              f"({'stationary' if d == 0 else 'non-stationary'})")
        print(f"  low-order candidate AIC: " +
              ", ".join(f"{o}={a:.1f}" for o, a in aics.items()))
        print(f"  lowest-AIC candidate: {best_order}  "
              f"(deployed (1,0,1)-with-d AIC = {fixed_aic:.1f})")

        # STEP 3 -- Ljung-Box residual check on the DEPLOYED order (1,0,1)
        lb_stat, lb_p = ljung_box_check(series, FIXED_ORDER)
        verdict = ("adequate (residuals ~ white noise)" if lb_p > 0.05
                   else "residual autocorrelation remains")
        print(f"  Ljung-Box on ARIMA(1,0,1) residuals: "
              f"stat={lb_stat:.1f}  p={lb_p:.4f}  -> {verdict}")

        rows.append({
            "station": station,
            "deployed_order": str(FIXED_ORDER),
            "adf_pvalue": round(adf_p, 4),
            "adf_d": d,
            "candidate_aics": "; ".join(
                f"{o}:{a:.1f}" for o, a in aics.items()),
            "lowest_aic_candidate": str(best_order),
            "deployed_order_aic": round(fixed_aic, 2),
            "ljungbox_stat": round(lb_stat, 2),
            "ljungbox_pvalue": round(lb_p, 4),
            "ljungbox_lags": LB_LAGS,
            "residual_verdict": verdict,
        })

    # tuned_orders.json -- (1,0,1) for every station (consumed by dacp_arima.py)
    out_json = os.path.join(RESULTS_DIR, "tuned_orders.json")
    with open(out_json, "w") as f:
        json.dump(tuned_orders, f, indent=2)
    print(f"\nFixed orders (1,0,1 for all)  -> {out_json}")

    # justification table -- the evidence for the paper's methodology section
    diag_csv = os.path.join(RESULTS_DIR, "order_justification.csv")
    pd.DataFrame(rows).to_csv(diag_csv, index=False)
    print(f"Order-justification table     -> {diag_csv}")
    print(f"Identification figures        -> {PLOT_DIR}/acf_pacf_*.pdf")


if __name__ == "__main__":
    main()