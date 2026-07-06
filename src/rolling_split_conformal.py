"""Rolling-window split conformal baseline.

At each step the point forecast is produced the SAME way as DACP and
split_conformal -- fit ARIMA on the most recent N_TRAIN points of the window
and forecast one step ahead. This keeps the base forecaster identical across
methods, so any coverage/width difference is attributable to the conformal
machinery, not to a stronger or weaker predictor (a fair ablation).

What makes this method "rolling" rather than just split conformal is the
calibration set: instead of a held-out block inside each window, the quantile
is taken from a causal sliding buffer of the most recent RESID_BUFFER one-step
residuals (the realized |y - yhat| from prior steps). The buffer slides
forward with the stream, giving natural responsiveness to non-stationarity,
but the quantile level (k) and the train/cal split (t) are both fixed -- there
is no adaptation. This is the standard non-adaptive precursor to ACI.

Run this on each (station, target) pair. The per-step CSVs go into
results/dacp/ and the summary numbers print to stdout for pasting into
your run log and into the paper's table.
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_W = 80
TRAIN_FRACTION = 0.40
N_TRAIN = round(TRAIN_FRACTION * WINDOW_W)
N_CAL = WINDOW_W - N_TRAIN

# Length of the causal residual buffer the rolling quantile is taken from.
# Matched to N_CAL so the quantile is estimated from the same number of scores
# as split_conformal's calibration block, keeping the comparison apples-to-apples.
RESID_BUFFER = N_CAL

EWM_ALPHA = 0.15
Q_CAP_MULT = 2.0
Q_CAP_WINDOW = 100

# Station-specific ARIMA orders, matching your run log
ARIMA_ORDERS = {
    "Swift_Creek":       (1, 0, 1),
    "Bunchgrass_Meadow": (1, 0, 1),
    "Touchet":           (1, 1, 1),
    "Paradise":          (1, 0, 1),
    "Easy_Pass":         (1, 0, 1),
}

DATA_DIR = "A2_Experiment data"   # adjust to your actual data folder
OUT_DIR = "results/dacp"


def log_transform(x):
    return np.log(1.0 + np.asarray(x, dtype=float))


def inverse_log(y):
    return np.exp(np.asarray(y, dtype=float)) - 1.0


def ewm_smooth(values, alpha=EWM_ALPHA):
    series = pd.Series(values)
    return series.ewm(alpha=alpha, adjust=False).mean().values


def apply_cap(q_smoothed, mult=Q_CAP_MULT, window=Q_CAP_WINDOW):
    series = pd.Series(q_smoothed)
    rolling_median = series.rolling(window=window, min_periods=1).median().values
    cap = mult * rolling_median
    return np.minimum(q_smoothed, cap)


def compute_station_pass(data, arima_order):
    """Run the target-independent forecast + residual pass once per station.

    At each step:
    1. Take the most recent 80 observations as the window.
    2. Fit ARIMA on the most recent N_TRAIN points (same forecaster as DACP /
       split_conformal) and forecast y[i] one step ahead.
    3. Record the realized one-step residual |y[i] - yhat_i|.

    The point forecasts and residuals do not depend on the target coverage, so
    we compute them once and let every target reuse the cache. The rolling
    quantile itself is formed later, in assemble_result, from a causal buffer
    of these residuals.

    Returns a dict with steps and log-space truths / predictions / residuals.
    """
    y = log_transform(data)
    n = len(y)

    steps = []
    truths_log = []
    preds_log = []
    resids_log = []

    for i in range(WINDOW_W, n):
        window = y[i - WINDOW_W:i]
        train = window[-N_TRAIN:]   # most recent points -- fair forecaster

        try:
            model = ARIMA(train, order=arima_order).fit()
            point_log = float(model.forecast(steps=1)[0])
        except Exception:
            point_log = float(train[-1])

        steps.append(i)
        truths_log.append(float(y[i]))
        preds_log.append(point_log)
        # Realized one-step residual; only ever used by LATER steps (causal).
        resids_log.append(abs(float(y[i]) - point_log))

    return {
        "steps": steps,
        "truths_log": truths_log,
        "preds_log": preds_log,
        "resids_log": resids_log,
    }


def assemble_result(cached, target_coverage):
    """Build the per-step interval DataFrame for one target from the cached
    station pass. This is the only part that depends on the target coverage:
    at each step take the (1 - alpha) quantile of the causal sliding buffer of
    the most recent RESID_BUFFER one-step residuals, then smooth and cap to
    match the DACP pipeline.

    Causality: the interval at step k uses residuals strictly before k (we have
    not observed y[k] yet). The very first step has no history and falls back to
    its own residual -- a single-step warm-up that is negligible over ~1400
    steps.
    """
    alpha = 1.0 - target_coverage
    resids = cached["resids_log"]

    raw_quantiles = []
    for k in range(len(resids)):
        buf = resids[max(0, k - RESID_BUFFER):k]   # strictly before k
        if len(buf) == 0:
            buf = resids[:1]                        # warm-up only
        raw_quantiles.append(float(np.quantile(buf, 1.0 - alpha)))

    # Apply EWM smoothing then median cap, matching the DACP pipeline
    q_smoothed = ewm_smooth(raw_quantiles)
    q_final = apply_cap(q_smoothed)

    preds_log = cached["preds_log"]
    preds_orig = inverse_log(preds_log)
    truths_orig = inverse_log(cached["truths_log"])
    lowers_log = np.array(preds_log) - q_final
    uppers_log = np.array(preds_log) + q_final
    lowers_orig = np.maximum(inverse_log(lowers_log), 0.0)
    uppers_orig = inverse_log(uppers_log)

    return pd.DataFrame({
        "step": cached["steps"],
        "true_value": truths_orig,
        "prediction": preds_orig,
        "lower_bound": lowers_orig,
        "upper_bound": uppers_orig,
    })


def summarize(df, target_coverage):
    in_band = ((df["true_value"] >= df["lower_bound"]) &
               (df["true_value"] <= df["upper_bound"]))
    coverage = float(in_band.mean())
    width = float((df["upper_bound"] - df["lower_bound"]).mean())
    cal_err = abs(coverage - target_coverage)
    return coverage, width, cal_err


def main():
    stations = [
        "Swift_Creek",
        "Bunchgrass_Meadow",
        "Touchet",
        "Paradise",
        "Easy_Pass",
    ]

    targets = [0.70, 0.80, 0.90, 0.95]

    os.makedirs(OUT_DIR, exist_ok=True)

    for station in stations:
        print("\n" + "="*60)
        print(f"  {station.replace('_', ' ')}")
        print("="*60)

        arima_order = ARIMA_ORDERS[station]
        data_path = os.path.join(DATA_DIR, f"{station}_Inflow_Data_FILLED.csv")
        print(f"Loading {data_path}...")
        df_raw = pd.read_csv(data_path)

        if "discharge" in df_raw.columns:
            discharge = df_raw["discharge"].values
        else:
            discharge = df_raw.iloc[:, 1].values

        # The expensive ARIMA + calibration pass is target-independent, so run
        # it once per station and reuse the cache across all four targets.
        print("  Running ARIMA + calibration pass (shared across targets)...")
        cached = compute_station_pass(discharge, arima_order)

        for target in targets:
            print(f"\n  --- target {int(target*100)}% ---")
            print(f"  ARIMA order: {arima_order}, window {WINDOW_W}, "
                  f"train {N_TRAIN}, cal {N_CAL}")

            result = assemble_result(cached, target)
            cov, w, cal_err = summarize(result, target)

            print(f"    Coverage: {cov:.3f} (target {target:.2f}, error {cal_err:.3f})")
            print(f"    Width:    {w:.2f}")

            out_path = os.path.join(
                OUT_DIR,
                f"rolling_split_{station}_{int(target*100)}%.csv"
            )
            result.to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
