"""Weighted conformal prediction baseline.

At each rolling step, computes residuals the same way as split_conformal
(walk ARIMA forward through the older 48 observations of the 80-step window).
The difference is in how the quantile is computed: instead of an unweighted
quantile, this uses an exponentially-weighted quantile that emphasizes more
recent calibration residuals. This gives natural responsiveness to changes
in the residual distribution without any explicit coverage feedback.

Reference: Barber, Candes, Ramdas, Tibshirani (2023), "Conformal prediction
beyond exchangeability," Annals of Statistics 51(2), 816-845.
"""

import os
import numpy as np
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_W = 80
N_TRAIN = 32
N_CAL = WINDOW_W - N_TRAIN

EWM_ALPHA = 0.15
Q_CAP_MULT = 5.0
Q_CAP_WINDOW = 100

# Weighted conformal decay parameter. Larger LAMBDA = more emphasis on recent
# residuals. lambda = 0.05 gives the most recent residual weight 1.0 and the
# oldest of 48 weight exp(-2.4) = 0.09, so the effective sample size is roughly
# half the block.
LAMBDA = 0.05

ARIMA_ORDERS = {
    "Swift_Creek":       (1, 0, 1),
    "Bunchgrass_Meadow": (1, 0, 1),
    "Touchet":           (1, 1, 1),
    "Paradise":          (1, 0, 1),
    "Easy_Pass":         (1, 0, 1),
}

DATA_DIR = "A2_Experiment data"
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


def weighted_quantile(values, weights, q):
    """Compute the weighted q-th quantile of values.
    
    values: array of observations
    weights: array of weights, same length as values
    q: target quantile in [0, 1]
    
    Returns the value v such that the sum of weights of values <= v is at least
    q * total_weight.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Sort by value
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    
    # Cumulative weights, normalized
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    if total_weight <= 0:
        return float(np.quantile(values, q))
    cum_fractions = cum_weights / total_weight
    
    # Find the smallest value where cumulative fraction reaches q
    idx = np.searchsorted(cum_fractions, q, side="left")
    idx = min(idx, len(sorted_values) - 1)
    return float(sorted_values[idx])


def compute_station_pass(discharge, arima_order):
    """Run the target-independent ARIMA + calibration pass once per station.
    
    Same logic as split_conformal: walk ARIMA forward through the older 48
    observations of each 80-step window, collecting one-step calibration
    residuals. These residuals will be weighted at quantile time.
    """
    y = log_transform(discharge)
    n = len(y)

    steps = []
    truths_log = []
    preds_log = []
    calib_resids_per_step = []

    for i in range(WINDOW_W, n):
        window = y[i - WINDOW_W:i]
        train = window[-N_TRAIN:]
        calib = window[:-N_TRAIN]

        try:
            model = ARIMA(train, order=arima_order).fit()
            point_log = float(model.forecast(steps=1)[0])

            calib_resids = []
            ext = model
            for v in calib:
                try:
                    yhat = float(ext.forecast(steps=1)[0])
                except Exception:
                    yhat = float(train[-1])
                calib_resids.append(abs(v - yhat))
                try:
                    ext = ext.append([float(v)], refit=False)
                except Exception:
                    pass
            calib_resids = np.asarray(calib_resids, dtype=float)
        except Exception:
            point_log = float(train[-1])
            calib_resids = np.abs(calib - np.mean(train))

        steps.append(i)
        truths_log.append(float(y[i]))
        preds_log.append(point_log)
        calib_resids_per_step.append(calib_resids)

    return {
        "steps": steps,
        "truths_log": truths_log,
        "preds_log": preds_log,
        "calib_resids_per_step": calib_resids_per_step,
    }


def assemble_result(cached, target_coverage):
    """Build per-step intervals using a weighted quantile of calibration
    residuals. Weights decay exponentially with distance from the prediction
    step (most recent calibration residual has weight 1, oldest has weight
    exp(-LAMBDA * (N_CAL - 1))).
    """
    alpha = 1.0 - target_coverage
    
    # Precompute weights once. The calibration block is ordered oldest-first,
    # so position 0 is the oldest residual (largest distance) and position
    # N_CAL - 1 is the most recent. Distance d_i for position i is (N_CAL - 1 - i).
    distances = np.arange(N_CAL - 1, -1, -1)
    weights = np.exp(-LAMBDA * distances)
    
    raw_quantiles = [
        weighted_quantile(resids, weights, 1.0 - alpha)
        for resids in cached["calib_resids_per_step"]
    ]

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
        print("\n" + "=" * 60)
        print(f"  {station.replace('_', ' ')}")
        print("=" * 60)

        arima_order = ARIMA_ORDERS[station]
        data_path = os.path.join(DATA_DIR, f"{station}_Inflow_Data_FILLED.csv")
        print(f"Loading {data_path}...")
        df_raw = pd.read_csv(data_path)
        if "DISCHRG" in df_raw.columns:
            discharge = df_raw["DISCHRG"].values
        else:
            discharge = df_raw.iloc[:, 1].values

        print("  Running ARIMA + calibration pass (shared across targets)...")
        cached = compute_station_pass(discharge, arima_order)

        for target in targets:
            print(f"\n  --- target {int(target * 100)}% ---")
            print(f"  ARIMA order: {arima_order}, window {WINDOW_W}, "
                  f"train {N_TRAIN}, cal {N_CAL}, lambda {LAMBDA}")

            result = assemble_result(cached, target)
            cov, w, cal_err = summarize(result, target)

            print(f"    Coverage: {cov:.3f} (target {target:.2f}, error {cal_err:.3f})")
            print(f"    Width:    {w:.2f}")

            target_str = f"{int(target * 100)}%"
            out_path = os.path.join(
                OUT_DIR, f"weighted_conformal_{station}_{target_str}.csv"
            )
            result.to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()