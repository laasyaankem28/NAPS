"""Split conformal baseline with deployment-order calibration.

At each rolling step the ARIMA model is fit on the OLDER 32 observations of
the 80-step window, then walked forward through the NEWER 48 observations to
generate one-step-ahead calibration residuals. This is the same forward
calibration direction used by the rolling-window and DACP pipelines, so all
three methods are now computing residuals consistently, and any coverage
difference reflects the method itself rather than how each one happens to
arrange its calibration block.
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

# Window and split (match DACP / rolling split)
WINDOW_W = 80
N_TRAIN = 32
N_CAL = WINDOW_W - N_TRAIN

# Quantile stabilizers (match DACP / rolling split)
EWM_ALPHA = 0.15
Q_CAP_MULT = 5.0
Q_CAP_WINDOW = 100

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


def compute_station_pass(discharge, arima_order):
    """Run the target-independent ARIMA + calibration pass once per station.

    For each rolling step:
        train  = OLDEST N_TRAIN observations of the 80-step window
        calib  = NEWER N_CAL observations of the 80-step window
        target = the next observation after the window

    The ARIMA model is fit on `train`, then walked forward through `calib`
    using one-step-ahead forecasts. Each forecast produces a residual that
    represents true out-of-sample prediction performance, in deployment order.
    """
    y = log_transform(discharge)
    n = len(y)

    steps = []
    truths_log = []
    preds_log = []
    calib_resids_per_step = []

    for i in range(WINDOW_W, n):
        window = y[i - WINDOW_W:i]
        train = window[:N_TRAIN]       # OLDER 32 for training
        calib = window[N_TRAIN:]       # NEWER 48 for calibration

        try:
            model = ARIMA(train, order=arima_order).fit()

            # Walk forward through calibration: each step produces a
            # one-step-ahead forecast for the next calibration observation,
            # then absorbs that observation into the model state for the
            # following step.
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

            # The point forecast for the target step (i) uses the model after
            # it has absorbed the full calibration set, which gives it the
            # full window of training plus calibration observations to forecast
            # from. This is also the deployment-order choice.
            try:
                point_log = float(ext.forecast(steps=1)[0])
            except Exception:
                point_log = float(calib[-1])

            calib_resids = np.asarray(calib_resids, dtype=float)
        except Exception:
            point_log = float(window[-1])
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
    """Build the per-step interval DataFrame for one target."""
    alpha = 1.0 - target_coverage

    raw_quantiles = [
        float(np.quantile(resids, 1.0 - alpha))
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
                  f"train {N_TRAIN}, cal {N_CAL}")

            result = assemble_result(cached, target)
            cov, w, cal_err = summarize(result, target)

            print(f"    Coverage: {cov:.3f} (target {target:.2f}, error {cal_err:.3f})")
            print(f"    Width:    {w:.2f}")

            target_str = f"{int(target * 100)}%"
            out_path = os.path.join(
                OUT_DIR, f"split_conformal_{station}_{target_str}.csv"
            )
            result.to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()