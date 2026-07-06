"""Compute and save baseline per-step CSVs for Touchet at the 95% target.

dacp_arima.py runs all three methods but only saves DACP's per-step trajectory
to disk -- baselines are evaluated and aggregated, then discarded. probe.py
needs the baseline per-step trajectories to draw the three-panel method
comparison. This script recomputes the two baselines for Touchet at 95% using
the same implementations as the main run (imported, not reimplemented) and
saves them with the column schema probe.py reads.

Outputs (in results/dacp/):
  single_aci_Touchet_95%.csv
  split_conformal_Touchet_95%.csv

Why import instead of reimplementing: the user's concern was that the baselines
recomputed here match what dacp_arima.py reported in the summary tables.
Importing run_single_aci and run_split_conformal eliminates that risk -- same
code path, same hyperparameters, same result. The computation is deterministic.

Runtime: roughly the time to fit ARIMA ~1380 times for each of the two
baselines on Touchet (1,1,1). Typically a few minutes on a laptop.
"""

import os

import pandas as pd

from dacp_arima import (
    COVERAGE_TARGETS,
    DATA_DIR,
    RESULTS_DIR,
    TARGET_LEVEL,
    load_tuned_orders,
    run_single_aci,
    run_split_conformal,
)

STATION = "Touchet"
STATION_FILE = "Touchet_Inflow_Data_FILLED.csv"
TARGET_LBL = "95%"


def _save(result, method):
    path = os.path.join(
        RESULTS_DIR, f"{method}_{STATION}_{TARGET_LBL}.csv")
    pd.DataFrame({
        "step": result["steps"],
        "true_value": result["truth"],
        "prediction": result["pred"],
        "lower_bound": result["lower"],
        "upper_bound": result["upper"],
        "width": result["width"],
        "covered": result["covered"],
    }).to_csv(path, index=False)
    cov = float(result["covered"].mean())
    width = float(result["width"].mean())
    print(f"  {method:16s} cov={cov:.3f}  width={width:.2f}  -> {path}")


def main():
    tuned_orders = load_tuned_orders()
    if STATION not in tuned_orders:
        raise KeyError(
            f"{STATION!r} not found in tuned_orders.json. "
            f"Run arima.py first or check the station name."
        )
    order = tuned_orders[STATION]
    print(f"Computing baselines for {STATION} at {TARGET_LBL} "
          f"with ARIMA order {order} ...")

    df = pd.read_csv(os.path.join(DATA_DIR, STATION_FILE))
    series = pd.Series(df["DISCHRG"].values, name="DISCHRG")

    cfg = COVERAGE_TARGETS[TARGET_LBL]
    level = TARGET_LEVEL[TARGET_LBL]
    k_init, hit, miss = cfg["k_init"], cfg["hit"], cfg["miss"]

    print("Running single_aci ...")
    r_sa = run_single_aci(series, order, k_init, hit, miss)
    _save(r_sa, "single_aci")

    print("Running split_conformal ...")
    r_sc = run_split_conformal(series, order, level)
    _save(r_sc, "split_conformal")

    print("Done.")


if __name__ == "__main__":
    main()
