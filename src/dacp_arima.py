"""DACP + ARIMA coverage experiment over 5 stations x 70/80/90/95% targets.

DACP has two adaptive regulators:
  k_adap - the quantile level taken from calibration residuals
  t_adap - the training-set fraction of the rolling window (calibration
           set is the remainder), following Zong et al.

Baselines (both conformal, both share DACP's split convention -- training
is the most recent points, calibration the older remainder). Together
with DACP they form a clean ablation:
  single_aci      - adapts k_adap only (DACP minus t_adap)
  split_conformal - fixed conformal quantile (DACP minus k_adap and t_adap)

70/80/90% targets are reported as a numeric table; 95% gets a per-station
forecast plot.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DATA_DIR = "A2_Experiment data"
ARIMA_RESULTS_DIR = "results/arima"
RESULTS_DIR = "results/dacp"
PLOT_DIR = "plots/dacp"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TRAINING_SIZE = 80        # rolling window length
EPS = 1e-10

# t_adap: n_train = round(t_adap * TRAINING_SIZE), n_cal is the remainder.
T_LOW, T_HIGH, T_INIT = 0.30, 0.50, 0.40
N_TRAIN_MIN = 12          # smallest training set we will fit ARIMA on
N_CAL_MIN = 5             # smallest calibration set we will leave
MU_T_HIT, MU_T_MISS = +0.004, -0.008       # t_adap update steps (Zong-style)

N_TRAIN_FIXED = 32        # fixed split for the non-adaptive baselines

# spike dampening for the residual quantile
EWM_ALPHA = 0.15
Q_CAP_MULT = 5
Q_CAP_WINDOW = 100

# hit/miss steps set so the k_adap loop converges to the coverage target.
SCALE = 0.01
COVERAGE_TARGETS = {
    "70%": {"k_init": 0.30, "hit": 0.30 * SCALE, "miss": 0.70 * SCALE},
    "80%": {"k_init": 0.20, "hit": 0.20 * SCALE, "miss": 0.80 * SCALE},
    "90%": {"k_init": 0.10, "hit": 0.10 * SCALE, "miss": 0.90 * SCALE},
    "95%": {"k_init": 0.05, "hit": 0.05 * SCALE, "miss": 0.95 * SCALE},
}
TARGET_LEVEL = {"70%": 0.70, "80%": 0.80, "90%": 0.90, "95%": 0.95}

# Okabe-Ito colorblind-safe palette
COLORS = {"70%": "#E69F00", "80%": "#56B4E9", "90%": "#009E73", "95%": "#0072B2"}
METHOD_COLORS = {
    "DACP": "#0072B2", "single_aci": "#D55E00",
    "split_conformal": "#009E73",
}

FILES = [
    "Swift_Creek_Inflow_Data_FILLED.csv",
    "Bunchgrass_Meadow_Inflow_Data_FILLED.csv",
    "Touchet_Inflow_Data_FILLED.csv",
    "Paradise_Inflow_Data_FILLED.csv",
    "Easy_Pass_Inflow_Data_FILLED.csv",
]


def log_t(x):
    return np.log1p(x)


def inv_t(x):
    return np.expm1(x)


def load_tuned_orders():
    path = os.path.join(ARIMA_RESULTS_DIR, "tuned_orders.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python arima.py` first to generate "
            f"tuned ARIMA orders before running this experiment."
        )
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def _split_window(w, n_train):
    # Training = most recent n_train points, calibration = older remainder.
    n_train = int(np.clip(n_train, N_TRAIN_MIN, TRAINING_SIZE - N_CAL_MIN))
    return w[-n_train:], w[:-n_train]


def _fit_and_forecast(fit_window, order):
    # Fit ARIMA in log space; return the 1-step forecast and fitted object.
    try:
        fit = ARIMA(fit_window, order=order).fit()
        yhat = float(np.asarray(fit.forecast(steps=1))[0])
        return yhat, fit
    except Exception:
        return float(fit_window[-1]), None


def _calibration_residuals(fit, calib_obs, fallback_window):
    # Roll forward through the calibration window for out-of-sample residuals.
    if fit is None or len(calib_obs) == 0:
        return np.array([float(np.std(fallback_window))])
    ext = fit
    cps = []
    for v in calib_obs:
        try:
            cps.append(float(np.asarray(ext.forecast(steps=1))[0]))
        except Exception:
            cps.append(float(calib_obs[0]))
        try:
            ext = ext.append([float(v)], refit=False)
        except Exception:
            pass
    return np.abs(np.array(cps) - calib_obs)


def run_dacp(series, order, k_init, hit_step, miss_step):
    #Dual adaptive CP: k_adap moves the quantile level, t_adap the split.
    raw = series.values
    y = log_t(raw)
    n = len(y)

    k_adap = k_init
    t_adap = T_INIT
    ewm_q = None
    q_history = []

    out = {key: [] for key in
           ("steps", "truth", "pred", "lower", "upper", "width",
            "covered", "k_trace", "t_trace", "ncal_trace")}

    for i in range(TRAINING_SIZE, n):
        w = y[i - TRAINING_SIZE:i]

        n_train = int(round(t_adap * TRAINING_SIZE))
        fit_part, calib_part = _split_window(w, n_train)
        n_cal = len(calib_part)

        yhat_next, fit = _fit_and_forecast(fit_part, order)
        abs_resid = _calibration_residuals(fit, calib_part, w)

        ql = float(np.clip(1.0 - k_adap, 0.01, 0.999))
        q_raw = float(np.quantile(abs_resid, ql))

        # EWM smoothing + rolling-median cap to damp spikes
        ewm_q = q_raw if ewm_q is None else EWM_ALPHA * q_raw + (1 - EWM_ALPHA) * ewm_q
        q_history.append(q_raw)
        if len(q_history) > Q_CAP_WINDOW:
            q_history.pop(0)
        q = min(ewm_q, Q_CAP_MULT * np.median(q_history))

        pred = float(inv_t(np.array([yhat_next]))[0])
        lo = max(float(inv_t(np.array([max(yhat_next - q, 0)]))[0]), 0.0)
        hi = float(inv_t(np.array([yhat_next + q]))[0])
        is_in = lo <= raw[i] <= hi

        # both regulators update after every step
        if is_in:
            k_adap = float(np.clip(k_adap + hit_step, 0.001, 0.999))
            t_adap = float(np.clip(t_adap + MU_T_HIT, T_LOW, T_HIGH))
        else:
            k_adap = float(np.clip(k_adap - miss_step, 0.001, 0.999))
            t_adap = float(np.clip(t_adap + MU_T_MISS, T_LOW, T_HIGH))

        for key, val in (("steps", i), ("truth", float(raw[i])), ("pred", pred),
                         ("lower", lo), ("upper", hi), ("width", hi - lo),
                         ("covered", is_in), ("k_trace", k_adap),
                         ("t_trace", t_adap), ("ncal_trace", n_cal)):
            out[key].append(val)

    return {k: np.array(v) for k, v in out.items()}


def run_single_aci(series, order, k_init, hit_step, miss_step):
    # DACP minus t_adap: adapts k_adap only, with a fixed split.
    raw = series.values
    y = log_t(raw)
    n = len(y)

    k_adap = k_init
    ewm_q = None
    q_history = []
    out = {key: [] for key in
           ("steps", "truth", "pred", "lower", "upper", "width", "covered")}

    for i in range(TRAINING_SIZE, n):
        w = y[i - TRAINING_SIZE:i]
        fit_part, calib_part = _split_window(w, N_TRAIN_FIXED)

        yhat_next, fit = _fit_and_forecast(fit_part, order)
        abs_resid = _calibration_residuals(fit, calib_part, w)

        ql = float(np.clip(1.0 - k_adap, 0.01, 0.999))
        q_raw = float(np.quantile(abs_resid, ql))
        ewm_q = q_raw if ewm_q is None else EWM_ALPHA * q_raw + (1 - EWM_ALPHA) * ewm_q
        q_history.append(q_raw)
        if len(q_history) > Q_CAP_WINDOW:
            q_history.pop(0)
        q = min(ewm_q, Q_CAP_MULT * np.median(q_history))

        pred = float(inv_t(np.array([yhat_next]))[0])
        lo = max(float(inv_t(np.array([max(yhat_next - q, 0)]))[0]), 0.0)
        hi = float(inv_t(np.array([yhat_next + q]))[0])
        is_in = lo <= raw[i] <= hi

        k_adap = float(np.clip(
            k_adap + (hit_step if is_in else -miss_step), 0.001, 0.999))

        for key, val in (("steps", i), ("truth", float(raw[i])), ("pred", pred),
                         ("lower", lo), ("upper", hi), ("width", hi - lo),
                         ("covered", is_in)):
            out[key].append(val)

    return {k: np.array(v) for k, v in out.items()}


def run_split_conformal(series, order, target_level):
    # Fixed conformal quantile, no adaptation. The rolling-median cap here
    # is numerical hygiene, not adaptation - it keeps the comparison fair and
    # stops a single large log-space residual from blowing up the interval.
    raw = series.values
    y = log_t(raw)
    n = len(y)
    q_history = []
    out = {key: [] for key in
           ("steps", "truth", "pred", "lower", "upper", "width", "covered")}

    for i in range(TRAINING_SIZE, n):
        w = y[i - TRAINING_SIZE:i]
        fit_part, calib_part = _split_window(w, N_TRAIN_FIXED)

        yhat_next, fit = _fit_and_forecast(fit_part, order)
        abs_resid = _calibration_residuals(fit, calib_part, w)

        q_raw = float(np.quantile(abs_resid, target_level))
        q_history.append(q_raw)
        if len(q_history) > Q_CAP_WINDOW:
            q_history.pop(0)
        q = min(q_raw, Q_CAP_MULT * np.median(q_history))

        pred = float(inv_t(np.array([yhat_next]))[0])
        lo = max(float(inv_t(np.array([max(yhat_next - q, 0)]))[0]), 0.0)
        hi = float(inv_t(np.array([yhat_next + q]))[0])
        is_in = lo <= raw[i] <= hi

        for key, val in (("steps", i), ("truth", float(raw[i])), ("pred", pred),
                         ("lower", lo), ("upper", hi), ("width", hi - lo),
                         ("covered", is_in)):
            out[key].append(val)

    return {k: np.array(v) for k, v in out.items()}


def nse(truth, pred):
    # Nash-Sutcliffe Efficiency. 1 = perfect, 0 = no better than the mean.
    denom = np.sum((truth - np.mean(truth)) ** 2)
    if denom < EPS:
        return float("nan")
    return float(1.0 - np.sum((truth - pred) ** 2) / denom)


def kge(truth, pred):
    # Kling-Gupta Efficiency: correlation, bias and variability combined.
    if np.std(truth) < EPS or np.std(pred) < EPS:
        return float("nan")
    r = float(np.corrcoef(truth, pred)[0, 1])
    alpha = float(np.std(pred) / np.std(truth))
    beta = float(np.mean(pred) / (np.mean(truth) + EPS))
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(truth, pred):
    # Percent bias. + = systematic over-prediction, - = under-prediction.
    denom = float(np.sum(truth))
    if abs(denom) < EPS:
        return float("nan")
    return float(100.0 * np.sum(pred - truth) / denom)


def interval_score(truth, lower, upper, target_level):
    # Winkler interval score - lower is better.
    alpha = 1.0 - target_level
    width = upper - lower
    below = (truth < lower)
    above = (truth > upper)
    penalty = (2.0 / alpha) * ((lower - truth) * below + (truth - upper) * above)
    return float(np.mean(width + penalty))


def per_step_interval_score(truth, lower, upper, target_level):
    # Per-step interval score array, for the Wilcoxon test.
    alpha = 1.0 - target_level
    width = upper - lower
    below = (truth < lower)
    above = (truth > upper)
    penalty = (2.0 / alpha) * ((lower - truth) * below + (truth - upper) * above)
    return width + penalty


def miss_magnitude(truth, lower, upper):
    # Per-step distance the truth fell outside the interval; 0 if covered.
    # Continuous form of the miss indicator -- binary 0/1 has too many ties
    # for Wilcoxon to be powerful, this lets the test see how *badly* misses
    # missed, not just whether they happened.
    return np.maximum.reduce([lower - truth, truth - upper,
                              np.zeros_like(truth)])


def conditional_coverage(r, target_level):
    # Coverage and mean width within low/medium/high observed-flow terciles.
    truth = r["truth"]
    lo_cut, hi_cut = np.quantile(truth, [1 / 3, 2 / 3])
    bands = {
        "low": truth <= lo_cut,
        "medium": (truth > lo_cut) & (truth <= hi_cut),
        "high": truth > hi_cut,
    }
    rows = {}
    for name, mask in bands.items():
        if mask.sum() == 0:
            rows[name] = {"coverage": float("nan"), "width": float("nan"),
                          "n": 0}
        else:
            rows[name] = {
                "coverage": float(np.mean(r["covered"][mask])),
                "width": float(np.mean(r["width"][mask])),
                "n": int(mask.sum()),
            }
    return rows


def evaluate(r, target_level):
    # Point and interval metrics. MRE is omitted - it is undefined for the
    # near-zero baseflow stations.
    truth, pred = r["truth"], r["pred"]
    coverage = float(np.mean(r["covered"]))
    return {
        "RMSE": float(np.sqrt(mean_squared_error(truth, pred))),
        "MAE": float(np.mean(np.abs(truth - pred))),
        "PBIAS": pbias(truth, pred),
        "NSE": nse(truth, pred),
        "KGE": kge(truth, pred),
        "Coverage": coverage,
        "CalibrationError": abs(coverage - target_level),
        "AvgWidth": float(np.mean(r["width"])),
        "IntervalScore": interval_score(
            truth, r["lower"], r["upper"], target_level),
    }


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


def plot_forecast(station, dacp_run, out_path):
    """Per-station forecast plot for the 95% target."""
    r = dacp_run
    time = r["steps"]
    true_value = r["truth"]
    prediction = r["pred"]
    lower_bound = r["lower"]
    upper_bound = r["upper"]

    plt.figure(figsize=(20, 5))
    plt.plot(time, true_value, 'b-', linewidth=2, label='True Value')
    plt.plot(time, prediction, 'g-', linewidth=2, label='Prediction (ARIMA)')
    plt.fill_between(
        time, lower_bound, upper_bound,
        color=(1, 0.647, 0), alpha=0.3,
        label='Coverage Area (DACP)',
    )
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Discharge', fontsize=14)
    plt.title(
        f'{station}: True Value and Prediction with Coverage (DACP + ARIMA)',
        fontsize=16,
    )
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  forecast figure (95%) -> {out_path}")


def plot_calibration(calib_table, out_path):
    """Nominal vs achieved coverage, one line per method."""
    _set_style()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    targets = [0.70, 0.80, 0.90, 0.95]

    ax.plot([0.65, 1.0], [0.65, 1.0], color="#999999", lw=0.8,
            ls=(0, (3, 3)), label="Ideal (y = x)", zorder=1)
    for method, achieved in calib_table.items():
        ax.plot(targets, achieved, marker="o", ms=4, lw=1.1,
                color=METHOD_COLORS.get(method, "#444444"),
                label=method.replace("_", " "), zorder=3)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Achieved coverage")
    ax.set_xticks(targets)
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_title("Calibration across methods", fontsize=9, pad=4)
    ax.legend(fontsize=6.8, loc="upper left", handlelength=1.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  calibration figure -> {out_path}")


def write_coverage_table(summary_rows, out_path,
                         targets=("70%", "80%", "90%")):
    """Pivot coverage into a station x method table for the lower targets."""
    df = pd.DataFrame(summary_rows)
    df = df[df["target"].isin(targets)]
    table = df.pivot_table(index=["station", "method"],
                           columns="target", values="Coverage")
    table = table.reindex(columns=list(targets))
    table.to_csv(out_path)
    print(f"\nCoverage table (70/80/90% targets) -> {out_path}")
    print(table.round(3).to_string())
    return table


class _Tee:
    """Mirror writes to multiple streams so stdout also lands in a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    print(f"Logging this run to {log_path}\n")

    tuned_orders = load_tuned_orders()
    print(f"Loaded tuned ARIMA orders for {len(tuned_orders)} stations.")

    summary_rows = []
    cond_rows = []
    wilcoxon_rows = []
    calib_accum = {m: {t: [] for t in COVERAGE_TARGETS}
                   for m in ("DACP", "single_aci", "split_conformal")}

    for fname in FILES:
        station = fname.replace("_Inflow_Data_FILLED.csv", "").replace("_", " ")
        print(f"\n{'=' * 56}\n  {station}\n{'=' * 56}")

        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        series = pd.Series(df["DISCHRG"].values, name="DISCHRG")
        order = tuned_orders.get(station, (1, 0, 1))
        print(f"  ARIMA order (from arima.py): {order}")

        for lbl, cfg in COVERAGE_TARGETS.items():
            level = TARGET_LEVEL[lbl]
            k_init, hit, miss = cfg["k_init"], cfg["hit"], cfg["miss"]
            print(f"\n  --- target {lbl} ---")

            methods = {
                "DACP": run_dacp(series, order, k_init, hit, miss),
                "single_aci": run_single_aci(series, order, k_init, hit, miss),
                "split_conformal": run_split_conformal(series, order, level),
            }

            for mname, r in methods.items():
                m = evaluate(r, level)
                calib_accum[mname][lbl].append(m["Coverage"])
                summary_rows.append({"station": station, "target": lbl,
                                     "method": mname, **m})
                print(f"    {mname:16s} cov={m['Coverage']:.3f}  "
                      f"calErr={m['CalibrationError']:.3f}  "
                      f"width={m['AvgWidth']:.2f}  IS={m['IntervalScore']:.2f}  "
                      f"PBIAS={m['PBIAS']:+.1f}%  NSE={m['NSE']:.3f}")

            for band, vals in conditional_coverage(methods["DACP"], level).items():
                cond_rows.append({"station": station, "target": lbl,
                                  "method": "DACP", "flow_band": band, **vals})

            # Wilcoxon: DACP vs each baseline on (a) per-step interval score
            # and (b) per-step miss magnitude. The first asks "are DACP's
            # intervals better overall?"; the second asks "when DACP misses,
            # does it miss less badly?" -- the calibration-quality claim.
            dacp_is = per_step_interval_score(
                methods["DACP"]["truth"], methods["DACP"]["lower"],
                methods["DACP"]["upper"], level)
            dacp_miss = miss_magnitude(
                methods["DACP"]["truth"], methods["DACP"]["lower"],
                methods["DACP"]["upper"])
            for base in ("single_aci", "split_conformal"):
                b_is = per_step_interval_score(
                    methods[base]["truth"], methods[base]["lower"],
                    methods[base]["upper"], level)
                b_miss = miss_magnitude(
                    methods[base]["truth"], methods[base]["lower"],
                    methods[base]["upper"])
                for metric_name, dacp_vals, base_vals in (
                    ("interval_score", dacp_is, b_is),
                    ("miss_magnitude", dacp_miss, b_miss),
                ):
                    try:
                        stat, p = wilcoxon(dacp_vals, base_vals)
                    except Exception:
                        stat, p = float("nan"), float("nan")
                    wilcoxon_rows.append({
                        "station": station, "target": lbl,
                        "comparison": f"DACP_vs_{base}",
                        "metric": metric_name,
                        "dacp_mean": float(np.mean(dacp_vals)),
                        "baseline_mean": float(np.mean(base_vals)),
                        "wilcoxon_stat": float(stat),
                        "p_value": float(p),
                    })

            rd = methods["DACP"]
            pd.DataFrame({
                "step": rd["steps"], "true_value": rd["truth"],
                "prediction": rd["pred"], "lower_bound": rd["lower"],
                "upper_bound": rd["upper"], "width": rd["width"],
                "covered": rd["covered"], "k_adap": rd["k_trace"],
                "t_adap": rd["t_trace"], "n_cal": rd["ncal_trace"],
            }).to_csv(os.path.join(
                RESULTS_DIR,
                f"DACP_{station.replace(' ', '_')}_{lbl}.csv"), index=False)

            # only the 95% target is plotted; 70/80/90% go in the table
            if lbl == "95%":
                ncal = methods["DACP"]["ncal_trace"]
                print(f"    [t_adap check] n_cal ranged {ncal.min()}-"
                      f"{ncal.max()} (mean {ncal.mean():.1f}) -- t_adap is live")
                plot_forecast(
                    station, methods["DACP"],
                    os.path.join(
                        PLOT_DIR,
                        f"forecast_{station.replace(' ', '_')}_95.png",
                    ),
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_all_methods.csv"),
                      index=False)
    pd.DataFrame(cond_rows).to_csv(
        os.path.join(RESULTS_DIR, "conditional_coverage.csv"), index=False)
    pd.DataFrame(wilcoxon_rows).to_csv(
        os.path.join(RESULTS_DIR, "wilcoxon_tests.csv"), index=False)

    write_coverage_table(summary_rows, os.path.join(
        RESULTS_DIR, "coverage_table_70_80_90.csv"))
    print(f"\n{'=' * 56}\nSummary tables written to {RESULTS_DIR}/")

    print("\nGenerating calibration figure ...")
    calib_table = {
        m: [float(np.mean(calib_accum[m][t])) for t in COVERAGE_TARGETS]
        for m in calib_accum
    }
    plot_calibration(calib_table, os.path.join(PLOT_DIR, "calibration.pdf"))

    print(f"\nDone. Terminal output saved to {log_path}")


if __name__ == "__main__":
    main()