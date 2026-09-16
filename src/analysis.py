"""Task 5: derive interpretation and figures from saved outputs, without refitting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def md_table(frame):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.6f}" if np.isfinite(v) else "not finite"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main():
    s = json.loads((RESULTS / "run_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "rmse_table.csv")
    pred = pd.read_csv(RESULTS / "test_predictions.csv", parse_dates=["datetime"])
    a, b = (metrics.loc[metrics.model == f"{key} normal"].iloc[0] for key in ("A", "B"))
    gap, peak = b.rmse_daytime - a.rmse_daytime, s["train_peak_kw"]
    ta, tb = (pd.read_csv(RESULTS / f"table3_set_{key}.csv") for key in ("a", "b"))
    nonbias = ta.loc[ta.feature != "intercept"]
    largest = nonbias.loc[nonbias.normal.abs().idxmax()]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, label, color in [("ac_power", "Actual", "#182b40"), ("pred_a_normal", "On-site sensors A", "#0b937e"), ("pred_b_normal", "Public weather B (provisional)", "#d99119")]:
        ax.plot(pred.datetime, pred[col], label=label, color=color, linewidth=1.3)
    ax.set(title="Held-out test week | 11–17 June 2020", xlabel="Local date (Asia/Kolkata assumption)", ylabel="Aggregate AC power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(RESULTS / "actual_vs_predicted.png", dpi=170)
    plt.close(fig)
    residual = pred.assign(hour=pred.datetime.dt.hour, residual=pred.ac_power - pred.pred_a_normal)
    hours = residual.groupby("hour").residual.agg(["count", "mean", "std", "min", "max"]).reset_index()
    hours.to_csv(RESULTS / "residuals_by_hour.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(residual.hour, residual.residual, s=20, alpha=.45, color="#688ca5", label="Individual test hours")
    ax.plot(hours.hour, hours["mean"], color="#107c6a", marker="o", label="Mean at each hour")
    ax.axhline(0, color="#222222", linewidth=.8)
    ax.set(title="Set A normal equation | residuals by hour", xlabel="Local hour", ylabel="Actual minus clipped predicted AC power (kW)", xticks=range(0, 24, 2))
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(RESULTS / "residuals_vs_hour.png", dpi=170)
    plt.close(fig)
    hi, lo = hours.loc[hours["mean"].idxmax()], hours.loc[hours["mean"].idxmin()]
    facts = {"daytime_gap_kw": float(gap), "gap_percent_training_peak": float(100*gap/peak), "a_daytime_percent_training_peak": float(100*a.rmse_daytime/peak), "b_daytime_percent_training_peak": float(100*b.rmse_daytime/peak), "largest_standardized_feature": str(largest.feature), "largest_standardized_coefficient_kw": float(largest.normal), "max_mean_residual_hour": int(hi.hour), "max_mean_residual_kw": float(hi['mean']), "min_mean_residual_hour": int(lo.hour), "min_mean_residual_kw": float(lo['mean'])}
    (RESULTS / "analysis_summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    convergence = pd.DataFrame([{"Set": key, "Batch iterations": v["batch_iterations"], "Batch max coefficient difference": v["batch_max_abs_theta_difference"], "Rounded coefficients agree": v["batch_round2_equal"], "SGD epochs": v["sgd_epochs"], "SGD max coefficient difference": v["sgd_max_abs_theta_difference"]} for key, v in s["solvers"].items()])
    convergence.to_csv(RESULTS / "table3_convergence_checks.csv", index=False)
    rates = pd.read_csv(RESULTS / "learning_rate_experiments.csv")
    text = f"""# Analysis of hourly solar power regression

## Question and scope

Can public weather replace on-site measurements for hourly aggregate AC power estimates at Plant 1? The normal-equation models give daytime RMSE of **{a.rmse_daytime:,.2f} kW for on-site sensors** versus **{b.rmse_daytime:,.2f} kW for public weather**. This historical observational comparison is not a causal experiment or a validated future forecast. Public-weather location and interval alignment remain unresolved; see [the investigation](location_check.md).

## Data quality

The hourly calendar contains {s['hourly_rows']} rows. Removing {s['excluded_hours']} missing hours without interpolation leaves {s['train_rows']} training rows (15 May–10 June) and {s['test_rows']} test rows (11–17 June). The same {s['test_daytime_rows']} daytime test rows, defined by on-site irradiation > 0, are used for both sets. Standardization uses training-only means and population standard deviations. The observed training peak is {peak:,.2f} kW, not a nameplate capacity.

Quarter-hour generation and sensor readings are matched before hourly averaging, so paired features and targets use the same samples. Incomplete inverter coverage and partial hours are reported, not silently repaired. [Table 1](preparation_table1.md) records counts and peak-hour checks. The unusual raw AC/DC ratio must not be interpreted as verified inverter efficiency.

## Table 2 Model comparison

Errors are in kW after clipping negative predictions to zero. For normal equation, alpha, iterations and epochs are not applicable (stored as zero). Learning rates and stopping rules use training data only.

{md_table(metrics[['feature_set','solver','alpha','iterations','epochs','rmse_all_hours','rmse_daytime']])}

## Table 3 Coefficients

All coefficients below use standardized features with an unscaled intercept. Batch GD must match normal-equation coefficients when rounded to two decimals AND have maximum absolute difference at most 0.00001 for each set separately. A difference below 0.005 alone would not guarantee matching rounding.

### Set A

{md_table(ta)}

### Set B

{md_table(tb)}

{md_table(convergence)}

## Physical interpretation

The largest absolute non-intercept Set A coefficient is **{largest.feature}**, {largest.normal:,.2f} kW per training standard deviation, holding other features fixed. A positive irradiation coefficient is consistent with more sunlight producing more power. Weather, temperature and time are correlated; their conditional coefficient signs do not establish a causal thermal efficiency effect. Standardized coefficients allow a scale-aware comparison, unlike a ranking of unscaled raw coefficients.

## Public weather and rooftop suitability

The B-minus-A daytime error gap is **{gap:,.2f} kW**, or **{100*gap/peak:.2f}% of training peak**. The respective daytime RMSE values equal {100*a.rmse_daytime/peak:.2f}% and {100*b.rmse_daytime/peak:.2f}% of peak; these ratios are not prediction accuracies. All-hours RMSE can look better because it includes many easier night hours.

Public weather is useful here as a provisional educational scenario input. One plant, 34 days, uncertain coordinates and imperfect records do not validate a rooftop yield, financial estimate or operational commitment. A roof would need known capacity, orientation, losses, confirmed location, seasonal data and independent validation with future-weather inputs. No transfer or real-time forecast performance is claimed.

## Solvers and scale

Normal equation is direct and needs no learning rate. The exercise uses the explicit inverse of X-transpose-X; a production solution would generally prefer a linear solve or QR/SVD for numerical stability. At alpha {s['selected_batch_alpha']:g}, batch GD needs **{s['solvers']['A']['batch_iterations']:,} iterations for A** and **{s['solvers']['B']['batch_iterations']:,} for B**, measured separately against the stated training-only convergence criterion.

Ordered SGD runs {s['solvers']['A']['sgd_epochs']:,} epochs at alpha {s['selected_sgd_alpha']:g}; it does not shuffle. A fixed step and fixed order leave nonzero coefficient error, so the final model is not claimed to equal the exact least-squares solution. Even if SGD happens to have a smaller held-out RMSE, this does not show that it optimized training least squares better. Its budget and rate were not selected from the test week.

For ten million rows and only six columns, normal-equation sufficient statistics can be accumulated in chunks: a 6-by-6 Gram matrix and a six-element target-product vector. The entire design matrix need not be kept in memory, though forming the Gram matrix still costs work and can worsen conditioning. Batch GD makes many full scans; SGD can stream row updates but needs careful rate and convergence control. When feature counts grow, quadratic Gram-matrix storage and cubic factorization cost favor iterative alternatives. No lecture-specific comparison is invented because the lecture matrix was not supplied.

## Cost curves

The objective is one half of the **sum** of squared training errors, not its mean, so batch updates have no division by sample count. The required batch rates are tested for 500 iterations and SGD rates for 50 epochs. Stable runs compete on their final Set A training cost; final model budgets are separate.

{md_table(rates)}

Batch curves record full cost after each simultaneous update. SGD curves record full-dataset cost once at the end of each epoch, hiding within-epoch row fluctuations; a smooth curve does not imply an averaged objective or smooth individual updates. Rates reported as divergent are excluded, while stability over a finite experiment is not a universal guarantee.

![Batch learning rates](batch_learning_rates.png)

![SGD learning rates](sgd_learning_rates.png)

## Residuals and test week

Residuals are actual minus clipped prediction: positive means underprediction, negative means overprediction. The largest hourly mean occurs at {int(hi.hour):02d}:00 ({hi['mean']:,.2f} kW), and the smallest at {int(lo.hour):02d}:00 ({lo['mean']:,.2f} kW). Each hour has just seven test observations, so hour-specific patterns are descriptive, not a validated seasonal correction. The test residuals were not used to tune a correction.

![Held-out predictions](actual_vs_predicted.png)

![Hourly residuals](residuals_vs_hour.png)

## Frontend and remaining work

The webpage loads saved Set B normal-equation weights and the training scaler. Hour enters as sine/cosine; weather inputs are radiation, temperature and cloud cover. Only negative output is clipped. A positive night prediction is a model limitation, not silently replaced with zero. The interface is an aggregate-plant historical scenario explorer.

Confirm plant coordinates and sensor averaging semantics before treating the location-check requirement as resolved. Review and understand the code and writing, and add genuine contributions and public repository/blog links for both group members. No external publication or fabricated commit history is included.
"""
    (RESULTS / "analysis.md").write_text(text, encoding="utf-8")
    print(f"Analysis saved. Daytime RMSE gap: {gap:.2f} kW ({100*gap/peak:.2f}% of training peak).")


if __name__ == "__main__":
    main()
