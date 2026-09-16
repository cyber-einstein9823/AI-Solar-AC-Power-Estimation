"""Task 2's four required plots and measured, reproducible observations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    hourly = pd.read_csv(ROOT / "data" / "plant1_hourly.csv", parse_dates=["datetime"])
    frame = hourly.dropna(subset=["ac_power", "dc_power", "irradiation", "ambient_temp", "module_temp"])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(frame.irradiation, frame.ac_power, s=15, alpha=.55, color="#167d8d", edgecolors="none")
    ax.set(xlabel="On-site irradiation (kW/m²)", ylabel="Aggregate AC power (kW)",
           title="Plant 1 | AC power and on-site irradiation")
    ax.grid(alpha=.18)
    save(fig, "plot1_ac_vs_irradiation.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    points = ax.scatter(frame.ambient_temp, frame.module_temp, c=frame.irradiation,
                        cmap="viridis", s=17, alpha=.75, edgecolors="none")
    fig.colorbar(points, ax=ax, label="On-site irradiation (kW/m²)")
    ax.set(xlabel="Ambient temperature (°C)", ylabel="Module temperature (°C)",
           title="Plant 1 | Panel heating in sunlight")
    ax.grid(alpha=.18)
    save(fig, "plot2_moduletemp_vs_ambienttemp.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(frame.dc_power, frame.ac_power, s=15, alpha=.55, color="#167d8d", edgecolors="none")
    ax.set(xlabel="Aggregate DC power (dataset-reported kW)", ylabel="Aggregate AC power (kW)",
           title="Plant 1 | AC versus DC (original scale retained)")
    ax.grid(alpha=.18)
    save(fig, "plot3_ac_vs_dc.png")

    hourly_mean = frame.groupby(frame.datetime.dt.hour).ac_power.mean()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(hourly_mean.index, hourly_mean, color="#167d8d", marker="o", markersize=4, linewidth=2)
    ax.fill_between(hourly_mean.index, hourly_mean, color="#167d8d", alpha=.12)
    ax.set(xlabel="Local hour (Asia/Kolkata)", ylabel="Mean aggregate AC power (kW)",
           title="Plant 1 | Average daily generation profile", xticks=range(0, 24, 2))
    ax.grid(alpha=.18)
    save(fig, "plot4_avg_power_by_hour.png")

    lit = frame.irradiation > 0
    ratio = frame.loc[frame.dc_power > 0, "ac_power"] / frame.loc[frame.dc_power > 0, "dc_power"]
    stats = {
        "hourly_rows": len(hourly), "complete_eda_rows": len(frame),
        "ac_irradiation_correlation": float(frame.ac_power.corr(frame.irradiation)),
        "ambient_module_correlation": float(frame.ambient_temp.corr(frame.module_temp)),
        "daylight_median_module_minus_ambient_c": float((frame.loc[lit, "module_temp"] - frame.loc[lit, "ambient_temp"]).median()),
        "ac_dc_correlation": float(frame.ac_power.corr(frame.dc_power)),
        "median_ac_dc_ratio_positive_dc": float(ratio.median()),
        "mean_profile_peak_hour": int(hourly_mean.idxmax()),
        "mean_profile_peak_kw": float(hourly_mean.max()),
    }
    text = f"""# Task 2 — Exploratory observations

These figures describe {len(frame)} complete hourly observations from 15 May–17 June 2020; {len(hourly)-len(frame)} missing hours are not filled with zero. Exploratory summaries cover the full period, but scaling, learning-rate selection and fitting use training dates only.

## 1. AC power versus irradiation

The hourly AC-power/irradiation correlation is {stats['ac_irradiation_correlation']:.3f}, with power generally rising as on-site sunlight increases. The scatter around that relation shows that irradiation alone is not a perfect description of output; temperature, inverter availability and changing conditions may contribute, but this plot cannot separate their effects.

![AC power versus irradiation](plot1_ac_vs_irradiation.png)

## 2. Module versus ambient temperature

Module and ambient temperatures are positively associated (correlation {stats['ambient_module_correlation']:.3f}), while brighter points tend to show hotter modules. During hours with positive irradiation, the median module-minus-ambient temperature is {stats['daylight_median_module_minus_ambient_c']:.2f} °C; heating under sunlight is consistent with that pattern rather than evidence of causation from ambient temperature alone.

![Module versus ambient temperature](plot2_moduletemp_vs_ambienttemp.png)

## 3. AC versus DC power

AC and DC power are closely associated (correlation {stats['ac_dc_correlation']:.5f}), but the median AC/DC ratio over positive-DC hours is only {stats['median_ac_dc_ratio_positive_dc']:.4f}. This unusual raw scale means the ratio must not be presented as a verified inverter conversion efficiency; the original values are retained and their units/scaling require source clarification before a physical efficiency claim.

![AC versus DC power](plot3_ac_vs_dc.png)

## 4. Average AC power by hour

The mean daily profile peaks at local hour {stats['mean_profile_peak_hour']:02d}:00 at {stats['mean_profile_peak_kw']:,.1f} kW. The curve rises during daylight and falls towards the night, supporting cyclic hour features, but a 34-day mean hides individual cloudy-day variation and is not an annual energy estimate.

![Average power by hour](plot4_avg_power_by_hour.png)
"""
    (RESULTS / "eda_observations.md").write_text(text, encoding="utf-8")
    (RESULTS / "eda_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
