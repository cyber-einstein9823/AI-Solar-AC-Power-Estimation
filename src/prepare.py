"""Prepare Plant 1 by aligning quarter-hour records before hourly averaging.

Keep all 816 hours, leaving missing hours explicit. Partial inverter coverage is
reported, never scaled up or treated as confirmed whole-plant generation.
"""
import json
from pathlib import Path

import pandas as pd

from load_data import load_raw

ROOT = Path(__file__).resolve().parents[1]
START = "2020-05-15 00:00:00"
END = "2020-06-17 23:00:00"
VALUES = ["ac_power", "dc_power", "ambient_temp", "module_temp", "irradiation"]


def prepare_frames(gen, sensor, start=START, end=END):
    """Return hourly values, per-hour coverage and auditable preparation counts."""
    gen, sensor = gen.copy(), sensor.copy()
    raw_examples = {"generation": str(gen["DATE_TIME"].iloc[0]), "sensor": str(sensor["DATE_TIME"].iloc[0])}
    raw_missing = {"generation": int(gen.isna().sum().sum()), "sensor": int(sensor.isna().sum().sum())}
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="raise")
    sensor["DATE_TIME"] = pd.to_datetime(sensor["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    if gen.duplicated(["DATE_TIME", "SOURCE_KEY"]).any():
        raise ValueError("Duplicate generation timestamp/inverter keys require investigation.")
    if sensor["DATE_TIME"].duplicated().any():
        raise ValueError("Expected one Plant 1 sensor row per timestamp.")
    grouped = gen.groupby("DATE_TIME")
    power = grouped[["AC_POWER", "DC_POWER"]].sum(min_count=1).rename(columns={"AC_POWER": "ac_power", "DC_POWER": "dc_power"})
    coverage = grouped["SOURCE_KEY"].nunique().rename("inverters_present")
    expected_inverters = int(gen["SOURCE_KEY"].nunique())
    power = power.join(coverage).reset_index()
    measured = sensor[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].rename(
        columns={"AMBIENT_TEMPERATURE": "ambient_temp", "MODULE_TEMPERATURE": "module_temp", "IRRADIATION": "irradiation"})
    audit = power.merge(measured, on="DATE_TIME", how="outer", indicator=True, validate="one_to_one")
    counts = audit["_merge"].value_counts()
    matched = audit.loc[audit["_merge"] == "both"].drop(columns="_merge").set_index("DATE_TIME").sort_index()
    calendar = pd.date_range(start, end, freq="h", name="datetime")
    # Every variable uses the same matched quarter-hour observation timestamps.
    hourly = matched[VALUES].resample("h", closed="left", label="left").mean().reindex(calendar)
    hourly_coverage = pd.DataFrame(index=calendar)
    hourly_coverage["matched_quarter_hours"] = matched["ac_power"].resample("h").size().reindex(calendar, fill_value=0)
    hourly_coverage["minimum_inverters_present"] = matched["inverters_present"].resample("h").min().reindex(calendar)
    hourly_coverage["complete_for_modelling"] = hourly.notna().all(axis=1)
    metadata = {
        "raw_rows": {"generation": len(gen), "sensor": len(sensor)},
        "raw_timestamp_examples": raw_examples,
        "timestamp_formats": {"generation": "%d-%m-%Y %H:%M", "sensor": "%Y-%m-%d %H:%M:%S"},
        "parsed_ranges": {"generation": [str(gen.DATE_TIME.min()), str(gen.DATE_TIME.max())], "sensor": [str(sensor.DATE_TIME.min()), str(sensor.DATE_TIME.max())]},
        "raw_missing_cells": raw_missing,
        "generation_timestamps": len(power), "sensor_timestamps": len(sensor),
        "generation_only_timestamps": int(counts.get("left_only", 0)), "sensor_only_timestamps": int(counts.get("right_only", 0)),
        "matched_quarter_hours": len(matched), "expected_inverters": expected_inverters,
        "timestamps_with_incomplete_inverter_coverage": int((coverage < expected_inverters).sum()),
        "hourly_rows": len(hourly), "hourly_rows_with_missing_values": int(hourly.isna().any(axis=1).sum()),
        "hourly_missing_cells": int(hourly.isna().sum().sum()), "complete_hourly_rows": int(hourly.notna().all(axis=1).sum()),
        "hours_with_1_to_3_matched_quarters": int(hourly_coverage.matched_quarter_hours.between(1, 3).sum()),
        "aggregation": "Sum observed inverter AC/DC per timestamp; inner-align sensor and generation quarter-hours; left-labelled hourly mean; reindex full calendar.",
        "missing_policy": "Preserve missing hourly values in CSV. Training drops incomplete model rows. No imputation, backfill or test-derived correction.",
        "coverage_caution": "Summed values can understate full-plant power when inverter records are absent. Partial hours/inverter coverage are retained and reported, not corrected by invented values.",
        "timezone_assumption": "Raw plant timestamps are naive; interpreted as local Asia/Kolkata by the assignment. Original timestamp timezone metadata is not provided."
    }
    return hourly.reset_index(), hourly_coverage.reset_index(), metadata


def table1_rows(meta):
    return [
        ("Raw Plant 1 generation rows", meta["raw_rows"]["generation"], "Inverter-level records"),
        ("Raw Plant 1 sensor rows", meta["raw_rows"]["sensor"], "One sensor record per timestamp"),
        ("Unique generation timestamps", meta["generation_timestamps"], "After inverter summation"),
        ("Generation-only timestamps", meta["generation_only_timestamps"], "Excluded before hourly averaging"),
        ("Sensor-only timestamps", meta["sensor_only_timestamps"], "Excluded before hourly averaging"),
        ("Total unmatched timestamps", meta["generation_only_timestamps"] + meta["sensor_only_timestamps"], "Outer-join audit only"),
        ("Matched quarter-hour timestamps", meta["matched_quarter_hours"], "Inner join used for all hourly variables"),
        ("Observed inverter identifiers", meta["expected_inverters"], "Expected coverage benchmark"),
        ("Timestamps below full inverter coverage", meta["timestamps_with_incomplete_inverter_coverage"], "Retained; not scaled up"),
        ("Hourly calendar rows", meta["hourly_rows"], "15 May-17 June, inclusive"),
        ("Hours with missing values", meta["hourly_rows_with_missing_values"], "Preserved as NaN; excluded during model fitting"),
        ("Missing hourly numeric cells", meta["hourly_missing_cells"], "Across five measured variables"),
        ("Complete hourly rows", meta["complete_hourly_rows"], "Before joining complete weather data"),
        ("Hours containing only 1-3 matched quarters", meta["hours_with_1_to_3_matched_quarters"], "Means use available matched measurements")
    ]


def write_table1(rows, results):
    table = pd.DataFrame(rows, columns=["measure", "value", "note"])
    table.to_csv(results / "preparation_table1.csv", index=False)
    lines = ["# Table 1 - preparation and timestamp checks", "", "| Measure | Value | Note |", "|---|---:|---|"]
    lines += [f"| {r.measure} | {r.value} | {r.note} |" for r in table.itertuples()]
    (results / "preparation_table1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    raw = load_raw(ROOT / "data")
    hourly, coverage, metadata = prepare_frames(raw["gen1"], raw["sensor1"])
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    hourly.to_csv(ROOT / "data" / "plant1_hourly.csv", index=False)
    coverage.to_csv(results / "preparation_hourly_coverage.csv", index=False)
    (results / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_table1(table1_rows(metadata), results)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
