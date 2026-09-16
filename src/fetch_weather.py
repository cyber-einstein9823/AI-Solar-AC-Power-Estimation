"""Validate cached public weather and investigate alignment, offline by default.

Use --refresh only to deliberately replace the cached weather with a new archive
request. Historical reanalysis can change, so refresh also saves its raw response.
Diagnostics do not move timestamps or select settings using the held-out week.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from prepare import table1_rows, write_table1

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
API = "https://archive-api.open-meteo.com/v1/archive"
DOCS = "https://open-meteo.com/en/docs/historical-weather-api"
LAT, LON, TZ = 14.82, 78.28, "Asia/Kolkata"
START_DATE, END_DATE = "2020-05-15", "2020-06-17"
HOURLY_VARS = "shortwave_radiation,temperature_2m,cloud_cover"
DAYS = ["2020-05-20", "2020-06-01", "2020-06-15"]
W_PER_KW = 1000.0
PARAMS = {"latitude": LAT, "longitude": LON, "start_date": START_DATE,
          "end_date": END_DATE, "hourly": HOURLY_VARS, "timezone": TZ}


def fetch_openmeteo():
    response = requests.get(API, params=PARAMS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get("timezone") != TZ or payload.get("utc_offset_seconds") != 19800:
        raise ValueError("Archive timezone/UTC offset did not match requested Asia/Kolkata.")
    return payload


def payload_to_frame(payload):
    hourly = payload["hourly"]
    return pd.DataFrame({"datetime": pd.to_datetime(hourly["time"]),
                         "sw_radiation": hourly["shortwave_radiation"],
                         "temp_2m": hourly["temperature_2m"],
                         "cloud_cover": hourly["cloud_cover"]})


def validate_weather(frame):
    expected = pd.date_range(START_DATE, END_DATE + " 23:00:00", freq="h")
    dates = pd.DatetimeIndex(frame["datetime"])
    if dates.tz is not None or not dates.equals(expected):
        raise ValueError("Weather must contain exactly the 816 ordered local hourly timestamps.")
    values = frame[["sw_radiation", "temp_2m", "cloud_cover"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cached weather contains missing or nonfinite values.")
    if (frame.sw_radiation < 0).any() or not frame.cloud_cover.between(0, 100).all():
        raise ValueError("Weather radiation/cloud-cover values are outside physical bounds.")


def peak_table(merged):
    records = []
    for day in DAYS:
        subset = merged.loc[merged.datetime.dt.strftime("%Y-%m-%d") == day]
        sensor = subset.dropna(subset=["sensor_sw_wm2"])
        weather = subset.dropna(subset=["sw_radiation"])
        if sensor.empty or weather.empty:
            raise ValueError(f"No valid irradiation observations on {day}")
        sensor_peak, public_peak = sensor.loc[sensor.sensor_sw_wm2.idxmax()], weather.loc[weather.sw_radiation.idxmax()]
        records.append({"date": day, "sensor_peak_hour": sensor_peak.datetime.hour,
                        "public_peak_hour": public_peak.datetime.hour,
                        "public_minus_sensor_hours": public_peak.datetime.hour - sensor_peak.datetime.hour,
                        "sensor_peak_w_m2": float(sensor_peak.sensor_sw_wm2),
                        "public_peak_w_m2": float(public_peak.sw_radiation)})
    return pd.DataFrame(records)


def train_only_lag_diagnostics(sensor, weather):
    """Hypothetical timestamp shifts; never returned as modelling features.

    Both original source timestamps must lie inside training before any shift,
    so no June 11 or later observations can influence the comparison.
    """
    cutoff = pd.Timestamp("2020-06-11")
    lower = pd.Timestamp(START_DATE)
    train_sensor = sensor.loc[sensor.datetime.between(lower, cutoff, inclusive="left")]
    train_weather = weather.loc[weather.datetime.between(lower, cutoff, inclusive="left")]
    records = []
    for shift in range(-3, 4):
        shifted = train_weather[["datetime", "sw_radiation"]].copy()
        shifted["datetime"] += pd.Timedelta(hours=shift)
        pairs = train_sensor.merge(shifted, on="datetime", validate="one_to_one").dropna()
        pairs = pairs.loc[pairs.sensor_sw_wm2 > 0]
        records.append({"public_timestamp_shift_hours": shift, "paired_daytime_training_hours": len(pairs),
                        "daytime_correlation": float(pairs.sensor_sw_wm2.corr(pairs.sw_radiation)),
                        "daytime_radiation_mae_w_m2": float(np.abs(pairs.sensor_sw_wm2 - pairs.sw_radiation).mean())})
    return pd.DataFrame(records)


def plot_three_days(merged):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for ax, day in zip(axes, DAYS):
        subset = merged.loc[merged.datetime.dt.strftime("%Y-%m-%d") == day]
        ax.plot(subset.datetime, subset.sensor_sw_wm2, color="#14847a", label="On-site sensor (converted to W/m²)", lw=2)
        ax.plot(subset.datetime, subset.sw_radiation, color="#da8a21", label="Public weather (W/m²)", lw=2)
        ax.set_title(day, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Irradiance (W/m²)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Local clock hour (Asia/Kolkata assumption)")
    fig.suptitle("Sensor and public radiation: original timestamp labels", fontsize=14)
    path = RESULTS / "openmeteo_vs_sensor_3days.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Download a fresh archive response and replace cached weather deliberately.")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    cache = DATA / "plant1_openmeteo.csv"
    provenance_path = RESULTS / "openmeteo_provenance.json"
    if args.refresh:
        payload = fetch_openmeteo()
        weather = payload_to_frame(payload)
        validate_weather(weather)
        weather.to_csv(cache, index=False)
        (RESULTS / "openmeteo_response.json").write_text(json.dumps(payload), encoding="utf-8")
        provenance = {"mode": "refreshed_api_response", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                      "response_metadata": {key: value for key, value in payload.items() if key != "hourly"},
                      "original_cached_response_available": False}
    else:
        if not cache.exists():
            raise FileNotFoundError("Cached weather missing. Run this script with --refresh once with internet access.")
        weather = pd.read_csv(cache, parse_dates=["datetime"])
        validate_weather(weather)
        provenance = {"mode": "supplied_cached_csv", "original_cached_response_available": False,
                      "warning": "Original API JSON, retrieval time and returned grid-cell metadata were not supplied. The requested settings below document reproducibility intent, not independently verified cached-response metadata."}
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            if previous.get("csv_sha256") == hashlib.sha256(cache.read_bytes()).hexdigest() and previous.get("mode") == "refreshed_api_response":
                provenance = previous
    provenance.update({"endpoint": API, "requested_settings": PARAMS,
                       "csv_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "validated_rows": len(weather),
                       "documentation": DOCS})
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    sensor = pd.read_csv(DATA / "plant1_hourly.csv", parse_dates=["datetime"])[["datetime", "irradiation"]]
    sensor["sensor_sw_wm2"] = sensor.irradiation * W_PER_KW
    merged = sensor.merge(weather, on="datetime", validate="one_to_one")
    peaks = peak_table(merged)
    peaks.to_csv(RESULTS / "location_check_peaks.csv", index=False)
    lags = train_only_lag_diagnostics(sensor[["datetime", "sensor_sw_wm2"]], weather)
    lags.to_csv(RESULTS / "location_check_train_lags.csv", index=False)
    best = lags.loc[lags.daytime_correlation.idxmax()]
    zero = lags.loc[lags.public_timestamp_shift_hours == 0].iloc[0]
    metadata = {
        "status": "unresolved_provisional", "modelling_timestamp_shift_hours": 0,
        "reason": "One- and multiple-hour peak mismatches remain; exact site coordinates and raw sensor interval convention are not known. No test-guided shift is applied.",
        "timezone": TZ, "timezone_utc_offset_seconds": 19800,
        "timezone_verification": "Requested API timezone is explicit; original cached response metadata is unavailable unless refreshed. Plant timezone remains an assignment assumption.",
        "sensor_hourly_bin": "Left-labelled mean of matched observations at h:00, h:15, h:30, h:45 when present.",
        "public_radiation_convention": "Documentation describes a mean over the preceding hour; temperature and cloud cover are instantaneous quantities.",
        "train_only_lag_investigation": {"cutoff_exclusive": "2020-06-11", "unshifted_daytime_correlation": float(zero.daytime_correlation),
                                         "highest_correlation_timestamp_shift_hours": int(best.public_timestamp_shift_hours),
                                         "highest_daytime_correlation": float(best.daytime_correlation),
                                         "interpretation": "Diagnostic evidence only: a correlation maximum does not establish the correct site or interval convention."},
        "peaks": peaks.to_dict(orient="records"), "documentation": DOCS
    }
    (RESULTS / "location_check_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    peak_lines = ["| Date | Sensor peak | Public peak | Public minus sensor |", "|---|---:|---:|---:|"]
    peak_lines += [f"| {r.date} | {r.sensor_peak_hour:02d}:00 | {r.public_peak_hour:02d}:00 | {r.public_minus_sensor_hours:+d} h |" for r in peaks.itertuples()]
    report = f"""# Location and timestamp investigation\n\nStatus: **unresolved; Set B results are provisional**.\n\n{chr(10).join(peak_lines)}\n\nThe sensor is converted from kW/m² to W/m² by multiplying by 1,000. The cached public series contains all 816 local hourly labels. Its requested configuration is 14.82° N, 78.28° E, Asia/Kolkata, 15 May–17 June 2020. The original response was not supplied, so its returned grid cell and timezone metadata cannot be independently authenticated from the CSV. A deliberate `--refresh` preserves a fresh response and its metadata.\n\nThe plant file has no timezone metadata. Asia/Kolkata is an explicit assignment assumption, not a discovered timezone. The [Open-Meteo historical documentation]({DOCS}) describes shortwave radiation as the preceding-hour mean; the sensor aggregation here uses a left-labelled hour. Temperature and cloud cover have different, instantaneous semantics. This interval mismatch is a plausible contributor, but the raw sensor averaging convention is not documented, so relabelling all weather variables would not establish a correct alignment.\n\nA separate diagnostic tests shifts from −3 to +3 hours using only source observations dated before 11 June. Positive shifts move the public timestamps later. Daytime correlation is {zero.daytime_correlation:.4f} with no shift; the highest is {best.daytime_correlation:.4f} at a {int(best.public_timestamp_shift_hours):+d}-hour shift. Correlation is not proof of geographic or temporal alignment, and these alternatives are not fed to training. The 15 June plot is descriptive, not used to select a correction.\n\nNo clock shift or coordinate search was applied to improve the held-out results. Resolving this requires confirmed plant coordinates and sensor interval metadata, followed by a predeclared temporal convention and a fresh untouched evaluation period. Until then, this is a transparent limitation—not a claim that the location-check requirement is fully resolved.\n"""
    (RESULTS / "location_check.md").write_text(report, encoding="utf-8")
    preparation_path = RESULTS / "preparation_metadata.json"
    if preparation_path.exists():
        rows = table1_rows(json.loads(preparation_path.read_text(encoding="utf-8")))
        rows += [(f"Peak hours {r.date}", f"sensor {r.sensor_peak_hour:02d}:00; public {r.public_peak_hour:02d}:00", f"Public minus sensor {r.public_minus_sensor_hours:+d} h; unresolved") for r in peaks.itertuples()]
        write_table1(rows, RESULTS)
    plot_three_days(merged)
    print(peaks.to_string(index=False))
    print(lags.to_string(index=False))
    print("Saved offline diagnostics; cached weather values and modelling timestamps unchanged.")


if __name__ == "__main__":
    main()
