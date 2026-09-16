"""Offline tests for matched timestamp aggregation and weather diagnostics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare import prepare_frames
from load_data import load_raw
from fetch_weather import payload_to_frame, validate_weather, train_only_lag_diagnostics


class PreparationTests(unittest.TestCase):
    def synthetic_frames(self):
        generation = pd.DataFrame({
            "DATE_TIME": ["15-05-2020 00:00", "15-05-2020 00:00", "15-05-2020 00:15"],
            "SOURCE_KEY": ["A", "B", "A"], "AC_POWER": [10., 20., 500.], "DC_POWER": [11., 22., 550.]})
        sensor = pd.DataFrame({
            "DATE_TIME": ["2020-05-15 00:00:00", "2020-05-15 00:30:00"],
            "AMBIENT_TEMPERATURE": [25., 99.], "MODULE_TEMPERATURE": [30., 99.], "IRRADIATION": [.1, .9]})
        return generation, sensor

    def test_inner_match_precedes_average_and_missing_hours_survive(self):
        hourly, coverage, meta = prepare_frames(*self.synthetic_frames(), end="2020-05-15 01:00:00")
        self.assertEqual(len(hourly), 2)
        self.assertEqual(hourly.iloc[0].ac_power, 30.)
        self.assertEqual(hourly.iloc[0].ambient_temp, 25.)
        self.assertTrue(hourly.iloc[1].drop("datetime").isna().all())
        self.assertEqual(coverage.iloc[0].matched_quarter_hours, 1)
        self.assertEqual(meta["generation_only_timestamps"], 1)
        self.assertEqual(meta["sensor_only_timestamps"], 1)
        self.assertEqual(meta["timestamps_with_incomplete_inverter_coverage"], 1)

    def test_duplicate_inverter_rows_rejected(self):
        generation, sensor = self.synthetic_frames()
        generation = pd.concat([generation, generation.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate generation"):
            prepare_frames(generation, sensor)

    def test_duplicate_sensor_timestamps_rejected(self):
        generation, sensor = self.synthetic_frames()
        sensor = pd.concat([sensor, sensor.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one Plant 1 sensor"):
            prepare_frames(generation, sensor)

    def test_real_data_audit(self):
        raw = load_raw(ROOT / "data")
        hourly, _, meta = prepare_frames(raw["gen1"], raw["sensor1"])
        self.assertEqual(len(hourly), 816)
        self.assertEqual(meta["matched_quarter_hours"], 3157)
        self.assertEqual(meta["hourly_rows_with_missing_values"], 20)
        self.assertEqual(meta["complete_hourly_rows"], 796)
        self.assertEqual(meta["timestamps_with_incomplete_inverter_coverage"], 101)
        self.assertEqual(meta["generation_only_timestamps"] + meta["sensor_only_timestamps"], 26)


class WeatherTests(unittest.TestCase):
    def test_supplied_cache_grid_and_values(self):
        weather = pd.read_csv(ROOT / "data" / "plant1_openmeteo.csv", parse_dates=["datetime"])
        validate_weather(weather)
        with self.assertRaises(ValueError):
            validate_weather(weather.iloc[:-1])

    def test_payload_names(self):
        payload = {"hourly": {"time": ["2020-05-15T00:00"], "shortwave_radiation": [0.],
                              "temperature_2m": [29.5], "cloud_cover": [13.]}}
        frame = payload_to_frame(payload)
        self.assertEqual(list(frame.columns), ["datetime", "sw_radiation", "temp_2m", "cloud_cover"])

    def test_lag_diagnostics_do_not_use_test_observations(self):
        hours = pd.date_range("2020-06-09", "2020-06-12", freq="h")
        signal = np.maximum(0., 600 * np.sin(np.pi * (hours.hour.to_numpy() - 6) / 12))
        sensor = pd.DataFrame({"datetime": hours, "sensor_sw_wm2": signal})
        weather = pd.DataFrame({"datetime": hours, "sw_radiation": signal * .95})
        original = train_only_lag_diagnostics(sensor, weather)
        sensor.loc[sensor.datetime >= "2020-06-11", "sensor_sw_wm2"] = 999999.
        weather.loc[weather.datetime >= "2020-06-11", "sw_radiation"] = -999999.
        pd.testing.assert_frame_equal(original, train_only_lag_diagnostics(sensor, weather))


if __name__ == "__main__":
    unittest.main()
