"""Run after training: python -m unittest discover -s tests -p test_app.py -v"""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from prediction import FEATURES, load_weights, predict

class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "theta": np.array([100.0, 2.0, 3.0, -1.0, 30.0, 50.0]),
            "train_means": np.array([400.0, 25.0, 50.0, 0.0, 0.0]),
            "train_stds": np.array([100.0, 5.0, 20.0, 1.0, 1.0]),
            "feature_names": FEATURES,
        }

    def test_exact_saved_transform_and_prediction(self):
        raw, scaled, unbounded, clipped = predict(6, 600, 30, 10, self.weights)
        expected = np.array([600, 30, 10, 1.0, 0.0])
        np.testing.assert_allclose(raw, expected, atol=1e-12)
        expected_scaled = (expected - self.weights["train_means"]) / self.weights["train_stds"]
        np.testing.assert_allclose(scaled, expected_scaled, atol=1e-12)
        self.assertAlmostEqual(unbounded, np.r_[1.0, expected_scaled] @ self.weights["theta"])
        self.assertEqual(clipped, max(0, unbounded))

    def test_hour_and_temperature_change_prediction(self):
        baseline = predict(12, 600, 30, 10, self.weights)[2]
        self.assertNotEqual(baseline, predict(6, 600, 30, 10, self.weights)[2])
        self.assertNotEqual(baseline, predict(12, 600, 35, 10, self.weights)[2])

    def test_negative_is_clipped_not_forced_by_hour(self):
        self.assertGreater(predict(0, 0, 24, 20, self.weights)[3], 0)
        negative = dict(self.weights, theta=np.array([-1000., 0, 0, 0, 0, 0]))
        self.assertEqual(predict(12, 900, 35, 10, negative)[3], 0)

    def test_bad_inputs_rejected(self):
        cases = [(24, 100, 30, 50), (1.5, 100, 30, 50), (12, -1, 30, 50),
                 (12, 100, np.nan, 50), (12, 100, 30, 101)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                predict(*values, self.weights)

    def test_unicode_archive_and_bad_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, **self.weights)
            loaded = load_weights(path)
            np.testing.assert_array_equal(loaded["theta"], self.weights["theta"])
            np.savez(path, **dict(self.weights, train_stds=np.zeros(5)))
            with self.assertRaises(ValueError):
                load_weights(path)

    def test_pickle_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            np.savez(path, **dict(self.weights, feature_names=np.array(FEATURES, dtype=object)))
            with self.assertRaises(ValueError):
                load_weights(path)

@unittest.skipUnless((ROOT / "results" / "set_b_normal_weights.npz").exists(), "Run training before dashboard integration tests.")
class DashboardTests(unittest.TestCase):
    def make_app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0, list(app.exception))
        self.assertEqual(len(app.error), 0, list(app.error))
        return app

    def assert_output_matches_model(self, app):
        expected = predict(*(app.session_state[field] for field in
                             ("hour", "sw_radiation", "temp_2m", "cloud_cover")), load_weights())[3]
        self.assertEqual(app.metric[0].value, f"{expected:,.1f} kW")

    def test_first_render_has_real_prediction(self):
        app = self.make_app()
        self.assert_output_matches_model(app)
        self.assertEqual(len(app.slider), 4)
        self.assertEqual(len(app.number_input), 4)

    def test_each_typed_input_updates_matching_slider(self):
        app = self.make_app()
        for field, value in (("hour", 7), ("sw_radiation", 650.0), ("temp_2m", 31.2), ("cloud_cover", 45.0)):
            app.number_input(key=f"{field}_number").set_value(value).run()
            self.assertEqual(len(app.exception), 0)
            self.assertAlmostEqual(app.slider(key=f"{field}_slider").value, value)
            self.assertAlmostEqual(app.session_state[field], value)
            self.assertEqual(app.radio(key="scenario").value, "Custom")
            self.assert_output_matches_model(app)

    def test_each_slider_updates_matching_typed_input(self):
        app = self.make_app()
        for field, value in (("hour", 9), ("sw_radiation", 500.0), ("temp_2m", 29.5), ("cloud_cover", 63.0)):
            app.slider(key=f"{field}_slider").set_value(value).run()
            self.assertEqual(len(app.exception), 0)
            self.assertAlmostEqual(app.number_input(key=f"{field}_number").value, value)
            self.assertAlmostEqual(app.session_state[field], value)
            self.assert_output_matches_model(app)

    def test_all_presets_apply_all_fields(self):
        app = self.make_app()
        presets = {
            "Nighttime": (0, 0.0, 24.0, 20.0),
            "Cloudy daytime": (13, 400.0, 28.0, 70.0),
            "Sunny midday": (13, 950.0, 35.0, 10.0),
        }
        for name, values in presets.items():
            app.radio(key="scenario").set_value(name).run()
            self.assertEqual(len(app.exception), 0)
            for field, value in zip(("hour", "sw_radiation", "temp_2m", "cloud_cover"), values):
                self.assertAlmostEqual(app.slider(key=f"{field}_slider").value, value)
                self.assertAlmostEqual(app.number_input(key=f"{field}_number").value, value)
            self.assert_output_matches_model(app)

    def test_changing_hour_and_temperature_changes_visible_prediction(self):
        app = self.make_app()
        baseline = app.metric[0].value
        app.number_input(key="hour_number").set_value(8).run()
        self.assertNotEqual(app.metric[0].value, baseline)
        baseline = app.metric[0].value
        app.number_input(key="temp_2m_number").set_value(40.0).run()
        self.assertNotEqual(app.metric[0].value, baseline)

    def test_nighttime_sunlight_warns_without_changing_values(self):
        app = self.make_app()
        app.number_input(key="hour_number").set_value(0).run()
        self.assertEqual(app.slider(key="sw_radiation_slider").value, 950.0)
        self.assertTrue(any("nighttime" in warning.value for warning in app.warning))
        self.assert_output_matches_model(app)

if __name__ == "__main__":
    unittest.main()
