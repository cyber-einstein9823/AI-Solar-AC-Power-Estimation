"""The dashboard's pure, testable inference path; no fitting happens here."""
from pathlib import Path
import numpy as np

FEATURES = ["sw_radiation", "temp_2m", "cloud_cover", "sin_hour", "cos_hour"]
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "results" / "set_b_normal_weights.npz"

def _vector(value, name, size):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers in one dimension.")
    return array

def load_weights(path=None):
    """Read numeric arrays and Unicode feature names without permitting pickle."""
    path = Path(path) if path is not None else DEFAULT_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError("Saved model is missing. Run the training pipeline first.")
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = archive["feature_names"]
            if names.ndim != 1 or names.tolist() != FEATURES:
                raise ValueError("Saved feature names or their order do not match Set B.")
            weights = {
                "theta": _vector(archive["theta"], "theta", 6),
                "train_means": _vector(archive["train_means"], "train_means", 5),
                "train_stds": _vector(archive["train_stds"], "train_stds", 5),
                "feature_names": names.tolist(),
            }
    except KeyError as exc:
        raise ValueError(f"Saved model is missing array {exc}.") from exc
    if np.any(weights["train_stds"] <= 0):
        raise ValueError("Training standard deviations must all be positive.")
    return weights

def predict(hour, sw_radiation, temp_2m, cloud_cover, weights):
    """Return raw features, scaled features, raw kW and nonnegative kW.

    Hour uses Asia/Kolkata local time. Only negative final predictions are
    clipped: a nighttime preset does not override the linear model.
    """
    values = np.asarray([hour, sw_radiation, temp_2m, cloud_cover], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Every input must be a finite number.")
    hr, radiation, temperature, cloud = values
    if not 0 <= hr <= 23 or not hr.is_integer():
        raise ValueError("Hour must be a whole number from 0 to 23.")
    if not 0 <= radiation <= 1500:
        raise ValueError("Shortwave radiation must be between 0 and 1500 W/m².")
    if not -20 <= temperature <= 60:
        raise ValueError("Temperature must be between -20 and 60 °C.")
    if not 0 <= cloud <= 100:
        raise ValueError("Cloud cover must be between 0 and 100 percent.")
    theta = _vector(weights["theta"], "theta", 6)
    means = _vector(weights["train_means"], "train_means", 5)
    stds = _vector(weights["train_stds"], "train_stds", 5)
    if np.any(stds <= 0):
        raise ValueError("Training standard deviations must all be positive.")
    features = np.array([radiation, temperature, cloud,
                         np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)])
    scaled = (features - means) / stds
    raw_prediction = float(np.r_[1.0, scaled] @ theta)
    if not np.isfinite(raw_prediction):
        raise ValueError("The model did not produce a finite prediction.")
    return features, scaled, raw_prediction, max(0.0, raw_prediction)
