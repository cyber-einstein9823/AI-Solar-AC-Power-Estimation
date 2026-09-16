"""Solar / Lab: historical plant-scale scenario explorer for AML Assignment 1."""
import csv
import io
import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
RESULTS = ROOT / "results"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from prediction import FEATURES, load_weights, predict

PRESETS = {
    "Sunny midday": {"hour": 13, "sw_radiation": 950.0, "temp_2m": 35.0, "cloud_cover": 10.0},
    "Cloudy daytime": {"hour": 13, "sw_radiation": 400.0, "temp_2m": 28.0, "cloud_cover": 70.0},
    "Nighttime": {"hour": 0, "sw_radiation": 0.0, "temp_2m": 24.0, "cloud_cover": 20.0},
    "Custom": None,
}
INPUTS = {
    "hour": ("Hour of day", "Local time · Asia/Kolkata (UTC+05:30)", 0, 23, 1),
    "sw_radiation": ("Shortwave radiation", "Solar energy arriving at the surface · W/m²", 0.0, 1500.0, 1.0),
    "temp_2m": ("Air temperature", "Temperature at 2 metres · °C", -20.0, 60.0, 0.1),
    "cloud_cover": ("Cloud cover", "Fraction of sky covered by cloud · %", 0.0, 100.0, 1.0),
}

def initialize_inputs():
    if "scenario" not in st.session_state:
        st.session_state.scenario = "Sunny midday"
    for field, default in PRESETS["Sunny midday"].items():
        if field not in st.session_state:
            st.session_state[field] = default
        for suffix in ("slider", "number"):
            key = f"{field}_{suffix}"
            if key not in st.session_state:
                st.session_state[key] = st.session_state[field]

def sync_input(field, source):
    value = st.session_state[source]
    value = int(value) if field == "hour" else float(value)
    st.session_state[field] = value
    for suffix in ("slider", "number"):
        key = f"{field}_{suffix}"
        if key != source:
            st.session_state[key] = value
    st.session_state.scenario = "Custom"

def apply_preset():
    preset = PRESETS[st.session_state.scenario]
    if preset:
        for field, value in preset.items():
            st.session_state[field] = value
            st.session_state[f"{field}_slider"] = value
            st.session_state[f"{field}_number"] = value

def input_pair(field):
    title, description, minimum, maximum, step = INPUTS[field]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        slider_column, number_column = st.columns([3, 1.3], vertical_alignment="center")
        with slider_column:
            st.slider(f"{title} slider", min_value=minimum, max_value=maximum, step=step,
                      key=f"{field}_slider", label_visibility="collapsed",
                      on_change=sync_input, args=(field, f"{field}_slider"))
        with number_column:
            st.number_input(f"{title} value", min_value=minimum, max_value=maximum, step=step,
                            key=f"{field}_number", label_visibility="collapsed",
                            on_change=sync_input, args=(field, f"{field}_number"))
        st.caption(description)

def read_evidence():
    path = RESULTS / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path = RESULTS / "rmse_table.csv"
    metrics = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return summary, metrics

def get_rmse(metrics, column):
    if "model" not in metrics or column not in metrics:
        return None
    row = metrics.loc[metrics["model"] == "B normal", column]
    return float(row.iloc[0]) if len(row) else None

def inject_style():
    st.markdown("""
    <style>
    .stApp {background:#f5f7fa; color:#15283d;}
    .block-container {max-width:1220px; padding-top:2.1rem; padding-bottom:2rem;}
    h1,h2,h3 {color:#102b42; letter-spacing:-0.035em;}
    h1 {font-size:2.55rem !important; line-height:1.13 !important;}
    .stCaption, .stCaption p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:#53667a !important;}
    [data-testid="stMarkdownContainer"] p {color:#334e68;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:15px;}
    [data-testid="stMetricValue"] {font-weight:700; color:#0d5260;}
    [data-testid="stMetricLabel"] {color:#425b70;}
    [data-testid="stTabs"] button {font-weight:600;}
    [data-testid="stNumberInput"] input {font-weight:600;}
    .brand {font-size:.8rem; font-weight:800; letter-spacing:.19em; color:#0d5260; margin-bottom:1.2rem;}
    .brand span {color:#d7910b; margin-right:.5rem; font-size:1.2rem;}
    .footnote {font-size:.8rem; color:#53667a; margin-top:1.4rem;}
    @media(max-width:640px) {
      .block-container {padding-top:1.4rem; padding-left:1rem; padding-right:1rem;}
      h1 {font-size:2rem !important;}
    }
    </style>
    """, unsafe_allow_html=True)

def predictor_tab(weights, summary, metrics):
    initialize_inputs()
    st.radio("Start with a scenario", list(PRESETS), key="scenario",
             horizontal=True, on_change=apply_preset)
    hour = st.session_state.hour
    radiation = st.session_state.sw_radiation
    temperature = st.session_state.temp_2m
    cloud = st.session_state.cloud_cover
    raw, scaled, raw_kw, predicted_kw = predict(hour, radiation, temperature, cloud, weights)
    # Keep the actual prediction above controls, including on narrow screens.
    with st.container(border=True):
        output_column, weather_column, time_column = st.columns([2.2, 1.3, 1.3])
        with output_column:
            st.metric("Predicted AC power", f"{predicted_kw:,.1f} kW")
            st.caption(f"{predicted_kw / 1000:,.3f} MW · plant-level hourly average")
        with weather_column:
            st.metric("Solar radiation", f"{radiation:,.0f} W/m²")
            st.caption(f"{cloud:.0f}% cloud cover")
        with time_column:
            st.metric("Scenario time", f"{hour:02d}:00 IST")
            st.caption(f"{temperature:.1f} °C air temperature")
    controls, context = st.columns([2.05, 1], gap="large")
    with controls:
        st.subheader("Set the conditions")
        st.caption("Move any slider or type a value. The prediction updates automatically.")
        first, second = st.columns(2)
        with first:
            input_pair("hour")
            input_pair("temp_2m")
        with second:
            input_pair("sw_radiation")
            input_pair("cloud_cover")
        if (hour <= 5 or hour >= 20) and radiation > 50:
            st.warning("These inputs combine a nighttime hour with substantial sunlight. Check the scenario; the model still uses your exact values.")
        elif 10 <= hour <= 15 and radiation == 0:
            st.info("Zero sunlight around midday is an unusual scenario. The linear model is not constrained to predict zero at night or without radiation.")
        if raw_kw < 0:
            st.caption(f"The unconstrained linear model returned {raw_kw:,.1f} kW; the displayed prediction is clipped to zero, as required by the assignment.")
        if any(abs(float(value)) > 3 for value in scaled[:3]):
            st.caption("At least one weather input is more than three training standard deviations from its mean. Treat this extrapolation cautiously.")
    with context:
        st.subheader("What this estimate means")
        with st.container(border=True):
            st.markdown("**A model of one solar plant**")
            st.markdown("This is the saved **Set B normal-equation model**, using public weather and cyclic hour features. Nothing is retrained when you move a slider.")
            rmse = get_rmse(metrics, "rmse_daytime")
            if rmse is not None:
                st.metric("Daytime test RMSE", f"{rmse:,.1f} kW", help="A dataset-level error measure, not a confidence interval for this prediction.")
            peak = summary.get("train_peak_kw")
            if peak is not None:
                st.caption(f"Observed training peak: {float(peak):,.1f} kW. This is not a rated capacity or a prediction cap.")
        st.caption("Research prototype · historical data from May–June 2020. Not a live forecast, rooftop-system estimate, or financial recommendation.")
    st.warning("Location check remains unresolved: the public-weather coordinates are approximate and selected days show peak-time offsets. Use these predictions as experimental estimates only.", icon="⚠️")
    with st.expander("How the prediction is calculated"):
        st.write("The hour becomes sin(2π × hour / 24) and cos(2π × hour / 24). Features are standardized using training-only means and standard deviations, then a bias term is added. The saved coefficients produce power in kW; only negative outputs are clipped.")
        st.dataframe(pd.DataFrame({"Feature": FEATURES, "Input": raw, "Standardized input": scaled}), hide_index=True, width="stretch")
        st.caption(f"Unclipped output: {raw_kw:,.4f} kW. Weather units: W/m², °C, and percent.")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["hour_IST", "sw_radiation_W_m2", "temperature_C", "cloud_cover_percent", "raw_prediction_kW", "predicted_AC_kW"])
    writer.writerow([hour, radiation, temperature, cloud, raw_kw, predicted_kw])
    st.download_button("Download this scenario", buffer.getvalue(), "solar_scenario.csv", "text/csv")

def evidence_tab(summary, metrics):
    st.subheader("Evidence behind the estimate")
    st.caption("Evaluation uses a later, untouched week—not a random split. These figures and tables are loaded from the saved pipeline outputs.")
    first, second, third = st.columns(3)
    with first:
        st.metric("Training period", "27 days")
        st.caption(f"15 May–10 June 2020 · {summary.get('train_rows', '—')} usable hours")
    with second:
        st.metric("Test period", "7 days")
        st.caption(f"11–17 June 2020 · {summary.get('test_rows', '—')} usable hours")
    with third:
        rmse = get_rmse(metrics, "rmse_all_hours")
        st.metric("Set B · all-hours RMSE", f"{rmse:,.1f} kW" if rmse is not None else "Not available")
        st.caption("Normal equation · negative predictions clipped")
    st.markdown("#### Sensor features versus public weather")
    st.write("Set A uses on-site irradiation, module temperature and ambient temperature. Set B uses public shortwave radiation, air temperature and cloud cover. Both include cyclic hour features and use exactly the same chronological split.")
    if not metrics.empty:
        columns = [name for name in ("model", "rmse_all_hours", "rmse_daytime") if name in metrics]
        display = metrics[columns].rename(columns={"model": "Model", "rmse_all_hours": "All-hours RMSE (kW)", "rmse_daytime": "Daytime RMSE (kW)"})
        st.dataframe(display, hide_index=True, width="stretch", column_config={
            "All-hours RMSE (kW)": st.column_config.NumberColumn(format="%.2f"),
            "Daytime RMSE (kW)": st.column_config.NumberColumn(format="%.2f"),
        })
        st.caption("Daytime means measured on-site irradiation > 0, for both feature sets. Smaller RMSE is better; it is not a percentage accuracy score.")
    else:
        st.info("Run the training pipeline to generate the evaluation table.")
    path = RESULTS / "test_predictions.csv"
    if path.exists():
        predictions = pd.read_csv(path)
        time_column = next((name for name in ("datetime", "timestamp", "date_time") if name in predictions), None)
        fields = [name for name in ("ac_power", "pred_a_normal", "pred_b_normal") if name in predictions]
        if time_column and len(fields) == 3:
            st.markdown("#### Held-out week: observed and predicted power")
            chart = predictions[[time_column] + fields].copy()
            chart[time_column] = pd.to_datetime(chart[time_column])
            chart = chart.set_index(time_column).rename(columns={"ac_power": "Observed AC (kW)", "pred_a_normal": "On-site model A (kW)", "pred_b_normal": "Weather model B (kW)"})
            st.line_chart(chart, color=["#162d46", "#148879", "#d88b10"], height=330)
    with st.expander("Inspect diagnostics and learning curves"):
        for filename, caption in [
            ("residuals_vs_hour.png", "Hourly residuals · Set A normal equation"),
            ("batch_learning_rates.png", "Batch gradient descent · training-only learning-rate experiment"),
            ("sgd_learning_rates.png", "Stochastic gradient descent · training-only learning-rate experiment"),
        ]:
            path = RESULTS / filename
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")
    st.info("Scope: one plant, 34 calendar days, imperfect sensor coverage, approximate public-weather location, and a linear relationship. This study does not establish future or rooftop performance. Weather scenarios may be physically inconsistent; the interface does not silently change them.")

def main():
    st.set_page_config(page_title="Solar / Lab — Power Explorer", page_icon="☀️", layout="wide", initial_sidebar_state="collapsed")
    inject_style()
    st.markdown('<div class="brand"><span>☀</span> SOLAR / LAB</div>', unsafe_allow_html=True)
    st.title("Explore the power of sunlight.")
    st.markdown("An interactive look at how weather and time relate to one solar plant’s output.")
    st.caption("AML Assignment 1  /  Plant 1  /  NumPy linear regression  /  Historical scenario explorer")
    try:
        weights = load_weights()
        summary, metrics = read_evidence()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"The saved model or evaluation files could not be loaded: {exc}")
        st.info("From the project folder, run the documented training pipeline, then refresh this page. Model-loading errors are not replaced by demo values.")
        st.stop()
    predictor, evidence = st.tabs(["Power explorer", "Model evidence"])
    with predictor:
        predictor_tab(weights, summary, metrics)
    with evidence:
        evidence_tab(summary, metrics)
    st.markdown('<div class="footnote">Built for understanding, not certainty. Source data: Kaggle Solar Power Generation + Open-Meteo historical weather.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
