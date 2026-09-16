"""Export real saved coefficients, then serve the custom HTML site on localhost.

The exported site also works by opening web/index.html directly; no CDN/API is
used for inference or animation. Starting this server refreshes the model copy.
"""
import argparse
import csv
import json
import math
import shutil
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
WEB = APP / "web"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from prediction import load_weights

PLOTS = (
    "openmeteo_vs_sensor_3days.png", "actual_vs_predicted.png",
    "batch_learning_rates.png", "sgd_learning_rates.png", "residuals_vs_hour.png",
    "plot1_ac_vs_irradiation.png", "plot2_moduletemp_vs_ambienttemp.png",
    "plot3_ac_vs_dc.png", "plot4_avg_power_by_hour.png",
)


def export_plots(destination=None):
    """Copy exact pipeline figures into the portable website, without re-encoding."""
    folder = Path(destination) if destination else WEB / "plots"
    missing = [name for name in PLOTS if not (ROOT / "results" / name).is_file()]
    if missing:
        raise FileNotFoundError("Rebuild the pipeline; missing figures: " + ", ".join(missing))
    folder.mkdir(parents=True, exist_ok=True)
    for name in PLOTS:
        shutil.copy2(ROOT / "results" / name, folder / name)
    return list(PLOTS)


def export_model(destination=None):
    weights = load_weights()
    results = ROOT / "results"
    summary = json.loads((results / "run_summary.json").read_text(encoding="utf-8"))
    with (results / "rmse_table.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    for row in metrics:
        for key in ("rmse_all_hours", "rmse_daytime"):
            row[key] = float(row[key])
    groups = [[] for _ in range(24)]
    mins, maxs = [math.inf] * 3, [-math.inf] * 3
    with (ROOT / "data" / "plant1_hourly.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["ac_power"]:
                value = float(row["ac_power"])
                if math.isfinite(value):
                    groups[int(row["datetime"][11:13])].append(value)
    complete_train = set()
    with (ROOT / "data" / "plant1_hourly.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["datetime"][:10] <= "2020-06-10" and all(row[key] and math.isfinite(float(row[key])) for key in ("ac_power", "irradiation", "module_temp", "ambient_temp")):
                complete_train.add(row["datetime"])
    with (ROOT / "data" / "plant1_openmeteo.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["datetime"] in complete_train:
                for i,key in enumerate(("sw_radiation", "temp_2m", "cloud_cover")):
                    value=float(row[key]); mins[i]=min(mins[i],value); maxs[i]=max(maxs[i],value)
    with (results / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        series = [{"datetime": r["datetime"], "actual": float(r["ac_power"]), "sensor": float(r["pred_a_normal"]), "weather": float(r["pred_b_normal"])} for r in csv.DictReader(stream)]
    data = {key: value.tolist() if hasattr(value,"tolist") else value for key,value in weights.items()}
    data.update(summary=summary, metrics=metrics, test_series=series,
                hourly_profile=[sum(values)/len(values) if values else 0 for values in groups],
                feature_min=mins, feature_max=maxs)
    encoded = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    path = Path(destination) if destination else WEB / "model-data.js"
    path.write_text("// Generated from this project's saved NumPy weights and results. Do not edit by hand.\nwindow.SOLAR_MODEL = " + encoded.replace("</", "<\\/") + ";\n", encoding="utf-8")
    return data


class LocalHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--open", action="store_true", help="Open your default browser")
    parser.add_argument("--export-only", action="store_true")
    args=parser.parse_args()
    export_model()
    export_plots()
    if args.export_only:
        print("Exported real model data and all nine figures into app/web/")
        return
    handler=partial(LocalHandler, directory=str(WEB))
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler)
    url=f"http://localhost:{args.port}/"
    print(f"Solar Lab HTML dashboard: {url}",flush=True)
    print("Local only. Ctrl+C stops the server.",flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
