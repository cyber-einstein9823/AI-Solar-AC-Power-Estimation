"""Rebuild derived data and results offline from the included raw data/cache."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aml_assignment_matplotlib"))

def main():
    for name in ("prepare.py", "fetch_weather.py", "eda.py", "train_eval.py", "analysis.py"):
        print(f"\nRunning {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "app" / "html_server.py"), "--export-only"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], cwd=ROOT, check=True)
    print("\nResults and tests complete. Run Run_App.cmd to open the dashboard.")

if __name__ == "__main__":
    main()
