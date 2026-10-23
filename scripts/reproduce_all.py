"""Run the scientific workflow and verify frozen numerical tables offline."""
from pathlib import Path
import os
import subprocess
import sys
import json
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent if SCRIPTS.name == "scripts" else SCRIPTS


def compare_tables(output, reference):
    results = []
    for expected in sorted(reference.glob("*.csv")):
        actual = output / expected.name
        if not actual.exists():
            raise AssertionError(f"Missing reproduced table: {expected.name}")
        left = pd.read_csv(expected, keep_default_na=False)
        right = pd.read_csv(actual, keep_default_na=False)
        if list(left.columns) != list(right.columns) or left.shape != right.shape:
            raise AssertionError(f"Table structure differs: {expected.name}")
        for column in left.columns:
            a = pd.to_numeric(left[column], errors="coerce")
            b = pd.to_numeric(right[column], errors="coerce")
            numeric = a.notna() & b.notna()
            if numeric.any() and not np.allclose(a[numeric], b[numeric], rtol=1e-9, atol=1e-10):
                raise AssertionError(f"Numerical mismatch: {expected.name}, {column}")
            if not left.loc[~numeric, column].astype(str).equals(right.loc[~numeric, column].astype(str)):
                raise AssertionError(f"Label mismatch: {expected.name}, {column}")
        results.append(dict(file=expected.name, rows=len(left), columns=len(left.columns), matched=True))
    (output / "frozen_table_verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Verified {len(results)} frozen CSV tables", flush=True)


def main():
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output = Path(environment.get("URBAN_SERVICE_OUTPUT_DIR", ROOT / "outputs" / "reproduced"))
    output.mkdir(parents=True, exist_ok=True)
    environment["URBAN_SERVICE_OUTPUT_DIR"] = str(output)
    environment["URBAN_SERVICE_USE_CACHE"] = "0"
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-p", "test_*.py"], cwd=ROOT, env=environment, check=True)
    subprocess.run([sys.executable, "-u", str(SCRIPTS / "run_revision_analysis.py"), "--stage", "all"], cwd=ROOT, env=environment, check=True)
    # The main scientific run has now written the canonical cache from the frozen cells.
    environment["URBAN_SERVICE_USE_CACHE"] = "1"
    for script in ("additional_audits.py", "final_statistical_checks.py"):
        subprocess.run([sys.executable, "-u", str(SCRIPTS / script)], cwd=ROOT, env=environment, check=True)
    compare_tables(output, ROOT / "outputs" / "frozen")


if __name__ == "__main__":
    main()
