from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_analysis.py"
SPEC = importlib.util.spec_from_file_location("urban_service_analysis", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load the analysis module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frame(rows: list[dict]) -> pd.DataFrame:
    columns = ["id", "created", "closed", "dept", "group", "duration"]
    data = pd.DataFrame(rows, columns=columns)
    data["created"] = pd.to_datetime(data["created"])
    data["closed"] = pd.to_datetime(data["closed"])
    data["duration"] = pd.to_numeric(data["duration"])
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    empty = frame([])
    zero = MODULE.simulate(
        empty,
        "2024-01-01",
        "2024-01-03",
        (1.0, 0.0, 0.0, 0.0),
    )
    require(zero.closed == 0, "Zero-arrival fixture produced closures")
    require(zero.final_backlog == 0, "Zero-arrival fixture produced backlog")
    require(np.isnan(zero.p95), "Zero-arrival P95 should be undefined")

    requests = frame(
        [
            {
                "id": "a",
                "created": "2024-01-01",
                "closed": "2024-01-01",
                "dept": "D1",
                "group": "D1|G1",
                "duration": 0,
            },
            {
                "id": "b",
                "created": "2024-01-01",
                "closed": "2024-01-01",
                "dept": "D1",
                "group": "D1|G1",
                "duration": 0,
            },
            {
                "id": "c",
                "created": "2024-01-02",
                "closed": "2024-01-02",
                "dept": "D1",
                "group": "D1|G2",
                "duration": 0,
            },
        ]
    )
    first = MODULE.simulate(
        requests,
        "2024-01-01",
        "2024-01-03",
        (0.5, 0.5, 0.0, 0.0),
        capacity_scale=10.0,
        demand_scale=1.5,
        seed=20260924,
    )
    second = MODULE.simulate(
        requests,
        "2024-01-01",
        "2024-01-03",
        (0.5, 0.5, 0.0, 0.0),
        capacity_scale=10.0,
        demand_scale=1.5,
        seed=20260924,
    )
    require(first.asdict() == second.asdict(), "Fixed-seed behavior is not deterministic")
    require(first.final_backlog >= 0, "Backlog became negative")
    require(first.closed + first.final_backlog == 5, "Arrival conservation failed")
    require(np.isfinite(first.p95), "P95 is invalid for a non-empty closure sample")

    full_capacity = MODULE.simulate(
        requests,
        "2024-01-01",
        "2024-01-03",
        (1.0, 0.0, 0.0, 0.0),
        capacity_scale=10.0,
        seed=20260924,
    )
    require(full_capacity.closed == 3, "High capacity failed to close the fixture")
    require(full_capacity.final_backlog == 0, "High capacity left fixture backlog")

    print("PASS: simulation invariants and deterministic seed behavior")


if __name__ == "__main__":
    main()
