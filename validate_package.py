from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(name: str) -> list[dict[str, str]]:
    path = ROOT / "outputs" / "final_tables" / name
    require(path.is_file(), f"Missing table: {name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        "README.md",
        "CITATION.cff",
        "requirements.txt",
        "scripts/run_full_analysis.py",
        "scripts/run_validation_sensitivity.py",
        "tests/test_simulation_invariants.py",
        "outputs/validation_sensitivity.json",
        "outputs/final_tables/Table_2_Validation_Quantiles.csv",
        "outputs/final_tables/Table_3_Validation_Initialization.csv",
        "outputs/final_tables/Table_4_Brussels_Representative_Tradeoff.csv",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing required file: {relative}")

    raw_names = {
        "fims.csv",
        "311_service_requests_from_2020_to_present.csv",
    }
    packaged_names = {path.name.lower() for path in ROOT.rglob("*") if path.is_file()}
    require(not raw_names.intersection(packaged_names), "A raw source file is present")

    result = json.loads(
        (ROOT / "outputs" / "validation_sensitivity.json").read_text(
            encoding="utf-8"
        )
    )
    expected_quantiles = {
        "observed_closure_year": (59468, 7.0, 264.0, 526.0),
        "observed_creation_cohort_closed_by_cutoff": (61711, 8.0, 423.0, 562.0),
        "observed_creation_cohort_closed_by_year_end": (49086, 5.0, 74.0, 115.0),
        "simulator_exact_original": (52955, 32.0, 74.0, 99.0),
        "simulator_one_year_warmup": (56006, 125.0, 294.0, 344.0),
        "simulator_with_initial_backlog": (59468, 293.0, 1309.0, 1587.0),
    }
    for key, expected in expected_quantiles.items():
        observed = result[key]
        actual = (
            observed["n"],
            observed["p50"],
            observed["p90"],
            observed["p95"],
        )
        require(actual == expected, f"Unexpected validation values for {key}: {actual}")

    require(result["observed_initial_backlog"] == 51645, "Initial backlog mismatch")
    require(result["observed_end_backlog"] == 68221, "End backlog mismatch")
    require(
        result["simulator_with_initial_backlog"]["final_backlog"] == 68221,
        "Full-initialization accounting backlog mismatch",
    )

    validation_rows = read_csv("Table_2_Validation_Quantiles.csv")
    require(len(validation_rows) == 6, "Validation table row count mismatch")
    require(
        validation_rows[3]["p95_days"] == "99",
        "Exact original P95 differs from the audited result",
    )

    policy_rows = read_csv("Table_4_Brussels_Representative_Tradeoff.csv")
    require(len(policy_rows) == 2, "Policy table row count mismatch")
    require(policy_rows[0]["p95_days"] == "114.5", "FIFO P95 mismatch")
    require(policy_rows[1]["p95_days"] == "101.0", "Representative P95 mismatch")
    require(
        float(policy_rows[1]["disparity_cv"]) > float(policy_rows[0]["disparity_cv"]),
        "Expected dispersion trade-off is absent",
    )

    candidate_rows = read_csv("Table_5_Brussels_Policy_Performance.csv")
    require(len(candidate_rows) == 5, "Candidate-policy table row count mismatch")
    require(
        any(row["policy"] == "Candidate 10" for row in candidate_rows),
        "Representative candidate is absent from the candidate-policy table",
    )

    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))

    print("PASS: package structure and frozen numerical results are consistent")


if __name__ == "__main__":
    main()
