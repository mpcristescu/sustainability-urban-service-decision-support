from __future__ import annotations

import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260924


def load_fims(path: Path) -> pd.DataFrame:
    cols = [
        "fims_id",
        "createddate",
        "closeddate",
        "updateddate",
        "head_category",
        "status",
        "responsible_dep",
    ]
    frames = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=150_000, low_memory=False):
        frames.append(chunk)
    data = pd.concat(frames, ignore_index=True)
    for column in ["createddate", "closeddate", "updateddate"]:
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    data["fims_id"] = data["fims_id"].astype(str)
    data = (
        data.sort_values(["fims_id", "updateddate", "closeddate"], na_position="first")
        .drop_duplicates("fims_id", keep="last")
        .copy()
    )
    data["event"] = data["closeddate"].notna() & data["status"].eq("CLOSED")
    cutoff = pd.Timestamp("2026-09-24")
    data["enddate"] = data["closeddate"].where(data["event"], cutoff)
    data["duration"] = (data["enddate"] - data["createddate"]).dt.days.clip(lower=0)
    data["responsible_dep"] = data["responsible_dep"].astype(str)
    data["head_category"] = data["head_category"].astype(str)
    data["group"] = data["responsible_dep"] + "|" + data["head_category"]
    eligible = (
        data[data.createddate.between("2019-01-01", "2022-12-31")]
        .groupby("group")
        .size()
    )
    keep = set(eligible[eligible >= 250].index)
    data["group"] = [
        group if group in keep else f"{department}|OTHER"
        for group, department in zip(data.group, data.responsible_dep)
    ]
    return data.rename(
        columns={
            "fims_id": "id",
            "createddate": "created",
            "closeddate": "closed",
            "responsible_dep": "dept",
        }
    )


def quantiles(values) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "p50": None, "p90": None, "p95": None}
    return {
        "n": int(len(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def simulate_fifo(requests: pd.DataFrame, start: str, end: str, initialize_backlog: bool) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = requests[
        (requests["created"] <= end_ts)
        & ((requests["closed"].isna()) | (requests["closed"] >= start_ts))
    ].copy()
    arrivals = defaultdict(list)
    for row in selected.itertuples(index=False):
        arrivals[row.created].append((row.id, row.dept, row.group, row.created))
    observed_capacity = (
        requests[requests["closed"].between(start_ts, end_ts)]
        .groupby(["closed", "dept"])
        .size()
        .to_dict()
    )
    queues = defaultdict(deque)
    department_groups = defaultdict(list)
    initial_request_ids = set()
    if initialize_backlog:
        initial = selected[selected["created"] < start_ts]
        for row in initial.sort_values(["created", "id"]).itertuples(index=False):
            queues[row.group].append((row.created, row.id))
            initial_request_ids.add(row.id)
            if row.group not in department_groups[row.dept]:
                department_groups[row.dept].append(row.group)
    latencies = []
    initial_closed = 0
    backlog_daily = []
    for day in pd.date_range(start_ts, end_ts, freq="D"):
        for request_id, department, group, created in arrivals.get(day, []):
            queues[group].append((created, request_id))
            if group not in department_groups[department]:
                department_groups[department].append(group)
        for department, groups in department_groups.items():
            capacity = int(observed_capacity.get((day, department), 0))
            for _ in range(capacity):
                active = [group for group in groups if queues[group]]
                if not active:
                    break
                ages = [(day - queues[group][0][0]).days for group in active]
                selected_group = active[int(np.argmax(ages))]
                created, request_id = queues[selected_group].popleft()
                latencies.append(max((day - created).days, 0))
                if request_id in initial_request_ids:
                    initial_closed += 1
        backlog_daily.append(sum(len(queue) for queue in queues.values()))
    return {
        **quantiles(latencies),
        "initial_backlog": int(len(initial_request_ids)),
        "initial_backlog_closed": int(initial_closed),
        "mean_backlog": float(np.mean(backlog_daily)),
        "final_backlog": int(backlog_daily[-1]),
    }


def main(data_path: Path, output_path: Path) -> None:
    requests = load_fims(data_path)
    requests_from_2022 = requests[requests.created >= "2022-01-01"].copy()
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2023-12-31")
    closure_year = requests[requests.closed.between(start, end)]
    creation_cohort = requests[requests.created.between(start, end)]
    cohort_closed_by_cutoff = creation_cohort[creation_cohort.event]
    cohort_closed_by_year_end = creation_cohort[
        creation_cohort.closed.between(start, end)
    ]
    observed_initial_backlog = requests[
        (requests.created < start)
        & ((requests.closed.isna()) | (requests.closed >= start))
    ]
    observed_end_backlog = requests[
        (requests.created <= end)
        & ((requests.closed.isna()) | (requests.closed > end))
    ]
    result = {
        "definitions": {
            "closure_year": "Requests administratively closed during 2023, regardless of creation date.",
            "creation_cohort_closed_by_cutoff": "Requests created during 2023 and administratively closed by the 24 September 2026 cutoff.",
            "creation_cohort_closed_by_year_end": "Requests created and administratively closed during 2023.",
            "simulator_as_implemented": "FIFO replay with 2023 arrivals only and an empty queue on 1 January 2023.",
            "simulator_exact_original": "FIFO replay receiving only records created from 2022 onward, using closure opportunities from that restricted set, and starting with an empty queue on 1 January 2023.",
            "simulator_one_year_warmup": "FIFO replay receiving records created from 2022 onward, initialized with the subset still open on 1 January 2023.",
            "simulator_with_initial_backlog": "FIFO replay initialized with requests created before 2023 that were still open on 1 January 2023.",
        },
        "observed_closure_year": quantiles(closure_year.duration),
        "observed_creation_cohort_closed_by_cutoff": quantiles(cohort_closed_by_cutoff.duration),
        "observed_creation_cohort_closed_by_year_end": quantiles(cohort_closed_by_year_end.duration),
        "observed_initial_backlog": int(len(observed_initial_backlog)),
        "observed_end_backlog": int(len(observed_end_backlog)),
        "simulator_as_implemented": simulate_fifo(requests, str(start.date()), str(end.date()), False),
        "simulator_exact_original": simulate_fifo(requests_from_2022, str(start.date()), str(end.date()), False),
        "simulator_one_year_warmup": simulate_fifo(requests_from_2022, str(start.date()), str(end.date()), True),
        "simulator_with_initial_backlog": simulate_fifo(requests, str(start.date()), str(end.date()), True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: run_validation_sensitivity.py FIMS.csv OUTPUT.json"
        )
    main(Path(sys.argv[1]), Path(sys.argv[2]))
