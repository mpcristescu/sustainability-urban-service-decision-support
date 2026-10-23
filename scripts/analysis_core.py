"""Censored administrative lifecycles and temporally separated queue replays.

The simulator conditions on observed closure opportunities. It is not a staffing
model. No future closure times are used in the priority score.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_packages = Path(__file__).resolve().parent / "scientific_packages"
if _packages.exists():
    sys.path.insert(0, ("\\\\?\\" if os.name == "nt" else "") + str(_packages))

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import chi2, norm

EPOCH = pd.Timestamp("1970-01-01")
SEED = 20261020


def day(value):
    return int((pd.Timestamp(value).normalize() - EPOCH).days)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def date_days(series):
    parsed = pd.to_datetime(series, errors="coerce", format="mixed").dt.normalize()
    return ((parsed - EPOCH).dt.days).to_numpy(dtype=float)


def canonicalize(d, cutoff, source):
    d = d.copy()
    d["id"] = d["id"].astype(str)
    for name in ("dept", "category", "status"):
        d[name] = d[name].fillna("MISSING").astype(str).str.strip()
    created = date_days(d["created"])
    closed = date_days(d["closed"])
    valid_creation = np.isfinite(created) & (created <= day(cutoff))
    # Validate full timestamps before coarsening to civil days.
    # Same-day inversions must not become zero-day closure events.
    precise_created=pd.to_datetime(d["created"],errors="coerce",format="mixed")
    precise_closed=pd.to_datetime(d["closed"],errors="coerce",format="mixed")
    negative=(precise_closed<precise_created).fillna(False).to_numpy()
    negative_day=np.isfinite(closed)&(closed<created)
    status_closed = d.status.str.upper().isin(["CLOSED", "CLOSE"]).to_numpy()
    valid_closed = np.isfinite(closed) & ~negative & (closed <= day(cutoff))
    # Closure timestamp is the endpoint. Current snapshot status is not an event log.
    ambiguous = status_closed & (~np.isfinite(closed) | negative)
    future_closed = np.isfinite(closed) & (closed > day(cutoff))
    audit = dict(source=source, input_unique_rows=len(d), invalid_creation=int((~valid_creation).sum()),
                 negative_closure=int(negative.sum()), same_day_negative_timestamp=int((negative&~negative_day).sum()),
                 missing_closure=int((~np.isfinite(closed)).sum()),
                 closed_status_missing_or_invalid_date=int(ambiguous.sum()),
                 closure_timestamp_nonclosed_status=int((valid_closed & ~status_closed).sum()),
                 closure_after_cutoff=int(future_closed.sum()),
                 snapshot_cutoff=cutoff, status_counts=d.status.value_counts().to_dict())
    d["created"] = np.where(valid_creation, created, -1).astype(np.int32)
    d["closed"] = np.where(valid_closed, closed, -1).astype(np.int32)
    d["event"] = valid_closed
    d["ambiguous"] = ambiguous
    d["duration"] = np.where(valid_closed, closed - created, day(cutoff) - created)
    d = d.loc[valid_creation].reset_index(drop=True)
    d["duration"] = d.duration.astype(np.int32)
    audit["primary_rows"] = int((~d.ambiguous).sum())
    return d, audit


def load_brussels(path, cutoff="2026-09-24"):
    cols = ["fims_id", "createddate", "closeddate", "updateddate", "responsible_dep", "head_category", "status"]
    raw = pd.read_csv(path, usecols=cols, low_memory=False, dtype={"fims_id": str})
    n_raw = len(raw)
    dup = int(raw.duplicated("fims_id", keep=False).sum())
    for name in ("createddate", "closeddate", "updateddate"):
        raw[name] = pd.to_datetime(raw[name], errors="coerce", format="mixed")
    raw = raw.sort_values(["fims_id", "updateddate", "closeddate"], na_position="first").drop_duplicates("fims_id", keep="last")
    d = raw.rename(columns={"fims_id": "id", "createddate": "created", "closeddate": "closed", "responsible_dep": "dept", "head_category": "category"})
    d, audit = canonicalize(d, cutoff, "FixMyStreet Brussels")
    audit.update(raw_rows=n_raw, duplicate_identifier_rows=dup, raw_sha256=sha256(path))
    return d, audit


def load_nyc(directory, cutoff="2026-10-20"):
    paths = sorted(Path(directory).glob("nyc_20*.csv")) + sorted(Path(directory).glob("nyc_inherited_*.csv"))
    parts = [pd.read_csv(p, dtype=str, keep_default_na=True).rename(columns={"unique_key": "id", "created_date": "created", "closed_date": "closed", "agency": "dept", "complaint_type": "category"}) for p in paths]
    raw = pd.concat(parts, ignore_index=True)
    duplicates = int(raw.duplicated("id", keep=False).sum())
    if duplicates:
        # Identical duplicate IDs across the historical/current boundary are audited.
        conflicted = raw.groupby("id")[["created", "closed", "dept", "category"]].nunique(dropna=False).max(axis=1)
        if (conflicted > 1).any():
            raise AssertionError("NYC duplicate identifiers have conflicting source representations")
    raw = raw.drop_duplicates("id", keep="last")
    d, audit = canonicalize(raw, cutoff, "NYC 311 current and historical official extracts")
    audit.update(download_rows=sum(len(p) for p in parts), duplicate_identifier_rows=duplicates,
                 source_files=[dict(name=p.name, sha256=sha256(p), bytes=p.stat().st_size) for p in paths])
    current = d[d.created.between(day("2023-01-01"), day("2025-12-31"))]
    audit["creation_year_counts"] = current.groupby(pd.to_datetime(current.created, unit="D").dt.year).size().to_dict()
    return d, audit


def stable_canonical_order(d):
    keys=["dept","category","created","closed","event","ambiguous","duration","status"]
    return d.sort_values(keys,kind="stable").reset_index(drop=True)


def load_derived_cells(path):
    cells=pd.read_csv(path,dtype={"dept":str,"category":str,"status":str})
    if (cells["count"]<=0).any():raise AssertionError("Nonpositive derived multiplicity")
    d=cells.loc[np.repeat(cells.index.to_numpy(),cells["count"].to_numpy())].drop(columns="count").reset_index(drop=True)
    for name in ("created","closed","duration"):d[name]=d[name].astype(np.int32)
    for name in ("event","ambiguous"):d[name]=d[name].astype(bool)
    d["id"]=[f"derived_{i}" for i in range(len(d))]
    return stable_canonical_order(d)


def km_curve(duration, event):
    t = np.asarray(duration, dtype=np.int32)
    e = np.asarray(event, dtype=bool)
    if len(t) == 0:
        return pd.DataFrame(columns=["t", "n", "d", "survival", "greenwood", "se"])
    counts = np.bincount(t)
    deaths = np.bincount(t[e], minlength=len(counts))
    times = np.flatnonzero(counts)
    risk = len(t) - np.cumsum(counts)[times] + counts[times]
    dd = deaths[times]
    sf = np.cumprod(1.0 - dd / risk)
    terms = np.divide(dd, risk * (risk - dd), out=np.zeros_like(dd, dtype=float), where=risk > dd)
    gw = np.cumsum(terms)
    return pd.DataFrame(dict(t=times, n=risk, d=dd, survival=sf, greenwood=gw, se=sf*np.sqrt(gw)))


def km_at(curve, horizon):
    loc = np.searchsorted(curve.t.to_numpy(), horizon, side="right") - 1
    if loc < 0:
        return 1.0, 1.0, 1.0
    row = curve.iloc[loc]
    s = float(row.survival)
    if not 0 < s < 1:
        return s, s, s
    z = norm.ppf(.975)
    se_loglog = math.sqrt(float(row.greenwood)) / abs(math.log(s))
    center = math.log(-math.log(s))
    low = math.exp(-math.exp(center + z * se_loglog))
    high = math.exp(-math.exp(center - z * se_loglog))
    return s, low, high


def rmst(curve, horizon=90):
    # Integrate the right-continuous KM curve on [0, horizon].
    previous, survival, area = 0, 1.0, 0.0
    for row in curve.itertuples(index=False):
        if row.t >= horizon:
            break
        area += (row.t - previous) * survival
        previous, survival = row.t, row.survival
    return float(area + (horizon - previous) * survival)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    out = np.empty(len(p))
    out[order] = np.minimum(adjusted, 1)
    return out


def logrank(t1, e1, t2, e2):
    max_t = max(int(np.max(t1)), int(np.max(t2))) + 1
    n1 = np.bincount(t1, minlength=max_t)
    n2 = np.bincount(t2, minlength=max_t)
    d1 = np.bincount(np.asarray(t1)[np.asarray(e1, bool)], minlength=max_t)
    d2 = np.bincount(np.asarray(t2)[np.asarray(e2, bool)], minlength=max_t)
    y1 = np.cumsum(n1[::-1])[::-1]
    y2 = np.cumsum(n2[::-1])[::-1]
    # Float before products: large risk-set products can overflow int64.
    y1,y2 = y1.astype(float),y2.astype(float)
    y, dd = y1+y2, (d1+d2).astype(float)
    expected = np.divide(dd*y1, y, out=np.zeros_like(y, float), where=y>0)
    var = np.divide(y1*y2*dd*(y-dd), y*y*(y-1), out=np.zeros_like(y, float), where=y>1)
    variance = float(var.sum())
    statistic = float((d1.sum()-expected.sum())**2/variance) if variance>0 else 0.0
    return statistic, float(chi2.sf(statistic, 1))


def group_model(d, dev_start, dev_end, threshold):
    development = d[(~d.ambiguous) & d.created.between(day(dev_start), day(dev_end))].copy()
    counts = development.groupby(["dept", "category"]).size()
    eligible = set(counts[counts >= threshold].index)
    labels = np.asarray([f"{a}|{b}" if (a,b) in eligible else f"{a}|OTHER" for a,b in zip(d.dept,d.category)])
    names, group_id = np.unique(labels, return_inverse=True)
    depts = sorted(d.dept.unique())
    dept_map = {name:i for i,name in enumerate(depts)}
    group_dept = np.array([dept_map[g.split("|",1)[0]] for g in names], np.int32)
    cutoff = day(dev_end)
    risk = np.zeros(len(names), float)
    dd = development.copy()
    dd["group_id"] = group_id[development.index.to_numpy()]
    for gid, x in dd.groupby("group_id"):
        event = x.event.to_numpy() & (x.closed.to_numpy() <= cutoff)
        duration = np.where(event, x.closed-x.created, cutoff-x.created).astype(np.int32)
        risk[gid] = rmst(km_curve(duration, event), 90)
    global_risk = float(np.median(risk[risk>0])) if (risk>0).any() else 45.0
    risk[risk == 0] = global_risk
    return dict(names=names, group_id=group_id.astype(np.int32), group_dept=group_dept,
                departments=depts, risk=risk, threshold=int(threshold), development_start=dev_start,
                development_end=dev_end, eligible_groups=len(eligible), pooled_development_requests=int(counts[counts<threshold].sum()),
                development_requests=len(development))


@dataclass
class Replay:
    created: np.ndarray
    group: np.ndarray
    original_index: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    group_dept: np.ndarray
    capacities: np.ndarray
    risk: np.ndarray
    start: int
    end: int
    group_names: np.ndarray


def prepare_replay(d, model, start, end, initialization="full", source_since=None, demand_scale=1.0, capacity_scale=1.0, seed=SEED, include_ambiguous=False):
    sd, ed = day(start), day(end)
    valid = ~d.ambiguous.to_numpy() if not include_ambiguous else np.ones(len(d), bool)
    if source_since is not None:
        valid &= d.created.to_numpy() >= day(source_since)
    base = np.flatnonzero(valid)
    x = d.iloc[base]
    pending = (x.created.to_numpy() <= ed) & ((~x.event.to_numpy()) | (x.closed.to_numpy() >= sd))
    if initialization == "empty":
        pending &= x.created.to_numpy() >= sd
    elif initialization == "one_year":
        pending &= x.created.to_numpy() >= day(f"{pd.Timestamp(start).year-1}-01-01")
    elif initialization != "full":
        raise ValueError(initialization)
    rows = base[pending]
    if demand_scale != 1:
        inherited = rows[d.created.to_numpy()[rows] < sd]
        arrivals = rows[d.created.to_numpy()[rows] >= sd]
        rng = np.random.default_rng(seed)
        if demand_scale >= 1:
            extras = rng.choice(arrivals, size=int(round(len(arrivals)*(demand_scale-1))), replace=True)
            rows = np.r_[rows, extras]
        else:
            rows = np.r_[inherited, rng.choice(arrivals, size=int(round(len(arrivals)*demand_scale)), replace=False)]
    group = model["group_id"][rows]
    created = d.created.to_numpy()[rows]
    order = np.lexsort((np.arange(len(rows)), created, group))
    rows, created, group = rows[order], created[order], group[order]
    counts = np.bincount(group, minlength=len(model["names"]))
    stops = np.cumsum(counts).astype(np.int64)
    starts = np.r_[0, stops[:-1]].astype(np.int64)
    closure_rows = valid & d.event.to_numpy() & d.closed.between(sd,ed).to_numpy()
    dep = model["group_dept"][model["group_id"][closure_rows]]
    flat = (d.closed.to_numpy()[closure_rows]-sd)*len(model["departments"]) + dep
    capacity = np.bincount(flat, minlength=(ed-sd+1)*len(model["departments"])).reshape(ed-sd+1,-1)
    capacity = np.rint(capacity*capacity_scale).astype(np.int32)
    return Replay(created.astype(np.int32),group.astype(np.int32),rows,starts,stops,
                  model["group_dept"],capacity,model["risk"],sd,ed,model["names"])


@njit(cache=True)
def queue_kernel(created, group, starts, stops, group_dept, capacity, risk, sd, weights):
    ng = len(starts)
    head, tail = starts.copy(), starts.copy()
    closed_day = np.full(len(created), -1, np.int32)
    sum_delay = np.zeros(ng, np.float64)
    count_closed = np.zeros(ng, np.int64)
    backlog = np.zeros(capacity.shape[0], np.int64)
    unused = np.zeros(capacity.shape[0], np.int64)
    lag_mean = np.zeros(ng, np.float64)
    for t in range(capacity.shape[0]):
        today = sd+t
        for g in range(ng):
            while tail[g] < stops[g] and created[tail[g]] <= today:
                tail[g] += 1
            lag_mean[g] = sum_delay[g]/count_closed[g] if count_closed[g] else 0.0
        for dep in range(capacity.shape[1]):
            for slot in range(capacity[t,dep]):
                total, max_age, max_mean, max_risk = 0, 1, 1.0, 1.0
                for g in range(ng):
                    if group_dept[g] == dep and tail[g] > head[g]:
                        total += tail[g]-head[g]
                        max_age = max(max_age, today-created[head[g]])
                        max_mean = max(max_mean, lag_mean[g])
                        max_risk = max(max_risk, risk[g])
                if total == 0:
                    unused[t] += capacity[t,dep]-slot
                    break
                best, best_score, best_age = -1, -1e100, -1
                for g in range(ng):
                    if group_dept[g] == dep and tail[g] > head[g]:
                        age = today-created[head[g]]
                        score = (weights[0]*age/max_age + weights[1]*(tail[g]-head[g])/total
                                 + weights[2]*risk[g]/max_risk + weights[3]*lag_mean[g]/max_mean)
                        if score > best_score+1e-12 or (abs(score-best_score)<=1e-12 and age>best_age):
                            best, best_score, best_age = g, score, age
                index = head[best]
                closed_day[index] = today
                sum_delay[best] += today-created[index]
                count_closed[best] += 1
                head[best] += 1
        for g in range(ng):
            backlog[t] += tail[g]-head[g]
    return closed_day, backlog, unused


def replay(reconstruction, weights):
    r = reconstruction
    return queue_kernel(r.created,r.group,r.starts,r.stops,r.group_dept,r.capacities,r.risk,r.start,np.asarray(weights,float))


def policy_metrics(r, result, maturity=90, minimum_group_closures=20, common_groups=None):
    cd, backlog, unused = result
    closed = cd >= 0
    latency = cd[closed]-r.created[closed]
    mature = (r.created>=r.start) & (r.created<=r.end-maturity)
    long_wait = (~closed) | ((cd-r.created)>maturity)
    ng = len(r.starts)
    group_n = np.bincount(r.group[closed], minlength=ng)
    medians = np.full(ng,np.nan)
    for g in np.flatnonzero(group_n >= minimum_group_closures):
        medians[g] = np.median(cd[(r.group==g)&closed]-r.created[(r.group==g)&closed])
    valid = np.isfinite(medians)
    if common_groups is not None:
        valid &= common_groups
    cv = float(np.std(medians[valid])/np.mean(medians[valid])) if valid.any() and np.mean(medians[valid])>0 else np.nan
    open_ages = r.end-r.created[~closed]
    q = np.quantile(latency,[.5,.9,.95]) if len(latency) else [np.nan]*3
    metric = dict(p50=float(q[0]),p90=float(q[1]),p95=float(q[2]),
                  closed=int(closed.sum()),initial_backlog=int((r.created<r.start).sum()),
                  arrivals=int((r.created>=r.start).sum()),mean_backlog=float(np.mean(backlog)),final_backlog=int(backlog[-1]),
                  unused_slots=int(unused.sum()),long_wait_share=float(np.mean(long_wait[mature])) if mature.any() else np.nan,
                  mature_cohort_n=int(mature.sum()),disparity_cv=cv,dispersion_groups=int(valid.sum()),
                  terminal_open_p95=float(np.quantile(open_ages,.95)) if len(open_ages) else 0.0,
                  terminal_open_over90_share=float(np.mean(open_ages>90)) if len(open_ages) else 0.0)
    assert metric["initial_backlog"]+metric["arrivals"]-metric["closed"] == metric["final_backlog"]
    assert (latency>=0).all() and (backlog>=0).all()
    return metric, medians, group_n


def nondominated(values):
    a = np.asarray(values,float)
    return np.array([not np.any(np.all(a<=row,axis=1)&np.any(a<row,axis=1)) for row in a])


def candidate_search(r, n=32, seed=SEED):
    baseline = [("FIFO",[1,0,0,0]),("Queue pressure",[0,1,0,0]),("Age and pressure",[.5,.5,0,0]),("Risk stratified",[.35,.25,.4,0])]
    weights = list(baseline)+[(f"Candidate {i+1}",v) for i,v in enumerate(np.random.default_rng(seed).dirichlet(np.ones(4),n))]
    records, medians, counts = [], [], []
    for name,w in weights:
        metrics, m, c = policy_metrics(r,replay(r,w))
        records.append(dict(policy=name,**{f"w{i+1}":float(v) for i,v in enumerate(w)},**metrics))
        medians.append(m); counts.append(c)
    common = np.all(np.asarray(counts)>=20,axis=0)
    if not common.any():
        raise AssertionError("No common supported groups across candidate policies")
    for row,m in zip(records,medians):
        row["disparity_cv"] = float(np.std(m[common])/np.mean(m[common]))
        row["dispersion_groups"] = int(common.sum())
    frame = pd.DataFrame(records)
    objectives = frame[["p95","long_wait_share","disparity_cv"]].to_numpy()
    finite = np.isfinite(objectives).all(axis=1)
    keep = np.zeros(len(frame),bool); keep[finite] = nondominated(objectives[finite])
    frame["nondominated"] = keep
    p = objectives[keep]
    span = np.ptp(p,axis=0); span[span==0] = 1
    distance = np.sqrt(np.sum(((p-np.min(p,axis=0))/span)**2,axis=1))
    winner = np.flatnonzero(keep)[int(np.argmin(distance))]
    frame["selected"] = np.arange(len(frame))==winner
    return frame, frame.iloc[winner], common
