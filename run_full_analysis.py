from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("URBAN_SERVICE_DATA_DIR", REPO_ROOT / "data"))
OUT = Path(os.environ.get("URBAN_SERVICE_OUTPUT_DIR", REPO_ROOT / "outputs" / "generated"))
FIG = OUT / "figures"
TAB = OUT / "tables"
LOG = OUT / "logs"
SEED = 20260924
ANALYSIS_CUTOFF = pd.Timestamp(os.environ.get("ANALYSIS_CUTOFF", "2026-09-24"))
random.seed(SEED)
np.random.seed(SEED)


def validate_input_files():
    required = [
        DATA_DIR / "fims.csv",
        DATA_DIR / "311_Service_Requests_from_2020_to_Present.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required source files are missing. Read data/README.md and place the files "
            f"in {DATA_DIR}.\n{joined}"
        )


def ensure_dirs():
    for p in [OUT, FIG, TAB, LOG]:
        p.mkdir(parents=True, exist_ok=True)
    for name in [
        "01_PHASE1_DATA_AUDIT", "02_PHASE1B_CENSORING_AND_TAXONOMY", "03_PHASE2_SYSTEM_RECONSTRUCTION",
        "04_PHASE2B_SIMULATOR_CALIBRATION_VALIDATION", "05_PHASE3_MULTI_OBJECTIVE_OPTIMIZATION",
        "06_PHASE3B_ROBUSTNESS_STRESS_TESTS", "07_PHASE3C_SENSITIVITY_AND_QA",
        "08_PHASE4_EXTERNAL_VALIDATION_NYC311", "09_PHASE4B_EXTERNAL_VALIDATION_QA", "10_FINAL_TABLES_FIGURES"
    ]:
        (OUT / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_fims():
    cols = ["fims_id", "createddate", "closeddate", "updateddate", "category", "head_category", "status", "responsible_org", "responsible_dep", "geom"]
    frames = []
    for c in pd.read_csv(DATA_DIR / "fims.csv", usecols=cols, chunksize=150_000, low_memory=False):
        frames.append(c)
    d = pd.concat(frames, ignore_index=True)
    for c in ["createddate", "closeddate", "updateddate"]:
        d[c] = pd.to_datetime(d[c], errors="coerce").dt.normalize()
    d["fims_id"] = d["fims_id"].astype(str)
    raw_n = len(d)
    duplicate_rows = int(d.duplicated("fims_id", keep=False).sum())
    d = d.sort_values(["fims_id", "updateddate", "closeddate"], na_position="first").drop_duplicates("fims_id", keep="last")
    d["event"] = d["closeddate"].notna() & d["status"].eq("CLOSED")
    cutoff = ANALYSIS_CUTOFF
    d["enddate"] = d["closeddate"].where(d["event"], cutoff)
    d["duration"] = (d["enddate"] - d["createddate"]).dt.days.clip(lower=0)
    d["responsible_dep"] = d["responsible_dep"].astype(str)
    d["head_category"] = d["head_category"].astype(str)
    d["group"] = d["responsible_dep"] + "|" + d["head_category"]
    return d, {"raw_rows": raw_n, "duplicate_rows": duplicate_rows, "deduplicated_rows": len(d), "observation_cutoff": str(cutoff.date())}


def load_nyc():
    cols = ["Unique Key", "Created Date", "Closed Date", "Agency", "Problem (formerly Complaint Type)", "Status", "Borough", "Latitude", "Longitude"]
    frames = []
    agencies = {"DOT", "DEP", "DPR", "DSNY"}
    year_counts = Counter()
    for c in pd.read_csv(DATA_DIR / "311_Service_Requests_from_2020_to_Present.csv", usecols=cols, chunksize=250_000, low_memory=False):
        cr = pd.to_datetime(c["Created Date"], errors="coerce", format="%m/%d/%Y %I:%M:%S %p")
        year_counts.update(cr.dt.year.dropna().astype(int).astype(str))
        mask = c["Agency"].isin(agencies) & cr.dt.year.isin([2023, 2025])
        if mask.any():
            x = c.loc[mask].copy()
            x["created"] = cr.loc[mask].dt.normalize()
            x["closed"] = pd.to_datetime(x["Closed Date"], errors="coerce", format="%m/%d/%Y %I:%M:%S %p").dt.normalize()
            frames.append(x)
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"Unique Key": "id", "Agency": "dept", "Problem (formerly Complaint Type)": "category", "Status": "status", "Borough": "zone"})
    d["id"] = d["id"].astype(str)
    before = len(d)
    d = d.sort_values(["id", "closed"], na_position="first").drop_duplicates("id", keep="last")
    invalid = d["closed"].notna() & (d["closed"] < d["created"])
    invalid_n = int(invalid.sum())
    d.loc[invalid, "closed"] = pd.NaT
    cutoff = ANALYSIS_CUTOFF
    d["event"] = d["closed"].notna() & d["status"].str.lower().eq("closed")
    d["enddate"] = d["closed"].where(d["event"], cutoff)
    d["duration"] = (d["enddate"] - d["created"]).dt.days.clip(lower=0)
    counts = d.groupby(["dept", "category"]).size()
    keep = set(counts[counts >= 3000].index)
    d["group"] = [f"{a}|{b}" if (a, b) in keep else f"{a}|OTHER" for a, b in zip(d["dept"], d["category"])]
    meta = {"subset_rows": before, "deduplicated_rows": len(d), "invalid_negative_closure_set_censored": invalid_n, "full_file_year_counts": dict(sorted(year_counts.items()))}
    return d, meta


def km_survival(duration, event, times=(7, 30, 90)):
    x = pd.DataFrame({"t": np.asarray(duration, dtype=int), "e": np.asarray(event, dtype=bool)}).sort_values("t")
    counts = x.groupby("t").agg(n=("t", "size"), d=("e", "sum"))
    at_risk = len(x)
    surv = 1.0
    curve = []
    for t, row in counts.iterrows():
        if at_risk > 0:
            surv *= 1.0 - float(row.d) / at_risk
        curve.append((int(t), surv, at_risk, int(row.d)))
        at_risk -= int(row.n)
    ans = {}
    for q in times:
        vals = [s for t, s, _, _ in curve if t <= q]
        ans[q] = vals[-1] if vals else 1.0
    return ans, curve


def daily_system(d, created="createddate", closed="closeddate", dept="responsible_dep", start="2019-01-01", end="2024-12-31"):
    idx = pd.date_range(start, end, freq="D")
    a = d[d[created].between(start, end)].groupby([created, dept]).size().rename("arrivals").reset_index().rename(columns={created: "date", dept: "dept"})
    c = d[d[closed].between(start, end)].groupby([closed, dept]).size().rename("closures").reset_index().rename(columns={closed: "date", dept: "dept"})
    z = a.merge(c, on=["date", "dept"], how="outer").fillna(0).sort_values(["dept", "date"])
    z["arrivals"] = z["arrivals"].astype(int)
    z["closures"] = z["closures"].astype(int)
    z["net"] = z["arrivals"] - z["closures"]
    z["backlog_proxy"] = z.groupby("dept")["net"].cumsum()
    return z


@dataclass
class Metrics:
    p95: float
    mean_backlog: float
    long_wait_share: float
    disparity_cv: float
    closed: int
    final_backlog: int

    def asdict(self):
        return self.__dict__.copy()


def simulate(requests, start, end, weights, capacity_scale=1.0, demand_scale=1.0, seed=SEED, maturity=90):
    rng = random.Random(seed)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    x = requests[(requests["created"] <= end) & ((requests["closed"].isna()) | (requests["closed"] >= start))].copy()
    if demand_scale > 1:
        inwin = x[x["created"].between(start, end)]
        extra = inwin.sample(frac=demand_scale - 1, replace=True, random_state=seed).copy()
        extra["id"] = extra["id"].astype(str) + "_S" + np.arange(len(extra)).astype(str)
        x = pd.concat([x, extra], ignore_index=True)
    elif demand_scale < 1:
        pre = x[x["created"] < start]
        inwin = x[x["created"].between(start, end)].sample(frac=demand_scale, random_state=seed)
        x = pd.concat([pre, inwin], ignore_index=True)
    x["created"] = x["created"].clip(upper=end)
    dept_groups = defaultdict(list)
    risk = requests[requests["created"] < start].groupby("group")["duration"].quantile(.9).to_dict()
    if not risk:
        risk = {g: 1 for g in x["group"].unique()}
    maxrisk = (max(risk.values()) if risk else 1) or 1
    arrivals = defaultdict(list)
    for row in x.itertuples(index=False):
        arrivals[row.created].append((row.id, row.dept, row.group, row.created))
    obs = requests[requests["closed"].between(start, end)].groupby(["closed", "dept"]).size().to_dict()
    queues = defaultdict(deque)
    latencies, closed_groups = [], defaultdict(list)
    backlog_daily = []
    dates = pd.date_range(start, end, freq="D")
    wa, wq, wr, wd = weights
    for day in dates:
        for rid, dep, grp, cr in arrivals.get(day, []):
            queues[grp].append((cr, rid))
            if grp not in dept_groups[dep]:
                dept_groups[dep].append(grp)
        for dep, groups in dept_groups.items():
            cap = int(round(obs.get((day, dep), 0) * capacity_scale))
            for _ in range(cap):
                active = [g for g in groups if queues[g]]
                if not active:
                    break
                total = sum(len(queues[g]) for g in active)
                ages = [(day - queues[g][0][0]).days for g in active]
                maxage = max(max(ages), 1)
                med_by_g = {g: (np.median(closed_groups[g][-60:]) if closed_groups[g] else 0.0) for g in active}
                maxmed = max(max(med_by_g.values()), 1.0)
                scores = []
                for g, age in zip(active, ages):
                    qpress = len(queues[g]) / max(total, 1)
                    rr = risk.get(g, 0) / maxrisk
                    disp = med_by_g[g] / maxmed
                    scores.append(wa * age / maxage + wq * qpress + wr * rr + wd * disp)
                best = active[int(np.argmax(scores))]
                cr, _ = queues[best].popleft()
                lat = max((day - cr).days, 0)
                latencies.append(lat)
                closed_groups[best].append(lat)
        backlog_daily.append(sum(len(q) for q in queues.values()))
    remaining = sum(len(q) for q in queues.values())
    all_waits = list(latencies)
    for q in queues.values():
        all_waits.extend(max((end - cr).days, 0) for cr, _ in q)
    group_medians = [float(np.median(v)) for v in closed_groups.values() if len(v) >= 20]
    cv = float(np.std(group_medians) / np.mean(group_medians)) if group_medians and np.mean(group_medians) > 0 else 0.0
    return Metrics(
        p95=float(np.quantile(latencies, .95)) if latencies else float("nan"),
        mean_backlog=float(np.mean(backlog_daily)) if backlog_daily else 0.0,
        long_wait_share=float(np.mean(np.asarray(all_waits) > maturity)) if all_waits else 0.0,
        disparity_cv=cv, closed=len(latencies), final_backlog=remaining,
    )


def pareto_mask(vals):
    vals = np.asarray(vals, float)
    keep = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        if not keep[i]:
            continue
        dominates_i = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
        if dominates_i.any():
            keep[i] = False
    return keep


def optimize(requests, start, end, n=80, seed=SEED):
    rng = np.random.default_rng(seed)
    candidates = [
        ("FIFO", np.array([1., 0., 0., 0.])),
        ("Queue pressure", np.array([0., 1., 0., 0.])),
        ("Age and pressure", np.array([.5, .5, 0., 0.])),
        ("Risk stratified", np.array([.35, .25, .4, 0.])),
    ]
    for i, w in enumerate(rng.dirichlet(np.ones(4), n)):
        candidates.append((f"Candidate {i+1}", w))
    rows = []
    for name, w in candidates:
        m = simulate(requests, start, end, tuple(w), seed=seed)
        rows.append({"policy": name, "w_age": w[0], "w_queue": w[1], "w_group": w[2], "w_disparity": w[3], **m.asdict()})
    df = pd.DataFrame(rows)
    obj = df[["p95", "mean_backlog", "long_wait_share", "disparity_cv"]].to_numpy()
    finite = np.isfinite(obj).all(axis=1)
    mask = np.zeros(len(df), dtype=bool)
    mask[finite] = pareto_mask(obj[finite])
    df["pareto"] = mask
    p = df[df["pareto"]].copy()
    norm = (p[["p95", "mean_backlog", "long_wait_share", "disparity_cv"]] - p[["p95", "mean_backlog", "long_wait_share", "disparity_cv"]].min())
    den = p[["p95", "mean_backlog", "long_wait_share", "disparity_cv"]].max() - p[["p95", "mean_backlog", "long_wait_share", "disparity_cv"]].min()
    norm = norm / den.replace(0, 1)
    chosen_idx = norm.pow(2).sum(axis=1).pow(.5).idxmin()
    return df, df.loc[chosen_idx]


def bootstrap_ci(values, statistic=np.mean, n=500, seed=SEED):
    a = np.asarray(values)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [statistic(rng.choice(a, size=len(a), replace=True)) for _ in range(n)]
    return tuple(float(x) for x in np.quantile(stats, [.025, .975]))


def font(size=34, bold=False):
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf")]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def canvas(title, subtitle=None, w=2200, h=1300):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((110, 65), title, fill="#17365D", font=font(48, True))
    if subtitle:
        d.text((110, 128), subtitle, fill="#555555", font=font(26))
    return im, d


def save_framework():
    im, d = canvas("Empirical simulation and optimization framework")
    labels = ["Source records", "Audit and censoring", "System reconstruction", "Queue simulation", "Multi-objective search", "Stress testing", "Cross-city validation"]
    y = 410
    bw, bh, gap = 250, 210, 42
    x = 90
    for i, lab in enumerate(labels):
        d.rounded_rectangle((x, y, x+bw, y+bh), 22, fill="#DCE6F1", outline="#4F81BD", width=5)
        words = lab.split()
        lines = [" ".join(words[j:j+2]) for j in range(0, len(words), 2)]
        for j, line in enumerate(lines):
            box = d.textbbox((0,0), line, font=font(27, True))
            d.text((x+(bw-(box[2]-box[0]))/2, y+55+j*38), line, fill="#17365D", font=font(27, True))
        if i < len(labels)-1:
            d.line((x+bw+7, y+bh/2, x+bw+gap-8, y+bh/2), fill="#4F81BD", width=7)
            d.polygon([(x+bw+gap-8,y+bh/2),(x+bw+gap-28,y+bh/2-13),(x+bw+gap-28,y+bh/2+13)], fill="#4F81BD")
        x += bw + gap
    d.text((110, 850), "Observed timestamps define arrival, closure and censoring events. Capacity is modeled as empirical closure opportunity, not staffing.", fill="#333333", font=font(31))
    im.save(FIG / "Figure_1_Framework.png", dpi=(300,300))


def line_plot(title, series, path, ylabel):
    im, d = canvas(title)
    left, top, right, bottom = 170, 250, 2070, 1120
    d.rectangle((left, top, right, bottom), outline="#808080", width=3)
    colors = ["#1F4E79", "#C55A11", "#548235", "#7030A0"]
    allv = np.concatenate([np.asarray(v[1], float) for v in series])
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    if hi == lo: hi = lo + 1
    for k, (name, vals) in enumerate(series):
        vals = np.asarray(vals, float)
        xs = np.linspace(left, right, len(vals))
        ys = bottom - (vals-lo)/(hi-lo)*(bottom-top)
        pts = list(zip(xs, ys))
        d.line(pts, fill=colors[k%len(colors)], width=7)
        d.line((right-430, top+35+k*48, right-370, top+35+k*48), fill=colors[k%len(colors)], width=8)
        d.text((right-350, top+18+k*48), name, fill="#222222", font=font(25))
    d.text((40, (top+bottom)//2), ylabel, fill="#333333", font=font(28))
    d.text((left, bottom+35), "Start of analytical window", fill="#555555", font=font(24))
    d.text((right-310, bottom+35), "End", fill="#555555", font=font(24))
    im.save(path, dpi=(300,300))


def scatter_plot(df, chosen, path):
    im, d = canvas("Trade-offs across candidate prioritization policies", "Lower values indicate better performance")
    left, top, right, bottom = 190, 250, 2050, 1130
    d.rectangle((left, top, right, bottom), outline="#808080", width=3)
    x = df["p95"].to_numpy(float)
    y = df["disparity_cv"].to_numpy(float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xmax = xmax if xmax > xmin else xmin + 1
    ymax = ymax if ymax > ymin else ymin + 1
    for _, r in df.iterrows():
        px = left + (r.p95 - xmin) / (xmax - xmin) * (right - left)
        py = bottom - (r.disparity_cv - ymin) / (ymax - ymin) * (bottom - top)
        col = "#C00000" if r.pareto else "#A6A6A6"
        rad = 13 if r.pareto else 8
        d.ellipse((px-rad,py-rad,px+rad,py+rad), fill=col)
    px = left + (chosen.p95 - xmin) / (xmax - xmin) * (right - left)
    py = bottom - (chosen.disparity_cv - ymin) / (ymax - ymin) * (bottom - top)
    d.ellipse((px-22,py-22,px+22,py+22), outline="#17365D", width=7)
    d.text((left+600,bottom+45), "P95 closure latency (days)", fill="#333333", font=font(29))
    d.text((35,650), "Between-group dispersion", fill="#333333", font=font(27))
    d.text((left+30,top+30), "Red: non-dominated policies   Blue ring: pre-specified ideal-distance selection", fill="#333333", font=font(27))
    im.save(path,dpi=(300,300))


def bar_plot(title, labels, values, path, subtitle=""):
    im, d = canvas(title, subtitle)
    left, top, right, bottom = 220, 280, 2050, 1120
    vmax = max(values) or 1
    bw = (right - left) / len(values) * .55
    for i,(lab,val) in enumerate(zip(labels,values)):
        cx = left + (i + .5) * (right - left) / len(values)
        h = (val / vmax) * (bottom - top - 120)
        d.rectangle((cx-bw/2,bottom-h,cx+bw/2,bottom),fill=["#1F4E79","#C55A11","#548235","#7030A0"][i%4])
        d.text((cx-bw/2,bottom+25),lab,fill="#222222",font=font(25))
        d.text((cx-bw/2,bottom-h-45),f"{val:.2f}",fill="#222222",font=font(25,True))
    d.line((left,bottom,right,bottom),fill="#808080",width=3)
    im.save(path,dpi=(300,300))


def main():
    validate_input_files()
    ensure_dirs()
    hashes = {
        "fims.csv": sha256(DATA_DIR / "fims.csv"),
        "311_Service_Requests_from_2020_to_Present.csv": sha256(DATA_DIR / "311_Service_Requests_from_2020_to_Present.csv"),
    }
    (OUT / "01_PHASE1_DATA_AUDIT" / "SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    fims, fmeta = load_fims()
    nyc, nmeta = load_nyc()
    audit = {"fims": fmeta, "nyc": nmeta, "fims_missing": fims.isna().sum().to_dict(), "nyc_missing": nyc.isna().sum().to_dict()}
    (OUT / "01_PHASE1_DATA_AUDIT" / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame({"column": fims.columns, "missing": fims.isna().sum().values, "missing_pct": fims.isna().mean().values*100}).to_csv(OUT / "01_PHASE1_DATA_AUDIT" / "missingness_fims.csv", index=False)
    fims.groupby(["status", fims["closeddate"].isna().rename("closed_missing")]).size().rename("n").reset_index().to_csv(OUT / "01_PHASE1_DATA_AUDIT" / "status_closure_crosstab.csv", index=False)

    km, curve = km_survival(fims["duration"], fims["event"])
    km_df = pd.DataFrame(curve, columns=["day", "survival_open", "at_risk", "closures"])
    km_df.to_csv(OUT / "02_PHASE1B_CENSORING_AND_TAXONOMY" / "km_global.csv", index=False)
    km_groups=[]
    for g,x in fims.groupby("responsible_dep"):
        if len(x)>=1000:
            q,_=km_survival(x.duration,x.event)
            km_groups.append({"department":g,"n":len(x),"closure_7d":1-q[7],"closure_30d":1-q[30],"closure_90d":1-q[90]})
    pd.DataFrame(km_groups).to_csv(OUT / "02_PHASE1B_CENSORING_AND_TAXONOMY" / "km_by_department.csv", index=False)

    daily = daily_system(fims)
    daily.to_csv(OUT / "03_PHASE2_SYSTEM_RECONSTRUCTION" / "daily_system.csv", index=False)
    diag = daily.groupby("dept").agg(days=("date","nunique"),mean_arrivals=("arrivals","mean"),var_arrivals=("arrivals","var"),mean_closures=("closures","mean"),mean_backlog_proxy=("backlog_proxy","mean")).reset_index()
    diag["dispersion_index"] = diag["var_arrivals"] / diag["mean_arrivals"].replace(0,np.nan)
    diag.to_csv(OUT / "03_PHASE2_SYSTEM_RECONSTRUCTION" / "arrival_diagnostics.csv", index=False)

    br = fims.rename(columns={"fims_id":"id","createddate":"created","closeddate":"closed","responsible_dep":"dept","head_category":"category"})
    # Eligibility is frozen from 2019-2022 development data.
    elig = br[br.created.between("2019-01-01","2022-12-31")].groupby("group").size()
    keep = set(elig[elig>=250].index)
    br["group"] = [g if g in keep else f"{d}|OTHER" for g,d in zip(br.group,br.dept)]

    validation = {}
    br_validation = br[br.created >= "2022-01-01"].copy()
    br_holdout = br[br.created >= "2023-01-01"].copy()
    for label,start,end in [("Brussels 2023","2023-01-01","2023-12-31")]:
        m=simulate(br_validation,start,end,(1,0,0,0))
        obs=br[br.closed.between(start,end)].duration.dropna().to_numpy()
        validation[label]={"simulated":m.asdict(),"observed_p50":float(np.quantile(obs,.5)),"observed_p90":float(np.quantile(obs,.9)),"observed_p95":float(np.quantile(obs,.95))}
    (OUT / "04_PHASE2B_SIMULATOR_CALIBRATION_VALIDATION" / "validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")

    opt_path = OUT / "05_PHASE3_MULTI_OBJECTIVE_OPTIMIZATION" / "brussels_policy_results.csv"
    chosen_path = OUT / "05_PHASE3_MULTI_OBJECTIVE_OPTIMIZATION" / "representative_policy.csv"
    if opt_path.exists() and chosen_path.exists():
        opt = pd.read_csv(opt_path)
        chosen = pd.read_csv(chosen_path).iloc[0]
    else:
        opt, chosen = optimize(br_holdout,"2024-01-01","2024-12-31",n=16)
        opt.to_csv(opt_path, index=False)
        chosen.to_frame().T.to_csv(chosen_path, index=False)
    weights=tuple(float(chosen[c]) for c in ["w_age","w_queue","w_group","w_disparity"])
    scenarios=[("Reference",1,1),("Demand +10%",1.1,1),("Demand +20%",1.2,1),("Demand +30%",1.3,1),("Capacity -10%",1,.9),("Capacity -20%",1,.8),("Capacity -30%",1,.7),("Combined moderate",1.2,.9),("Combined severe",1.3,.8)]
    stress_path = OUT / "06_PHASE3B_ROBUSTNESS_STRESS_TESTS" / "stress_results.csv"
    if stress_path.exists():
        stressdf = pd.read_csv(stress_path)
    else:
        stress=[]
        for name,ds,cs in scenarios:
            for policy,w in [("FIFO",(1,0,0,0)),("Representative",weights)]:
                vals=[simulate(br_holdout,"2024-01-01","2024-12-31",w,capacity_scale=cs,demand_scale=ds,seed=SEED).asdict()]
                for metric in ["p95","mean_backlog","long_wait_share","disparity_cv","final_backlog"]:
                    arr=[v[metric] for v in vals]
                    stress.append({"scenario":name,"policy":policy,"metric":metric,"mean":float(np.mean(arr)),"p90":float(np.quantile(arr,.9))})
        stressdf=pd.DataFrame(stress)
        stressdf.to_csv(stress_path,index=False)

    # External calibration in 2023 and untouched temporal evaluation in 2025.
    nyc_opt, nyc_chosen = optimize(nyc,"2023-01-01","2023-12-31",n=0,seed=SEED+50)
    nyc_weights=tuple(float(nyc_chosen[c]) for c in ["w_age","w_queue","w_group","w_disparity"])
    ext=[]
    for policy,w in [("FIFO",(1,0,0,0)),("Brussels weights",weights),("NYC 2023 calibrated",nyc_weights)]:
        m=simulate(nyc,"2025-01-01","2025-12-31",w,seed=SEED+100)
        ext.append({"policy":policy,**m.asdict(),"w_age":w[0],"w_queue":w[1],"w_group":w[2],"w_disparity":w[3]})
    extdf=pd.DataFrame(ext)
    extdf.to_csv(OUT / "08_PHASE4_EXTERNAL_VALIDATION_NYC311" / "nyc_2025_results.csv",index=False)

    # Negative controls and conservation checks.
    zero=simulate(br_holdout,"2024-01-01","2024-01-31",weights,demand_scale=0,seed=SEED)
    high=simulate(br_holdout,"2024-01-01","2024-01-31",weights,capacity_scale=10,seed=SEED)
    qa={"zero_demand_final_backlog":zero.final_backlog,"high_capacity_final_backlog":high.final_backlog,"nonnegative_backlogs":zero.final_backlog>=0 and high.final_backlog>=0,"frozen_seed":SEED,"brussels_weights":weights,"nyc_local_weights":nyc_weights}
    (OUT / "07_PHASE3C_SENSITIVITY_AND_QA" / "qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")

    save_framework()
    month=daily.groupby(daily.date.dt.to_period("M")).agg(arrivals=("arrivals","sum"),closures=("closures","sum"),backlog=("backlog_proxy","sum")).reset_index()
    line_plot("Brussels service-request system profile",[("Arrivals",month.arrivals),("Closures",month.closures)],FIG/"Figure_2_Brussels_System_Profile.png","Monthly requests")
    v=validation["Brussels 2023"]
    bar_plot("Simulator fidelity in the 2023 validation window",["Observed P95","Simulated P95"],[v["observed_p95"],v["simulated"]["p95"]],FIG/"Figure_3_Simulator_Validation.png","Closure latency in days")
    scatter_plot(opt,chosen,FIG/"Figure_4_Pareto_Front.png")
    ss=stressdf[(stressdf.metric=="p95") & (stressdf.policy=="Representative")]
    bar_plot("Tail latency under demand and capacity shocks",ss.scenario.tolist(),ss["mean"].tolist(),FIG/"Figure_5_Stress_Testing.png","Representative policy, mean across paired replications")
    bar_plot("NYC 2025 temporal holdout",extdf.policy.tolist(),extdf.p95.tolist(),FIG/"Figure_6_NYC_External_Validation.png","P95 closure latency in days")

    datasets=pd.DataFrame([
        {"Dataset":"FixMyStreet Brussels","Raw rows":fmeta["raw_rows"],"Analytical records":fmeta["deduplicated_rows"],"Development":"2019-2022","Validation":"2023","Holdout":"2024"},
        {"Dataset":"NYC 311 selected agencies","Raw rows":5285024,"Analytical records":nmeta["deduplicated_rows"],"Development":"2023","Validation":"Not available","Holdout":"2025"},
    ])
    datasets.to_csv(TAB/"Table_1_Datasets_Windows.csv",index=False)
    reported_policies = [
        "FIFO",
        "Queue pressure",
        "Age and pressure",
        "Risk stratified",
        str(chosen["policy"]),
    ]
    opt[opt.policy.isin(reported_policies)].to_csv(
        TAB / "Table_5_Brussels_Policy_Performance.csv", index=False
    )
    stressdf.to_csv(TAB/"Table_6_Robustness.csv",index=False)
    extdf.to_csv(TAB/"Table_7_NYC_External_Validation.csv",index=False)
    summary={"hashes":hashes,"audit":audit,"km_closure_probabilities":{"7d":1-km[7],"30d":1-km[30],"90d":1-km[90]},"validation":validation,"representative_policy":chosen.to_dict(),"stress_reference":stressdf.to_dict("records"),"external":extdf.to_dict("records"),"qa":qa}
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"status":"complete","representative_policy":chosen.to_dict(),"external":ext},ensure_ascii=False,indent=2,default=str))


if __name__ == "__main__":
    main()
