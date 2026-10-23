"""Download scoped public NYC records and verify every monthly agency count.

Raw source snapshots and partial downloads remain outside the release package.
Only explicit, non-identifying fields are requested. No addresses or free text.
"""
from __future__ import annotations

import concurrent.futures as futures
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORK = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(WORK / "scientific_packages"))
import truststore
truststore.inject_into_ssl()
DATA = Path(os.environ.get("URBAN_SERVICE_DATA_DIR", WORK / "data"))
DATA.mkdir(exist_ok=True)
AGENCIES = "agency in ('DOT','DEP','DPR','DSNY')"
FIELDS = "unique_key,created_date,closed_date,agency,complaint_type,status,borough"


def fetch(url: str, timeout=180) -> bytes:
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UrbanServiceResearch/1.0", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last = exc
            print(f"Retry {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed request: {url}") from last


def url(dataset, fmt="json", **params):
    return f"https://data.cityofnewyork.us/resource/{dataset}.{fmt}?" + urllib.parse.urlencode({f"${k}": v for k, v in params.items()})


def query(dataset, **params):
    return json.loads(fetch(url(dataset, **params)))

def inherited_scopes():
    prefix=f"{AGENCIES} AND created_date < '2023-01-01T00:00:00' AND "
    return dict(undated_or_late=prefix+"(closed_date IS NULL OR closed_date >= '2023-01-01T00:00:00')",
        invalid_date=prefix+"closed_date < created_date")


def download_job(dataset, label, where, expected):
    path = DATA / f"nyc_{label}.csv"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old["rows"] == expected and old["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest():
            print(f"Cached {label}: {expected:,}", flush=True)
            return old
    request_url = url(dataset, "csv", select=FIELDS, where=where, order="unique_key", limit=2_000_000)
    raw = fetch(request_url)
    decoded = raw.decode("utf-8-sig")
    count = sum(1 for _ in csv.DictReader(io.StringIO(decoded)))
    if count != expected:
        raise AssertionError(f"{label}: downloaded {count}, expected {expected}")
    path.write_bytes(raw)
    metadata = dict(dataset=dataset, label=label, url=request_url, rows=count,
                    sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Downloaded {label}: {count:,} rows, {len(raw)/1e6:.1f} MB", flush=True)
    return metadata


def main():
    dataset = "erm2-nwe9"
    archive = None
    discovery_url = "https://api.us.socrata.com/api/catalog/v1?" + urllib.parse.urlencode({
        "search_context": "data.cityofnewyork.us", "q": "311 Service Requests from 2010 to 2019", "limit": 10})
    discovery = json.loads(fetch(discovery_url))
    for item in discovery.get("results", []):
        resource = item.get("resource", {})
        if "2010 to 2019" in resource.get("name", ""):
            archive = resource["id"]
            print(f"Historical archive: {archive}, {resource['name']}", flush=True)
            break
    (DATA / "catalog_discovery.json").write_text(json.dumps(discovery, indent=2), encoding="utf-8")
    window = "created_date >= '2023-01-01T00:00:00' AND created_date < '2026-01-01T00:00:00'"
    monthly = query(dataset, select="date_trunc_ym(created_date) as month,agency,count(*) as n", where=f"{AGENCIES} AND {window}", group="month,agency", order="month,agency", limit=10000)
    (DATA / "nyc_official_monthly_counts.json").write_text(json.dumps(monthly, indent=2), encoding="utf-8")
    jobs = []
    for year in (2023, 2024, 2025):
        for agency in ("DOT", "DEP", "DPR", "DSNY"):
            count = sum(int(x["n"]) for x in monthly if x["agency"] == agency and x["month"].startswith(str(year)))
            where = f"agency='{agency}' AND created_date >= '{year}-01-01T00:00:00' AND created_date < '{year+1}-01-01T00:00:00'"
            jobs.append((dataset, f"{year}_{agency}", where, count))
    inherited = inherited_scopes()["undated_or_late"]
    invalid_inherited = inherited_scopes()["invalid_date"]
    for source in [dataset] + ([archive] if archive else []):
        n = int(query(source, select="count(*) as n", where=inherited)[0]["n"])
        print(f"Observable inherited records in {source}: {n:,}", flush=True)
        jobs.append((source, f"inherited_{source}", inherited, n))
        invalid_n = int(query(source, select="count(*) as n", where=invalid_inherited)[0]["n"])
        jobs.append((source, f"inherited_invalid_{source}", invalid_inherited, invalid_n))
    if archive is None:
        raise RuntimeError("Historical 2010-2019 source could not be identified.")
    before = datetime.now(timezone.utc).isoformat()
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(lambda x: download_job(*x), jobs))
    after = query(dataset, select="date_trunc_ym(created_date) as month,agency,count(*) as n", where=f"{AGENCIES} AND {window}", group="month,agency", order="month,agency", limit=10000)
    if monthly != after:
        raise AssertionError("Official counts changed during extraction. Repeat and reconcile the snapshot.")
    result = dict(started_at_utc=before, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  fields=FIELDS.split(","), monthly_counts_verified=True, archived_dataset=archive,
                  files=manifest, current_creation_rows=sum(int(x["n"]) for x in monthly))
    (DATA / "nyc_download_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "files"}), flush=True)

def supplement_invalid_inherited():
    where=inherited_scopes()["invalid_date"]
    before={}
    for source in ("erm2-nwe9","76ig-c548"):
        before[source]=query(source,select="agency,status,count(*) as n",where=where,group="agency,status",order="agency,status")
    jobs=[(source,f"inherited_invalid_{source}",where,sum(int(x["n"]) for x in counts)) for source,counts in before.items()]
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        downloads=list(pool.map(lambda x:download_job(*x),jobs))
    after={source:query(source,select="agency,status,count(*) as n",where=where,group="agency,status",order="agency,status") for source in before}
    if before!=after:raise AssertionError("Invalid-date inherited counts changed during acquisition.")
    audit=dict(where=where,before=before,after=after,counts_verified=True,
        fields=FIELDS.split(","),files=downloads,completed_at_utc=datetime.now(timezone.utc).isoformat())
    (DATA/"nyc_inherited_invalid_date_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    target=DATA/"nyc_download_manifest.json"
    old=json.loads(target.read_text(encoding="utf-8"))
    labels={x["label"] for x in downloads}
    old["files"]=[x for x in old["files"] if x["label"] not in labels]+downloads
    old["invalid_date_inherited_extension"]=audit
    old["completed_at_utc"]=audit["completed_at_utc"]
    target.write_text(json.dumps(old,indent=2),encoding="utf-8")
    print(json.dumps(dict(supplemental_rows=sum(x["rows"] for x in downloads),counts_verified=True)),flush=True)


if __name__ == "__main__":
    if "--supplement-invalid-inherited" in sys.argv:supplement_invalid_inherited()
    else:main()
