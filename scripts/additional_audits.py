from __future__ import annotations
import json
from pathlib import Path
from analysis_core import *
from run_revision_analysis import load_data, OUT, WORK, DATA_DIR, PROTOCOL, SCRIPT_DIR, say, dump
from prepare_data import fetch


def main():
    br,ny,ba,na=load_data()
    for key in ("department","category","organisation"):
        target=DATA_DIR/f"brussels_{key}_labels.json"
        if not target.exists():
            raw=fetch(f"https://data.mobility.brussels/data/attributevalues/?tid=fims_code_{key}")
            target.write_bytes(raw)
    departments=json.loads((DATA_DIR/"brussels_department_labels.json").read_text(encoding="utf-8"))
    names={str(x["value"]):x["label_fr"].strip() for x in departments}
    frame=pd.read_csv(OUT/"lifecycle_by_department.csv",dtype={"department":str})
    frame["label"]=[names.get(str(dep),str(dep)) if city=="Brussels" else str(dep) for city,dep in zip(frame.city,frame.department)]
    frame.to_csv(OUT/"lifecycle_by_department_named.csv",index=False)
    rows=[]
    for city,d,lo in [("Brussels",br,"2019-01-01"),("NYC",ny,"2023-01-01")]:
        cohort=d[d.created.between(day(lo),day("2025-12-31"))]
        for scenario,mask in [("Primary ambiguity exclusion",~cohort.ambiguous),
                              ("Ambiguous closed records censored",pd.Series(True,index=cohort.index)),
                              ("Exclude status-date disagreements",~cohort.ambiguous&(~cohort.event|cohort.status.str.upper().eq("CLOSED")))]:
            x=cohort[mask]
            c=km_curve(x.duration,x.event)
            rows.append(dict(city=city,scenario=scenario,n=len(x),closure7=1-km_at(c,7)[0],closure30=1-km_at(c,30)[0],closure90=1-km_at(c,90)[0],rmst90=rmst(c,90)))
    pd.DataFrame(rows).to_csv(OUT/"endpoint_coding_sensitivity.csv",index=False)
    # Matched creation cohorts with at least ninety days of potential follow-up.
    annual=[]
    for city,d in [("Brussels",br),("NYC",ny)]:
        for year in (2023,2024,2025):
            x=d[(~d.ambiguous)&d.created.between(day(f"{year}-01-01"),day(f"{year}-12-31")-90)]
            annual.append(dict(city=city,year=year,mature_cohort_n=len(x),observed_long_wait90=float(np.mean((~x.event)|(x.duration>90))),
                observed_closed_n=int(x.event.sum()),observed_closed_p95=float(np.quantile(x.loc[x.event,"duration"],.95))))
    pd.DataFrame(annual).to_csv(OUT/"observed_mature_creation_cohorts.csv",index=False)
    for city,model in [("Brussels",group_model(br,"2019-01-01","2022-12-31",250)),("NYC",group_model(ny,"2023-01-01","2023-12-31",3000))]:
        data=br if city=="Brussels" else ny
        for year in ([2023,2024,2025] if city=="Brussels" else [2024,2025]):
            records=[]
            x=data[(~data.ambiguous)&data.created.between(day(f"{year}-01-01"),day(f"{year}-12-31"))]
            for grp,g in x.groupby("dept"):
                event=g.event.to_numpy()&(g.closed.to_numpy()<=day(f"{year}-12-31"))
                duration=np.where(event,g.closed-g.created,day(f"{year}-12-31")-g.created).astype(np.int32)
                c=km_curve(duration,event)
                records.append(dict(city=city,year=year,department=grp,n=len(g),closure90=1-km_at(c,90)[0],rmst90=rmst(c,90)))
            pd.DataFrame(records).to_csv(OUT/f"observed_{city}_{year}_matched_cohort.csv",index=False)
    package_meta=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,analysis_protocol=sha256(PROTOCOL),
        core_code_sha256=sha256(SCRIPT_DIR/"analysis_core.py"),main_code_sha256=sha256(SCRIPT_DIR/"run_revision_analysis.py"))
    dump("runtime_and_code_manifest.json",package_meta)
    say("Metadata labels and endpoint sensitivities completed")


if __name__=="__main__":main()
