from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis_core import *
from scipy import sparse
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE = SCRIPT_DIR.name == "scripts"
WORK = SCRIPT_DIR.parent if RELEASE else SCRIPT_DIR
DATA_DIR = Path(os.environ.get("URBAN_SERVICE_DATA_DIR", WORK / "data"))
OUT = Path(os.environ.get("URBAN_SERVICE_OUTPUT_DIR", WORK / "outputs" / "reproduced" if RELEASE else WORK / "analysis"))
OUT.mkdir(exist_ok=True)
FIMS = Path(os.environ.get("URBAN_SERVICE_FIMS_CSV", DATA_DIR / "fims.csv"))
PROTOCOL = WORK / "metadata" / "analysis_specification.md" if RELEASE else WORK / "ANALYSIS_SPECIFICATION_v5.md"


def say(message):
    print(datetime.now().strftime("%H:%M:%S"), message, flush=True)


def dump(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")


def load_data():
    cache = OUT/"canonical_data_v2.pkl"
    if cache.exists() and os.environ.get("URBAN_SERVICE_USE_CACHE","0" if RELEASE else "1")=="1":
        say("Load local canonical cache")
        return pickle.loads(cache.read_bytes())
    if os.environ.get("URBAN_SERVICE_SOURCE_MODE","derived")!="raw" and (DATA_DIR/"brussels_cohort_cells.csv.gz").exists() and (DATA_DIR/"nyc_cohort_cells.csv.gz").exists():
        say("Read frozen derived cohort cells for offline reproduction")
        br=load_derived_cells(DATA_DIR/"brussels_cohort_cells.csv.gz")
        ny=load_derived_cells(DATA_DIR/"nyc_cohort_cells.csv.gz")
        audit_path=WORK/"metadata"/"source_audit.json" if RELEASE else OUT/"source_audit.json"
        audit=json.loads(audit_path.read_text(encoding="utf-8"));ba,na=audit["brussels"],audit["nyc"]
    else:
        say("Read Brussels source snapshot")
        br, ba = load_brussels(FIMS)
        say("Read complete official NYC extracts")
        ny, na = load_nyc(DATA_DIR)
        br,ny=stable_canonical_order(br),stable_canonical_order(ny)
    dump("source_audit.json",dict(brussels=ba,nyc=na,protocol_sha256=sha256(PROTOCOL),run_started_utc=datetime.now(timezone.utc).isoformat()))
    cache.write_bytes(pickle.dumps((br,ny,ba,na),protocol=5))
    for label,d in [("brussels",br),("nyc",ny)]:
        d.groupby(["dept","status","ambiguous","event"]).size().rename("n").reset_index().to_csv(OUT/f"{label}_status_audit.csv",index=False)
        say(f"{label}: total {len(d):,}, primary {(~d.ambiguous).sum():,}, ambiguous {d.ambiguous.sum():,}")
    return br,ny,ba,na


def cluster_lifecycle_ci(x, horizon=90, draws=2000, block=28, seed=SEED):
    # In these mature creation cohorts everyone has >=horizon follow-up.
    c = ((x.created.to_numpy()-int(x.created.min()))//block).astype(int)
    _,c = np.unique(c,return_inverse=True)
    n = np.bincount(c)
    success = x.event.to_numpy() & (x.duration.to_numpy()<=horizon)
    z = np.minimum(x.duration.to_numpy(),horizon)
    ss = np.bincount(c,weights=success)
    zz = np.bincount(c,weights=z)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(n),np.full(len(n),1/len(n)),size=draws)
    denominator = w@n
    p = (w@ss)/denominator
    m = (w@zz)/denominator
    return (*np.quantile(p,[.025,.975]),*np.quantile(m,[.025,.975]),len(n))


def describe_lifecycles(br,ny):
    records=[]
    for label,d in [("Brussels",br),("NYC",ny)]:
        lo = day("2019-01-01") if label=="Brussels" else day("2023-01-01")
        primary = d[(~d.ambiguous)&d.created.between(lo,day("2025-12-31"))]
        for dep,x in [("ALL",primary)]+list(primary.groupby("dept")):
            curve=km_curve(x.duration,x.event)
            row=dict(city=label,department=dep,n=len(x),closures=int(x.event.sum()),censored=int((~x.event).sum()),rmst90=rmst(curve,90))
            for h in (7,30,90):
                s,low,high=km_at(curve,h)
                row.update({f"closure{h}":1-s,f"closure{h}_low":1-high,f"closure{h}_high":1-low})
            pl,pu,ml,mu,nb=cluster_lifecycle_ci(x,seed=SEED+len(records))
            row.update(closure90_block_low=float(pl),closure90_block_high=float(pu),rmst90_low=float(ml),rmst90_high=float(mu),blocks=nb)
            records.append(row)
            if dep=="ALL" or len(x)>=1000:
                safe="".join(c if c.isalnum() else "_" for c in str(dep))[:70]
                curve.to_csv(OUT/f"km_{label}_{safe}.csv",index=False)
        say(f"Censored lifecycle statistics completed for {label}")
    table=pd.DataFrame(records)
    table.to_csv(OUT/"lifecycle_by_department.csv",index=False)
    x=br[(~br.ambiguous)&br.created.between(day("2019-01-01"),day("2025-12-31"))]
    supported=[(dep,g) for dep,g in x.groupby("dept") if len(g)>=1000]
    pairs=[]
    for (a,da),(b,db) in itertools.combinations(supported,2):
        statistic,p=logrank(da.duration.to_numpy(),da.event.to_numpy(),db.duration.to_numpy(),db.event.to_numpy())
        pa=float(np.mean(da.event & (da.duration<=90))); pb=float(np.mean(db.event & (db.duration<=90)))
        pairs.append(dict(department_a=a,department_b=b,n_a=len(da),n_b=len(db),logrank_chi2=statistic,p=p,closure90_difference=pa-pb))
    ptable=pd.DataFrame(pairs)
    ptable["p_holm"]=holm(ptable.p)
    ptable.to_csv(OUT/"department_pairwise_logrank.csv",index=False)
    # Standardize supported pair contrasts only over service categories present in both.
    standardized=[]
    for (a,da),(b,db) in itertools.combinations(supported,2):
        ca=da.groupby("category").size(); cb=db.groupby("category").size()
        common=sorted(set(ca[ca>=50].index)&set(cb[cb>=50].index))
        if not common: continue
        weights=np.array([ca[k]+cb[k] for k in common],float); weights/=weights.sum()
        ps=[]; vs=[]
        for data in (da,db):
            prop=[];var=[]
            for cat in common:
                g=data[data.category==cat]; p=float(np.mean(g.event&(g.duration<=90)))
                prop.append(p);var.append(p*(1-p)/len(g))
            ps.append(float(weights@prop));vs.append(float((weights**2)@var))
        diff=ps[0]-ps[1];se=math.sqrt(sum(vs))
        standardized.append(dict(department_a=a,department_b=b,common_categories=len(common),category_names=" | ".join(common),
            standardized_closure90_a=ps[0],standardized_closure90_b=ps[1],difference=diff,low=diff-1.96*se,high=diff+1.96*se,
            p=2*norm.sf(abs(diff/se)) if se else 1.0,covered_a=float(da.category.isin(common).mean()),covered_b=float(db.category.isin(common).mean())))
    st=pd.DataFrame(standardized)
    if len(st): st["p_holm"]=holm(st.p)
    st.to_csv(OUT/"department_service_mix_standardized.csv",index=False)
    maturity=[]
    for year in range(2019,2026):
        g=x[x.created.between(day(f"{year}-01-01"),day(f"{year}-12-31"))]
        for dep,z in [("ALL",g)]+list(g.groupby("dept")):
            cv=km_curve(z.duration,z.event)
            maturity.append(dict(year=year,department=dep,n=len(z),closure90=1-km_at(cv,90)[0],rmst90=rmst(cv,90)))
    pd.DataFrame(maturity).to_csv(OUT/"brussels_year_department_lifecycles.csv",index=False)


def diagnostic_quantile_ci(latency, closure_days, draws=2000, seed=SEED):
    labels=(np.asarray(closure_days)-int(np.min(closure_days)))//7
    nblock=int(labels.max())+1
    values=np.asarray(latency,np.int32)
    matrix=sparse.csr_matrix((np.ones(len(values)),(labels,values)),shape=(nblock,int(values.max())+1))
    rng=np.random.default_rng(seed)
    results=[]
    for _ in range(draws//50):
        w=rng.multinomial(nblock,np.full(nblock,1/nblock),size=50)
        hist=np.asarray(w@matrix);cdf=hist.cumsum(axis=1);total=cdf[:,-1]
        qs=[]
        for q in (.5,.9,.95):
            pos=q*(total-1);low=np.floor(pos);high=np.ceil(pos)
            li=(cdf>low[:,None]).argmax(axis=1);hi=(cdf>high[:,None]).argmax(axis=1)
            qs.append(li+(hi-li)*(pos-low))
        results.extend(np.asarray(qs).T)
    a=np.asarray(results)
    return {f"p{q}_{bound}":float(v) for q,col in zip((50,90,95),a.T) for bound,v in zip(("low","high"),np.quantile(col,[.025,.975]))}


def validation_diagnostics(br):
    model=group_model(br,"2019-01-01","2022-12-31",250)
    obs=br[(~br.ambiguous)&br.event&br.closed.between(day("2023-01-01"),day("2023-12-31"))]
    observed=np.quantile(obs.duration,[.5,.9,.95])
    ci=diagnostic_quantile_ci(obs.duration,obs.closed)
    dump("observed_2023_quantile_uncertainty.json",dict(zip(("p50","p90","p95"),observed))|ci)
    configurations=[("Restricted source, empty", "empty","2022-01-01"),
                    ("Restricted source, one-year inherited", "one_year","2022-01-01"),
                    ("Full source, empty", "empty",None),
                    ("Full source, one-year inherited", "one_year",None),
                    ("Full source, complete inherited", "full",None)]
    rows=[];dept_rows=[];band_rows=[]
    for label,initial,restriction in configurations:
        r=prepare_replay(br,model,"2023-01-01","2023-12-31",initialization=initial,source_since=restriction)
        result=replay(r,[1,0,0,0]);m,_,_=policy_metrics(r,result)
        simulated=np.array([m["p50"],m["p90"],m["p95"]]);error=abs(simulated-observed)
        lat=result[0][result[0]>=0]-r.created[result[0]>=0]
        from scipy.stats import ks_2samp, wasserstein_distance
        row=dict(configuration=label,**m,ks_distance=float(ks_2samp(obs.duration.to_numpy(),lat).statistic),wasserstein_days=float(wasserstein_distance(obs.duration,lat)))
        for k,o,s,e in zip((50,90,95),observed,simulated,error):row.update({f"observed_p{k}":float(o),f"error_p{k}":float(e),f"relative_error_p{k}":float(e/max(o,1))})
        rows.append(row)
        for epsilon in (.1,.2,.3):
            passes=error<=np.maximum(7,epsilon*observed)
            band_rows.append(dict(configuration=label,relative_band=epsilon,absolute_floor_days=7,p50_pass=bool(passes[0]),p90_pass=bool(passes[1]),p95_pass=bool(passes[2]),all_quantiles_pass=bool(passes.all())))
        for dep in model["departments"]:
            gid=np.flatnonzero(np.array([g.split("|",1)[0]==dep for g in r.group_names]))
            mask=np.isin(r.group,gid)&(result[0]>=0)
            ov=obs[obs.dept==dep]
            if mask.sum()>=20 and len(ov)>=20:
                sq=np.quantile(result[0][mask]-r.created[mask],[.5,.9,.95]);oq=np.quantile(ov.duration,[.5,.9,.95])
                dept_rows.append(dict(configuration=label,department=dep,observed_n=len(ov),simulated_n=int(mask.sum()),**{f"observed_p{k}":float(v) for k,v in zip((50,90,95),oq)},**{f"simulated_p{k}":float(v) for k,v in zip((50,90,95),sq)}))
        # Compare the same created-2023 cohort with censoring at the same year end.
        cohorts=(r.created>=r.start)
        cd=result[0][cohorts];created=r.created[cohorts]
        ev=cd>=0;duration=np.where(ev,cd-created,r.end-created)
        curve=km_curve(duration,ev)
        row.update(simulated_creation_cohort_closure90=1-km_at(curve,90)[0],simulated_creation_cohort_rmst90=rmst(curve,90))
        if initial=="full":curve.to_csv(OUT/"validation_simulated_cohort_km.csv",index=False)
        say(f"Validation {label}: P50/P90/P95 {simulated.tolist()}, n={m['closed']:,}")
    primary=br[(~br.ambiguous)&br.created.between(day("2023-01-01"),day("2023-12-31"))]
    ev=primary.event.to_numpy()&(primary.closed.to_numpy()<=day("2023-12-31"))
    dur=np.where(ev,primary.closed-primary.created,day("2023-12-31")-primary.created).astype(np.int32)
    curve=km_curve(dur,ev);curve.to_csv(OUT/"validation_observed_cohort_km.csv",index=False)
    dump("validation_observed_cohort.json",dict(n=len(primary),closure90=1-km_at(curve,90)[0],rmst90=rmst(curve,90)))
    pd.DataFrame(rows).to_csv(OUT/"validation_controlled_ablation.csv",index=False)
    pd.DataFrame(dept_rows).to_csv(OUT/"validation_department_errors.csv",index=False)
    pd.DataFrame(band_rows).to_csv(OUT/"validation_retrospective_bands.csv",index=False)
    return model


def policy_bootstrap(r, results, draws=1000, seed=SEED, common=None):
    # Paired resampling of creation-week clusters, conditional on fitted replays.
    _,clusters=np.unique(r.created//7,return_inverse=True)
    nb=int(clusters.max())+1
    ng=len(r.starts)
    maxlat=max(int((z[0][z[0]>=0]-r.created[z[0]>=0]).max()) for z in results.values())+1
    matrices={};long={};den={};point=[]
    mature=(r.created>=r.start)&(r.created<=r.end-90)
    counts=np.bincount(clusters[mature],minlength=nb)
    if common is None:
        common=np.all([policy_metrics(r,z)[2]>=20 for z in results.values()],axis=0)
    for name,z in results.items():
        cd=z[0];closed=cd>=0
        col=r.group[closed]*maxlat+(cd[closed]-r.created[closed])
        matrices[name]=sparse.csr_matrix((np.ones(closed.sum()),(clusters[closed],col)),shape=(nb,ng*maxlat))
        lw=(~closed)|((cd-r.created)>90)
        long[name]=np.bincount(clusters[mature],weights=lw[mature],minlength=nb)
    boot={k:[] for k in results}
    rng=np.random.default_rng(seed)
    for begin in range(0,draws,10):
        w=rng.multinomial(nb,np.full(nb,1/nb),size=min(10,draws-begin))
        denominator=w@counts
        for name,matrix in matrices.items():
            hist=np.asarray(w@matrix).reshape(len(w),ng,maxlat)
            gcdf=hist.cumsum(axis=2);gn=gcdf[:,:,-1]
            totalcdf=hist.sum(axis=1).cumsum(axis=1);total=totalcdf[:,-1]
            pos=.95*(total-1);lo=np.floor(pos);hi=np.ceil(pos)
            qlo=(totalcdf>lo[:,None]).argmax(axis=1);qhi=(totalcdf>hi[:,None]).argmax(axis=1)
            p95=qlo+(qhi-qlo)*(pos-lo)
            gmpos=.5*(gn-1);glo=np.floor(gmpos);ghi=np.ceil(gmpos)
            medlo=(gcdf>glo[:,:,None]).argmax(axis=2);medhi=(gcdf>ghi[:,:,None]).argmax(axis=2)
            med=medlo+(medhi-medlo)*(gmpos-glo)
            chosen=med[:,common]
            cv=chosen.std(axis=1)/chosen.mean(axis=1)
            bad=(gn[:,common]==0).any(axis=1);cv[bad]=np.nan
            share=(w@long[name])/np.where(denominator,denominator,np.nan)
            boot[name].extend(np.column_stack((p95,share,cv)))
    rows=[]
    for name,values in boot.items():
        a=np.asarray(values)
        for j,metric in enumerate(("p95","long_wait_share","disparity_cv")):
            valid=a[:,j][np.isfinite(a[:,j])]
            low,high=np.quantile(valid,[.025,.975])
            rows.append(dict(policy=name,comparison="level",metric=metric,low=float(low),high=float(high),draws=len(valid),creation_week_clusters=nb))
        if name!="FIFO":
            delta=a-np.asarray(boot["FIFO"])
            for j,metric in enumerate(("p95","long_wait_share","disparity_cv")):
                valid=delta[:,j][np.isfinite(delta[:,j])];low,high=np.quantile(valid,[.025,.975])
                rows.append(dict(policy=name,comparison="minus_FIFO",metric=metric,low=float(low),high=float(high),draws=len(valid),creation_week_clusters=nb))
    return pd.DataFrame(rows)


def evaluate_policies(br,ny,br_model):
    all_rows=[];ci_rows=[];selections={};models={}
    cities=[("Brussels",br,br_model,"2023-01-01","2023-12-31",[2024,2025]),
            ("NYC",ny,group_model(ny,"2023-01-01","2023-12-31",3000),"2024-01-01","2024-12-31",[2025])]
    for city,d,model,train_start,train_end,years in cities:
        models[city]=model
        r=prepare_replay(d,model,train_start,train_end)
        say(f"Search {city}: {len(r.created):,} cases, {len(r.starts)} groups")
        starttime=time.time();candidates,chosen,common=candidate_search(r,n=32,seed=SEED+(city=="NYC"))
        candidates.to_csv(OUT/f"{city}_development_candidates.csv",index=False)
        weights=[float(chosen[f"w{i}"]) for i in range(1,5)]
        selections[city]=dict(weights=weights,selected_label=str(chosen.policy),selection_start=train_start,selection_end=train_end,
            common_development_groups=int(common.sum()),candidate_count=len(candidates),search_seconds=time.time()-starttime,
            group_threshold=model["threshold"],development_requests=model["development_requests"],pooled_development_requests=model["pooled_development_requests"])
        say(f"Selected {city}: {chosen.policy}, weights {weights}, search {time.time()-starttime:.1f}s")
        for year in years:
            rr=prepare_replay(d,model,f"{year}-01-01",f"{year}-12-31")
            policies={"FIFO":[1,0,0,0],"Queue pressure":[0,1,0,0],"Frozen local rule":weights}
            if city=="NYC":policies["Frozen Brussels rule"]=selections["Brussels"]["weights"]
            results={name:replay(rr,w) for name,w in policies.items()}
            common_groups=np.all([policy_metrics(rr,z)[2]>=20 for z in results.values()],axis=0)
            for name,z in results.items():
                m,med,n=policy_metrics(rr,z,common_groups=common_groups)
                all_rows.append(dict(city=city,year=year,policy=name,**m))
                for horizon in (60,90,180):
                    mm,_,_=policy_metrics(rr,z,maturity=horizon,common_groups=common_groups)
                    selections.setdefault("maturity",[]).append(dict(city=city,year=year,policy=name,horizon=horizon,share=mm["long_wait_share"],eligible=mm["mature_cohort_n"]))
                np.savez_compressed(OUT/f"outcomes_{city}_{year}_{name.replace(' ','_')}.npz",created=rr.created,group=rr.group,closed=z[0],backlog=z[1],original_index=rr.original_index)
                say(f"{city} {year} {name}: P95={m['p95']:.1f}, mature90={m['long_wait_share']:.4f}, CV={m['disparity_cv']:.4f}")
            # CIs for principal fixed-policy comparison, no repeated policy selection.
            bci=policy_bootstrap(rr,{k:results[k] for k in ("FIFO","Frozen local rule")},common=common_groups)
            bci["city"]=city;bci["year"]=year;ci_rows.append(bci)
    pd.DataFrame(all_rows).to_csv(OUT/"temporal_policy_evaluation.csv",index=False)
    pd.concat(ci_rows,ignore_index=True).to_csv(OUT/"policy_paired_bootstrap_intervals.csv",index=False)
    pd.DataFrame(selections.pop("maturity")).to_csv(OUT/"maturity_threshold_sensitivity.csv",index=False)
    dump("frozen_policy_parameters.json",selections)
    return selections,models


def robustness(br,ny,selections,models):
    sensitivity=[];stress=[]
    for city,d in [("Brussels",br),("NYC",ny)]:
        year=2024 if city=="Brussels" else 2025
        dev=("2019-01-01","2022-12-31") if city=="Brussels" else ("2023-01-01","2023-12-31")
        thresholds=(125,250,500) if city=="Brussels" else (1500,3000,6000)
        weights=selections[city]["weights"]
        for threshold in thresholds:
            model=group_model(d,*dev,threshold)
            for initial in ("empty","full"):
                r=prepare_replay(d,model,f"{year}-01-01",f"{year}-12-31",initialization=initial)
                results={"FIFO":replay(r,[1,0,0,0]),"Frozen local rule":replay(r,weights)}
                # Keep the same comparator roster as the main evaluation so that
                # the reference threshold/full-initialization CV agrees exactly.
                support_results=list(results.values())+[replay(r,[0,1,0,0])]
                if city=="NYC":support_results.append(replay(r,selections["Brussels"]["weights"]))
                common=np.all([policy_metrics(r,z)[2]>=20 for z in support_results],axis=0)
                for name,z in results.items():
                    m,_,_=policy_metrics(r,z,common_groups=common)
                    sensitivity.append(dict(city=city,year=year,threshold=threshold,initialization=initial,policy=name,**m))
        # Thirty paired input realizations, distinct from deterministic replay uncertainty.
        for scenario,ds,cs,repetitions in [("Demand +30%",1.3,1,30),("Combined severe",1.3,.8,30),("Throughput -30%",1,.7,1)]:
            for k in range(repetitions):
                r=prepare_replay(d,models[city],f"{year}-01-01",f"{year}-12-31",demand_scale=ds,capacity_scale=cs,seed=SEED+1000+k)
                results={"FIFO":replay(r,[1,0,0,0]),"Frozen local rule":replay(r,weights)}
                common=np.all([policy_metrics(r,z)[2]>=20 for z in results.values()],axis=0)
                for name,z in results.items():
                    m,_,_=policy_metrics(r,z,common_groups=common)
                    stress.append(dict(city=city,year=year,scenario=scenario,replicate=k+1,seed=SEED+1000+k,policy=name,**m))
            say(f"Stress {city} {scenario}: {repetitions} paired realizations complete")
    pd.DataFrame(sensitivity).to_csv(OUT/"grouping_initialization_sensitivity.csv",index=False)
    s=pd.DataFrame(stress);s.to_csv(OUT/"paired_stress_realizations.csv",index=False)
    summary=[]
    for (city,scenario),x in s.groupby(["city","scenario"]):
        a=x[x.policy=="FIFO"].set_index("replicate");b=x[x.policy=="Frozen local rule"].set_index("replicate")
        for metric in ("p95","long_wait_share","disparity_cv","mean_backlog","final_backlog"):
            delta=(b[metric]-a[metric]).to_numpy();n=len(delta);mean=float(delta.mean())
            se=float(delta.std(ddof=1)/math.sqrt(n)) if n>1 else np.nan
            margin=float(student_t.ppf(.975,n-1)*se) if n>1 else np.nan
            summary.append(dict(city=city,scenario=scenario,metric=metric,n_paired=n,FIFO_mean=float(a[metric].mean()),rule_mean=float(b[metric].mean()),delta_mean=mean,
                delta_mc_se=se,delta_mean_low=mean-margin,delta_mean_high=mean+margin,delta_p05=float(np.quantile(delta,.05)),delta_p95=float(np.quantile(delta,.95)),fraction_lower=float(np.mean(delta<0))))
    pd.DataFrame(summary).to_csv(OUT/"stress_paired_summary.csv",index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["audit","descriptive","validation","policies","robustness","all"],default="all")
    stage=parser.parse_args().stage
    br,ny,ba,na=load_data()
    if stage=="audit":
        print(json.dumps(dict(brussels=ba,nyc=na),indent=2,default=str),flush=True)
        return
    if stage in ("descriptive","all"):describe_lifecycles(br,ny)
    bm=group_model(br,"2019-01-01","2022-12-31",250)
    if stage in ("validation","all"):validation_diagnostics(br)
    if stage in ("policies","all"):selections,models=evaluate_policies(br,ny,bm)
    if stage=="robustness":
        selections=json.loads((OUT/"frozen_policy_parameters.json").read_text(encoding="utf-8"))
        models={"Brussels":bm,"NYC":group_model(ny,"2023-01-01","2023-12-31",3000)}
    if stage in ("robustness","all"):robustness(br,ny,selections,models)
    say(f"Stage {stage} complete")


if __name__=="__main__":main()
