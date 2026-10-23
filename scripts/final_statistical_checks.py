from analysis_core import *
from run_revision_analysis import WORK,OUT,DATA_DIR,load_data,say,dump
from prepare_data import fetch,url


def main():
    br,ny,ba,na=load_data()
    cohort=br[(~br.ambiguous)&br.created.between(day("2019-01-01"),day("2025-12-31"))]
    supported=cohort.groupby("dept").size();supported=supported[supported>=1000]
    reference=supported.idxmax()
    _,blocks=np.unique((cohort.created.to_numpy()-day("2019-01-01"))//28,return_inverse=True)
    nb=int(blocks.max())+1
    rng=np.random.default_rng(SEED+500)
    w=rng.multinomial(nb,np.full(nb,1/nb),size=2000)
    probs={};points={}
    for dep in supported.index:
        mask=cohort.dept.to_numpy()==dep
        total=np.bincount(blocks[mask],minlength=nb)
        success=cohort.event.to_numpy()[mask]&(cohort.duration.to_numpy()[mask]<=90)
        yes=np.bincount(blocks[mask],weights=success,minlength=nb)
        denominator=w@total
        probs[dep]=np.divide(w@yes,denominator,out=np.full(2000,np.nan),where=denominator>0)
        points[dep]=float(success.mean())
    comparisons=[]
    for dep in supported.index:
        if dep==reference:continue
        delta=probs[dep]-probs[reference]
        valid=delta[np.isfinite(delta)]
        estimate=points[dep]-points[reference];se=float(np.std(valid,ddof=1))
        low,high=np.quantile(valid,[.025,.975])
        comparisons.append(dict(department=dep,reference=reference,n=int(supported[dep]),closure90=points[dep],reference_closure90=points[reference],
            difference=estimate,low=float(low),high=float(high),block_bootstrap_se=se,p=float(2*norm.sf(abs(estimate/se))) if se else 0.0,draws=len(valid)))
    frame=pd.DataFrame(comparisons);frame["p_holm"]=holm(frame.p)
    frame.to_csv(OUT/"department_reference_block_contrasts.csv",index=False)
    # Reproduce the submitted hybrid status/date endpoint and identify its source.
    oldmask=br.closed.between(day("2023-01-01"),day("2023-12-31"))
    old=br[oldmask]
    oldstatus=old.status.str.upper().eq("CLOSED")
    hybrid=np.where(oldstatus,old.closed-old.created,day("2026-09-24")-old.created)
    endpoint=dict(closure_year_records=len(old),nonclosed_status_with_recorded_closure=int((~oldstatus).sum()),
        recorded_timestamp_quantiles=dict(zip(["p50","p90","p95"],np.quantile(old.closed-old.created,[.5,.9,.95]).tolist())),
        submitted_hybrid_quantiles=dict(zip(["p50","p90","p95"],np.quantile(hybrid,[.5,.9,.95]).tolist())))
    dump("submitted_endpoint_reconciliation.json",endpoint)
    # Official SLA metadata is audited, never used to assign unmatched deadlines.
    sla_path=DATA_DIR/"nyc_sla_snapshot.json"
    if not sla_path.exists():
        sla_path.write_bytes(fetch(url("cs9t-e3x8",select="agency,problem,problem_details,additional_details,sla_days",limit=10000)))
    data=sla_path.read_bytes()
    sla=pd.DataFrame(json.loads(data))
    agency_codes={"Department of Transportation":"DOT","Department of Environmental Protection":"DEP",
        "Department of Parks and Recreation":"DPR","Department of Sanitation":"DSNY"}
    sla["agency"]=sla.agency.map(agency_codes)
    sla=sla[sla.agency.notna()]
    counts=sla.groupby(["agency","problem"]).sla_days.nunique(dropna=False)
    ambiguous=int((counts>1).sum());single=int((counts==1).sum())
    # Metadata varies by detailed category and has no historical validity interval.
    matched=ny[(~ny.ambiguous)&ny.created.between(day("2023-01-01"),day("2025-12-31"))].groupby(["dept","category"]).size()
    matched_n=sum(int(n) for key,n in matched.items() if key in counts.index)
    unambiguous_n=sum(int(n) for key,n in matched.items() if key in counts.index and counts.loc[key]==1)
    dump("sla_mapping_audit.json",dict(selected_agency_rows=len(sla),agency_problem_pairs=len(counts),
        pairs_with_multiple_sla_values=ambiguous,pairs_with_single_sla_value=single,
        matched_primary_requests=matched_n,single_value_primary_requests=unambiguous_n,total_primary_current_requests=int(matched.sum()),
        statement="SLA is not used as a historical closure deadline: detailed matching fields and historical validity are not established."))
    say(f"Reference department {reference}; endpoint reconciliation and SLA audit complete")


if __name__=="__main__":main()
