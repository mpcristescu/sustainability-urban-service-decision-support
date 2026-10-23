"""Freeze identifier-free cohort counts and verify exact expansion of analysis fields."""
from analysis_core import *
import pickle
from run_revision_analysis import WORK,OUT,DATA_DIR,dump,say


def main():
    br,ny,ba,na=pickle.loads((OUT/"canonical_data_v2.pkl").read_bytes())
    fields=["dept","category","created","closed","event","ambiguous","duration","status"]
    manifest=[]
    for name,d in [("brussels",br),("nyc",ny)]:
        ordered=stable_canonical_order(d)
        cells=ordered.groupby(fields,dropna=False,sort=True).size().rename("count").reset_index()
        path=DATA_DIR/f"{name}_cohort_cells.csv.gz"
        cells.to_csv(path,index=False,compression={"method":"gzip","mtime":0},encoding="utf-8")
        expanded=load_derived_cells(path)
        pd.testing.assert_frame_equal(ordered[fields].astype({"dept":str,"category":str,"status":str}).reset_index(drop=True),
            expanded[fields].astype({"dept":str,"category":str,"status":str}).reset_index(drop=True),check_dtype=False)
        manifest.append(dict(file=path.name,cohort_cells=len(cells),expanded_records=len(expanded),sha256=sha256(path),bytes=path.stat().st_size,
            contains_request_identifiers=False,contains_addresses=False,contains_narratives=False))
        say(f"Frozen {name}: {len(cells):,} cells, {len(expanded):,} records, {path.stat().st_size/1e6:.1f} MB")
    dump("derived_cells_manifest.json",dict(fields=fields+["count"],files=manifest,source="Frozen canonical public administrative snapshots",exact_expansion_verified=True))


if __name__=="__main__":main()
