import unittest
from collections import deque
import sys
from pathlib import Path
if Path(__file__).resolve().parent.name=="tests":
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analysis_core import *


def independent_fifo(r):
    waiting={g:deque() for g in range(len(r.starts))}
    closed=np.full(len(r.created),-1,np.int32)
    for today in range(r.start,r.end+1):
        for i in np.flatnonzero(r.created==today):waiting[int(r.group[i])].append(int(i))
        if today==r.start:
            for g in waiting:
                prior=np.flatnonzero((r.group==g)&(r.created<r.start)).tolist()
                waiting[g]=deque(prior+list(waiting[g]))
        for dep in range(r.capacities.shape[1]):
            for _ in range(r.capacities[today-r.start,dep]):
                active=[g for g in waiting if waiting[g] and r.group_dept[g]==dep]
                if not active:break
                selected=min(active,key=lambda g:(r.created[waiting[g][0]],g))
                closed[waiting[selected].popleft()]=today
    return closed


class ScientificChecks(unittest.TestCase):
    def toy(self):
        return Replay(np.array([0,2,1,2,1],np.int32),np.array([0,0,1,1,2],np.int32),np.arange(5),
            np.array([0,2,4]),np.array([2,4,5]),np.array([0,0,1],np.int32),
            np.array([[1,0],[1,1],[1,1],[2,0]],np.int32),np.array([20.,40.,30.]),1,4,np.array(["A|1","A|2","B|1"]))

    def test_fifo_matches_independent_reference(self):
        r=self.toy();np.testing.assert_array_equal(replay(r,[1,0,0,0])[0],independent_fifo(r))

    def test_conservation_and_unused_slots(self):
        r=self.toy();z=replay(r,[.1,.7,.1,.1]);m,_,_=policy_metrics(r,z,maturity=1,minimum_group_closures=1)
        self.assertEqual(m["initial_backlog"]+m["arrivals"]-m["closed"],m["final_backlog"])
        self.assertEqual(m["closed"]+m["unused_slots"],r.capacities.sum())

    def test_no_close_before_arrival(self):
        r=self.toy();cd=replay(r,[0,1,0,0])[0];closed=cd>=0
        self.assertTrue(np.all(cd[closed]>=r.created[closed]))

    def test_deterministic_score_ties(self):
        r=self.toy();np.testing.assert_array_equal(replay(r,[0,0,1,0])[0],replay(r,[0,0,1,0])[0])

    def test_zero_throughput_preserves_all_requests(self):
        r=self.toy();r.capacities[:]=0;z=replay(r,[1,0,0,0]);self.assertEqual(z[1][-1],len(r.created))

    def test_kaplan_meier_and_integral(self):
        c=km_curve([1,3],[True,False]);self.assertAlmostEqual(km_at(c,1)[0],.5)
        self.assertAlmostEqual(rmst(c,3),2.0)
        self.assertTrue(np.all(np.diff(c.survival)<=0))

    def test_complete_observation_rmst_equals_truncated_mean(self):
        values=np.array([0,7,30,95,300]);c=km_curve(values,np.ones(5,bool))
        self.assertAlmostEqual(rmst(c,90),np.minimum(values,90).mean())

    def test_logrank_identical_groups(self):
        statistic,p=logrank(np.array([1,2,3]),np.array([1,0,1],bool),np.array([1,2,3]),np.array([1,0,1],bool))
        self.assertAlmostEqual(statistic,0);self.assertAlmostEqual(p,1)

    def test_holm_monotone_adjustment(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_large_logrank_risk_sets_do_not_overflow(self):
        t1=np.r_[np.ones(100000,dtype=int),np.full(100000,2)]
        t2=np.r_[np.ones(50000,dtype=int),np.full(150000,2)]
        e1=t1==1;e2=t2==1
        statistic,p=logrank(t1,e1,t2,e2)
        variance=200000.0*200000.0*150000.0*250000.0/(400000.0**2*399999.0)
        self.assertAlmostEqual(statistic,25000.0**2/variance,places=5)
        self.assertGreaterEqual(p,0);self.assertLess(p,.001)

    def test_future_closures_cannot_change_development_risk(self):
        df=pd.DataFrame(dict(id=["a","b"],dept=["D","D"],category=["C","C"],status=["Closed","Closed"],
            created=[day("2022-01-01"),day("2022-02-01")],closed=[day("2024-01-01"),day("2025-01-01")],event=[True,True],ambiguous=[False,False],duration=[730,1000]))
        a=group_model(df,"2022-01-01","2022-12-31",1)
        df["closed"]+=400;df["duration"]+=400
        b=group_model(df,"2022-01-01","2022-12-31",1)
        np.testing.assert_array_equal(a["risk"],b["risk"])

    def test_nondominance(self):
        np.testing.assert_array_equal(nondominated([[1,2],[2,1],[3,3]]),[True,True,False])

    def test_impossible_legacy_closure_does_not_remove_pending_stock(self):
        from prepare_data import inherited_scopes
        self.assertIn("closed_date < created_date",inherited_scopes()["invalid_date"])
        raw=pd.DataFrame(dict(id=["pending","closed_bad","closed_valid","intraday_bad"],dept=["DOT"]*4,category=["C"]*4,
            status=["Pending","Closed","Closed","Closed"],created=["2010-01-01","2010-01-01","2010-01-01","2010-01-01T12:00:00"],
            closed=["1900-01-01","1900-01-01","2015-01-01","2010-01-01T11:00:00"]))
        d,audit=canonicalize(raw,"2026-10-20","fixture")
        self.assertEqual(audit["negative_closure"],3)
        self.assertEqual(audit["same_day_negative_timestamp"],1)
        self.assertEqual(audit["closed_status_missing_or_invalid_date"],2)
        model=group_model(d,"2023-01-01","2023-12-31",1)
        r=prepare_replay(d,model,"2025-01-01","2025-12-31")
        self.assertEqual(len(r.created),1)
        self.assertEqual(int(r.original_index[0]),0)
        self.assertEqual(int(replay(r,[1,0,0,0])[1][-1]),1)


if __name__=="__main__":unittest.main(verbosity=2)
