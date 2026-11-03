#!/usr/bin/env python3
"""single_track_v2.py [--root R] -- Paper I: the two loss groups of the skeptical single-track sample, looked at closely
(Dylan 2026-11-02 p.m.), and the revised single-track efficiency. Run after single_track.py (reads its per-event files).

1. One-view second deposits (det3 sees >= 2 candidates in one view, one in the other). Per event:
   the muon candidate and the second candidate in the multi view (by distance to the M3 ray), the single candidate
   of the other view; charge ratio ln(q_x/q_y) under "the other view holds the muon alone" vs "the two merged".
   Classes: 'merged in projection' (other-view candidate >= 15 strips [single tracks: 12], or its charge fits the sum
   and not the muon alone, or it sits 5-15 mm off the muon, i.e. between the two within the 12 mm seed gap);
   'faint, one view only' (on the muon, charge = muon alone); 'unclear'.
2. Spark-vetoed deposits (> 50 strips over the floor; rerun/satstudy/sparks.py classes discharge / muon+other /
   shower): extra M3 clusters in the upper (z 1302/1185) and lower (144/24) doublets, against detected muons and
   events with an independent second particle (sat_det3 only: the M3 cluster trees exist for that row).
3. Single-track efficiency v2: removes (i) independent second particles [single_track.py], (ii) two particles merged
   in one projection, (iii) clean muon + separate deposit vetoed together; the delta-ray deposits stay as losses
   (and a variant without them).
Writes results/single_track_v2.json, results/oneview_events.parquet.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import side_tree

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
CREDIT = ('detected', 'multi:m3_picked', 'multi:other_picked')
LQ_MED, LQ_SIG, NSTRIP_SINGLE = -0.056, 0.062, 12


def oneview(root, key, rel, D, analysis=side_tree.PASS1, eff=side_tree.EFF):
    B = pd.read_parquet(os.path.join(side_tree.eff_dir(root, eff), 'budget', rel, 'rays_det3.parquet')).drop_duplicates('event_id').set_index('event_id')
    C = pd.read_parquet(side_tree.cand_path(root, analysis, rel))
    C = C[C['rank'] >= 0]
    out = []
    for e in D.index[D.one_view_only & D.det3_multi]:
        g = C[C.event_id == e]; b = B.loc[e]
        def view(v):
            gv = g[g.plane == v].copy(); gv['d'] = (gv.p0 - b[f'mesh_{v}']).abs(); return gv.sort_values('d')
        gx, gy = view('x'), view('y')
        vm, gm, go = ('x', gx, gy) if len(gx) >= 2 else ('y', gy, gx)
        mu, sec = gm.iloc[0], gm.iloc[1]; oth = go.iloc[0] if len(go) else None
        q_o = oth.q_sum if oth is not None else np.nan
        lq_mu = np.log(mu.q_sum / q_o) if vm == 'x' else np.log(q_o / mu.q_sum)
        lq_mg = np.log((mu.q_sum + sec.q_sum) / q_o) if vm == 'x' else np.log(q_o / (mu.q_sum + sec.q_sum))
        out.append(dict(row=key, event_id=int(e), recat=D.recat[e], multi_view=vm, sec_sep=abs(sec.p0 - mu.p0),
                        q_mu=mu.q_sum, q_sec=sec.q_sum, q_other=q_o, other_d=oth.d if oth is not None else np.nan,
                        other_nstrips=oth.n_strips if oth is not None else np.nan, sec_nstrips=sec.n_strips,
                        z_mu=(lq_mu - LQ_MED) / LQ_SIG, z_mg=(lq_mg - LQ_MED) / LQ_SIG))
    O = pd.DataFrame(out)
    if not len(O):
        return O
    merged = (O.other_nstrips >= 15) | ((O.z_mg.abs() < 3) & (O.z_mu.abs() >= 3)) | ((O.other_d >= 5) & (O.other_d < 15))
    faint = ~merged & (O.other_d < 5) & (O.z_mu.abs() < 3)
    O['ov_cls'] = np.select([merged, faint], ['merged in projection', 'faint, one view only'], 'unclear')
    return O


def m3_doublets(root):
    import uproot
    rows = []
    for f in sorted(glob.glob(os.path.join(root, 'x17rerun/m3clus/*saturday*_analyse.root'))):
        a = uproot.open(f)['T'].arrays(['evn', 'MGv2_NClus'], library='np'); n = a['MGv2_NClus']
        up, dn = [1, 3, 0, 2], [5, 6, 4, 7]                      # plane index -> layers 0,1 (upper) / 2,3 (lower)
        rows += [dict(event_id=int(e), up_extra=int(np.maximum(n[i, up] - 1, 0).sum()),
                      dn_extra=int(np.maximum(n[i, dn] - 1, 0).sum())) for i, e in enumerate(a['evn'])]
    return pd.DataFrame(rows).set_index('event_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--analysis', default=side_tree.PASS1, help='tree holding the candidate tables')
    ap.add_argument('--eff', default=side_tree.EFF, help='efficiency-chain dir (budget/, a1/) under --root, or absolute; '
                    'paper1_eff_twotrack = the chamber A two-track re-pass products (read directly, no overlay)')
    ap.add_argument('--m3root', default=None, help='root holding x17rerun/m3clus (default --root)')
    ap.add_argument('--res', default=RES, help='directory of single_track_events_*.parquet (read) and the outputs (written)')
    a = ap.parse_args(); RESD = a.res; os.makedirs(RESD, exist_ok=True)
    res, acc, Os = {'oneview': {}, 'single_track_v2': {}}, {}, []
    for key, rel in ROWS.items():
        D = pd.read_parquet(os.path.join(RESD, f'single_track_events_{key}.parquet'))
        O = oneview(a.root, key, rel, D, a.analysis, a.eff); Os.append(O)
        res['oneview'][key] = {c: dict(n=int(len(g)), credited=float(g.recat.isin(CREDIT[1:]).mean()),
                                       q_sec_over_q_mu=float((g.q_sec / g.q_mu).median()),
                                       other_nstrips=float(g.other_nstrips.median()), sec_sep_mm=float(g.sec_sep.median()))
                               for c, g in O.groupby('ov_cls')}
        o = O.set_index('event_id').ov_cls.reindex(D.index) if len(O) else pd.Series(index=D.index, dtype=object)
        drop = D.recat.isin(['far:POS_TAIL', 'far:EDGE']); cr = D.recat.isin(CREDIT)
        base = ~D.indep_2nd & ~o.eq('merged in projection') & ~D.recat.eq('spark:muon+other') & ~drop
        sels = {'v1: (i) removed': ~D.indep_2nd & ~drop, 'v2: (i)-(iii) removed': base,
                'v2, delta deposits (spark:shower) removed too': base & ~D.recat.eq('spark:shower')}
        r = {}
        for nm, m in sels.items():
            n, k = int(m.sum()), int((cr & m).sum()); p = k / n
            r[nm] = dict(n=n, k=k, eff=100 * p, err=100 * np.sqrt(p * (1 - p) / n))
            acc.setdefault(nm, [0, 0]); acc[nm][0] += n; acc[nm][1] += k
        r['v2 losses, % of sample'] = {c: 100 * v / base.sum() for c, v in D[base & ~cr].recat.value_counts().items()}
        res['single_track_v2'][key] = r
        if key == 'sat_det3':
            M = m3_doublets(a.m3root or a.root); X = D.join(M)
            grp = np.where(X.indep_2nd, 'independent 2nd particle', X.recat)
            res['m3_doublet_extra_clusters_sat_det3'] = {
                gname: dict(n=int(len(g)), upper=float(g.up_extra.mean()), lower=float(g.dn_extra.mean()))
                for gname, g in X.groupby(grp) if gname in ('detected', 'independent 2nd particle', 'spark:shower',
                                                             'spark:muon+other', 'spark:discharge', 'far:DELTA', 'far:FIT_SAT')}
    res['single_track_v2']['pooled'] = {nm: dict(n=n, eff=100 * k / n, err=100 * np.sqrt(k / n * (1 - k / n) / n))
                                        for nm, (n, k) in acc.items()}
    pd.concat(Os).to_parquet(os.path.join(RESD, 'oneview_events.parquet'))
    json.dump(res, open(os.path.join(RESD, 'single_track_v2.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'oneview'}, indent=1)[:3500])
    print(json.dumps(res['oneview'], indent=1))


if __name__ == '__main__':
    main()
