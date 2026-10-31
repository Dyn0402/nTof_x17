#!/usr/bin/env python3
"""a1_job.py <key> <det> --table T --alignment AL --cand C --rays RAYS --out DIR -- the waveform side of Paper I
appendix A1 (loss categories) on one golden row's PASS table, run through rerun/stage_job.py (run_config, m3, hits
staged; the decoded waveforms read straight from EOS, rerun/inputs.plan, generation rr1).

The same code as the pre-pass study, with its data directory pointed at a scratch dir (olib.BASE) instead of
~/x17/cosmic_bench/outliers:
  1. rerun/outliers/join.py      the per-event join (wft + M3 as 03_angles + every ray + hits QA)
  2. display.extract             waveforms of the C22 outliers + 300 controls, of every interior 'spark' ray and of
                                 every event with >= 2 particle candidates in a view
  3. wffeat.py, classify.py      the C22 failure classes (M3 cluster trees, m3feat, are NOT rebuilt: they only fill
                                 the 'why' text of POS_TAIL / M3_ELSEWHERE, never the class)
  4. sparks (satstudy/sparks.py rules)   discharge / muon+other / shower for the interior spark rays
  5. candidate time extent (satstudy/multitrack.py feat)   particle (<= 24 samples above 20 % of peak) vs band
Writes DIR/{joined,classified,wffeat}_<det>.parquet, sparks_<det>.parquet, cand_feat_<det>.parquet.
"""
import argparse, os, runpy, shutil, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OL = os.path.join(REPO, 'rerun', 'outliers')
sys.path[:0] = [OL, REPO, os.path.join(REPO, 'rerun')]
EXT_MAX, DIS_ON, DIS_NEG, WIDE = 24, 150, 100, 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('det')
    for k in ('--table', '--alignment', '--cand', '--rays', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--n-controls', type=int, default=300)
    a = ap.parse_args()
    import inputs
    from wft import io as wio
    row = inputs.row_for(a.key)
    files, _, _, _ = inputs.plan(row, 'rr1')
    eos = {}
    for src, name in files['decoded_root']:
        eos.setdefault(int(name[-7:-5]), []).append(src)
    # decoded files from EOS, not from a staged tree (display.extract asks wio.subrun_files)
    wio.subrun_files = lambda base, run, sub, feu: sorted(eos.get(int(feu), []))

    work = os.path.join(os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'), 'a1'); os.makedirs(work, exist_ok=True)
    # det4's dead / low-gain band map (C22, keyed by DREAM event id: the same raw events as the pass row)
    live = os.path.expanduser('~/x17/cosmic_bench/outliers/an/live_det4.parquet')
    if a.det == 'det4' and os.path.exists(live):
        os.makedirs(os.path.join(work, 'an'), exist_ok=True)
        shutil.copyfile(live, os.path.join(work, 'an', 'live_det4.parquet'))
    import olib
    olib.BASE = work
    import display as D
    D.BASE = work
    D.KEYS[a.det] = a.key
    J = os.path.join(work, f'joined_{a.det}.parquet')
    sys.argv = ['join.py', a.key, '--table', a.table, '--alignment', a.alignment, '--out', J]
    runpy.run_path(os.path.join(OL, 'join.py'), run_name='__main__')

    import classify as K
    K.BASE = work
    out_ids, ctl = K.outlier_ids(a.det)
    rng = np.random.default_rng(1)
    ctl = rng.choice(ctl, size=min(a.n_controls, len(ctl)), replace=False).tolist()
    R = pd.read_parquet(a.rays)
    Ri = R[(R.region == 'interior') & ~R.not_measured]
    spark_ids = Ri.event_id[Ri.cat_xy_5 == 'spark'].astype(int).tolist()
    C = pd.read_parquet(a.cand); C = C[(C['rank'] >= 0) & C.plausible & C.event_id.isin(set(Ri.event_id))]
    nc = C.groupby(['event_id', 'plane']).size()
    multi_ids = sorted(set(nc[nc >= 2].index.get_level_values(0).astype(int)))
    want = sorted(set(out_ids) | set(ctl) | set(spark_ids) | set(multi_ids))
    print(f'outliers {len(out_ids)}  controls {len(ctl)}  sparks {len(spark_ids)}  multi {len(multi_ids)}', flush=True)
    wf = D.extract(a.det, want)

    import wffeat as F
    F.BASE = work
    sys.argv = ['wffeat.py', a.det]; F.main()
    pd.DataFrame(dict(event_id=pd.Series([], dtype='int64'))).to_parquet(os.path.join(work, f'm3feat_{a.det}.parquet'))
    sys.argv = ['classify.py', a.det]; K.main()

    pm = D.posmap(a.det)
    # 4. sparks (satstudy/sparks.py, unchanged rules)
    rows = []
    Rs = Ri.set_index('event_id')
    for e in spark_ids:
        if e not in wf:
            continue
        r = Rs.loc[e]; ev = wf[e]; o = dict(event_id=e, mult=int(r.mult)); clean = True; dis = False
        for v in 'xy':
            if v not in ev:
                clean = False; continue
            W = ev[v].astype(np.float32); n = np.maximum(ev['noise_' + v].astype(np.float32), 3.0); pos = pm[v]
            ok = np.isfinite(pos); snr = W.max(1) / n
            o[f'{v}_non'] = int((ok & (snr > 6)).sum()); o[f'{v}_nneg'] = int((ok & (W.min(1) / n < -10)).sum())
            dis |= o[f'{v}_non'] > DIS_ON or o[f'{v}_nneg'] > DIS_NEG
            f = F.feats(W, n, pos, r[f'mesh_{v}'], r[f'tan_{v}'])
            lo, hi = sorted((r[f'mesh_{v}'], r[f'mesh_{v}'] + 30 * r[f'tan_{v}']))
            single = f['snr_ref'] > 10 and f['dfoot1'] == 0 and f['wid1'] <= (hi - lo) + 5 + WIDE
            if f['dfoot1'] > 0 and f['snr_ref'] > 10:
                single = True
            clean &= single
        o['cls'] = 'discharge' if dis else ('muon+other' if clean else 'shower')
        rows.append(o)
    pd.DataFrame(rows).to_parquet(os.path.join(work, f'sparks_{a.det}.parquet'))
    # 5. candidate time extent (satstudy/multitrack.py feat)
    rows = []
    for (e, v), g in C[C.event_id.isin(multi_ids)].groupby(['event_id', 'plane']):
        ev = wf.get(int(e))
        if ev is None or v not in ev:
            continue
        W = ev[v].astype(np.float32); p = pm[v]
        for _, c in g.iterrows():
            mid = c.p0 + 15 * c.tan_theta
            sel = np.isfinite(p) & (np.abs(p - mid) < max(4.0, abs(15 * c.tan_theta) + 3))
            prof = W[sel].sum(0)
            pk = float(prof.max()) if len(prof) else 0.0
            rows.append(dict(event_id=int(e), plane=v, rank=int(c['rank']),
                             ext=int((prof > 0.2 * pk).sum()) if pk > 0 else 0, peak=float(W[sel].max()) if sel.any() else 0.0))
    pd.DataFrame(rows).to_parquet(os.path.join(work, f'cand_feat_{a.det}.parquet'))
    os.makedirs(a.out, exist_ok=True)
    for n in os.listdir(work):
        if n.endswith('.parquet'):
            shutil.copyfile(os.path.join(work, n), os.path.join(a.out, n))
    shutil.rmtree(work, ignore_errors=True)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
