#!/usr/bin/env python3
"""zs_scan_job.py <registry key> --out DIR [--per-plateau 1500] [--views x] [--jobs 4] [--work DIR]

The head-on footprint grid (headon_sigma.py) on a REAL zero-suppressed flat SPS row, with the suppressed samples fitted
as measured zeros (--censor 0) -- the treatment that reproduces RAW for the X view (sigma0_X -0.02 mm, Dp_X unchanged;
Y's sigma0/D_rc split is not recoverable from ZS, ws_sps 2026-10-27).  Builds the row's events from the decoded FEU 03
files on EOS (sps_build_cache.build_cache, self-seeded), re-windows each view to the full +-4 strips by position around
its brightest strip (channels the FEU suppressed entirely become zero rows), then runs headon_sigma.py fit on it.
Rows: the flat ZS plateaus of rerun/sps_registry.csv (run_55 CO2 drift ladder, run_63 flat700, run_66 resist scan,
run_70 drift scan).
"""
import argparse, csv, os, pickle, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'rerun', 'model_audit'), os.path.join(REPO, 'sps_beam_test_26', 'analysis'),
                os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

NOISE_ADC = 10.0


def rewindow(rec, nsmp):
    out = dict(eid=rec['eid'], tan_x=0.0, tan_y=0.0, ftst_x=0, ftst_y=0)
    for v in 'xy':
        P = rec[v]; W = np.asarray(P['W'], np.float32); ch = np.asarray(P['ch'], int)
        c = int(ch[int(np.argmax(W.max(1)))]); pc = POSITION_MM[c]
        want = np.flatnonzero((VIEW == v) & (np.abs(POSITION_MM - pc) <= 4.5 * PITCH_MM))
        want = want[np.argsort(POSITION_MM[want])]
        if len(want) != 9:
            return None
        Wn = np.zeros((9, nsmp), np.float32)
        for i, w in enumerate(want):
            j = np.flatnonzero(ch == w)
            if len(j):
                Wn[i] = W[j[0]]
        out[v] = dict(ch=want.astype(np.int16), pos=POSITION_MM[want].astype(np.float32), W=Wn,
                      noise=np.full(9, NOISE_ADC, np.float32))
        out[f'ref_mesh_{v}'] = float(pc)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('key'); ap.add_argument('--out', required=True)
    ap.add_argument('--per-plateau', type=int, default=1500); ap.add_argument('--views', default='x')
    ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', '/tmp'))
    a = ap.parse_args()
    row = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}[a.key]
    import sps_build_cache as sbc
    import datasets
    nsmp = int(datasets.get(row['dataset'])['n_samples'])
    evs = {}
    for sub in row['subruns'].split():
        got = sbc.build_cache(row['run'], sub, row['dataset'], per_plateau=a.per_plateau - len(evs), half_win=6, tan=0.0,
                              only=[row['plateau']], verbose=False).get(row['plateau'], {})
        for k, r in got.items():
            w = rewindow(r, nsmp)
            if w is not None:
                evs[k] = w
        print(f'[zs_scan] {sub}: {len(evs)} events', flush=True)
        if len(evs) >= a.per_plateau:
            break
    cache = os.path.join(a.work, f'zsscan_{a.key}.pkl'); pickle.dump(evs, open(cache, 'wb'), protocol=4)
    cmd = [sys.executable, os.path.join(HERE, 'headon_sigma.py'), 'fit', '--cache', cache, '--tag', f'zsscan_{a.key}',
           '--censor', '0', '--views', a.views, '--nev', str(a.per_plateau), '--jobs', str(a.jobs), '--out', a.out]
    rc = subprocess.call(cmd)
    os.remove(cache)
    sys.exit(rc)


if __name__ == '__main__':
    main()
