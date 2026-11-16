#!/usr/bin/env python3
"""vt_acq.py -- P7a: the model-free column duration T on the pass's clean tracks, one (acquisition, arm).

    vt_acq.py <run> <subrun> <tag> <arm> --base RUNS_DIR --pass PASS_EVENTS.parquet --tmpl TMPL.npz --out OUT.npz
              [--jobs 4] [--max-events 0]

NTOF_PASS_PLAN §3 P7a.  The estimator is FA's colsum.py, unchanged (as gc_gas_time/ntof_colsum.py uses it): per
plane, the main cluster of >= 3 strips above 5 sigma, +-2 strips; strip sum S(t) = a box of duration T = gap / v
convolved with the electronics template (the pass-1 X template of the arm's bench bundle); (t0, T) on a grid, (a, b)
linear.  The sample is the pass's own: events of this acquisition whose pass row has exactly one track, x and y ok and
quality_ok, one candidate per plane (`clean`), and one colsum cluster per plane.  Saved per event: event_id, DREAM
timestamp, ftst, cluster summary, saturation flag, and per view the fitted t0, T, a, b, chi2/dof, plus the pass track's
tan_theta, and the strip sum S itself (vt_table.py fits the box to the stacked median S per acquisition).

The window is 20 x 60 ns = 1.2 us, so the box end can sit near the window end (C closest): absolute T is window-limited;
T against time within one arm and run condition is the product, and the gap's error carries into v (plan §6 11-02).
Inputs are local files (the condor wrapper vt_stage.sh stages them over xrootd).
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
import colsum                          # noqa: E402

T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)
T_GRID = np.arange(300.0, 1500.0 + 1e-9, 10.0)
SAT_RAW = 3700.0
FIELDS = ['ts', 'ftst', 'nsig', 'ncl', 'cen', 'rms', 'q', 'smax', 'sat']


def _grids():
    colsum.T0_GRID, colsum.T_GRID = T0_GRID, T_GRID


def fit_chunk(args):
    S, grid, tmpl, ns = args
    _grids()                                                   # a forkserver child re-imports colsum
    g = colsum.GridFit(colsum.Kern(grid, tmpl), ns)
    return np.array([g.fit(s.astype(float)) for s in S]) if len(S) else np.zeros((0, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=int); ap.add_argument('subrun'); ap.add_argument('tag'); ap.add_argument('arm')
    ap.add_argument('--base', required=True, help='runs dir: <base>/run_<r>/<subrun>/decoded_root/*.root')
    ap.add_argument('--pass', dest='pass_', required=True, help="the pass's events parquet of this (acquisition, arm)")
    ap.add_argument('--tmpl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--max-events', type=int, default=0)
    a = ap.parse_args()
    _grids()
    os.environ['WFT_BEAM_BASE'] = a.base.rstrip('/') + '/'
    import pandas as pd
    from ntof_tracking import wft_beam as wb
    from wft import io as wio

    ev = pd.read_parquet(a.pass_)
    clean = ((ev.n_tracks == 1) & ev.x_ok.astype(bool) & ev.y_ok.astype(bool) & ev.x_quality_ok.astype(bool)
             & ev.y_quality_ok.astype(bool) & (ev.x_n_candidates == 1) & (ev.y_n_candidates == 1))
    sel = ev[clean].set_index('event_id')
    want = set(sel.index.tolist())
    cfg = wb.beam_config(a.arm, f'run_{a.run}', a.subrun)
    pmaps = wio.strip_position_map(cfg)
    z = np.load(a.tmpl)
    grid, tmpl = z['grid'], z['tmpl_x'] / z['tmpl_x'].max()
    per = {}
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        files = [f for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu) if wio.file_tag(f) == a.tag]
        if len(files) != 1:
            raise SystemExit(f'{len(files)} decoded files for tag {a.tag} FEU {feu}')
        _, out = colsum.plane_pass((files[0], pmaps[feu], [], SAT_RAW - 330.0))
        per[plane] = {e: v for e, v in out.items() if e in want}
        print(f'[vt] {a.arm} {plane}: {len(out)} events with a cluster, {len(per[plane])} of the {len(want)} clean',
              flush=True)
    keys = sorted(set(per['x']) & set(per['y']))
    if a.max_events:
        keys = keys[:a.max_events]
    n = len(keys)
    rec = dict(eid=np.array(keys, np.int64))
    for p in 'xy':
        rec[f'{p}_tan_pass'] = sel.loc[keys, f'{p}_tan_theta'].to_numpy(np.float32) if n else np.zeros(0, np.float32)
        ns = len(next(iter(per[p].values()))[-1]) if per[p] else 20
        S = np.zeros((n, ns), np.float32)
        cols = {k: [] for k in FIELDS}
        for i, k in enumerate(keys):
            v = per[p][k]
            for j, name in enumerate(FIELDS):
                cols[name].append(v[j])
            S[i] = v[-1]
        for name, arr in cols.items():
            rec[f'{p}_{name}'] = np.asarray(arr)
        chunks = np.array_split(np.arange(n), max(1, a.jobs * 4))
        res = np.full((n, 5), np.nan)
        with ProcessPoolExecutor(a.jobs) as ex:
            for ch, r in zip(chunks, ex.map(fit_chunk, [(S[c], grid, tmpl, ns) for c in chunks])):
                if len(ch):
                    res[ch] = r
        for j, name in enumerate(['t0', 'T', 'a', 'b', 'chi2dof']):
            rec[f'{p}_{name}'] = res[:, j].astype(np.float32)
        # the strip sums themselves: the per-acquisition v is the box fit to their stacked median (vt_table.py, plan
        # §6 11-02); the per-event fits above are too noisy to average (T spread ~150 ns rms)
        rec[f'{p}_S'] = S
        print(f'[vt] {a.arm} {p}: {n} events fitted, median T {np.nanmedian(res[:, 1]) if n else np.nan:.0f} ns',
              flush=True)
    rec['meta'] = np.array([f'run={a.run} subrun={a.subrun} tag={a.tag} arm={a.arm} n_clean_pass={len(want)} '
                            f'T_grid={T_GRID[0]:.0f}-{T_GRID[-1]:.0f}/10 t0_grid={T0_GRID[0]:.0f}-{T0_GRID[-1]:.0f}/5'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **rec)
    print(f'[vt] wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
