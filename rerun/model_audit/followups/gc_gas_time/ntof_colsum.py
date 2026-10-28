#!/usr/bin/env python3
"""ntof_colsum.py <run> <subrun> --tmpl-dir D --out OUT.npz [--arms ABCD] [--jobs 4]

GC: model-free column duration T of the four n_TOF chambers vs time, one sub-run.  FA's colsum.py estimator unchanged
(pedestal + CM as wft.io.FeuReader reads them; main cluster of >= 3 strips above 5 sigma, +-2 strips; strip sum S(t) =
box of duration gap/v convolved with the electronics template; (t0, T) grid + linear (a, b)), on the beam decoded files
(<EOS>/july_beam/runs/<run>/<subrun>/decoded_root, read through the EOS fuse mount) with the beam strip maps
(ntof_tracking.wft_beam.beam_config).  Template: each arm's bench seed bundle tmpl_x (tmpl_<arm>.npz).  No dead list
(the beam bundles' masks are applied by the reconstruction, not needed for a strip sum of a bright cluster), saturation
at SAT_ADC_BEAM - pedestal.

n_TOF window: 20 samples x 60 ns = 1.2 us against a column of ~0.7-0.9 us, so the box end can sit near the window end:
the absolute T is window-limited; T vs time within one arm and run condition is what is used.
Per arm and trigger saved: acquisition index (file stem), eventId, DREAM timestamp, ftst, cluster summary, saturation
flag, fitted t0, T, a, b, chi2/dof per view.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CODE = os.environ.get('WFT_REPO') or '/eos/user/d/dneff/x17rerun/followups/fa/code_2ce1dcc'
sys.path[:0] = [CODE, os.path.join(CODE, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
os.environ.setdefault('WFT_BEAM_BASE', '/eos/experiment/ntof/data/x17/july_beam/runs/')
import colsum                          # noqa: E402

colsum.T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)
colsum.T_GRID = np.arange(300.0, 1500.0 + 1e-9, 10.0)
SAT_RAW = 3700.0


def fit_chunk(args):
    S, grid, tmpl, ns = args
    colsum.T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)      # set again: a forkserver child re-imports colsum
    colsum.T_GRID = np.arange(300.0, 1500.0 + 1e-9, 10.0)
    g = colsum.GridFit(colsum.Kern(grid, tmpl), ns)
    return np.array([g.fit(s.astype(float)) for s in S])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('subrun'); ap.add_argument('--tmpl-dir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--arms', default='ABCD')
    ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--max-files', type=int, default=0)
    a = ap.parse_args()
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    rec = {}
    for arm in a.arms:
        cfg = wb.beam_config(arm, a.run, a.subrun)
        pmaps = wio.strip_position_map(cfg)
        z = np.load(os.path.join(a.tmpl_dir, f'tmpl_{arm}.npz'))
        grid, tmpl = z['grid'], z['tmpl_x'] / z['tmpl_x'].max()
        tasks, tags = [], {}
        for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            files = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
            if a.max_files:
                files = files[:a.max_files]
            for f in files:
                tags[f] = (plane, wio.file_tag(f))
                tasks.append(f)
        per = {'x': {}, 'y': {}}; acq = {}; ped = {}
        with ProcessPoolExecutor(a.jobs) as ex:
            # saturation threshold in pedestal-subtracted ADC: the rail minus a typical pedestal (~330)
            args = [(f, pmaps[cfg.MX17_FEU_X if tags[f][0] == 'x' else cfg.MX17_FEU_Y], [], SAT_RAW - 330.0)
                    for f in tasks]
            for path, out in ex.map(colsum.plane_pass, args):
                plane, tag = tags[path]
                k = acq.setdefault(tag, len(acq))
                for eid, v in out.items():
                    per[plane][(k, eid)] = v
                print(f'[ntof_colsum] {arm} {os.path.basename(path)}: {len(out)} events with a cluster', flush=True)
        keys = sorted(set(per['x']) | set(per['y']))
        n = len(keys)
        if n == 0:
            continue
        ns = next(len(v[-1]) for p in 'xy' for v in per[p].values())
        rec[f'{arm}_acq'] = np.array([k[0] for k in keys], np.int16)
        rec[f'{arm}_eid'] = np.array([k[1] for k in keys], np.int64)
        rec[f'{arm}_acq_tags'] = np.array(sorted(acq, key=acq.get))
        for p in 'xy':
            cols = dict(ts=np.full(n, -1, np.int64), ftst=np.full(n, -1, np.int16), nsig=np.zeros(n, np.int16),
                        ncl=np.zeros(n, np.int16), cen=np.full(n, np.nan, np.float32),
                        rms=np.full(n, np.nan, np.float32), q=np.full(n, np.nan, np.float32),
                        smax=np.full(n, np.nan, np.float32), sat=np.zeros(n, bool), has=np.zeros(n, bool))
            S = np.zeros((n, ns), np.float32)
            for i, k in enumerate(keys):
                v = per[p].get(k)
                if v is None:
                    continue
                cols['has'][i] = True
                for j, name in enumerate(['ts', 'ftst', 'nsig', 'ncl', 'cen', 'rms', 'q', 'smax', 'sat']):
                    cols[name][i] = v[j]
                S[i] = v[-1]
            idx = np.flatnonzero(cols['has'])
            res = np.full((n, 5), np.nan)
            chunks = np.array_split(idx, max(1, a.jobs * 4))
            with ProcessPoolExecutor(a.jobs) as ex:
                for ch, r in zip(chunks, ex.map(fit_chunk, [(S[c], grid, tmpl, ns) for c in chunks])):
                    if len(ch):
                        res[ch] = r
            for name, arr in cols.items():
                rec[f'{arm}_{p}_{name}'] = arr
            for j, name in enumerate(['t0', 'T', 'a', 'b', 'chi2dof']):
                rec[f'{arm}_{p}_{name}'] = res[:, j].astype(np.float32)
            # the stacked strip sum is refitted offline from per-block medians of the normalised S
            rec[f'{arm}_{p}_S'] = S
            print(f'[ntof_colsum] {arm} {p}: {n} triggers, fitted {np.isfinite(res[:, 1]).sum()}', flush=True)
    rec['meta'] = np.array([f'run={a.run} subrun={a.subrun} arms={a.arms} ns_grid={len(colsum.T_GRID)}'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **rec)
    print(f'[ntof_colsum] wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
