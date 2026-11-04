#!/usr/bin/env python3
"""centroids.py <run> <subrun> --out OUT.npz [--arms ABCD] [--jobs 3]

Per trigger, arm and view: the main cluster of a plane as colsum.py finds it (pedestal + CM as wft.io.FeuReader reads
them; >= 3 strips above 5 sigma, gaps of <= 2 strips bridged, +-2 strips padding; the bundle's dead list left out),
its charge centroid [strip-map mm], rms width, strip count, cluster count, saturation flag, DREAM ftst, and its strip
sum S(t) (20 samples).  No fit.  Used by the D/B studies (rerun/ntof/tests/d_b/FINDINGS.md):

  * the B-D and A-C "centroid lines" (positions only, no angle fit in the truth),
  * D's column shape on beam vs cosmics (S stacks; column start per event).

Inputs: the staged sub-runs under /media/ucla/work/ntof_pass1/runs (decoded_root = fa80c06), strip maps through
ntof_tracking.wft_beam.beam_config, dead lists from the v1 bundles.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
BASE = os.environ.setdefault('WFT_BEAM_BASE', '/media/ucla/work/ntof_pass1/runs/')
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
import colsum  # noqa: E402

BUNDLES = '/media/ucla/work/ntof_pass1/bundles'
SAT_RAW = 3700.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('subrun'); ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='ABCD'); ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle
    rec = {}
    for arm in a.arms:
        cfg = wb.beam_config(arm, a.run, a.subrun)
        cfg.BASE_PATH = BASE
        pmaps = wio.strip_position_map(cfg)
        cal = CalibrationBundle.load(os.path.join(BUNDLES, f'v1_{arm}'))
        tasks, tags = [], {}
        for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            for f in wio.subrun_files(BASE, cfg.RUN, cfg.SUB_RUN, feu):
                tags[f] = (plane, wio.file_tag(f))
                tasks.append((f, pmaps[feu], list(cal.dead.get(plane, [])), SAT_RAW - 330.0))
        per = {'x': {}, 'y': {}}; acq = {}
        with ProcessPoolExecutor(a.jobs) as ex:
            for path, out in ex.map(colsum.plane_pass, tasks):
                plane, tag = tags[path]
                k = acq.setdefault(tag, len(acq))
                for eid, v in out.items():
                    per[plane][(k, eid)] = v
        keys = sorted(set(per['x']) | set(per['y']))
        n = len(keys)
        print(f'[centroids] {a.run} {a.subrun} {arm}: {n} triggers with a cluster', flush=True)
        if not n:
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
            for name, arr in cols.items():
                rec[f'{arm}_{p}_{name}'] = arr
            rec[f'{arm}_{p}_S'] = S
    rec['meta'] = np.array([f'run={a.run} subrun={a.subrun} arms={a.arms} base={BASE} dead=v1 bundles'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **rec)
    print(f'[centroids] wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
