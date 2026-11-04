#!/usr/bin/env python3
"""ntof_planes.py -- n_TOF single-track planes as a waveform cache, in the bench head-on cache format.

    ntof_planes.py --reco R.parquet --run 145 --subrun stat090_0000 --arm A --out C.pkl [--tan-max 0.08] [--planes x]

Selection, from the reconstruction (ntof_reco.py events table): the plane was fitted (``<p>_ok``), the event has at
most one track, and |fitted tan| < --tan-max (the head-on cut proper is applied by the consumers).  For each selected
plane the window is re-read from decoded_root through wft.io (pedestal + plain block CM, the reco's own recipe): every
strip within PAD_MM of the fitted track's extent [p0, p0 + w T] (T = the plane's q_uend), as
sps_beam_test_26/analysis/cloud_basics/headon_cache.py does around the M3 track on the bench.

Event format (rc_diffusion.stack / template_resid.py): {eid: {'eid', 'tan_<p>', 'ref_mesh_<p>' (= the fitted p0),
'ftst_<p>', <p>: {ch, pos, W, noise}, 'fit_<p>': {p0, w, t0, tan, q_sum, q_uend, chi2, dof, n_sat_strips}, 'acq_tag'}}.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'rerun', 'ntof'))
PAD_MM = 5.0


def build(reco, run, subrun, arm, base, tan_max=0.08, planes='x'):
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from ntof_reco import FEUS
    d = pd.read_parquet(reco)
    if 'acq_tag' not in d:                 # tables older than the per-row acquisition column: one-tag runs only
        import json
        tags = json.load(open(reco.replace('.parquet', '.meta.json'))).get('tags_done') or []
        if len(tags) != 1:
            raise SystemExit(f'{reco}: no acq_tag column and {len(tags)} tags')
        d['acq_tag'] = tags[0]
    fx, fy = FEUS[arm]
    cfg = wb.BeamConfig(KEY=f'run_{run}_{subrun}_mx17_{arm}', RUN=f'run_{run}', SUB_RUN=subrun, DET_NAME=f'mx17_{arm}',
                        MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=base, OUT_BASE='/tmp')
    pos_maps = wio.strip_position_map(cfg)
    out = {}
    for plane, feu in (('x', fx), ('y', fy)):
        if plane not in planes:
            continue
        m = d[f'{plane}_ok'].astype(bool) & (d['n_tracks'].fillna(0) <= 1) & (d[f'{plane}_tan_theta'].abs() < tan_max)
        sel = d[m]
        pm = pos_maps[feu]
        for tag, g in sel.groupby('acq_tag'):
            f = [x for x in wio.subrun_files(base, f'run_{run}', subrun, feu) if wio.file_tag(x) == tag]
            if not f:
                continue
            rows = {int(r.event_id): r for r in g.itertuples()}
            rdr = wio.FeuReader(f[0])
            for eid, ftst, W in rdr.iter_events(set(rows)):
                r = rows[eid]
                p0, w, T = getattr(r, f'{plane}_p0'), getattr(r, f'{plane}_w'), getattr(r, f'{plane}_q_uend')
                a, b = p0, p0 + w * (T if np.isfinite(T) else 0.0)
                ch = np.where((pm >= min(a, b) - PAD_MM) & (pm <= max(a, b) + PAD_MM))[0]
                if len(ch) < 5:
                    continue
                ch = ch[np.argsort(pm[ch])]
                e = out.setdefault(eid, dict(eid=eid, acq_tag=tag))
                e[f'tan_{plane}'] = float(getattr(r, f'{plane}_tan_theta'))
                e[f'ref_mesh_{plane}'] = float(p0)
                e[f'ftst_{plane}'] = int(ftst)
                e[plane] = dict(ch=ch.astype(np.int16), pos=pm[ch].astype(np.float32), W=W[ch].astype(np.float32),
                                noise=np.maximum(rdr.noise[ch], 3.0).astype(np.float32))
                e[f'fit_{plane}'] = {k: float(getattr(r, f'{plane}_{k}')) for k in
                                     ('p0', 'w', 't0', 'tan_theta', 'q_sum', 'q_uend', 'chi2', 'dof', 'n_sat_strips')}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reco', required=True)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tan-max', type=float, default=0.08)
    ap.add_argument('--planes', default='x')
    ap.add_argument('--base', default='/media/ucla/work/ntof_pass1/runs/')
    a = ap.parse_args()
    ev = build(a.reco, a.run, a.subrun, a.arm, a.base, a.tan_max, a.planes)
    pickle.dump(ev, open(a.out, 'wb'))
    print(f'{len(ev)} events -> {a.out}')


if __name__ == '__main__':
    main()
