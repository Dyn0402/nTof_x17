#!/usr/bin/env python3
"""boot_extract.py <key> --ids IDS.json --out OUT.pkl -- the data side of the residual bootstrap (BOOTSTRAP_2026-11-01.md).

Runs inside rerun/stage_job.py (--no-sync: nothing is written to the row's Analysis tree).  For the clean events of one
calibration row (boot_ids.py), the production reconstruction is replayed event by event exactly as wft.reco does it
(the same hits, seeds, windows, fit_plane_candidates with the bundle's t0 prior), and for every clean plane it keeps
the window and the fitted model on it:

  W, M      the gain-corrected window (wft.model.prep_plane) and the pass model on it (wft.model.model_waveforms at the
            fit's p0, w, t0 and depth profile q)  ->  the plane's fit residual R = W - M
  noise, censor, gain, pos, ch, ftst, t0_prior     to rebuild a raw window around another model
  p0, w, t0, chi2, dof, q   the fit, and the table's own w for the same event (closure: the replay must reproduce it)

READ-ONLY: the row's events.parquet and calib_bundle are read from the pulled Analysis tree; the output goes to OUT
(an xrootd URL is copied with xrdcp).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import pandas as pd
    import qa_config
    from wft import io as wio, model as wm, reco, seed as wseed
    cfg = qa_config.RUNS[a.key]
    wdir = os.path.join(cfg.OUT_BASE, 'wft')
    bundle = os.path.join(wdir, 'calib_bundle')
    reco._worker_init(bundle)
    cal = reco._CAL
    ids = json.load(open(a.ids))
    want_p = {p: set(int(e) for e in ids[p]['event_id']) for p in ('x', 'y')}
    tab = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    tab = tab[tab.event_id.isin(want_p['x'] | want_p['y'])]
    tw = {(int(r.event_id), str(r.acq_tag), p): float(getattr(r, f'{p}_w')) for r in tab.itertuples() for p in 'xy'}

    pos_maps = wio.strip_position_map(cfg)
    hits = reco._load_hits(cfg)
    hits, _pg = reco.pedestal_guard(hits, cfg)
    dead = (getattr(cal, 'dead', None) or {}) if wseed.DEAD_SEED else {}
    seeds = wseed.seeds_from_hits(hits, pos_maps, cfg.MX17_FEU_X, cfg.MX17_FEU_Y, hot=cal.hot, dead=dead)
    wanted = (want_p['x'] | want_p['y']) & set(seeds)
    wanted = {e for e in wanted if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    if wm.BAND_NSIG > 0:
        hits = hits.set_index('eventId')
        hits = hits[hits.index.isin(wanted)]
    else:
        hits = None
    print(f'[boot] {a.key}: {len(wanted)} clean events to replay', flush=True)
    out, evtime = [], {}
    for payloads in reco._stream_windows(cfg, pos_maps, seeds, wanted, 3, verbose=False, dead=dead, hits=hits,
                                         hot=cal.hot, evtime=evtime):
        for eid, wins, sds, _nh, _sp, ftst in payloads:
            ftst = ftst if isinstance(ftst, dict) else {}
            tag = evtime.get(int(eid), (-1, ''))[1]
            for p in ('x', 'y'):
                if eid not in want_p[p] or not wins.get(p):
                    continue
                tp = reco.t0_prior_for(cal, p, ftst.get(p))
                try:
                    best, _r = reco.fit_plane_candidates(wins[p], p, cal, seeds=sds.get(p), return_all=True,
                                                         t0_prior=tp)
                except Exception:
                    best = None
                if best is None:
                    continue
                P = wins[p][best._win]
                Wg, nz, ps, st = wm.prep_plane(P, p)
                M = wm.model_waveforms(p, ps, best.p0, best.w, best.t0, best._q, cal.hyper)
                ch = np.asarray(P['ch'], int)
                out.append(dict(event_id=int(eid), acq_tag=tag, view=p, W=Wg.astype(np.float32), M=M.astype(np.float32),
                                noise=np.asarray(nz, np.float32), censor=np.asarray(st, bool),
                                gain=np.asarray(wm.GAIN[p][ch], np.float32), pos=np.asarray(ps, np.float32), ch=ch,
                                ftst=ftst.get(p), t0_prior=tp, p0=best.p0, w=best.w, t0=best.t0, chi2=best.chi2,
                                dof=best.dof, q=np.asarray(best._q, np.float32), w_table=tw.get((int(eid), tag, p))))
        print(f'[boot]   {len(out)} planes', flush=True)
    meta = dict(key=a.key, bundle=bundle, v_drift=float(cal.v_drift), kw=dict(cal.kw), w0=dict(cal.w0),
                nsamp=int(wm.NSAMP), n=len(out))
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(dict(meta=meta, planes=out), f)
        tmp = f.name
    if a.out.startswith('root://'):
        subprocess.run(['xrdcp', '-f', tmp, a.out], check=True)
        os.remove(tmp)
    else:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        os.replace(tmp, a.out)
    ok = [r for r in out if r['w_table'] is not None and np.isfinite(r['w_table'])]
    dw = np.array([abs(r['w'] - r['w_table']) for r in ok])
    print(f'[boot] wrote {a.out}: {len(out)} planes; replay vs table |dw| < 1e-6 on '
          f'{np.mean(dw < 1e-6) * 100 if len(dw) else float("nan"):.1f} % of {len(ok)}', flush=True)


if __name__ == '__main__':
    main()
