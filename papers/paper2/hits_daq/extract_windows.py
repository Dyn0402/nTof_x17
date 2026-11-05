#!/usr/bin/env python3
"""extract_windows.py <key> --ids IDS.json --out OUT.pkl [--part i/n] -- Arm D input of the hits-DAQ comparison.

Runs inside rerun/stage_job.py --no-sync (READ-ONLY on Analysis_pass1), like rerun/paper1_floor/boot_extract.py, whose
replay it reuses: for the wanted events of one row the production reconstruction is replayed exactly as wft.reco does
it (same hits, seeds, windows, fit_plane_candidates with the bundle's t0 prior), and for every plane with a fit it keeps
what a hit-DAQ emulator and every hits tier need:

  W        the window as the pass sees it: pedestal + 64-channel common mode subtracted, NOT gain corrected (ADC)
  raw      the undecoded ADC samples of the same strips (uint16): raw - ped is the window an analog front end sees
           (no common-mode subtraction); ped per strip is stored too
  noise, ch, pos, gain, censor (the fit's censor mask)
  fit      p0, w, t0, q (depth profile), chi2, dof, and the table's w for the same event (closure)
  M        the pass model on the window (ADC, gain applied back: comparable with W)
  hits     the production combined_hits of the window's channels for this event (time, time_of_max, amplitude, ToT,
           significance, sample, max_sample, flags): what V0 must reproduce

--part i/n keeps the events with event_id % n == i (the row's events split over n jobs).
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
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

HIT_COLS = ['eventId', 'feu', 'channel', 'amplitude', 'time', 'time_of_max', 'sample', 'max_sample',
            'time_over_threshold', 'integral', 'significance', 'saturated', 'trunc_left', 'trunc_right']


def load_full_hits(cfg, wanted):
    import uproot
    import pandas as pd
    files = [f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f]
    parts = []
    for f in files:
        t = uproot.open(f'{cfg.combined_hits_dir}{f}:hits')
        cols = [c for c in HIT_COLS if c in t.keys()]
        d = t.arrays(cols, library='pd')
        d = d[d['feu'].isin(cfg.MX17_FEUS) & d['eventId'].isin(wanted)]
        d['file_tag'] = f
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--part', default='0/1')
    a = ap.parse_args()
    ip, npart = (int(x) for x in a.part.split('/'))
    import pandas as pd
    import qa_config
    from wft import io as wio, model as wm, reco, seed as wseed
    wio.CARRY_RAILS = True                      # windows carry rail_hi = W + 4095 - raw  ->  raw is recoverable
    cfg = qa_config.RUNS[a.key]
    wdir = os.path.join(cfg.OUT_BASE, 'wft')
    bundle = os.path.join(wdir, 'calib_bundle')
    reco._worker_init(bundle)
    cal = reco._CAL
    ids = json.load(open(a.ids))
    want_p = {p: set(int(e) for e in ids[p] if int(e) % npart == ip) for p in ('x', 'y')}
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
    full = load_full_hits(cfg, wanted)
    feu_of = {'x': cfg.MX17_FEU_X, 'y': cfg.MX17_FEU_Y}
    # pedestals per file tag and plane (the reader's own: median of the file's first events)
    ped_cache = {}

    def peds(tag):
        if tag not in ped_cache:
            d = {}
            for p in ('x', 'y'):
                f = [x for x in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu_of[p]) if wio.file_tag(x) == tag]
                d[p] = wio.FeuReader(f[0]).ped.astype(np.float32) if f else None
            ped_cache[tag] = d
        return ped_cache[tag]

    print(f'[xw] {a.key} part {a.part}: {len(wanted)} events to replay', flush=True)
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
                ch = np.asarray(P['ch'], int)
                g = np.asarray(wm.GAIN[p][ch], np.float32)
                M = wm.model_waveforms(p, ps, best.p0, best.w, best.t0, best._q, cal.hyper) * g[:, None]
                W = np.asarray(P['W'], np.float32)
                rh = P.get('rail_hi') if isinstance(P, dict) else getattr(P, 'rail_hi', None)
                raw = None if rh is None else np.rint(W + wio.ADC_MAX - np.asarray(rh, np.float32)).astype(np.uint16)
                pd_ = peds(tag).get(p)
                h = None
                if full is not None:
                    hh = full[(full.eventId == eid) & (full.feu == feu_of[p]) & full.channel.isin(ch)]
                    if tag:
                        ht = hh[hh.file_tag.str.contains(tag, regex=False)]
                        hh = ht if len(ht) or not len(hh) else hh.iloc[:0]   # eventId repeats across acquisitions
                    h = {c: hh[c].to_numpy() for c in hh.columns if c not in ('eventId', 'feu', 'file_tag')}
                out.append(dict(event_id=int(eid), acq_tag=tag, view=p, W=W, raw=raw,
                                ped=None if pd_ is None else pd_[ch], M=M.astype(np.float32),
                                noise=np.asarray(P['noise'], np.float32), censor=np.asarray(st, bool), gain=g,
                                pos=np.asarray(ps, np.float32), ch=ch, ftst=ftst.get(p), t0_prior=tp,
                                p0=best.p0, w=best.w, t0=best.t0, chi2=best.chi2, dof=best.dof,
                                q=np.asarray(best._q, np.float32), w_table=tw.get((int(eid), tag, p)), hits=h))
        print(f'[xw]   {len(out)} planes', flush=True)
    meta = dict(key=a.key, part=a.part, bundle=bundle, v_drift=float(cal.v_drift), kw=dict(cal.kw), w0=dict(cal.w0),
                hyper={k: v for k, v in cal.hyper.items() if isinstance(v, (int, float, str))},
                nsamp=int(wm.NSAMP), sns=float(wm.SNS), n=len(out))
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
    nh = sum(1 for r in out if r['hits'] is not None and len(r['hits'].get('time', [])))
    nr = sum(1 for r in out if r['raw'] is not None)
    print(f'[xw] wrote {a.out}: {len(out)} planes ({nh} with production hits, {nr} with raw); replay vs table '
          f'|dw| < 1e-6 on {np.mean(dw < 1e-6) * 100 if len(dw) else float("nan"):.1f} % of {len(ok)}', flush=True)


if __name__ == '__main__':
    main()
