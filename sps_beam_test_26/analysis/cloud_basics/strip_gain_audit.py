#!/usr/bin/env python3
"""strip_gain_audit.py -- dead/weak strips and local gain structure under the head-on width stacks.

The X-width stacks (rc_diffusion.stack: 9 strips around the event's charge maximum) and the toy (xw_sim)
assume every strip in the window reads the same gain and that the avalanche gain is the same over the
+-4 strips.  det4 is 62 % dead in 2-38 mm stripes along X (det4_sps_assessment), det6/det7 have
cell-efficiency spreads 3-4x binomial; none of the cloud_basics fits masks anything.  Here:

  1. Gain map per chamber, at strip scale across X: median of ln Q, Q = sqrt(x_q_sum y_q_sum) / path,
     wft events (x_ok & y_ok, no spark) at (x_p0, y_p0), anisotropic Gaussian kernel (SX across the X
     strips, SY along them).  Survivor-biased where nothing reconstructs; used here only to locate the
     gradients.
  2. Strip health from the head-on cache itself: for each channel, the median ratio of its peak to the
     cluster's peak strip when it sits at offset +-1 (rel1), and its noise against the median.  dead =
     rel1 < DEAD_REL x the chamber median; weak = < WEAK_REL x.  Also per view.
  3. Per head-on event: local gain, gain change across the window, |d ln g / dx| x 3 mm (from the map
     at the cluster centre), and whether a dead/weak channel sits within +-4 of the centre.

    strip_gain_audit.py [--keys g_det4,...]  -> results/strip_gain_audit.json  (+ event flags pickled
                                                  next to the caches: gainflags_<key>.pkl)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
import x_width as xw                                          # noqa: E402

KEYS = ['sat_det3__long_run_resist_490V_drift_1000V', 'sat_det3', 'o22_long_det2', 'g_det4',
        'g_det6_long', 'g_det6_longer', 'g_det7_long', 'g_det7_longer']
# the wft events.parquet a cache's gain map comes from (the det3 long-run sub-run shares sat_det3's)
MAP_KEY = {'sat_det3__long_run_resist_490V_drift_1000V': 'sat_det3'}
SX, SY, STEP = 2.0, 15.0, 1.0
DEAD_REL, WEAK_REL = 0.25, 0.6
HALF = 4


def gain_map(key):
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(MAP_KEY.get(key, key))
    d = pd.read_parquet(os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet'))
    d = d[d.x_ok & d.y_ok & ~d.spark.astype(bool)]
    qx, qy = d.x_q_sum.to_numpy(float), d.y_q_sum.to_numpy(float)
    tx = np.clip(d.x_tan_theta.fillna(0).to_numpy(float), -1, 1)
    ty = np.clip(d.y_tan_theta.fillna(0).to_numpy(float), -1, 1)
    ok = np.isfinite(qx) & np.isfinite(qy) & (qx > 0) & (qy > 0) & (qx < 1e7) & (qy < 1e7)
    x, y = d.x_p0.to_numpy(float)[ok], d.y_p0.to_numpy(float)[ok]
    lq = np.log(np.sqrt(qx[ok] * qy[ok]) / np.sqrt(1 + tx[ok] ** 2 + ty[ok] ** 2))
    lq -= np.median(lq)
    gx = np.arange(np.floor(x.min()), np.ceil(x.max()) + STEP, STEP)
    gy = np.arange(np.floor(y.min()), np.ceil(y.max()) + 5.0, 5.0)
    wx = np.exp(-0.5 * ((gx[:, None] - x[None, :]) / SX) ** 2)        # (nx, n)
    G = np.full((len(gx), len(gy)), np.nan)
    N = np.zeros_like(G)
    for j, yy in enumerate(gy):
        wy = np.exp(-0.5 * ((y - yy) / SY) ** 2)
        w = wx * wy[None, :]
        s = w.sum(1)
        N[:, j] = s
        G[:, j] = np.where(s > 4.0, (w * lq[None, :]).sum(1) / np.maximum(s, 1e-9), np.nan)
    # 1-D profile across X (all y), for chambers with stripe structure
    w = wx
    s = w.sum(1)
    prof = np.where(s > 8.0, (w * lq[None, :]).sum(1) / np.maximum(s, 1e-9), np.nan)
    return dict(gx=gx, gy=gy, G=G, N=N, prof=prof, n=int(ok.sum()), rms_lnq=float(np.std(lq)))


def lookup(m, x, y):
    i = np.clip(np.round((x - m['gx'][0]) / STEP).astype(int), 0, len(m['gx']) - 1)
    j = np.clip(np.round((y - m['gy'][0]) / 5.0).astype(int), 0, len(m['gy']) - 1)
    return m['G'][i, j]


def health(ev, plane):
    rel, nz, fire = {}, {}, {}
    for e in ev.values():
        if plane not in e:
            continue
        W = np.asarray(e[plane]['W'], float)
        ch = e[plane]['ch']
        pk = W.max(1)
        c = int(np.argmax(W.sum(1)))
        # firing rate at +-1 from the charge maximum (any amplitude, saturated events included): a dead
        # channel next to a real cluster essentially never clears 5 sigma
        if pk[c] >= 200:
            for o in (-1, 1):
                if 0 <= c + o < len(W):
                    fire.setdefault(int(ch[c + o]), []).append(pk[c + o] > 5.0 * max(e[plane]['noise'][c + o], 3.0))
        if pk[c] < 200 or pk[c] >= 3500:
            continue
        for o in (-1, 1):
            if 0 <= c + o < len(W):
                rel.setdefault(int(ch[c + o]), []).append(pk[c + o] / pk[c])
        for k, n in zip(ch, e[plane]['noise']):
            nz.setdefault(int(k), []).append(float(n))
    med = {k: float(np.median(v)) for k, v in rel.items() if len(v) >= 8}
    ref = np.median(list(med.values()))
    noise = {k: float(np.median(v)) for k, v in nz.items()}
    nref = np.median(list(noise.values()))
    frate = {k: float(np.mean(v)) for k, v in fire.items() if len(v) >= 8}
    fref = np.median(list(frate.values()))
    dead = sorted(set(k for k, v in med.items() if v < DEAD_REL * ref) |
                  set(k for k, v in frate.items() if v < DEAD_REL * fref))
    weak = sorted(k for k, v in med.items() if DEAD_REL * ref <= v < WEAK_REL * ref)
    hot = sorted(k for k, v in noise.items() if v > 2.0 * nref)
    return dict(rel1_median=float(ref), n_judged=len(med), fire1_median=float(fref), n_fire_judged=len(frate), dead=dead, weak=weak, hot=hot,
                rel1={str(k): v for k, v in med.items()}, noise_median=float(nref),
                fire1={str(k): v for k, v in frate.items()})


def holes(ev, plane, nmin=8):
    """Dead-strip test with inclined tracks (gap_extent caches): a channel whose two spatial neighbours both fire
    (above 5 sigma) inside an event's track span should read about as much as they do; a dead one is a hole.
    Returns the per-channel median peak / neighbours' mean and the channels below DEAD_REL x the median.  Covers far more channels than the
    head-on +-1 test, saturated events included."""
    inside = {}
    for e in ev.values():
        if plane not in e:
            continue
        W = np.asarray(e[plane]['W'], float)
        pk = W.max(1)
        fire = pk > 5.0 * np.maximum(np.asarray(e[plane]['noise'], float), 3.0)
        idx = np.flatnonzero(fire)
        if len(idx) < 4:
            continue
        # judged only where both spatial neighbours fire: a live strip between two firing ones fires too
        # whatever the local gain; a dead one does not
        for j in range(max(idx[0], 1), min(idx[-1], len(W) - 2) + 1):
            if fire[j - 1] and fire[j + 1] and max(pk[j - 1], pk[j + 1]) < 3500:
                # peak over the neighbours' mean: a noisy channel has a higher 5-sigma threshold and would
                # look like a hole in a firing test; the ratio is noise-blind
                inside.setdefault(int(e[plane]['ch'][j]), []).append(pk[j] / (0.5 * (pk[j - 1] + pk[j + 1])))
    rate = {k: float(np.median(v)) for k, v in inside.items() if len(v) >= nmin}
    ref = float(np.median(list(rate.values()))) if rate else float('nan')
    dead = sorted(k for k, v in rate.items() if v < DEAD_REL * ref)
    return dict(rate={str(k): v for k, v in rate.items()}, median=ref, n_judged=len(rate), dead=dead)


INCL = os.path.expanduser('~/x17/cosmic_bench/cloud_basics/gap')
INCL_KEY = {'sat_det3__long_run_resist_490V_drift_1000V': 'sat_det3'}


def bad_set(entry, plane):
    """Every flagged channel of one audit entry: head-on dead + weak, and inclined-track holes."""
    h = entry['health'][plane]
    out = set(h['dead']) | set(h['weak'])
    if 'holes' in entry:
        out |= set(entry['holes'][plane]['dead'])
    return out


def flags(ev, m, H):
    bad = {p: set(H[p]['dead']) | set(H[p]['weak']) for p in ('x', 'y')}
    out = {}
    for k, e in ev.items():
        f = {}
        for p in ('x', 'y'):
            if p not in e:
                continue
            W = np.asarray(e[p]['W'], float)
            c = int(np.argmax(W.sum(1)))
            ch = e[p]['ch']
            win = set(int(q) for q in ch[max(c - HALF, 0):c + HALF + 1])
            f[f'bad_{p}'] = len(win & bad[p])
            f[f'pos_{p}'] = float(e[p]['pos'][c])
        if 'pos_x' in f and 'pos_y' in f:
            x, y = f['pos_x'], f['pos_y']
            g0 = lookup(m, np.array([x]), np.array([y]))[0]
            gl = lookup(m, np.array([x - 3.0]), np.array([y]))[0]
            gr = lookup(m, np.array([x + 3.0]), np.array([y]))[0]
            f.update(lng=float(g0), dlng_x=float((gr - gl) / 2.0))        # change in ln g per 3 mm
        out[k] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    a = ap.parse_args()
    path = f'{HERE}/results/strip_gain_audit.json'
    res = json.load(open(path)) if os.path.exists(path) else {}
    for key in a.keys.split(','):
        ev = pickle.load(open(f'{xw.CACHES}/headon_{key}.pkl', 'rb'))
        m = gain_map(key)
        H = {p: health(ev, p) for p in ('x', 'y')}
        F = flags(ev, m, H)
        pickle.dump(F, open(f'{xw.CACHES}/gainflags_{key}.pkl', 'wb'))
        prof = m['prof'][np.isfinite(m['prof'])]
        d = np.array([f['dlng_x'] for f in F.values() if np.isfinite(f.get('dlng_x', np.nan))])
        bx = np.array([f.get('bad_x', 0) for f in F.values()])
        hx = [e for k, e in ev.items() if abs(e['tan_x']) < xw.TAN]
        res[key] = dict(det=xw.det_of(key), n_map=m['n'], rms_lnq_event=m['rms_lnq'],
                        xprofile=dict(gx=m['gx'].tolist(), lng=[None if not np.isfinite(v) else float(v) for v in m['prof']]),
                        xprofile_p90_p10=float(np.exp(np.percentile(prof, 90) - np.percentile(prof, 10))),
                        dlng3mm_abs_quantiles=[float(q) for q in np.quantile(np.abs(d), [0.25, 0.5, 0.75, 0.9])],
                        frac_bad_x_in_window=float(np.mean(bx > 0)),
                        n_headon_x=len(hx),
                        health={p: {k: v for k, v in H[p].items() if k not in ('rel1', 'fire1')} for p in ('x', 'y')},
                        rel1={p: H[p]['rel1'] for p in ('x', 'y')}, fire1={p: H[p]['fire1'] for p in ('x', 'y')})
        h = res[key]['health']
        print(f'{key:44s} map n {m["n"]:5d}  x-profile p90/p10 {res[key]["xprofile_p90_p10"]:5.2f}  '
              f'|dlng/3mm| q50/q90 {res[key]["dlng3mm_abs_quantiles"][1]:.2f}/{res[key]["dlng3mm_abs_quantiles"][3]:.2f}  '
              f'fire1 {h["x"]["fire1_median"]:.2f} (n {h["x"]["n_fire_judged"]})  X dead {h["x"]["dead"]} weak {len(h["x"]["weak"])} hot {len(h["x"]["hot"])} | Y dead {h["y"]["dead"]} weak '
              f'{len(h["y"]["weak"])} hot {len(h["y"]["hot"])}  | events with a bad X strip in +-4: {res[key]["frac_bad_x_in_window"]:.2f}',
              flush=True)
        ip = f'{INCL}/incl_{INCL_KEY.get(key, key)}.pkl'
        if os.path.exists(ip):
            iev = pickle.load(open(ip, 'rb'))
            res[key]['holes'] = {p: holes(iev, p) for p in ('x', 'y')}
            hx = res[key]['holes']['x']
            print(f'{"":44s} inclined-track holes: X judged {hx["n_judged"]}, dead {hx["dead"]}', flush=True)
        json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
