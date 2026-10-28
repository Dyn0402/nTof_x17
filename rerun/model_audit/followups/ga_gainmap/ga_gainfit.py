#!/usr/bin/env python3
"""ga_gainfit.py -- GA (gain-map follow-up): a per-channel gain map measured INSIDE the production waveform fit, on two
disjoint halves of a calibration cache, and its cross-validated effect on angles and chi2.

Why in-fit: the fit's model already says how much charge each strip should see for the fitted track (footprint,
diffusion, sharing kernel, censoring); the ratio data/model per channel, accumulated over many tracks at random
offsets, is the strip's gain.  The per-event depth profile q is free (NNLS), so only the strip-to-strip part of a
gain variation can show up in the ratio: a gain that is common to the whole cluster is absorbed by q.

Algorithm (rerun/det4/gain_fit.py's update, on wft.reco.fit_plane -- the production fitter, as heldout_arm.py):
  g_c <- g_c * sum_ev sum_t (W'M / n'^2) / sum_ev sum_t (M^2 / n'^2)   over unsaturated, uncensored samples,
  W', n' the gain-corrected data and noise, M the model of the fitted track; channels with sum M^2/n'^2 < MIN_INFO keep
  their gain; normalised to median 1 over updated channels (the scale is free: q absorbs it).

Passes over the cache events [0, N) split by parity into halves A (even eid order index) and B (odd):
  pass 0   every event with unit gain                      -> arm 'base' + first update of map_A (from A) / map_B (B)
  pass i   A with map_A, B with map_B                       -> next update (training refits only)
  eval     A with map_B, B with map_A                       -> arm 'cv'          (no event is fitted with a map it made)
           same with the SMOOTH part of the other map       -> arm 'cv_smooth'   (ln g Gaussian-smoothed, SMOOTH_MM)
           same with the FINE part (full / smooth)          -> arm 'cv_fine'
           A with map_A, B with map_B                       -> arm 'self'        (circular, for scale only)

    ga_gainfit.py --det det4 [--cache ws_t0/caches/det4.pkl] [--bundle det4/calib_bundle] [--iters 2] [--jobs 8]
Output: results/gainfit_<det>.npz (maps per half and iteration, per-arm rows) + results/gainfit_<det>.json (summary).
"""
import argparse, json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit'))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
MIN_INFO = 50.0
SMOOTH_MM = 4.0          # Gaussian sigma of the smooth part [mm] (5 strips); the footprint is +-2 strips
PITCH = 0.78
_EV = None; _MAPS = None


def load_cache(path):
    """the calibration cache; lists (a numpy-free copy for LCG_105's numpy 1.x on condor) become arrays again"""
    E = pickle.load(open(path, 'rb'))
    for e in E.values():
        for p in 'xy':
            if p in e:
                for k in ('ch', 'pos', 'W', 'noise', 'missing'):
                    if k in e[p] and not isinstance(e[p][k], np.ndarray):
                        e[p][k] = np.asarray(e[p][k], dtype=int if k == 'ch' else (bool if k == 'missing' else float))
    return E


def _init(cache, bundle, maps):
    global _EV, _MAPS
    sys.path.insert(0, REPO)
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    _EV = load_cache(cache)
    cal = CalibrationBundle.load(bundle)
    wm.use_calibration(cal); wm.set_depth_bins(24)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01
    wr.PIVOT_START = True
    _MAPS = maps


def one(task):
    eid, mapname, want = task
    from wft import model as wm, reco as wr
    cal = wm.CAL; ev = _EV[eid]
    wm.GAIN = {p: np.asarray(_MAPS[mapname][p], float) for p in 'xy'}
    rows, upd = [], {}
    for vi, plane in enumerate('xy'):
        if plane not in ev:
            continue
        P = ev[plane]
        W0 = np.asarray(P['W'])
        if W0.shape[1] != wm.NSAMP:
            wm.set_nsamp(W0.shape[1])
        try:
            f = wr.fit_plane(P, plane, cal)
        except Exception:
            f = None
        if f is None:
            continue
        rows.append((int(eid), vi, float(ev[f'tan_{plane}']), f.w * 1e3, f.chi2, f.dof, float(W0.max()),
                     int((W0 >= wm.SAT).sum()), f.t0, f.p0, float(ev.get(f'ref_mesh_{plane}', np.nan)), f.q_sum))
        if want:
            W, noise, pos, sat = wm.prep_plane(P, plane)
            M = wm.model_waveforms(plane, pos, f.p0, f.w, f.t0, f._q, cal.hyper)
            ok = ~sat
            iw = 1.0 / noise[:, None] ** 2
            a = (np.where(ok, W * M, 0) * iw).sum(1)
            b = (np.where(ok, M * M, 0) * iw).sum(1)
            upd[plane] = (np.asarray(P['ch'], int), a, b)
    return rows, upd


def run_pass(cache, bundle, maps, tasks, jobs):
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(cache, bundle, maps)) as pool:
        return list(pool.map(one, tasks, chunksize=2))


def update(g, results):
    out = {}
    for p in 'xy':
        A, B = np.zeros(512), np.zeros(512)
        for _, u in results:
            if p in u:
                ch, a, b = u[p]
                np.add.at(A, ch, a); np.add.at(B, ch, b)
        ok = B > MIN_INFO
        r = np.where(ok, np.clip(A / np.where(ok, B, 1), 0.1, 10.0), 1.0)
        gn = g[p] * r
        gn /= np.median(gn[ok]) if ok.any() else 1.0
        out[p] = gn
        out[p + '_info'] = B
    return out


def smooth_part(g, info, pos_of_ch):
    """ln g Gaussian-smoothed in strip POSITION over updated channels (info weights), per plane."""
    lg = np.log(np.clip(g, 1e-3, None))
    ok = info > MIN_INFO
    pos = pos_of_ch
    s = np.ones(512)
    for c in range(512):
        if not np.isfinite(pos[c]):
            continue
        d = pos[ok] - pos[c]
        w = np.exp(-0.5 * (d / SMOOTH_MM) ** 2)
        if w.sum() > 1e-6:
            s[c] = np.exp((w * lg[ok]).sum() / w.sum())
    return s


def positions(cache):
    C = load_cache(cache)
    pos = {p: np.full(512, np.nan) for p in 'xy'}
    for e in C.values():
        for p in 'xy':
            if p in e:
                pos[p][np.asarray(e[p]['ch'], int)] = np.asarray(e[p]['pos'], float)
    # fill channels never seen by linear interpolation in channel number (positions are monotone per connector)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--cache'); ap.add_argument('--bundle')
    ap.add_argument('--iters', type=int, default=2); ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    cache = a.cache or os.path.join(AUDIT, 'ws_t0', 'caches', f'{a.det}.pkl')
    bundle = a.bundle or os.path.join(AUDIT, a.det, 'calib_bundle')
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    out = os.path.join(HERE, 'results', f'gainfit_{a.det}{a.tag}')
    C = pickle.load(open(cache, 'rb'))
    ids = sorted(C)[:a.n]
    del C
    half = {e: ('A' if i % 2 == 0 else 'B') for i, e in enumerate(ids)}
    one_ = {p: np.ones(512) for p in 'xy'}
    store = dict(ids=np.array(ids), half=np.array([half[e] for e in ids]))
    t = time.time()
    # pass 0: unit gain everywhere = the base arm and the first update
    res = run_pass(cache, bundle, {'unit': one_}, [(e, 'unit', True) for e in ids], a.jobs)
    store['rows_base'] = np.array([r for rs, _ in res for r in rs], float)
    maps = {}
    for h in 'AB':
        maps[h] = update(one_, [rr for e, rr in zip(ids, res) if half[e] == h])
    print('pass 0', time.time() - t, flush=True)
    log = []
    for it in range(1, a.iters):
        res = run_pass(cache, bundle, {h: {p: maps[h][p] for p in 'xy'} for h in 'AB'},
                       [(e, half[e], True) for e in ids], a.jobs)
        store[f'rows_self{it}'] = np.array([r for rs, _ in res for r in rs], float)
        for h in 'AB':
            prev = maps[h]
            maps[h] = update({p: prev[p] for p in 'xy'}, [rr for e, rr in zip(ids, res) if half[e] == h])
            log.append(dict(it=it, half=h, **{p: float(np.std(np.log(maps[h][p][maps[h][p + '_info'] > MIN_INFO] /
                                                                      prev[p][maps[h][p + '_info'] > MIN_INFO])))
                                               for p in 'xy'}))
        print('pass', it, time.time() - t, log[-2:], flush=True)
    pos = positions(cache)
    for h in 'AB':
        for p in 'xy':
            store[f'map_{h}_{p}'] = maps[h][p]; store[f'info_{h}_{p}'] = maps[h][p + '_info']
            sm = smooth_part(maps[h][p], maps[h][p + '_info'], pos[p])
            store[f'smooth_{h}_{p}'] = sm
    for p in 'xy':
        store[f'pos_{p}'] = pos[p]
    other = {'A': 'B', 'B': 'A'}
    mm = {}
    for h in 'AB':
        mm['full' + h] = {p: store[f'map_{h}_{p}'] for p in 'xy'}
        mm['smooth' + h] = {p: store[f'smooth_{h}_{p}'] for p in 'xy'}
        mm['fine' + h] = {p: store[f'map_{h}_{p}'] / store[f'smooth_{h}_{p}'] for p in 'xy'}
    tasks = ([(e, 'full' + other[half[e]], False) for e in ids] + [(e, 'smooth' + other[half[e]], False) for e in ids] +
             [(e, 'fine' + other[half[e]], False) for e in ids])
    res = run_pass(cache, bundle, mm, tasks, a.jobs)
    n = len(ids)
    store['rows_cv'] = np.array([r for rs, _ in res[:n] for r in rs], float)
    store['rows_cv_smooth'] = np.array([r for rs, _ in res[n:2 * n] for r in rs], float)
    store['rows_cv_fine'] = np.array([r for rs, _ in res[2 * n:] for r in rs], float)
    if a.iters > 1:
        store['rows_self'] = store[f'rows_self{a.iters - 1}']
    np.savez(out + '.npz', **store)
    json.dump(dict(args=vars(a), cache=cache, bundle=bundle, log=log, seconds=time.time() - t,
                   smooth_mm=SMOOTH_MM, min_info=MIN_INFO), open(out + '.json', 'w'), indent=1)
    print('wrote', out, time.time() - t, flush=True)


if __name__ == '__main__':
    main()
