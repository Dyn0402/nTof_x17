#!/usr/bin/env python3
"""Portability test: full waveform-first protocol on det2 (g_det2 day run).

Stages (all outputs under <det2 Analysis>/mx17_4/waveform_first/):
  1. corridor waveform cache (long-run alignment chain, FEU 6=X / 8=Y)
  2. per-plane impulse templates
  3. 8-hyper ref-pinned calibration (gains=1, 130 medium-angle events)
  4. free fits on a disjoint 800-event test set (forward_model3 fitter)
  5. benchmark numbers (angle/mesh/census/v-flatness)
"""
import os, sys, glob, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa')
from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles
from common.Mx17StripMap import RunConfig

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)

CFG = get_config('g_det2')
OUT_DIR = CFG.out_dir('waveform_first')
CACHE_PKL = os.path.join(OUT_DIR, 'wfcache.pkl')
VETO, RES_CUT, PAD = 50, 10.0, 5.0
N_PED, SNS = 300, 60.0

# ---------------- stage 1: cache ----------------
def build_cache():
    if os.path.exists(CACHE_PKL):
        return pickle.load(open(CACHE_PKL, 'rb'))['events']
    cache = os.path.join(CFG.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}',
                              'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    events = {}
    for r in results:
        if not (r.has_x and r.has_y):
            continue
        if not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > RES_CUT:
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, best)
        events[int(r.event_id)] = dict(
            eid=int(r.event_id), ref_mesh_x=float(r.ref_mesh_x_mm),
            ref_mesh_y=float(r.ref_mesh_y_mm), tan_x=float(tx), tan_y=float(ty))
    print(f'{len(events)} matched det2 events', flush=True)
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    pos_map = {}
    for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
        p = np.full(512, np.nan)
        for ch in range(512):
            hit = det.map_hit(feu, ch)
            if hit is not None and hit[axis] is not None:
                p[ch] = hit[axis]
        pos_map[feu] = p
    for ev in events.values():
        for plane, feu in (('x', CFG.MX17_FEU_X), ('y', CFG.MX17_FEU_Y)):
            p0 = ev[f'ref_mesh_{plane}']
            tn = ev[f'tan_{plane}']
            a, b = p0 - 3 * abs(tn), p0 + 33 * abs(tn)
            lo, hi = min(a, b) - PAD, max(a, b) + PAD
            pm = pos_map[feu]
            ch = np.where((pm >= lo) & (pm <= hi))[0]
            o = np.argsort(pm[ch])
            ev[plane] = dict(ch=ch[o].astype(np.int16),
                             pos=pm[ch][o].astype(np.float32))
    dec = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
    for feu, plane in ((CFG.MX17_FEU_X, 'x'), (CFG.MX17_FEU_Y, 'y')):
        for fn in sorted(glob.glob(os.path.join(dec, f'*_{feu:02d}.root'))):
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED, library='np')['amplitude']
            lens = np.array([len(a) for a in a0])
            ns = int(np.bincount(lens).argmax() // 512)
            keep = [a for a in a0 if len(a) == ns * 512]
            stack = np.stack([a.reshape(ns, 512) for a in keep]).astype(np.float32)
            ped = np.median(stack, axis=(0, 1))
            s0 = stack - ped[None, None, :]
            cms = np.median(s0.reshape(len(keep), ns, 8, 64), axis=3)
            s0 -= np.repeat(cms, 64, axis=2)
            sig = 1.4826 * np.median(np.abs(s0), axis=(0, 1))
            want = np.where(np.isin(eids_all,
                                    np.fromiter(events.keys(), np.int64)))[0]
            for lo_i in range(0, len(want), 400):
                idx = want[lo_i:lo_i + 400]
                arr = t.arrays(['eventId', 'amplitude'], entry_start=int(idx[0]),
                               entry_stop=int(idx[-1]) + 1, library='np')
                base_i = int(idx[0])
                for i in idx:
                    j = i - base_i
                    eid = int(arr['eventId'][j])
                    if eid not in events:
                        continue
                    ns_ev = len(arr['amplitude'][j]) // 512
                    if ns_ev < ns:
                        continue
                    wfm = arr['amplitude'][j].reshape(ns_ev, 512)[:ns].astype(np.float32) - ped
                    cms2 = np.median(wfm.reshape(ns, 8, 64), axis=2)
                    wfm -= np.repeat(cms2, 64, axis=1)
                    ev = events[eid]
                    ch = ev[plane]['ch']
                    ev[plane]['W'] = wfm[:, ch].T.astype(np.float16)
                    ev[plane]['noise'] = sig[ch]
            print(f'  {os.path.basename(fn)}', flush=True)
    events = {k: v for k, v in events.items()
              if 'W' in v.get('x', {}) and 'W' in v.get('y', {})}
    pickle.dump(dict(events=events), open(CACHE_PKL, 'wb'))
    print(f'{len(events)} det2 events cached', flush=True)
    return events

# ---------------- stage 2: templates ----------------
def build_templates(events):
    outf = os.path.join(OUT_DIR, 'templates_perplane.npz')
    if os.path.exists(outf):
        return
    GRID = np.arange(-360, 1400, 10.0)
    def t50(w):
        ipk = int(np.argmax(w)); a = w[ipk]
        for k in range(1, ipk + 1):
            if w[k] >= 0.5 * a > w[k - 1]:
                return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
        return np.nan
    tm = {}
    for plane in ('x', 'y'):
        acc = []
        for ev in events.values():
            tn = abs(ev[f'tan_{plane}'])
            if tn < 0.20:                      # det2 lower stats: looser
                continue
            W = ev[plane]['W'].astype(np.float32)
            for i in np.argsort(W.max(axis=1))[::-1][:2]:
                w = W[i]; a = w.max(); ipk = int(np.argmax(w))
                if a < 250 or a > 3550 or ipk < 6 or ipk > W.shape[1] - 10:
                    continue
                c = t50(w)
                if np.isfinite(c):
                    tt = (np.arange(len(w)) - c) * SNS
                    acc.append(np.interp(GRID, tt, w / a, left=np.nan, right=np.nan))
        t = np.nanmedian(np.array(acc), axis=0)
        t -= np.nanmedian(t[GRID < -250])
        tm[plane] = np.nan_to_num(t)
        print(f'det2 template {plane}: n={len(acc)}', flush=True)
    np.savez(outf, grid=GRID, tmpl_x=tm['x'], tmpl_y=tm['y'])

# ---------------- stage 3-5 run in __main__ after fm patch ----------------
if __name__ == '__main__':
    events = build_cache()
    build_templates(events)
    import forward_model2 as fm2
    import forward_model3 as fm3
    ns_det = next(iter(events.values()))['x']['W'].shape[1]
    fm2.set_nsamp(ns_det)
    fm3._TT_CACHE.clear()
    print(f'det2 window: {ns_det} samples', flush=True)
    tz = np.load(os.path.join(OUT_DIR, 'templates_perplane.npz'))
    fm2.TGRID = tz['grid']
    fm2.TMPL = {'x': tz['tmpl_x'], 'y': tz['tmpl_y']}
    fm2.GAIN = {'x': np.ones(512), 'y': np.ones(512)}
    fm2._smear_cache.clear()
    fm3._TT_CACHE.clear()

    cand = []
    for eid, ev in events.items():
        t3 = np.hypot(ev['tan_x'], ev['tan_y'])
        if ev['x']['W'].size == 0 or ev['y']['W'].size == 0:
            continue
        if 0.10 < t3 < 0.40 and ev['x']['W'].max() > 250 and ev['y']['W'].max() > 250:
            cand.append(eid)
    rng = np.random.default_rng(2222)
    rng.shuffle(cand)
    train, test = cand[:130], cand[130:130 + 800]
    print(f'det2: {len(cand)} candidates, train {len(train)}, test {len(test)}',
          flush=True)

    def solve_t0(ev, plane, p0l, wline, hyper, warm):
        P = ev[plane]
        W, noise, pos, sat = fm2.prep_plane(P, plane)
        wt0 = warm.get(plane)
        for _ in range(3):
            grid = (np.arange(150.0, 900.0, 30.0) if wt0 is None
                    else np.arange(wt0 - 60.0, wt0 + 61.0, 15.0))
            chis = [fm3.chi2_plane_fast(plane, W, noise, pos, sat, p0l, wline,
                                        float(t0), hyper)[0] for t0 in grid]
            j = int(np.argmin(chis))
            t0b = float(grid[j])
            if j not in (0, len(grid) - 1) or wt0 is None:
                return chis[j], t0b
            wt0 = t0b
        return chis[j], t0b

    def event_chi2(args):
        eid, hyper, v, warm = args
        ev = events[eid]
        tot, t0s = 0.0, {}
        for plane in ('x', 'y'):
            tn = ev[f'tan_{plane}']
            p0l = ev[f'ref_mesh_{plane}']
            chi, t0b = solve_t0(ev, plane, p0l, tn * v * 1e-3, hyper, warm)
            if np.isfinite(chi):
                tot += chi; t0s[plane] = t0b
        return eid, tot, t0s

    warm = {e: {} for e in train}
    pool = ProcessPoolExecutor(max_workers=12)
    def total_chi2(hv):
        c1, c2, kY, tau, sig_s, sp0, Dp, v = hv
        hyper = dict(c1=c1, c2=c2, kY=kY, tau_s=tau, sigma_s=sig_s,
                     sigma_p0=sp0, Dp=Dp)
        c = 0.0
        for eid, tot, t0s in pool.map(event_chi2,
                                      [(e, hyper, v, warm[e]) for e in train],
                                      chunksize=6):
            c += tot; warm[eid] = t0s
        return c

    x0 = np.array([0.30, 0.05, 1.0, 47.0, 87.0, 0.10, 0.010, 36.0])
    scale = np.array([0.06, 0.03, 0.15, 15.0, 20.0, 0.06, 0.005, 2.5])
    c0 = total_chi2(x0)
    print(f'det2 initial chi2 {c0:.4e}', flush=True)
    neval = [0]
    def obj(x):
        x = np.asarray(x)
        if (x[:3] < 0).any() or x[3] < 0 or x[4] < 0 or x[5] < 0.03 or \
                x[6] < 0 or not (15 < x[7] < 60):
            return 2 * c0
        c = total_chi2(x); neval[0] += 1
        if neval[0] % 10 == 0:
            print(f'  eval{neval[0]}', np.round(x, 3), f'{c:.4e}', flush=True)
        return c
    simplex = [x0] + [x0 + np.eye(8)[i] * scale[i] for i in range(8)]
    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(initial_simplex=np.array(simplex),
                                xatol=1e-3, fatol=c0 * 2e-4, maxiter=110))
    NAMES = ['c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp', 'v']
    out = {k: float(v_) for k, v_ in zip(NAMES, res.x)}
    out['chi2'] = float(res.fun)
    json.dump(out, open(os.path.join(OUT_DIR, 'hyper_det2.json'), 'w'), indent=1)
    print('det2 hypers:', np.round(res.x, 3), flush=True)

    H = {k: out[k] for k in NAMES[:-1]}
    V4 = out['v']
    def fit_free(eid):
        ev = events[eid]
        o = dict(eid=eid)
        try:
            for plane in ('x', 'y'):
                tn = ev[f'tan_{plane}']
                p0r = ev[f'ref_mesh_{plane}']
                g = fm2.init_guess(ev[plane], plane, tn, p0r, V4 * 1e-3)
                r = fm3.fit_plane(ev[plane], plane, *g, hyper=H)
                o[plane] = dict(tan_ref=tn, p0_ref=p0r, p0=r['p0'], w=r['w'],
                                t0=r['t0'], chi2=r['chi2'], dof=r['dof'],
                                amax=float(ev[plane]['W'].max()))
        except Exception as ex:
            o['error'] = str(ex)
        return o
    t_ = time.time()
    resf = list(pool.map(fit_free, test, chunksize=4))
    print(f'det2 freefit {len(resf)} in {time.time()-t_:.0f}s', flush=True)
    pickle.dump(resf, open(os.path.join(OUT_DIR, 'freefit_det2.pkl'), 'wb'))

    def rs(a):
        a = a[np.isfinite(a)]
        return 1.4826 * np.median(np.abs(a - np.median(a)))
    for p in ('x', 'y'):
        tan, w, p0, p0r = [], [], [], []
        for r in resf:
            if p in r and 'error' not in r:
                d = r[p]
                tan.append(d['tan_ref']); w.append(d['w'])
                p0.append(d['p0']); p0r.append(d['p0_ref'])
        tan, w, p0, p0r = map(np.array, (tan, w, p0, p0r))
        tanf = w * 1e3 / V4
        dth = np.degrees(np.arctan(tanf)) - np.degrees(np.arctan(tan))
        dp = p0 - p0r
        dev = np.maximum(np.abs(dp), np.abs(dp + (tanf - tan) * 29))
        at = np.abs(tan)
        vimp = [np.nanmedian((w * 1e3 / tan)[(at >= a) & (at < b)])
                for a, b in ((0.08, 0.15), (0.15, 0.25), (0.25, 0.45))]
        print(f'det2 {p}: angle med {np.nanmedian(dth):+.2f} sig {rs(dth):.2f} deg; '
              f'mesh sig {rs(dp):.3f} mm; <1mm {np.mean(dev<1)*100:.0f}%; '
              f'v(angle bins) {np.round(vimp,1)}', flush=True)
    pool.shutdown()
