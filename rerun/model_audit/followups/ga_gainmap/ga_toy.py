#!/usr/bin/env python3
"""ga_toy.py -- GA task 2: what a gain non-uniformity does to the production wft fit, by spatial scale.

Same generator and fitter as FD's realism ladder (rerun/model_audit/followups/fd_floor_realism/ladder.py: ws_floor's
electron-by-electron X-view toy, fitted by ``wft.reco.fit_plane`` with the chamber's promoted bundle), ideal events
plus ONE gain pattern, X view.  The gain acts either per strip after the footprint (readout / coupling gain: ``strip``)
or per electron at its landing point before the resistive spread (amplification gain: ``amp``).  Arms:

  none            no gain pattern (FD rung L1)
  white<s>        iid ln g ~ N(0, s) per strip (FD rung L4's model: det4 s = 0.31, det3 0.06)
  meas            the chamber's MEASURED charge profile across X (ga_chargemap: 1 mm median ln q, 1.5 mm smoothing),
                  per electron, unknown to the fit                                     (det4 = the gain stripes)
  meas_strip      same pattern applied per strip
  meas_large      only its 8 mm-smoothed part (large scale), per electron
  meas_known      meas, and the fit's bundle carries the same pattern per strip (what a measured map would give)
  infit_known     meas per electron, the fit carries the IN-FIT map measured on data (ga_gainfit, half A) -- needs
                  results/gainfit_<det>.npz
Angles 0 (head-on: only the diffusion-growth term a*Dp^2/v can act) and 3/9/16/25 deg.

    ga_toy.py run --det det4 [--arms none,meas,...] [--n 150] [--angles 0,3,9,16,25] [--jobs 8]
          -> results/toy_<det>.json (+ per-arm rows .npy)
    ga_toy.py job ... --out DIR   (condor: one det x arm x angle)
    ga_toy.py collect --inp DIR
"""
import argparse, json, os, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'ws_floor'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism'))
sys.path.insert(0, REPO)
import floor_toy as ft  # noqa: E402

AUDIT = os.environ.get('FD_AUDIT') or os.path.expanduser('~/x17/cosmic_bench/model_audit')
ft.AUDIT = AUDIT
os.environ['FD_AUDIT'] = AUDIT
import ladder as LD  # noqa: E402

OUT = os.path.join(HERE, 'results')
NBLK = 64
PITCH = ft.PITCH
_CAL = None; _G = None


def measured_lng(det, part='S'):
    """ln g at the 512 toy strip positions (strip n at n * pitch) from the charge profile; outside the measured
    range (or in bins without events) the chamber median (0)."""
    R = json.load(open(os.path.join(HERE, 'results', 'chargemap.json')))['res'][det]['x']
    c = np.array(R['centers']); v = np.array(R[part], dtype=float)
    ok = np.isfinite(v)
    x = np.arange(512) * PITCH
    lg = np.interp(x, c[ok], v[ok], left=0.0, right=0.0)
    return lg - np.median(lg)


def infit_lng(det, half='A'):
    z = np.load(os.path.join(HERE, 'results', f'gainfit_{det}.npz'))
    g = z[f'map_{half}_x']; pos = z['pos_x']
    ok = np.isfinite(pos) & (z[f'info_{half}_x'] > 50)
    x = np.arange(512) * PITCH
    return np.interp(x, pos[ok], np.log(np.clip(g[ok], 1e-3, None)))


def arm_state(det, arm):
    """(lng_true [512], mode 'amp'|'strip', lng_known_by_fit [512] or None)"""
    if arm == 'none':
        return np.zeros(512), 'strip', None
    if arm.startswith('white'):
        s = float(arm[5:]) / 100
        rs = np.random.default_rng(777)
        lg = rs.normal(0, s, 512)
        return lg - lg.mean(), 'strip', None
    if arm == 'meas':
        return measured_lng(det), 'amp', None
    if arm == 'meas_strip':
        return measured_lng(det), 'strip', None
    if arm == 'meas_large':
        return measured_lng(det, 'Lg'), 'amp', None
    if arm == 'meas_known':
        lg = measured_lng(det)
        return lg, 'amp', lg
    if arm == 'infit_known':
        return measured_lng(det), 'amp', infit_lng(det)
    raise ValueError(arm)


def _init(det, arm):
    global _CAL, _G
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(os.path.join(AUDIT, det, 'calib_bundle'))
    lg, mode, known = arm_state(det, arm)
    if known is not None:
        _CAL.gain['x'] = np.exp(known)
    wm.use_calibration(_CAL)
    _G = (lg, mode)


def gen_event(det, tan, adc, tmpl, rng):
    """ws_floor/ladder ideal X event (ladder.gen_event with no flags) + the gain pattern, per strip or per electron."""
    P = ft.params(det, 'full')
    g_t, h_t = tmpl
    z, x = ft.electrons(tan, P, rng)
    xl, ul = ft.drift(z, x, P, rng)
    gain = rng.gamma(1 + ft.THETA, 1.0 / (1 + ft.THETA), len(z))
    t0 = 300.0 + rng.uniform(0, ft.DT_SAMP)
    xmax = P['G'] * tan
    span = int(np.ceil(abs(xmax) / PITCH)) + 12
    blk = int(rng.integers(0, 8))
    s_lo = int(rng.integers(0, max(NBLK - span, 1)))
    x_off = (s_lo + 6 + (0 if xmax >= 0 else span - 12)) * PITCH + rng.uniform(-PITCH / 2, PITCH / 2)
    xl = xl + x_off
    pos = np.arange(NBLK) * PITCH
    lg, mode = _G
    gstrip = np.arange(NBLK) + NBLK * blk
    if mode == 'amp':      # amplification gain at the landing point (chamber position = (blk*64) * pitch + xl)
        gain = gain * np.exp(np.interp(xl + blk * NBLK * PITCH, np.arange(512) * PITCH, lg))
    F = ft.footprint(pos, xl, P['s0'])
    A = F * (gain * adc)[None, :]
    ts = np.arange(ft.NSAMP) * ft.DT_SAMP
    H = np.interp(ts[None, :] - (t0 + ul)[:, None], g_t, h_t, left=0, right=0)
    W = A @ H
    if mode == 'strip':
        W *= np.exp(lg[gstrip])[:, None]
    noise = np.full(NBLK, P['noise'])
    W = W + rng.normal(0, 1, W.shape) * noise[:, None]
    W = np.minimum(W, ft.SAT)
    over = np.flatnonzero(W.max(axis=1) > 5 * noise)
    if not len(over):
        return None
    lo, hi = max(over.min() - 3, 0), min(over.max() + 3, NBLK - 1)
    sl = slice(lo, hi + 1)
    win = dict(W=W[sl], pos=pos[sl], noise=noise[sl], ch=gstrip[sl])
    xmid = blk * NBLK * PITCH + x_off + 0.5 * P['G'] * tan
    return win, dict(tan=tan, t0=t0, xmid=xmid, p0=x_off)


def one(args):
    det, arm, ang, seed, n = args
    if _CAL is None:
        _init(det, arm)
    from wft import reco
    rng = np.random.default_rng(seed)
    tmpl = ft.load_template(det)
    adc = ft.adc_scale(ft.params(det, 'full'), np.random.default_rng(12345))
    tan = np.tan(np.radians(ang))
    v_um = float(_CAL.v_drift)
    rows = []
    for _ in range(n):
        ev = gen_event(det, tan, adc, tmpl, rng)
        if ev is None:
            rows.append((ang, np.nan, np.nan, 0, np.nan, 0, np.nan)); continue
        win, tr = ev
        tp = reco.t0_prior_for(_CAL, 'x', 0)
        tp = (tr['t0'], tp[1]) if tp is not None else None
        f = reco.fit_plane(win, 'x', _CAL, t0_prior=tp)
        if f is None:
            rows.append((ang, np.nan, np.nan, 0, tr['xmid'], 1, np.nan)); continue
        tfit = f.w * 1e3 / v_um
        rows.append((ang, np.degrees(np.arctan(tfit)) - ang, f.chi2 / max(f.dof, 1), f.dof, tr['xmid'], 1, f.p0 - tr['p0']))
    return rows


def p0_stats(r):
    if r.shape[1] < 7:
        return {}
    dp = r[:, 6][np.isfinite(r[:, 6]) & (np.abs(r[:, 1]) <= 5)]
    dp = dp[np.abs(dp - np.median(dp)) < 2] if len(dp) else dp
    if len(dp) < 6:
        return {}
    return dict(p0_bias=float(np.median(dp)), p0_s68=float(np.percentile(np.abs(dp - np.median(dp)), 68)),
                p0_rms=float(np.std(dp)))


def summarise(rows):
    rows = np.array(rows, float)
    out = {}
    for ang in sorted(set(rows[:, 0])):
        r = rows[rows[:, 0] == ang]; d = r[:, 1]
        out[str(int(ang))] = dict(n=len(r), n_detected=int(np.sum(r[:, 5] > 0)), s68=LD.s68(d), s68_err=LD.boot_err(d),
                                  bias=float(np.nanmedian(d)), rms_c=float(np.nanstd(np.clip(d[np.isfinite(d)], -5, 5))),
                                  far5=float(np.mean(~(np.abs(d) <= 5))), chi2dof=float(np.nanmedian(r[:, 2])),
                                  **p0_stats(r))
    d = rows[:, 1]
    out['all'] = dict(n=len(rows), s68=LD.s68(d), s68_err=LD.boot_err(d), chi2dof=float(np.nanmedian(rows[:, 2])))
    return out


def jobs_for(det, arm, angles, n, seed_offset=0):
    return [(det, arm, ang, 1000 * int(ang) + c + 7 + 100000 * seed_offset, 10) for ang in angles for c in range(n // 10)]


def run(a):
    os.makedirs(OUT, exist_ok=True)
    angles = [float(x) for x in a.angles.split(',')]
    path = os.path.join(OUT, f'toy_{a.det}.json')
    res = json.load(open(path)) if os.path.exists(path) else dict(det=a.det, arms={})
    for arm in a.arms.split(','):
        t = time.time()
        with Pool(a.jobs, initializer=_init, initargs=(a.det, arm)) as pool:
            rows = [r for part in pool.map(one, jobs_for(a.det, arm, angles, a.n), chunksize=1) for r in part]
        np.save(path.replace('.json', f'_{arm}.npy'), np.array(rows, float))
        res['arms'][arm] = summarise(rows)
        json.dump(res, open(path, 'w'), indent=1)
        print(a.det, f'{arm:12s}', ' '.join(f"{k}: {v['s68']:.2f}/{v['bias']:+.2f}" for k, v in res['arms'][arm].items()
                                            if k != 'all'), f'{time.time() - t:.0f} s', flush=True)


def job(a):
    os.makedirs(a.out, exist_ok=True)
    for arm in a.arms.split(','):
        for ang in [float(x) for x in a.angles.split(',')]:
            _init(a.det, arm)
            rows = [r for j in jobs_for(a.det, arm, [ang], a.n, a.seed_offset) for r in one(j)]
            sfx = f'_s{a.seed_offset}' if a.seed_offset else ''
            np.save(os.path.join(a.out, f'{a.det}_{arm}_a{int(ang)}{sfx}.npy'), np.array(rows, float))
            print(a.det, arm, ang, 'done', flush=True)


def collect(a):
    import glob
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        fs = sorted(glob.glob(os.path.join(a.inp, f'{det}_*_a*.npy')))
        if not fs:
            continue
        res = dict(det=det, arms={})
        import re
        arms = sorted(set(re.sub(r'_a\d+(_s\d+)?\.npy$', '', os.path.basename(f)[len(det) + 1:]) for f in fs))
        for arm in arms:
            rows = np.concatenate([np.load(f) for f in fs if re.fullmatch(rf'{det}_{arm}_a\d+(_s\d+)?\.npy', os.path.basename(f))])
            np.save(os.path.join(OUT, f'toy_{det}_{arm}.npy'), rows)
            res['arms'][arm] = summarise(rows)
            print(det, f'{arm:12s}', ' '.join(f"{k}: {v['s68']:.2f}/{v['bias']:+.2f}" for k, v in res['arms'][arm].items()
                                                if k != 'all'))
        json.dump(res, open(os.path.join(OUT, f'toy_{det}.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'job', 'collect'))
    ap.add_argument('--det', default='det4'); ap.add_argument('--arms', default='none,white31,meas,meas_strip,meas_large,meas_known')
    ap.add_argument('--n', type=int, default=150); ap.add_argument('--angles', default='0,3,9,16,25')
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--out'); ap.add_argument('--inp')
    ap.add_argument('--seed-offset', type=int, default=0)
    a = ap.parse_args()
    dict(run=run, job=job, collect=collect)[a.cmd](a)


if __name__ == '__main__':
    main()
