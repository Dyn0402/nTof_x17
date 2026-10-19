#!/usr/bin/env python3
"""gap_extent.py -- the drift gap from the spatial extent of inclined tracks' charge, with no timing.

Every earlier gap number (mx_june_wft GAP_STUDY / GAP_CONSISTENCY, bench_fit.py) is a time: the end
of the charge-arrival profile, T_end, times a drift speed.  T_end depends on the kernel (+-1 mm across
five legitimate bundles) and on the signal amplitude (+1.5 mm det3, +9 mm det4).

Here no time enters.  For a track with reference slope tan_x through the mesh at x_mesh, the X strip
at position x collects the ionisation from depth z = (x - x_mesh) / tan_x, over a depth interval
pitch / |tan_x|.  Stacking the per-strip charge density against z over many tracks gives a profile
that is flat from the mesh (z = 0) to the cathode (z = gap) and blurred at both ends by the footprint
and diffusion (sigma_x / tan) and the strip interval.  Those blurs are symmetric, so the half-height
points of the two edges are unbiased, and

    gap = z_cathode - z_mesh,

independent of v, the kernel, t0 and the time window.  An offset in x_mesh (alignment) moves both
edges together and cancels.  What does not cancel: the reference slope's scale (a scale error k reads
gap / k) and a tilt between the drift field and the M3 frame, which shifts tan by delta and makes
the + and - tan halves differ (their mean is first-order free of it).

Per-strip charge: the window integral (sum of samples) and, as a check, the peak sample.  Events with
any X sample >= SAT are dropped; dead/weak strips (strip_gain_audit.py) are left out of both the charge
and the exposure.  Each event must keep >= CONTAIN of its charge inside z in (-6, 36) and is normalised
to its total charge; |tan| >= 0.2 (TAN_MAIN).  The edges are the half-height crossings (half()).

    gap_extent.py build <key> [--tan-min 0.12] [--tan-max 0.8]   -> <OUT>/incl_<key>.pkl
    gap_extent.py fit [--keys k1,k2]                             -> results/gap_extent.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headon_cache as HC                                    # noqa: E402

OUT = '/home/dylan/x17/cosmic_bench/cloud_basics/gap'
KEYS = {'sat_det3': 'det3', 'o22_long_det3': 'det3', 'o22_long_det2': 'det2', 'g_det4': 'det4',
        'g_det6_long': 'det6', 'g_det6_longer': 'det6', 'g_det7_long': 'det7', 'g_det7_longer': 'det7'}
PITCH = {'x': 0.78, 'y': 0.78}
SAT = 3500.0
DZ = 0.25
ZG = np.arange(-15.0, 50.0 + DZ, DZ)
NORM_Z = (4.0, 22.0)
CONTAIN = 0.85
TAN_BINS = (0.12, 0.2, 0.3, 0.45, 0.8)
TAN_MAIN = (0.2, 0.8)            # below 0.2 a strip is >= 4 mm of depth and the neighbour tails dominate the edges
FIT_Z = (-8.0, 42.0)


def build(key, tan_min, tan_max, subrun=None, base=None):
    """subrun/base: another sub-run of the key's run (e.g. the det3 drift scan), reference from its own M3
    rays through the key's alignment (headon_cache ref-only mode); cached as incl_<key>__<subrun>."""
    import copy
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(key)
    sel = lambda tx, ty: tan_min <= abs(tx) <= tan_max or tan_min <= abs(ty) <= tan_max
    # z window -8..40 mm along the track plus 5 mm in x
    if subrun:
        acfg, cfg = cfg, copy.copy(cfg)
        cfg.SUB_RUN = subrun
        if base:
            cfg.BASE_PATH = base
        ev = HC.build(cfg, tan_max=None, select=sel, z_lo=-8.0, z_hi=40.0, refonly=(acfg, cfg.m3_tracking_dir))
        key = f'{key}__{subrun}'
    else:
        ev = HC.build(cfg, tan_max=None, select=sel, z_lo=-8.0, z_hi=40.0)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(ev, open(f'{OUT}/incl_{key}.pkl', 'wb'), protocol=4)
    print('wrote', f'{OUT}/incl_{key}.pkl', len(ev))


def bad_channels(key, plane):
    path = f'{HERE}/results/strip_gain_audit.json'
    if not os.path.exists(path):
        return set()
    J = json.load(open(path))
    if key not in J:
        key = key.split('__')[0]                # a sub-run uses its parent run's strip health
    if key not in J:
        return set()
    h = J[key]['health'][plane]
    return set(h['dead']) | set(h['weak'])


def profile(evs, plane, bad, qmode='int', tsign=0, tbin=TAN_MAIN, pitch=0.78):
    S = np.zeros(len(ZG)); E = np.zeros(len(ZG)); n = 0
    for e in evs:
        if plane not in e:
            continue
        tn = e[f'tan_{plane}']
        if not (tbin[0] <= abs(tn) < tbin[1]) or (tsign and np.sign(tn) != tsign):
            continue
        W = np.asarray(e[plane]['W'], float)
        if W.max() >= SAT:
            continue
        pos = np.asarray(e[plane]['pos'], float)
        z = (pos - e[f'ref_mesh_{plane}']) / tn
        dz = pitch / abs(tn)
        if z.min() > -6.0 or z.max() < 38.0:            # window must cover both edges with margin
            continue
        q = W.sum(1) if qmode == 'int' else W.max(1)
        good = np.array([int(c) not in bad for c in e[plane]['ch']])
        dens = q / dz
        lo, hi = z - dz / 2, z + dz / 2
        ov = np.clip(np.minimum(hi[:, None], ZG[None, :] + DZ / 2) - np.maximum(lo[:, None], ZG[None, :] - DZ / 2), 0, None) / DZ
        ov = ov * good[:, None]
        s = (dens[:, None] * ov).sum(0); x = ov.sum(0)
        # the event's charge must follow its reference: CONTAIN of it inside the expected column
        # (rejects wrong references, two-track and delta-ray-dominated events), then normalised to its
        # total, so no event can carry an outsized weight
        tot = q[good].sum()
        inside = q[good & (z > -6.0) & (z < 36.0)].sum()
        if tot <= 0 or inside < CONTAIN * tot:
            continue
        norm = tot / 30.0                       # density / (total / 30 mm): plateau ~1 for a 30 mm column
        S += s / norm; E += x; n += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        P = S / E
    return n, P, E


def model(p, z):
    A, z0, s0, z1, s1, b, c = p
    return A * 0.5 * (erf((z - z0) / (np.sqrt(2) * s0)) - erf((z - z1) / (np.sqrt(2) * s1))) * (1 + b * (z - 15.0)) + c


def fit(P, E):
    m = np.isfinite(P) & (E > 5) & (ZG >= FIT_Z[0]) & (ZG <= FIT_Z[1])
    z, y = ZG[m], P[m]
    w = np.sqrt(E[m])
    p0 = [1.0, 0.0, 1.0, 28.0, 2.0, 0.0, 0.0]
    r = least_squares(lambda p: (model(p, z) - y) * w / w.mean(), p0,
                      bounds=([0.2, -6, 0.05, 15, 0.05, -0.05, -0.3], [3, 6, 8, 40, 10, 0.05, 0.3]))
    J = r.jac
    dof = max(len(z) - len(p0), 1)
    s2 = (r.fun ** 2).sum() / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        err = np.sqrt(np.diag(cov))
        gerr = float(np.sqrt(cov[1, 1] + cov[3, 3] - 2 * cov[1, 3]))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan); gerr = np.nan
    A, z0, s0, z1, s1, b, c = r.x
    return dict(gap=float(z1 - z0), gap_err_fit=gerr, z0=float(z0), z1=float(z1), s0=float(s0), s1=float(s1),
                A=float(A), slope=float(b), c=float(c), chi2dof=float(s2))


def half(P, E, plateau=(6.0, 20.0)):
    """Model-free edges: the half-height crossings of the profile (floor 0, plateau = median over
    `plateau`).  A symmetric blur of any shape, or a mixture of them (the tan range), keeps the
    half-height point of a step at the step."""
    ok = np.isfinite(P) & (E > 5)
    m = ok & (ZG >= plateau[0]) & (ZG <= plateau[1])
    H = 0.5 * np.median(P[m])
    z, y = ZG[ok], P[ok]
    up = [i for i in range(1, len(z)) if y[i - 1] < H <= y[i] and -10 <= z[i] <= 10]
    dn = [i for i in range(1, len(z)) if y[i - 1] >= H > y[i] and 15 <= z[i] <= 48]
    if not up or not dn:
        return dict(gap=np.nan)
    i, j = up[-1], dn[-1]
    z0 = z[i - 1] + (H - y[i - 1]) / (y[i] - y[i - 1]) * (z[i] - z[i - 1])
    z1 = z[j - 1] + (y[j - 1] - H) / (y[j - 1] - y[j]) * (z[j] - z[j - 1])
    return dict(gap=float(z1 - z0), z0=float(z0), z1=float(z1), plateau=float(2 * H))


def boot(evs, plane, bad, nb, est=None, **kw):
    rng = np.random.default_rng(7)
    evs = list(evs)
    g = []
    for _ in range(nb):
        sub = [evs[i] for i in rng.integers(0, len(evs), len(evs))]
        n, P, E = profile(sub, plane, bad, **kw)
        if n > 30:
            g.append((est or fit)(P, E)['gap'])
    return float(np.std(g)) if g else np.nan


def run_fit(keys, nb):
    path = f'{HERE}/results/gap_extent.json'
    out = json.load(open(path)) if os.path.exists(path) else {}
    for key in keys:
        f = f'{OUT}/incl_{key}.pkl'
        if not os.path.exists(f):
            print('no cache', f); continue
        ev = list(pickle.load(open(f, 'rb')).values())
        r = dict(det=KEYS.get(key, key))
        for plane in ('x', 'y'):
            bad = bad_channels(key, plane)
            R = {}
            for qmode in ('int', 'peak'):
                if plane == 'y' and qmode == 'int':
                    continue                            # Y's RC spread drains: its integral is not the charge
                n, P, E = profile(ev, plane, bad, qmode=qmode)
                if n < 50:
                    continue
                q = fit(P, E); q['n'] = n
                q['gap_err'] = boot(ev, plane, bad, nb, qmode=qmode)
                q['half'] = half(P, E)
                q['half']['gap_err'] = boot(ev, plane, bad, nb, est=half, qmode=qmode)
                q['profile'] = [None if not np.isfinite(v) else round(float(v), 4) for v in P]
                q['by_sign'] = {}
                for sg in (1, -1):
                    ns, Ps, Es = profile(ev, plane, bad, qmode=qmode, tsign=sg)
                    if ns >= 40:
                        qs = fit(Ps, Es); qs['n'] = ns; qs['half'] = half(Ps, Es)
                        q['by_sign'][str(sg)] = qs
                q['by_tan'] = []
                for lo, hi in zip(TAN_BINS[:-1], TAN_BINS[1:]):
                    nt, Pt, Et = profile(ev, plane, bad, qmode=qmode, tbin=(lo, hi))
                    if nt >= 40:
                        qt = fit(Pt, Et); qt.update(n=nt, lo=lo, hi=hi, half=half(Pt, Et))
                        q['by_tan'].append(qt)
                R[qmode] = q
                sg = q['by_sign']
                hh = q['half']
                print(f'{key:16s} {plane} {qmode:4s} n {n:5d}  HALF gap {hh["gap"]:6.2f} ± {hh["gap_err"]:.2f} '
                      f'(z0 {hh.get("z0", np.nan):+.2f} z1 {hh.get("z1", np.nan):.2f}; +tan {sg.get("1", {}).get("half", {}).get("gap", np.nan):.2f} '
                      f'/ -tan {sg.get("-1", {}).get("half", {}).get("gap", np.nan):.2f}; by tan '
                      + ' '.join(f'{t["half"]["gap"]:.2f}' for t in q['by_tan']) + ')', flush=True)
                print(f'{"":16s}   erf fit gap {q["gap"]:6.2f} ± {q["gap_err"]:.2f}  '
                      f'(z0 {q["z0"]:+.2f}, z1 {q["z1"]:.2f}, s0 {q["s0"]:.2f}, s1 {q["s1"]:.2f}, slope {q["slope"]:+.4f}, '
                      f'chi2 {q["chi2dof"]:.1f})  +tan {sg.get("1", {}).get("gap", np.nan):.2f} / -tan {sg.get("-1", {}).get("gap", np.nan):.2f}  '
                      'by tan ' + ' '.join(f'{t["gap"]:.2f}' for t in q['by_tan']), flush=True)
            r[plane] = R
        r['zgrid'] = [float(ZG[0]), DZ, len(ZG)]
        out[key] = r
        json.dump(out, open(path, 'w'), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('build', 'fit'))
    ap.add_argument('key', nargs='?')
    ap.add_argument('--keys', default=','.join(KEYS))
    ap.add_argument('--tan-min', type=float, default=0.12)
    ap.add_argument('--tan-max', type=float, default=0.8)
    ap.add_argument('--nboot', type=int, default=30)
    ap.add_argument('--subrun')
    ap.add_argument('--base')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.key, a.tan_min, a.tan_max, a.subrun, a.base)
    else:
        run_fit(a.keys.split(','), a.nboot)


if __name__ == '__main__':
    main()
