"""q15_windows.py -- items 15 and 18 with Dylan's note (11-24): the drift windows (cathode assemblies) may have been swapped
between chambers between runs.  If so, a window's cathode relief should reappear in the gap map of another chamber, possibly
turned by a multiple of 90 deg or flipped.

Gap maps (delta ln q_u90 P-spline, as r3_gap, K = 8) of every mounting, evaluated on the inscribed disc |(U, W)| <= 1.35
(units of 100 mm).  For every pair (a, b): the correlation of map a with map b turned by an angle phi (every 5 deg, 0-355),
without and with a mirror.  The fine part is the map minus a quadratic fitted on the disc, after the turn.  A real match
is a peak at a multiple of 90 deg; the correlations at the other angles (more than 20 deg from a multiple of 90) give the
null distribution of the same pair, which has the same smoothness.  z = (r_peak - mean_null) / std_null.
Split-half correlations of each map give the noise ceiling. DETILT: ln q_u90 is first corrected for its dependence on
|tan| (pass-1 angles; the telescope acceptance makes the angle mix vary with position, a pattern common to all chambers).  C uses its last row only (the first two are censored).
-> $WORK/audit/q16/q15_windows.json
"""
import json
import sys
import numpy as np
from r3_lib import load_mounting, R1

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
RAD = 1.35
g = np.linspace(-RAD, RAD, 41)
GU, GW = [a.ravel() for a in np.meshgrid(g, g, indexing='ij')]
IN = GU ** 2 + GW ** 2 <= RAD ** 2
PU, PW = GU[IN], GW[IN]
Q = np.c_[np.ones(PU.size), PU, PW, PU ** 2, PW ** 2, PU * PW]
rng = np.random.default_rng(15)
NAMES = ['A625', 'A628', 'B', 'C', 'D', 'E623', 'E624']
ANG = np.arange(0, 360, 5)


DETILT = True    # remove the dependence of ln q_u90 on the track angle first (telescope acceptance varies with position)


def angle_free(D, ok):
    y = D.lnq.copy()
    if not DETILT:
        return y
    X = np.c_[np.ones(D.n), np.abs(D.tpx), np.abs(D.tpy), D.tpx ** 2, D.tpy ** 2]
    keep = ok.copy()
    for _ in range(3):
        c = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - X @ c
        keep = ok & (np.abs(r) < 3.5 * 1.4826 * np.median(np.abs(r[ok])))
    return y - X[:, 1:] @ c[1:]


def fit(D, m, K=8):
    ok = m & np.isfinite(D.lnq) & np.isfinite(D.tpx) & np.isfinite(D.tpy)
    y = angle_free(D, ok)
    spl = R1.Spl(K)
    c, _ = spl.fit(D.U[ok], D.W[ok], y[ok] - np.mean(y[ok]), 0.1)
    return spl, c


def ev(sc, phi=0.0, mirror=False, fine=True):
    spl, c = sc
    u, w = (PU, PW) if not mirror else (-PU, PW)
    t = np.radians(phi)
    uu, ww = np.cos(t) * u - np.sin(t) * w, np.sin(t) * u + np.cos(t) * w
    z = spl.basis(uu, ww) @ c
    if fine:
        z = z - Q @ np.linalg.lstsq(Q, z, rcond=None)[0]
    return z - z.mean()


def corr(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


fits, ceil = {}, {}
for n in NAMES:
    D, info, _ = load_mounting(n)
    keep = np.ones(D.n, bool)
    if n == 'C':
        keep = D.row == [i for i, r in enumerate(info) if r['key'] == 'det6:long'][0]
    fits[n] = fit(D, keep)
    cs = []
    for _ in range(8):
        h = rng.random(D.n) < 0.5
        a, b = fit(D, keep & h), fit(D, keep & ~h)
        cs.append((corr(ev(a, fine=False), ev(b, fine=False)), corr(ev(a), ev(b))))
    ceil[n] = np.mean(cs, axis=0).tolist()
    print(n, 'N', int(keep.sum()), 'split-half full/fine', np.round(ceil[n], 2), flush=True)

res = dict(ceiling=ceil, pairs={})
near90 = np.array([min(a % 90, 90 - a % 90) for a in ANG])
for i, a in enumerate(NAMES):
    for b in NAMES[i + 1:]:
        o = {}
        for part, fine in (('fine', True), ('full', False)):
            A = ev(fits[a], fine=fine)
            curves = {m: np.array([corr(A, ev(fits[b], phi, m, fine)) for phi in ANG]) for m in (False, True)}
            null = np.concatenate([curves[m][near90 > 20] for m in curves])
            best = None
            for m in curves:
                for k in np.where(near90 == 0)[0]:
                    r = curves[m][k]
                    if best is None or r > best[0]:
                        best = (r, int(ANG[k]), m)
            z = (best[0] - null.mean()) / null.std()
            # is the 90-multiple a local maximum of the curve (within +-10 deg)?
            cm = curves[best[2]]; k = int(best[1] // 5)
            peak_off = int(ANG[(k + np.argmax([cm[(k + d) % len(ANG)] for d in (-2, -1, 0, 1, 2)]) - 2) % len(ANG)])
            o[part] = dict(r=float(best[0]), angle=best[1], mirror=bool(best[2]), z=float(z), null_mean=float(null.mean()),
                           null_std=float(null.std()), null_max=float(null.max()), local_peak_at=peak_off,
                           ceiling=float(np.sqrt(max(ceil[a][0 if part == 'full' else 1], 0) * max(ceil[b][0 if part == 'full' else 1], 0))))
        res['pairs'][f'{a}|{b}'] = o
        f, F = o['fine'], o['full']
        print(f"{a:5s}-{b:5s} fine: r {f['r']:+.2f} at {f['angle']:3d}{' m' if f['mirror'] else '  '} z {f['z']:+.1f} "
              f"(null {f['null_mean']:+.2f}+-{f['null_std']:.2f}, max {f['null_max']:+.2f}; peak at {f['local_peak_at']}) | "
              f"full: r {F['r']:+.2f} at {F['angle']}{' m' if F['mirror'] else ''} z {F['z']:+.1f}", flush=True)
json.dump(res, open(f'{OUT}/q15_windows{"_detilt" if DETILT else ""}.json', 'w'), indent=1)
