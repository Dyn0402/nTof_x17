"""T11: is the lateral footprint Gaussian?  Model-free.

Unsaturated near-head-on planes (M3 |tan| < 0.05, max < 3550 ADC, brightest strip not at the window edge, no dead
strip within +-2): time-integrated charge per strip q_i = sum over samples, relative to the brightest strip:
R1 = (q-1 + q+1)/q0, R2 = (q-2 + q+2)/q0 (medians over planes).  A Gaussian footprint of width s landing uniformly
within the strip (pitch 0.78 mm, the brightest strip = the one under the landing point most of the time; the toy
takes the max strip exactly as the data do) gives R1(s), R2(s); invert each.  s(R1) = s(R2) <=> Gaussian.
Also the head-on column adds drift diffusion: s^2 = s0^2 + Dp^2 z, z uniform over the gap, included in the toy.
Usage: python -I t11_footprint.py  -> results/t11_footprint.json"""
import json, os, pickle
import numpy as np
from scipy.special import erf
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
BUN = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/bundles/{}/calib_bundle_fe_nobow/bundle.json')
P = 0.78
DP = dict(det2=0.025, det3=0.023, det4=0.020, det6=0.018, det7=0.0225)
GAP = dict(det2=31.12, det3=28.08, det4=30.48, det6=28.0, det7=29.44)


def data_ratios(det, plane, tanmax=0.05):
    ev = pickle.load(open(CACHE.format(det), 'rb')); dead = set(json.load(open(BUN.format(det)))['dead'].get(plane, []))
    r1, r2, q0s = [], [], []
    for e in ev.values():
        if plane not in e or abs(e[f'tan_{plane}']) >= tanmax:
            continue
        W = np.asarray(e[plane]['W'], float); ch = np.asarray(e[plane]['ch']); pos = np.asarray(e[plane]['pos'])
        if W.max() >= 3550:
            continue
        q = W.sum(axis=1); c = int(np.argmax(W.max(axis=1)))
        if c < 2 or c > len(q) - 3 or W[c].max() < 400:
            continue
        if dead & set(ch[c - 2:c + 3].tolist()):
            continue
        if not np.allclose(np.diff(pos[c - 2:c + 3]), np.diff(pos[c - 2:c + 3]).mean(), atol=0.05):
            continue                                          # contiguous strips only
        r1.append((q[c - 1] + q[c + 1]) / q[c]); r2.append((q[c - 2] + q[c + 2]) / q[c]); q0s.append(q[c])
    return np.array(r1), np.array(r2)


def toy_ratios(s0, dp, gap, n=20000, rng=np.random.default_rng(5)):
    u = rng.uniform(-P / 2, P / 2, n); z = rng.uniform(0, gap, (n, 24))
    edges = (np.arange(-3, 4) - 0.5) * P                     # strips -3..+3 around strip 0
    q = np.zeros((n, 7))
    for k in range(24):                                       # the column: 24 depth slices
        s = np.sqrt(s0 ** 2 + dp ** 2 * z[:, k])[:, None]
        cdf = 0.5 * (1 + erf((np.r_[edges, 3.5 * P][None, :] - u[:, None]) / (np.sqrt(2) * s)))
        q += np.diff(cdf, axis=1)
    c = np.argmax(q[:, 1:6], axis=1) + 1
    i = np.arange(n)
    r1 = (q[i, c - 1] + q[i, c + 1]) / q[i, c]; r2 = (q[i, c - 2] + q[i, np.minimum(c + 2, 6)]) / q[i, c]
    return np.median(r1), np.median(r2)


def invert(target, idx, dp, gap, grid=np.arange(0.10, 0.80, 0.01)):
    vals = np.array([toy_ratios(s, dp, gap)[idx] for s in grid])
    return float(np.interp(target, vals, grid)) if vals[0] < target < vals[-1] else float('nan')


if __name__ == '__main__':
    out = {}
    for det in DP:
        for plane in 'xy':
            r1, r2 = data_ratios(det, plane)
            if len(r1) < 30:
                continue
            m1, m2 = float(np.median(r1)), float(np.median(r2))
            s1 = invert(m1, 0, DP[det], GAP[det]); s2 = invert(m2, 1, DP[det], GAP[det])
            t1, t2 = toy_ratios(s1, DP[det], GAP[det]) if np.isfinite(s1) else (np.nan, np.nan)
            out[f'{det}_{plane}'] = dict(n=len(r1), R1=m1, R2=m2, s_from_R1=s1, s_from_R2=s2, R2_pred_from_s1=float(t2))
            print(f'{det} {plane}: n={len(r1):4d} R1 {m1:.3f} R2 {m2:.3f} | Gaussian s0 from R1 {s1:.3f} mm, from R2 {s2:.3f} mm | '
                  f'R2 predicted by s(R1): {t2:.3f}')
    json.dump(out, open(f'{HERE}/results/t11_footprint.json', 'w'), indent=1)
