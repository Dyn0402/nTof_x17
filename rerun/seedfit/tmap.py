#!/usr/bin/env python3
"""tmap.py <det> -- the drift column duration T per 3x3 region, from near-head-on tracks (|tan_x|, |tan_y| < 0.15,
no saturated sample): X-view waveforms deconvolved with the electronics template only (fitters.arrival_image), summed
over strips -> arrival-time distribution; T = its width between the 50 % crossings of the plateau-normalised
cumulative... (here: full width at half maximum of the arrival distribution, 15 ns bins). Then G = v_ladder x T per
region, with v_ladder from ladder_map.py (mean of the X and Y views)."""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', '..')]
from wft.calib import CalibrationBundle
from wft import model as wm
import fitters as F
L = os.path.expanduser('~/x17/cosmic_bench/vladder/')
det = sys.argv[1]
ev = pickle.load(open(L + det + '/cache.pkl', 'rb'))
wm.use_calibration(CalibrationBundle.load(L + det + '/bundle'))
LM = json.load(open(L + f'ladder_map_{det}.json'))
edges = [0, 133, 266, 400]


def fwhm(ta, a):
    a = np.convolve(a, np.ones(3) / 3, mode='same')
    pk = a.max(); i = np.flatnonzero(a >= 0.5 * pk)
    if len(i) < 2:
        return np.nan
    lo, hi = i[0], i[-1]
    tl = ta[lo - 1] + (0.5 * pk - a[lo - 1]) / (a[lo] - a[lo - 1]) * (ta[lo] - ta[lo - 1]) if lo > 0 else ta[lo]
    th = ta[hi] + (a[hi] - 0.5 * pk) / (a[hi] - a[hi + 1]) * (ta[hi + 1] - ta[hi]) if hi + 1 < len(a) else ta[hi]
    return th - tl


T = {}
STK = {}
cal = wm.CAL
for k, e in ev.items():
    if abs(e['tan_x']) > 0.15 or abs(e['tan_y']) > 0.15:
        continue
    P = e['x']
    if np.asarray(P['W']).max() >= wm.SAT:
        continue
    wm.set_nsamp(np.asarray(P['W']).shape[1])
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    ta, Q = F.arrival_image(W, noise, sat, 'x', step=15.0)
    a = Q.sum(0)
    if a.sum() <= 0:
        continue
    ft = e['ftst_x']; t0p = cal.t0_abs['x'].get(str(ft), cal.t0_abs['x'].get(ft))
    if t0p is None:
        continue
    tg = np.arange(-300, 2000, 15.0)
    st = np.interp(tg, ta - float(t0p), a / a.sum(), left=0, right=0)
    ix = min(int(np.searchsorted(edges, e['ref_mesh_x'], 'right')) - 1, 2)
    iy = min(int(np.searchsorted(edges, e['ref_mesh_y'], 'right')) - 1, 2)
    STK.setdefault((ix, iy), []).append(st)
    STK.setdefault('all', []).append(st)
tg = np.arange(-300, 2000, 15.0)


def stack_T(sts, nboot=60):
    sts = np.array(sts)
    def width(S):
        S = np.convolve(S, np.ones(3) / 3, mode='same')
        i0 = int(np.argmax(S))
        # plateau level: median of the central part above 50 % of the max
        hi = np.flatnonzero(S > 0.5 * S.max())
        plat = np.median(S[hi[0]:hi[-1] + 1])
        h = 0.5 * plat
        up = np.flatnonzero((S[:-1] < h) & (S[1:] >= h)); dn = np.flatnonzero((S[:-1] >= h) & (S[1:] < h))
        if not len(up) or not len(dn):
            return np.nan
        a, b = up[0], dn[-1]
        ta_ = tg[a] + (h - S[a]) / (S[a + 1] - S[a]) * 15.0
        tb_ = tg[b] + (S[b] - h) / (S[b] - S[b + 1]) * 15.0
        return tb_ - ta_
    t = width(sts.sum(0))
    rng = np.random.default_rng(0)
    bs = [width(sts[rng.integers(0, len(sts), len(sts))].sum(0)) for _ in range(nboot)]
    return t, float(np.nanstd(bs))


for key, sts in STK.items():
    if len(sts) >= 8:
        T[key] = stack_T(sts) + (len(sts),)
print(f"{det}: head-on stacked T (half-height width, trigger-aligned) all: {T['all'][0]:.0f} ± {T['all'][1]:.0f} ns, n {T['all'][2]}")
rows = []
for iy in (2, 1, 0):
    line = f'   y {edges[iy]:3d}-{edges[iy + 1]:3d} '
    for ix in range(3):
        tt = T.get((ix, iy))
        gx = [g for g in LM['x']['grid'] if g['ix'] == ix and g['iy'] == iy][0]
        gy = [g for g in LM['y']['grid'] if g['ix'] == ix and g['iy'] == iy][0]
        v = np.nanmean([gx['v'], gy['v']])
        if tt is not None and np.isfinite(tt[0]):
            tm, te = tt[0], tt[1]
            rows.append(dict(ix=ix, iy=iy, T=tm, Terr=te, v=v, G=v * tm / 1e3))
            line += f'| T {tm:4.0f}±{te:3.0f} v {v:4.1f} G {v * tm / 1e3:4.1f} '
        else:
            line += '|        .                 '
    print(line)
v = np.array([r['v'] for r in rows]); t = np.array([r['T'] for r in rows]); G = np.array([r['G'] for r in rows])
print(f'   corr(v, T) {np.corrcoef(v, t)[0, 1]:+.2f}   relative spread: v {np.std(v) / np.mean(v) * 100:.1f}%  T {np.std(t) / np.mean(t) * 100:.1f}%  G=vT {np.std(G) / np.mean(G) * 100:.1f}%')
json.dump(rows, open(L + f'tmap_{det}.json', 'w'), indent=1, default=float)
