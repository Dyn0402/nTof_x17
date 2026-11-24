"""q12_null.py -- item 12, fourth round: how significant is the part of the charge-gap link that the path length and the
drift field do not explain?

q12_gain.py measured d ln q / d ln T = 1.30 +- 0.19 on the fine 8 x 8 cell maps of the golden mounting (unsaturated
tracks, cross-run, noise-free denominator); the path length and the field predict 0.60-0.67. Its error is an event
bootstrap. It does not include the chance alignment of the fine charge map's own reproducible structure (strip gains,
pillars, resistive layer: about 7 % rms, most of the fine charge map) with the fine drift-time map, since both are the
same in every bootstrap replica.

Null: shift the drift-time maps (both runs together) cyclically on the cell grid by (du, dw) != (0, 0), and recompute the
slope with the charge maps unshifted. A cyclic shift keeps the smoothness and the rms of each map and breaks their
spatial relation. The spread of the null slopes is the uncertainty from chance alignment. Repeated for 6 x 6 and 10 x 10.

-> $WORK/audit/q16/q12_null.json
"""
import json
import numpy as np
from r3_lib import load_mounting
import r3_lib
import d_lib as DL

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
D, info, _ = load_mounting('A628')
qs = np.concatenate([np.log(0.5 * (DL.load(k).x_q_sum.values + DL.load(k).y_q_sum.values)) for k in r3_lib.MOUNT['A628']['rows']])
sat = ((D.satx > 0) | (D.saty > 0))
sel = ~sat & np.isfinite(D.lnq) & np.isfinite(qs)


def run(N):
    eu, ew = np.linspace(-1.75, 1.75, N + 1), np.linspace(-1.55, 1.55, N + 1)
    ci = np.searchsorted(eu, D.U) - 1; cj = np.searchsorted(ew, D.W) - 1
    inside = (ci >= 0) & (ci < N) & (cj >= 0) & (cj < N)
    Uc = 0.5 * (eu[1:] + eu[:-1]); Wc = 0.5 * (ew[1:] + ew[:-1])
    UU, WW = np.meshgrid(Uc, Wc, indexing='ij')
    Q = np.c_[np.ones(N * N), UU.ravel(), WW.ravel(), UU.ravel() ** 2, WW.ravel() ** 2, UU.ravel() * WW.ravel()]

    def cmap(x, r):
        m = sel & inside & (D.row == r)
        cell = (ci * N + cj)[m]
        s = np.bincount(cell, weights=x[m], minlength=N * N); c = np.bincount(cell, minlength=N * N)
        return np.where(c >= 20, s / np.maximum(c, 1), np.nan).reshape(N, N)

    def fine(v):
        v = v.ravel(); ok = np.isfinite(v); r = np.full_like(v, np.nan)
        r[ok] = v[ok] - Q[ok] @ np.linalg.lstsq(Q[ok], v[ok], rcond=None)[0]
        return r.reshape(N, N)

    T = {r: fine(cmap(D.lnq, r)) for r in (0, 1)}
    G = {r: fine(cmap(qs, r)) for r in (0, 1)}

    def cov(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.mean((a[ok] - a[ok].mean()) * (b[ok] - b[ok].mean())), ok.sum()

    def k(Ts):
        num = 0.5 * (cov(G[0], Ts[1])[0] + cov(G[1], Ts[0])[0])
        return num / cov(T[0], T[1])[0]          # denominator: the unshifted signal variance of T

    k0 = k(T)
    null = [k({r: np.roll(np.roll(T[r], du, 0), dw, 1) for r in (0, 1)})
            for du in range(N) for dw in range(N) if (du, dw) != (0, 0)]
    null = np.array(null)
    ncell = int(np.isfinite(T[0]).sum())
    out = dict(N=N, cells=ncell, k=float(k0), null_mean=float(null.mean()), null_sd=float(null.std()),
               frac_null_above=float(np.mean(null >= k0)),
               rms_fine_q=float(np.sqrt(abs(cov(G[0], G[1])[0]))), rms_fine_T=float(np.sqrt(abs(cov(T[0], T[1])[0]))))
    print(json.dumps(out))
    return out


res = {N: run(N) for N in (6, 8, 10)}
json.dump(res, open(f'{OUT}/q12_null.json', 'w'), indent=1)
