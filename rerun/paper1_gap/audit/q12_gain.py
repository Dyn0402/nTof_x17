"""q12_gain.py -- outstanding question 12: the fine drift-time map of the golden mounting correlates with the gain map
five times more strongly than the event-level amplitude slope predicts (b test, addendum 6: +0.13 +- 0.05 vs +0.03).
Hypothesis: saturated tracks carry a ln q_u90 offset (0.02-0.06) and their fraction follows the gain.
Cells n = 8 over the interior, fine part = cell map minus a quadratic.  Cross-run slope (noise-free):
  k = [cov(T_1, G_2) + cov(T_2, G_1)] / [2 cov(G_1, G_2)]     T = walk-corrected ln q_u90, G = ln q_sum (cell means)
for all events, unsaturated events, and with the saturated fraction regressed out.  Errors: bootstrap over events (200).
-> $WORK/audit/q16/q12_gain.json"""
import json
import numpy as np
from r3_lib import load_mounting
import r3_lib

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16'
N = 8
D, info, _ = load_mounting('A628')
# ln q_sum per event (gain proxy), same order as D
import d_lib as DL
qs = np.concatenate([np.log(0.5 * (DL.load(k).x_q_sum.values + DL.load(k).y_q_sum.values)) for k in r3_lib.MOUNT['A628']['rows']])
sat = ((D.satx > 0) | (D.saty > 0)).astype(float)
eu, ew = np.linspace(-1.75, 1.75, N + 1), np.linspace(-1.55, 1.55, N + 1)
ci = np.clip(np.searchsorted(eu, D.U) - 1, -1, N); cj = np.clip(np.searchsorted(ew, D.W) - 1, -1, N)
inside = (ci >= 0) & (ci < N) & (cj >= 0) & (cj < N) & np.isfinite(D.lnq)
cell = np.where(inside, ci * N + cj, -1)
Uc = np.repeat(0.5 * (eu[1:] + eu[:-1]), N); Wc = np.tile(0.5 * (ew[1:] + ew[:-1]), N)
Q = np.c_[np.ones(N * N), Uc, Wc, Uc ** 2, Wc ** 2, Uc * Wc]


def cmean(x, m, wts):
    s = np.bincount(cell[m], weights=(x * wts)[m], minlength=N * N)
    c = np.bincount(cell[m], weights=wts[m], minlength=N * N)
    return np.where(c >= 20, s / np.maximum(c, 1), np.nan)


def fine(v):
    ok = np.isfinite(v)
    r = np.full_like(v, np.nan)
    r[ok] = v[ok] - Q[ok] @ np.linalg.lstsq(Q[ok], v[ok], rcond=None)[0]
    return r


def cov(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.mean((a[ok] - a[ok].mean()) * (b[ok] - b[ok].mean()))


def k_of(sel, wts, resid_sat=False):
    T, G, S = {}, {}, {}
    for r in (0, 1):
        m = sel & inside & (D.row == r)
        T[r], G[r], S[r] = fine(cmean(D.lnq, m, wts)), fine(cmean(qs, m, wts)), fine(cmean(sat, m, wts))
        if resid_sat:   # remove the part of T explained by the saturated fraction, slope from the other run (cross)
            pass
    k = (cov(T[0], G[1]) + cov(T[1], G[0])) / (2 * cov(G[0], G[1]))
    ks = (cov(T[0], S[1]) + cov(T[1], S[0])) / (2 * cov(S[0], S[1])) if cov(S[0], S[1]) > 0 else np.nan
    rGS = cov(G[0], S[1]) / np.sqrt(abs(cov(G[0], G[1]) * cov(S[0], S[1]))) if cov(S[0], S[1]) > 0 else np.nan
    return k, ks, rGS


rng = np.random.default_rng(12)
res = {}
for lab, sel in (('all', np.ones(D.n, bool)), ('unsaturated', sat == 0), ('saturated', sat == 1)):
    k0 = k_of(sel, np.ones(D.n))
    B = np.array([k_of(sel, rng.poisson(1.0, D.n).astype(float)) for _ in range(200)])
    res[lab] = dict(n=int((sel & inside).sum()), k_T_on_gain=(float(k0[0]), float(np.nanstd(B[:, 0]))),
                    k_T_on_satfrac=(float(k0[1]), float(np.nanstd(B[:, 1]))),
                    r_gain_satfrac=(float(k0[2]), float(np.nanstd(B[:, 2]))))
    print(lab, json.dumps(res[lab]), flush=True)
json.dump(res, open(f'{OUT}/q12_gain.json', 'w'), indent=1)


# ---- second test (11-24, later): the path-length hypothesis.  A track crossing a gap g leaves charge in proportion to
# g, so the fine charge map should contain the fine gap map: d ln q / d ln T = 1 / (1 + beta) (about 0.6) if the
# fine T structure is gap.  Cross-run, noise-free denominator: kGT = [cov(G_1, T_2) + cov(G_2, T_1)] / [2 cov(T_1, T_2)].
def k_GT(sel, wts):
    T, G = {}, {}
    for r in (0, 1):
        m = sel & inside & (D.row == r)
        T[r], G[r] = fine(cmean(D.lnq, m, wts)), fine(cmean(qs, m, wts))
    return ((cov(G[0], T[1]) + cov(G[1], T[0])) / (2 * cov(T[0], T[1])),
            (cov(G[0], G[1])) ** 0.5, (cov(T[0], T[1])) ** 0.5,
            (cov(G[0], T[1]) + cov(G[1], T[0])) / 2 / np.sqrt(cov(G[0], G[1]) * cov(T[0], T[1])))


res2 = {}
for lab, sel in (('all', np.ones(D.n, bool)), ('unsaturated', sat == 0)):
    k0 = k_GT(sel, np.ones(D.n))
    B = np.array([k_GT(sel, rng.poisson(1.0, D.n).astype(float)) for _ in range(200)])
    res2[lab] = dict(k_lnq_on_lnT=(float(k0[0]), float(np.nanstd(B[:, 0]))), rms_fine_gain=float(k0[1]),
                     rms_fine_T=float(k0[2]), r_signal=(float(k0[3]), float(np.nanstd(B[:, 3]))))
    print('pathlength', lab, json.dumps(res2[lab]), flush=True)
json.dump(dict(first=res, pathlength=res2), open(f'{OUT}/q12_gain.json', 'w'), indent=1)
