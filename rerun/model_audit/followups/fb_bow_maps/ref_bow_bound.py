#!/usr/bin/env python3
"""ref_bow_bound.py -- what the EXISTING data say about the telescope's (M3) contribution to the DUT mesh height h.

Replaces the unsourced "0.5-1.3 x the quoted value" of 08b_bow.tex.  Output: ref_bow/ref_bow_bound.json

Framework.  M3 plane i (z_i = 1302, 1185, 144, 24 mm) has surface z_i + d_i(x,y), d_i = kxx x^2 + kyy y^2 + kxy x y (+ linear
terms, absorbed by alignment).  A plane at height d reads a track of slope t at x_track + t*d.  The reference track is the
straight line through the four points; at height z it has the weights w_i(z) = 1/4 + (z_i-zm)(z-zm)/Sxx, so the shift it
inherits is  h_tel(z) = sum_i w_i(z) k_i  (times tan).  w(z) lies in span{1, z_i}.  Hence:
  * only the part of the 4-vector k_i along {1, z_i-zm} reaches the DUT:  k_i = a + b (z_i-zm)   [a = COMMON dish, b = a
    dish that grows linearly along z];  h_tel(z) = a + b (z-zm).  The orthogonal part of k_i is seen in the self-residuals
    (tan * x^2 terms) but never reaches the DUT.
  * a and b are invisible to the linear-in-position self-residual terms (a line absorbs them).  They show only through the
    NON-LINEARITY of d in the track's own position: k t (xc + t dz_i)^2 has a  t^3 dz_i^2  piece, whose pattern over the four
    planes, (I-H) diag(dz^2) k, is the only handle (observed via the per-plane tan^3 coefficients c_i, rank 2).
  * the a-mode shows as the (+,-,-,+) pattern of c_i, the b-mode as (+,-,+,-).  The latter is degenerate with each plane's own
    angle response (tan^3 non-linearity, mirror-mounted doublets), so b is NOT measured by the self-residuals; a is, under
    the assumption that no plane's tan^3 response has a (+,-,-,+) part.
  * a is the same at every DUT height, so the two-slot comparison cannot see it.  The slot comparison sees b:
    h_tel(702)-h_tel(232) = 470 mm * b.
Both view residuals are fitted jointly (X residual: t_x^3 <- kxx, t_x t_y^2 <- kyy, t_x^2 t_y <- kxy; Y residual likewise), per
plane, then the 4-vectors c are fitted for (a_xx,a_yy,a_xy,b_xx,b_yy,b_xy).  Depth-equivalent at 150 mm, same convention as
summary.py:  depth = k * 150^2 (mm), pooled over the four terms = (kxx+kyy)/2 * 150^2.

Part C (M3-independent, prior dependent): the w0 gradient does not involve M3.  r_f = ratio measured/predicted gradient;
if the telescope adds T to the X-strip kuu, r_true*(kuu - T) = r_meas*kuu with r_true in [0.5, 1] (flat cathode .. cathode
following the mesh).  Intersect over chambers of the top slot.
Runtime: seconds.
"""
import json, os, sys
import numpy as np, uproot, awkward as ak

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RAYS = os.path.join(REPO, 'mx_june_cosmic_qa', 'm3_self_resolution', 'sat_perplane_0to3.root')
RES = os.path.join(HERE, 'results_final')
OUT = os.path.join(HERE, 'ref_bow', 'ref_bow_bound.json')
Z = np.array([1302., 1185., 144., 24.])
R = 150.0
ZS = {'top': 702.0, 'bottom': 232.0}


def load():
    t = uproot.open(RAYS)['T']
    br = ['Chi2X', 'Chi2Y', 'NClusX', 'NClusY'] + [f'm{c}{k}' for c in 'XY' for k in range(4)]
    a = t.arrays(br, library='ak'); a = a[ak.num(a['Chi2X'], axis=1) == 1]
    return {k: ak.to_numpy(ak.flatten(a[k])) for k in br}


def design(d, chi, tmax):
    ok = np.ones(len(d['Chi2X']), bool)
    for c in 'XY':
        for k in range(4):
            ok &= np.isfinite(d[f'm{c}{k}'])
        ok &= (d[f'NClus{c}'] >= 4) & (d[f'Chi2{c}'] < chi)
    X = np.c_[[d[f'mX{k}'][ok] for k in range(4)]].T
    Y = np.c_[[d[f'mY{k}'][ok] for k in range(4)]].T
    zm = Z.mean(); dz = Z - zm
    def lf(P):
        return P.mean(1), (P * dz).sum(1) / (dz ** 2).sum()
    xc, tx = lf(X); yc, ty = lf(Y)
    s = (np.abs(tx) < tmax) & (np.abs(ty) < tmax)
    return X[s], Y[s], xc[s], tx[s], yc[s], ty[s], dz


def plane_coefs(P, own_t, oth_t, xc, yc, own_is_x, k):
    """tan^3-type signal coefficients (own^3, own*oth^2, own^2*oth) for plane k of one view; nuisance absorbs the rest."""
    dz = Z - Z.mean()
    res = P - (P.mean(1)[:, None] + ((P * dz).sum(1) / (dz ** 2).sum())[:, None] * dz[None, :])
    r = res[:, k]; g = np.abs(r) < 2.0
    ux, uy = ((xc, yc) if own_is_x else (yc, xc))
    ux = (ux - np.median(ux)) / 100; uy = (uy - np.median(uy)) / 100
    t, o = own_t, oth_t
    sig = [t ** 3, t * o ** 2, t ** 2 * o]
    nui = [np.ones_like(t), t, t ** 2, o, o ** 2, t * o, ux, ux * t, ux * o, uy, uy * t, uy * o,
           t * ux ** 2, t * uy ** 2, t * ux * uy]
    A = np.c_[tuple(sig + nui)]
    beta, *_ = np.linalg.lstsq(A[g], r[g], rcond=None)
    rr = r[g] - A[g] @ beta
    cov = np.linalg.inv(A[g].T @ A[g]) * (rr ** 2).mean()
    return beta[:3], cov[:3, :3]


def fit_modes(Xp, Yp, xc, tx, yc, ty, dz, free_b=True):
    """returns theta = (a_xx,a_yy,a_xy,b_xx,b_yy,b_xy) in 1/mm (and 1/mm^2 for b) from GLS over the 4 planes x 2 views."""
    H = np.eye(4) - np.c_[np.ones(4), dz] @ np.linalg.pinv(np.c_[np.ones(4), dz])
    C = {}
    for k in range(4):
        C[('X', k)] = plane_coefs(Xp, tx, ty, xc, yc, True, k)
        C[('Y', k)] = plane_coefs(Yp, ty, tx, xc, yc, False, k)
    # which k-term enters which regressor: X: (t^3,t o^2,t^2 o) <- (kxx,kyy,kxy); Y: (t^3, t o^2, t^2 o)=(ty^3, ty tx^2, ty^2 tx) <- (kyy,kxx,kxy)
    perm = {'X': [0, 1, 2], 'Y': [1, 0, 2]}      # index of (kxx,kyy,kxy) feeding each regressor
    rows, y, W = [], [], []
    for v in 'XY':
        for k in range(4):
            beta, cov = C[(v, k)]
            for j in range(3):
                term = perm[v][j]                 # 0:xx 1:yy 2:xy
                row = np.zeros(6)
                # c_ik = sum_m H[k,m] k_m dz_m^2 ,  k_m = a + b dz_m
                row[term] = np.sum(H[k] * dz ** 2)
                row[3 + term] = np.sum(H[k] * dz ** 3)
                rows.append(row); y.append(beta[j]); W.append(1 / cov[j, j])
    rows, y, W = np.array(rows), np.array(y), np.array(W)
    cols = list(range(6)) if free_b else [0, 1, 2]
    A = rows[:, cols] * np.sqrt(W)[:, None]; bb = y * np.sqrt(W)
    th, *_ = np.linalg.lstsq(A, bb, rcond=None)
    out = np.zeros(6); out[cols] = th
    return out, np.linalg.inv(A.T @ A)


def depth(th, z, zm=Z.mean()):
    """pooled (kxx+kyy)/2 at height z, as depth at 150 mm (mm)"""
    k = lambda i: th[i] + th[3 + i] * (z - zm)
    return 0.5 * (k(0) + k(1)) * R ** 2


def analyse(d, chi, tmax, nboot=0, seed=1):
    Xp, Yp, xc, tx, yc, ty, dz = design(d, chi, tmax)
    out = {'n': int(len(Xp)), 'chi2cut': chi, 'tmax': tmax}
    for lab, fb in (('b_free', True), ('b_zero', False)):
        th, cov = fit_modes(Xp, Yp, xc, tx, yc, ty, dz, fb)
        rec = {'a_depth150': 0.5 * (th[0] + th[1]) * R ** 2,
               'a_xx_depth150': th[0] * R ** 2, 'a_yy_depth150': th[1] * R ** 2,
               'h702': depth(th, 702.), 'h232': depth(th, 232.),
               'b_slot_diff_470': 0.5 * (th[3] + th[4]) * R ** 2 * 470.}
        if nboot:
            rng = np.random.default_rng(seed); n = len(Xp); bs = []
            for _ in range(nboot):
                i = rng.integers(0, n, n)
                t2, _ = fit_modes(Xp[i], Yp[i], xc[i], tx[i], yc[i], ty[i], dz, fb)
                bs.append([0.5 * (t2[0] + t2[1]) * R ** 2, depth(t2, 702.), depth(t2, 232.),
                           0.5 * (t2[3] + t2[4]) * R ** 2 * 470.])
            bs = np.array(bs)
            for j, nm in enumerate(['a_depth150', 'h702', 'h232', 'b_slot_diff_470']):
                rec[nm + '_err'] = float(bs[:, j].std())
        out[lab] = {k: float(v) for k, v in rec.items()}
    return out


def slot_constraint():
    S = json.load(open(os.path.join(RES, 'summary_final.json')))
    rows = {'top': ['det3', 'det3_p2'], 'bottom': ['det3_day625', 'det3_o23']}
    m = {}
    for s, names in rows.items():
        v = np.array([S[n]['depth_mm'] for n in names]); e = np.array([S[n]['depth_err'] for n in names])
        w = 1 / e ** 2; mean = (w * v).sum() / w.sum(); err = 1 / np.sqrt(w.sum())
        chi2 = (w * (v - mean) ** 2).sum(); infl = max(1., np.sqrt(chi2 / (len(v) - 1)))
        m[s] = dict(names=names, depth=v.tolist(), err=e.tolist(), mean=mean, err_stat=err, err_inflated=err * infl)
    dtb = m['top']['mean'] - m['bottom']['mean']
    m['top_minus_bottom'] = dict(value=dtb, err_stat=float(np.hypot(m['top']['err_stat'], m['bottom']['err_stat'])),
                                 err_inflated=float(np.hypot(m['top']['err_inflated'], m['bottom']['err_inflated'])),
                                 note='includes any change of the chamber bow on remount; = 470 mm * b only if the chamber is identical in both slots')
    return m


def rf_constraint(nmc=20000, seed=3):
    """Part C. X-strip kuu, r_f from the 2-D fit; telescope adds T (kuu units) to every top-slot chamber."""
    rng = np.random.default_rng(seed)
    names = {'A': 'det3', 'B': 'det2', 'D': 'det7', 'E': 'det4'}   # top slot only (C is in the bottom slot)
    info = {}; Tgrid = np.linspace(-1.5, 1.5, 601)   # in mm of depth (kuu*1.125)
    ok = np.ones((nmc, len(Tgrid)), bool)
    for lab, n in names.items():
        X = json.load(open(os.path.join(RES, f'xcheck_{n}.json')))
        k, ke = X['pos_x']['kuu'], X['pos_x']['kuu_err']
        cu, cue, B, Be = X['ang_x']['cu'], X['ang_x']['cu_err'], X['ang_x']['B'], X['ang_x']['B_err']
        kd, cud, Bd = rng.normal(k, ke, nmc), rng.normal(cu, cue, nmc), rng.normal(B, Be, nmc)
        rmeas = cud * 100 / (Bd * kd)
        # kuu_true = k_meas - T_kuu ; r_true = cu*100/(B*kuu_true) in [0.5,1]
        # T in depth mm = 0.5*kuu_T*1.5^2
        T_k = Tgrid / (0.5 * 1.5 ** 2)
        kt = kd[:, None] - T_k[None, :]
        rt = (cud * 100 / Bd)[:, None] / kt
        okc = (rt >= 0.5) & (rt <= 1.0) & (kt > 0)
        info[lab] = dict(chamber_row=n, kuu=k, kuu_err=ke, r_f_meas=float(np.median(rmeas)),
                         T_interval_mm_68=None)
        frac = okc.mean(0)
        ok &= okc
        okn = np.where(frac > 0.5)[0]
        info[lab]['T_range_mm_where_physical_prob_gt_half'] = [float(Tgrid[okn[0]]), float(Tgrid[okn[-1]])] if len(okn) else None
    # joint: probability that all four top-slot chambers sit in the physical range for a common T
    pj = ok.mean(0)
    best = float(Tgrid[np.argmax(pj)])
    # joint acceptance region (each MC draw must satisfy all four): report T where pj > 0.5*max and > 0.05
    sel = np.where(pj > 0.1 * pj.max())[0]
    return dict(per_chamber=info, joint_best_T_mm=best, joint_peak_prob=float(pj.max()),
                joint_T_range_mm_prob_gt_10pct_of_peak=[float(Tgrid[sel[0]]), float(Tgrid[sel[-1]])],
                joint_prob_at_T0=float(pj[np.argmin(abs(Tgrid))]),
                note='T = telescope contribution to the X-strip kuu, in depth-mm at 150 mm; prior r_f in [0.5,1]; physical-range only, no likelihood on r_f')


def main():
    d = load()
    res = {'cuts': {}}
    for chi, tmax in ((5, 9), (2, 9), (5, 0.3), (5, 0.25)):
        res['cuts'][f'chi2<{chi},|tan|<{tmax}'] = analyse(d, chi, tmax, nboot=100 if (chi, tmax) == (5, 9) else 0)
    # bootstraps for all cuts quickly
    for chi, tmax in ((2, 9), (5, 0.3), (5, 0.25)):
        res['cuts'][f'chi2<{chi},|tan|<{tmax}'] = analyse(d, chi, tmax, nboot=100)
    res['slot_constraint'] = slot_constraint()
    res['rf_constraint'] = rf_constraint()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    for k, v in res['cuts'].items():
        for lab in ('b_free', 'b_zero'):
            r = v[lab]
            print(f"{k:22s} n={v['n']:6d} {lab:7s} a={r['a_depth150']:+.2f}±{r['a_depth150_err']:.2f}  h(702)={r['h702']:+.2f}±{r['h702_err']:.2f}  "
                  f"h(232)={r['h232']:+.2f}±{r['h232_err']:.2f}  470b={r['b_slot_diff_470']:+.2f}±{r['b_slot_diff_470_err']:.2f}  "
                  f"[a_xx {r['a_xx_depth150']:+.2f} a_yy {r['a_yy_depth150']:+.2f}]")
    print(json.dumps(res['slot_constraint'], indent=1))
    print(json.dumps(res['rf_constraint'], indent=1))


if __name__ == '__main__':
    main()
