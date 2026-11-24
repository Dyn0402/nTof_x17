"""q16_mountings.py -- outstanding question 16: are chamber E's two mountings (6-23, 6-24) the same surfaces?
Control: chamber A's two mountings (6-25, 6-27/28), where the cathode is known to move and the mesh is not.

For each pair of mountings:
 1. gap map (delta ln q_u90 spline, as r3_gap) -- correlation of the full map and of the fine part (quadratic removed),
    on the common coverage; error from a bootstrap over events.
 2. mesh height h from the position route (p0 - ref_mesh = offset(U, W) + tan_ref * h(U, W)), h quadratic, per view;
    the shape coefficients compared between mountings with bootstrap errors.
 3. angle-map transfer: a tilt-only spline map (r3 'dir_tilt') fitted on the whole of one mounting and applied to the
    other (constants refit per row), against the other mounting's own time-block CV map and its constant arm.
-> $WORK/audit/q16/q16.json
"""
import json
import os
import numpy as np
from r3_lib import (DL, R1, MOUNT, G0, BETA0, load_mounting, gap_field, select, fit_generic, tans, dth, s68, Spec)
import track_test as TT

WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/q16'
os.makedirs(OUT, exist_ok=True)
PAIRS = {'E': ('E623', 'E624'), 'A': ('A625', 'A628')}
UG, WG = np.linspace(-1.6, 1.6, 33), np.linspace(-1.4, 1.4, 29)
GU, GW = np.meshgrid(UG, WG, indexing='ij')
FAC = G0 / (1 + BETA0)
NB = 100
rng = np.random.default_rng(16)


def coverage(D):
    H, eu, ew = np.histogram2d(D.U, D.W, bins=[np.linspace(-1.75, 1.75, 15), np.linspace(-1.55, 1.55, 13)])
    ui = np.clip(np.searchsorted(eu, GU) - 1, 0, 13); wi = np.clip(np.searchsorted(ew, GW) - 1, 0, 11)
    return H[ui, wi] >= 15


def gapgrid(D, w=None, K=8):
    ok = np.isfinite(D.lnq)
    spl = R1.Spl(K)
    c, _ = spl.fit(D.U[ok], D.W[ok], D.lnq[ok] - np.mean(D.lnq[ok]), 0.1, wts=None if w is None else w[ok])
    return (spl.basis(GU.ravel(), GW.ravel()) @ c).reshape(GU.shape) * FAC


def quadfit(Z, cov):
    X = np.vstack([np.ones(cov.sum()), GU[cov], GW[cov], GU[cov] ** 2, GW[cov] ** 2, GU[cov] * GW[cov]]).T
    c = np.linalg.lstsq(X, Z[cov], rcond=None)[0]
    Q = c[0] + c[1] * GU + c[2] * GW + c[3] * GU ** 2 + c[4] * GW ** 2 + c[5] * GU * GW
    return Q, c


def corr(A, B, cov):
    a, b = A[cov] - A[cov].mean(), B[cov] - B[cov].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def mesh_h(key_list, w=None):
    """event-level fit per view: p0 - ref_mesh = o0 + o1 U + o2 W + tan_ref * (h0 + gu U + gw W + kuu U^2/2 + kww W^2/2 + kuw UW)
    (mm; U, W in 100 mm, so gu is mm per 100 mm and kuu mm per (100 mm)^2).  Returns {view: coef[3:]} on the concatenated rows,
    with a separate offset per row."""
    out = {}
    ds = [DL.load(k) for k in key_list]
    for p in 'xy':
        Xs, ys = [], []
        nrow = len(ds)
        for i, d in enumerate(ds):
            U = d.Uref.values; W = d.Wref.values; t = d[f'tan_ref_{p}'].values
            r = d[f'{p}_p0'].values - d[f'ref_mesh_{p}'].values
            s = np.abs(t) <= 0.6
            U, W, t, r = U[s], W[s], t[s], r[s]
            off = np.zeros((len(r), 3 * nrow)); off[:, 3 * i] = 1; off[:, 3 * i + 1] = U; off[:, 3 * i + 2] = W
            H = np.vstack([t, t * U, t * W, t * U ** 2 / 2, t * W ** 2 / 2, t * U * W]).T
            Xs.append(np.hstack([off, H])); ys.append(r)
        X, y = np.vstack(Xs), np.concatenate(ys)
        out[p] = (X, y, 3 * nrow)
    return out


def lsq_clip_w(X, y, wts, clip=3.5, passes=3):
    keep = wts > 0
    for _ in range(passes + 1):
        ww = wts * keep
        c = np.linalg.lstsq(X * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None)[0]
        r = y - X @ c
        rs = 1.4826 * np.median(np.abs(r[wts > 0] - np.median(r[wts > 0])))
        keep = np.abs(r - np.median(r[wts > 0])) < clip * rs
    return c


HN = ['h0', 'gu', 'gw', 'kuu', 'kww', 'kuw']
res = {}
for ch, (m1, m2) in PAIRS.items():
    R = dict(mountings=[m1, m2])
    D1, i1, b1 = load_mounting(m1)
    D2, i2, b2 = load_mounting(m2)
    cov = coverage(D1) & coverage(D2)
    # ---- 1. gap maps
    Z1, Z2 = gapgrid(D1), gapgrid(D2)
    Q1, q1 = quadfit(Z1, cov); Q2, q2 = quadfit(Z2, cov)
    full, fine, fullS, fineS = [], [], [], []
    for b in range(NB):
        w1 = rng.poisson(1.0, D1.n).astype(float); w2 = rng.poisson(1.0, D2.n).astype(float)
        a, bb = gapgrid(D1, w1), gapgrid(D2, w2)
        qa, _ = quadfit(a, cov); qb, _ = quadfit(bb, cov)
        full.append(corr(a, bb, cov)); fine.append(corr(a - qa, bb - qb, cov))
        # same-mounting split-half reference for the noise ceiling: independent halves of mounting 2 (the larger one)
        h = rng.random(D2.n) < 0.5
        za = gapgrid(D2, w2 * h); zb = gapgrid(D2, w2 * ~h)
        qa2, _ = quadfit(za, cov); qb2, _ = quadfit(zb, cov)
        fullS.append(corr(za, zb, cov)); fineS.append(corr(za - qa2, zb - qb2, cov))
    names = ['c', 'gu', 'gw', 'guu/2', 'gww/2', 'guw']
    R['gap'] = dict(ncov=int(cov.sum()),
                    corr_full=(corr(Z1, Z2, cov), float(np.std(full))), corr_fine=(corr(Z1 - Q1, Z2 - Q2, cov), float(np.std(fine))),
                    splithalf_full_m2=(float(np.mean(fullS)), float(np.std(fullS))),
                    splithalf_fine_m2=(float(np.mean(fineS)), float(np.std(fineS))),
                    quad_m1={n: float(v) for n, v in zip(names, q1)}, quad_m2={n: float(v) for n, v in zip(names, q2)},
                    rms_full=(float(Z1[cov].std()), float(Z2[cov].std())), rms_diff=float((Z1 - Z2)[cov].std()))
    print(ch, 'gap', R['gap']['corr_full'], R['gap']['corr_fine'], R['gap']['splithalf_full_m2'], flush=True)
    # ---- 2. mesh height
    mh = {}
    for m in (m1, m2):
        H = mesh_h(MOUNT[m]['rows'])
        mh[m] = {}
        for p, (X, y, no) in H.items():
            c0 = lsq_clip_w(X, y, np.ones(len(y)))[no:]
            bs = np.array([lsq_clip_w(X, y, rng.poisson(1.0, len(y)).astype(float))[no:] for _ in range(NB)])
            mh[m][p] = {n: (float(c0[k]), float(bs[:, k].std())) for k, n in enumerate(HN)}
    R['mesh'] = mh
    R['mesh_diff'] = {p: {n: (mh[m2][p][n][0] - mh[m1][p][n][0],
                              float(np.hypot(mh[m2][p][n][1], mh[m1][p][n][1]))) for n in HN} for p in 'xy'}
    print(ch, 'mesh', json.dumps(R['mesh_diff']), flush=True)
    # ---- 3. angle-map transfer (tilt-only map, whole training mounting)
    tr = {}
    for a, b, Da, Db in ((m1, m2, D1, D2), (m2, m1, D2, D1)):
        K, KS, lam, sc = select(Da, 'T')
        m = fit_generic(Spec('phi', K, KS, 1.0), Da, lam)
        tx, ty = tans(Db, m)
        cx, cy = tans(Db, None)
        cv = np.load(f'{WORK}/audit/r3/cv_{b}.npz')
        out = {}
        for p, t, c, trr, tp in (('x', tx, cx, Db.trx, Db.tpx), ('y', ty, cy, Db.trY, Db.tpy)):
            arr = {'pass1': dth(tp, trr), 'const': dth(c, trr), 'transfer': dth(t, trr),
                   'own_cv': cv[f'time5|dir_tilt|{p}'].astype(float)}
            idx = rng.integers(0, len(trr), (300, len(trr)))
            S = {k: np.array([s68(v[i]) for i in idx]) for k, v in arr.items()}
            out[p] = {k: (s68(v), float(S[k].std())) for k, v in arr.items()}
            out[p]['transfer-const'] = (s68(arr['transfer']) - s68(arr['const']), float((S['transfer'] - S['const']).std()))
            out[p]['own_cv-const'] = (s68(arr['own_cv']) - s68(arr['const']), float((S['own_cv'] - S['const']).std()))
        tr[f'{a}->{b}'] = dict(sel=[K, KS, lam], **out)
        print(ch, f'{a}->{b}', json.dumps(out), flush=True)
    R['transfer'] = tr
    res[ch] = R
json.dump(res, open(f'{OUT}/q16.json', 'w'), indent=1, default=float)
print('done')
