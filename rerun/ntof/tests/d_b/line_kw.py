#!/usr/bin/env python3
"""line_kw.py -- an angle truth for a chamber with no fitted partner: the "centroid line" (D/B studies, FINDINGS.md).

A cosmic muon through the chamber under test (T) and a partner chamber (P) is one straight line.  The A-C line
(rerun/ntof/ac_line.py) uses both chambers' FITTED tracks.  Here the partner contributes only a POSITION: the charge
centroid of its main cluster in each view (centroids.py), placed at its mid-gap depth (gap_P / 2), which needs no
angle, no v and no time ladder.  The chamber under test contributes its fitted p0 at the mesh (v-independent) or, as
a check, its own centroid at mid-gap.  The line through the two points is the truth; T's fitted tan per view is
compared with it, and the scale ratio fitted/true is the median over |true tan| in [0.15, 0.45] (ac_line's window),
so kw_new = kw_bundle x ratio (rerun/ntof NTOF_PASS_PLAN §6, "kw in situ").

Partners for D: B (opposite, parallel, 468 mm: B has no field-shaping ring but its positions need none), A and C (at
90 deg: the line is steep in the partner, which costs nothing because only its centroid is used).
Validation: the same construction for A (partner C's centroid) and C (partner A's centroid) on the run_149 cos_0001
v1k reconstructions, against ac_line.py's fitted-line ratio on the same events.

    line_kw.py validate      # A-C: centroid line vs fitted line
    line_kw.py measure       # D: B-D, A-D, C-D centroid lines on the d_b/reco D fits
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'ntof')]

W = '/media/ucla/work/ntof_pass1'
CEN = f'{W}/d_b/centroids'
OUT = f'{W}/d_b/line_kw'
RC149 = f'{W}/runs/run_149/run_config.json'
#: bench timing-free gaps (physics_inputs.json, as used for v in situ)
GAP = {'A': 28.08, 'B': 31.12, 'C': 28.0, 'D': 29.44}
KW_WIN = (0.15, 0.45)
#: v1 bundle v [um/ns] of the partners whose depth is taken from time (A and C are calibrated; B has no time ladder)
V_PART = {'A': 38.6, 'C': 28.1}
SAMPLE_NS = 60.0
_TMID = {}


def transforms():
    from ntof_tracking.reco import geometry as G
    return G.detector_transforms(json.load(open(RC149)))


def fitted(path, arm, tr, single=True):
    """ntof_reco output -> one row per gated track, local + global line (k = 1: the bundle's v is the working v)."""
    from sept26_prelim_analysis import build_tracks as BT
    cand = pd.read_parquet(path.replace('.parquet', '.candidates.parquet'))
    if 'tag' not in cand:
        cand['tag'] = 'all'
    df = BT.local_and_global(BT.pair_rows(cand), tr, k=1.0)
    df = df[df.gated]
    if single:
        df = df.groupby('event_id').filter(lambda g: len(g) == 1)
    return df.reset_index(drop=True)


_HOT = {}


def hot_spots(arm, view, binw=0.5, factor=15.0):
    """centroid positions that repeat far above the flat cosmic rate (noisy strip groups: B x 207.5, B y 8 / 122 /
    127, C x 158, C y 5 / 296 / 386, D x 80, D y 4 / 147 mm on run_149): 0.5 mm bins above 15 x the median bin."""
    k = (arm, view)
    if k not in _HOT:
        c = np.concatenate([np.load(f)[f'{arm}_{view}_cen'][np.load(f)[f'{arm}_{view}_has']]
                            for f in sorted(glob.glob(f'{CEN}/r149_*.npz'))])
        h, e = np.histogram(c, bins=np.arange(0, 401, binw))
        med = max(np.median(h[h > 0]), 1)
        _HOT[k] = e[:-1][h > factor * med] + binw / 2
    return _HOT[k]


def not_hot(arm, view, cen, tol=1.0):
    hs = hot_spots(arm, view)
    if not len(hs):
        return np.ones(len(cen), bool)
    return np.min(np.abs(np.asarray(cen)[:, None] - hs[None, :]), axis=1) > tol


def centroids(sub, arm, strict=True, min_strips=3, veto_hot=True):
    """per event: the partner's centroid (strip-map mm) in both views, from centroids.py."""
    z = np.load(f'{CEN}/r149_{sub}.npz')
    m = z[f'{arm}_x_has'] & z[f'{arm}_y_has']
    if veto_hot:
        m &= not_hot(arm, 'x', z[f'{arm}_x_cen']) & not_hot(arm, 'y', z[f'{arm}_y_cen'])
    m &= (z[f'{arm}_x_nsig'] >= min_strips) & (z[f'{arm}_y_nsig'] >= min_strips)
    if strict:
        m &= (z[f'{arm}_x_ncl'] == 1) & (z[f'{arm}_y_ncl'] == 1)
    tb = {}
    for p in 'xy':
        S = np.clip(z[f'{arm}_{p}_S'][m].astype(float), 0, None)
        tb[p] = (S * np.arange(S.shape[1]) * SAMPLE_NS).sum(1) / np.maximum(S.sum(1), 1e-9)
    return pd.DataFrame(dict(event_id=z[f'{arm}_eid'][m], cx=z[f'{arm}_x_cen'][m], cy=z[f'{arm}_y_cen'][m],
                             tbx=tb['x'], tby=tb['y'],
                             rx=z[f'{arm}_x_rms'][m], ry=z[f'{arm}_y_rms'][m],
                             nx=z[f'{arm}_x_nsig'][m], ny=z[f'{arm}_y_nsig'][m],
                             satx=z[f'{arm}_x_sat'][m], saty=z[f'{arm}_y_sat'][m]))


def tmid(arm):
    """the charge-weighted mean time of a FULL column (the depth gap/2), per view: the median over the cos_0001 A-C
    line muons (shallow, full-gap through-goers, both chambers calibrated)."""
    if arm not in _TMID:
        import ac_line
        H = f'{W}/heldout'
        _, M = ac_line.score(f'{H}/r149s1_A_v1k.parquet', f'{H}/r149s1_C_v1k.parquet', RC149)
        c = centroids('cosbounce_cos_0001', arm, strict=True)
        c = c[c.event_id.isin(M.event_id)]
        _TMID[arm] = dict(x=float(np.median(c.tbx)), y=float(np.median(c.tby)), n=int(len(c)))
    return _TMID[arm]


def centroid_global(c, arm, tr, depth='mid', tan_local=None):
    """the partner's point.  depth='mid': both centroids at gap/2.  depth='time': each view's centroid at the depth of
    its own visible charge, w = gap/2 + v (tbar - tmid), then carried along the line (local tans ``tan_local``) to
    gap/2 -- a steep track lights only part of its column above threshold, so its centroid is not at mid-gap."""
    from sept26_prelim_analysis import build_tracks as BT
    xl = BT.IN_PLANE_SIGN * (c.cx.to_numpy(float) - BT.STRIP_MAP_HALF)
    yl = BT.IN_PLANE_SIGN_Y * (c.cy.to_numpy(float) - BT.STRIP_MAP_HALF)
    h = GAP[arm] / 2
    if depth == 'time' and arm in V_PART and tan_local is not None:
        tm = tmid(arm)
        wx = np.clip(h + V_PART[arm] * 1e-3 * (c.tbx.to_numpy(float) - tm['x']), 0, GAP[arm])
        wy = np.clip(h + V_PART[arm] * 1e-3 * (c.tby.to_numpy(float) - tm['y']), 0, GAP[arm])
        tx, ty = tan_local
        # local_and_global: position at depth w = p_mesh - tan w  =>  p(h) = p(w) - tan (h - w)
        xl = xl - tx * (h - wx)
        yl = yl - ty * (h - wy)
    return xl, yl, tr.local_to_global(xl, yl, np.full(len(xl), h))


def line_table(T, tarm, C, parm, trs, own_centroid=None, depth='mid'):
    """T: fitted tracks of the arm under test; C: partner centroids.  Returns the joined table with the truth tans."""
    M = T.merge(C, on='event_id')
    if not len(M):
        return M
    trT, trP = trs[f'mx17_{tarm}'], trs[f'mx17_{parm}']
    if own_centroid is not None:
        M = M.merge(own_centroid.rename(columns={k: f'{k}_own' for k in own_centroid.columns if k != 'event_id'}),
                    on='event_id')
        oc = M[['event_id', 'cx_own', 'cy_own']].rename(columns={'cx_own': 'cx', 'cy_own': 'cy'})
        Pt = centroid_global(oc, tarm, trT)[2]
    else:
        Pt = M[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    Pp = centroid_global(M, parm, trP)[2]
    if depth == 'time':
        for _ in range(4):
            Lp = (Pt - Pp) @ trP.R
            Pp = centroid_global(M, parm, trP, 'time', (Lp[:, 0] / Lp[:, 2], Lp[:, 1] / Lp[:, 2]))[2]
    L = Pt - Pp
    M['lever_mm'] = np.linalg.norm(L, axis=1)
    L /= M['lever_mm'].to_numpy()[:, None]
    d = M[['d_x', 'd_y', 'd_z']].to_numpy(float)
    # distance of the partner point from the fitted line of T (the loose consistency cut, like ac_line's sep)
    Pt0 = M[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    w = Pp - Pt0
    M['miss_mm'] = np.linalg.norm(w - np.einsum('ij,ij->i', w, d)[:, None] * d, axis=1)
    M['res_deg'] = np.degrees(np.arccos(np.abs(np.einsum('ij,ij->i', d, L)).clip(0, 1)))
    Ll = L @ trT.R
    M['true_tanx'] = Ll[:, 0] / Ll[:, 2]
    M['true_tany'] = Ll[:, 1] / Ll[:, 2]
    return M


def ratios(M, miss_max=60.0, win=KW_WIN, nboot=400, seed=0):
    """per view: sign (corr), ratio fitted/true (median over |true| in win), bootstrap error, n."""
    out = {}
    keep = M.miss_mm.to_numpy() < miss_max
    rng = np.random.default_rng(seed)
    for ax in 'xy':
        fit = M[f'tan{ax}'].to_numpy(float)
        tt = M[f'true_tan{ax}'].to_numpy(float)
        r = np.corrcoef(fit[keep], tt[keep])[0, 1] if keep.sum() > 5 else np.nan
        tt = tt * (np.sign(r) if np.isfinite(r) else 1.0)
        m = keep & (np.abs(tt) > win[0]) & (np.abs(tt) < win[1]) & np.isfinite(fit)
        q = fit[m] / tt[m]
        if len(q) >= 8:
            ratio = float(np.median(q))
            boot = [np.median(q[rng.integers(0, len(q), len(q))]) for _ in range(nboot)]
            err = float(np.std(boot))
        else:
            ratio, err = np.nan, np.nan
        res = fit[keep] - tt[keep]
        out[ax] = dict(corr=float(r), n=int(m.sum()), ratio=ratio, err=err,
                       res_tan_68=float(np.percentile(np.abs(res), 68)) if keep.any() else np.nan)
    out['n_line'] = int(keep.sum())
    out['n_joined'] = int(len(M))
    return out


def validate():
    """A and C on cos_0001 (v1k bundles, kw from cos_0000): centroid line vs ac_line's fitted line."""
    import ac_line
    trs = transforms()
    H = f'{W}/heldout'
    sub = 'cosbounce_cos_0001'
    s, Mline = ac_line.score(f'{H}/r149s1_A_v1k.parquet', f'{H}/r149s1_C_v1k.parquet', RC149)
    rows = []
    for ax in 'xy':
        for arm in 'AC':
            rows.append(dict(method='fitted A-C line (ac_line.py)', arm=arm, view=ax, n=s[f'{arm}{ax}']['n_kw'],
                             ratio=s[f'{arm}{ax}']['ratio'], err=s[f'{arm}{ax}']['ratio_err']))
    for tarm, parm in (('A', 'C'), ('C', 'A')):
        T = fitted(f'{H}/r149s1_{tarm}_v1k.parquet', tarm, trs[f'mx17_{tarm}'])
        for strict in (True, False):
            for own in (False, True):
                C = centroids(sub, parm, strict=strict)
                oc = centroids(sub, tarm, strict=False)[['event_id', 'cx', 'cy']] if own else None
                for depth in ('mid', 'time'):
                    if own and depth == 'time':
                        continue
                    M = line_table(T, tarm, C, parm, trs, own_centroid=oc, depth=depth)
                    for miss in (30.0, 60.0, 100.0):
                        R = ratios(M, miss)
                        for ax in 'xy':
                            rows.append(dict(method=f'centroid line: {parm} centroid ({depth}) -> {tarm} '
                                                    f'{"centroid" if own else "fitted p0"}',
                                             partner_strict=strict, miss_max=miss, arm=tarm, view=ax, n=R[ax]['n'],
                                             ratio=R[ax]['ratio'], err=R[ax]['err'], n_line=R['n_line'],
                                             res_tan_68=R[ax]['res_tan_68']))
                    if strict and not own:
                        M.to_parquet(f'{OUT}/validate_{tarm}_from_{parm}_{depth}.parquet')
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/validate.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print(R.round(4).to_string(index=False))


def measure(reco_glob=f'{W}/d_b/reco/r149_*_D_v1_[a-z].parquet', tarm='D', label='v1'):
    """D against B (parallel, opposite; B centroid at mid-gap) and A / C (perpendicular: the line is steep in the
    partner, so its centroid is placed by the depth of its visible charge, depth='time')."""
    trs = transforms()
    rows, keepM = [], []
    files = sorted(glob.glob(reco_glob))
    Tall = {}
    for f in files:
        sub = 'cosbounce_' + os.path.basename(f).split('_cosbounce_')[1].split(f'_{tarm}_')[0]
        T = fitted(f, tarm, trs[f'mx17_{tarm}'])
        Tall[sub] = pd.concat([Tall[sub], T], ignore_index=True) if sub in Tall else T
    for parm in 'BAC':
        for strict in (True, False):
            for own in (False, True):
                for depth in (('mid',) if parm == 'B' else ('mid', 'time')):
                    parts = []
                    for sub, T in Tall.items():
                        C = centroids(sub, parm, strict=strict)
                        oc = centroids(sub, tarm, strict=False)[['event_id', 'cx', 'cy']] if own else None
                        M = line_table(T, tarm, C, parm, trs, own_centroid=oc, depth=depth)
                        if len(M):
                            parts.append(M.assign(subrun=sub))
                    if not parts:
                        continue
                    M = pd.concat(parts, ignore_index=True)
                    if strict and not own and depth == ('mid' if parm == 'B' else 'time'):
                        keepM.append(M.assign(partner=parm, depth=depth))
                    for lever_min in (0.0, 250.0):
                        Ml = M[M.lever_mm >= lever_min]
                        for miss in (30.0, 60.0, 100.0):
                            R = ratios(Ml, miss)
                            for ax in 'xy':
                                rows.append(dict(partner=parm, partner_strict=strict, depth=depth,
                                                 point=f'{tarm} {"centroid" if own else "fitted p0"}',
                                                 lever_min=lever_min, miss_max=miss, view=ax, n=R[ax]['n'],
                                                 corr=R[ax]['corr'], ratio=R[ax]['ratio'], err=R[ax]['err'],
                                                 n_line=R['n_line'], n_joined=R['n_joined'],
                                                 res_tan_68=R[ax]['res_tan_68']))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/measure_{tarm}_{label}.csv', index=False)
    if keepM:
        pd.concat(keepM, ignore_index=True).to_parquet(f'{OUT}/lines_{tarm}_{label}.parquet')
    with pd.option_context('display.width', 250, 'display.max_rows', 400):
        print(R.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['validate', 'measure'])
    ap.add_argument('--reco-glob', default=f'{W}/d_b/reco/r149_*_D_v1_[a-z].parquet')
    ap.add_argument('--arm', default='D')
    ap.add_argument('--label', default='v1')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.step == 'validate':
        validate()
    else:
        measure(a.reco_glob, a.arm, a.label)


if __name__ == '__main__':
    main()
