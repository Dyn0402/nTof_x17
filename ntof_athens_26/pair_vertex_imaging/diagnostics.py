#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics.py -- the seven measurements that take the pair vertex apart.

Each writes one CSV into ``<out>/pair_vertex/``.  They are ordered as the
argument runs: confirm the observation, show what the vertex algebraically IS,
then take away one candidate explanation at a time until what is left is
measured rather than asserted.

  1 ``classes.csv``      the observation.  Per arm pair, real and mixed: the
                         legs' own pointing, the 3D vertex, the transverse
                         crossing, the y mismatch, the conditioning.
  2 ``decomposition.csv``the algebra.  ``v_r_xz`` is shown to be exactly
                         ``|leg-miss combination| / |sin psi|`` -- the pair
                         vertex is a FUNCTION of the two single-track
                         pointings, not an independent measurement of the
                         source, and the function amplifies.
  3 ``leg_scan.csv``     tighten the per-leg pointing cut from 60 mm to 5 mm
                         and watch the vertex follow it.  If the vertex were
                         limited by something other than the legs it would not.
  4 ``pointing.csv``     the single-track pointing resolution, per arm, with a
                         TAN-SHUFFLED null that has the source information
                         removed.  This is where the centroid-versus-event
                         distinction becomes a number.
  5 ``scale.csv``        multiply every tan by s.  The band crossing is
                         scale-free and does not move; the vertex is not and
                         has an optimum.  Where that optimum sits is a second,
                         independent read on the angle scale.
  6 ``ybudget.csv``      how much of the gap between ``v_r`` and ``v_r_xz`` is
                         the y plane, by substituting the y information away.
  7 ``floor.csv``        the ideal-leg substitution.  Replace one or both legs'
                         directions with a direction that points exactly at a
                         random point in the capsule and re-vertex: the floor
                         set by the source size and the crossing geometry, and
                         therefore how much imaging is available at all.

    python -m pair_vertex_imaging.diagnostics
    python -m pair_vertex_imaging.diagnostics --only pointing,scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from pair_vertex_imaging import vertex_lab as VL  # noqa: E402

#: The published leg cut.  Every table that needs one point rather than a scan
#: uses this, so a number here and a number in `pair_qa` describe one sample.
DCA_PUB = 30.0

CLASSES = ('intra', 'perpendicular', 'opposing')


def _q(v, qs=(0.25, 0.5, 0.75, 0.9)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, [100 * x for x in qs]) if len(v) else [np.nan] * len(qs)


def signed_miss(P: pd.DataFrame, leg: int) -> np.ndarray:
    """Signed transverse miss of one leg at the beam axis [mm].

    ``dca_axis_mm`` is its absolute value; the SIGN is what makes the vertex
    algebra work, because two legs that miss on the same side and two that miss
    on opposite sides put the crossing in completely different places.
    """
    px = P[f'p0_x_{leg}'].to_numpy(float)
    pz = P[f'p0_z_{leg}'].to_numpy(float)
    dx = P[f'd_x_{leg}'].to_numpy(float)
    dz = P[f'd_z_{leg}'].to_numpy(float)
    n = np.hypot(dx, dz)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (px * dz - pz * dx) / np.where(n > 1e-12, n, np.nan)


# --------------------------------------------------------------------------- #
# 1 -- the observation
# --------------------------------------------------------------------------- #
def classes(P: pd.DataFrame, dca: float = DCA_PUB) -> pd.DataFrame:
    """Per arm pair, real and mixed: legs, vertex, crossing, y, conditioning.

    ``f_10`` columns are the fraction of pairs whose vertex lands inside the
    capsule bore (10 mm).  That is the number the question is really about: an
    imaging estimator that puts 4 % of its pairs on a 10 mm source is not
    imaging it.
    """
    d = P[P.dca_worst < dca]
    rows = []
    # The per-class rows carry ``pair == 'all'`` and come FIRST, because they
    # are the numbers the report quotes: an arm pair with 546 entries and one
    # with 17 744 are not equal votes, and taking iloc[0] of a per-pair table
    # would quietly quote the smallest one.
    groups = [(t, 'all', g) for t, g in d.groupby('topo')]
    groups += [(t, p_, g) for (t, p_), g in d.groupby(['topo', 'pair'])]
    for topo, pair, g in groups:
        for mx in (False, True):
            h = g[g.mixed == mx]
            if len(h) < 50:
                continue
            rows.append(dict(
                topology=topo, pair=pair, mixed=mx, n=len(h),
                leg_dca_best=np.nanmedian(h.dca_best),
                leg_dca_worst=np.nanmedian(h.dca_worst),
                v_r=np.nanmedian(h.v_r), v_r_xz=np.nanmedian(h.v_r_xz),
                v_r_p90=_q(h.v_r, (0.9,))[0], v_r_xz_p90=_q(h.v_r_xz, (0.9,))[0],
                sep_mm=np.nanmedian(h.sep_mm),
                abs_dy_cross=np.nanmedian(np.abs(h.dy_cross)),
                sin_psi_xz=np.nanmedian(h.sin_psi_xz),
                open_deg=np.nanmedian(h.open_deg),
                f_vr_10=float(np.mean(h.v_r < 10)),
                f_vrxz_10=float(np.mean(h.v_r_xz < 10)),
                f_legbest_10=float(np.mean(h.dca_best < 10)),
                f_behind=float(np.mean(h.s_xz < 0))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2 -- the algebra
# --------------------------------------------------------------------------- #
def decomposition(P: pd.DataFrame, dca: float = DCA_PUB) -> pd.DataFrame:
    """``v_r_xz`` = |combined leg miss| / |sin psi|, verified, then split.

    Two lines in a plane, each a known perpendicular distance from the origin,
    meet at a point whose distance from the origin is fixed by those two
    distances and the angle between them:

        |c| = sqrt(e1^2 + e2^2 - 2 e1 e2 cos psi) / |sin psi|

    Nothing else enters.  So the transverse pair vertex is not a second,
    independent look at the source -- it is a deterministic combination of the
    two legs' own pointing, and ``1/|sin psi|`` is the factor by which it
    multiplies their error.  The residual column is the check that this module
    and `vertex_lab` agree; it is ~1e-16 and if it ever is not, one of the two
    has a sign wrong.
    """
    d = P[(P.dca_worst < dca) & (~P.mixed)].copy()
    e1, e2 = signed_miss(d, 1), signed_miss(d, 2)
    d1 = np.array(d[['d_x_1', 'd_z_1']].to_numpy(float), copy=True)
    d2 = np.array(d[['d_x_2', 'd_z_2']].to_numpy(float), copy=True)
    d1 /= np.linalg.norm(d1, axis=1)[:, None]
    d2 /= np.linalg.norm(d2, axis=1)[:, None]
    cos = np.einsum('ij,ij->i', d1, d2)
    sin = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    comb = np.sqrt(np.clip(e1 ** 2 + e2 ** 2 - 2 * e1 * e2 * cos, 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        pred = comb / np.abs(sin)
        amp = 1.0 / np.abs(sin)
    d = d.assign(comb=comb, amp=amp, pred=pred,
                 same_side=np.sign(e1) == np.sign(e2))
    rows = []
    for topo, g in d.groupby('topo'):
        rel = np.abs(g.pred - g.v_r_xz) / np.maximum(g.v_r_xz, 1e-9)
        rows.append(dict(
            topology=topo, n=len(g),
            closed_form_max_rel_resid=float(np.nanmax(rel)),
            leg_comb_med=float(np.nanmedian(g.comb)),
            amp_p25=_q(g.amp, (0.25,))[0], amp_med=float(np.nanmedian(g.amp)),
            amp_p75=_q(g.amp, (0.75,))[0], amp_p90=_q(g.amp, (0.9,))[0],
            v_r_xz_med=float(np.nanmedian(g.v_r_xz)),
            v_r_med=float(np.nanmedian(g.v_r)),
            frac_amp_gt2=float(np.mean(g.amp > 2)),
            frac_amp_gt5=float(np.mean(g.amp > 5)),
            frac_same_side=float(np.mean(g.same_side))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3 -- the leg cut scan
# --------------------------------------------------------------------------- #
def leg_scan(P: pd.DataFrame, cuts=(5, 10, 15, 20, 30, 40, 60)) -> pd.DataFrame:
    """Tighten the per-leg pointing cut and watch the vertex follow.

    The vertex tracking the cut is the signature of a vertex that is limited by
    its legs and by nothing else.  A vertex limited by, say, a broken y plane
    or by a wrong pairing would sit still while the legs improved.
    """
    rows = []
    for c in cuts:
        d = P[P.dca_worst < c]
        for topo in CLASSES:
            for mx in (False, True):
                g = d[(d.topo == topo) & (d.mixed == mx)]
                if len(g) < 50:
                    continue
                rows.append(dict(
                    leg_cut_mm=c, topology=topo, mixed=mx, n=len(g),
                    leg_comb_med=float(np.nanmedian(
                        np.hypot(g.dca_axis_mm_1, g.dca_axis_mm_2))),
                    v_r_xz_med=float(np.nanmedian(g.v_r_xz)),
                    v_r_med=float(np.nanmedian(g.v_r)),
                    f_vrxz_10=float(np.mean(g.v_r_xz < 10)),
                    f_vr_10=float(np.mean(g.v_r < 10))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4 -- the single-track pointing, and its null
# --------------------------------------------------------------------------- #
def _robust_line(u, t, n_iter=8):
    A = np.vstack([u, np.ones_like(u)]).T
    w = np.ones_like(u)
    c = np.array([0.0, 0.0])
    for _ in range(n_iter):
        c, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        r = t - (c[0] * u + c[1])
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * max(s, 1e-4))) ** 2)
    return float(c[0]), float(c[1])


DCA_EDGES = np.r_[np.arange(0, 120, 2.0), np.arange(120, 320, 10.0)]

#: The tan multipliers every scale scan uses.  1.33 is on the grid on purpose:
#: it is the factor the arm-A scintillator wall independently asks for
#: (`sept26_prelim_analysis/STATUS.md`, 2026-09-10), so the scan answers what
#: the wall's k would do to the vertex without a second pass.
SCALE_FACTORS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.33, 1.5, 1.8)


def _pointing_one(run: str, subruns, src: str, seed: int) -> tuple:
    """Per arm, for ONE run: the dca histogram, and the same with the source
    information destroyed.

    THE NULL.  Shuffling ``tan`` among the tracks of one arm keeps every
    marginal -- the same impact points, the same angular distribution, the same
    acceptance -- and destroys only the CORRELATION between where a track lands
    and which way it was going.  That correlation is the entire content of
    "this track came from the capsule", so the shuffled sample is the same
    chamber reading the same rates with no source.  The difference between the
    two histograms is the pointing information, in units anyone can check.
    """
    import numpy as np
    import pandas as pd
    cols = ['arm', 'gated', 'angle_calibrated', 'p0_x', 'p0_y', 'p0_z',
            'd_x', 'd_y', 'd_z', 'x_local', 'tanx', 'dca_axis_mm']
    out = []
    for sub in subruns:
        p = Path(src) / f'tracks_{run}_{sub}.parquet'
        out.append(pd.read_parquet(p, columns=cols))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated & t.angle_calibrated]
    rng = np.random.default_rng(seed)
    rows, hists, focus = [], {}, []
    for arm, g in t.groupby('arm'):
        P0 = g[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        D = g[['d_x', 'd_y', 'd_z']].to_numpy(float)
        u, w = VL.U_HAT[arm], VL.W_HAT[arm]
        # The miss distance, rebuilt from tan so the shuffled version can use
        # the identical formula: e = p.n with n the XZ normal of the line.
        du, dw = D @ u, D @ w
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = du / dw
        pu, pw = P0 @ u, P0 @ w
        # In the chamber's transverse (u, w) frame the line through (pu, pw)
        # has direction proportional to (tx, 1), so its signed distance from
        # the beam axis is (pu*1 - pw*tx)/|(tx,1)|.  A track that really came
        # from the axis has tx = pu/pw and this is identically zero, which is
        # the check below -- it must reproduce the stage-3 ``dca_axis_mm``
        # column to machine precision, and it does (< 1e-9 mm).
        e = (pu - pw * tx) / np.hypot(tx, 1.0)
        ok = np.isfinite(e) & np.isfinite(tx)
        resid = float(np.nanmax(np.abs(
            np.abs(e[ok]) - g.dca_axis_mm.to_numpy(float)[ok])))
        if resid > 1e-6:
            raise AssertionError(
                f'{run}/{arm}: rebuilt miss disagrees with dca_axis_mm by '
                f'{resid:.3g} mm -- the transverse frame is wrong')
        e, tx, pu, pw = e[ok], tx[ok], pu[ok], pw[ok]
        if len(e) < 200:
            continue
        txs = rng.permutation(tx)
        es = (pu - pw * txs) / np.hypot(txs, 1.0)
        hists[arm] = (np.histogram(np.abs(e), DCA_EDGES)[0],
                      np.histogram(np.abs(es), DCA_EDGES)[0])
        # The focus scan, on the UNCUT gated sample.  This is the version of
        # the scale scan that is free of the circularity in `scale_scan`:
        # nothing here has been selected on pointing, so a minimum in the miss
        # distance against ``s`` is a property of the angle scale and not of
        # the cut that made the sample.
        for sc in SCALE_FACTORS:
            ee = (pu - pw * (sc * tx)) / np.hypot(sc * tx, 1.0)
            focus.append(dict(run=run, arm=arm, scale=sc, n=len(ee),
                              med_dca=float(np.median(np.abs(ee))),
                              f10=float(np.mean(np.abs(ee) < 10)),
                              f30=float(np.mean(np.abs(ee) < 30))))
        sl, ic = _robust_line(pu, tx)
        rows.append(dict(run=run, arm=arm, n=len(e),
                         med_dca=float(np.median(np.abs(e))),
                         med_dca_null=float(np.median(np.abs(es))),
                         f10=float(np.mean(np.abs(e) < 10)),
                         f10_null=float(np.mean(np.abs(es) < 10)),
                         f30=float(np.mean(np.abs(e) < 30)),
                         f30_null=float(np.mean(np.abs(es) < 30)),
                         band_slope_x_d=float(sl * VL.D_PERP_MM),
                         band_x0=float(-ic / sl) if sl else np.nan))
    return run, pd.DataFrame(rows), hists, pd.DataFrame(focus)


def pointing(src: Path, jobs: int, include_pre_access: bool, seed: int
             ) -> tuple:
    rs = VL.discover(src, include_pre_access)
    rows, H, F = [], {}, []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_pointing_one, r, s, str(src), seed): r
                for r, s in rs.items()}
        for f in as_completed(futs):
            run, df, hists, fc = f.result()
            rows.append(df)
            F.append(fc)
            for a, (h, hn) in hists.items():
                if a not in H:
                    H[a] = [np.zeros(len(DCA_EDGES) - 1, np.int64),
                            np.zeros(len(DCA_EDGES) - 1, np.int64)]
                H[a][0] += h
                H[a][1] += hn
            print(f'  {run:<10} pointing ok', flush=True)
    R = pd.concat(rows, ignore_index=True)
    hist = pd.DataFrame({'lo': DCA_EDGES[:-1], 'hi': DCA_EDGES[1:]})
    for a, (h, hn) in sorted(H.items()):
        hist[f'{a}_data'] = h
        hist[f'{a}_null'] = hn
    # Pool the focus scan weighting each run by its own track count, so a
    # 500 k-track run is not one vote against a 5 k-track one.
    Fc = pd.concat(F, ignore_index=True)
    G = (Fc.assign(_m=Fc.med_dca * Fc.n, _f=Fc.f10 * Fc.n)
         .groupby(['arm', 'scale'])[['_m', '_f', 'n']].sum().reset_index())
    G['med_dca'] = G._m / G.n
    G['f10'] = G._f / G.n
    return R, hist, G[['arm', 'scale', 'n', 'med_dca', 'f10']].rename(
        columns={'n': 'n_tracks'})


def pointing_summary(R: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Per arm, pooled: the resolution, the null, and the excess over it.

    ``purity_10mm`` is the fraction of tracks inside 10 mm that the null does
    not account for -- i.e. how much of the "pointing" population is genuinely
    pointing.  It is a lower bound on the target fraction and an upper bound on
    nothing, because a target track can also land outside 10 mm.
    """
    rows = []
    for a in sorted(x.split('_')[0] for x in hist.columns if x.endswith('_data')):
        d, n = hist[f'{a}_data'].to_numpy(), hist[f'{a}_null'].to_numpy()
        ctr = 0.5 * (hist.lo + hist.hi).to_numpy()
        tot = d.sum()
        i10 = ctr < 10
        i30 = ctr < 30
        g = R[R.arm == a]
        rows.append(dict(
            arm=a, n_tracks=int(tot), n_runs=int(g.run.nunique()),
            med_dca=float(g.med_dca.median()),
            med_dca_null=float(g.med_dca_null.median()),
            f10=float(d[i10].sum() / tot), f10_null=float(n[i10].sum() / tot),
            f30=float(d[i30].sum() / tot), f30_null=float(n[i30].sum() / tot),
            excess_10=float((d[i10].sum() - n[i10].sum()) / tot),
            purity_10mm=float(1 - n[i10].sum() / max(d[i10].sum(), 1)),
            band_slope_x_d=float(g.band_slope_x_d.median()),
            band_x0=float(g.band_x0.median())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5 -- the angle scale
# --------------------------------------------------------------------------- #
def scale_scan(P: pd.DataFrame, T: pd.DataFrame, factors=SCALE_FACTORS,
               dca: float = DCA_PUB) -> pd.DataFrame:
    """Multiply every chamber's tan by ``s`` and re-measure both estimators.

    The band crossing is scale-free by construction (`source_imaging`'s opening
    paragraph): scaling every angle scales the fitted slope and intercept
    together and ``-intercept/slope`` does not move.  The vertex has no such
    protection -- it uses the angle's MAGNITUDE, not just its gradient -- so if
    the campaign's ``k`` is wrong by the 33 % the arm-A scintillator wall
    measures, the crossing does not care and the vertex is displaced.

    Both are computed here on ONE sample so the contrast is not a sample
    difference.  The scan is applied to every arm at once; a per-arm scan is a
    later refinement and is not what this question needs.

    **THE CIRCULARITY, AND WHERE THE HONEST VERSION IS.**  This sample was
    selected at ``dca_worst < dca``, and that cut was evaluated at s = 1.  So
    the sample is enriched in tracks that point well AT THE CURRENT SCALE and
    the scan is biased toward finding its minimum at s = 1 whatever the truth
    is.  The size of the bias is measurable -- run the scan at two leg cuts and
    watch the minimum move -- but it is not removable here.  The version with
    no such selection is the focus scan inside :func:`pointing`, which runs on
    every gated track with no pointing cut at all: read ``scale_focus.csv`` for
    the angle scale and this table for what a scale error does to the VERTEX.
    """
    d = P[(P.dca_worst < dca) & (~P.mixed)].copy()
    p1 = d[['p0_x_1', 'p0_y_1', 'p0_z_1']].to_numpy(float)
    p2 = d[['p0_x_2', 'p0_y_2', 'p0_z_2']].to_numpy(float)
    D1 = d[['d_x_1', 'd_y_1', 'd_z_1']].to_numpy(float)
    D2 = d[['d_x_2', 'd_y_2', 'd_z_2']].to_numpy(float)
    a1, a2 = d.arm1.to_numpy(), d.arm2.to_numpy()
    e = max(VL.check_rescale_identity(a1, D1), VL.check_rescale_identity(a2, D2))
    print(f'   rescale identity at f=1: max |dd| = {e:.2e}')
    rows = []
    for s in factors:
        n1 = VL.rescale_dirs(a1, D1, s)
        n2 = VL.rescale_dirs(a2, D2, s)
        cx, cz, dy, cy, sp, sx, tx = VL.cross_xz(p1, n1, p2, n2)
        V, sep, _, _ = VL.dca_3d(p1, n1, p2, n2)
        vrxz = np.hypot(cx, cz)
        vr = np.hypot(V[:, 0], V[:, 2])
        e1 = _miss_from(p1, n1)
        e2 = _miss_from(p2, n2)
        for topo in CLASSES:
            m = (d.topo == topo).to_numpy()
            rows.append(dict(scale=s, topology=topo, n=int(m.sum()),
                             leg_dca_med=float(np.nanmedian(
                                 np.abs(np.r_[e1[m], e2[m]]))),
                             v_r_xz_med=float(np.nanmedian(vrxz[m])),
                             v_r_med=float(np.nanmedian(vr[m])),
                             f_vrxz_10=float(np.nanmean(vrxz[m] < 10)),
                             f_vr_10=float(np.nanmean(vr[m] < 10))))
        # the band crossing on the SAME tracks, per arm
        for arm, g in T.groupby('arm'):
            if arm not in VL.U_HAT or len(g) < 500:
                continue
            u, w = VL.U_HAT[arm], VL.W_HAT[arm]
            pu = g.x_local.to_numpy(float)
            tx_ = g.tanx.to_numpy(float) * s
            ok = np.isfinite(pu) & np.isfinite(tx_)
            if ok.sum() < 500:
                continue
            sl, ic = _robust_line(pu[ok], tx_[ok])
            rows.append(dict(scale=s, topology=f'band_{arm}', n=int(ok.sum()),
                             band_x0=float(-ic / sl) if sl else np.nan,
                             band_slope_x_d=float(sl * VL.D_PERP_MM)))
    return pd.DataFrame(rows)


def _miss_from(p, d):
    n = np.hypot(d[:, 0], d[:, 2])
    with np.errstate(divide='ignore', invalid='ignore'):
        return (p[:, 0] * d[:, 2] - p[:, 2] * d[:, 0]) / np.where(n > 1e-12, n, np.nan)


# --------------------------------------------------------------------------- #
# 6 -- what the y plane costs
# --------------------------------------------------------------------------- #
def ybudget(P: pd.DataFrame, dca: float = DCA_PUB) -> pd.DataFrame:
    """How much of ``v_r``'s excess over ``v_r_xz`` is the y information.

    The 3D closest approach is free to slide both tracks along themselves to
    reduce a y mismatch, and every millimetre it slides moves the vertex
    transversely as well.  So a y plane that disagrees by 150 mm does not just
    give a wrong ``vy`` -- it drags ``vx`` and ``vz`` off the answer the
    transverse information alone would have given.  Measured by comparing the
    two on the same pairs, and by conditioning on |dy| to show the mechanism.
    """
    d = P[(P.dca_worst < dca) & (~P.mixed)].copy()
    d['dy_abs'] = np.abs(d.dy_cross)
    rows = []
    for topo, g in d.groupby('topo'):
        bins = [0, 25, 50, 100, 200, 400, np.inf]
        lab = ['0-25', '25-50', '50-100', '100-200', '200-400', '>400']
        g = g.assign(b=pd.cut(g.dy_abs, bins, labels=lab))
        for b, h in g.groupby('b', observed=True):
            if len(h) < 30:
                continue
            rows.append(dict(topology=topo, dy_bin=str(b), n=len(h),
                             frac=len(h) / len(g),
                             v_r_xz_med=float(np.nanmedian(h.v_r_xz)),
                             v_r_med=float(np.nanmedian(h.v_r)),
                             drag=float(np.nanmedian(h.v_r - h.v_r_xz)),
                             sep_med=float(np.nanmedian(h.sep_mm)),
                             vy_xz_med=float(np.nanmedian(h.vy_xz))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7 -- the floor
# --------------------------------------------------------------------------- #
def _capsule_points(n, rng):
    """Uniform points in the He-3 active gas, as a cylinder of the bounding
    radius over the gas length.  The polycone tapers at both ends, so this is
    slightly LARGER than the true source -- which is the right way round for a
    floor: it cannot flatter the reconstruction."""
    r = VL.HE3_R_MAX * np.sqrt(rng.random(n))
    ph = 2 * np.pi * rng.random(n)
    y = rng.uniform(VL.HE3_Y[0], VL.HE3_Y[1], n)
    return np.c_[r * np.cos(ph), y, r * np.sin(ph)]


def floor(P: pd.DataFrame, dca: float = DCA_PUB, seed: int = 11
          ) -> pd.DataFrame:
    """Replace one or both legs with a leg that points exactly at the capsule.

    The substituted leg keeps its MEASURED impact point -- the strips are
    precise and are not in question -- and gets the direction from a random
    point in the capsule to that impact point.  So the substitution changes the
    angle and nothing else, which is the quantity under suspicion.

    Three samples come out of it and the ordering is the result:

      ``both``   perfect angles.  The residual width is the source size folded
                 through the crossing geometry -- the FLOOR, i.e. the best any
                 reconstruction of this apparatus could do with this estimator.
      ``one``    one leg perfect, one measured.  Half the error, and it says
                 whether the two legs contribute alike.
      ``none``   the data.
    """
    rng = np.random.default_rng(seed)
    d = P[(P.dca_worst < dca) & (~P.mixed)].copy()
    p1 = d[['p0_x_1', 'p0_y_1', 'p0_z_1']].to_numpy(float)
    p2 = d[['p0_x_2', 'p0_y_2', 'p0_z_2']].to_numpy(float)
    D1 = d[['d_x_1', 'd_y_1', 'd_z_1']].to_numpy(float)
    D2 = d[['d_x_2', 'd_y_2', 'd_z_2']].to_numpy(float)
    S = _capsule_points(len(d), rng)
    I1 = S - p1
    I1 /= np.linalg.norm(I1, axis=1)[:, None]
    I2 = S - p2
    I2 /= np.linalg.norm(I2, axis=1)[:, None]
    variants = dict(none=(D1, D2), one=(I1, D2), both=(I1, I2))
    rows = []
    for name, (n1, n2) in variants.items():
        cx, cz, dy, cy, sp, sx, tx = VL.cross_xz(p1, n1, p2, n2)
        V, sep, _, _ = VL.dca_3d(p1, n1, p2, n2)
        vrxz = np.hypot(cx, cz)
        vr = np.hypot(V[:, 0], V[:, 2])
        for topo in CLASSES:
            m = (d.topo == topo).to_numpy()
            rows.append(dict(
                variant=name, topology=topo, n=int(m.sum()),
                v_r_xz_med=float(np.nanmedian(vrxz[m])),
                v_r_xz_p90=float(np.nanpercentile(vrxz[m], 90)),
                v_r_med=float(np.nanmedian(vr[m])),
                f_vrxz_10=float(np.nanmean(vrxz[m] < 10)),
                f_vr_10=float(np.nanmean(vr[m] < 10)),
                abs_dy_med=float(np.nanmedian(np.abs(dy[m]))),
                sep_med=float(np.nanmedian(sep[m]))))
    return pd.DataFrame(rows)



def yband(T: pd.DataFrame) -> pd.DataFrame:
    """The pointing band in BOTH views, per chamber, on the pair sample.

    A track from a point source at the perpendicular foot has
    ``tan = (u - u0)/d_perp``, so the band slope times ``d_perp`` is 1 for a
    point source and 0 for a sample carrying no pointing at all.  Fitted here
    in the in-plane view and in the y view on the same tracks.

    **Only the y row is a measurement.**  The sample is selected on
    ``dca_axis_mm``, which is the miss distance IN THE XZ PROJECTION -- it is
    computed from the in-plane angle alone and is blind to the y slope
    (`build_tracks.pointing`).  So the x band is pinned near 1 by that cut and
    says nothing, while the y band is uncut and says everything.  The x row is
    printed anyway, as the demonstration that the cut does what is claimed.

    The irreducible column is the other half of the statement: the capsule is
    80 mm long along y, so even a perfect y angle carries a spread of
    ``L/sqrt(12)`` at the lever arm and there is no y image to be had.
    """
    rows = []
    for arm, g in T.groupby('arm'):
        if len(g) < 1000:
            continue
        r = dict(arm=arm, n=len(g))
        for view, pos, tan in (('x', g.x_local, g.tanx),
                               ('y', g.y_local, g.tany)):
            u, t = pos.to_numpy(float), tan.to_numpy(float)
            m = np.isfinite(u) & np.isfinite(t)
            sl, ic = _robust_line(u[m], t[m])
            r[f'{view}_slope'] = float(sl * VL.D_PERP_MM)
            r[f'{view}_x0'] = float(-ic / sl) if sl else np.nan
        r['y_irreducible_mm'] = float(
            (VL.HE3_Y[1] - VL.HE3_Y[0]) / np.sqrt(12))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 8 -- is there any COINCIDENCE information in the vertex at all?
# --------------------------------------------------------------------------- #
def lift(P: pd.DataFrame, dca: float = DCA_PUB,
         cuts=(5, 10, 20, 30, 50, 80)) -> pd.DataFrame:
    """Does cutting on the vertex enrich real pairs over mixed ones?

    Everything above is about RESOLUTION.  This is the other question, and it
    is the one the physics wants: even at 25 mm, does a tight vertex pick out
    pairs that shared a trigger?  It would if the two legs of a real pair came
    from ONE decay, because then they share a point and the mixed pairs do not.

    The comparison is of FRACTIONS, not counts -- after a leg cut the real and
    mixed samples are no longer the same size (the mixing is matched at build
    time, at the build's looser ceiling) -- so the statistic is

        lift = [N_real(vertex cut) / N_real] / [N_mixed(vertex cut) / N_mixed]

    which is 1 for no information and is what `source_imaging.vertex_summary`
    computes at one fixed cut.  The error is binomial on the real numerator,
    which dominates.

    A lift of 1 here is NOT the same statement as "the imaging failed".  Event
    mixing moves the trigger, not the capsule: both legs of a mixed pair still
    came out of the same 10 mm source, so a perfect vertex detector would put
    mixed pairs on the capsule too.  What a lift of 1 says is narrower and more
    useful -- the vertex carries no information about whether the two legs are
    the same event, so it cannot be used as a pair selection.
    """
    d = P[P.dca_worst < dca]
    rows = []
    for topo, g in d.groupby('topo'):
        r, m = g[~g.mixed], g[g.mixed]
        if len(r) < 100 or len(m) < 100:
            continue
        for c in cuts:
            for var in ('v_r_xz', 'v_r'):
                kr = int((r[var] < c).sum())
                km = int((m[var] < c).sum())
                fr, fm = kr / len(r), km / len(m)
                err = np.sqrt(max(kr, 1)) / len(r)
                rows.append(dict(
                    topology=topo, variable=var, cut_mm=c,
                    n_real=len(r), n_mixed=len(m), k_real=kr, k_mixed=km,
                    frac_real=fr, frac_mixed=fm,
                    lift=fr / fm if fm > 0 else np.nan,
                    sigma=(fr - fm) / err if err > 0 else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 9 -- resolution or background?  the scintillator-confirmed legs
# --------------------------------------------------------------------------- #
def _scint_A(runs, root: Path) -> pd.DataFrame:
    """Per-track arm-A scintillator confirmation, from `det_a_scint`.

    The join key is ``(run, subrun, event_id, dca_axis_mm)``.  ``event_id``
    alone is NOT enough -- 17 % of arm-A triggers hold more than one gated
    track and joining on the trigger would confirm all of them because one of
    them matched.  ``dca_axis_mm`` is a continuous double computed from the
    same line in the same pass, so it identifies the track exactly; the
    duplicate count is printed rather than assumed away.
    """
    out = []
    for r in runs:
        p = root / f'scint_{r}.parquet'
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=['run', 'subrun', 'event_id',
                                        'dca_axis_mm', 'match_wall',
                                        'match_plas'])
        out.append(d)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d['jk'] = (d.run + ':' + d.subrun + ':' + d.event_id.astype(str) + ':'
               + d.dca_axis_mm.round(9).astype(str))
    n0 = len(d)
    d = d.drop_duplicates('jk')
    print(f'   scint tracks {n0:,} -> {len(d):,} unique join keys '
          f'({n0 - len(d)} ambiguous, dropped)')
    d['confirmed'] = d.match_wall.astype(bool) & d.match_plas.astype(bool)
    return d[['jk', 'confirmed']]


def scint_purity(P: pd.DataFrame, root: Path, dca: float = DCA_PUB
                 ) -> pd.DataFrame:
    """Does CONFIRMING a leg sharpen the vertex, at a fixed pointing cut?

    The leg scan says a tighter ``dca`` cut sharpens the vertex.  That is not
    the same statement as "the sample is dirty": a cut on pointing sharpens the
    vertex even on a perfectly pure sample, because it selects the tracks whose
    angle happened to come out well.  The question this answers is the other
    one -- **at a FIXED pointing cut, is a confirmed leg better than an
    unconfirmed one?**  If it is, what is left after the cut is background and
    the fix is purity.  If it is not, what is left is resolution and no
    selection will fix it.

    Arm A only, because `det_a_scint` is arm A only.  So the classes here are
    A-A (both legs confirmable), A-C, A-D and A-B (one leg confirmable).
    """
    d = P[(P.dca_worst < dca) & (~P.mixed)].copy()
    runs = sorted(d.run.unique())
    S = _scint_A(runs, root)
    if S.empty:
        print('   det_a_scint tracks absent -- skipping')
        return pd.DataFrame()
    key = dict(zip(S.jk, S.confirmed))
    for leg in (1, 2):
        jk = (d.run + ':' + d[f'key{leg}'].str.split(':').str[0] + ':'
              + d[f'key{leg}'].str.split(':').str[1] + ':'
              + d[f'dca_axis_mm_{leg}'].round(9).astype(str))
        isA = d[f'arm{leg}'] == 'A'
        d[f'conf_{leg}'] = np.where(isA, jk.map(key).fillna(False), np.nan)
    rows = []
    for pair, g in d.groupby('pair'):
        if 'A' not in pair:
            continue
        both = pair == 'A-A'
        c = (g.conf_1.astype(float) + g.conf_2.astype(float)) if both \
            else g[['conf_1', 'conf_2']].max(axis=1).astype(float)
        for lab, m in (('confirmed', c >= (2 if both else 1)),
                       ('not confirmed', c == 0)):
            h = g[m.fillna(False)]
            if len(h) < 50:
                continue
            rows.append(dict(
                pair=pair, leg_state=lab, n=len(h),
                leg_dca_med=float(np.nanmedian(
                    np.r_[h.dca_axis_mm_1, h.dca_axis_mm_2])),
                v_r_xz_med=float(np.nanmedian(h.v_r_xz)),
                v_r_med=float(np.nanmedian(h.v_r)),
                f_vrxz_10=float(np.mean(h.v_r_xz < 10)),
                abs_dy_med=float(np.nanmedian(np.abs(h.dy_cross))),
                sin_psi_med=float(np.nanmedian(h.sin_psi_xz)),
                amp_med=float(np.nanmedian(1.0 / h.sin_psi_xz))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--include-pre-access', action='store_true')
    ap.add_argument('--only', default='',
                    help='comma list: classes,decomposition,leg_scan,'
                         'pointing,scale,ybudget,yband,floor,lift,'
                         'scint_purity')
    a = ap.parse_args()
    want = set(x.strip() for x in a.only.split(',') if x.strip())

    od = paths.out('pair_vertex')
    P = pd.read_parquet(od / 'pairs_vertex.parquet')
    T = pd.read_parquet(od / 'tracks_vertex.parquet')
    print(f'{len(P):,} pair rows, {len(T):,} track rows from {od}\n')

    def run(name, fn):
        if want and name not in want:
            return
        print(f'-- {name}')
        out = fn()
        if isinstance(out, tuple):
            for sub, df in out:
                df.to_csv(od / f'{sub}.csv', index=False)
                print(f'   wrote {sub}.csv  ({len(df)} rows)')
        else:
            out.to_csv(od / f'{name}.csv', index=False)
            print(out.to_string(index=False, float_format=lambda x: f'{x:9.3f}'))
        print()

    run('classes', lambda: classes(P))
    run('decomposition', lambda: decomposition(P))
    run('leg_scan', lambda: leg_scan(P))
    if not want or 'pointing' in want:
        print('-- pointing (streams every stage-3 file)')
        R, H, Fo = pointing(paths.require(Path(a.src), 'stage-3 tracks'),
                            a.jobs, a.include_pre_access, a.seed)
        S = pointing_summary(R, H)
        R.to_csv(od / 'pointing_per_run.csv', index=False)
        H.to_csv(od / 'pointing_hist.csv', index=False)
        S.to_csv(od / 'pointing.csv', index=False)
        Fo.to_csv(od / 'scale_focus.csv', index=False)
        print(S.to_string(index=False, float_format=lambda x: f'{x:9.4f}'))
        print()
        print('  focus scan (every gated track, NO pointing cut) '
              '-- median miss [mm]:')
        print(Fo.pivot_table(index='scale', columns='arm', values='med_dca')
              .to_string(float_format=lambda x: f'{x:8.2f}'))
        print()
    if not want or 'scale' in want:
        print('-- scale')
        out = pd.concat([scale_scan(P, T, dca=c).assign(leg_cut_mm=c)
                         for c in (DCA_PUB, 60.0)], ignore_index=True)
        out.to_csv(od / 'scale.csv', index=False)
        print(out[out.topology.isin(CLASSES)].pivot_table(
            index='scale', columns=['leg_cut_mm', 'topology'],
            values='v_r_xz_med').to_string(float_format=lambda x: f'{x:8.2f}'))
        print()
    run('ybudget', lambda: ybudget(P))
    run('yband', lambda: yband(T))
    run('floor', lambda: floor(P))
    run('lift', lambda: lift(P))
    if not want or 'scint_purity' in want:
        print('-- scint_purity')
        out = scint_purity(P, paths.spell('out', 'det_a_scint', 'tracks'))
        if len(out):
            out.to_csv(od / 'scint_purity.csv', index=False)
            print(out.to_string(index=False,
                                float_format=lambda x: f'{x:9.3f}'))
        print()

    json.dump(dict(schema='athens26/pair_vertex_diag/1', dca_pub=DCA_PUB,
                   src=a.src, seed=a.seed),
              open(od / 'diagnostics.meta.json', 'w'), indent=1)
    print(f'wrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
