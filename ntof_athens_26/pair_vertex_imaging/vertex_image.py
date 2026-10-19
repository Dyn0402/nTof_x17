#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vertex_image.py -- the pair vertices as an IMAGE: where two tracks of one
trigger meet, in 3D, against a sample with no source in it.

`vertex_lab` / `diagnostics` asked why the pair vertex is not a 10 mm image.
This asks the complementary question: **is it a blurred image of the capsule
at all, and how blurred?**  A blur is a measurement, not a failure, provided
the blob is centred where the capsule is and is not what the chambers and the
selection would draw on their own.  Both of those need a null.

THE NULL.  Every track keeps its MEASURED impact point, and its direction
(both tans, jointly) is replaced by the direction of another track of the same
chamber in the same run.  Every marginal survives -- the impact points, the
angular distribution, the acceptance, the trigger multiplicity, which tracks
share a trigger -- and the only thing destroyed is the correlation between
where a track landed and which way it was going, i.e. "this track came from
the capsule".  The null is then pushed through the IDENTICAL selection and
pairing: the pointing cut is re-evaluated on the shuffled directions, so the
null is what this selection makes of chambers that see no source.  Two
independent shuffles are kept (``variant`` 1, 2) so the null carries twice the
statistics of the data.

This is a different null from the event-mixed one `pair_qa` carries, and the
difference matters here: event mixing keeps each track's own direction, so both
legs of a mixed pair still came out of the capsule and a mixed pair images it
exactly as well as a real one.  For imaging the question is "is there a source",
and only a null with the pointing removed answers it.

THE IMAGE IS FITTED ONE COORDINATE AT A TIME, and this is forced by the data,
not chosen.  In a perpendicular pair the crossing's x is set by the A or C leg
and its z by the D leg, so the two coordinates are two different chambers'
pointing.  x shows a clean excess over the null peaked at the capsule; z shows
chamber D's strip-plane structure (narrow spikes that the null only partly
reproduces) and no excess at the capsule at all.  A 2D fit with a free z term
was the first version and it chased that structure to z = +45 mm with
sigma_z = 65 mm -- a model absorbing a background mismatch, not an image.  So
x is fitted with a free centre, and z is TESTED: centre fixed at the capsule,
and the question is only whether any source fraction is wanted there.

TWO CENTRES FOR THE CUT.  ``dca_axis_mm`` is the miss distance from the
nominal beam axis (x = z = 0), but the capsule is measured at
X = -9.3, Z = -3.5 mm (`imaging_campaign`, 33 runs, +-0.3 / +-0.8 mm).  A tight
axis-centred cut therefore selects tracks through a disc that only partly
overlaps the source and drags the image toward the axis.  So both misses are
stored -- ``e*`` about the axis, ``ec*`` about the measured capsule -- and the
cut is scanned in both.

CHAMBER B IS NOT BUILT.  No field-shaping rings, no usable angle.

    python -m pair_vertex_imaging.vertex_image --jobs 8        # build + measure
    python -m pair_vertex_imaging.vertex_image --derive-only   # measure, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
from pair_vertex_imaging import vertex_lab as VL  # noqa: E402

SCHEMA = 'athens26/pair_vertex_image/2'
ARMS_IMG = ('A', 'C', 'D')

#: Leg ceiling at build time, about the beam axis [mm].  Wide enough that the
#: 64 mm image window is never truncated by it, so "150" in every table means
#: "no pointing cut" as far as the image is concerned.
BUILD_CEIL_MM = 150.0
N_SHUFFLE = 2
CUTS = (150.0, 60.0, 30.0, 20.0, 10.0)
CLASSES = ('perpendicular', 'intra', 'opposing')

#: The profile window and binning used by the fits [mm].
WIN_MM = 64.0
BIN_MM = 2.0
SUPERSAMPLE = 4
#: Half-width of the model-free "is there an excess at the capsule" band [mm].
BAND_MM = 15.0

#: Which chamber sets which coordinate of a perpendicular crossing.
COORD_CHAMBER = {('A-D', 'x'): 'A', ('C-D', 'x'): 'C',
                 ('A-D', 'z'): 'D', ('C-D', 'z'): 'D',
                 ('perpendicular', 'x'): 'A+C', ('perpendicular', 'z'): 'D'}

TRACK_COLS = ['event_id', 'arm', 'gated', 'angle_calibrated',
              'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'dca_axis_mm']

F32 = ('e1', 'e2', 'ec1', 'ec2', 'vx', 'vy', 'vz', 'sep_mm', 'vx_xz', 'vz_xz',
       'vy_xz', 'dy_cross', 'sin_psi_xz', 's_xz', 't_xz', 'open_deg')


# --------------------------------------------------------------------------- #
# the capsule
# --------------------------------------------------------------------------- #
def capsule_centre() -> tuple[float, float]:
    """Measured (X, Z) of the capsule from the single-track band crossing.

    Read from `imaging_campaign`'s verdict rather than typed in, so a re-run of
    the single-track imaging moves this analysis with it.
    """
    p = paths.spell('out', 'imaging_campaign', 'campaign_imaging.meta.json')
    v = json.loads(Path(p).read_text())['verdict']
    return float(v['x_source_mm']), float(v['z_D_mm'])


def capsule_profile():
    """The He-3 active gas polycone, (y, r) [mm], from the geometry module.

    Loaded from its FILE, not as ``ntof_tracking.reco.geometry``: that
    package's ``__init__`` imports the July-beam io and raises on any machine
    without the July data tree, and a capsule outline does not need it.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_ntof_geometry', Path(REPO) / 'ntof_tracking' / 'reco' / 'geometry.py')
    G = sys.modules.get(spec.name)
    if G is None:
        G = importlib.util.module_from_spec(spec)
        # registered BEFORE exec: geometry.py's dataclasses look their module
        # up in sys.modules while the class body is being processed
        sys.modules[spec.name] = G
        spec.loader.exec_module(G)
    return np.asarray(G.HE3_GAS_Y, float), np.asarray(G.HE3_GAS_R, float)


def capsule_y_centroid() -> float:
    """Volume centroid of the gas along the beam [mm] -- where a uniform
    capture density puts the mean vertex y."""
    y0, r0 = capsule_profile()
    y = np.linspace(y0[0], y0[-1], 4001)
    r = np.interp(y, y0, r0)
    return float((y * r * r).sum() / (r * r).sum())


def source_profile_1d(dx=0.01):
    """The gas projected onto ONE transverse coordinate: (u, rho(u)).

    A uniform capture density in the polycone, integrated over the beam
    direction and over the other transverse coordinate, is
    ``rho(u) = sum_y 2 sqrt(R(y)^2 - u^2)`` -- a rounded profile 20 mm wide
    at the base, not a box, which is what a 1D slice of the image should be
    compared with.
    """
    y0, r0 = capsule_profile()
    y = np.linspace(y0[0], y0[-1], 4001)
    R = np.interp(y, y0, r0)
    dy = y[1] - y[0]
    u = np.arange(-float(R.max()) - 0.5, float(R.max()) + 0.5 + dx, dx)
    rho = (2 * np.sqrt(np.clip(R[None, :] ** 2 - u[:, None] ** 2, 0, None))
           ).sum(1) * dy
    return u, rho


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def tans(arm: str, D: np.ndarray):
    u, w = VL.U_HAT[arm], VL.W_HAT[arm]
    du, dv, dw = D @ u, D @ VL.V_HAT, D @ w
    with np.errstate(divide='ignore', invalid='ignore'):
        return du / dw, dv / dw, np.sign(dw)


def rebuild(arm: str, tx, ty, sgn) -> np.ndarray:
    """Unit direction from the chamber-frame tans.  ``sgn`` is sign(d.w) and
    must be carried, or the rebuilt track is reflected (see
    `vertex_lab.rescale_dirs`)."""
    u, w = VL.U_HAT[arm], VL.W_HAT[arm]
    nd = tx[:, None] * u + ty[:, None] * VL.V_HAT + w[None, :]
    nd = nd * sgn[:, None]
    return nd / np.linalg.norm(nd, axis=1)[:, None]


def signed_miss(P: np.ndarray, D: np.ndarray, c=(0.0, 0.0)) -> np.ndarray:
    """Signed transverse distance of each line from the vertical line through
    ``(x, z) = c`` [mm].  About the origin its absolute value is the stage-3
    ``dca_axis_mm``; that is asserted on every run."""
    n = np.hypot(D[:, 0], D[:, 2])
    with np.errstate(divide='ignore', invalid='ignore'):
        return (((P[:, 0] - c[0]) * D[:, 2] - (P[:, 2] - c[1]) * D[:, 0])
                / np.where(n > 1e-12, n, np.nan))


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def one_run(run: str, subruns, src: str, ceil: float, cap, seed: int) -> tuple:
    try:
        out = []
        for sub in subruns:
            p = Path(src) / f'tracks_{run}_{sub}.parquet'
            out.append(pd.read_parquet(p, columns=TRACK_COLS).assign(subrun=sub))
        t = pd.concat(out, ignore_index=True)
        t = t[t.gated & t.angle_calibrated & t.arm.isin(ARMS_IMG)]
        t = t.reset_index(drop=True)
        if t.empty:
            return run, None, 'no angle-calibrated A/C/D tracks'
        t['key'] = t.subrun + ':' + t.event_id.astype(str)
        P = t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        D = t[['d_x', 'd_y', 'd_z']].to_numpy(float)
        arms = t.arm.to_numpy()

        e = signed_miss(P, D)
        ok = np.isfinite(e)
        resid = float(np.max(np.abs(np.abs(e[ok])
                                    - t.dca_axis_mm.to_numpy(float)[ok])))
        if resid > 1e-6:
            raise AssertionError(f'signed miss != dca_axis_mm by {resid:.3g} mm')

        TX = np.full(len(t), np.nan)
        TY = np.full(len(t), np.nan)
        SG = np.zeros(len(t))
        for arm in ARMS_IMG:
            m = arms == arm
            if m.any():
                TX[m], TY[m], SG[m] = tans(arm, D[m])
                back = rebuild(arm, TX[m], TY[m], SG[m])
                good = np.isfinite(back).all(axis=1)
                err = float(np.max(np.abs(back[good] - D[m][good]))) if good.any() else 0
                if err > 1e-9:
                    raise AssertionError(f'{arm}: tan rebuild is not the identity '
                                         f'({err:.3g})')

        rng = np.random.default_rng(seed)
        frames = []
        for v in range(N_SHUFFLE + 1):
            if v == 0:
                Dv = D
            else:
                Dv = np.full_like(D, np.nan)
                for arm in ARMS_IMG:
                    idx = np.flatnonzero((arms == arm) & np.isfinite(TX)
                                         & np.isfinite(TY))
                    if len(idx) < 2:
                        continue
                    perm = rng.permutation(idx)
                    Dv[idx] = rebuild(arm, TX[perm], TY[perm], SG[idx])
            ev = signed_miss(P, Dv)
            ec = signed_miss(P, Dv, cap)
            keep = np.isfinite(ev) & (np.abs(ev) < ceil)
            pr = VL.pairs_real(t.loc[keep, ['key']])
            if pr.empty:
                continue
            i, j = pr.i.to_numpy(), pr.j.to_numpy()
            swap = arms[i] > arms[j]
            i, j = np.where(swap, j, i), np.where(swap, i, j)
            p1, p2, d1, d2 = P[i], P[j], Dv[i], Dv[j]
            V, sep, _, _ = VL.dca_3d(p1, d1, p2, d2)
            cx, cz, dy, cy, sp, sxz, txz = VL.cross_xz(p1, d1, p2, d2)
            dot = np.einsum('ij,ij->i', d1, d2).clip(-1, 1)
            f = pd.DataFrame(dict(
                variant=np.int8(v), arm1=arms[i], arm2=arms[j],
                e1=ev[i], e2=ev[j], ec1=ec[i], ec2=ec[j],
                vx=V[:, 0], vy=V[:, 1], vz=V[:, 2], sep_mm=sep,
                vx_xz=cx, vz_xz=cz, vy_xz=cy, dy_cross=dy,
                sin_psi_xz=sp, s_xz=sxz, t_xz=txz,
                open_deg=np.degrees(np.arccos(dot))))
            frames.append(f)
        if not frames:
            return run, None, 'no pairs'
        d = pd.concat(frames, ignore_index=True)
        for c in F32:
            d[c] = d[c].astype(np.float32)
        d['run'] = run
        return run, d, ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def build(src: Path, jobs: int, include_pre_access: bool, seed: int, cap
          ) -> tuple[pd.DataFrame, dict]:
    rs = VL.discover(src, include_pre_access)
    print(f'{len(rs)} run(s) from {src}   leg ceiling {BUILD_CEIL_MM:.0f} mm   '
          f'capsule at X={cap[0]:+.2f} Z={cap[1]:+.2f} mm\n')
    out, bad = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, s, str(src), BUILD_CEIL_MM, cap,
                          seed + k): r
                for k, (r, s) in enumerate(sorted(rs.items()))}
        for f in as_completed(futs):
            run, d, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            out.append(d)
            print(f'  {run:<10} ok   {int((d.variant == 0).sum()):>9,} data '
                  f'{int((d.variant > 0).sum()):>9,} null pairs', flush=True)
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    for c in ('arm1', 'arm2', 'run'):
        if c in d:
            d[c] = d[c].astype('category')
    return d, bad


# --------------------------------------------------------------------------- #
# measure
# --------------------------------------------------------------------------- #
def load(od: Path) -> pd.DataFrame:
    d = pd.read_parquet(od / 'pairs_image.parquet')
    a1, a2 = d.arm1.astype(str).to_numpy(), d.arm2.astype(str).to_numpy()
    pair = pd.Series(np.char.add(np.char.add(a1.astype(str), '-'), a2.astype(str)))
    topo = {p: VL.topology(p[0], p[2]) for p in pair.unique()}
    d['pair'] = pair.astype('category').to_numpy()
    d['topo'] = pair.map(topo).astype('category').to_numpy()
    d['null'] = d.variant > 0
    d['worst_axis'] = np.maximum(np.abs(d.e1), np.abs(d.e2))
    d['worst_cap'] = np.maximum(np.abs(d.ec1), np.abs(d.ec2))
    return d


def select(d: pd.DataFrame, sel: str, cut: float, centre: str = 'axis'
           ) -> pd.DataFrame:
    g = d[(d.topo == sel) if sel in CLASSES else (d.pair == sel)]
    return g[g['worst_axis' if centre == 'axis' else 'worst_cap'] < cut]


def verify(d: pd.DataFrame, od: Path) -> pd.DataFrame:
    """The data variant at the published 30 mm cut must be `vertex_lab`'s
    real-pair sample, arm pair for arm pair."""
    p = od / 'pairs_vertex.parquet'
    if not p.exists():
        return pd.DataFrame()
    a = pd.read_parquet(p, columns=['pair', 'mixed', 'dca_worst'])
    a = a[(~a.mixed) & (a.dca_worst < 30) & ~a.pair.str.contains('B')]
    a = a.groupby('pair').size().rename('n_vertex_lab')
    b = d[(d.variant == 0) & (d.worst_axis < 30)].groupby(
        'pair', observed=True).size().rename('n_image')
    b.index = b.index.astype(str)
    c = pd.concat([a, b], axis=1).fillna(0).astype(int)
    c['delta'] = c.n_image - c.n_vertex_lab
    return c.reset_index(names='pair')


def _rsig(v):
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def stats(d: pd.DataFrame, cap) -> pd.DataFrame:
    """Model-free image statistics, per class, sample, cut and estimator."""
    rows = []
    for topo in CLASSES + ('all', 'A-D', 'C-D'):
        g0 = d if topo == 'all' else (d[d.topo == topo] if topo in CLASSES
                                      else d[d.pair == topo])
        for kind, col in (('axis', 'worst_axis'), ('capsule', 'worst_cap')):
            for cut in CUTS:
                g1 = g0[g0[col] < cut]
                for null in (False, True):
                    h = g1[g1.null == null]
                    if len(h) < 30:
                        continue
                    for est, (cx, cy, cz) in (('xz', ('vx_xz', 'vy_xz', 'vz_xz')),
                                              ('3d', ('vx', 'vy', 'vz'))):
                        x = h[cx].to_numpy(float)
                        y = h[cy].to_numpy(float)
                        z = h[cz].to_numpy(float)
                        rc = np.hypot(x - cap[0], z - cap[1])
                        sx, sz = _rsig(x), _rsig(z)
                        rows.append(dict(
                            topology=topo, cut_centre=kind, cut_mm=cut,
                            sample='null' if null else 'data', estimator=est,
                            n=len(h),
                            n_per_sample=len(h) / (N_SHUFFLE if null else 1),
                            x_med=float(np.nanmedian(x)),
                            z_med=float(np.nanmedian(z)),
                            y_med=float(np.nanmedian(y)),
                            x_med_err=1.2533 * sx / np.sqrt(len(h)),
                            z_med_err=1.2533 * sz / np.sqrt(len(h)),
                            x_rsig=sx, z_rsig=sz, y_rsig=_rsig(y),
                            f_xband=float(np.nanmean(np.abs(x - cap[0]) < BAND_MM)),
                            f_zband=float(np.nanmean(np.abs(z - cap[1]) < BAND_MM)),
                            r_cap_med=float(np.nanmedian(rc)),
                            f_cap_10=float(np.nanmean(rc < 10)),
                            f_axis_10=float(np.nanmean(np.hypot(x, z) < 10)),
                            sep_p25=float(np.nanpercentile(h.sep_mm, 25)),
                            sep_med=float(np.nanmedian(h.sep_mm)),
                            sep_p75=float(np.nanpercentile(h.sep_mm, 75)),
                            dy_abs_med=float(np.nanmedian(np.abs(h.dy_cross)))))
    return pd.DataFrame(rows)


# ---- the fit ---------------------------------------------------------------- #
def edges():
    return np.arange(-WIN_MM, WIN_MM + 0.5 * BIN_MM, BIN_MM)


class ProfileModel:
    """``f * (capsule profile (x) Gauss(sigma)) + (1 - f) * null`` in 1D.

    The source term is the gas projected onto the coordinate
    (:func:`source_profile_1d`), shifted to ``c`` and blurred; the background
    is the null through the same selection, as a fixed shape.  Normalised in
    the window, so ``f`` is the in-window fraction the source claims.
    """

    def __init__(self, data_vals: np.ndarray, null_vals: np.ndarray):
        from scipy.ndimage import gaussian_filter1d
        self.gf = gaussian_filter1d
        e = edges()
        self.n = np.histogram(data_vals[np.isfinite(data_vals)], e)[0].astype(float)
        b = np.histogram(null_vals[np.isfinite(null_vals)], e)[0].astype(float)
        B = gaussian_filter1d(b, 1.0) + 1e-9
        self.B = B / B.sum()
        self.N = self.n.sum()
        self.nb = len(self.n)
        self.fine = BIN_MM / SUPERSAMPLE
        self.xf = -WIN_MM + self.fine * (np.arange(self.nb * SUPERSAMPLE) + 0.5)
        self.u, self.rho = source_profile_1d()
        self.centres = 0.5 * (e[:-1] + e[1:])

    def source(self, c, s):
        S = np.interp(self.xf - c, self.u, self.rho, left=0.0, right=0.0)
        S = self.gf(S, s / self.fine, mode='constant', truncate=4.0)
        S = S.reshape(self.nb, SUPERSAMPLE).sum(1)
        return S / max(S.sum(), 1e-300)

    def mu(self, p):
        c, s, f = p
        return self.N * (f * self.source(c, s) + (1 - f) * self.B)

    def nll(self, p):
        c, s, f = p
        if not (0.3 < s < 80 and 0 <= f <= 1 and -WIN_MM < c < WIN_MM):
            return 1e30
        m = self.mu(p)
        return float((m - self.n * np.log(m)).sum())

    def hess_err(self, p, free):
        h = np.array([0.1, 0.1, 0.003])
        idx = [k for k in range(3) if free[k]]
        H = np.zeros((len(idx), len(idx)))
        f0 = self.nll(p)
        for a_, a in enumerate(idx):
            for b_, b in enumerate(idx):
                if b_ < a_:
                    continue
                if a == b:
                    q = np.array(p, float); q[a] += h[a]; fp = self.nll(q)
                    q[a] -= 2 * h[a]; fm = self.nll(q)
                    H[a_, a_] = (fp - 2 * f0 + fm) / h[a] ** 2
                else:
                    v = []
                    for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        q = np.array(p, float); q[a] += sa * h[a]; q[b] += sb * h[b]
                        v.append(self.nll(q))
                    H[a_, b_] = H[b_, a_] = (v[0] - v[1] - v[2] + v[3]) / (4 * h[a] * h[b])
        err = np.full(3, np.nan)
        try:
            C = np.linalg.inv(H)
            err[idx] = np.sqrt(np.clip(np.diag(C), 0, None))
        except np.linalg.LinAlgError:
            pass
        return err


def fit_profile(data_vals, null_vals, c0, fix_c=None, fix_s=None, fix_f=None,
                model_cls=None) -> dict:
    """Fit one coordinate.  Any of centre / blur / fraction may be fixed.

    ``model_cls`` swaps in a `ProfileModel` subclass (e.g. with tighter
    parameter bounds); the default is `ProfileModel` itself.
    """
    from scipy.optimize import minimize
    M = (model_cls or ProfileModel)(np.asarray(data_vals, float),
                                    np.asarray(null_vals, float))
    if M.N < 100:
        return dict(n_in_window=int(M.N))
    fixed = [fix_c, fix_s, fix_f]
    free = [v is None for v in fixed]

    def full(q):
        it = iter(q)
        return [next(it) if free[k] else fixed[k] for k in range(3)]

    starts = [(c0, 12.0, 0.3), (c0, 30.0, 0.1), (0.0, 20.0, 0.2)]
    best = None
    for s0 in starts:
        q0 = [s0[k] for k in range(3) if free[k]]
        if not q0:
            break
        r = minimize(lambda q: M.nll(full(q)), q0, method='Nelder-Mead',
                     options=dict(maxiter=3000, xatol=1e-4, fatol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    p = full(best.x) if best is not None else full([])
    err = M.hess_err(p, free)
    nll_best = M.nll(p)
    nll_none = M.nll((c0, 10.0, 0.0))
    m = M.mu(p)
    good = m > 5
    chi2 = float((((M.n - m) ** 2) / m)[good].sum())
    return dict(n_in_window=int(M.N), c=p[0], s=p[1], f=p[2],
                c_err=err[0], s_err=err[1], f_err=err[2],
                two_dnll_vs_none=2 * (nll_none - nll_best),
                chi2=chi2, ndf=int(good.sum()) - int(sum(free)))


FIT_SELECTIONS = ('A-D', 'C-D', 'perpendicular')
FIT_CUTS = (150.0, 60.0)


def fits(d: pd.DataFrame, cap) -> tuple[pd.DataFrame, pd.DataFrame]:
    """x fitted with a free centre; z tested with the centre at the capsule.

    Plus, per row, the model-free number the fit must agree with: the fraction
    of vertices within +-15 mm of the capsule coordinate, data against null.
    """
    rows = []
    for sel in FIT_SELECTIONS:
        for cut in FIT_CUTS:
            g = select(d, sel, cut)
            for coord, col, c0 in (('x', 'vx_xz', cap[0]), ('z', 'vz_xz', cap[1])):
                a = g.loc[~g.null, col].to_numpy(float)
                b = g.loc[g.null, col].to_numpy(float)
                r = fit_profile(a, b, c0, fix_c=(c0 if coord == 'z' else None))
                band_d = float(np.mean(np.abs(a - c0) < BAND_MM))
                band_n = float(np.mean(np.abs(b - c0) < BAND_MM))
                rows.append(dict(selection=sel, cut_mm=cut, coord=coord,
                                 chamber=COORD_CHAMBER[(sel, coord)],
                                 centre_free=coord == 'x', capsule=c0,
                                 n_data=len(a), band_data=band_d,
                                 band_null=band_n, band_excess=band_d - band_n,
                                 **r))
                print(f'   {sel:<14} cut {cut:>4.0f} {coord}: '
                      + ', '.join(f'{k}={v:.3g}' for k, v in rows[-1].items()
                                  if isinstance(v, float)), flush=True)
    F = pd.DataFrame(rows)

    # Per run, per chamber: the x centre alone free, blur and fraction fixed
    # to that arm pair's pooled fit, against its pooled null.  A-D measures
    # chamber A and C-D chamber C, so each is compared with that chamber's own
    # single-track crossing in that run.
    ref = paths.spell('out', 'imaging_campaign', 'axis_per_run.csv')
    A = pd.read_csv(ref) if Path(ref).exists() else pd.DataFrame()
    pr = []
    for sel, chamber, refcol in (('A-D', 'A', 'x_A_mm'), ('C-D', 'C', 'x_C_mm')):
        fr = F[(F.selection == sel) & (F.cut_mm == 60.0) & (F.coord == 'x')].iloc[0]
        g = select(d, sel, 60.0)
        null = g.loc[g.null, 'vx_xz'].to_numpy(float)
        for run, h in g[~g.null].groupby('run', observed=True):
            r = fit_profile(h.vx_xz.to_numpy(float), null, cap[0],
                            fix_s=fr.s, fix_f=fr.f)
            row = dict(run=str(run), selection=sel, chamber=chamber, **r)
            if len(A) and refcol in A:
                m = A[A.run == str(run)]
                row['single_track_x'] = float(m[refcol].iloc[0]) if len(m) else np.nan
            pr.append(row)
    return F, pd.DataFrame(pr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--include-pre-access', action='store_true')
    ap.add_argument('--derive-only', action='store_true')
    a = ap.parse_args()

    od = paths.out('pair_vertex')
    cap = capsule_centre()
    if not a.derive_only:
        src = paths.require(Path(a.src), 'the stage-3 track tables')
        d, bad = build(src, a.jobs, a.include_pre_access, a.seed, cap)
        if d.empty:
            print('nothing built')
            return 1
        d.to_parquet(od / 'pairs_image.parquet', index=False)
        json.dump(dict(schema=SCHEMA, src=str(src), seed=a.seed,
                       build_ceil_mm=BUILD_CEIL_MM, n_shuffle=N_SHUFFLE,
                       arms=list(ARMS_IMG), capsule_xz=list(cap),
                       capsule_y_centroid=capsule_y_centroid(),
                       include_pre_access=a.include_pre_access,
                       n_runs=int(d.run.nunique()),
                       n_data=int((d.variant == 0).sum()),
                       n_null=int((d.variant > 0).sum()), runs_failed=bad),
                  open(od / 'pairs_image.meta.json', 'w'), indent=1)
        del d

    d = load(od)
    print(f'\n{int((~d.null).sum()):,} data pairs, {int(d.null.sum()):,} null '
          f'pairs ({N_SHUFFLE} shuffles)\n')
    V = verify(d, od)
    if len(V):
        V.to_csv(od / 'image_verify.csv', index=False)
        print('SAME SAMPLE AS vertex_lab at the 30 mm cut?')
        print(V.to_string(index=False))
        print('  all match\n' if (V.delta == 0).all() else
              '  *** MISMATCH ***\n')

    print('-- stats')
    S = stats(d, cap)
    S.to_csv(od / 'image_stats.csv', index=False)
    print(S[(S.estimator == 'xz') & (S.topology.isin(['A-D', 'C-D']))][
        ['topology', 'cut_centre', 'cut_mm', 'sample', 'n', 'x_med', 'z_med',
         'x_rsig', 'z_rsig', 'f_xband', 'f_zband', 'sep_med']].to_string(
        index=False, float_format=lambda x: f'{x:8.2f}'))
    print('\n-- fits')
    F, PR = fits(d, cap)
    F.to_csv(od / 'image_fit.csv', index=False)
    PR.to_csv(od / 'image_fit_per_run.csv', index=False)
    print(F.to_string(index=False, float_format=lambda x: f'{x:9.3f}'))
    for ch, g in PR.groupby('chamber'):
        g = g.dropna(subset=['c'])
        print(f'\nper-run x from chamber {ch}: median {g.c.median():+.2f}, '
              f'spread {g.c.std():.2f}, median err {g.c_err.median():.2f}; '
              f'single-track median {g.single_track_x.median():+.2f}, '
              f'corr {g.c.corr(g.single_track_x):+.2f}')
    json.dump(dict(schema=SCHEMA, capsule_xz=list(cap), band_mm=BAND_MM,
                   fit_window_mm=WIN_MM, fit_bin_mm=BIN_MM),
              open(od / 'image_derive.meta.json', 'w'), indent=1)
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
