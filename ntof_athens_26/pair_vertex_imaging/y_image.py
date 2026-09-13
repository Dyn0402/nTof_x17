#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
y_image.py -- the vertex along the beam: why the pair y had three peaks, and a
y image that means something.

WHERE THE THREE PEAKS CAME FROM.  The vertex y of the companion notes is the
MEAN of the two legs' y at the transverse crossing.  In an A-D or C-D pair the
A/C leg's y there is a single peak near +35 mm; the D leg's y is noise --
spikes at fixed y from D's noisy y columns, and the tan_y ~ 0 pile-up of
slopes the drift timing did not measure -- and the two are uncorrelated.
Averaging a good number with a junk one draws the junk's structure at half
scale around the good one's peak.  A-D and C-D "carried" the y distribution
only because they are 84 % of the excess near the capsule.

WHAT IS DONE ABOUT IT, in three steps:

1. THE y PLANE GETS ITS OWN CLEANING, the mirror of the x plane's:
   ``y_slope_reliable`` (|tan_y| >= 0.08, below which `wft` measures no
   slope) and noisy y columns found per run from the data (same 2 mm / 4x
   running-median rule as x).  Five flag bits per track now: x slope measured
   (1), x noisy column (2), own scintillators (4), y slope measured (8),
   y noisy column (16).  The null is shuffled within all 32 strata.

2. EACH LEG'S y IS KEPT SEPARATELY, as a function of the y angle scale.  A
   leg's y at a transverse point is ``p0_y + tan_y * dw``, with ``dw`` the
   displacement along that chamber's drift axis to the point.  The transverse
   crossing does not depend on tan_y, so ``dw`` is fixed and y can be
   re-evaluated under any y scale at analysis time.  A pair's y is then taken
   from the legs whose y is informative: the A or C leg in A-D and C-D, both
   D legs in D-D.

3. THE y ANGLE SCALE IS MEASURED, not assumed.  `k_arm` calibrates tan_x;
   tan_y's scale cannot be calibrated against the capsule in the usual way
   because the source is 80 mm long along y (STATUS.md, "kY cannot be
   fitted").  But the per-chamber y BAND still has a slope: a track from a
   source at y_s has ``tan_y * dw = s (y_s - p0_y)`` with s the scale error,
   so a robust line of ``tan_y * dw`` against ``p0_y`` has slope ``-s``
   and crosses zero at ``y_s`` whatever s is -- the y band crossing, which is
   the scale-free ensemble y.  The scale itself is measured two ways, and
   THE TWO DISAGREE IN OPPOSITE DIRECTIONS, as their biases predict: the band
   slope comes out below 1 (0.58-0.86), pulled down by background tracks that
   flatten the line, and dividing tan_y by it WIDENS every image; the FOCUS
   scan (the s that makes the single-track y image narrowest) comes out above
   1 (1.2-1.4), pushed up because a larger s also shrinks the tan_y
   resolution term.  Neither is adopted: images are made at the scale as
   reconstructed (s = 1), and the other two are carried as a bracket.  The x
   selection (pointing < 30 mm) is blind to tan_y, so none of this is
   circular in y.

THE y SOURCE MODEL is the He-3 polycone's cross-sectional area along the beam,
pi R(y)^2, placed with its volume centroid at the fitted ``c`` and blurred by
a Gaussian, on top of the null shape.  The nominal gas centroid is at
y = +0.8 mm in the detector frame.

    python -m pair_vertex_imaging.y_image --jobs 8        # build + measure
    python -m pair_vertex_imaging.y_image --derive-only
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
from pair_vertex_imaging import vertex_image as VI  # noqa: E402
from pair_vertex_imaging import z_image as Z  # noqa: E402

SCHEMA = 'athens26/pair_vertex_y/1'
ARMS = VI.ARMS_IMG
REL, HOT, CONF, YREL, YHOT = 1, 2, 4, 8, 16
N_CODES = 32
TRACK_COLS = Z.TRACK_COLS + ['y_local', 'y_slope_reliable', 'target_y_mm']
L_MM = VL.D_PERP_MM
CUT = 60.0
WIN_X = 16.0       # transverse half-window around the capsule for y profiles
Y_WIN, Y_BIN = 250.0, 10.0
SCALE_GRID = np.round(np.arange(0.30, 1.51, 0.05), 2)


def ok_x(code):
    code = np.asarray(code)
    return ((code & REL) > 0) & ((code & HOT) == 0)


def ok_y(code):
    code = np.asarray(code)
    return ((code & YREL) > 0) & ((code & YHOT) == 0)


def ok_conf(code):
    return (np.asarray(code) & CONF) > 0


W_X = {'A': 0.0, 'C': 0.0, 'D': 1.0}
W_Z = {'A': 1.0, 'C': -1.0, 'D': 0.0}


# --------------------------------------------------------------------------- #
def one_run(run: str, subruns, src: str, seed: int) -> tuple:
    try:
        out = []
        for sub in subruns:
            p = Path(src) / f'tracks_{run}_{sub}.parquet'
            out.append(pd.read_parquet(p, columns=TRACK_COLS).assign(subrun=sub))
        t = pd.concat(out, ignore_index=True)
        t = t[t.gated & t.angle_calibrated & t.arm.isin(ARMS)].reset_index(drop=True)
        if t.empty:
            return run, None, 'no angle-calibrated A/C/D tracks'
        t['key'] = t.subrun + ':' + t.event_id.astype(str)
        P = np.array(t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), copy=True)
        D = np.array(t[['d_x', 'd_y', 'd_z']].to_numpy(float), copy=True)
        arms = t.arm.to_numpy()
        for arm in ARMS:   # the frame assumed below, checked rather than trusted
            m = arms == arm
            if m.any() and not (np.allclose(VL.W_HAT[arm], [W_X[arm], 0, W_Z[arm]])):
                raise AssertionError(f'{arm}: drift axis is not the assumed one')
        wx = np.array([W_X[a] for a in arms])
        wz = np.array([W_Z[a] for a in arms])

        e0 = VI.signed_miss(P, D)
        rel = t.x_slope_reliable.fillna(False).astype(bool).to_numpy()
        yrel = t.y_slope_reliable.fillna(False).astype(bool).to_numpy()
        conf = pd.to_numeric(t.coinc_this_arm, errors='coerce').fillna(-1).to_numpy() == 1
        xhot, _ = Z.hot_columns(t.x_local.to_numpy(float), arms, run)
        yhot, _ = Z.hot_columns(t.y_local.to_numpy(float), arms, run)
        code = (rel * REL + xhot * HOT + conf * CONF + yrel * YREL + yhot * YHOT).astype(np.int8)

        TX = np.full(len(t), np.nan)
        TY = np.full(len(t), np.nan)
        SG = np.zeros(len(t))
        for arm in ARMS:
            m = arms == arm
            if m.any():
                TX[m], TY[m], SG[m] = VI.tans(arm, D[m])

        # ---- single tracks: y at the beam axis as p0_y + tan_y * dw
        dxz2 = D[:, 0] ** 2 + D[:, 2] ** 2
        s_ax = -(P[:, 0] * D[:, 0] + P[:, 2] * D[:, 2]) / dxz2
        dw_ax = s_ax * (D[:, 0] * wx + D[:, 2] * wz)
        y_ax = P[:, 1] + TY * dw_ax
        ty_ref = t.target_y_mm.to_numpy(float)
        okc = np.isfinite(y_ax) & np.isfinite(ty_ref)
        err = float(np.max(np.abs(y_ax[okc] - ty_ref[okc]))) if okc.any() else 0.0
        if err > 1e-6:
            raise AssertionError(f'y at axis rebuild != target_y_mm by {err:.3g} mm')
        keep_t = ok_x(code) & (np.abs(e0) < 30)
        tracks = pd.DataFrame(dict(
            arm=arms[keep_t], code=code[keep_t],
            py=P[keep_t, 1].astype(np.float32), ty=TY[keep_t].astype(np.float32),
            dw=dw_ax[keep_t].astype(np.float32),
            ylocal=t.y_local.to_numpy(float)[keep_t].astype(np.float32)))
        tracks['run'] = run

        # ---- pairs
        rng = np.random.default_rng(seed)
        dirs = [D]
        for _ in range(VI.N_SHUFFLE):
            Dv = np.full_like(D, np.nan)
            for arm in ARMS:
                for c in range(N_CODES):
                    idx = np.flatnonzero((arms == arm) & (code == c)
                                         & np.isfinite(TX) & np.isfinite(TY))
                    if len(idx) < 2:
                        continue
                    perm = rng.permutation(idx)
                    Dv[idx] = VI.rebuild(arm, TX[perm], TY[perm], SG[idx])
            dirs.append(Dv)

        frames = []
        for v, Dv in enumerate(dirs):
            TYv = np.full(len(t), np.nan)
            for arm in ARMS:
                m = arms == arm
                if m.any():
                    _, TYv[m], _ = VI.tans(arm, Dv[m])
            ev = VI.signed_miss(P, Dv)
            keep = np.isfinite(ev) & (np.abs(ev) < VI.BUILD_CEIL_MM)
            pr = VL.pairs_real(t.loc[keep, ['key']])
            if pr.empty:
                continue
            i, j = pr.i.to_numpy(), pr.j.to_numpy()
            swap = arms[i] > arms[j]
            i, j = np.where(swap, j, i), np.where(swap, i, j)
            cx, cz, dy, cy, sp, _, _ = VL.cross_xz(P[i], Dv[i], P[j], Dv[j])
            dw1 = (cx - P[i, 0]) * wx[i] + (cz - P[i, 2]) * wz[i]
            dw2 = (cx - P[j, 0]) * wx[j] + (cz - P[j, 2]) * wz[j]
            y1 = P[i, 1] + TYv[i] * dw1
            y2 = P[j, 1] + TYv[j] * dw2
            if v == 0:
                good = np.isfinite(y1) & np.isfinite(cy)
                e1 = float(np.max(np.abs((y1 - y2)[good] - dy[good]))) if good.any() else 0.0
                e2 = float(np.max(np.abs((0.5 * (y1 + y2))[good] - cy[good]))) if good.any() else 0.0
                if max(e1, e2) > 1e-6:
                    raise AssertionError(f'leg y rebuild disagrees with cross_xz ({e1:.3g}, {e2:.3g})')
            frames.append(pd.DataFrame(dict(
                variant=np.int8(v), arm1=arms[i], arm2=arms[j],
                code1=code[i], code2=code[j],
                e1=ev[i].astype(np.float32), e2=ev[j].astype(np.float32),
                vx_xz=cx.astype(np.float32), vz_xz=cz.astype(np.float32),
                sin_psi_xz=sp.astype(np.float32),
                py1=P[i, 1].astype(np.float32), ty1=TYv[i].astype(np.float32),
                dw1=dw1.astype(np.float32),
                py2=P[j, 1].astype(np.float32), ty2=TYv[j].astype(np.float32),
                dw2=dw2.astype(np.float32))))
        d = pd.concat(frames, ignore_index=True)
        d['run'] = run
        return run, dict(pairs=d, tracks=tracks), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def build(src: Path, jobs: int, seed: int):
    rs = VL.discover(src, False)
    print(f'{len(rs)} run(s) from {src}\n')
    P, T, bad = [], [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, s, str(src), seed + k): r
                for k, (r, s) in enumerate(sorted(rs.items()))}
        for f in as_completed(futs):
            run, res, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            P.append(res['pairs'])
            T.append(res['tracks'])
            print(f'  {run:<10} ok   {int((res["pairs"].variant == 0).sum()):>9,} data pairs '
                  f'{len(res["tracks"]):>8,} tracks', flush=True)
    d = pd.concat(P, ignore_index=True)
    tr = pd.concat(T, ignore_index=True)
    for df in (d, tr):
        for c in ('arm1', 'arm2', 'arm', 'run'):
            if c in df:
                df[c] = df[c].astype('category')
    return d, tr, bad


# --------------------------------------------------------------------------- #
def capsule_y_profile(dy=0.05):
    """pi R(y)^2 of the gas, on a grid relative to its volume centroid."""
    y0, r0 = VI.capsule_profile()
    y = np.arange(y0[0], y0[-1] + dy, dy)
    a = np.pi * np.interp(y, y0, r0) ** 2
    cen = float((y * a).sum() / a.sum())
    return y - cen, a, cen


class YProfileModel(VI.ProfileModel):
    """The 1D profile model along the beam: gas area (x) Gauss + null, +-250 mm."""

    def __init__(self, data_vals, null_vals):
        from scipy.ndimage import gaussian_filter1d
        self.gf = gaussian_filter1d
        e = np.arange(-Y_WIN, Y_WIN + 0.5 * Y_BIN, Y_BIN)
        dv = data_vals[np.isfinite(data_vals)]
        nv = null_vals[np.isfinite(null_vals)]
        self.n = np.histogram(dv, e)[0].astype(float)
        B = gaussian_filter1d(np.histogram(nv, e)[0].astype(float), 1.0) + 1e-9
        self.B = B / B.sum()
        self.N = self.n.sum()
        self.nb = len(self.n)
        self.fine = Y_BIN / VI.SUPERSAMPLE
        self.xf = -Y_WIN + self.fine * (np.arange(self.nb * VI.SUPERSAMPLE) + 0.5)
        self.u, self.rho, _ = capsule_y_profile()
        self.centres = 0.5 * (e[:-1] + e[1:])

    def nll(self, p):
        c, s, f = p
        # blur held above half a 10 mm bin: below it the fit puts a spike on
        # one bin, which is not a measurement of anything
        if not (5.0 < s < 150 and 0 <= f <= 1 and -150 < c < 150):
            return 1e30
        m = self.mu(p)
        return float((m - self.n * np.log(m)).sum())


def fit_y(data_vals, null_vals, c0=25.0):
    data_vals = np.asarray(data_vals, float)
    null_vals = np.asarray(null_vals, float)
    base = dict(n_data=int(np.isfinite(data_vals).sum()),
                median_data=float(np.nanmedian(data_vals)) if len(data_vals) else np.nan,
                rsig_data=VI._rsig(data_vals), rsig_null=VI._rsig(null_vals))
    if base['n_data'] < 200 or np.isfinite(null_vals).sum() < 200:
        return base
    r = VI.fit_profile(data_vals, null_vals, c0, model_cls=YProfileModel)
    # blank both "nothing wanted" and a blur parked on its lower bound -- the
    # latter is a one-bin spike (seen on the D-leg y), not an image
    if r.get('f', 0) < Z.F_NONE or r.get('s', 99.0) <= 5.5:
        for k in ('c', 's', 'c_err', 's_err'):
            r[k] = np.nan
    return dict(base, **r)


# ---- the y scale ------------------------------------------------------------ #
def band_scale(tr: pd.DataFrame) -> pd.DataFrame:
    """Per arm, per y tier: the robust y band slope (-> scale s) and crossing."""
    from pair_vertex_imaging.diagnostics import _robust_line
    rows = []
    for arm in ARMS:
        g = tr[tr.arm == arm]
        for tier, m in (('x-clean', np.ones(len(g), bool)),
                        ('y-clean', ok_y(g.code)),
                        ('y-clean, confirmed', ok_y(g.code) & ok_conf(g.code))):
            h = g[m]
            if len(h) < 2000:
                continue
            q = (h.ty * h.dw).to_numpy(float)
            yy = h.py.to_numpy(float)
            ok = np.isfinite(q) & np.isfinite(yy)
            sl, ic = _robust_line(yy[ok], q[ok])
            rows.append(dict(arm=arm, tier=tier, n=int(ok.sum()), band_scale=-sl,
                             band_y0=-ic / sl if sl else np.nan))
    return pd.DataFrame(rows)


def shuffled_ty(tr: pd.DataFrame, seed=41) -> np.ndarray:
    """tan_y permuted within (run, arm, code): the single-track y null."""
    rng = np.random.default_rng(seed)
    out = tr.ty.to_numpy(float).copy()
    keys = tr.groupby(['run', 'arm', 'code'], observed=True).indices
    for idx in keys.values():
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def focus_scan(tr: pd.DataFrame, tyn: np.ndarray) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        m = (tr.arm == arm).to_numpy() & ok_y(tr.code)
        py, ty, dw = (tr.py.to_numpy(float)[m], tr.ty.to_numpy(float)[m],
                      tr.dw.to_numpy(float)[m])
        tn = tyn[m]
        for s in SCALE_GRID:
            yd = py + ty / s * dw
            yn = py + tn / s * dw
            rows.append(dict(arm=arm, scale=s, rsig_data=VI._rsig(yd), rsig_null=VI._rsig(yn),
                             median_data=float(np.nanmedian(yd))))
    return pd.DataFrame(rows)


# ---- the pairs --------------------------------------------------------------- #
def pair_load(od: Path) -> pd.DataFrame:
    d = pd.read_parquet(od / 'pairs_y.parquet')
    a1, a2 = d.arm1.astype(str).to_numpy(), d.arm2.astype(str).to_numpy()
    d['pair'] = pd.Series(np.char.add(np.char.add(a1, '-'), a2)).astype('category').to_numpy()
    d['null'] = d.variant > 0
    d['worst_axis'] = np.maximum(np.abs(d.e1), np.abs(d.e2))
    return d


def leg_y(d, leg, scale_map):
    arm = d[f'arm{leg}'].astype(str).to_numpy()
    s = np.array([scale_map.get(a, 1.0) for a in arm])
    return (d[f'py{leg}'].to_numpy(float)
            + d[f'ty{leg}'].to_numpy(float) / s * d[f'dw{leg}'].to_numpy(float))


#: (label, pair, which legs give y, which legs must be y-clean, window)
PAIR_ESTIMATORS = (
    ('A–D, mean of both legs (as before)', 'A-D', 'mean', (), 'x'),
    ('C–D, mean of both legs (as before)', 'C-D', 'mean', (), 'x'),
    ('A–D, D leg alone', 'A-D', 2, (2,), 'x'),
    ('C–D, D leg alone', 'C-D', 2, (2,), 'x'),
    ('A–D, A leg', 'A-D', 1, (1,), 'x'),
    ('C–D, C leg', 'C-D', 1, (1,), 'x'),
    ('D–D, mean of both D legs', 'D-D', 'mean', (1, 2), 'z'),
)


def pair_select(d, pair, yclean_legs, window, cap):
    g = d[d.pair == pair]
    m = ((g.worst_axis < CUT).to_numpy() & ok_x(g.code1) & ok_x(g.code2))
    for leg in yclean_legs:
        m &= ok_y(g[f'code{leg}'])
    if window == 'x':
        m &= (np.abs(g.vx_xz - cap[0]) < WIN_X).to_numpy()
    else:
        m &= (np.abs(g.vz_xz - cap[1]) < WIN_X).to_numpy()
    return g[m]


def pair_y(g, which, scale_map):
    if which == 'mean':
        return 0.5 * (leg_y(g, 1, scale_map) + leg_y(g, 2, scale_map))
    return leg_y(g, which, scale_map)


# ---- the 3D map with y from the informative legs ----------------------------- #
EX3 = np.arange(-64.0, 64.0 + 4.0, 4.0)
EY3 = np.arange(-Y_WIN, Y_WIN + Y_BIN, Y_BIN)


def map3d(d, cap, scale_map):
    """x slab (A-D, C-D; y from the A/C leg) + z slab (D-D; y from both D legs),
    each sideband-normalised against its null and scaled to unit excess."""
    corner_x = lambda g: (np.abs(g.vx_xz - cap[0]) > 35) & (np.abs(g.vz_xz - cap[1]) > 35)  # noqa: E731
    out = {}
    for name, pairs, which, ylegs in (('x slab (A–D, C–D)', ('A-D', 'C-D'), 1, (1,)),
                                      ('z slab (D–D)', ('D-D',), 'mean', (1, 2))):
        g = d[d.pair.isin(pairs)]
        m = (g.worst_axis < CUT).to_numpy() & ok_x(g.code1) & ok_x(g.code2)
        for leg in ylegs:
            m &= ok_y(g[f'code{leg}'])
        g = g[m]
        y = pair_y(g, which, scale_map)
        H = {}
        for v in (0, 1, 2):
            s = (g.variant == v).to_numpy()
            H[v] = np.histogramdd(np.c_[g.vx_xz.to_numpy(float)[s], y[s],
                                        g.vz_xz.to_numpy(float)[s]],
                                  bins=(EX3, EY3, EX3))[0]
        cm = ((np.abs(0.5 * (EX3[:-1] + EX3[1:]) - cap[0]) > 35)[:, None]
              & (np.abs(0.5 * (EX3[:-1] + EX3[1:]) - cap[1]) > 35)[None, :])
        Hn = H[1] + H[2]
        k = H[0].sum(1)[cm].sum() / max(Hn.sum(1)[cm].sum(), 1)
        E = H[0] - k * Hn
        w = 1.0 / max(E.sum(), 1.0)
        out[name] = dict(excess=E, weight=w, n=int((g.variant == 0).sum()), k=k)
    out['balanced'] = dict(excess=sum(o['excess'] * o['weight'] for o in out.values()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=53)
    ap.add_argument('--derive-only', action='store_true')
    a = ap.parse_args()

    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    if not a.derive_only:
        src = paths.require(Path(a.src), 'the stage-3 track tables')
        d, tr, bad = build(src, a.jobs, a.seed)
        d.to_parquet(od / 'pairs_y.parquet', index=False)
        tr.to_parquet(od / 'tracks_y.parquet', index=False)
        json.dump(dict(schema=SCHEMA, src=str(src), seed=a.seed, runs_failed=bad,
                       n_runs=int(d.run.nunique()), n_data=int((d.variant == 0).sum()),
                       n_tracks=len(tr), code_bits=dict(x_rel=REL, x_hot=HOT, conf=CONF,
                                                        y_rel=YREL, y_hot=YHOT)),
                  open(od / 'pairs_y.meta.json', 'w'), indent=1)
        del d, tr

    tr = pd.read_parquet(od / 'tracks_y.parquet')
    tyn = shuffled_ty(tr)
    BS = band_scale(tr)
    FS = focus_scan(tr, tyn)
    BS.to_csv(od / 'y_band_scale.csv', index=False)
    FS.to_csv(od / 'y_focus.csv', index=False)
    print('-- y band scale and crossing')
    print(BS.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
    best = FS.loc[FS.groupby('arm').rsig_data.idxmin()]
    print('\n-- focus scan minimum')
    print(best.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
    band_s = BS[BS.tier == 'y-clean'].set_index('arm').band_scale.to_dict()
    focus_s = best.set_index('arm').scale.to_dict()
    scales = {'raw (s = 1)': {a_: 1.0 for a_ in ARMS},
              'band scale': band_s, 'focus scale': focus_s}

    # single-track y images
    rows = []
    for sname, smap in scales.items():
        for arm in ARMS:
            for tier, m in (('x-clean', np.ones(len(tr), bool)), ('y-clean', ok_y(tr.code))):
                mm = (tr.arm == arm).to_numpy() & m
                s = smap.get(arm, 1.0)
                yd = tr.py.to_numpy(float)[mm] + tr.ty.to_numpy(float)[mm] / s * tr.dw.to_numpy(float)[mm]
                yn = tr.py.to_numpy(float)[mm] + tyn[mm] / s * tr.dw.to_numpy(float)[mm]
                r = fit_y(yd, yn)
                rows.append(dict(kind='single track', selection=f'chamber {arm}', tier=tier,
                                 scale_set=sname, scale=s, **r))
    del tr

    d = pair_load(od)
    for sname, smap in scales.items():
        for label, pair, which, ylegs, win in PAIR_ESTIMATORS:
            g = pair_select(d, pair, ylegs, win, cap)
            y = pair_y(g, which, smap)
            nul = g.null.to_numpy()
            r = fit_y(y[~nul], y[nul])
            rows.append(dict(kind='pair', selection=label, tier='x-clean both legs'
                             + ('; y-clean on the y legs' if ylegs else ''),
                             scale_set=sname, scale=np.nan, **r))
            print(f'   {sname:12s} {label:36s} n={r["n_data"]:>7,} '
                  + (f'c={r["c"]:+6.1f}±{r["c_err"]:.1f} s={r["s"]:5.1f} f={r["f"]:.2f} '
                     f'chi2/ndf={r["chi2"] / max(r["ndf"], 1):.1f}' if np.isfinite(r.get('c', np.nan))
                     else f'no capsule term  rsig={r["rsig_data"]:.0f}'), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / 'y_fits.csv', index=False)
    print()
    print(R[R.kind == 'single track'][['selection', 'tier', 'scale_set', 'scale', 'n_data',
                                        'c', 'c_err', 's', 'f', 'rsig_data', 'rsig_null']]
          .to_string(index=False, float_format=lambda x: f'{x:8.2f}'))

    # the 3D map at the scale as reconstructed: the band scale widens the
    # images and the focus scale is biased high (module docstring, point 3)
    M = map3d(d, cap, scales['raw (s = 1)'])
    np.savez_compressed(od / 'y_map3d.npz', ex=EX3, ey=EY3,
                        **{f'{k}|excess': v['excess'] for k, v in M.items()})
    json.dump(dict(scales={k: {a_: float(v) for a_, v in s.items()} for k, s in scales.items()},
                   capsule_xz=list(cap), capsule_y_centroid=capsule_y_profile()[2],
                   cut_mm=CUT, win_x_mm=WIN_X, y_win=Y_WIN, y_bin=Y_BIN,
                   map3d={k: dict(n=v.get('n'), k=v.get('k'), weight=v.get('weight'))
                          for k, v in M.items() if k != 'balanced'}),
              open(od / 'y_derive.meta.json', 'w'), indent=1)
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
