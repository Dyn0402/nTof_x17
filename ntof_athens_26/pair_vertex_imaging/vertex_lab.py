#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vertex_lab.py -- why the pair vertex does not image the capsule and the single
tracks do.

THE QUESTION.  `source_imaging.py` locates the He-3 capsule to a few tenths of
a millimetre and the answer repeats over 33 runs.  `pair_qa.py` then forms every
in-trigger pair of those same tracks, takes the closest approach of the two
lines, and the resulting vertex sits **tens of millimetres** off the beam axis
with a distribution the event-mixed null reproduces.  One of those two results
looks wrong.  Neither is: they are **different measurements of different things
on different samples**, and this module takes the difference apart.

WHAT IS BUILT HERE.  One pair table, wider than `pair_qa`'s, carrying the
geometry the diagnosis needs and `pair_qa` throws away -- both legs' full
(p0, d), the transverse-only crossing, and the y mismatch at that crossing.
Four quantities per pair instead of one:

  ``v_r``        the published 3D quantity: radius of the DCA midpoint of the
                 two lines.  What the QA figure plots.
  ``v_r_xz``     the same thing computed **in the XZ projection only** -- the
                 two lines' crossing point in the transverse plane.  This uses
                 the in-plane angles and nothing else, which is *exactly* the
                 information `source_imaging`'s band uses.
  ``dy_cross``   the two legs' y, evaluated at that transverse crossing,
                 subtracted.  The y information, isolated, with the transverse
                 information divided out.
  ``sin_psi_xz`` the sine of the transverse crossing angle: the conditioning.
                 A pair that crosses at 5 deg localises nothing however well
                 each leg is measured, and this is the number that says so.

The split is the point.  ``v_r`` mixes three separate failure modes -- per-leg
angular resolution, sample purity, and the geometry of the crossing -- and a
single histogram of it cannot say which is biting.  ``v_r_xz`` and
``dy_cross`` separate the transverse and longitudinal information;
``sin_psi_xz`` separates out the conditioning.

NO CUT IS BAKED IN.  `pair_qa` pre-selects legs at ``dca_axis_mm < 30``, which
is a *pointing* cut, so its pair sample is already conditioned on the very
quantity the vertex is supposed to measure.  Here the cut is a column, applied
at analysis time and scanned (:func:`scan_leg_dca`), so "does a tighter leg cut
sharpen the vertex?" is answerable rather than assumed.  The build default is a
loose 60 mm so the whole 10-60 mm range is available downstream.

THE EVENT-MIXED NULL DOES NOT MEAN HERE WHAT IT MEANS IN THE ANGLE SPECTRUM.
Mixing decorrelates the *trigger*, not the *origin*.  Two tracks from two
different neutron captures in the same capsule still both come from the capsule,
so a mixed pair has a real common source and must image it just as well as a
real pair does.  Real == mixed is therefore **not** evidence that the imaging
failed; it is evidence that the imaging carries no coincidence information,
which is a different (and expected) statement.  The null is kept because it is
the reference for every *shape*, and because its agreement with the data is
itself one of the measurements below.

    python -m pair_vertex_imaging.vertex_lab --jobs 8      # build the table
    python -m pair_vertex_imaging.vertex_lab --derive-only # re-measure, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

SCHEMA = 'athens26/pair_vertex/1'
ARMS = ('A', 'B', 'C', 'D')

#: Perpendicular distance from the beam axis to every strip plane [mm].  The
#: lever arm that turns an angle error into a pointing error, and the single
#: most important number in this module: a track's miss distance at the axis is
#: ``D_PERP * (tan_measured - tan_true)`` to first order, so a tan error of 0.1
#: is a 23 mm miss no matter how well the strips are read.
D_PERP_MM = 234.6

#: The He-3 active gas, from ``ntof_tracking.reco.geometry`` -- 10 mm radius,
#: 80 mm long along the beam.  Transversely this is the thing being imaged, and
#: it sets the floor: no estimator can beat 10 mm in ``v_r`` because the source
#: really is 10 mm across.
HE3_R_MAX = 10.0
HE3_Y = (-29.50, 50.70)

#: Nominal chamber basis.  Verified against the stage-3 columns rather than
#: trusted: recomputing ``tanx``/``tany`` from ``(d_x, d_y, d_z)`` through these
#: axes reproduces the stored values to 1e-15 on run_145, so the run_config
#: y-rotations are zero for this campaign and the nominal axes are exact.
U_HAT = {'A': np.array([1., 0., 0.]), 'B': np.array([0., 0., 1.]),
         'C': np.array([-1., 0., 0.]), 'D': np.array([0., 0., -1.])}
W_HAT = {'A': np.array([0., 0., 1.]), 'B': np.array([-1., 0., 0.]),
         'C': np.array([0., 0., -1.]), 'D': np.array([1., 0., 0.])}
V_HAT = np.array([0., 1., 0.])

#: Runs on the other side of the 27 July access.  Same exclusion as
#: `tight_coincidence.PRE_ACCESS_RUNS`, spelled here so this module can be run
#: without importing the campaign chain.
PRE_ACCESS_RUNS = ('run_79', 'run_81')

#: Stage-3 columns carried onto both legs.  Everything the diagnosis needs and
#: nothing it does not: the line, the pointing, the fit quality, the depth.
LEG_COLS = ['p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'tanx', 'tany',
            'x_local', 'y_local', 'dca_axis_mm', 'target_x_mm', 'target_y_mm',
            'target_z_mm', 'chi2dof_x', 'chi2dof_y', 'x_n_strips',
            'y_n_strips', 'drift_railed', 'drift_len_mm', 'q_total',
            'angle_to_beam_deg', 'k_arm', 't_since_flash_ns']

TRACK_COLS = (['event_id', 'arm', 'gated', 'angle_calibrated'] + LEG_COLS)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def dca_3d(p1, d1, p2, d2):
    """Closest approach of two 3D lines: (midpoint, distance, s, t).

    ``s``/``t`` are the path lengths from each ``p0`` to that line's own closest
    point, and they are returned because they are a diagnosis in themselves: a
    pair whose vertex sits 400 mm *behind* one of the chambers has an ``s`` that
    says so, and the radius alone does not.
    """
    w0 = p1 - p2
    a = np.einsum('ij,ij->i', d1, d1)
    b = np.einsum('ij,ij->i', d1, d2)
    c = np.einsum('ij,ij->i', d2, d2)
    d = np.einsum('ij,ij->i', d1, w0)
    e = np.einsum('ij,ij->i', d2, w0)
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    q1 = p1 + s[:, None] * d1
    q2 = p2 + t[:, None] * d2
    return 0.5 * (q1 + q2), np.linalg.norm(q1 - q2, axis=1), s, t


def cross_xz(p1, d1, p2, d2):
    """The two lines' crossing **in the transverse (XZ) plane**.

    Two lines in a plane always meet unless they are parallel, so there is no
    miss distance here and no midpoint to take -- the crossing is a point, and
    it is the transverse image the pair actually supports.  Returned with it:

      ``sin_psi``  sine of the crossing angle.  The conditioning: a transverse
                   displacement of one leg by ``eps`` moves the crossing by
                   ``eps / sin_psi``, so this is the amplification factor and
                   it is why A-C (nearly anti-parallel in XZ) and the intra
                   pairs (nearly parallel) cannot localise anything.
      ``dy``       ``y1 - y2`` evaluated at the crossing.  The two legs agree
                   transversely by construction; whether they agree in y is a
                   free test, and it is the one that isolates the y plane.
      ``s``/``t``  path length along each line to the crossing.

    Signs: ``d`` points from the strip plane INWARD (verified on run_145, all
    four chambers), so a physical crossing has ``s, t > 0`` and a pair whose
    crossing sits at negative ``s`` is extrapolating backwards through the
    chamber.  That population is kept and counted, not silently dropped.
    """
    a1, b1 = d1[:, 0], d1[:, 2]
    a2, b2 = d2[:, 0], d2[:, 2]
    det = a2 * b1 - a1 * b2
    n1 = np.hypot(a1, b1)
    n2 = np.hypot(a2, b2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sin_psi = np.abs(det) / np.where((n1 * n2) > 1e-12, n1 * n2, np.nan)
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        dx = p2[:, 0] - p1[:, 0]
        dz = p2[:, 2] - p1[:, 2]
        s = (a2 * dz - b2 * dx) / det
        t = (a1 * dz - b1 * dx) / det
    cx = p1[:, 0] + s * d1[:, 0]
    cz = p1[:, 2] + s * d1[:, 2]
    y1 = p1[:, 1] + s * d1[:, 1]
    y2 = p2[:, 1] + t * d2[:, 1]
    return cx, cz, y1 - y2, 0.5 * (y1 + y2), sin_psi, s, t


def rescale_dirs(arms: np.ndarray, d: np.ndarray, scale) -> np.ndarray:
    """Rebuild unit directions with every in-plane and out-of-plane tan scaled.

    ``scale`` is a float or a dict ``{arm: factor}``.  The rebuild goes through
    the chamber basis rather than through the global components, because the
    angle scale ``k`` is a property of the DRIFT -- it multiplies ``tan`` -- and
    scaling a global component instead would mix the two chambers' conventions
    and rotate the track rather than tilt it.
    """
    out = np.empty_like(d)
    for arm in np.unique(arms):
        m = arms == arm
        if arm not in U_HAT:
            out[m] = d[m]
            continue
        f = scale.get(arm, 1.0) if isinstance(scale, dict) else float(scale)
        u, w = U_HAT[arm], W_HAT[arm]
        du, dv, dw = d[m] @ u, d[m] @ V_HAT, d[m] @ w
        with np.errstate(divide='ignore', invalid='ignore'):
            tx, ty = du / dw, dv / dw
        # d = dw * (tx, ty, 1) in the (u, v, w) basis, and dw < 0 for every
        # chamber (d points inward from the strip plane), so the sign has to
        # be carried or the rebuilt track is REFLECTED rather than tilted --
        # which costs nothing at f = 1 in |d| and everything in where it
        # points.  The f = 1 identity is asserted by the caller.
        nd = (f * tx)[:, None] * u + (f * ty)[:, None] * V_HAT + w[None, :]
        nd = nd * np.sign(dw)[:, None]
        out[m] = nd / np.linalg.norm(nd, axis=1)[:, None]
    return out


def check_rescale_identity(arms, d, tol=1e-9) -> float:
    """``rescale_dirs(.., 1.0)`` must return the input.  Raises if it does not.

    The scan below is only a measurement if the identity holds: a rebuild that
    is subtly wrong at f = 1 would show a vertex 'optimum' that is an artefact
    of the rebuild and not of the angle scale.  Cheap, so it runs every time.
    """
    r = rescale_dirs(arms, d, 1.0)
    ok = np.isfinite(d).all(axis=1)
    err = float(np.nanmax(np.abs(r[ok] - d[ok])))
    if err > tol:
        raise AssertionError(f'rescale_dirs(f=1) is not the identity: {err:.3g}')
    return err


def vertex_columns(a: pd.DataFrame, b: pd.DataFrame, d1=None, d2=None
                   ) -> pd.DataFrame:
    """Every vertex quantity for one aligned pair of leg frames."""
    p1 = a[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    p2 = b[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    d1 = a[['d_x', 'd_y', 'd_z']].to_numpy(float) if d1 is None else d1
    d2 = b[['d_x', 'd_y', 'd_z']].to_numpy(float) if d2 is None else d2
    V, sep, s3, t3 = dca_3d(p1, d1, p2, d2)
    cx, cz, dy, cy, sin_psi, sxz, txz = cross_xz(p1, d1, p2, d2)
    dot = np.einsum('ij,ij->i', d1, d2).clip(-1, 1)
    return pd.DataFrame(dict(
        vx=V[:, 0], vy=V[:, 1], vz=V[:, 2],
        v_r=np.hypot(V[:, 0], V[:, 2]), sep_mm=sep, s3=s3, t3=t3,
        vx_xz=cx, vz_xz=cz, v_r_xz=np.hypot(cx, cz),
        dy_cross=dy, vy_xz=cy, sin_psi_xz=sin_psi, s_xz=sxz, t_xz=txz,
        open_deg=np.degrees(np.arccos(dot))))


# --------------------------------------------------------------------------- #
# the sample
# --------------------------------------------------------------------------- #
def track_table(run: str, subruns, src: Path, dca_max: float) -> pd.DataFrame:
    """Gated, angle-calibrated tracks of one run.

    Deliberately the SAME selection as `source_imaging._track_table` -- gated,
    ``angle_calibrated``, a ``dca_axis_mm`` ceiling -- so that any difference
    found downstream is a difference of estimator and not of sample.  The only
    change is that ``dca_max`` defaults loose here; the published 30 mm is one
    point on a scan rather than the boundary of the universe.
    """
    out = []
    for sub in subruns:
        p = src / f'tracks_{run}_{sub}.parquet'
        if not p.exists():
            raise FileNotFoundError(f'missing stage-3 tracks: {p}')
        out.append(pd.read_parquet(p, columns=TRACK_COLS).assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated & t.angle_calibrated & (t.dca_axis_mm < dca_max)].copy()
    t['key'] = t.subrun + ':' + t.event_id.astype(str)
    return t.reset_index(drop=True)


def pairs_real(t: pd.DataFrame) -> pd.DataFrame:
    """Every unordered pair of distinct tracks inside one trigger.

    Vectorised rather than ``itertools`` over groups -- the loose ``dca_max``
    here makes the sample several times the published one and the per-group
    Python loop stops being free.  Checked against the loop on run_145: same
    pairs, same order after a sort.
    """
    idx = t.index.to_numpy()
    order = np.argsort(t.key.to_numpy(), kind='stable')
    ks = t.key.to_numpy()[order]
    ids = idx[order]
    bnd = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    L, R = [], []
    for lo, hi in zip(bnd[:-1], bnd[1:]):
        n = hi - lo
        if n < 2:
            continue
        i, j = np.triu_indices(n, k=1)
        L.append(ids[lo:hi][i])
        R.append(ids[lo:hi][j])
    if not L:
        return pd.DataFrame(dict(i=np.zeros(0, np.int64), j=np.zeros(0, np.int64)))
    return pd.DataFrame(dict(i=np.concatenate(L).astype(np.int64),
                             j=np.concatenate(R).astype(np.int64)))


def pairs_mixed(t: pd.DataFrame, real: pd.DataFrame, seed=5) -> pd.DataFrame:
    """The null, matched pair-for-pair in arm composition.

    Identical rule to `source_imaging._pairs_mixed` -- draw from the tracks that
    actually form real pairs, reject same-trigger draws -- so the null here and
    the null in the published QA are the same construction and can be compared.
    Vectorised, with a rejection loop only for the collisions.
    """
    if real.empty:
        return pd.DataFrame(dict(i=np.zeros(0, np.int64), j=np.zeros(0, np.int64)))
    rng = np.random.default_rng(seed)
    arms = t.arm.to_numpy()
    keys = t.key.to_numpy()
    used = np.unique(np.concatenate([real.i.to_numpy(), real.j.to_numpy()]))
    pools = {a: used[arms[used] == a] for a in np.unique(arms[used])}
    a1, a2 = arms[real.i.to_numpy()], arms[real.j.to_numpy()]
    L = np.full(len(real), -1, np.int64)
    R = np.full(len(real), -1, np.int64)
    for arm in np.unique(a1):
        m = a1 == arm
        if len(pools.get(arm, ())) >= 2:
            L[m] = rng.choice(pools[arm], m.sum())
    for arm in np.unique(a2):
        m = a2 == arm
        if len(pools.get(arm, ())) >= 2:
            R[m] = rng.choice(pools[arm], m.sum())
    ok = (L >= 0) & (R >= 0)
    for _ in range(20):
        bad = ok & ((L == R) | (keys[np.where(ok, R, 0)]
                                == keys[np.where(ok, L, 0)]))
        if not bad.any():
            break
        for arm in np.unique(a2[bad]):
            m = bad & (a2 == arm)
            R[m] = rng.choice(pools[arm], m.sum())
    ok &= ~((L == R) | (keys[np.where(ok, R, 0)] == keys[np.where(ok, L, 0)]))
    return pd.DataFrame(dict(i=L[ok], j=R[ok]))


def topology(a: str, b: str) -> str:
    """intra / perpendicular / opposing, by the chambers' azimuth."""
    if a == b:
        return 'intra'
    return 'opposing' if {a, b} in ({'A', 'C'}, {'B', 'D'}) else 'perpendicular'


def pair_frame(t: pd.DataFrame, pr: pd.DataFrame, mixed: bool) -> pd.DataFrame:
    """Vertex quantities plus both legs, in ``arm1 <= arm2`` order.

    The leg order follows the arms and not the storage order, for the reason
    `pair_qa` gives: otherwise an A-D pair puts A's numbers in the ``_1``
    columns only half the time and every per-chamber plot is a mixture.
    """
    if pr.empty:
        return pd.DataFrame()
    a = t.loc[pr.i.to_numpy()].reset_index(drop=True)
    b = t.loc[pr.j.to_numpy()].reset_index(drop=True)
    swap = (a.arm.to_numpy() > b.arm.to_numpy())
    a2 = a.copy()
    for c in a.columns:
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        a2[c] = np.where(swap, bv, av)
        b[c] = np.where(swap, av, bv)
    a = a2
    out = vertex_columns(a, b)
    out['arm1'] = a.arm.to_numpy()
    out['arm2'] = b.arm.to_numpy()
    out['key1'] = a.key.to_numpy()
    out['key2'] = b.key.to_numpy()
    for c in LEG_COLS:
        out[f'{c}_1'] = a[c].to_numpy()
        out[f'{c}_2'] = b[c].to_numpy()
    out['dca_worst'] = np.nanmax(
        np.vstack([out.dca_axis_mm_1, out.dca_axis_mm_2]), axis=0)
    out['dca_best'] = np.nanmin(
        np.vstack([out.dca_axis_mm_1, out.dca_axis_mm_2]), axis=0)
    out['chi2dof_worst'] = np.nanmax(np.vstack([
        out.chi2dof_x_1, out.chi2dof_y_1, out.chi2dof_x_2, out.chi2dof_y_2]),
        axis=0)
    out['mixed'] = mixed
    return out


def one_run(run: str, subruns, src: str, dca_max: float, seed: int) -> tuple:
    """Worker: the whole pair frame of one run, real and mixed."""
    try:
        t = track_table(run, subruns, Path(src), dca_max)
        real = pairs_real(t)
        if real.empty:
            return run, None, 'no pairs (no angle-calibrated tracks?)'
        mix = pairs_mixed(t, real, seed)
        d = pd.concat([pair_frame(t, real, False), pair_frame(t, mix, True)],
                      ignore_index=True)
        d['run'] = run
        d['topo'] = [topology(x, y) for x, y in zip(d.arm1, d.arm2)]
        d['pair'] = d.arm1 + '-' + d.arm2
        # Per-track table too, pooled downsampled: the single-track image is
        # half the comparison and rebuilding it from the pairs would weight
        # every track by how many pairs it is in.
        tk = t[['arm', 'dca_axis_mm', 'target_x_mm', 'target_y_mm',
                'target_z_mm', 'x_local', 'y_local', 'tanx', 'tany',
                'drift_railed']].copy()
        tk['run'] = run
        return run, (d, tk), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def discover(src: Path, include_pre_access: bool) -> dict:
    rs: dict = {}
    for p in sorted(src.glob('tracks_run_*_stat090_*.parquet')):
        m = re.match(r'tracks_(run_\d+)_(stat090_\d+)\.parquet$', p.name)
        if m:
            rs.setdefault(m.group(1), []).append(m.group(2))
    if not include_pre_access:
        for r in PRE_ACCESS_RUNS:
            rs.pop(r, None)
    return rs


def build(src: Path, dca_max: float, jobs: int, include_pre_access: bool,
          seed: int) -> tuple:
    rs = discover(src, include_pre_access)
    print(f'{len(rs)} run(s), {sum(len(v) for v in rs.values())} sub-runs '
          f'from {src}   leg dca < {dca_max:.0f} mm\n')
    pairs, tracks, bad = [], [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, s, str(src), dca_max, seed): r
                for r, s in rs.items()}
        for f in as_completed(futs):
            run, res, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            d, tk = res
            pairs.append(d)
            tracks.append(tk)
            print(f'  {run:<10} ok   {int((~d.mixed).sum()):>8,} real pairs'
                  f'   {len(tk):>8,} tracks', flush=True)
    P = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    T = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame()
    return P, T, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--dca', type=float, default=60.0,
                    help='leg pointing ceiling at BUILD time; the published '
                         '30 mm is then one point on a scan (default 60)')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--include-pre-access', action='store_true')
    a = ap.parse_args()

    src = paths.require(Path(a.src), 'the stage-3 track tables')
    od = paths.out('pair_vertex')
    P, T, bad = build(src, a.dca, a.jobs, a.include_pre_access, a.seed)
    if P.empty:
        print('no pairs at all -- nothing written')
        return 1
    P.to_parquet(od / 'pairs_vertex.parquet', index=False)
    T.to_parquet(od / 'tracks_vertex.parquet', index=False)
    json.dump(dict(schema=SCHEMA, src=str(src), dca_max=a.dca, seed=a.seed,
                   include_pre_access=a.include_pre_access,
                   d_perp_mm=D_PERP_MM, he3_r_max=HE3_R_MAX,
                   n_runs=int(P.run.nunique()),
                   n_pairs_real=int((~P.mixed).sum()),
                   n_pairs_mixed=int(P.mixed.sum()),
                   n_tracks=int(len(T)), runs_failed=bad),
              open(od / 'pair_vertex.meta.json', 'w'), indent=1)
    print(f'\nREAL pairs by class')
    print(P[~P.mixed].groupby(['topo', 'pair']).size().to_string())
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
