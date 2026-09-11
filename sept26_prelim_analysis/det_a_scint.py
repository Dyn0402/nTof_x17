#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
det_a_scint.py -- every arm-A track, pointed back at the scintillators behind it.

ONE QUESTION.  Take every gated track the chamber found -- not the pair sample,
not the pointing subset, every track -- extrapolate its line to each
scintillator layer behind the chamber, and ask whether the channel it lands on
is a channel that actually fired.

That is a per-track, POSITIONAL confirmation, and the analysis did not have one.
What it had was three things that are each less than this:

  * ``wall_A`` / ``plastic_A`` in the track table are PER TRIGGER booleans from
    stage 1 -- "arm A's wall fired somewhere".  A track on the far side of the
    chamber from the fired bar carries the same flag as one that points at it.
    `det_a_intra`'s ``tag_A`` is this.
  * `build_tracks.predictions` writes ``pred_sipm_bar`` / ``pred_plastic``
    for every track and nothing in the chain ever compares them with the slim.
    The prediction has been sitting in all 586 track files unused.
  * `run145_target_imaging.pointing_coincidence` does compare them, but in the
    x plane ONLY: it never checks that the track is inside the bar's 500 mm
    length in v, it returns a single boolean with no residual, and its
    consumers (`funnel`, `campaign_efficiency`) use it as a purity cut rather
    than as a measurement.

WHAT IS NEW HERE, in the order the report reads it.

**The extrapolation is 3D and comes from the DAQ's own survey.**  Every layer
position is read from ``run_config.json`` -- ``sipm_A_01..16`` at w = 332.0 mm,
``plastic_A_L/R`` at 425.22, ``liquid_A`` at 483.1, all in the global frame the
tracks already live in -- so the geometry in this module and the geometry the
DAQ recorded cannot drift apart.  The arm-A configuration is IDENTICAL in all
36 runs and the module asserts that rather than assuming it.

**The v coordinate is checked, and it is not free.**  Two thirds of arm-A
tracks land inside the plastic's +-150 mm half-length; the rest leave the
layer through its end and can never be confirmed by it.  A u-only test counts
those as failures.

**The position tolerance is MEASURED, not assumed.**  The formal fit error
extrapolates to 1.8 mm at the wall and 3.5 mm at the plastic, which is far
smaller than the 100 mm the wall's read-out granularity can resolve and is not
the relevant width.  The relevant width is how sharply the identity of the
fired group switches as the predicted crossing moves across a group boundary,
and :func:`edge_profile` measures it in situ on single-track, single-group
events.  ``EDGE_SIGMA_MM`` is the fitted value; a track closer than
``EDGE_N_SIGMA`` times it to a boundary is scored against BOTH groups
(``match_tol``), because for that track the prediction genuinely does not
choose between them.

**THE WALL IS FOUR CHANNELS IN u, NOT SIXTEEN.**  16 bars are instrumented but
the slim carries ``detn`` 1..8 = four groups of four bars x two ends, so the
wall resolves u to 100 mm and nothing finer.  The bar-level prediction is kept
in the table because the two ends are the y handle (`scintillators.py`), but
every match here is at group level and says so.

    python -m sept26_prelim_analysis.det_a_scint --runs run_145
    python -m sept26_prelim_analysis.det_a_scint --jobs 6
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    PRE_ACCESS_RUNS, condition, in_block, run_number, subruns_of)

SCHEMA = 'sept26_prelim/det_a_scint/1'
ARM = 'A'

#: THE WINDOWS, and every one of them is the SAME WIDTH as the signal window
#: it controls.  This is not a detail: the first version of this module used a
#: 700 ns off-time control against a 160 ns signal window and duly found that
#: the plastic's confirmation rate (0.591) equalled its accidental floor
#: (0.576) -- an artefact of comparing a wide window with a narrow one, not a
#: result.  A control window must have the width of the thing it controls.
#:
#: ``prod`` is the production accept window `efficiency` and
#: `run145_target_imaging` use, so a rate here is readable against a rate
#: there.  ``core`` is the peak's own core: HANDOFF_ACCIDENTAL_TIMING.md
#: measures the prompt peak at roughly +-15 ns on a flat pedestal and says the
#: production window admits ~100 ns of background it does not need.  Both are
#: carried, because the difference between them IS the accidental content.
#:
#: Each has two controls.  ``ctrl`` re-runs the identical test on the slim's
#: own ``is_control`` hits -- the n_TOF processing's random-coincidence sample,
#: dead flat across +-1000 ns and, per that handoff, used nowhere in this
#: analysis except to be cut away.  ``off`` re-runs it on real hits in a window
#: of the same width displaced past the peak.  Two independent accidental
#: floors that should agree, and do.
DT_WINDOW = (-100.0, 60.0)
CORE_WINDOW = (-30.0, 30.0)
#: Displaced 400 ns BEFORE the trigger, same width as ``prod``.  Before, not
#: after, and that is measured rather than stylistic: on run_145 the wall's
#: ``dt_ns`` is flat at ~1 100 hits per 100 ns everywhere outside a single
#: 100 ns peak, but the PLASTIC decays for the best part of a microsecond after
#: it -- 27 154 hits in (0, 100) falling through 9 149, 6 223, 3 431 and still
#: ~15 % above its own pedestal at +900 ns.  A control window at +400 ns
#: therefore sits in a real, trigger-correlated tail and reports the plastic's
#: accidental floor as 21 % when it is 5 %.  The pre-trigger side is flat for
#: both layers and equals the ``is_control`` level exactly, which is what makes
#: the two floors an independent check on each other rather than two names for
#: the same assumption.
OFF_WINDOW = (-560.0, -400.0)
OFF_CORE_WINDOW = (-460.0, -400.0)

#: name -> (is_control value, lo, hi).  The order is the order the report reads.
WINDOWS = {
    'prod': (0, DT_WINDOW[0], DT_WINDOW[1]),
    'core': (0, CORE_WINDOW[0], CORE_WINDOW[1]),
    'ctrl': (1, DT_WINDOW[0], DT_WINDOW[1]),
    'ctrl_core': (1, CORE_WINDOW[0], CORE_WINDOW[1]),
    'off': (0, OFF_WINDOW[0], OFF_WINDOW[1]),
    'off_core': (0, OFF_CORE_WINDOW[0], OFF_CORE_WINDOW[1]),
}
#: The window every unqualified column means.
PRIMARY = 'prod'
#: Its two accidental floors, in the order the report quotes them.
CONTROLS = ('ctrl', 'off')

#: ``det`` codes: wall 0-3, plastic 4-7, liquid 8-11, in A B C D order.
ARMS = ('A', 'B', 'C', 'D')
WAL_CODE = {a: i for i, a in enumerate(ARMS)}

#: Wall read-out: ``detn`` 1..8 = 4 groups x 2 ends, group = (detn - 1) // 2,
#: group g covering instrumented bars 4g+1 .. 4g+4 ASCENDING in u.  The
#: ascending sense is empirical (`run145_target_imaging`, 2026-08-20) because
#: the SiPM bars are the one family `run_config.json` gives no ``ntof_daq``
#: block for; :func:`group_confusion` re-measures it here, per run, so the
#: convention is checked on every sample rather than inherited.
BARS_PER_GROUP = 4
N_WALL_GROUP = 4
#: Plastic read-out: ``detn`` 1 = L, 2 = R.  This one IS in ``run_config`` --
#: ``plastic_A_L`` carries ``ntof_daq.detn = 1`` -- so it is read, not assumed.

#: Half-widths of the active volumes, mm (geometry.py, ported from the Geant
#: build and cross-checked against the DAQ config).
SIPM_BAR_HALF_U = 12.5
SIPM_HALF_V = 250.0
PLASTIC_HALF_U = 100.0
PLASTIC_HALF_V = 150.0
LS_HALF_U = 225.6
LS_HALF_V = 225.3

#: The in-situ edge width at the wall, mm -- how far the predicted crossing has
#: to be from a group boundary before the fired group is unambiguous.  MEASURED
#: by :func:`edge_profile`; this is the default the per-track tolerant match
#: uses when no fit is supplied, and the fit overwrites it in the products.
EDGE_SIGMA_MM = 25.0
#: How many edge widths count as "on the boundary".
EDGE_N_SIGMA = 1.0

#: Map binning, mm.  20 mm on the chamber (the strip pitch is 0.4 mm, so this
#: is set by how many tracks a cell needs, not by resolution) and 25 mm on the
#: wall so a cell is exactly one bar wide in u.
MM_U_EDGES = np.arange(-200.0, 200.1, 20.0)
MM_V_EDGES = np.arange(-200.0, 200.1, 20.0)
WALL_U_EDGES = np.arange(-250.0, 250.1, 25.0)
WALL_V_EDGES = np.arange(-300.0, 300.1, 25.0)
PLAS_U_EDGES = np.arange(-250.0, 250.1, 25.0)
PLAS_V_EDGES = np.arange(-250.0, 250.1, 25.0)
#: Fine binning for the edge profile: 10 mm, ~2.5 x finer than the width being
#: measured.
EDGE_EDGES = np.arange(-260.0, 260.1, 10.0)
#: Minimum tracks in a map cell before its match fraction is reported.
MIN_CELL = 20

#: The chamber's ACTIVE area, mm from the plane centre (geometry.MM_SIZE_U/V).
#: A reconstructed impact point outside this is not a position on the detector,
#: and a fifth of the arm-A track table is outside it in v -- see
#: :func:`rail_census`.
FIDUCIAL_U = 190.0
FIDUCIAL_V = 170.0
#: The v rail itself: the fitted y position piles up here, just outside the
#: active area, and this window is what counts it.
RAIL_V = (-205.0, -185.0)

#: The columns read from the stage-3 table.  Narrow on purpose: the campaign
#: table is 24 GB and this module wants ~2 M arm-A rows out of it.
TRACK_COLS = [
    'run', 'subrun', 'event_id', 'arm', 'gated', 'angle_calibrated',
    'x_local', 'y_local', 'tanx', 'tany',
    'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
    'x_p0_err', 'x_tan_err', 'y_p0_err', 'y_tan_err',
    'x_slope_reliable', 'y_slope_reliable', 'chi2dof_x', 'chi2dof_y',
    'q_total', 'dca_axis_mm', 'drift_railed',
    'pred_sipm_bar', 'pred_plastic', 'pred_ls',
    'wall_A', 'plastic_A', 'k_arm',
]
#: Columns the per-track product carries out.  Everything a downstream question
#: about position, tolerance or channel identity needs, and nothing else.
KEEP_COLS = [
    'run', 'subrun', 'event_id', 'n_trk',
    'u_mm', 'v_mm', 'tanx', 'tany', 'slope_ok', 'dca_axis_mm',
    'u_wall', 'v_wall', 'u_plas', 'v_plas', 'u_ls', 'v_ls',
    'sig_u_wall', 'sig_v_wall', 'sig_u_plas', 'sig_v_plas',
    'bar_pred', 'grp_pred', 'grp_alt', 'd_edge_wall', 'on_wall',
    'plas_pred', 'd_edge_plas', 'on_plas', 'on_ls',
    'n_grp_fired', 'grp_fired', 'n_plas_fired', 'plas_fired', 'ls_fired',
    'match_wall_alt', 'match_plas_alt', 'res_u_wall', 'res_u_plas',
] + [f'{c}{"" if w == PRIMARY else "_" + w}'
     for w in WINDOWS for c in ('wall_hit', 'plas_hit',
                                'match_wall', 'match_plas')]


# --------------------------------------------------------------------------- #
# geometry, from the DAQ's own survey
# --------------------------------------------------------------------------- #
def layer_geometry(run: str, arm: str = ARM) -> dict:
    """Where every layer of one arm is, in the frame the tracks live in.

    Read from ``run_config.json``, which places each scintillator in global
    coordinates and carries the ``ntof_daq.detn`` code for the two families
    that have one.  Nothing here is a constant copied from a note.

    Returns ``w_*`` (the global coordinate along the arm's outward normal),
    ``bar_u`` (instrumented SiPM bar -> u on the structure), ``plas_u``
    (``detn`` -> u), and the unit vectors.
    """
    from ntof_tracking.reco import geometry as G
    cfg = json.loads(paths.require(paths.root('runs') / run / 'run_config.json',
                                   f'DAQ config for {run}').read_text())
    tr = G.detector_transforms(cfg)[f'mx17_{arm}']
    uh, vh, wh = G.U_HAT[arm], G.V_HAT, G.W_HAT[arm]
    pos = {d['name']: np.array([d['det_center_coords'][k] for k in 'xyz'],
                               float) for d in cfg['detectors']}
    daq = {d['name']: (d.get('ntof_daq') or {}).get('detn')
           for d in cfg['detectors']}

    bar_u, bar_w = {}, []
    for name, p in pos.items():
        if name.startswith(f'sipm_{arm}_'):
            bar_u[int(name.split('_')[-1])] = float(p @ uh)
            bar_w.append(float(p @ wh))
    if len(bar_u) != BARS_PER_GROUP * N_WALL_GROUP:
        raise ValueError(f'{run} arm {arm}: {len(bar_u)} SiPM bars in the DAQ '
                         f'config, expected {BARS_PER_GROUP * N_WALL_GROUP}')
    if max(bar_w) - min(bar_w) > 1e-6:
        raise ValueError(f'{run} arm {arm}: SiPM bars are not coplanar')

    plas_u = {}
    for side in ('L', 'R'):
        n = daq.get(f'plastic_{arm}_{side}')
        if n is None:
            raise ValueError(f'{run}: plastic_{arm}_{side} has no ntof_daq.detn '
                             'in the DAQ config -- the L/R to detn map cannot '
                             'be assumed, it is what fixes the readout order')
        plas_u[int(n)] = float(pos[f'plastic_{arm}_{side}'] @ uh)

    return dict(
        arm=arm, u_hat=uh, v_hat=vh, w_hat=wh,
        w_strip=float(tr.center @ wh), u_mm=float(tr.center @ uh),
        v_mm=float(tr.center @ vh),
        w_wall=float(bar_w[0]),
        w_plas=float(pos[f'plastic_{arm}_L'] @ wh),
        w_ls=float(pos[f'liquid_{arm}'] @ wh),
        u_ls=float(pos[f'liquid_{arm}'] @ uh),
        v_ls=float(pos[f'liquid_{arm}'] @ vh),
        bar_u=bar_u, plas_u=plas_u)


def group_u(geo: dict) -> dict:
    """{group: (u_lo, u_centre, u_hi)} of each wall read-out group."""
    out = {}
    for g in range(N_WALL_GROUP):
        us = [geo['bar_u'][BARS_PER_GROUP * g + 1 + i]
              for i in range(BARS_PER_GROUP)]
        out[g] = (min(us) - SIPM_BAR_HALF_U, float(np.mean(us)),
                  max(us) + SIPM_BAR_HALF_U)
    return out


def assert_same_geometry(geos: dict) -> None:
    """Raise unless every run placed arm A identically.

    The 27 July access moved things inside the chamber; it did not move the
    scintillator structure, and pooling 36 runs onto one map is only legitimate
    if that is true rather than assumed.
    """
    def sig(g):
        return json.dumps(dict(
            w=[round(g[k], 3) for k in ('w_strip', 'w_wall', 'w_plas', 'w_ls')],
            bar=[round(g['bar_u'][b], 3) for b in sorted(g['bar_u'])],
            plas=[round(g['plas_u'][n], 3) for n in sorted(g['plas_u'])]))
    seen = {}
    for run, g in geos.items():
        seen.setdefault(sig(g), []).append(run)
    if len(seen) > 1:
        groups = ' | '.join(f'{len(v)} runs ({v[0]}...)' for v in seen.values())
        raise ValueError(
            f'arm {ARM} is placed differently in different runs: {groups}\n'
            '  the pooled maps in this module assume one geometry. Split the '
            'pass by configuration rather than averaging two of them.')


# --------------------------------------------------------------------------- #
# tracks, projected
# --------------------------------------------------------------------------- #
def project(P0: np.ndarray, D: np.ndarray, geo: dict, w: float) -> tuple:
    """(u, v) where each track's LINE crosses the plane at global ``w``.

    The intersection of a line with a plane does not depend on which way the
    direction vector points, so no outward orientation is needed here -- and
    every layer sits outside the chamber, on the far side from the beam axis,
    so the crossing is on the physical path for any track that reaches it.
    """
    uh, vh, wh = geo['u_hat'], geo['v_hat'], geo['w_hat']
    dw = D @ wh
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (w - P0 @ wh) / np.where(np.abs(dw) > 1e-9, dw, np.nan)
    X = P0 + s[:, None] * D
    return X @ uh, X @ vh


def fired_sets(slim: pd.DataFrame, arm: str, window: tuple) -> tuple:
    """({(subrun, event): wall groups}, {...: plastic detn}, {...: LS}).

    ``window`` is ``(is_control, lo, hi)`` from :data:`WINDOWS` -- a SIGNED
    range on ``dt_ns`` and which hit collection to take it from.  Signed, not a
    magnitude: a magnitude window silently doubles its own width and that is
    precisely the error this module was built to avoid making.
    """
    ctrl, lo, hi = window
    it = slim[(slim.is_control == ctrl) & (slim.dt_ns >= lo)
              & (slim.dt_ns <= hi)]
    code = WAL_CODE[arm]

    def by_event(d, key):
        if not len(d):
            return {}
        g = d.groupby(['subrun', 'eventId'])[key].apply(
            lambda s: frozenset(int(x) for x in s))
        return dict(g)

    wal = it[it.det == code].assign(grp=lambda d: (d.detn - 1) // 2)
    pss = it[it.det == code + 4]
    liq = it[it.det == code + 8]
    return (by_event(wal, 'grp'), by_event(pss, 'detn'),
            by_event(liq, 'detn'))


def read_arm_slim(run: str, subruns, arm: str, slim_dir: Path | None
                  ) -> pd.DataFrame:
    """The exported slim, cut to ONE arm's three families and the windows used.

    `slim_export.read_export` returns every family of every arm over the full
    +-1000 ns, which for a 29 sub-run run is about a gigabyte and eleven twelfths
    of it is other arms.  This reads the same files, filters each one as it
    lands, and never holds the whole run.  Identical rows out, so no result
    depends on the difference.
    """
    d = Path(slim_dir) if slim_dir else paths.out('slim')
    code = WAL_CODE[arm]
    keep = {code, code + 4, code + 8}
    lo = min(min(w[1], w[2]) for w in WINDOWS.values())
    hi = max(max(w[1], w[2]) for w in WINDOWS.values())
    cols = ['eventId', 'det', 'detn', 'dt_ns', 'is_control', 'subrun']
    out, missing = [], []
    for sub in subruns:
        p = d / f'ntof_hits_{run}_{sub}.parquet'
        if not p.exists():
            missing.append(sub)
            continue
        x = pd.read_parquet(p, columns=cols)
        out.append(x[x.det.isin(keep) & (x.dt_ns >= lo) & (x.dt_ns <= hi)])
    if missing:
        raise FileNotFoundError(
            f'no exported slim for {run} sub-run(s) {", ".join(missing)} under '
            f'{d}\n  run `sept26_prelim_analysis.slim_export` for them, or '
            f'fetch the campaign slim pass. A sub-run that is silently skipped '
            f'here reads as a sub-run in which no scintillator ever fired.')
    x = pd.concat(out, ignore_index=True)
    x['subrun'] = x.subrun.astype(str)
    return x


def match_run(run: str, subruns, src: Path, slim_dir: Path | None,
              geo: dict) -> tuple:
    """(one row per gated arm-A track, tracks dropped for want of a k).

    The second return is not decoration: a run can lose every track to a
    missing angle calibration, and the count is what lets the report say so.
    """
    T = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {run}/{sub}')
        d = pd.read_parquet(p, columns=TRACK_COLS)
        T.append(d[(d.arm == geo['arm']) & d.gated].assign(subrun=sub))
    t = pd.concat(T, ignore_index=True)
    if not len(t):
        return pd.DataFrame(columns=KEEP_COLS), 0
    # EVERY quantity on this page is an extrapolation, so every one of them
    # needs the DIRECTION, and the direction needs the angle scale.  A run
    # whose stage-3 table was built before its `k_arm_<run>.json` existed
    # carries `angle_calibrated = False` and NULL `d_x/d_y/d_z`, and projecting
    # those gives a NaN crossing that lands on no channel -- which arrives at
    # the far end of this module as a confirmation rate of exactly 0.0 %, the
    # one wrong answer that looks like a measurement.  Refuse it here instead.
    n_all = len(t)
    t = t[t.angle_calibrated.astype('boolean').fillna(False)]
    if not len(t):
        raise ValueError(
            f'{run}: none of its {n_all:,} gated arm-{geo["arm"]} tracks are '
            f'angle-calibrated, so no track has a direction and nothing can be '
            f'extrapolated. Its k_arm_{run}.json may well exist now -- stage 3 '
            f'for this run predates it. Rebuild stage 3 for {run} '
            f'(campaign_tracks.py) and re-run; do NOT read the zero this would '
            f'otherwise produce as a rate.')
    n_uncal = n_all - len(t)
    t['run'] = run
    for c in ('x_slope_reliable', 'y_slope_reliable', 'drift_railed'):
        t[c] = t[c].astype('boolean').fillna(False).astype(bool)
    t['slope_ok'] = t.x_slope_reliable & t.y_slope_reliable
    t['n_trk'] = t.groupby(['subrun', 'event_id']).event_id.transform('size')
    t = t.rename(columns={'x_local': 'u_mm', 'y_local': 'v_mm'})

    P0 = t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    D = t[['d_x', 'd_y', 'd_z']].to_numpy(float)
    for tag, w in (('wall', geo['w_wall']), ('plas', geo['w_plas']),
                   ('ls', geo['w_ls'])):
        t[f'u_{tag}'], t[f'v_{tag}'] = project(P0, D, geo, w)
    # The FORMAL error, propagated along the lever arm.  It is a floor and is
    # labelled as one everywhere it is used: it carries the plane fit only, not
    # the angle scale `k`, not multiple scattering in 100-250 mm of air and
    # structure, and not the survey.  The width that matters is measured by
    # `edge_profile` instead.
    for tag, w in (('wall', geo['w_wall']), ('plas', geo['w_plas'])):
        lever = w - geo['w_strip']
        t[f'sig_u_{tag}'] = np.hypot(t.x_p0_err, lever * t.x_tan_err)
        t[f'sig_v_{tag}'] = np.hypot(t.y_p0_err, lever * t.y_tan_err)

    # --- predicted channel, per layer ---------------------------------------
    gu = group_u(geo)
    u_w = t.u_wall.to_numpy()
    grp = np.full(len(t), -1)
    for g, (lo, _c, hi) in gu.items():
        grp[(u_w >= lo) & (u_w < hi)] = g
    on_v_wall = np.abs(t.v_wall.to_numpy() - geo['v_mm']) <= SIPM_HALF_V
    t['grp_pred'] = grp
    t['on_wall'] = (grp >= 0) & on_v_wall
    # the instrumented bar, kept because the wall's two ends are the y handle
    bars = np.array([geo['bar_u'][b] for b in sorted(geo['bar_u'])])
    bar_idx = np.argmin(np.abs(u_w[:, None] - bars[None, :]), axis=1)
    on_bar = np.abs(u_w - bars[bar_idx]) <= SIPM_BAR_HALF_U
    t['bar_pred'] = np.where(on_bar & on_v_wall, bar_idx + 1, -1)
    # distance to the nearer boundary of the predicted group, and the group on
    # the other side of it -- this pair is what makes a TOLERANT match possible
    # without re-reading the data.
    d_edge = np.full(len(t), np.nan)
    g_alt = np.full(len(t), -1)
    for g, (lo, _c, hi) in gu.items():
        m = grp == g
        dl, dh = u_w[m] - lo, hi - u_w[m]
        d_edge[m] = np.minimum(dl, dh)
        g_alt[m] = np.where(dl < dh, g - 1, g + 1)
    g_alt[(g_alt < 0) | (g_alt >= N_WALL_GROUP)] = -1
    t['d_edge_wall'] = d_edge
    t['grp_alt'] = g_alt

    pn = np.array(sorted(geo['plas_u']))
    pu = np.array([geo['plas_u'][n] for n in pn])
    p_idx = np.argmin(np.abs(t.u_plas.to_numpy()[:, None] - pu[None, :]), axis=1)
    d_pl = np.abs(t.u_plas.to_numpy() - pu[p_idx])
    on_pl = ((d_pl <= PLASTIC_HALF_U)
             & (np.abs(t.v_plas.to_numpy() - geo['v_mm']) <= PLASTIC_HALF_V))
    t['plas_pred'] = np.where(on_pl, pn[p_idx], -1)
    t['d_edge_plas'] = PLASTIC_HALF_U - d_pl
    t['on_plas'] = on_pl
    t['on_ls'] = ((np.abs(t.u_ls.to_numpy() - geo['u_ls']) <= LS_HALF_U)
                  & (np.abs(t.v_ls.to_numpy() - geo['v_ls']) <= LS_HALF_V))

    # --- what actually fired -------------------------------------------------
    # The identical test is run in every window of `WINDOWS`, so the accidental
    # floor of each rate is measured by the same code path that measured the
    # rate.  Nothing here is modelled.
    slim = read_arm_slim(run, subruns, geo['arm'], slim_dir)
    key = list(zip(t.subrun, t.event_id))
    on_w = t.on_wall.to_numpy()
    on_p = t.on_plas.to_numpy()
    gp, pp, ga = (t.grp_pred.to_numpy(), t.plas_pred.to_numpy(),
                  t.grp_alt.to_numpy())
    for name, window in WINDOWS.items():
        W, P, L = fired_sets(slim, geo['arm'], window)
        wf = [W.get(k, frozenset()) for k in key]
        pf = [P.get(k, frozenset()) for k in key]
        lf = [L.get(k, frozenset()) for k in key]
        sfx = '' if name == PRIMARY else f'_{name}'
        t[f'match_wall{sfx}'] = np.fromiter(
            (g >= 0 and g in s for g, s in zip(gp, wf)), bool, len(t)) & on_w
        t[f'match_plas{sfx}'] = np.fromiter(
            (p > 0 and p in s for p, s in zip(pp, pf)), bool, len(t)) & on_p
        t[f'wall_hit{sfx}'] = [len(s) > 0 for s in wf]
        t[f'plas_hit{sfx}'] = [len(s) > 0 for s in pf]
        if name != PRIMARY:
            continue
        t['match_wall_alt'] = np.fromiter(
            (g >= 0 and g in s for g, s in zip(ga, wf)), bool, len(t)) & on_w
        t['match_plas_alt'] = np.fromiter(
            (p > 0 and any(x != p for x in s) for p, s in zip(pp, pf)),
            bool, len(t)) & on_p
        t['n_grp_fired'] = [len(s) for s in wf]
        t['grp_fired'] = [min(s) if len(s) == 1 else -1 for s in wf]
        t['n_plas_fired'] = [len(s) for s in pf]
        t['plas_fired'] = [min(s) if len(s) == 1 else -1 for s in pf]
        t['ls_fired'] = [len(s) > 0 for s in lf]
        # residual to the fired channel, single-channel events only: the
        # continuous version of the boolean, and what the edge fit uses.
        gc = {g: c for g, (_l, c, _h) in gu.items()}
        gf = t.grp_fired.to_numpy()
        t['res_u_wall'] = np.where(
            gf >= 0, u_w - np.array([gc.get(int(g), np.nan) for g in gf]),
            np.nan)
        pf1 = t.plas_fired.to_numpy()
        t['res_u_plas'] = np.where(
            pf1 > 0,
            t.u_plas.to_numpy()
            - np.array([geo['plas_u'].get(int(p), np.nan) for p in pf1]),
            np.nan)
    return t[KEEP_COLS].reset_index(drop=True), n_uncal


# --------------------------------------------------------------------------- #
# the measured position tolerance
# --------------------------------------------------------------------------- #
def edge_profile(M: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """P(group g is the one that fired) against the predicted crossing u.

    Restricted to events with exactly ONE arm-A track and exactly ONE wall
    group lit, which is the only sample in which the association is unambiguous
    without assuming the answer.  The width of the rise and fall of each curve
    is the extrapolation resolution the wall can actually see, and it is the
    number the tolerant match uses.
    """
    d = M[(M.n_trk == 1) & (M.n_grp_fired == 1) & M.on_wall]
    if not len(d):
        return pd.DataFrame(columns=['group', 'u', 'n', 'p', 'p_err'])
    gu = group_u(geo)
    idx = np.digitize(d.u_wall.to_numpy(), EDGE_EDGES) - 1
    c = 0.5 * (EDGE_EDGES[:-1] + EDGE_EDGES[1:])
    rows = []
    for g in range(N_WALL_GROUP):
        hit = (d.grp_fired.to_numpy() == g)
        for i in range(len(c)):
            m = idx == i
            n = int(m.sum())
            if n < 10:
                continue
            k = int(hit[m].sum())
            p = k / n
            rows.append(dict(group=g, u=float(c[i]), n=n, k=k, p=p,
                             p_err=float(np.sqrt(max(p * (1 - p), 1e-6) / n)),
                             u_lo=gu[g][0], u_hi=gu[g][2]))
    return pd.DataFrame(rows)


def _erf_edge(u, u0, sigma, sign):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * (1.0 + sign * erf((u - u0) / (sqrt(2.0) * sigma)))


def edge_width(E: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """Fit each group boundary with a plateau x error function -> sigma, mm.

    Four groups x two boundaries = eight edges, of which the two outermost are
    the wall's own ends and are fitted anyway so the report can show that they
    behave like the internal ones.  A boundary whose fit does not converge is
    reported as NaN rather than dropped.
    """
    from scipy.optimize import curve_fit
    gu = group_u(geo)
    rows = []
    for g in range(N_WALL_GROUP):
        d = E[E.group == g].sort_values('u')
        if len(d) < 8:
            continue
        lo, _c, hi = gu[g]
        for side, u_edge, sign in (('low', lo, +1.0), ('high', hi, -1.0)):
            m = np.abs(d.u - u_edge) <= 80.0
            x, y, w = (d.u[m].to_numpy(), d.p[m].to_numpy(),
                       d.p_err[m].to_numpy())
            if len(x) < 5:
                continue
            def f(u, u0, s, a):
                return a * _erf_edge(u, u0, s, sign)
            try:
                p, cov = curve_fit(f, x, y, p0=[u_edge, 20.0, 0.9],
                                   sigma=np.maximum(w, 0.01),
                                   bounds=([u_edge - 60, 2.0, 0.2],
                                           [u_edge + 60, 120.0, 1.0]),
                                   maxfev=20000)
                err = float(np.sqrt(np.diag(cov))[1])
            except Exception:
                p, err = [np.nan] * 3, np.nan
            rows.append(dict(group=g, side=side, u_edge=float(u_edge),
                             u_fit=float(p[0]), sigma_mm=float(p[1]),
                             sigma_err=err, plateau=float(p[2]),
                             n_bins=int(len(x)),
                             interior=bool(0 < g < N_WALL_GROUP - 1
                                           or (g == 0 and side == 'high')
                                           or (g == N_WALL_GROUP - 1
                                               and side == 'low'))))
    return pd.DataFrame(rows)


#: The interior group boundaries are the only ones with data on both sides, and
#: they are the ones the angle-scale test uses.  Two adjacent groups each see
#: the same boundary, so every boundary is measured twice.
def interior_boundaries(geo: dict) -> list:
    gu = group_u(geo)
    return [0.5 * (gu[g][2] + gu[g + 1][0]) for g in range(N_WALL_GROUP - 1)]


def _fit_edge(u, p, perr, u0, sign, half=80.0):
    """(u_fit, err) of one plateau x error-function edge, or (nan, nan)."""
    from scipy.optimize import curve_fit
    m = np.abs(u - u0) <= half
    if m.sum() < 5:
        return np.nan, np.nan
    def f(x, c, sg, a):
        return a * _erf_edge(x, c, sg, sign)
    try:
        q, cov = curve_fit(f, u[m], p[m], p0=[u0, 20.0, 0.9],
                           sigma=np.maximum(perr[m], 0.01),
                           bounds=([u0 - 60, 2.0, 0.2], [u0 + 60, 120.0, 1.0]),
                           maxfev=20000)
        return float(q[0]), float(np.sqrt(np.diag(cov))[0])
    except Exception:
        return np.nan, np.nan


def edge_vs_tan(M: pd.DataFrame, geo: dict, n_bins: int = 5) -> pd.DataFrame:
    """Where each wall boundary APPEARS to be, in bins of the track's slope.

    THIS IS AN ANGLE-SCALE MEASUREMENT, and it is independent of the target
    imaging that currently sets ``k``.

    The predicted crossing is ``u_wall = u_strip + tan * L`` with L the 97.4 mm
    lever arm from the strip plane to the wall.  If the calibrated ``tan`` is
    wrong by a factor ``1 + eps`` then

        u_wall(predicted) - u_wall(true) = eps * tan * L

    -- a displacement PROPORTIONAL TO THE TRACK'S OWN SLOPE and zero for a
    normal-incidence track.  The wall's group boundaries are surveyed, fixed
    and known, so fitting where a boundary appears to sit separately in bins of
    ``tan`` and regressing the shift on the bin's mean ``tan`` measures
    ``eps * L`` directly, with no assumption about where the tracks came from.

    A pure survey or alignment error gives the same shift in every bin and so
    contributes only to the intercept.  That is the whole point of the split:
    the two defects are degenerate in the pooled fit and separated here.
    """
    d = M[(M.n_trk == 1) & (M.n_grp_fired == 1) & M.on_wall].copy()
    if len(d) < 5000:
        return pd.DataFrame()
    gu = group_u(geo)
    qs = np.nanquantile(d.tanx, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    d['tbin'] = np.digitize(d.tanx, qs[1:-1])
    rows = []
    for b, g in d.groupby('tbin'):
        idx = np.digitize(g.u_wall.to_numpy(), EDGE_EDGES) - 1
        c = 0.5 * (EDGE_EDGES[:-1] + EDGE_EDGES[1:])
        for grp in range(N_WALL_GROUP):
            hit = (g.grp_fired.to_numpy() == grp)
            u, p, pe = [], [], []
            for i in range(len(c)):
                m = idx == i
                n = int(m.sum())
                if n < 10:
                    continue
                f = hit[m].mean()
                u.append(c[i]), p.append(f)
                pe.append(np.sqrt(max(f * (1 - f), 1e-6) / n))
            if len(u) < 8:
                continue
            u, p, pe = np.array(u), np.array(p), np.array(pe)
            for side, u0, sign in (('low', gu[grp][0], +1.0),
                                   ('high', gu[grp][2], -1.0)):
                if not any(abs(u0 - b0) < 1.0 for b0 in interior_boundaries(geo)):
                    continue                      # skip the wall's own ends
                uf, ue = _fit_edge(u, p, pe, u0, sign)
                rows.append(dict(tbin=int(b), group=grp, side=side,
                                 n=int(len(g)), tan_mean=float(g.tanx.mean()),
                                 tan_lo=float(qs[b]), tan_hi=float(qs[b + 1]),
                                 u_nominal=float(u0), u_fit=uf, u_fit_err=ue,
                                 shift_mm=uf - u0))
    return pd.DataFrame(rows)


def plastic_edge_vs_tan(M: pd.DataFrame, geo: dict, n_bins: int = 8
                        ) -> pd.DataFrame:
    """The same test at the plastic, whose lever arm is nearly twice the wall's.

    THIS IS THE DECISIVE CHECK ON :func:`edge_vs_tan`.  The plastic has exactly
    one boundary -- the gap between the L and R bars, which sits on the chamber
    centre because the plastics are mounted on the MM and the wall is not --
    but it sits 190.6 mm past the strip plane against the wall's 97.4 mm.

    A wrong angle scale displaces the predicted crossing by ``eps * tan * L``,
    so the shift per unit ``tan`` must be nearly TWICE as large here, while
    ``eps`` itself comes out the same.  Any defect that is not proportional to
    the lever arm -- a survey offset, a mis-mapped read-out order, a plane-fit
    bias -- cannot do that.
    """
    d = M[(M.n_trk == 1) & (M.n_plas_fired == 1) & M.on_plas].copy()
    if len(d) < 5000:
        return pd.DataFrame()
    u0 = 0.5 * (geo['plas_u'][1] + geo['plas_u'][2])
    qs = np.nanquantile(d.tanx, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    d['tbin'] = np.digitize(d.tanx, qs[1:-1])
    edges = np.arange(u0 - 200.0, u0 + 200.1, 10.0)
    c = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for b, g in d.groupby('tbin'):
        idx = np.digitize(g.u_plas.to_numpy(), edges) - 1
        hit = (g.plas_fired.to_numpy() == 2)      # the +u bar
        u, p, pe = [], [], []
        for i in range(len(c)):
            m = idx == i
            n = int(m.sum())
            if n < 10:
                continue
            f = hit[m].mean()
            u.append(c[i]), p.append(f)
            pe.append(np.sqrt(max(f * (1 - f), 1e-6) / n))
        if len(u) < 8:
            continue
        uf, ue = _fit_edge(np.array(u), np.array(p), np.array(pe), u0, +1.0,
                           half=120.0)
        rows.append(dict(tbin=int(b), group=-1, side='gap', n=int(len(g)),
                         tan_mean=float(g.tanx.mean()),
                         tan_lo=float(qs[b]), tan_hi=float(qs[b + 1]),
                         u_nominal=float(u0), u_fit=uf, u_fit_err=ue,
                         shift_mm=uf - u0))
    return pd.DataFrame(rows)


def angle_scale(EV: pd.DataFrame, lever: float) -> dict:
    """Regress the boundary shift on the bin's mean slope -> eps and k_wall.

    ``eps`` is the fractional error in the calibrated tangent, so the angle
    scale the wall prefers is ``k_wall = k_applied * (1 + eps)``: a positive
    ``eps`` means the tangents in the track table are too LARGE and ``k`` is
    too small.  The intercept absorbs a rigid survey offset and is reported
    beside it rather than folded in.
    """
    d = EV.dropna(subset=['u_fit', 'shift_mm'])
    d = d[np.isfinite(d.u_fit_err) & (d.u_fit_err < 30.0)]
    if len(d) < 4:                    # two free parameters, two degrees left
        return dict(n=int(len(d)), eps=np.nan)
    x = d.tan_mean.to_numpy()
    y = d.shift_mm.to_numpy()
    w = 1.0 / np.maximum(d.u_fit_err.to_numpy(), 1.0) ** 2
    X = np.c_[x, np.ones_like(x)]
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ (X.T @ W @ y)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    chi2 = float((w * resid ** 2).sum())
    scale = max(chi2 / dof, 1.0)                  # inflate on a poor fit
    err = np.sqrt(np.diag(cov) * scale)
    eps = float(beta[0] / lever)
    return dict(n=int(len(d)), slope_mm=float(beta[0]),
                slope_err=float(err[0]), intercept_mm=float(beta[1]),
                intercept_err=float(err[1]), lever_mm=float(lever),
                eps=eps, eps_err=float(err[0] / lever),
                k_ratio=1.0 + eps, chi2=chi2, dof=int(dof),
                chi2dof=chi2 / dof)


#: Selections the angle-scale measurement is repeated on.  ``pointing`` is
#: deliberately in the list and deliberately flagged: the beam-axis cut is
#: computed FROM the reconstructed direction, so selecting on it selects on
#: ``tan`` in a way that is correlated with position, and the test is not valid
#: there.  Leaving it out would hide that; leaving it in unmarked would invite
#: reading it as a disagreement.
ROBUST_CUTS = ('all', 'fiducial', 'slope', 'fiducial+slope', 'pointing')
BIASED_CUTS = ('pointing',)


def angle_scale_robustness(M: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """The angle-scale measurement repeated on each selection, both layers.

    A 33 % error in a calibrated quantity is a large claim, and the thing that
    would most easily fake it is a selection that correlates ``tan`` with
    position.  Repeating the fit on samples that cut on position (fiducial),
    on fit quality (slope) and on both is the check, and the answer has to be
    the same number each time or it is not an angle scale.
    """
    lev = {'wall': geo['w_wall'] - geo['w_strip'],
           'plas': geo['w_plas'] - geo['w_strip']}
    rows = []
    for name in ROBUST_CUTS:
        m = np.ones(len(M), bool)
        for part in name.split('+'):
            m &= _sel(M, part)
        g = M[m]
        if len(g) < 20000:
            continue
        for lay, fn in (('wall', edge_vs_tan), ('plas', plastic_edge_vs_tan)):
            fit = angle_scale(fn(g, geo), lev[lay])
            # `fit` carries its own `n` (the number of fitted edges) and its
            # own `lever_mm`, so the sample size goes in under a name of its
            # own and the lever is not restated.
            rows.append(dict(selection=name, layer=lay,
                             n_tracks=int(len(g)),
                             biased=name in BIASED_CUTS, **fit))
    return pd.DataFrame(rows)


def group_confusion(M: pd.DataFrame) -> pd.DataFrame:
    """Predicted wall group against the single group that fired.

    The diagonal is the check on the ASCENDING read-out convention, which is
    the one piece of this geometry the DAQ config does not record.  A
    descending map would put the whole population on the anti-diagonal.
    """
    d = M[(M.n_grp_fired == 1) & M.on_wall]
    if not len(d):
        return pd.DataFrame()
    x = pd.crosstab(d.grp_pred, d.grp_fired)
    x.index.name, x.columns.name = 'predicted', 'fired'
    return x.reset_index()


def rail_census(M: pd.DataFrame) -> pd.DataFrame:
    """How much of the track table lands outside the chamber, and does it confirm?

    FOUND BY LOOKING AT THE WALL PROJECTION, which is the point of drawing it.
    The map carries a hard horizontal stripe at v ~ -195 mm that no physical
    feature of the apparatus sits at, and the strip-plane histogram behind it
    shows the cause: the fitted y position RAILS just outside the active area.
    The chamber is 340 mm tall, so |v| <= 170 is the whole of it.

    The scintillators are an EXTERNAL arbiter of what those tracks are, and
    they are unambiguous about it: a railed track is confirmed at a small
    fraction of the rate of one inside the chamber, on both layers, with the
    accidental floor going the other way.  This is not something the
    reconstruction can be asked about from inside itself.

    Reported, not cut.  Whether a railed y should be gated out is a decision
    for `wft`'s gate and not for this page, which only has to say how large it
    is and what it does to everything measured here.
    """
    rows = []
    v = M.v_mm.to_numpy()
    sets = (('inside the active area', np.abs(v) <= FIDUCIAL_V),
            ('outside it in v', np.abs(v) > FIDUCIAL_V),
            ('in the v rail', (v >= RAIL_V[0]) & (v <= RAIL_V[1])),
            ('outside it in u', M.u_mm.abs().to_numpy() > FIDUCIAL_U))
    n_all = len(M)
    for name, m in sets:
        g = M[m]
        npw, npp = int(g.on_wall.sum()), int(g.on_plas.sum())
        rows.append(dict(
            sample=name, n=int(len(g)), frac_of_all=len(g) / max(n_all, 1),
            n_pred_wall=npw, n_pred_plas=npp,
            frac_match_wall=int(g.match_wall.sum()) / npw if npw else np.nan,
            frac_match_plas=int(g.match_plas.sum()) / npp if npp else np.nan,
            frac_ctrl_wall=int(g.match_wall_ctrl.sum()) / npw if npw else np.nan,
            frac_slope=float(g.slope_ok.mean()) if len(g) else np.nan))
    return pd.DataFrame(rows)


def v_profile(M: pd.DataFrame) -> pd.DataFrame:
    """Tracks and their confirmation rate against v, in 10 mm bins.

    The abscissa the rail is visible on, with the confirmation rate beside it
    so the figure can show that the pile-up is not confirmed rather than only
    that it is there.
    """
    e = np.arange(-230.0, 230.1, 10.0)
    i = np.digitize(M.v_mm.to_numpy(), e) - 1
    ok = (i >= 0) & (i < len(e) - 1)
    d = M[ok].assign(iv=i[ok])
    g = d.groupby('iv').agg(n=('v_mm', 'size'), n_pred=('on_wall', 'sum'),
                            n_match=('match_wall', 'sum'),
                            n_ctrl=('match_wall_ctrl', 'sum')).reset_index()
    g['v'] = 0.5 * (e[g.iv] + e[g.iv + 1])
    g['frac_match'] = g.n_match / g.n_pred.replace(0, np.nan)
    g['frac_ctrl'] = g.n_ctrl / g.n_pred.replace(0, np.nan)
    g.loc[g.n < MIN_CELL, ['frac_match', 'frac_ctrl']] = np.nan
    return g


# --------------------------------------------------------------------------- #
# the maps
# --------------------------------------------------------------------------- #
def _cells(u, v, ue, ve):
    iu = np.digitize(u, ue) - 1
    iv = np.digitize(v, ve) - 1
    ok = (iu >= 0) & (iu < len(ue) - 1) & (iv >= 0) & (iv < len(ve) - 1)
    return iu, iv, ok


def surface_map(M: pd.DataFrame, plane: str, ue, ve) -> pd.DataFrame:
    """Per-cell counts on one plane, in the long form the figures consume.

    One row per cell, carrying the denominators (tracks, tracks that can be
    predicted at all) beside every numerator, so a fraction is never formed
    without its own denominator in the same row.  The per-channel columns
    ``n_w0..n_w3`` and ``n_p1``/``n_p2`` are what colour-codes a cell by WHICH
    scintillator confirmed it.
    """
    cols = {'mm': ('u_mm', 'v_mm'), 'wall': ('u_wall', 'v_wall'),
            'plas': ('u_plas', 'v_plas')}[plane]
    iu, iv, ok = _cells(M[cols[0]].to_numpy(), M[cols[1]].to_numpy(), ue, ve)
    d = M[ok].copy()
    d['iu'], d['iv'] = iu[ok], iv[ok]
    agg = d.groupby(['iu', 'iv']).agg(
        n=('u_mm', 'size'),
        n_pred_w=('on_wall', 'sum'), n_match_w=('match_wall', 'sum'),
        n_match_w_alt=('match_wall_alt', 'sum'),
        n_match_w_ctrl=('match_wall_ctrl', 'sum'),
        n_match_w_core=('match_wall_core', 'sum'),
        n_hit_w=('wall_hit', 'sum'),
        n_pred_p=('on_plas', 'sum'), n_match_p=('match_plas', 'sum'),
        n_match_p_ctrl=('match_plas_ctrl', 'sum'),
        n_match_p_core=('match_plas_core', 'sum'),
        n_hit_p=('plas_hit', 'sum'),
        n_ls=('ls_fired', 'sum'), n_pred_ls=('on_ls', 'sum'),
    ).reset_index()
    # matched-by-channel: the colour code
    for g in range(N_WALL_GROUP):
        s = d[d.match_wall & (d.grp_pred == g)].groupby(['iu', 'iv']).size()
        agg[f'n_w{g}'] = agg.set_index(['iu', 'iv']).index.map(s).fillna(0).astype(int)
    for n in (1, 2):
        s = d[d.match_plas & (d.plas_pred == n)].groupby(['iu', 'iv']).size()
        agg[f'n_p{n}'] = agg.set_index(['iu', 'iv']).index.map(s).fillna(0).astype(int)
    agg['plane'] = plane
    agg['u'] = 0.5 * (ue[agg.iu] + ue[agg.iu + 1])
    agg['v'] = 0.5 * (ve[agg.iv] + ve[agg.iv + 1])
    return agg


MAPS = {'mm': (MM_U_EDGES, MM_V_EDGES),
        'wall': (WALL_U_EDGES, WALL_V_EDGES),
        'plas': (PLAS_U_EDGES, PLAS_V_EDGES)}


def all_maps(M: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([surface_map(M, k, *v) for k, v in MAPS.items()],
                     ignore_index=True)


# --------------------------------------------------------------------------- #
# the rates
# --------------------------------------------------------------------------- #
SELECTIONS = ('all', 'fiducial', 'slope', 'single', 'pointing')


def _sel(M: pd.DataFrame, name: str) -> np.ndarray:
    if name == 'all':
        return np.ones(len(M), bool)
    if name == 'fiducial':
        return ((M.u_mm.abs() <= FIDUCIAL_U)
                & (M.v_mm.abs() <= FIDUCIAL_V)).to_numpy(copy=True)
    if name == 'slope':
        return M.slope_ok.to_numpy(copy=True)
    if name == 'single':
        return (M.n_trk == 1).to_numpy(copy=True)
    if name == 'pointing':
        return (M.dca_axis_mm < 30.0).to_numpy(copy=True)
    raise KeyError(name)


def rates(M: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Confirmation rates, LONG FORM: one row per selection, layer and window.

    Long form because the interesting comparison is a rate against its own
    accidental floor, and those are the same quantity measured in two windows
    -- putting them in one row would invite reading one as a correction to the
    other rather than as the same test run twice.

    Every denominator is named in the row that uses it:

      ``frac_pred``       of all tracks, how many land on an instrumented
                          channel and can be confirmed at all
      ``frac_match_pred`` THE HEADLINE -- of those, how many point at a channel
                          that fired
      ``frac_match_hit``  of those whose layer fired at all, how many point at
                          the right channel: the pointing, with the layer's own
                          efficiency divided out
      ``excess``          ``frac_match_pred`` minus the same quantity in the
                          matching control window, which is what is left when
                          the accidental floor is taken off
    """
    rows = []
    keys = [(None, M)] if by is None else list(M.groupby(by))
    LAYERS = (('wall', 'on_wall'), ('plas', 'on_plas'))
    for k, d in keys:
        for sel in SELECTIONS:
            g = d[_sel(d, sel)]
            n = len(g)
            if not n:
                continue
            for lay, pred in LAYERS:
                npred = int(g[pred].sum())
                for win in WINDOWS:
                    sfx = '' if win == PRIMARY else f'_{win}'
                    nh = int((g[pred] & g[f'{lay}_hit{sfx}']).sum())
                    nm = int(g[f'match_{lay}{sfx}'].sum())
                    r = dict(selection=sel, layer=lay, window=win, n=n,
                             n_pred=npred, n_hit=nh, n_match=nm,
                             frac_pred=npred / n,
                             frac_hit=nh / npred if npred else np.nan,
                             frac_match_pred=nm / npred if npred else np.nan,
                             frac_match_hit=nm / nh if nh else np.nan,
                             err_match_pred=(np.sqrt(max(nm, 1)) / npred
                                             if npred else np.nan))
                    if by is not None:
                        r[by] = k
                    rows.append(r)
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    # the excess over each window's own control, attached to the signal row
    idx = ['selection', 'layer'] + ([by] if by is not None else [])
    for sig, ctl in (('prod', 'ctrl'), ('prod', 'off'),
                     ('core', 'ctrl_core'), ('core', 'off_core')):
        c = (R[R.window == ctl].set_index(idx).frac_match_pred
             .rename(f'floor_{ctl}'))
        R = R.merge(c, left_on=idx, right_index=True, how='left')
    R['floor'] = R.floor_ctrl.where(R.window == 'prod', R.floor_ctrl_core)
    R['excess'] = R.frac_match_pred - R.floor
    R.loc[~R.window.isin(('prod', 'core')), ['floor', 'excess']] = np.nan
    return R


def both_layers(M: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """How often BOTH layers confirm the same track, and how often neither.

    Two independent layers at different depths agreeing is far stronger than
    either alone, and the four-way partition is the only place the report can
    say what fraction of tracks the scintillators corroborate outright.
    """
    rows = []
    keys = [(None, M)] if by is None else list(M.groupby(by))
    for k, d in keys:
        for sel in SELECTIONS:
            g = d[_sel(d, sel) & d.on_wall & d.on_plas]
            n = len(g)
            if not n:
                continue
            w, p = g.match_wall.to_numpy(), g.match_plas.to_numpy()
            wc = g.match_wall_ctrl.to_numpy()
            pc = g.match_plas_ctrl.to_numpy()
            r = dict(selection=sel, n=n,
                     both=int((w & p).sum()) / n,
                     wall_only=int((w & ~p).sum()) / n,
                     plas_only=int((~w & p).sum()) / n,
                     neither=int((~w & ~p).sum()) / n,
                     both_ctrl=int((wc & pc).sum()) / n,
                     n_both=int((w & p).sum()))
            if by is not None:
                r[by] = k
            rows.append(r)
    return pd.DataFrame(rows)


def tolerant(M: pd.DataFrame, sigma: float, n_sigma: float = EDGE_N_SIGMA
             ) -> pd.Series:
    """The match with the MEASURED position tolerance folded in.

    A track whose predicted crossing sits within ``n_sigma`` edge widths of a
    group boundary is confirmed by EITHER of the two groups that boundary
    separates, because at that distance the extrapolation does not choose
    between them.  Everywhere else the strict match stands.
    """
    near = M.d_edge_wall.to_numpy() < n_sigma * sigma
    return pd.Series(M.match_wall.to_numpy()
                     | (near & M.match_wall_alt.to_numpy()), index=M.index)


# --------------------------------------------------------------------------- #
# per run
# --------------------------------------------------------------------------- #
def one_run(run: str, src: str, reco: str, slim_dir: str | None,
            out_dir: str, reuse: bool = False) -> tuple:
    """(run, maps, rates, edge, confusion, headline, error).

    ``reuse`` rebuilds this run's aggregates from the per-track parquet a
    previous pass wrote, without touching the slim.  The matching is the
    expensive step and its result is already on disk, so a change to a summary
    -- a new table, a different binning -- should not cost another read of ten
    gigabytes of n_TOF hits.  It is not a cache in the dangerous sense: the
    parquet is a stated product with its own schema, and `--from-tracks` says
    on the tin that no track was re-matched.
    """
    try:
        src, reco = Path(src), Path(reco)
        od = Path(out_dir) / 'tracks'
        cached = od / f'scint_{run}.parquet'
        if reuse and cached.exists():
            M = pd.read_parquet(cached)
            geo = layer_geometry(run)
            have, dropped, n_uncal = [], {}, -1
            if not len(M):
                return run, None, None, None, None, None, 'empty cached table'
        else:
            subs, dropped = subruns_of(reco, run)
            if not subs:
                return run, None, None, None, None, None, 'no usable sub-runs'
            have = [s for s in subs
                    if (src / f'tracks_{run}_{s}.parquet').exists()]
            if not have:
                return run, None, None, None, None, None, 'no stage-3 tracks'
            geo = layer_geometry(run)
            M, n_uncal = match_run(run, have, src,
                                   Path(slim_dir) if slim_dir else None, geo)
            if not len(M):
                return run, None, None, None, None, None, 'no gated arm-A tracks'
            od.mkdir(parents=True, exist_ok=True)
            M.to_parquet(cached, index=False, compression='snappy')

        MP = all_maps(M).assign(run=run)
        R = rates(M).assign(run=run, condition=condition(run),
                            k_block=in_block(run))
        E = edge_profile(M, geo).assign(run=run)
        C = group_confusion(M).assign(run=run)
        one = M[_sel(M, 'all')]
        H = dict(run=run, n_subruns=len(have) or int(M.subrun.nunique()),
                 dropped_subruns=dropped,
                 n_tracks=int(len(M)), n_uncalibrated=int(n_uncal),
                 n_pred_wall=int(M.on_wall.sum()),
                 n_match_wall=int(M.match_wall.sum()),
                 n_pred_plas=int(M.on_plas.sum()),
                 n_match_plas=int(M.match_plas.sum()),
                 frac_match_wall=float(M.match_wall.sum()
                                       / max(M.on_wall.sum(), 1)),
                 frac_match_plas=float(M.match_plas.sum()
                                       / max(M.on_plas.sum(), 1)),
                 frac_ctrl_wall=float(M.match_wall_ctrl.sum()
                                      / max(M.on_wall.sum(), 1)),
                 condition=condition(run), k_block=in_block(run),
                 n_single=int((one.n_trk == 1).sum()))
        return run, MP, R, E, C, H, ''
    except Exception:
        return run, None, None, None, None, None, traceback.format_exc(limit=4)


def _sum_maps(MP: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in MP.columns if c.startswith('n')]
    G = MP.groupby(['plane', 'iu', 'iv'], as_index=False)[num].sum()
    ref = MP.drop_duplicates(['plane', 'iu', 'iv'])[['plane', 'iu', 'iv',
                                                     'u', 'v']]
    G = G.merge(ref, on=['plane', 'iu', 'iv'], how='left')
    G['frac_match_w'] = G.n_match_w / G.n_pred_w.replace(0, np.nan)
    G['frac_match_w_ctrl'] = G.n_match_w_ctrl / G.n_pred_w.replace(0, np.nan)
    G['frac_match_p'] = G.n_match_p / G.n_pred_p.replace(0, np.nan)
    G['frac_match_p_ctrl'] = G.n_match_p_ctrl / G.n_pred_p.replace(0, np.nan)
    G['frac_match_w_net'] = G.frac_match_w - G.frac_match_w_ctrl
    G['frac_match_p_net'] = G.frac_match_p - G.frac_match_p_ctrl
    G.loc[G.n < MIN_CELL,
          ['frac_match_w', 'frac_match_p', 'frac_match_w_ctrl',
           'frac_match_p_ctrl', 'frac_match_w_net',
           'frac_match_p_net']] = np.nan
    wc = [f'n_w{g}' for g in range(N_WALL_GROUP)]
    G['grp_mode'] = np.where(G[wc].sum(1) > 0, np.argmax(G[wc].to_numpy(), 1),
                             -1)
    G['grp_purity'] = (G[wc].max(1) / G[wc].sum(1).replace(0, np.nan))
    pc = ['n_p1', 'n_p2']
    G['plas_mode'] = np.where(G[pc].sum(1) > 0,
                              np.argmax(G[pc].to_numpy(), 1) + 1, -1)
    G['plas_purity'] = (G[pc].max(1) / G[pc].sum(1).replace(0, np.nan))
    return G


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.out('stage3_fullpass')))
    ap.add_argument('--reco', default=str(paths.out('reco_fullpass')))
    ap.add_argument('--slim', default=None)
    ap.add_argument('--runs', default='')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--include-pre-access', action='store_true')
    ap.add_argument('--from-tracks', action='store_true',
                    help='rebuild every summary from the per-run tables a '
                         'previous pass wrote, without re-reading the slim. '
                         'No track is re-matched; `n_uncalibrated` reads -1 '
                         'because it is a property of that pass, not of these '
                         'tables.')
    a = ap.parse_args()

    src = Path(paths.require(a.src, 'the stage-3 track table'))
    reco = Path(paths.require(a.reco, 'the full-pass reco tree'))
    runs = ([r for r in a.runs.split(',') if r] or
            sorted((p.name for p in reco.iterdir()
                    if p.is_dir() and p.name.startswith('run_')),
                   key=run_number))
    if not a.include_pre_access:
        runs = [r for r in runs if r not in PRE_ACCESS_RUNS]
    od = paths.out('det_a_scint')
    print(f'detector {ARM} -> scintillators, {len(runs)} run(s)\n')

    geos = {r: layer_geometry(r) for r in runs}
    assert_same_geometry(geos)
    geo = geos[runs[0]]
    print('arm A layers, from the DAQ config (global w, mm):')
    print(f'  strip plane {geo["w_strip"]:.1f}   SiPM wall {geo["w_wall"]:.1f}'
          f'   plastic {geo["w_plas"]:.1f}   LS {geo["w_ls"]:.1f}')
    print('  lever arms past the strips: wall '
          f'{geo["w_wall"] - geo["w_strip"]:.1f}, plastic '
          f'{geo["w_plas"] - geo["w_strip"]:.1f}, LS '
          f'{geo["w_ls"] - geo["w_strip"]:.1f}\n')

    if a.from_tracks:
        runs = [r for r in runs if (od / 'tracks' / f'scint_{r}.parquet').exists()]
        print(f'--from-tracks: rebuilding summaries for {len(runs)} run(s) '
              f'from stored per-track tables; nothing is re-matched\n')
    args = (str(src), str(reco), a.slim, str(od), a.from_tracks)
    if a.jobs <= 1:
        results = [one_run(r, *args) for r in runs]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            fut = {ex.submit(one_run, r, *args): r for r in runs}
            results = [f.result() for f in as_completed(fut)]

    MPs, Rs, Es, Cs, Hs, failed = [], [], [], [], [], {}
    for run, MP, R, E, C, H, err in sorted(results,
                                           key=lambda r: run_number(r[0])):
        if err:
            failed[run] = err.strip().splitlines()[-1]
            print(f'  {run}: FAILED -- {failed[run]}', flush=True)
            continue
        MPs.append(MP), Rs.append(R), Es.append(E), Cs.append(C), Hs.append(H)
        print(f'  {run}: {H["n_tracks"]:>7,} tracks, wall match '
              f'{100 * H["frac_match_wall"]:.1f} % (control '
              f'{100 * H["frac_ctrl_wall"]:.1f} %), plastic '
              f'{100 * H["frac_match_plas"]:.1f} %', flush=True)
    if not MPs:
        print('\nno runs produced tracks')
        return 1

    MP = _sum_maps(pd.concat(MPs, ignore_index=True))
    RUN = pd.concat(Rs, ignore_index=True)
    E = pd.concat(Es, ignore_index=True)
    C = pd.concat(Cs, ignore_index=True)
    H = pd.DataFrame(Hs)

    # pooled: re-read the per-track tables one run at a time so the pooled
    # numbers are the pooled sample and not an average of per-run fractions.
    # Read back ONLY the runs that succeeded this pass.  A per-run file from an
    # earlier pass survives in `tracks/` -- an earlier build of this module wrote
    # one for each of the three uncalibrated runs before it learned to refuse
    # them -- and a glob would silently pool it back in.
    M = pd.concat([pd.read_parquet(od / 'tracks' / f'scint_{r}.parquet')
                   for r in H.run], ignore_index=True)
    stale = sorted(p.name for p in (od / 'tracks').glob('scint_run_*.parquet')
                   if p.name[len('scint_'):-len('.parquet')] not in set(H.run))
    if stale:
        print(f'\n  NOTE: {len(stale)} per-run file(s) in tracks/ are not from '
              f'this pass and were NOT pooled: {", ".join(stale)}')
    POOL = rates(M)
    BOTH = both_layers(M)
    EP = edge_profile(M, geo)
    EW = edge_width(EP, geo)
    CONF = group_confusion(M)
    RAIL = rail_census(M)
    VP = v_profile(M)
    lev_w = geo['w_wall'] - geo['w_strip']
    lev_p = geo['w_plas'] - geo['w_strip']
    EV = edge_vs_tan(M, geo).assign(layer='wall', lever_mm=lev_w)
    PV = plastic_edge_vs_tan(M, geo).assign(layer='plas', lever_mm=lev_p)
    AS = angle_scale(EV, lev_w) if len(EV) else {}
    ROB = angle_scale_robustness(M, geo)
    AP = angle_scale(PV, lev_p) if len(PV) else {}
    EV = pd.concat([EV, PV], ignore_index=True)
    sigma = float(np.nanmedian(EW.sigma_mm[EW.interior])) if len(EW) else np.nan
    TOL = pd.DataFrame()
    if np.isfinite(sigma):
        M['match_wall_tol'] = tolerant(M, sigma)
        TOL = pd.DataFrame([
            dict(n_sigma=ns, sigma_mm=sigma,
                 n_near=int((M.d_edge_wall < ns * sigma).sum()),
                 frac_match=float(tolerant(M, sigma, ns).sum()
                                  / max(M.on_wall.sum(), 1)))
            for ns in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)])
    RES = M[['res_u_wall', 'res_u_plas', 'sig_u_wall', 'sig_u_plas',
             'd_edge_wall', 'on_wall', 'on_plas', 'slope_ok',
             'n_trk']].dropna(subset=['res_u_wall'], how='all')

    for name, df in (('maps', MP), ('rates_run', RUN), ('rates', POOL),
                     ('edge_profile', EP), ('edge_width', EW),
                     ('confusion', CONF), ('headline', H),
                     ('both_layers', BOTH), ('tolerance', TOL),
                     ('edge_vs_tan', EV), ('rail_census', RAIL),
                     ('angle_scale_robustness', ROB),
                     ('v_profile', VP),
                     ('edge_profile_run', E), ('confusion_run', C)):
        df.to_csv(od / f'{name}.csv', index=False)
    RES.sample(min(len(RES), 400_000), random_state=3).to_parquet(
        od / 'residuals.parquet', index=False)

    meta = dict(schema=SCHEMA, arm=ARM, runs=list(H.run), failed=failed,
                dt_window=list(DT_WINDOW), off_window=list(OFF_WINDOW),
                n_tracks=int(len(M)),
                layers={k: float(geo[k]) for k in ('w_strip', 'w_wall',
                                                   'w_plas', 'w_ls')},
                lever_wall=float(geo['w_wall'] - geo['w_strip']),
                lever_plas=float(geo['w_plas'] - geo['w_strip']),
                group_u={str(g): list(map(float, v))
                         for g, v in group_u(geo).items()},
                plas_u={str(k): float(v) for k, v in geo['plas_u'].items()},
                edge_sigma_mm=sigma, edge_n_sigma=EDGE_N_SIGMA,
                angle_scale=AS, angle_scale_plastic=AP,
                boundaries=list(map(float, interior_boundaries(geo))),
                sig_u_wall_median=float(np.nanmedian(M.sig_u_wall)),
                sig_u_plas_median=float(np.nanmedian(M.sig_u_plas)))
    (od / 'det_a_scint.meta.json').write_text(json.dumps(meta, indent=1,
                                                         default=float))

    print(f'\n{len(M):,} gated arm-{ARM} tracks over {len(H)} runs')
    print('\nPOOLED confirmation rates -- signal window against its control')
    show = POOL[POOL.window.isin(('prod', 'ctrl', 'core', 'ctrl_core'))]
    print(show[['selection', 'layer', 'window', 'n', 'n_pred', 'frac_pred',
                'frac_match_pred', 'frac_match_hit', 'excess']]
          .round(3).to_string(index=False))
    print('\nTHE v RAIL -- what lands outside the chamber, and does it confirm?')
    print(RAIL[['sample', 'n', 'frac_of_all', 'frac_match_wall',
                'frac_match_plas', 'frac_ctrl_wall']].round(4)
          .to_string(index=False))
    print('\nBOTH LAYERS, of tracks that can be confirmed by both')
    print(BOTH.round(3).to_string(index=False))
    print(f'\nMEASURED edge width at the wall: {sigma:.1f} mm '
          f'(formal fit error {np.nanmedian(M.sig_u_wall):.1f} mm)')
    if AS and np.isfinite(AS.get('eps', np.nan)):
        print(f'\nANGLE SCALE FROM THE WALL (independent of the imaging): '
              f'eps = {100 * AS["eps"]:+.1f} +- {100 * AS["eps_err"]:.1f} %, '
              f'chi2/dof {AS["chi2dof"]:.1f}')
        print(f'  boundary shift = {AS["slope_mm"]:+.1f} mm per unit tan, '
              f'rigid offset {AS["intercept_mm"]:+.1f} mm')
        print(f'  -> the wall prefers k(arm A) x {AS["k_ratio"]:.3f}')
        if len(ROB):
            print('  repeated per selection (eps, %):')
            for _, r in ROB.iterrows():
                flag = '  <- BIASED, tan-correlated cut' if r.biased else ''
                print(f'    {r.selection:<15} {r.layer:<5} '
                      f'{100 * r.eps:+6.1f} +- {100 * r.eps_err:4.1f}{flag}')
    if AP and np.isfinite(AP.get('eps', np.nan)):
        print(f'  PLASTIC, lever {AP["lever_mm"]:.0f} mm (wall '
              f'{AS.get("lever_mm", float("nan")):.0f}): shift '
              f'{AP["slope_mm"]:+.1f} mm per unit tan -> eps = '
              f'{100 * AP["eps"]:+.1f} +- {100 * AP["eps_err"]:.1f} %')
    print('\nPREDICTED vs FIRED wall group, single-group events')
    print(CONF.to_string(index=False) if len(CONF) else '  (none)')
    print(f'\n-> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
