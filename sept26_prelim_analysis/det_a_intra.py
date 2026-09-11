#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
det_a_intra.py -- one chamber, one topology: intra-chamber pairs in detector A.

A deliberately narrow analysis.  Everything here concerns pairs of tracks that
both land in chamber A, and nothing else enters: no other chamber, no other
topology, no pooled campaign spectrum.  The purpose is to establish, on the
simplest possible sample, what the intra-chamber opening-angle distribution
actually is and what it can be compared against.

WHY CHAMBER A.  It is the only chamber with no dead channels at all
(`source_imaging.dead_ranges` finds none in any run), it has the highest
scintillator-tagged efficiency of the four, and it is the arm whose per-run
angle scale `k` is stable to 3.4 % where C and D move by 10-13 %.  Every
confound that a chamber can add, A adds least of.

FOUR NESTED SELECTIONS, and the differences between them are the result:

  ``all``       every unordered pair of gated, angle-calibrated arm-A tracks
                inside one trigger.
  ``slope``     both legs carry a reliable slope on BOTH planes
                (``x_slope_reliable & y_slope_reliable``).  A track whose
                charge column arrives at every strip simultaneously has no
                timing slope, so its direction -- and therefore the pair's
                opening angle -- is not a measurement.  **On run_145 only 19 %
                of intra-A pairs pass this**, and the median opening angle of
                those that do is ~50 deg against ~32 deg for the rest.  This is
                the single largest effect on this page.
  ``pointing``  both legs within ``DCA_MAX`` of the beam axis.  The existing
                campaign chain applies this silently inside
                `source_imaging._track_table`; here it is a named selection
                because it removes ~97 % of intra-A two-track triggers.
  ``prompt``    the two legs agree in time inside the chamber,
                ``|dt0| <= PROMPT_NS``.

THE IN-CHAMBER CLOCK, AND WHAT IT CAN AND CANNOT SETTLE.  ``t0`` is the fitted
arrival time of the charge that starts at the mesh, so two legs born in the same
instant have the same ``t0`` whatever their angles, and ``dt0 = t0(1) - t0(2)``
is a coincidence test that needs no scintillator.  It shows an unambiguous
prompt peak, ~60 ns wide, on a ~200-270 ns pedestal.

**The prompt FRACTION is not identifiable from that shape.**  Five
two-component models with essentially the same likelihood
(:func:`dt0_models`) return prompt fractions from 0.27 to 0.69.  The two widths
differ by a factor of four, which is not enough to separate them.  So this
module never quotes a subtracted spectrum: it quotes the ``prompt`` and
``offtime`` samples side by side and lets the difference between their shapes
carry whatever the timing has to say.

THE PEAK LIVES ENTIRELY BELOW THE SLOPE THRESHOLD.  :func:`slope_profile` splits
the pairs by the smallest in-plane slope anywhere in the pair and the whole
prompt excess sits on one side of ``wft.reco.TAN_MIN_SLOPE``; above it the
prompt-to-off-time ratio is consistent with one, meaning no peak.  The
scintillators do not see the peak either: arm-A tagged pairs have the SMALLER
core-to-wing ratio (1.52 against 1.85 on run_145), the wrong way round if it
were the population the scintillators call prompt.

**Its origin is left open, and one explanation is explicitly falsified here.**
The natural reading -- a slope-less fit has a degenerate ``t0`` that collapses
onto its prior -- fails: single-track ``t0`` is WIDEST for the low-slope tracks
(sd 293 ns against 105 ns for the well-measured ones), so nothing is collapsing.
What the module does establish is operational: on the pairs whose opening angle
is a measurement at all, there is no prompt excess to select on.

THE EFFICIENCY MAP IS TWO-DIMENSIONAL AND IN THE TOY'S OWN FRAME.  `efficiency.py`
bins in the in-plane coordinate only, and it bins in the lever from the beam-axis
foot (``local_x - PINWHEEL``) while `acceptance.Chambers.cross` returns the
offset from the PLANE CENTRE.  For arm A those differ by 16.35 mm, so the
published map is indexed 0.4 bins off from the toy that consumes it.  Here the
map is built in ``(u, v) = (x_local, y_local)``, which is the plane-centre frame
the toy uses, and it is binned in both.

    python -m sept26_prelim_analysis.det_a_intra --runs run_145
    python -m sept26_prelim_analysis.det_a_intra --jobs 6
"""
from __future__ import annotations

import argparse
import itertools
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

SCHEMA = 'sept26_prelim/det_a_intra/1'
ARM = 'A'
#: Pointing cut, the same 30 mm the campaign chain applies inside
#: `source_imaging._track_table` -- named here rather than buried.
DCA_MAX = 30.0
#: Half-width of the in-chamber prompt window, ns.  Chosen at roughly one
#: fitted prompt sigma (~60 ns) so the window is dominated by the peak; the
#: analysis never divides by it, so its exact value is not a tuned parameter.
PROMPT_NS = 75.0
#: The off-time control window, ns.  Starts well past the prompt peak and stops
#: inside the range where the pedestal is still well populated.
OFFTIME_NS = (150.0, 700.0)
#: Opening-angle binning.  10 deg over the range one chamber can reach: the
#: geometric maximum for two tracks in a 380 x 340 mm plane at 235 mm is ~90 deg.
BINS = np.arange(0.0, 100.1, 10.0)
#: (u, v) grid for the efficiency map, mm about the PLANE CENTRE.  40 mm, the
#: same pitch `efficiency.U_EDGES` uses, so the two are comparable bin for bin.
U_EDGES = np.arange(-180.0, 180.1, 40.0)
V_EDGES = np.arange(-160.0, 160.1, 40.0)
#: Minimum tagged-and-seeded events in a cell before its efficiency is reported.
MIN_CELL = 25
#: The selections, in the order the report reads them.
SELECTIONS = ('all', 'mixed', 'slope', 'slope+mixed', 'prompt',
              'slope+prompt', 'slope+offtime', 'vertex', 'slope+vertex',
              'slope+vertex+mixed', 'pointing', 'slope+pointing',
              'pointing+mixed', 'slope+pointing+mixed')
#: A pair counts as vertexed if its two lines approach each other inside
#: ``VERTEX_DCA_MM`` and do so within ``VERTEX_R_MM`` of the beam axis -- the
#: same 20/30 mm `source_imaging.vertex_summary` uses, so the two are readable
#: against each other.
VERTEX_R_MM = 20.0
VERTEX_DCA_MM = 30.0


# --------------------------------------------------------------------------- #
# tracks and pairs
# --------------------------------------------------------------------------- #
TRACK_COLS = [
    'event_id', 'arm', 'gated', 'angle_calibrated', 'dca_axis_mm',
    'x_local', 'y_local', 'd_x', 'd_y', 'd_z', 'x_t0', 'y_t0',
    'p0_x', 'p0_y', 'p0_z',
    'drift_t_end_ns', 'drift_len_mm', 'drift_railed',
    'x_slope_reliable', 'y_slope_reliable', 'x_isochronous', 'y_isochronous',
    'chi2dof_x', 'chi2dof_y', 'q_total', 'tanx', 'tany',
    'wall_A', 'plastic_A', 'coinc_A', 'n_coinc_arms', 'k_arm',
]


def tracks(run: str, subruns, src: Path) -> pd.DataFrame:
    """Gated, angle-calibrated arm-A tracks for one run.

    No pointing cut: it is applied later as a named selection, because on this
    sample it is not a quality cut but a physics one and it removes most of the
    data.
    """
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {run}/{sub}')
        d = pd.read_parquet(p, columns=TRACK_COLS)
        out.append(d[(d.arm == ARM) & d.gated & d.angle_calibrated]
                   .assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    # The per-trigger scintillator flags are NULL for any sub-run whose stage-1
    # arm flags were not available, and a null propagates into `~` as a
    # TypeError three functions later.  Coerced once, here, with the count kept
    # so a run that silently has no tag cannot pass for a run with no hits.
    for c in ('wall_A', 'plastic_A', 'coinc_A', 'x_slope_reliable',
              'y_slope_reliable', 'x_isochronous', 'y_isochronous',
              'drift_railed', 'gated', 'angle_calibrated'):
        if c in t.columns:
            t[c] = t[c].astype('boolean').fillna(False).astype(bool)
    t['t0'] = 0.5 * (t.x_t0 + t.y_t0)
    t['slope_ok'] = t.x_slope_reliable & t.y_slope_reliable
    t['key'] = t.subrun + ':' + t.event_id.astype(str)
    return t.reset_index(drop=True)


def _mixed_index(t: pd.DataFrame, n_real: int, seed: int = 11,
                 factor: int = 6, same_class: bool = True) -> tuple:
    """Index pairs of arm-A tracks drawn from DIFFERENT triggers.

    The geometric null.  Two tracks that never shared a trigger cannot share a
    vertex, so their opening angle, their closest approach and their in-plane
    separation are what the apparatus produces with no pair physics in it at
    all.  Drawn within a sub-run, so the beam and detector conditions match.

    ``same_class`` restricts the pool to tracks that came from a trigger with
    at least two arm-A tracks -- the same population the real pairs are drawn
    from.  Without it the null is built from every track in the run, including
    the single-track triggers, and any difference in occupancy between busy and
    quiet triggers enters the null as if it were physics.  It is the like-for-
    like choice and it is the default.

    **Not a null for `dt0`.**  ``t0`` carries a per-trigger offset -- the
    trigger-mean spread is 235 ns against a 142 ns within-trigger spread on
    run_145 -- so a cross-trigger difference is broader than a within-trigger
    accidental one by construction.  Every timing quantity on a mixed pair is
    therefore written as NaN rather than left to be misread.
    """
    rng = np.random.default_rng(seed)
    pool = t
    if same_class:
        n_per = t.groupby('key').event_id.transform('size')
        pool = t[n_per >= 2]
        if len(pool) < 4:
            pool = t
    mi, mj = [], []
    for _, g in pool.groupby('subrun', sort=False):
        idx = g.index.to_numpy()
        ev = g.event_id.to_numpy()
        if len(idx) < 4:
            continue
        n = max(factor * n_real // max(pool.subrun.nunique(), 1), 100)
        A = rng.integers(0, len(idx), n)
        B = rng.integers(0, len(idx), n)
        ok = ev[A] != ev[B]
        mi.append(idx[A[ok]])
        mj.append(idx[B[ok]])
    if not mi:
        return np.array([], np.int64), np.array([], np.int64)
    return np.concatenate(mi), np.concatenate(mj)


def _dca(p1, d1, p2, d2) -> tuple:
    """(midpoint of closest approach, line separation) for two 3D lines."""
    from sept26_prelim_analysis.source_imaging import _dca_two_lines
    return _dca_two_lines(p1, d1, p2, d2)


def pairs(t: pd.DataFrame, run: str = '', with_mixed: bool = True
          ) -> pd.DataFrame:
    """Every unordered pair of arm-A tracks inside one trigger, with its observables.

    ``dt0`` is the in-chamber coincidence variable, ``sep_plane_mm`` the
    in-plane separation of the two impact points, and ``dca_pair_mm`` how close
    the two reconstructed lines actually come to each other in space.  The
    event-mixed rows (``mixed = True``) are the same quantities for tracks that
    never shared a trigger, which is the geometric null for all three.
    """
    L, R = [], []
    for _, g in t.groupby('key', sort=False):
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(g.index, 2):
            L.append(i)
            R.append(j)
    if not L:
        return pd.DataFrame()
    li = np.asarray(L, np.int64)
    ri = np.asarray(R, np.int64)
    if with_mixed:
        xi, xj = _mixed_index(t, len(li))
        is_mixed = np.r_[np.zeros(len(li), bool), np.ones(len(xi), bool)]
        li = np.r_[li, xi]
        ri = np.r_[ri, xj]
    else:
        is_mixed = np.zeros(len(li), bool)
    a = t.loc[li].reset_index(drop=True)
    b = t.loc[ri].reset_index(drop=True)
    d1 = a[['d_x', 'd_y', 'd_z']].to_numpy(float)
    d2 = b[['d_x', 'd_y', 'd_z']].to_numpy(float)
    n1 = np.linalg.norm(d1, axis=1)
    n2 = np.linalg.norm(d2, axis=1)
    cos = np.clip((d1 * d2).sum(1) / np.where(n1 * n2 > 0, n1 * n2, np.nan),
                  -1, 1)
    du = (a.x_local - b.x_local).to_numpy()
    dv = (a.y_local - b.y_local).to_numpy()
    V, dca = _dca(a[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), d1,
                  b[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), d2)
    P = pd.DataFrame(dict(
        run=run or a.get('run', ''), subrun=a.subrun.to_numpy(),
        event_id=a.event_id.to_numpy(), mixed=is_mixed,
        open_deg=np.degrees(np.arccos(cos)),
        du_mm=du, dv_mm=dv, sep_plane_mm=np.hypot(du, dv),
        vx_mm=V[:, 0], vy_mm=V[:, 1], vz_mm=V[:, 2],
        v_r_mm=np.hypot(V[:, 0], V[:, 2]), dca_pair_mm=dca,
        u1=a.x_local.to_numpy(), v1=a.y_local.to_numpy(),
        u2=b.x_local.to_numpy(), v2=b.y_local.to_numpy(),
        dt0_ns=(a.t0 - b.t0).to_numpy(),
        dt0_x_ns=(a.x_t0 - b.x_t0).to_numpy(),
        dt0_y_ns=(a.y_t0 - b.y_t0).to_numpy(),
        dt_end_ns=(a.drift_t_end_ns - b.drift_t_end_ns).to_numpy(),
        both_slope=(a.slope_ok & b.slope_ok).to_numpy(),
        # the smallest in-plane slope anywhere in the pair: four planes, two
        # legs.  A track at |tan| below TAN_MIN_SLOPE delivers its charge to
        # every strip at once, so its t0 and its direction are both degenerate,
        # and this is the variable that turns out to control the dt0 peak.
        tan_min_pair=np.minimum(
            np.minimum(a.tanx.abs(), a.tany.abs()),
            np.minimum(b.tanx.abs(), b.tany.abs())).to_numpy(),
        n_isochronous=(a.x_isochronous.astype(int)
                       + a.y_isochronous.astype(int)
                       + b.x_isochronous.astype(int)
                       + b.y_isochronous.astype(int)).to_numpy(),
        either_railed=(a.drift_railed | b.drift_railed).to_numpy(),
        dca_max_mm=np.maximum(a.dca_axis_mm, b.dca_axis_mm).to_numpy(),
        chi2_max=np.maximum(np.maximum(a.chi2dof_x, a.chi2dof_y),
                            np.maximum(b.chi2dof_x, b.chi2dof_y)).to_numpy(),
        q_min=np.minimum(a.q_total, b.q_total).to_numpy(),
        drift_len_max=np.maximum(a.drift_len_mm, b.drift_len_mm).to_numpy(),
        wall_A=a.wall_A.to_numpy(), plastic_A=a.plastic_A.to_numpy(),
        tag_A=(a.wall_A & a.plastic_A).to_numpy(),
        n_coinc_arms=a.n_coinc_arms.to_numpy(),
        k_arm=a.k_arm.to_numpy()))
    # timing is meaningless across triggers: blank it rather than let a mixed
    # dt0 be read as an accidental template it is not (see `_mixed_index`)
    for c in ('dt0_ns', 'dt0_x_ns', 'dt0_y_ns', 'dt_end_ns'):
        P.loc[P.mixed, c] = np.nan
    P['prompt'] = (P.dt0_ns.abs() <= PROMPT_NS) & ~P.mixed
    P['offtime'] = ((P.dt0_ns.abs() > OFFTIME_NS[0])
                    & (P.dt0_ns.abs() <= OFFTIME_NS[1]) & ~P.mixed)
    P['pointing'] = P.dca_max_mm < DCA_MAX
    P['vertex'] = (P.v_r_mm < VERTEX_R_MM) & (P.dca_pair_mm < VERTEX_DCA_MM)
    return P


def select(P: pd.DataFrame, name: str) -> np.ndarray:
    """The boolean mask for one named selection.  One place, so nothing drifts.

    Every selection is on the REAL pairs unless it names ``mixed``, so a caller
    can never pool the two by accident.
    """
    parts = name.split('+')
    # `Series.to_numpy()` may return a VIEW of the frame's own buffer, and the
    # `&=` below is in place -- taking it without a copy silently overwrote the
    # `mixed` column the first time this ran.  Every mask here is a fresh array.
    if 'mixed' in P.columns:
        mx = P.mixed.to_numpy(copy=True)
        m = mx if 'mixed' in parts else ~mx
    else:
        m = np.ones(len(P), bool)
    m = np.array(m, dtype=bool, copy=True)
    for part in parts:
        if part in ('all', 'mixed'):
            continue
        if part == 'slope':
            m &= P.both_slope.to_numpy()
        elif part == 'prompt':
            m &= P.prompt.to_numpy()
        elif part == 'offtime':
            m &= P.offtime.to_numpy()
        elif part == 'pointing':
            m &= P.pointing.to_numpy()
        elif part == 'vertex':
            m &= P.vertex.to_numpy()
        else:
            raise KeyError(f'unknown selection component {part!r}')
    return m


def census(P: pd.DataFrame) -> pd.DataFrame:
    """How many pairs survive each selection, and what it does to the spectrum."""
    rows = []
    n_real = int((~P.mixed).sum()) if 'mixed' in P.columns else len(P)
    n_mix = int(P.mixed.sum()) if 'mixed' in P.columns else 0
    for name in SELECTIONS:
        m = select(P, name)
        g = P[m]
        # fraction of the selection's OWN class: a mixed selection against the
        # mixed total and a real one against the real total, or the numbers
        # read as if the two samples were one
        base = n_mix if 'mixed' in name.split('+') else n_real
        rows.append(dict(
            selection=name, n=int(m.sum()),
            frac_of_all=float(m.sum() / base) if base else np.nan,
            median_open_deg=float(g.open_deg.median()) if len(g) else np.nan,
            mean_open_deg=float(g.open_deg.mean()) if len(g) else np.nan,
            p90_open_deg=float(g.open_deg.quantile(0.9)) if len(g) else np.nan,
            median_sep_mm=float(g.sep_plane_mm.median()) if len(g) else np.nan,
            median_dca_pair_mm=float(g.dca_pair_mm.median()) if len(g) else np.nan,
            frac_vertex=float(g.vertex.mean()) if len(g) else np.nan,
            frac_tag_A=float(g.tag_A.mean()) if len(g) else np.nan,
            frac_railed=float(g.either_railed.mean()) if len(g) else np.nan,
            median_dca_max=float(g.dca_max_mm.median()) if len(g) else np.nan,
            median_chi2_max=float(g.chi2_max.median()) if len(g) else np.nan))
    return pd.DataFrame(rows)


def spectra(P: pd.DataFrame, bins=BINS) -> pd.DataFrame:
    """The opening-angle histogram of every selection, counts and shape."""
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for name in SELECTIONS:
        g = P[select(P, name)]
        h, _ = np.histogram(g.open_deg.to_numpy(float), bins=bins)
        tot = h.sum()
        for i, v in enumerate(h):
            rows.append(dict(selection=name, theta=mid[i], lo=bins[i],
                             hi=bins[i + 1], n=int(v),
                             frac=float(v / tot) if tot else np.nan,
                             err=float(np.sqrt(max(v, 1)) / tot)
                             if tot else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the double-track resolution, measured in situ
# --------------------------------------------------------------------------- #
#: Separation above which the real and event-mixed samples agree, mm.  Only a
#: fallback for the resolution normalisation, and never a cut on the physics.
SEP_PLATEAU_MM = 200.0
#: The window the two-track efficiency is declared complete over, mm.  Two
#: clusters this far apart on a 400 mm plane read at 0.78 mm pitch are resolved
#: by any reasonable definition; see :func:`two_track_efficiency`.
SEP_RESOLVED = (140.0, 200.0)
SEP_EDGES = np.arange(0.0, 561.0, 20.0)


def two_track_efficiency(P: pd.DataFrame, edges=SEP_EDGES,
                         plateau: float = SEP_PLATEAU_MM) -> pd.DataFrame:
    """P(both tracks found | their impact points are ``sep`` apart), in situ.

    **The single largest thing missing from the acceptance toy**, and the
    reason its folded prediction is far more collimated than anything measured.
    `acceptance.py` treats the two legs as independently reconstructed, so a
    pair whose impact points are 10 mm apart is accepted at full efficiency.
    The data says otherwise: **no intra-A pair at all is reconstructed below
    40 mm separation**, and the sample is still depleted out to ~120 mm.

    Measured as the real-over-mixed ratio.  Event-mixed pairs are two tracks
    from different triggers, so they sample the same single-track occupancy with
    no resolution loss and no correlation; normalising the ratio on the plateau
    above ``plateau`` mm turns it into an efficiency.

    **What that assumes, and it is not free.**  If real pairs are genuinely
    correlated in position -- which a pair from a common vertex is -- the ratio
    mixes that correlation into the efficiency.  The hard zero below 40 mm
    cannot be correlation (no physics gives exactly zero), so that part is
    resolution; between 40 and 120 mm the curve is an upper bound on the loss if
    real pairs cluster at small separation, and a lower bound if they avoid it.
    """
    mid = 0.5 * (edges[:-1] + edges[1:])
    r, _ = np.histogram(P.loc[~P.mixed, 'sep_plane_mm'].to_numpy(float),
                        bins=edges)
    m, _ = np.histogram(P.loc[P.mixed, 'sep_plane_mm'].to_numpy(float),
                        bins=edges)
    hi = (mid >= SEP_RESOLVED[0]) & (mid <= SEP_RESOLVED[1])
    if m[hi].sum() <= 0 or r[hi].sum() <= 0:
        return pd.DataFrame()
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.where(m > 0, r / m, np.nan)
    # Normalised on a FIXED PHYSICAL window, not on a plateau, because there is
    # no plateau: the raw ratio keeps rising to the largest separations, which
    # is correlation and not efficiency.  Two clusters 140-200 mm apart on a
    # 400 mm plane with a 0.78 mm pitch are resolved by any reasonable
    # definition, so the resolution is DECLARED complete there and the further
    # rise is left in `ratio_raw`.
    #
    # **No conclusion on this page depends on the choice.**  Every comparison
    # is shape-normalised, and as long as the curve does not clip at one, a
    # different constant rescales the acceptance uniformly and cancels.
    k = float(np.nansum(r[hi]) / np.nansum(m[hi]))
    eff = raw / k
    err = np.where((m > 0) & (r > 0),
                   eff * np.sqrt(1 / np.clip(r, 1, None)
                                 + 1 / np.clip(m, 1, None)), np.nan)
    return pd.DataFrame(dict(sep_mid=mid, sep_lo=edges[:-1], sep_hi=edges[1:],
                             n_real=r, n_mixed=m, scale=k,
                             ratio_raw=raw, eff=np.clip(eff, 0.0, 1.0),
                             err=err,
                             above_plateau=np.where(np.isfinite(eff),
                                                    eff > 1.0, False)))


#: When measuring one axis's turn-on, the OTHER axis is held above this, so the
#: curve is not contaminated by the other view's own loss.  100 mm is four
#: times the hard edge and well onto the flat part of the 2D map.
OTHER_MIN_MM = 100.0
#: Per-axis grid for the two-track efficiency, mm.
AXIS_EDGES = np.arange(0.0, 401.0, 20.0)
#: Where each view's resolution is declared complete, mm.  Past the turn-on,
#: which finishes by ~50 mm on both views, and before the decline beyond
#: 200 mm that is geometry and leg-to-leg correlation, not efficiency.
AXIS_PLATEAU = (60.0, 200.0)


def two_track_efficiency_axis(P: pd.DataFrame, axis: str = 'u',
                              other_min: float = OTHER_MIN_MM,
                              edges=AXIS_EDGES) -> pd.DataFrame:
    """P(both tracks found) against the separation in ONE view.

    **Why one view at a time.**  The measured loss is not a disc around zero
    separation, it is a CROSS: on the 2D map of real over event-mixed the whole
    first row and the whole first column sit at 0.00-0.03 of the plateau,
    whatever the other coordinate does.  A pair 12 mm apart in u and 300 mm
    apart in v is lost as completely as one 12 mm apart in both.  That is what
    two independent strip planes should do -- two tracks sharing an x-strip band
    merge in the x view, and the y view cannot rescue them because the fit needs
    both planes -- and it is invisible to a radial parameterisation, which calls
    that pair well separated.

    Measured with the OTHER view held above ``other_min`` so the curve is the
    one view's own turn-on.  Normalised over ``AXIS_PLATEAU`` -- past the
    turn-on, which completes by ~50 mm on both views, and before the slow
    decline and the last-bin spike that set in beyond 200 mm and are geometry
    and correlation rather than efficiency.  Then taken as a running maximum
    from the left, because a resolution can only rise with separation.
    """
    col = 'du_mm' if axis == 'u' else 'dv_mm'
    oth = 'dv_mm' if axis == 'u' else 'du_mm'
    mid = 0.5 * (edges[:-1] + edges[1:])
    sel = P[oth].abs() >= other_min
    r, _ = np.histogram(P.loc[sel & ~P.mixed, col].abs(), bins=edges)
    m, _ = np.histogram(P.loc[sel & P.mixed, col].abs(), bins=edges)
    if r.sum() <= 0 or m.sum() <= 0:
        return pd.DataFrame()
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.where(m > 0, r / m, np.nan)
    solid = (np.isfinite(raw) & (r >= 200)
             & (mid >= AXIS_PLATEAU[0]) & (mid <= AXIS_PLATEAU[1]))
    if not solid.any():
        return pd.DataFrame()
    k = float(np.nanmean(raw[solid]))
    rel = raw / k
    # running maximum: the largest turn-on consistent with a monotonic
    # resolution, and the only part of the ratio that can BE a resolution
    eff = np.fmax.accumulate(np.nan_to_num(rel, nan=0.0))
    return pd.DataFrame(dict(
        axis=axis, sep_lo=edges[:-1], sep_hi=edges[1:], sep_mid=mid,
        n_real=r, n_mixed=m, scale=k, ratio_raw=raw, ratio_rel=rel,
        eff=np.clip(eff, 0.0, 1.0),
        err=np.where((m > 0) & (r > 0),
                     rel * np.sqrt(1 / np.clip(r, 1, None)
                                   + 1 / np.clip(m, 1, None)), np.nan),
        other_min=other_min))


def separability_check(P: pd.DataFrame, EU: pd.DataFrame, EV: pd.DataFrame,
                       edges=AXIS_EDGES, floor: float = 50.0,
                       ceil: float = 300.0) -> pd.DataFrame:
    """Does the product of the two per-axis curves reproduce the 2D map?

    The separable model is an assumption, so it is tested rather than asserted.
    Compared only where both coordinates are past the hard edge and inside
    ``ceil``: below the edge both the model and the measurement are zero and
    agreement there is free, and beyond it the ratio carries the leg-to-leg
    correlation the efficiency deliberately excludes.
    """
    if EU.empty or EV.empty:
        return pd.DataFrame()
    mid = 0.5 * (edges[:-1] + edges[1:])
    def h(g):
        out, _, _ = np.histogram2d(g.du_mm.abs(), g.dv_mm.abs(),
                                   bins=[edges, edges])
        return out
    hr, hm = h(P[~P.mixed]), h(P[P.mixed])
    with np.errstate(divide='ignore', invalid='ignore'):
        obs = np.where(hm > 0, hr / hm, np.nan)
    fu = np.interp(mid, EU.sep_mid, EU.eff)
    fv = np.interp(mid, EV.sep_mid, EV.eff)
    model = np.outer(fu, fv)
    live = ((mid[:, None] >= floor) & (mid[None, :] >= floor)
            & (mid[:, None] <= ceil) & (mid[None, :] <= ceil) & (hr >= 100))
    if not live.any():
        return pd.DataFrame()
    k = float(np.nansum(obs[live] * model[live]) / np.nansum(model[live] ** 2))
    rows = []
    for i, u in enumerate(mid):
        for j, v in enumerate(mid):
            if not live[i, j]:
                continue
            rows.append(dict(du_mid=u, dv_mid=v, n_real=int(hr[i, j]),
                             observed=float(obs[i, j] / k),
                             separable_model=float(model[i, j]),
                             residual=float(obs[i, j] / k - model[i, j])))
    return pd.DataFrame(rows)


def _uv_lookup(EU: pd.DataFrame, EV: pd.DataFrame):
    """A (|du|, |dv|) -> two-track efficiency interpolator, separable."""
    if EU is None or EV is None or EU.empty or EV.empty:
        return lambda du, dv: np.ones(len(np.atleast_1d(du)))
    xu, yu = EU.sep_mid.to_numpy(float), np.clip(EU.eff.to_numpy(float), 0, 1)
    xv, yv = EV.sep_mid.to_numpy(float), np.clip(EV.eff.to_numpy(float), 0, 1)
    return lambda du, dv: (
        np.interp(np.abs(np.asarray(du, float)), xu, yu, left=0.0,
                  right=yu[-1])
        * np.interp(np.abs(np.asarray(dv, float)), xv, yv, left=0.0,
                    right=yv[-1]))


def _sep_lookup(E: pd.DataFrame):
    """A separation -> two-track efficiency interpolator, 0 below the first bin."""
    if E is None or E.empty:
        return lambda s: np.ones(len(np.atleast_1d(s)))
    g = E[np.isfinite(E.eff)]
    x = g.sep_mid.to_numpy(float)
    y = np.clip(g.eff.to_numpy(float), 0.0, 1.0)
    if len(x) < 2:
        return lambda s: np.ones(len(np.atleast_1d(s)))
    return lambda s: np.interp(np.asarray(s, float), x, y,
                               left=0.0, right=y[-1])


# --------------------------------------------------------------------------- #
# the in-chamber clock
# --------------------------------------------------------------------------- #
def _shapes():
    """The two-component densities the dt0 fit is tried with."""
    from scipy.special import gamma as _gam

    def gauss(x, s):
        return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))

    def lap(x, b):
        return np.exp(-np.abs(x) / b) / (2 * b)

    def tri(x, W):
        return np.clip(1 - np.abs(x) / W, 0, None) / W

    def gnorm(x, a, b):
        return b / (2 * a * _gam(1 / b)) * np.exp(-np.abs(x / a) ** b)

    return {
        'gauss + gauss':    (lambda p, x: p[0] * gauss(x, p[1])
                             + (1 - p[0]) * gauss(x, p[2]),
                             [0.5, 60, 270],
                             [(0.01, .99), (5, 150), (100, 700)]),
        'gauss + triangle': (lambda p, x: p[0] * gauss(x, p[1])
                             + (1 - p[0]) * tri(x, p[2]),
                             [0.5, 60, 600],
                             [(0.01, .99), (5, 150), (200, 1600)]),
        'gauss + Laplace':  (lambda p, x: p[0] * gauss(x, p[1])
                             + (1 - p[0]) * lap(x, p[2]),
                             [0.5, 60, 180],
                             [(0.01, .99), (5, 150), (30, 600)]),
        'gauss + gen-norm': (lambda p, x: p[0] * gauss(x, p[1])
                             + (1 - p[0]) * gnorm(x, p[2], p[3]),
                             [0.5, 60, 250, 1.5],
                             [(0.01, .99), (5, 150), (50, 900), (0.4, 4)]),
        'Laplace + gen-norm': (lambda p, x: p[0] * lap(x, p[1])
                               + (1 - p[0]) * gnorm(x, p[2], p[3]),
                               [0.5, 40, 250, 1.5],
                               [(0.01, .99), (5, 150), (50, 900), (0.4, 4)]),
    }


def dt0_models(dt: np.ndarray, window: float = 800.0,
               bins: np.ndarray = None) -> pd.DataFrame:
    """Fit the family of two-component models to ``dt0``, and report the spread.

    The point of running five shapes is not to choose one.  It is that they
    agree on the likelihood and disagree on the prompt fraction, which is the
    statement that the fraction is not measurable this way.  Both components are
    centred on zero by construction: the accidental term is the difference of
    two exchangeable times and so must be symmetric.
    """
    from scipy.optimize import minimize
    d = np.asarray(dt, float)
    d = d[np.isfinite(d) & (np.abs(d) < window)]
    if len(d) < 200:
        return pd.DataFrame()
    if bins is None:
        bins = np.arange(-window, window + 1, 25.0)
    mid = 0.5 * (bins[:-1] + bins[1:])
    wb = np.diff(bins)
    h, _ = np.histogram(d, bins=bins)
    rows = []
    for name, (f, p0, bnds) in _shapes().items():
        def nll(p, f=f, bnds=bnds):
            for v, (lo, hi) in zip(p, bnds):
                if not lo < v < hi:
                    return 1e9
            return -np.sum(np.log(np.clip(f(p, d), 1e-300, None)))
        best = None
        for scale in (0.7, 1.0, 1.4):
            start = [p0[0]] + [x * scale for x in p0[1:]]
            r = minimize(nll, start, method='Nelder-Mead',
                         options=dict(maxiter=8000, maxfev=8000,
                                      xatol=1e-5, fatol=1e-5))
            if best is None or r.fun < best.fun:
                best = r
        pred = len(d) * f(best.x, mid) * wb
        chi2 = float(np.sum((h - pred) ** 2 / np.clip(pred, 1, None)))
        dof = int(len(h) - len(best.x))
        # what the model says the off-time window buys you: the accidental
        # weight inside the prompt window over its weight in the control
        acc = (lambda x: f(np.r_[0.0, best.x[1:]] * 0 + best.x, x))
        xx = np.linspace(-window, window, 20001)
        a_only = f(np.r_[0.0, best.x[1:]], xx)      # f -> 0 leaves the pedestal
        core = a_only[np.abs(xx) <= PROMPT_NS].sum()
        wing = a_only[(np.abs(xx) > OFFTIME_NS[0])
                      & (np.abs(xx) <= OFFTIME_NS[1])].sum()
        rows.append(dict(model=name, n=int(len(d)),
                         prompt_fraction=float(best.x[0]),
                         par1=float(best.x[1]),
                         par2=float(best.x[2]),
                         par3=float(best.x[3]) if len(best.x) > 3 else np.nan,
                         nll=float(best.fun), chi2=chi2, dof=dof,
                         chi2dof=chi2 / max(dof, 1),
                         acc_core_over_wing=float(core / wing)
                         if wing > 0 else np.nan))
    return pd.DataFrame(rows).sort_values('nll', ignore_index=True)


def dt0_histogram(P: pd.DataFrame, window: float = 800.0,
                  step: float = 25.0) -> pd.DataFrame:
    """dt0, binned, split by scintillator tag -- the cross-check that fails."""
    bins = np.arange(-window, window + 1, step)
    mid = 0.5 * (bins[:-1] + bins[1:])
    real = ~P.mixed.to_numpy()
    rows = []
    for name, m in (('all', real),
                    ('tagged', real & P.tag_A.to_numpy()),
                    ('untagged', real & ~P.tag_A.to_numpy()),
                    ('slope', real & P.both_slope.to_numpy()),
                    ('no slope', real & ~P.both_slope.to_numpy())):
        h, _ = np.histogram(P.dt0_ns.to_numpy(float)[m], bins=bins)
        for i, v in enumerate(h):
            rows.append(dict(sample=name, dt0=mid[i], n=int(v),
                             n_total=int(m.sum())))
    return pd.DataFrame(rows)


def tag_test(P: pd.DataFrame) -> pd.DataFrame:
    """Core-over-wing in dt0, split by the arm-A scintillator tag.

    If the ``dt0`` peak were the population the scintillators call prompt, the
    tagged pairs would have the larger core-to-wing ratio.  They do not.
    """
    real = ~P.mixed.to_numpy()
    rows = []
    for name, m in (('tagged (wall AND plastic A)', real & P.tag_A.to_numpy()),
                    ('untagged', real & ~P.tag_A.to_numpy()),
                    ('wall A only', real & (P.wall_A & ~P.plastic_A).to_numpy()),
                    ('plastic A only', real & (P.plastic_A & ~P.wall_A).to_numpy())):
        g = P[m]
        if len(g) < 50:
            continue
        core = int(g.prompt.sum())
        wing = int(g.offtime.sum())
        rows.append(dict(
            sample=name, n=int(len(g)), n_core=core, n_wing=wing,
            core_over_wing=core / max(wing, 1),
            err=(core / max(wing, 1)) * np.sqrt(1 / max(core, 1)
                                                + 1 / max(wing, 1)),
            median_open_core=float(g[g.prompt].open_deg.median())
            if core else np.nan,
            median_open_wing=float(g[g.offtime].open_deg.median())
            if wing else np.nan))
    return pd.DataFrame(rows)


def vertex_excess(P: pd.DataFrame) -> pd.DataFrame:
    """Do the two legs converge on the beam axis more often than chance?

    The rate at which a pair's two lines approach each other inside
    ``VERTEX_DCA_MM`` and do so within ``VERTEX_R_MM`` of the axis, for the real
    sample and for the event-mixed one drawn from the same trigger class.  The
    ratio is the only statement on this page that a pair population would move
    and an uncorrelated one would not.
    """
    rows = []
    for name, base in (('all', ''), ('slope', 'slope')):
        r = P[select(P, base or 'all')]
        m = P[select(P, (base + '+mixed') if base else 'mixed')]
        if len(r) < 50 or len(m) < 50:
            continue
        kr, km = int(r.vertex.sum()), int(m.vertex.sum())
        fr, fm = kr / len(r), km / len(m)
        # Poisson error on the ratio of two rates
        e = (fr / fm) * np.sqrt(1 / max(kr, 1) + 1 / max(km, 1)) if fm else np.nan
        rows.append(dict(
            selection=name, n_real=int(len(r)), n_mixed=int(len(m)),
            k_real=kr, k_mixed=km, rate_real=fr, rate_mixed=fm,
            ratio=fr / fm if fm else np.nan, err=e,
            excess_pairs=float(kr - fm * len(r)),
            excess_sigma=((kr - fm * len(r)) / np.sqrt(max(kr, 1))
                          if kr else np.nan),
            median_open_real=float(r[r.vertex].open_deg.median())
            if kr else np.nan,
            median_open_mixed=float(m[m.vertex].open_deg.median())
            if km else np.nan))
    return pd.DataFrame(rows)


def dt_sep_grid(P: pd.DataFrame, mask: np.ndarray = None,
                sep_edges=None, dt_edges=None) -> pd.DataFrame:
    """The 2D correlation the QA turns on: dt0 against in-plane separation.

    A pair from a common vertex has its separation set by its opening angle and
    the drift depth of its two legs; an accidental pair has no such link.  So
    structure in this plane is the model-free statement about whether the
    sample contains pairs at all.
    """
    if sep_edges is None:
        sep_edges = np.arange(0.0, 561.0, 40.0)
    if dt_edges is None:
        dt_edges = np.arange(-500.0, 501.0, 50.0)
    g = P[~P.mixed] if mask is None else P[mask & ~P.mixed.to_numpy()]
    H, _, _ = np.histogram2d(g.sep_plane_mm.to_numpy(float),
                             g.dt0_ns.to_numpy(float),
                             bins=[sep_edges, dt_edges])
    rows = []
    smid = 0.5 * (sep_edges[:-1] + sep_edges[1:])
    tmid = 0.5 * (dt_edges[:-1] + dt_edges[1:])
    for i, s in enumerate(smid):
        for j, tt in enumerate(tmid):
            rows.append(dict(sep_mm=s, dt0_ns=tt, n=int(H[i, j])))
    return pd.DataFrame(rows)


def slope_profile(P: pd.DataFrame) -> pd.DataFrame:
    """Core fraction and opening angle against the pair's smallest slope.

    The test that identifies what the ``dt0`` peak is.  A track below
    ``TAN_MIN_SLOPE`` in either plane has no timing slope, so its fitted ``t0``
    is degenerate -- and if the peak is that degeneracy rather than a
    coincidence, the peak must vanish as the smallest slope in the pair grows.
    """
    from wft import reco as WR
    real = P[~P.mixed]
    edges = np.array([0.0, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 3.0])
    rows = []
    b = np.digitize(real.tan_min_pair.to_numpy(float), edges) - 1
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() < 20:
            continue
        g = real[m]
        rows.append(dict(
            tan_lo=edges[i], tan_hi=edges[i + 1],
            tan_mid=float(np.sqrt(max(edges[i], 1e-3) * edges[i + 1])),
            n=int(m.sum()),
            frac_prompt=float(g.prompt.mean()),
            frac_offtime=float(g.offtime.mean()),
            core_over_wing=float(g.prompt.sum() / max(g.offtime.sum(), 1)),
            median_open_deg=float(g.open_deg.median()),
            median_sep_mm=float(g.sep_plane_mm.median()),
            frac_tag_A=float(g.tag_A.mean()),
            below_tan_min=bool(edges[i + 1] <= WR.TAN_MIN_SLOPE)))
    return pd.DataFrame(rows)


def dt_sep_profile(P: pd.DataFrame) -> pd.DataFrame:
    """Core fraction and median opening angle against separation, per selection.

    The profile of the 2D grid, which is what a reader can actually check a
    number against.
    """
    edges = np.array([0, 60, 100, 150, 200, 260, 330, 420, 600.0])
    rows = []
    for name in ('all', 'slope'):
        g = P[select(P, name)]
        if not len(g):
            continue
        b = np.digitize(g.sep_plane_mm.to_numpy(float), edges) - 1
        for i in range(len(edges) - 1):
            m = b == i
            if m.sum() < 10:
                continue
            q = g[m]
            rows.append(dict(
                selection=name, sep_lo=edges[i], sep_hi=edges[i + 1],
                sep_mid=0.5 * (edges[i] + edges[i + 1]), n=int(m.sum()),
                frac_prompt=float(q.prompt.mean()),
                frac_offtime=float(q.offtime.mean()),
                core_over_wing=float(q.prompt.sum() / max(q.offtime.sum(), 1)),
                median_open_deg=float(q.open_deg.median())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the n_TOF timing, read from the slim
# --------------------------------------------------------------------------- #
WAL_A, PSS_A = 0, 4                     # det codes, `efficiency.WAL_CODE`


def ntof_timing(run: str, subruns, P: pd.DataFrame,
                slim_dir: Path = None, step: float = 10.0,
                window: float = 500.0) -> pd.DataFrame:
    """dt_ns of the arm-A wall and plastic hits, on and off the pair sample.

    Three populations per family: hits in triggers that produced an intra-A
    pair, hits in every trigger of the run, and the n_TOF processing's own
    random-coincidence control (``is_control``), which is flat by construction
    and is the only accidental normalisation in the problem that was not built
    here.
    """
    from sept26_prelim_analysis import slim_export as SE
    slim = SE.read_export(run, subruns, slim_dir)
    keys = set(map(tuple, P.loc[~P.mixed, ['subrun', 'event_id']]
                   .drop_duplicates().values))
    in_pair = np.fromiter(
        ((s, int(e)) in keys for s, e in zip(slim.subrun, slim.eventId)),
        bool, len(slim))
    bins = np.arange(-window, window + 1, step)
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for fam, code in (('wall A', WAL_A), ('plastic A', PSS_A)):
        d = slim[slim.det == code]
        m_pair = in_pair[slim.det.to_numpy() == code]
        for name, sel in (('pair triggers', (d.is_control == 0).to_numpy()
                           & m_pair),
                          ('all triggers', (d.is_control == 0).to_numpy()),
                          ('control', (d.is_control == 1).to_numpy())):
            h, _ = np.histogram(d.dt_ns.to_numpy(float)[sel], bins=bins)
            tot = h.sum()
            for i, v in enumerate(h):
                rows.append(dict(family=fam, sample=name, dt_ns=mid[i],
                                 n=int(v), n_total=int(tot),
                                 frac=float(v / tot) if tot else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the two-dimensional efficiency map
# --------------------------------------------------------------------------- #
def _response(run: str, subruns, reco: Path) -> pd.DataFrame:
    """What chamber A did per event, with BOTH fitted plane positions.

    `campaign_efficiency.chamber_response` reads ``x_p0`` only, because its map
    is one-dimensional.  A two-dimensional map needs ``y_p0`` as well, so the
    read is done here rather than widening the shared helper and changing what
    every existing consumer loads.
    """
    out = []
    for sub in subruns:
        p = reco / run / sub / f'mx17_{ARM}' / 'events_prelim.parquet'
        if not p.exists():
            raise FileNotFoundError(f'no merged full pass for {ARM}/{sub}: {p}')
        out.append(pd.read_parquet(
            p, columns=['event_id', 'x_ok', 'y_ok', 'n_tracks',
                        'x_p0', 'y_p0']).assign(subrun=sub))
    return pd.concat(out, ignore_index=True)


def eff_map_2d(run: str, subruns, reco: Path, slim_dir: Path = None
               ) -> tuple:
    """P(one gated track | tagged AND seeded) on a (u, v) grid, arm A.

    Returns (map frame, headline dict).  ``u`` and ``v`` are the offsets from
    the PLANE CENTRE -- the frame `acceptance.Chambers.cross` returns and the
    frame `build_tracks` writes ``x_local``/``y_local`` in -- so the map and the
    toy that consumes it index the same coordinate.  `efficiency.py` bins in
    ``local_x - PINWHEEL`` instead, which for arm A is 16.35 mm away.

    The denominator is a scintillator tag, so the absolute scale is
    Micromegas-independent; but a cell's position comes from the reconstruction,
    which only exists for a seeded event.  So the MAP is the fit-and-gate
    efficiency at a position and the HEADLINE is the absolute scale, and the
    two are reported separately rather than multiplied here.
    """
    from sept26_prelim_analysis import slim_export as SE
    from sept26_prelim_analysis.campaign_efficiency import (
        n_triggers, tagged_events)
    from sept26_prelim_analysis.efficiency import corrected
    from ntof_tracking import run145_target_imaging as TI
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y

    slim = SE.read_export(run, subruns, slim_dir)
    tag, _ = tagged_events(run, subruns, ARM, slim)
    resp = _response(run, subruns, reco)
    n_trig = n_triggers(run, subruns)

    key = list(map(tuple, resp[['subrun', 'event_id']].values))
    is_tag = np.fromiter((k in tag for k in key), bool, len(key))
    trk = resp.n_tracks.to_numpy() > 0
    one = resp.n_tracks.to_numpy() == 1
    multi = resp.n_tracks.to_numpy() > 1
    n_tag = len(tag)
    n_untag = n_trig - n_tag
    p0 = float((trk & ~is_tag).sum()) / max(n_untag, 1)
    p_tag = float((trk & is_tag).sum()) / max(n_tag, 1)
    headline = dict(run=run, arm=ARM, n_triggers=int(n_trig),
                    n_tagged=int(n_tag), n_tag_seeded=int(is_tag.sum()),
                    n_tag_tracked=int((trk & is_tag).sum()),
                    p_track_given_tag=p_tag, p0_track=p0,
                    efficiency=corrected(p_tag, p0))

    u = TI.IN_PLANE_SIGN * (resp.x_p0.to_numpy(float) - TI.STRIP_MAP_HALF)
    v = IN_PLANE_SIGN_Y * (resp.y_p0.to_numpy(float) - TI.STRIP_MAP_HALF)
    has = np.isfinite(u) & np.isfinite(v) & is_tag
    rows = []
    for iu in range(len(U_EDGES) - 1):
        for iv in range(len(V_EDGES) - 1):
            cell = (has & (u >= U_EDGES[iu]) & (u < U_EDGES[iu + 1])
                    & (v >= V_EDGES[iv]) & (v < V_EDGES[iv + 1]))
            n = int(cell.sum())
            if n < MIN_CELL:
                continue
            sden = cell & ~multi
            e_all = float(trk[cell].mean())
            e_one = (float(one[sden].mean()) if sden.sum() >= MIN_CELL
                     else np.nan)
            rows.append(dict(
                run=run, u_lo=U_EDGES[iu], u_hi=U_EDGES[iu + 1],
                u_mid=0.5 * (U_EDGES[iu] + U_EDGES[iu + 1]),
                v_lo=V_EDGES[iv], v_hi=V_EDGES[iv + 1],
                v_mid=0.5 * (V_EDGES[iv] + V_EDGES[iv + 1]),
                n=n, n_single_den=int(sden.sum()),
                eff_track=e_all, eff_single=e_one,
                err=float(np.sqrt(max(e_all * (1 - e_all), 0) / n)),
                condition=condition(run), k_block=in_block(run)))
    return pd.DataFrame(rows), headline


def map_stability(M: pd.DataFrame) -> pd.DataFrame:
    """How much each (u, v) cell moves run to run, normalised per run.

    The acceptance uses the map as a shape, so the question is not whether the
    efficiency moved but whether its PATTERN across the plane moved.
    """
    d = M[np.isfinite(M.eff_single) & ~M.run.isin(PRE_ACCESS_RUNS)].copy()
    if d.empty:
        return pd.DataFrame()
    d['shape'] = d.groupby('run').eff_single.transform(
        lambda s: s / s.mean() if s.mean() > 0 else s)
    rows = []
    for (uu, vv), g in d.groupby(['u_mid', 'v_mid']):
        if len(g) < 3:
            continue
        rows.append(dict(u_mid=uu, v_mid=vv, n_runs=int(g.run.nunique()),
                         shape_mean=float(g['shape'].mean()),
                         shape_sd=float(g['shape'].std(ddof=1)),
                         eff_mean=float(g.eff_single.mean()),
                         eff_sd=float(g.eff_single.std(ddof=1))))
    cols = ['u_mid', 'v_mid', 'n_runs', 'shape_mean', 'shape_sd',
            'eff_mean', 'eff_sd']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(['u_mid', 'v_mid'],
                                          ignore_index=True)


# --------------------------------------------------------------------------- #
# the acceptance, chamber A only
# --------------------------------------------------------------------------- #
def _eff_lookup(M: pd.DataFrame, col: str = 'eff_single'):
    """A (u, v) -> efficiency-shape interpolator, normalised to its own mean.

    Nearest-cell, not bilinear: the map is a 40 mm grid of binomial estimates
    and interpolating between them would invent structure the measurement does
    not have.  Cells the map never filled return the map's mean rather than
    zero -- a cell with no measurement is not a cell with no efficiency, and
    `acceptance.Chambers` turning that NaN into a zero is exactly the artefact
    this module exists partly to avoid.
    """
    g = M[np.isfinite(M[col])]
    if g.empty:
        return lambda u, v: np.ones(len(np.atleast_1d(u)))
    uu = np.sort(g.u_mid.unique())
    vv = np.sort(g.v_mid.unique())
    grid = np.full((len(uu), len(vv)), np.nan)
    for r in g.itertuples():
        grid[np.searchsorted(uu, r.u_mid), np.searchsorted(vv, r.v_mid)] = \
            getattr(r, col)
    mean = np.nanmean(grid)
    shape = grid / mean if mean > 0 else grid

    def f(u, v):
        iu = np.clip(np.searchsorted(uu, np.asarray(u)) - 0, 0, len(uu) - 1)
        iu = np.clip(np.abs(np.asarray(u)[:, None] - uu).argmin(1),
                     0, len(uu) - 1)
        iv = np.clip(np.abs(np.asarray(v)[:, None] - vv).argmin(1),
                     0, len(vv) - 1)
        s = shape[iu, iv]
        return np.where(np.isfinite(s), s, 1.0)
    return f


def acceptance(run: str, subruns, reco: Path, M: pd.DataFrame,
               eff: float, offset=(0.0, 0.0, 0.0), n: int = 4_000_000,
               seed: int = 41, chunk: int = 1_000_000,
               tan_min: float = None, sep_eff: pd.DataFrame = None,
               eff_u: pd.DataFrame = None, eff_v: pd.DataFrame = None
               ) -> pd.DataFrame:
    """A(theta) for intra-A pairs, thrown flat in opening angle.

    Both legs must land inside chamber A's active area and survive the
    efficiency draw; the trigger requires at least one leg to reach arm A's
    plastic bars, which is what a wall-and-plastic coincidence in A means for a
    track that already crossed the chamber.

    Three curves come back per angle bin.  ``acc`` applies the per-leg
    efficiency at each leg's own (u, v).  ``acc_slope`` additionally requires
    both planes of both legs to carry a usable timing slope -- the ``slope``
    selection in the data, applied to the toy so the two are like for like.
    ``acc_sep`` additionally applies the measured two-track resolution
    (:func:`two_track_efficiency`) at the pair's own in-plane separation, which
    is the term `acceptance.py` omits entirely and the one that dominates the
    small-angle end.
    """
    from sept26_prelim_analysis import acceptance as AC
    from sept26_prelim_analysis.normal_incidence import tan_in_plane
    from ntof_tracking.reco import geometry as G
    from wft import reco as WR

    tan_min = WR.TAN_MIN_SLOPE if tan_min is None else tan_min
    ch = AC.Chambers(run, subruns, str(reco / run), eff_map=None,
                     eff_headline={ARM: eff})
    look = _eff_lookup(M)
    # per-axis if it is available, radial only as a fallback: the loss is a
    # cross and a radial curve passes pairs that share one view's strips
    sep_look = (_uv_lookup(eff_u, eff_v)
                if eff_u is not None and eff_v is not None
                else (lambda du, dv: _sep_lookup(sep_eff)(np.hypot(du, dv))))
    bins = BINS
    mid = 0.5 * (bins[:-1] + bins[1:])
    n0 = np.zeros(len(mid))
    k_acc = np.zeros(len(mid))
    k_slope = np.zeros(len(mid))
    k_sep = np.zeros(len(mid))
    k_sep_slope = np.zeros(len(mid))
    done = c = 0
    while done < n:
        m = min(chunk, n - done)
        rng = np.random.default_rng(seed + c)
        P = AC._vertices(rng, m, offset)
        d1 = AC._isotropic(rng, m)
        th = rng.uniform(0.0, np.radians(bins[-1]), m)
        d2 = AC._rotate_by(rng, d1, th)
        keep = {}
        rel = {}
        plast = {}
        uv = {}
        for leg, D in ((1, d1), (2, d2)):
            u, v, plane, plastic = ch.cross(P, D, ARM)
            e = np.clip(eff * look(u, v), 0.0, 1.0)
            keep[leg] = plane & (rng.uniform(0, 1, m) < e)
            plast[leg] = plane & plastic
            uv[leg] = (u, v)
            tu, tv = tan_in_plane(D, ARM)
            rel[leg] = (tu >= tan_min) & (tv >= tan_min)
        both = keep[1] & keep[2] & (plast[1] | plast[2])
        du = uv[1][0] - uv[2][0]
        dv = uv[1][1] - uv[2][1]
        resolved = both & (rng.uniform(0, 1, m) < sep_look(du, dv))
        slope = rel[1] & rel[2]
        deg = np.degrees(th)
        h0, _ = np.histogram(deg, bins=bins)
        n0 += h0
        k_acc += np.histogram(deg[both], bins=bins)[0]
        k_slope += np.histogram(deg[both & slope], bins=bins)[0]
        k_sep += np.histogram(deg[resolved], bins=bins)[0]
        k_sep_slope += np.histogram(deg[resolved & slope], bins=bins)[0]
        done += m
        c += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        def rate(k):
            return np.where(n0 > 0, k / n0, np.nan)

        def err(k):
            return np.where(n0 > 0, np.sqrt(np.clip(k, 1, None)) / n0, np.nan)
    return pd.DataFrame(dict(
        run=run, group='intra_A', theta=mid, n_thrown=n0,
        n_acc=k_acc, n_acc_slope=k_slope, n_acc_sep=k_sep,
        n_acc_sep_slope=k_sep_slope,
        acc=rate(k_acc), acc_slope=rate(k_slope), acc_sep=rate(k_sep),
        acc_sep_slope=rate(k_sep_slope),
        err=err(k_acc), err_slope=err(k_slope), err_sep=err(k_sep),
        err_sep_slope=err(k_sep_slope)))


# --------------------------------------------------------------------------- #
# the fold
# --------------------------------------------------------------------------- #
def fold_models(A: pd.DataFrame, acc_col: str = 'acc', bins=BINS,
                cache: Path = None) -> pd.DataFrame:
    """The capsule and gas continua x the intra-A acceptance, binned.

    Multiplies on the 1 deg physics grid and bins afterwards, because the
    acceptance falls by more than an order of magnitude across the range and
    multiplying already-binned quantities would be wrong wherever it varies
    inside a bin.
    """
    from sept26_prelim_analysis.campaign_fold import shapes
    fine, S, prov = shapes(cache)
    g = A.sort_values('theta')
    a = np.interp(fine, g.theta.to_numpy(), g[acc_col].to_numpy(),
                  left=0.0, right=0.0)
    mid = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(fine, bins) - 1
    rows = []
    for name, y in S.items():
        w = y * a
        out = np.array([w[idx == i].sum() for i in range(len(mid))])
        tot = out.sum()
        for i, v in enumerate(out):
            rows.append(dict(model=name, acceptance=acc_col, theta=mid[i],
                             frac=float(v / tot) if tot > 0 else np.nan))
    return pd.DataFrame(rows), prov


def compare(S: pd.DataFrame, F: pd.DataFrame, selection: str,
            bins=BINS, mixed_of: str = None) -> pd.DataFrame:
    """Each folded model against one measured selection, one free normalisation.

    The event-mixed spectrum of the MATCHING selection enters the same table on
    the same footing.  It is not folded -- it is already an
    acceptance-times-whatever-the-data-is product -- and it is the only
    candidate here that contains no pair physics at all.  If it describes the
    data better than every folded continuum, that is the result.
    """
    o = S[S.selection == selection].sort_values('theta')
    if o.empty or o.n.sum() < 20:
        return pd.DataFrame()
    obs = o.n.to_numpy(float)
    err = np.sqrt(np.clip(obs, 1, None))
    cand = [(m, g.sort_values('theta').frac.to_numpy())
            for m, g in F.groupby('model')]
    mx_name = mixed_of or (selection + '+mixed' if 'mixed' not in selection
                           else None)
    if mx_name:
        mo = S[S.selection == mx_name].sort_values('theta')
        if len(mo) and mo.n.sum() > 0:
            cand.append(('event-mixed (no pair physics)',
                         mo.n.to_numpy(float) / mo.n.sum()))
    rows = []
    for model, p in cand:
        if not np.isfinite(p).any() or np.nansum(p) <= 0:
            continue
        pred = np.nan_to_num(p) * obs.sum()
        live = (obs + pred) > 0
        chi2 = float(np.sum(((obs[live] - pred[live]) / err[live]) ** 2))
        dof = int(live.sum() - 1)
        cdf = np.cumsum(pred) / pred.sum()
        rows.append(dict(
            selection=selection, model=model, n_obs=int(obs.sum()),
            chi2=chi2, dof=dof, chi2dof=chi2 / max(dof, 1),
            median_pred=float(np.interp(0.5, cdf, o.theta.to_numpy())),
            median_obs=float(np.interp(0.5, np.cumsum(obs) / obs.sum(),
                                       o.theta.to_numpy()))))
    return pd.DataFrame(rows).sort_values('chi2dof', ignore_index=True)


# --------------------------------------------------------------------------- #
# per run, and the campaign
# --------------------------------------------------------------------------- #
def one_run(run: str, src: str, reco: str, slim_dir: str | None,
            imaging_csv: str) -> tuple:
    """(run, pairs, map, headline, error) -- PASS ONE, no acceptance.

    The acceptance is thrown in a second pass because it needs the two-track
    resolution curve, and that is measured on the pooled sample: it is a
    property of the reconstruction, not of a run, and one run's pairs do not
    determine it.
    """
    try:
        src = Path(src)
        reco = Path(reco)
        subs, dropped = subruns_of(reco, run)
        if not subs:
            return run, None, None, None, 'no usable sub-runs'
        have = [s for s in subs if (src / f'tracks_{run}_{s}.parquet').exists()]
        if not have:
            return run, None, None, None, 'no stage-3 tracks'
        t = tracks(run, have, src)
        P = pairs(t, run)
        if len(P):
            P['run'] = run
            P['condition'] = condition(run)
            P['k_block'] = in_block(run)
        M, H = eff_map_2d(run, have, reco,
                          Path(slim_dir) if slim_dir else None)
        H['n_pairs'] = int((~P.mixed).sum()) if len(P) else 0
        H['n_mixed'] = int(P.mixed.sum()) if len(P) else 0
        H['n_subruns'] = len(have)
        H['subruns'] = have
        H['offset_mm'] = list(_offset(run, imaging_csv))
        H['dropped_subruns'] = dropped
        return run, P, M, H, ''
    except Exception:
        return run, None, None, None, traceback.format_exc(limit=3)


def one_throw(run: str, subruns, reco: str, M_csv: str, eff: float,
              offset, n_throw: int, sep_csv: str) -> tuple:
    """(run, acceptance, error) -- PASS TWO, the toy with everything measured."""
    try:
        M = pd.read_csv(M_csv)
        M = M[M.run == run] if 'run' in M.columns else M
        E = pd.read_csv(sep_csv) if sep_csv else None
        EU = E[E.axis == 'u'] if E is not None and 'axis' in E.columns else None
        EV = E[E.axis == 'v'] if E is not None and 'axis' in E.columns else None
        A = acceptance(run, subruns, Path(reco), M, float(eff),
                       offset=tuple(offset), n=n_throw, sep_eff=E,
                       eff_u=EU, eff_v=EV)
        A['offset_x_mm'] = offset[0]
        A['offset_z_mm'] = offset[2]
        return run, A, ''
    except Exception:
        return run, None, traceback.format_exc(limit=3)


def _offset(run: str, imaging_csv: str) -> tuple:
    """(X, 0, Z) mm from the campaign pointing crossings, campaign median if absent."""
    PR = pd.read_csv(imaging_csv)
    g = PR[PR.run == run]
    x = g[(g.axis == 'X') & (g.arm.isin(('A', 'C')))].mm
    z = g[(g.axis == 'Z') & (g.arm == 'D')].mm
    mx = PR[(PR.axis == 'X') & (PR.arm.isin(('A', 'C')))].mm.median()
    mz = PR[(PR.axis == 'Z') & (PR.arm == 'D')].mm.median()
    return (float(x.mean()) if len(x) else float(mx), 0.0,
            float(z.mean()) if len(z) else float(mz))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.out('stage3_fullpass')))
    ap.add_argument('--reco', default=str(paths.out('reco_fullpass')))
    ap.add_argument('--slim', default=None)
    ap.add_argument('--imaging', default=str(paths.out('imaging_campaign')
                                             / 'per_run.csv'))
    ap.add_argument('--runs', default='')
    ap.add_argument('--n', type=int, default=4_000_000)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--include-pre-access', action='store_true')
    a = ap.parse_args()

    src = Path(paths.require(a.src, 'the stage-3 track table'))
    reco = Path(paths.require(a.reco, 'the full-pass reco tree'))
    pr = str(paths.require(a.imaging, 'the campaign pointing crossings'))
    runs = ([r for r in a.runs.split(',') if r] or
            sorted((p.name for p in reco.iterdir()
                    if p.is_dir() and p.name.startswith('run_')),
                   key=run_number))
    if not a.include_pre_access:
        runs = [r for r in runs if r not in PRE_ACCESS_RUNS]
    print(f'detector {ARM}, intra-chamber pairs -- {len(runs)} run(s)\n')

    args = (str(src), str(reco), a.slim, pr)
    results = []
    if a.jobs <= 1:
        results = [one_run(r, *args) for r in runs]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            fut = {ex.submit(one_run, r, *args): r for r in runs}
            for f in as_completed(fut):
                results.append(f.result())
    PP, MM, HH, failed = [], [], [], {}
    for run, P, M, H, err in sorted(results, key=lambda r: run_number(r[0])):
        if err:
            failed[run] = err.strip().splitlines()[-1]
            print(f'  {run}: FAILED -- {failed[run]}', flush=True)
            continue
        if P is not None and len(P):
            PP.append(P)
        MM.append(M)
        HH.append(H)
        print(f'  {run}: {H["n_pairs"]:>5,} pairs, {len(M):>2} map cells, '
              f'eff {100 * H["efficiency"]:.1f} %', flush=True)
    if not PP:
        print('\nno pairs')
        return 1

    P = pd.concat(PP, ignore_index=True)
    M = pd.concat(MM, ignore_index=True)
    H = pd.DataFrame(HH)

    od = paths.out('det_a_intra')
    # the two-track resolution, measured on the pooled sample, THEN thrown.
    # Per axis, because the loss is a cross and not a disc; the radial curve is
    # kept beside it because it is what the first version of this page used.
    E2 = two_track_efficiency(P)
    EU = two_track_efficiency_axis(P, 'u')
    EV = two_track_efficiency_axis(P, 'v')
    SEP = separability_check(P, EU, EV)
    E2.to_csv(od / 'two_track_efficiency.csv', index=False)
    pd.concat([EU, EV], ignore_index=True).to_csv(
        od / 'two_track_efficiency_axis.csv', index=False)
    SEP.to_csv(od / 'two_track_separability.csv', index=False)
    M.to_csv(od / 'eff_map_2d.csv', index=False)
    print(f'\ntwo-track resolution measured on {int((~P.mixed).sum()):,} real '
          f'and {int(P.mixed.sum()):,} mixed pairs')
    if not EU.empty and not EV.empty:
        print(f'{"separation":>14} {"eff in u":>9} {"eff in v":>9}')
        # NOT `a`: that is the argparse namespace, and shadowing it here broke
        # the throw stage three lines later
        for ru, rv in zip(EU[EU.sep_mid < 220].itertuples(),
                          EV[EV.sep_mid < 220].itertuples()):
            print(f'{ru.sep_lo:6.0f}-{ru.sep_hi:<7.0f}{ru.eff:>9.3f} '
                  f'{rv.eff:>9.3f}')
    if not SEP.empty:
        rms = float(np.sqrt((SEP.residual ** 2).mean()))
        print(f'  separable model vs the 2D map: rms residual {rms:.3f} over '
              f'{len(SEP)} cells')

    targs = [(h['run'], h['subruns'], str(reco), str(od / 'eff_map_2d.csv'),
              float(h['efficiency']), h['offset_mm'], a.n,
              str(od / 'two_track_efficiency_axis.csv')) for h in HH]
    print(f'\nthrowing {a.n:,} pairs per run through the measured acceptance\n')
    if a.jobs <= 1:
        tres = [one_throw(*t) for t in targs]
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            fut = [ex.submit(one_throw, *t) for t in targs]
            tres = [f.result() for f in as_completed(fut)]
    AA = []
    for run, A, err in sorted(tres, key=lambda r: run_number(r[0])):
        if err:
            failed[run] = err.strip().splitlines()[-1]
            print(f'  {run}: THROW FAILED -- {failed[run]}', flush=True)
            continue
        AA.append(A)
    if not AA:
        print('\nno acceptance thrown')
        return 1
    A = pd.concat(AA, ignore_index=True)

    # the campaign acceptance: each run's curve weighted by its own pairs
    w = P[~P.mixed].groupby('run').size()
    A['w'] = A.run.map(w).astype(float)
    ACC_COLS = ('acc', 'acc_slope', 'acc_sep', 'acc_sep_slope')
    pooled = (A[A.w.notna()].groupby('theta')
              .apply(lambda g: pd.Series(
                  {c: float((g.w * g[c]).sum() / g.w.sum()) for c in ACC_COLS}
                  | dict(n_runs=int(len(g)),
                         sd=float(g.acc_sep.std(ddof=1)) if len(g) > 1
                         else 0.0)),
                  include_groups=False)
              .reset_index())

    K = census(P)
    S = spectra(P)
    real = ~P.mixed.to_numpy()
    D = dt0_models(P.dt0_ns.to_numpy()[real])
    Dslope = dt0_models(P.dt0_ns.to_numpy()[real & P.both_slope.to_numpy()])
    if not Dslope.empty:
        Dslope.insert(0, 'sample', 'slope')
        D.insert(0, 'sample', 'all')
        D = pd.concat([D, Dslope], ignore_index=True)
    Hd = dt0_histogram(P)
    Tg = tag_test(P)
    G2 = dt_sep_grid(P)
    G2s = dt_sep_grid(P, real & P.both_slope.to_numpy())
    G2s.insert(0, 'selection', 'slope')
    G2.insert(0, 'selection', 'all')
    G2 = pd.concat([G2, G2s], ignore_index=True)
    Pr = dt_sep_profile(P)
    Sl = slope_profile(P)
    VX = vertex_excess(P)
    MS = map_stability(M)

    FF = []
    for c in ACC_COLS:
        f, prov = fold_models(pooled, c, cache=od / 'shape_cache.csv')
        FF.append(f)
    F = pd.concat(FF, ignore_index=True)
    # each measured selection is compared against the acceptance that matches
    # it: the slope selection against the slope-required toy, and everything
    # against the toy that carries the measured two-track resolution
    PAIRING = {'all': 'acc_sep', 'prompt': 'acc_sep', 'offtime': 'acc_sep',
               'pointing': 'acc_sep', 'slope': 'acc_sep_slope',
               'slope+prompt': 'acc_sep_slope',
               'slope+offtime': 'acc_sep_slope',
               'slope+pointing': 'acc_sep_slope', 'mixed': 'acc_sep',
               'slope+mixed': 'acc_sep_slope'}
    MIXED_OF = {'all': 'mixed', 'slope': 'slope+mixed',
                'prompt': 'mixed', 'offtime': 'mixed',
                'slope+prompt': 'slope+mixed',
                'slope+offtime': 'slope+mixed',
                'pointing': 'pointing+mixed',
                'slope+pointing': 'slope+pointing+mixed'}
    C = pd.concat([compare(S, F[F.acceptance == acc], sel,
                           mixed_of=MIXED_OF.get(sel))
                   .assign(acceptance=acc)
                   for sel, acc in PAIRING.items()]
                  + [compare(S, F[F.acceptance == 'acc'], sel,
                             mixed_of=MIXED_OF.get(sel))
                     .assign(acceptance='acc')
                     for sel in ('all', 'slope')],
                  ignore_index=True)

    # the n_TOF timing, on one representative run: the slim read is the
    # expensive part and the shape is a property of the trigger, not the run
    ref = 'run_145' if 'run_145' in set(P.run) else sorted(set(P.run))[0]
    subs = sorted(set(P[P.run == ref].subrun))
    N = ntof_timing(ref, subs, P[P.run == ref],
                    Path(a.slim) if a.slim else None)
    N.insert(0, 'run', ref)

    P.to_parquet(od / 'pairs.parquet', index=False)
    MS.to_csv(od / 'eff_map_stability.csv', index=False)
    H.to_csv(od / 'headline_per_run.csv', index=False)
    A.drop(columns=['w']).to_csv(od / 'acceptance_per_run.csv', index=False)
    pooled.to_csv(od / 'acceptance_pooled.csv', index=False)
    K.to_csv(od / 'census.csv', index=False)
    S.to_csv(od / 'spectra.csv', index=False)
    D.to_csv(od / 'dt0_models.csv', index=False)
    Hd.to_csv(od / 'dt0_histogram.csv', index=False)
    Tg.to_csv(od / 'tag_test.csv', index=False)
    G2.to_csv(od / 'dt_sep_grid.csv', index=False)
    Pr.to_csv(od / 'dt_sep_profile.csv', index=False)
    Sl.to_csv(od / 'slope_profile.csv', index=False)
    VX.to_csv(od / 'vertex_excess.csv', index=False)
    F.to_csv(od / 'folded.csv', index=False)
    C.to_csv(od / 'compare.csv', index=False)
    N.to_csv(od / 'ntof_timing.csv', index=False)
    json.dump(dict(
        schema=SCHEMA, arm=ARM, src=str(src), reco=str(reco),
        runs=sorted(set(P.run)), n_runs=int(P.run.nunique()),
        n_pairs=int((~P.mixed).sum()), n_mixed=int(P.mixed.sum()),
        n_throw=a.n, vertex_r_mm=VERTEX_R_MM, vertex_dca_mm=VERTEX_DCA_MM,
        dca_max_mm=DCA_MAX, prompt_ns=PROMPT_NS, offtime_ns=list(OFFTIME_NS),
        bins=[float(x) for x in BINS],
        u_edges=[float(x) for x in U_EDGES],
        v_edges=[float(x) for x in V_EDGES],
        min_cell=MIN_CELL, selections=list(SELECTIONS),
        two_track_model='separable per-axis, eff(|du|) x eff(|dv|); the loss is a cross, not a disc',
        other_min_mm=OTHER_MIN_MM,
        mixed_pool='tracks from triggers with >= 2 arm-A tracks',
        ntof_reference_run=ref, include_pre_access=a.include_pre_access,
        runs_failed=failed, shape_provenance=prov,
        frame='u, v are offsets from the PLANE CENTRE, the frame '
              'acceptance.Chambers.cross returns; efficiency.py bins in '
              'local_x - PINWHEEL, which for arm A is 16.35 mm away'),
        open(od / 'det_a_intra.meta.json', 'w'), indent=1)

    print(f'\n{int((~P.mixed).sum()):,} real and {int(P.mixed.sum()):,} '
          f'event-mixed intra-{ARM} pairs over {P.run.nunique()} runs\n')
    print(f'{"selection":>20} {"n":>8} {"median":>8} {"p90":>7} '
          f'{"vertexed":>9} {"med dca":>8}')
    for r in K.itertuples():
        print(f'{r.selection:>20} {r.n:>8,} {r.median_open_deg:>7.1f}d '
              f'{r.p90_open_deg:>6.1f}d {r.frac_vertex:>9.3f} '
              f'{r.median_dca_pair_mm:>7.1f}mm')

    if not D.empty:
        print('\nthe in-chamber clock: five two-component fits to dt0, '
              'all-pairs sample\n')
        print(f'{"model":>20} {"prompt f":>9} {"nll":>10} {"chi2/dof":>9}')
        for r in D[D['sample'] == 'all'].itertuples():
            print(f'{r.model:>20} {r.prompt_fraction:>9.3f} {r.nll:>10.1f} '
                  f'{r.chi2dof:>9.2f}')
        g = D[D['sample'] == 'all']
        print(f'  prompt fraction spans {g.prompt_fraction.min():.2f} to '
              f'{g.prompt_fraction.max():.2f} over a likelihood range of '
              f'{g.nll.max() - g.nll.min():.1f}: NOT IDENTIFIABLE')

    if not Sl.empty:
        print('\nand what the peak actually tracks: the pair\'s smallest '
              'in-plane slope\n')
        print(f'{"|tan| range":>16} {"n":>7} {"core/wing":>10} '
              f'{"median open":>12} {"median sep":>11}')
        for r in Sl.itertuples():
            mark = ' *' if r.below_tan_min else '  '
            print(f'{r.tan_lo:7.3f}-{r.tan_hi:<7.3f}{mark}{r.n:>7,} '
                  f'{r.core_over_wing:>10.3f} {r.median_open_deg:>11.1f}d '
                  f'{r.median_sep_mm:>10.1f}mm')
        print('  * below TAN_MIN_SLOPE: no timing slope, so t0 is degenerate')
    if not Tg.empty:
        print('\nthe scintillator cross-check on that peak\n')
        print(f'{"sample":>28} {"n":>7} {"core/wing":>10}')
        for r in Tg.itertuples():
            print(f'{r.sample:>28} {r.n:>7,} {r.core_over_wing:>10.3f}')

    if not VX.empty:
        print('\ndo the two legs converge on the axis more often than '
              'chance?\n')
        print(f'{"selection":>10} {"real":>9} {"mixed":>9} {"ratio":>14} '
              f'{"excess":>9}')
        for r in VX.itertuples():
            print(f'{r.selection:>10} {r.k_real:>4}/{r.n_real:<7,} '
                  f'{r.k_mixed:>4}/{r.n_mixed:<7,} '
                  f'{r.ratio:>7.2f} +- {r.err:<4.2f} '
                  f'{r.excess_sigma:>8.1f}s')
    print('\nfolded models against the measured spectrum\n')
    print(f'{"selection":>16} {"model":>28} {"chi2/dof":>9} '
          f'{"median pred":>12} {"median obs":>11}')
    for sel in ('all', 'slope', 'slope+prompt'):
        for r in C[(C.selection == sel)
                   & (C.acceptance == PAIRING[sel])].head(3).itertuples():
            print(f'{r.selection:>16} {r.model:>28} {r.chi2dof:>9.1f} '
                  f'{r.median_pred:>11.1f}d {r.median_obs:>10.1f}d')
    if failed:
        print(f'\n{len(failed)} run(s) failed: {", ".join(failed)}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
