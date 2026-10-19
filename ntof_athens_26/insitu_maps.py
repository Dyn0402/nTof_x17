#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insitu_maps.py -- what each chamber surface sees, with and without its own trigger.

THE PROBLEM THIS SOLVES.  Every hit map in this analysis so far is a map of the
TRIGGER as much as of the detector.  The production trigger is a wall AND
plastic coincidence in one arm, so a chamber is only ever read out when
something crossed the two scintillator layers *behind it* -- and those layers do
not cover the chamber.  The plastic is two 200 mm-wide bars with a gap between
them, which prints a shadow at u ~ +7 mm on every chamber; the bars are 300 mm
long against the chamber's 400, which cuts v at +-150; the wall's four read-out
groups span u = -225..+175, which blinds the last 24 mm of the surface.  None of
that is the Micromegas, and all of it is in the map.

THE MEASUREMENT THAT REMOVES IT, and it needs no model and no subtraction.
**The trigger is a single-arm coincidence OR'd over the four arms**, and of the
read-outs this analysis can tag offline, **97 % have exactly ONE arm lit**
(measured here, :func:`trigger_census`: 8.66 M of 8.94 M tagged events, against
274 k with two arms and 6.7 k with three).  So for any chamber X there is a
large sample of events in which X's own scintillators were silent and some
*other* arm made the trigger.  In those events chamber X is a **bystander**: it
is read out for reasons that have nothing to do with its own surface, and its
occupancy is illumination x efficiency over the WHOLE chamber with no
scintillator acceptance folded in.

A further **14.6 % of read-outs carry no offline tag in any arm** and are the
``no_tag`` class below -- the DAQ's own discriminator and this 160 ns window are
not the same cut, and the per-run n_TOF join is not perfect (the zero-arm
fraction runs from 0.2 % to 35 % run to run, which is join quality and not
physics).  They are dropped from BOTH samples alike, so they cannot bias the
comparison; what they could in principle do is let a partial slim loss put a
self-triggered track into the bystander sample.  The data bounds that directly:
if it happened at any scale, the trigger's gap shadow would partly survive in
chamber A's bystander map, and it does not -- it inverts.

That is the unbiased map, and three things make it better than it looks:

  * **It needs no track removal.**  The bystander sample never contained the
    triggering particle in the first place, so there is nothing to identify and
    nothing to subtract -- the ambiguity of "which track caused the trigger"
    simply does not arise.  The removal method is built anyway, as ``leftover``,
    because an independent check on a new method is worth more than the
    argument that it is unnecessary.
  * **It needs no angle.**  A bystander is defined by which arm's scintillators
    fired, not by where a track points, so the selection is pure n_TOF.  That
    makes it the ONE map in this analysis that works identically in all four
    chambers -- **chamber B included**, which has no drift field, no angle and
    nothing to extrapolate (STATUS.md, 2026-09-08).
  * **It has a falsifiable prediction.**  The plastic-gap shadow at u ~ +7 mm is
    the trigger's, so it must VANISH in the bystander map -- except where a
    chamber has dead readout channels in the same place, and C and D do
    (~130 dead channels across D's connectors, ~10 on C, STATUS.md) while
    chamber A has none at all.  So A's dip must disappear and D's must stay.
    That is the test, declared before it is run, and :func:`shadow_profile`
    measures it.

THE FIVE SAMPLES, and every track lands in exactly one so the census is a
partition and every fraction has an honest denominator:

  ``triggered``       X's own wall AND plastic fired, and this track points at
                      the group and the bar that fired.  The trigger-biased map,
                      at its purest -- this is (very probably) the particle that
                      caused the read-out.
  ``leftover``        X self-triggered and some *other* track in the same event
                      took the match.  This is the "remove the triggering track
                      and see what is left" sample.  It is small -- a
                      self-trigger with a second reconstructed track is rare --
                      and it is kept as an INDEPENDENT CHECK on the bystander
                      map rather than as the main result.
  ``bystander``       X's scintillators silent, another arm's fired.  **The
                      unbiased map.**
  ``self_unmatched``  X self-triggered and NO track in the event matched the
                      fired channel.  Ambiguous by construction -- the
                      triggering particle was not reconstructed, or the match
                      failed -- so it is counted, reported, and mapped nowhere.
  ``no_tag``          no arm reconstructs a wall+plastic coincidence offline at
                      all (~4 % of read-outs: the DAQ's own discriminator and
                      this window are not the same cut).  Counted, never mapped.

WHAT THE BYSTANDER MAP IS NOT, and there are three things, all measured rather
than conceded.

**It is occupancy, not efficiency.**  It has no independent denominator, so a
cold cell is "less illuminated or less efficient" and this module never says
which.  What it removes is the chamber's own scintillator acceptance, and that
is all it claims to remove.

**It is not purity-selected.**  Nothing confirms any track in it -- that is the
price of not using the chamber's own scintillators -- so it is diluted with junk
that the matched samples drop.  Measured on run_145: median cluster width is
38/46/49/42 strips on the bystander sample against 25/43/34/42 on all gated
clusters.  Position is unaffected, which is why the maps stand; *shape* is not,
which is why :func:`shape` is deliberately computed on a different sample.

**It still shares a beam.**  A bystander sits in an event that some other arm
triggered, so the illumination is the beam's own and is not uniform, and an
opposing arm's trigger selects a direction through the target which weights the
surface gently.  Neither is the chamber's own trigger footprint, which is the
thing being removed.

    python ntof_athens_26/insitu_maps.py --runs run_145
    python ntof_athens_26/insitu_maps.py --jobs 6          # the campaign

On the Windows box point the tree at the drive letter first::

    X17_ROOT=D:/x17 X17_BEAM_JULY=D:/x17/beam_july \
        python ntof_athens_26/insitu_maps.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                      # noqa: E402
from sept26_prelim_analysis import det_a_scint as DAS         # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (         # noqa: E402
    condition, in_block, run_number, subruns_of)

SCHEMA = 'athens26/insitu_maps/1'
ARMS = ('A', 'B', 'C', 'D')

#: The accept window, and it is the SAME one `det_a_scint` and `efficiency` use,
#: so a rate here is readable against a rate there.
DT_WINDOW = DAS.DT_WINDOW

#: Map binning, mm.  The active area is 398.58 mm square; 8 mm cells give 50x50,
#: which is set by how many tracks a cell needs and not by resolution -- the
#: strip pitch is 0.4 mm and the extrapolation is good to ~2 mm.
BIN_MM = 8.0
EDGES = np.arange(-200.0, 200.0 + 0.5 * BIN_MM, BIN_MM)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])

#: The samples, in the order every table and every figure reads them.
SAMPLES = ('triggered', 'leftover', 'bystander', 'self_unmatched', 'no_tag')
#: The three that are mapped.  The other two are censused and not drawn.
MAPPED = ('triggered', 'bystander', 'leftover')

#: The plastic-gap shadow.  Predicted centre u = +7.3/7.0/7.7/6.9 mm for
#: A/B/C/D from the Geant geometry (STATUS.md), so one window serves all four.
#: The flanks are where the occupancy reads as "unshadowed", set wide enough to
#: clear the ~30 mm smearing the dca selection alone imposes.  Every bound is a
#: BIN EDGE, because the shadow is measured on the masked map rather than on the
#: tracks -- a window that cut a bin in half would have to guess how the bin's
#: contents are distributed inside it.
SHADOW_U = (-16.0, 32.0)
FLANK_U = ((-88.0, -40.0), (56.0, 104.0))
#: Read the shadow only where the plastic could cast one: inside its own v.
SHADOW_V = 152.0

#: The FIDUCIAL half-width, mm.  Outside it a reconstructed position is not a
#: position on the detector: the plane fit RAILS at the edge of the strip map
#: and piles up there.  STATUS.md records it for arm A in v ("a fifth of the
#: arm-A track table lands outside the chamber in v, 14 % of all tracks in one
#: 20 mm window") and `det_a_scint.RAIL_V` is that window.  Measured again here
#: on the pooled maps, the pile-up is confined to |v| >= 180 mm (arm A 14x and
#: 5x its own median in the outer two bins, arm D 26x and 7x) and |u| >= 184
#: (arm D 20x and 9x); everything inside is ordinary.  176 mm is the bin edge
#: that clears both with a bin to spare, and it keeps 77 % of the active area.
#: Cells outside it are MASKED, not dropped, so the maps still show the full
#: surface and the reader can see where the detector stops being readable.
FIDUCIAL = 176.0

#: A map cell is HOT when it exceeds this multiple of its neighbourhood median.
#: 8x is far above anything illumination does -- the plastic gap, the deepest
#: real structure on these maps, is a factor of ~4 the other way -- and it is
#: the same local-neighbourhood logic `noisy_channels.py` uses at channel
#: granularity, for the same reason: a plane-wide median is dragged by the
#: trigger's own two-lobe illumination, which is real and must not be flagged.
#: 5x, over a 9-cell neighbourhood.  Both were chosen by scanning window in
#: (5, 9, 11, 13) against factor in (4, 5, 8) and looking at what each setting
#: costs the HEALTHY chambers, because the unhealthy one cannot be rescued by
#: any of them: chamber D's ratio of 99th percentile to median cell occupancy
#: is 207 at 5/8 and still 68 at 13/4, where the mask has taken 45 % of its
#: sample.  A, B and C sit at 4.7-5.5 throughout and are insensitive to the
#: choice.  So the setting is picked to avoid over-masking A and B -- at
#: factor 4 chamber B loses 40 cells and 7 % of its sample to rows that
#: `noisy_channels.py` does not call hot, and which are more likely real -- and
#: **D's map is reported as noise-dominated rather than tuned until it looks
#: clean.**  A 5-cell window was the first choice and is too narrow: D's hot
#: bands are whole connectors, 3 map cells wide, so a 5x5 window centred inside
#: one is half hot and its median is dragged up with it.
HOT_FACTOR = 5.0
#: Neighbourhood, in cells, over which that median is taken.
HOT_WINDOW = 9
#: Never flag a cell on small numbers: a 3-count cell beside empty neighbours is
#: Poisson noise, not a hot channel.
HOT_MIN_COUNT = 25

#: Columns read from the stage-3 table.  Narrow on purpose -- the campaign table
#: is 24 GB and this module wants positions and flags.
TRACK_COLS = [
    'run', 'subrun', 'event_id', 'arm', 'track_id', 'gated', 'angle_calibrated',
    'x_local', 'y_local', 'tanx', 'tany',
    'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
    'q_total', 'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips',
]


# --------------------------------------------------------------------------- #
# the trigger, read from the n_TOF slim
# --------------------------------------------------------------------------- #
def trigger_sets(run: str, subruns, slim_dir: Path | None,
                 window: tuple = DT_WINDOW) -> dict:
    """``{(subrun, event_id): frozenset of arms}`` -- who made the trigger.

    An arm is "lit" when BOTH its wall and its plastic have a hit in ``window``,
    which is the production trigger's own condition and the same one stage 1
    applies.  Read once per run for all four arms at once, because the whole
    method turns on comparing one arm's tag with the other three.
    """
    d = Path(slim_dir) if slim_dir else paths.out('slim')
    lo, hi = window
    cols = ['eventId', 'det', 'dt_ns', 'is_control', 'subrun']
    out, missing = [], []
    for sub in subruns:
        p = d / f'ntof_hits_{run}_{sub}.parquet'
        if not p.exists():
            missing.append(sub)
            continue
        x = pd.read_parquet(p, columns=cols)
        out.append(x[(x.is_control == 0) & (x.det < 8)
                     & (x.dt_ns >= lo) & (x.dt_ns <= hi)])
    if missing:
        raise FileNotFoundError(
            f'no exported slim for {run} sub-run(s) {", ".join(missing)} under '
            f'{d}\n  A sub-run silently skipped here would read as a sub-run in '
            f'which no scintillator ever fired, which moves every track in it '
            f'into the bystander sample. Refusing.')
    x = pd.concat(out, ignore_index=True)
    x['subrun'] = x.subrun.astype(str)

    lit = {}
    for i, arm in enumerate(ARMS):
        w = set(map(tuple, x.loc[x.det == i, ['subrun', 'eventId']]
                    .drop_duplicates().to_numpy()))
        p = set(map(tuple, x.loc[x.det == i + 4, ['subrun', 'eventId']]
                    .drop_duplicates().to_numpy()))
        lit[arm] = w & p                       # wall AND plastic: the trigger
    keys = set().union(*lit.values())
    return {k: frozenset(a for a in ARMS if k in lit[a]) for k in keys}


def trigger_census(trig: dict, keys: set) -> pd.DataFrame:
    """How many arms are lit per read-out -- the fact the method rests on.

    If this were not overwhelmingly ONE, the bystander sample would be a
    footnote instead of a map.

    ``keys`` is the population this is a partition OF, and it is the events that
    carry a gated track in at least one arm -- the population the maps are drawn
    from, and the only denominator this module can state honestly.  It is NOT
    every DREAM read-out: an event in which nothing reconstructed anywhere never
    reaches this module, and the stage-1 candidate ledger is where a
    per-read-out denominator lives (`candidate_filter`).  So the ``n_arms = 0``
    row here is "mapped events whose scintillators say nothing", not "triggers
    with no coincidence".
    """
    n = pd.Series([len(trig.get(k, ())) for k in keys]).value_counts()
    d = (pd.DataFrame([dict(n_arms=int(k), n=int(v)) for k, v in n.items()])
         .groupby('n_arms', as_index=False).n.sum())
    d['frac'] = d.n / max(d.n.sum(), 1)
    return d.sort_values('n_arms').reset_index(drop=True)


# --------------------------------------------------------------------------- #
# one arm, one run
# --------------------------------------------------------------------------- #
def arm_tracks(run: str, subruns, src: Path, arm: str, trig: dict,
               geo: dict | None, slim_dir: Path | None) -> pd.DataFrame:
    """Every gated track of one arm, positioned and sampled.

    ``geo`` is ``None`` for chamber B: it has no field-shaping ring chain, so no
    uniform drift field, no time-to-depth ladder and no direction to
    extrapolate.  B therefore gets position and the sample label -- which is all
    the bystander map needs -- and no pointing match.  That is not a degraded
    version of the others' treatment; it is the same treatment with the one
    ingredient B cannot supply left out, and it is why B appears on the unbiased
    map at all.

    ``x_local``/``y_local`` are used for position throughout, in every chamber,
    because they descend from the plane-fit intercepts and the strip map and so
    do **not** depend on the drift velocity or on ``k``
    (``build_tracks.ANGLE_DERIVED`` lists what does).  The extrapolated
    crossings are used only to decide which scintillator channel a track points
    at, never to place it on the chamber.
    """
    T = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {run}/{sub}')
        d = pd.read_parquet(p, columns=TRACK_COLS)
        T.append(d[(d.arm == arm) & d.gated].assign(subrun=sub))
    t = pd.concat(T, ignore_index=True)
    if not len(t):
        return t
    t['run'] = run
    t = t.rename(columns={'x_local': 'u_mm', 'y_local': 'v_mm'})
    t['n_trk'] = t.groupby(['subrun', 'event_id']).event_id.transform('size')

    key = list(zip(t.subrun, t.event_id))
    lit = [trig.get(k, frozenset()) for k in key]
    t['self_trig'] = np.fromiter((arm in s for s in lit), bool, len(t))
    t['other_trig'] = np.fromiter((bool(s - {arm}) for s in lit), bool, len(t))
    t['n_trig_arms'] = np.fromiter((len(s) for s in lit), int, len(t))

    # --- the pointing match, where there is a direction to point ------------
    if geo is None:
        t['has_dir'] = False
        t['matched'] = False
        for c in ('u_wall', 'v_wall', 'u_plas', 'v_plas'):
            t[c] = np.nan
        t['grp_pred'] = -1
        t['plas_pred'] = -1
    else:
        cal = t.angle_calibrated.astype('boolean').fillna(False).to_numpy()
        D = t[['d_x', 'd_y', 'd_z']].to_numpy(float)
        t['has_dir'] = cal & np.isfinite(D).all(axis=1)
        P0 = t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        for tag, w in (('wall', geo['w_wall']), ('plas', geo['w_plas'])):
            t[f'u_{tag}'], t[f'v_{tag}'] = DAS.project(P0, D, geo, w)

        gu = DAS.group_u(geo)
        u_w = t.u_wall.to_numpy()
        grp = np.full(len(t), -1)
        for g, (lo, _c, hi) in gu.items():
            grp[(u_w >= lo) & (u_w < hi)] = g
        on_w = ((grp >= 0)
                & (np.abs(t.v_wall.to_numpy() - geo['v_mm']) <= DAS.SIPM_HALF_V)
                & t.has_dir.to_numpy())
        t['grp_pred'] = np.where(on_w, grp, -1)

        pn = np.array(sorted(geo['plas_u']))
        pu = np.array([geo['plas_u'][n] for n in pn])
        pi = np.argmin(np.abs(t.u_plas.to_numpy()[:, None] - pu[None, :]),
                       axis=1)
        on_p = ((np.abs(t.u_plas.to_numpy() - pu[pi]) <= DAS.PLASTIC_HALF_U)
                & (np.abs(t.v_plas.to_numpy() - geo['v_mm'])
                   <= DAS.PLASTIC_HALF_V)
                & t.has_dir.to_numpy())
        t['plas_pred'] = np.where(on_p, pn[pi], -1)

        # Which channel actually fired, this arm, this event.
        slim = DAS.read_arm_slim(run, subruns, arm, slim_dir)
        W, P, _L = DAS.fired_sets(slim, arm,
                                  (0, DT_WINDOW[0], DT_WINDOW[1]))
        gp, pp = t.grp_pred.to_numpy(), t.plas_pred.to_numpy()
        wf = [W.get(k, frozenset()) for k in key]
        pf = [P.get(k, frozenset()) for k in key]
        mw = np.fromiter((g >= 0 and g in s for g, s in zip(gp, wf)),
                         bool, len(t))
        mp = np.fromiter((p > 0 and p in s for p, s in zip(pp, pf)),
                         bool, len(t))
        t['matched'] = mw & mp

    t['sample'] = assign_sample(t)
    return t


def assign_sample(t: pd.DataFrame) -> np.ndarray:
    """One label per track.  A partition -- every track gets exactly one.

    The order matters and is stated: a self-triggered event is classified by
    whether ANY of its tracks took the match, so ``leftover`` is exactly "the
    tracks that are not the one that fired the trigger" and nothing else falls
    into it by default.
    """
    self_t = t.self_trig.to_numpy()
    other = t.other_trig.to_numpy()
    matched = t.matched.to_numpy()
    has_match = (t.assign(_m=matched)
                 .groupby(['subrun', 'event_id'])._m.transform('any')
                 .to_numpy())

    s = np.full(len(t), 'no_tag', dtype=object)
    s[~self_t & other] = 'bystander'
    s[self_t & ~has_match] = 'self_unmatched'
    s[self_t & has_match & ~matched] = 'leftover'
    s[self_t & matched] = 'triggered'
    return s


# --------------------------------------------------------------------------- #
# maps and profiles
# --------------------------------------------------------------------------- #
def histogram(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Long-format 2D occupancy, (arm, sample, u, v) -> n."""
    rows = []
    for s in MAPPED:
        d = t[t['sample'] == s]
        H, _, _ = np.histogram2d(d.u_mm.to_numpy(), d.v_mm.to_numpy(),
                                 bins=[EDGES, EDGES])
        iu, iv = np.nonzero(H)
        rows.append(pd.DataFrame(dict(
            arm=arm, sample=s, u=CENTRES[iu], v=CENTRES[iv],
            n=H[iu, iv].astype(np.int64))))
    return pd.concat(rows, ignore_index=True)


def as_grid(maps: pd.DataFrame, arm: str, sample: str) -> np.ndarray:
    """The long map table back into a dense (u, v) array, zero-filled."""
    d = maps[(maps.arm == arm) & (maps['sample'] == sample)]
    H = np.zeros((len(CENTRES), len(CENTRES)))
    if len(d):
        iu = np.searchsorted(CENTRES, d.u.to_numpy())
        iv = np.searchsorted(CENTRES, d.v.to_numpy())
        H[iu, iv] = d.n.to_numpy()
    return H


def fiducial_mask() -> np.ndarray:
    """Cells outside :data:`FIDUCIAL` -- where the plane fit rails."""
    out = np.abs(CENTRES) > FIDUCIAL
    return out[:, None] | out[None, :]


def hot_mask(H: np.ndarray) -> np.ndarray:
    """Cells that are a hot channel rather than a feature of the chamber.

    WHY THIS IS NEEDED, and it is not cosmetic.  The bystander sample is the
    one place hot-channel junk accumulates, **by construction**: a cluster
    manufactured by a noisy strip points at no scintillator, so it can never be
    matched, so it lands in ``bystander`` or ``self_unmatched`` and never in
    ``triggered``.  On chamber D that is not a small effect -- 28 % of the
    bystander sample sits in ten cells of 2 500, and `noisy_channels.py`
    independently finds 8.2 % of D's x channels and 11.1 % of its y channels
    hot, carrying over half its raw hits.  Left in, they set the colour scale
    and the map shows nothing else.

    THE DEFINITION IS COMMON TO EVERY SAMPLE.  The mask is derived once, from
    the occupancy summed over all mapped samples, and then applied to each of
    them alike -- so it cannot manufacture a difference between the two samples
    the figures compare.  That is the whole reason it is not derived per sample.

    A cell is hot when it holds at least :data:`HOT_MIN_COUNT` tracks AND
    exceeds :data:`HOT_FACTOR` times the median of the occupied cells in its
    :data:`HOT_WINDOW`-square neighbourhood.  Local, not plane-wide, because a
    plane-wide median is dragged by the trigger's own two-lobe illumination.
    """
    n = len(CENTRES)
    r = HOT_WINDOW // 2
    out = np.zeros_like(H, dtype=bool)
    for i in range(n):
        lo_i, hi_i = max(0, i - r), min(n, i + r + 1)
        for j in range(n):
            if H[i, j] < HOT_MIN_COUNT:
                continue
            w = H[lo_i:hi_i, max(0, j - r):min(n, j + r + 1)]
            occ = w[w > 0]
            if len(occ) < 3:
                continue
            med = np.median(occ)
            if med > 0 and H[i, j] > HOT_FACTOR * med:
                out[i, j] = True
    return out


def profiles_from_map(H: dict, mask: np.ndarray, arm: str) -> pd.DataFrame:
    """1D projections on u and on v, per sample -- where the cliffs are.

    The v projection carries the plastic's 300 mm bar length against the
    chamber's 400 mm; the u projection carries both the gap shadow and the
    wall's four groups.  All three are the trigger's, and all three should
    flatten in the bystander sample.

    Taken from the MASKED map, so a hot channel cannot put a spike in a
    projection that is about to be read as detector structure.
    """
    rows = []
    for s, G in H.items():
        g = np.where(mask, 0.0, G)
        for axis, n in (('u', g.sum(axis=1)), ('v', g.sum(axis=0))):
            live = (~mask).sum(axis=1 if axis == 'u' else 0)
            rows.append(pd.DataFrame(dict(
                arm=arm, sample=s, axis=axis, coord=CENTRES,
                n=n.astype(np.int64), n_live_cells=live.astype(np.int64))))
    return pd.concat(rows, ignore_index=True)


def _bin_slice(lo: float, hi: float) -> slice:
    """Bins whose centres lie in [lo, hi) -- the windows are edge-aligned."""
    return slice(int(np.searchsorted(EDGES, lo, 'left')),
                 int(np.searchsorted(EDGES, hi, 'left')))


def shadow_from_map(H: dict, mask: np.ndarray, arm: str) -> pd.DataFrame:
    """The declared test: how deep is the plastic-gap dip, per sample.

    ``depth = 1 - (rate in the shadow window) / (rate in the flanks)``, both per
    LIVE cell rather than per unit u -- so neither the unequal widths of the two
    windows nor an unequal number of masked cells inside them can fake a result.

    The prediction, made before this ran: the dip is the TRIGGER's, so it goes
    away in the bystander sample -- **except** where a chamber has dead readout
    channels under it, and C and D do while A does not (STATUS.md).
    """
    v = _bin_slice(-SHADOW_V, SHADOW_V)
    sh = _bin_slice(*SHADOW_U)
    fl = [_bin_slice(a, b) for a, b in FLANK_U]
    live = ~mask
    rows = []
    for s, G in H.items():
        g = np.where(mask, 0.0, G)
        n_sh = float(g[sh, v].sum())
        c_sh = int(live[sh, v].sum())
        n_fl = float(sum(g[f, v].sum() for f in fl))
        c_fl = int(sum(live[f, v].sum() for f in fl))
        rows.append(dict(arm=arm, sample=s, n_shadow=n_sh, n_flank=n_fl,
                         cells_shadow=c_sh, cells_flank=c_fl))
    return _shadow_depth(pd.DataFrame(rows))


def _shadow_depth(g: pd.DataFrame) -> pd.DataFrame:
    """Counts per live cell -> depth, Poisson propagated through the ratio."""
    g = g.copy()
    g['rate_shadow'] = g.n_shadow / g.cells_shadow.clip(lower=1)
    g['rate_flank'] = g.n_flank / g.cells_flank.clip(lower=1)
    r = g.rate_shadow / g.rate_flank.replace(0, np.nan)
    g['depth'] = 1.0 - r
    g['depth_err'] = r * np.hypot(1 / np.sqrt(g.n_shadow.clip(lower=1)),
                                  1 / np.sqrt(g.n_flank.clip(lower=1)))
    return g


#: Strip-count bins: one strip each, to the 512 channels of a plane.
STRIP_EDGES = np.arange(0.0, 513.0, 1.0)
#: Charge bins: logarithmic, ~2 % per bin over the eight decades the fits span.
Q_EDGES = np.concatenate(([0.0], np.logspace(0.0, 8.0, 401), [np.inf]))

#: The cluster-shape variables, and the binning each needs.
SHAPE_VARS = (('width_x', 'x_n_strips', STRIP_EDGES),
              ('width_y', 'y_n_strips', STRIP_EDGES),
              ('q_x', 'x_q_sum', Q_EDGES),
              ('q_y', 'y_q_sum', Q_EDGES),
              ('q_per_strip_x', None, Q_EDGES),
              ('q_per_strip_y', None, Q_EDGES))


def shape(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Cluster width and charge density per arm -- chamber B's diagnosis.

    A fringing drift field spreads the same charge over more strips, so the
    signature of a missing field-shaping ring chain is a **wide, dilute**
    cluster -- not a weak one, since the amplification stage is untouched.
    That is a prediction of the hardware fault and it is testable against the
    three chambers that do have ring chains.

    **The sample is every gated cluster, NOT the bystander sample**, and the
    difference was measured rather than assumed.  On run_145 the bystander
    subset gives median widths of 38/46/49/42 strips against 25/43/34/42 on all
    gated clusters, and it reverses the ordering.  The reason is purity, not
    position: the bystander sample is unbiased in *where* it looks but nothing
    confirms any track in it, so it is diluted with junk that the
    scintillator-matched samples drop.  That is fine for an occupancy map --
    illumination is what it is measuring -- and fatal for a shape comparison.
    So this table stays on the basis `chamber_b.cluster_shape` established and
    STATUS.md quotes, and extends it from run_145 to the whole campaign.

    **The statistic is the MEDIAN**, and that is not stylistic either: the
    fitted charge carries a catastrophic tail -- ``x_q_sum`` reaches 1e29 on
    run_145 against a median of 1.5e3, from fits that blow up rather than from
    anything in the chamber -- so a mean measures the tail and nothing else.

    HISTOGRAMS are accumulated rather than sums, so the pooled median over 36
    runs is the median of the pooled sample and not an average of medians, and
    a 3 sub-run run does not weigh like a 29 sub-run one.  Strip counts bin at
    one strip, which is exact; charge bins logarithmically at ~2 % per bin,
    which resolves the median far finer than the spread it is quoted against.
    """
    rows = []
    nx = pd.to_numeric(t.x_n_strips, errors='coerce').to_numpy(float)
    ny = pd.to_numeric(t.y_n_strips, errors='coerce').to_numpy(float)
    qx = pd.to_numeric(t.x_q_sum, errors='coerce').to_numpy(float)
    qy = pd.to_numeric(t.y_q_sum, errors='coerce').to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = dict(width_x=nx, width_y=ny, q_x=qx, q_y=qy,
                    q_per_strip_x=qx / np.where(nx > 0, nx, np.nan),
                    q_per_strip_y=qy / np.where(ny > 0, ny, np.nan))
    for name, _col, edges in SHAPE_VARS:
        v = vals[name]
        v = v[np.isfinite(v)]
        n, _ = np.histogram(v, bins=edges)
        rows.append(pd.DataFrame(dict(arm=arm, var=name,
                                      lo=edges[:-1], hi=edges[1:],
                                      n=n.astype(np.int64))))
    return pd.concat(rows, ignore_index=True)


def _median(lo: np.ndarray, hi: np.ndarray, n: np.ndarray) -> float:
    """Median of a binned sample, interpolated inside the bin that contains it.

    Returns NaN if the median falls in an unbounded overflow bin rather than
    inventing an edge for it -- which, for a distribution whose tail reaches
    1e32, is a real possibility and not a theoretical one.
    """
    tot = n.sum()
    if tot == 0:
        return float('nan')
    c = np.cumsum(n)
    i = int(np.searchsorted(c, tot / 2.0))
    if not np.isfinite(hi[i]) or not np.isfinite(lo[i]):
        return float('nan')
    before = c[i - 1] if i else 0
    frac = (tot / 2.0 - before) / max(n[i], 1)
    return float(lo[i] + frac * (hi[i] - lo[i]))


def _pool_shape(sp: pd.DataFrame) -> pd.DataFrame:
    """Pooled medians, and the charge density that separates chamber B.

    Columns are named as `chamber_b.cluster_shape` names them, so the campaign
    number and the published run_145 number can be put side by side without a
    translation step.
    """
    g = sp.groupby(['arm', 'var', 'lo', 'hi'], as_index=False).n.sum()
    med = {}
    for (arm, var), d in g.groupby(['arm', 'var']):
        d = d.sort_values('lo')
        med[(arm, var)] = _median(d.lo.to_numpy(), d.hi.to_numpy(),
                                  d.n.to_numpy())
    arms = sorted({a for a, _ in med})
    out = pd.DataFrame(dict(arm=arms))
    out['n'] = [int(g[(g.arm == a) & (g['var'] == 'width_x')].n.sum())
                for a in arms]
    for name, _c, _e in SHAPE_VARS:
        out[name] = [med[(a, name)] for a in arms]
    ref = out[out.arm == 'A']
    if len(ref):
        for c in ('width_x', 'q_per_strip_x'):
            out[f'{c}_vs_A'] = out[c] / float(ref[c].iloc[0])
    return out


def census(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per-sample counts -- the partition, so every fraction has a denominator."""
    g = (t.groupby('sample')
         .agg(n_tracks=('event_id', 'size'),
              n_events=('event_id', 'nunique'),
              mean_ntrk=('n_trk', 'mean'),
              mean_q=('q_total', 'mean'),
              mean_nstrip_x=('x_n_strips', 'mean'))
         .reindex(SAMPLES).fillna(0).reset_index())
    g.insert(0, 'arm', arm)
    g['frac'] = g.n_tracks / max(g.n_tracks.sum(), 1)
    return g


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
def one_run(run: str, src: str, reco: str, slim_dir: str | None) -> tuple:
    """(run, maps, census, shape, trigger census, error).

    The per-run pass produces only what has to be built from the tracks.
    Profiles and the shadow test are projections of the map and are derived
    AFTER pooling, in :func:`main`, because the hot-cell mask they both depend
    on is only meaningful on campaign statistics -- a low-count run's mask would
    be Poisson noise, and a mask that differed run to run would make the pooled
    projection a sum of differently-censored maps.
    """
    try:
        src = Path(src)
        sd = Path(slim_dir) if slim_dir else None
        subs, dropped = subruns_of(Path(reco), run)
        if not subs:
            return (run,) + (None,) * 4 + ('no usable sub-runs',)
        have = [s for s in subs if (src / f'tracks_{run}_{s}.parquet').exists()]
        if not have:
            return (run,) + (None,) * 4 + ('no stage-3 tracks',)

        trig = trigger_sets(run, have, sd)
        geos = {a: (None if a == 'B' else DAS.layer_geometry(run, a))
                for a in ARMS}

        MP, CE, SP, keys = [], [], [], set()
        for arm in ARMS:
            t = arm_tracks(run, have, src, arm, trig, geos[arm], sd)
            if not len(t):
                continue
            keys |= set(zip(t.subrun, t.event_id))
            MP.append(histogram(t, arm))
            CE.append(census(t, arm))
            SP.append(shape(t, arm))
        if not CE:
            return (run,) + (None,) * 4 + ('no gated tracks in any arm',)

        tag = dict(run=run, condition=condition(run), k_block=in_block(run),
                   n_subruns=len(have), dropped_subruns=len(dropped))
        return (run,
                pd.concat(MP, ignore_index=True).assign(run=run),
                pd.concat(CE, ignore_index=True).assign(**tag),
                pd.concat(SP, ignore_index=True).assign(run=run),
                trigger_census(trig, keys).assign(**tag), '')
    except Exception:
        return (run,) + (None,) * 4 + (traceback.format_exc(limit=4),)


def derive(maps: pd.DataFrame) -> tuple:
    """Pooled maps -> (masked cells, profiles, shadow test, mask census).

    ONE place where the mask is built and applied, so nothing downstream can
    accidentally read an unmasked projection.  The mask has two reasons and the
    census reports them separately, because they are different statements about
    the detector: ``fiducial`` is where the fit rails and a position means
    nothing, ``hot`` is a channel that fires without a particle.
    """
    fid = fiducial_mask()
    MK, PR, SH, MC = [], [], [], []
    for arm in sorted(maps.arm.unique()):
        H = {s: as_grid(maps, arm, s) for s in MAPPED}
        total = sum(H.values())
        # The hot search runs INSIDE the fiducial only: the rail bins are
        # 10-30x their neighbours by construction, so including them would
        # spend the hot flag on an artefact that is already masked, and would
        # drag the neighbourhood median for the cells beside them.
        hot = hot_mask(np.where(fid, 0.0, total))
        mask = fid | hot
        PR.append(profiles_from_map(H, mask, arm))
        SH.append(shadow_from_map(H, mask, arm))
        iu, iv = np.nonzero(hot)
        MK.append(pd.DataFrame(dict(arm=arm, u=CENTRES[iu], v=CENTRES[iv],
                                    n=total[iu, iv].astype(np.int64),
                                    reason='hot')))
        row = dict(arm=arm, n_cells=int(mask.size), n_hot=int(hot.sum()),
                   n_fiducial=int(fid.sum()),
                   n_occupied=int((total > 0).sum()))
        for s, G in H.items():
            tot = G.sum()
            row[f'frac_hot_{s}'] = float(G[hot].sum() / tot) if tot else 0.0
            row[f'frac_rail_{s}'] = float(G[fid].sum() / tot) if tot else 0.0
        MC.append(row)
    return (pd.concat(MK, ignore_index=True) if MK else pd.DataFrame(),
            pd.concat(PR, ignore_index=True),
            pd.concat(SH, ignore_index=True),
            pd.DataFrame(MC))


def _runs_on_disk(reco: Path) -> list:
    return sorted((p.name for p in reco.iterdir()
                   if p.is_dir() and p.name.startswith('run_')), key=run_number)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--runs', nargs='*', default=None,
                    help='runs to process (default: every run on disk)')
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')),
                    help='stage-3 track tables')
    ap.add_argument('--reco', default=str(paths.spell('out', 'reco_fullpass')),
                    help='the full reconstruction pass (for the sub-run list)')
    ap.add_argument('--slim', default=None, help='exported n_TOF slim')
    ap.add_argument('--out', default=str(paths.spell('out', 'athens_insitu')))
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument(
        '--derive-only', action='store_true',
        help='rebuild the pooled tables from the per-run products a previous '
             'pass wrote, without re-reading a track or a slim file. The '
             'matching is the expensive step and its result is already on '
             'disk, so changing the mask, a window or a summary should not '
             'cost another read of the campaign.')
    a = ap.parse_args()

    runs = a.runs or _runs_on_disk(Path(a.reco))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed = {}

    if a.derive_only:
        print(f'in-situ maps: re-deriving from {out}\n')
        maps = pd.read_parquet(paths.require(
            out / 'maps_run.parquet', 'the per-run maps a previous pass wrote'))
        cen = pd.read_csv(paths.require(out / 'census_run.csv', 'the census'))
        sp = pd.read_csv(paths.require(out / 'shape_run.csv', 'the shapes'))
        tc = pd.read_csv(paths.require(out / 'trigger_census_run.csv',
                                       'the trigger census'))
    else:
        print(f'in-situ maps: {len(runs)} run(s), {a.jobs} job(s)\n')
        MP, CE, SP, TC = [], [], [], []
        args = [(r, a.src, a.reco, a.slim) for r in runs]
        if a.jobs > 1:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futs = {ex.submit(one_run, *g): g[0] for g in args}
                for i, f in enumerate(as_completed(futs), 1):
                    _collect(f.result(), MP, CE, SP, TC, failed, i, len(runs))
        else:
            for i, g in enumerate(args, 1):
                _collect(one_run(*g), MP, CE, SP, TC, failed, i, len(runs))

        if not CE:
            print('\nnothing produced -- every run failed', file=sys.stderr)
            for r, e in failed.items():
                print(f'  {r}: {e.splitlines()[-1]}', file=sys.stderr)
            return 1

        maps = pd.concat(MP, ignore_index=True)
        cen = pd.concat(CE, ignore_index=True)
        sp = pd.concat(SP, ignore_index=True)
        tc = pd.concat(TC, ignore_index=True)

    # Pool over runs.  Legitimate because the scintillator structure did not
    # move across the 27 July access -- `det_a_scint.assert_same_geometry`
    # checks exactly that, per run, and this pass reads the same config.
    pool_m = maps.groupby(['arm', 'sample', 'u', 'v'], as_index=False).n.sum()
    pool_c = (cen.groupby(['arm', 'sample'], as_index=False)
              .agg(n_tracks=('n_tracks', 'sum'), n_events=('n_events', 'sum')))
    pool_c['frac'] = (pool_c.n_tracks
                      / pool_c.groupby('arm').n_tracks.transform('sum'))
    pool_sp = _pool_shape(sp)
    pool_t = tc.groupby('n_arms', as_index=False).n.sum()
    pool_t['frac'] = pool_t.n / pool_t.n.sum()

    # The mask, and everything that depends on it, from the POOLED map.
    hot, pool_p, pool_s, mask_census = derive(pool_m)

    pool_m.to_parquet(out / 'maps.parquet', index=False)
    maps.to_parquet(out / 'maps_run.parquet', index=False)
    for name, d in (('profiles', pool_p), ('census', pool_c),
                    ('shadow', pool_s), ('shape', pool_sp),
                    ('masked_cells', hot), ('mask_census', mask_census),
                    ('trigger_census', pool_t), ('shape_run', sp),
                    ('census_run', cen), ('trigger_census_run', tc)):
        d.to_csv(out / f'{name}.csv', index=False)
        print(f'  -> {out / f"{name}.csv"}')

    meta = dict(schema=SCHEMA, runs=sorted(set(cen.run), key=run_number),
                failed={k: v.splitlines()[-1] for k, v in failed.items()},
                bin_mm=BIN_MM, dt_window=list(DT_WINDOW),
                shadow_u=list(SHADOW_U), flank_u=[list(f) for f in FLANK_U],
                shadow_v=SHADOW_V, fiducial=FIDUCIAL,
                hot_factor=HOT_FACTOR,
                hot_window=HOT_WINDOW, hot_min_count=HOT_MIN_COUNT,
                n_tracks=int(cen.n_tracks.sum()),
                seconds=round(time.time() - t0, 1))
    (out / 'insitu_maps.meta.json').write_text(json.dumps(meta, indent=1))

    print(f'\n{cen.n_tracks.sum():,} gated tracks, {len(set(cen.run))} runs, '
          f'{time.time() - t0:.0f}s')
    print('\nper arm, per sample (tracks):')
    print(pool_c.pivot(index='arm', columns='sample', values='n_tracks')
          .reindex(columns=list(SAMPLES)).fillna(0).astype(int).to_string())
    print('\nmasked cells, and what fraction of each sample they held:')
    print(mask_census.round(4).to_string(index=False))
    print('\ncluster shape, all gated clusters (chamber_b.py basis):')
    print(pool_sp.round(3).to_string(index=False))
    print('\nplastic-gap shadow depth (1 = fully blind):')
    print(pool_s.pivot(index='arm', columns='sample', values='depth')
          .reindex(columns=list(MAPPED)).round(3).to_string())
    if failed:
        print(f'\n{len(failed)} run(s) failed:', file=sys.stderr)
        for r, e in failed.items():
            print(f'  {r}: {e.splitlines()[-1]}', file=sys.stderr)
    return 0


def _collect(res, MP, CE, SP, TC, failed, i, n) -> None:
    run, mp, ce, sp, tc, err = res
    if err:
        failed[run] = err
        print(f'  [{i}/{n}] {run}: FAILED -- {err.splitlines()[-1]}')
        return
    MP.append(mp)
    CE.append(ce)
    SP.append(sp)
    TC.append(tc)
    by = ce.groupby('sample').n_tracks.sum()
    print(f'  [{i}/{n}] {run}: {int(ce.n_tracks.sum()):>8,} tracks  '
          f'trig {int(by.get("triggered", 0)):>7,}  '
          f'byst {int(by.get("bystander", 0)):>7,}')


if __name__ == '__main__':
    raise SystemExit(main())
