#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_acceptance.py -- A(theta), per run, and what the incidence does to it.

`acceptance.py` throws the toy once, for run_145, with run_145's efficiency map.
`campaign_angle.py` then applies that one curve to the pooled campaign spectrum
and stamps it ``acceptance_source = run_145 (BORROWED)``, calling it the leading
systematic on a set of fits that nothing describes.  This module removes the
borrowing and, more importantly, tests the OTHER half of that suspicion.

TWO SEPARABLE WORRIES, AND THEY HAVE DIFFERENT ANSWERS.

  1. **Is one run's acceptance a fair stand-in for the campaign's?**  Run the
     same toy 36 times, each with its own efficiency (`campaign_efficiency.py`),
     its own dead-channel ranges and its own measured source offset
     (`campaign_imaging.py`), and compare.  This is a run-to-run question.
  2. **Is the efficiency applied in the right VARIABLE?**  The toy applies it as
     a function of the in-plane position `u` only.  The measured effect that
     actually bites is the **head-on dip**: a track arriving perpendicular to the
     strip plane delivers its charge to every strip at once, carries no timing
     slope, and is reconstructed less often -- campaign-wide by a factor
     0.83 (A), 0.68 (C), 0.73 (D) against the same chamber's oblique tracks, and
     below both positional neighbours in 100 % of runs.  A point source
     CORRELATES incidence with opening angle, so this is a theta-dependent
     distortion and not a normalisation.  This is a variable question, and no
     amount of per-run measurement touches it.

THREE VARIANTS, RUN SIDE BY SIDE, because the difference between them IS the
systematic:

  ``flat``       the headline efficiency, uniform over the plane.  The baseline.
  ``u_map``      headline x the single-track efficiency map's shape in `u`.
                 This is what `acceptance.py` does today.
  ``incidence``  headline x the measured response versus incidence, interpolated
                 at each thrown leg's own true in-plane slope and normalised so
                 the tagged-weighted mean is 1 -- so the absolute scale is the
                 same measured number and only the shape moves.

ONE ARTEFACT IN ``u_map``, INHERITED AND DELIBERATELY NOT FIXED.  The measured
map spans |u| <= 160 mm because the scintillators stop covering the plane beyond
about 150 mm, while the active area runs to |u| = 190 mm.  `acceptance.Chambers`
interpolates the map with ``left=nan, right=nan`` and turns the NaN into a zero,
so in the ``u_map`` variant **a leg landing in the outer 30 mm of the plane is
given zero efficiency** -- an acceptance cut wearing an efficiency's clothes.
That is what every published number so far did, so it is reproduced here rather
than corrected, and it is part of why ``flat`` and ``incidence`` integrate
higher.  :func:`edge_cost` measures what it costs.

**`u_map` and `incidence` must never be multiplied together.**  They are two
views of the SAME four-point measurement: the wall groups that supply the
incidence abscissa are themselves four bands of `u`.  Applying both would count
the head-on dip twice.  Which one is right depends on whether the chamber
responds to where the track is or to how steeply it crosses, and the answer
(from the mechanism -- simultaneous arrival kills the timing slope) is
incidence; `u_map` is kept because it is what every published number so far
used and the comparison is the point.

WHAT IS STILL NOT IN ANY VARIANT, all of them making the acceptance an
over-estimate: multiple scattering, energy loss, the leptons' own energies,
pile-up, and the double-track finding efficiency at small separations.  So this
is a preliminary shape and never a rate -- unchanged from `acceptance.py`.

    python -m sept26_prelim_analysis.campaign_acceptance --jobs 6
    python -m sept26_prelim_analysis.campaign_acceptance --runs run_86,run_145
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
from sept26_prelim_analysis import acceptance as AC  # noqa: E402
from sept26_prelim_analysis.campaign_imaging import (  # noqa: E402
    PRE_ACCESS_RUNS, condition, in_block, run_number, subruns_of)

SCHEMA = 'sept26_prelim/campaign_acceptance/1'
ARMS = AC.ARMS
ANGLE_ARMS = AC.ANGLE_ARMS
TOPOLOGIES = ('intra', 'perpendicular', 'opposing')
#: The three ways the measured efficiency can enter the toy.  See the docstring:
#: `u_map` and `incidence` are alternatives, never a product.
VARIANTS = ('flat', 'u_map', 'incidence')
#: 5 deg, `acceptance.THETA_BINS` -- fine enough that `campaign_angle.fold`
#: can multiply on the physics grid and bin afterwards.
THETA_BINS = AC.THETA_BINS
#: Throws per run, in chunks of this size.  2 M gives ~50 accepted pairs in a
#: 5 deg opposing bin, which is a 14 % statistical error on the acceptance in
#: that bin -- below the run-to-run spread it is being used to measure, and the
#: chunking keeps the peak array allocation at n_chunk and not n_total.
N_THROW = 8_000_000
N_CHUNK = 1_000_000


# --------------------------------------------------------------------------- #
# the per-run inputs
# --------------------------------------------------------------------------- #
def source_offset(run: str, per_run: pd.DataFrame) -> tuple:
    """(X, 0, Z) mm for one run, from the campaign pointing crossings.

    X is the mean of A and C, which face each other along global X and so give
    the source and their own alignment together; Z is D alone, because B's
    crossing is a few hundred tracks with millimetres of scatter and
    `campaign_imaging` never averages it into a verdict.  A run missing either
    falls back to the campaign median rather than to zero -- zero is a real
    position 9 mm from the measured one, and would silently be the largest
    single change this module makes.
    """
    g = per_run[per_run.run == run]
    x = g[(g.axis == 'X') & (g.arm.isin(('A', 'C')))].mm
    z = g[(g.axis == 'Z') & (g.arm == 'D')].mm
    med = per_run[per_run.axis == 'X'] \
        .query('arm in ("A", "C")').mm.median()
    medz = per_run[(per_run.axis == 'Z') & (per_run.arm == 'D')].mm.median()
    src = ['measured' if len(x) else 'campaign median',
           'measured' if len(z) else 'campaign median']
    return ((float(x.mean()) if len(x) else float(med)), 0.0,
            (float(z.mean()) if len(z) else float(medz))), src


def incidence_response(I: pd.DataFrame) -> dict:
    """{arm: (tan nodes, relative response)} normalised to the tagged mean.

    The four wall groups give four (incidence, tracking rate) points per arm.
    Dividing by the tagged-weighted mean rate rather than by the best group's
    rate is what keeps the absolute scale equal to the headline efficiency: the
    headline is itself an average over the tagged incidence distribution, so a
    factor whose tagged-weighted mean is 1 rescales nothing and only tilts.

    The abscissa is |tan| because the response cannot depend on the sign of the
    slope -- the chamber does not know which way the track leans -- so the nodes
    are folded and sorted.
    """
    out = {}
    for arm, g in I.groupby('arm'):
        g = g[g.n_tagged > 0]
        if len(g) < 3:
            continue
        w = g.n_tagged.to_numpy(float)
        bar = float((w * g.p_tracked.to_numpy()).sum() / w.sum())
        if bar <= 0:
            continue
        t = np.abs(g.tan_expected.to_numpy())
        r = g.p_tracked.to_numpy() / bar
        o = np.argsort(t)
        t, r = t[o], r[o]
        # fold: two groups can land at the same |tan|; average their response
        tu, idx = np.unique(t, return_inverse=True)
        ru = np.array([r[idx == i].mean() for i in range(len(tu))])
        out[arm] = (tu, ru)
    return out


def vertices_wall(rng, n: int, offset) -> np.ndarray:
    """Vertices on the capsule WALL -- the skin of the He-3 polycone.

    A gas pair is born in the volume `acceptance._vertices` samples; a capsule
    pair is born in the aluminium and carbon-fibre shell around it, so its
    vertex distribution is a SURFACE.  Sampled proportional to the local lateral
    area, `2 pi r(y) ds` with `ds = sqrt(dy^2 + dr^2)`, which is what a roughly
    uniform thermal flux on a thin shell makes.

    The shell's own thickness (0.5 mm of aluminium plus the fibre) is ignored:
    it is 5 % of the 10 mm radius and 0.2 % of the 235 mm lever to the nearest
    strip plane, so it cannot move a 40 mm efficiency bin.  End caps are ignored
    for the same reason the lateral surface dominates -- the polycone is 80 mm
    long and 20 mm across.
    """
    from ntof_tracking.reco import geometry as G
    ys, rs = G.HE3_GAS_Y, G.HE3_GAS_R
    # a fine y grid, weighted by the local slant area
    yy = np.linspace(ys.min(), ys.max(), 4000)
    rr = np.interp(yy, ys, rs)
    dr = np.gradient(rr, yy)
    w = rr * np.sqrt(1.0 + dr ** 2)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return AC._vertices(rng, n, offset)
    cdf = np.cumsum(w) / w.sum()
    y = np.interp(rng.uniform(0, 1, n), cdf, yy)
    r = np.interp(y, ys, rs)
    ph = rng.uniform(0, 2 * np.pi, n)
    return (np.column_stack([r * np.cos(ph), y, r * np.sin(ph)])
            + np.asarray(offset, float))


class RunChambers(AC.Chambers):
    """`acceptance.Chambers` with the run's own map, plus an incidence factor.

    The base class interpolates the map's shape in `u` and multiplies the
    headline; this adds the third variant and switches between them, so the
    three curves come out of ONE geometry and one vertex model and differ only
    in how the same measured efficiency is applied.
    """

    def __init__(self, run, subruns, merged_dir, eff_map, eff_headline,
                 inc_resp, variant: str = 'u_map'):
        super().__init__(run, subruns, merged_dir, eff_map=eff_map,
                         eff_headline=eff_headline)
        self.inc = inc_resp
        self.variant = variant
        # The single-track column is this module's map basis; the base class
        # reads `eff_track_given_seed`, so rename rather than reimplement.
        if eff_map is not None and 'eff_single_given_seed' in eff_map.columns:
            m = eff_map.copy()
            good = np.isfinite(m.eff_single_given_seed)
            m.loc[good, 'eff_track_given_seed'] = \
                m.loc[good, 'eff_single_given_seed']
            self.eff_map = m

    def leg_efficiency(self, arm, u, tan_u):
        base = float(self.eff.get(arm, 0.0))
        if self.variant == 'flat':
            return np.full(len(u), base)
        if self.variant == 'u_map':
            return super().efficiency(arm, u)
        node, resp = self.inc.get(arm, (None, None))
        if node is None:
            return np.full(len(u), base)
        # flat extrapolation outside the measured span: the four groups cover
        # |tan| 0.08-0.53 and a leg outside that is not a measurement, so it
        # gets the nearest measured response and not an invented slope
        r = np.interp(np.abs(tan_u), node, resp, left=resp[0], right=resp[-1])
        return np.clip(base * r, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# the throw
# --------------------------------------------------------------------------- #
def throw_chunk(ch: RunChambers, n: int, offset, seed: int,
                vertex: str = 'gas') -> tuple:
    """(accepted frame, thrown theta) for one chunk, one variant.

    A copy of `acceptance.throw` with three changes: the efficiency call gets
    the leg's true in-plane slope as well as its position, the vertex can come
    from the capsule wall instead of the gas volume, and the thrown angles come
    back as an array rather than a frame so the caller can histogram chunks
    without concatenating them.
    """
    from sept26_prelim_analysis.normal_incidence import tan_in_plane
    rng = np.random.default_rng(seed)
    P = (vertices_wall(rng, n, offset) if vertex == 'wall'
         else AC._vertices(rng, n, offset))
    d1 = AC._isotropic(rng, n)
    th = rng.uniform(0.0, np.pi, n)          # FLAT in theta: this is an
    d2 = AC._rotate_by(rng, d1, th)          # acceptance, not a prediction

    hit = {}
    for leg, D in ((1, d1), (2, d2)):
        for a in ARMS:
            u, v, plane, plastic = ch.cross(P, D, a)
            tu, _ = tan_in_plane(D, a)
            keep = plane & (rng.uniform(0, 1, n)
                            < ch.leg_efficiency(a, u, tu))
            hit[(leg, a)] = dict(plane=plane, plastic=plastic, reco=keep)

    # the trigger: at least one leg makes a wall+plastic coincidence in an arm
    # it also crossed
    trig = np.zeros(n, bool)
    for leg in (1, 2):
        for a in ARMS:
            trig |= hit[(leg, a)]['plane'] & hit[(leg, a)]['plastic']

    rows = []
    for a1 in ANGLE_ARMS:
        for a2 in ANGLE_ARMS:
            if a2 < a1:
                continue
            both = hit[(1, a1)]['reco'] & hit[(2, a2)]['reco'] & trig
            if a1 != a2:
                both |= hit[(1, a2)]['reco'] & hit[(2, a1)]['reco'] & trig
            if not both.any():
                continue
            rows.append(pd.DataFrame(dict(
                theta=np.degrees(th[both]), arm1=a1, arm2=a2,
                topology=AC.topology(a1, a2))))
    acc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        dict(theta=[], arm1=[], arm2=[], topology=[]))
    return acc, np.degrees(th)


def curves(ch: RunChambers, n: int, offset, seed: int,
           bins=THETA_BINS, vertex: str = 'gas') -> pd.DataFrame:
    """A(theta) per topology and arm pair, accumulated over chunks."""
    mid = 0.5 * (bins[:-1] + bins[1:])
    n0 = np.zeros(len(mid))
    counts = {}
    done = 0
    k = 0
    while done < n:
        m = min(N_CHUNK, n - done)
        acc, th = throw_chunk(ch, m, offset, seed + k, vertex)
        h, _ = np.histogram(th, bins=bins)
        n0 += h
        if len(acc):
            groups = [('all', acc)]
            groups += [(t, g) for t, g in acc.groupby('topology')]
            groups += [(f'{a1}-{a2}', g)
                       for (a1, a2), g in acc.groupby(['arm1', 'arm2'])]
            for name, g in groups:
                hh, _ = np.histogram(g.theta, bins=bins)
                counts[name] = counts.get(name, np.zeros(len(mid))) + hh
        done += m
        k += 1
    rows = []
    for name, kk in counts.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.where(n0 > 0, kk / n0, np.nan)
            e = np.where(n0 > 0, np.sqrt(np.clip(kk, 1, None)) / n0, np.nan)
        rows.append(pd.DataFrame(dict(group=name, theta=mid, n_acc=kk,
                                      n_thrown=n0, acc=a, err=e)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# one run, all three variants
# --------------------------------------------------------------------------- #
def one_run(run: str, reco: str, eff_dir: str, per_run_csv: str,
            n: int, seed: int, vertices=('gas',)) -> tuple:
    """(run, curves frame, meta dict, error text).  Runs in a worker."""
    try:
        reco = Path(reco)
        subs, dropped = subruns_of(reco, run)
        if not subs:
            return run, None, None, 'no usable sub-runs under the reco tree'
        ed = Path(eff_dir)
        H = pd.read_csv(paths.require(
            ed / f'efficiency_headline_{run}.csv',
            f'per-run efficiency headline for {run} -- run '
            f'campaign_efficiency.py first'))
        M = pd.read_csv(paths.require(ed / f'efficiency_map_{run}.csv',
                                      f'per-run efficiency map for {run}'))
        ip = ed / f'efficiency_incidence_{run}.csv'
        I = pd.read_csv(ip) if ip.exists() else pd.DataFrame()
        # B's headline is a HIT efficiency and is not a track efficiency, so it
        # enters as zero -- B is excluded from every measurable topology anyway
        eff = {r.arm: (r.efficiency if r.basis == 'tracks' else 0.0)
               for r in H.itertuples()}
        inc = incidence_response(I) if not I.empty else {}
        PR = pd.read_csv(per_run_csv)
        offset, src = source_offset(run, PR)

        out = []
        for vtx in vertices:
            for v in VARIANTS:
                ch = RunChambers(run, subs, str(reco / run), M, eff, inc,
                                 variant=v)
                c = curves(ch, n, offset, seed, vertex=vtx)
                if c.empty:
                    continue
                c.insert(0, 'run', run)
                c.insert(1, 'variant', v)
                c.insert(2, 'vertex', vtx)
                out.append(c)
        meta = dict(run=run, n_subruns=len(subs), n_thrown=n,
                    vertices=list(vertices),
                    offset_mm=list(offset), offset_source=src,
                    efficiency={k: float(v) for k, v in eff.items()},
                    dead_u_ranges={k: [[round(x, 1) for x in r] for r in val]
                                   for k, val in ch.dead.items()},
                    incidence_arms=sorted(inc),
                    dropped_subruns=dropped,
                    condition=condition(run), k_block=in_block(run))
        return run, (pd.concat(out, ignore_index=True) if out else None), \
            meta, ''
    except Exception:
        return run, None, None, traceback.format_exc(limit=3)


def edge_pass(reco: Path, runs, eff_dir: Path, per_run_csv: str,
              n: int = 2_000_000) -> pd.DataFrame:
    """`edge_cost` for every run -- cheap, and independent of the throws.

    Its own pass so that measuring the `u_map` edge artefact never costs a
    re-throw of the acceptance itself.
    """
    PR = pd.read_csv(per_run_csv)
    rows = []
    for run in runs:
        subs, _ = subruns_of(reco, run)
        if not subs:
            continue
        h = eff_dir / f'efficiency_headline_{run}.csv'
        m = eff_dir / f'efficiency_map_{run}.csv'
        if not (h.exists() and m.exists()):
            continue
        H, M = pd.read_csv(h), pd.read_csv(m)
        eff = {r.arm: (r.efficiency if r.basis == 'tracks' else 0.0)
               for r in H.itertuples()}
        ch = RunChambers(run, subs, str(reco / run), M, eff, {})
        offset, _ = source_offset(run, PR)
        e = edge_cost(ch, n, offset)
        e.insert(0, 'run', run)
        rows.append(e)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def campaign(reco: Path, runs, eff_dir: Path, per_run_csv: str, n: int,
             jobs: int, seed: int = 17, vertices=('gas',)) -> tuple:
    C, meta, failed = [], {}, {}
    args = (str(reco), str(eff_dir), per_run_csv, n, seed, tuple(vertices))
    if jobs <= 1:
        results = [one_run(r, *args) for r in runs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(one_run, r, *args): r for r in runs}
            for f in as_completed(fut):
                results.append(f.result())
    for run, c, m, err in sorted(results, key=lambda r: run_number(r[0])):
        if err:
            failed[run] = err.strip().splitlines()[-1]
            print(f'  {run}: FAILED -- {failed[run]}', flush=True)
            continue
        C.append(c)
        meta[run] = m
        opp = c[(c.group == 'opposing') & (c.variant == 'u_map')
                & (c.vertex == vertices[0])]
        print(f'  {run}: offset ({m["offset_mm"][0]:+.1f}, 0, '
              f'{m["offset_mm"][2]:+.1f}) mm, opposing A = '
              f'{1e4 * opp.acc.mean():.2f}e-4', flush=True)
    return (pd.concat(C, ignore_index=True) if C else pd.DataFrame(),
            meta, failed)


# --------------------------------------------------------------------------- #
# what the curves say
# --------------------------------------------------------------------------- #
def edge_cost(ch: RunChambers, n: int, offset, seed: int = 991) -> pd.DataFrame:
    """How many accepted legs land where the efficiency map does not reach.

    The `u_map` variant zeroes them (see the docstring), so this is the size of
    that artefact: the fraction of legs inside the active area but outside the
    map's |u| <= 160 mm span, per arm and per topology.
    """
    from ntof_tracking.reco import geometry as G
    rng = np.random.default_rng(seed)
    P = AC._vertices(rng, n, offset)
    d1 = AC._isotropic(rng, n)
    th = rng.uniform(0.0, np.pi, n)
    d2 = AC._rotate_by(rng, d1, th)
    edge = float(np.max(np.abs(ch.eff_map.u_mid.to_numpy()))
                 + 0.5 * 40.0) if ch.eff_map is not None else 160.0
    rows = []
    for a in ANGLE_ARMS:
        n_in = n_out = 0
        for D in (d1, d2):
            u, v, plane, plastic = ch.cross(P, D, a)
            n_in += int((plane & (np.abs(u) <= edge)).sum())
            n_out += int((plane & (np.abs(u) > edge)).sum())
        rows.append(dict(arm=a, u_edge_mm=edge, n_inside=n_in,
                         n_outside=n_out,
                         frac_outside=n_out / max(n_in + n_out, 1)))
    return pd.DataFrame(rows)


def pooled(C: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """The campaign acceptance: each run's curve, weighted by its real pairs.

    Weighting by pair count and not equally is the whole point -- the campaign
    spectrum is the sum of the runs' pairs, so its acceptance is the
    pair-weighted mean of the runs' acceptances.  run_116 alone carries 11 % of
    the pairs and an equal-weighted mean would understate it by 30x.
    """
    d = C[C.run.isin(weights.index)].copy()
    d['w'] = d.run.map(weights).astype(float)
    rows = []
    for (variant, group, theta), g in d.groupby(['variant', 'group', 'theta']):
        w = g.w.to_numpy()
        a = g.acc.to_numpy()
        ok = np.isfinite(a) & (w > 0)
        if not ok.any():
            continue
        rows.append(dict(variant=variant, group=group, theta=theta,
                         acc=float((w[ok] * a[ok]).sum() / w[ok].sum()),
                         n_runs=int(ok.sum()),
                         sd=float(a[ok].std(ddof=1)) if ok.sum() > 1 else 0.0))
    return pd.DataFrame(rows).sort_values(['variant', 'group', 'theta'],
                                          ignore_index=True)


def run_spread(C: pd.DataFrame, variant: str = 'u_map') -> pd.DataFrame:
    """How much the acceptance SHAPE moves run to run, per topology.

    Each run's curve is normalised to its own integral first, because a change
    in the overall efficiency is a normalisation the spectrum comparison does
    not care about, and a change in shape is one it cares about entirely.
    """
    d = C[(C.variant == variant) & ~C.run.isin(PRE_ACCESS_RUNS)].copy()
    d = d[np.isfinite(d.acc)]
    d['shape'] = d.groupby(['run', 'group']).acc.transform(
        lambda s: s / s.sum() if s.sum() > 0 else s)
    rows = []
    for (group, theta), g in d.groupby(['group', 'theta']):
        s = g['shape'].to_numpy()
        if len(s) < 3 or s.mean() <= 0:
            continue
        rows.append(dict(group=group, theta=theta, n_runs=len(s),
                         shape_mean=float(s.mean()),
                         shape_cv=float(s.std(ddof=1) / s.mean())))
    cols = ['group', 'theta', 'n_runs', 'shape_mean', 'shape_cv']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(['group', 'theta'],
                                          ignore_index=True)


def variant_shift(P: pd.DataFrame) -> pd.DataFrame:
    """Each variant against `u_map`, as the shape change it makes.

    Two numbers per topology: the integrated acceptance ratio (a
    normalisation, which the spectrum fits absorb) and the median opening angle
    of the acceptance curve (a shape, which they cannot).  If the second moves
    and the first does not, the variant matters.
    """
    rows = []
    for group, g in P.groupby('group'):
        ref = g[g.variant == 'u_map'].sort_values('theta')
        if ref.empty:
            continue
        r_int = ref.acc.sum()
        for v, h in g.groupby('variant'):
            h = h.sort_values('theta')
            a = h.acc.to_numpy()
            if a.sum() <= 0:
                continue
            cdf = np.cumsum(a) / a.sum()
            rows.append(dict(
                group=group, variant=v,
                integral_ratio=float(a.sum() / r_int) if r_int else np.nan,
                median_deg=float(np.interp(0.5, cdf, h.theta.to_numpy())),
                frac_above_109=float(a[h.theta.to_numpy() > 109.0].sum()
                                     / a.sum())))
    d = pd.DataFrame(rows)
    ref = d[d.variant == 'u_map'].set_index('group')
    d['median_shift_deg'] = [r.median_deg - ref.median_deg.get(r.group, np.nan)
                             for r in d.itertuples()]
    d['frac109_ratio'] = [
        r.frac_above_109 / ref.frac_above_109.get(r.group, np.nan)
        if ref.frac_above_109.get(r.group, 0) else np.nan
        for r in d.itertuples()]
    return d.sort_values(['group', 'variant'], ignore_index=True)


def borrow_test(C: pd.DataFrame, weights: pd.Series,
                ref: str = 'run_145', variant: str = 'u_map') -> pd.DataFrame:
    """run_145's own curve against the pair-weighted campaign one.

    The direct test of what every published spectrum so far assumed.  Quoted as
    the shape ratio in each bin, because a pure normalisation difference cancels
    in a fit with one free normalisation and a shape difference does not.
    """
    P = pooled(C, weights)
    rows = []
    for group in TOPOLOGIES + ('all',):
        r = C[(C.run == ref) & (C.variant == variant) & (C.group == group)] \
            .sort_values('theta')
        c = P[(P.variant == variant) & (P.group == group)].sort_values('theta')
        if r.empty or c.empty:
            continue
        m = np.isfinite(r.acc.to_numpy()) & np.isfinite(c.acc.to_numpy())
        ra, ca = r.acc.to_numpy()[m], c.acc.to_numpy()[m]
        if ra.sum() <= 0 or ca.sum() <= 0:
            continue
        sr, sc = ra / ra.sum(), ca / ca.sum()
        rows.append(dict(
            group=group, n_bins=int(m.sum()),
            integral_ratio=float(ra.sum() / ca.sum()),
            max_shape_dev=float(np.nanmax(np.abs(
                sr / np.where(sc > 0, sc, np.nan) - 1))),
            rms_shape_dev=float(np.sqrt(np.nanmean((sr / np.where(
                sc > 0, sc, np.nan) - 1) ** 2))),
            frac109_ref=float(sr[r.theta.to_numpy()[m] > 109].sum()),
            frac109_campaign=float(sc[c.theta.to_numpy()[m] > 109].sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--reco', default=str(paths.out('reco_fullpass')))
    ap.add_argument('--eff', default=str(paths.out('efficiency_campaign',
                                                   'per_run')))
    ap.add_argument('--imaging', default=str(paths.out('imaging_campaign')
                                             / 'per_run.csv'))
    ap.add_argument('--pairs', default=str(paths.out('angle_campaign')
                                           / 'pairs.parquet'))
    ap.add_argument('--runs', default='')
    ap.add_argument('--n', type=int, default=N_THROW)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--summarise-only', action='store_true',
                    help='re-derive the summary tables from the throw already '
                         'in <out>/acceptance_campaign, without re-throwing')
    ap.add_argument('--vertex', default='gas,wall',
                    help='vertex models to throw: gas (the He-3 volume, where '
                         'a gas pair is born) and/or wall (the capsule shell, '
                         'where an aluminium pair is born)')
    a = ap.parse_args()

    reco = Path(paths.require(a.reco, 'the full-pass reco tree'))
    eff = Path(paths.require(a.eff, 'the per-run efficiency products -- run '
                                    'campaign_efficiency.py first'))
    pr = str(paths.require(a.imaging, 'the campaign pointing crossings -- run '
                                      'campaign_imaging.py first'))
    runs = ([r for r in a.runs.split(',') if r] or
            sorted((p.name for p in reco.iterdir()
                    if p.is_dir() and p.name.startswith('run_')),
                   key=run_number))
    vtx = tuple(v for v in a.vertex.split(',') if v)
    od = paths.out('acceptance_campaign')
    if a.summarise_only:
        C = pd.read_csv(paths.require(od / 'acceptance_per_run.csv',
                                      'a previous throw to summarise'))
        meta = json.loads((od / 'campaign_acceptance.meta.json').read_text()
                          ).get('per_run', {})
        failed = {}
        print(f'summarising the existing throw: {C.run.nunique()} runs\n')
    else:
        print(f'{len(runs)} run(s), {a.n:,} pairs each, {len(VARIANTS)} '
              f'efficiency variants x {len(vtx)} vertex model(s) {vtx}\n')
        C, meta, failed = campaign(reco, runs, eff, pr, a.n, a.jobs,
                                   vertices=vtx)
    if C.empty:
        print('\nnothing thrown')
        return 1

    # pair weights: the campaign spectrum is the sum of the runs' pairs, so the
    # campaign acceptance is their pair-weighted mean
    W = pd.read_parquet(paths.require(a.pairs, 'the campaign pair table'))
    w = W[~W.mixed].groupby('run').size()

    C.to_csv(od / 'acceptance_per_run.csv', index=False)
    EC = edge_pass(reco, sorted(C.run.unique(), key=run_number), eff, pr)
    EC.to_csv(od / 'edge_cost.csv', index=False)
    # The gas vertex is the default product name, because it is what every
    # existing consumer means by "the acceptance"; the wall throw sits beside
    # it under an explicit suffix so nothing picks it up by accident.
    Cg = C[C.vertex == vtx[0]]
    P = pooled(Cg, w)
    S = run_spread(Cg)
    V = variant_shift(P)
    B = borrow_test(Cg, w)
    P.to_csv(od / 'acceptance_pooled.csv', index=False)
    S.to_csv(od / 'run_spread.csv', index=False)
    V.to_csv(od / 'variant_shift.csv', index=False)
    B.to_csv(od / 'borrow_test.csv', index=False)
    for other in vtx[1:]:
        Co = C[C.vertex == other]
        if Co.empty:
            continue
        pooled(Co, w).to_csv(od / f'acceptance_pooled_{other}.csv',
                             index=False)
        variant_shift(pooled(Co, w)).to_csv(
            od / f'variant_shift_{other}.csv', index=False)
    (od / 'per_run').mkdir(exist_ok=True)
    for (run, variant, vv), g in C.groupby(['run', 'variant', 'vertex']):
        g.drop(columns=['run', 'variant', 'vertex']).to_csv(
            od / 'per_run' / f'acceptance_{run}_{variant}_{vv}.csv',
            index=False)
    json.dump(dict(schema=SCHEMA, reco=str(reco), eff_dir=str(eff),
                   imaging=pr, pairs=a.pairs, n_thrown=a.n,
                   variants=list(VARIANTS), vertices=list(vtx),
                   theta_bins=[float(x) for x in THETA_BINS],
                   n_runs=int(C.run.nunique()),
                   pair_weights={k: int(v) for k, v in w.items()},
                   runs_failed=failed, per_run=meta,
                   caveat='no multiple scattering, no energy loss, no lepton '
                          'energies, no pile-up, no double-track finding '
                          'efficiency -- a shape, never a rate'),
              open(od / 'campaign_acceptance.meta.json', 'w'), indent=1)

    print('\nSHAPE spread over the runs, per topology '
          '(cv of the normalised curve)\n')
    for group, g in S[S.group.isin(TOPOLOGIES)].groupby('group') \
            if not S.empty else []:
        live = g[g.shape_mean > 1e-4]
        print(f'  {group:>14}  {len(live):>3} live bins, median cv '
              f'{live.shape_cv.median():.3f}, max {live.shape_cv.max():.3f}')

    print('\nrun_145 BORROWED against the pair-weighted campaign acceptance\n')
    print(f'{"topology":>14} {"integral":>9} {"rms shape":>10} '
          f'{"frac>109 r145":>14} {"campaign":>9}')
    for r in B.itertuples():
        print(f'{r.group:>14} {r.integral_ratio:>9.3f} '
              f'{r.rms_shape_dev:>10.3f} {r.frac109_ref:>14.4f} '
              f'{r.frac109_campaign:>9.4f}')

    print('\nthe INCIDENCE variant against the u-map one -- the theta-'
          'dependent distortion\n')
    print(f'{"topology":>14} {"variant":>10} {"integral":>9} {"median":>8} '
          f'{"d median":>9} {"frac>109":>9}')
    for r in V[V.group.isin(TOPOLOGIES)].itertuples():
        print(f'{r.group:>14} {r.variant:>10} {r.integral_ratio:>9.3f} '
              f'{r.median_deg:>7.1f}d {r.median_shift_deg:>+8.1f}d '
              f'{r.frac_above_109:>9.4f}')
    if not EC.empty:
        print('\nthe u_map edge artefact -- accepted legs the map does not '
              'reach, and so gets zero efficiency\n')
        for arm, g in EC.groupby('arm'):
            print(f'  {arm}: {100 * g.frac_outside.mean():.1f} % of legs '
                  f'beyond |u| = {g.u_edge_mm.iloc[0]:.0f} mm')
    if failed:
        print(f'\n{len(failed)} run(s) failed: {", ".join(failed)}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
