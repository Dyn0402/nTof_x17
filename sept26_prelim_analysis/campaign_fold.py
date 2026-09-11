#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_fold.py -- the aluminium capsule, folded, against the measured angles.

`campaign_angle.py` compares the campaign opening-angle spectra against the
3He gas prediction (`ipc_channels.thermal_spectrum`) and X17, and nothing fits:
chi2/dof 13-85 with one free normalisation, the event-mixed accidental template
still the best description.  **But the gas is not where most of the pairs come
from.**  `ipc_aluminium.py` puts the capsule wall's own internal-pair continuum
at 10^4-10^6 times the gas's wide-angle yield, in a spectrum whose shape differs
from the gas's by about a factor of two, and PLAN sec S4 says in as many words
that nothing in the angle chain splits gas from capsule.  This module folds the
capsule in.

WHAT IS FOLDED, and every one of them through the SAME acceptance:

  ``Al capsule (birth)``       `ipc_aluminium.capsule_spectrum`, 27Al and 12C
                               weighted by their captures.
  ``Al capsule (after wall)``  the same, after escape and multiple scattering in
                               the wall the pair was born inside.  **This is the
                               physical one** -- an aluminium pair has to get out
                               of the aluminium -- and it is the one to read.
  ``3He gas M1+E0``            what `campaign_angle` already folds, kept for the
                               comparison.
  ``3He gas (after wall)``     the gas pair crosses the whole wall rather than a
                               random part of it, so it smears less.
  ``X17``                      `pair_physics.x17_angles`.
  ``event-mixed``              the measured accidental shape, not folded -- it is
                               already an acceptance-times-physics product,
                               which is exactly why it fits best and why folding
                               it would be double-counting.

TWO VERTEX DISTRIBUTIONS, AND THE ALUMINIUM NEEDS THE SECOND ONE.  The
acceptance toy draws vertices by volume from the He-3 gas polycone, because that
is where a gas pair is born.  A capsule pair is born in the WALL -- the skin of
the same polycone -- so its vertex distribution is a surface and not a volume.
`campaign_acceptance.py --vertex wall` throws it; this module reads whichever is
available and uses the wall acceptance for the capsule components and the gas
acceptance for the gas ones.  The difference is small (a 10 mm radius against a
235 mm lever) and it is applied rather than argued about.

THE FIT, AND WHY IT IS TWO COMPONENTS.  A single-shape chi2 asks the wrong
question: nobody claims the sample is pure pairs.  The measured accidental
fraction is **f = 29 % [17, 40] overall and 42 % [26, 58] opposing**
(`HANDOFF_ACCIDENTAL_TIMING.md`), so the honest comparison floats the accidental
share and asks whether the pair half looks like the capsule.  :func:`two_comp`
profiles chi2 over that share on a grid and reports the best fit, its 1-sigma
interval, and whether the interval contains the independently measured f -- which
is the actual test, because the timing measurement did not use the angles at all.

WHAT A GOOD FIT HERE WOULD AND WOULD NOT MEAN.  It would mean the wide-angle
background is understood as capsule internal pairs plus accidentals, which is
what an X17 search needs before it can quote anything.  It would NOT identify
the pairs as aluminium: the one handle that separates a 2-4 MeV capsule pair
from a 20.6 MeV gas pair is the total pair energy, and this setup does not
measure it (`IPC_MISSING.md`).  Opening angle alone cannot do it, and this
module's own numbers say why -- after the wall the two shapes' medians are 51
and 37 degrees, distinguishable in principle and not in 1 000 pairs.

    python -m sept26_prelim_analysis.campaign_fold
    python -m sept26_prelim_analysis.campaign_fold --variant incidence
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis.campaign_angle import (  # noqa: E402
    BINS, TOPOLOGIES, X17_MIN_DEG, fold)

SCHEMA = 'sept26_prelim/campaign_fold/1'
#: Which acceptance each physics component is entitled to.  The capsule pair is
#: born in the wall and the gas pair in the gas; X17 is a gas transition.
VERTEX_OF = {
    'Al capsule (birth)': 'wall',
    'Al capsule (after wall)': 'wall',
    '3He gas M1+E0': 'gas',
    '3He gas M1+E0 (after wall)': 'gas',
    '3He gas M1 only': 'gas',
    '3He gas E0 only': 'gas',
    'X17 (17 MeV boson)': 'gas',
}
#: The accidental share grid the two-component fit profiles over.
F_GRID = np.linspace(0.0, 1.0, 201)
#: The timing measurement this fit is tested against
#: (`HANDOFF_ACCIDENTAL_TIMING.md`), as (best, lo, hi).
F_TIMING = {'all': (0.29, 0.17, 0.40), 'opposing': (0.42, 0.26, 0.58),
            'perpendicular': (0.16, 0.00, 0.32)}


# --------------------------------------------------------------------------- #
# the physics shapes, on the fine grid
# --------------------------------------------------------------------------- #
def shapes(cache: Path | None = None, assume: str = 'M1') -> tuple:
    """(theta_mid, {name: dN/dtheta}) on the 1 deg grid, plus a provenance dict.

    `ipc_aluminium.shape_comparison` samples multiple scattering line by line
    and takes ~50 s, so it is cached: the inputs are nuclear data files and a
    fixed seed, so the result is a constant of the analysis and re-deriving it
    on every report build buys nothing.
    """
    from sept26_prelim_analysis import ipc_aluminium as AL
    from sept26_prelim_analysis import ipc_born as IB
    from sept26_prelim_analysis import ipc_channels as IC
    from sept26_prelim_analysis import pair_physics as PP

    if cache is not None and cache.exists():
        R = pd.read_csv(cache)
        prov = json.loads((cache.with_suffix('.meta.json')).read_text())
    else:
        R = AL.shape_comparison(assume)
        prov = dict(assume=assume, **{k: float(v) if isinstance(v, float)
                                      else v for k, v in R.attrs.items()})
        if cache is not None:
            R.to_csv(cache, index=False)
            cache.with_suffix('.meta.json').write_text(json.dumps(prov,
                                                                  indent=1))
    fine = R.theta_mid.to_numpy()
    P = IC.thermal_spectrum()
    out = {
        'Al capsule (birth)': R.capsule_birth.to_numpy(),
        'Al capsule (after wall)': R.capsule_after_wall.to_numpy(),
        '3He gas M1+E0': P.total.to_numpy(),
        '3He gas M1+E0 (after wall)': R.he3_after_wall.to_numpy(),
        '3He gas M1 only': P.M1.to_numpy(),
        '3He gas E0 only': P.E0.to_numpy(),
    }
    h17, _ = np.histogram(PP.x17_angles(400_000),
                          bins=np.arange(0.0, 180.01, 1.0), density=True)
    out['X17 (17 MeV boson)'] = h17
    # every shape on the same grid, or the fold silently interpolates a
    # different physics than the one named
    for k, v in out.items():
        if len(v) != len(fine):
            raise ValueError(f'shape {k!r} is {len(v)} bins, grid is '
                             f'{len(fine)}')
    prov['medians_deg'] = {
        k: float(np.interp(0.5, np.cumsum(v * np.diff(IB.THETA_BINS)),
                           fine)) for k, v in out.items()}
    prov['frac_above_109'] = {k: float(IB.frac_above(v, X17_MIN_DEG))
                              for k, v in out.items()}
    return fine, out, prov


# --------------------------------------------------------------------------- #
# folding
# --------------------------------------------------------------------------- #
def read_acceptance(acc_dir: Path, variant: str) -> dict:
    """{vertex: acceptance frame} for one efficiency variant.

    The pooled curve is the pair-weighted campaign one, so it is the acceptance
    the pooled spectrum is entitled to.  A missing wall throw is not fatal --
    the capsule then folds through the gas acceptance and the meta says so,
    because a report that silently substitutes one vertex model for another is
    worse than one that says it did.
    """
    out = {}
    for vtx in ('gas', 'wall'):
        p = acc_dir / (f'acceptance_pooled.csv' if vtx == 'gas'
                       else f'acceptance_pooled_wall.csv')
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.variant == variant]
        if not d.empty:
            out[vtx] = d
    if 'gas' not in out:
        raise FileNotFoundError(
            f'no pooled acceptance for variant {variant!r} under {acc_dir}\n'
            f'  run campaign_acceptance.py first')
    return out


def folded(A: dict, fine: np.ndarray, S: dict, bins=BINS) -> pd.DataFrame:
    """Every shape x its own vertex's acceptance, per topology, binned.

    `campaign_angle.fold` is reused rather than reimplemented: it multiplies on
    the fine grid and bins afterwards, which is the part that has to be right
    when the acceptance varies by an order of magnitude inside a 15 deg bin.
    """
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for name, s in S.items():
        vtx = VERTEX_OF.get(name, 'gas')
        acc = A.get(vtx, A['gas'])
        used = vtx if vtx in A else 'gas'
        for topo in TOPOLOGIES + ('all',):
            y = fold(acc, topo, s, fine, bins)
            for i, v in enumerate(y):
                rows.append(dict(topology=topo, model=name, vertex=used,
                                 theta=mid[i], frac=float(v)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# comparing
# --------------------------------------------------------------------------- #
def _obs(S: pd.DataFrame, topo: str, selection: str, bins=BINS) -> tuple:
    """(counts, bin centres) of one measured spectrum, on the fold's binning."""
    o = S[(S.topology == topo) & (S.selection == selection)] \
        .sort_values('theta')
    if o.empty:
        return None, None
    return o.n.to_numpy(float), o.theta.to_numpy()


def one_comp(S: pd.DataFrame, F: pd.DataFrame, selection: str,
             mixed: str = 'mixed') -> pd.DataFrame:
    """Each folded shape alone against one selection, one free normalisation.

    The same test `campaign_angle.compare` runs, with the capsule components
    added, so the new rows sit in the same table as the old ones on the same
    footing and the numbers are directly comparable to the ones already
    published.
    """
    rows = []
    for topo in TOPOLOGIES:
        obs, mid = _obs(S, topo, selection)
        if obs is None or obs.sum() < 20:
            continue
        err = np.sqrt(np.clip(obs, 1, None))
        cand = [(r.model, r.vertex, g.sort_values('theta').frac.to_numpy())
                for (r, g) in [(g.iloc[0], g) for _, g in
                               F[F.topology == topo].groupby('model')]]
        mx, _ = _obs(S, topo, mixed)
        if mx is not None and mx.sum() > 0:
            cand.append(('event-mixed (accidental shape)', 'measured',
                         mx / mx.sum()))
        for name, vtx, p in cand:
            if not np.isfinite(p).any() or p.sum() <= 0:
                continue
            pred = p * obs.sum()
            live = (obs + pred) > 0
            chi2 = float(np.sum(((obs[live] - pred[live]) / err[live]) ** 2))
            dof = int(live.sum() - 1)
            rows.append(dict(
                topology=topo, selection=selection, model=name, vertex=vtx,
                n_obs=int(obs.sum()), chi2=chi2, dof=dof,
                chi2dof=chi2 / max(dof, 1),
                frac_above_x17_obs=float(obs[mid > X17_MIN_DEG].sum()
                                         / obs.sum()),
                frac_above_x17_pred=float(p[mid > X17_MIN_DEG].sum())))
    return pd.DataFrame(rows).sort_values(['topology', 'chi2dof'],
                                          ignore_index=True)


def two_comp(S: pd.DataFrame, F: pd.DataFrame, selection: str,
             pair_model: str = 'Al capsule (after wall)',
             mixed: str = 'mixed') -> pd.DataFrame:
    """Pair shape + accidental shape, profiling chi2 over the accidental share.

    One free parameter beyond the normalisation, and it is a parameter that was
    already MEASURED by a route that never touched the opening angle -- the
    arm-to-arm scintillator timing.  So the interesting output is not the chi2
    on its own but whether the angle-fitted share agrees with the timing one.
    """
    rows = []
    for topo in TOPOLOGIES:
        obs, mid = _obs(S, topo, selection)
        mx, _ = _obs(S, topo, mixed)
        if obs is None or mx is None or obs.sum() < 20 or mx.sum() <= 0:
            continue
        g = F[(F.topology == topo) & (F.model == pair_model)] \
            .sort_values('theta')
        if g.empty:
            continue
        p_pair = g.frac.to_numpy()
        p_acc = mx / mx.sum()
        if p_pair.sum() <= 0:
            continue
        p_pair = p_pair / p_pair.sum()
        err = np.sqrt(np.clip(obs, 1, None))
        chi2, nlive = [], []
        for f in F_GRID:
            pred = ((1 - f) * p_pair + f * p_acc) * obs.sum()
            live = (obs + pred) > 0
            chi2.append(float(np.sum(((obs[live] - pred[live])
                                      / err[live]) ** 2)))
            nlive.append(int(live.sum()))
        chi2 = np.array(chi2)
        i = int(np.argmin(chi2))
        # two fitted parameters: the normalisation and the mixture.  Counted
        # over the LIVE bins only, the way `one_comp` does -- a bin where both
        # the data and the model are empty is not a constraint.
        dof = int(nlive[i] - 2)
        # 1 sigma on one parameter is delta chi2 = 1 about the minimum
        inside = np.flatnonzero(chi2 <= chi2[i] + 1.0)
        lo, hi = float(F_GRID[inside[0]]), float(F_GRID[inside[-1]])
        tim = F_TIMING.get(topo)
        rows.append(dict(
            topology=topo, selection=selection, pair_model=pair_model,
            n_obs=int(obs.sum()), f_acc=float(F_GRID[i]), f_lo=lo, f_hi=hi,
            chi2=float(chi2[i]), dof=dof, chi2dof=float(chi2[i] / max(dof, 1)),
            chi2dof_pure_pair=float(chi2[0] / max(dof + 1, 1)),
            chi2dof_pure_acc=float(chi2[-1] / max(dof + 1, 1)),
            f_timing=tim[0] if tim else np.nan,
            f_timing_lo=tim[1] if tim else np.nan,
            f_timing_hi=tim[2] if tim else np.nan,
            timing_consistent=(bool(hi >= tim[1] and lo <= tim[2])
                               if tim else None)))
    return pd.DataFrame(rows)


def ratio_test(S: pd.DataFrame, F: pd.DataFrame, selection: str) -> pd.DataFrame:
    """opposing / intra above the X17 threshold -- the least model-dependent test.

    The acceptance normalisation, the vertex model and the efficiency scale are
    largely common to the two topologies and divide out; the shape of the physics
    does not.  Kept identical in form to `campaign_angle.ratio_test` so the
    capsule rows drop straight into the same comparison.
    """
    rows = []
    for topo in TOPOLOGIES:
        obs, mid = _obs(S, topo, selection)
        if obs is None or obs.sum() < 1:
            continue
        k = float(obs[mid > X17_MIN_DEG].sum())
        n = float(obs.sum())
        r = dict(topology=topo, selection=selection, n=int(n), k_above=int(k),
                 frac_obs=k / n,
                 err=float(np.sqrt(max(k, 1)) / n))
        for model, g in F[F.topology == topo].groupby('model'):
            p = g.sort_values('theta').frac.to_numpy()
            r[f'frac[{model}]'] = (float(p[mid > X17_MIN_DEG].sum() / p.sum())
                                   if p.sum() > 0 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def corrected(A: dict, S: pd.DataFrame, selection: str,
              vertex: str = 'gas', bins=BINS) -> pd.DataFrame:
    """The measured spectrum divided by the acceptance -- the plan's third plot.

    PLAN sec S4 asks for the raw spectrum, the acceptance-corrected one and the
    category ratios, in that order.  **Folding the model forward is the better
    test and it is what every chi2 on this page does**; dividing is here because
    it is the only form in which the data can be compared against a birth-level
    median.  A bin where the acceptance is a thousandth of its peak divides a
    handful of counts by nearly nothing, so bins below ``FLOOR`` of the group's
    own peak acceptance are dropped rather than plotted as a spike -- and the
    fraction of the sample dropped travels with the table.
    """
    FLOOR = 0.02
    mid = 0.5 * (bins[:-1] + bins[1:])
    acc = A.get(vertex, A['gas'])
    rows = []
    for topo in TOPOLOGIES:
        obs, om = _obs(S, topo, selection, bins)
        if obs is None or obs.sum() < 20:
            continue
        g = acc[acc.group == topo].sort_values('theta')
        if g.empty:
            continue
        a = np.interp(mid, g.theta.to_numpy(), g.acc.to_numpy(),
                      left=0.0, right=0.0)
        peak = a.max()
        live = a > FLOOR * peak
        c = np.where(live, obs / np.where(a > 0, a, np.nan), np.nan)
        e = np.where(live, np.sqrt(np.clip(obs, 1, None))
                     / np.where(a > 0, a, np.nan), np.nan)
        tot = np.nansum(c)
        if tot <= 0:
            continue
        cdf = np.nancumsum(c) / tot
        for i in range(len(mid)):
            rows.append(dict(topology=topo, selection=selection, theta=mid[i],
                             n_obs=float(obs[i]), acc=float(a[i]),
                             live=bool(live[i]),
                             corrected=float(c[i]) if live[i] else np.nan,
                             err=float(e[i]) if live[i] else np.nan,
                             frac=float(c[i] / tot) if live[i] else np.nan))
        rows.append(dict(topology=topo, selection=selection, theta=np.nan,
                         n_obs=float(obs.sum()),
                         acc=np.nan, live=True,
                         corrected=float(tot), err=np.nan,
                         frac=np.nan,
                         median_deg=float(np.interp(0.5, cdf[live], mid[live]))
                         if live.any() else np.nan,
                         frac_above_109=float(np.nansum(c[mid > X17_MIN_DEG])
                                              / tot),
                         frac_dropped=float(obs[~live].sum() / obs.sum())))
    return pd.DataFrame(rows)


def agreement(U: pd.DataFrame, fine: np.ndarray, S: dict, selection: str,
              bins=BINS) -> pd.DataFrame:
    """The corrected spectrum over each birth model, bin by bin.

    The figure that comes out of :func:`corrected` shows the perpendicular
    topology following the capsule curve over the middle of its range and
    rising above it past the X17 threshold.  This puts a number on both halves:
    the ratio in each bin, and a chi2 over the bins BELOW the threshold, which
    is the region where a signal cannot contribute and the comparison is
    therefore a background test rather than a search.

    Normalised over the sub-threshold region only, for the same reason: a
    normalisation fitted over the whole range would let an excess above 109 deg
    pull the level below it and hide itself.

    ``excess_sigma`` counts the corrected bins' own errors and **not** the
    uncertainty on that normalisation, which is a few per cent on four
    sub-threshold bins and would soften the number rather than change its sign.
    Read it as an indication of size, not as a significance.
    """
    mid = 0.5 * (bins[:-1] + bins[1:])
    dth = np.diff(bins)
    sub = mid < X17_MIN_DEG
    rows = []
    for topo in TOPOLOGIES:
        g = U[(U.topology == topo) & (U.selection == selection)
              & U.theta.notna() & U.live].sort_values('theta')
        if g.empty:
            continue
        th = g.theta.to_numpy()
        c = g.corrected.to_numpy()
        e = g.err.to_numpy()
        for name, y in S.items():
            # the model, integrated into the same coarse bins
            idx = np.digitize(fine, bins) - 1
            m = np.array([y[idx == i].sum() for i in range(len(mid))])
            m = np.interp(th, mid, m)
            live_sub = (th < X17_MIN_DEG) & (m > 0) & (c > 0)
            if live_sub.sum() < 2 or m[live_sub].sum() <= 0:
                continue
            k = c[live_sub].sum() / m[live_sub].sum()      # sub-threshold only
            pred = k * m
            chi2 = float(np.sum(((c[live_sub] - pred[live_sub])
                                 / e[live_sub]) ** 2))
            dof = int(live_sub.sum() - 1)
            above = th >= X17_MIN_DEG
            rows.append(dict(
                topology=topo, selection=selection, model=name,
                n_bins_sub=int(live_sub.sum()), norm=float(k),
                chi2_sub=chi2, dof_sub=dof,
                chi2dof_sub=chi2 / max(dof, 1),
                ratio_sub=float(c[live_sub].sum() / pred[live_sub].sum()),
                ratio_above=(float(c[above].sum() / pred[above].sum())
                             if above.any() and pred[above].sum() > 0
                             else np.nan),
                excess_above=(float(c[above].sum() - pred[above].sum())
                              if above.any() else np.nan),
                excess_sigma=(float((c[above].sum() - pred[above].sum())
                                    / np.sqrt(np.sum(e[above] ** 2)))
                              if above.any() and np.any(e[above] > 0)
                              else np.nan)))
    return pd.DataFrame(rows).sort_values(
        ['topology', 'chi2dof_sub'], ignore_index=True)


def acceptance_distortion(A: dict, fine: np.ndarray, S: dict,
                          bins=BINS) -> pd.DataFrame:
    """What the acceptance does to each shape: birth median vs folded median.

    The point of the whole exercise in one table.  A shape whose median moves
    100 degrees on folding is being reported by the apparatus rather than by the
    physics, and the fraction above 109 degrees -- the number an X17 claim would
    rest on -- moves with it.
    """
    from sept26_prelim_analysis import ipc_born as IB
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for name, s in S.items():
        vtx = VERTEX_OF.get(name, 'gas')
        acc = A.get(vtx, A['gas'])
        b_med = float(np.interp(0.5, np.cumsum(s * np.diff(IB.THETA_BINS)),
                                fine))
        b_109 = float(IB.frac_above(s, X17_MIN_DEG))
        for topo in TOPOLOGIES + ('all',):
            y = fold(acc, topo, s, fine, bins)
            if not np.isfinite(y).any() or y.sum() <= 0:
                continue
            cdf = np.cumsum(y) / y.sum()
            rows.append(dict(
                model=name, vertex=vtx if vtx in A else 'gas', topology=topo,
                birth_median_deg=b_med, birth_frac109=b_109,
                folded_median_deg=float(np.interp(0.5, cdf, mid)),
                folded_frac109=float(y[mid > X17_MIN_DEG].sum() / y.sum())))
    d = pd.DataFrame(rows)
    d['median_shift_deg'] = d.folded_median_deg - d.birth_median_deg
    d['frac109_ratio'] = np.where(d.birth_frac109 > 0,
                                  d.folded_frac109 / d.birth_frac109, np.nan)
    return d


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--acceptance', default=str(paths.out(
        'acceptance_campaign')))
    ap.add_argument('--spectra', default=str(paths.out('angle_campaign')
                                             / 'spectra.csv'))
    ap.add_argument('--variant', default='incidence',
                    help='which efficiency variant of the acceptance to fold '
                         'through: flat, u_map or incidence')
    ap.add_argument('--assume', default='M1',
                    help='multipole for aluminium lines with no assignment')
    a = ap.parse_args()

    acc_dir = Path(paths.require(a.acceptance, 'the campaign acceptance -- run '
                                               'campaign_acceptance.py first'))
    S = pd.read_csv(paths.require(a.spectra, 'the campaign angle spectra -- '
                                             'run campaign_angle.py first'))
    od = paths.out('fold_campaign')
    fine, shp, prov = shapes(od / 'shape_cache.csv', a.assume)
    A = read_acceptance(acc_dir, a.variant)
    print(f'acceptance {a.variant!r}, vertex models {sorted(A)}\n')

    F = folded(A, fine, shp)
    D = acceptance_distortion(A, fine, shp)
    C = pd.concat([one_comp(S, F, s) for s in
                   ('all_no_b2b', 'tagged', 'tight_pair')], ignore_index=True)
    T = pd.concat([two_comp(S, F, s, m) for s in ('all_no_b2b', 'tight_pair')
                   for m in ('Al capsule (after wall)', '3He gas M1+E0')],
                  ignore_index=True)
    R = pd.concat([ratio_test(S, F, s) for s in ('all_no_b2b', 'tight_pair')],
                  ignore_index=True)
    U = pd.concat([corrected(A, S, s) for s in ('all_no_b2b', 'tight_pair')],
                  ignore_index=True)
    G = pd.concat([agreement(U, fine, shp, s)
                   for s in ('all_no_b2b', 'tight_pair')], ignore_index=True)

    F.to_csv(od / 'folded.csv', index=False)
    D.to_csv(od / 'distortion.csv', index=False)
    C.to_csv(od / 'one_component.csv', index=False)
    T.to_csv(od / 'two_component.csv', index=False)
    R.to_csv(od / 'ratio.csv', index=False)
    U.to_csv(od / 'corrected.csv', index=False)
    G.to_csv(od / 'agreement.csv', index=False)
    json.dump(dict(schema=SCHEMA, acceptance=str(acc_dir),
                   variant=a.variant, spectra=a.spectra,
                   vertex_models=sorted(A), vertex_of=VERTEX_OF,
                   bins=[float(x) for x in BINS], shape_provenance=prov,
                   f_timing=F_TIMING,
                   caveat='the after-wall curves carry a multiple-scattering '
                          'Gaussian that is not trustworthy for '
                          f'{prov.get("capsule_unreliable", float("nan")):.0%} '
                          'of the capsule weight; opening angle alone cannot '
                          'separate a 2-4 MeV capsule pair from a 20.6 MeV gas '
                          'pair -- only the total pair energy can, and this '
                          'setup does not measure it'),
              open(od / 'campaign_fold.meta.json', 'w'), indent=1)

    print('what the ACCEPTANCE does to each shape (topology "all")\n')
    print(f'{"model":>28} {"birth":>7} {"folded":>7} {"shift":>7} '
          f'{">109 birth":>11} {">109 folded":>12}')
    for r in D[D.topology == 'all'].itertuples():
        print(f'{r.model:>28} {r.birth_median_deg:>6.1f}d '
              f'{r.folded_median_deg:>6.1f}d {r.median_shift_deg:>+6.1f}d '
              f'{r.birth_frac109:>11.4f} {r.folded_frac109:>12.4f}')

    for sel in ('all_no_b2b', 'tight_pair'):
        g = C[C.selection == sel]
        if g.empty:
            continue
        print(f'\none free normalisation, selection {sel!r} -- '
              f'chi2/dof, best first\n')
        for topo, h in g.groupby('topology'):
            print(f'  {topo}  (n = {int(h.n_obs.iloc[0]):,})')
            for r in h.sort_values('chi2dof').itertuples():
                print(f'      {r.chi2dof:>9.2f}  {r.model}')

    print('\nTWO components: pair shape + the measured accidental shape\n')
    print(f'{"selection":>12} {"topology":>14} {"pair model":>26} '
          f'{"f_acc":>16} {"chi2/dof":>9} {"timing f":>9} {"ok":>4}')
    for r in T.itertuples():
        print(f'{r.selection:>12} {r.topology:>14} {r.pair_model:>26} '
              f'{r.f_acc:>5.2f} [{r.f_lo:.2f},{r.f_hi:.2f}] '
              f'{r.chi2dof:>9.2f} {r.f_timing:>9.2f} '
              f'{"yes" if r.timing_consistent else "NO":>4}')
    if 'median_deg' in U.columns:
        print('\nthe data DIVIDED by the acceptance -- birth-level medians, '
              'to compare against the shapes above\n')
        h = U[U.theta.isna()]
        print(f'{"selection":>12} {"topology":>14} {"median":>8} '
              f'{">109 deg":>9} {"dropped":>8}')
        for r in h.itertuples():
            print(f'{r.selection:>12} {r.topology:>14} '
                  f'{r.median_deg:>7.1f}d {r.frac_above_109:>9.4f} '
                  f'{r.frac_dropped:>8.3f}')
    if not G.empty:
        print('\nthe CORRECTED spectrum against each birth model, normalised '
              'BELOW 109 deg only\n')
        print(f'{"selection":>12} {"topology":>14} {"model":>28} '
              f'{"chi2/dof <109":>13} {"ratio >109":>11} {"sigma":>7}')
        for sel in ('all_no_b2b', 'tight_pair'):
            for topo in TOPOLOGIES:
                h = G[(G.selection == sel) & (G.topology == topo)]
                for r in h.head(3).itertuples():
                    print(f'{sel:>12} {topo:>14} {r.model:>28} '
                          f'{r.chi2dof_sub:>13.1f} {r.ratio_above:>11.2f} '
                          f'{r.excess_sigma:>7.1f}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
