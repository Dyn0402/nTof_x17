#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_angle.py -- the opening-angle distribution, over the whole campaign.

`opening_angle.py` does this for ONE run against `pair_physics.VARIANTS`.  This
module does it for all of them, on the condor full pass, against the thermal
Born prediction that superseded those variants (`ipc_channels.thermal_spectrum`,
CLAUDE.md and PLAN.md sec S4).

THREE SAMPLES, AND THE DIFFERENCE BETWEEN THEM IS THE RESULT.  A "pair" gets
progressively harder to be, and the spectrum has to be read at every step
because each step removes a different background:

  ``all``         every unordered pair of selected tracks inside one trigger.
                  Dominated by two unrelated particles: the production accept
                  window is 160 ns wide and the plastic family's own accidental
                  rate lands something in it almost every time
                  (`HANDOFF_ACCIDENTAL_TIMING.md`).
  ``tagged``      both arms carry a scintillator time.  Still not a
                  coincidence -- only a sample in which one can be tested.
  ``tight_pair``  both arms prompt (|t| <= 30 ns) and prompt with EACH OTHER
                  (|t1-t2| <= 20 ns), and not the back-to-back single particle.
                  This is the coincident sample.

THE BACK-TO-BACK CUT IS NOT COSMETIC AND IT LANDS ON THE SIGNAL REGION.  One
charged particle that crosses the target and punches through BOTH opposing
chambers reads as a pair at ~180 deg, and it is perfectly time-coincident
because it is one particle.  Campaign-wide it is **40 % of the tight opposing
sample**.  X17's own region starts at 109 deg, so this background sits inside
it, and every number here is quoted with the cut applied and without.

WHAT THE INTRA-CHAMBER CONTROL CAN AND CANNOT HAVE.  Both legs in one chamber
means there is only ONE arm and therefore no ``t1 - t2``: the mutual half of
the tight cut does not exist for it.  So intra is carried at the ``all`` level
only, and the comparison between an intra spectrum with no timing cut and an
opposing spectrum with one is NOT like for like.  Said here rather than
buried, because PLAN sec S4 makes intra the background normalisation.

WHAT IS BORROWED, AND IT IS THE LEADING SYSTEMATIC.  The acceptance is
`acceptance.py`'s run_145 curve -- the only run with the efficiency map it
needs.  It is applied to the pooled campaign spectrum and flagged
``acceptance_source = run_145`` everywhere.  A campaign acceptance needs a
per-run efficiency measurement that does not exist yet.

    python -m sept26_prelim_analysis.campaign_angle --jobs 8
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

SCHEMA = 'sept26_prelim/campaign_angle/1'
#: 15 deg, the binning `opening_angle.py` publishes in.
BINS = np.arange(0.0, 181.0, 15.0)
#: The X17 threshold: a 17 MeV boson from a 20.6 MeV transition cannot make a
#: pair below this.  Everything below it is background by construction.
X17_MIN_DEG = 109.0
#: Opposing and nearly collinear: one particle through both chambers, not a
#: pair.  The same constant `tight_coincidence.py` uses, imported rather than
#: retyped so the two can never drift apart.
from sept26_prelim_analysis.tight_coincidence import (  # noqa: E402
    BACK_TO_BACK_DEG, PRE_ACCESS_RUNS)

TOPOLOGIES = ('intra', 'perpendicular', 'opposing')


# --------------------------------------------------------------------------- #
# the pairs
# --------------------------------------------------------------------------- #
def one_run(run: str, subruns, src: str, dca_max: float) -> tuple:
    """Every real pair of one run, and its event-mixed null.  In a worker."""
    from sept26_prelim_analysis import source_imaging as SI
    try:
        real, mixed = SI.vertices(run, subruns, dca_max, src=Path(src))
        out = []
        for d, is_mixed in ((real, False), (mixed, True)):
            if not len(d):
                continue
            d = d[['arm1', 'arm2', 'open_deg', 'v_r', 'sep_mm']].copy()
            d['topo'] = [AC.topology(a, b) for a, b in zip(d.arm1, d.arm2)]
            d['mixed'] = is_mixed
            d['run'] = run
            out.append(d)
        if not out:
            return run, None, 'no pairs -- check angle_calibrated for this run'
        d = pd.concat(out, ignore_index=True)
        d['back_to_back'] = ((d.topo == 'opposing')
                             & (d.open_deg > BACK_TO_BACK_DEG))
        return run, d, ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def campaign_pairs(src: Path, dca_max: float, jobs: int,
                   include_pre_access: bool) -> tuple:
    """All real pairs and their mixed null, pooled, with the runs that failed."""
    import re
    rs = {}
    for p in sorted(src.glob('tracks_run_*_stat090_*.parquet')):
        m = re.match(r'tracks_(run_\d+)_(stat090_\d+)\.parquet$', p.name)
        if m:
            rs.setdefault(m.group(1), []).append(m.group(2))
    if not include_pre_access:
        for r in PRE_ACCESS_RUNS:
            rs.pop(r, None)
    print(f'{len(rs)} run(s) from {src}'
          f'{"" if include_pre_access else " (pre-access excluded)"}\n')
    out, bad = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, subs, str(src), dca_max): r
                for r, subs in rs.items()}
        for f in as_completed(futs):
            run, d, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --      {err}', flush=True)
                continue
            out.append(d)
            n = int((~d.mixed).sum())
            print(f'  {run:<10} ok      {n:>7,} real pairs', flush=True)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), bad


def attach_tight(P: pd.DataFrame, tight: Path) -> pd.DataFrame:
    """Fold the tight-coincidence census in as per-topology COUNTS.

    The tight sample is built by `tight_coincidence.py` from the scintillator
    slim and cannot be rederived here; what this does is line its counts up
    with the same topologies and bins so the three samples are one table.
    """
    m = pd.read_parquet(tight)
    m['topo'] = [AC.topology(a, b) for a, b in zip(m.arm1, m.arm2)]
    return m


# --------------------------------------------------------------------------- #
# the spectra
# --------------------------------------------------------------------------- #
def spectrum(theta, bins=BINS) -> np.ndarray:
    h, _ = np.histogram(np.asarray(theta, float), bins=bins)
    return h


def spectra(P: pd.DataFrame, M: pd.DataFrame, bins=BINS) -> pd.DataFrame:
    """One row per (topology, selection, bin) -- the deliverable table.

    ``all`` and ``mixed`` come from the track table; ``tagged``, ``tight`` and
    ``tight_pair`` from the tight-coincidence product, which exists only for
    inter-chamber pairs.  A topology with no rows in a selection is absent, not
    zero-filled: an intra ``tight_pair`` count of zero would read as a
    measurement rather than as a cut that cannot be formed.
    """
    real = P[~P.mixed]
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []

    def add(topo, sel, theta, note=''):
        h = spectrum(theta, bins)
        for i, v in enumerate(h):
            rows.append(dict(topology=topo, selection=sel, theta=mid[i],
                             lo=bins[i], hi=bins[i + 1], n=int(v), note=note))

    for topo in TOPOLOGIES:
        r = real[real.topo == topo]
        x = P[P.mixed & (P.topo == topo)]
        add(topo, 'all', r.open_deg)
        add(topo, 'all_no_b2b', r[~r.back_to_back].open_deg)
        if len(x):
            add(topo, 'mixed', x.open_deg)
        if M is not None:
            g = M[M.topo == topo]
            if len(g):
                add(topo, 'tagged', g.open_deg)
                add(topo, 'tight', g[g.tight].open_deg)
                add(topo, 'tight_pair', g[g.tight_pair].open_deg)
    return pd.DataFrame(rows)


def census(P: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """How many pairs survive each step, per topology, and how many are high.

    ``frac_above_x17`` is the number the analysis turns on: X17 cannot make a
    pair below 109 deg, so a signal moves this fraction and nothing else.
    """
    real = P[~P.mixed]
    rows = []
    for topo in TOPOLOGIES + ('(all)',):
        r = real if topo == '(all)' else real[real.topo == topo]
        x = P[P.mixed] if topo == '(all)' else P[P.mixed & (P.topo == topo)]
        g = (M if M is not None and topo == '(all)' else
             (M[M.topo == topo] if M is not None else None))
        def above(s):
            s = s.dropna()
            return float((s > X17_MIN_DEG).mean()) if len(s) else np.nan
        row = dict(topology=topo, n_all=int(len(r)),
                   n_all_no_b2b=int((~r.back_to_back).sum()),
                   n_mixed=int(len(x)),
                   n_b2b=int(r.back_to_back.sum()),
                   frac_above_x17_all=above(r.open_deg),
                   frac_above_x17_mixed=above(x.open_deg),
                   median_all=float(r.open_deg.median()) if len(r) else np.nan)
        if g is not None and len(g):
            row.update(n_tagged=int(len(g)), n_tight=int(g.tight.sum()),
                       n_tight_pair=int(g.tight_pair.sum()),
                       frac_above_x17_tight_pair=above(g[g.tight_pair].open_deg),
                       median_tight_pair=float(g[g.tight_pair].open_deg.median())
                       if g.tight_pair.any() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the expectation
# --------------------------------------------------------------------------- #
def fold(acc: pd.DataFrame, group: str, shape: np.ndarray,
         shape_theta: np.ndarray, bins=BINS) -> np.ndarray:
    """physics x acceptance, integrated into the coarse bins, normalised to 1.

    The two multiply pointwise on the FINE physics grid and are binned
    afterwards.  Multiplying already-binned quantities would be wrong wherever
    either varies inside a bin, and the acceptance varies by an order of
    magnitude across 15 deg near the edges.
    """
    g = acc[acc.group == group].sort_values('theta')
    if g.empty:
        return np.full(len(bins) - 1, np.nan)
    a = np.interp(shape_theta, g.theta.to_numpy(), g.acc.to_numpy(),
                  left=0.0, right=0.0)
    w = shape * a
    idx = np.digitize(shape_theta, bins) - 1
    out = np.array([w[idx == i].sum() for i in range(len(bins) - 1)])
    s = out.sum()
    return out / s if s > 0 else out


def expectation(acc: pd.DataFrame, bins=BINS, n_x17: int = 400_000) -> pd.DataFrame:
    """The thermal Born prediction, folded, per topology -- and X17 beside it.

    The IPC shape is `ipc_channels.thermal_spectrum()`: the M1 and E0 Born
    multipoles in the mixture the <2 eV window makes, which is what CLAUDE.md
    and PLAN sec S4 require and what `pair_physics.VARIANTS` was replaced by.
    The two components are folded SEPARATELY as well as summed, because the E0
    fraction is the one free parameter and a fit has to be able to float it.
    """
    from sept26_prelim_analysis import ipc_channels as IC
    from sept26_prelim_analysis import pair_physics as PP
    P = IC.thermal_spectrum()
    fine = P.theta_mid.to_numpy()
    shapes = {'IPC thermal (M1+E0)': P.total.to_numpy(),
              'IPC M1 only': P.M1.to_numpy(),
              'IPC E0 only': P.E0.to_numpy()}
    h17, e17 = np.histogram(PP.x17_angles(n_x17),
                            bins=np.arange(0.0, 180.01, 1.0), density=True)
    shapes['X17 (17 MeV boson)'] = h17
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for topo in TOPOLOGIES + ('all',):
        for name, s in shapes.items():
            y = fold(acc, topo, s, fine, bins)
            for i, v in enumerate(y):
                rows.append(dict(topology=topo, model=name, theta=mid[i],
                                 frac=float(v)))
    return pd.DataFrame(rows)


def compare(S: pd.DataFrame, E: pd.DataFrame, selection: str,
            bins=BINS) -> pd.DataFrame:
    """Each model against one selection's shape, per topology.

    No background is subtracted.  Both candidate normalisations failed
    (`opening_angle.compare`): the mixed sample is a shape and carries no rate,
    and the Poisson product over-predicts the observed pair count 2-4x because
    the trigger correlates the arms.  So the missing normalisation is stated as
    the leading systematic rather than guessed, and the event-mixed shape enters
    the SAME table as the physics models, on the same footing -- if the data
    looks more like two unrelated tracks than like any pair spectrum, that has
    to be visible here.
    """
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for topo in TOPOLOGIES:
        o = S[(S.topology == topo) & (S.selection == selection)] \
            .sort_values('theta')
        if o.empty or o.n.sum() < 20:
            continue
        obs = o.n.to_numpy(float)
        err = np.sqrt(np.clip(obs, 1, None))
        cand = [(n, g.sort_values('theta').frac.to_numpy())
                for n, g in E[E.topology == topo].groupby('model')]
        mx = S[(S.topology == topo) & (S.selection == 'mixed')] \
            .sort_values('theta').n.to_numpy(float)
        if mx.sum() > 0:
            cand.append(('event-mixed (accidental shape)', mx / mx.sum()))
        for name, p in cand:
            if not np.isfinite(p).any() or p.sum() <= 0:
                continue
            pred = p * obs.sum()
            live = (obs + pred) > 0
            chi2 = float(np.sum(((obs[live] - pred[live]) / err[live]) ** 2))
            dof = int(live.sum() - 1)
            rows.append(dict(
                topology=topo, selection=selection, model=name,
                n_obs=int(obs.sum()), chi2=chi2, dof=dof,
                chi2dof=chi2 / max(dof, 1),
                frac_above_x17_obs=float(obs[mid > X17_MIN_DEG].sum()
                                         / max(obs.sum(), 1)),
                frac_above_x17_pred=float(p[mid > X17_MIN_DEG].sum())))
    return pd.DataFrame(rows)


def ratio_test(S: pd.DataFrame, E: pd.DataFrame, selection: str,
               bins=BINS) -> pd.DataFrame:
    """opposing / intra: the test that depends least on the model.

    The acceptance normalisation, the vertex model and the efficiency scale are
    largely common to the two topologies and divide out; the shape of the
    physics does not.  Quoted as the fraction above the X17 threshold, which is
    the quantity a signal actually moves.
    """
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for topo in TOPOLOGIES:
        o = S[(S.topology == topo) & (S.selection == selection)] \
            .sort_values('theta')
        if o.empty or o.n.sum() < 20:
            continue
        obs = o.n.to_numpy(float)
        n, k = obs.sum(), obs[mid > X17_MIN_DEG].sum()
        row = dict(topology=topo, selection=selection, n=int(n),
                   k_above=int(k), frac_obs=float(k / n),
                   err=float(np.sqrt(max(k, 1)) / n))
        for name, g in E[E.topology == topo].groupby('model'):
            p = g.sort_values('theta').frac.to_numpy()
            row[f'frac[{name}]'] = float(p[mid > X17_MIN_DEG].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(K: pd.DataFrame, C: pd.DataFrame) -> dict:
    """The headline, from the census ``K`` and the shape comparison ``C``."""
    opp = K[K.topology == 'opposing']
    tp = C[C.selection == 'tight_pair']
    best = (tp.sort_values('chi2dof').iloc[0] if len(tp) else None)
    allrow = K[K.topology == '(all)']
    return dict(
        n_pairs_all=int(allrow.n_all.iloc[0]) if len(allrow) else 0,
        n_tight_pair=(int(opp.n_tight_pair.iloc[0])
                      if len(opp) and 'n_tight_pair' in opp else None),
        opposing_frac_above_x17_all=(float(opp.frac_above_x17_all.iloc[0])
                                     if len(opp) else None),
        opposing_frac_above_x17_tight_pair=(
            float(opp.frac_above_x17_tight_pair.iloc[0])
            if len(opp) and 'frac_above_x17_tight_pair' in opp else None),
        b2b_frac_of_opposing=(float(opp.n_b2b.iloc[0] / opp.n_all.iloc[0])
                              if len(opp) and opp.n_all.iloc[0] else None),
        best_fitting_model=(None if best is None else
                            dict(topology=str(best.topology),
                                 model=str(best.model),
                                 chi2dof=float(best.chi2dof))),
        acceptance_source='run_145 (BORROWED -- no campaign acceptance exists)')


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=None,
                    help='stage-3 tracks; default <out>/stage3_fullpass')
    ap.add_argument('--tight', default=None,
                    help='the tight-coincidence pair table; default '
                         '<out>/tight_coincidence/pairs_tight_campaign.parquet')
    ap.add_argument('--acceptance', default=None,
                    help='default <out>/angle/acceptance_run_145.csv -- '
                         'BORROWED, there is no campaign acceptance')
    ap.add_argument('--out', default=None, help='default <out>/angle_campaign')
    ap.add_argument('--dca', type=float, default=30.0)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--include-pre-access', action='store_true',
                    help='include run_79/run_81 -- a DIFFERENT detector '
                         'condition, ~2x the INTER fraction')
    a = ap.parse_args()

    src = Path(a.src) if a.src else paths.out('stage3_fullpass')
    tight = Path(a.tight) if a.tight else (
        paths.out('tight_coincidence') / 'pairs_tight_campaign.parquet')
    accp = Path(a.acceptance) if a.acceptance else (
        paths.out('angle') / 'acceptance_run_145.csv')
    od = Path(a.out) if a.out else paths.out('angle_campaign')
    od.mkdir(parents=True, exist_ok=True)
    paths.require(src, 'the stage-3 track tree')
    paths.require(accp, 'the acceptance -- run acceptance.py')

    P, bad = campaign_pairs(src, a.dca, a.jobs, a.include_pre_access)
    if P.empty:
        print('no pairs at all -- nothing to do')
        return 1
    M = attach_tight(P, tight) if tight.exists() else None
    if M is None:
        print(f'\n!! no tight-coincidence table at {tight}; the timing-cut '
              f'selections will be absent')

    acc = pd.read_csv(accp)
    S = spectra(P, M)
    K = census(P, M)
    E = expectation(acc)
    Call = compare(S, E, 'all_no_b2b')
    Ctp = compare(S, E, 'tight_pair')
    C = pd.concat([Call, Ctp], ignore_index=True)
    R = pd.concat([ratio_test(S, E, 'all_no_b2b'),
                   ratio_test(S, E, 'tight_pair')], ignore_index=True)

    P.drop(columns=['v_r', 'sep_mm']).to_parquet(od / 'pairs.parquet',
                                                 index=False)
    S.to_csv(od / 'spectra.csv', index=False)
    K.to_csv(od / 'census.csv', index=False)
    E.to_csv(od / 'expectation.csv', index=False)
    C.to_csv(od / 'compare.csv', index=False)
    R.to_csv(od / 'ratio.csv', index=False)
    vd = verdict(K, C)
    json.dump(dict(schema=SCHEMA, src=str(src), tight=str(tight),
                   acceptance=str(accp), dca_max=a.dca,
                   bins=BINS.tolist(), x17_min_deg=X17_MIN_DEG,
                   back_to_back_deg=BACK_TO_BACK_DEG,
                   include_pre_access=bool(a.include_pre_access),
                   runs_failed=bad, verdict=vd),
              open(od / 'campaign_angle.meta.json', 'w'), indent=1,
              default=float)

    print('\nCENSUS -- how many pairs survive each step')
    print(K.round(3).to_string(index=False))
    print('\nSPECTRUM, tight_pair (counts per 15 deg)')
    piv = S[S.selection == 'tight_pair'].pivot_table(
        index='theta', columns='topology', values='n', aggfunc='sum')
    print(piv.to_string() if len(piv) else '  (none)')
    print('\nSPECTRUM, all pairs minus back-to-back')
    piv = S[S.selection == 'all_no_b2b'].pivot_table(
        index='theta', columns='topology', values='n', aggfunc='sum')
    print(piv.to_string())
    print('\nSHAPE COMPARISON (no subtraction -- see the docstring)')
    print(C.round(3).to_string(index=False))
    print('\nTHE MODEL-LIGHT TEST: fraction above 109 deg')
    print(R.round(4).to_string(index=False))
    print('\nVERDICT')
    print(json.dumps(vd, indent=1, default=float))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
