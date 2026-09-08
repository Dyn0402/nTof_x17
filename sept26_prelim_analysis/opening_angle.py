#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opening_angle.py -- the final observable, against something.

    measured(theta | topology)   vs   physics(theta) x acceptance(theta | topology)

The two factors come from the two modules either side of this one:
`pair_physics.py` for the first -- carried as a BAND, because the internal-pair
matrix element is not ours -- and `acceptance.py` for the second, which is ours
and is measured.

THE TOPOLOGIES, and why the split is not cosmetic.  Four chambers at 90 degrees
means the opening angle a pair can have is decided almost entirely by which two
chambers it lands in:

  intra          both legs in one chamber.  theta <~ 90 deg.  No X17 can appear
                 here -- its minimum is 109 deg -- so this is the IPC continuum
                 measured directly, and it is the strongest background test we
                 have.
  perpendicular  neighbouring chambers.  theta ~ 60-120 deg.
  opposing       facing chambers (A-C; B-D, which we cannot use).  theta >~ 110
                 deg.  The signal region.

Splitting them is what makes the comparison meaningful, because the acceptance
is wildly different between them and a pooled spectrum is mostly a picture of
the geometry.

THE TEST THAT DEPENDS LEAST ON THE MODEL is the RATIO between topologies.  A
steeply falling IPC continuum and a 110-140 deg X17 peak differ most between
`intra` and `opposing`, and much of the acceptance uncertainty -- the vertex
model, the efficiency scale, the trigger -- is common and divides out.

CHAMBER B IS NOT IN ANY OF THIS.  No field cage, no angle.  That costs the
B-D opposing channel entirely, which is half the signal topology.

    python -m sept26_prelim_analysis.opening_angle --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402
from sept26_prelim_analysis import acceptance as AC  # noqa: E402
from sept26_prelim_analysis import pair_physics as PP  # noqa: E402

SCHEMA = 'sept26_prelim/opening_angle/1'
BINS = np.arange(0.0, 181.0, 15.0)
X17_MIN_DEG = 109.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def measured(run: str, subruns, dca_max: float = 30.0) -> tuple:
    """Observed pairs and their event-mixed control, with the fine topology.

    Both come from `source_imaging.vertices`, so the pairing rule, the track
    selection and the mixing are the ones already validated there -- and the
    mixed sample here is the accidental background, measured rather than
    modelled.
    """
    from sept26_prelim_analysis import source_imaging as SI
    real, mixed = SI.vertices(run, subruns, dca_max)
    for d in (real, mixed):
        if len(d):
            d['topo'] = [AC.topology(a, b) for a, b in zip(d.arm1, d.arm2)]
    return real, mixed


def n_triggers(run: str, subruns) -> int:
    """Total DAQ triggers, from the stage-1 census -- the only honest denominator."""
    n = 0
    for sub in subruns:
        c = pd.read_csv(paths.require(
            paths.out('stage1') / f'census_{run}_{sub}.csv',
            f'stage-1 census for {sub}'))
        n += int(c.loc[c.cls == '(total)', 'n'].iloc[0])
    return n


def accidentals(run: str, subruns, real: pd.DataFrame,
                dca_max: float = 30.0) -> pd.DataFrame:
    """How many pairs are two unrelated tracks that happened to share a trigger.

    THE MIXED SAMPLE IS A SHAPE, NOT A NORMALISATION.  It is built pair for
    pair with the data, so subtracting all of it subtracts the signal too --
    which is exactly what it did the first time this ran, leaving every model
    with the same chi2 because every prediction had been scaled to zero.

    The normalisation is a rate calculation instead, and it needs nothing but
    the single-track rates already measured:

        N_acc(i,j) = N_trig * p_i * p_j          i != j
        N_acc(i,i) = N_trig * p_i^2 / 2          same chamber, unordered

    where p_i is the per-trigger probability of one selected track in arm i.
    That is a prediction with no free parameter, and comparing it to the
    observed pair count per arm combination is itself the test of whether
    anything correlated is present at all.
    """
    from sept26_prelim_analysis import source_imaging as SI
    t = SI._track_table(run, subruns, dca_max)
    ntrig = n_triggers(run, subruns)
    p = {a: t[t.arm == a].key.nunique() / ntrig for a in t.arm.unique()}
    rows = []
    for (a1, a2), g in real.groupby(['arm1', 'arm2']):
        exp = (ntrig * p.get(a1, 0) * p.get(a2, 0)
               * (0.5 if a1 == a2 else 1.0))
        rows.append(dict(arm1=a1, arm2=a2, topology=AC.topology(a1, a2),
                         n_obs=int(len(g)), n_acc=float(exp),
                         p1=p.get(a1, 0.0), p2=p.get(a2, 0.0),
                         n_trig=ntrig,
                         acc_frac=float(min(exp / max(len(g), 1), 1.0)),
                         excess=float(len(g) - exp),
                         sigma=float((len(g) - exp) / np.sqrt(max(exp, 1.0)))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the fold
# --------------------------------------------------------------------------- #
def fold(acc: pd.DataFrame, group: str, shape: np.ndarray,
         shape_theta: np.ndarray, bins=BINS) -> np.ndarray:
    """physics x acceptance, rebinned onto ``bins`` and normalised to 1.

    The acceptance is a probability per angle, so the two multiply pointwise on
    the FINE physics grid and are integrated into the coarse bins afterwards.
    Multiplying binned quantities instead would be wrong wherever either factor
    varies inside a bin, and the acceptance varies by an order of magnitude
    across 15 degrees near the edges.
    """
    g = acc[acc.group == group].sort_values('theta')
    if g.empty:
        return np.full(len(bins) - 1, np.nan)
    a = np.interp(shape_theta, g.theta.to_numpy(), g.acc.to_numpy(),
                  left=0.0, right=0.0)
    w = shape * a
    out = np.zeros(len(bins) - 1)
    idx = np.digitize(shape_theta, bins) - 1
    for i in range(len(out)):
        out[i] = w[idx == i].sum()
    s = out.sum()
    return out / s if s > 0 else out


def expectations(acc: pd.DataFrame, n: int = 600_000,
                 bins=BINS) -> pd.DataFrame:
    """Folded expectation per (topology, physics variant), shape-normalised."""
    edges = np.arange(0.0, 180.01, 0.5)
    fine = 0.5 * (edges[:-1] + edges[1:])
    src = {'X17': PP.x17_angles(n)}
    for name, kw in PP.VARIANTS.items():
        src[f'IPC · {name}'] = PP.ipc_angles(n, **kw)
    rows = []
    for topo in ('intra', 'perpendicular', 'opposing', 'all'):
        for name, ang in src.items():
            h, _ = np.histogram(ang, bins=edges, density=True)
            y = fold(acc, topo, h, fine, bins)
            for i, v in enumerate(y):
                rows.append(dict(topology=topo, model=name,
                                 theta=0.5 * (bins[i] + bins[i + 1]),
                                 frac=v))
    return pd.DataFrame(rows)


def compare(real: pd.DataFrame, mixed: pd.DataFrame, E: pd.DataFrame,
            ACC: pd.DataFrame, bins=BINS) -> pd.DataFrame:
    """Per topology: the observed shape, and how each model fits it.

    NO BACKGROUND IS SUBTRACTED, and that is a decision with a measurement
    behind it.  Two candidate normalisations for the accidental component were
    tried and both fail:

      * the event-mixed sample is built pair for pair with the data, so
        subtracting all of it subtracts the signal as well -- it is a SHAPE
        template and carries no rate;
      * the Poisson rate ``N_trig * p_i * p_j`` **over-predicts the observed
        pair count by a factor 2-4 in every arm combination**
        (:func:`accidentals`), because the trigger is a scintillator
        coincidence in one arm and that correlates the arms rather than leaving
        them independent.  The funnel report reached the same conclusion for
        the two-chamber rate and solved it with a control chamber instead.

    So the comparison is against the raw observed shape, the mixed shape is
    reported alongside as what accidentals would LOOK like, and the missing
    normalisation is stated as the leading systematic instead of being guessed.
    """
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    for topo in ('intra', 'perpendicular', 'opposing', 'all'):
        r = real if topo == 'all' else real[real.topo == topo]
        x = mixed if topo == 'all' else mixed[mixed.topo == topo]
        if len(r) < 10:
            continue
        obs, _ = np.histogram(r.open_deg, bins=bins)
        mshape, _ = np.histogram(x.open_deg, bins=bins)
        err = np.sqrt(np.clip(obs, 1, None))
        a = ACC if topo == 'all' else ACC[ACC.topology == topo]
        # The event-mixed shape enters the SAME comparison as the physics
        # models, on the same footing.  If the data looks more like two
        # unrelated tracks than like any pair spectrum, that has to be visible
        # in the same table and not left as a remark.
        cand = [(n, g.sort_values('theta').frac.to_numpy())
                for n, g in E[E.topology == topo].groupby('model')]
        if mshape.sum() > 0:
            cand.append(('event-mixed (accidental shape)',
                         mshape / mshape.sum()))
        for name, p in cand:
            if not np.isfinite(p).any() or p.sum() <= 0:
                continue
            pred = p * obs.sum()
            live = obs + pred > 0
            chi2 = float(np.sum(((obs[live] - pred[live]) / err[live]) ** 2))
            dof = int(live.sum() - 1)
            rows.append(dict(
                topology=topo, model=name, n_obs=int(obs.sum()),
                n_acc_poisson=float(a.n_acc.sum()),
                acc_over_obs=float(a.n_acc.sum() / max(obs.sum(), 1)),
                chi2=chi2, dof=dof, chi2dof=chi2 / max(dof, 1),
                frac_above_x17_obs=float(obs[mid > X17_MIN_DEG].sum()
                                         / max(obs.sum(), 1)),
                frac_above_x17_pred=float(p[mid > X17_MIN_DEG].sum()),
                frac_above_x17_mixed=float(mshape[mid > X17_MIN_DEG].sum()
                                           / max(mshape.sum(), 1))))
    return pd.DataFrame(rows)


def ratio_test(real: pd.DataFrame, E: pd.DataFrame,
               bins=BINS) -> pd.DataFrame:
    """opposing / intra, in data and in each model -- the model-light test.

    The acceptance normalisation, the vertex model and the efficiency scale are
    largely common to the two topologies and divide out; what does not divide
    out is the shape of the physics, which is what is being tested.  Quoted as
    the fraction of each topology's pairs above the X17 threshold, because that
    is the ratio the signal actually moves.
    """
    rows = []
    mid = 0.5 * (bins[:-1] + bins[1:])
    for topo in ('intra', 'perpendicular', 'opposing'):
        r = real[real.topo == topo]
        if len(r) < 10:
            continue
        k = int((r.open_deg > X17_MIN_DEG).sum())
        n = int(len(r))
        f = k / n
        rows.append(dict(topology=topo, n=n, k_above=k, frac_obs=f,
                         err=float(np.sqrt(max(k, 1)) / n),
                         **{f'frac_{name}': float(
                             g.sort_values('theta').frac.to_numpy()[
                                 mid > X17_MIN_DEG].sum())
                            for name, g in E[E.topology == topo].groupby('model')}))
    return pd.DataFrame(rows)


def projection(C: pd.DataFrame, run_scale: float = 50.0) -> pd.DataFrame:
    """How many pairs the shape comparison needs to separate the models.

    The separation between two shapes p and q over N pairs is
    N * sum((p-q)^2 / (p+q)) in the Gaussian limit, so N for a given
    significance follows directly.  Quoted for the pair that matters: the
    IPC-only expectation against IPC plus enough X17 to double the fraction
    above threshold.
    """
    rows = []
    for topo, g in C.groupby('topology'):
        base = g[g.model.str.startswith('IPC')]
        if base.empty:
            continue
        lo = float(base.frac_above_x17_pred.min())
        hi = float(base.frac_above_x17_pred.max())
        n = int(base.n_obs.iloc[0])
        # to see an X17 excess of size `d` on a background fraction `lo`
        # at nsig: n >= nsig^2 * lo (1-lo) / d^2
        rows.append(dict(topology=topo, n_now=n,
                         band_lo=lo, band_hi=hi,
                         band_width=hi - lo,
                         n_for_3sigma_10pct=float(
                             9.0 * lo * (1 - lo) / max(0.10 * lo, 1e-6) ** 2),
                         campaign_n=n * run_scale))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--dca', type=float, default=30.0)
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    od = paths.out('angle')
    acc = pd.read_csv(paths.require(od / f'acceptance_{a.run}.csv',
                                    'the acceptance -- run acceptance.py'))
    real, mixed = measured(a.run, subs, a.dca)
    E = expectations(acc)
    ACC = accidentals(a.run, subs, real, a.dca)
    C = compare(real, mixed, E, ACC)
    R = ratio_test(real, E)
    P = projection(C)

    real.to_parquet(od / f'pairs_{a.run}.parquet', index=False)
    mixed.to_parquet(od / f'pairs_mixed_{a.run}.parquet', index=False)
    E.to_csv(od / f'expected_{a.run}.csv', index=False)
    C.to_csv(od / f'compare_{a.run}.csv', index=False)
    ACC.to_csv(od / f'accidentals_{a.run}.csv', index=False)
    R.to_csv(od / f'ratio_{a.run}.csv', index=False)
    P.to_csv(od / f'angle_projection_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs, dca_max=a.dca,
                   bins=BINS.tolist(), x17_min_deg=X17_MIN_DEG,
                   n_pairs=int(len(real)),
                   by_topology=real.topo.value_counts().to_dict()),
              open(od / f'angle_{a.run}.meta.json', 'w'), indent=1)

    print('MEASURED pairs by topology')
    print(real.topo.value_counts().to_string())
    print('\nBY ARM PAIR')
    print(real.groupby(['arm1', 'arm2']).size().to_string())
    print('\nACCIDENTALS -- predicted from the single-track rates, no free '
          'parameter')
    print(ACC.to_string(index=False))
    print('\nSHAPE COMPARISON (no subtraction -- see the docstring)')
    print(C[['topology', 'model', 'n_obs', 'chi2dof', 'frac_above_x17_obs',
             'frac_above_x17_pred', 'frac_above_x17_mixed']].to_string(index=False))
    print('\nTHE MODEL-LIGHT TEST: fraction above 109 deg, per topology')
    print(R.to_string(index=False))
    print('\nWHAT THE SHAPE COMPARISON NEEDS')
    print(P.to_string(index=False))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
