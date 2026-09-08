#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_chain.py -- the drift velocity along the gas line, and the water it implies.

The four chambers share ONE gas line, daisy-chained

    supply -> A -> B -> C -> D -> exhaust

(`mx_july_beam_qa/DRIFT_WINDOW_HANDOFF.md` sec 0, confirmed twice in July).  Each
chamber outgasses into the gas that feeds the next one, so a contaminant that
comes from the detectors themselves gets *worse* down the chain.  That is a
prediction with a direction, and the in-situ drift velocities test it.

WHY IT IS GAS AND NOT FIELD.  In run_145 all four drift cathodes sit at the
same voltage -- read here from the run's own `hv_monitor.csv`, not assumed --
so E is the same in every chamber and the velocity ladder cannot be a field
effect.  :func:`drift_field` is what makes that a measurement in this module
rather than a sentence in a report.

WHAT IS INVERTED, AND WHAT THAT COSTS.  Magboltz v(E) curves for Ar/iso 90/10
with H2O, N2, O2 and air already exist at CERN pressure
(`garfield_sim/results/drift_9010_contam_cern.json`, 2026-07-20).  v falls
monotonically with contaminant fraction at fixed E, so it inverts.  Three
caveats travel with every number this module produces and are printed with
them:

  * the in-situ v is ``v_prior / k`` and k is the angle scale, whose focus
    objective is flat over ~+-20 % -- so the ORDERING along the chain is much
    better established than the absolute level;
  * v assumes a 30 mm effective gap.  A smaller effective gap shrinks every
    deficit and every implied fraction together;
  * chamber B has no field-shaping ring chain, so it has no velocity at all.
    It is a hole in the middle of the chain, and the figure must show it as a
    hole rather than omit it.

WHAT IS EXCLUDED, AND WHY IT IS FREE.  O2 and air barely move v (1 % O2 still
gives 41.6 um/ns at 233 V/cm) but attach at eta = 2-4 /cm, which would strip
the cathode-side charge to a per cent of the anode side.
`garfield_sim/attachment_run58.py` measured the opposite on real data.  So the
attachment column is not decoration: it is what rules out the alternatives.

    python -m sept26_prelim_analysis.gas_chain --run run_145
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

SCHEMA = 'sept26_prelim/gas_chain/1'

#: The order gas reaches the chambers.  Not alphabetical by accident -- the
#: line was plumbed A -> B -> C -> D and the analysis depends on the order.
CHAIN = ('A', 'B', 'C', 'D')

#: HV channel per chamber's DRIFT electrode, from run_config `hv_channels`.
DRIFT_CHAN = {'A': '9:0', 'B': '9:1', 'C': '9:2', 'D': '9:3'}
DRIFT_GAP_MM = 30.0

#: Magboltz suite: base mixture, and the contaminant ladders inside it.
CONTAM_JSON = 'garfield_sim/results/drift_9010_contam_cern.json'
BASE_KEY = 'Ar90_iso10'
LADDERS = {
    'H2O': [('Ar_iso10_H2O0.3', 0.3), ('Ar_iso10_H2O0.5', 0.5),
            ('Ar_iso10_H2O1.0', 1.0), ('Ar_iso10_H2O1.5', 1.5),
            ('Ar_iso10_H2O2.0', 2.0), ('Ar_iso10_H2O3.0', 3.0)],
    'N2':  [('Ar_iso10_N2_1', 1.0), ('Ar_iso10_N2_2', 2.0),
            ('Ar_iso10_N2_5', 5.0)],
    'O2':  [('Ar_iso10_O2_0.5', 0.5), ('Ar_iso10_O2_1.0', 1.0)],
    'air': [('Ar_iso10_air1', 1.0), ('Ar_iso10_air2', 2.0),
            ('Ar_iso10_air3', 3.0)],
}


# --------------------------------------------------------------------------- #
# the field -- measured from the run, so "same E everywhere" is not an
# assumption the rest of the module quietly leans on
# --------------------------------------------------------------------------- #
def drift_field(run: str, subruns) -> pd.DataFrame:
    """Per-chamber drift voltage and field, from the run's own HV monitor.

    One row per chamber: the set point, the mean monitored voltage, the mean
    monitored current, and E = V / gap.  A chamber whose channel is missing
    gets NaN rather than a guess.
    """
    frames = []
    for sub in subruns:
        p = os.path.join(str(paths.root('runs')), run, sub, 'hv_monitor.csv')
        frames.append(pd.read_csv(paths.require(p, f'HV monitor for {sub}')))
    hv = pd.concat(frames, ignore_index=True)

    rows = []
    for arm in CHAIN:
        ch = DRIFT_CHAN[arm]
        cols = {k: f'{ch} {k}' for k in ('v0', 'vmon', 'imon')}
        if not all(c in hv.columns for c in cols.values()):
            rows.append(dict(arm=arm, v0_V=np.nan, vmon_V=np.nan,
                             imon_uA=np.nan, E_Vcm=np.nan, n=0))
            continue
        rows.append(dict(
            arm=arm,
            v0_V=float(hv[cols['v0']].median()),
            vmon_V=float(hv[cols['vmon']].mean()),
            imon_uA=float(hv[cols['imon']].mean()),
            E_Vcm=float(hv[cols['vmon']].mean()) / (DRIFT_GAP_MM / 10.0),
            n=int(len(hv))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the Magboltz side
# --------------------------------------------------------------------------- #
def _curve(mix: list, E: float) -> tuple:
    """(v, eta) of one mixture at field ``E``, linearly interpolated."""
    e = np.array([p['E_Vcm'] for p in mix], float)
    v = np.array([p['v_um_per_ns'] for p in mix], float)
    a = np.array([p['eta_per_cm'] for p in mix], float)
    o = np.argsort(e)
    return float(np.interp(E, e[o], v[o])), float(np.interp(E, e[o], a[o]))


def ladder(E: float, path: str = None) -> pd.DataFrame:
    """v and eta at field ``E`` for the base mix and every contaminant point.

    Returned sorted by species then fraction, with the base mixture as the
    ``frac = 0`` row of every species -- which is what makes each species a
    curve that can be inverted rather than a scatter of labels.
    """
    p = paths.require(os.path.join(REPO, path or CONTAM_JSON),
                      'the Magboltz contamination suite')
    d = json.load(open(p))
    mixes = d['mixtures']
    v0, a0 = _curve(mixes[BASE_KEY], E)
    rows = []
    for species, points in LADDERS.items():
        rows.append(dict(species=species, frac_pct=0.0, key=BASE_KEY,
                         v_um_ns=v0, eta_per_cm=a0))
        for key, frac in points:
            v, a = _curve(mixes[key], E)
            rows.append(dict(species=species, frac_pct=frac, key=key,
                             v_um_ns=v, eta_per_cm=a))
    return pd.DataFrame(rows).sort_values(['species', 'frac_pct'],
                                          ignore_index=True)


def implied_fraction(v_measured: float, lad: pd.DataFrame,
                     species: str = 'H2O') -> float:
    """Contaminant fraction whose Magboltz v matches ``v_measured``.

    NaN -- not an extrapolation -- when the measured velocity is outside what
    the ladder covers.  A velocity *above* the pure-gas curve is not a small
    negative contamination, it is a sign something else is wrong, and this
    returns NaN so it shows up as a gap instead of as a number.
    """
    g = lad[lad.species == species].sort_values('v_um_ns')
    v, f = g.v_um_ns.to_numpy(), g.frac_pct.to_numpy()
    if not np.isfinite(v_measured) or v_measured > v.max() or v_measured < v.min():
        return float('nan')
    return float(np.interp(v_measured, v, f))


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def build(run: str, subruns, kcal: dict) -> dict:
    """Everything the figure and the report need, in one dict."""
    hv = drift_field(run, subruns)
    E = float(np.nanmean(hv.E_Vcm))
    same_E = bool(np.nanmax(hv.E_Vcm) - np.nanmin(hv.E_Vcm) < 1.0)
    lad = ladder(E)
    v_prior = float(kcal['v_bundle'])

    rows = []
    for i, arm in enumerate(CHAIN):
        a = kcal['arms'].get(arm, {})
        k = a.get('k')
        verdict = a.get('verdict', 'NO DATA')
        # An arm without a certified scale has no velocity.  B is the case
        # this exists for: it has a k, and that k is not a drift velocity,
        # because B has no drift field to have a velocity in.
        usable = verdict in ('CALIBRATED', 'PROVISIONAL') and k
        v = v_prior / k if usable else float('nan')
        pl = a.get('focus_plateau') or ([k, k] if k else [np.nan, np.nan])
        v_lo = v_prior / pl[1] if usable else float('nan')
        v_hi = v_prior / pl[0] if usable else float('nan')
        rows.append(dict(
            position=i + 1, arm=arm, verdict=verdict, k=k,
            v_um_ns=v, v_lo=v_lo, v_hi=v_hi,
            deficit_pct=100.0 * (v / v_prior - 1.0) if usable else float('nan'),
            h2o_pct=implied_fraction(v, lad, 'H2O'),
            h2o_pct_lo=implied_fraction(v_hi, lad, 'H2O'),
            h2o_pct_hi=implied_fraction(v_lo, lad, 'H2O'),
            n2_pct=implied_fraction(v, lad, 'N2'),
            E_Vcm=float(hv.loc[hv.arm == arm, 'E_Vcm'].iloc[0]),
            v0_V=float(hv.loc[hv.arm == arm, 'v0_V'].iloc[0]),
            imon_uA=float(hv.loc[hv.arm == arm, 'imon_uA'].iloc[0])))
    chain = pd.DataFrame(rows)

    m = chain[np.isfinite(chain.v_um_ns)]
    monotone = bool(len(m) > 1 and np.all(np.diff(m.v_um_ns.to_numpy()) < 0))

    return dict(schema=SCHEMA, run=run, subruns=list(subruns),
                E_Vcm=E, same_field=same_E, v_prior=v_prior,
                monotone_down_chain=monotone,
                chain=chain, ladder=lad, hv=hv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    kp = paths.require(paths.out('kcal') / f'k_arm_{a.run}.json',
                       'the angle-scale calibration -- run k_arm.py first')
    res = build(a.run, subs, json.load(open(kp)))

    od = paths.out('gaschain')
    res['chain'].to_csv(od / f'gas_chain_{a.run}.csv', index=False)
    res['ladder'].to_csv(od / f'magboltz_ladder_{a.run}.csv', index=False)
    res['hv'].to_csv(od / f'drift_field_{a.run}.csv', index=False)
    meta = {k: v for k, v in res.items()
            if k not in ('chain', 'ladder', 'hv')}
    json.dump(meta, open(od / f'gas_chain_{a.run}.meta.json', 'w'), indent=1)

    print(f'field: E = {res["E_Vcm"]:.1f} V/cm, '
          f'same on every chamber: {res["same_field"]}')
    print(res['hv'].to_string(index=False))
    print()
    print(res['chain'][['position', 'arm', 'verdict', 'v_um_ns',
                        'deficit_pct', 'h2o_pct', 'h2o_pct_lo', 'h2o_pct_hi',
                        'n2_pct']].to_string(index=False))
    print(f'\nmonotone down the chain: {res["monotone_down_chain"]}')
    print(f'wrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
