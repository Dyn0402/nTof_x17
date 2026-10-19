#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_hot_wildcards.py -- attach noisy_channels.py's hot-channel classification
to a beam calibration bundle, as the ``hot`` wildcard wft/model.py and
wft/seed.py now read (HANDOFF_D_NOISY_CHANNELS.md item 4: never seed on a
flagged channel, keep it in the fit down-weighted).

WHERE THE BASELINE BUNDLE COMES FROM.  The bundle run_145/D's frozen tracks
were actually built from lived on a condor worker
(``/pool/condor/dir_1937099/...`` -- see ``events_prelim.meta.json``'s
``calibration`` field) and no longer exists on disk anywhere. It does not need
to: ``ntof_tracking.wft_beam.make_bundle`` is a PURE function of the bench
source bundle (``calib_bundle_r06`` for the arm) and this run's own
``run_config.json`` -- no fit, no randomness -- so calling it here reproduces
the identical bundle byte-for-byte in every field this module does not touch.
Verified against the embedded meta.json 2026-09-08: c1, kY, tau_s, sigma_s,
sigma_p0, Dp, v_drift, sat_adc all match to full float precision.

**Does not overwrite the baseline.** Writes a new variant
(``calib_bundle_hotmasked`` by default) alongside ``calib_bundle_prelim`` --
per CLAUDE.md, a calibration bundle is per detector AND per run condition, and
silently mutating the one the frozen products already point at would make
those products' own provenance lie.

**Does not launch a reconstruction.** This only builds and saves the bundle.
Re-running run_145's reconstruction with it (to compare against the frozen
products, per HANDOFF_D_NOISY_CHANNELS.md Sec. 4 step 4) needs
``ntof_tracking.wft_beam reco`` on lxplus/condor -- a separate, explicit step.

    python -m sept26_prelim_analysis.apply_hot_wildcards --run run_145 --arm D
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

#: Hyper fields make_bundle's re-derivation must match exactly against the
#: frozen run's own record -- if any of these drift, it means wft_beam.py's
#: bench source table or V_DRIFT_PRIOR changed since the frozen reco ran, and
#: blindly re-deriving would silently confound "add hot wildcards" with
#: "also changed the calibration", defeating the point of the comparison.
_MUST_MATCH_HYPER = ('c1', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')


def frozen_reference(run: str, arm: str) -> dict | None:
    """v_drift/sat_adc/hyper this run's OWN frozen ``events_prelim.meta.json``
    actually used, read back from whichever sub-run has one -- the ground
    truth to re-derive against, not the current (possibly since-changed)
    defaults in wft_beam.BEAM_DETS/V_DRIFT_PRIOR."""
    base = paths.out('fullpass') / run
    if not base.is_dir():
        return None
    for sub in sorted(base.iterdir()):
        p = sub / f'mx17_{arm}' / 'events_prelim.meta.json'
        if p.exists():
            m = json.loads(p.read_text())
            b = m.get('bundle') or {}
            if b:
                return dict(sub_run=sub.name, v_drift=b.get('v_drift'),
                           sat_adc=b.get('sat_adc'), hyper=b.get('hyper', {}))
    return None


def hot_channels(run: str, arm: str) -> dict:
    """{'x': [...], 'y': [...]} channel numbers -- the ``hot`` class only
    (HANDOFF_D_NOISY_CHANNELS.md's ``noisy`` shape class is not wired in: it
    is threshold-fragile and, on D-x, empty at every setting tried -- see
    noisy_channels.py's module docstring)."""
    p = paths.out('noisy_channels') / f'noisy_channels_{run}.csv'
    df = pd.read_csv(paths.require(
        p, 'noisy_channels.py output -- run that first'))
    g = df[(df.arm == arm) & (df.cls == 'hot')]
    return {plane: sorted(int(c) for c in g.loc[g.plane == plane, 'channel'])
           for plane in ('x', 'y')}


def export_json(run: str, arms, out_path: str) -> dict:
    """``{'D': {'x': [...], 'y': [...]}, ...}`` for the given arms, written to
    ``out_path`` -- what ``ntof_tracking/condor/run_beam_job.py --hot`` reads
    on a condor worker (it has no ``sept26_prelim_analysis`` on its path, only
    ``wft``/``ntof_tracking``/``common``, so the classification has to cross
    as data, the same way ``--allow`` ships a stage-2 allowlist)."""
    doc = {arm: hot_channels(run, arm) for arm in arms}
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=1)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub-run', default='stat090_0000')
    ap.add_argument('--arm', default='D')
    ap.add_argument('--out', default=None,
                    help='bundle output dir (default: calib_bundle_hotmasked '
                         'next to calib_bundle_prelim)')
    ap.add_argument('--export-json', default=None,
                    help='instead of building a local bundle, write '
                         '{arm: {plane: [channels]}} for --arms (comma-'
                         'separated) to this path -- for shipping to condor '
                         'via run_beam_job.py --hot')
    ap.add_argument('--arms', default='D',
                    help='with --export-json: comma-separated arms to include')
    a = ap.parse_args()

    if a.export_json:
        doc = export_json(a.run, a.arms.split(','), a.export_json)
        for arm, h in doc.items():
            print(f'{arm}: x={len(h["x"])} y={len(h["y"])}')
        print(f'wrote -> {a.export_json}')
        return 0

    from ntof_tracking import wft_beam
    from wft.calib import CalibrationBundle

    hot = hot_channels(a.run, a.arm)
    n = sum(len(v) for v in hot.values())
    print(f'{a.arm}: {n} hot channels from noisy_channels.py '
         f'(x={len(hot["x"])}, y={len(hot["y"])})')

    baseline_path = os.path.join(wft_beam.ANALYSIS_BASE, a.run, a.sub_run,
                                 f'mx17_{a.arm}', 'calib_bundle_prelim')
    if os.path.exists(os.path.join(baseline_path, 'bundle.json')):
        print(f'baseline already on disk -> {baseline_path}')
        cal = CalibrationBundle.load(baseline_path)
    else:
        ref = frozen_reference(a.run, a.arm)
        if ref is None:
            raise SystemExit(
                f'no events_prelim.meta.json found for {a.run}/mx17_{a.arm} to '
                f'read the frozen v_drift from -- refusing to guess it from '
                f'wft_beam.V_DRIFT_PRIOR, which may have changed since the '
                f'frozen reco ran. Pass an explicit bundle via --out/manual load.')
        print(f'baseline not on disk (expected -- it lived on condor); '
             f're-deriving it from the bench source bundle + run_config.json, '
             f'v_drift={ref["v_drift"]} read back from the frozen '
             f'{ref["sub_run"]}/events_prelim.meta.json')
        wft_beam.make_bundle(a.arm, run=a.run, sub_run=a.sub_run,
                             v_drift=ref['v_drift'], out=baseline_path)
        cal = CalibrationBundle.load(baseline_path)
        mism = {k: (cal.hyper.get(k), ref['hyper'].get(k))
               for k in _MUST_MATCH_HYPER
               if abs(cal.hyper.get(k, float('nan')) - ref['hyper'].get(k, float('nan'))) > 1e-6}
        if mism:
            raise SystemExit(
                f're-derived bundle does NOT match the frozen run\'s own '
                f'hyper -- wft_beam.py\'s bench source table has moved since '
                f'{a.run} was reconstructed. Mismatches (re-derived, frozen): '
                f'{mism}. Fix the source table or load the real bundle '
                f'manually before trusting this comparison.')
        if abs(cal.sat_adc - (ref['sat_adc'] or cal.sat_adc)) > 1e-6:
            raise SystemExit(f'sat_adc mismatch: re-derived {cal.sat_adc} vs '
                             f'frozen {ref["sat_adc"]}')
        print('re-derived bundle verified against the frozen run\'s own '
             'record: hyper, v_drift and sat_adc all match.')

    if cal.dead:
        overlap = {p: sorted(set(hot[p]) & set(cal.dead.get(p, [])))
                  for p in ('x', 'y')}
        if any(overlap.values()):
            print(f'!! {sum(len(v) for v in overlap.values())} channels are '
                 f'BOTH dead and hot -- dead wins (censored), see prep_plane: '
                 f'{overlap}')

    cal.hot = hot
    cal.provenance = dict(cal.provenance)
    cal.provenance['hot_wildcards'] = dict(
        source='sept26_prelim_analysis/noisy_channels.py',
        run=a.run, n_hot_x=len(hot['x']), n_hot_y=len(hot['y']),
        note='hot channels only (occupancy-based); the shape-only "noisy" '
             'class is not included -- see noisy_channels.py docstring')

    out = a.out or os.path.join(wft_beam.ANALYSIS_BASE, a.run, a.sub_run,
                                f'mx17_{a.arm}', 'calib_bundle_hotmasked')
    cal.save(out, note=f'{baseline_path} + hot wildcards from noisy_channels.py')
    print(cal.summary())
    print(f'wrote -> {out}')
    print(f'\nbaseline for comparison (unchanged) -> {baseline_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
