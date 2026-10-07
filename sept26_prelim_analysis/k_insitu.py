#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_insitu.py -- per-run angle scale for a re-pass reconstructed with the in-situ
(cosmic-calibrated) bundles, written as k_arm-style JSONs into their OWN kcal
directory so the production calibration is never touched.

    k_eff(arm, run) = norm(arm) x k_band(arm, run) / k_band(arm, REF_RUN)

- ``norm``: what the in-situ bundle reads on in-beam cosmic muons (A-C line
  truth), gas-normalised to REF_RUN and pooled over the campaign:
  `ntof_cosmics/inbeam_through_goers.py pooled` -> pooled_norm.csv (sep < 10),
  mean of the x and y views (k is one scalar per arm: it also sets v/k).
- ``k_band(run) / k_band(REF_RUN)``: the gas drift between runs, from the
  production k_arm capsule band of each run.  Only the RATIO is used: the band
  depends on the source model (HANDOFF_TRACKING_2026-10-06.md §13), but the
  source does not change run to run, and in-beam muons per period confirm
  the ratio on C (1.08 +- .05 vs 1.09).
- Arms without an in-situ bundle (B, D) get no k here: their angles stay null
  in the re-pass table, not silently borrowed.
- Runs with no production k_arm band for an arm get no k for it (null angles).

    python -m sept26_prelim_analysis.k_insitu --version is2_v1 [--ref 145] [--sep 10]
    python -m sept26_prelim_analysis.campaign_tracks --fullpass <reco_is2_v1> \\
        --kcal <out>/kcal_is2_v1 --out <out>/stage3_is2_v1
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths

NORM_CSV = Path('/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/pooled_norm.csv')
INSITU_ARMS = ('A', 'C')


def production_bands() -> dict:
    """{run: {arm: band}} from the production k_arm JSONs."""
    out = {}
    for f in paths.out('kcal').glob('k_arm_run_*.json'):
        if not re.fullmatch(r'k_arm_run_\d+\.json', f.name):
            continue
        d = json.loads(f.read_text())
        out[d['run']] = {a: d['arms'].get(a, {}).get('per_estimator', {}).get('band')
                         for a in 'ABCD'}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--version', required=True, help='re-pass tag, e.g. is2_v1')
    ap.add_argument('--ref', type=int, default=145, help='run the muon norm is gas-normalised to')
    ap.add_argument('--sep', type=int, default=10, help='pooled_norm.csv sep cut to use')
    a = ap.parse_args()

    N = pd.read_csv(NORM_CSV)
    N = N[N.sep_max == a.sep]
    norm = {arm: float(N[N.arm == arm].norm.mean()) for arm in INSITU_ARMS}
    norm_err = {arm: float(np.sqrt((N[N.arm == arm].norm_err ** 2).mean())) for arm in INSITU_ARMS}
    view_spread = {arm: float(N[N.arm == arm].norm.max() - N[N.arm == arm].norm.min())
                   for arm in INSITU_ARMS}
    bands = production_bands()
    ref = bands.get(f'run_{a.ref}')
    if not ref:
        raise SystemExit(f'no production k_arm for run_{a.ref}')

    out = paths.out(f'kcal_{a.version}')
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    n = 0
    for run, b in sorted(bands.items(), key=lambda kv: int(kv[0][4:])):
        apply, arms = {}, {}
        for arm in INSITU_ARMS:
            if b.get(arm) is None or ref.get(arm) is None or not np.isfinite(b[arm]):
                continue
            gas = float(b[arm]) / float(ref[arm])
            apply[arm] = norm[arm] * gas
            arms[arm] = dict(k=apply[arm], norm=norm[arm], norm_err=norm_err[arm],
                             norm_view_spread=view_spread[arm], gas_ratio=gas,
                             band_run=float(b[arm]), band_ref=float(ref[arm]))
        doc = dict(schema='sept26_prelim/k_insitu/1', run=run, version=a.version,
                   apply=apply, arms=arms, ref_run=f'run_{a.ref}', sep_max=a.sep,
                   formula='k = norm(arm) x band(run)/band(ref); tan = k * tan_raw (build_tracks)',
                   norm_source=str(NORM_CSV), built=stamp,
                   note='for tracks reconstructed with the in-situ bundles ONLY; '
                        'B and D have no in-situ bundle and get no k')
        (out / f'k_arm_{run}.json').write_text(json.dumps(doc, indent=1))
        n += 1
    print(f'norm (in-beam muons, sep<{a.sep}, mean x/y): '
          + ', '.join(f'{k} {v:.3f}±{norm_err[k]:.3f} (x-y spread {view_spread[k]:.3f})'
                      for k, v in norm.items()))
    print(f'wrote {n} k JSONs -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
