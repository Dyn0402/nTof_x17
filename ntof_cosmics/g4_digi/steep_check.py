#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steep_check.py -- pre-launch check 1 of the is2_v1 re-pass (HANDOFF_TRACKING
§14): does the reconstruction measure tracks above |tan| 0.6, and what does
the raw-tan plausibility cut (wft.reco.TAN_MAX) do to them?

Synthetic straight MIP tracks (run_digi.py muons --tan-u 0 1.1), digitised
into quiet run_145 overlay triggers and reconstructed by the production code
under the in-situ bundles: TAN_MAX 0.6 (as staged); 1.2 (the cut out of the
way); 1.2 with the fit's start scan W_SCAN_HALF doubled to 0.042 mm/ns (as
staged it spans |tan| 0.55 on A and 0.73 on C).  Same seed, same events.

Per arm, cut and |true tan_u| bin: the gated-track efficiency (an x/y pair
passed the gate), the x-fit efficiency, median raw/true (x view), MAD
resolution, wrong-sign fraction.

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/steep_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_cosmics.g4_digi import analyse as AN  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results' / 'repass_readiness'
EDGES = np.round(np.arange(0.0, 1.11, 0.1), 2)


#: (label suffix, TAN_MAX raw, W_SCAN_HALF mm/ns)
VARIANTS = (('tm0.6', 0.6, 0.021), ('tm1.2', 1.2, 0.021), ('tm1.2_ws042', 1.2, 0.042))


def table(arm: str, suffix: str, tm: float, ws: float) -> pd.DataFrame:
    R = AN.load(f'steep_{arm}_is2_{suffix}')
    t, r = R.tan_u.to_numpy(float), R.x_tan_theta.to_numpy(float)
    fit = (R.x_ok & R.x_quality_ok).to_numpy(bool) & np.isfinite(r)
    gated = (R.n_tracks.fillna(0) > 0).to_numpy(bool)
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = (np.abs(t) >= lo) & (np.abs(t) < hi)
        m = b & fit
        x = r[m] * np.sign(t[m])
        right = x > 0
        rows.append(dict(arm=arm, variant=suffix, tan_max=tm, w_scan_half=ws, lo=lo, hi=hi, n=int(b.sum()),
                         eff_gated=float(gated[b].mean()), eff_xfit=float(fit[b].mean()),
                         med_ratio=float(np.median(x / np.abs(t[m]))) if lo > 0 and m.sum() > 10 else np.nan,
                         sigma=float(1.4826 * np.median(np.abs((r[m] - t[m]) - np.median(r[m] - t[m]))))
                         if m.sum() > 10 else np.nan,
                         wrong_sign=float((x < 0).mean()) if m.sum() else np.nan,
                         ratio_right=float(np.median(x[right] / np.abs(t[m][right]))) if lo > 0 and right.sum() > 10 else np.nan,
                         ratio_wrong=float(np.median(-x[~right] / np.abs(t[m][~right]))) if lo > 0 and (~right).sum() > 10 else np.nan,
                         raw_over_0p6=float((np.abs(r[m]) >= 0.6).mean()) if m.sum() else np.nan))
    return pd.DataFrame(rows)


def main() -> int:
    T = pd.concat([table(a, *v) for a in 'AC' for v in VARIANTS
                   if (AN.OUT / f'steep_{a}_is2_{v[0]}.parquet').exists()], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / 'steep_muons.csv', index=False)
    with pd.option_context('display.width', 200):
        for a in 'AC':
            print(f'--- arm {a}')
            print(T[T.arm == a].pivot_table(index=['lo'], columns='variant',
                                            values=['eff_gated', 'wrong_sign', 'ratio_right', 'sigma']).round(3))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
