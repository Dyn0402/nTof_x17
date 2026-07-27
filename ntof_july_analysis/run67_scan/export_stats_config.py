#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the run_67 STATISTICS run configuration to ~/daq/calibrations/mm/.

This is the "bank as many reconstructed tracks per spill as possible over the
WHOLE 1-76 ms gate" operating point. It is deliberately NOT the same thing as
`resist_hv_run67.json`, which recommends fine-scan ranges optimised for the
2-8 ms IPC signal window; see the ROLE note below and in the emitted JSON.

Sources of the numbers (all recomputed here, nothing hand-copied):
  * per-cell efficiency from stats.per_cell_stats on NON-OVERLAPPING dt windows.
    NB: do NOT pool the boxcar table (slide_curves.csv) for this — its windows
    overlap by design, so summing k/n would count each event ~W/step times and
    shrink the errors by ~sqrt(6).
  * drift is FIXED at 700 V (operator choice, 2026-07-26), so each detector's
    resist optimum is evaluated AT drift 700, not pooled over drift.
  * plastic threshold chosen on THROUGHPUT (tracks per spill), not per-trigger
    efficiency: a lower threshold buys more triggers, and the question is
    whether the extra ones are worth keeping.

Run: .venv/bin/python ntof_july_analysis/run67_scan/export_stats_config.py
     [--out <dir>]   default ~/daq/calibrations/mm
"""
import argparse
import json
import os
import sys
from datetime import date

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import stats as ST  # noqa: E402

DEFAULT_OUT = os.path.expanduser('~/daq/calibrations/mm')
OUT_NAME = 'statistics_run_config_run67.json'

DRIFT_V = 700               # operator choice 2026-07-26 (also raises drift speed)
MIP_TAG = {141: '1.41 MIP', 113: '1.13 MIP', 90: '0.90 MIP'}

# Non-overlapping, spanning the MEASURED acceptance (1-76 ms; the nominal gate
# claims 81 ms but run_67 stops accepting at ~76 — see slide.gate_edges).
WINDOWS = [(1.0, 6.0), (6.0, 12.0), (12.0, 30.0), (30.0, 76.0)]


def plateau(resists, p, e, best_i, nsig=1.0):
    """Resist values statistically indistinguishable from the best.

    The surfaces are flat near the top (per-cell error ~0.0017 on p~0.10), so
    quoting a single peak overstates what was measured. A point is kept when it
    is within `nsig` of the peak using the error on the DIFFERENCE of two
    independent binomials, sqrt(e_best^2 + e_i^2) — not the peak's own error.
    Using the peak error alone is ~sqrt(2) too strict and collapses every
    plateau to a single point, which reads as far more precision than the
    measurement supports.

    Points are collected by VALUE, not by walking outward from the peak: these
    profiles have flat tops with statistical dips in them (Det C reads
    525:0.1336, 530:0.1300, 535:0.1318 — 535 is closer to the peak than 530 is),
    and a contiguous walk halts at the dip and reports a spuriously narrow
    plateau. The returned span is [min, max] of everything compatible with the
    peak.
    """
    ok = [i for i in range(len(p))
          if (p[best_i] - p[i]) <= nsig * np.hypot(e[best_i], e[i])]
    return [int(resists[min(ok)]), int(resists[max(ok)])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    ev, _ = ST.load()
    st = ST.per_cell_stats(ev, WINDOWS)
    st = st[st.drift != 400]          # fragmentary block, 1 resist point
    st['k'] = st.k_pair

    # ---- plastic threshold, decided on throughput ----
    thr = {}
    for m in (141, 113, 90):
        s = st[st.mip == m]
        row = {}
        for det in 'ABCD':
            d = s[s.det == det]
            ncell = max(d.drop_duplicates(['drift', 'resist']).shape[0], 1)
            row[det] = {
                'eff_p_pair': round(float(d.k.sum() / d.n.sum()), 4),
                'triggers_per_cell': int(round(d.n.sum() / ncell)),
                'tracks_per_cell': int(round(d.k.sum() / ncell)),
            }
        thr[str(m)] = row
    best_mip = max((90, 113, 141),
                   key=lambda m: sum(thr[str(m)][d]['tracks_per_cell']
                                     for d in 'ABCD'))

    # ---- per-detector resist optimum AT the fixed drift ----
    dets = {}
    at = st[st.drift == DRIFT_V]
    for det in 'ABCD':
        g = (at[at.det == det].groupby('resist', as_index=False)
             .agg(k=('k', 'sum'), n=('n', 'sum')))
        g = g.sort_values('resist')
        r = g.resist.to_numpy()
        p, e = ST.binom_err(g.k.to_numpy(float), g.n.to_numpy(float))
        i = int(np.argmax(p))
        # how the optimum MOVES with time since flash — the headline caveat
        per_win = {}
        for lo, hi in WINDOWS:
            w = at[(at.det == det) & (at.win_lo == lo)]
            gw = w.groupby('resist', as_index=False).agg(k=('k', 'sum'),
                                                         n=('n', 'sum'))
            if gw.empty or gw.n.sum() == 0:
                continue
            pw = (gw.k / gw.n.replace(0, np.nan)).to_numpy()
            per_win[f'{lo:g}-{hi:g} ms'] = {
                'best_resist_V': int(gw.resist.to_numpy()[int(np.nanargmax(pw))]),
                'p_pair': [round(float(x), 4) for x in pw],
            }
        dets[det] = {
            'drift_hv_V': DRIFT_V,
            'resist_hv_V': int(r[i]),
            'resist_1sigma_plateau_V': plateau(r, p, e, i, nsig=1.0),
            # 2 sigma = the practical "set it anywhere in here" band. Det A
            # genuinely peaks (535 sits at exactly 1.0 sigma), so its 1-sigma
            # plateau is a single point; the 2-sigma band is the operational one.
            'resist_2sigma_band_V': plateau(r, p, e, i, nsig=2.0),
            'expected_p_pair': round(float(p[i]), 4),
            'expected_p_pair_err': round(float(e[i]), 4),
            'measured_profile_at_drift': {
                'resist_V': [int(x) for x in r],
                'p_pair': [round(float(x), 4) for x in p],
                'p_err': [round(float(x), 4) for x in e],
                'n': [int(x) for x in g.n.to_numpy()],
            },
            'resist_optimum_by_window': per_win,
        }

    doc = {
        'provenance': {
            'run': 'run_67 (2026-07-22/23)',
            'exported': str(date.today()),
            'source': 'nTof_x17/ntof_july_analysis/run67_scan/export_stats_config.py',
            'reco': ('ntof_tracking.reco on the 2026-07-24 re-decoded hits '
                     '(small-pulse reprocessing); cache re-reco\'d 2026-07-25/26'),
            'purpose': ('STATISTICS run: maximise reconstructed tracks per spill '
                        'over the whole 1-76 ms gate.'),
            'metric': ('P(3D x/y pair) per recorded trigger. Denominator = events '
                       'the detector was READ OUT for (post-flash blindness stays '
                       'in the denominator).'),
            'method': (f'stats.per_cell_stats on non-overlapping windows '
                       f'{[list(w) for w in WINDOWS]} ms; resist optimum evaluated '
                       f'at the fixed drift {DRIFT_V} V; plastic threshold chosen '
                       f'on tracks/spill, not per-trigger efficiency.'),
            'caveats': [
                'RELATIVE efficiency (single-arm events dominate; no absolute '
                'normalisation). Det A is the clean-M1 reference.',
                'The surfaces are FLAT near the optimum (per-cell error ~0.0016 '
                'on p~0.10): use the 1-sigma plateau, do not chase the peak.',
                'The resist optimum MOVES with time since flash — low gain early, '
                'high gain late (see resist_optimum_by_window). A single setting '
                'is a compromise; this file optimises the WHOLE gate.',
                'Efficiency only. run_67 carries no spark/stability information, '
                'so if a listed resist sits near a chamber\'s discharge threshold, '
                'that overrides this file.',
                'Det B works ONLY at drift 700 V (10-30x better than 500/600 at '
                'every resist) — treat as a hardware symptom, not a setting.',
                'Several optima sit at the EDGE of the scanned ladder '
                f'(threshold {MIP_TAG[best_mip]}, drift {DRIFT_V} V, Det D resist '
                '520 V); the true optimum may lie outside the scan.',
            ],
        },
        'role': {
            'this_file': ('STATISTICS run — most tracks per spill across the full '
                          'gate.'),
            'not_this_file': ('resist_hv_run67.json recommends per-detector fine '
                              'SCAN RANGES optimised for the 2-8 ms IPC signal '
                              'window, and is deliberately LOWER in resist. It was '
                              'also generated 2026-07-23 from the PRE-reprocessing '
                              'reco, so its absolute efficiencies are ~10x low; its '
                              'shapes still hold. Do not mix the two.'),
        },
        'plastic_threshold': {
            'recommended_mip': best_mip / 100.0,
            'tag': MIP_TAG[best_mip],
            'chosen_on': 'throughput (tracks per spill), pooled over the HV grid',
            'note': ('Lowest threshold on the run_67 ladder. Throughput rises '
                     'toward lower threshold on ALL four detectors; on A and D the '
                     'per-trigger efficiency rises too, so the extra triggers are '
                     'better than average, not junk. The optimum is at the ladder '
                     'edge — extend below 0.90 MIP in the next scan.'),
            'evidence_per_threshold': thr,
        },
        'drift_hv_V': {
            'value': DRIFT_V,
            'applies_to': ['A', 'B', 'C', 'D'],
            'chosen_by': 'operator 2026-07-26',
            'note': ('Uniform 700 V; also raises drift speed. Det A is flat in '
                     'drift (0.105-0.107 over 500/600/700), C prefers 600-700, D '
                     'prefers 700, and B only works at 700 — so 700 costs A '
                     'nothing and helps the rest. 700 V is the TOP of the scanned '
                     'range; the optimum may be higher.'),
        },
        'detectors': dets,
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, OUT_NAME)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2)
    print('wrote', path)
    print(f'  plastic threshold: {MIP_TAG[best_mip]}')
    for det in 'ABCD':
        d = dets[det]
        print(f'  Det {det}: drift {d["drift_hv_V"]} V, resist '
              f'{d["resist_hv_V"]} V '
              f'(1sig {d["resist_1sigma_plateau_V"][0]}-'
              f'{d["resist_1sigma_plateau_V"][1]}, '
              f'2sig {d["resist_2sigma_band_V"][0]}-'
              f'{d["resist_2sigma_band_V"][1]}), '
              f'P(3D pair) = {d["expected_p_pair"]:.4f} '
              f'± {d["expected_p_pair_err"]:.4f}')
    return doc


if __name__ == '__main__':
    main()
