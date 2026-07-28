#!/usr/bin/env python3
"""
digest.py — waveform-first results against the hits-chain baseline.

Reads the wft outputs for one or more run keys and compares them with
``mx_june_cosmic_qa/rerun_baseline.json`` (the pre-rerun hits-chain numbers,
same events, same M3 recipe), in the layout of RERUN_RESULTS_20260725_011307.md.

Read the comparison with the basis difference in mind — some columns are
*expected* to move:

  * within 5 mm / core sigma / median |r| : should hold or improve. These are
    the gate.
  * sigma_theta                           : should improve a lot (the hits
                                            ladder is compressed).
  * v_drift                               : WILL move (34.3 -> 36.6 on det3);
                                            that is the correction, not a
                                            regression.
  * has_any / spark_frac                  : should not move at all — detection
                                            is still hits-defined.

    ../.venv/bin/python mx_june_wft/digest.py sat_det3 [g_det3_wknd ...]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths     # noqa: E402
setup_paths()

BASELINE = os.path.join(REPO, 'mx_june_cosmic_qa', 'rerun_baseline.json')

# metric -> (baseline key, wft source file, wft key path, format, direction)
#   direction: +1 higher is better, -1 lower is better, 0 informational
METRICS = [
    ('rays',            'n_rays',            'eff',  ('n_rays',),        '{:.0f}',  0),
    ('has_any %',       'has_any',           'eff',  ('has_any',),       '{:.1f}', +1),
    ('within 5 mm %',   'within5',        'eff',  ('within_R',),      '{:.1f}', +1),
    ('reco-at-all %',   'reco_at_all',       'eff',  ('reco_at_all',),   '{:.1f}', +1),
    ('reco_far %',      'reco_far',          'eff',  ('reco_far',),      '{:.1f}', -1),
    ('core sigma r mm', 'core_sigma_mm',     'eff',  ('core_sigma_mm',), '{:.2f}', -1),
    ('median r mm',     'median_r_mm',       'eff',  ('median_r_mm',),   '{:.2f}', -1),
    ('spark_frac %',    'spark_frac',        'eff',  ('spark_frac',),    '{:.1f}',  0),
    ('sigma_theta X',   'sigma_theta_x_deg', 'ang',  ('planes', 'x', 'sigma_deg'), '{:.2f}', -1),
    ('sigma_theta Y',   'sigma_theta_y_deg', 'ang',  ('planes', 'y', 'sigma_deg'), '{:.2f}', -1),
    ('bias X deg',      None,                'ang',  ('planes', 'x', 'bias_deg'),  '{:+.2f}', 0),
    ('bias Y deg',      None,                'ang',  ('planes', 'y', 'bias_deg'),  '{:+.2f}', 0),
    ('implied-v spread X', None,             'ang',  ('planes', 'x', 'implied_v_spread'), '{:.2f}', -1),
    ('implied-v spread Y', None,             'ang',  ('planes', 'y', 'implied_v_spread'), '{:.2f}', -1),
    ('v_drift um/ns',   'v_drift_um_ns',     'ang',  ('v_cal_um_ns',),   '{:.1f}',  0),
]

# gate thresholds for det3 (DET3_RECO_FIX_2026-07-25.md, same events)
GATE = {'within_R': ('>=', 93.0), 'core_sigma_mm': ('<=', 0.50),
        'median_r_mm': ('<=', 0.85), 'has_any': ('>=', 99.0)}


def _dig(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def load_wft(cfg):
    out = {}
    p = os.path.join(cfg.OUT_BASE, 'wft', 'efficiency', 'efficiency_breakdown.json')
    if os.path.exists(p):
        out['eff'] = json.load(open(p))
    p = os.path.join(cfg.OUT_BASE, 'wft', 'angles', 'angular_resolution.json')
    if os.path.exists(p):
        out['ang'] = json.load(open(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--out', default=None, help='write a markdown digest here')
    args = ap.parse_args()

    base = json.load(open(BASELINE))['keys']
    lines = ['| quantity | ' + ' | '.join(args.keys) + ' |',
             '|---|' + '---|' * len(args.keys)]
    gate_fail = []
    data = {}
    for key in args.keys:
        data[key] = load_wft(get_config(key))

    for label, bkey, src, path, fmt, direction in METRICS:
        cells = []
        for key in args.keys:
            new = _dig(data[key].get(src), path)
            old = base.get(key, {}).get(bkey) if bkey else None
            if new is None:
                cells.append('—')
                continue
            cell = fmt.format(new)
            if old is not None and np.isfinite(old):
                d = new - old
                cell += f'  (was {fmt.format(old)}'
                if direction:
                    better = (d > 0) == (direction > 0)
                    cell += ', better' if abs(d) > 1e-9 and better else (
                        ', worse' if abs(d) > 1e-9 else '')
                cell += ')'
            cells.append(cell)
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    for key in args.keys:
        eff = data[key].get('eff', {})
        for gk, (op, thr) in GATE.items():
            v = eff.get(gk)
            if v is None:
                continue
            ok = v >= thr if op == '>=' else v <= thr
            if not ok:
                gate_fail.append(f'{key}: {gk} = {v:.3f} fails {op} {thr}')

    txt = '\n'.join(lines)
    print(txt)
    print()
    if gate_fail:
        print('GATE FAILURES:')
        for g in gate_fail:
            print('  ' + g)
    else:
        print('GATE: all thresholds met')
    if args.out:
        with open(args.out, 'w') as f:
            f.write('# Waveform-first vs hits-chain digest\n\n' + txt + '\n\n')
            f.write('GATE: ' + ('FAILED\n- ' + '\n- '.join(gate_fail)
                                if gate_fail else 'all thresholds met') + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
