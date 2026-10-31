#!/usr/bin/env python3
"""vtc_summary.py [results dir] -- print the per-row lines and blocks of vtc_measure.py's JSONs (CROSSCHECK_2026-10-31)."""
import glob
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'measure')
SHOW = ('lad', 'col', 'col_std', 'col_sat', 'col_head', 'stack', 'stack_in30')
for f in sorted(glob.glob(os.path.join(D, '*.json'))):
    d = json.load(open(f))
    sv = d.get('survey', {})
    print(f"\n== {d['key']}  {d['duration_h']:.2f} h  stack head-on unsat {d['stack']['n_unsat']} (in30 "
          f"{d['stack']['n_unsat_in30']})  survey dv {100 * (sv.get('dv') or 0):+.2f} ± {100 * (sv.get('dv_err') or 0):.2f} %")
    for n, v in d['blk']['lines'].items():
        print(f"  {n:15s} dv {100 * v['dv']:+6.2f} ± {100 * v['dv_err']:.2f} %   chi2 {v['chi2']:.1f}/{v['dof']}")
    for b in d['blk']['blocks']:
        print(f"   t {b['t']:.2f} gain {b['gain']:.3f} in30 {b['in30_share']:.3f} | " + ' '.join(
            f"{k} {b[k]['r']:.3f}±{b[k]['r_err']:.3f}({b[k].get('n', 0)})" for k in SHOW if k in b))
