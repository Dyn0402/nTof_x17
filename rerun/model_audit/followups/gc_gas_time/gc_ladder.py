#!/usr/bin/env python3
"""gc_ladder.py -- GC: the reconstruction's own angle scale per bench row (FA tilt_rows.json: ladder v = slope of the
fitted w against the M3 reference tan, both views averaged) against the row's calibration row (registry calib_key),
next to the column-duration offset of gc_bench.  eps_lad = v_lad(row) / v_lad(calib) - 1 is the tan-scale error the
row carries if it inherits its group's kw/v (tan_reco = (1 + eps_lad) tan_true).  Writes results/ladder_rows.json."""
import datetime as dt
import json
import os

import numpy as np

from gc_common import RES

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, '..', 'fa_gas_v_tilt', 'results', 'tilt_rows.json')))
B = {(r['run'], r['subrun'], r['det']): r for r in json.load(open(os.path.join(RES, 'bench_rows.json')))
     if not r.get('missing')}
rows = []
for t in T:
    b = B.get((t['run'], t['subrun'], 'mx17_' + t['det'][-1]))
    if not b:
        continue
    rows.append(dict(key=b['key'], det=t['det'], t_start=b['t_start'], t_end=b['t_end'], subrun=t['subrun'],
                     run=t['run'], drift_V=t['drift_V'], resist_V=t['resist_V'],
                     v_lad=0.5 * (t['ang_x']['v'] + t['ang_y']['v']), q=b['q_unsat'], calib_key=b['calib_key'],
                     refit=b['refit'], valid=bool(t['angle_valid']), eps_T_pct=b['eps_group_pct']))
lad = {r['key']: r['v_lad'] for r in rows}
for r in sorted(rows, key=lambda r: (r['det'], r['t_start'])):
    cal = lad.get(r['calib_key'])
    r['eps_lad_pct'] = (r['v_lad'] / cal - 1) * 100 if cal else np.nan
    print(r['det'], dt.datetime.fromtimestamp(r['t_start']).strftime('%m-%d %H:%M'),
          dt.datetime.fromtimestamp(r['t_end']).strftime('%H:%M'), r['subrun'][:34].ljust(34),
          f"drift={r['drift_V']:5.0f} vlad={r['v_lad']:6.2f} q={r['q']:7.0f}", r['refit'].ljust(6),
          'valid ' if r['valid'] else 'BROKEN', f"eps_lad={r['eps_lad_pct']:6.1f}% epsT={r['eps_T_pct']:6.1f}%")
json.dump(rows, open(os.path.join(RES, 'ladder_rows.json'), 'w'), indent=1)
