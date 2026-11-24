#!/usr/bin/env python
"""verify_bundle.py -- install the shipped block and row constants in the real calibration bundle of a row (saved and reloaded
through wft.calib.CalibrationBundle), run wft.reco.bow_pairs on the whole gate table, and compare with the x_tan_map / y_tan_map
columns of the final parquet.  One row per map unit.  -> $JOB/phimap/verify_bundle.json"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib as L  # noqa: E402
import rows as R  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402

OUT = R.OUTDIR
rc = json.load(open(f'{OUT}/row_constants.json'))
res = {}
for u in sorted(set(c['unit'] for c in rc.values() if c.get('unit'))):
    ks = [k for k, c in rc.items() if c.get('unit') == u and c.get('map_trained_here')]
    # a calibration row (has a calib_bundle directory of its own) with the most events
    ks = [k for k in ks if os.path.isdir(os.path.dirname(R.bundle_path(k))) and R.full().loc[k].calib_key == k] or ks
    k = max(ks, key=lambda k_: rc[k_]['n_gate'])
    c = rc[k]
    bdir = os.path.dirname(R.bundle_path(k))
    cal = CalibrationBundle.load(bdir)
    phi = json.load(open(f"{OUT}/{c['block']}"))
    cal.bow = dict(cal.bow or {}, phi=phi)
    cal.w0 = {'x': c['w0']['x'], 'y': c['w0']['y']}
    cal.kw = {'x': c['kw']['x'], 'y': c['kw']['y']}
    tmp = tempfile.mkdtemp(prefix='phimap_verify_')
    cal.save(tmp, note='verify_bundle')
    cal2 = CalibrationBundle.load(tmp)
    shutil.rmtree(tmp)
    t = R.load_full(k)
    d = pd.read_parquet(f'{OUT}/final/{k}.parquet')
    r = L.reco.bow_pairs(cal2, t.x_w.values, t.x_p0.values, t.y_w.values, t.y_p0.values)
    m = np.isfinite(r['tan_x']) & np.isfinite(d.x_tan_map.values)
    dx = np.abs(r['tan_x'] - d.x_tan_map.values)[m]
    dy = np.abs(r['tan_y'] - d.y_tan_map.values)[np.isfinite(r['tan_y']) & np.isfinite(d.y_tan_map.values)]
    res[u] = dict(row=k, n=int(m.sum()), max_abs_diff_x=float(dx.max()), max_abs_diff_y=float(dy.max()), block=c['block'], variant=c['map_variant'])
    print(u, k[-45:], res[u]['n'], f"{dx.max():.1e} {dy.max():.1e}", flush=True)
json.dump(res, open(f'{OUT}/verify_bundle.json', 'w'), indent=1)
