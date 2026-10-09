#!/usr/bin/env python3
"""make_rc_arms.py -- arm JSONs for plane_bench.py: production hypers + the
physical kernel with every constant taken from the head-on measurement
(results/bench_uniform.json, Dd free), no chi2 refit.

  rcm   sigma_p0 = sigma_0, Dp = sqrt(2 Dd), rc_D_y = D_rc, rc_D_x = 0, X template
  rcmd  same, Dp from dry Magboltz at the bundle v (the gas-only prediction)
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench_attach import BUNDLES, BUNDLE
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/dylan/x17/cosmic_bench/cloud_basics/arms'
U = json.load(open(f'{HERE}/results/bench_uniform.json'))
P = json.load(open(f'{HERE}/results/bench_pinned.json'))
for det, b in BUNDLE.items():
    base = json.load(open(f'{BUNDLES}/{b}/bundle.json'))['hyper']
    base = {k: float(v) for k, v in base.items() if k != 'kTauY'}
    f = U[det]['Dd_free']
    for tag, Dd in (('rcm', f['Dd']), ('rcmd', P[det]['dry']['Dd'])):
        h = dict(base, sigma_p0=f['sig0'], Dp=float(np.sqrt(2 * Dd)), rc_D_y=f['Drc'],
                 rc_D_x=0.0, rc_tmpl=1.0)
        name = f'{det}_{tag}'
        json.dump(dict(arm=name, hyper=h, model_frac=0.0,
                       source='cloud_basics head-on measurement, no refit'),
                  open(f'{OUT}/arm_{name}.json', 'w'), indent=1)
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in h.items()
                     if k in ('sigma_p0', 'Dp', 'rc_D_y', 'rc_tmpl')})
