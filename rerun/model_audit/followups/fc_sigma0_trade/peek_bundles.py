#!/usr/bin/env python3
"""peek_bundles.py [root] -- print the width-law hypers, K, v, t0_abs of the five bench bundles under <root>/<det>."""
import json, sys

root = sys.argv[1] if len(sys.argv) > 1 else '/home/dylan/x17/cosmic_bench/model_audit/ws_t0/bundles'
for d in ['det2', 'det3', 'det4', 'det6', 'det7']:
    b = json.load(open(f'{root}/{d}/bundle.json')); h = b['hyper']
    print(d, {k: h[k] for k in h if k.startswith('sigma') or k in ('Dp', 'rc_D_y', 'rc_D_x', 'rc_tmpl')},
          'K', b.get('n_depth_bins'), 'v', b.get('v_drift'), 'pitch', b.get('pitch_mm'), 't0_abs', b.get('t0_abs'),
          'amap', bool(b.get('angle_map')))
