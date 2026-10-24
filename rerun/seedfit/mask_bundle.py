#!/usr/bin/env python3
"""mask_bundle.py --src <bundle> --health health.json --out <bundle> [--gain]
Copy a bundle with strip_health.py's dead channels as the bundle's censored DEAD list, and (--gain) its relative
strip gains in gain_x/gain_y.  Everything else unchanged."""
import argparse, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True); ap.add_argument('--health', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gain', action='store_true')
a = ap.parse_args()
H = json.load(open(a.health))
shutil.rmtree(a.out, ignore_errors=True); shutil.copytree(a.src, a.out)
b = json.load(open(os.path.join(a.out, 'bundle.json')))
b['dead'] = {p: [int(c) for c in H[p]['dead']] for p in ('x', 'y')}
b.setdefault('provenance', {})
if isinstance(b['provenance'], dict):
    b['provenance']['strip_health'] = dict(source=os.path.abspath(a.health), gain=a.gain)
json.dump(b, open(os.path.join(a.out, 'bundle.json'), 'w'), indent=1)
if a.gain:
    z = dict(np.load(os.path.join(a.out, 'arrays.npz')))
    for p in ('x', 'y'):
        g = np.ones(512)
        for c, v in H[p]['gain'].items():
            g[int(c)] = v
        z[f'gain_{p}'] = g
    np.savez(os.path.join(a.out, 'arrays.npz'), **z)
print('wrote', a.out, {p: len(b['dead'][p]) for p in ('x', 'y')}, 'gain' if a.gain else '')
