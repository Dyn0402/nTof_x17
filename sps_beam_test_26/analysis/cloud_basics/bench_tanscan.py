#!/usr/bin/env python3
"""bench_tanscan.py -- does track inclination leak into the bench drift diffusion?

A head-on cut |tan| < 0.03 lets the track walk up to 0.9 mm over the 30 mm gap, which a stack aligned
on the peak strip reads as width growing with depth -- i.e. as drift diffusion.  If that inflated the
bench Dd (FINDINGS §27: widths look dry while v(E) says ~0.5 % water), Dd would fall as the cut
tightens.  The bench_pinrc joint fit (all free) at |tan| < 0.01 ... 0.05.

    bench_tanscan.py det3,det4,det7 -> results/bench_tanscan_<det>.json
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bench_pinrc import fit
from bench_attach import template, GAP
from width_vs_time import SOURCES
from rc_diffusion import stack

for det in sys.argv[1].split(','):
    pin = json.load(open(f'{HERE}/results/bench_pinned.json'))[det]
    T = GAP / (pin['v'] * 1e-3)
    h, _ = template(det)
    ev = pickle.load(open(SOURCES[det], 'rb'))
    out = {}
    for tm in (0.01, 0.015, 0.02, 0.03, 0.05):
        st = {p: stack(ev, p, tan_max=tm) for p in ('x', 'y')}
        Ms = {p: st[p][1] for p in st}
        sels = {p: np.all(np.isfinite(M), axis=0) for p, M in Ms.items()}
        r = fit(h, T, Ms, sels)
        r['n'] = [st['x'][0], st['y'][0]]
        out[tm] = r
        print(f'{det} |tan|<{tm:.3f} n {st["x"][0]:4d}/{st["y"][0]:4d}: sig0 {r["sig0"]:.3f}±{r["sig0_err"]:.3f} '
              f'Dd {r["Dd"]:.2e} Drc {r["Drc"]:.2e} rms {r["rms"]:.4f}', flush=True)
    json.dump(out, open(f'{HERE}/results/bench_tanscan_{det}.json', 'w'), indent=1)
