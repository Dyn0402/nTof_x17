#!/usr/bin/env python3
"""ga_figures.py -- GA toy figure: angle s68 per arm and angle (results/toy_<det>.json) -> figures/toy_s68.png,
and the quadrature excess over 'none' table -> results/toy_excess.md"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
ARMS = ['none', 'white', 'meas', 'meas_strip', 'meas_large', 'meas_known']
f, axs = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
md = ['| det | arm | ' + ' | '.join(f'{a}°' for a in (0, 3, 9, 16, 25)) + ' |', '|---|---|' + '---|' * 5]
for ax, det in zip(axs, DETS):
    R = json.load(open(os.path.join(HERE, 'results', f'toy_{det}.json')))['arms']
    for i, arm in enumerate(ARMS):
        key = arm if arm != 'white' else [k for k in R if k.startswith('white')][0]
        if key not in R:
            continue
        angs = [k for k in R[key] if k != 'all']
        x = np.array([float(a) for a in angs]) + (i - 2.5) * 0.25
        y = [R[key][a]['s68'] for a in angs]; e = [R[key][a]['s68_err'] for a in angs]
        ax.errorbar(x, y, e, fmt='o-', ms=4, label=key)
        if key != 'none':
            n = R['none']
            ex = []
            for a in angs:
                d2 = R[key][a]['s68'] ** 2 - n[a]['s68'] ** 2
                err = np.hypot(2 * R[key][a]['s68'] * R[key][a]['s68_err'], 2 * n[a]['s68'] * n[a]['s68_err'])
                ex.append(f"{np.sign(d2) * np.sqrt(abs(d2)):+.2f} (±{err / max(2 * np.sqrt(abs(d2)), 0.05):.2f})")
            md.append(f'| {det} | {key} | ' + ' | '.join(ex) + ' |')
    ax.set_title(det); ax.set_xlabel('track angle [deg]')
axs[0].set_ylabel('toy angle s68 [deg] (X view, production fit)'); axs[0].legend(fontsize=7)
f.tight_layout(); f.savefig(os.path.join(HERE, 'figures', 'toy_s68.png'), dpi=110)
open(os.path.join(HERE, 'results', 'toy_excess.md'), 'w').write('quadrature excess over none [deg] (signed; approx error)\n\n' + '\n'.join(md) + '\n')
print('\n'.join(md))
