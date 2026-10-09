import json, pickle, sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack, GRID, OFFS
from bench_joint import predict
J = json.load(open(f'{HERE}/results/bench_joint.json'))
names = sys.argv[1:] or ['det3', 'det4']
fig, axs = plt.subplots(len(names), 2, figsize=(11, 3.4 * len(names)), squeeze=False)
for r, name in enumerate(names):
    ev = pickle.load(open(SOURCES[name], 'rb')); q = J[name]
    for c, p in enumerate('xy'):
        M = stack(ev, p)[1]; ok = np.all(np.isfinite(M), axis=0)
        S = np.nan_to_num(M).sum(0); S[~ok] = 0
        P = predict(S, q['sig0'], q['Dd'], q['Drc'] if p == 'y' else 0, q['T'], q[f'dt_{p}'])
        ax = axs[r, c]
        for k, o in enumerate(OFFS):
            if o < 0 or o > 3: continue
            col = f'C{o}'
            ax.plot(GRID[ok], M[k, ok] if o == 0 else 0.5 * (M[k, ok] + M[list(OFFS).index(-o), ok]), col, lw=1.5, label=f'data ±{o}')
            ax.plot(GRID[ok], P[k, ok] if o == 0 else 0.5 * (P[k, ok] + P[list(OFFS).index(-o), ok]), col, ls='--', lw=1)
        ax.set_title(f'{name} {p.upper()}  (dashed = joint model)'); ax.axhline(0, c='0.7', lw=.5)
        ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{HERE}/figures/joint_{"_".join(names)}.png', dpi=110)
print('ok')
