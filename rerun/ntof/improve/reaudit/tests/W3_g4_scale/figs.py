#!/usr/bin/env python3
"""Figure: binned residual centre (measured - expected) against expected tan, data (mapped) vs Geant4 truth
(full, point source, no scattering), per arm.  Slope of each set = k - 1."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 10.5})
D = pd.read_csv('data_profiles.csv')
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for i, arm in enumerate('ACD'):
    a = ax[i]
    d = D[(D.arm == arm) & (D.col == 'shm')]
    a.plot(d.e, d.c, 'ks-', ms=4, lw=1, label='data, mapped')
    for lib, mk in (('nose', 'o'), ('ear2_L19p5', '^')):
        try:
            P = pd.read_csv(f'decompose_profiles_{lib}.csv')
        except FileNotFoundError:
            continue
        for case, col in (('full', 'C0'), ('nosc', 'C2')):
            p = P[(P.arm == arm) & (P.case == case)]
            a.plot(p.e, p.c, marker=mk, ls='none', color=col, ms=4, alpha=0.8,
                   label=f'G4 {lib} truth' + (' (no scattering)' if case == 'nosc' else ''))
    a.axhline(0, color='0.6', lw=0.6)
    a.set_title(f'chamber {arm}')
    a.set_xlabel('expected tan e')
ax[0].set_ylabel('residual centre (measured - e)')
ax[0].set_ylim(-0.08, 0.08)
ax[2].legend(fontsize=7.5, loc='lower left')
fig.tight_layout()
fig.savefig('fig_profiles.png', dpi=140)
