"""a1_fig.py -- signal rms against cell size (plane- and quadratic-subtracted), variance table.  -> fig_signal_vs_size.png, sigtab.json"""
import json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a1_lib import *  # noqa
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
maps = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))['maps']
tab = {}
for n in NS:
    for p in 'xy':
        for kind in ('plane', 'quad'):
            m1, m2 = maps[n]['run1'][p], maps[n]['run2'][p]
            R1 = residual(m1, 'tau', 'etau', kind)[0]; R2 = residual(m2, 'tau', 'etau', kind)[0]
            s = cross_stats(R1, m1['etau'], R2, m2['etau'])
            tab[f'{n}_{p}_{kind}'] = dict(cov=s['cov'], cov_err=s['cov_err'], sig=s['sig'], r=s['r'], r_err=s['r_err'], ncell=s['ncell'],
                                          size=LU / n, err_mean=0.5 * (s['err1'] + s['err2']), rms1=s['rms1'], rms2=s['rms2'])
json.dump(tab, open(f'{OUT}/sigtab.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
for p, c in (('x', 'C0'), ('y', 'C1')):
    for kind, ls in (('plane', '-'), ('quad', '--')):
        sz = np.array([tab[f'{n}_{p}_{kind}']['size'] for n in NS])
        sg = np.array([tab[f'{n}_{p}_{kind}']['sig'] for n in NS]); ce = np.array([tab[f'{n}_{p}_{kind}']['cov_err'] for n in NS])
        cv = np.array([tab[f'{n}_{p}_{kind}']['cov'] for n in NS])
        lo = np.sqrt(np.clip(cv - ce, 0, None)); hi = np.sqrt(np.clip(cv + ce, 0, None))
        ax[0].errorbar(sz, 1e3 * sg, [1e3 * (sg - lo), 1e3 * (hi - sg)], fmt='o' + ls, color=c, ms=4, label=f'tau_{p}, minus {kind}')
        nz = np.array([tab[f'{n}_{p}_{kind}']['err_mean'] for n in NS])
        if kind == 'plane':
            ax[0].plot(sz, 1e3 * nz, ':', color=c, label=f'tau_{p} mean cell error')
ax[0].set_xlabel('cell size (mm)'); ax[0].set_ylabel('run1 x run2 signal rms of tau (1e-3 tan)'); ax[0].invert_xaxis(); ax[0].legend(fontsize=7)
for p, c in (('x', 'C0'), ('y', 'C1')):
    sz = np.array([tab[f'{n}_{p}_plane']['size'] for n in NS]); rr = np.array([tab[f'{n}_{p}_plane']['r'] for n in NS]); re = np.array([tab[f'{n}_{p}_plane']['r_err'] for n in NS])
    ax[1].errorbar(sz, rr, re, fmt='o-', color=c, label=f'tau_{p} minus plane')
ax[1].set_xlabel('cell size (mm)'); ax[1].set_ylabel('Pearson r, run 1 vs run 2'); ax[1].invert_xaxis(); ax[1].set_ylim(0.8, 1.01); ax[1].legend(fontsize=8)
fig.savefig(f'{OUT}/fig_signal_vs_size.png'); plt.close(fig)
for p in 'xy':
    for kind in ('plane', 'quad'):
        prev = 0
        for n in NS:
            t = tab[f'{n}_{p}_{kind}']
            print(p, kind, n, f"{t['size']:.0f}mm ncell {t['ncell']} cov {t['cov']*1e6:.1f}+-{t['cov_err']*1e6:.1f} e-6 sig {t['sig']*1e3:.2f} ({np.degrees(np.arctan(t['sig'])):.2f} deg) r {t['r']:.3f}+-{t['r_err']:.3f} rmserr {t['err_mean']*1e3:.2f}")
