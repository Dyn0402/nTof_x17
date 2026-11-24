import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/c'
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
K = json.load(open(f'{OUT}/kfit.json')); R = json.load(open(f'{OUT}/res625.json')); Z = np.load(f'{OUT}/maps_imgs.npz')
# Fig 1: fine maps images golden vs 6-25 in-sample, n=4 and 6
for n in (4, 6):
    fig, ax = plt.subplots(2, 3, figsize=(9, 5.6), constrained_layout=True)
    for r, p in enumerate('xy'):
        g = Z[f'{p}{n}__gold__Afine'] * 1e3; o = Z[f'{p}{n}__r625__Afine'] * 1e3
        Uc, Wc = Z[f'{p}{n}__gold__Uc'], Z[f'{p}{n}__gold__Wc']
        vm = max(np.nanmax(abs(g)), np.nanmax(abs(o)))
        ext = [Uc.min() - (Uc[1, 0] - Uc[0, 0]) / 2, Uc.max() + (Uc[1, 0] - Uc[0, 0]) / 2, Wc.min() - (Wc[0, 1] - Wc[0, 0]) / 2, Wc.max() + (Wc[0, 1] - Wc[0, 0]) / 2]
        for c, (m, t) in enumerate(((g, 'golden (run 1+2)'), (o, '6-25 in-sample'), (o - g, '6-25 minus golden'))):
            im = ax[r, c].imshow(m.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='equal')
            ax[r, c].set_title(f'{p.upper()} view: {t}', fontsize=9); ax[r, c].set_xlabel('U (100 mm)'); ax[r, c].set_ylabel('W (100 mm)')
        fig.colorbar(im, ax=ax[r, :], shrink=0.8, label=r'$A_{fine}$ ($10^{-3}$ in tan)')
    fig.savefig(f'{OUT}/fig1_fine_maps_n{n}.png'); plt.close(fig)
# Fig 2: scatter of cell values
fig, ax = plt.subplots(1, 4, figsize=(11, 3.2), constrained_layout=True)
for a, (p, n) in zip(ax, [('x', 4), ('y', 4), ('x', 6), ('y', 6)]):
    g = Z[f'{p}{n}__gold__Afine'].ravel() * 1e3; o = Z[f'{p}{n}__r625__Afine'].ravel() * 1e3
    r1, r2 = Z[f'{p}{n}__run1__Afine'].ravel() * 1e3, Z[f'{p}{n}__run2__Afine'].ravel() * 1e3
    a.plot(r1, r2, '.', c='gray', label='run 1 vs run 2'); a.plot(g, o, 'o', c='C3', ms=4, label='golden vs 6-25')
    l = max(np.nanmax(abs(g)), np.nanmax(abs(o))) * 1.1; a.plot([-l, l], [-l, l], 'k:', lw=0.8)
    a.set_title(f'{p.upper()} view, {n}x{n} cells', fontsize=9); a.set_xlabel(r'$A_{fine}$ first map ($10^{-3}$)'); a.set_ylabel(r'$A_{fine}$ second map ($10^{-3}$)')
a = ax[0]; a.legend(fontsize=7)
fig.savefig(f'{OUT}/fig2_fine_scatter.png'); plt.close(fig)
# Fig 3: k values (ratio to control)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
for a, (q, t) in zip(ax, (('kp', r'$k_{plane}$'), ('kf', r'$k_{fine}$'), ('ks', r'$k_s$'))):
    for p, c, off in (('x', 'C0', -0.08), ('y', 'C1', 0.08)):
        ns = [4, 6, 8]
        kk = [K['k'][f'{n}_{p}'][q] for n in ns]; ee = [K['k'][f'{n}_{p}'][q + '_e'] for n in ns]
        cc = [0.5 * (K['ctrl'][f'{n}_{p}_run1->run2'][q] + K['ctrl'][f'{n}_{p}_run2->run1'][q]) for n in ns]
        a.errorbar(np.array(ns) + off, kk, ee, fmt='o', c=c, label=f'{p.upper()} 6-25 raw')
        a.plot(np.array(ns) + off, cc, 's', mfc='none', c=c, label=f'{p.upper()} control (golden to golden)')
        a.plot(np.array(ns) + off, np.array(kk) / np.array(cc), '^', c=c, label=f'{p.upper()} 6-25 / control')
    a.axhline(1, c='k', lw=0.8); a.axhline(1.63, c='gray', ls='--', lw=0.8)
    if q == 'ks': a.axhline(2.25, c='C2', ls=':', lw=0.8)
    a.set_title(t); a.set_xlabel('cells per side n'); a.set_xticks([4, 6, 8])
ax[0].legend(fontsize=6.5); ax[0].set_ylabel('amplitude k'); fig.suptitle('solid line 1 = geometric; dashed 1.63 = fixed rate; dotted (k_s) = beta ratio 2.25', fontsize=8)
fig.savefig(f'{OUT}/fig3_k_values.png'); plt.close(fig)
# Fig 4: s68
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for a, p in zip(ax, 'xy'):
    r = R[p]['r625']; labs = ['pass1', 'const']
    for n in (4, 6, 8): labs += [f'gold{n} w0+v', f'gold{n} w0 (fixed rate)', f'own2fold{n} w0+v']
    v = [r[l]['s68'] for l in labs]; e = [r[l]['dif_e'] for l in labs]
    col = ['gray' if l in ('pass1', 'const') else ('C3' if 'own' in l else ('C2' if 'fixed' in l else 'C0')) for l in labs]
    a.bar(range(len(labs)), v, yerr=e, color=col); a.set_xticks(range(len(labs))); a.set_xticklabels(labs, rotation=70, ha='right', fontsize=7)
    a.set_ylim(0.6, max(v) * 1.05); a.set_ylabel('s68 of theta - theta_ref (deg)'); a.set_title(f'{p.upper()} view, 6-25 row')
fig.savefig(f'{OUT}/fig4_s68.png'); plt.close(fig)
print('ok')
