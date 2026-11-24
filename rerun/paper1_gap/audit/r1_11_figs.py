import pickle, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'figure.dpi': 110})
O = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/r1'
P1 = pickle.load(open(f'{O}/part1_maps.pkl', 'rb')); J1 = json.load(open(f'{O}/part1.json'))
F = pickle.load(open(f'{O}/fields.pkl', 'rb'))
GU, GW = P1['GU'], P1['GW']
ext = [GU[0] * 100 + 200, GU[-1] * 100 + 200, GW[0] * 100 + 200, GW[-1] * 100 + 200]
# fig 1
names = ['T_run1', 'T_run2', 'q_run1', 'q_run2', 'q_r625']
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2))
Z = {}
for n in names:
    z = P1['CEN'][n]; z = z - z.mean(); Z[n] = z * 28.2 / (1 + F[n]['beta'])
for a, n in zip(ax.ravel(), names):
    im = a.imshow(Z[n].T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-1.8, vmax=1.8)
    a.set_title(f"{n}  (beta {F[n]['beta']})", fontsize=9); a.set_xlabel('x (mm)'); a.set_ylabel('y (mm)')
zg = 0.5 * (Z['q_run1'] + Z['q_run2'])
d = Z['q_r625'] - zg
ax[1, 2].imshow(d.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-1.8, vmax=1.8); ax[1, 2].set_title('6-25 minus golden q (mm)', fontsize=9)
ax[1, 2].set_xlabel('x (mm)')
fig.colorbar(im, ax=ax, label='delta g (mm)', shrink=0.8)
fig.savefig(f'{O}/fig1_gap_maps.png', bbox_inches='tight'); plt.close(fig)
# fig 2: predicted vs measured tilt maps, n = 8
R = pickle.load(open(f'{O}/resid.pkl', 'rb'))
def panel(Rd, labels, fn, title):
    fig, ax = plt.subplots(len(Rd), 3, figsize=(9.5, 3.0 * len(Rd)))
    for r, (key, lab) in enumerate(zip(Rd, labels)):
        d = key
        M = d['M']; ext2 = [(M['Uc'].min() - 0.2125) * 100 + 200, (M['Uc'].max() + 0.2125) * 100 + 200, (M['Wc'].min() - 0.19) * 100 + 200, (M['Wc'].max() + 0.19) * 100 + 200]
        for c, (k, t) in enumerate((('A', 'measured'), ('P', 'predicted'), ('res', 'measured - predicted'))):
            im = ax[r, c].imshow(d[k].T * 1e3, origin='lower', extent=ext2, cmap='RdBu_r', vmin=-25, vmax=25)
            ax[r, c].set_title(f'{lab}: {t}', fontsize=8); ax[r, c].set_xlabel('x (mm)')
        ax[r, 0].set_ylabel('y (mm)')
    fig.colorbar(im, ax=ax, label='tilt (mrad, constant removed)', shrink=0.8); fig.suptitle(title, fontsize=10)
    fig.savefig(fn, bbox_inches='tight'); plt.close(fig)
panel([R['T:perview'][('run2', 'x')], R['T:perview'][('run2', 'y')]], ['golden run 2, X', 'golden run 2, Y'], f'{O}/fig2_tilt_golden.png',
      'Golden mounting: tilt of run 2 against the prediction from the T gap field of run 1 (per-view mesh, b = 0.5)')
panel([R['r625:perview']['x'], R['r625:perview']['y']], ['6-25, X', '6-25, Y'], f'{O}/fig3_tilt_625.png',
      '6-25 row: tilt against the prediction from its own q_u90 gap field (golden mesh, b = 0.5, beta 1.28)')
# fig 4: scatter of fine parts
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
for a, (lab, items) in zip(ax, (('golden, T field (cross-run), X', [R['T:perview'][('run1', 'x')], R['T:perview'][('run2', 'x')]]),
                                ('golden, T field (cross-run), Y', [R['T:perview'][('run1', 'y')], R['T:perview'][('run2', 'y')]]),
                                ('6-25, q_u90 field, X (o) and Y (+)', [R['r625:perview']['x'], R['r625:perview']['y']]))):
    for k, d in enumerate(items):
        a.plot(d['Pf'].ravel() * 1e3, d['Af'].ravel() * 1e3, 'o' if k == 0 else '+', ms=4)
    a.axline((0, 0), slope=1, color='k', lw=0.8); a.set_title(lab, fontsize=9)
    a.set_xlabel('predicted fine tilt (mrad)'); a.set_ylabel('measured fine tilt (mrad)')
fig.savefig(f'{O}/fig4_fine_scatter.png', bbox_inches='tight'); plt.close(fig)
# fig 5: s68
G = json.load(open(f'{O}/part3_golden.json')); S = json.load(open(f'{O}/part3_625.json'))
fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.2))
sel = [('pass1', 'pass1'), ('const', 'const'), ('mesh', 'mesh'), ('mesh+gap other (T)', 'mesh+gap (other T)'), ('mesh+gap same (T)', 'mesh+gap (same T)'),
       ('mesh+gap + v_rel, other (T)', '+v_rel (other T)'), ('direct 4x4 w0 (other run)', 'direct 4x4 w0'), ('direct 8x8 w0 (other run)', 'direct 8x8 w0'), ('direct 8x8 w0+v (other run)', 'direct 8x8 w0+v')]
for a, key in zip(ax[:2], ('run1', 'run2')):
    for k, p in enumerate('xy'):
        vals = [G[f'{key}:{p}']['res'][s]['s68'] for s, _ in sel]
        a.bar(np.arange(len(sel)) + 0.4 * k, vals, 0.4, label=p.upper())
    a.set_xticks(np.arange(len(sel)) + 0.2); a.set_xticklabels([l for _, l in sel], rotation=70, ha='right', fontsize=7); a.set_ylim(0.55, 1.1)
    a.set_title(f'golden {key}'); a.set_ylabel('s68 (deg)'); a.legend()
sel2 = [('pass1', 'pass1'), ('const', 'const'), ('golden direct 4x4 w0+v', 'golden direct 4x4'), ('mesh', 'mesh'), ('mesh+gap own q_u90, nominal (b=0.5/(1+1.28) scale)', 'mesh+gap nominal'),
        ('mesh+gap own q_u90, free b (a=1)', 'mesh+gap free b'), ('mesh+gap + v_rel (beta 1.28)', '+v_rel'), ('own 2-fold direct 4x4 w0+v', 'own 2-fold 4x4'), ('own 2-fold direct 8x8 w0+v', 'own 2-fold 8x8')]
for k, p in enumerate('xy'):
    vals = [S[f'{p}:arms'][s]['s68'] for s, _ in sel2]
    ax[2].bar(np.arange(len(sel2)) + 0.4 * k, vals, 0.4, label=p.upper())
ax[2].set_xticks(np.arange(len(sel2)) + 0.2); ax[2].set_xticklabels([l for _, l in sel2], rotation=70, ha='right', fontsize=7); ax[2].set_ylim(0.55, 1.35); ax[2].set_title('6-25 row'); ax[2].legend()
fig.savefig(f'{O}/fig5_s68.png', bbox_inches='tight'); plt.close(fig)
print('ok')
