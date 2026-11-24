"""r4_09_figs: figures of R4."""
import sys, json, pickle
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import r4_lib as L
import r1_lib as R1
import r4_06_check_template as T6
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25})
S = json.load(open(f'{L.OUT}/summary.json')); A = json.load(open(f'{L.OUT}/analysis.json')); U = A['units']
rows = {r['label']: r for r in S['rows']}
dr = [r for r in S['rows'] if 'drift scan' in r['label']]
g1 = rows['6-27 long run (golden run 1)']; g2 = rows['6-28 p2 overnight (golden run 2)']; r625 = rows['6-25 day (bottom slot)']; r622 = rows['6-22 overnight long run']
hv = [r for r in S['rows'] if r['label'].startswith('6-27 hv_scan')]
bf = lambda V: np.interp(V, [int(k) for k in S['beta_eff']], list(S['beta_eff'].values()))
# ---------------- fig1
fig, ax = plt.subplots(2, 3, figsize=(11, 6.2)); ax = ax.ravel()
specs = [('guu', 3, 'guu (ln q per (100 mm)^2)'), ('gww', 4, 'gww (ln q per (100 mm)^2)'), ('guw', 5, 'guw (ln q per (100 mm)^2)')]
Vg = np.linspace(450, 1150, 50)
for a, (n, j, lab) in zip(ax[:3], specs):
    V = [r['drift'] for r in dr]; y = [r['c'][j] for r in dr]; e = [r['e'][j] for r in dr]
    a.errorbar(V, y, e, fmt='o', color='C0', label='6-27 drift scan (490 V mesh)')
    a.errorbar(1000, g1['c'][j], g1['e'][j], fmt='D', color='k', ms=7, label='golden run 1 (6-27 long run)')
    a.errorbar(1000, g2['c'][j], g2['e'][j], fmt='d', color='gray', ms=6, label='golden run 2 (6-28)')
    a.errorbar(500, r625['c'][j], r625['e'][j], fmt='*', color='C3', ms=12, label='6-25, bottom slot (500 V)')
    a.errorbar(1000, r622['c'][j], r622['e'][j], fmt='s', color='C2', ms=6, label='6-22 overnight (1000 V)')
    if n == 'gww':
        a.plot(Vg, g1['c'][j] * (1 + bf(Vg)) / (1 + bf(1000)), 'k--', lw=1, label='fixed gap shape (beta from the scan)')
        sl = (g1['c'][j] - r625['c'][j]) / .75
        a.plot(Vg, r625['c'][j] + sl * ((Vg / 1000) ** 2 - .25), 'C3:', lw=1.2, label='V^2 line, 6-25 to golden')
    a.set_xlabel('drift voltage (V)'); a.set_ylabel(lab); a.axhline(0, color='k', lw=.5)
ax[0].legend(fontsize=6.5, loc='lower left')
# mm
a = ax[3]
f = lambda r, V, j: (r['c'][j] * 28.2 / (1 + bf(V)), r['e'][j] * 28.2 / (1 + bf(V)))
for rr, mk, col, lab in ((dr, 'o', 'C0', 'drift scan'),):
    a.errorbar([r['drift'] for r in rr], [f(r, r['drift'], 4)[0] for r in rr], [f(r, r['drift'], 4)[1] for r in rr], fmt=mk, color=col, label='drift scan, beta(V) from the scan')
a.errorbar(1000, f(g1, 1000, 4)[0], f(g1, 1000, 4)[1], fmt='D', color='k', ms=7, label='golden run 1')
a.errorbar(500, r625['c'][4] * 28.2 / 2.28, r625['e'][4] * 28.2 / 2.28, fmt='*', color='C3', ms=12, label='6-25 (beta 1.28)')
a.set_xlabel('drift voltage (V)'); a.set_ylabel('gww of the gap (mm per (100 mm)^2)'); a.legend(fontsize=6.5); a.axhline(0, color='k', lw=.5)
# amp_fine
a = ax[4]
a.errorbar([r['drift'] for r in dr], [r['amp'][0] for r in dr], [r['amp'][1] for r in dr], fmt='o', color='C0')
a.errorbar(1000, g1['amp'][0], g1['amp'][1], fmt='D', color='k', ms=7); a.errorbar(1000, g2['amp'][0], g2['amp'][1], fmt='d', color='gray', ms=6)
a.errorbar(500, r625['amp'][0], r625['amp'][1], fmt='*', color='C3', ms=12); a.errorbar(1000, r622['amp'][0], r622['amp'][1], fmt='s', color='C2', ms=6)
a.plot(Vg, g1['amp'][0] * (1 + bf(Vg)) / (1 + bf(1000)), 'k--', lw=1)
a.set_xlabel('drift voltage (V)'); a.set_ylabel('amplitude of the golden fine map (ridge)'); a.axhline(0, color='k', lw=.5)
# q mean
a = ax[5]
a.plot([r['drift'] for r in dr], [r['mean_q'] for r in dr], 'o', color='C0'); a.plot(1000, g1['mean_q'], 'D', color='k'); a.plot(500, r625['mean_q'], '*', color='C3', ms=12)
a.plot(1000, r622['mean_q'], 's', color='C2'); a.axhline(1700, color='r', lw=1, ls='--'); a.text(320, 1650, 'window end', color='r', fontsize=7, va='top')
a.set_xlabel('drift voltage (V)'); a.set_ylabel('mean q_u90 (ns)')
fig.suptitle('Chamber A: shape of ln q_u90 against drift voltage (error bars: bootstrap)'); fig.tight_layout(); fig.savefig(f'{L.OUT}/fig1_coef_vs_drift.png', dpi=130)
# ---------------- fig2 maps (quadratic surfaces, percent of q_u90) and spline maps of long runs
Ug = np.linspace(-1.7, 1.68, 60); Wg = np.linspace(-1.5, 1.5, 54); UU, WW = np.meshgrid(Ug, Wg, indexing='ij')
def quad(r):
    c = r['c']; return 100 * (c[1] * UU + c[2] * WW + c[3] * UU ** 2 / 2 + c[4] * WW ** 2 / 2 + c[5] * UU * WW)
panels = [('6-27 1100 V', rows['6-27 drift scan 1100 V']), ('6-27 900 V', rows['6-27 drift scan 900 V']), ('6-27 700 V', rows['6-27 drift scan 700 V']),
          ('6-27 500 V', rows['6-27 drift scan 500 V']), ('6-27 300 V', rows['6-27 drift scan 300 V']), ('golden run 1, 1000 V', g1),
          ('6-25, 500 V', r625), ('6-22, 1000 V', r622)]
fig, ax = plt.subplots(2, 4, figsize=(12, 5.6))
for a, (t, r) in zip(ax.ravel(), panels):
    m = quad(r); m = m - m.mean()
    im = a.pcolormesh(200 + 100 * Ug, 200 + 100 * Wg, m.T, cmap='RdBu_r', vmin=-12, vmax=12, shading='auto')
    a.set_title(f'{t} (N={r["N"]})', fontsize=8); a.set_xlabel('x (mm)'); a.set_ylabel('y (mm)'); a.set_aspect('equal'); a.grid(False)
fig.colorbar(im, ax=ax, shrink=.8, label='quadratic part of ln q_u90 (percent)'); fig.savefig(f'{L.OUT}/fig2_maps_quadratic.png', dpi=130)
# spline maps
fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
sets = [('golden run 1', L.GOLD1), ('golden run 2', L.GOLD2), ('6-25 (500 V)', L.R625), ('6-22 (1000 V)', 'mx17_det2_det3_overnight_6-22-26/long_run'),
        ('6-27 700+900+1100 V', None)]
for a, (t, rel) in zip(ax, sets):
    if rel is None:
        import pandas as pd
        ds = []
        for V in (700, 900, 1100):
            d = L.load_events(f'{L.SC}/drift_scan_resist_490V_drift_{V}V'); f_ = L.fit_shape(d, nboot=1); A_ = L.design(d, ('walk', 'tan'))
            d = d.copy(); d['lq'] = d.lq.values - A_[:, 6:] @ f_['c'][6:9]; d['lq'] -= d.lq.mean(); ds.append(d)
        d = pd.concat(ds); spl = R1.Spl(6); B = spl.basis(d.U.values, d.W.values); co_, _ = spl.fit(d.U.values, d.W.values, d.lq.values, 0.1, B=B)
        m = (spl.basis(T6.UU.ravel(), T6.WW.ravel()) @ co_).reshape(T6.UU.shape)
        tt = t + f' (K=6, N={len(d)})'
    else:
        d = L.load_events(rel); m, _ = T6.spline_map(d); tt = f'{t} (N={len(d)})'
    m = 100 * (m - m.mean())
    im = a.pcolormesh(200 + 100 * T6.GRID_U, 200 + 100 * T6.GRID_W, m.T, cmap='RdBu_r', vmin=-12, vmax=12, shading='auto')
    a.set_title(tt, fontsize=8); a.set_xlabel('x (mm)'); a.set_aspect('equal'); a.grid(False)
ax[0].set_ylabel('y (mm)'); fig.colorbar(im, ax=ax, shrink=.9, label='ln q_u90 map (percent)'); fig.savefig(f'{L.OUT}/fig2b_maps_spline.png', dpi=130)
# ---------------- fig3 mesh scan
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
mv = [r['mesh'] for r in hv]
for a, (n, j) in zip(ax[:3], (('guu', 3), ('gww', 4), ('guw', 5))):
    a.errorbar(mv, [r['c'][j] for r in hv], [r['e'][j] for r in hv], fmt='o', color='C0')
    a.axhline(g1['c'][j], color='k', lw=1); a.axhspan(g1['c'][j] - g1['e'][j], g1['c'][j] + g1['e'][j], color='k', alpha=.15)
    a.set_xlabel('mesh voltage (V), drift 1000 V'); a.set_ylabel(n + ' (ln per (100 mm)^2)')
ax[3].plot(mv, [r['mean_q'] for r in hv], 'o'); ax[3].set_xlabel('mesh voltage (V)'); ax[3].set_ylabel('mean q_u90 (ns)')
fig.suptitle('6-27 mesh-voltage scans at 1000 V drift (band: golden run 1)', fontsize=9); fig.tight_layout(); fig.savefig(f'{L.OUT}/fig3_mesh_scan.png', dpi=130)
# ---------------- fig4 time blocks
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for a, (n, j) in zip(ax, (('guu', 3), ('gww', 4), ('guw', 5))):
    x0 = 0
    for key, col in (('run6-25', 'C3'), ('run6-27', 'k'), ('run6-28', 'C0')):
        b = A['blocks'][key]
        a.errorbar(np.arange(len(b)) + x0, [q['c'][j] for q in b], [q['err'][j] for q in b], fmt='o', color=col, label=key)
        x0 += len(b) + 1
    a.set_xlabel('block (one acquisition file each, in time order)'); a.set_ylabel(n + ' (ln per (100 mm)^2)')
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{L.OUT}/fig4_time_blocks.png', dpi=130)
print('ok')
