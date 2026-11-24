"""e_06_figs.py -- figures and the weight a -> $WORK/audit/e/fig_*.png, a_weight.json"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import e_lib as L, e_03_main as M
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
R = json.load(open(f'{L.OUT}/results.json'))
S = pickle.load(open(f'{L.OUT}/slices_default.pkl', 'rb'))
H = json.load(open(f'{L.OUT}/headon.json'))
cols = plt.cm.viridis(np.linspace(0, 0.9, 5))

# ---- 1: calibration
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for (v, axis), a in zip((('x', 'U'), ('y', 'W')), ax):
    for i, r in enumerate(S[(v, axis, 'both')]):
        sel = r['n'] > 40
        a.plot(r['tmid'][sel], r['z'][sel], 'o-', ms=3, color=cols[i], label=f'{axis} slice {i+1} ({r["coord"]:+.2f})')
    a.axvspan(0, 810, color='0.9', zorder=0)
    a.set_xlabel('time since t0 (ns)'); a.set_ylabel('effective height z_k (mm)'); a.set_title(f'{v.upper()} view, 5 slices along {axis}')
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{L.OUT}/fig_zk_calibration.png'); plt.close(fig)

# ---- 2: delta profiles
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
for j, (v, axis) in enumerate((('x', 'U'), ('y', 'W'))):
    res = S[(v, axis, 'both')]
    for i, r in enumerate(res):
        sel = r['sel']; z = r['z']; dl = r['delta']; a, b, q = r['par']
        zz = np.linspace(0, 25, 100)
        ax[0, j].errorbar(z[sel], (dl - a)[sel], yerr=r['ed'][sel], fmt='o', ms=3, color=cols[i], label=f'slice {i+1}, {axis}={r["coord"]:+.2f}')
        ax[0, j].plot(zz, b * zz + q * zz ** 2, '-', color=cols[i], lw=1)
        ax[1, j].plot(z[sel], (dl - a - b * z)[sel], 'o', ms=3, color=cols[i])
        ax[1, j].plot(zz, q * zz ** 2, '-', color=cols[i], lw=1)
    ax[0, j].set_title(f'{v.upper()} view along {axis}'); ax[0, j].set_ylabel('displacement delta - a (mm)'); ax[0, j].legend(fontsize=7)
    ax[1, j].set_ylabel('delta - a - b z  (mm), the curvature'); ax[1, j].set_xlabel('effective height z (mm)')
    ax[1, j].axhline(0, color='0.5', lw=0.5)
fig.tight_layout(); fig.savefig(f'{L.OUT}/fig_delta_profiles.png'); plt.close(fig)

# ---- 3: b and D vs slice with predictions
fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
for j, (v, axis) in enumerate((('x', 'U'), ('y', 'W'))):
    P = R['pred'][f'{v}|{axis}']; d = R['default'][f'{v}|{axis}|both']
    xs = np.array(d['b']['x'])
    for i, (k, lab) in enumerate((('b', 'b: slope at the mesh end'), ('D', 'D = c_slope - h_slope (extrapolated to 28.2 mm)'))):
        a = ax[i, j]
        a.errorbar(xs, d[k]['y'], yerr=d[k]['yerr'], fmt='ko', label='waveforms, both runs')
        hh = H[f'{v}|{axis}'][k]; a.plot(xs + 0.04, hh['y'], 's', mfc='none', color='C1', label='head-on tracks only')
        xx = np.linspace(xs.min() - .1, xs.max() + .1, 10)
        c0 = d[k]['icpt']
        if k == 'b':
            a.plot(xx, c0 + P['b'] * xx, 'C0-', label=f'mesh dish (positions): {P["b"]*1e4:.0f}e-4 per 100 mm')
        else:
            for kk, ls, nm in (('D_audit', 'C0-', 'audited fit'), ('D_flat', 'C3--', 'flat cathode'), ('D_par', 'C2:', 'parallel')):
                a.plot(xx, c0 + P[kk] * xx, ls, label=f'{nm}: {P[kk]*1e4:+.0f}e-4')
        a.set_xlabel(f'{axis} (100 mm)'); a.set_ylabel(lab, fontsize=8); a.legend(fontsize=6.5)
        a.set_title(f'{v.upper()} view  gradient {d[k]["grad"]*1e4:+.0f} +- {d[k]["err"]*1e4:.0f} (e-4 per 100 mm)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{L.OUT}/fig_gradients.png'); plt.close(fig)

# ---- weight a
out = {}
for v, axis in (('x', 'U'), ('y', 'W')):
    d = R['default'][f'{v}|{axis}|both']; w = R['w0v'][f'{v}|{axis}']
    gb, gD = np.array(d['b']['boots']), np.array(d['D']['boots'])
    rng = np.random.default_rng(1)
    gw = w['grad'] + w['err'] * rng.standard_normal(len(gb))
    a = (gw - gb) / gD
    a0 = (w['grad'] - d['b']['grad']) / d['D']['grad']
    q = np.percentile(a, [16, 50, 84])
    out[f'{v}|{axis}'] = dict(a=a0, p16=q[0], p50=q[1], p84=q[2], a_range_midpoint=float(0.5 * (d['zr'][0] + d['zr'][1]) / 28.2),
                              a_mean_of_my_range=float((d['zr'][0] + d['zr'][1]) / 2 / 28.2))
    print(v, axis, out[f'{v}|{axis}'])
json.dump(out, open(f'{L.OUT}/a_weight.json', 'w'), indent=1)
