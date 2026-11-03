#!/usr/bin/env python3
"""timing_vs_hv_check.py [--rows DIR] -- is chamber A's improvement of (tX-tY)/sqrt2 with the resist HV (109 -> 23 ns,
425 -> 490 V; paper1_ops/hv_metrics) significant, and is it the chamber or the measurement?  (Dylan 2026-11-03)

Per-trigger edge-start times of the 6-27 HV-scan rows (paper1_timing timing_row.py npz: tx, ty, amp = early amplitude,
the strip sum at the rising-edge cut sample, X+Y [ADC]).  Tests:
  1. significance: s68 with bootstrap errors per HV, the 425 vs 490 V difference in sigmas, and a straight-line
     log-log fit of sigma_xy against HV;
  2. the selection: clean triggers / all triggers per HV (saturated and < 5-strip triggers are dropped);
  3. noise over slope: sigma_xy in bins of the early amplitude, per HV: if the HV only acts through the signal size, the
     per-HV curves lie on one sigma(amp) curve; and sigma_xy in a fixed amplitude window common to all HV;
  4. the amplitude distribution per HV (median) against the gain proxy.
Writes results/timing_vs_hv_check.json and figures/timing_vs_hv_check.{pdf,png} (diagnostic, not in the paper unless asked).
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers')); sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_timing'))
import paper_style as ps  # noqa: E402
from timing_row import s68  # noqa: E402

D = os.environ.get('TIMING_ROWS', '/media/ucla/work/claude-jobs/pairfix/timing_rows')
RUN = 'mx17_det3_saturday_scan_6-27-26'
rng = np.random.default_rng(3)


def sxy(tx, ty):
    return s68(tx - ty) / np.sqrt(2)


def boot(tx, ty, n=300):
    v = [sxy(tx[i], ty[i]) for i in (rng.integers(0, len(tx), len(tx)) for _ in range(n))]
    return float(np.nanstd(v))


rows = {}
for f in glob.glob(f'{D}/c26__{RUN}__hv_scan*_drift_1000V__mx17_3.npz'):
    j = json.load(open(f[:-4] + '.json'))
    z = np.load(f)
    ok = np.isfinite(z['tx']) & np.isfinite(z['ty'])
    rows[int(j['resist_V'])] = dict(tx=z['tx'][ok].astype(float), ty=z['ty'][ok].astype(float), amp=z['amp'][ok].astype(float),
                                    n_trig=j['n_triggers'], n_clean=j['n_clean'])
Vs = sorted(rows)
out = dict(per_hv={})
print('V   n_trig n_clean frac  amp50   sxy   err')
for V in Vs:
    r = rows[V]
    r['sxy'] = sxy(r['tx'], r['ty']); r['err'] = boot(r['tx'], r['ty'])
    out['per_hv'][V] = dict(n_trig=r['n_trig'], n_clean=r['n_clean'], frac_clean=r['n_clean'] / r['n_trig'],
                            amp50=float(np.median(r['amp'])), sigma_xy=r['sxy'], err=r['err'])
    print(f"{V} {r['n_trig']:6d} {r['n_clean']:6d} {r['n_clean']/r['n_trig']:.2f} {np.median(r['amp']):7.0f} {r['sxy']:6.1f} {r['err']:.1f}")
a, b = rows[425], rows[490]
out['diff_425_490'] = dict(a=a['sxy'], b=b['sxy'], sigmas=(a['sxy'] - b['sxy']) / np.hypot(a['err'], b['err']))
# power law sigma ~ amp50^-p across HV
V = np.array(Vs); s = np.array([rows[v]['sxy'] for v in Vs]); am = np.array([np.median(rows[v]['amp']) for v in Vs])
k = V <= 490
p = np.polyfit(np.log(am[k]), np.log(s[k]), 1); out['sigma_vs_median_amp_slope_le490'] = float(p[0])
# amplitude bins, pooled across HV and per HV
edges = np.geomspace(100, 20000, 14)
curves = {}
for V in Vs:
    r = rows[V]; c = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r['amp'] >= lo) & (r['amp'] < hi)
        if m.sum() >= 80:
            c.append((float(np.median(r['amp'][m])), sxy(r['tx'][m], r['ty'][m]), boot(r['tx'][m], r['ty'][m], 100), int(m.sum())))
    curves[V] = c
out['by_amp_bin'] = {str(V): c for V, c in curves.items()}
# fixed windows common to the HV points
win = {}
for lo, hi in ((600, 1000), (1000, 1600), (1600, 2600)):
    w = {}
    for V in Vs:
        r = rows[V]; m = (r['amp'] >= lo) & (r['amp'] < hi)
        if m.sum() >= 80:
            w[V] = (sxy(r['tx'][m], r['ty'][m]), boot(r['tx'][m], r['ty'][m], 150), int(m.sum()))
    win[f'{lo}-{hi}'] = w
out['fixed_amp_window'] = {k_: {str(v): w[v] for v in w} for k_, w in win.items()}
out['chi2_dof_const'] = {}
for k_, w in win.items():
    vs = sorted(w)
    print('window', k_, ' '.join(f'{v}:{w[v][0]:.0f}+-{w[v][1]:.0f}(n{w[v][2]})' for v in vs))
    x = np.array([w[v][0] for v in vs]); e = np.array([w[v][1] for v in vs]); mu = np.sum(x / e**2) / np.sum(1 / e**2)
    out['chi2_dof_const'][k_] = float(np.sum(((x - mu) / e) ** 2) / (len(x) - 1))
    print('   chi2/dof vs constant', out['chi2_dof_const'][k_], 'mean', mu)
# figure
ps.use()
fig, ax = ps.fig_for('full', ncols=2, height=2.3)
import matplotlib.cm as cmm
for V in Vs:
    c = np.array(curves[V], float)
    if len(c):
        ax[0].errorbar(c[:, 0], c[:, 1], yerr=c[:, 2], marker='o', ms=2.5, lw=0.7, elinewidth=0.6,
                       color=cmm.viridis((V - 425) / 100), label=f'{V} V' if V in (425, 455, 485, 505, 525) else None)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('early amplitude X+Y [ADC]')
ax[0].set_ylabel(r'$(t_X-t_Y)/\sqrt{2}$ $s_{68}$ [ns]'); ax[0].legend(fontsize=6.5, ncol=1)
ax[1].errorbar(np.array(Vs), [rows[v]['sxy'] for v in Vs], yerr=[rows[v]['err'] for v in Vs], marker='o', ms=3, lw=0.8, label='all clean triggers')
for (k_, w), mk in zip(win.items(), ('s', '^', 'v')):
    vs = sorted(w); ax[1].errorbar(vs, [w[v][0] for v in vs], yerr=[w[v][1] for v in vs], marker=mk, ms=3, lw=0.8, ls='--', mfc='white', label=f'amp {k_} ADC')
ax[1].set_xlabel('resist HV [V]'); ax[1].set_ylabel(r'$\sigma_{XY}$ [ns]'); ax[1].set_yscale('log'); ax[1].legend(fontsize=6.5)
ps.panel_labels(ax)
ps.save(fig, os.path.join(HERE, 'figures', 'timing_vs_hv_check'))
json.dump(out, open(os.path.join(HERE, 'results', 'timing_vs_hv_check.json'), 'w'), indent=1, default=float)
print(json.dumps(out['diff_425_490']), out['sigma_vs_median_amp_slope_le490'])
