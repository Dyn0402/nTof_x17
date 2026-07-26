#!/usr/bin/env python3
"""Threading metrics from batch free fits.

Inputs: freefit.pkl (list of per-event dicts from wf8) + hyper json.
Outputs (all in waveform_first/):
  ff_w_vs_tan.png     fitted w vs tan_ref, regression slope = v_eff per plane
  ff_angle_res.png    per-event angle residuals (deg) vs production numbers
  ff_mesh_res.png     p0 - ref_mesh residuals
  ff_dchi2.png        chi2(ref line) - chi2(free)  vs  chi2(prod line) - chi2(free)
  ff_threading.png    full-depth max track-line deviation from ref, census
Prints a summary table.
"""
import os, sys, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
FF = sys.argv[1] if len(sys.argv) > 1 else 'freefit.pkl'
res = pickle.load(open(os.path.join(BASE, FF), 'rb'))
hj = json.load(open(os.path.join(BASE, 'hyper_ref.json')))
V = hj['v']  # µm/ns? no: mm/ns*1e3... stored as v in um/ns units (34 scale)
print(f'v_cal = {V:.2f} um/ns, n events {len(res)}')

def robust_sigma(a):
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))

rows = {p: dict(tan=[], w=[], p0=[], p0r=[], dchi_ref=[], dchi_prod=[],
                chi2=[], dof=[], wprod=[], amax=[]) for p in ('x', 'y')}
for r in res:
    for p in ('x', 'y'):
        if p not in r or 'error' in r[p]:
            continue
        d = r[p]
        rows[p]['tan'].append(d['tan_ref'])
        rows[p]['w'].append(d['w'])
        rows[p]['p0'].append(d['p0'])
        rows[p]['p0r'].append(d['p0_ref'])
        rows[p]['chi2'].append(d['chi2']); rows[p]['dof'].append(d['dof'])
        rows[p]['dchi_ref'].append(d['chi2_ref'] - d['chi2'])
        rows[p]['dchi_prod'].append(d['chi2_prod'] - d['chi2'])
        rows[p]['wprod'].append(d.get('w_prod', np.nan))
        rows[p]['amax'].append(d['amax'])
for p in rows:
    for k in rows[p]:
        rows[p][k] = np.asarray(rows[p][k], float)

# ---- 1. w vs tan_ref ----
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
v_eff = {}
for i, p in enumerate(('x', 'y')):
    R = rows[p]
    ax = axs[0, i]
    ax.plot(R['tan'], R['w'] * 1e3, '.', ms=2, alpha=0.4, label='free fit')
    ax.plot(R['tan'], R['wprod'] * 1e3, '.', ms=2, alpha=0.2, color='orange',
            label='production 1/slope')
    # robust regression through origin-ish: fit w = v*tan + b
    ok = np.isfinite(R['w']) & (np.abs(R['tan']) < 0.6)
    A = np.vstack([R['tan'][ok], np.ones(ok.sum())]).T
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A, R['w'][ok] * 1e3, rcond=None)
        pred = A @ coef
        rres = R['w'][ok] * 1e3 - pred
        s = robust_sigma(rres)
        keep = np.abs(rres - np.median(rres)) < 3 * s
        A = A[keep]; ok_idx = np.where(ok)[0][keep]
        ok = np.zeros_like(ok); ok[ok_idx] = True
    coef = np.asarray(coef).ravel()
    v_eff[p] = float(coef[0])
    tt = np.linspace(-0.6, 0.6, 3)
    ax.plot(tt, coef[0] * tt + coef[1], 'r-', lw=1.5,
            label=f'v_eff={coef[0]:.1f} um/ns (b={coef[1]:.2f})')
    ax.plot(tt, 34 * tt, 'k--', lw=1, label='v=34')
    ax.set_xlabel('tan_ref'); ax.set_ylabel('fitted w [um/ns]')
    ax.set_title(f'{p}: fitted transverse speed vs ref tangent')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-25, 25)

    # angle residual: tan_fit - tan_ref using v_cal
    ax = axs[1, i]
    tan_fit = R['w'] * 1e3 / V
    dth = (np.degrees(np.arctan(tan_fit)) - np.degrees(np.arctan(R['tan'])))
    tanp = R['wprod'] * 1e3 / V
    dthp = (np.degrees(np.arctan(tanp)) - np.degrees(np.arctan(R['tan'])))
    b = np.linspace(-8, 8, 90)
    ax.hist(dth, bins=b, histtype='step', lw=2,
            label=f'forward fit: med {np.nanmedian(dth):+.2f}, '
                  f'sig {robust_sigma(dth):.2f} deg')
    ax.hist(dthp, bins=b, histtype='step', lw=1.5, color='orange',
            label=f'production: med {np.nanmedian(dthp):+.2f}, '
                  f'sig {robust_sigma(dthp):.2f} deg')
    ax.set_xlabel('angle - ref angle [deg]  (v=v_cal)')
    ax.set_title(f'{p}: per-event angle residual'); ax.legend(fontsize=8)
    print(f'{p}: v_eff {v_eff[p]:.2f}  angle med {np.nanmedian(dth):+.3f} '
          f'sig {robust_sigma(dth):.3f} deg   (prod med {np.nanmedian(dthp):+.3f} '
          f'sig {robust_sigma(dthp):.3f})')
fig.tight_layout(); fig.savefig(os.path.join(BASE, 'ff_w_vs_tan.png'), dpi=110)

# ---- 1b. angle-independence of implied v (the 46-series killer test) ----
fig1b, ax1b = plt.subplots(1, 2, figsize=(12, 4.5))
for i, p in enumerate(('x', 'y')):
    R = rows[p]
    at = np.abs(R['tan'])
    vimp = R['w'] * 1e3 / R['tan']
    vprod = R['wprod'] * 1e3 / R['tan']
    bins = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
    for series, lab, mk in ((vimp, 'forward fit', 'o'), (vprod, 'production', 's')):
        med = [np.nanmedian(series[(at >= a) & (at < b)]) for a, b in bins]
        err = [robust_sigma(series[(at >= a) & (at < b)]) /
               max(np.sqrt(np.sum((at >= a) & (at < b))), 1) for a, b in bins]
        ax1b[i].errorbar([0.5 * (a + b) for a, b in bins], med, yerr=err,
                         fmt=mk + '-', capsize=3, label=lab)
    ax1b[i].axhline(34, color='k', ls='--', lw=1, label='v_geom 34')
    ax1b[i].set_xlabel('|tan_ref|'); ax1b[i].set_ylabel('median w/tan [um/ns]')
    ax1b[i].set_title(f'{p}: implied v vs angle (flat = physical)')
    ax1b[i].legend(fontsize=8); ax1b[i].grid(alpha=0.3)
fig1b.tight_layout(); fig1b.savefig(os.path.join(BASE, 'ff_v_vs_angle.png'), dpi=110)

# ---- 2. mesh position residual ----
fig2, axs2 = plt.subplots(1, 2, figsize=(11, 4.5))
for i, p in enumerate(('x', 'y')):
    R = rows[p]
    dp = R['p0'] - R['p0r']
    axs2[i].hist(dp, bins=np.linspace(-4, 4, 100), histtype='step', lw=2,
                 label=f'med {np.nanmedian(dp):+.3f}, sig {robust_sigma(dp):.3f} mm')
    axs2[i].set_xlabel(f'{p}: p0_fit - ref_mesh [mm]'); axs2[i].legend()
    print(f'{p}: mesh residual med {np.nanmedian(dp):+.3f} sig {robust_sigma(dp):.3f} mm')
fig2.tight_layout(); fig2.savefig(os.path.join(BASE, 'ff_mesh_res.png'), dpi=110)

# ---- 3. delta-chi2 discrimination ----
fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
for i, p in enumerate(('x', 'y')):
    R = rows[p]
    ndof = R['dof']
    x = R['dchi_ref'] / ndof
    y = R['dchi_prod'] / ndof
    ok = np.isfinite(x) & np.isfinite(y)
    axs3[i].plot(x[ok], y[ok], '.', ms=3, alpha=0.4)
    axs3[i].plot([0, 50], [0, 50], 'k--', lw=1)
    axs3[i].set_xlim(-1, 30); axs3[i].set_ylim(-1, 30)
    axs3[i].set_xlabel('[chi2(ref line) - chi2(free)]/dof')
    axs3[i].set_ylabel('[chi2(prod line) - chi2(free)]/dof')
    frac_ref_better = np.mean(x[ok] < y[ok])
    axs3[i].set_title(f'{p}: ref line fits better in {frac_ref_better*100:.0f}% of events')
    print(f'{p}: median dchi2/dof ref {np.nanmedian(x):+.3f} prod {np.nanmedian(y):+.3f}; '
          f'ref-better frac {frac_ref_better:.3f}')
fig3.tight_layout(); fig3.savefig(os.path.join(BASE, 'ff_dchi2.png'), dpi=110)

# ---- 4. full-depth threading census (track-line deviation over 30 mm) ----
fig4, ax4 = plt.subplots(1, 2, figsize=(12, 4.5))
for i, p in enumerate(('x', 'y')):
    R = rows[p]
    tan_fit = R['w'] * 1e3 / V
    dev0 = np.abs(R['p0'] - R['p0r'])
    dev30 = np.abs((R['p0'] + tan_fit * 29) - (R['p0r'] + R['tan'] * 29))
    worst = np.maximum(dev0, dev30)
    ax4[i].hist(worst[np.isfinite(worst)], bins=np.linspace(0, 5, 100),
                cumulative=True, density=True, histtype='step', lw=2)
    for T in (0.5, 1.0, 1.5):
        f = np.nanmean(worst < T)
        print(f'{p}: full-depth line-dev < {T} mm: {f*100:.1f}%')
        ax4[i].axvline(T, color='gray', ls=':', lw=0.7)
    ax4[i].set_xlabel(f'{p}: max |line - ref| over 30 mm depth [mm]')
    ax4[i].set_ylabel('cumulative fraction'); ax4[i].grid(alpha=0.3)
fig4.tight_layout(); fig4.savefig(os.path.join(BASE, 'ff_threading.png'), dpi=110)
print('done')
