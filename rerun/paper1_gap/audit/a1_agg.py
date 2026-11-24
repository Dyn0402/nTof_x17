"""a1_agg.py -- aggregate the Part 2 bootstrap with the baseline -> agg.json and printed tables"""
import json, pickle
import numpy as np
from a1_lib import OUT
base = json.load(open(f'{OUT}/part2.json'))
boots = pickle.load(open(f'{OUT}/part2_boot.pkl', 'rb'))
agg = {}
def col(f):
    return np.array([f(b) for b in boots], float)
for n in (4, 6, 8, 10, 12):
    B = [b[n] for b in boots]; b0 = base[str(n)]
    d = {}
    def stat(name, f):
        v = np.array([f(b) for b in B], float)
        d[name] = dict(base=float(f(b0)), mean=float(np.nanmean(v)), std=float(np.nanstd(v)), p16=float(np.nanpercentile(v, 16)), p84=float(np.nanpercentile(v, 84)))
    for k in ('rxx', 'ryy', 'rxy', 'frac', 'slope', 'slope12', 'slope21', 'ncorner', 'rms_cx', 'rms_cy'):
        stat(k, lambda x, k=k: x['curl'].get(k, np.nan))
    stat('slope_reg', lambda x: x['curl']['cxy'] / x['curl']['cxx'] if x['curl']['cxx'] > 0 else np.nan)
    stat('slope_inv', lambda x: x['curl']['cyy'] / x['curl']['cxy'] if x['curl']['cxy'] > 0 else np.nan)
    stat('slope_geo', lambda x: np.sqrt(x['curl']['cyy'] / x['curl']['cxx']) if x['curl']['cxx'] > 0 and x['curl']['cyy'] > 0 else np.nan)
    for kind in ('plane', 'none'):
        for k in ('sol_sig', 'tot_sig', 'sol_cov', 'sol_frac_var'):
            stat(f'{kind}_{k}', lambda x, k=k, kind=kind: x[f'helm_{kind}'][k])
        for t in ('both', 'run1', 'run2'):
            stat(f'{kind}_chi2dof_{t}', lambda x, k=kind, t=t: x[f'helm_{k}'][t]['chi2_dof'])
        d[f'{kind}_dof_both'] = b0[f'helm_{kind}']['both']['dof']
        d[f'{kind}_noise_rms_both'] = b0[f'helm_{kind}']['both']['exp_noise_rms']
        d[f'{kind}_obs_rms_both'] = b0[f'helm_{kind}']['both']['obs_rms']
    for r in ('run1', 'run2'):
        stat(f'cx_minus_by_{r}', lambda x, r=r: x['plane'][r]['cx'] - x['plane'][r]['by'])
        d[f'cx_{r}'] = b0['plane'][r]['cx']; d[f'by_{r}'] = b0['plane'][r]['by']
        d[f'cx_err_cell_{r}'] = b0['plane'][r]['cx_err']; d[f'by_err_cell_{r}'] = b0['plane'][r]['by_err']
        stat(f'cx_{r}_b', lambda x, r=r: x['plane'][r]['cx']); stat(f'by_{r}_b', lambda x, r=r: x['plane'][r]['by'])
        stat(f'bx_{r}', lambda x, r=r: x['plane'][r]['bx']); stat(f'cy_{r}', lambda x, r=r: x['plane'][r]['cy'])
    agg[n] = d
json.dump(agg, open(f'{OUT}/agg.json', 'w'), indent=1)
f = lambda s: f"{s['base']:.3f}+-{s['std']:.3f}"
print('n  | ncorner  rxx  ryy  rxy  frac  slope(Deming)  slope_reg  slope_geo')
for n, d in agg.items():
    print(n, f(d['ncorner']), f(d['rxx']), f(d['ryy']), f(d['rxy']), f(d['frac']), f(d['slope']), f(d['slope_reg']), f(d['slope_geo']), f(d['slope_inv']))
print('rms_cx rms_cy')
for n, d in agg.items(): print(n, f(d['rms_cx']), f(d['rms_cy']))
print('Helmholtz: kind, chi2dof both/run1/run2, dof, sol_sig, tot_sig, sol_frac_var, noise rms, obs rms')
for n, d in agg.items():
    for kind in ('plane', 'none'):
        print(n, kind, f(d[f'{kind}_chi2dof_both']), f(d[f'{kind}_chi2dof_run1']), f(d[f'{kind}_chi2dof_run2']), d[f'{kind}_dof_both'], 'sol', f(d[f'{kind}_sol_sig']), 'cov', f"{d[f'{kind}_sol_cov']['base']:.2e}+-{d[f'{kind}_sol_cov']['std']:.1e}", 'tot', f(d[f'{kind}_tot_sig']), 'frac', f(d[f'{kind}_sol_frac_var']), round(d[f'{kind}_noise_rms_both'],4), round(d[f'{kind}_obs_rms_both'],4))
print('plane level: cx, by, diff +- boot std')
for n, d in agg.items():
    for r in ('run1', 'run2'):
        s = d[f'cx_minus_by_{r}']
        print(n, r, f"cx={d[f'cx_{r}']:.4f} by={d[f'by_{r}']:.4f} cell_err {d[f'cx_err_cell_{r}']:.4f}/{d[f'by_err_cell_{r}']:.4f} diff={s['base']:.4f} bootstd={s['std']:.4f} ({s['base']/s['std']:.1f} sig boot)  bx={d[f'bx_{r}']['base']:.4f} cy={d[f'cy_{r}']['base']:.4f}")
