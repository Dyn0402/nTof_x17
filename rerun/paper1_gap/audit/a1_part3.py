"""a1_part3.py -- Part 3: systematics on the offset maps.  (a) acceptance leak: cubic term, symmetrised tan_ref per cell,
|tan_ref| <= 0.3.  (b) cell index from the reconstructed mid-gap position.  -> part3.json"""
import json
import pickle
import numpy as np
from a1_lib import *  # noqa

RUNS = ('run1', 'run2')
base = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))['maps']


def build(variant, n, index='ref'):
    mp = {}
    for run in RUNS:
        mp[run] = {}
        for p in 'xy':
            tmax = 0.3 if variant == 't03' else GM.TAN_LADDER
            U, W, t, w = view_arrays(run, p, index=index, tmax=tmax)
            cub = variant == 'cubic'
            vrun = rline(t, w, cubic=cub, nmin=40)[1]
            cm = cellmap(U, W, t, w, n, cubic=cub, symmetrize=(variant == 'sym'))
            mp[run][p] = to_tau(cm, vrun)
            mp[run][p]['vrun'] = vrun
    return mp


def summarize(mp, ref=None):
    out = {}
    for p in 'xy':
        R = {r: residual(mp[r][p], 'tau', 'etau', 'plane') for r in RUNS}
        s = cross_stats(R['run1'][0], mp['run1'][p]['etau'], R['run2'][0], mp['run2'][p]['etau'], nb=300)
        d = dict(sig=s.get('sig'), sig_deg=float(np.degrees(np.arctan(s['sig']))) if 'sig' in s else None,
                 sig_err=s.get('cov_err'), r12=s.get('r'), ncell=s.get('ncell'),
                 rms1=s.get('rms1'), rms2=s.get('rms2'))
        # own-axis gradient: b_x = dtau_x/dU, c_y = dtau_y/dW  (and the cross one)
        ia = 1 if p == 'x' else 2; ic = 2 if p == 'x' else 1
        for r in RUNS:
            d[f'own_grad_{r}'] = float(R[r][1][ia]); d[f'own_grad_err_{r}'] = float(np.sqrt(R[r][2][ia, ia]))
            d[f'cross_grad_{r}'] = float(R[r][1][ic]); d[f'cross_grad_err_{r}'] = float(np.sqrt(R[r][2][ic, ic]))
            d[f'vrun_{r}'] = float(mp[r][p].get('vrun', np.nan))
        if ref is not None:
            # correlation of the residual maps (and of the full tau maps) with the baseline map
            for r in RUNS:
                Rb = residual(ref[r][p], 'tau', 'etau', 'plane')[0]
                ok = np.isfinite(Rb) & np.isfinite(R[r][0])
                d[f'r_vs_base_{r}'] = float(np.corrcoef(Rb[ok], R[r][0][ok])[0, 1])
                d[f'rms_diff_vs_base_{r}'] = float(np.sqrt(np.mean((Rb[ok] - R[r][0][ok]) ** 2)))
                d[f'ncell_common_{r}'] = int(ok.sum())
        out[p] = d
    return out


res = {}
for n in (4, 8):
    mpb = build('base', n)
    res[f'{n}_base'] = summarize(mpb, mpb)
    for variant in ('cubic', 'sym', 't03'):
        res[f'{n}_{variant}'] = summarize(build(variant, n), mpb)
        print(n, variant, flush=True)
mpb = build('base', 8)
mpr = build('base', 8, index='recon')
res['8_recon'] = summarize(mpr, mpb)
# recon-indexed map against reference-indexed map, cell by cell, also on full tau (plane included)
for p in 'xy':
    for r in RUNS:
        a, b = mpb[r][p]['tau'], mpr[r][p]['tau']
        ok = np.isfinite(a) & np.isfinite(b)
        res['8_recon'][p][f'r_fulltau_{r}'] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        res['8_recon'][p][f'rms_fulltau_diff_{r}'] = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
# cross-run reproducibility of the (recon-indexed) map vs the (ref-indexed) cross-run reproducibility: already in sig, r12.
# and the cross index: run1 ref-indexed vs run2 recon-indexed residuals
for p in 'xy':
    R1 = residual(mpb['run1'][p], 'tau', 'etau', 'plane')[0]; R2 = residual(mpr['run2'][p], 'tau', 'etau', 'plane')[0]
    s = cross_stats(R1, mpb['run1'][p]['etau'], R2, mpr['run2'][p]['etau'], nb=300)
    res['8_recon'][p]['cross_ref1_recon2'] = dict(r=s['r'], sig=s['sig'])
    R1 = residual(mpr['run1'][p], 'tau', 'etau', 'plane')[0]; R2 = residual(mpb['run2'][p], 'tau', 'etau', 'plane')[0]
    s = cross_stats(R1, mpr['run1'][p]['etau'], R2, mpb['run2'][p]['etau'], nb=300)
    res['8_recon'][p]['cross_recon1_ref2'] = dict(r=s['r'], sig=s['sig'])
json.dump(res, open(f'{OUT}/part3.json', 'w'), indent=1, default=float)
for k, v in res.items():
    for p in 'xy':
        d = v[p]
        print(k, p, 'sig', round(d['sig'], 5), 'r12', round(d['r12'], 3), 'ncell', d['ncell'],
              'own', round(d['own_grad_run1'], 4), round(d['own_grad_run2'], 4), '+-', round(d['own_grad_err_run1'], 4),
              'r_base', [round(d.get(f'r_vs_base_{r}', np.nan), 3) for r in RUNS], 'rmsdiff', [round(d.get(f'rms_diff_vs_base_{r}', np.nan), 4) for r in RUNS])
print(json.dumps({p: {k: v for k, v in res['8_recon'][p].items() if 'full' in k or 'cross_' in k} for p in 'xy'}, default=float))
