#!/usr/bin/env python3
"""b_02_part1.py: reproducibility of the fine maps and the charge-time proxies (Part 1)."""
import json, numpy as np
from b_lib import *

QS = [('T', 'T_err', 2, 'T (ns)'), ('vx', 'vx_err', 2, 'v_x'), ('vy', 'vy_err', 2, 'v_y'), ('hpx', 'hpx_err', 2, 'hp_x (mm)'),
      ('hpy', 'hpy_err', 2, 'hp_y (mm)'), ('w0x', 'w0x_err', 1, 'w0_x'), ('w0y', 'w0y_err', 1, 'w0_y')]
for v in 'xy':
    for k in ('u50', 'u90', 'uend'):
        QS.append((f'{v}{k}', f'{v}{k}_err', 2, f'{v}_q_{k} (ns)'))
rows = []
RES = {}
for n in (4, 6, 8, 10):
    for key, ek, deg, lab in QS:
        r = {}
        for run in RUNS:
            M = MAPS[run][n]
            fit, res, _ = wfit(M[key], M[ek], n, deg)
            r[run] = (M[key], M[ek], fit, res)
        res1, res2 = r['run1'][3].ravel(), r['run2'][3].ravel()
        ok = np.isfinite(res1) & np.isfinite(res2)
        c = cov(res1, res2)
        # full-map (no surface removed) reproducibility for reference
        rf = pear(r['run1'][0].ravel(), r['run2'][0].ravel())
        # error of r and signal rms by bootstrap over cells
        _, sr, _, _, N = boot_cells(lambda a, b: pear(a, b), (res1, res2), nb=500)
        _, ss, _, _, _ = boot_cells(lambda a, b: np.sqrt(max(cov(a, b), 0)), (res1, res2), nb=500)
        row = dict(n=n, q=key, label=lab, N=int(N),
                   rms1=float(np.nanstd(res1)), rms2=float(np.nanstd(res2)),
                   err1=float(np.nanmean(r['run1'][1])), err2=float(np.nanmean(r['run2'][1])),
                   r_resid=pear(res1, res2), r_err=sr, sig=float(np.sqrt(max(c, 0))), sig_err=ss, r_full=rf)
        rows.append(row)
        RES[(n, key)] = (r['run1'], r['run2'])
json.dump(rows, open(f'{OUT}/part1_repro.json', 'w'), indent=1)
for n in (4, 6, 8, 10):
    print(f'--- n={n}')
    for r in rows:
        if r['n'] == n:
            print(f"{r['label']:14s} N{r['N']:3d} rms {r['rms1']:8.3f} {r['rms2']:8.3f} err {r['err1']:7.3f} {r['err2']:7.3f}  r_full {r['r_full']:+.2f}  r_res {r['r_resid']:+.2f}±{r['r_err']:.2f}  sig {r['sig']:.3f}±{r['sig_err']:.3f}")

# --- proxies versus T
print('\nPROXY vs T')
prox_rows = []
for n in (4, 6, 8):
    for v in ('x', 'y', 'xy'):
        for k in ('u50', 'u90', 'uend'):
            out = dict(n=n, view=v, prox=k)
            fl, rs = {}, {}
            for run in RUNS:
                M = MAPS[run][n]
                if v == 'xy':
                    p = 0.5 * (M[f'x{k}'] + M[f'y{k}']); pe = 0.5 * np.hypot(M[f'x{k}_err'], M[f'y{k}_err'])
                else:
                    p, pe = M[f'{v}{k}'], M[f'{v}{k}_err']
                _, rp, _ = wfit(p, pe, n, 2)
                _, rT, _ = wfit(M['T'], M['T_err'], n, 2)
                fl[run] = (p, M['T']); rs[run] = (rp, rT)
            # cross-run: proxy run A vs T run B, full maps and residual maps
            for lab, d in (('full', fl), ('resid', rs)):
                pa, Ta = d['run1']; pb, Tb = d['run2']
                rc = 0.5 * (pear(pa.ravel(), Tb.ravel()) + pear(pb.ravel(), Ta.ravel()))
                # noise-free correlation: cov(pA,TB)/sqrt(cov(pA,pB) cov(TA,TB))
                a, b = pa.ravel(), pb.ravel(); ta, tb = Ta.ravel(), Tb.ravel()
                num = 0.5 * (cov(a, tb) + cov(b, ta)); den = np.sqrt(max(cov(a, b), 1e-12) * max(cov(ta, tb), 1e-12))
                # slope in d ln: d ln T / d ln proxy (cross-run IV, proxy mean from the maps)
                la1, la2 = np.log(a) if lab == 'full' else a / np.nanmean(a), np.log(b) if lab == 'full' else b / np.nanmean(b)
                if lab == 'full':
                    lt1, lt2 = np.log(ta), np.log(tb)
                else:
                    lt1, lt2 = ta / np.nanmean(MAPS['run1'][n]['T']), tb / np.nanmean(MAPS['run2'][n]['T'])
                    la1 = a / np.nanmean(MAPS['run1'][n][f'{v if v!="xy" else "x"}{k}'])
                    la2 = b / np.nanmean(MAPS['run2'][n][f'{v if v!="xy" else "x"}{k}'])
                s = boot_cells(lambda p1, p2, t1, t2: iv_slope(t1, t2, p1, p2), (la1, la2, lt1, lt2), nb=500)
                out[lab] = dict(r_cross=rc, r_nf=float(np.clip(num / den, -9, 9)), slope=s[0], slope_err=s[1])
            prox_rows.append(out)
            print(n, v, k, {l: {kk: round(vv, 2) for kk, vv in out[l].items()} for l in ('full', 'resid')})
json.dump(prox_rows, open(f'{OUT}/part1_proxy.json', 'w'), indent=1)
