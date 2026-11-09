#!/usr/bin/env python3
"""universal.py --out DIR -- the single-time angle bias as a function of the column length, on one axis for data, model arms,
pitch scan and the literature (the review's 'universal curve').

Inputs (existing tables): the per-plane hit-tier tables of det3 Run 1 and Run 2 (tiers_*: u per family against M3's tan_ref),
the re-simulation tables (resim_*: u per variant against the exact generating slope) and results/gang_scan.json (pitch x angle).
Per |tan| bin: ratio = median(u / (v tan_ref)) [data] or median(u / w_true) [arms]; Delta tan = (ratio - 1) |tan|.  The column
holds N = L |tan| / p strips (L = 28.08 mm).  Each curve is fitted with Delta tan = a / tan + b tan: 'a' is the strip-end term
(it scales as 1/N^2 in the ratio), 'b' the interior compression (a constant ratio excess).  Literature points are converted
from (set angle, reconstructed angle) with their own L and p.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
L_GAP = 28.08
PITCH = 0.78
V = 35.8e-3
EDGES = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17, 0.20, 0.24, 0.28, 0.33, 0.40, 0.50])
KW = {'x': 1.0368355217021707, 'y': 1.0149061257816572}      # det3 pass-1 angle scale per view (bundle w0/kw)
W0 = {'x': -0.002496696974859436, 'y': -0.1651376992772256}   # um/ns
# literature: label, true deg, reco deg, L mm, pitch mm, D_T um/sqrt(cm) (assumed where noted), statistic
LIT = [
    ('Bortfeldt 2014 FSM 10° (meas.)', 10.0, 15.0, 6.0, 0.5, 365, 'MPV; D_T assumed'),
    ('Bortfeldt 2014 FSM 10° (LTspice)', 10.0, 19.3, 6.0, 0.5, 365, 'circuit model, no diffusion'),
    ('Bortfeldt LTspice 20°', 20.0, 24.3, 6.0, 0.5, 365, 'circuit model'),
    ('Bortfeldt LTspice 30°', 30.0, 32.0, 6.0, 0.5, 365, 'circuit model'),
    ('Bortfeldt LTspice 40°', 40.0, 40.6, 6.0, 0.5, 365, 'circuit model'),
    ('Flierl 2018 GEM 15°', 15.0, 19.0, 8.0, 0.4, 365, 'MPV; D_T assumed'),
    ('Flierl 2018 GEM 30°', 30.0, 32.0, 8.0, 0.4, 365, 'mean; D_T assumed'),
    ('Iodice 2015 NSW-type 20°', 20.0, 23.0, 5.0, 0.4, 365, 'peak'),
    ('Ntekas 2016 gas-only MC 10°', 10.0, 14.0, 5.0, 0.4, 365, 'simulation, no circuit'),
    ('Ntekas 2016 gas-only MC 30°', 30.0, 30.2, 5.0, 0.4, 365, 'simulation, no circuit'),
]


def binned(tan, ratio, edges=EDGES, nmin=60):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = np.isfinite(ratio) & (tan >= lo) & (tan < hi)
        if k.sum() < nmin:
            continue
        r = ratio[k]
        med = float(np.median(r))
        err = float(1.2533 * 1.4826 * np.median(np.abs(r - med)) / np.sqrt(k.sum()))
        tc = float(np.median(tan[k]))
        rows.append(dict(lo=lo, hi=hi, tan=tc, n=int(k.sum()), ratio=med, ratio_err=err, dtan=(med - 1) * tc, dtan_err=err * tc,
                         N=L_GAP * tc / PITCH))
    return rows


def fit_ab(rows):
    """Delta tan = a / tan + b tan, weighted least squares on the bins"""
    t = np.array([r['tan'] for r in rows])
    y = np.array([r['dtan'] for r in rows])
    e = np.array([max(r['dtan_err'], 1e-4) for r in rows])
    A = np.vstack([1 / t, t]).T / e[:, None]
    c, *_ = np.linalg.lstsq(A, y / e, rcond=None)
    res = (A @ c - y / e)
    return dict(a=float(c[0]), b=float(c[1]), chi2=float(res @ res), ndf=int(len(t) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='/media/ucla/work/paper2_hits/eos_tables')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'universal.json'))
    ap.add_argument('--fig', default=os.path.join(HERE, 'report_coupling', 'figures', 'universal.png'))
    a = ap.parse_args()
    out = dict(edges=EDGES.tolist(), L=L_GAP, pitch=PITCH, curves={}, fits={}, lit=[])
    # ---- data: both runs, families dream (V0), ideal (V6), fd (V4), with the robust and edge variants
    for run, pat in (('run1', 'tiers_sat_det3_p*.parquet'), ('run2', 'tiers_p2_det3_p*.parquet')):
        T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.tables, pat)))], ignore_index=True)
        T = T[np.isfinite(T.tan_ref)]
        for view in 'xy':
            g = T[T.view == view]
            tan = np.abs(g.tan_ref.to_numpy())
            sgn = np.sign(g.tan_ref.to_numpy())
            # the truth slope in mm/ns on the reference's angle scale: w_ref = kw v tan_ref + w0 (the bundle's angle mapping),
            # i.e. the waveform fit's calibrated scale.  Using the timing v (35.8) alone inflates every ratio by kw - 1 (3.7 % on X).
            w_ref = (KW[view] * V * g.tan_ref.to_numpy() + W0[view] * 1e-3)
            for col in ('u_dream', 'ur_dream', 'ue_dream', 'u_ideal', 'u_fd', 'u_peak', 'u_thr'):
                if col not in g:
                    continue
                key = f'data|{run}|{view}|{col}'
                rows = binned(tan, g[col].to_numpy() / w_ref)
                out['curves'][key] = rows
                if len(rows) >= 4:
                    out['fits'][key] = fit_ab(rows)
            # the waveform fit on the same planes (already on the calibrated scale)
            tw = g.tan_W.to_numpy() * sgn
            key = f'data|{run}|{view}|W'
            rows = binned(tan, tw / np.where(tan > 0, tan, np.nan))
            out['curves'][key] = rows
    # ---- re-simulation arms (exact truth)
    R = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.tables, 'resim_sat_det3_*.parquet')))], ignore_index=True)
    for view in 'xy':
        for var, g in R[R.view == view].groupby('variant'):
            tan = np.abs(g.tan_true.to_numpy())
            ratio = (g.u_dream / g.w_true).to_numpy()
            key = f'resim|{view}|{var}|u_dream'
            rows = binned(tan, ratio)
            out['curves'][key] = rows
            if len(rows) >= 4:
                out['fits'][key] = fit_ab(rows)
            if 'w_fit' in g and g.w_fit.notna().any():
                rows = binned(tan, (g.w_fit / g.w_true).to_numpy())
                out['curves'][f'resim|{view}|{var}|W'] = rows
    # ---- pitch scan
    G = json.load(open(os.path.join(HERE, 'results', 'gang_scan.json')))
    bins = (('0.05-0.1', 0.075), ('0.1-0.2', 0.15), ('0.2-0.3', 0.25), ('0.3-0.5', 0.38))
    for view in G:
        for arm in G[view]:
            for k in G[view][arm]:
                r = G[view][arm][k]['dream']
                rows = []
                sc = 1.0 / KW[view] if arm == 'data' else 1.0      # gang_scan divided the data by the timing v: put it on the kw scale
                for lab, tc in bins:
                    if r.get(lab) is None:
                        continue
                    rr = r[lab] * sc
                    rows.append(dict(tan=tc, ratio=rr, dtan=(rr - 1) * tc, dtan_err=0.01 * tc, N=L_GAP * tc / (PITCH * int(k)), pitch=PITCH * int(k)))
                out['curves'][f'gang|{view}|{arm}|k{k}'] = rows
                if len(rows) >= 3:
                    out['fits'][f'gang|{view}|{arm}|k{k}'] = fit_ab(rows)
    # ---- literature points
    for lab, t_true, t_reco, L, p, D, note in LIT:
        tt, tr = np.tan(np.radians(t_true)), np.tan(np.radians(t_reco))
        out['lit'].append(dict(label=lab, tan_true=tt, ratio=tr / tt, dtan=tr - tt, N=L * tt / p, Lambda=D * 1e-4 * np.sqrt(L / 10) / (L * tt), L=L, pitch=p, note=note))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    # ---- print
    def show(key):
        rows = out['curves'].get(key, [])
        f = out['fits'].get(key)
        s = ' '.join(f'{r["tan"]:.2f}:{r["ratio"]:.3f}' for r in rows)
        fs = f'   a={f["a"]:.4f} b={f["b"]:.4f} chi2/ndf={f["chi2"]:.1f}/{f["ndf"]}' if f else ''
        print(f'{key:34s} {s}{fs}')
    for key in ['data|run1|x|u_dream', 'data|run2|x|u_dream', 'data|run1|y|u_dream', 'data|run2|y|u_dream', 'data|run1|x|ue_dream', 'data|run1|x|u_ideal',
                'data|run1|x|W', 'resim|x|base|u_dream', 'resim|x|metal|u_dream', 'resim|x|point|u_dream', 'resim|x|dp0|u_dream', 'resim|x|s0_07|u_dream',
                'resim|x|s0_45|u_dream', 'resim|y|base|u_dream', 'resim|y|rc0|u_dream', 'resim|y|metal|u_dream', 'resim|y|point|u_dream', 'resim|y|rc2|u_dream',
                'resim|x|base|W', 'resim|y|rc5|W']:
        show(key)
    print('pitch scan fits (data X):', {k: (round(v['a'], 4), round(v['b'], 4)) for k, v in out['fits'].items() if k.startswith('gang|x|data')})
    print('pitch scan fits (point X):', {k: (round(v['a'], 4), round(v['b'], 4)) for k, v in out['fits'].items() if k.startswith('gang|x|point')})
    print('pitch scan fits (metal X):', {k: (round(v['a'], 4), round(v['b'], 4)) for k, v in out['fits'].items() if k.startswith('gang|x|metal')})
    for l in out['lit']:
        print(f'  LIT {l["label"]:36s} tan {l["tan_true"]:.3f} ratio {l["ratio"]:.3f} dtan {l["dtan"]:.3f} N {l["N"]:.1f} Lambda {l["Lambda"]:.3f}')
    # ---- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for key, lab, st in (('data|run1|x|u_dream', 'data X (V0 vs M3)', dict(color='k', marker='o')), ('data|run1|y|u_dream', 'data Y', dict(color='k', marker='s', mfc='none')),
                         ('resim|x|base|u_dream', 'model X (base)', dict(color='C0')), ('resim|x|metal|u_dream', 'metallic (sigma0 0.07, no RC)', dict(color='C3')),
                         ('resim|x|dp0|u_dream', 'no diffusion', dict(color='C1')), ('resim|x|point|u_dream', 'point (geometry only)', dict(color='C2')),
                         ('resim|y|base|u_dream', 'model Y (base)', dict(color='C0', ls='--')), ('resim|y|rc0|u_dream', 'Y, RC off', dict(color='C4', ls='--'))):
        rows = out['curves'].get(key, [])
        if not rows:
            continue
        t = [r['tan'] for r in rows]
        ax[0].errorbar(t, [r['ratio'] for r in rows], [r['ratio_err'] for r in rows], label=lab, **st)
        ax[1].errorbar([r['N'] for r in rows], [r['ratio'] - 1 for r in rows], [r['ratio_err'] for r in rows], label=lab, **st)
        ax[2].errorbar(t, [r['dtan'] for r in rows], [r['dtan_err'] for r in rows], label=lab, **st)
    for l in out['lit']:
        mk = 'D' if 'LTspice' in l['label'] or 'MC' in l['label'] else '*'
        ax[1].plot(l['N'], l['ratio'] - 1, mk, color='C5', ms=8)
        ax[1].annotate(l['label'].split(' ')[0] + ' ' + l['label'].split(' ')[-2 if '(' in l['label'] else -1], (l['N'], l['ratio'] - 1), fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax[0].set(xlabel='|tan theta| (true)', ylabel='tan_reco / tan_true (median)', title='ladder compression against angle', ylim=(0.95, 2.2))
    ax[1].set(xscale='log', yscale='log', xlabel='N = L tan theta / p  (strips in the column)', ylabel='ratio - 1', title='against the column length in strips; literature points (stars/diamonds)')
    ax[2].set(xlabel='|tan theta|', ylabel='Delta tan = tan_reco - tan_true', title='Delta tan = a / tan + b tan')
    for axx in ax:
        axx.grid(alpha=0.3, which='both')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    fig.savefig(a.fig, dpi=130)
    print('wrote', a.out, a.fig)


if __name__ == '__main__':
    main()
