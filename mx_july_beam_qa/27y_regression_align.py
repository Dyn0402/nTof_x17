#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27y_regression_align.py — apply the frozen June-bench hits6 |tan| regression
(restandardized in-situ) + Fisher sign model to the run_55 real-track features
(27x), and re-do the source-hypothesis fit with the CALIBRATED angle.

Purpose: the raw anchored time-fit angle is charge-sharing compressed (~0.62x,
27c).  The bench regression was built to remove exactly that (bench: raw 6.9 deg
-> regression 1.8 deg).  Per-chamber frozen models: A=mx17_3, B=mx17_2,
C=mx17_6, D=mx17_7, each plane restandardized on THIS data (weights+calibration
frozen; the README-validated 'frozen_rs' day-1 strategy).

Deliverables:
  * calibrated tan(theta) per cluster (sign from the Fisher model);
  * source fit slope vs the ideal 1/R -> does the scale go to ~1?  a clean
    in-situ drift velocity if it does;
  * angular resolution (sigma68 of source residual) calibrated vs raw;
  * source LOCATION from the per-chamber u0 (transverse x, and the y = beam/
    vertical offset = bottom of the He-3 capsule).

Outputs: figures/27_tracks/{09_regression,10_source_loc}.png, calib/27_regression.json.

Run:  venv/bin/python mx_july_beam_qa/27y_regression_align.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import trackcache as tc
from ntof_tracking import microtpc_lib as mt

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
CALIB = os.path.join(HERE, 'calib')
MODELS = os.path.join(HERE, '..', 'ntof_tracking', 'models')
BENCH = {'A': 3, 'B': 2, 'C': 6, 'D': 7}
FIDUCIAL = (70.0, 330.0)
STRIP_CENTER = 199.3
FEATS = list(mt.FEATS_HITS6)


def load_features():
    d = np.load(os.path.join(CALIB, '27_features.npz'), allow_pickle=True)
    return pd.DataFrame({k: d[k] for k in d.files})


def robust_profile(x, y, edges, minc=12):
    ctr, med, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= minc:
            ctr.append(np.median(x[m])); med.append(np.median(y[m]))
            err.append(1.253 * np.std(y[m]) / np.sqrt(m.sum()))
    return map(np.array, (ctr, med, err))


def wls(x, y, w):
    W = w.sum(); Wx = (w*x).sum(); Wy = (w*y).sum()
    Wxx = (w*x*x).sum(); Wxy = (w*x*y).sum(); den = W*Wxx-Wx*Wx
    s = (W*Wxy-Wx*Wy)/den; b = (Wy-s*Wx)/W
    return s, b


def s68(a):
    a = a[np.isfinite(a)]
    return 0.5*(np.percentile(a, 84)-np.percentile(a, 16)) if len(a) > 20 else np.nan


def main():
    F = load_features()
    F = F[F[FEATS].notna().all(axis=1)].copy()
    F['v'] = F['detn'].map(lambda d: tc.DRIFT[d]['v_um_ns'])
    # raw anchored angle (low-threshold refit if available, else gate slope)
    sl = np.where(np.isfinite(F['slope_lo']), F['slope_lo'], F['slope'])
    F['tan_raw'] = np.where(np.abs(sl) > 1e-6, 1000.0/(sl*F['v'].values), np.nan)

    calib = {}
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    print(f'{"plane":6s} {"n":>5s} {"scale_raw":>9s} {"scale_reg":>9s} '
          f'{"v_reg":>7s} {"u0_reg":>7s} {"sig_raw":>7s} {"sig_reg":>7s} {"signacc?":>8s}')
    summ = []
    for di, dn in enumerate('ABCD'):
        R = tc.DRIFT[dn]['R']; vnom = tc.DRIFT[dn]['v_um_ns']
        model = mt.load_model(os.path.join(MODELS, f'mx17_{BENCH[dn]}_hits6.json'))
        for pi, pn in enumerate('xy'):
            d = F[(F['detn'] == dn) & (F['plnn'] == pn)].copy()
            if len(d) < 40:
                continue
            Fmat = d[FEATS].values.astype(float)
            pm = model['planes'][pn]
            tan_abs, ok = mt.apply_tan_regression(pm, Fmat, restandardize=True)
            sign = mt.apply_sign(pm['wg'], d['a_asym_sgn'].values,
                                 d['t_asym_sgn'].values)
            d['tan_reg'] = sign * tan_abs
            x = d['cen'].values
            a = ax[pi, di]
            scale_raw = sig_raw = np.nan
            scale_reg = u0_reg = v_reg = sig_reg = np.nan
            for col, key, lab in [('0.6', 'tan_raw', 'raw'), ('C0', 'tan_reg', 'reg')]:
                y = d[key].values
                g = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 0.6)
                fid = g & (x > FIDUCIAL[0]) & (x < FIDUCIAL[1])
                edges = np.linspace(*FIDUCIAL, 9)
                ctr, med, err = robust_profile(x[fid], y[fid], edges)
                if len(ctr) < 4:
                    continue
                s, b = wls(ctr, med, 1.0/err**2)
                scale = -s*R; u0 = -b/s
                sig = s68(y[fid] - (s*x[fid]+b))
                sigdeg = np.degrees(np.arctan(abs(sig)))
                mk = 'o' if key == 'tan_reg' else 's'
                a.errorbar(ctr, med, yerr=err, fmt=mk, color=col, ms=4, capsize=2, label=lab)
                xx = np.linspace(0, 398, 40); a.plot(xx, s*xx+b, color=col, lw=1)
                if key == 'tan_raw':
                    scale_raw, sig_raw = scale, sigdeg
                else:
                    scale_reg, u0_reg, v_reg = scale, u0, vnom*(1.0/R)/abs(s)
                    sig_reg = sigdeg
            a.plot(xx, -(xx-STRIP_CENTER)/R, 'g--', lw=1)
            a.set_ylim(-0.35, 0.35); a.set_title(f'{dn}{pn}: raw x{scale_raw:.2f} reg x{scale_reg:.2f}')
            a.set_xlabel('strip pos [mm]'); a.set_ylabel('tan θ'); a.legend(fontsize=6)
            rec = dict(det=dn, plane=pn, n=int(len(d)), R=R, v_nom=vnom,
                       scale_raw=float(scale_raw), scale_reg=float(scale_reg),
                       v_reg=float(v_reg), u0_reg=float(u0_reg),
                       sig_raw_deg=float(sig_raw), sig_reg_deg=float(sig_reg),
                       bench_sig=pm['holdout']['s68'])
            summ.append(rec); calib[f'{dn}{pn}'] = rec
            print(f'{dn}{pn:5s} {len(d):5d} {scale_raw:9.2f} {scale_reg:9.2f} '
                  f'{v_reg:7.1f} {u0_reg:7.1f} {sig_raw:7.1f} {sig_reg:7.1f} '
                  f'{pm["holdout"]["sign_acc"]:8.2f}')
    fig.suptitle('27y regression vs raw angle, source-pointing (green=ideal 1/R)', y=1.005)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '09_regression.png'), dpi=95)
    plt.close(fig)

    S = pd.DataFrame(summ)
    # source location: y-plane u0 -> source vertical (beam) offset
    print('\n=== source location (u0 - strip_center = source offset in that coord) ===')
    for pn, lab in [('x', 'transverse'), ('y', 'beam/vertical')]:
        s = S[S['plane'] == pn]
        off = s['u0_reg'] - STRIP_CENTER
        print(f'  {lab:14s}: per-chamber offset '
              f'{dict(zip(s.det, off.round(0).tolist()))} mm | mean {off.mean():.0f} mm')

    # summary fig
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    lab = S['det']+S['plane']
    w = 0.38
    ax[0].bar(np.arange(len(S))-w/2, S['scale_raw'], w, label='raw', color='0.6')
    ax[0].bar(np.arange(len(S))+w/2, S['scale_reg'], w, label='regression', color='C0')
    ax[0].axhline(1, color='g', ls='--'); ax[0].set_xticks(range(len(S)))
    ax[0].set_xticklabels(lab); ax[0].legend(); ax[0].set_title('angle scale (1=source-ideal)')
    ax[1].bar(np.arange(len(S))-w/2, S['sig_raw_deg'], w, label='raw', color='0.6')
    ax[1].bar(np.arange(len(S))+w/2, S['sig_reg_deg'], w, label='regression', color='C0')
    ax[1].axhline(1.8, color='r', ls='--', label='bench reg'); ax[1].set_xticks(range(len(S)))
    ax[1].set_xticklabels(lab); ax[1].legend(); ax[1].set_title('source-residual σθ [deg]')
    vt = S[S['det'] != 'B']
    ax[2].bar(lab, S['v_reg']); ax[2].axhline(40.5, color='C0', ls='--')
    ax[2].axhline(44.1, color='C1', ls='--'); ax[2].set_title('v_drift implied (regression) [µm/ns]')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '10_source_loc.png'), dpi=95)
    plt.close(fig)

    with open(os.path.join(CALIB, '27_regression.json'), 'w') as f:
        json.dump(calib, f, indent=1)
    print(f'\nscale: raw median {S["scale_raw"].median():.2f} -> '
          f'regression median {S["scale_reg"].median():.2f} (1=ideal)')
    print(f'σθ: raw {S["sig_raw_deg"].median():.1f}° -> regression {S["sig_reg_deg"].median():.1f}° '
          f'(includes ~5° extended-source spread)')
    print(f'v_drift (A/C/D, regression): {vt["v_reg"].median():.1f} µm/ns (garfield 40-44)')
    print('saved figures 09/10 + calib/27_regression.json')


if __name__ == '__main__':
    main()
