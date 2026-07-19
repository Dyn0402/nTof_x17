#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27c_source_align.py — source-hypothesis micro-TPC calibration & alignment for
run_55, per detector, per plane, using the clean inclined-track sample.

Physics
-------
Each chamber is a single micro-TPC 30 mm from the He-3 target ("the source").
A track from the source hitting the readout at transverse strip position u makes
an angle theta from the drift normal with
        tan(theta) = (u - u0) / R                              (source model)
where R is the chamber radius (~234 mm) and u0 is the strip position that a
NORMAL-incidence (radial) track hits = the transverse ALIGNMENT of the chamber.

The micro-TPC angle is measured from the amplitude-weighted per-strip
drift-time (max_sample) trail:  slope = d(t)/d(u) [ns/mm], and
        tan(theta)_meas = 1000 / (slope * v_drift).
(NB the raw anchored-time slope carries the ~40 % charge-sharing compression
seen on the bench; here it appears as an overall SCALE on tan.)

Fitting tan_meas vs u in a fiducial window gives, per plane:
  * u0        -> transverse alignment (compare to the mechanical center);
  * slope s   -> compared to the ideal -1/R: s = -(1/R)*(v_nom/v_true)*kappa,
                where kappa is the angle-scale (charge-sharing) bias.  The
                cross-detector CONSISTENCY of s (same gas) separates a common
                angle-scale bias from a per-chamber v_drift/distortion effect;
  * residual(u) = tan_meas - source_fit  OUTSIDE the fiducial -> the
                position-dependent DISTORTION map (fringe field at the edges) —
                this is the "bending vs position" seen when plotting an hour of
                tracks.  Applying it back shrinks the residual (before/after).

Consumes calib/27_tracks.npz is NOT used here; we re-select inclined clusters
straight from the 27_run55 cache (more per-plane statistics than matched pairs).

Outputs: calib/27_align.json, figures/27_tracks/{04_align,05_distortion,
06_summary}.png, console summary.

Run:  venv/bin/python mx_july_beam_qa/27c_source_align.py
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
import trackcache as tc

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
CALIB = os.path.join(HERE, 'calib')
os.makedirs(FIGDIR, exist_ok=True)

STRIP_CENTER_MM = 199.3            # (512-1)/2 * 0.78
FIDUCIAL = (70.0, 330.0)           # strip-position fiducial [mm] (edge fringe cut)
# mechanical transverse offset of each chamber centre from the beam axis, taken
# from run_config det_center_coords (the in-plane component); sign resolved by
# best agreement with the fitted u0.  For x-plane use the transverse (non-beam)
# center component; for the y-plane the source sits on the beam axis (offset 0).
MECH_OFFSET = {  # (x-plane transverse offset, y-plane offset) mm, |value|
    'A': (16.35, 0.0), 'B': (15.75, 0.0), 'C': (17.3, 0.0), 'D': (15.5, 0.0),
}


def robust_profile(x, y, edges):
    ctr, med, err, cnt = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 12:
            ctr.append(np.median(x[m]))
            med.append(np.median(y[m]))
            err.append(1.253 * np.std(y[m]) / np.sqrt(m.sum()))
            cnt.append(int(m.sum()))
    return map(np.array, (ctr, med, err, cnt))


def wls_line(x, y, w):
    W = w.sum(); Wx = (w * x).sum(); Wy = (w * y).sum()
    Wxx = (w * x * x).sum(); Wxy = (w * x * y).sum()
    den = W * Wxx - Wx * Wx
    s = (W * Wxy - Wx * Wy) / den
    b = (Wy - s * Wx) / W
    return s, b


def s68(a):
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan
    q = np.percentile(a, [16, 84])
    return 0.5 * (q[1] - q[0])


def main():
    cl, ss, ev = tc.load_all()
    cl = tc.add_derived(cl, ss)
    evk = ev.set_index(['sub', 'eid'])
    cl = cl.join(evk[['t_ms']], on=['sub', 'eid'])
    cl['b12'] = (cl['t_ms'] > 7) & (cl['t_ms'] < 30)
    cl = tc.tag_tracklike(cl)
    inc = cl[cl['b12'] & cl['inclined']].copy()
    inc['v'] = inc['detn'].map(lambda d: tc.DRIFT[d]['v_um_ns'])
    slope = inc['slope'].values
    inc['tan'] = np.where(np.abs(slope) > 1e-6,
                          1000.0 / (slope * inc['v'].values), np.nan)

    calib = {}
    fig1, ax1 = plt.subplots(2, 4, figsize=(19, 9))   # profiles + fit
    fig2, ax2 = plt.subplots(2, 4, figsize=(19, 9))   # distortion residual
    summ = []
    print(f'{"plane":6s} {"n_fid":>5s} {"u0_mm":>7s} {"mech":>6s} {"align":>6s} '
          f'{"s*(-R)":>7s} {"v_true":>7s} {"res_raw":>7s} {"res_cor":>7s} '
          f'{"sigma_deg":>9s}')
    for di, dn in enumerate('ABCD'):
        R = tc.DRIFT[dn]['R']; vnom = tc.DRIFT[dn]['v_um_ns']
        for pi, pn in enumerate('xy'):
            d = inc[(inc['detn'] == dn) & (inc['plnn'] == pn)]
            x = d['cen'].values
            y = d['tan'].values
            g = np.isfinite(x) & np.isfinite(y) & (np.abs(y) < 0.6)
            x, y = x[g], y[g]
            a1 = ax1[pi, di]; a2 = ax2[pi, di]
            a1.scatter(x, y, s=3, alpha=0.08, color='gray')
            # fiducial fit
            fid = (x > FIDUCIAL[0]) & (x < FIDUCIAL[1])
            edges = np.linspace(FIDUCIAL[0], FIDUCIAL[1], 9)
            ctr, med, err, cnt = robust_profile(x[fid], y[fid], edges)
            rec = dict(det=dn, plane=pn, n=int(len(x)), n_fid=int(fid.sum()),
                       R=R, v_nom=vnom, fiducial=list(FIDUCIAL))
            if len(ctr) >= 4:
                s, b = wls_line(ctr, med, 1.0 / err**2)
                u0 = -b / s
                v_true = vnom * (1.0 / R) / abs(s)
                s_R = -s * R                       # ideal 1.0
                # distortion map over full range: residual of profile vs the
                # fiducial source line
                fedg = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 13)
                fc, fm, fe, fn = robust_profile(x, y, fedg)
                model = s * fc + b
                resid = fm - model
                # per-track residual (fiducial) -> angular resolution
                rt = y[fid] - (s * x[fid] + b)
                sig = s68(rt)
                sig_deg = np.degrees(np.arctan(abs(sig))) if np.isfinite(sig) else np.nan
                # before/after over the FULL position range (where the edge
                # distortion lives): raw = source model only; corrected also
                # subtracts the distortion map.  Shows the map's value at edges.
                res_raw = s68(y - (s * x + b))
                dcorr = np.interp(x, fc, resid)
                res_cor = s68(y - (s * x + b) - dcorr)
                mech = MECH_OFFSET[dn][pi]
                # expected u0: strip centre +- mechanical offset (sign by best fit)
                exp_hi = STRIP_CENTER_MM + mech
                exp_lo = STRIP_CENTER_MM - mech
                exp = exp_lo if abs(u0 - exp_lo) < abs(u0 - exp_hi) else exp_hi
                align = u0 - exp
                rec.update(u0_mm=float(u0), slope=float(s), scale_sR=float(s_R),
                           v_true_umns=float(v_true), sigma_tan=float(sig),
                           sigma_deg=float(sig_deg), align_resid_mm=float(align),
                           mech_expected_u0=float(exp),
                           dist_pos=fc.tolist(), dist_dtan=resid.tolist(),
                           res_raw=float(res_raw), res_cor=float(res_cor))
                # plots
                xx = np.linspace(0, 398, 60)
                a1.errorbar(ctr, med, yerr=err, fmt='o', color='C0', capsize=2, ms=4)
                a1.plot(xx, s * xx + b, 'r-', lw=1.5, label=f's·(-R)={s_R:.2f}')
                a1.plot(xx, -(xx - u0) / R, 'g--', lw=1, label='source 1/R')
                for fx in FIDUCIAL:
                    a1.axvline(fx, color='0.8', ls=':')
                a1.set_title(f'{dn}{pn}: u0={u0:.0f} v_t={v_true:.0f} σ={sig_deg:.1f}°')
                a1.set_ylim(-0.35, 0.35); a1.legend(fontsize=6)
                a1.set_xlabel('strip pos [mm]'); a1.set_ylabel('tan θ')
                # distortion residual panel
                a2.axhline(0, color='k', lw=0.6)
                a2.errorbar(fc, resid, yerr=fe, fmt='o-', color='C3', capsize=2, ms=4)
                for fx in FIDUCIAL:
                    a2.axvline(fx, color='0.8', ls=':')
                a2.set_ylim(-0.2, 0.2)
                a2.set_title(f'{dn}{pn}: distortion (res_raw {res_raw:.3f}->{res_cor:.3f})')
                a2.set_xlabel('strip pos [mm]'); a2.set_ylabel('Δtan (meas-model)')
                summ.append(rec)
                print(f'{dn}{pn:5s} {fid.sum():5d} {u0:7.1f} {exp:6.1f} {align:6.1f} '
                      f'{s_R:7.2f} {v_true:7.1f} {res_raw:7.3f} {res_cor:7.3f} '
                      f'{sig_deg:9.2f}')
            calib[f'{dn}{pn}'] = rec
    fig1.suptitle('27c source-pointing calibration (inclined tracks, b1/b2)', y=1.005)
    fig1.tight_layout(); fig1.savefig(os.path.join(FIGDIR, '04_align.png'), dpi=95)
    fig2.suptitle('27c fringe-field distortion map (residual vs position)', y=1.005)
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGDIR, '05_distortion.png'), dpi=95)
    plt.close('all')

    # summary figure: cross-detector consistency
    S = pd.DataFrame(summ)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    lab = S['det'] + S['plane']
    ax[0].bar(lab, S['scale_sR']); ax[0].axhline(1.0, color='g', ls='--')
    ax[0].set_ylabel('s·(−R)  (1 = source-ideal)'); ax[0].set_title('angle-scale (charge-sharing) per plane')
    ax[1].bar(lab, S['v_true_umns']);
    ax[1].axhline(40.5, color='C0', ls='--', label='A garfield')
    ax[1].axhline(44.1, color='C1', ls='--', label='BCD garfield')
    ax[1].set_ylabel('v_true implied [µm/ns]'); ax[1].legend(); ax[1].set_title('implied drift velocity')
    ax[2].bar(lab, S['align_resid_mm'].abs()); ax[2].axhline(10, color='r', ls='--', label='1 cm')
    ax[2].set_ylabel('|alignment residual| [mm]'); ax[2].legend()
    ax[2].set_title('transverse alignment vs mechanical')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '06_summary.png'), dpi=95)
    plt.close(fig)

    with open(os.path.join(CALIB, '27_align.json'), 'w') as f:
        json.dump(calib, f, indent=1)

    # cross-detector v_true (exclude B — known pathology)
    vt = S[S['det'] != 'B']['v_true_umns']
    print(f'\nv_true (A/C/D planes): median {vt.median():.1f} µm/ns '
          f'range {vt.min():.0f}-{vt.max():.0f}  (garfield 40-44)')
    print(f'angle-scale s·(−R): A/C/D median {S[S.det!="B"]["scale_sR"].median():.2f} '
          f'(charge-sharing compresses raw angle if <1)')
    print(f'alignment: max |residual| {S["align_resid_mm"].abs().max():.1f} mm '
          f'across planes (mechanical ~1 cm offsets)')
    print(f'angular resolution (raw anchored fit) median '
          f'{S["sigma_deg"].median():.1f}°  (bench raw ~6.9°, regression ~1.8°)')
    print('\nsaved calib/27_align.json, figures/27_tracks/{04_align,05_distortion,06_summary}.png')


if __name__ == '__main__':
    main()
