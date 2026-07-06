#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_tpc_angle_bias.py

Diagnose the systematic off-diagonal offset in the micro-TPC angle correlation
(θ_det consistently ABOVE |θ_ref|, seen identically in every detector).

Hypothesis under test
---------------------
Every cosmic crosses the FULL drift gap, so the time span of a cluster is fixed
at ~gap/v_drift regardless of track angle, while the SPATIAL cluster width has a
non-geometric floor (resistive-layer charge spreading + transverse diffusion +
threshold effects).  The strip fit therefore sees
    dx/dt  ≈  v · (tanθ_ref + w_extra/gap)
i.e. an additive, roughly angle-independent excess in tan-space:
    tanθ_det ≈ tanθ_ref ± w_extra/gap        (sign follows the track sign)
which appears as a ridge PARALLEL to the diagonal, displaced outward — exactly
what the correlation plots show.

Checks
------
 1. Median profile θ_det vs θ_ref (signed) and Δtan = |tanθ_det|−|tanθ_ref|
    vs |θ_ref| — is the excess constant in tan-space?
 2. Cluster time span vs |θ_ref| — is it angle-independent (→ gap/v)?
 3. Cluster spatial extent vs |θ_ref| — geometric slope + non-geometric floor?
 4. Bias split by n_strips — do wider (more spread) clusters carry more bias?
 5. Direct drift-velocity estimator immune to the additive offset: per-sign
    straight-line fit of the (rotated) strip-fit slope [µm/ns] vs tanθ_ref;
    the fitted SLOPE is v_drift, the INTERCEPT is v·w_extra/gap.

Usage:  ../.venv/bin/python 13_tpc_angle_bias.py [run_key] [--veto=50]
Output: <Analysis>/<run>/<subrun>/<det>/alignment_tpc_veto50/bias_study/
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from qa_config import config_from_argv, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
MIN_STRIPS = 4
RES_CUT_MM = 10.0
PITCH_MM = 0.78
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking); was 20 pre-v2


def robust_line(x, y, n_iter=4, clip=3.0):
    """OLS with iterative sigma-clipping. Returns slope, intercept, slope_err, n_used."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 10:
            return np.nan, np.nan, np.nan, 0
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    n = int(keep.sum())
    xk = x[keep]
    resid = y[keep] - np.polyval(p, xk)
    se = np.sqrt(np.sum(resid**2) / max(n - 2, 1) / np.sum((xk - xk.mean())**2))
    return float(p[0]), float(p[1]), float(se), n


def main():
    tag = f'_veto{VETO}' if VETO is not None else ''
    out_dir = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
    cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')

    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    print(f'{len(results):,} cached events; alignment {best}')

    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, yang, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)

    # ---- Quality selection (mirrors plot_angle_correlation) ----
    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and np.isfinite(r.x_fit.slope_mm_per_ns) and np.isfinite(r.y_fit.slope_mm_per_ns)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    print(f'{len(sel):,} quality events (n_strips>={MIN_STRIPS}, r<{RES_CUT_MM} mm)')

    s_x = np.array([r.x_fit.slope_mm_per_ns for r in sel]) * 1000.0   # µm/ns
    s_y = np.array([r.y_fit.slope_mm_per_ns for r in sel]) * 1000.0
    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    ns_x = np.array([r.x_fit.n_strips for r in sel])
    ns_y = np.array([r.y_fit.n_strips for r in sel])
    dur_x = np.array([r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns for r in sel])
    dur_y = np.array([r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns for r in sel])

    # Rotate detector slopes into the M3 frame (same transform as production code)
    th = np.deg2rad(best.theta_deg)
    S_x = np.cos(th) * s_x - np.sin(th) * s_y     # pairs with tan_ref_x
    S_y = np.sin(th) * s_x + np.cos(th) * s_y     # pairs with tan_ref_y
    # For per-plane physics (θ≈90°): S_x ≈ −s_y (FEU Y plane), S_y ≈ +s_x (FEU X plane)

    # ---- 5. Direct v_drift estimator: slope of S vs tanθ_ref, per sign ----
    lines = {}
    print('\nDirect v_drift fit  S[µm/ns] = v * tanθ_ref + v*(w/gap)*sign :')
    for name, S, tr in [('X', S_x, tan_rx), ('Y', S_y, tan_ry)]:
        for sign, lo, hi in [('+', 0.06, 0.55), ('-', -0.55, -0.06)]:
            m = (tr > lo) & (tr < hi)
            v, b, ve, n = robust_line(tr[m], S[m])
            lines[(name, sign)] = (v, b, ve, n)
            print(f'  {name}{sign}: v = {v:7.2f} ± {ve:.2f} µm/ns   '
                  f'intercept = {b:+7.2f} µm/ns   (n={n:,})')
    v_mean = np.nanmean([lines[k][0] for k in lines])
    print(f'  mean v_drift = {v_mean:.2f} µm/ns')
    # implied extra width: intercept = ±v*w/gap → w/gap = |b|/v
    wg = np.nanmean([abs(lines[k][1]) for k in lines]) / v_mean
    print(f'  mean |intercept|/v = w_extra/gap = {wg:.4f}')

    # gap estimate from time span (95th pct ~ full-gap crossing incl. jitter)
    t_gap = np.nanmedian(np.concatenate([dur_x, dur_y]))
    print(f'  median cluster time span = {t_gap:.0f} ns  →  gap ≈ v*T = '
          f'{v_mean * t_gap / 1000.0:.1f} mm  →  w_extra ≈ {wg * v_mean * t_gap / 1000.0:.2f} mm')

    # ---- Figures ----
    deg_rx = np.degrees(np.arctan(tan_rx)); deg_ry = np.degrees(np.arctan(tan_ry))

    # 1) profile of θ_det vs θ_ref at the fitted v (per plane) + Δtan vs |θ_ref|
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    bins = np.linspace(-30, 30, 31)
    for j, (name, S, tr, deg_r, col) in enumerate([
            ('X', S_x, tan_rx, deg_rx, 'red'), ('Y', S_y, tan_ry, deg_ry, 'blue')]):
        deg_d = np.degrees(np.arctan(S / v_mean))
        ax = axes[0, j]
        med, lo16, hi84, ctr = [], [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (deg_r >= b0) & (deg_r < b1)
            if m.sum() < 30:
                continue
            q = np.percentile(deg_d[m], [16, 50, 84])
            ctr.append(0.5 * (b0 + b1)); lo16.append(q[0]); med.append(q[1]); hi84.append(q[2])
        ctr = np.array(ctr)
        ax.plot([-30, 30], [-30, 30], 'k--', lw=1, label='y = x')
        ax.fill_between(ctr, lo16, hi84, alpha=0.25, color=col, label='16–84%')
        ax.plot(ctr, med, 'o-', color=col, ms=4, label='median θ_det')
        ax.set_xlabel(f'θ_{name.lower()} reference [deg]')
        ax.set_ylabel(f'θ_{name.lower()} detector [deg]  (v={v_mean:.1f} µm/ns)')
        ax.set_title(f'{name}: median detector angle vs reference')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # Δtan vs |θ_ref|, split by n_strips of the PHYSICAL plane that feeds S
        ns_phys = ns_x if name == 'Y' else ns_y   # θ≈90°: S_x←s_y(FEU-Y), S_y←s_x(FEU-X)
        ax2 = axes[1, j]
        dtan = np.abs(S / v_mean) - np.abs(tr)
        adeg = np.abs(deg_r)
        for nmin, nmax, c in [(4, 6, 'green'), (7, 9, 'darkorange'), (10, 99, 'purple')]:
            mm = (ns_phys >= nmin) & (ns_phys <= nmax)
            med2, ctr2 = [], []
            for b0, b1 in zip(np.arange(0, 30, 2), np.arange(2, 32, 2)):
                m = mm & (adeg >= b0) & (adeg < b1)
                if m.sum() < 30:
                    continue
                ctr2.append(0.5 * (b0 + b1)); med2.append(np.median(dtan[m]))
            ax2.plot(ctr2, med2, 'o-', ms=4, color=c, label=f'{nmin}–{nmax if nmax<99 else "+"} strips')
        ax2.axhline(0, color='k', lw=1)
        ax2.axhline(wg, color='gray', ls=':', lw=1.2, label=f'w/gap = {wg:.3f}')
        ax2.set_xlabel(f'|θ_{name.lower()} reference| [deg]')
        ax2.set_ylabel('median |tanθ_det| − |tanθ_ref|')
        ax2.set_title(f'{name}: additive tan-space excess vs angle, by cluster width')
        ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f'{CFG.RUN} / {CFG.SUB_RUN} — angle-correlation bias study', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, 'bias_profiles.png'), dpi=150)
    plt.close(fig)

    # 2) time span + spatial extent vs |θ_ref| (per physical plane; X plane pairs tan_ry)
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9))
    for j, (plane, dur, ns, adeg) in enumerate([
            ('X plane (FEU %d)' % CFG.MX17_FEU_X, dur_x, ns_x, np.abs(deg_ry)),
            ('Y plane (FEU %d)' % CFG.MX17_FEU_Y, dur_y, ns_y, np.abs(deg_rx))]):
        ax = axes2[0, j]
        h = ax.hist2d(adeg, dur, bins=[30, 60], range=[[0, 30], [0, 1200]],
                      norm=LogNorm(), cmap='viridis')
        plt.colorbar(h[3], ax=ax, label='events')
        med, ctr = [], []
        for b0, b1 in zip(np.arange(0, 30, 2), np.arange(2, 32, 2)):
            m = (adeg >= b0) & (adeg < b1)
            if m.sum() < 30:
                continue
            ctr.append(0.5 * (b0 + b1)); med.append(np.median(dur[m]))
        ax.plot(ctr, med, 'r.-', label='median')
        ax.set_xlabel('|θ_ref| [deg]'); ax.set_ylabel('cluster time span [ns]')
        ax.set_title(f'{plane}: time span vs angle'); ax.legend(fontsize=8)

        ax = axes2[1, j]
        ext = (ns - 1) * PITCH_MM
        h = ax.hist2d(adeg, ext, bins=[30, 40], range=[[0, 30], [0, 30]],
                      norm=LogNorm(), cmap='viridis')
        plt.colorbar(h[3], ax=ax, label='events')
        med, ctr = [], []
        for b0, b1 in zip(np.arange(0, 30, 2), np.arange(2, 32, 2)):
            m = (adeg >= b0) & (adeg < b1)
            if m.sum() < 30:
                continue
            ctr.append(0.5 * (b0 + b1)); med.append(np.median(ext[m]))
        ax.plot(ctr, med, 'r.-', label='median extent')
        gap_mm = v_mean * t_gap / 1000.0
        th_line = np.linspace(0, 30, 61)
        ax.plot(th_line, gap_mm * np.tan(np.radians(th_line)), 'w--', lw=1.5,
                label=f'geometric: gap·tanθ (gap={gap_mm:.1f} mm)')
        ax.set_xlabel('|θ_ref| [deg]'); ax.set_ylabel('cluster spatial extent [mm]')
        ax.set_title(f'{plane}: spatial extent vs angle'); ax.legend(fontsize=8)
    fig2.suptitle(f'{CFG.RUN} / {CFG.SUB_RUN} — cluster geometry vs track angle', fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(os.path.join(out_dir, 'cluster_geometry.png'), dpi=150)
    plt.close(fig2)

    # 3) corrected correlation at the ridge-fit v (sanity: ridge should be parallel,
    #    offset ±v*w/gap in tan space)
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, S, tr, name in [(axes3[0], S_x, tan_rx, 'X'), (axes3[1], S_y, tan_ry, 'Y')]:
        deg_d = np.degrees(np.arctan(S / v_mean))
        deg_r = np.degrees(np.arctan(tr))
        m = (np.abs(deg_d) < 40) & (np.abs(deg_r) < 40)
        h = ax.hist2d(deg_d[m], deg_r[m], bins=120, norm=LogNorm(), cmap='viridis', cmin=1)
        ax.plot([-40, 40], [-40, 40], 'r--', lw=1)
        # predicted ridge: tan_det = tan_ref ± w/gap
        tt = np.linspace(-40, 40, 200)
        pred = np.degrees(np.arctan(np.tan(np.radians(tt)) + np.sign(tt) * wg))
        ax.plot(pred, tt, 'w:', lw=1.5, label='tanθ_det = tanθ_ref ± w/gap')
        ax.set_xlabel(f'θ_{name.lower()} detector [deg] (v={v_mean:.1f})')
        ax.set_ylabel(f'θ_{name.lower()} reference [deg]')
        ax.set_title(f'{name} correlation + additive-offset prediction')
        ax.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, 'correlation_with_prediction.png'), dpi=150)
    plt.close(fig3)

    print(f'\nPlots in {out_dir}')


if __name__ == '__main__':
    main()
