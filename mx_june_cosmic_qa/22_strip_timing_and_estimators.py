#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_strip_timing_and_estimators.py

Understand HOW the resistive-strip time spreading distorts the micro-TPC
track fit, and find an estimator that recovers the geometry-based velocity.

Background (20/21_*.py): the production per-strip fit (amplitude-weighted,
ANCHORED at the earliest hit, using the pulse-start `time`) gives
v_ridge = 28 um/ns, while the geometry-only extent-slope estimator gives
v_geom = 33.9 +/- 0.3 um/ns at 1000 V. Suspect: RC charge spreading on the
resistive strips makes low-amplitude skirt strips at the cluster's spatial
ends LATE (delayed copies), adding time span without geometric lever arm.

Parts
-----
 A. Pulse-structure event displays: strips as [left,right] time-over-threshold
    bars vs strip position, colored by amplitude, for reference tracks at
    ~0 / ~15 / ~30 deg. (No raw waveforms on disk; ToT bars + start/peak
    markers are the next best thing.)
 B. Skirt timing: cluster core (amp >= 30% of max) vs skirt strips; skirt
    time residuals from the core line, vs distance beyond the core edge.
 C. Core time-walk: time residual vs relative amplitude within the core.
 D. Estimator shoot-out: per-event slope refit with 6 variants, then the
    detector-frame ridge fit for each -> v per variant, compared to v_geom.

Usage: ../.venv/bin/python 22_strip_timing_and_estimators.py sat_det3 [--veto=50]
Output: <out>/alignment_tpc_veto50/bias_study/
        strip_timing_displays.png, skirt_timing.png, estimator_shootout.png
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
MIN_STRIPS = 4
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
GAP_MM_CLUSTER = getattr(cm, 'GAP_THRESHOLD_MM', 2.0)
CORE_FRAC = 0.30
V_GEOM = 33.9   # from 21_geometry_vdrift_scan.py (long run, weighted planes)

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')


def robust_line(x, y, n_iter=4, clip=3.0):
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


def largest_cluster(pos, order=None):
    """Indices of the largest spatial cluster (gap threshold like production)."""
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > GAP_MM_CLUSTER)[0]
    groups = np.split(o, breaks + 1)
    return max(groups, key=len)


# ---------- variant slope estimators (return dx/dt in mm/ns or nan) ----------
def slope_prod(pos, t, amp, **kw):
    """Production: amplitude-weighted, anchored at earliest hit."""
    i0 = np.argmin(t)
    dx, dt = pos - pos[i0], t - t[i0]
    w = amp + 1e-9
    den = np.sum(w * dx * dx)
    if den <= 0:
        return np.nan
    m = np.sum(w * dx * dt) / den            # ns/mm
    return 1.0 / m if m != 0 else np.nan


def slope_ols(pos, t, amp, **kw):
    if len(pos) < 3 or np.ptp(pos) == 0:
        return np.nan
    m = np.polyfit(pos, t, 1)[0]
    return 1.0 / m if m != 0 else np.nan


def slope_lead(pos, t, amp, tlead=None, **kw):
    return slope_ols(pos, tlead, amp)


def slope_peak(pos, t, amp, tpeak=None, **kw):
    return slope_ols(pos, tpeak, amp)


def slope_core(pos, t, amp, **kw):
    m = amp >= CORE_FRAC * amp.max()
    if m.sum() < 3:
        return np.nan
    return slope_ols(pos[m], t[m], amp[m])


def slope_endpoints(pos, t, amp, **kw):
    i_lo, i_hi = np.argmin(t), np.argmax(t)
    dt = t[i_hi] - t[i_lo]
    if dt <= 0:
        return np.nan
    return (pos[i_hi] - pos[i_lo]) / dt


VARIANTS = {
    'prod (anchored, amp-wt, t_start)': slope_prod,
    'OLS unweighted, t_start': slope_ols,
    'OLS, leading edge': slope_lead,
    'OLS, peak time': slope_peak,
    f'OLS, core only (>{CORE_FRAC:.0%} max)': slope_core,
    'endpoints Δx/Δt': slope_endpoints,
}


def main():
    cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)

    th = np.deg2rad(best.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    print(f'{len(ref):,} matched events')

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate(
        [f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'time',
                     'time_of_max', 'left_sample', 'right_sample'], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)]
    hpe = df.groupby('eventId')['eventId'].transform('size')
    df = df[(hpe <= VETO) & df['eventId'].isin(ref)].copy()
    df = cm._map_strip_positions(df, det)
    df['t_lead'] = df['left_sample'] * SAMPLE_NS
    df['t_right'] = df['right_sample'] * SAMPLE_NS

    planes = {'x': ('x_position_mm', 0), 'y': ('y_position_mm', 1)}

    # ================= A. event displays =================
    bands = [('~0°', 0.00, 0.04), ('~15°', 0.24, 0.30), ('~30°', 0.52, 0.62)]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13))
    picked = {b[0]: [] for b in bands}
    for eid, g in df.groupby('eventId'):
        tx, _ = ref[eid]
        gx = g[g['x_position_mm'].notna()]
        if len(gx) < 5:
            continue
        for bname, lo, hi in bands:
            if lo <= abs(tx) < hi and len(picked[bname]) < 3:
                picked[bname].append((eid, gx, tx))
        if all(len(v) >= 3 for v in picked.values()):
            break
    norm = mcolors.LogNorm(vmin=80, vmax=4000)
    for j, (bname, lo, hi) in enumerate(bands):
        for i, (eid, gx, tx) in enumerate(picked[bname]):
            ax = axes[i, j]
            idx = largest_cluster(gx['x_position_mm'].to_numpy())
            gc = gx.iloc[idx].sort_values('x_position_mm')
            for _, s in gc.iterrows():
                c = plt.cm.viridis(norm(max(s['amplitude'], 80)))
                ax.plot([s['x_position_mm']] * 2, [s['t_lead'], s['t_right']],
                        '-', color=c, lw=5, alpha=0.85, solid_capstyle='butt')
                ax.plot(s['x_position_mm'], s['time'], 'o', color='crimson', ms=4, zorder=5)
                ax.plot(s['x_position_mm'], s['time_of_max'], 'x', color='k', ms=4, zorder=5)
            # geometric slope guide through the earliest strip: dt/dx = 1000/(v·tanθ)
            s0 = gc.loc[gc['time'].idxmin()]
            xx = np.linspace(gc['x_position_mm'].min() - 1, gc['x_position_mm'].max() + 1, 10)
            if abs(tx) > 0.02:
                ax.plot(xx, s0['time'] + (xx - s0['x_position_mm']) * 1000.0 / (V_GEOM * tx),
                        '--', color='gray', lw=1.2)
            ax.set_title(f'evt {eid}  tanθ={tx:+.2f} ({bname})', fontsize=9)
            ax.set_xlabel('x strip position [mm]')
            ax.set_ylabel('time [ns]')
            ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    fig.colorbar(sm, ax=axes, label='amplitude [ADC]', shrink=0.7)
    fig.suptitle(f'{CFG.RUN} — X-plane strip pulses: ToT bar [left→right], '
                 '● pulse-start time, × peak time; dashed = geometric slope '
                 f'(v={V_GEOM} µm/ns)', fontsize=12)
    fig.savefig(os.path.join(OUT, 'strip_timing_displays.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('displays done')

    # ============ B/C. skirt + core timing, and D. estimator variants ============
    dt_skirt_lo, dt_skirt_hi, dpos_skirt = [], [], []
    core_relamp, core_resid = [], []
    var_slopes = {k: {'x': [], 'y': []} for k in VARIANTS}
    var_tans = {'x': [], 'y': []}

    for eid, g in df.groupby('eventId'):
        tref = ref[eid]
        for p, (pcol, ti) in planes.items():
            gp = g[g[pcol].notna()]
            if len(gp) < MIN_STRIPS:
                continue
            pos_all = gp[pcol].to_numpy()
            idx = largest_cluster(pos_all)
            if len(idx) < MIN_STRIPS:
                continue
            gc = gp.iloc[idx]
            o = np.argsort(gc[pcol].to_numpy())
            pos = gc[pcol].to_numpy()[o]
            t = gc['time'].to_numpy()[o]
            amp = gc['amplitude'].to_numpy()[o]
            tlead = gc['t_lead'].to_numpy()[o]
            tpeak = gc['time_of_max'].to_numpy()[o]

            # ---- D. variants ----
            var_tans[p].append(tref[ti])
            for name, fn in VARIANTS.items():
                s = fn(pos, t, amp, tlead=tlead, tpeak=tpeak)
                var_slopes[name][p].append(s * 1000.0 if np.isfinite(s) else np.nan)

            # ---- B/C. skirt & core (inclined tracks only) ----
            if not (0.2 < abs(tref[ti]) < 0.55):
                continue
            mcore = amp >= CORE_FRAC * amp.max()
            if mcore.sum() < 4 or (~mcore).sum() < 1:
                continue
            pc = np.polyfit(pos[mcore], t[mcore], 1)
            resid = t - np.polyval(pc, pos)
            core_relamp.append(amp[mcore] / amp.max())
            core_resid.append(resid[mcore])
            lo_edge, hi_edge = pos[mcore].min(), pos[mcore].max()
            for k in np.where(~mcore)[0]:
                d = lo_edge - pos[k] if pos[k] < lo_edge else pos[k] - hi_edge
                if d <= 0:
                    continue
                dpos_skirt.append(d)
                (dt_skirt_lo if pos[k] < lo_edge else dt_skirt_hi).append(resid[k])

    # ---- B/C figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    bins = np.linspace(-400, 700, 56)
    ax.hist(dt_skirt_lo, bins=bins, histtype='step', lw=2, color='tab:blue',
            label=f'skirt BEFORE core (n={len(dt_skirt_lo):,}), med='
                  f'{np.median(dt_skirt_lo):.0f} ns')
    ax.hist(dt_skirt_hi, bins=bins, histtype='step', lw=2, color='tab:red',
            label=f'skirt AFTER core (n={len(dt_skirt_hi):,}), med='
                  f'{np.median(dt_skirt_hi):.0f} ns')
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('strip time − core-line prediction [ns]')
    ax.set_ylabel('skirt strips')
    ax.set_title('skirt-strip time residuals (inclined tracks)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    relamp = np.concatenate(core_relamp) if core_relamp else np.array([])
    cres = np.concatenate(core_resid) if core_resid else np.array([])
    hb = ax.hexbin(relamp, cres, gridsize=40, extent=(0.3, 1.0, -250, 250),
                   norm=mcolors.LogNorm(), cmap='viridis')
    med, ctr = [], []
    for b0 in np.arange(0.3, 1.0, 0.05):
        m = (relamp >= b0) & (relamp < b0 + 0.05)
        if m.sum() > 100:
            ctr.append(b0 + 0.025); med.append(np.median(cres[m]))
    ax.plot(ctr, med, 'r.-', lw=1.5, label='median')
    plt.colorbar(hb, ax=ax, label='core strips')
    ax.axhline(0, color='w', lw=0.8)
    ax.set_xlabel('strip amplitude / cluster max')
    ax.set_ylabel('time residual from core line [ns]')
    ax.set_title('CORE strips: time-walk vs relative amplitude')
    ax.legend(fontsize=9)

    # ---- D figure: ridge fit per variant ----
    ax = axes[2]
    print('\n== Estimator shoot-out (detector-frame ridge, 0.06<|tan|<0.55) ==')
    labels, vvals, verrs = [], [], []
    for name in VARIANTS:
        vs = []
        for p in ('x', 'y'):
            S = np.array(var_slopes[name][p])
            tr = np.array(var_tans[p])
            for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
                m = (tr > lo) & (tr < hi) & np.isfinite(S)
                v, b, ve, n = robust_line(tr[m], S[m])
                if np.isfinite(v) and n > 100:
                    vs.append(v)
        v_m, v_e = np.mean(vs), np.std(vs, ddof=1) / np.sqrt(len(vs))
        labels.append(name); vvals.append(v_m); verrs.append(v_e)
        print(f'  {name:34s} v = {v_m:6.2f} ± {v_e:4.2f} µm/ns   ({len(vs)} fits)')
    ypos = np.arange(len(labels))
    ax.barh(ypos, vvals, xerr=verrs, color='steelblue', alpha=0.8)
    ax.axvline(V_GEOM, color='crimson', lw=2, label=f'v_geom = {V_GEOM} µm/ns')
    ax.axvline(28.1, color='gray', ls='--', lw=1.5, label='old ridge = 28.1')
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('ridge-fit v [µm/ns]')
    ax.set_title('slope-estimator variants vs geometry reference')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(20, 45); ax.grid(alpha=0.3, axis='x')

    fig.suptitle(f'{CFG.RUN} — resistive-strip timing structure & estimator test',
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'skirt_timing_and_estimators.png'), dpi=160,
                bbox_inches='tight')
    print(f'\nPlots in {OUT}')


if __name__ == '__main__':
    main()
