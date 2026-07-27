#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46_vdrift_ref_metric_scan.py — back-to-basics drift-velocity scan against the
M3 reference track, on the RAW hits.

Premise (user request, 7-17): take the raw strip hits of each matched muon,
place them in space with z = (t - t0) * v for a GUESSED drift velocity v, and
ask how close the charge cluster comes to the independent M3 reference line,
using a metric that carries the reference-track uncertainty. Then scan v and
see whether the data picks a minimum, and how well clusters agree there.
Drift gap assumed 29 mm (user: "29 cm" read as 2.9 cm; mechanical gap 30 mm).

Geometry (identical to the 3-D displays):
    z_i(v)   = (t_i - t0) * v / 1000          [mm], t0 = earliest cluster hit
    ref_a(z) = ref_mesh_a + z * tan_a_raw     a in {x, y}, tangents rotated
                                              into the raw strip frame
    d_i(v)   = pos_i - ref_a(z_i(v))          distance in the measured axis

Metric (per hit -> per cluster):
    pull_i = d_i / sqrt(sig_ref_a^2 + (sig_slope*z)^2 + sig_hit(z)^2)
    cluster closeness chi_c = sqrt(mean pull^2) over the cluster's strips
  sig_ref  : M3 pointing at the DUT plane (m3_self_resolution/results.json,
             rotated to raw frame; ~0.21/0.24 mm)
  sig_slope: M3 slope error ~0.35/0.45 mrad (X/Y) — <15 um over the gap
  sig_hit  : measured depth-resolved core width of this same data,
             sig_hit(z) ~ 0.9 + 0.06 z mm (threading report §3)

Scan objectives (v = 4..60 um/ns):
    J_med   (v) = median |d|            all hits, ANCHORED (no free params)
    J_pull  (v) = mean min(pull^2, 25)  all hits, ANCHORED
    J_float (v) = median |d - <d>_c|    per-cluster offset floated (slope-only)
  computed for ALL events and for the INCLINED subset (|tan_ref| > 0.08 in the
  hit's own axis) — vertical tracks carry no v information by construction.

Outputs -> CFG.out_dir('vdrift_ref_metric_scan'):
    scan_curves.png / scan_curves.csv      the v scan + gap overlay
    per_event_v.png                        per-cluster floated v* distribution
    agreement_at_min.png                   pull distributions at the minimum
    event_<eid>_vcompare.png (x3)          hits + ref band at 3 velocities
    unshared_scan.png                      same scan, raw vs unshared subset

Usage (from mx_june_cosmic_qa/):
    ../.venv/bin/python 46_vdrift_ref_metric_scan.py [--res-cut 6]
        [--displays] [--unshared N] [--no-scan]
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'engineer_package'))
from qa_config import get_config, setup_paths            # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                   # noqa: E402
from make_event_displays import (                        # noqa: E402
    CFG, VETO, load_hits, largest_cluster, CLUSTER_GAP_MM)
from make_event_displays_3d import (                     # noqa: E402
    load_full_reference, REF_SIGMA_ALIGNED_X_MM, REF_SIGMA_ALIGNED_Y_MM)

GAP_MM = 29.0                     # user-specified working drift gap
V_SCAN = np.arange(4.0, 60.01, 0.25)
TAN_MIN = 0.08                    # "inclined" threshold (~4.6 deg) per axis
SIG_SLOPE = {'x': 0.00035, 'y': 0.00045}   # M3 slope error [rad], raw frame
SIG_HIT_0, SIG_HIT_K = 0.9, 0.06  # sig_hit(z) = 0.9 + 0.06 z  [mm]
V_NOMINAL = 34.0                  # geometry/unshared/Magboltz consensus
OUT = CFG.out_dir('vdrift_ref_metric_scan')


# ---------------------------------------------------------------- flat table
def build_hit_table(hits, ref, by_eid, best, res_cut):
    """One row per raw cluster hit of every matched event.

    Columns: eid, plane, pos, trel [ns], tan (ref tangent, this axis, raw
    frame), mesh (ref anchor, this axis), sig_ref (pointing sigma, this axis).
    """
    sx, sy = cm.ref_sigma_raw_frame(best, REF_SIGMA_ALIGNED_X_MM,
                                    REF_SIGMA_ALIGNED_Y_MM)
    eids = [e for e in ref
            if np.isfinite(by_eid[e].radial_residual_mm)
            and by_eid[e].radial_residual_mm <= res_cut]
    hv = hits[hits['nrows'] <= VETO]
    hv = hv[hv['eventId'].isin(set(eids))]
    rows = []
    for eid, g in hv.groupby('eventId'):
        r = by_eid[int(eid)]
        tanx, tany = cm._rotate_ref_tangents(r, best)
        planes = {}
        for pl, col in (('x', 'x_position_mm'), ('y', 'y_position_mm')):
            gp = g[g[col].notna()]
            if len(gp) < 3:
                continue
            pos = gp[col].to_numpy()
            idx = largest_cluster(pos, gap=CLUSTER_GAP_MM)
            planes[pl] = (pos[idx], gp['time'].to_numpy()[idx],
                          gp['amplitude'].to_numpy()[idx])
        if 'x' not in planes or 'y' not in planes:
            continue
        t0 = min(planes['x'][1].min(), planes['y'][1].min())
        for pl, (pos, t, amp) in planes.items():
            tan = tanx if pl == 'x' else tany
            mesh = r.ref_mesh_x_mm if pl == 'x' else r.ref_mesh_y_mm
            sref = sx if pl == 'x' else sy
            for p, ti, a in zip(pos, t, amp):
                rows.append((int(eid), pl, p, ti - t0, tan, mesh, sref, a))
    df = pd.DataFrame(rows, columns=['eid', 'plane', 'pos', 'trel', 'tan',
                                     'mesh', 'sig_ref', 'amp'])
    df['core'] = df['amp'] >= 0.30 * df.groupby(
        [df['eid'], df['plane']])['amp'].transform('max')
    print(f'hit table: {len(df):,} hits ({df.core.mean():.0%} core), '
          f'{df.eid.nunique():,} events (res_cut {res_cut} mm)')
    return df


def residuals(df, v):
    """d [mm] and z [mm] for all hits at drift velocity v [um/ns]."""
    z = df['trel'].to_numpy() * v / 1000.0
    d = df['pos'].to_numpy() - (df['mesh'].to_numpy()
                                + z * df['tan'].to_numpy())
    return d, z


def pulls(df, v):
    d, z = residuals(df, v)
    sig_sl = np.where(df['plane'].to_numpy() == 'x',
                      SIG_SLOPE['x'], SIG_SLOPE['y'])
    sig = np.sqrt(df['sig_ref'].to_numpy() ** 2 + (sig_sl * z) ** 2
                  + (SIG_HIT_0 + SIG_HIT_K * np.abs(z)) ** 2)
    return d / sig, d, z, sig


# --------------------------------------------------------------------- scans
def scan(df):
    """Return DataFrame of the three objectives vs v, all + inclined."""
    incl = np.abs(df['tan'].to_numpy()) > TAN_MIN
    # cluster ids for the floated variant
    cid = df['eid'].astype(str) + '_' + df['plane']
    cid = cid.astype('category').cat.codes.to_numpy()
    order = np.argsort(cid)
    counts = np.bincount(cid)
    rec = []
    for v in V_SCAN:
        p, d, z, _ = pulls(df, v)
        # per-cluster demean (floating offset)
        sums = np.bincount(cid, weights=d)
        dfloat = d - (sums / counts)[cid]
        rec.append(dict(
            v=v,
            j_med=np.median(np.abs(d)),
            j_pull=np.mean(np.minimum(p ** 2, 25.0)),
            j_float=np.median(np.abs(dfloat)),
            j_med_incl=np.median(np.abs(d[incl])),
            j_pull_incl=np.mean(np.minimum(p[incl] ** 2, 25.0)),
            j_float_incl=np.median(np.abs(dfloat[incl])),
            f_over_gap=np.mean(z > GAP_MM),
        ))
    return pd.DataFrame(rec)


def per_cluster_v(df, direction='xt'):
    """Floating-offset per-cluster best v divided by the reference tangent
    (inclined clusters only).  direction 'xt' = OLS pos-on-time (dx/dt);
    't x' = OLS time-on-pos, inverted (1/(dt/dx)) — the two differ by
    regression dilution when the per-strip times are noisy."""
    out = []
    for (eid, pl), g in df.groupby(['eid', 'plane']):
        tan = g['tan'].iloc[0]
        if abs(tan) < TAN_MIN or len(g) < 4 or np.ptp(g['trel']) <= 0 \
                or np.ptp(g['pos']) <= 0:
            continue
        if direction == 'xt':
            s = np.polyfit(g['trel'], g['pos'], 1)[0]      # mm/ns
        else:
            m = np.polyfit(g['pos'], g['trel'], 1)[0]      # ns/mm
            if m == 0:
                continue
            s = 1.0 / m
        out.append(dict(eid=eid, plane=pl, v=1000.0 * s / tan,
                        tan=tan, n=len(g)))
    return pd.DataFrame(out)


def gap_filling_v(df):
    """v that would stretch the median inclined-track time span to GAP_MM."""
    spans = df.groupby('eid')['trel'].agg(np.ptp)
    incl = df.groupby('eid')['tan'].agg(lambda t: np.abs(t).max())
    spans = spans[incl > TAN_MIN]
    t_med = spans.median()
    return GAP_MM * 1000.0 / t_med, t_med


# --------------------------------------------------------------------- plots
def plot_scan(sc, v_star, extras, out=OUT):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for a, suff, ttl in ((ax[0], '', 'all matched hits'),
                         (ax[1], '_incl', f'inclined only (|tan ref| > {TAN_MIN})')):
        a2 = a.twinx()
        a.plot(sc.v, sc['j_med' + suff], '-', color='#c0392b', lw=2,
               label='anchored  median |d|  [mm]')
        a.plot(sc.v, sc['j_float' + suff], '-', color='#2e86c1', lw=2,
               label='offset-floated  median |d|  [mm]')
        a2.plot(sc.v, sc['j_pull' + suff], '--', color='#7d3c98', lw=1.6,
                label='anchored  mean pull$^2$ (capped)')
        vm = sc.v[np.argmin(sc['j_float' + suff])]
        a.axvline(vm, color='#2e86c1', ls=':', lw=1.4)
        a.axvline(V_NOMINAL, color='#1a9850', ls='--', lw=1.6)
        a.axvline(extras['v_gap'], color='#888', ls='-.', lw=1.4)
        a.annotate(f'scan min {vm:.1f}', xy=(vm, a.get_ylim()[1]),
                   xytext=(vm + 0.5, np.max(sc['j_med' + suff]) * 0.96),
                   color='#2e86c1', fontsize=10)
        a.annotate('v$_{geom}$ 34', xy=(V_NOMINAL, 0), xytext=(V_NOMINAL + 0.5,
                   np.max(sc['j_med' + suff]) * 0.88),
                   color='#1a9850', fontsize=10)
        a.annotate(f'gap-filling {extras["v_gap"]:.0f}',
                   xy=(extras['v_gap'], 0), xytext=(extras['v_gap'] + 0.5,
                   np.max(sc['j_med' + suff]) * 0.80), color='#888', fontsize=10)
        a.set_xlabel('drift velocity  v  [µm/ns]')
        a.set_ylabel('median |distance to reference line|  [mm]')
        a2.set_ylabel('mean capped pull²')
        a.set_title(ttl, fontsize=12)
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = a2.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper right')
    fig.suptitle('Raw hits vs M3 reference: drift-velocity scan '
                 f'(det3 sat long run, 1000 V, gap {GAP_MM:.0f} mm assumed)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, 'scan_curves.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}')


def plot_per_event(pev, out=OUT):
    fig, ax = plt.subplots(figsize=(8, 5))
    good = pev.v[(pev.v > -20) & (pev.v < 100)]
    ax.hist(good, bins=120, color='#666')
    med = np.median(good)
    ax.axvline(med, color='#c0392b', lw=2, label=f'median {med:.1f} µm/ns')
    ax.axvline(V_NOMINAL, color='#1a9850', ls='--', lw=2, label='v_geom 34')
    ax.set_xlabel('per-cluster floated v* = (dpos/dt) / tan_ref  [µm/ns]')
    ax.set_ylabel('clusters')
    ax.set_title(f'Per-cluster raw-ladder velocity, inclined clusters '
                 f'(N={len(good):,})')
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out, 'per_event_v.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}  (median {med:.2f}, p16/p84 '
          f'{np.percentile(good, 16):.1f}/{np.percentile(good, 84):.1f})')
    return med


def plot_agreement(df, v_min, out=OUT):
    """Pull distributions + per-cluster closeness at the scan minimum and at
    v_geom=34, with full and ref-only normalisation."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for v, colr in ((v_min, '#2e86c1'), (V_NOMINAL, '#1a9850')):
        p, d, z, sig = pulls(df, v)
        ax[0].hist(np.clip(p, -8, 8), bins=160, histtype='step', lw=1.8,
                   color=colr, density=True,
                   label=f'v={v:.1f}: med|pull| {np.median(np.abs(p)):.2f}')
        chi = pd.DataFrame(dict(cid=df.eid.astype(str) + df.plane, p2=p ** 2)) \
            .groupby('cid')['p2'].mean() ** 0.5
        ax[1].hist(np.clip(chi, 0, 6), bins=120, histtype='step', lw=1.8,
                   color=colr, density=True,
                   label=(f'v={v:.1f}: med {np.median(chi):.2f}, '
                          f'frac<1 {np.mean(chi < 1):.0%}'))
        pref = d / df['sig_ref'].to_numpy()
        ax[2].hist(np.clip(np.abs(pref), 0, 30), bins=120, histtype='step',
                   lw=1.8, color=colr, density=True,
                   label=f'v={v:.1f}: med {np.median(np.abs(pref)):.1f}σ_ref')
    ax[0].set_xlabel('per-hit pull  d / σ(total)')
    ax[0].set_title('per-hit pulls (full σ: ref ⊕ hit(z))')
    ax[1].set_xlabel('per-cluster closeness  √(mean pull²)')
    ax[1].set_title('cluster agreement metric')
    ax[2].set_xlabel('|d| / σ_ref   (reference uncertainty only)')
    ax[2].set_title('distance in units of M3 pointing σ (~0.2 mm)')
    for a in ax:
        a.legend(fontsize=9)
    fig.suptitle('Cluster-vs-reference agreement at the scan minimum and at '
                 'v_geom', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, 'agreement_at_min.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}')


def plot_event_vcompare(df, by_eid, best, eid, v_list, out=OUT):
    """One event: x-z and y-z views of the raw cluster hits with the reference
    line ± its pointing band, drawn at each candidate velocity."""
    g = df[df.eid == eid]
    fig, ax = plt.subplots(2, len(v_list), figsize=(4.6 * len(v_list), 8.4),
                           sharey='row')
    r = by_eid[int(eid)]
    for j, v in enumerate(v_list):
        for i, pl in enumerate(('x', 'y')):
            a = ax[i, j]
            gp = g[g.plane == pl]
            z = gp['trel'].to_numpy() * v / 1000.0
            pos = gp['pos'].to_numpy()
            tan, mesh = gp['tan'].iloc[0], gp['mesh'].iloc[0]
            sref = gp['sig_ref'].iloc[0]
            zz = np.linspace(0, max(GAP_MM, z.max() * 1.05), 50)
            a.fill_betweenx(zz, mesh + zz * tan - sref, mesh + zz * tan + sref,
                            color='#1a9850', alpha=0.25, lw=0)
            a.plot(mesh + zz * tan, zz, color='#1a9850', lw=2)
            sc = a.scatter(pos, z, c=gp['amp'], cmap='viridis', s=55,
                           edgecolors='k', linewidths=0.4, zorder=4)
            a.axhline(0, color='k', lw=0.8)
            a.axhline(GAP_MM, color='#888', lw=1.2, ls='-.')
            p_, d_, _, _ = pulls(gp, v)
            if i == 0:
                a.set_title(f'v = {v:.1f} µm/ns\n'
                            f'x: √mean pull² = {np.sqrt(np.mean(p_**2)):.2f}',
                            fontsize=11)
            else:
                a.set_title(f'y: √mean pull² = {np.sqrt(np.mean(p_**2)):.2f}',
                            fontsize=10)
            a.set_xlabel(f'{pl} strip position [mm]')
            if j == 0:
                a.set_ylabel('drift depth z [mm]')
    fig.suptitle(f'event {eid} — raw hits vs M3 reference (green ±1σ '
                 f'pointing), gap {GAP_MM:.0f} mm dash-dot', fontsize=13,
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, f'event_{eid}_vcompare.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}')


def sota_hit_table(eids, by_eid, best, det, do_unshare):
    """46-format hit table from the waveform chain (sota_reco), raw or
    unshared, for the given events."""
    import sota_reco as sota
    from make_event_displays_3d import (REF_SIGMA_ALIGNED_X_MM as SRX,
                                        REF_SIGMA_ALIGNED_Y_MM as SRY)
    sx, sy = cm.ref_sigma_raw_frame(best, SRX, SRY)
    h = sota.sota_hits(eids, CFG, det, do_unshare=do_unshare)
    rows = []
    for eid, g in h.groupby('eventId'):
        r = by_eid[int(eid)]
        tanx, tany = cm._rotate_ref_tangents(r, best)
        t0 = g['time'].min()
        for pl, col in (('x', 'x_position_mm'), ('y', 'y_position_mm')):
            gp = g[g.plane == pl]
            if len(gp) < 3:
                continue
            tan = tanx if pl == 'x' else tany
            mesh = r.ref_mesh_x_mm if pl == 'x' else r.ref_mesh_y_mm
            sref = sx if pl == 'x' else sy
            for p, ti, a in zip(gp[col], gp['time'], gp['amplitude']):
                rows.append((int(eid), pl, p, ti - t0, tan, mesh, sref, a))
    return pd.DataFrame(rows, columns=['eid', 'plane', 'pos', 'trel', 'tan',
                                       'mesh', 'sig_ref', 'amp'])


def plot_recon_compare(df_raw, by_eid, best, det, eid, v_scan_min, out=OUT):
    """One event, three reconstructions: raw hits at the raw scan minimum,
    raw hits at v_geom, unshared hits at v_geom.  The honest three-way: raw@41
    'threads by construction' (v stretched to fit), unshared@34 threads with
    the physical velocity and stops at the visible column."""
    cols = [('RAW hits\nv = %.1f (raw scan min)' % v_scan_min,
             df_raw[df_raw.eid == eid], v_scan_min, '#555555'),
            ('RAW hits\nv = 34 (physical)', df_raw[df_raw.eid == eid],
             V_NOMINAL, '#555555'),
            ('UNSHARED hits\nv = 34 (physical)',
             sota_hit_table([eid], by_eid, best, det, True), V_NOMINAL,
             '#c0392b')]
    fig, ax = plt.subplots(2, 3, figsize=(13.8, 8.6), sharey='row')
    for j, (title, tab, v, _c) in enumerate(cols):
        g = tab[tab.eid == eid]
        for i, pl in enumerate(('x', 'y')):
            a = ax[i, j]
            gp = g[g.plane == pl]
            if len(gp) == 0:
                continue
            z = gp['trel'].to_numpy() * v / 1000.0
            tan, mesh = gp['tan'].iloc[0], gp['mesh'].iloc[0]
            sref = gp['sig_ref'].iloc[0]
            zz = np.linspace(0, GAP_MM * 1.05, 50)
            a.fill_betweenx(zz, mesh + zz * tan - sref, mesh + zz * tan + sref,
                            color='#1a9850', alpha=0.25, lw=0)
            a.plot(mesh + zz * tan, zz, color='#1a9850', lw=2)
            a.scatter(gp['pos'], z, c=gp['amp'], cmap='viridis', s=55,
                      edgecolors='k', linewidths=0.4, zorder=4)
            a.axhline(0, color='k', lw=0.8)
            a.axhline(GAP_MM, color='#888', lw=1.2, ls='-.')
            p_, _, _, _ = pulls(gp, v)
            a.set_title(f'{title if i == 0 else ""}\n{pl}: √mean pull² = '
                        f'{np.sqrt(np.mean(p_ ** 2)):.2f}', fontsize=10)
            a.set_xlabel(f'{pl} strip position [mm]')
            if j == 0:
                a.set_ylabel('drift depth z [mm]')
    fig.suptitle(f'event {eid} — three reconstructions vs the M3 reference '
                 f'(green ±1σ pointing; {GAP_MM:.0f} mm gap dash-dot)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, f'event_{eid}_recon_compare.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}')


# ----------------------------------------------------------------- unshared
def unshared_scan(df_raw, ref, by_eid, best, det, n_events, res_cut):
    """Repeat the floated scan on UNSHARED hits for a subset of inclined
    events, against the raw hits of the same events."""
    import sota_reco as sota
    sx, sy = cm.ref_sigma_raw_frame(best, REF_SIGMA_ALIGNED_X_MM,
                                    REF_SIGMA_ALIGNED_Y_MM)
    # most-inclined matched events first: more v information per event
    cand = [(abs(by_eid[e].ref_tan_theta_x) + abs(by_eid[e].ref_tan_theta_y), e)
            for e in ref if np.isfinite(by_eid[e].radial_residual_mm)
            and by_eid[e].radial_residual_mm <= res_cut]
    eids = [e for _, e in sorted(cand, reverse=True)[:n_events]]
    print(f'unshared scan: {len(eids)} most-inclined events, loading waveforms...')
    wf = sota.load_waveforms(eids, CFG, det)

    def table(do_unshare):
        h = sota.sota_hits(eids, CFG, det, do_unshare=do_unshare, wf_cache=wf)
        rows = []
        for eid, g in h.groupby('eventId'):
            r = by_eid[int(eid)]
            tanx, tany = cm._rotate_ref_tangents(r, best)
            t0 = g['time'].min()
            for pl, col in (('x', 'x_position_mm'), ('y', 'y_position_mm')):
                gp = g[g.plane == pl]
                if len(gp) < 3:
                    continue
                tan = tanx if pl == 'x' else tany
                mesh = r.ref_mesh_x_mm if pl == 'x' else r.ref_mesh_y_mm
                sref = sx if pl == 'x' else sy
                for p, ti, a in zip(gp[col], gp['time'], gp['amplitude']):
                    rows.append((int(eid), pl, p, ti - t0, tan, mesh, sref, a))
        return pd.DataFrame(rows, columns=['eid', 'plane', 'pos', 'trel',
                                           'tan', 'mesh', 'sig_ref', 'amp'])

    curves = {}
    for name, tab in (('raw (wf re-extract)', table(False)),
                      ('unshared', table(True)),
                      ('raw (production hits)',
                       df_raw[df_raw.eid.isin(set(eids))])):
        sc = scan(tab)
        curves[name] = sc
        vm = sc.v[np.argmin(sc['j_float_incl'])]
        med = np.median(per_cluster_v(tab).v)
        print(f'  {name:24s}: floated-scan min {vm:5.1f}  '
              f'per-cluster median v* {med:5.1f}')
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    colors = {'raw (wf re-extract)': '#888888', 'unshared': '#c0392b',
              'raw (production hits)': '#333333'}
    for name, sc in curves.items():
        j = sc['j_float_incl'] / sc['j_float_incl'].min()
        ax.plot(sc.v, j, lw=2, color=colors[name],
                ls='--' if 'production' in name else '-',
                label=f'{name}  (min {sc.v[np.argmin(j)]:.1f})')
    ax.axvline(V_NOMINAL, color='#1a9850', ls='--', lw=2, label='v_geom 34')
    ax.set_xlabel('drift velocity v [µm/ns]')
    ax.set_ylabel('offset-floated median |d|, normalised to min')
    ax.set_xlim(4, 60)
    ax.set_title(f'Same scan, raw vs unshared hits ({len(eids)} inclined events)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, 'unshared_scan.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  -> {p}')


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res-cut', type=float, default=6.0,
                    help='max radial ref match [mm] for the population')
    ap.add_argument('--displays', action='store_true',
                    help='render per-event velocity-comparison displays')
    ap.add_argument('--unshared', type=int, default=0, metavar='N',
                    help='also run the scan on unshared hits for N events')
    ap.add_argument('--recon-displays', action='store_true',
                    help='three-way per-event displays raw@scanmin / raw@34 / '
                         'unshared@34 (needs decoded_root waveforms)')
    ap.add_argument('--no-scan', action='store_true')
    args = ap.parse_args()

    results, best, ref, by_eid = load_full_reference()
    hits, det = load_hits()
    df = build_hit_table(hits, ref, by_eid, best, args.res_cut)

    if not args.no_scan:
        sc = scan(df)
        v_gap, t_med = gap_filling_v(df)
        vm_anch = sc.v[np.argmin(sc.j_med_incl)]
        vm_float = sc.v[np.argmin(sc.j_float_incl)]
        vm_pull = sc.v[np.argmin(sc.j_pull_incl)]
        print('\n================ SCAN RESULT (raw hits) ================')
        print(f'  anchored  median|d| minimum : v = {vm_anch:.2f} um/ns')
        print(f'  floated   median|d| minimum : v = {vm_float:.2f} um/ns')
        print(f'  anchored  pull^2    minimum : v = {vm_pull:.2f} um/ns')
        print(f'  gap-filling velocity (29 mm / median inclined t-span '
              f'{t_med:.0f} ns): {v_gap:.1f} um/ns')
        print(f'  consensus physics value      : v = {V_NOMINAL} um/ns')
        d_, z_ = residuals(df, vm_float)
        print(f'  at scan min: frac hits above gap {np.mean(z_ > GAP_MM):.2%}')
        # core-strip variant: drop the low-amplitude RC-skirt strips whose
        # times prior work flagged as corrupted, same metric
        sc_core = scan(df[df.core])
        print(f'  CORE strips only ({df.core.mean():.0%} of hits):')
        print(f'    anchored median|d| min {sc_core.v[np.argmin(sc_core.j_med_incl)]:.2f}, '
              f'floated {sc_core.v[np.argmin(sc_core.j_float_incl)]:.2f}, '
              f'pull^2 {sc_core.v[np.argmin(sc_core.j_pull_incl)]:.2f} um/ns')
        sc_core.to_csv(os.path.join(OUT, 'scan_curves_core.csv'), index=False)
        for tag, sub in (('all strips', df), ('core strips', df[df.core])):
            for direction in ('xt', 'tx'):
                pv = per_cluster_v(sub, direction)
                print(f'    per-cluster v* [{tag}, {"pos-on-t" if direction=="xt" else "t-on-pos inv"}]: '
                      f'median {np.median(pv.v):5.1f}  '
                      f'(p16/p84 {np.percentile(pv.v,16):5.1f}/{np.percentile(pv.v,84):5.1f})')
        plot_scan(sc, vm_float, dict(v_gap=v_gap))
        sc.to_csv(os.path.join(OUT, 'scan_curves.csv'), index=False)
        med = plot_per_event(per_cluster_v(df))
        plot_agreement(df, vm_float)

        if args.displays or args.recon_displays:
            # three most-inclined tight-match events with decent clusters
            pool = []
            for eid in df.eid.unique():
                r = by_eid[int(eid)]
                t = max(abs(df[df.eid == eid]['tan'].max()),
                        abs(df[df.eid == eid]['tan'].min()))
                n = len(df[df.eid == eid])
                if r.radial_residual_mm < 1.0 and n >= 14 and t > 0.25:
                    pool.append((r.radial_residual_mm, eid))
            for _, eid in sorted(pool)[:3]:
                if args.displays:
                    plot_event_vcompare(df, by_eid, best, eid,
                                        [vm_float, V_NOMINAL, v_gap])
                if args.recon_displays:
                    plot_recon_compare(df, by_eid, best, det, eid, vm_float)

    if args.unshared:
        unshared_scan(df, ref, by_eid, best, det,
                      args.unshared, args.res_cut)

    print(f'\nall outputs in {OUT}')


if __name__ == '__main__':
    main()
