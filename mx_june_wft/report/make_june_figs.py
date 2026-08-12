#!/usr/bin/env python3
"""
make_june_figs.py — the june_grand_qa.pdf per-detector figures, rebuilt on the
waveform-first reconstruction.

The June PDF's per-detector pages were built from hits-chain figures
(efficiency/efficiency_map_sliding.png, efficiency/scatter_within_5mm.png,
alignment_tpc_veto50/angle_correlation_corrected_hist.png, ...). This script
produces the same figure types from the campaign wft tables so the remade
report can follow the PDF layout without quoting the superseded chain:

    <OUT_BASE>/wft/efficiency/ray_hit_miss_list.csv        per-ray accounting
    <OUT_BASE>/wft/efficiency/efficiency_map_sliding.png    within|has_any|rays trio
    <OUT_BASE>/wft/efficiency/efficiency_map_sliding.json
    <OUT_BASE>/wft/efficiency/scatter_within_5mm.png
    <OUT_BASE>/wft/efficiency/efficiency_breakdown_wide.png
    <OUT_BASE>/wft/angles_w0corr/angle_correlation_hist.png (falls back to
        the uncorrected table with a warning if angles_w0corr is absent)

The per-ray accounting replicates 02_efficiency.py exactly (same M3 recipe,
same significance-floor spark tag, same 0.5-99.5 percentile active box, no
cluster cut), so the integrated numbers here cannot drift from the breakdown
JSONs the report quotes. Sliding-kernel conventions follow the hits chain's
12_efficiency_map_sliding.py (kernel 25 mm, 120x120 grid, >=30 rays).

    ../../.venv/bin/python mx_june_wft/report/make_june_figs.py [keys ...]
Default keys = the five June-best runs (the june_grand_qa.pdf pages).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from common.mx17_active_area import draw_outlines, alignment_transform    # noqa: E402
from wft import compat                                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402

JUNE_KEYS = ['g_det3_wknd', 'o22_long_det2', 'g_det6_long', 'g_det7_long', 'g_det4']
KERNEL, GRID, MIN_RAYS, R = 25.0, 120, 30, 5.0


def per_ray_table(cfg):
    """02_efficiency.py's accounting, kept per ray instead of aggregated."""
    align_path = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
    params = cm.load_alignment(align_path)
    table = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    df = compat.load_table(table, max_dropped=None)
    results = compat.as_event_results(df)

    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'significance'], library='pd')
    det_raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    fired = set(int(e) for e in det_raw['eventId'].unique())
    det_lo, det_hi = int(det_raw['eventId'].min()), int(det_raw['eventId'].max())
    mult_raw = det_raw.groupby('eventId').size()
    mult = (cm.apply_significance_floor(det_raw, rel=SIG_REL_FLOOR)
            .groupby('eventId').size().reindex(mult_raw.index).fillna(0).astype(int))
    mult_by_ev = mult.to_dict()

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    evn = [int(v) for v in evn]

    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])

    rows = []
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1
                and ay0 <= y <= ay1):
            continue
        spark = mult_by_ev.get(e, 0) > SPARK_VETO_HITS
        det_x, det_y = reco.get(e, (np.nan, np.nan))
        r_mm = (float(np.hypot(x - det_x, y - det_y))
                if (e in reco and not spark) else np.nan)
        rows.append(dict(event_id=e, x=x, y=y,
                         det_x=det_x, det_y=det_y,
                         spark=spark,
                         has_any=e in fired,
                         within=bool(np.isfinite(r_mm) and r_mm <= R),
                         r_mm=r_mm))
    box = dict(x0=float(ax0), x1=float(ax1), y0=float(ay0), y1=float(ay1))
    return pd.DataFrame(rows), box, params


def sliding_map(x, y, val, x_grid, y_grid, kernel, min_n):
    r2 = kernel ** 2
    eff = np.full((len(x_grid), len(y_grid)), np.nan)
    cnt = np.zeros_like(eff, dtype=int)
    for i, xg in enumerate(x_grid):
        dx2 = (x - xg) ** 2
        for j, yg in enumerate(y_grid):
            mask = (dx2 + (y - yg) ** 2) <= r2
            n = int(mask.sum())
            cnt[i, j] = n
            if n >= min_n:
                eff[i, j] = float(val[mask].mean())
    return eff, cnt


def fig_sliding(cfg, d, box, params):
    x, y = d['x'].to_numpy(float), d['y'].to_numpy(float)
    within = d['within'].to_numpy(float)
    has_any = d['has_any'].to_numpy(float)
    ax0, ax1, ay0, ay1 = box['x0'], box['x1'], box['y0'], box['y1']
    tr = alignment_transform(params)
    pad = KERNEL
    x_grid = np.linspace(ax0 - pad, ax1 + pad, GRID)
    y_grid = np.linspace(ay0 - pad, ay1 + pad, GRID)
    eff_w, cnt = sliding_map(x, y, within, x_grid, y_grid, KERNEL, MIN_RAYS)
    eff_a, _ = sliding_map(x, y, has_any, x_grid, y_grid, KERNEL, MIN_RAYS)

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    rect = dict(xy=(ax0, ay0), width=ax1 - ax0, height=ay1 - ay0)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgrey')
    cmap_c = plt.get_cmap('plasma').copy(); cmap_c.set_bad('lightgrey')
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, data, label in [(axes[0], eff_w, f'efficiency (reco within {R:g} mm)'),
                            (axes[1], eff_a, 'has_any (fired any strip)')]:
        im = ax.imshow(data.T, origin='lower', extent=extent, aspect='equal',
                       cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.add_patch(plt.Rectangle(**rect, fill=False, ec='black', lw=1.3))
        draw_outlines(ax, transform=tr, det_name=cfg.DET_NAME)
        ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]')
        ax.set_title(f'{cfg.DET_NAME}  {label}\nsliding kernel r={KERNEL:.1f} mm')
    cnt_m = np.where(cnt >= MIN_RAYS, cnt, np.nan)
    im3 = axes[2].imshow(cnt_m.T, origin='lower', extent=extent, aspect='equal',
                         cmap=cmap_c)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='rays in kernel')
    axes[2].add_patch(plt.Rectangle(**rect, fill=False, ec='black', lw=1.3))
    draw_outlines(axes[2], transform=tr, det_name=cfg.DET_NAME)
    axes[2].set_xlabel('reference X [mm]'); axes[2].set_ylabel('reference Y [mm]')
    axes[2].set_title(f'rays per kernel\n(grey < {MIN_RAYS})')
    fig.suptitle(f'{cfg.DET_NAME} sliding-window efficiency (waveform-first) — '
                 f'{cfg.RUN}/{cfg.SUB_RUN}', y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(cfg.OUT_BASE, 'wft', 'efficiency')
    fig.savefig(os.path.join(out, 'efficiency_map_sliding.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    inact = np.ones(len(d), bool)   # rows are already box-cut in per_ray_table
    summary = dict(det=cfg.DET_NAME, run=cfg.RUN, sub_run=cfg.SUB_RUN,
                   basis='waveform-first (wft)', feus=cfg.MX17_FEUS,
                   det_z=cfg.DET_PLANE_Z, r_mm=R, kernel_mm=KERNEL,
                   min_rays=MIN_RAYS, grid=GRID,
                   n_rays=int(len(d)), n_rays_active=int(inact.sum()),
                   integrated_within=float(within.mean()),
                   integrated_has_any=float(has_any.mean()),
                   active_box=box)
    with open(os.path.join(out, 'efficiency_map_sliding.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def fig_scatter(cfg, d, box, params):
    hit = d[d['within']]
    miss = d[~d['within']]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(hit['x'], hit['y'], s=2.5, c='#2ca02c', alpha=0.45,
               label=f'hit within {R:g} mm ({len(hit):,})', lw=0)
    ax.scatter(miss['x'], miss['y'], s=2.5, c='#d62728', alpha=0.45,
               label=f'no hit within {R:g} mm ({len(miss):,})', lw=0)
    ax.add_patch(plt.Rectangle((box['x0'], box['y0']), box['x1'] - box['x0'],
                               box['y1'] - box['y0'], fill=False, ec='black',
                               lw=1.3, label='empirical footprint'))
    draw_outlines(ax, transform=alignment_transform(params), det_name=cfg.DET_NAME)
    ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'{cfg.DET_NAME} efficiency scatter — hit within {R:g} mm\n'
                 f'{cfg.RUN}/{cfg.SUB_RUN} (waveform-first)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                             'scatter_within_5mm.png'), dpi=130)
    plt.close(fig)


def fig_wide_breakdown(cfg):
    j = json.load(open(os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                                    'efficiency_breakdown.json')))
    cats = [('reco_near (≤5mm)', j['within_R'], '#2ca02c'),
            ('reco_far (>5mm)', j['reco_far'], '#ff9f1c'),
            ('spark (>50 strips)', j['spark_cat'], '#7b2d8e'),
            ('hit_no_reco', j['hit_no_reco'], '#e6c229'),
            ('no_hit (silent)', j['no_hit'], '#d62728')]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ys = np.arange(len(cats))[::-1]
    for yv, (lab, val, col) in zip(ys, cats):
        ax.barh(yv, val, color=col, height=0.62)
        ax.text(val + 0.4, yv, f'{val:.1f}%', va='center', fontsize=8)
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in cats], fontsize=8)
    ax.set_xlabel(f'% of crossing muons in active area (R={R:g} mm)', fontsize=8)
    ax.set_title(f'{cfg.DET_NAME} efficiency breakdown — where do the crossing '
                 f'muons go? (waveform-first, n={j["n_rays"]:,} rays)', fontsize=9)
    ax.set_xlim(0, max(45.0, j['within_R'] * 1.18))
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                             'efficiency_breakdown_wide.png'), dpi=130)
    plt.close(fig)


def fig_pos_corr(cfg, d):
    """Detector-vs-reference position density per axis, axes clipped to the
    footprint (the alignment stage's own figure lets single pathological fits
    blow the axis range)."""
    m = np.isfinite(d['det_x']) & np.isfinite(d['det_y'])
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for i, (ax_name, det_c, ref_c) in enumerate([('X', 'det_x', 'x'),
                                                 ('Y', 'det_y', 'y')]):
        dv = d.loc[m, det_c].to_numpy(float)
        rv = d.loc[m, ref_c].to_numpy(float)
        lo = min(np.percentile(dv, 0.2), np.percentile(rv, 0.2)) - 15
        hi = max(np.percentile(dv, 99.8), np.percentile(rv, 99.8)) + 15
        ax = axs[i]
        hb = ax.hist2d(dv, rv, bins=[np.linspace(lo, hi, 140)] * 2,
                       norm=matplotlib.colors.LogNorm(), cmap='viridis')
        fig.colorbar(hb[3], ax=ax, fraction=0.046, pad=0.04, label='events / bin')
        ax.plot([lo, hi], [lo, hi], color='red', lw=0.8, ls='--')
        ax.set_xlabel(f'detector {ax_name} [mm]')
        ax.set_ylabel(f'reference {ax_name} [mm]')
        ax.set_title(f'{ax_name} position density (n={m.sum():,})', fontsize=9)
    fig.suptitle(f'{cfg.DET_NAME} position correlation vs M3 — '
                 f'{cfg.RUN}/{cfg.SUB_RUN} (waveform-first)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                             'position_correlation_density.png'), dpi=130)
    plt.close(fig)


def fig_angle_corr(cfg):
    """Ref-vs-reconstructed angle density, per plane, on the w0/kw-corrected
    angles (03_angles conventions: slope_reliable only)."""
    W = os.path.join(cfg.OUT_BASE, 'wft')
    table = os.path.join(W, 'angles_w0corr', 'events_w0corr.parquet')
    out_dir = os.path.join(W, 'angles_w0corr')
    tag = 'w0/kw corrected'
    if not os.path.exists(table):
        print(f'  WARNING: {table} missing — falling back to frozen angles')
        table = os.path.join(W, 'events.parquet')
        out_dir = os.path.join(W, 'angles')
        tag = 'frozen (w0/kw NOT applied)'
    params = cm.load_alignment(os.path.join(W, 'alignment', 'alignment.json'))
    df = compat.load_table(table)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    ref = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        ref[int(r.event_id)] = (tx, ty)
    idx = df.drop_duplicates('event_id').set_index('event_id')

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for i, plane in enumerate(('x', 'y')):
        ok = idx[f'{plane}_ok'].astype(bool)
        rel = idx[f'{plane}_slope_reliable'].astype(bool)
        eids = [e for e in idx.index[ok & rel] if e in ref]
        th_ref = np.degrees(np.arctan([ref[e][0 if plane == 'x' else 1]
                                       for e in eids]))
        th_det = np.degrees(np.arctan(idx.loc[eids, f'{plane}_tan_theta']
                                      .to_numpy(float)))
        m = np.isfinite(th_ref) & np.isfinite(th_det)
        ax = axs[i]
        hb = ax.hist2d(th_det[m], th_ref[m], bins=[np.linspace(-30, 30, 121)] * 2,
                       norm=matplotlib.colors.LogNorm(), cmap='viridis')
        fig.colorbar(hb[3], ax=ax, fraction=0.046, pad=0.04, label='events / bin')
        ax.plot([-30, 30], [-30, 30], color='red', lw=0.8, ls='--')
        ax.set_xlabel(f'{plane} detector angle [deg]')
        ax.set_ylabel(f'{plane} reference angle [deg]')
        ax.set_title(f'{plane.upper()} angle correlation (n={m.sum():,})\n{tag}',
                     fontsize=9)
    fig.suptitle(f'{cfg.DET_NAME} angular correlation vs M3 — '
                 f'{cfg.RUN}/{cfg.SUB_RUN}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'angle_correlation_hist.png'), dpi=130)
    plt.close(fig)


def main():
    keys = [a for a in sys.argv[1:] if not a.startswith('-')] or JUNE_KEYS
    for key in keys:
        cfg = get_config(key)
        print(f'== {key} ({cfg.DET_NAME})')
        d, box, params = per_ray_table(cfg)
        out = cfg.out_dir('wft', 'efficiency')
        d.to_csv(os.path.join(out, 'ray_hit_miss_list.csv'), index=False)
        print(f'  {len(d):,} rays in active box; integrated within '
              f'{100 * d["within"].mean():.1f}%')
        fig_sliding(cfg, d, box, params)
        fig_pos_corr(cfg, d)
        fig_scatter(cfg, d, box, params)
        fig_wide_breakdown(cfg)
        fig_angle_corr(cfg)
        print(f'  figures written under {cfg.OUT_BASE}/wft/')


if __name__ == '__main__':
    main()
