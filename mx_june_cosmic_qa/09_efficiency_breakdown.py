#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_efficiency_breakdown.py

Standard efficiency breakdown QA. For clean M3 rays inside the active area, split
each crossing muon into:
  no_hit      : detector fired no strip            (has_any = False)
  hit_no_reco : fired strips, NO valid X+Y reco point
  reco_far    : reconstructed X+Y but |r| > R
  reco_near   : reconstructed X+Y and |r| <= R     (= the "within R" efficiency)

Outputs (output/<run>/<det>/efficiency/):
  efficiency_breakdown_table.png  - the enshrined breakdown table
  efficiency_breakdown.png        - bar chart
  radial_residual_reco.png        - |r| of reconstructed hits (core + tail)
  reco_positions_detector.png     - DETECTOR positions of well-reconstructed (reco_near) hits
  nonreco_ray_positions.png       - RAY projections of non-reconstructed muons
  efficiency_breakdown.txt        - the numbers

Usage: python 09_efficiency_breakdown.py ovn_det1 [--r=5]
"""
import os, sys, pickle
import matplotlib; matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

R = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--r=')), 5.0)
SPARK_THRESH = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--spark=')), 50)


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v); k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10: break
        v = v[k]
    return float(np.std(v))


def main():
    out_dir = CFG.out_dir('efficiency')
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20.0)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    det_raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    det1_hit = set(int(e) for e in det_raw['eventId'].unique())
    # spark rate: fraction of detector-firing events with > SPARK_THRESH strips (one row
    # per hit) firing at once (a full-detector discharge; same threshold as the align veto)
    _mult = det_raw.groupby('eventId').size()
    n_firing = int(len(_mult)); n_spark = int((_mult > SPARK_THRESH).sum())
    spark_frac = 100.0 * n_spark / n_firing if n_firing else float('nan')

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr); evn = [int(v) for v in evn]

    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5]); ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    box = dict(xy=(ax0, ay0), width=ax1 - ax0, height=ay1 - ay0)

    # categorise; keep ray positions per category + detector positions for reco
    cat = {k: [] for k in ('no_hit', 'hit_no_reco', 'reco_far', 'reco_near')}   # -> ray (x,y)
    reco_near_det, reco_far_det, rlist = [], [], []
    for e, x, y in zip(evn, px, py):
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        if e in reco:
            r = float(np.hypot(x - reco[e][0], y - reco[e][1])); rlist.append(r)
            if r <= R:
                cat['reco_near'].append((x, y)); reco_near_det.append(reco[e])
            else:
                cat['reco_far'].append((x, y)); reco_far_det.append(reco[e])
        elif e in det1_hit:
            cat['hit_no_reco'].append((x, y))
        else:
            cat['no_hit'].append((x, y))
    n = sum(len(v) for v in cat.values()); rlist = np.array(rlist)
    pct = {k: 100 * len(v) / n for k, v in cat.items()}
    has_any = 100 * (n - len(cat['no_hit'])) / n
    reco_all = 100 * (len(cat['reco_near']) + len(cat['reco_far'])) / n
    frac_in = 100 * np.mean(rlist <= R) if len(rlist) else 0
    sig = rstd(rlist[rlist < 15]) if len(rlist) else float('nan')
    med = float(np.median(rlist)) if len(rlist) else float('nan')

    lines = [f'Efficiency breakdown — {CFG.DET_NAME}  {CFG.RUN}/{CFG.SUB_RUN}  (R={R:g} mm)',
             f'active-area clean M3 rays: {n}']
    for k in ('reco_near', 'reco_far', 'hit_no_reco', 'no_hit'):
        lines.append(f'  {k:12s}: {len(cat[k]):5d}  ({pct[k]:5.1f}%)')
    lines += [f'  has_any={has_any:.1f}%  within{R:g}mm={pct["reco_near"]:.1f}%  reco-at-all={reco_all:.1f}%',
              f'  of reconstructed: {frac_in:.1f}% within {R:g}mm, core sigma(|r|<15)={sig:.2f} mm, median |r|={med:.2f} mm',
              f'  spark_frac={spark_frac:.1f}%  (>{SPARK_THRESH} strips: {n_spark}/{n_firing} firing events)']
    txt = '\n'.join(lines); print(txt)
    open(f'{out_dir}/efficiency_breakdown.txt', 'w').write(txt + '\n')

    # ---- enshrined table figure ----
    fig, ax = plt.subplots(figsize=(8.5, 3.4)); ax.axis('off')
    rows = [['reco_near', 'reconstructed, |r|<=%gmm  (the efficiency)' % R, f'{len(cat["reco_near"])}', f'{pct["reco_near"]:.1f}%'],
            ['reco_far', 'reconstructed but |r|>%gmm  (competing cluster)' % R, f'{len(cat["reco_far"])}', f'{pct["reco_far"]:.1f}%'],
            ['hit_no_reco', 'fired strips, no valid X+Y reco', f'{len(cat["hit_no_reco"])}', f'{pct["hit_no_reco"]:.1f}%'],
            ['no_hit', 'detector silent (true miss)', f'{len(cat["no_hit"])}', f'{pct["no_hit"]:.1f}%'],
            ['', 'TOTAL crossings in active area', f'{n}', '100%'],
            ['has_any', 'detector responded (any strip)', '', f'{has_any:.1f}%'],
            ['reco-at-all', 'formed a valid X+Y point', '', f'{reco_all:.1f}%']]
    colcol = ['green', 'orange', 'gold', 'red', 'white', 'lightsteelblue', 'lightsteelblue']
    t = ax.table(cellText=rows, colLabels=['category', 'meaning', 'N', '% crossings'],
                 colWidths=[0.18, 0.5, 0.12, 0.16], loc='center', cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for i, c in enumerate(colcol):
        t[(i + 1, 0)].set_facecolor(c)
    ax.set_title(f'{CFG.DET_NAME} efficiency breakdown — {CFG.RUN}/{CFG.SUB_RUN}\n'
                 f'reco residual: {frac_in:.0f}% within {R:g}mm, core σ={sig:.2f}mm, median |r|={med:.2f}mm',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f'{out_dir}/efficiency_breakdown_table.png', dpi=150, bbox_inches='tight')

    # ---- bar chart ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ks = ['reco_near', 'reco_far', 'hit_no_reco', 'no_hit']; cols = ['green', 'orange', 'gold', 'red']
    ax.bar(ks, [pct[k] for k in ks], color=cols)
    for i, k in enumerate(ks):
        ax.text(i, pct[k] + 1, f'{pct[k]:.0f}%', ha='center')
    ax.set_ylabel('% of crossing muons'); ax.set_title(f'{CFG.DET_NAME} where do the muons go? (active area, R={R:g}mm)')
    fig.tight_layout(); fig.savefig(f'{out_dir}/efficiency_breakdown.png', dpi=150, bbox_inches='tight')

    # ---- |r| residual of reconstructed hits ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(rlist, bins=np.linspace(0, 50, 100), color='steelblue')
    axes[0].axvline(R, color='r', ls='--', label=f'{R:g} mm cut'); axes[0].legend()
    axes[0].set_xlabel('|r| residual [mm]'); axes[0].set_ylabel('reconstructed events'); axes[0].set_title('|r| (0-50 mm)')
    axes[1].hist(rlist, bins=np.linspace(0, max(rlist.max(), 50), 120), color='steelblue')
    axes[1].axvline(R, color='r', ls='--'); axes[1].set_yscale('log')
    axes[1].set_xlabel('|r| residual [mm]'); axes[1].set_ylabel('events (log)'); axes[1].set_title('|r| full range (core + tail)')
    fig.suptitle(f'{CFG.DET_NAME} radial residual of reconstructed hits (active area) — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/radial_residual_reco.png', dpi=150, bbox_inches='tight')

    rng = [[ax0 - 40, ax1 + 40], [ay0 - 40, ay1 + 40]]

    # ---- (a) DETECTOR positions of well-reconstructed (reco_near) hits ----
    rn = np.array(reco_near_det)
    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(rn[:, 0], rn[:, 1], bins=50, range=rng, cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='well-reconstructed hits')
    ax.add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.5, label='active area'))
    ax.set_xlabel('detector X (aligned) [mm]'); ax.set_ylabel('detector Y (aligned) [mm]'); ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'{CFG.DET_NAME} DETECTOR positions of well-reconstructed tracks (|r|<={R:g}mm)\n'
                 f'{len(rn)} hits — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/reco_positions_detector.png', dpi=150, bbox_inches='tight')

    # ---- (b) RAY projections of NON-reconstructed muons (fired-no-reco vs silent) ----
    hnr = np.array(cat['hit_no_reco']); nh = np.array(cat['no_hit'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axx, arr, ttl in [(axes[0], hnr, f'fired strips, no reco  ({len(hnr)})'),
                          (axes[1], nh, f'detector silent  ({len(nh)})')]:
        h = axx.hist2d(arr[:, 0], arr[:, 1], bins=50, range=rng, cmap='inferno')
        plt.colorbar(h[3], ax=axx, label='rays')
        axx.add_patch(plt.Rectangle(**box, fill=False, ec='cyan', lw=1.5))
        axx.set_xlabel('reference X [mm]'); axx.set_ylabel('reference Y [mm]'); axx.set_aspect('equal')
        axx.set_title(ttl)
    fig.suptitle(f'{CFG.DET_NAME} RAY projections of NON-reconstructed muons — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/nonreco_ray_positions.png', dpi=150, bbox_inches='tight')

    # ---- (b') same NON-reconstructed muons as a scatter (copy of (b), per-point) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axx, arr, color, ttl in [(axes[0], hnr, '#d4a017', f'fired strips, no reco  ({len(hnr)})'),
                                 (axes[1], nh, '#cc2a2a', f'detector silent  ({len(nh)})')]:
        axx.scatter(arr[:, 0], arr[:, 1], s=6, alpha=0.3, color=color, edgecolors='none')
        axx.add_patch(plt.Rectangle(**box, fill=False, ec='cyan', lw=1.5))
        axx.set_xlim(rng[0]); axx.set_ylim(rng[1])
        axx.set_xlabel('reference X [mm]'); axx.set_ylabel('reference Y [mm]'); axx.set_aspect('equal')
        axx.set_title(ttl)
    fig.suptitle(f'{CFG.DET_NAME} RAY projections of NON-reconstructed muons (scatter) — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout(); fig.savefig(f'{out_dir}/nonreco_ray_positions_scatter.png', dpi=150, bbox_inches='tight')

    print(f'\nWritten to: {out_dir}')


if __name__ == '__main__':
    main()
