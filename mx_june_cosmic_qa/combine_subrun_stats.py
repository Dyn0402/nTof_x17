#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_subrun_stats.py

Pool the ray-based efficiency breakdown across subruns that share the SAME run
parameters (here: mx17_det2_det3_overnight_6-22-26 short_run + longer_run).

We do NOT merge the combined_hits/rays directories — event numbers (`evn`/`eventId`)
restart at 1 in each subrun, so a merged directory would cross-pair events. Instead
each subrun is run through the full pipeline independently (its own alignment + active
box), and here we sum the per-category ray COUNTS from each subrun's
efficiency_breakdown.txt and recompute the pooled percentages. This is the
statistically correct pooled efficiency = (total reco_near) / (total active rays).

Resolution is pooled as the reco-count-weighted mean of the per-subrun core sigmas.
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ANALYSIS = '/home/dylan/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_6-22-26'
SUBRUNS = ['short_run', 'longer_run']
CATS = ['reco_near', 'reco_far', 'hit_no_reco', 'no_hit']
# Detector -> (label, FEU string, z)
DETS = {
    'mx17_3': ('mx17_3  (bottom, FEU 3/4, z=232)', 3, 4, 232),
    'mx17_2': ('mx17_2  (top, FEU 6/8, z=702)', 6, 8, 702),
}


def parse_breakdown(path):
    """Return dict: active rays, per-category counts, core_sigma, within_of_reco frac."""
    txt = open(path).read()
    out = {}
    m = re.search(r'active-area clean M3 rays:\s*(\d+)', txt)
    out['active'] = int(m.group(1))
    for c in CATS:
        m = re.search(rf'{c}\s*:\s*(\d+)', txt)
        out[c] = int(m.group(1))
    m = re.search(r'core sigma\(\|r\|<15\)=([\d.]+)', txt)
    out['core_sigma'] = float(m.group(1)) if m else float('nan')
    m = re.search(r'of reconstructed:\s*([\d.]+)% within', txt)
    out['within_of_reco'] = float(m.group(1)) if m else float('nan')
    return out


def pooled(det):
    per = {sr: parse_breakdown(os.path.join(ANALYSIS, sr, det, 'efficiency',
                                            'efficiency_breakdown.txt'))
           for sr in SUBRUNS}
    tot = {k: sum(per[sr][k] for sr in SUBRUNS) for k in ['active'] + CATS}
    reco = tot['reco_near'] + tot['reco_far']
    tot['eff'] = 100.0 * tot['reco_near'] / tot['active']
    tot['reco_at_all'] = 100.0 * reco / tot['active']
    tot['has_any'] = 100.0 * (reco + tot['hit_no_reco']) / tot['active']
    # reco-count-weighted core sigma
    w = np.array([per[sr]['reco_near'] + per[sr]['reco_far'] for sr in SUBRUNS], float)
    s = np.array([per[sr]['core_sigma'] for sr in SUBRUNS], float)
    tot['core_sigma'] = float(np.sum(w * s) / np.sum(w))
    tot['within_of_reco'] = 100.0 * tot['reco_near'] / reco
    return per, tot


def main():
    out_dir = os.path.join(ANALYSIS, 'combined_short_longer')
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (det, (label, fx, fy, z)) in zip(axes, DETS.items()):
        per, tot = pooled(det)
        # ---- grouped bar: subruns + combined, category fractions ----
        groups = SUBRUNS + ['COMBINED']
        data = []
        for sr in SUBRUNS:
            a = per[sr]['active']
            data.append([100.0 * per[sr][c] / a for c in CATS])
        data.append([100.0 * tot[c] / tot['active'] for c in CATS])
        data = np.array(data)            # (3 groups, 4 cats)
        x = np.arange(len(groups))
        w = 0.2
        colors = ['#2a9d8f', '#e9c46a', '#f4a261', '#c1c1c1']
        for i, c in enumerate(CATS):
            ax.bar(x + (i - 1.5) * w, data[:, i], w, label=c, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}\n(N={per[g]["active"] if g in per else tot["active"]})'
                            for g in groups])
        ax.set_ylabel('% of active-area M3 crossings')
        ax.set_title(f'{label}\ncombined eff={tot["eff"]:.1f}%  '
                     f'reco-at-all={tot["reco_at_all"]:.1f}%')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        lines.append(f'===== {label} =====')
        for sr in SUBRUNS:
            p = per[sr]
            lines.append(f'  {sr:11s} active={p["active"]:6d}  '
                         f'eff={100*p["reco_near"]/p["active"]:5.1f}%  '
                         f'reco_far={100*p["reco_far"]/p["active"]:4.1f}%  '
                         f'hit_no_reco={100*p["hit_no_reco"]/p["active"]:4.1f}%  '
                         f'no_hit={100*p["no_hit"]/p["active"]:4.1f}%  '
                         f'sigma={p["core_sigma"]:.2f}mm')
        lines.append(f'  {"COMBINED":11s} active={tot["active"]:6d}  '
                     f'eff={tot["eff"]:5.1f}%  '
                     f'reco_far={100*tot["reco_far"]/tot["active"]:4.1f}%  '
                     f'hit_no_reco={100*tot["hit_no_reco"]/tot["active"]:4.1f}%  '
                     f'no_hit={100*tot["no_hit"]/tot["active"]:4.1f}%  '
                     f'sigma~{tot["core_sigma"]:.2f}mm')
        lines.append(f'    reco-at-all={tot["reco_at_all"]:.1f}%  has_any={tot["has_any"]:.1f}%  '
                     f'(of reco {tot["within_of_reco"]:.1f}% within 5mm)')
        lines.append('')

    fig.suptitle('mx17_det2_det3_overnight_6-22-26  —  short_run + longer_run combined '
                 '(pooled ray-based efficiency, R=5 mm)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'combined_efficiency_breakdown.png'), dpi=130)

    report = '\n'.join(lines)
    print(report)
    open(os.path.join(out_dir, 'combined_efficiency_summary.txt'), 'w').write(report + '\n')
    print(f'\nWritten to: {out_dir}')


if __name__ == '__main__':
    main()
