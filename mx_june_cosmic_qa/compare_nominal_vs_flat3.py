#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_nominal_vs_flat3.py

Compare the nominal processing (5sigma x per-channel pedestal RMS) against the
flat-threshold reprocessing (3 x per-FEU median pedestal RMS) for the 6-22 run, for
both detectors, per subrun and pooled (short+longer). Reads each run's
efficiency_breakdown.txt and pools by summing per-category ray counts.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOM = '/home/dylan/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_6-22-26'
FL3 = '/home/dylan/x17/cosmic_bench/det2_det3/Analysis/mx17_det2_det3_overnight_6-22-26'
OUT = '/home/dylan/x17/cosmic_bench/det2_det3/Analysis/mx17_det2_det3_overnight_6-22-26/_compare_nominal_flat3'
SUBRUNS = ['short_run', 'longer_run']
CATS = ['reco_near', 'reco_far', 'hit_no_reco', 'no_hit']
DETS = {'mx17_3': 'mx17_3 (bottom, FEU 3/4)', 'mx17_2': 'mx17_2 (top, FEU 6/8)'}


def parse(path):
    txt = open(path).read()
    o = {'active': int(re.search(r'active-area clean M3 rays:\s*(\d+)', txt).group(1))}
    for c in CATS:
        o[c] = int(re.search(rf'{c}\s*:\s*(\d+)', txt).group(1))
    o['core_sigma'] = float(re.search(r'core sigma\(\|r\|<15\)=([\d.]+)', txt).group(1))
    return o


def get(base, det):
    per = {sr: parse(os.path.join(base, sr, det, 'efficiency', 'efficiency_breakdown.txt'))
           for sr in SUBRUNS}
    tot = {k: sum(per[sr][k] for sr in SUBRUNS) for k in ['active'] + CATS}
    reco = tot['reco_near'] + tot['reco_far']
    tot['eff'] = 100 * tot['reco_near'] / tot['active']
    tot['reco_at_all'] = 100 * reco / tot['active']
    w = np.array([per[sr]['reco_near'] + per[sr]['reco_far'] for sr in SUBRUNS], float)
    s = np.array([per[sr]['core_sigma'] for sr in SUBRUNS], float)
    tot['core_sigma'] = float(np.sum(w * s) / np.sum(w))
    return per, tot


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ['Nominal (5sigma x per-ch RMS)  vs  Flat3 (3 x per-FEU median RMS)  —  6-22 run', '']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (det, label) in zip(axes, DETS.items()):
        pn, tn = get(NOM, det)
        pf, tf = get(FL3, det)
        groups = ['short', 'longer', 'COMBINED']
        nom_eff = [100 * pn[s]['reco_near'] / pn[s]['active'] for s in SUBRUNS] + [tn['eff']]
        fl3_eff = [100 * pf[s]['reco_near'] / pf[s]['active'] for s in SUBRUNS] + [tf['eff']]
        x = np.arange(len(groups)); w = 0.38
        b1 = ax.bar(x - w/2, nom_eff, w, label='nominal (5σ×RMS)', color='#bdbdbd')
        b2 = ax.bar(x + w/2, fl3_eff, w, label='flat3 (3×median)', color='#2a9d8f')
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel('efficiency within 5 mm (%)'); ax.set_ylim(0, 100)
        ax.set_title(f'{label}\nCOMBINED {tn["eff"]:.0f}% → {tf["eff"]:.0f}%  '
                     f'(σ {tn["core_sigma"]:.2f}→{tf["core_sigma"]:.2f} mm)')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

        lines.append(f'===== {label} =====')
        for s in SUBRUNS:
            lines.append(f'  {s:11s} eff {100*pn[s]["reco_near"]/pn[s]["active"]:5.1f}% -> '
                         f'{100*pf[s]["reco_near"]/pf[s]["active"]:5.1f}%   '
                         f'no_hit {100*pn[s]["no_hit"]/pn[s]["active"]:4.1f}% -> '
                         f'{100*pf[s]["no_hit"]/pf[s]["active"]:4.1f}%   '
                         f'hit_no_reco {100*pn[s]["hit_no_reco"]/pn[s]["active"]:4.1f}% -> '
                         f'{100*pf[s]["hit_no_reco"]/pf[s]["active"]:4.1f}%   '
                         f'σ {pn[s]["core_sigma"]:.2f}->{pf[s]["core_sigma"]:.2f}')
        lines.append(f'  COMBINED    eff {tn["eff"]:5.1f}% -> {tf["eff"]:5.1f}%   '
                     f'reco-at-all {tn["reco_at_all"]:.1f}% -> {tf["reco_at_all"]:.1f}%   '
                     f'σ~{tn["core_sigma"]:.2f}->{tf["core_sigma"]:.2f} mm')
        lines.append('')

    fig.suptitle('6-22 run: flat 3×median-RMS threshold recovers suppressed micro-TPC signal '
                 '(resolution preserved)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'nominal_vs_flat3_efficiency.png'), dpi=140)
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT, 'comparison_summary.txt'), 'w').write(report + '\n')
    print('Written to:', OUT)


if __name__ == '__main__':
    main()
