#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shower_test.py

Test the SHOWER hypothesis for the "no clean M3 track" spark population.

Competing explanations for why sparks are over-represented in events where M3
finds no clean single track:
  (SHOWER)     a real multi-particle shower crosses BOTH M3 (-> messy, no clean
               single track) AND our detector (-> many strips -> read as a spark)
  (CROSS-TALK) the discharge corrupts the shared readout so M3 loses the track
  (GEOMETRIC)  the muon simply misses M3 / M3 is inefficient (unrelated to the spark)

Discriminator: a shower of several charged particles leaves M3 with MULTIPLE
straight-track candidates (ncand >= 2). Cross-talk / geometric failure leaves
ncand = 0 (M3 blank) or ncand = 1 (one marginal track that failed quality cuts).
We also compare the DISCHARGE MORPHOLOGY of no-track vs muon-tracked sparks: if
identical, the no-track sparks are the same physical discharge, not a distinct
shower population.

Run:  ../../.venv/bin/python shower_test.py [KEY]   (default g_det3_wknd)
"""
import os, sys, json
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
from M3RefTracking import get_ray_data

KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    CFG = get_config(KEY)
    e = np.load(os.path.join(HERE, 'events.npz'))
    h = np.load(os.path.join(HERE, 'spark_hits.npz'))
    meta = json.load(open(os.path.join(HERE, 'spark_meta.json')))

    # ---- raw M3 candidate-track count per event (unfiltered) ----
    rd = get_ray_data(CFG.m3_tracking_dir, 'all', None)
    m3evn = ak.to_numpy(rd['evn']).astype(np.int64)
    ncand = ak.to_numpy(ak.num(rd['Chi2X'], axis=1)).astype(np.int64)
    ncand_by_ev = dict(zip(m3evn, ncand))

    eid = e['eventId']; spark = e['spark']; hasray = e['has_ray']
    nc = np.array([ncand_by_ev.get(int(x), -1) for x in eid])   # -1 = event absent from M3 file
    inM3 = nc >= 0

    def frac(mask, cond):
        return 100 * np.mean(cond[mask]) if mask.sum() else float('nan')

    # cross-tab ncand for spark vs non-spark (events present in M3)
    nonsp = inM3 & ~spark
    sp = inM3 & spark
    notrk_sp = sp & ~hasray                              # the population in question
    print(f'=== {KEY}  ({CFG.DET_NAME}) ===')
    print(f'events in M3: {inM3.sum()}   sparks: {sp.sum()}   no-track sparks: {notrk_sp.sum()}')
    rows = {}
    for nm, msk in [('all non-spark', nonsp), ('all sparks', sp), ('no-track sparks', notrk_sp)]:
        d = {k: frac(msk, nc == k) for k in (0, 1)}
        d['ge2'] = frac(msk, nc >= 2)
        rows[nm] = d
        print(f'  {nm:16s}: ncand=0 {d[0]:5.1f}%   ncand=1 {d[1]:5.1f}%   ncand>=2 {d["ge2"]:5.2f}%')

    # the quantitative bound: how many no-track sparks COULD be showers (ncand>=2)?
    n_shower_like = int((notrk_sp & (nc >= 2)).sum())
    max_shower_frac = 100 * n_shower_like / max(1, notrk_sp.sum())
    n_multi_total = int((inM3 & (nc >= 2)).sum())
    print(f'  no-track sparks with ncand>=2 (shower-like): {n_shower_like} '
          f'({max_shower_frac:.2f}% of no-track sparks)')
    print(f'  ALL multi-track events in run: {n_multi_total} ({100*n_multi_total/inM3.sum():.2f}%)')

    # ---- morphology: no-track vs tracked sparks (are they the same discharge?) ----
    tim = h['time']; ok = (tim >= -50) & (tim <= 1600)
    dfh = pd.DataFrame({'eid': h['eventId'][ok], 'isx': h['is_x'][ok], 'pos': h['pos'][ok],
                        't': tim[ok], 'sat': h['sat'][ok]})
    hasray_by_ev = dict(zip(eid.astype(np.int64), hasray))
    dfh['tracked'] = dfh['eid'].map(lambda x: bool(hasray_by_ev.get(int(x), False)))
    grp = dfh.groupby('eid')
    per = pd.DataFrame({
        'n': grp.size(),
        'span': grp['pos'].apply(lambda p: np.nanmax(p) - np.nanmin(p)),
        'tstd': grp['t'].std(),
        'satfrac': grp['sat'].mean(),
        'edgefrac': grp['pos'].apply(lambda p: np.mean((p < 40) | (p > 360))),
        'tracked': grp['tracked'].first(),
    })
    mor = {}
    for nm, msk in [('tracked spark', per['tracked']), ('no-track spark', ~per['tracked'])]:
        s = per[msk]
        mor[nm] = dict(n=int(len(s)), strips_med=float(s['n'].median()),
                       span_med=float(s['span'].median()), tstd_med=float(s['tstd'].median()),
                       satfrac_med=float(s['satfrac'].median()), edgefrac_med=float(s['edgefrac'].median()))
        print(f'  MORPH {nm:16s}: strips {mor[nm]["strips_med"]:.0f}  span {mor[nm]["span_med"]:.0f}mm  '
              f'tstd {mor[nm]["tstd_med"]:.0f}ns  edgefrac {mor[nm]["edgefrac_med"]:.2f}  satfrac {mor[nm]["satfrac_med"]:.2f}')

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) ncand distribution
    x = np.arange(3); w = 0.27
    for i, (nm, msk, col) in enumerate([('non-spark', nonsp, 'grey'),
                                        ('all sparks', sp, 'purple'),
                                        ('no-track sparks', notrk_sp, 'crimson')]):
        vals = [frac(msk, nc == 0), frac(msk, nc == 1), frac(msk, nc >= 2)]
        axes[0].bar(x + (i - 1) * w, vals, w, label=nm, color=col)
    axes[0].set_xticks(x); axes[0].set_xticklabels(['0\n(M3 blank)', '1\n(single)', '≥2\n(SHOWER)'])
    axes[0].set_ylabel('% of events'); axes[0].legend(fontsize=8)
    axes[0].set_title('(a) M3 candidate-track count\nno-track sparks are ncand=0, NOT ≥2')
    # (b) zoom on the ≥2 (shower) bin, log
    for i, (nm, msk, col) in enumerate([('non-spark', nonsp, 'grey'),
                                        ('all sparks', sp, 'purple'),
                                        ('no-track sparks', notrk_sp, 'crimson')]):
        axes[1].bar(i, frac(msk, nc >= 2), 0.6, color=col)
        axes[1].text(i, frac(msk, nc >= 2) + 0.002, f'{frac(msk, nc>=2):.2f}%', ha='center', fontsize=8)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(['non-spark', 'all\nsparks', 'no-track\nsparks'], fontsize=8)
    axes[1].set_ylabel('% with ≥2 M3 tracks (shower)')
    axes[1].set_title('(b) shower signature (≥2 M3 tracks)\nNOT enriched in sparks')
    # (c) morphology: no-track vs tracked spark, normalised discharge size
    nt = per[~per['tracked']]['n']; tr = per[per['tracked']]['n']
    bb = np.arange(50, 260, 8)
    axes[2].hist(tr, bins=bb, density=True, alpha=0.6, color='seagreen', label='muon-tracked spark')
    axes[2].hist(nt, bins=bb, density=True, alpha=0.6, color='crimson', label='no-track spark')
    axes[2].set_xlabel('strips fired per spark'); axes[2].set_ylabel('norm.'); axes[2].legend(fontsize=8)
    axes[2].set_title('(c) discharge morphology identical\ntracked vs no-track sparks overlap')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_shower_test.png'), dpi=140, bbox_inches='tight')
    plt.close(fig); print('wrote fig_shower_test.png')

    meta['shower_test'] = dict(ncand=rows, n_shower_like_notrk=n_shower_like,
                               max_shower_frac_pct=max_shower_frac,
                               n_multi_total=n_multi_total,
                               multi_total_pct=100 * n_multi_total / inM3.sum(),
                               morphology=mor)
    json.dump(meta, open(os.path.join(HERE, 'spark_meta.json'), 'w'), indent=2)
    print('updated spark_meta.json')


if __name__ == '__main__':
    main()
