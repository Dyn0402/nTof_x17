#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p2_channel_qa.py

Channel-level raw QA for the P2 detector (det_type 'P2'), which has NO strip->position
mapping yet — so this stops BEFORE any alignment / micro-TPC step. It reads the
combined-hits ROOT file(s) directly and characterises P2's DREAM FEUs in raw
(feu, channel) space:

  - occupancy (hits per channel) per FEU            -> alive / dead / noisy channels
  - pulse height (local_max) vs channel, per FEU    -> gain uniformity, hot channels
  - local_max distribution per FEU                  -> signal vs noise
  - hits-per-event multiplicity (P2 channels)       -> clustering / sparking
  - saturated-hit fraction per channel              -> sparking map
  - rate vs time                                    -> stability

P2 reads out on FEUs 3, 4, 6 (run_config dream_feus). Usage:
  python p2_channel_qa.py [hits_dir] [--feus=3,4,6] [--out=DIR]
"""
import os
import sys
import glob

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot

RUN = 'mx17_det3_p2_det1_overnight_6-27-26'
SUB = 'long_run_p2_det1_sanity_check'
DEF_HITS = f'/home/dylan/x17/cosmic_bench/det3_p2/{RUN}/{SUB}/combined_hits_root'
DEF_OUT = f'/home/dylan/x17/cosmic_bench/Analysis/{RUN}/p2_qa'


def _arg(flag, default):
    return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(flag)), default)


def main():
    hits_dir = next((a for a in sys.argv[1:] if not a.startswith('-')), DEF_HITS)
    feus = [int(x) for x in _arg('--feus=', '3,4,6').split(',')]
    out = _arg('--out=', DEF_OUT)
    os.makedirs(out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(hits_dir, '*.root')))
    if not files:
        print(f'No .root in {hits_dir}'); sys.exit(1)
    cols = ['eventId', 'feu', 'channel', 'local_max', 'integral',
            'saturated', 'trigger_timestamp_ns', 'time_over_threshold']
    df = uproot.concatenate([f'{f}:hits' for f in files], expressions=cols, library='pd')
    n_all_evt = df['eventId'].nunique()
    p2 = df[df['feu'].isin(feus)].copy()
    print(f'{len(files)} file(s): {len(df):,} hits / {n_all_evt:,} events total; '
          f'P2 (FEU {feus}): {len(p2):,} hits / {p2["eventId"].nunique():,} events, '
          f'saturated {p2["saturated"].mean()*100:.1f}%')

    t0 = df['trigger_timestamp_ns'].astype('float64').min()
    p2['t_s'] = (p2['trigger_timestamp_ns'].astype('float64') - t0) / 1e9
    dur = (df['trigger_timestamp_ns'].astype('float64').max() - t0) / 1e9
    NCH = 512
    colors = {3: 'tab:blue', 4: 'tab:green', 6: 'tab:red'}

    # ---------- Figure 1: per-FEU occupancy + pulse-height-vs-channel ----------
    fig1, axes = plt.subplots(len(feus), 2, figsize=(15, 3.2 * len(feus)), squeeze=False)
    for i, fe in enumerate(feus):
        s = p2[p2['feu'] == fe]
        c = colors.get(fe, 'tab:gray')
        ax = axes[i][0]
        ax.hist(s['channel'], bins=NCH, range=(0, NCH), color=c, histtype='stepfilled')
        nlive = s['channel'].nunique()
        ax.set_title(f'FEU {fe}: occupancy  ({len(s):,} hits, {nlive}/{NCH} channels fired)')
        ax.set_xlabel('channel'); ax.set_ylabel('hits'); ax.set_xlim(0, NCH)

        ax2 = axes[i][1]
        if len(s) > 5:
            h = ax2.hist2d(s['channel'], s['local_max'], bins=[NCH // 2, 80],
                           range=[[0, NCH], [0, np.nanpercentile(s['local_max'], 99.5)]],
                           cmap='viridis', norm=LogNorm(), cmin=1)
            fig1.colorbar(h[3], ax=ax2, label='hits', fraction=0.046, pad=0.04)
        ax2.set_title(f'FEU {fe}: pulse height (local_max) vs channel')
        ax2.set_xlabel('channel'); ax2.set_ylabel('local_max [ADC]'); ax2.set_xlim(0, NCH)
    fig1.suptitle(f'P2 channel QA — {RUN}/{SUB}  (no mapping; raw FEU/channel space)',
                  fontsize=13, y=1.0)
    fig1.tight_layout()
    fig1.savefig(f'{out}/p2_occupancy_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ---------- Figure 2: summary (amplitude, multiplicity, saturation, rate) ----------
    fig2, ax = plt.subplots(2, 2, figsize=(14, 10))

    for fe in feus:
        s = p2[p2['feu'] == fe]['local_max']
        s = s[np.isfinite(s)]
        ax[0, 0].hist(s, bins=120, range=(0, np.nanpercentile(p2['local_max'], 99.5)),
                      histtype='step', lw=1.6, color=colors.get(fe), label=f'FEU {fe}')
    ax[0, 0].set_yscale('log'); ax[0, 0].set_xlabel('local_max [ADC]'); ax[0, 0].set_ylabel('hits')
    ax[0, 0].set_title('Pulse-height distribution'); ax[0, 0].legend()

    mult = p2.groupby('eventId').size()
    ax[0, 1].hist(mult, bins=range(1, int(mult.quantile(0.995)) + 2),
                  color='tab:purple', histtype='stepfilled')
    ax[0, 1].set_xlabel('P2 hits per event'); ax[0, 1].set_ylabel('events')
    ax[0, 1].set_title(f'Hit multiplicity  (median {int(mult.median())}, max {mult.max()})')

    # saturated fraction per (feu,channel)
    g = p2.groupby(['feu', 'channel'])['saturated'].agg(['mean', 'size']).reset_index()
    g = g[g['size'] >= 3]
    for fe in feus:
        gg = g[g['feu'] == fe]
        ax[1, 0].scatter(gg['channel'], gg['mean'] * 100, s=8, color=colors.get(fe),
                         label=f'FEU {fe}', alpha=0.7)
    ax[1, 0].set_xlabel('channel'); ax[1, 0].set_ylabel('saturated hits [%]')
    ax[1, 0].set_title('Saturation per channel (sparking map)'); ax[1, 0].legend(); ax[1, 0].set_xlim(0, NCH)

    nb = max(10, int(dur / 30))
    edges = np.linspace(0, p2['t_s'].max(), nb + 1)
    hrate, _ = np.histogram(p2['t_s'], bins=edges)
    erate = p2.groupby(pd.cut(p2['t_s'], edges, labels=False))['eventId'].nunique().reindex(range(nb), fill_value=0)
    bw = edges[1] - edges[0]
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax[1, 1].plot(ctr, hrate / bw, '-', color='tab:blue', label='P2 hits/s')
    ax[1, 1].plot(ctr, erate.values / bw, '-', color='tab:orange', label='P2 events/s')
    ax[1, 1].set_xlabel('time [s]'); ax[1, 1].set_ylabel('rate [/s]')
    ax[1, 1].set_title(f'Rate vs time  (run ~{dur/60:.0f} min)'); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    fig2.suptitle(f'P2 channel QA summary — {RUN}/{SUB}', fontsize=13, y=1.0)
    fig2.tight_layout()
    fig2.savefig(f'{out}/p2_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'written: {out}/p2_occupancy_amplitude.png  and  p2_summary.png')


if __name__ == '__main__':
    main()
