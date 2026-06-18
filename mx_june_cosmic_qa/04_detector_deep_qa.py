#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_detector_deep_qa.py

Deep noise / pathology QA for a single mx17 detector, aimed at understanding a
noisy detector (e.g. det1/mx17_1 in mx17_det1_det2_overnight_6-17-26).

Usage:
    python 04_detector_deep_qa.py ovn_det1

Products (output/<run>/<det>/deep_qa/):
  surface_hitmap.png        2D hitmap on the detector surface (time-paired X-Y),
                            linear + log; hot strips show as lines, sparks as fill.
  strip_firing_fraction.png fraction of events each strip fires (X & Y) vs position
                            — flags always-firing / hot strips.
  event_multiplicity.png    # strips firing per event (X, Y, total), log y —
                            the high tail = spark-like events.
  multiplicity_vs_time.png  mean strips/event over the run (are sparks time-clustered?)
  deep_qa_summary.txt       text summary: hottest strips, spark fraction, etc.
"""

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()

import uproot
import detector_qa as dq
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig

SPARK_FRAC = 0.20   # an event is "spark-like" if >this fraction of either plane's
                    # instrumented strips fire (also report absolute thresholds)


def load_hits():
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    files = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                   if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in files], library='pd')
    df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
    df = cm._map_strip_positions(df, det)
    return df


def plot_surface_hitmap(df, out_dir):
    """2D hitmap on the detector surface from time-paired X-Y hits."""
    paired = dq.get_hit_positions_time_paired(df)
    if paired.empty:
        print('No X-Y pairs for surface hitmap'); return
    x, y = paired['x_mm'].values, paired['y_mm'].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, norm, tag in [(axes[0], None, 'linear'),
                          (axes[1], matplotlib.colors.LogNorm(), 'log')]:
        h = ax.hist2d(x, y, bins=120, range=[[0, 400], [0, 400]],
                      cmap='inferno', norm=norm)
        fig.colorbar(h[3], ax=ax, label=f'paired hits ({tag})')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Y [mm]'); ax.set_aspect('equal')
        ax.set_title(f'Surface hitmap ({tag})')
    fig.suptitle(f'{CFG.DET_NAME} surface hitmap (time-paired) — {CFG.RUN}/{CFG.SUB_RUN}\n'
                 f'{len(paired):,} pairs from {paired["event_id"].nunique():,} events')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/surface_hitmap.png', dpi=150, bbox_inches='tight')


def strip_firing(df, n_events):
    """Per-strip firing fraction (events in which the strip fires / all events)."""
    g = df.groupby(['feu', 'channel'])
    fire = (g['eventId'].nunique() / n_events).rename('fire_frac')
    pos = g[['x_position_mm', 'y_position_mm']].first()
    return fire.to_frame().join(pos).reset_index()


def plot_strip_firing(stripdf, out_dir, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, feu, poscol, lbl in [(axes[0], CFG.MX17_FEU_X, 'x_position_mm', 'X'),
                                 (axes[1], CFG.MX17_FEU_Y, 'y_position_mm', 'Y')]:
        s = stripdf[stripdf['feu'] == feu]
        ax.bar(s[poscol], s['fire_frac'], width=0.8, color='steelblue')
        med = s['fire_frac'].median()
        ax.axhline(med, color='green', ls='--', lw=1, label=f'median {med:.3f}')
        ax.set_xlabel(f'{lbl} strip position [mm]')
        ax.set_ylabel('Fraction of events strip fires')
        ax.set_title(f'{lbl} strips (FEU {feu})')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        top = s.sort_values('fire_frac', ascending=False).head(10)
        summary.append(f'  {lbl} hottest strips (ch, pos_mm, fire_frac):')
        for _, r in top.iterrows():
            summary.append(f'    ch {int(r.channel):3d}  pos {r[poscol]:6.1f}  '
                           f'{r.fire_frac:.3f}')
    fig.suptitle(f'{CFG.DET_NAME} per-strip firing fraction — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/strip_firing_fraction.png', dpi=150, bbox_inches='tight')


def multiplicity(df):
    """Per-event count of distinct strips firing in X and Y."""
    nx = (df[df['feu'] == CFG.MX17_FEU_X].groupby('eventId')['channel'].nunique())
    ny = (df[df['feu'] == CFG.MX17_FEU_Y].groupby('eventId')['channel'].nunique())
    ev = df['eventId'].drop_duplicates()
    mult = pd.DataFrame({'eventId': ev}).set_index('eventId')
    mult['nx'] = nx.reindex(mult.index).fillna(0).astype(int)
    mult['ny'] = ny.reindex(mult.index).fillna(0).astype(int)
    mult['ntot'] = mult['nx'] + mult['ny']
    return mult


def plot_multiplicity(mult, out_dir, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, max(mult['ntot'].max(), 10) + 2)
    axes[0].hist(mult['nx'], bins=bins, histtype='step', lw=1.6, color='red', label='X strips')
    axes[0].hist(mult['ny'], bins=bins, histtype='step', lw=1.6, color='blue', label='Y strips')
    axes[0].set_yscale('log'); axes[0].set_xlabel('strips firing per event')
    axes[0].set_ylabel('events'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Per-plane strip multiplicity')

    axes[1].hist(mult['ntot'], bins=bins, color='purple', alpha=0.8)
    axes[1].set_yscale('log'); axes[1].set_xlabel('total strips firing per event')
    axes[1].set_ylabel('events'); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Total strip multiplicity')
    fig.suptitle(f'{CFG.DET_NAME} event multiplicity — {CFG.RUN}/{CFG.SUB_RUN}')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/event_multiplicity.png', dpi=150, bbox_inches='tight')

    for thr in (20, 50, 100):
        frac = (mult['ntot'] > thr).mean()
        summary.append(f'  events with >{thr} total strips: {frac*100:.2f}%')
    summary.append(f'  median total strips/event: {mult["ntot"].median():.0f}, '
                   f'mean: {mult["ntot"].mean():.1f}, max: {mult["ntot"].max()}')


def plot_mult_vs_time(df, mult, out_dir):
    ts = df.groupby('eventId')['trigger_timestamp_ns'].first()
    m = mult.join(ts.rename('ts'))
    t = (m['ts'] - m['ts'].min()) / 1e9
    from scipy.stats import binned_statistic
    mean_m, edges, _ = binned_statistic(t, m['ntot'], statistic='mean', bins=80)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ctr, mean_m, 'o-', ms=3, color='purple')
    ax.set_xlabel('Time since run start [s]'); ax.set_ylabel('mean strips/event')
    ax.set_title(f'{CFG.DET_NAME} mean multiplicity vs time — {CFG.RUN}/{CFG.SUB_RUN}')
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(f'{out_dir}/multiplicity_vs_time.png', dpi=150, bbox_inches='tight')


def main():
    out_dir = CFG.out_dir('deep_qa')
    df = load_hits()
    n_events = df['eventId'].nunique()
    summary = [f'Deep QA — {CFG.DET_NAME}  {CFG.RUN}/{CFG.SUB_RUN}',
               f'  total events with any hit: {n_events:,}',
               f'  total hits: {len(df):,}  ({len(df)/n_events:.1f} hits/event)',
               f'  X strips firing: {df[df.feu==CFG.MX17_FEU_X]["channel"].nunique()} distinct, '
               f'Y strips: {df[df.feu==CFG.MX17_FEU_Y]["channel"].nunique()} distinct']

    plot_surface_hitmap(df, out_dir)
    stripdf = strip_firing(df, n_events)
    plot_strip_firing(stripdf, out_dir, summary)
    mult = multiplicity(df)
    plot_multiplicity(mult, out_dir, summary)
    plot_mult_vs_time(df, mult, out_dir)

    txt = '\n'.join(summary)
    print(txt)
    with open(f'{out_dir}/deep_qa_summary.txt', 'w') as f:
        f.write(txt + '\n')
    print(f'\nDeep QA written to: {out_dir}')


if __name__ == '__main__':
    main()
