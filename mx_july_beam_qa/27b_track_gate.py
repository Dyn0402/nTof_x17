#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27b_track_gate.py — stringent "real track" selection for run_55, and its
validation.

The gate (trackcache.GATE / tag_tracklike / match_xy) requires, per chamber:
  * FULL-GAP TRAVERSAL — a cluster whose drift-time occupancy
    t_occ = max(right_sample) - min(left_sample) reaches the full ~30 mm gap
    (>= 8 of the ~11-12-sample gap crossing), realised EITHER as
      - an inclined MONOTONIC micro-TPC trail (4-20 strips, <=25 mm,
        |corr(pos,drift-sample)| >= 0.8), OR
      - a HEAD-ON long/saturated single pulse (<=4 strips, pulse width
        >= 8 samples) — the cosmics "hybrid" regime;
  * X<->Y CONSISTENCY — the x and y track-like clusters of the same event must
    overlap in ABSOLUTE drift-time window (IoU >= 0.3), match in time-length
    (|Δt_occ| <= 6 samples), and balance charge (|f-0.49|/0.10 <= 3).

Validation (this script):
  * event-shuffled NULL (pair each event's X with a random other event's Y in
    the same det) — real pairs must beat accidentals in IoU and charge balance;
  * per-detector survival cascade candidates -> compact -> full-gap ->
    monotonic -> X/Y matched;
  * charge-balance of matched pairs vs the June bench value (independent
    real-track check);
  * yield vs HV and vs in-gate time (b1 8-12 ms / b2 16-28 ms).

Outputs: figures/27_tracks/*.png, calib/27_tracks.npz (the clean 3D-segment
table consumed by 27c_source_align.py), console summary.

Run:  venv/bin/python mx_july_beam_qa/27b_track_gate.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import trackcache as tc

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, 'figures', '27_tracks')
CALIB = os.path.join(HERE, 'calib')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CALIB, exist_ok=True)


def b12(t):
    return (t > 7) & (t < 30)


def main():
    cl, ss, ev = tc.load_all()
    cl = tc.add_derived(cl, ss)
    evk = ev.set_index(['sub', 'eid'])
    cl = cl.join(evk[['t_ms', 'flash_ok']], on=['sub', 'eid'])
    cl['b12'] = b12(cl['t_ms'])
    cl = tc.tag_tracklike(cl)

    # restrict physics to b1/b2 (the only sampled, recovered windows)
    clw = cl[cl['b12']].copy()

    # ---- survival cascade per detector ----
    print('\n=== survival cascade (b1/b2 windows) ===')
    casc = []
    for dn in 'ABCD':
        d = clw[clw['detn'] == dn]
        n_all = len(d)
        n_comp = int(((d['n'] >= 4) & (d['n'] <= 20) & (d['extent'] <= 25)).sum())
        n_fg = int(((d['n'] >= 4) & (d['n'] <= 20) & (d['extent'] <= 25) &
                    (d['dur'] >= 6)).sum())
        n_tl = int(d['tracklike'].sum())
        n_inc = int(d['inclined'].sum())
        n_ho = int(d['headon'].sum())
        casc.append((dn, n_all, n_comp, n_fg, n_tl, n_inc, n_ho))
        print(f'  {dn}: candidates {n_all:6d} -> compact {n_comp:5d} -> '
              f'full-gap {n_fg:4d} -> track-like {n_tl:4d} '
              f'(inclined {n_inc}, head-on {n_ho})')

    # ---- X/Y matching: real + null ----
    tl = clw[clw['tracklike']].copy()
    M = tc.match_xy(tl)
    N = pd.concat([tc.match_xy(tl, shuffle_y=True, seed=s) for s in range(5)],
                  ignore_index=True)
    print(f'\n=== X/Y matched 3D segments: {len(M)} real, '
          f'{len(N)//5:.0f}/shuffle null ===')
    print(M.groupby('detn').agg(
        n=('iou', 'size'), iou_med=('iou', 'median'),
        fbal_med=('fbal', 'median'), fbal_sd=('fbal', 'std'),
        durdiff_med=('durdiff', 'median')).round(3))

    # The right discriminant is NOT match count (shared trigger timing means a
    # shuffled Y always finds an overlapping partner) but the CHARGE-BALANCE
    # WIDTH: real same-particle pairs cluster tightly at the bench f_bal, the
    # accidental (event-shuffled) background is broad.  Plus: multiplicity ~1
    # per plane makes the true pairing unambiguous.
    mult = tl.groupby(['sub', 'eid', 'det', 'pln']).size()
    print(f'\n  track-like multiplicity per plane: mean {mult.mean():.2f} '
          f'(1: {int((mult==1).sum())}, >=2: {int((mult>=2).sum())})')
    print('  charge-balance width, real vs event-shuffled null:')
    for dn in 'ABCD':
        r = M[M['detn'] == dn]; a = N[N['detn'] == dn]
        if len(r) == 0:
            continue
        print(f'    {dn}: real f_bal {r["fbal"].median():.3f}+-{r["fbal"].std():.3f}'
              f'  null {a["fbal"].median():.3f}+-{a["fbal"].std():.3f}  '
              f'(tighter = real correlation)')

    # ================= figures =================
    # Fig 1: gate discriminators per det
    fig, ax = plt.subplots(3, 4, figsize=(19, 12))
    for di, dn in enumerate('ABCD'):
        d = clw[clw['detn'] == dn]
        comp = d[(d['n'] >= 4) & (d['n'] <= 20) & (d['extent'] <= 25)]
        tlk = d[d['tracklike']]
        ax[0, di].hist(comp['dur'], bins=np.arange(0, 34), histtype='step',
                       density=True, label='compact')
        ax[0, di].hist(tlk['dur'], bins=np.arange(0, 34), histtype='stepfilled',
                       alpha=0.3, density=True, label='track-like')
        ax[0, di].axvline(6, color='r', ls='--')
        ax[0, di].set_title(f'{dn}: peak-time trail span dur [smp]')
        ax[0, di].legend(fontsize=7)
        ax[1, di].hist(comp['mono'].dropna(), bins=np.linspace(-1, 1, 41),
                       histtype='step', density=True, label='compact')
        cfg = comp[comp['dur'] >= 6]
        ax[1, di].hist(cfg['mono'].dropna(), bins=np.linspace(-1, 1, 41),
                       histtype='stepfilled', alpha=0.3, density=True,
                       label='compact+full-gap')
        ax[1, di].axvspan(-1, -0.8, color='g', alpha=0.1)
        ax[1, di].axvspan(0.8, 1, color='g', alpha=0.1)
        ax[1, di].set_title(f'{dn}: trail monotonicity'); ax[1, di].legend(fontsize=7)
        ax[2, di].scatter(d['n'], d['wfmax'], s=2, alpha=0.1)
        ho = d[d['headon']]
        ax[2, di].scatter(ho['n'], ho['wfmax'], s=6, color='r', label='head-on')
        ax[2, di].axhline(8, color='r', ls='--'); ax[2, di].set_xlim(0, 25)
        ax[2, di].set_xlabel('n_strips'); ax[2, di].set_ylabel('max pulse width')
        ax[2, di].set_title(f'{dn}: head-on population'); ax[2, di].legend(fontsize=7)
    fig.suptitle('27b gate discriminators (b1/b2)', y=1.005)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '01_discriminators.png'), dpi=95)
    plt.close(fig)

    # Fig 2: real vs null (IoU, charge balance) — the key validation
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(M['iou'], bins=np.linspace(0, 1, 30), histtype='step',
               density=True, color='C0', lw=2, label='matched (real)')
    ax[0].hist(N['iou'], bins=np.linspace(0, 1, 30), histtype='step',
               density=True, color='k', ls=':', label='event-shuffled null')
    ax[0].set_xlabel('X/Y drift-window IoU'); ax[0].legend(); ax[0].set_title('window overlap')
    ax[1].hist(M['fbal'], bins=np.linspace(0, 1, 30), histtype='step',
               density=True, color='C0', lw=2, label='matched')
    ax[1].hist(N['fbal'], bins=np.linspace(0, 1, 30), histtype='step',
               density=True, color='k', ls=':', label='null')
    ax[1].axvline(0.49, color='r', ls='--', label='bench f_bal')
    ax[1].set_xlabel('charge balance f = qX/(qX+qY)'); ax[1].legend()
    ax[1].set_title('charge balance (real tracks -> bench value)')
    # yield vs HV per det
    yv = M.groupby(['detn', 'resist_v']).size().reset_index(name='ntrk')
    for dn in 'ABCD':
        s = yv[yv['detn'] == dn]
        ax[2].plot(s['resist_v'], s['ntrk'], 'o-', label=dn)
    ax[2].set_xlabel('resist HV [V]'); ax[2].set_ylabel('clean tracks (b1/b2, all subruns)')
    ax[2].legend(); ax[2].set_title('clean-track yield vs HV')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '02_validation.png'), dpi=95)
    plt.close(fig)

    # Fig 3: survival cascade bar + source-pointing preview
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cdf = pd.DataFrame(casc, columns=['det', 'cand', 'compact', 'fullgap',
                                      'tracklike', 'inc', 'ho'])
    xpos = np.arange(4)
    for i, col in enumerate(['cand', 'compact', 'fullgap', 'tracklike']):
        ax[0].bar(xpos + i * 0.2, cdf[col], width=0.2, label=col)
    ax[0].set_yscale('log'); ax[0].set_xticks(xpos + 0.3); ax[0].set_xticklabels(cdf['det'])
    ax[0].set_ylabel('clusters (b1/b2)'); ax[0].legend(); ax[0].set_title('survival cascade')
    # source-pointing preview: x-slope vs xcen for matched inclined pairs
    mi = M[~M['x_headon']]
    ax[1].scatter(mi['xcen'], mi['xslope'], s=8, alpha=0.4, c=mi['det'], cmap='tab10')
    ax[1].set_xlabel('x centroid [mm]'); ax[1].set_ylabel('x anchored slope [ns/mm]')
    ax[1].set_title('source-pointing preview (matched inclined)')
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, '03_cascade.png'), dpi=95)
    plt.close(fig)

    # ---- save the clean 3D-segment table for 27c ----
    outp = os.path.join(CALIB, '27_tracks.npz')
    np.savez_compressed(outp, **{c: M[c].values for c in M.columns})
    print(f'\nsaved {len(M)} clean 3D segments -> {outp}')
    print('figures -> figures/27_tracks/{01_discriminators,02_validation,03_cascade}.png')


if __name__ == '__main__':
    main()
