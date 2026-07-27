#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_plastic_amplitude.py -- is the plastic PSA cutting into the real spectrum?

dream_trigger found that ~60 % of DREAM events have no plastic hit near the
trigger at all, while 96.5 % have a wall segment sum over threshold. One candidate
explanation is a processing threshold set too high on the plastic trees, which
would show up as the amplitude spectrum being truncated in its bulk rather than
dying away.

The spectra are truncated -- hard zero below a sharp edge, with the distribution
still at full height there -- and the plastic edge is twice the wall edge:

    PSSA-D   100.0 ADC        WALA-D   50.0 ADC

so the plastic PSA really is cutting the spectrum, and harder than the wall's. But
that edge sits at ~3.1 mV, while the DREAM plastic discriminator ran at 118-157 mV
(3800-5100 ADC) -- more than an order of magnitude above it. So the cut is real and
worth fixing, but it CANNOT be what removes trigger-level plastic pulses: nothing
near the trigger threshold is anywhere near the edge.

Panels: the spectra with both the PSA edge and the trigger threshold marked, on a
log-x axis so both scales are visible at once, plus the amplitudes of the plastic
hits that genuinely match a DREAM event.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.beam_july_paths import ANALYSIS_DIR                       # noqa: E402
from ntof_dream_merge.ntof_io import read_bunches                     # noqa: E402
from ntof_dream_merge.bunch_join import dream_event_to_bunch          # noqa: E402
from ntof_dream_merge.dream_trigger import load_thresholds, load_adc_mv  # noqa: E402

ARM_COLOR = {'A': '#3b82f6', 'B': '#f59e0b', 'C': '#10b981', 'D': '#ef4444'}
INK, MUTED, GRID = '#1f2328', '#5b6169', '#d8dbe0'
K, T0 = 1.089e-4, -197.5
BANDS = ((-150.0, 150.0), (250.0, 450.0))


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def collect(ntof_run, bunches, ev, thr, adc):
    out = {}
    inband = lambda r: ((r >= BANDS[0][0]) & (r <= BANDS[0][1])) | \
                       ((r >= BANDS[1][0]) & (r <= BANDS[1][1]))
    late = ev[ev['t_since_flash_ns'] > 20e6]
    for kind in ('PSS', 'WAL'):
        for arm in 'ABCD':
            tree = f'{kind}{arm}'
            h = read_bunches(ntof_run, tree, bunches,
                             branches=('BunchNumber', 'detn', 'amp'))
            sel = h['t_since_flash_ns'] > 20e6
            allamp = h['amp'][sel]
            o = np.lexsort((h['t_since_flash_ns'], h['BunchNumber']))
            cb, ct, ca = h['BunchNumber'][o], h['t_since_flash_ns'][o], h['amp'][o]
            got = []
            for b, g in late.groupby('BunchNumber'):
                s, e = np.searchsorted(cb, [b, b + 1])
                tt, aa = ct[s:e], ca[s:e]
                if tt.size == 0:
                    continue
                et = g['t_since_flash_ns'].to_numpy().astype(float)
                pred = et + K * et + T0
                lo = np.searchsorted(tt, pred - 500.)
                hi = np.searchsorted(tt, pred + 500.)
                for j in range(et.size):
                    if hi[j] <= lo[j]:
                        continue
                    m = inband(tt[lo[j]:hi[j]] - pred[j])
                    if m.any():
                        got.append(aa[lo[j]:hi[j]][m])
            out[tree] = (allamp, np.concatenate(got) if got else np.array([]))
    return out


def make_figure(data, thr, adc, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.patch.set_facecolor('white')
    bins = np.geomspace(30, 16384, 160)

    for ax, kind, title in ((axes[0], 'PSS', '1. plastics: hard edge at 100 ADC'),
                            (axes[1], 'WAL', '2. walls: hard edge at 50 ADC')):
        for arm in 'ABCD':
            a = data[f'{kind}{arm}'][0]
            if a.size:
                ax.hist(a, bins=bins, histtype='step', lw=1.8,
                        color=ARM_COLOR[arm], label=f'{kind}{arm}')
        edge = 100 if kind == 'PSS' else 50
        ax.axvline(edge, color=INK, lw=1.5, ls='--')
        ax.annotate(f'PSA edge\n{edge} ADC', xy=(edge * 1.12, ax.get_ylim()[1] * 0.35),
                    fontsize=8, color=INK)
        if kind == 'PSS':
            for arm in 'ABCD':
                t_adc = thr['plastic'][arm] / adc[f'PSS{arm}'][0]
                ax.axvline(t_adc, color=ARM_COLOR[arm], lw=1.2, ls=':', alpha=0.9)
            ax.annotate('DREAM trigger\nthreshold\n(118-157 mV)',
                        xy=(4300, ax.get_ylim()[1] * 0.30), fontsize=8, color=INK)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('pulse amplitude [ADC]')
        ax.set_ylabel('hits')
        ax.set_title(title, fontsize=9.5, color=INK, loc='left')
        ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc='upper right')
        _style(ax)

    ax = axes[2]
    for arm in 'ABCD':
        a = data[f'PSS{arm}'][1]
        if a.size:
            ax.hist(a, bins=bins, histtype='step', lw=1.8,
                    color=ARM_COLOR[arm], label=f'PSS{arm}')
    ax.axvline(100, color=INK, lw=1.5, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('pulse amplitude [ADC]')
    ax.set_ylabel('hits matching a DREAM event')
    ax.set_title('3. plastic hits that DO match a DREAM event', fontsize=9.5,
                 color=INK, loc='left')
    ax.annotate('trigger-level hits sit here,\nfar above the PSA edge',
                xy=(0.30, 0.86), xycoords='axes fraction', fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc='upper left')
    _style(ax)

    fig.suptitle('n_TOF plastic vs wall amplitude spectra (run224572, t > 20 ms)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white')
    print(f'wrote {out}')


if __name__ == '__main__':
    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    sel = ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
    thr, adc = load_thresholds(run, sub), load_adc_mv()
    data = collect(nt, bunches, sel, thr, adc)
    for tree in ('PSSA', 'PSSC', 'WALA', 'WALC'):
        a, m = data[tree]
        print(f'  {tree}: {a.size:8,} hits, min {a.min():6.1f} ADC | '
              f'{m.size:6,} matched, median {np.median(m) if m.size else float("nan"):8.1f} ADC')
    make_figure(data, thr, adc,
                ANALYSIS_DIR / 'ntof_dream_merge' / 'figures' /
                f'plastic_amplitude_{nt}.png')
