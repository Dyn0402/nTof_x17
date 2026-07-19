"""21_y88_spectra.py — Y-88 source-scan amplitude spectra, per channel, mV.

Y-88 source placed between the two plastic bars of ONE arm per run (source-run
analysis, NOT a beam analysis — see HANDOFF_Y88_SCAN.md). Run -> illuminated
arm:  224476=A  224477=B  224478=C  224479=D.

This is T1 of the handoff: per-channel linear + log amplitude spectra, mV-
calibrated, for the source arm's PSS (2 ch) and WAL (8 ch). The non-illuminated
arms in the SAME run are recorded too and used as a background/no-strong-source
shape reference (same acquisition, different physical detector).

Because the source is bright the illuminated-arm plastics have 5-6.5 M hits, so
statistics are not the limitation the handoff feared — the spectra are smooth.
The observable landmark is the Compton edge (organic scintillator, no photopeak):
898 keV gamma -> 699 keVee edge, 1836 keV gamma -> 1612 keVee edge.

Work in LINEAR amplitude for edge extraction (dN/dlogA peaks sit high — the
lesson already paid for in the run224489 MIP analysis). Log bins are for the
overview only.

Outputs:
  cache/21_y88_<run>.npz     fine linear hist (mV) + log hist per tree/channel
  figures/21_y88/spectra_<run>.png    source-arm overview (PSS + WAL)
Usage:
  python 21_y88_spectra.py [run_stem ...]   (default: all four)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hitcache
from adc_mv import mv_factors

BASE = Path(__file__).parent
DATA = Path.home() / 'x17' / 'beam_july' / 'data'
CACHE = BASE / 'cache'
OUT = BASE / 'figures' / '21_y88'
CACHE.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

RUN_ARM = {'run224476': 'A', 'run224477': 'B', 'run224478': 'C', 'run224479': 'D'}
ARMS = 'ABCD'
NCH = {'PSS': 2, 'WAL': 8, 'LIQ': 1}

# fine linear grid for edge extraction (T2 rebins / bootstraps these bins);
# 0.2 mV bins to 160 mV covers every source feature seen (plastic tail ~80 mV,
# wall 1612 keV edge expected ~80-105 mV). Log grid is overview-only.
LIN_EDGES = np.arange(0.0, 160.0 + 1e-9, 0.2)
LOG_EDGES = np.geomspace(0.5, 1000.0, 121)


def spectra_for_run(run_stem):
    run = DATA / f'{run_stem}.root'
    fac = mv_factors(run)
    arm = RUN_ARM[run_stem]
    store = {'run': run_stem, 'source_arm': arm,
             'lin_edges': LIN_EDGES, 'log_edges': LOG_EDGES}
    for kind in ('WAL', 'PSS', 'LIQ'):
        for a in ARMS:
            tree = f'{kind}{a}'
            d = hitcache.load(run, tree, ['amp', 'detn'])
            amp = d['amp'].astype(np.float64)
            detn = d['detn']
            f = fac[tree]
            lin = np.zeros((NCH[kind], len(LIN_EDGES) - 1))
            log = np.zeros((NCH[kind], len(LOG_EDGES) - 1))
            for c in range(NCH[kind]):
                m = detn == (c + 1)
                a_mv = amp[m] * f[c]
                lin[c] = np.histogram(a_mv, bins=LIN_EDGES)[0]
                log[c] = np.histogram(a_mv, bins=LOG_EDGES)[0]
            store[f'{tree}_lin'] = lin
            store[f'{tree}_log'] = log
    np.savez_compressed(CACHE / f'21_y88_{run_stem}.npz', **store)
    print(f'{run_stem} (arm {arm}) cached -> cache/21_y88_{run_stem}.npz')
    return store


def overview_figure(store):
    run, arm = store['run'], store['source_arm']
    le = store['lin_edges']
    cen = 0.5 * (le[:-1] + le[1:])
    bg_arm = 'B' if arm != 'B' else 'C'      # a non-illuminated arm, shape ref

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.25, 1.25, 1, 1])

    # --- plastics: the clean, source-dominated spectra (2 channels) ---
    for c in range(2):
        ax = fig.add_subplot(gs[0, c * 2:c * 2 + 2])
        sig = store[f'PSS{arm}_lin'][c]
        bg = store[f'PSS{bg_arm}_lin'][c]
        ax.step(cen, sig, where='mid', color='crimson', lw=1.3,
                label=f'PSS{arm}{c + 1} (source arm)')
        ax.step(cen, bg, where='mid', color='gray', lw=1.0, alpha=0.8,
                label=f'PSS{bg_arm}{c + 1} (no source, shape ref)')
        ax.set_yscale('log')
        ax.set_xlim(0, 120)
        ax.set_ylim(bottom=1)
        ax.set_xlabel('amplitude [mV]')
        ax.set_ylabel('hits / 0.2 mV')
        ax.set_title(f'PSS{arm}{c + 1}  (n={sig.sum():,.0f})', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # --- liquid: single channel, source arm (first-ever LIQ Y-88 spectrum) ---
    ax = fig.add_subplot(gs[1, 0:2])
    sigL = store[f'LIQ{arm}_lin'][0]
    bgL = store[f'LIQ{bg_arm}_lin'][0]
    ax.step(cen, sigL, where='mid', color='darkgreen', lw=1.3,
            label=f'LIQ{arm} (source arm)')
    ax.step(cen, bgL, where='mid', color='gray', lw=1.0, alpha=0.8,
            label=f'LIQ{bg_arm} (no source)')
    ax.set_yscale('log')
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=1)
    ax.set_xlabel('amplitude [mV]')
    ax.set_ylabel('hits / 0.2 mV')
    ax.set_title(f'LIQ{arm}  (n={sigL.sum():,.0f})', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- walls: 8 channels, source arm (rows: gs[1,2], gs[1,3], gs[2,*], gs[3,*]) ---
    wall_slots = [(1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)]
    for c in range(8):
        ax = fig.add_subplot(gs[wall_slots[c]])
        sig = store[f'WAL{arm}_lin'][c]
        ax.step(cen, sig, where='mid', color='steelblue', lw=1.0)
        ax.set_yscale('log')
        ax.set_xlim(0, 120)
        ax.set_ylim(bottom=1)
        ax.set_title(f'WAL{arm}{c + 1}  (n={sig.sum():,.0f})', fontsize=8)
        ax.tick_params(labelsize=7)
        row, col = wall_slots[c]
        if row == 3:
            ax.set_xlabel('amplitude [mV]', fontsize=8)
        if col == 0:
            ax.set_ylabel('hits / 0.2 mV', fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f'{run}: Y-88 source between the plastic bars of arm {arm} — '
                 f'per-channel amplitude spectra (linear mV, log-y). '
                 f'Compton edges expected: 699 & 1612 keVee.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / f'spectra_{run}.png'
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f'  -> {p}')


def main():
    stems = sys.argv[1:] or list(RUN_ARM)
    for s in stems:
        store = spectra_for_run(s)
        overview_figure(store)


if __name__ == '__main__':
    main()
