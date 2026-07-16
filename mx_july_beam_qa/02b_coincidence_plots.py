"""
02b_coincidence_plots.py — Figures from the 02 coincidence cache: dt matrices,
dt by time-since-flash region, and the coincidence-window scan.

Usage: python 02b_coincidence_plots.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '02_coincidence'
OUT.mkdir(parents=True, exist_ok=True)

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
REGION_COLORS = ['#9ca3af', '#7c3aed', '#1d4ed8', '#0d9488', '#ca8a04', '#c2410c']

d = np.load(BASE / 'cache' / f'02_coinc_{RUN_STEM}.npz')
DT_CEN = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
LABELS = list(d['region_labels'])
SIDE = np.abs(DT_CEN) > 100


def peak_pos(hh):
    base = hh[SIDE].mean()
    return DT_CEN[np.argmax(hh - base)]


def dt_matrix():
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i, w in enumerate(WALL_TREES):
        for j, p in enumerate(PSS_TREES):
            ax = axes[i][j]
            hh = d[f'{w}_{p}'].sum(axis=(0, 1))
            ax.plot(DT_CEN, hh, lw=0.9, color='#1d4ed8')
            base = hh[SIDE].mean()
            ax.axhline(base, color='#9ca3af', lw=0.8, ls='--')
            exc = hh.max() - base
            ax.set_title(f'{w} - {p}   (peak excess {exc:,.0f} @ {peak_pos(hh):+.0f} ns)',
                         fontsize=9)
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            if i == 3:
                ax.set_xlabel('t(wall) - t(pss) [ns]')
            if j == 0:
                ax.set_ylabel('pairs / ns')
    fig.suptitle(f'{RUN_STEM}: WAL-PSS time differences, all regions summed '
                 '(dashed = combinatorial level from |dt|>100 ns)')
    fig.tight_layout()
    fig.savefig(OUT / 'dt_matrix.png', dpi=140)
    plt.close(fig)


def dt_by_region():
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i, w in enumerate(WALL_TREES):
        for j, p in enumerate(PSS_TREES):
            ax = axes[i][j]
            hr = d[f'{w}_{p}'].sum(axis=0)  # (region, dt)
            for r, lab in enumerate(LABELS):
                if hr[r].sum() < 100:
                    continue
                ax.plot(DT_CEN, hr[r], lw=0.8, color=REGION_COLORS[r], label=lab)
            ax.set_title(f'{w} - {p}', fontsize=9)
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            if i == 0 and j == 3:
                ax.legend(fontsize=7, frameon=False)
            if i == 3:
                ax.set_xlabel('t(wall) - t(pss) [ns]')
            if j == 0:
                ax.set_ylabel('pairs / ns')
    fig.suptitle(f'{RUN_STEM}: WAL-PSS dt by time since gamma flash (pss hit)')
    fig.tight_layout()
    fig.savefig(OUT / 'dt_by_region.png', dpi=140)
    plt.close(fig)


def window_scan():
    """Counts inside |dt - dt_peak| <= W vs W: total, combinatorial expectation, excess."""
    ws = np.arange(1, 101)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for i, w in enumerate(WALL_TREES):
        for j, p in enumerate(PSS_TREES):
            ax = axes[i][j]
            hh = d[f'{w}_{p}'].sum(axis=(0, 1))
            dpk = peak_pos(hh)
            base = hh[SIDE].mean()
            tot = np.array([hh[np.abs(DT_CEN - dpk) <= W].sum() for W in ws])
            nbin = np.array([(np.abs(DT_CEN - dpk) <= W).sum() for W in ws])
            ax.plot(ws, tot, lw=1.2, color='#1d4ed8', label='total in window')
            ax.plot(ws, base * nbin, lw=1.2, ls='--', color='#9ca3af', label='combinatorial')
            ax.plot(ws, tot - base * nbin, lw=1.2, color='#c2410c', label='excess')
            ax.set_title(f'{w} - {p}', fontsize=9)
            ax.set_yscale('log')
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
            if i == 3:
                ax.set_xlabel('window half-width W [ns]')
            if j == 0:
                ax.set_ylabel('pairs')
    fig.suptitle(f'{RUN_STEM}: coincidence-window scan (centered on each pair dt peak)')
    fig.tight_layout()
    fig.savefig(OUT / 'window_scan.png', dpi=140)
    plt.close(fig)


def window_scan_regions(pairs):
    """Per-region window scan for selected pairs: purity & excess vs W."""
    ws = np.arange(1, 101)
    for w, p in pairs:
        hr = d[f'{w}_{p}'].sum(axis=0)
        hh = hr.sum(axis=0)
        dpk = peak_pos(hh)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for r, lab in enumerate(LABELS):
            if hr[r].sum() < 1000:
                continue
            base = hr[r][SIDE].mean()
            tot = np.array([hr[r][np.abs(DT_CEN - dpk) <= W].sum() for W in ws])
            nbin = np.array([(np.abs(DT_CEN - dpk) <= W).sum() for W in ws])
            exc = tot - base * nbin
            axes[0].plot(ws, tot, lw=1.1, color=REGION_COLORS[r], label=lab)
            axes[1].plot(ws, exc, lw=1.1, color=REGION_COLORS[r], label=lab)
            with np.errstate(divide='ignore', invalid='ignore'):
                axes[2].plot(ws, np.where(tot > 0, exc / tot, np.nan), lw=1.1,
                             color=REGION_COLORS[r], label=lab)
        axes[0].set_ylabel('total pairs in window')
        axes[1].set_ylabel('excess (true) pairs')
        axes[2].set_ylabel('purity = excess / total')
        axes[2].set_ylim(0, 1.05)
        for ax in axes[:2]:
            ax.set_yscale('log')
        for ax in axes:
            ax.set_xlabel('window half-width W [ns]')
            ax.grid(alpha=0.25)
        axes[1].legend(fontsize=8, frameon=False)
        fig.suptitle(f'{RUN_STEM}: {w}-{p} window scan by time since flash '
                     f'(peak at {dpk:+.0f} ns)')
        fig.tight_layout()
        fig.savefig(OUT / f'window_scan_regions_{w}_{p}.png', dpi=140)
        plt.close(fig)


def strongest_pairs(n=4):
    scored = []
    for w in WALL_TREES:
        for p in PSS_TREES:
            hh = d[f'{w}_{p}'].sum(axis=(0, 1))
            base = hh[SIDE].mean()
            dpk = peak_pos(hh)
            win = np.abs(DT_CEN - dpk) <= 10
            scored.append((hh[win].sum() - base * win.sum(), w, p))
    scored.sort(reverse=True)
    return [(w, p) for _, w, p in scored[:n]]


if __name__ == '__main__':
    dt_matrix()
    dt_by_region()
    window_scan()
    window_scan_regions(strongest_pairs())
    print(f'Figures -> {OUT}')
