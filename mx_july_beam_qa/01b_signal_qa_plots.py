"""
01b_signal_qa_plots.py — Render QA figures from the histogram cache produced by
01_signal_qa.py. Kept separate so plots can be iterated without re-reading 13 GB.

Usage: python 01b_signal_qa_plots.py [run_stem]   (default run224404)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '01_signal_qa'
OUT.mkdir(parents=True, exist_ok=True)

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
ALL_TREES = WALL_TREES + PSS_TREES
N_CH = {**{t: 8 for t in WALL_TREES}, **{t: 2 for t in PSS_TREES}}

# Fixed channel colors (identity follows the channel, never re-cycled)
CH_COLORS = plt.get_cmap('tab10').colors
DEDICATED_THRESH = 6e12

d = np.load(BASE / 'cache' / f'01_qa_{RUN_STEM}.npz')
centers = {k: np.sqrt(d[f'{k}_edges'][:-1] * d[f'{k}_edges'][1:]) for k in ('amp', 'area', 'fwhm', 'tof')}
flash_centers = 0.5 * (d['flash_edges'][:-1] + d['flash_edges'][1:])
ded = d['bunch_intensity'] > DEDICATED_THRESH


def spectra_grid(kind, xlabel, fname):
    to_mv = None
    if kind == 'amp':
        try:
            from adc_mv import mv_factors
            to_mv = mv_factors()
            xlabel = 'amplitude [mV]'
        except Exception:
            pass
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey='row')
    for ax, tn in zip(axes.flat, ALL_TREES):
        h = d[f'{tn}_{kind}']
        for ch in range(N_CH[tn]):
            x = centers[kind] * (to_mv[tn][ch] if to_mv is not None else 1.0)
            ax.plot(x, h[ch], lw=1, color=CH_COLORS[ch], label=f'ch{ch + 1}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(tn)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel('hits / bin')
    fig.suptitle(f'{RUN_STEM}: per-channel {kind} spectra (top: SiPM walls, bottom: plastics)')
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def tof_grid():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey='row')
    for ax, tn in zip(axes.flat, ALL_TREES):
        h = d[f'{tn}_tof']
        widths = np.diff(d['tof_edges'])
        for ch in range(N_CH[tn]):
            ax.plot(centers['tof'] * 1e-3, h[ch] / widths * 1e3, lw=0.8,
                    color=CH_COLORS[ch], label=f'ch{ch + 1}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(tn)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('time since acquisition start [us]')
    for ax in axes[:, 0]:
        ax.set_ylabel('hits / us')
    fig.suptitle(f'{RUN_STEM}: hit-time distributions, full 20 ms window (gamma flash at ~11 us)')
    fig.tight_layout()
    fig.savefig(OUT / 'tof_per_channel.png', dpi=150)
    plt.close(fig)


def flash_zoom():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey='row')
    for ax, tn in zip(axes.flat, ALL_TREES):
        h = d[f'{tn}_flash']
        for ch in range(N_CH[tn]):
            ax.plot(flash_centers * 1e-3, h[ch], lw=0.8, color=CH_COLORS[ch], label=f'ch{ch + 1}')
        ax.set_yscale('log')
        ax.set_title(tn)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('time since acquisition start [us]')
    for ax in axes[:, 0]:
        ax.set_ylabel('hits / 10 ns')
    fig.suptitle(f'{RUN_STEM}: gamma-flash region zoom')
    fig.tight_layout()
    fig.savefig(OUT / 'tof_flash_zoom.png', dpi=150)
    plt.close(fig)


def amp_vs_tof():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, tn in zip(axes.flat, ALL_TREES):
        h = d[f'{tn}_amp_vs_tof']
        m = ax.pcolormesh(d['tof_edges'] * 1e-3, d['amp_edges'], h.T,
                          norm=LogNorm(vmin=1), cmap='viridis', rasterized=True)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(tn)
    fig.colorbar(m, ax=axes, label='hits / bin', shrink=0.85)
    for ax in axes[1]:
        ax.set_xlabel('time since acquisition start [us]')
    for ax in axes[:, 0]:
        ax.set_ylabel('amplitude [ADC]')
    fig.suptitle(f'{RUN_STEM}: amplitude vs hit time (all channels summed)')
    fig.savefig(OUT / 'amp_vs_tof.png', dpi=150)
    plt.close(fig)


def rates_per_bunch():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex='col')
    for ax, tn in zip(axes.flat, ALL_TREES):
        hpb = d[f'{tn}_hits_per_bunch'].sum(axis=0)  # summed over channels
        bins = np.linspace(0, np.percentile(hpb, 99.8) * 1.1, 80)
        ax.hist(hpb[ded], bins=bins, histtype='step', lw=1.5, color='#c2410c',
                label=f'dedicated (med {np.median(hpb[ded]):.0f})')
        ax.hist(hpb[~ded], bins=bins, histtype='step', lw=1.5, color='#1d4ed8',
                label=f'parasitic (med {np.median(hpb[~ded]):.0f})')
        ax.set_title(tn)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('hits per bunch (all channels)')
    for ax in axes[:, 0]:
        ax.set_ylabel('bunches')
    fig.suptitle(f'{RUN_STEM}: hits per bunch by pulse type')
    fig.tight_layout()
    fig.savefig(OUT / 'hits_per_bunch.png', dpi=150)
    plt.close(fig)


def channel_stability():
    """Mean hits/bunch per channel vs bunch number (trend over the run), dedicated only."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    ded_idx = np.where(ded)[0]
    for ax, tn in zip(axes.flat, ALL_TREES):
        hpb = d[f'{tn}_hits_per_bunch'][:, ded_idx]
        n_grp = max(len(ded_idx) // 40, 1)
        for ch in range(N_CH[tn]):
            n_full = (len(ded_idx) // n_grp) * n_grp
            trend = hpb[ch, :n_full].reshape(-1, n_grp).mean(axis=1)
            x = ded_idx[:n_full].reshape(-1, n_grp).mean(axis=1)
            ax.plot(x, trend, lw=1, color=CH_COLORS[ch], label=f'ch{ch + 1}')
        ax.set_title(tn)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('bunch number')
    for ax in axes[:, 0]:
        ax.set_ylabel('hits / dedicated bunch')
    fig.suptitle(f'{RUN_STEM}: per-channel rate stability over the run (dedicated bunches)')
    fig.tight_layout()
    fig.savefig(OUT / 'rate_stability.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    spectra_grid('amp', 'amplitude [ADC]', 'amp_spectra.png')
    spectra_grid('area', 'area [ADC*ns]', 'area_spectra.png')
    spectra_grid('fwhm', 'FWHM [ns]', 'fwhm_spectra.png')
    tof_grid()
    flash_zoom()
    amp_vs_tof()
    rates_per_bunch()
    channel_stability()
    print(f'Figures -> {OUT}')
