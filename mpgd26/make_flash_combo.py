#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_flash_combo.py -- the two DAQ switch-on times and the X17 spectrum, on one
time axis.  Variations on ``status_two_readouts_op`` (make_flash_slides.py),
which is left as it is.

    ../.venv/bin/python make_flash_combo.py [--only stack,overlay,lines]

The X17 curve is slide 31's (``make_x17_rate.load()``: the table's own
numbers, one point per decade of neutron energy, PCHIP through them), pared
down to the curve, its points and the two windows.  The switch-on times are the
ones ``make_flash_slides.fig_two_readouts_op`` draws.

    stack    spectrum on top, the two bars below it, one shared time axis
    overlay  spectrum is the plot; the two bars ride in a strip above the curve
    lines    spectrum is the plot; each DAQ's switch-on is a vertical line and
             the blind stretches are shaded (darker = more of the chain blind)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plotstyle as P                       # noqa: E402
import make_x17_rate as R                   # noqa: E402
import make_flash_slides as F               # noqa: E402

XLIM = (5e-5, 4e1)                          # ms since the flash
GREEN, BLUE = P.DET_COLOR['C'], P.DET_COLOR['A']
YLAB = 'X17 pairs per day\n(nominal ³He cell)'


def switch_on(mm_us=None):
    """``mm_us`` overrides the measured 2 µs (the '~1 µs' variants)."""
    measured_ms = F.mm_recovery_ns() / 1e6
    mm_ms = mm_us / 1e3 if mm_us else measured_ms
    dream_ms = F.dream_recovery_ms('A', F.OP_RESIST)
    if not np.isfinite(dream_ms):               # run_57 cache is on the Linux box
        dream_ms = 2435 * measured_ms           # from the MEASURED value, not the override
    return mm_ms, dream_ms


def spectrum(ax):
    """Slide 31's curve in ms, pared to curve + points."""
    elo, ehi, y = R.load()
    t_lo, t_hi = R.t_of_E(ehi) * 1e3, R.t_of_E(elo) * 1e3
    t_mid = 0.5 * (t_lo + t_hi)
    o = np.argsort(t_mid)
    t_mid, y, t_lo, t_hi = t_mid[o], y[o], t_lo[o], t_hi[o]
    cs = PchipInterpolator(np.log(t_mid), np.log(y))
    ts = np.logspace(np.log10(t_mid[0]), np.log10(t_mid[-1]), 600)
    ax.plot(ts, np.exp(cs(np.log(ts))), color=P.ACCENT, lw=2.4, zorder=5)
    lit = ((t_lo >= R.t_of_E(R.MEV_HI_EV) * 1e3 / 1.01)
           & (t_hi <= R.t_of_E(R.MEV_LO_EV) * 1e3 * 1.01))
    ax.plot(t_mid[~lit], y[~lit], 'o', ms=5, color=P.MUTED, zorder=6,
            markeredgecolor=P.SURFACE, markeredgewidth=0.8)
    ax.plot(t_mid[lit], y[lit], 'o', ms=8, color=P.ACCENT, zorder=6,
            markeredgecolor=P.SURFACE, markeredgewidth=0.8)


def windows(axes, labels_on=None):
    """MeV (grey) and thermal (purple) bands across every axes given."""
    mev = (F.MEV_LO_MS, F.MEV_HI_MS)
    th = tuple(np.array([R.t_of_E(R.TH_HI_EV), R.t_of_E(R.TH_LO_EV)]) * 1e3)
    for ax in axes:
        ax.axvspan(*mev, color=P.INK, alpha=0.10, lw=0, zorder=0)
        ax.axvspan(*th, color=P.BAND_SIGNAL, alpha=0.10, lw=0, zorder=0)
    if labels_on is not None:
        ax, ytop = labels_on
        ax.text(np.sqrt(mev[0] * mev[1]), ytop, 'MeV', color=P.INK,
                fontsize=12, fontweight='bold', ha='center', va='top')
        ax.text(np.sqrt(th[0] * th[1]), ytop, 'thermal', color=P.BAND_SIGNAL,
                fontsize=12, fontweight='bold', ha='center', va='top')


def numbers_on_curve(ax, y_mev, y_th):
    n = R.numbers()
    ax.text(np.sqrt(F.MEV_LO_MS * F.MEV_HI_MS), y_mev,
            f"{n['mev']:.0f} / day", ha='center', va='center', fontsize=13,
            fontweight='bold', color=P.ACCENT, zorder=7)
    ax.text(np.sqrt(np.prod(n['thermal_t'])), y_th, f"{n['thermal']:.1f} / day",
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=P.ACCENT, zorder=7)


def energy_axis(ax):
    top = ax.twiny()
    top.set_xscale('log')
    top.set_xlim(*ax.get_xlim())
    e = np.array([1e-2, 1e0, 1e2, 1e4, 1e6, 1e8])
    t = R.t_of_E(e) * 1e3
    k = (t > XLIM[0]) & (t < XLIM[1])
    top.set_xticks(t[k])
    top.set_xticklabels([R._ev(v) for v in e[k]])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xlabel('neutron energy', labelpad=6)
    for s in ('right', 'left', 'bottom'):
        top.spines[s].set_visible(False)


def time_axis(ax, label=True):
    ax.set_xscale('log')
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel('time since the γ flash  [ms, log scale]')


def bar(ax, y, H, name, col, t_end, fs=13):
    """One DAQ's blind-then-alive bar (same look as status_two_readouts_op)."""
    ax.fill_between([XLIM[0], t_end], y - H / 2, y + H / 2, color=P.BAND_DEAD,
                    alpha=0.22, lw=0, zorder=3)
    ax.fill_between([t_end, XLIM[1]], y - H / 2, y + H / 2, color=col,
                    alpha=0.92, lw=0, zorder=3)
    ax.plot([t_end, t_end], [y - H / 2 - 0.05 * H, y + H / 2 + 0.05 * H],
            color=P.INK, lw=2.2, zorder=4, solid_capstyle='butt')


def tag_digitiser(ax, y, mm_ms, fs):
    ax.text(np.sqrt(XLIM[0] * mm_ms), y, 'blind', fontsize=fs - 1,
            color=P.BAND_DEAD, fontweight='bold', ha='center', va='center',
            zorder=5)
    ax.text(mm_ms * 1.6, y, f'detector alive from {mm_ms * 1e3:.0f} µs',
            fontsize=fs, color='white', fontweight='bold', ha='left',
            va='center', zorder=5)


def tag_dream(ax, y, dream_ms, fs):
    ax.text(dream_ms / 1.35, y, f'DREAM read-out: alive from {dream_ms:.0f} ms →',
            fontsize=fs, color=BLUE, fontweight='bold', ha='right',
            va='center', zorder=5)


# --------------------------------------------------------------------------- #

def fig_stack(mm_us=None, tag=''):
    mm_ms, dream_ms = switch_on(mm_us)
    fig = plt.figure(figsize=(8.4, 5.58))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.06,
                          left=0.095, right=0.975, top=0.875, bottom=0.115)
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1], sharex=a0)
    for a in (a0, a1):
        time_axis(a, label=False)
    a0.set_ylim(0, 21)
    spectrum(a0)
    windows((a0, a1), labels_on=(a0, 20.6))
    numbers_on_curve(a0, 8.5, 5.5)
    a0.set_ylabel(YLAB)
    a0.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    P.strip(a0)

    a1.set_ylim(0, 4.0)
    bar(a1, 2.95, 1.4, 'n_TOF digitiser', GREEN, mm_ms)
    bar(a1, 1.05, 1.4, 'DREAM read-out', BLUE, dream_ms)
    tag_digitiser(a1, 2.95, mm_ms, 12)
    tag_dream(a1, 1.05, dream_ms, 12)
    a1.text(np.sqrt(XLIM[0] * dream_ms) / 6, 1.05, 'blind', fontsize=11,
            color=P.BAND_DEAD, fontweight='bold', ha='center', va='center',
            zorder=5)
    a1.set_yticks([])
    a1.grid(axis='y', visible=False)
    a1.set_xlabel('time since the γ flash  [ms, log scale]')
    P.strip(a1, left=False)
    energy_axis(a0)
    F.save(fig, 'status_flash_combo_stack' + tag)


def fig_overlay(mm_us=None, tag=''):
    mm_ms, dream_ms = switch_on(mm_us)
    fig, ax = plt.subplots(figsize=(8.4, 5.58))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.875, bottom=0.12)
    time_axis(ax)
    ax.set_ylim(0, 28)
    ax.set_yticks([0, 5, 10, 15, 20])
    windows((ax,))
    for x, txt, col in ((np.sqrt(F.MEV_LO_MS * F.MEV_HI_MS), 'MeV', P.INK),
                        (np.sqrt(np.prod(np.array([R.t_of_E(R.TH_HI_EV), R.t_of_E(R.TH_LO_EV)]) * 1e3)), 'thermal', P.BAND_SIGNAL)):
        ax.text(x, 0.5 if txt == 'MeV' else 9.0, txt, color=col, fontsize=12,
                fontweight='bold', ha='center', va='bottom', zorder=7)
    spectrum(ax)
    numbers_on_curve(ax, 8.5, 5.5)
    bar(ax, 26.0, 2.2, 'n_TOF digitiser', GREEN, mm_ms)
    bar(ax, 23.0, 2.2, 'DREAM read-out', BLUE, dream_ms)
    tag_digitiser(ax, 26.0, mm_ms, 11.5)
    tag_dream(ax, 23.0, dream_ms, 11.5)
    ax.set_ylabel(YLAB)
    ax.yaxis.set_label_coords(-0.075, 0.36)
    P.strip(ax)
    energy_axis(ax)
    F.save(fig, 'status_flash_combo_overlay' + tag)


def fig_lines(mm_us=None, tag=''):
    mm_ms, dream_ms = switch_on(mm_us)
    fig, ax = plt.subplots(figsize=(8.4, 5.58))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.875, bottom=0.12)
    time_axis(ax)
    ax.set_ylim(0, 21)
    # darker = more of the chain blind: left of the digitiser line both are
    ax.axvspan(XLIM[0], dream_ms, color=P.BAND_DEAD, alpha=0.13, lw=0, zorder=0)
    ax.axvspan(XLIM[0], mm_ms, color=P.BAND_DEAD, alpha=0.13, lw=0, zorder=0)
    windows((ax,))
    spectrum(ax)
    for x, col, name in ((mm_ms, GREEN, 'detector\nalive from %.0f µs' % (mm_ms * 1e3)),
                         (dream_ms, BLUE, 'DREAM\nalive from %.0f ms' % dream_ms)):
        ax.axvline(x, color=col, lw=2.6, zorder=4)
        ax.text(x * 1.18, 20.4, name, color=col, fontsize=12, fontweight='bold',
                ha='left', va='top', zorder=7, linespacing=1.1,
                bbox=dict(fc=P.SURFACE, ec='none', alpha=0.7, pad=1.5))
    n = R.numbers()
    ax.text(4.5e-4, 12.5, f"{n['mev']:.0f} / day\n0.1–10 MeV",
            ha='right', va='center', fontsize=12.5, fontweight='bold',
            color=P.ACCENT, zorder=7, linespacing=1.3)
    ax.text(dream_ms * 1.15, 7.0, f"{n['thermal']:.1f} / day\nthermal",
            ha='left', va='bottom', fontsize=12, fontweight='bold',
            color=P.ACCENT, zorder=7, linespacing=1.3)
    ax.text(XLIM[0] * 1.35, 0.45, 'both blind', fontsize=11, color=P.BAND_DEAD,
            fontweight='bold', va='bottom', ha='left')
    ax.text(np.sqrt(mm_ms * dream_ms) * 3.0, 0.45, 'DREAM still blind', fontsize=11,
            color=P.BAND_DEAD, fontweight='bold', va='bottom', ha='center')
    ax.set_ylabel(YLAB)
    P.strip(ax)
    energy_axis(ax)
    F.save(fig, 'status_flash_combo_lines' + tag)


FIGURES = dict(stack=fig_stack, overlay=fig_overlay, lines=fig_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    P.use()
    for n in (args.only.split(',') if args.only else FIGURES):
        print(n)
        FIGURES[n]()
        print(n, '(1 µs)')
        FIGURES[n](mm_us=1.0, tag='_1us')


if __name__ == '__main__':
    main()
