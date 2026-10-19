#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_couplings.py -- the two X17-theory teaching figures, for the backup slides.

    ../.venv/bin/python make_couplings.py

Writes, per figure:

  figures/x17_epsilon_light.png/.pdf            titled -- standalone/report copy
  figures/x17_couplings_light.png/.pdf
  slides/assets/img/x17_epsilon.png             bare -- what the slides use
  slides/assets/img/x17_couplings.png

Two figures because they answer two different questions (added 2026-08-12,
Dylan: "I need to have epsilon explained clearly in general" + "visualize/
explain these constraints"):

  * ``epsilon``   -- WHAT the coupling epsilon is: the X-charge of each fermion
                     in units of e, one vertex diagram against the photon's, and
                     where eps_n / eps_e enter our own signal chain.  Drawn in
                     the scenes_x17 visual language (same nuclei, same
                     squiggles, same lepton fork) so it reads as a sibling of
                     the signature figure two slides earlier.
  * ``couplings`` -- WHERE each coupling is allowed: one shared log axis,
                     three lanes (eps_n / eps_p / eps_e), exclusions as hatched
                     grey (status encoding, not categorical -- the hatch is the
                     secondary encoding, so the statement never rests on hue),
                     the fit/surviving windows as the one x17-purple accent.
                     Reading DOWN the lanes is the whole argument: the n-lane
                     band sits above the p-lane ceiling (protophobia), and the
                     e-lane keeps a thin gap between the beam dumps and g-2.
                     Labels over hatch always sit on a white bbox -- the hatch
                     is unreadable underneath text otherwise (v1 tried).

EVERY NUMBER here is the verified set from the theory backup slides
(index.html, comment block "Backup -- X17 theory", 2026-08-12): NA48/2
|eps_p| < 1.2e-3; 8Be fit |eps_n| = (2-10)e-3; n-Pb ceiling 2.5e-2; E141 floor
2e-4; NA64 exclusion 1.2e-4..6.8e-4 (2020; the 2018 edge was 4.2e-4); ceiling
1.4e-3 from electron g-2 / KLOE-2.  The eps_p / eps_n numbers are Feng et
al.'s translation (arXiv:1604.07411, 1608.03591); NA64 from arXiv:1803.07748
and 1912.11389.  If a number changes on the slide, change it HERE too -- the
slide table is the record, this drawing repeats it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Circle, Rectangle   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenes_x17 as X                             # noqa: E402
import plotstyle as PS                             # noqa: E402

FIG = os.path.join(HERE, 'figures')
SLIDE_IMG = os.path.join(HERE, 'slides', 'assets', 'img')

FONT = X.FONT

# ---- the verified numbers (see module docstring for provenance) ----------- #
EPS_P_MAX = 1.2e-3          # NA48/2 pi0 -> gamma X, Feng et al. translation
EPS_N_LO, EPS_N_HI = 2e-3, 1e-2   # what the 8Be rate needs (mass-dependent)
EPS_N_MAX = 2.5e-2          # n-Pb scattering ceiling
E141_EDGE = 2e-4            # SLAC E141 beam-dump upper edge (the old floor)
NA64_2018 = 4.2e-4          # NA64 PRL 120, 231802
NA64_2020 = 6.8e-4          # NA64 PRD 101, 071101(R) -- the current floor
EPS_E_MAX = 1.4e-3          # electron g-2 + KLOE-2 ceiling

XLO, XHI = 8e-5, 6e-2       # shared axis range


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, f'{name}_light.{ext}')
        fig.savefig(p)
        print(f'  -> {p}')


def _save_slide(fig, name):
    os.makedirs(SLIDE_IMG, exist_ok=True)
    p = os.path.join(SLIDE_IMG, f'{name}.png')
    fig.savefig(p)
    print(f'  -> {p}')


# =========================================================================== #
# Figure 1 -- what epsilon is
# =========================================================================== #
W = 160.0


def draw_epsilon(title=True):
    """One fixed content layout; the titled copy just extends the canvas up.

    All y coordinates are absolute.  Content occupies 0..80; the title block,
    when drawn, occupies 80..98.  The figure height follows the ylim span so
    the isotropic aspect is preserved in both variants.
    """
    P = X.palette('light')
    y_max = 98.0 if title else 80.0
    fig, ax = plt.subplots(figsize=(12.8, y_max / W * 12.8), dpi=160)
    fig.patch.set_facecolor(P['page'])
    ax.set_xlim(0, W); ax.set_ylim(0, y_max)
    ax.set_aspect('equal'); ax.axis('off')

    if title:
        ax.text(0, 96.0, 'What ε is', fontsize=21, fontweight='bold',
                color=P['ink'], ha='left', va='top', **FONT)
        ax.text(0, 88.0, 'each fermion gets its own small "X-charge" '
                '$\\varepsilon_f$, measured in units of the electron charge e',
                fontsize=10.5, color=P['muted'], ha='left', va='top', **FONT)

    # ---- top row: two vertex cards, photon vs X ---------------------------- #
    y_cards = 78.0              # card top
    card_h, card_w = 30.0, 66.0
    for x0, which in ((6.0, 'gamma'), (88.0, 'x17')):
        ax.add_patch(Rectangle((x0, y_cards - card_h), card_w, card_h,
                               facecolor=P['card'], edgecolor=P['rule'],
                               lw=0.8, zorder=1))
        yv = y_cards - card_h * 0.55
        xf0, xv = x0 + 6.0, x0 + 30.0
        # the fermion line, through the vertex
        ax.plot([xf0, xv], [yv - 6.5, yv], color=P['ink'], lw=1.9,
                solid_capstyle='round', zorder=4)
        ax.plot([xv, xf0], [yv, yv + 6.5], color=P['ink'], lw=1.9,
                solid_capstyle='round', zorder=4)
        ax.annotate('', xy=(x0 + 19.0, yv + 3.35), xytext=(x0 + 16.0, yv + 4.2),
                    arrowprops=dict(arrowstyle='-|>', color=P['ink'], lw=0,
                                    mutation_scale=9), zorder=5)
        ax.text(xf0 - 1.6, yv - 7.0, r'$f$', fontsize=11, color=P['ink'],
                ha='right', va='center', **FONT)
        ax.text(xf0 - 1.6, yv + 7.0, r'$f$', fontsize=11, color=P['ink'],
                ha='right', va='center', **FONT)
        # the boson off the vertex
        col = P['gamma'] if which == 'gamma' else P['x17']
        X.squiggle(ax, xv, yv, xv + 22.0, yv, col, n_wave=5, amp=1.15, lw=2.0)
        ax.add_patch(Circle((xv, yv), 1.05, facecolor=col,
                            edgecolor='none', zorder=6))
        boson = r'$\gamma$' if which == 'gamma' else r'$X$'
        ax.text(xv + 11.0, yv + 4.4, boson, fontsize=13, color=col,
                ha='center', va='center', **FONT)
        if which == 'gamma':
            cap = ('the photon: couples to electric charge,\n'
                   'the same strength  e  for every unit of charge')
            vtx, vcol = r'$e\,Q_f$', P['ink']
        else:
            cap = ('the X: couples to its own charge  $\\varepsilon_f$,\n'
                   'a different small number for each fermion')
            vtx, vcol = r'$\varepsilon_f\, e$', P['x17']
        ax.text(x0 + card_w / 2, y_cards - card_h - 3.0, cap, fontsize=9.5,
                color=P['muted'], ha='center', va='top', **FONT)
        ax.text(xv, yv - 5.2, vtx, fontsize=12.5, color=vcol, ha='center',
                va='top', fontweight='bold', **FONT)

    ax.text(80.0, y_cards - card_h * 0.5, 'vs', fontsize=12,
            color=P['muted'], ha='center', va='center', **FONT)

    # ---- bottom row: where each epsilon enters our signal ------------------ #
    y_chain = 24.0
    ax.text(6.0, y_chain + 11.0, 'where each ε enters the signal',
            fontsize=11, fontweight='bold', color=P['ink'], ha='left',
            va='center', **FONT)

    xn = 14.0
    X.nucleus(ax, xn, y_chain, 2, 2, r=1.6, P=P)
    X.excitation_waves(ax, xn, y_chain, P, r=1.6)
    ax.text(xn, y_chain - 8.0, '$^{4}$He$^{*}$', fontsize=10.5,
            color=P['ink'], ha='center', va='center', **FONT)
    xv1 = xn + 16.0
    ax.plot([xn + 5.4, xv1], [y_chain, y_chain], color=P['muted'], lw=1.3,
            ls=(0, (1.5, 2.2)), zorder=3)
    ax.add_patch(Circle((xv1, y_chain), 1.05, facecolor=P['x17'],
                        edgecolor='none', zorder=6))
    X.squiggle(ax, xv1, y_chain, xv1 + 20.0, y_chain, P['x17'], n_wave=5,
               amp=1.15, lw=2.0)
    ax.text(xv1 + 10.0, y_chain + 4.4, r'$X$', fontsize=12, color=P['x17'],
            ha='center', va='center', **FONT)
    xv2 = xv1 + 20.0
    ax.add_patch(Circle((xv2, y_chain), 1.05, facecolor=P['x17'],
                        edgecolor='none', zorder=6))
    X.lepton_fork(ax, xv2, y_chain, 13.0, 26.0, P, fs=10)

    # the two vertex callouts, kept inside the canvas (v1 clipped them)
    ax.annotate('production\nrate $\\propto \\varepsilon_n^{\\,2}$'
                '  (and $\\varepsilon_p \\approx 0$)',
                xy=(xv1, y_chain - 1.8), xytext=(xv1 - 4.0, y_chain - 11.5),
                fontsize=9.5, color=P['ink'], ha='center', va='top',
                arrowprops=dict(arrowstyle='-', color=P['muted'], lw=0.9,
                                shrinkA=2, shrinkB=3), **FONT)
    ax.annotate('decay to $e^+e^-$\nlifetime $\\propto 1/\\varepsilon_e^{\\,2}$',
                xy=(xv2, y_chain - 1.8), xytext=(xv2 + 8.0, y_chain - 11.5),
                fontsize=9.5, color=P['ink'], ha='center', va='top',
                arrowprops=dict(arrowstyle='-', color=P['muted'], lw=0.9,
                                shrinkA=2, shrinkB=3), **FONT)

    # the scale statement, right-hand side of the chain row
    ax.text(92.0, y_chain - 1.0,
            'small charge, tiny force:  $\\varepsilon \\sim 10^{-3}$  means\n'
            'every rate carries  $\\varepsilon^2 \\sim 10^{-6}$  of the\n'
            'electromagnetic one — why a 17 MeV boson can hide at all.\n'
            'Each experiment pins down a different $\\varepsilon_f$;\n'
            'the next slide draws where each one is allowed.',
            fontsize=9.5, color=P['muted'], ha='left', va='center',
            linespacing=1.7, **FONT)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    return fig


# =========================================================================== #
# Figure 2 -- the coupling windows
# =========================================================================== #
LANE_Y = {'n': 3.1, 'p': 1.55, 'e': 0.0}   # generous gaps: annotations live
LANE_H = 0.62                              # between the lanes, not on them
LANE_LABEL = {'n': ('$\\varepsilon_n$', 'neutrons'),
              'p': ('$\\varepsilon_p$', 'protons'),
              'e': ('$\\varepsilon_e$', 'electrons')}


def _band(ax, y, x0, x1, kind, P):
    """One horizontal band: 'excl' = hatched grey status, 'fit' = the accent."""
    if kind == 'excl':
        ax.add_patch(Rectangle((x0, y - LANE_H / 2), x1 - x0, LANE_H,
                               facecolor='#eceff2', edgecolor=PS.MUTED,
                               hatch='///', lw=0.6, zorder=3))
    else:
        ax.add_patch(Rectangle((x0, y - LANE_H / 2), x1 - x0, LANE_H,
                               facecolor=P['x17'], edgecolor='none',
                               alpha=0.88, zorder=3))


def _band_label(ax, x, y, text, fs=10):
    """A label on top of hatching -- always on a white plate, or it drowns."""
    ax.text(x, y, text, fontsize=fs, color=PS.MUTED, ha='center', va='center',
            linespacing=1.3, zorder=5,
            bbox=dict(facecolor='#ffffff', edgecolor='none',
                      boxstyle='round,pad=0.32'))


def draw_couplings(title=True):
    PS.use()
    P = X.palette('light')
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    ax.set_xscale('log')
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-0.95, 3.85)
    ax.set_yticks([])
    ax.grid(False)
    ax.grid(True, axis='x', color=PS.LINE, lw=0.7, alpha=0.7)
    PS.strip(ax, left=False)
    ax.set_xlabel(r'coupling strength  $|\varepsilon|$   '
                  '(X-charge, in units of the electron charge)')

    for k, (sym, word) in LANE_LABEL.items():
        ax.text(-0.012, LANE_Y[k], f'{sym}\n{word}',
                transform=ax.get_yaxis_transform(), fontsize=13, color=PS.INK,
                ha='right', va='center', fontweight='bold', linespacing=1.3)

    ink, mut = PS.INK, PS.MUTED

    # ---- eps_n lane -------------------------------------------------------- #
    y = LANE_Y['n']
    _band(ax, y, EPS_N_LO, EPS_N_HI, 'fit', P)
    _band(ax, y, EPS_N_MAX, XHI, 'excl', P)
    ax.text(np.sqrt(EPS_N_LO * EPS_N_HI), y + LANE_H / 2 + 0.10,
            'what the ⁸Be rate needs', fontsize=12, color=P['x17'],
            fontweight='bold', ha='center', va='bottom')
    _band_label(ax, np.sqrt(EPS_N_MAX * XHI), y,
                'excluded:\nn–Pb scattering', fs=9.5)

    # ---- eps_p lane -------------------------------------------------------- #
    y = LANE_Y['p']
    _band(ax, y, EPS_P_MAX, XHI, 'excl', P)
    _band_label(ax, np.sqrt(EPS_P_MAX * XHI), y,
                'excluded:  NA48/2,  $\\pi^0 \\to \\gamma\\,(X \\to e^+e^-)$')
    # plain text INSIDE the allowed white region -- an arrow at the exclusion
    # edge read as pointing at the excluded side (v2 tried)
    ax.text(np.sqrt(XLO * EPS_P_MAX), y,
            'protons allowed only in here — "protophobic"',
            fontsize=10.5, color=ink, ha='center', va='center')

    # the protophobia gap: the n-band starts above the p-ceiling
    ax.annotate('', xy=(EPS_N_LO, LANE_Y['n'] - LANE_H / 2 - 0.05),
                xytext=(EPS_P_MAX, LANE_Y['p'] + LANE_H / 2 + 0.05),
                arrowprops=dict(arrowstyle='<|-|>', color=P['x17'], lw=1.4,
                                shrinkA=0, shrinkB=0))
    ax.text(1.05e-3, (LANE_Y['n'] + LANE_Y['p']) / 2 + 0.05,
            'the fit needs neutrons coupled\nwell above the proton ceiling —\n'
            'it must see neutrons, not protons',
            fontsize=10.5, color=P['x17'], ha='right', va='center',
            fontweight='bold', linespacing=1.35)

    # ---- eps_e lane -------------------------------------------------------- #
    y = LANE_Y['e']
    _band(ax, y, XLO, NA64_2020, 'excl', P)
    _band(ax, y, NA64_2020, EPS_E_MAX, 'fit', P)
    _band(ax, y, EPS_E_MAX, XHI, 'excl', P)
    for edge, lab, dy in ((E141_EDGE, 'E141', 0.0),
                          (NA64_2018, 'NA64 2018', 0.0),
                          (NA64_2020, 'NA64 2020', -0.16)):
        ax.plot([edge, edge], [y - LANE_H / 2, y + LANE_H / 2], color=mut,
                lw=1.1, ls=(0, (3, 2)) if edge != NA64_2020 else '-', zorder=4)
        ax.text(edge, y - LANE_H / 2 - 0.07 + dy, lab, fontsize=9, color=mut,
                ha='center', va='top')
    _band_label(ax, 1.55e-4, y,
                'excluded: beam dumps\n(too weakly coupled —\n'
                'X lives long, escapes)', fs=9.5)
    _band_label(ax, np.sqrt(EPS_E_MAX * XHI), y,
                'excluded:  electron $g-2$,  KLOE-2')
    # the squeeze from below, drawn above the lane where there is room
    ax.annotate('', xy=(NA64_2020 * 0.97, y + LANE_H / 2 + 0.13),
                xytext=(NA64_2018 * 0.82, y + LANE_H / 2 + 0.13),
                arrowprops=dict(arrowstyle='-|>', color=ink, lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(NA64_2018 * 0.76, y + LANE_H / 2 + 0.13, 'NA64 keeps pushing',
            fontsize=9.5, color=ink, ha='right', va='center')
    ax.text(np.sqrt(NA64_2020 * EPS_E_MAX), y + LANE_H / 2 + 0.10,
            'survives', fontsize=12, color=P['x17'], fontweight='bold',
            ha='center', va='bottom')

    if title:
        PS.title(ax, 'Where a 17 MeV vector is still allowed',
                 sub='hatched = excluded; purple = what the anomaly needs '
                     '(nucleons) and what survives (electrons)')
        PS.note(fig, 'Numbers: Feng et al., PRL 117, 071803 (2016) and PRD 95, '
                     '035017 (2017) [ε$_p$, ε$_n$, E141, g−2]; '
                     'NA64, PRL 120, 231802 (2018) and PRD 101, 071101(R) '
                     '(2020) [ε$_e$ floor]. One-dimensional projection of '
                     'the published 2-D limits at m$_X$ = 17 MeV.')

    fig.subplots_adjust(left=0.085, right=0.985,
                        top=0.90 if title else 0.985, bottom=0.13)
    return fig


def main():
    print('epsilon explainer:')
    fig = draw_epsilon(title=True)
    _save(fig, 'x17_epsilon')
    plt.close(fig)
    fig = draw_epsilon(title=False)
    _save_slide(fig, 'x17_epsilon')
    plt.close(fig)

    print('coupling windows:')
    fig = draw_couplings(title=True)
    _save(fig, 'x17_couplings')
    plt.close(fig)
    fig = draw_couplings(title=False)
    _save_slide(fig, 'x17_couplings')
    plt.close(fig)


if __name__ == '__main__':
    main()
