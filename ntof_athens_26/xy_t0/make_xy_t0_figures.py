#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_xy_t0_figures.py -- the four figures for the x/y t0 investigation.

Draws only; every number comes from ``xy_t0.py``'s tables or is recomputed from
the same track table with the same helpers, so nothing here can disagree with
the report.  Figures are PNG+CSV pairs (``figstyle.save``), referenced from
``report.html`` with relative links.

    X17_ROOT=D:/x17 python ntof_athens_26/xy_t0/make_xy_t0_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / 'sept26_prelim_analysis'))
import figstyle as fs                                             # noqa: E402
from sept26_prelim_analysis import paths                          # noqa: E402
import xy_t0 as X                                                 # noqa: E402

ARMS3 = ('A', 'C', 'D')


# --------------------------------------------------------------------------- #
def fig_agreement(d, insitu, out):
    """THE FIGURE. x_t0 - y_t0 per ftst class: a broad peak that translates.

    Two things at once, which is the whole result.  Across a row the shape
    slides left as ``ftst_diff`` rises -- that is ``dt_xy``, and it is linear.
    Within a panel it is ~200 ns wide, not a coincidence spike -- that
    is the t0 resolution, and no offset fixes it.  The scrambled pairing is
    drawn under each so the accidental pedestal is visible rather than asserted.
    """
    un = d[(d.n_cand_x == 1) & (d.n_cand_y == 1) & np.isfinite(d.raw)]
    rng = np.random.default_rng(3)
    rows = []
    fig, axes = plt.subplots(3, 5, figsize=(13.0, 6.6), sharex=True, sharey='row')
    for i, arm in enumerate(ARMS3):
        g = un[un.arm == arm]
        ks = sorted(g.ftst_diff.unique())[:5]
        for j, k in enumerate(ks):
            ax = axes[i, j]
            gk = g[g.ftst_diff == k]
            ht = X._hist(gk.raw.values) * 100
            hb = X._scrambled_hist(gk.x_t0.values, gk.y_t0.values, rng) * 100
            tail = np.abs(X.HIST_CENTRES) > X.PEDESTAL_NS
            f = ht[tail].sum() / hb[tail].sum() if hb[tail].sum() else 0.0
            c = X.HIST_CENTRES
            ax.fill_between(c, 0, hb * f, color='#b9c0c9', alpha=.55, lw=0,
                            label='scrambled pairing' if (i == 0 and j == 0) else None)
            ax.plot(c, ht, color=fs.DET_COLOR[arm], lw=1.25,
                    label='true pairing' if (i == 0 and j == 0) else None)
            sh = insitu[(insitu.arm == arm) & (insitu.ftst_diff == k)]
            if len(sh):
                ax.axvline(float(sh.dt_insitu.iloc[0]), color=fs.DET_COLOR[arm],
                           ls='--', lw=.9, alpha=.85)
            ax.axvline(X.FALLBACK_DT, color='#555', ls=':', lw=.9, alpha=.9)
            ax.set_xlim(-300, 300)
            ax.tick_params(labelsize=8)
            ax.set_title(f'{arm}   ftst diff {k:+d}', fontsize=9.5, pad=3)
            rows.append(pd.DataFrame({'arm': arm, 'ftst_diff': k, 'centre_ns': c,
                                      'pct_true': ht, 'pct_scrambled': hb * f}))
        axes[i, 0].set_ylabel('% of tracks / 5 ns', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('$t_{0}^{x}-t_{0}^{y}$  [ns]', fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color='#555', ls=':', lw=1),
          plt.Line2D([], [], color='#333', ls='--', lw=1)]
    l += ['$-18.8$ ns fallback (what ran)', 'in-situ $dt_{xy}$']
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.015))
    fs.fig_title(fig, 'The x/y time difference is 200 ns wide, and it slides with the readout phase',
                 'unambiguous tracks, no gate  ·  two planes seeing one track at once would give a spike, not this')
    fig.tight_layout(rect=(0, .045, 1, .93))
    fs.save(fig, out / 'xy_agreement', data=pd.concat(rows, ignore_index=True))
    plt.close(fig)


def fig_dt_law(insitu, law, meta_dt, out):
    """dt_xy is linear in ftst_diff, on every arm, at ~the 10 ns phase quantum.

    The bench points are drawn where they sit, which is the other half of the
    story: two classes per arm, and on A, B and C neither of them ever occurs.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3),
                                  gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = []
    for arm in ('A', 'B', 'C', 'D'):
        g = insitu[insitu.arm == arm].sort_values('ftst_diff')
        st = fs.det_style(arm)
        ax.plot(g.ftst_diff, g.shift_ns, lw=1.5, ms=5.5, **st)
        lw = law[law.arm == arm].iloc[0]
        xs = np.linspace(g.ftst_diff.min() - .4, g.ftst_diff.max() + .4, 20)
        ax.plot(xs, lw.slope_ns_per_unit * xs + lw.intercept_ns,
                color=fs.DET_COLOR[arm], lw=.8, alpha=.45, zorder=1)
        rows.append(g.assign(slope=lw.slope_ns_per_unit))
    xs = np.array([-5, 5])
    ax.plot(xs, -X.FTST_QUANTUM_NS * xs, color='#333', ls='--', lw=1.2,
            label=f'$-{X.FTST_QUANTUM_NS:.0f}$ ns/unit (one phase of a 60 ns sample)')
    ax.axhline(0, color='#aaa', lw=.7)
    ax.set_xlabel('$ftst_x - ftst_y$')
    ax.set_ylabel('shift of the distribution  [ns]')
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fs.title(ax, 'One readout-clock effect, not four chamber constants',
             'measured in situ, referenced to each arm\'s central class')

    for arm in ('A', 'B', 'C', 'D'):
        st = fs.det_style(arm)
        seen = sorted(insitu[insitu.arm == arm].ftst_diff.unique())
        st2 = dict(st); st2.pop('label'); ax2.plot(seen, [arm] * len(seen), ls='none', ms=7, **st2)
        keys = sorted(int(k) for k in meta_dt.get(arm, {}))
        ax2.plot(keys, [arm] * len(keys), ls='none', marker='x', ms=9, mew=2,
                 color='#c0392b')
    ax2.set_xlim(-6, 6)
    ax2.set_xlabel('$ftst_x - ftst_y$')
    ax2.set_xticks(range(-5, 6))
    ax2.grid(axis='x', alpha=.25)
    ax2.plot([], [], ls='none', marker='x', ms=8, mew=2, color='#c0392b',
             label='bench bundle $dt_{xy}$ key')
    ax2.plot([], [], ls='none', marker='o', ms=6, color='#666',
             label='class seen in beam data')
    ax2.legend(frameon=False, fontsize=9, loc='lower center',
               bbox_to_anchor=(.5, -.42), ncol=2)
    fs.title(ax2, 'The bench keys and the beam classes never meet',
             'A, B, C: keys odd, data even  ·  D overlaps at $\\pm3$ only')
    fig.tight_layout(rect=(0, .06, 1, 1))
    fs.save(fig, out / 'dt_xy_law', data=pd.concat(rows, ignore_index=True))
    plt.close(fig)


def fig_mechanism(deg, geo, out):
    """The two mechanism tests, side by side, each against its own control."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    w = .26
    xs = np.arange(len(ARMS3))
    for i, (col, lab, col_c) in enumerate([
            ('null_mean', 'arbitrary periods (no structure)', '#b9c0c9'),
            ('amp_60ns', '60 ns — the depth bin', '#c0392b'),
            ('amp_5ns_control', '5 ns — the known $T_0$ snap', '#2c7fb8')]):
        v = [float(deg.loc[deg.arm == a, col].iloc[0]) for a in ARMS3]
        ax.bar(xs + (i - 1) * w, v, w, color=col_c, label=lab)
    for j, a in enumerate(ARMS3):
        r = deg[deg.arm == a].iloc[0]
        ax.errorbar(xs[j], r.null_mean, yerr=r.null_sd, color='#555', capsize=3, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARMS3)
    ax.set_ylabel('Rayleigh $|R|$ of the residual')
    ax.legend(frameon=False, fontsize=9)
    fs.title(ax, 'No comb at the depth bin',
             'the 5 ns control proves the test can see one')

    rows = []
    for a in ARMS3:
        g = geo[geo.arm == a].sort_values('tan_med')
        st = fs.det_style(a)
        ax2.plot(g.tan_med, g.halfwidth, lw=1.5, ms=5.5, **st)
        rows.append(g)
    ax2.set_ylim(0, None)
    ax2.set_xlabel(r'track inclination  $\tan\theta$')
    ax2.set_ylabel('residual half-width  [ns]')
    ax2.legend(frameon=False, fontsize=9)
    fs.title(ax2, 'Geometry is real and is ~10 ns of a 66–77 ns spread',
             'a plane separation crossed by an inclined track must vanish at $\\tan\\theta\\to0$; '
             'this does not')
    fig.tight_layout()
    fs.save(fig, out / 'mechanism', data=pd.concat(rows, ignore_index=True))
    plt.close(fig)


def fig_circularity(circ, disc, out):
    """The gate is the metric, and what the gate is actually worth."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    arms = list(circ.arm)
    xs = np.arange(len(arms))
    for i, (col, lab, c) in enumerate([
            ('hw_gated', 'gated (what Sec. 1 measured)', '#2c7fb8'),
            ('hw_all', 'all stage-3 tracks', '#8a3f8f'),
            ('hw_ungated', 'ungated', '#d18a44')]):
        ax.bar(xs + (i - 1) * .26, circ[col], .26, color=c, label=lab)
    ax.axhline(X.TOL_NS, color='#c0392b', ls='--', lw=1.1)
    ax.text(-.42, X.TOL_NS + 8, f'the $\\pm${X.TOL_NS:.0f} ns cut',
            color='#c0392b', fontsize=9, ha='left')
    ax.set_xticks(xs)
    ax.set_xticklabels(arms)
    ax.set_ylabel('residual half-width  [ns]')
    ax.legend(frameon=False, fontsize=9)
    fs.title(ax, 'The published sample is this distribution, truncated',
             'the gate IS $|(t_0^x-t_0^y)-dt|\\leq 120$ ns — 0 of 2.11 M gated tracks sit outside')

    w = .34
    ax2.bar(xs - w / 2, disc.acc_true, w, color='#2c7fb8', label='true pairing')
    ax2.bar(xs + w / 2, disc.acc_scrambled, w, color='#b9c0c9', label='scrambled pairing')
    for j, r in enumerate(disc.itertuples()):
        ax2.text(j, max(r.acc_true, r.acc_scrambled) + .03,
                 f'{r.enhancement:.1f}$\\times$', ha='center', fontsize=9.5,
                 color='#1b2430')
    ax2.set_xticks(xs)
    ax2.set_xticklabels(list(disc.arm))
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('fraction inside the window')
    ax2.legend(frameon=False, fontsize=9, loc='upper right')
    fs.title(ax2, 'The coincidence test buys a factor of 2',
             'for 30 % of the real tracks — not the discriminator its docstring describes')
    fig.tight_layout()
    fs.save(fig, out / 'gate', data=circ.merge(disc, on='arm', suffixes=('', '_disc')))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None, help='xy_t0.py output directory')
    ap.add_argument('--tracks', default=None)
    ap.add_argument('--bundles', default=None)
    ap.add_argument('--out', default=None, help='figure directory')
    a = ap.parse_args()

    src = Path(a.src) if a.src else paths.spell('out', 'xy_t0')
    paths.require(src / 'insitu_dt.csv', 'xy_t0.py tables (run xy_t0.py first)')
    out = Path(a.out) if a.out else HERE / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fs.use()

    tracks = Path(a.tracks) if a.tracks else paths.spell(
        'out', 'stage3_campaign', 'tracks_campaign.parquet')
    broot = Path(a.bundles) if a.bundles else paths.spell(
        'x17', 'sept26_fullpass', 'bundles')
    dt_xy = X.bundle_dt_xy(broot)
    d = X.load_tracks(tracks, dt_xy)

    insitu = pd.read_csv(src / 'insitu_dt.csv')
    law = pd.read_csv(src / 'insitu_law.csv')
    deg = pd.read_csv(src / 'degeneracy_test.csv')
    geo = pd.read_csv(src / 'geometry_test.csv')
    circ = pd.read_csv(src / 'circularity.csv')
    disc = pd.read_csv(src / 'discrimination.csv')

    fig_agreement(d, insitu, out)
    fig_dt_law(insitu, law, dt_xy, out)
    fig_mechanism(deg, geo, out)
    fig_circularity(circ, disc, out)
    print(f'[xy_t0] figures -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
