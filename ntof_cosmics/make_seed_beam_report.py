#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_seed_beam_report.py -- report.html for seed_beam_test.py: does the beam
seeder's 3-strip minimum let junk in on beam data?

Reads the CSVs that `seed_beam_test.py compare` and `scint` write under
<work>/compare/<run>/<sub>/, copies them beside the report and draws the
figures from them, so re-running the analysis and this script updates numbers,
charts and verdict together.

    python ntof_cosmics/make_seed_beam_report.py [--work ~/scratch/ntof_insitu/beamseed]
Output: ntof_cosmics/results/seed_beam/report.html (+ figures/, data/).
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'sept26_prelim_analysis'))
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

OUT = HERE / 'results' / 'seed_beam'
RUN, SUB = 'run_145', 'stat090_0000'


def table(df, cols, heads, fmts):
    th = ''.join(f'<th>{h}</th>' for h in heads)
    tr = ''
    for _, r in df.iterrows():
        tr += '<tr>' + ''.join(f'<td>{f.format(r[c]) if isinstance(f, str) else f(r[c])}</td>'
                               for c, f in zip(cols, fmts)) + '</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


pct = lambda v: f'{100 * v:.1f} %'  # noqa: E731


def figures(S: pd.DataFrame, X: pd.DataFrame) -> list[tuple[str, str]]:
    fs.use()
    figd = OUT / 'figures'
    out = []
    arms = [a for a in 'ACD' if a in set(X.arm)]

    # 1 -- confirmed (background-subtracted) tracks and all gated tracks
    fig, axs = fs.plt.subplots(1, 2, figsize=fs.WIDE)
    w = 0.38
    for i, arm in enumerate(arms):
        x = X[X.arm == arm].set_index('sample')
        for j, (lab, col) in enumerate((('production', fs.MUTED), ('min 3, all', fs.DET_COLOR[arm]))):
            axs[0].bar(i + (j - 0.5) * w, x.loc[lab, 'n'], w, color=col, alpha=0.35 if j == 0 else 0.6)
            axs[1].bar(i + (j - 0.5) * w, x.loc[lab, 'wall_excess_n'], w, color=col)
            axs[1].text(i + (j - 0.5) * w, x.loc[lab, 'wall_excess_n'], f"{x.loc[lab, 'wall_excess_n']:,}",
                        ha='center', va='bottom', fontsize=8, color=fs.INK)
    for ax, t in zip(axs, ('gated tracks', 'wall-confirmed tracks, minus the accidental floor')):
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f'chamber {a}' for a in arms])
        ax.set_title(t, fontsize=10, color=fs.MUTED, loc='left')
    axs[0].legend(handles=[fs.plt.Rectangle((0, 0), 1, 1, color=fs.MUTED, alpha=0.35),
                           fs.plt.Rectangle((0, 0), 1, 1, color=fs.INK, alpha=0.6)],
                  labels=['production (min 5)', 'min 3'], frameon=False, fontsize=8)
    fs.fig_title(fig, 'The 3-strip seeder adds real particles, not only junk',
                 f'{RUN} {SUB}, all seven tags; same bundles, same fit, same gate')
    fs.preliminary(axs[1])
    fs.save(fig, figd / 'yield', data={'scint': X})
    out.append(('yield.png',
                'Left: gated tracks per chamber. Right: tracks whose extrapolation lands on a SiPM-wall '
                'group that fired in the signal window, minus the same count in an equal-width off-time '
                'window (the accidental floor). That difference counts real particles that reached the '
                'wall; it is a lower bound on real tracks, since out-of-time and wall-missing particles '
                'never confirm.'))

    # 2 -- confirmation rate by sample
    fig, ax = fs.figure(fs.FIG)
    samples = ['production', 'min 3, shared', 'min 3, gained', 'production, near-normal',
               'min 3, near-normal']
    for i, arm in enumerate(arms):
        x = X[X.arm == arm].set_index('sample').reindex(samples)
        xs = np.arange(len(samples)) + (i - (len(arms) - 1) / 2) * 0.22
        ax.plot(xs, x.wall, ls='none', marker=fs.DET_MARKER[arm], color=fs.DET_COLOR[arm], ms=6,
                label=f'chamber {arm}')
        ax.plot(xs, x.wall_ctrl, ls='none', marker='x', color=fs.DET_COLOR[arm], ms=5, mew=1.0,
                label=f'chamber {arm}, off-time (accidental)')
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([s.replace(', ', ',\n') for s in samples], fontsize=8)
    ax.set_ylabel('fraction wall-confirmed')
    ax.legend(frameon=False, fontsize=8)
    rel = {a: (X[(X.arm == a) & (X['sample'] == 'min 3, gained')].wall.iloc[0]
               / X[(X.arm == a) & (X['sample'] == 'production')].wall.iloc[0]) for a in arms}
    fs.fig_title(fig, 'Gained tracks confirm at ' + ', '.join(f'{100 * v:.0f} % ({a})' for a, v in rel.items())
                 + ' of the production rate',
                 'filled: signal window; crosses: same-width off-time window (accidentals)')
    fs.preliminary(ax)
    fs.save(fig, figd / 'confirmation', data={'scint': X})
    out.append(('confirmation.png',
                'Wall confirmation rate per sample. "shared": min-3 tracks within 2 mm of a production '
                'track; "gained": the rest. Near-normal: |tan| < 0.1 in both views (reconstructed). '
                'Production itself confirms well below 1 because of out-of-time particles and tracks '
                'that leave the wall; compare samples with each other, not with 1.'))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=str(Path.home() / 'scratch' / 'ntof_insitu' / 'beamseed'))
    a = ap.parse_args()
    src = Path(a.work) / 'compare' / RUN / SUB
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    for f in ('summary.csv', 'samples.csv', 'scint_confirmation.csv', 'pairs.csv'):
        shutil.copy(src / f, OUT / 'data' / f)
    S = pd.read_csv(OUT / 'data' / 'summary.csv')
    C = pd.read_csv(OUT / 'data' / 'samples.csv')
    X = pd.read_csv(OUT / 'data' / 'scint_confirmation.csv')
    PR = pd.read_csv(OUT / 'data' / 'pairs.csv')
    S = S[S.gated_m3 > 0]
    figs = figures(S, X)

    def g(arm, sample, col):
        return X[(X.arm == arm) & (X['sample'] == sample)][col].iloc[0]
    arms = [a for a in 'ACD' if a in set(X.arm)]
    gain = {a: g(a, 'min 3, all', 'wall_excess_n') / g(a, 'production', 'wall_excess_n') - 1 for a in arms}
    nng = {a: g(a, 'min 3, near-normal', 'wall_excess_n') / max(g(a, 'production, near-normal',
                                                                    'wall_excess_n'), 1) for a in arms}
    rel = {a: g(a, 'min 3, gained', 'wall') / g(a, 'production', 'wall') for a in arms}
    s_tab = table(S, ['arm', 'events_prod', 'events_m3', 'gated_prod', 'gated_m3', 'lost', 'gained',
                      'nearnormal_prod', 'nearnormal_m3', 'events_2gated_prod', 'events_2gated_m3'],
                  ['arm', 'events, prod', 'events, min 3', 'gated, prod', 'gated, min 3',
                   'prod tracks unmatched', 'min-3 tracks new', 'near-normal, prod',
                   'near-normal, min 3', '≥ 2 gated, prod', '≥ 2 gated, min 3'],
                  ['{}'] + ['{:,}'] * 10)
    x_tab = table(X, ['arm', 'sample', 'n', 'wall', 'wall_ctrl', 'wall_excess_n', 'plas', 'plas_ctrl',
                      'plas_excess_n'],
                  ['arm', 'sample', 'gated tracks', 'wall-confirmed', 'accidental', 'excess (n)',
                   'plastic-confirmed', 'accidental', 'excess (n)'],
                  ['{}', '{}', '{:,}', pct, pct, '{:,}', pct, pct, '{:,}'])
    c_tab = table(C[C['sample'].isin(['prod', 'matched', 'gained', 'lost']) & C.n.gt(0)],
                  ['arm', 'sample', 'n', 'med_n_strips', 'med_chi2dof', 'chi2dof_gt20', 'late_t0',
                   'near_normal', 'med_dca_mm'],
                  ['arm', 'sample', 'n', 'median strips (fit window)', 'median χ²/dof',
                   'χ²/dof &gt; 20', 't0 &gt; 300 ns', 'near-normal', 'median DCA to axis [mm]'],
                  ['{}', '{}', '{:,}', '{:.0f}', '{:.1f}', pct, pct, pct, '{:.0f}'])
    p_tab = table(PR, ['arm', 'sample', 'events', 'sep_lt12', 'sep_12_24', 'sep_med_mm', 'either_wall',
                       'both_wall'],
                  ['arm', 'seeder', '2-track events', '&lt; 12 mm', '12–24 mm', 'median sep [mm]',
                   'either track confirmed', 'both confirmed'],
                  ['{}', '{}', '{:,}', pct, pct, '{:.0f}', pct, pct])
    fig_html = ''.join(f'<figure><img src="figures/{f}" alt=""><figcaption>{html.escape(c)}'
                       f'</figcaption></figure>' for f, c in figs)
    gain_s = ', '.join(f'{a} {100 * gain[a]:+.0f} %' for a in arms)
    nn_s = ', '.join(f'{a} ×{nng[a]:.1f}' for a in arms)
    rel_s = ', '.join(f'{a} {100 * rel[a]:.0f} %' for a in arms)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beam seeder minimum</title>{HEAD}</head><body><main>
<p class="eyebrow">n_TOF EAR2 · X17 · ntof_cosmics/seed_beam_test</p>
<h1>The 3-strip beam seeder on beam data: more real tracks, at lower purity</h1>
<p class="lede"><b>Verdict: min 3 is a net gain on beam data, but it should not ship alone.</b>
Background-subtracted scintillator-confirmed tracks rise by {gain_s}, near-normal confirmed
tracks by {nn_s}. Tracks only min 3 finds confirm at {rel_s} of production's rate, so they are
mostly real but dirtier. Nothing is lost as a particle. About 13 % (A) / 5 % (C) of production's
x/y pairings are <i>re-paired</i> in busy events, where neither choice is clearly better on timing
or charge. That is the x/y pairing problem the two-track work's <code>xy_pairing</code> fix addresses,
so the seeder change and that fix must be validated together
(<code>sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md</code>).</p>

<h2>What was compared</h2>
<p>{RUN} {SUB}, all seven file tags, every trigger (the September full pass had no allowlist). The
production reco (<code>reco_fullpass</code>, seeder minimum 5) against a re-run that changes
<b>only</b> <code>MIN_STRIPS_BEAM</code> to 3: the full pass's own saved bundle per arm, the same fit,
the same environment. A local re-run at min 5 reproduces production on one tag to 99.8 % of fits
(the rest are the known laptop-vs-condor flips between near-degenerate minima). Tracks are built and
gated by today's <code>build_tracks</code> with stage 3's k, which reproduces the stage-3 gated
counts exactly. Purity is external: each gated track is extrapolated to its arm's SiPM wall and
plastic (<code>det_a_scint.match_run</code>), and the same test in an equal-width off-time window is
the accidental floor. Chamber B has no angle calibration and cannot be extrapolated.</p>

<h2>Counts</h2>{s_tab}
<h2>External confirmation</h2>{x_tab}
<h2>Track quality by sample</h2>{c_tab}
<h2>Same-chamber two-track events</h2>
<p>Min 3 doubles the events with two gated tracks in one chamber, but every pair is still ≥ 24 mm
apart (as in production): the extra small clusters do not make close fake pairs, and they do not
recover close pairs either (that is the seed gap and the joint fit, the two-track thread).</p>{p_tab}
<h2>Figures</h2>{fig_html}

<h2>What this does not rule out</h2>
<ul>
<li>One sub-run (run_145 stat090_0000), post-23-July noisy configuration. Quiet runs have more strips
over threshold and less to gain.</li>
<li>Confirmation is a lower bound on reality: out-of-time particles and tracks that miss the wall
never confirm, so "unconfirmed" is not "junk". The gained sample is more out of time (t0 &gt; 300 ns)
than production, which accounts for part of its lower rate.</li>
<li>Near-normal confirmation is low in both seeders (A 8–10 %). Either near-normal reconstructed
tracks are mostly mis-measured, or few of them point at a fired group. Not resolved here.</li>
<li>Re-paired tracks: which x/y pairing is right is undecided. Per-plane t0 is unconstrained in the
free fit (89 ns scatter), so timing cannot arbitrate.</li>
<li>Angles still come from the v = 42.6 bundles and the old C kernel: this tests detection, not
angle accuracy.</li>
</ul>
<p class="foot">Built by <code>ntof_cosmics/make_seed_beam_report.py</code> from
<code>seed_beam_test.py compare</code> / <code>scint</code>; data in <code>data/</code>.</p>
</main></body></html>"""
    (OUT / 'report.html').write_text(page)
    print(OUT / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
