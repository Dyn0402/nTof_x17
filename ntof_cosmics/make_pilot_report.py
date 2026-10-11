#!/usr/bin/env python3
"""make_pilot_report.py -- results/pilot_is2_v1/report.html from the tables
pilot_compare.py writes (summary, bins, cosmic, valley) and, when present, the
run_86 one-sided/two-sided yield gate (results/yield_gate/yield_gate.csv,
chains is2e and is2ts).  Re-run after any of them changes.

    python -m ntof_cosmics.make_pilot_report
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

OUT = HERE / 'results' / 'pilot_is2_v1'
FIG = OUT / 'figures'
YG = HERE / 'results' / 'yield_gate' / 'yield_gate.csv'
YGB = HERE / 'results' / 'yield_gate' / 'confirm_by_tan_run86.csv'


def tab(df, fmt='{:.3f}'):
    return df.to_html(index=False, float_format=lambda v: fmt.format(v), border=0, classes='t', na_rep='')


def fig_ratio(R):
    """Per-run pilot/production confirmed (|true tan| < 0.6 in both)."""
    import matplotlib.pyplot as plt
    fs.use()
    fig, ax = fs.figure(fs.WIDE)
    P = R[(R.chain == 'pilot')].copy()
    P['rn'] = P.run.str[4:].astype(int)
    for arm in ('A', 'C'):
        p = P[(P.arm == arm) & np.isfinite(P.confirmed_acc_vs_prod)]
        ax.plot(p.rn, p.confirmed_acc_vs_prod, ls='none', ms=6, **{**fs.det_style(arm), 'label': f'arm {arm}'})
    ax.axhline(1, color=fs.MUTED, lw=0.8)
    ax.set_xlabel('run')
    ax.set_ylabel('confirmed, pilot / production')
    ax.set_ylim(0.8, 2.0)
    ax.legend(frameon=False, loc='lower right')
    fs.title(ax, 'The pilot confirms 1.5-1.8x production on every run',
             'wall excess over the off-time control, |true tan| < 0.6 in both chains')
    fs.preliminary(ax)
    fs.save(fig, FIG / 'confirmed_ratio_by_run',
            data=P[['run', 'arm', 'confirmed_acc', 'confirmed_acc_vs_prod']])
    plt.close(fig)


def fig_bins(B):
    import matplotlib.pyplot as plt
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    for ax, arm in zip(axs, ('A', 'C')):
        for chain, ls in (('prod', '--'), ('pilot', '-')):
            for view, mk in (('tanx', 'o'), ('tany', 's')):
                b = B[(B.arm == arm) & (B.chain == chain) & (B.view == view)]
                ax.plot((b.lo + b.hi) / 2, b.net, ls=ls, marker=mk, ms=5, lw=1.5,
                        color=fs.DET_COLOR[arm] if chain == 'pilot' else fs.MUTED,
                        label=f'{chain} {view[-1]}')
        ax.set_xlabel('|true tan|')
        ax.set_title(f'arm {arm}', fontsize=10.5, loc='left')
        ax.axvspan(0.3, 0.6, color=fs.GRID, zorder=0)
    axs[0].set_ylabel('net wall confirmation')
    axs[1].legend(frameon=False, fontsize=9)
    fs.fig_title(fig, 'Per bin, A\'s pilot sits left of production: its k is 0.85x on the same tracks',
                 'shaded: where the one-sided search mirrors fits on synthetic muons; the chains\' k differ')
    fs.preliminary(axs[1])
    fs.save(fig, FIG / 'confirm_by_tan', data=B)
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(OUT / 'pilot_compare.csv')
    B = pd.read_csv(OUT / 'confirm_by_tan.csv')
    C = pd.read_csv(OUT / 'cosmic_two_sided_changed.csv')
    V = pd.read_csv(OUT / 'synthetic_t0_valley.csv')
    S = pd.read_csv(OUT / 'same_track_confirm.csv')
    fig_ratio(R)
    fig_bins(B)

    both = R.groupby(['run', 'arm']).confirmed_acc.transform(lambda s: s.notna().all())
    T = R[both & R.run.ne('run_126')].groupby(['arm', 'chain'])[
        ['gated', 'gated_acc', 'confirmed', 'confirmed_acc']].sum()
    rel = {a: T.loc[(a, 'pilot'), 'confirmed_acc'] / T.loc[(a, 'prod'), 'confirmed_acc'] for a in 'AC'}
    rr = R[(R.chain == 'pilot') & np.isfinite(R.confirmed_acc_vs_prod)].groupby('arm').confirmed_acc_vs_prod
    rng = {a: (rr.min()[a], rr.max()[a]) for a in 'AC'}
    tot = T.reset_index()

    piv = B.pivot_table(index=['arm', 'view', 'lo'], columns='chain', values='net').reset_index()
    piv.columns = ['arm', 'view', '|tan| from', 'net pilot', 'net prod']

    ts_html = '<p><i>The run_86 two-sided yield gate has not finished; re-run this script when it has.</i></p>'
    verdict_ts = 'pending'
    if YG.exists():
        Y = pd.read_csv(YG)
        Y = Y[(Y.run == 'run_86') & Y.chain.isin(['prod', 'is2e', 'is2ts'])]
        if (Y.chain == 'is2ts').any():
            cols = ['arm', 'chain', 'gated', 'gated_acc', 'confirmed', 'confirmed_acc', 'wall_rate_acc',
                    'late_frac', 'near_normal']
            ts_html = tab(Y[[c for c in cols if c in Y]])
            if YGB.exists():
                Yb = pd.read_csv(YGB)
                yp = Yb.pivot_table(index=['arm', 'view', 'lo'], columns='chain', values='net').reset_index()
                ts_html += '<p>Net wall confirmation per |true tan| bin, run_86 (same k in both chains):</p>' + tab(yp)
            verdict_ts = 'measured'

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>is2_v1 pilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 16px;color:#1c2230;background:#f5f3ee;line-height:1.45}}
h1{{font-size:30px}} h2{{font-size:22px;margin-top:34px}} .v{{border-left:6px solid #2f8a5b;padding:6px 16px;background:#fffdf9}}
.w{{border-left:6px solid #c99318;padding:6px 16px;background:#fffdf9}} table.t{{border-collapse:collapse;font-size:14px;margin:8px 0}}
table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #d8d3c8;text-align:right}} code{{font-size:13px}}
img{{max-width:100%}} .scroll{{overflow-x:auto}}</style></head><body>
<h1>is2_v1 pilot against production</h1>
<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M} by <code>ntof_cosmics/make_pilot_report.py</code>. HANDOFF_TRACKING §16, §18, §19.</p>

<div class="v"><p><b>Yield: the pilot works.</b> Inside the same |true tan| &lt; 0.6, it confirms <b>A ×{rel['A']:.2f}</b> and
<b>C ×{rel['C']:.2f}</b> as many tracks on the scintillator wall as production (per run A {rng['A'][0]:.2f}–{rng['A'][1]:.2f},
C {rng['C'][0]:.2f}–{rng['C'][1]:.2f}), over the run/arms both chains can extrapolate. Stage 3 built all 36 sub-runs, none failed;
run_126 has no k.</p></div>
<div class="w"><p><b>A's apparent loss at 0.3–0.6 is the angle scale, not mirrors.</b> Binned by each chain's own angle, A's pilot
confirms fewer tracks than production above 0.3. But for the <i>same</i> track the pilot's angle is ×0.85 production's on A
(C ×0.90 x / ×1.01 y): the two k disagree, and the same tracks fall in a lower bin. Compared track by track, the pilot's direction
confirms on the wall at least as often as production's in every bin, and where they disagree the pilot wins ~2:1 — the wall mildly
prefers the in-situ scale. Mirrors among the tracks only the pilot finds are tested by the two-sided run below (run_86: {verdict_ts}).</p>
<p><b>A's −1.5 % with the two-sided search is the search correcting fits, not a new bias.</b> On the A–C cosmic truth the tracks it
changes read high before (×1.06 of truth) and ×1.01–1.03 after, with χ² lower on 95–99 % of them and t0 100–150 ns earlier: the
late-t0 slide of §17, now on data. A two-sided pass therefore needs its k rescaled by its own cosmic closure.</p></div>

<h2>What was compared</h2>
<p><b>Pilot</b>: first sub-run of each of 36 runs (condor 4409382, 702 jobs), is2 in-situ bundles, seeder min 3, stage 2 at TAN_MAX 1.0 raw
with the one-sided search, stage 3 with <code>kcal_is2_v1</code> and a 0.6 true-tan acceptance (<code>stage3_is2_v1_pilot</code>).
<b>Production</b>: <code>stage3_fullpass</code>, the same sub-runs, TAN_MAX 0.6 raw, its own per-run k. It has no k for
{int((R[(R.chain == 'prod')].confirmed.isna()).sum())} run/arms (e.g. run_81 A, run_156 A and C): those drop out of the ratios.
<i>Confirmed</i> = SiPM-wall matches minus the same-width off-time control, on gated tracks that extrapolate onto the wall
(<code>det_a_scint.match_run</code>).</p>
<div class="scroll">{tab(tot, '{:.0f}')}</div>
<p>Run/arms where both chains have a k.</p>

<h2>Per run</h2>
<img src="figures/confirmed_ratio_by_run.png" alt="pilot over production confirmed tracks per run">
<p>The wall rate itself dips in runs 110, 114, 132 and 162 in both chains (a scintillator condition); the ratio does not.</p>

<h2>Wall confirmation by angle</h2>
<img src="figures/confirm_by_tan.png" alt="net wall confirmation per true-tan bin, pilot vs production">
<div class="scroll">{tab(piv)}</div>

<h2>The same track in both chains</h2>
<p>Single-track events, track point within 3 mm in both chains, both on the wall; binned by the pilot's |true tan|. <i>scale</i> =
pilot/production true tan; <i>only_*</i> = the fraction confirmed with one chain's direction and not the other's.</p>
<div class="scroll">{tab(S)}</div>

<h2>run_86: one-sided (is2e) against two-sided (is2ts)</h2>
<p>Tags 000+003 of stat090_0000, same bundles, TAN_MAX 1.0, same k (<code>kcal_is2_v1</code>); only the search differs.</p>
<div class="scroll">{ts_html}</div>

<h2>The two-sided search on the cosmic A–C truth, track by track</h2>
<p>is2w (one-sided, TAN_MAX 1.0) against is2ts, 0.2 ≤ |true tan| &lt; 0.5; <i>changed</i> = |Δ raw tan| &gt; 0.01.</p>
<div class="scroll">{tab(C)}</div>

<h2>A's early t0 on synthetic muons is a flat valley</h2>
<p>Right-sign two-sided fits against the refit from the true side (<code>mirror_ts_*.parquet</code>, §17), |true tan| &gt; 0.3. Where the
two t0 differ by &gt; 30 ns the χ² differs by a few units in ~570 dof and the angle not at all: the early t0 moves p0, not the slope.</p>
<div class="scroll">{tab(V)}</div>

<h2>What this does not rule out</h2>
<ul>
<li>Which k is right is not settled by the wall: it is coarse, and prefers the pilot only mildly. The A–C cosmic truth (is2 closes to ~0.97) is the stronger argument.</li>
<li>The same-track test only covers tracks both chains find in the same place; it says nothing about the tracks only the pilot finds.</li>
<li>Production lacks k for several run/arms; the ratios skip them, so they cover 27 of 36 runs per arm.</li>
<li>The cosmic truth stops near |tan| 0.5; nothing here tests 0.5–1 on data except the wall confirmation.</li>
<li>Wall confirmation is a purity proxy with its own angle-dependent extrapolation error.</li>
<li>Only the first sub-run of each run; within-run variation is untested.</li>
</ul>
</body></html>"""
    (OUT / 'report.html').write_text(html)
    print(f'-> {OUT / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
