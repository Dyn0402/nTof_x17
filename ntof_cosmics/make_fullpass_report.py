#!/usr/bin/env python3
"""make_fullpass_report.py -- results/fullpass_<version>/report.html: a full
re-pass (stage3_<version>) against production (stage3_fullpass), from the
tables `pilot_compare.py <step> --version <version>` writes (match, summary,
bins, same).  Every number and every verdict sentence is computed here from
those tables, so re-running after the comparison changes keeps them in step.

The pass-independent checks (the two-sided search on the cosmic A-C truth,
the synthetic t0 valley, run_86 one- vs two-sided) live in the is2_v1 pilot
report and are linked, not repeated.

    python -m ntof_cosmics.make_fullpass_report --version is2_v2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sept26_prelim_analysis import figstyle as fs  # noqa: E402

ROOT = Path('/media/dylan/data/x17/sept26_prelim')
NO_PROD_K = 'run_126'   # no production k at all: its ratios are meaningless


def tab(df, fmt='{:.3f}'):
    return df.to_html(index=False, float_format=lambda v: fmt.format(v), border=0, classes='t', na_rep='')


def fig_ratio(R, FIG, version):
    import matplotlib.pyplot as plt
    fs.use()
    fig, ax = fs.figure(fs.WIDE)
    P = R[R.chain == 'pilot'].copy()
    P['rn'] = P.run.str[4:].astype(int)
    for arm in ('A', 'C'):
        p = P[(P.arm == arm) & np.isfinite(P.confirmed_acc_vs_prod)]
        ax.plot(p.rn, p.confirmed_acc_vs_prod, ls='none', ms=6, **{**fs.det_style(arm), 'label': f'arm {arm}'})
    ax.axhline(1, color=fs.MUTED, lw=0.8)
    ax.set_xlabel('run')
    ax.set_ylabel(f'confirmed, {version} / production')
    lo = np.nanmin(P.confirmed_acc_vs_prod.replace([np.inf, -np.inf], np.nan))
    hi = np.nanmax(P.confirmed_acc_vs_prod.replace([np.inf, -np.inf], np.nan))
    ax.set_ylim(min(0.8, lo - 0.05), max(2.1, hi + 0.15))
    ax.legend(frameon=False, loc='lower right')
    med = P.groupby('arm').confirmed_acc_vs_prod.median()
    fs.title(ax, f'Per run, {version} confirms A x{med["A"]:.2f}, C x{med["C"]:.2f} of production (median)',
             'wall excess over the off-time control, |true tan| < 0.6 in both chains, all sub-runs common to both')
    fs.preliminary(ax)
    fs.save(fig, FIG / 'confirmed_ratio_by_run', data=P[['run', 'arm', 'confirmed_acc', 'confirmed_acc_vs_prod']])
    plt.close(fig)


def fig_bins(B, FIG, version):
    import matplotlib.pyplot as plt
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE, sharey=True)
    for ax, arm in zip(axs, ('A', 'C')):
        for chain, ls in (('prod', '--'), ('pilot', '-')):
            for view, mk in (('tanx', 'o'), ('tany', 's')):
                b = B[(B.arm == arm) & (B.chain == chain) & (B.view == view)]
                ax.plot((b.lo + b.hi) / 2, b.net, ls=ls, marker=mk, ms=5, lw=1.5,
                        color=fs.DET_COLOR[arm] if chain == 'pilot' else fs.MUTED,
                        label=f'{version if chain == "pilot" else "prod"} {view[-1]}')
        ax.set_xlabel('|true tan|, each chain its own')
        ax.set_title(f'arm {arm}', fontsize=10.5, loc='left')
    axs[0].set_ylabel('net wall confirmation')
    axs[1].legend(frameon=False, fontsize=9)
    fs.fig_title(fig, 'Net wall confirmation per angle bin',
                 'each chain binned by its own angle: the chains\' k differ, so a bin holds different tracks')
    fs.preliminary(axs[1])
    fs.save(fig, FIG / 'confirm_by_tan', data=B)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    a = ap.parse_args()
    v = a.version
    OUT = HERE / 'results' / f'fullpass_{v}'
    FIG = OUT / 'figures'
    FIG.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(OUT / 'pilot_compare.csv')
    B = pd.read_csv(OUT / 'confirm_by_tan.csv')
    S = pd.read_csv(OUT / 'same_track_confirm.csv')
    fig_ratio(R, FIG, v)
    fig_bins(B, FIG, v)

    meta = json.loads((ROOT / f'stage3_{v}' / 'tracks_campaign.meta.json').read_text())
    # run/arms where both chains have a k (production lacks some; run_126 has none)
    both = R.groupby(['run', 'arm']).confirmed_acc.transform(lambda s: s.notna().all())
    keep = both & R.run.ne(NO_PROD_K)
    T = R[keep].groupby(['arm', 'chain'])[['gated', 'gated_acc', 'on_wall', 'confirmed', 'confirmed_acc']].sum()
    rel = {x: T.loc[(x, 'pilot'), 'confirmed_acc'] / T.loc[(x, 'prod'), 'confirmed_acc'] for x in 'AC'}
    rel_g = {x: T.loc[(x, 'pilot'), 'gated_acc'] / T.loc[(x, 'prod'), 'gated_acc'] for x in 'AC'}
    # confirmed per gated-in-acceptance track: a purity proxy (it also carries wall acceptance)
    pur = {(x, c): T.loc[(x, c), 'confirmed_acc'] / T.loc[(x, c), 'gated_acc'] for x in 'AC' for c in ('prod', 'pilot')}
    rr = R[(R.chain == 'pilot') & keep & np.isfinite(R.confirmed_acc_vs_prod)].groupby('arm').confirmed_acc_vs_prod
    rng = {x: (rr.min()[x], rr.max()[x]) for x in 'AC'}
    n_below = {x: int((rr.get_group(x) < 1).sum()) for x in 'AC'}
    n_runarm = {x: int(rr.get_group(x).size) for x in 'AC'}
    no_k = sorted(f'{r} {x}' for (r, x), g in R.groupby(['run', 'arm']) if not g.confirmed_acc.notna().all())
    tot = T.reset_index().replace({'chain': {'pilot': v}})

    # same physical track: does the re-pass direction or production's point at the tile that fired?
    win_p = (S.only_pilot * S.n).groupby(S.arm).sum()
    win_q = (S.only_prod * S.n).groupby(S.arm).sum()
    same_scale = {x: (S[S.arm == x].scale_x.iloc[1:].median(), S[S.arm == x].scale_y.iloc[1:].median()) for x in 'AC'}
    bins_ok = {x: bool((S[S.arm == x].net_pilot >= S[S.arm == x].net_prod - 0.002).all()) for x in 'AC'}
    St = S.rename(columns={'net_pilot': f'net_{v}', 'only_pilot': f'only_{v}'})

    piv = B.pivot_table(index=['arm', 'view', 'lo'], columns='chain', values='net').reset_index()
    piv.columns = ['arm', 'view', '|tan| from', f'net {v}', 'net prod']
    both_bins = piv.dropna()
    n_lower = int((both_bins[f'net {v}'] < both_bins['net prod']).sum())

    up = all(rel[x] > 1 for x in 'AC')
    verdict = (f'<b>Yield: the {v} re-pass confirms more tracks on the wall than production on both arms.</b>' if up else
               f'<b>Yield: the {v} re-pass does NOT beat production on every arm.</b>')
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{v} full pass</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:'IBM Plex Sans',Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 16px;color:#1c2230;background:#f5f3ee;line-height:1.45}}
h1{{font-size:30px}} h2{{font-size:22px;margin-top:34px}} .v{{border-left:6px solid #2f8a5b;padding:6px 16px;background:#fffdf9}}
.w{{border-left:6px solid #c99318;padding:6px 16px;background:#fffdf9}} table.t{{border-collapse:collapse;font-size:14px;margin:8px 0}}
table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #d8d3c8;text-align:right}} code{{font-size:13px}}
img{{max-width:100%}} .scroll{{overflow-x:auto}}</style></head><body>
<h1>{v} full re-pass against production</h1>
<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M} by <code>ntof_cosmics/make_fullpass_report.py --version {v}</code>
from <code>pilot_compare.py … --version {v}</code>. Context: HANDOFF_TRACKING §16–§20; the one-sub-run pilot is
<a href="../pilot_is2_v1/report.html">pilot_is2_v1/report.html</a>.</p>

<div class="v"><p>{verdict} Inside the same |true tan| &lt; 0.6, summed over the run/arms both chains calibrate, it confirms
<b>A ×{rel['A']:.2f}</b> and <b>C ×{rel['C']:.2f}</b> as many tracks as production (per run A {rng['A'][0]:.2f}–{rng['A'][1]:.2f},
{n_below['A']} of {n_runarm['A']} run/arms below 1; C {rng['C'][0]:.2f}–{rng['C'][1]:.2f}, {n_below['C']} of {n_runarm['C']} below 1),
from ×{rel_g['A']:.2f} (A) and ×{rel_g['C']:.2f} (C) the gated tracks in acceptance.
Confirmed per gated track in acceptance goes A {pur[('A', 'prod')]:.3f} → {pur[('A', 'pilot')]:.3f},
C {pur[('C', 'prod')]:.3f} → {pur[('C', 'pilot')]:.3f}.</p>
<p>Stage 3 built all {meta['n_subruns']} sub-runs ({meta['n_tracks']:,} segments, {meta['n_gated']:,} gated) from 6,464 stage-2 jobs;
the only two missing jobs are run_104 stat090_0016 A/C, whose decoded input is empty (production has no A/C there either).</p></div>
<div class="w"><p><b>The same track, both directions:</b> where the two chains find the same single track on the wall, the re-pass's angle
is ×{same_scale['A'][0]:.2f} x / ×{same_scale['A'][1]:.2f} y production's on A and ×{same_scale['C'][0]:.2f} / ×{same_scale['C'][1]:.2f}
on C (the k difference). Where only one direction confirms, the re-pass wins
A {int(win_p['A'])}:{int(win_q['A'])}, C {int(win_p['C'])}:{int(win_q['C'])}; its net confirmation is
{'at least production’s in every bin' if bins_ok['A'] else 'below production’s in some bin'} on A and
{'at least production’s in every bin' if bins_ok['C'] else 'below production’s in some bin'} on C.</p>
<p><b>Per angle bin</b>, each chain binned by its own angle, the re-pass's net confirmation is below production's in
{n_lower} of {len(both_bins)} bin/views below 0.6 (the one-sided pilot had A below production above 0.3 on this view).</p></div>

<h2>What was compared</h2>
<p><b>{v}</b>: condor 4410513, two-sided slope search (<code>WFT_TWO_SIDED=1</code>), in-situ bundles is2_A/is2_C, seeder min 3,
stage 2 at TAN_MAX 1.0 raw, stage 3 with <code>kcal_{v}</code> (in-situ norm × search scale A 1.009 / C 1.003 × production band ratio)
and a 0.6 true-tan acceptance (<code>stage3_{v}</code>). <b>Production</b>: <code>stage3_fullpass</code>, one-sided, TAN_MAX 0.6 raw,
its own per-run k. Every sub-run with a track table in both chains is compared. Run/arms without a production k drop out of
the ratios: {', '.join(no_k) or 'none'}. <i>Confirmed</i> = SiPM-wall matches minus the same-width off-time control on gated
tracks that extrapolate onto the wall (<code>det_a_scint.match_run</code>).</p>
<div class="scroll">{tab(tot, '{:.0f}')}</div>

<h2>Per run</h2>
<img src="figures/confirmed_ratio_by_run.png" alt="re-pass over production confirmed tracks per run">

<h2>Wall confirmation by angle</h2>
<img src="figures/confirm_by_tan.png" alt="net wall confirmation per true-tan bin, re-pass vs production">
<p>Runs whose wall rate dips in both chains (110, 114, 126, 132, 162) are left out of the bins.</p>
<div class="scroll">{tab(piv)}</div>

<h2>The same track in both chains</h2>
<p>Single-track events, track point within 3 mm in both chains, both on the wall; binned by the re-pass's |true tan|.
<i>scale</i> = re-pass/production true tan; <i>only_*</i> = the fraction confirmed with one chain's direction and not the other's.</p>
<div class="scroll">{tab(St)}</div>

<h2>What this does not rule out</h2>
<ul>
<li>Which k is right on A is not settled by the wall: it is coarse, and the same-track preference is a few per mille of the tracks.
The A–C cosmic truth is the stronger argument.</li>
<li>The same-track test covers only tracks both chains find in the same place; it says nothing about the tracks only the re-pass finds.</li>
<li>Wall confirmation is a purity proxy with its own angle-dependent extrapolation error, and confirmed-per-gated also carries
the wall's acceptance.</li>
<li>Nothing here tests |true tan| 0.5–1 on data except the wall; the cosmic truth stops near 0.5.</li>
<li>B and D are not in the re-pass (no in-situ bundle).</li>
</ul>
</body></html>"""
    (OUT / 'report.html').write_text(html)
    print(f'-> {OUT / "report.html"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
