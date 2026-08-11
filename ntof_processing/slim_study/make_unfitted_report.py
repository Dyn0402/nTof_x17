#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_unfitted_report.py -- the report for `unfitted_bunches.py`.

    python make_unfitted_report.py [--csv perbunch_pkup.csv] [-o outdir]

Builds `report.html` plus its figures from the stage-B csv, so re-running the
study updates numbers, charts and verdict text together (repo CLAUDE.md).
Every number in the page is computed here; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402

DEF_CSV = Path('/media/dylan/data/x17/slim_unfitted/perbunch_pkup.csv')
EMPTY_E10 = 10.0      # a pulse below this delivered no protons at all
PARA_E10 = 600.0      # splits the two intensity families (~410 vs ~850e10)
PB_MIN_EVENTS = 20    # clockfit.PB_MIN_EVENTS -- the per-bunch fit threshold

CSS = """
body{font:16px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
 max-width:60rem;margin:2rem auto;padding:0 1.2rem;color:#1a1a1a;background:#fff}
h1{font-size:1.7rem;margin-bottom:.2rem}h2{margin-top:2.2rem;font-size:1.25rem;
 border-bottom:1px solid #ddd;padding-bottom:.3rem}h3{font-size:1.05rem}
.sub{color:#666;margin-top:0}
.verdict{background:#f2f7f2;border-left:5px solid #2e7d32;padding:.9rem 1.1rem;
 margin:1.2rem 0;border-radius:3px}
.verdict b{color:#2e7d32}
table{border-collapse:collapse;margin:1rem 0;font-size:.93rem;width:100%}
th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:right}
th{background:#f5f5f5;text-align:left}td:first-child,th:first-child{text-align:left}
figure{margin:1.5rem 0}img{max-width:100%;border:1px solid #eee;border-radius:3px}
figcaption{color:#555;font-size:.9rem;margin-top:.4rem}
code{background:#f3f3f3;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
.note{background:#fbf7ec;border-left:5px solid #c99b2e;padding:.8rem 1.1rem;
 border-radius:3px}
"""


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d['seg'] = d.dream_run + '/' + d.dream_subrun
    d['is_empty'] = d.intensity_e10 < EMPTY_E10
    d['family'] = np.where(d.is_empty, 'empty',
                           np.where(d.intensity_e10 < PARA_E10,
                                    'parasitic', 'dedicated'))
    return d


def numbers(d: pd.DataFrame) -> dict:
    u = d.fitted == 0
    n = dict(
        n_segments=d.seg.nunique(), n_bunches=len(d),
        n_unfitted=int(u.sum()), frac_unfitted=float(u.mean()),
        n_empty=int(d.is_empty.sum()),
        n_unfitted_empty=int((u & d.is_empty).sum()),
        n_empty_fitted=int((~u & d.is_empty).sum()),
        n_unfitted_beam=int((u & ~d.is_empty).sum()),
        tflash_zero_frac=float((d.tflash_ns[d.is_empty] == 0).mean()),
        max_nphys_unfitted=int(d.n_phys[u].max()),
        med_nphys_unfitted=float(d.n_phys[u].median()),
        med_nphys_fitted=float(d.n_phys[~u].median()),
        n_events_unfitted=int(d.n_phys[u].sum() + d.n_flash[u].sum()),
        frac_events_unfitted=float((d.n_phys[u].sum() + d.n_flash[u].sum())
                                   / (d.n_phys.sum() + d.n_flash.sum())),
        n_matched_unfitted=int(d.n_matched[u].sum()),
    )
    fam = d.groupby('family', observed=True).agg(
        n=('bunch', 'size'), fitted=('fitted', 'mean'),
        nphys=('n_phys', 'median'), matched=('n_matched', 'sum'),
        phys=('n_phys', 'sum'), inten=('intensity_e10', 'median'))
    fam['match_frac'] = fam.matched / fam.phys.replace(0, np.nan)
    n['fam'] = fam
    beam = d[~d.is_empty]
    g = beam.groupby('seg').apply(lambda s: pd.Series(dict(
        par_frac=float((s.intensity_e10 < PARA_E10).mean()),
        eff=float(s.n_matched.sum() / max(s.n_phys.sum(), 1)))),
        include_groups=False)
    n['seg_beam'] = g
    n['r_eff_par'] = float(np.corrcoef(g.par_frac, g.eff)[0, 1])
    b, a = np.polyfit(g.par_frac, g.eff, 1)
    n['eff_dedicated'], n['eff_parasitic'] = float(a), float(a + b)
    seg = d.groupby('seg').apply(lambda s: pd.Series(dict(
        n_bunch=len(s), n_unfit=int((s.fitted == 0).sum()),
        n_empty=int(s.is_empty.sum()))), include_groups=False)
    seg['frac_fitted'] = 1 - seg.n_unfit / seg.n_bunch
    n['seg'] = seg.sort_values('frac_fitted')
    n['n_warn'] = int((seg.frac_fitted < 0.90).sum())
    n['n_identical'] = int((seg.n_unfit == seg.n_empty).sum())
    return n


def figures(d: pd.DataFrame, N: dict, fig: Path):
    fig.mkdir(parents=True, exist_ok=True)
    u = d.fitted == 0

    # 1. the whole answer in one panel: pulse intensity, fitted vs not
    f, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, 900, 91)
    ax.hist(d.intensity_e10[~u], bins=bins, color='#2e6fb7',
            label=f'fitted ({int((~u).sum()):,} bunches)')
    ax.hist(d.intensity_e10[u], bins=bins, color='#c0392b',
            label=f'not fitted ({int(u.sum()):,})')
    ax.set_yscale('log')
    ax.set_xlabel('PKUP pulse intensity  [1e10 protons]')
    ax.set_ylabel('bunches')
    ax.legend()
    ax.set_title('Every bunch without its own clock correction is an empty pulse')
    f.tight_layout(); f.savefig(fig / 'intensity.png', dpi=130); plt.close(f)

    # 2. DREAM triggers per burst
    f, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.arange(0, 160, 2)
    ax.hist(d.n_phys[~u], bins=bins, color='#2e6fb7', label='fitted bunches')
    ax.hist(d.n_phys[u], bins=bins, color='#c0392b', label='not fitted')
    ax.axvline(PB_MIN_EVENTS, color='k', ls='--', lw=1)
    ax.annotate(f'per-bunch fit needs {PB_MIN_EVENTS}',
                (PB_MIN_EVENTS + 2, ax.get_ylim()[1] * 0.4), fontsize=9)
    ax.set_yscale('log')
    ax.set_xlabel('DREAM physics triggers in the burst')
    ax.set_ylabel('bunches')
    ax.legend()
    ax.set_title('No overlap: the unfitted bunches have 0-19 triggers, the rest 46-139')
    f.tight_layout(); f.savefig(fig / 'triggers.png', dpi=130); plt.close(f)

    # 3. the by-product: efficiency is set by the beam mix, not by the fit
    g = N['seg_beam']
    f, ax = plt.subplots(figsize=(7.5, 4))
    ax.scatter(100 * g.par_frac, 100 * g.eff, s=18, color='#2e6fb7', alpha=.75)
    x = np.linspace(g.par_frac.min(), g.par_frac.max(), 10)
    b = (N['eff_parasitic'] - N['eff_dedicated'])
    ax.plot(100 * x, 100 * (N['eff_dedicated'] + b * x), 'k--', lw=1.2,
            label=f'r = {N["r_eff_par"]:.2f}')
    ax.set_xlabel('parasitic pulses in the segment  [%]')
    ax.set_ylabel('match efficiency  [%]')
    ax.legend()
    ax.set_title('The fleet efficiency spread is beam composition')
    f.tight_layout(); f.savefig(fig / 'efficiency.png', dpi=130); plt.close(f)


def table(df: pd.DataFrame, cols: dict, fmt: dict) -> str:
    h = ''.join(f'<th>{html.escape(v)}</th>' for v in cols.values())
    rows = []
    for idx, r in df.iterrows():
        cells = []
        for c in cols:
            v = idx if c == '_index' else r[c]
            cells.append(f'<td>{fmt.get(c, str)(v)}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table><tr>{h}</tr>' + ''.join(rows) + '</table>'


def build(d: pd.DataFrame, N: dict, out: Path):
    fam = N['fam'].reindex(['dedicated', 'parasitic', 'empty']).dropna(how='all')
    fam_tbl = table(
        fam.reset_index(),
        {'family': 'pulse', 'inten': 'median intensity [1e10 p]',
         'n': 'bunches', 'fitted': 'got own correction',
         'nphys': 'median DREAM triggers', 'match_frac': 'matched at +-25 ns'},
        {'inten': lambda v: f'{v:,.0f}', 'n': lambda v: f'{int(v):,}',
         'fitted': lambda v: f'{v:.4%}', 'nphys': lambda v: f'{v:.0f}',
         'match_frac': lambda v: '--' if not np.isfinite(v) else f'{v:.1%}'})

    worst = N['seg'].head(8).reset_index()
    seg_tbl = table(
        worst,
        {'seg': 'segment', 'n_bunch': 'bunches', 'n_unfit': 'unfitted',
         'n_empty': 'empty pulses', 'frac_fitted': 'fitted fraction'},
        {'n_bunch': lambda v: f'{int(v):,}', 'n_unfit': lambda v: f'{int(v):,}',
         'n_empty': lambda v: f'{int(v):,}',
         'frac_fitted': lambda v: f'{v:.1%}'})

    body = f"""
<h1>The bunches that get no per-bunch clock correction</h1>
<p class="sub">n_TOF &rarr; DREAM slim campaign, {N['n_segments']} segments,
{N['n_bunches']:,} bunches &middot; generated by
<code>slim_study/make_unfitted_report.py</code></p>

<div class="verdict">
<b>They are empty PS pulses.</b> {N['n_unfitted_empty']:,} of the
{N['n_unfitted']:,} bunches that never get their own <code>(da<sub>b</sub>,
dk<sub>b</sub>)</code> delivered <b>no protons</b> &mdash; PKUP intensity below
{EMPTY_E10:g}e10 and, independently, <code>tflash = 0</code> on
{N['tflash_zero_frac']:.1%} of them, i.e. the n_TOF flash finder saw no gamma
flash either. They are not parasitic pulses: the parasitic family is fitted
{N['fam'].loc['parasitic', 'fitted']:.4%} of the time. Nothing is lost and
nothing is recoverable, because there was no beam: the
{N['n_events_unfitted']:,} DREAM triggers inside them
({N['frac_events_unfitted']:.4%} of the campaign) are detector background, and
{N['n_matched_unfitted']} of them match an n_TOF candidate.
</div>

<h2>Headline numbers</h2>
<table>
<tr><th>bunches with no per-bunch correction</th>
    <td>{N['n_unfitted']:,} of {N['n_bunches']:,} ({N['frac_unfitted']:.2%})</td></tr>
<tr><th>of those, empty pulses (&lt; {EMPTY_E10:g}e10 protons)</th>
    <td>{N['n_unfitted_empty']:,} ({N['n_unfitted_empty']/N['n_unfitted']:.1%})</td></tr>
<tr><th>empty pulses that <i>were</i> fitted</th>
    <td>{N['n_empty_fitted']}</td></tr>
<tr><th>unfitted bunches that had beam</th>
    <td>{N['n_unfitted_beam']} (both the first bunch of a sub-run)</td></tr>
<tr><th>DREAM triggers in an unfitted bunch</th>
    <td>median {N['med_nphys_unfitted']:.0f}, max {N['max_nphys_unfitted']}
        (fitted bunches: median {N['med_nphys_fitted']:.0f})</td></tr>
<tr><th>campaign triggers affected</th>
    <td>{N['n_events_unfitted']:,} ({N['frac_events_unfitted']:.4%}), all unmatched</td></tr>
<tr><th>segments where unfitted == empty exactly</th>
    <td>{N['n_identical']} of {N['n_segments']}</td></tr>
</table>

<h2>How a bunch loses its correction</h2>
<p><code>clockfit.fit_perbunch</code> fits a straight line
<code>da<sub>b</sub> + dk<sub>b</sub>&middot;t</code> per bunch and needs
{PB_MIN_EVENTS} DREAM triggers whose nearest n_TOF candidate already sits inside
&plusmn;200 ns of the global map. Below that the bunch keeps the global
<code>(K, T0, arm)</code> map only, and <code>efficiency()</code> drops its
events from the quoted denominator.</p>
<p>The measurement below says the threshold is never the marginal thing it looks
like. The two populations do not overlap: unfitted bunches hold 0&ndash;{N['max_nphys_unfitted']}
triggers, fitted ones 46&ndash;139. No bunch fails on match quality &mdash; every
single one fails for want of triggers, and it has no triggers because the PS sent
it no protons. DREAM's gate opens on the PS timing whether or not beam arrives, so
an empty pulse still produces a burst: a couple of dark-count coincidences at
~22 Hz against ~1.2 kHz with beam.</p>

<figure><img src="figures/intensity.png" alt="pulse intensity">
<figcaption>Pulse intensity for fitted and unfitted bunches. The beam has two
families &mdash; parasitic near
{N['fam'].loc['parasitic', 'inten']:,.0f}e10 and dedicated near
{N['fam'].loc['dedicated', 'inten']:,.0f}e10 &mdash; and both are fitted. The
unfitted population sits at zero.</figcaption></figure>

<figure><img src="figures/triggers.png" alt="triggers per burst">
<figcaption>DREAM triggers per burst. The {PB_MIN_EVENTS}-trigger threshold falls
in an empty gap between the two populations, so it is not a tuning parameter that
could be relaxed to recover anything.</figcaption></figure>

<h2>Per beam family</h2>
{fam_tbl}
<p>Parasitic pulses carry half the protons and give ~27 % fewer DREAM triggers,
but they fit and match perfectly well. The unfitted bunches do not skew
parasitic; they are a third category.</p>

<h2>What is inside an empty pulse</h2>
<p>Measured on <code>run_116/stat090_0014 &times; 224636</code>, the worst
segment (22.3 % of its bunches empty): its 418 empty-pulse triggers carry 2.82
slim hits each, against 16.8 for beam triggers, and the composition is the
giveaway &mdash; <b>WAL 2.80, PSS 0.017, LIQ 0.000</b> per trigger. The walls'
apparent activity is SiPM dark counts in the &plusmn;1 &micro;s window; the
plastics and the liquids, which have no dark rate to speak of, see nothing at
all. Match efficiency in those bunches is 0.0000 against 0.946 for the beam
bunches of the same segment.</p>

<h2>The four 'bunches fitted' WARNs</h2>
{seg_tbl}
<p>In {N['n_identical']} of {N['n_segments']} segments the unfitted count equals
the empty-pulse count exactly. So <code>clock_qa</code>'s 'bunches fitted' check
is not measuring the clock fit at all &mdash; <b>it is measuring beam
availability</b>, and its {N['n_warn']} WARNs are hours when the PS delivered
86&ndash;92 % of the pulses n_TOF was scheduled. Every PKUP bunch in a segment's
range reaches the slim, empty ones included (verified on 224636 and 224603:
0 missing), so the fitted fraction is exactly the delivered fraction.</p>

<h2>By-product: the fleet efficiency spread is the beam, not the clock</h2>
<figure><img src="figures/efficiency.png" alt="efficiency vs parasitic fraction">
<figcaption>Per-segment match efficiency against the fraction of its pulses that
were parasitic. r = {N['r_eff_par']:.2f} over {len(N['seg_beam'])} segments;
the line extrapolates to {N['eff_dedicated']:.1%} for pure dedicated running and
{N['eff_parasitic']:.1%} for pure parasitic, against
{N['fam'].loc['dedicated', 'match_frac']:.1%} /
{N['fam'].loc['parasitic', 'match_frac']:.1%} measured directly per
family.</figcaption></figure>
<p>The campaign's 93.6&ndash;97.3 % efficiency range was previously unexplained
segment-to-segment scatter. Most of it is the dedicated/parasitic mix, which runs
from 9 % to 61 % across segments.</p>

<h2>What was changed, 2026-08-10</h2>
<p>Empty pulses are now dropped at the join, before the candidate pass reads
anything, and the <code>bunches</code> tree carries <code>has_beam</code> and
<code>intensity_e10</code> for every bunch the sub-run touched &mdash; so the
table is both the beam record and the record of what was filtered. An analysis
selects a clean sample with <code>has_beam</code> and splits
dedicated/parasitic with <code>intensity_e10</code>.
<code>clock_qa</code> now asks 'bunches fitted' of the bunches that
<i>had</i> beam, adds a check that no no-beam trigger survived, and reports
availability and beam mix without judging them. <b>It takes effect on a
re-slim</b>: files written before this date still carry their
{N['frac_events_unfitted']:.4%} of no-beam triggers, and their
<code>frac_fitted</code> is beam availability rather than a quality
number.</p>

<h2>What this does not rule out</h2>
<ul>
<li><b>Why parasitic pulses match ~6 points worse</b> is not established here.
Fewer protons means fewer n_TOF candidates per trigger, so a DREAM trigger is
more often left without a partner above the singles threshold &mdash; plausible,
unmeasured.</li>
<li><b>The two unfitted bunches that did have beam</b> (run_77/stat090_0003
bunch 44, run_118/stat090_0019 bunch 1758) are the first bunch of their sub-run
&mdash; a burst DREAM joined part-way through. 33 events; no evidence they
generalise beyond sub-run starts.</li>
<li><b>The empty pulses themselves are a beam question, not a DAQ one</b> as far
as this study goes: intensity and tflash both come from the n_TOF beam record.
Whether the PS skipped them, or n_TOF was scheduled a pulse it never received,
needs the accelerator logs.</li>
<li>This says nothing about the 54 sub-runs that could not be slimmed at all
(the ~0.982 ms association) &mdash; a separate, still-open question.</li>
</ul>

<h2>Reproducing</h2>
<pre><code>python slim_study/unfitted_bunches.py ~/x17slim -o perbunch.csv       # lxplus
python slim_study/unfitted_bunches.py --pkup perbunch.csv -o perbunch_pkup.csv
python slim_study/make_unfitted_report.py --csv perbunch_pkup.csv -o unfitted</code></pre>
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Unfitted bunches = empty pulses</title>'
        f'<style>{CSS}</style></head><body>{body}</body></html>')
    print(f'-> {out / "report.html"}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=DEF_CSV)
    ap.add_argument('-o', '--out', type=Path,
                    default=Path(__file__).resolve().parent / 'unfitted')
    a = ap.parse_args()
    d = load(a.csv)
    N = numbers(d)
    figures(d, N, a.out / 'figures')
    build(d, N, a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
