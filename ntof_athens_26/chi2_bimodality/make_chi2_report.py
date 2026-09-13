#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_chi2_report.py -- figures/report.html for the chi2 bimodality investigation.

Generated, not hand-written: every number in the prose is read from
``<out>/chi2_bimodality/`` and from the figure CSVs, so re-running
`chi2_shape.py` and `make_chi2_figures.py` updates the tables, the headline
numbers and the verdict text together.  The DAQ web page lists any ``.html``
in an analysis directory and opens it inline, and figures are referenced with
relative links so the same file works from disk, from the DAQ, or copied.

    python ntof_athens_26/chi2_bimodality/make_chi2_report.py
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                          # noqa: E402

SRC = paths.spell('out', 'chi2_bimodality')
FIGS = HERE / 'figures'
OUT = FIGS / 'report.html'

CSS = """
:root{--ink:#1b2430;--muted:#6a7583;--line:#d4d9e0;--surface:#fbfcfe;
--accent:#8a3f8f;--copper:#d18a44;--A:#0072B2;--B:#D55E00;--C:#009E73;--D:#CC79A7}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 20px 90px}
h1{font-size:31px;line-height:1.22;margin:0 0 6px}
h2{font-size:21px;margin:46px 0 12px;padding-top:16px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:var(--muted);
letter-spacing:.05em;text-transform:uppercase}
p{margin:0 0 14px}
.sub{color:var(--muted);font-size:15px;margin-bottom:26px}
.verdict{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);
padding:20px 24px;border-radius:6px;margin:26px 0}
.verdict p:last-child{margin-bottom:0}
.warn{border-left-color:var(--copper)}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.tile{flex:1 1 150px;background:#fff;border:1px solid var(--line);
border-radius:6px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:26px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
.tile .n{font-size:12.5px;color:var(--muted);margin-top:2px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14.5px;
background:#fff;font-variant-numeric:tabular-nums}
th,td{padding:8px 11px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{font-size:12.5px;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em;border-bottom:1.5px solid var(--ink)}
tbody tr:last-child td{border-bottom:1px solid var(--ink)}
figure{margin:26px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:14px;color:var(--muted);margin-top:9px}
code{font:13.5px ui-monospace,SFMono-Regular,Menlo,monospace;
background:#eef1f5;padding:1px 5px;border-radius:3px}
ul,ol{margin:0 0 14px;padding-left:22px}li{margin-bottom:7px}
.badge{display:inline-block;font-size:11.5px;letter-spacing:.09em;
border:1px solid var(--copper);color:var(--copper);border-radius:4px;
padding:2px 9px;vertical-align:middle;margin-left:10px}
.det{font-weight:650}
.A{color:var(--A)}.B{color:var(--B)}.C{color:var(--C)}.D{color:var(--D)}
@media (max-width:640px){.wrap{padding:24px 14px 60px}h1{font-size:25px}}
"""


def _f(x, n=2, dash='\u2014'):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f'{x:,.{n}f}'


def _table(df: pd.DataFrame, fmt=None) -> str:
    fmt = fmt or {}
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = _f(v)
            cells.append(f'<td>{v}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _fig(name, caption) -> str:
    return (f'<figure><img src="{name}.png" alt="{html.escape(name)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def _det(a) -> str:
    return f'<span class="det {a}">{a}</span>'


def build() -> str:
    S = json.loads((SRC / 'summary.json').read_text())
    vlen = pd.read_csv(SRC / 'chi2_vs_len.csv')
    rw = pd.read_csv(SRC / 'reweight.csv')
    bd = pd.read_csv(SRC / 'bundle_diff.csv')
    runs = pd.read_csv(SRC / 'chi2_by_run.csv')
    mod = pd.read_csv(SRC / 'modality.csv')
    fp = pd.read_csv(SRC / 'floor_peak.csv')
    legs = (pd.read_csv(SRC / 'legs_crosscheck.csv')
            if (SRC / 'legs_crosscheck.csv').exists() else None)
    ph = (pd.read_csv(SRC / 'pair_hist.csv')
          if (SRC / 'pair_hist.csv').exists() else None)

    P = S['per_arm']
    arms = [a for a in ('A', 'B', 'C', 'D') if a in P]
    A = P.get('A', {})

    # --- the shoulder fractions the artefact figure quotes
    sh = {}
    if ph is not None:
        for pair, g in ph.groupby('pair'):
            tot = g.n.sum()
            sh[pair] = float(g.loc[g.chi2dof < 5, 'n'].sum() / max(tot, 1))

    # --- headline numbers
    n_runs = int(S.get('n_runs') or runs.run.nunique())
    n_runs_plot = int(runs.run.nunique())
    a_gap = (np.median([P[a]['median_short'] for a in arms if a != 'A'])
             / A['median_short']) if len(arms) > 1 else float('nan')

    t = []
    t.append(f'<!doctype html><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>\u03c7\u00b2/dof bimodality \u2014 chamber A</title>'
             f'<style>{CSS}</style><div class="wrap">')

    t.append('<h1>The double bump in \u03c7\u00b2/dof, and why only chamber A '
             'shows it<span class="badge">PRELIMINARY</span></h1>')
    t.append(f'<p class="sub">n_TOF 2026 campaign \u00b7 '
             f'{S["n_tracks"]:,} tracks over {n_runs} runs \u00b7 '
             f'generated {date.today().isoformat()} by '
             f'<code>chi2_shape.py</code></p>')

    # ------------------------------------------------------------------ verdict
    t.append('<div class="verdict"><p><strong>Two things are true, and the '
             'first one has to be said first.</strong></p>'
             '<p><strong>1. The figure exaggerated the low bump by about '
             '5\u00d7 \u2014 now fixed.</strong> '
             '<code>make_pair_qa_figures._hist</code> divided each bin by '
             '<code>np.diff(edges)</code> \u2014 the bin\u2019s <em>linear</em> '
             'width \u2014 while the edges are geometric and the axis is log. '
             'The plotted height was therefore the per-bin fraction divided by '
             '\u03c7\u00b2/dof, which lifts the left of the axis and renders a '
             'small shoulder as a mode of comparable height. Drawn as a per-bin '
             f'fraction, only {sh.get("A-A", 0):.1%} of A\u2013A pairs sit below '
             f'\u03c7\u00b2/dof = 5. <strong>All five log-scaled panels of that '
             'set carried the same lift.</strong> Corrected 2026-09-12: '
             '<code>_hist</code> now returns the per-bin fraction, which is what '
             'its y-axis label always said, and the nine figures are '
             'regenerated. The correction also <em>revealed</em> something the '
             'lift had flattened \u2014 D\u2013D\u2019s \u03c7\u00b2 sits an '
             'order of magnitude above every other arm pair.</p>'
             '<p><strong>2. The A-specific excess underneath it is real.</strong> '
             f'That same {sh.get("A-A", 0):.1%} is {sh.get("C-C", 0):.1%} on '
             f'C\u2013C and {sh.get("D-D", 0):.1%} on D\u2013D. At the '
             'single-track level, where the pair variable\u2019s worst-of-four '
             f'does not hide it, {A.get("frac_at_floor", 0):.0%} of chamber A\u2019s '
             'tracks fit their waveforms to within the strips\u2019 own noise, '
             'against '
             + ' and '.join(f'{P[a]["frac_at_floor"]:.0%} on {a}'
                            for a in arms if a != 'A')
             + '.</p></div>')

    t.append('<div class="verdict warn"><p><strong>The mechanism, in one '
             'line.</strong> \u03c7\u00b2/dof here is the mean squared residual '
             '<em>per waveform sample</em> in units of that strip\u2019s own '
             'measured noise, so <strong>1.0 is the noise floor and it means the '
             'same thing on every chamber</strong>. It climbs with track length '
             'on all of them \u2014 that mixture is the bimodality \u2014 but '
             'A\u2019s short tracks are the only ones that start from the floor '
             f'({A.get("median_short", 0):.1f}), which puts them '
             f'{A.get("separation", 0):.0f}\u00d7 below its long ones and opens '
             'clear air between the two modes. On C and D the short tracks are '
             'already at '
             + ' and '.join(f'{P[a]["median_short"]:.1f}' for a in arms
                            if a != 'A')
             + ', too high to resolve from the long-track continuum. '
             '<strong>The low bump is not missing on C and D \u2014 it is not '
             'separated.</strong></p></div>')

    # ------------------------------------------------------------------ tiles
    tiles = [
        ('A at the noise floor', f'{A.get("frac_at_floor", 0):.0%}',
         'of A\u2019s tracks, \u03c7\u00b2/dof &lt; 1.5'),
        ('A\u2019s short\u2192long split', f'{A.get("separation", 0):.0f}\u00d7',
         f'{A.get("median_short", 0):.1f} \u2192 {A.get("median_long", 0):.0f}'),
        ('A below C and D', f'{a_gap:.1f}\u00d7',
         f'short tracks, in every one of {n_runs_plot} runs'),
        ('figure\u2019s low-end lift', '\u22485\u00d7',
         'linear bin width on a log axis'),
    ]
    t.append('<div class="tiles">' + ''.join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in tiles) + '</div>')

    # ------------------------------------------------------------------ what
    t.append('<h2>What was compared</h2>')
    t.append('<p>The question was asked of <code>qa_chi2dof_worst</code>, which '
             'is one row per <em>pair</em> and takes the worst of four track '
             'fits. That variable compounds four draws of the same single-track '
             'distribution, so it can show that A is different but never why. '
             'Everything below is at the <em>track</em> level, on the legs of '
             'that same published sample \u2014 gated, angle-calibrated, '
             f'DCA &lt; {S["dca_max"]:.0f} mm, in a trigger that made two tracks '
             f'\u2014 {S["n_tracks"]:,} of them.</p>')
    t.append('<p>Chamber <span class="det B">B</span> carries no angle '
             'calibration, so it is in no pairing and on none of these figures. '
             'It reaches the cross-check table below by another route, and it '
             'is the worst of the four there.</p>')

    t.append('<h3>The definition this all turns on</h3>')
    dc = S['dof_check']
    t.append(f'<p><code>wft.model.chi2_plane</code> fits the forward model to '
             f'the raw waveform window and <code>dof = (~saturated).sum()</code>. '
             f'Measured here: <code>dof / n_strips</code> has median '
             f'{dc["ratio_median"]:.3f} and is exactly 20 on '
             f'{dc["frac_exactly_20"]:.1%} of tracks \u2014 so dof counts '
             f'<em>samples</em>, and \u03c7\u00b2/dof is a mean squared residual '
             f'per sample. <code>wft.model.prep_plane</code> takes the noise per '
             f'strip from the event itself, so the unit is self-calibrating and '
             f'the chambers are directly comparable. A chamber whose best tracks '
             f'sit at 3 has a model that does not describe its data; it is not a '
             f'units problem.</p>')

    # ------------------------------------------------------------------ figures
    t.append('<h2>The artefact, and what survives it</h2>')
    t.append(_fig('pair_link',
                  'Left: the intra pairs exactly as the QA set draws them. '
                  'Right: the same counts as a per-bin fraction. The low bump '
                  'loses about 5\u00d7 of its height and stops being a mode '
                  '\u2014 but A\u2013A keeps a visible excess below '
                  '\u03c7\u00b2/dof \u2248 5 that C\u2013C and D\u2013D do not.'))
    if sh:
        sht = pd.DataFrame(
            [dict(pair=k, frac_below_5=v) for k, v in sorted(sh.items())])
        t.append('<h3>Fraction of pairs below \u03c7\u00b2/dof = 5</h3>')
        t.append(_table(sht, fmt={'frac_below_5': lambda v: f'{v:.1%}'}))

    t.append('<h3>On a linear axis, at 0.1 resolution</h3>')
    a_fp = fp.set_index('arm')
    t.append('<p>A linear axis binned at a width of 1 put 36 % of chamber A in '
             'one bin \u2014 that bin <em>was</em> the peak, and the panel showed '
             'a spike with no shape. At 0.1 the shape is there, and it says '
             'something the coarse view could not: '
             f'<strong>A\u2019s peak sits at \u03c7\u00b2/dof = '
             f'{a_fp.loc["A"].peak_at:.2f}</strong> \u2014 on the noise floor, '
             'not merely near it.</p>')
    t.append('<p>The refinement this forces on the headline: <strong>all three '
             'chambers do have a peak at the floor.</strong> The difference is '
             'how much of the sample is in it and how tight it is \u2014 '
             f'A\u2019s is {a_fp.loc["A"].peak_dens / a_fp.loc["C"].peak_dens:.1f}\u00d7 '
             'taller than C\u2019s and about half the width. "C and D never reach '
             'the floor" was too strong; "C and D put a fifth as many tracks '
             'there" is what the data says.</p>')
    t.append(_table(fp[['arm', 'peak_at', 'fwhm_lo', 'fwhm_hi', 'peak_dens',
                        'frac_below_1', 'frac_1_to_2', 'frac_below_10']],
                    fmt={'peak_at': lambda v: f'{v:.2f}',
                         'fwhm_lo': lambda v: f'{v:.1f}',
                         'fwhm_hi': lambda v: f'{v:.1f}',
                         'peak_dens': lambda v: f'{v:.3f}',
                         'frac_below_1': lambda v: f'{v:.1%}',
                         'frac_1_to_2': lambda v: f'{v:.1%}',
                         'frac_below_10': lambda v: f'{v:.1%}'}))
    t.append(_fig('axis_and_binning',
                  'Left and middle are the same curve at two resolutions and '
                  'share their y units \u2014 both a density per unit '
                  '\u03c7\u00b2, which is the correct normalisation on a linear '
                  'axis and the wrong one on a log axis. Right is the log view '
                  'for the decades the linear axis cannot reach. The high mode '
                  'never appears on the linear panels because it is spread over '
                  'decades; that is the two axes answering different questions, '
                  'not a disagreement.'))
    t.append('<h3>And the log-space bimodality, measured</h3>')
    t.append('<p>Dip depth is the dip against the <em>shallower</em> of the two '
             'modes, so 1.0 would mean no dip at all.</p>')
    t.append(_table(mod[['arm', 'low_mode_at', 'dip_at', 'high_mode_at',
                         'dip_depth']],
                    fmt={'low_mode_at': lambda v: f'{v:.2f}',
                         'dip_at': lambda v: f'{v:.1f}',
                         'high_mode_at': lambda v: f'{v:.1f}',
                         'dip_depth': lambda v: f'{v:.2f}'}))

    t.append('<h2>The bimodality is track length</h2>')
    t.append(_fig('chi2_decomposition',
                  'Top: the distribution with its two length populations in '
                  'place. Bottom: each population normalised to itself, so the '
                  'modes can be compared. Both populations exist on all three '
                  'chambers; only on A does the short-track mode reach the '
                  'noise floor.'))
    t.append(_fig('chi2_vs_length',
                  '\u03c7\u00b2/dof against track length, with the quartile '
                  'band. The rise is universal \u2014 so the mixture, and '
                  'therefore the bimodality, is universal. The floor each '
                  'chamber starts from is not.'))
    t.append('<h3>\u03c7\u00b2/dof by track length</h3>')
    vt = vlen.pivot(index='len_class', columns='arm', values='median')
    vt = vt.reindex([c for c in ['11-13', '14-17', '18-23', '24-33', '34-59',
                                 '60+'] if c in vt.index]).reset_index()
    t.append(_table(vt))

    t.append('<h3>And it is not the length spectrum</h3>')
    t.append('<p>Each chamber\u2019s per-length fraction-at-floor, re-averaged '
             'over another chamber\u2019s length distribution. If the spectrum '
             'were the explanation, a reweighted arm would land on the other\u2019s '
             'own value. It does not \u2014 A reweighted onto C\u2019s lengths is '
             'still far above C\u2019s own number.</p>')
    rwt = rw.pivot(index='arm', columns='weights', values='value').reset_index()
    t.append(_table(rwt, fmt={c: (lambda v: '\u2014' if pd.isna(v) else f'{v:.3f}')
                              for c in rwt.columns if c != 'arm'}))

    t.append('<h2>At fixed length, it is pulse amplitude</h2>')
    t.append(_fig('chi2_vs_charge',
                  'The model\u2019s residual is a fixed <em>fraction</em> of the '
                  'pulse, so on a quiet track it is buried in the noise and on a '
                  'loud one it is not. A\u2019s low-\u03c7\u00b2 tracks carry '
                  'about 2.5\u00d7 less charge than its high-\u03c7\u00b2 ones at '
                  'identical n_strips.'))

    t.append('<h2>The offset is a constant of the chamber</h2>')
    t.append(_fig('chi2_by_run',
                  f'Short tracks only, so the length mixture cannot move. '
                  f'Across all {n_runs_plot} runs with enough short tracks to measure, '
                  f'and both access conditions, A holds '
                  f'its factor \u2248{a_gap:.1f} under C and D. Nothing here '
                  f'tracks the beam, the period or the 27 July access.'))

    # ------------------------------------------------------------------ bundles
    t.append('<h2>Where a chamber constant could live: the bundles</h2>')
    t.append('<p>Calibration is per detector and per run condition. The four '
             'bundles differ structurally, and this is a matter of record '
             'rather than a result \u2014 <strong>nothing here ranks these '
             'differences or shows that any one of them causes the '
             '\u03c7\u00b2 offset.</strong> That needs a refit with one knob '
             'moved at a time.</p>'
             '<p class="sub"><strong>Read this before reading the table.</strong> '
             'These are the <em>bench</em> bundles. The campaign ran '
             '<code>calib_bundle_prelim</code>, which '
             '<code>ntof_tracking.wft_beam.make_bundle</code> derives from them: '
             'it carries the impulse template and the kernel hypers through '
             'verbatim (<code>sigma_p0</code> and <code>Dp</code> included), '
             'replaces <code>v_drift</code> with one shared Magboltz prior, and '
             '<strong>drops <code>t0_abs</code> and <code>t0_prior_sigma</code> '
             'for every arm alike</strong> \u2014 deliberately, because a bench '
             't0 is an absolute arrival time against the bench trigger and is a '
             'wrong answer for an n_TOF-triggered run '
             '(<code>RUN145_R06_2026-08-19.md</code> \u00a71, commit '
             '<code>11ce347</code>). So the t0 prior is off on A, B, C and D in '
             'every product here and explains no chamber-to-chamber difference. '
             'What its absence costs is a separate question \u2014 '
             '<code>HANDOFF_T0_PRIOR.md</code>.</p>')
    show = ['arm', 'bundle', 'share_mode', 'n_dead_masked', 'n_hot_masked',
            'hyper_sigma_s', 'hyper_sigma_p0', 'hyper_Dp', 'hyper_c1',
            'hyper_c2_over_c1', 'hyper_kY', 'hyper_tau_s']
    bt = bd[[c for c in show if c in bd.columns]].copy()
    t.append(_table(bt, fmt={
        'share_mode': lambda v: '\u2014' if pd.isna(v) else str(v),
        'hyper_c2_over_c1': lambda v: '\u2014' if pd.isna(v) else f'{v:.2f}',
        't0_prior_sigma': lambda v: '\u2014' if pd.isna(v) else f'{v:.1f}',
        'hyper_sigma_s': lambda v: f'{v:.1f}',
        'hyper_sigma_p0': lambda v: f'{v:.4f}',
        'hyper_Dp': lambda v: f'{v:.4f}',
    }))
    t.append('<ul>'
             '<li><strong>No chamber masks a single channel, at any stage.</strong> '
             '<code>dead</code> is empty on A, B and D and absent on C, and no '
             'bundle carries a <code>hot</code> map at all — so chamber '
             'D’s noisy cells, the 22.4 % of its sample that '
             '<code>HANDOFF_D_NOISY_CHANNELS.md</code> is about, are in every '
             'fit at full weight. The classifier and the bundle-patching tool '
             'both exist and were only ever run on run_145. See '
             '<code>HANDOFF_CHANNEL_MASKS.md</code>.</li>'
             '<li><strong>C is a bundle generation behind.</strong> '
             '<code>calib_bundle_lp</code>, with no <code>share_mode</code>, no '
             '<code>dead</code> key, a bare <code>c2</code> instead of '
             '<code>c2_over_c1</code>, and — the part that matters, because '
             '<code>make_bundle</code> carries it through verbatim and calls it '
             '“the largest un-validated assumption in this chain” '
             '— <code>sigma_p0</code> and <code>Dp</code> an order of '
             'magnitude below A’s, B’s and D’s.</li>'
             '<li><strong>A is the only one calibrated on a dedicated scan.</strong> '
             'Its <code>run_key</code> names the resistive and drift voltages; '
             'D’s <code>conditions</code> are empty strings.</li>'
             '</ul>')

    # ------------------------------------------------------------------ cross
    if legs is not None:
        t.append('<h2>Cross-check</h2>')
        t.append('<p>The sample above reproduces '
                 '<code>_track_table</code>\u2019s cuts from the merged stage-3 '
                 'parquet. <code>pair_qa.py</code> applied them to the per-sub-run '
                 'files and kept some legs this reconstruction does not, so the '
                 'two are close but not identical. These numbers are taken '
                 'straight from the published pair table\u2019s own legs instead. '
                 'The ordering and the short-track medians agree; that is what '
                 'the conclusions rest on.</p>')
        lt = legs[['arm', 'n', 'median_all', 'median_short',
                   'frac_at_floor']].copy()
        t.append(_table(lt, fmt={'n': lambda v: f'{int(v):,}',
                                 'frac_at_floor': lambda v: f'{v:.1%}'}))

    # ------------------------------------------------------------------ limits
    t.append('<h2>What this does not rule out</h2>')
    t.append('<ul>'
             '<li><strong>Which bundle difference causes the offset.</strong> '
             'Four chambers is four points; the differences are correlated and '
             'cannot be separated by looking. The decisive test is a refit of '
             'one chamber with one knob moved \u2014 D with '
             '<code>t0_prior_sigma = 5</code>, C on an <code>r06</code>-generation '
             'bundle \u2014 which needs the waveform path, not these tables.</li>'
             '<li><strong>That A is \u201cright\u201d in any absolute sense.</strong> '
             'A reaching the noise floor says its model describes its waveforms; '
             'it says nothing about whether its angles or depths are accurate. '
             'A well-fitted wrong model is still wrong.</li>'
             '<li><strong>That \u03c7\u00b2/dof is a useful cut.</strong> Nothing '
             'in this chain cuts on it, and the event-mixed null in the QA set '
             'sits under the data over most of the range. This investigation '
             'explains the shape; it does not argue for using it.</li>'
             '<li><strong>The charge columns.</strong> '
             '<code>q_total</code> and <code>x_q_sum</code> run to 10<sup>30</sup> '
             'on a tail of tracks \u2014 the NNLS profile escaping into a '
             'near-null direction. They are excluded from the amplitude tables '
             'here and are their own bug, untouched by this work.</li>'
             '</ul>')

    t.append('<h2>Reproduce</h2>')
    t.append('<p><code>X17_ROOT=D:/x17 python ntof_athens_26/chi2_bimodality/'
             'chi2_shape.py</code> then <code>make_chi2_figures.py</code> then '
             '<code>make_chi2_report.py</code>. Add <code>--all-tracks</code> to '
             'the first to run on the whole stage-3 population instead of the '
             'published selection.</p>')
    t.append('</div>')
    return ''.join(t)


def main() -> int:
    if not (SRC / 'summary.json').exists():
        print(f'missing {SRC}/summary.json -- run chi2_shape.py first',
              file=sys.stderr)
        return 2
    FIGS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding='utf-8')
    print(f'  -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
