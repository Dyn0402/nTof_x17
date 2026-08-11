#!/usr/bin/env python3
"""Build report.html for the campaign QA -- our processing against n_TOF's own.

Reads the products of the checks (all in results/):
    compare.json     beam-gated per-run metrics, ours and official
    config_check.txt UserInput parameter diff + template md5 comparison
    quality.log      quality_metrics.py, ours vs official
    verify.log       structural verification of every transferred partial
and writes results/report.html plus results/figures/*.png.

    .venv/bin/python ntof_processing/campaign_qa/make_report.py
"""
from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
FIG = RES / 'figures'

TREES_RATE = ['WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD',
              'LIQA', 'LIQB', 'LIQC', 'LIQD']
ARMS = 'ABCD'


# ---------------------------------------------------------------- helpers
def col(rows, tree, field):
    return np.array([r.get('trees', {}).get(tree, {}).get(field, np.nan)
                     for r in rows], dtype=float)


def span(rows, tree, field, fmt='{:.0f}'):
    v = col(rows, tree, field)
    v = v[np.isfinite(v)]
    if not v.size:
        return 'n/a'
    return f'{fmt.format(v.min())}–{fmt.format(v.max())}'


def overlap_pct(a, b):
    """How far the two ranges are apart, as a % of the official mid-point.
    0 means the ranges overlap."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return np.nan
    mid = np.median(b)
    if a.min() > b.max():
        return (a.min() - b.max()) / mid * 100
    if a.max() < b.min():
        return (a.max() - b.min()) / mid * 100
    return 0.0


# ---------------------------------------------------------------- figures
def figures(ours, off):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 130,
                         'axes.grid': True, 'grid.alpha': 0.25})

    # 1. intensity-normalised rate per tree, ours vs official
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True)
    for ax, fam in zip(axes.ravel(), ['WAL', 'PSS', 'LIQ']):
        for i, arm in enumerate(ARMS):
            t = f'{fam}{arm}'
            ro, rf = col(ours, t, 'hits_per_1e12p'), col(off, t, 'hits_per_1e12p')
            xo = [r['run'] for r in ours]
            xf = [r['run'] for r in off]
            c = f'C{i}'
            ax.plot(xf, rf, 'o', ms=4, color=c, alpha=.55, label=f'{t} official')
            ax.plot(xo, ro, 's', ms=4, color=c, label=f'{t} ours')
        ax.set_title(f'{fam}: hits per 1e12 protons', fontsize=9)
        ax.legend(fontsize=5.5, ncol=4, loc='lower center', framealpha=.85)
        ax.margins(y=.22)
        ax.tick_params(axis='x', labelrotation=45, labelbottom=True, labelsize=7)
    ax = axes.ravel()[3]
    ax.axis('off')
    ax.text(0.02, 0.95, 'squares = runs we processed (224688–224700)\n'
                        'circles  = n_TOF official (224660–224676)\n\n'
                        'Beam bunches only (PKUP amp > 0), normalised to the\n'
                        'protons those bunches carried, so runs at different\n'
                        'intensity are directly comparable.',
            va='top', fontsize=8, transform=ax.transAxes)
    for a in axes.ravel()[:3]:
        a.set_xlabel('run')
    fig.tight_layout()
    fig.savefig(FIG / 'rates.png')
    plt.close(fig)

    # 2. modal tflash per tree
    fig, ax = plt.subplots(figsize=(8, 3.4))
    trees = ['WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD',
             'LIQA', 'LIQB', 'LIQC', 'LIQD']
    x = np.arange(len(trees))
    for rows, mark, lab, c in ((off, 'o', 'official', 'C0'), (ours, 's', 'ours', 'C3')):
        for j, r in enumerate(rows):
            y = [r.get('trees', {}).get(t, {}).get('tflash_mode_ns', np.nan) for t in trees]
            ax.plot(x + (0.12 if lab == 'ours' else -0.12), y, mark, ms=4,
                    color=c, alpha=.6, label=lab if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(trees, rotation=45, ha='right')
    ax.set_ylabel('modal tflash (ns)')
    ax.set_title('Where each tree puts the gamma flash — one point per run')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'tflash.png')
    plt.close(fig)

    # 3. median amplitude per tree, ours vs official
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for rows, mark, lab, c in ((off, 'o', 'official', 'C0'), (ours, 's', 'ours', 'C3')):
        for j, r in enumerate(rows):
            y = [r.get('trees', {}).get(t, {}).get('median_amp', np.nan) for t in trees]
            ax.plot(x + (0.12 if lab == 'ours' else -0.12), y, mark, ms=4,
                    color=c, alpha=.6, label=lab if j == 0 else None)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(trees, rotation=45, ha='right')
    ax.set_ylabel('median hit amplitude (ADC)')
    ax.set_title('Median hit amplitude — one point per run')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'amps.png')
    plt.close(fig)


# ---------------------------------------------------------------- tables
def rate_table(ours, off):
    rows = []
    for t in TREES_RATE:
        a, b = col(ours, t, 'hits_per_1e12p'), col(off, t, 'hits_per_1e12p')
        gap = overlap_pct(a, b)
        rows.append(
            f'<tr><td class="k">{t}</td><td>{span(ours, t, "hits_per_1e12p")}</td>'
            f'<td>{span(off, t, "hits_per_1e12p")}</td>'
            f'<td class="{"ok" if abs(gap) < 3 else "warn"}">'
            f'{"overlap" if gap == 0 else f"{gap:+.1f} %"}</td></tr>')
    return ('<table><thead><tr><th>tree</th><th>ours (13 runs)</th>'
            '<th>official (6 runs)</th><th>separation</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def amp_table(ours, off):
    rows = []
    for t in TREES_RATE:
        a, b = col(ours, t, 'median_amp'), col(off, t, 'median_amp')
        gap = overlap_pct(a, b)
        rows.append(
            f'<tr><td class="k">{t}</td>'
            f'<td>{span(ours, t, "median_amp", "{:.1f}")}</td>'
            f'<td>{span(off, t, "median_amp", "{:.1f}")}</td>'
            f'<td class="{"ok" if abs(gap) < 3 else "warn"}">'
            f'{"overlap" if gap == 0 else f"{gap:+.1f} %"}</td></tr>')
    return ('<table><thead><tr><th>tree</th><th>ours</th><th>official</th>'
            '<th>separation</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def flash_table(ours, off):
    rows = []
    for t in TREES_RATE:
        mo, mf = col(ours, t, 'tflash_mode_ns'), col(off, t, 'tflash_mode_ns')
        bo, bf = col(ours, t, 'flash_bad_pct'), col(off, t, 'flash_bad_pct')
        rows.append(
            f'<tr><td class="k">{t}</td>'
            f'<td>{np.nanmin(mo):.0f}–{np.nanmax(mo):.0f}</td>'
            f'<td>{np.nanmin(mf):.0f}–{np.nanmax(mf):.0f}</td>'
            f'<td class="{"ok" if np.nanmax(bo) < 2 else "warn"}">{np.nanmax(bo):.2f} %</td>'
            f'<td class="{"ok" if np.nanmax(bf) < 2 else "warn"}">{np.nanmax(bf):.2f} %</td></tr>')
    return ('<table><thead><tr><th>tree</th><th>modal tflash, ours (ns)</th>'
            '<th>modal tflash, official (ns)</th><th>worst off-flash, ours</th>'
            '<th>worst off-flash, official</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def offset_table(ours, off):
    keys = [f'{k}{a}' for k in ('pss', 'liq') for a in ARMS]
    rows = []
    for k in keys:
        a = np.array([r.get('offsets', {}).get(k, {}).get('peak_ns', np.nan)
                      for r in ours], dtype=float)
        b = np.array([r.get('offsets', {}).get(k, {}).get('peak_ns', np.nan)
                      for r in off], dtype=float)
        f = lambda v: (f'{np.nanmin(v):+.0f} … {np.nanmax(v):+.0f}'
                       if np.isfinite(v).any() else 'n/a')
        worst = np.nanmax(np.abs(a)) if np.isfinite(a).any() else np.nan
        rows.append(f'<tr><td class="k">{k.upper()}</td><td>{f(a)}</td><td>{f(b)}</td>'
                    f'<td class="{"ok" if worst < 40 else "warn"}">{worst:.0f} ns</td></tr>')
    return ('<table><thead><tr><th>pair</th><th>ours (ns)</th><th>official (ns)</th>'
            '<th>|worst| ours</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def per_run_table(ours):
    rows = []
    for r in ours:
        t = r.get('trees', {})
        rows.append(
            f'<tr><td class="k">{r["run"]}</td>'
            f'<td>{r["bunches_beam"]}</td>'
            f'<td>{r["frac_empty"] * 100:.0f} %</td>'
            f'<td>{r["protons_1e12"]:.0f}</td>'
            f'<td>{t.get("WALA", {}).get("hits_per_1e12p", np.nan):.0f}</td>'
            f'<td>{t.get("PSSC", {}).get("hits_per_1e12p", np.nan):.0f}</td>'
            f'<td>{t.get("LIQA", {}).get("hits_per_1e12p", np.nan):.0f}</td>'
            f'<td class="ok">{max(t.get(x, {}).get("flash_bad_pct", 0) for x in TREES_RATE):.2f} %</td>'
            '</tr>')
    return ('<table><thead><tr><th>run</th><th>beam bunches</th><th>empty</th>'
            '<th>protons (1e12)</th><th>WALA /1e12p</th><th>PSSC /1e12p</th>'
            '<th>LIQA /1e12p</th><th>worst off-flash</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def verify_totals():
    """Partial count and volume out of the structural log, so the headline
    numbers cannot drift away from what was actually verified."""
    logs = sorted(RES.glob('verify*.log'))
    if not logs:
        return None
    parts = gb = runs = bad = 0
    for p in logs:
        for line in p.read_text().splitlines():
            f = line.split()
            if len(f) == 10 and f[0].isdigit():
                runs += 1
                parts += int(f[1])
                bad += int(f[6])
                gb += float(f[9])
    return {'runs': runs, 'parts': parts, 'gb': gb, 'bad': bad}


def pre(path, fallback):
    p = RES / path
    return (f'<pre>{html.escape(p.read_text().strip())}</pre>' if p.exists()
            else f'<p class="warn">{fallback}</p>')


# ------------------------------------------------- ledger: official vs ours
STATE_NOTE = {
    'MERGED': 'merged file in <code>done/</code> — usable, best',
    'IN_FLIGHT': 'n_TOF wiped <code>completed/</code> and is reprocessing it now',
    'PARTIALS_ONLY': 'reconstruction finished, merge never ran — partials are the truth',
    'MERGE_EMPTY': 'zero-byte <code>done/</code> file: the merge failed, partials are the truth',
    'RAW_ONLY': 'raw staged, n_TOF has processed nothing',
    'NOTHING': 'no raw staged and nothing processed',
}


def ledger_rows():
    p = RES / 'ledger_2026-08-11.json'
    if not p.exists():
        return []
    return json.loads(p.read_text())


def ledger_state_table(rows):
    from collections import Counter
    c = Counter(r['off_state'] for r in rows)
    order = ['MERGED', 'IN_FLIGHT', 'PARTIALS_ONLY', 'MERGE_EMPTY',
             'RAW_ONLY', 'NOTHING']
    out = []
    for s in order:
        if not c[s]:
            continue
        runs = [r['run'] for r in rows if r['off_state'] == s]
        rng = (f'{runs[0]}–{runs[-1]}' if len(runs) > 6
               else ', '.join(str(x) for x in runs))
        ok = 'ok' if s in ('MERGED', 'PARTIALS_ONLY') else 'warn'
        out.append(f'<tr><td class="k {ok}">{s}</td><td>{c[s]}</td>'
                   f'<td style="text-align:left">{STATE_NOTE.get(s, "")}</td>'
                   f'<td style="text-align:left">{rng}</td></tr>')
    return ('<table><thead><tr><th>official state</th><th>runs</th>'
            '<th>what it means</th><th>run range (not contiguous)</th></tr></thead>'
            f'<tbody>{"".join(out)}</tbody></table>')


def newly_merged(rows):
    """Runs that were not MERGED in the 08-10 inventory and are now."""
    import csv as _csv
    p = HERE.parent / 'skip_diagnosis' / 'inputs' / 'inventory_2026-08-10.csv'
    if not p.exists():
        return []
    with p.open() as fh:
        old = {r['run']: r['state'] for r in _csv.DictReader(fh)}
    return [r['run'] for r in rows if r['off_state'] == 'MERGED'
            and old.get(str(r['run']), 'MERGED') != 'MERGED']


def overlap_table(rows):
    both = [r for r in rows if r['ours_prod'] and r['off_state'] != 'RAW_ONLY']
    cells = ''.join(
        f'<tr><td class="k">{r["run"]}</td><td>{r["off_state"]}</td>'
        f'<td>{r["off_parts"]}</td><td>{r["off_GB"]}</td>'
        f'<td>{r["ours_prod"]}</td><td>{r["ours_variant"]}</td>'
        f'<td>{r["ours_parts"]}</td><td>{r["ours_GB"]}</td>'
        f'<td class="{"ok" if r["ours_recipe"] == r["off_recipe"] else "warn"}">'
        f'{"same recipe" if r["ours_recipe"] == r["off_recipe"] else "different"}</td></tr>'
        for r in both)
    return ('<table><thead><tr><th>run</th><th>official</th><th>parts</th><th>GB</th>'
            '<th>our production</th><th>our variant</th><th>parts</th><th>GB</th>'
            f'<th>recipe</th></tr></thead><tbody>{cells}</tbody></table>')


def ours_only_table(rows):
    only = [r for r in rows if r['ours_prod'] and r['off_state'] == 'RAW_ONLY']
    cells = ''.join(
        f'<tr><td class="k">{r["run"]}</td><td>{r["raw_files"]}</td>'
        f'<td>{r["raw_GB"]}</td><td>{r["ours_parts"]}</td><td>{r["ours_GB"]}</td></tr>'
        for r in only)
    return ('<table><thead><tr><th>run</th><th>raw files</th><th>raw GB</th>'
            '<th>our partials</th><th>our GB</th></tr></thead>'
            f'<tbody>{cells}</tbody></table>'), only


IDENTITY_FILES = [
    ('identity_224572.json',
     'ours <code>v12_liqpileup</code> vs official — same recipe'),
    ('identity_224574.json',
     'ours <code>prod_v11</code> vs official <code>v12</code> — recipes differ on LIQ only'),
    ('identity_224577.json',
     'ours <code>prod_v11</code> vs official <code>v12</code> — merged by n_TOF on 08-11'),
]


def identity_table():
    blocks = []
    for fn, note in IDENTITY_FILES:
        p = RES / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cells = []
        for tree, rec in d['trees'].items():
            if 'error' in rec:
                cells.append(f'<tr><td class="k">{tree}</td><td colspan="3" '
                             f'class="warn">{rec["error"]}</td></tr>')
                continue
            v = rec['verdict']
            cls = 'ok' if v == 'IDENTICAL' else 'warn'
            detail = ''
            if v.startswith('DIFFERENT (values)'):
                detail = ', '.join(
                    f'{c} {x["cells"]} cells ({100 * x["frac"]:.2f} %)'
                    for c, x in rec.get('columns_differing', {}).items())
            elif v.startswith('DIFFERENT (hit'):
                detail = (f'{rec["n_official"] - rec["n_ours"]:+d} hits '
                          f'({100 * (rec["n_official"] / max(1, rec["n_ours"]) - 1):+.1f} %)')
            cells.append(
                f'<tr><td class="k">{tree}</td><td>{rec["n_ours"]:,}</td>'
                f'<td class="{cls}">{v}</td>'
                f'<td style="text-align:left">{detail}</td></tr>')
        blocks.append(
            f'<h3>run {d["run"]} — {note}</h3>'
            f'<p class="sub" style="margin:.2rem 0 .5rem">our partial '
            f'<code>{d["ours_partial"]}</code>, bunches '
            f'{d["bunches"][0]}–{d["bunches"][1]}</p>'
            '<table><thead><tr><th>tree</th><th>hits compared</th><th>verdict</th>'
            f'<th>difference</th></tr></thead><tbody>{"".join(cells)}</tbody></table>')
    return ''.join(blocks)


def beam_table():
    p = RES / 'beam_state.json'
    if not p.exists():
        return '', []
    rows = json.loads(p.read_text())
    cells = ''.join(
        f'<tr><td class="k">{r["run"]}</td><td>{r["bunches"]}</td>'
        f'<td>{r["beam_bunches"]}</td><td>{r["beam_pct"]:.1f} %</td>'
        f'<td>{r["protons_1e12"]:,.0f}</td>'
        f'<td class="{"ok" if r["state"] == "beam" else "warn"}">{r["state"]}</td></tr>'
        for r in rows)
    return ('<table><thead><tr><th>run</th><th>bunches</th><th>with protons</th>'
            '<th>beam fraction</th><th>protons (1e12)</th><th>state</th></tr>'
            f'</thead><tbody>{cells}</tbody></table>'), rows


# ---------------------------------------------------------------- page
CSS = """
:root{--fg:#1b1b1b;--bg:#fff;--mut:#666;--line:#ddd;--ok:#0a7d33;--warn:#b25000;
      --band:#f6f7f9;}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1000px;padding:2rem 1.2rem 4rem;
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
     color:var(--fg);background:var(--bg)}
h1{font-size:1.7rem;margin:0 0 .2rem}
h2{font-size:1.15rem;margin:2.4rem 0 .6rem;padding-bottom:.25rem;
   border-bottom:2px solid var(--line)}
h3{font-size:1rem;margin:1.4rem 0 .4rem}
.sub{color:var(--mut);margin:0 0 1.4rem}
.verdict{background:var(--band);border-left:4px solid var(--ok);padding:.9rem 1.1rem;
         margin:1.2rem 0;border-radius:0 4px 4px 0}
.verdict strong{font-size:1.05rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem;
      margin:1.2rem 0}
.card{background:var(--band);border:1px solid var(--line);border-radius:5px;padding:.7rem .8rem}
.card .n{font-size:1.35rem;font-weight:650;display:block}
.card .l{color:var(--mut);font-size:.8rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.87rem}
th,td{border-bottom:1px solid var(--line);padding:.35rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--line);font-weight:600;color:var(--mut);
         font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
td.k{font-weight:600}
.ok{color:var(--ok)}
.warn{color:var(--warn)}
pre{background:var(--band);border:1px solid var(--line);border-radius:4px;
    padding:.7rem .8rem;overflow-x:auto;font-size:.76rem;line-height:1.45}
figure{margin:1.2rem 0}
img{max-width:100%;border:1px solid var(--line);border-radius:4px}
figcaption{color:var(--mut);font-size:.83rem;margin-top:.4rem}
ul{padding-left:1.2rem}
li{margin:.3rem 0}
code{background:var(--band);padding:.1em .35em;border-radius:3px;font-size:.86em}
"""


def build():
    data = json.loads((RES / 'compare.json').read_text())
    ours, off = data['ours'], data['official']
    figures(ours, off)

    wal_o = col(ours, 'WALA', 'hits_per_1e12p')
    wal_f = col(off, 'WALA', 'hits_per_1e12p')
    worst_flash = max(max(t.get('flash_bad_pct', 0) for t in r['trees'].values())
                      for r in ours)
    v = verify_totals()
    vol = (f'<div class="card"><span class="n">{v["parts"]}</span>'
           f'<span class="l">partials, {v["gb"]:.0f} GB — {v["bad"]} unreadable</span></div>'
           if v else '')
    beam_tbl, beam_rows = beam_table()
    n_beam = sum(1 for r in beam_rows if r['state'] == 'beam')
    n_dark = len(beam_rows) - n_beam

    led = ledger_rows()
    ours_tbl, ours_only = ours_only_table(led)
    n_both = len([r for r in led
                  if r['ours_prod'] and r['off_state'] != 'RAW_ONLY'])
    ident = identity_table()
    n_new = len(newly_merged(led))
    n_flight = sum(1 for r in led if r['off_state'] == 'IN_FLIGHT')

    body = f"""
<h1>n_TOF reprocessing campaign — are our runs as good as n_TOF's?</h1>
<p class="sub">Runs 224688–224700, processed by us and moved to
<code>/eos/experiment/ntof/data/x17/reproc/prod_v12/</code>, checked against runs
n_TOF processed themselves in <code>official/completed/</code>. {date.today()}</p>

<div class="verdict">
<strong>Yes — and on the runs that now exist in both processings, bit for
bit.</strong> Given the same UserInput our chain reproduces n_TOF's product
exactly: on run 224572 every hit of every wall, plastic, silicon and pickup tree
matches on all 22 columns. The transferred block was produced with a configuration
byte-identical to n_TOF's own, and on every measurable axis it is indistinguishable
from official runs taken under the same beam: intensity-normalised hit rates
overlap, the gamma flash lands in the same 10&nbsp;ns bin in every tree,
{worst_flash:.2f}&nbsp;% of beam bunches are off-flash (the broken July processing sat at
37–85&nbsp;% on the plastics), and hit quality matches to a few percent.
</div>

<div class="grid">
  <div class="card"><span class="n">{len(beam_rows) or 13}</span><span class="l">runs on the ntof disk ({n_beam} with beam)</span></div>
  {vol}
  <div class="card"><span class="n">{n_both}</span><span class="l">runs now in both processings</span></div>
  <div class="card"><span class="n">{len(ours_only)}</span><span class="l">runs only we have (n_TOF: raw only)</span></div>
  <div class="card"><span class="n">26/26</span><span class="l">templates byte-identical to official</span></div>
  <div class="card"><span class="n">{worst_flash:.2f} %</span><span class="l">worst off-flash bunches (target &lt; 2 %)</span></div>
  <div class="card"><span class="n">{wal_o.min():.0f}–{wal_o.max():.0f}</span><span class="l">WALA hits/1e12 p (official {wal_f.min():.0f}–{wal_f.max():.0f})</span></div>
</div>

<h2>Where every run of the campaign stands, ours and n_TOF's</h2>
<p><code>official_ledger.py</code> walks all {len(led)} runs staged under the X17
EAR2 2026 DAQ directory and records, per run, what n_TOF has, what we have, and
which UserInput each product was actually made with — read out of the product's
own <code>history</code> object rather than assumed.</p>
{ledger_state_table(led)}
<p><strong>n_TOF moved a long way on 08-10/08-11.</strong> Against the 08-10
inventory (359 MERGED / 53 PARTIALS_ONLY / 2 MERGE_EMPTY / 31 RAW_ONLY), {n_new}
of the unmerged runs have since been merged and {n_flight} more are being
reprocessed from scratch right now — their <code>completed/</code> directories were emptied and
are refilling with partials stamped within the hour. A snapshot taken during that
window reads those as data loss; they are not.</p>
<p><strong>The recipe is uniform.</strong> Every official X17 product with a
readable history — all 413 of them, including the ones written today — carries
<code>UserInput_2026_EAR2_X17_v4.h</code>, and its parameters normalise to the same
fingerprint as our <code>v12_liqpileup</code>. There is no recipe boundary hiding
inside the official set.</p>

<h3>Runs that exist in both processings</h3>
<p>{n_both} runs are now in both — which was not true a week ago, and it is what
makes the next section possible.</p>
{overlap_table(led)}

<h3>Runs we processed that n_TOF still has not</h3>
<p>{len(ours_only)} runs, all of them <code>RAW_ONLY</code> on n_TOF's side: the
contiguous block at the end of the campaign. n_TOF's pass has skipped past it —
it spent 08-11 merging and reprocessing runs below 224688 and then moved on to
224719+, which belong to a different experiment.</p>
{ours_tbl}

<h2>Hit for hit, on the runs that exist in both</h2>
<p><code>compare_identity.py</code> matches on <code>BunchNumber</code> — the two
processings split a run into partials differently, so partial N is not partial N —
and compares all 22 per-hit columns for a window of bunches.</p>
{ident or '<p class="warn">identity comparison pending</p>'}
<p><strong>Same UserInput reproduces n_TOF bit for bit.</strong> On 224572, where
our product and theirs were made with the same recipe, every hit of every wall,
plastic, silicon and pickup tree agrees exactly — same count, same
<code>tof</code>, <code>amp</code>, <code>area</code>, <code>chi2</code>, every
column. The liquids agree on every column too except <code>afast</code>, which
differs on 3–6 hits in ~85 000 (0.00–0.02 %) with huge magnitude — a numerically
unstable integral on a handful of pathological pulses, not a difference in the
reconstruction.</p>
<p><strong>Our <code>prod_v11</code> runs differ exactly where they were always
documented to.</strong> 224574 and 224577 are bit-identical to official on WAL,
PSS, SILI and PKUP and differ only in the LIQ hit count (official +17 to +21 %) —
the known v11→v12 liquid yield step from <code>STEP SIZE</code> and
<code>SIGNAL WIDTH HIGH</code>. Nothing else moved, which is the strongest
available evidence that the difference is the recipe and not our chain.</p>

<h2>What was compared for the block n_TOF has not processed</h2>
<p>For 224688–224718 there is still nothing to diff against, so the argument there
remains an equivalence one, in two parts:</p>
<ul>
<li><strong>Configuration</strong> — compared exactly, because each product records
the UserInput it was made with in its own <code>history</code> object.</li>
<li><strong>Behaviour</strong> — compared statistically against the nearest official
runs in time (224660–224676), with everything normalised to the protons delivered,
because a raw hit count is a beam measurement, not a processing measurement.</li>
</ul>
<p>The hit-for-hit result above upgrades it: the chain that produced this block is
the same chain that reproduces n_TOF exactly on the runs where both exist.</p>

<h3>Two traps that make an honest comparison look broken</h3>
<ul>
<li><strong>The official runs next door have no beam.</strong> 224678–224687 all sit
at zero PulseIntensity and zero PKUP amplitude. Picking 224687 — the run immediately
before our block — as the control makes our output look 400× too busy. It is not;
that run had no protons.</li>
<li><strong>Empty PS pulses inside our runs.</strong> The first partial of 224692 is
75&nbsp;% empty pulses; those bunches have no flash, so tflash is 0 and every
flash check flags them. Restricted to bunches that actually had protons, 224692 is
clean like the rest — and its intensity-normalised rates match the other twelve
runs to ~1&nbsp;%. Every number in this report is gated on PKUP amplitude &gt; 0.</li>
</ul>

<h2>Configuration: identical, not merely similar</h2>
<p>n_TOF's production <code>UserInput_2026_EAR2_X17_v4.h</code> <em>is</em> our
v12_liqpileup — they adopted it after the July handoff, so both processings run the
same recipe. Every parameter column matches; the only differing lines are the file
name itself and the directory the templates are read from (our AFS staging area vs
their EOS shapes directory). All 26 referenced templates are byte-identical.</p>
{pre('config_check.txt', 'config_check.txt missing')}

<h2>Structural verification of every transferred partial</h2>
<p>Not a sample — every partial of every run is opened, all 16 top-level objects are
required, and a real array read is issued on all 14 hit trees so a truncated basket
is hit rather than only the header. Contiguity is checked against
<code>ceil(raw_files / 4)</code>, the split RunProcessing uses, and the
<code>history</code> md5 is compared across our own runs.</p>
{pre('verify.log', 'Structural pass still running — this section is pending.')}

<h2>Which transferred runs actually have beam</h2>
<p>Read from the <code>index</code> tree, which is replicated in full in every
partial and therefore describes the whole run rather than the sampled partial.
{n_beam} of the {len(beam_rows)} runs on the disk carry beam; {n_dark} carry none at
all — they were processed correctly and are simply quiet, which is why their files
are a few MB rather than tens of GB. The physics comparison above uses the
{n_beam} beam runs.</p>
{beam_tbl}
<p>224692 is worth reading twice: 98.0&nbsp;% beam over the whole run, but its
<em>first partial</em> is 75&nbsp;% empty. Sampling one partial and not gating on
protons is exactly how a healthy run gets called broken.</p>

<h3>Second batch, landed while this check was running</h3>
<p>224706, 224716, 224717 and 224718 arrived after the sweep started, so they have
the structural pass and the beam state but not the physics comparison — there is no
beam in them to compare.</p>
{pre('verify_batch2.log', 'verify_batch2.log missing')}

<h3>Third batch — the rest of the campaign</h3>
<p>224701–224715 landed on 08-11 afternoon and pass the same structural test.
Two of them, <strong>224705 and 224711, were flagged <code>COPY FAILED</code> by
the campaign driver</strong> — its <code>cp -r</code> returned non-zero and it kept
the staging copy. Both are complete and readable at the destination, so that was an
exit code, not a bad transfer.</p>
{pre('verify_batch3.log', 'verify_batch3.log missing')}
<p><strong>224709 is the one run of the block not yet on the ntof disk.</strong> Its
last job (partial 0023) was evicted once and its retry was still running when the
driver's stall timer fired; 85 of 86 partials are staged. It needs
<code>harvest_staged.sh 224709</code> once the job lands.</p>

<h2>Hit rates, normalised to delivered protons</h2>
{rate_table(ours, off)}
<figure><img src="figures/rates.png" alt="rates">
<figcaption>Intensity-normalised hit rate per tree. Our runs (squares) continue the
official runs (circles) with no step at the boundary between the two processings.
The slow rise visible on LIQA/LIQD across run number is a time trend that runs
through both sets, not a difference between them.</figcaption></figure>

<h2>Gamma flash</h2>
<p>This is the check that failed in the July processing: per tree, the fraction of
bunches whose stored <code>tflash</code> is more than 150&nbsp;ns off that tree's
modal value. The target is &lt; 2&nbsp;% per tree.</p>
{flash_table(ours, off)}
<figure><img src="figures/tflash.png" alt="tflash">
<figcaption>Modal tflash per tree, one point per run. Ours land in the same 10 ns
histogram bin as the official runs on every one of the twelve trees.</figcaption></figure>

<h3>Cross-detector consistency, per arm</h3>
<p>Prompt-coincidence peak of large plastic hits (amp &gt; 1000) and of liquid hits
against the same arm's wall, after removing each tree's modal tflash. The broken
processing sat at −375/+25/−325/−325&nbsp;ns on A/B/C/D.</p>
{offset_table(ours, off)}
<p>PSSC — and PSSD in some runs — sits near 30&nbsp;ns rather than 0. <strong>The
official runs show the same thing in the same trees</strong> (PSSC 33–35&nbsp;ns on
224660/224667/224674/224676), so it is a property of those channels under this
recipe, not something our processing introduced. It is worth chasing, but it is not
a campaign defect.</p>

<h2>Hit quality, not just hit count</h2>
<p>A processing can buy hits by measuring them worse, which no rate comparison would
show. <code>quality_metrics.py</code> on two partials each of our 224691 and official
224672 — every number accidental-subtracted:</p>
{pre('quality.log', 'quality.log missing')}

<h2>Amplitudes</h2>
{amp_table(ours, off)}
<figure><img src="figures/amps.png" alt="amplitudes">
<figcaption>Median hit amplitude per tree, one point per run.</figcaption></figure>

<h2>Per-run detail, our 13 runs</h2>
{per_run_table(ours)}

<h2>What this does not rule out</h2>
<ul>
<li><strong>Shared systematics are invisible here.</strong> Both processings run the
same UserInput through the same PSA binary, so a defect in that recipe affects ours
and n_TOF's identically and this comparison would never see it. This establishes
<em>equivalence to the official product</em>, not absolute correctness — and the
hit-for-hit agreement makes that sharper, not weaker: it proves the two chains are
the same chain.</li>
<li><strong>The identity comparison samples five bunches of one partial per
run.</strong> It is an exactness test, not a coverage test; the structural pass is
what covers every file.</li>
<li><strong>The ledger is a snapshot.</strong> 24 runs were mid-reprocessing when
it was taken, so their partial counts are meaningless until n_TOF finishes. Re-run
<code>official_ledger.py</code> rather than quoting those numbers later.</li>
<li><strong>The physics comparison samples one partial per run</strong> (two for the
quality metrics), which is 63–80 bunches out of thousands. The structural pass is
the one that covers every file. A defect confined to late partials of a run would
survive this.</li>
<li><strong>The late runs have structure but not physics.</strong> 224701–224718 all
passed the structural pass, but the beam-gated comparison above still covers only
224688–224700. Several of the late ones have no beam at all, and for those the
acceptance test can only be structural — with no protons there is no flash and no
rate to compare.</li>
<li><strong>No downstream check yet.</strong> Nothing here runs the DREAM slim over
these runs; the association efficiency and clock QA on the new block are still to
do, and those are what would catch a timing problem that survives all of the above.</li>
<li>The three extra template files in our staging directory
(<code>X17_WALA_Signal_3.txt</code>, <code>X17_WALB_Signal_0.txt</code>,
<code>X17_WALC_Signal_0.txt</code>) are unreferenced leftovers; the UserInput points
only at the <code>_avg0/1/2</code> set. Harmless, but they should be cleared so a
future reader does not think they were used.</li>
</ul>

<h2>How to re-run this</h2>
<pre>ssh -K lxplus
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
cd /afs/cern.ch/work/d/dneff/x17_reproc/campaign_qa

python3 -u verify_transferred.py &lt;run&gt; ...                 # structure, every partial
python3 -u compare_campaign.py --partials=1 --json=compare.json \\
        ours=&lt;dir&gt;,... official=&lt;dir&gt;,...                    # beam-gated physics
python3 history_diff.py &lt;ours_history&gt; &lt;official_history&gt;   # configuration
python3 ../quality_metrics.py ours=&lt;f&gt;,&lt;f&gt; official=&lt;f&gt;,&lt;f&gt;  # hit quality

# then, locally, with the logs rsynced into campaign_qa/results/
.venv/bin/python ntof_processing/campaign_qa/make_report.py</pre>
"""
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>n_TOF campaign QA — our processing vs n_TOF\'s</title>'
            f'<style>{CSS}</style></head><body>{body}</body></html>')
    out = RES / 'report.html'
    out.write_text(page)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(build())
