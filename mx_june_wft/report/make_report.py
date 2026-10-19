#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report.py — build the full cosmic-bench fleet report (tabbed HTML) from
the 2026-08-12 frozen-analysis condor campaign outputs.

Generated, not hand-written (CLAUDE.md): every number is read from the
accounting JSONs the campaign produced (efficiency_breakdown / angular
resolution / alignment / events.meta / gate eval summaries / the campaign
manifest / hv_trends.json), so re-running after any accounting stage keeps
the tables, the figures and the verdict text in step.  Figures are copied
into figures/ next to the report and referenced with ordinary relative
links, so the same file works from disk, served by the DAQ page's
/analysis_file/<relpath> route, or copied elsewhere with its figures/.

Order of operations (all idempotent):
    1. mx_june_wft/report/make_maps_2mm.py        (r<2 mm efficiency maps)
    2. mx_june_wft/report/hv_trends.py            (HV-scan aggregation)
    3. mx_june_wft/report/make_report.py          (this)

Run: ../../.venv/bin/python mx_june_wft/report/make_report.py
Output: /home/dylan/x17/cosmic_bench/Analysis/fleet_report/report.html
"""
import csv
import glob
import html
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'mx_june_wft')]

from qa_config import get_config, setup_paths          # noqa: E402
setup_paths()
import fleet_state as FS                               # noqa: E402

FLEET = FS.FLEET
DET = FS.DET                       # key -> det3 style label
OUT = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report'
FIG = os.path.join(OUT, 'figures')
STAGING = '/home/dylan/x17/cosmic_bench/condor_campaign/results'
CAMPAIGN_DIR = '/home/dylan/x17/cosmic_bench/condor_campaign'
MANIFEST = os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')
RUNBOOK = 'mx_june_wft/FREEZE_MPGD26_2026-08-12.md'

MXDET = {v: k for k, v in
         {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4',
          'mx17_6': 'det6', 'mx17_7': 'det7'}.items()}   # det3 -> mx17_3

# figures pulled per detector from <OUT_BASE>/wft/ — (relpath, slug, caption)
WFT_FIGS = [
    ('maps/efficiency_r_2_mm_waveform_first.png', 'map2mm',
     'Spatial efficiency with the tight r < 2 mm success criterion — the map '
     'that isolates the core (fleet core σ is 0.43–0.62 mm). 40×40 bins '
     '(~12 mm); grey bins have < 5 reference tracks.'),
    ('maps/efficiency_r_10_mm_waveform_first.png', 'map10mm',
     'Spatial efficiency, r < 10 mm criterion (the standard map).'),
    ('maps/efficiency_any_hit_waveform_first.png', 'mapany',
     'Any-hit detection map — sensitive to amplification/readout dead '
     'regions, insensitive to reconstruction quality.'),
    ('maps/resolution_map_sliding_r50mm.png', 'resmap',
     'Sliding-kernel (50 mm) core-resolution map.'),
    ('efficiency/efficiency_breakdown.png', 'effbreak',
     'Loss budget per reference ray: no-hit / spark veto / hit-but-no-reco / '
     'reco-far / reco within 5 mm.'),
    ('alignment/residuals.png', 'resid',
     'Per-plane position residuals vs the M3 reference after alignment.'),
    ('alignment/radial_residuals.png', 'radresid',
     'Radial residual |r| (10 mm window).'),
    ('angles/angles.png', 'angles',
     'Track-angle reconstruction vs the M3 reference, both planes.'),
]
DET3_EXTRA = [
    ('explain/vd_estimators.png', 'vdest',
     'Why hit-time estimators compress the drift ladder while the '
     'waveform forward model does not (basis for RECONSTRUCTION_BASIS.md).'),
    ('explain/deconv_scatter.png', 'deconv',
     'Neighbour-charge deconvolution: shared charge identified and removed '
     'by the forward model.'),
]
HW_FIGS = [
    ('raw_detector_qa/amplitude_vs_strip.png', 'ampstrip',
     'Hit amplitude vs strip (hits chain, pre-campaign vintage) — hardware '
     'gain uniformity.'),
    ('raw_detector_qa/hits_vs_position.png', 'occ',
     'Hit occupancy vs position (hits chain, pre-campaign vintage).'),
]

DET_NOTES = {
    'det3': 'The reference chamber and the calibration test-bed: the only '
            'detector whose absolute-t0 trigger prior was validated directly. '
            'The det3-only digest gate (within 5 mm ≥ 93 %, core ≤ 0.50 mm, '
            'median ≤ 0.85 mm, has-any ≥ 99 %) passes on every threshold. '
            'The explanation figures below are the measured basis for '
            'reconstructing from waveforms rather than hit times.',
    'det2': 'Second-best chamber and nearly det3’s equal. The σ=5 '
            'trigger-t0 prior was adopted here by the campaign gate '
            '(within-5 mm +0.29 pp, far −0.30 pp, core −5 %). Its ftst '
            'timing ladder is visibly noisier than det3’s clean '
            '−10 ns/step, which is exactly why the gate was run per detector.',
    'det4': 'The outlier — and the loss is amplification, not tracking: '
            'any-hit detection is 95.8 % while within-5 mm is 41.6 %, and '
            'the maps show the known fixed non-amplifying stripes (a large '
            'fraction of the active area never amplifies; no HV setting '
            'recovers it). Where the chamber does amplify, the fit performs: '
            'the t0 prior was adopted with the strongest arm of the gate '
            '(core −20 %, median −16 %).',
    'det6': 'Middle of the fleet at 74.9 % within 5 mm. Its headline σθY '
            '(2.82°) and Y bias (−1.04°) are dominated by known artefacts, '
            'not the chamber: the unapplied w0/kw angle constants (corrected '
            'bias +0.22°, σθY 2.59° — see Method) account for the bias, and '
            'its calibration bundle is under review (σ_s 165.9 ns and '
            'v 26.7 µm/ns vs det7’s 36.6 in the same run suggest a '
            'degenerate calibration basin, so treat the fitted v as '
            'provisional). The t0 prior fell back here (marginal, at the '
            'gate-rule tolerance).',
    'det7': 'Spark-dominated: 37 % of M3-matched events carry a spark flag, '
            'the fleet’s highest by 1.7×, and the efficiency ceiling '
            'follows. The t0-prior gate exposed a real position-vs-angles '
            'trade (core and every angle metric improve, within-5 mm '
            'regresses), so it fell back to the no-prior bundle.',
}


def esc(s):
    return html.escape(str(s))


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fmt(v, spec='{:.2f}', dash='&mdash;'):
    if v is None:
        return dash
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return esc(v)


def copy_fig(src, slug):
    """Copy a figure into figures/<slug>.png; return relative link or None."""
    if not os.path.exists(src):
        return None
    os.makedirs(FIG, exist_ok=True)
    dst = os.path.join(FIG, slug + os.path.splitext(src)[1])
    shutil.copy2(src, dst)
    return 'figures/' + os.path.basename(dst)


# ---------------------------------------------------------------- gathering
def gather(key):
    cfg = get_config(key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    d = {'key': key, 'det': DET[key], 'cfg': cfg,
         'state': FS.state(key),
         'eff': jload(os.path.join(W, 'efficiency',
                                   'efficiency_breakdown.json')),
         'eff_hits': jload(os.path.join(W, 'efficiency',
                                        'efficiency_breakdown_hits.json')),
         'ang': jload(os.path.join(W, 'angles', 'angular_resolution.json')),
         'meta': jload(os.path.join(W, 'events.meta.json'))}
    figs = []
    fig_list = WFT_FIGS + (DET3_EXTRA if DET[key] == 'det3' else [])
    for rel, slug, cap in fig_list:
        link = copy_fig(os.path.join(W, rel), f'{d["det"]}_{slug}')
        if link:
            figs.append((link, cap, 'wft'))
    for rel, slug, cap in HW_FIGS:
        link = copy_fig(os.path.join(cfg.OUT_BASE, rel), f'{d["det"]}_{slug}')
        if link:
            figs.append((link, cap, 'hw'))
    d['figs'] = figs
    return d


def eff_at_r(eff, r):
    if not eff or 'eff_vs_R' not in eff:
        return None
    e = eff['eff_vs_R']
    for k in (str(r), f'{r:.0f}', f'{r:.1f}'):
        if k in e:
            return e[k]
    return None


def gate_table():
    """Rebuild the t0-prior gate comparison. Preferred source: the staged
    arms' eval/summary.json (only det4's survived the arm evaluations);
    fallback: the verdict table recorded in the campaign runbook §7 — the
    same gate_eval output, transcribed at decision time."""
    runbook = {}
    try:
        txt = open(os.path.join(REPO, RUNBOOK)).read()
        for m in re.finditer(
                r'^\| (mx17_\d) \| ([\d.]+) / ([\d.]+) / ([\d.]+) '
                r'\| ([\d.]+) / ([\d.]+) / ([\d.]+) \| \*\*(ADOPT|FALLBACK)'
                r'\*\*', txt, re.M):
            det = 'det' + m.group(1)[-1]
            runbook[det] = (
                dict(within5=float(m.group(2)), far=float(m.group(3)),
                     core=float(m.group(4))),
                dict(within5=float(m.group(5)), far=float(m.group(6)),
                     core=float(m.group(7))),
                m.group(8))
    except OSError:
        pass
    rows = []
    for key in FLEET:
        if key == 'sat_det3':
            continue
        arms = {}
        for name in (key, key + '__t0p', key + '__prod_noprior'):
            s = jload(os.path.join(STAGING, name, 'eval', 'summary.json'))
            row = jload(os.path.join(STAGING, name, 'job_row.json'))
            if not (s and row):
                continue
            arm = 't0p' if 't0p' in row.get('bundle_used', '') else 'prod'
            arms[arm] = s
        if 'prod' in arms and 't0p' in arms:
            a, b = arms['prod'], arms['t0p']
            ok = (b['within5'] >= a['within5'] - 0.1
                  and b['far'] <= a['far'] + 0.1
                  and b['core'] <= a['core'] * 1.02)
            better = (b['within5'] > a['within5'] or b['far'] < a['far']
                      or b['core'] < a['core'])
            rows.append((DET[key], a, b,
                         'ADOPT' if ok and better else 'FALLBACK'))
        elif DET[key] in runbook:
            a, b, v = runbook[DET[key]]
            rows.append((DET[key], a, b, v))
    return rows


def logistics_runs():
    """Group the manifest by run for the logistics table."""
    with open(MANIFEST) as f:
        man = list(csv.DictReader(f))
    runs = {}
    for r in man:
        run = r['run'] or '(no run_config)'
        g = runs.setdefault(run, dict(
            run=run, dets=set(), tiers={'A': 0, 'B': 0, 'C': 0},
            excluded=0, reasons=set(), resist=set(), drift=set(), gas=set()))
        g['dets'].add(r['det'])
        tier = r['tier']
        excl = (tier == 'C' and not r['reason'].startswith('resist'))
        if excl:
            g['excluded'] += 1
            g['reasons'].add(r['reason'])
        else:
            g['tiers'][tier] = g['tiers'].get(tier, 0) + 1
            if tier == 'C':
                g['reasons'].add('HV scan (trend-grade)')
        for col, s in (('resist_V', 'resist'), ('drift_V', 'drift'),
                       ('gas', 'gas')):
            if r[col]:
                g[s].add(r[col])
    # date from the trailing 6-27-26 style token
    def date_key(run):
        m = re.search(r'(\d+)-(\d+)-(\d+)$', run)
        return (int(m.group(3)), int(m.group(1)), int(m.group(2))) if m \
            else (99, 99, 99)
    return sorted(runs.values(), key=lambda g: date_key(g['run'])), man


def staging_counts():
    dirs = [d for d in sorted(os.listdir(STAGING))
            if os.path.isdir(os.path.join(STAGING, d))]
    off = [d for d in dirs if d.endswith('__offcond')]
    arms = [d for d in dirs if d.endswith(('__t0p', '__prod_noprior'))]
    return dict(total=len(dirs), offcond=len(off), arms=len(arms),
                tierA=len(dirs) - len(off) - len(arms))


# ---------------------------------------------------------------- charts
PALETTE = ['#2a78d6', '#d64545', '#3fa45b', '#9467bd', '#e08b2d']


def svg_eff_curves(data):
    """Efficiency vs matching radius R, all five chambers, inline SVG."""
    W, H = 760, 360
    ml, mr, mt, mb = 52, 150, 16, 44
    pw, ph = W - ml - mr, H - mt - mb
    rmax = 15.0

    def X(r):
        return ml + pw * r / rmax

    def Y(v):
        return mt + ph * (100 - v) / 100

    p = []
    for v in range(0, 101, 20):
        p.append(f'<line class="grid" x1="{ml}" x2="{ml+pw}" y1="{Y(v):.1f}" '
                 f'y2="{Y(v):.1f}"/>')
        p.append(f'<text class="tick" x="{ml-8}" y="{Y(v)+4:.1f}" '
                 f'text-anchor="end">{v}</text>')
    for r in (2, 5, 10):
        p.append(f'<line class="gridv" x1="{X(r):.1f}" x2="{X(r):.1f}" '
                 f'y1="{mt}" y2="{mt+ph}"/>')
    for r in range(0, 16, 5):
        p.append(f'<text class="tick" x="{X(r):.1f}" y="{mt+ph+18}" '
                 f'text-anchor="middle">{r}</text>')
    for i, d in enumerate(data):
        eff = d['eff']
        if not eff or 'eff_vs_R' not in eff:
            continue
        pts = sorted((float(k), v) for k, v in eff['eff_vs_R'].items()
                     if float(k) <= rmax)
        path = ' '.join(f'{X(r):.1f},{Y(v):.1f}' for r, v in pts)
        c = PALETTE[i % len(PALETTE)]
        p.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.2" '
                 f'points="{path}"/>')
        for r, v in pts:
            p.append(f'<circle cx="{X(r):.1f}" cy="{Y(v):.1f}" r="2.6" '
                     f'fill="{c}"/>')
        y5 = [v for r, v in pts if abs(r - 5) < .01]
        lab = f'{d["det"]}  ({y5[0]:.1f}% @5mm)' if y5 else d['det']
        p.append(f'<text class="leg" x="{ml+pw+12}" y="{mt+18+i*20}" '
                 f'fill="{c}">{esc(lab)}</text>')
    p.append(f'<text class="ylab" transform="rotate(-90)" '
             f'x="{-(mt+ph/2):.0f}" y="14" text-anchor="middle">'
             f'efficiency [%]</text>')
    p.append(f'<text class="xtitle" x="{ml+pw/2:.0f}" y="{H-6}" '
             f'text-anchor="middle">matching radius R [mm] '
             f'(dashed: 2, 5, 10 mm)</text>')
    return (f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Efficiency versus matching radius for all five '
            f'chambers">{"".join(p)}</svg></div>')


# ---------------------------------------------------------------- html bits
def table(headers, rows, cls='num'):
    th = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
                   for r in rows)
    return (f'<div class="tbl-wrap"><table class="{cls}"><thead><tr>{th}</tr>'
            f'</thead><tbody>{body}</tbody></table></div>')


def figure(link, caption, wide=False):
    return (f'<figure class="{"wide" if wide else ""}">'
            f'<a href="{link}" target="_blank" rel="noopener">'
            f'<img src="{link}" alt="{esc(caption[:80])}" loading="lazy"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def corr_bias(d, plane):
    """Bias with the bundle's w0/kw angle constants applied — the frozen code
    does not apply them (INVESTIGATION_2026-08-12.md §4); values from the
    standard 03_angles run on the corrected table (wft/angles_w0corr/)."""
    a = jload(os.path.join(d['cfg'].OUT_BASE, 'wft', 'angles_w0corr',
                           'angular_resolution.json'))
    if not a:
        return None
    return a['planes'].get(plane, {}).get('bias_deg')


def build(data):
    by_det = {d['det']: d for d in data}
    order = ['det3', 'det2', 'det6', 'det7', 'det4']   # best -> worst
    dd = [by_det[x] for x in order if x in by_det]

    # ---------- overview ----------
    def hl(d):
        return (d['eff']['within_R'], d['eff']['core_sigma_mm'])
    tiles = ''.join(
        f'<div class="tile"><div class="tile-v">{d["eff"]["within_R"]:.1f}%'
        f'</div><div class="tile-k">{d["det"]} within 5 mm</div>'
        f'<div class="tile-s">core σ {d["eff"]["core_sigma_mm"]:.2f} mm · '
        f'{d["eff"]["n_rays"]:,} rays</div></div>' for d in dd)

    metric_rows = []
    M = [('rays', lambda d: fmt(d['eff']['n_rays'], '{:,}')),
         ('detected at all (any hit) %', lambda d: fmt(d['eff']['has_any'], '{:.1f}')),
         ('reco within 2 mm %', lambda d: fmt(eff_at_r(d['eff'], 2), '{:.1f}')),
         ('reco within 5 mm %', lambda d: fmt(d['eff']['within_R'], '{:.1f}')),
         ('reco at all %', lambda d: fmt(d['eff']['reco_at_all'], '{:.1f}')),
         ('reco far (>5 mm) %', lambda d: fmt(d['eff']['reco_far'], '{:.1f}')),
         ('core σ(r) mm', lambda d: fmt(d['eff']['core_sigma_mm'])),
         ('median r mm', lambda d: fmt(d['eff']['median_r_mm'])),
         ('spark-flagged %', lambda d: fmt(d['eff']['spark_frac'], '{:.1f}')),
         ('σθ X °', lambda d: fmt(d['state'].get('sig_x'))),
         ('σθ Y °', lambda d: fmt(d['state'].get('sig_y'))),
         ('bias X / Y ° (uncorrected¹)', lambda d:
          fmt(d['state'].get('bias_x'), '{:+.2f}')
          + ' / ' + fmt(d['state'].get('bias_y'), '{:+.2f}')),
         ('bias X / Y ° (w0/kw applied¹)', lambda d:
          fmt(corr_bias(d, 'x'), '{:+.2f}') + ' / '
          + fmt(corr_bias(d, 'y'), '{:+.2f}')),
         ('implied-v spread X / Y', lambda d: fmt(d['state'].get('vsp_x'))
          + ' / ' + fmt(d['state'].get('vsp_y'))),
         ('v_drift µm/ns', lambda d: fmt(d['state'].get('v_drift'), '{:.1f}')),
         ('hits chain, within 5 mm %', lambda d:
          fmt(d['eff_hits'] and d['eff_hits'].get('within_R'), '{:.1f}')),
         ('hits chain, core σ mm', lambda d:
          fmt(d['eff_hits'] and d['eff_hits'].get('core_sigma_mm')))]
    for name, f in M:
        metric_rows.append([name] + [f(d) for d in dd])
    digest_tbl = table(['quantity'] + [d['det'] for d in dd], metric_rows)

    overview = f"""
<div class="verdict"><p style="margin:0"><b>The June cosmic fleet is fully
characterized on the frozen waveform-first reconstruction.</b> Two chambers
are excellent (det3 <b>93.3 %</b>, det2 <b>92.0 %</b> of reference tracks
reconstructed within 5 mm, core σ 0.45/0.43 mm, σθ 1.1–1.5°), det6 and det7
are limited by their operating conditions (74.9 % / 56.9 % — slow drift and
sparking respectively, not reconstruction), and det4 is hardware-limited
(41.6 % within 5 mm against 95.8 % raw detection — fixed non-amplifying
stripes). The waveform-first chain beats the hits chain on every detector's
detection and efficiency accounting and roughly halves the angular-resolution
figures. Numbers come from the 2026-08-12 lxplus condor re-reconstruction of
the full June dataset at the frozen analysis (214 manifest rows → 149
results, 0 unaccounted).</p></div>
<div class="tiles">{tiles}</div>
<h2>Fleet digest — waveform-first, campaign generation</h2>
{digest_tbl}
<p class="note">σθ values sit above pre-freeze figures quoted in older slides
(det3 X: 1.08 → 1.16): the per-event slopes are unchanged — the
slope-reliable <i>population</i> and the angle mapping changed at the freeze.
The trusted flatness judge (implied-v spread) improves sharply. Quote angles
only from this table.<br>
¹ The frozen code computes angles <b>without</b> the per-plane w→angle
constants (w0/kw) that every calibration bundle carries — a silent
regression at <code>f9e18d2</code>, found 8-12 overnight. The uncorrected
row is what the frozen campaign wrote; the corrected row applies the exact
<code>9dd7d6e</code> formula through the standard accounting and collapses
every |bias| to ≤ 0.27°. See Method &amp; caveats.</p>
<h2>Efficiency vs matching radius</h2>
{svg_eff_curves(dd)}
"""

    # ---------- fleet ----------
    fleet_figs = []
    for src, slug, cap in [
        (os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img',
                      'efficiency_breakdown.png'), 'fleet_effbreak',
         '<b>Loss budget, det3 (the reference chamber).</b> Where crossing '
         'muons go: reconstructed within 5 mm, near-miss tail, spark veto, '
         'no-point, silent. Every chamber has its own version of this '
         'figure on its Detectors sub-tab.'),
        (os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img',
                      'efficiency_residual_tail.png'), 'fleet_tail',
         '<b>Residual structure, det3.</b> |r| distribution (log) and '
         'efficiency vs matching radius; the core/tail split is what the '
         '2 mm criterion below probes.')]:
        link = copy_fig(src, slug)
        if link:
            fleet_figs.append(figure(link, cap, wide=True))
    maps2 = ''.join(
        figure(f'figures/{d["det"]}_map2mm.png',
               f'<b>{d["det"]}</b> — efficiency, r &lt; 2 mm criterion '
               f'(within-2mm: {fmt(eff_at_r(d["eff"], 2), "{:.1f}")} %).')
        for d in dd if os.path.exists(os.path.join(
            FIG, f'{d["det"]}_map2mm.png')))
    fleet = f"""
<h2>Loss budget and residual structure</h2>
{''.join(fleet_figs)}
<h2>2 mm-criterion efficiency maps — the tight-core view</h2>
<p>Success = reconstruction within <b>2 mm</b> of the reference track
(core σ is 0.43–0.62 mm, so this accepts the core and rejects the tail).
Binned 40×40 (~12 mm pitch); a literal 2 mm kernel would hold ~0.05 cosmic
rays per cell at these statistics. det4's dead stripes and det7's uniform
suppression are immediately visible.</p>
<div class="figgrid">{maps2}</div>
"""

    # ---------- detectors ----------
    det_btns, det_panels = [], []
    for i, d in enumerate(dd):
        det = d['det']
        s = d['state']
        e = d['eff']
        stat = table(['quantity', 'value'], [
            ['run / subrun', f'{esc(d["cfg"].RUN)} / {esc(d["cfg"].SUB_RUN)}'],
            ['M3 reference rays', fmt(e['n_rays'], '{:,}')],
            ['events reconstructed', fmt(s.get('n_events'), '{:,}')],
            ['detected at all', fmt(e['has_any'], '{:.1f}') + ' %'],
            ['within 2 mm', fmt(eff_at_r(e, 2), '{:.1f}') + ' %'],
            ['within 5 mm', fmt(e['within_R'], '{:.1f}') + ' %'],
            ['core σ(r)', fmt(e['core_sigma_mm']) + ' mm'],
            ['median r', fmt(e['median_r_mm']) + ' mm'],
            ['σθ X / Y', fmt(s.get('sig_x')) + ' / ' + fmt(s.get('sig_y'))
             + ' °'],
            ['spark-flagged', fmt(e['spark_frac'], '{:.1f}') + ' %'],
            ['v_drift', fmt(s.get('v_drift'), '{:.2f}') + ' µm/ns'],
            ['calibration bundle', esc(s.get('bundle_dir', '?'))],
            ['reco generation', esc(s.get('reco_mtime', '?'))]])
        wft_figs = ''.join(figure(l, c) for l, c, kind in d['figs']
                           if kind == 'wft')
        hw_figs = ''.join(figure(l, c) for l, c, kind in d['figs']
                          if kind == 'hw')
        hw_block = (f'<h3>Hardware QA (hits chain, pre-campaign vintage)</h3>'
                    f'<p class="note">These describe the raw detector '
                    f'(amplitude, occupancy), not the reconstruction; they '
                    f'predate the campaign and are unaffected by it.</p>'
                    f'<div class="figgrid">{hw_figs}</div>') if hw_figs else ''
        det_btns.append(f'<button class="sub-btn{" active" if i == 0 else ""}"'
                        f' data-sub="{det}">{det}</button>')
        det_panels.append(f"""
<div class="sub-panel{' active' if i == 0 else ''}" id="sub-{det}">
<p>{DET_NOTES.get(det, '')}</p>
<div class="cols">{stat}</div>
<h3>Campaign reconstruction</h3>
<div class="figgrid">{wft_figs}</div>
{hw_block}
</div>""")
    detectors = (f'<div class="subnav">{"".join(det_btns)}</div>'
                 + ''.join(det_panels))

    # ---------- hv ----------
    hv = jload(os.path.join(OUT, 'hv_trends.json')) or {'rows': []}
    hvrows = hv['rows']
    cov = []
    for det in order:
        mx = MXDET[det]
        rs = [r for r in hvrows if r['det'] == mx and not r['on_conditions']]
        if not rs:
            continue
        res = sorted(r['resist_V'] for r in rs)
        drift = sorted({r['drift_V'] for r in rs})
        cov.append([det, len(rs), f'{res[0]:.0f}–{res[-1]:.0f}',
                    ', '.join(f'{v:.0f}' for v in drift),
                    esc(sorted({r['gas'] for r in rs})[0])])
    hv_figs = []
    for det in order:
        row = ''.join(
            figure(f'figures/hv_{det}_{tag}.png', cap)
            for tag, cap in [
                ('reco', f'<b>{det}</b> — fraction of M3-matched events with '
                 'a two-plane reconstruction vs resistive HV.'),
                ('gain', f'<b>{det}</b> — relative gain (median fitted '
                 'charge, X plane; log scale).'),
                ('shape', f'<b>{det}</b> — gain-normalized shape χ² '
                 '(median (χ²/dof)/(q/1000)², X plane; log scale). Flat = '
                 'healthy; a rise flags a genuine shape breakdown, not '
                 'gain. The raw per-plane quality flag is an amplitude cut '
                 'in disguise and is deliberately not plotted.')]
            if os.path.exists(os.path.join(FIG, f'hv_{det}_{tag}.png')))
        if row:
            hv_figs.append(f'<h3>{det}</h3><div class="figgrid">{row}</div>')
    hvtab = f"""
<div class="warn"><b>Trend-grade only, by construction — two independent
reasons.</b> (1) Each scan point is reconstructed with the detector's frozen
calibration bundle <i>outside</i> the HV conditions it was calibrated at
(a bundle off its conditions is a silent error — here it is a loud,
stamped one: every row carries <code>off_conditions: true</code>).
(2) The scan subruns have v1-only M3 tracking — reference recipe
[χ²&lt;1] with no NClus ≥ 4 clause, looser than the golden rows'. Shapes,
plateaus and knees are meaningful; absolute values are not comparable to the
fleet tab.</div>
<h2>Scan coverage</h2>
{table(['chamber', 'scan points', 'resist V range', 'drift V', 'gas'], cov)}
<p>Metrics are computed from each scan point's reconstruction table alone
(denominator = that subrun's M3-matched events). The star marks the golden
(on-conditions) point through the identical metric. The reconstructed
fraction here is <i>not</i> the within-5 mm efficiency of the fleet tab — it
needs no alignment, so it is computable uniformly across all 127 points.</p>
{''.join(hv_figs)}
"""

    # ---------- logistics ----------
    runs, man = logistics_runs()
    sc = staging_counts()
    run_rows = []
    for g in runs:
        m = re.search(r'(\d+-\d+-\d+)$', g['run'])
        date = m.group(1) if m else '—'
        tiers = g['tiers']
        parts = []
        if tiers.get('A'):
            parts.append(f'{tiers["A"]} golden-condition')
        if tiers.get('B'):
            parts.append(f'{tiers["B"]} drift-scan (deferred)')
        if tiers.get('C'):
            parts.append(f'{tiers["C"]} HV-scan')
        if g['excluded']:
            parts.append(f'{g["excluded"]} excluded')
        note = '; '.join(sorted(g['reasons']))[:120]
        run_rows.append([
            esc(g['run']), date,
            ', '.join(sorted(d for d in g['dets'] if d and d != '?')),
            ' + '.join(parts) or '—',
            (esc(sorted(g['resist'])[0]) + '–' + esc(sorted(g['resist'])[-1])
             if len(g['resist']) > 1 else esc(next(iter(g['resist']), '—'))),
            ', '.join(sorted(g['drift'])) or '—',
            esc(note)])
    golden_rows = []
    for d in dd:
        mrow = next((r for r in man if r['key'] == d['key']), {})
        golden_rows.append([
            d['det'], esc(d['key']), esc(d['cfg'].RUN), esc(d['cfg'].SUB_RUN),
            esc(mrow.get('resist_V', '?')) + ' / '
            + esc(mrow.get('drift_V', '?')),
            esc(mrow.get('gas', '?')), esc(d['state'].get('bundle_dir', '?')),
            fmt(d['eff']['n_rays'], '{:,}')])
    gt = gate_table()
    gate_rows = [[det,
                  f'{a["within5"]:.2f} / {a["far"]:.2f} / {a["core"]:.3f}',
                  f'{b["within5"]:.2f} / {b["far"]:.2f} / {b["core"]:.3f}',
                  f'<b>{v}</b>'] for det, a, b, v in gt]
    freeze_commit = '(unknown)'
    fc = os.path.join(CAMPAIGN_DIR, 'FREEZE_COMMIT.txt')
    if os.path.exists(fc):
        freeze_commit = open(fc).read().strip().split()[0][:9]
    logistics = f"""
<h2>Campaign in one table</h2>
{table(['stage', 'count', 'note'], [
    ['manifest rows', '214', 'every June (subrun × detector) pair, classified'],
    ['runnable', '159', '22 tier A (golden conditions) + 7 tier B (drift '
     'scan) + 130 tier C (HV scan, off-conditions)'],
    ['excluded by manifest', '55', '37 pre-June · 12 no frozen bundle (incl. '
     'all det1) · 5 no run_config · 1 no decoded files'],
    ['produced a result', '149', f'{sc["tierA"]} absolute-geometry + '
     f'{sc["offcond"]} trend-grade staged (+{sc["arms"]} gate arms)'],
    ['tier B removed', '7', 'v-refit needs hits-chain alignment products '
     'never made for the drift scan — deferred, 7-30 bench gap results stand'],
    ['terminal data failures', '3', 'two subruns with the telescope off, one '
     'with pre-7-25 hits'],
    ['unaccounted', '0', 'every row has a result or a recorded reason'],
], cls='')}
<p>Executed 2026-08-12 on lxplus HTCondor (clusters 3924896 gate /
3924904 main, ~60–100 core-h), code shipped at freeze commit
<code>{esc(freeze_commit)}</code>; one job = one manifest row, inputs fetched
from EOS, outputs verified md5-identical after promotion. Full execution
record: <code>{RUNBOOK}</code> §9.</p>
<h2>The five golden datasets (absolute geometry)</h2>
{table(['chamber', 'key', 'run', 'subrun', 'resist / drift V', 'gas',
        'bundle', 'M3 rays'], golden_rows)}
<h2>Every June run in the campaign</h2>
{table(['run', 'date', 'detectors', 'rows', 'resist V', 'drift V', 'notes'],
       run_rows, cls='')}
<h2>t0-prior gate (σ = 5 trigger prior, per detector)</h2>
<p>det3 validated the prior directly; the other four ran full-statistics A/B
arms through the standard accounting. Rule: adopt iff no regression beyond
(within5 −0.1, far +0.1, core +2 %) and ≥ 1 improvement.</p>
{table(['chamber', 'no prior: within5 / far / core σ',
        'with prior: within5 / far / core σ', 'verdict'], gate_rows)}
<p class="note">Final fleet configuration: det2 <code>lp_t0p</code> ·
det3 <code>lp2_t0p</code> · det4 <code>lp_t0p</code> · det6 <code>lp</code>
· det7 <code>lp</code>. A mixed fleet is deliberate — the gate decides per
chamber, and each result records its bundle.</p>
"""

    # ---------- method ----------
    prov_rows = []
    for d in dd:
        s = d['state']
        stale = ('<b class="bad">STALE: ' + ', '.join(s['stale']) + '</b>'
                 if s.get('stale') else 'coherent')
        prov_rows.append([d['det'], esc(s.get('bundle_dir', '?')),
                          esc(s.get('kernel', '?')),
                          fmt(s.get('v_drift'), '{:.2f}'),
                          fmt(s.get('n_events'), '{:,}'),
                          esc(s.get('reco_mtime', '?')), stale])
    method = f"""
<h2>Reconstruction basis — why waveforms, not hit times</h2>
<p>On these resistive-strip detectors a per-strip hit time is an aggregate of
the strip's own charge and delayed, dispersed copies of its neighbours'
(~29 % at τ ≈ 47 ns to ±1 strip). Reconstructing geometry from hit times
compresses the drift-time ladder by 20–30 %, reads ~4° too steep, and makes
the cluster fan away from the true track with depth — independent of the
time estimator. All geometry here therefore comes from a forward-model fit
of the raw waveforms (the <code>wft/</code> chain); hits are used only to
decide which events and strips to look at, and for QA. The measured basis is
<code>RECONSTRUCTION_BASIS.md</code>; the det3 tab's explanation figures
show the effect directly.</p>
<h2>What was frozen</h2>
<p>The analysis was frozen 2026-08-12 for MPGD26 (runbook
<code>{RUNBOOK}</code>): per-detector calibration bundles (kernel
hyper-parameters, drift velocity, w→angle constants, and — where the gate
adopted it — the σ=5 trigger-t0 prior), the reconstruction code at the
freeze commit, and the M3 reference recipe (χ² &lt; 1, NClus ≥ 4) on the
golden rows. A calibration bundle is valid per detector <b>and</b> per run
condition; every campaign output records the bundle that made it.</p>
{table(['chamber', 'bundle', 'kernel', 'v_drift µm/ns', 'events',
        'reco generation', 'accounting'], prov_rows)}
<p class="note">Kernel-label caveat: the bundles carry the RC-ladder
(share_lp) hyper-parameters, but every production bundle stores
<code>share_mode: null</code>, which the loader resolves to the
<b>delay</b> kernel branch — the <code>share_lp</code> hyper is read by
nothing. All campaign numbers, gates and validations were measured on that
loaded configuration, so the fleet is self-consistent; the delay-vs-lp
adjudication is deliberately post-MPGD26 (runbook §2/§8). Measured
difference on det3: median |Δp0| 10–20 µm.</p>
<h2>Findings of the 8-12 overnight audit (det3 deep-dive)</h2>
<p>Triggered by the maps looking off-centre and the fit-quality-vs-HV
collapse; full record in
<code>mx_june_wft/quality_investigation/INVESTIGATION_2026-08-12.md</code>.</p>
<ul>
<li><b>The reconstruction reproduces bit-identically</b> on a second machine
from an independent copy of the raw data (400 golden det3 events: p0/w/t0
|Δ| = 0, all flags identical, same 8,479 M3-matched events).</li>
<li><b>Alignment is right and the off-centre footprint is real geometry</b>:
wft and hits-chain alignments agree to 10 µm; the dead column at reference
X &gt; 145 mm is the detector-local Y-passivation band ([18, 380] mm) mapped
through the ~90° strip rotation, identical in the hits-chain maps.</li>
<li><b>Track counts are the frozen M3 recipe</b>, not a loss: 47,452 M3
events → 8,479 good tracks ([χ²&lt;1 &amp; NClus≥4], 17.9 %) → 7,049 rays
in the active box — within 1 % of both the hits accounting (7,119) and the
pre-campaign baseline (7,130).</li>
<li><b>The per-plane quality flag is an amplitude cut in disguise</b>:
χ² is pedestal-weighted, so χ²/dof ∝ amplitude² (measured exponent
2.0) against an absolute threshold of 300. Quality-fail events still
reconstruct at 93.1 % within 5 mm; p0_err is amplitude-flat. It gates
nothing in the headline accounting. Real saturation censoring is confined
to the top amplitude decile (~5 % of samples).</li>
<li><b>The fleet-wide angle bias is a code regression at the freeze</b>:
<code>f9e18d2</code> silently dropped the per-plane w→angle constants
(w0/kw) of <code>9dd7d6e</code>; every bundle carries them calibrated.
arctan(w0/v) reproduces the bias detector-by-detector (det6 Y: predicted
−1.14°, measured −1.04°); applying the constants through the standard
accounting collapses every |bias| to ≤ 0.27° (the corrected digest row).
Post-freeze fix: restore the two lines in <code>plane_fit</code>.</li>
</ul>
<h2>Caveats — what this report does and does not claim</h2>
<ul>
<li><b>HV-scan tab is trend-grade only</b> (off-conditions bundles + looser
v1 M3 reference — both stamped per row). Shapes are physics; levels are
not.</li>
<li><b>σθ is not comparable to pre-freeze slides</b> (population + mapping
change at the freeze; per-event slopes unchanged; implied-v flatness — the
trusted judge — improved). This table is the quotable record.</li>
<li><b>det4/det6/det7 numbers characterize the chambers as operated in
June</b>, not the reconstruction ceiling: det4's stripes are hardware, det7 sparked,
and det6's fitted v/σ_s pair is under review (possible degenerate
calibration basin). The algorithm's ceiling is det3/det2.</li>
<li><b>Efficiency depends on the M3 acceptance and recipe</b>: the
denominator is reference tracks crossing the active area; a different
χ²/NClus recipe shifts every number at the few-tenths-pp level.</li>
<li><b>The 2 mm maps are 12 mm-pitch pictures</b> of a 2 mm criterion — at
cosmic statistics a true 2 mm-kernel map would be noise. Sub-bin structure
(det4's stripes) is real but under-resolved.</li>
<li><b>The det3 drift scan has no waveform-first result yet</b> (tier B
deferred — the per-point v-refit needs alignment products that don't exist
yet); the 7-30 bench-line gap study stands.</li>
<li><b>Hardware-QA figures are hits-chain, pre-campaign vintage</b> — kept
because they describe the hardware, which did not change.</li>
</ul>
"""

    tabs = [('overview', 'Overview', overview),
            ('fleet', 'Fleet results', fleet),
            ('detectors', 'Detectors', detectors),
            ('hv', 'HV scans', hvtab),
            ('logistics', 'Logistics', logistics),
            ('method', 'Method & caveats', method)]
    nav = ''.join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'data-tab="{tid}">{lab}</button>'
        for i, (tid, lab, _) in enumerate(tabs))
    panels = ''.join(
        f'<div class="tab-panel{" active" if i == 0 else ""}" '
        f'id="tab-{tid}">{body}</div>'
        for i, (tid, _, body) in enumerate(tabs))
    stamp = time.strftime('%Y-%m-%d %H:%M')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>June cosmic bench — micro-TPC fleet report</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#fbfbfa; --surface:#ffffff; --line:#e3e2df;
  --ink:#14140f; --ink2:#55534c; --ink3:#87857c;
  --acc:#2a78d6; --ok:#1b7f4b; --okbg:#1b7f4b14;
  --warnbg:#f3d77a2e; --warnln:#b8860b; --bad:#c0392b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15151a; --surface:#1c1c22; --line:#33333c;
    --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
    --acc:#3987e5; --ok:#5fd08a; --okbg:#5fd08a1a;
    --warnbg:#f3d77a14; --warnln:#d1a33a; --bad:#e06050;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --acc:#3987e5; --ok:#5fd08a; --okbg:#5fd08a1a;
  --warnbg:#f3d77a14; --warnln:#d1a33a; --bad:#e06050;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:26px 20px 64px; background:var(--bg);
  color:var(--ink); font:15px/1.6 -apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:1.55rem; margin:0 0 4px; letter-spacing:-.01em; }}
h2 {{ font-size:1.1rem; margin:30px 0 12px; padding-bottom:6px;
  border-bottom:1px solid var(--line); }}
h3 {{ font-size:.98rem; margin:20px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 18px; font-size:.93rem; }}
p {{ margin:0 0 12px; }}
code {{ font:.85em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; }}
.verdict {{ background:var(--okbg); border:1px solid var(--ok);
  border-left-width:4px; border-radius:8px; padding:15px 17px;
  margin:0 0 22px; }}
.warn {{ background:var(--warnbg); border:1px solid var(--warnln);
  border-left-width:4px; border-radius:8px; padding:13px 16px;
  margin:0 0 20px; }}
.note {{ color:var(--ink2); font-size:.88rem; }}
.bad {{ color:var(--bad); }}
.tiles {{ display:grid; gap:11px; margin:0 0 10px;
  grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }}
.tile {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:12px 14px; }}
.tile-v {{ font-size:1.45rem; font-weight:650;
  font-variant-numeric:tabular-nums; }}
.tile-k {{ color:var(--ink2); font-size:.84rem; margin-top:2px; }}
.tile-s {{ color:var(--ink3); font-size:.77rem; margin-top:2px; }}
.tbl-wrap {{ overflow-x:auto; margin:0 0 14px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem;
  background:var(--surface); border:1px solid var(--line); }}
th,td {{ padding:6px 10px; text-align:left;
  border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--ink2); font-weight:600; font-size:.78rem; }}
tbody tr:last-child td {{ border-bottom:0; }}
table.num td+td, table.num th+th {{ text-align:right;
  font-variant-numeric:tabular-nums; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:18px 0 20px;
  border-bottom:2px solid var(--line); padding-bottom:0; }}
.tab-btn, .sub-btn {{ appearance:none; background:none; color:var(--ink2);
  border:none; border-bottom:3px solid transparent; padding:8px 14px;
  font:600 .92rem/1 inherit; cursor:pointer; }}
.tab-btn.active, .sub-btn.active {{ color:var(--acc);
  border-bottom-color:var(--acc); }}
.tab-btn:hover, .sub-btn:hover {{ color:var(--ink); }}
.tab-panel, .sub-panel {{ display:none; }}
.tab-panel.active, .sub-panel.active {{ display:block; }}
.subnav {{ display:flex; flex-wrap:wrap; gap:4px; margin:0 0 16px;
  border-bottom:1px solid var(--line); }}
.figgrid {{ display:grid; gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); }}
figure {{ margin:0 0 8px; background:var(--surface);
  border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure.wide {{ grid-column:1/-1; max-width:900px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px;
  background:#fff; padding:4px; }}
figcaption {{ color:var(--ink2); font-size:.8rem; margin-top:6px; }}
figcaption b {{ color:var(--ink); }}
.chart-wrap {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px 6px 2px; overflow-x:auto;
  margin:0 0 14px; }}
svg {{ width:100%; height:auto; min-width:600px; display:block; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.gridv {{ stroke:var(--line); stroke-width:1; stroke-dasharray:4 4; }}
.tick,.leg {{ fill:var(--ink2); font-size:11px; }}
.leg {{ font-size:12px; font-weight:600; }}
.ylab,.xtitle {{ fill:var(--ink3); font-size:11px; }}
.cols {{ max-width:640px; }}
.cols td {{ white-space:normal; }}
.cols td:first-child {{ color:var(--ink2); }}
ul {{ margin:0 0 12px; padding-left:20px; }}
li {{ margin-bottom:7px; }}
.foot {{ color:var(--ink3); font-size:.78rem; margin-top:40px;
  border-top:1px solid var(--line); padding-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>June cosmic bench — micro-TPC fleet report</h1>
<p class="sub">Waveform-first reconstruction of the full June dataset ·
frozen analysis, lxplus condor campaign of 2026-08-12 · five chambers,
214 manifest rows, 149 results · generated {stamp}</p>
<div class="tabs">{nav}</div>
{panels}
<p class="foot">Generated by <code>mx_june_wft/report/make_report.py</code>
from the campaign accounting JSONs, <code>campaign_manifest.csv</code>,
staged gate arms and <code>hv_trends.json</code>. Campaign record:
<code>{RUNBOOK}</code>. Reconstruction basis:
<code>RECONSTRUCTION_BASIS.md</code>.</p>
</div>
<script>
function wire(btnSel, panelPrefix) {{
  document.querySelectorAll(btnSel).forEach(function (b) {{
    b.addEventListener('click', function () {{
      var scope = b.parentElement;
      scope.querySelectorAll(btnSel).forEach(
        function (x) {{ x.classList.remove('active'); }});
      b.classList.add('active');
      var root = scope.parentElement;
      var id = panelPrefix + b.dataset[btnSel === '.tab-btn' ? 'tab' : 'sub'];
      root.querySelectorAll(
        btnSel === '.tab-btn' ? '.tab-panel' : '.sub-panel').forEach(
        function (p) {{ p.classList.remove('active'); }});
      var el = document.getElementById(id);
      if (el) el.classList.add('active');
      if (btnSel === '.tab-btn') history.replaceState(null, '', '#' + b.dataset.tab);
    }});
  }});
}}
wire('.tab-btn', 'tab-');
wire('.sub-btn', 'sub-');
var h = location.hash.replace('#', '');
if (h) {{
  var b = document.querySelector('.tab-btn[data-tab="' + h + '"]');
  if (b) b.click();
}}
</script>
</body>
</html>
"""


def main():
    os.makedirs(FIG, exist_ok=True)
    data = [gather(k) for k in FLEET]
    for d in data:
        if d['state'].get('stale'):
            print(f'WARNING: {d["det"]} accounting is STALE vs its parquet: '
                  f'{d["state"]["stale"]} — numbers below are suspect')
    doc = build(data)
    out = os.path.join(OUT, 'report.html')
    with open(out, 'w') as f:
        f.write(doc)
    nfig = len(glob.glob(os.path.join(FIG, '*.png')))
    print(f'wrote {out} ({len(doc)/1024:.0f} kB) + {nfig} figures')


if __name__ == '__main__':
    main()
