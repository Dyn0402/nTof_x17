#!/usr/bin/env python3
"""
make_grand_report.py — the june_grand_qa.pdf layout, rebuilt as the fleet
report.html on the waveform-first campaign results.

Page order mirrors the PDF exactly: a fleet-summary section (four bar charts,
efficiency/spark vs resist HV, the MM layout diagram, the fleet table), then
one section per detector A-E in the June letter order (A=det3, B=det2, C=det6,
D=det7, E=det4), each with the June stat-card strip, the info box, and the
same eight figure slots. Detector pages use the *June best runs* — for det3
that is the 6-27 weekend run (g_det3_wknd, 22k clean rays), not the saturday
scan the campaign digest keyed on.

Numbers come from the frozen campaign accounting JSONs under each key's
<OUT_BASE>/wft/ (efficiency_breakdown.json, alignment/alignment.json,
angles_w0corr/angular_resolution.json). Angles are quoted from angles_w0corr
ONLY (the frozen reco omits the bundle w0/kw constants — INVESTIGATION_
2026-08-12.md §4). Figures must exist beforehand:

    01_alignment/02_efficiency/03_angles/04_maps per key   (run_chain stages)
    quality_investigation/corrected_angles.py              (angles_w0corr)
    report/make_june_figs.py                               (June figure set)

    ../../.venv/bin/python mx_june_wft/report/make_grand_report.py
Output: /home/dylan/x17/cosmic_bench/Analysis/fleet_report/report.html
        (+ figures/), then publish_selfcontained.py for the single-file copy.
"""
import html
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths      # noqa: E402
setup_paths()
import matplotlib                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import pandas as pd                                 # noqa: E402

OUT_DIR = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report'
FIG = os.path.join(OUT_DIR, 'figures')
HV_JSON = os.path.join(OUT_DIR, 'hv_trends.json')

# June letter order (det_labels.py): A=det3, B=det2, C=det6, D=det7, E=det4.
DETS = [
    dict(letter='A', det='det3', key='g_det3_wknd'),
    dict(letter='B', det='det2', key='o22_long_det2'),
    dict(letter='C', det='det6', key='g_det6_long'),
    dict(letter='D', det='det7', key='g_det7_long'),
    dict(letter='E', det='det4', key='g_det4'),
]
DET_COLOR = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c',
             'D': '#d62728', 'E': '#9467bd'}
MXDET = {'det3': 'mx17_3', 'det2': 'mx17_2', 'det6': 'mx17_6',
         'det7': 'mx17_7', 'det4': 'mx17_4'}
GOLDEN_DRIFT = {'det3': 1000, 'det2': 1000, 'det6': 700, 'det7': 700,
                'det4': 600}

# The June PDF's numbers (june_grand_qa.pdf, generated 2026-07-25, hits chain,
# crossing-muon denominator). Kept as the fixed continuity reference — the PDF
# is a historical artifact, these do not update.
JUNE_PDF = {
    'det3': dict(eff=82.3, sigma=0.66, theta=1.63, spark=10.6, rays=22417),
    'det2': dict(eff=80.3, sigma=0.60, theta=2.16, spark=12.3, rays=3772),
    'det6': dict(eff=42.8, sigma=0.60, theta=3.63, spark=43.5, rays=10366),
    'det7': dict(eff=16.7, sigma=1.30, theta=2.61, spark=66.5, rays=10651),
    'det4': dict(eff=35.3, sigma=0.88, theta=2.48, spark=13.7, rays=12570),
}

LAYOUT_DIAGRAM_CANDIDATES = [
    '/home/dylan/CLionProjects/MX17_Full_Geant/scripts/mx17_mm_layout_topdown.png',
    os.path.join(REPO, 'mx_june_cosmic_qa', 'mx17_mm_layout_topdown.png'),
]

# spark-vs-HV curves: the hits-chain scan CSVs the June PDF used (spark is a
# hits-level quantity — multiplicity > 50 strips — so these remain current).
HV_SPARK_KEYS = ['g_det2', 'g_det3', 'g_det6_hv', 'g_det6_long',
                 'g_det7_hv', 'g_det7_long']


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
        if isinstance(v, float) and not np.isfinite(v):
            return dash
        return spec.format(v)
    except (TypeError, ValueError):
        return esc(v)


def copy_fig(src, slug):
    if not src or not os.path.exists(src):
        return None
    os.makedirs(FIG, exist_ok=True)
    dst = os.path.join(FIG, slug + os.path.splitext(src)[1])
    shutil.copy2(src, dst)
    return 'figures/' + os.path.basename(dst)


def hv_settings(cfg):
    """(resist_V, drift_V) from run_config — build_final_pdf.py's recipe."""
    try:
        d = json.load(open(cfg.run_config_path))
    except OSError:
        return None, None
    det = next((x for x in d.get('detectors', [])
                if (x.get('det_name') or x.get('name')) == cfg.DET_NAME), None)
    ch = (det or {}).get('hv_channels', {})
    subs = d.get('sub_runs', []) or []
    sr = next((s for s in subs if s.get('sub_run_name') == cfg.SUB_RUN),
              subs[0] if subs else {})
    hvs = sr.get('hvs', {})

    def look(name):
        if name not in ch:
            return None
        m, c = ch[name]
        v = hvs.get(str(m), {}).get(str(c))
        return int(v) if isinstance(v, (int, float)) else None
    return look('resist'), look('drift')


# ---------------------------------------------------------------- gathering
def gather(entry):
    cfg = get_config(entry['key'])
    W = os.path.join(cfg.OUT_BASE, 'wft')
    d = dict(entry, cfg=cfg,
             eff=jload(os.path.join(W, 'efficiency', 'efficiency_breakdown.json')),
             align=jload(os.path.join(W, 'alignment', 'alignment.json')),
             ang=jload(os.path.join(W, 'angles_w0corr', 'angular_resolution.json')),
             ang_frozen=jload(os.path.join(W, 'angles', 'angular_resolution.json')),
             meta=jload(os.path.join(W, 'events.meta.json')))
    d['resist'], d['drift'] = hv_settings(cfg)
    a = d['ang'] or d['ang_frozen'] or {}
    planes = a.get('planes', {})
    sig = [planes.get(p, {}).get('sigma_deg') for p in ('x', 'y')]
    sig = [s for s in sig if isinstance(s, (int, float)) and np.isfinite(s)]
    d['theta'] = float(np.mean(sig)) if sig else None
    d['theta_corr'] = bool(d['ang'])
    figs = {}
    for slug, rel in [
            ('sliding', 'wft/efficiency/efficiency_map_sliding.png'),
            ('poscorr', 'wft/efficiency/position_correlation_density.png'),
            ('poscorr_fallback', 'wft/alignment/position_correlation_hist.png'),
            ('angcorr', 'wft/angles_w0corr/angle_correlation_hist.png'),
            ('angcorr_fallback', 'wft/angles/angle_correlation_hist.png'),
            ('resmap', 'wft/maps/resolution_map_sliding_r50mm.png'),
            ('radresid', 'wft/alignment/radial_residuals.png'),
            ('ampstrip', 'raw_detector_qa/amplitude_vs_strip.png'),
            ('scatter', 'wft/efficiency/scatter_within_5mm.png'),
            ('breakdown', 'wft/efficiency/efficiency_breakdown_wide.png')]:
        figs[slug] = copy_fig(os.path.join(cfg.OUT_BASE, rel),
                              f"{d['det']}_{slug}")
    if not figs['angcorr']:
        figs['angcorr'] = figs['angcorr_fallback']
    if not figs['poscorr']:
        figs['poscorr'] = figs['poscorr_fallback']
    d['figs'] = figs
    return d


# ---------------------------------------------------------------- fleet figs
def bar_chart(dd, field, title, ylabel, spec, fname):
    letters = [d['letter'] for d in dd]
    vals = [d.get(field) for d in dd]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(letters))
    cols = [DET_COLOR[L] for L in letters]
    for x, v, c in zip(xs, vals, cols):
        if v is None or not np.isfinite(v):
            ax.text(x, 0.02, 'n/a', ha='center', color='grey',
                    transform=ax.get_xaxis_transform())
            continue
        ax.bar(x, v, color=c)
        ax.text(x, v, spec.format(v), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d['letter']} ({d['det']})" for d in dd], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    vmax = max([v for v in vals if v is not None and np.isfinite(v)] or [1])
    ax.set_ylim(0, vmax * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)
    return 'figures/' + fname


def hv_eff_fig(dd):
    """Efficiency vs resist HV — wft campaign off-conditions trend
    (reconstructed fraction of M3-matched events; trend-grade), golden run
    anchored as a star."""
    hv = jload(HV_JSON)
    if not hv:
        return None
    rows = pd.DataFrame(hv['rows'])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    drew = False
    for d in dd:
        det = d['det']
        mx = MXDET[det]
        sub = rows[(rows['det'] == mx) & (~rows['on_conditions'].astype(bool))]
        sub = sub[np.isfinite(sub['resist_V']) &
                  (sub['drift_V'] == GOLDEN_DRIFT[det]) &
                  (sub['n_events'] > 100)]
        sub = (sub.sort_values(['resist_V', 'n_events'])
               .drop_duplicates('resist_V', keep='last'))
        col = DET_COLOR[d['letter']]
        if len(sub) >= 3:
            ax.plot(sub['resist_V'], 100 * sub['frac_reco'], 'o-', ms=3.5,
                    color=col, label=f"{d['letter']} ({det})")
            drew = True
        g = rows[(rows['det'] == mx) & rows['on_conditions'].astype(bool)]
        if len(g) and d.get('resist'):
            ax.plot([d['resist']], [100 * g.iloc[0]['frac_reco']], '*',
                    ms=15, color=col, mec='black', mew=0.5)
    if not drew:
        plt.close(fig)
        return None
    ax.set_xlabel('resist HV [V]')
    ax.set_ylabel('reconstructed fraction [%]')
    ax.set_title('Efficiency vs resist HV', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.02, 'wft, M3-matched basis (trend-grade); ★ = golden run',
            transform=ax.transAxes, ha='right', fontsize=6.5, color='grey')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fleet_eff_vs_hv.png'), dpi=130)
    plt.close(fig)
    return 'figures/fleet_eff_vs_hv.png'


def hv_spark_fig(dd):
    """Spark rate vs resist HV from the hits-chain scan CSVs (June source;
    spark = >50 strips is hits-defined, so these curves remain current)."""
    curves = {}
    for key in HV_SPARK_KEYS:
        try:
            cfg = get_config(key)
        except KeyError:
            continue
        csv = os.path.join(os.path.dirname(cfg.BASE_PATH.rstrip('/')),
                           'Analysis', cfg.RUN, 'hv_scan', cfg.DET_NAME,
                           'efficiency_vs_hv.csv')
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if 'hv' not in df or 'spark_frac' not in df:
            continue
        det = next((x['det'] for x in DETS if MXDET[x['det']] == cfg.DET_NAME),
                   None)
        if det is None:
            continue
        curves.setdefault(det, []).append(df[['hv', 'spark_frac']])
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for d in dd:
        det = d['det']
        if det not in curves:
            continue
        df = (pd.concat(curves[det]).dropna()
              .drop_duplicates('hv').sort_values('hv'))
        ax.plot(df['hv'], 100 * df['spark_frac'], 'o-', ms=3.5,
                color=DET_COLOR[d['letter']], label=f"{d['letter']} ({det})")
    ax.set_xlabel('resist HV [V]')
    ax.set_ylabel('spark fraction [%]')
    ax.set_title('Spark rate vs resist HV', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.02, 'firing-event basis (hits-chain scans)',
            transform=ax.transAxes, ha='right', fontsize=6.5, color='grey')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fleet_spark_vs_hv.png'), dpi=130)
    plt.close(fig)
    return 'figures/fleet_spark_vs_hv.png'


# ---------------------------------------------------------------- html bits
def card(value, label, kind):
    """June stat card with its colour thresholds."""
    cls = ''
    if value is not None and np.isfinite(value):
        if kind == 'eff':
            cls = 'c-good' if value >= 50 else 'c-warn' if value >= 25 else 'c-bad'
        elif kind == 'spark':
            cls = 'c-bad' if value >= 20 else 'c-warn' if value >= 8 else 'c-good'
    unit = 'mm' if kind == 'mm' else '%'
    txt = fmt(value, '{:.2f}' if kind == 'mm' else '{:.1f}')
    return (f'<div class="stat"><div class="stat-v {cls}">{txt}'
            f'<span class="stat-u">{unit}</span></div>'
            f'<div class="stat-k">{esc(label)}</div></div>')


def figure(link, caption, wide=False):
    if not link:
        return (f'<figure{" class=\"wide\"" if wide else ""}>'
                f'<div class="missing">(missing)</div>'
                f'<figcaption>{caption}</figcaption></figure>')
    return (f'<figure{" class=\"wide\"" if wide else ""}>'
            f'<a href="{link}" target="_blank" rel="noopener">'
            f'<img src="{link}" alt="" loading="lazy"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def detector_section(d):
    eff = d['eff'] or {}
    al = d['align'] or {}
    cfg = d['cfg']
    L, det = d['letter'], d['det']
    hv = ''
    if d['resist']:
        hv += f'<span class="chip">Resist {d["resist"]} V</span>'
    if d['drift']:
        hv += f'<span class="chip">Drift {d["drift"]} V</span>'
    cards = (card(eff.get('within_R'), 'Efficiency (≤5 mm)', 'eff')
             + card(eff.get('core_sigma_mm'), 'Resolution (core σ)', 'mm')
             + card(eff.get('has_any'), 'Fired any strip', 'plain')
             + card(eff.get('reco_at_all'), 'Reconstructed', 'plain')
             + card(eff.get('spark_cat'), 'Spark rate (>50 strips)', 'spark'))
    info = (f"Detector {L}  ({cfg.DET_NAME})\n"
            f"{cfg.RUN}\n"
            f"subrun: {cfg.SUB_RUN}\n"
            f"FEU X/Y: {cfg.MX17_FEUS[0]}/{cfg.MX17_FEUS[1]}    "
            f"z: {cfg.DET_PLANE_Z:.0f} mm\n"
            f"align: θ={al.get('theta_deg', '—')}°  "
            f"z={al.get('z_x', '—')} mm\n"
            f"clean M3 rays: {eff.get('n_rays', '—')}\n"
            f"median |r|: {fmt(eff.get('median_r_mm'))} mm\n"
            f"loss: hit-no-reco {fmt(eff.get('hit_no_reco'), '{:.1f}')}%  "
            f"silent {fmt(eff.get('no_hit'), '{:.1f}')}%")
    f = d['figs']
    theta_note = ('w0/kw-corrected' if d['theta_corr']
                  else '<span class="bad">frozen — w0/kw NOT applied</span>')
    ang = (d['ang'] or {}).get('planes', {})
    ang_line = ''
    if ang:
        ang_line = (f"σθ X/Y = {fmt(ang.get('x', {}).get('sigma_deg'))} / "
                    f"{fmt(ang.get('y', {}).get('sigma_deg'))}°, bias "
                    f"{fmt(ang.get('x', {}).get('bias_deg'), '{:+.2f}')} / "
                    f"{fmt(ang.get('y', {}).get('bias_deg'), '{:+.2f}')}° "
                    f"({theta_note})")
    return f"""
<section class="det" id="det-{L}">
<div class="det-head">
  <div>
    <h2 class="det-title" style="border-color:{DET_COLOR[L]}">Detector {L}</h2>
    <div class="chips">{hv}</div>
  </div>
  <pre class="infobox">{esc(info)}</pre>
</div>
<div class="stats">{cards}</div>
<p class="note">{ang_line}</p>
<div class="figgrid">
{figure(f['sliding'], 'Sliding-window efficiency map — reco within 5 mm | '
        'has_any | rays/kernel (waveform-first, kernel 25 mm).', wide=True)}
{figure(f['poscorr'], 'Position correlation density (detector vs M3).')}
{figure(f['angcorr'], 'Angular correlation density (detector vs M3), '
        'w0/kw-corrected angles.')}
{figure(f['resmap'], 'Sliding-window spatial resolution map (r=50 mm kernel).')}
{figure(f['radresid'], 'Alignment residuals — radial residual, full range and '
        'zoom.')}
{figure(f['ampstrip'], 'Pulse height vs strip (raw QA, hits chain — '
        'reconstruction-independent).')}
{figure(f['scatter'], 'Hit/miss scatter — reco within 5 mm (green) vs not '
        '(red).')}
{figure(f['breakdown'], 'Efficiency breakdown — where do the crossing muons '
        'go?', wide=True)}
</div>
</section>"""


def build(dd, fleet_figs, gen_notes):
    stamp = time.strftime('%Y-%m-%d %H:%M')
    rows = ''.join(
        f'<tr><td style="border-left:6px solid {DET_COLOR[d["letter"]]}">'
        f'{d["letter"]} ({d["det"]})</td>'
        f'<td>{fmt((d["eff"] or {}).get("within_R"), "{:.1f}")}</td>'
        f'<td>{fmt((d["eff"] or {}).get("core_sigma_mm"))}</td>'
        f'<td>{fmt(d.get("theta"))}</td>'
        f'<td>{fmt((d["eff"] or {}).get("spark_cat"), "{:.1f}")}</td>'
        f'<td>{fmt((d["eff"] or {}).get("n_rays"), "{:d}")}</td></tr>'
        for d in dd)
    june_rows = ''.join(
        f'<tr><td>{d["letter"]} ({d["det"]})</td>'
        f'<td>{JUNE_PDF[d["det"]]["eff"]:.1f} &rarr; '
        f'{fmt((d["eff"] or {}).get("within_R"), "{:.1f}")}</td>'
        f'<td>{JUNE_PDF[d["det"]]["sigma"]:.2f} &rarr; '
        f'{fmt((d["eff"] or {}).get("core_sigma_mm"))}</td>'
        f'<td>{JUNE_PDF[d["det"]]["theta"]:.2f} &rarr; '
        f'{fmt(d.get("theta"))}</td>'
        f'<td>{JUNE_PDF[d["det"]]["spark"]:.1f} &rarr; '
        f'{fmt((d["eff"] or {}).get("spark_cat"), "{:.1f}")}</td>'
        f'<td>{JUNE_PDF[d["det"]]["rays"]:,} &rarr; '
        f'{fmt((d["eff"] or {}).get("n_rays"), "{:d}")}</td></tr>'
        for d in dd)
    toc = ' · '.join(f'<a href="#det-{d["letter"]}">Detector {d["letter"]} '
                     f'({d["det"]})</a>' for d in dd)
    det_html = ''.join(detector_section(d) for d in dd)
    ff = fleet_figs
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MX17 June 2026 cosmic-bench — fleet summary</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#fbfbfa; --surface:#ffffff; --line:#e3e2df;
  --ink:#14140f; --ink2:#55534c; --ink3:#87857c;
  --acc:#2a78d6; --good:#1a7f37; --warn:#b26a00; --bad:#b3261e;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#15151a; --surface:#1c1c22; --line:#33333c;
    --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
    --acc:#3987e5; --good:#5fd08a; --warn:#d1a33a; --bad:#e06050;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#15151a; --surface:#1c1c22; --line:#33333c;
  --ink:#f2f2ef; --ink2:#b6b4ab; --ink3:#87857c;
  --acc:#3987e5; --good:#5fd08a; --warn:#d1a33a; --bad:#e06050;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:26px 20px 64px; background:var(--bg);
  color:var(--ink); font:15px/1.6 -apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 2px; letter-spacing:-.01em;
  text-align:center; }}
.sub {{ color:var(--ink2); margin:0 0 20px; font-size:.92rem;
  text-align:center; }}
h2 {{ font-size:1.12rem; margin:30px 0 12px; padding-bottom:6px;
  border-bottom:1px solid var(--line); }}
p {{ margin:0 0 12px; }}
code {{ font:.85em ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--surface); border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; }}
.note {{ color:var(--ink2); font-size:.86rem; }}
.bad {{ color:var(--bad); }}
.toc {{ text-align:center; font-size:.9rem; margin:0 0 18px; }}
a {{ color:var(--acc); }}
.grid2 {{ display:grid; gap:14px; grid-template-columns:repeat(2,1fr); }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
figure {{ margin:0 0 8px; background:var(--surface);
  border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure.wide {{ grid-column:1/-1; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px;
  background:#fff; padding:4px; }}
figcaption {{ color:var(--ink2); font-size:.8rem; margin-top:6px; }}
.missing {{ color:var(--ink3); padding:40px; text-align:center;
  border:1px dashed var(--line); border-radius:6px; }}
.figgrid {{ display:grid; gap:14px;
  grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); }}
.tbl-wrap {{ overflow-x:auto; margin:0 0 14px; }}
table {{ border-collapse:collapse; width:100%; font-size:.86rem;
  background:var(--surface); border:1px solid var(--line); }}
th,td {{ padding:6px 10px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; }}
th:first-child, td:first-child {{ text-align:left; }}
th {{ color:var(--ink2); font-weight:600; font-size:.78rem; }}
tbody tr:last-child td {{ border-bottom:0; }}
.det {{ margin-top:44px; border-top:3px solid var(--line);
  padding-top:16px; }}
.det-head {{ display:flex; justify-content:space-between; gap:18px;
  align-items:flex-start; flex-wrap:wrap; }}
.det-title {{ font-size:1.9rem; margin:0 0 6px; border-bottom:4px solid;
  display:inline-block; padding-bottom:2px; }}
.chips {{ margin:4px 0 10px; }}
.chip {{ display:inline-block; background:var(--surface);
  border:1px solid var(--line); border-radius:20px; padding:2px 12px;
  font-size:.85rem; font-weight:600; color:var(--ink2);
  margin-right:6px; }}
.infobox {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px 13px; margin:0;
  font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink2); }}
.stats {{ display:grid; gap:11px; margin:8px 0 12px;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); }}
.stat {{ background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:12px 14px; }}
.stat-v {{ font-size:1.6rem; font-weight:700;
  font-variant-numeric:tabular-nums; }}
.stat-u {{ font-size:.85rem; font-weight:500; color:var(--ink3);
  margin-left:3px; }}
.stat-k {{ color:var(--ink2); font-size:.8rem; margin-top:2px; }}
.c-good {{ color:var(--good); }} .c-warn {{ color:var(--warn); }}
.c-bad {{ color:var(--bad); }}
.foot {{ color:var(--ink3); font-size:.78rem; margin-top:40px;
  border-top:1px solid var(--line); padding-top:12px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>MX17 June 2026 cosmic-bench — fleet summary</h1>
<p class="sub">M3 v2 reference tracking (NClus=4 &amp; χ²&lt;1.0); best long
run per detector · waveform-first reconstruction (frozen campaign 2026-08-12)
· generated {stamp}</p>
<p class="toc">{toc}</p>

<div class="grid2">
{figure(ff.get('eff'), 'Best-run efficiency (within 5 mm), crossing-muon '
        'denominator incl. sparks — the June PDF convention.')}
{figure(ff.get('sigma'), 'Spatial resolution (core σ of |r| &lt; 15 mm).')}
{figure(ff.get('theta'), 'micro-TPC angular resolution (waveform-first σθ, '
        'mean of X/Y, |tanθ| ≥ 0.08, w0/kw applied).')}
{figure(ff.get('spark'), 'Spark rate (&gt;50 strips), % of crossing muons in '
        'active area.')}
{figure(ff.get('effhv'), 'Efficiency vs resist HV — wft off-conditions trend '
        '(reconstructed fraction, M3-matched basis; trend-grade). '
        '★ = golden run.')}
{figure(ff.get('sparkhv'), 'Spark rate vs resist HV (hits-chain scans, '
        'firing-event basis — spark is hits-defined).')}
{figure(ff.get('layout'), 'MX17 Micromegas layout (top-down).')}
<div>
<h2 style="margin-top:0">Fleet summary</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>Det</th><th>Eff %</th><th>σ mm</th><th>θ°</th>
<th>Spark %</th><th>rays</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>Continuity with june_grand_qa.pdf (7-25 hits chain &rarr; tonight, wft)</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>Det</th><th>Eff %</th><th>σ mm</th><th>θ°</th>
<th>Spark %</th><th>rays</th></tr></thead>
<tbody>{june_rows}</tbody></table></div>
<p class="note">Same runs, same crossing-muon denominator. What moved and
why: <b>θ</b> — June's hybrid-hits estimator vs tonight's waveform-first fit
(roughly 2× better fleet-wide); <b>spark</b> — the 7-25 significance floor
re-classifies discharge-adjacent crossings that the old veto discarded, so
det6/det7 recover 20–40 points of efficiency from the spark bucket;
<b>rays</b> — each accounting draws its own active box from its own reco
footprint (±2–7 %). det3's June page run (22.4k rays) is restored as the
Detector A page below.</p>
</div>
</div>

{det_html}

<h2>Method &amp; caveats</h2>
<ul class="note" style="padding-left:20px">
{gen_notes}
</ul>
<p class="foot">Generated by <code>mx_june_wft/report/make_grand_report.py</code>
from the frozen-campaign accounting JSONs (layout follows
<code>june_grand_qa.pdf</code> / <code>build_final_pdf.py</code>).
Campaign record: <code>mx_june_wft/FREEZE_MPGD26_2026-08-12.md</code> ·
overnight audit: <code>mx_june_wft/quality_investigation/INVESTIGATION_2026-08-12.md</code> ·
reconstruction basis: <code>RECONSTRUCTION_BASIS.md</code>.</p>
</div>
</body>
</html>
"""


NOTES = [
    'Angles are quoted from <code>angles_w0corr/</code> only: the frozen reco '
    'computes angles without the per-plane w→angle constants (w0/kw) every '
    'bundle carries — a silent regression at <code>f9e18d2</code>, found in '
    'the 8-12 overnight audit. Applying the exact <code>9dd7d6e</code> formula '
    'collapses every |bias| to ≤ 0.27°.',
    'Detector A uses the June golden run <code>g_det3_wknd</code> (22.4k clean '
    'rays). Its campaign table contained 620 duplicate-event-id rows from the '
    'false-start acquisition (01H29) the reco job should have excluded; both '
    'copies of every colliding id were dropped (0.9 % of events; originals in '
    '<code>wft/pre_clean_falsestart/</code>).',
    'Efficiency denominators include spark-vetoed crossings (June PDF '
    'convention). The <i>spark</i> stat card is the crossing-based category; '
    'the firing-event-basis spark fraction is higher and appears only in the '
    'HV figure.',
    'The per-plane <code>quality_ok</code> flag is an amplitude cut in '
    'disguise (χ²/dof ∝ gain²) and gates nothing here.',
    'What this does not rule out: det6\'s bundle v/σ_s degeneracy is an open '
    'item (its v=26.7 µm/ns is not a settled gas fact); merged-cluster double '
    'tracks are not split; off-conditions HV points use the golden bundle and '
    'are trend-grade only.',
]


def main():
    os.makedirs(FIG, exist_ok=True)
    dd = [gather(e) for e in DETS]
    for d in dd:
        d['eff_pct'] = (d['eff'] or {}).get('within_R')
        d['core'] = (d['eff'] or {}).get('core_sigma_mm')
        d['spark_pct'] = (d['eff'] or {}).get('spark_cat')
    fleet_figs = dict(
        eff=bar_chart(dd, 'eff_pct', 'Best-run efficiency (within 5 mm)', '%',
                      '{:.1f}', 'fleet_eff.png'),
        sigma=bar_chart(dd, 'core', 'Spatial resolution (core σ)', 'mm',
                        '{:.2f}', 'fleet_sigma.png'),
        theta=bar_chart(dd, 'theta',
                        'micro-TPC angular resolution (wft, w0/kw applied)',
                        'deg', '{:.2f}', 'fleet_theta.png'),
        spark=bar_chart(dd, 'spark_pct', 'Spark rate (>50 strips)', '%',
                        '{:.1f}', 'fleet_spark.png'),
        effhv=hv_eff_fig(dd),
        sparkhv=hv_spark_fig(dd),
        layout=copy_fig(next((p for p in LAYOUT_DIAGRAM_CANDIDATES
                              if os.path.exists(p)), None), 'fleet_layout'),
    )
    notes = ''.join(f'<li>{n}</li>' for n in NOTES)
    html_doc = build(dd, fleet_figs, notes)
    out = os.path.join(OUT_DIR, 'report.html')
    with open(out, 'w') as f:
        f.write(html_doc)
    print(f'wrote {out} ({len(html_doc) / 1024:.0f} kB)')
    for d in dd:
        missing = [k for k, v in d['figs'].items()
                   if v is None and not k.endswith('_fallback')]
        state = 'OK' if not missing else f'MISSING {missing}'
        print(f"  {d['letter']} ({d['det']}, {d['key']}): "
              f"eff {fmt(d['eff_pct'], '{:.1f}')} core {fmt(d['core'])} "
              f"theta {fmt(d['theta'])} [{state}]")


if __name__ == '__main__':
    main()
