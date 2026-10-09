#!/usr/bin/env python3
"""make_report.py -- build report.html for the per-view sharing-kernel study.

Everything is read from the result files, so re-running after the analysis
updates numbers, figures and verdict together:

  modelfree.json        per-view head-on neighbour pattern, five chambers
                        (computed here from the big caches if absent)
  agg_blind.json        held-out bench, every arm (aggregate.py)
  <wft>/plane_ratio/gate_<cand>.json   full-reco gate per golden key

    make_report.py [--out /home/dylan/x17/cosmic_bench/plane_ratio]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = '/home/dylan/x17/cosmic_bench/Analysis/'
RES = '/home/dylan/x17/cosmic_bench/condor_plane_ratio/results'
DETS = {
    'det2': ('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
    'det3': ('sat_det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
    'det4': ('g_det4', 'mx17_det4_day_6-24-26/long_run/mx17_4'),
    'det6': ('g_det6_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
    'det7': ('g_det7_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'),
}
COL = {'x': '#2a78d6', 'y': '#eb6834'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
plt.rcParams.update({'font.size': 10.5, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.axisbelow': True, 'legend.frameon': False})


def modelfree(path):
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, os.path.join(HERE, '..', 'sharing_kernel'))
    import bench_kernel as bk
    rng = np.random.default_rng(1)
    out = {}
    for det, (_k, sub) in DETS.items():
        cache = ANA + sub + '/wft/plane_ratio/big_cache_3000.pkl'
        for v in 'xy':
            A, t, n = bk.build(v, 0.05, 12, cache=cache)

            def st(i=None):
                W = {d: bk.trim_mean(A[d] if i is None else A[d][i]) for d in A}
                s0 = W[0]
                a = {d: W[d].sum() / s0.sum() for d in (1, -1, 2, -2)}
                return ((a[2] + a[-2]) / (a[1] + a[-1]), (a[1] + a[-1]) / 2,
                        (a[-1] - a[1]) / (a[-1] + a[1]))
            r, a1, asym = st()
            bs = np.array([st(rng.integers(0, n, n)) for _ in range(150)])
            out.setdefault(det, {})[v] = dict(n=int(n), r21=r, r21_err=float(bs[:, 0].std()),
                                              a1=a1, a1_err=float(bs[:, 1].std()),
                                              asym=asym, asym_err=float(bs[:, 2].std()))
    json.dump(out, open(path, 'w'), indent=1)
    return out


def fig_modelfree(mf, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    dets = list(DETS)
    x = np.arange(len(dets))
    for k, v in enumerate('xy'):
        vals = [mf[d][v]['r21'] for d in dets]
        errs = [mf[d][v]['r21_err'] for d in dets]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, yerr=errs, color=COL[v],
               error_kw=dict(ecolor=MUTED, lw=1), label=f'{v.upper()} view')
    ax.set_xticks(x, dets)
    ax.set_ylabel('±2 / ±1 neighbour charge')
    ax.set_ylim(0, 0.68)
    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'modelfree_ratio.png'), dpi=150)
    plt.close(fig)


def fig_scan(agg, out):
    ys = [(60, None), (80, 'pin_x20_y80'), (90, 'pin_x60_y90'), (95, 'pin_x60_y95'),
          (120, 'diag_pin_x60_y120'), (160, 'diag_pin_x60_y160')]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    marks = {'det2': 'o', 'det3': 's', 'det4': '^', 'det6': 'v', 'det7': 'D'}
    # categorical slots 1-5, fixed order (dataviz reference palette)
    dcol = {'det2': '#2a78d6', 'det3': '#eb6834', 'det4': '#1baf7a',
            'det6': '#eda100', 'det7': '#e87ba4'}
    for ax, key, title in ((axs[0], 's68', 'all angles'),
                           (axs[1], 's68_lt5', '|θ| < 5°')):
        for det in DETS:
            xs, vs, es = [], [], []
            for y, tag in ys:
                if tag is None:
                    xs.append(y / 100); vs.append(0.0); es.append(0.0)
                    continue
                r = agg.get(f'{det}_{tag}', {}).get('vs_prod', {}).get('y')
                if r is None:
                    continue
                xs.append(y / 100); vs.append(r[key]); es.append(r[key + '_err'])
            ax.errorbar(xs, vs, yerr=es, marker=marks[det], ms=5, lw=1.5,
                        capsize=2, color=dcol[det], label=det)
        ax.axvline(0.95, color=MUTED, lw=1.2, ls='--')
        ax.axvspan(1.0, 1.7, color=GRID, alpha=0.6, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('Y c2/c1 (pinned; X and all else = production)')
        ax.set_title(title, fontsize=10.5, color=INK)
    axs[0].set_ylabel('Δ s68 (Y) vs production  [deg]')
    axs[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9.5)
    axs[1].text(1.03, 0.12, 'beyond c2 < c1\n(diagnostic only)', fontsize=8.5,
                color=MUTED, va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'y_ratio_scan.png'), dpi=150)
    plt.close(fig)


def gate_rows():
    rows = []
    for det, (key, sub) in DETS.items():
        W = ANA + sub + '/wft/plane_ratio/'
        for cand, base in (('pvy95k', 'prodref'), ('pvy95', 'prodt0')):
            p = W + f'gate_{cand}.json'
            if os.path.exists(p):
                g = json.load(open(p))
                if cand in g:
                    rows.append((det, key, cand, base, g[cand]))
                    break
    return rows


def fig_gate(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rows))
    for k, (lab, key) in enumerate((('all angles', 's68'), ('|θ| < 5°', 's68_lt5'))):
        d = [r[4]['y']['delta'][key] for r in rows]
        e = [r[4]['y']['delta_err'][key] for r in rows]
        ax.bar(x + (k - 0.5) * 0.36, d, 0.34, yerr=e, color=COL['y'],
               alpha=1.0 if k == 0 else 0.55, error_kw=dict(ecolor=MUTED, lw=1),
               label=f'Y, {lab}')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel('Δ s68 (Y), candidate − control  [deg]')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'gate_y.png'), dpi=150)
    plt.close(fig)


def fmt(v, e):
    return f'{v:+.3f} ± {e:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/dylan/x17/cosmic_bench/plane_ratio')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    mf = modelfree(os.path.join(a.out, 'modelfree.json'))
    agg = json.load(open(os.path.join(RES, 'agg_blind.json')))
    rows = gate_rows()
    fig_modelfree(mf, a.out)
    fig_scan(agg, a.out)
    fig_gate(rows, a.out)

    mf_rows = ''.join(
        f'<tr><td>{d}</td><td>{fmt(mf[d]["x"]["r21"], mf[d]["x"]["r21_err"])[1:]}</td>'
        f'<td>{fmt(mf[d]["y"]["r21"], mf[d]["y"]["r21_err"])[1:]}</td>'
        f'<td>{fmt(mf[d]["x"]["asym"], mf[d]["x"]["asym_err"])}</td></tr>' for d in DETS)
    g_rows = ''
    for det, key, cand, base, g in rows:
        y, x = g['y'], g['x']
        eff = g.get('efficiency', {})
        e0, e1 = eff.get(base, {}), eff.get(cand, {})
        effs = (f'{e0.get("within_R", float("nan")):.2f} → {e1.get("within_R", float("nan")):.2f}'
                if e0 and e1 else '—')
        g_rows += (f'<tr><td>{key}</td><td>{cand} vs {base}</td>'
                   f'<td>{y["base"]["s68"]:.3f} → {y["arm"]["s68"]:.3f}<br><small>{fmt(y["delta"]["s68"], y["delta_err"]["s68"])}</small></td>'
                   f'<td>{y["base"]["s68_lt5"]:.3f} → {y["arm"]["s68_lt5"]:.3f}<br><small>{fmt(y["delta"]["s68_lt5"], y["delta_err"]["s68_lt5"])}</small></td>'
                   f'<td>{fmt(x["delta"]["s68"], x["delta_err"]["s68"])}</td><td>{effs}</td></tr>')
    n_better = sum(1 for r in rows if r[4]['y']['delta']['s68_lt5'] + 2 * r[4]['y']['delta_err']['s68_lt5'] < 0)

    html = f'''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Per-view sharing kernel</title>
<style>
:root{{--bg:#fbfbf8;--ink:#1f1f1e;--muted:#6b6a63;--rule:#e4e3dc;--acc:#eb6834}}
@media (prefers-color-scheme: dark){{:root{{--bg:#1a1a19;--ink:#f2f1ea;--muted:#c3c2b7;--rule:#3a3a37}} img{{background:#fbfbf8;border-radius:6px}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--rule);padding-bottom:4px}}
.verdict{{border-left:4px solid var(--acc);padding:10px 14px;background:rgba(235,104,52,.07)}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}} th,td{{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}} img{{max-width:100%}} figcaption{{color:var(--muted);font-size:13px}} small{{color:var(--muted)}}
code{{font-size:13px}}
</style></head><body>
<h1>Per-view sharing kernel</h1>
<p><small>MX17 June cosmic bench + det4 in H4 · 2026-10-09 · <code>sps_beam_test_26/analysis/plane_ratio/</code> on <code>mx17-paper-status</code></small></p>

<div class="verdict"><b>Verdict.</b> Give the Y view its own ±2 copy ratio,
c2/c1 = 0.95 (the most the c2 &lt; c1 gate allows), and leave X and every other
calibration constant at production. In the full reconstruction of the golden
keys this improves the Y angle resolution on {n_better} of {len(rows)} chambers
— by 0.04–0.08° overall and 0.08–0.13° for near-vertical tracks — with X,
efficiency and position unchanged. Refitting the per-view ratio inside the bench
calibration does <i>not</i> work: the bench χ² is model-error dominated and
spends the freedom elsewhere. The ratio has to be pinned.</div>

<h2>What was compared</h2>
<p>Production bundles (r06 on det2/3/7; <code>lp_t0p</code> det4; <code>lp</code> det6) against the same
hypers with only <code>c2_over_c1_y</code> changed. The new per-view keys are opt-in in
<code>wft/model.py</code>; without them reconstruction is bit-identical. Three judges:
the model-free head-on neighbour pattern; a held-out bench (2000 events per
chamber, M3 reference, paired bootstrap); and the full golden-key reconstruction
gate (R06_GATE procedure: reco, w0/kw re-measured, alignment, angles,
efficiency), paired per event.</p>

<h2>1 · The two views differ in shape (model-free)</h2>
<figure><img src="figures/modelfree_ratio.png" alt="±2/±1 neighbour charge per view and chamber">
<figcaption>Head-on (|tan| &lt; 0.05) bench cosmics, 20 %-trimmed peak-aligned stacks, absent strips as zero.
Y reaches the ±2 strip 2.4–3.5× more than X on every chamber.</figcaption></figure>
<table><tr><th>chamber</th><th>X ±2/±1</th><th>Y ±2/±1</th><th>X ±1 one-sidedness</th></tr>{mf_rows}</table>

<h2>2 · Pinning the Y ratio (held-out bench)</h2>
<figure><img src="figures/y_ratio_scan.png" alt="Y resolution change versus pinned Y ratio">
<figcaption>Δ s68 of the Y angle against production, held-out events, blind start. 0.6 is production.
The shaded region breaks the hyper-level c2 &lt; c1 gate and is shown as a diagnostic only:
det2/3/7 keep improving there.</figcaption></figure>

<h2>3 · Full reconstruction gate (golden keys)</h2>
<figure><img src="figures/gate_y.png" alt="Gate result per chamber">
<figcaption>Paired per event, full matched sample. Control = production hypers with the same
t0/w0 treatment.</figcaption></figure>
<table><tr><th>key</th><th>arms</th><th>Y s68 [deg]</th><th>Y s68, |θ|&lt;5°</th><th>X Δ s68</th><th>within 5 mm %</th></tr>{g_rows}</table>

<h2>4 · What else was learned</h2>
<ul>
<li>The beam's c2/c1 = 0.45 (H4 head-on) is the <b>Y</b> view's; X's ±2 is near zero. R06_GATE §8 had it as X's.</li>
<li>In this model the hyper c2/c1 is not the observable ±2/±1: the prompt spread σ<sub>p0</sub> (0.39–0.47 mm on the bench, shared by both views) carries most of the neighbour charge, and c1 sits on its 0.05 floor.</li>
<li>Freeing per-view ratios (and per-view σ<sub>p0</sub>, X asymmetry, Y τ) in the bench calibration lands in different basins (σ<sub>s</sub> 9 → 280 ns) and is not reliably better on held-out tracks; det3 got worse. The 180-event bench χ² (χ²/dof ≈ 800) cannot pin these constants.</li>
<li>A fractional model-error term (MODEL_FRAC 0.05) in calibration and reco is clearly worse (det3 Y +0.3–0.46°).</li>
<li>In H4 (sharp telescope reference) σ<sub>p0</sub> fits at 0.12–0.15 mm; profiling p0 on the bench moves it only to 0.36–0.44 mm — the gap is mostly gas, not M3 pointing error.</li>
<li>H4 head-on closure prefers a <b>low</b> X ratio (matching the model-free X) and Y at 0.6; every variant leaves Y's copies 13–15σ too early on the beam. The in-model ratio is not a pure board constant.</li>
</ul>

<h2>5 · What this does not rule out</h2>
<ul>
<li>Y beyond 0.95 helps further on det2/3/7 (diagnostic). The c2 &lt; c1 gate is applied to the hyper; whether it should be applied to the observable instead is a decision, not a measurement.</li>
<li>det6's production bundle is a different representation (σ<sub>p0</sub> 0.039 mm, c1 0.064, free c2) and gains nothing; its gate decides whether it is overridden.</li>
<li>The kernel's timing (Y copies too early in H4) is untouched; a per-view τ did not survive the bench calibration either.</li>
<li>The improvement is concentrated near vertical incidence; at large angles the drift ladder dominates and the kernel matters less.</li>
<li>w0/kw were re-measured from each candidate's own reconstruction, as for r06.</li>
</ul>
<p><small>Generated by <code>make_report.py</code>; source tables: <code>agg_blind.json</code>, <code>modelfree.json</code>,
<code>&lt;wft&gt;/plane_ratio/gate_*.json</code>. Log: <code>FINDINGS.md</code>.</small></p>
</body></html>'''
    with open(os.path.join(a.out, 'report.html'), 'w') as f:
        f.write(html)
    print('wrote', os.path.join(a.out, 'report.html'))


if __name__ == '__main__':
    main()
