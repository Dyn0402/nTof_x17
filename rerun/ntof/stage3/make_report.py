#!/usr/bin/env python3
"""make_report.py -- vt_score/report.html: the n_TOF stage-3 v(t) correction and its check against the A-C line.

Reads vt_score/{line_campaign.json, line_campaign_subruns.csv, vt_kappa.json, vt_score.json} (vt_line_campaign.py,
vt_score.py) and writes report.html + figures/ beside them.  Generated: re-run after either script.

    make_report.py [--dir rerun/ntof/stage3/vt_score]
"""
import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)


def fig_line(d, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sept26_prelim_analysis import figstyle as F
    F.use()
    g = pd.read_csv(f'{d}/line_campaign_subruns.csv')
    g = g[(g.n >= 20) & g.vt_ratio.notna()]
    kap = json.load(open(f'{d}/vt_kappa.json'))['kappa']
    fits = json.load(open(f'{d}/line_campaign.json'))['fits']
    fig, axs = plt.subplots(1, 2, figsize=F.WIDE, sharey=True)
    for ax, arm in zip(axs, 'AC'):
        for view, mk in (('x', 'o'), ('y', 's')):
            s = g[(g.arm == arm) & (g.view == view)]
            ax.errorbar(s.vt_ratio, s.ratio, yerr=s.ratio_err, fmt=mk, ms=2.5, lw=0.5, alpha=0.45,
                        color=F.DET_COLOR[arm] if view == 'x' else F.MUTED, label=f'{arm} {view}')
            xs = np.linspace(s.vt_ratio.min(), s.vt_ratio.max(), 2)
            k = kap[f'{arm}{view}']['kappa']
            ax.plot(xs, k * xs, color=F.INK if view == 'x' else F.MUTED, lw=1.2,
                    ls='-' if view == 'x' else '--')
            f = fits[f'{arm}{view}']
            ax.text(0.03, 0.95 - 0.08 * (view == 'y'), f"{view}: slope {f['slope']:.2f} ± {f['slope_err']:.2f}, "
                    f"κ {k:.4f}", transform=ax.transAxes, fontsize=8, va='top')
        xs = np.linspace(g.vt_ratio.min(), g.vt_ratio.max(), 2)
        ax.plot(xs, xs, color=F.COPPER, lw=0.8, ls=':')
        ax.set_xlabel('v_t / v_bundle  (timing v, per sub-run)')
        ax.set_title(f'Chamber {arm}', loc='left')
        ax.legend(loc='lower right', fontsize=7, frameon=False)
    axs[0].set_ylabel('A-C line: fitted / true tan')
    fig.tight_layout()
    os.makedirs(f'{out}/figures', exist_ok=True)
    fig.savefig(f'{out}/figures/line_vs_vt.png', dpi=160)
    plt.close(fig)


def table(rows, head):
    h = ''.join(f'<th>{html.escape(c)}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(HERE, 'vt_score'))
    a = ap.parse_args()
    d = a.dir
    fits = json.load(open(f'{d}/line_campaign.json'))
    kap = json.load(open(f'{d}/vt_kappa.json'))
    sc = json.load(open(f'{d}/vt_score.json'))
    fig_line(d, d)
    L, S = sc['line_run149'], sc['srcy_run145']
    t_fit = table([[k, v['n_subruns'], v['n_events'], f"{v['slope']:.3f} ± {v['slope_err']:.3f}",
                    f"{v['intercept_at_1']:.4f} ± {v['intercept_err']:.4f}", f"{v['chi2']:.0f} / {v['ndof']}"]
                   for k, v in fits['fits'].items()],
                  ['view', 'sub-runs', 'line events', 'slope (1 = timing v is the angle v)', 'ratio at v_t = v_bundle',
                   'χ² / dof'])
    t_kap = table([[k, f"{v['kappa']:.4f} ± {v['err']:.4f}", f"{v['chi2_dof']:.2f}",
                    f"{kap['syst_early_late'][k]['early']:.4f} / {kap['syst_early_late'][k]['late']:.4f}"]
                   for k, v in kap['kappa'].items()],
                  ['view', 'κ = v_angle / v_t', 'χ² / dof (slope 1)', 'early / late half'])
    t_line = table([[k] + [f"{L[v][k]['ratio']:.4f} ± {L[v][k]['ratio_err']:.4f}"
                           for v in ('frozen', 'vt_nokappa', 'vt', 'vt_acc05')] for k in ('Ax', 'Ay', 'Cx', 'Cy')],
                   ['view', 'frozen v', 'v(t), no κ', 'v(t) with κ', 'v(t) with κ, |tan| < 0.5'])
    t_src = table([[v] + [f"{S[v][arm]['y_cross_mm']} ± {S[v][arm]['err_mm']} (n {S[v][arm]['n']})" for arm in 'AC']
                   for v in ('frozen', 'frozen_acc05', 'vt')],
                  ['columns', 'A (ref −31)', 'C (ref −35)'])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>n_TOF stage-3 v(t)</title><style>
:root{{--ink:#1b2430;--muted:#6a7583;--bg:#fff;--line:#c9ced6}}
@media (prefers-color-scheme:dark){{:root{{--ink:#e6e9ee;--muted:#9aa4b1;--bg:#161a20;--line:#39414c}}}}
body{{font:15px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg);max-width:980px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;margin:8px 0 18px;font-size:13px}}td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;background:#fff}}.muted{{color:var(--muted)}}
.verdict{{border-left:4px solid #8a3f8f;padding:6px 12px;margin:12px 0}}</style></head><body>
<h1>n_TOF stage 3: the v(t) angle correction</h1>
<div class="verdict"><b>Verdict.</b> The angle scale follows the timing v(t) in time (slope ≈ 1 on all four views over
~240 beam-off sub-runs), but sits a constant 2–3 % below it. The table correction is therefore
<code>tan_vt = tan · v_bundle / (κ · v_t)</code> with one κ per chamber and view; without κ it overshoots
(run_149 C: 1.04 steep → 0.98 shallow). With κ the A–C line reads 1.000–1.004 on run_149 and the beam source y is
unchanged.</div>
<h2>Does the angle scale follow v(t)?</h2>
<p>Per beam-off cosmic sub-run: the A–C through-going line (built from mesh positions only, so independent of v)
against the chamber's fitted tan on |true tan| 0.15–0.45, versus the sub-run's smooth-model v_t / v_bundle.</p>
<img src="figures/line_vs_vt.png" alt="line ratio vs v_t">
<p class="muted">Dots: sub-runs (x coloured, y grey). Lines: κ · (v_t / v_bundle), slope fixed at 1. Dotted: κ = 1.</p>
{t_fit}
<h2>κ = v_angle / v_t (slope fixed at 1)</h2>{t_kap}
<p class="muted">D has no A–C line: κ_D = 1 (timing ratio only); D's absolute angle scale needs an anchor.</p>
<h2>run_149 A–C line, frozen vs corrected</h2>{t_line}
<p class="muted">run_149's 87 sub-runs are part of κ's fit (174 of 476 sub-run views), so the corrected line is not
independent. The slope above and the beam source y below are.</p>
<h2>run_145 beam: late source-y band crossing</h2>{t_src}
<h2>What this does not rule out</h2><ul>
<li>κ is measured on cosmic triggers and applied to beam rows; a beam/cosmic difference in the timing–angle relation
would not show here (the class offset in v_t itself is ~1 %, opposite sign in A and C).</li>
<li>κ drifts by 0.3–0.4 % between the campaign's two halves (~3σ): a sub-percent systematic on the angle scale.</li>
<li>The ~1 % candidate reshuffle a v(t) re-reconstruction would make at the fastest times is not undone by a table
correction (A/B 4677861).</li>
<li>D: no angle truth, κ_D = 1; D beam v_t is predicted from D's cosmic sub-runs.</li></ul>
</body></html>"""
    open(f'{d}/report.html', 'w').write(page)
    print(f'{d}/report.html')


if __name__ == '__main__':
    main()
