#!/usr/bin/env python3
"""make_report.py [--res results] [--out report] -- the SPS gas-fit report (report.html + figures/) from gasfit.json
and its systematic arms (results/gasfit_<arm>.json).  Re-run after gasfit.py; numbers, figures and verdict text
come from the JSONs."""
import argparse, glob, html, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
T = np.arange(64) * 60.0
MAIN = ['co2_run55', 'co2_run56', 'co2_run57', 'co2_run59', 'cf4_run62_ladder', 'cf4_run63_ladder', 'cf4_run63_flat', 'cf4_run71_raw']
WHEN = {'co2_run55': 'Sat 08-01 14:25', 'co2_run56': 'Sat 15:47', 'co2_run57': 'Sat 18:14', 'co2_run59': 'Sat 20:02',
        'cf4_run62_ladder': 'Sun 22:07', 'cf4_run63_ladder': 'Sun 23:53', 'cf4_run63_flat': 'Mon 01:00', 'cf4_run71_raw': 'Mon 05:22'}
ARMS = {'zs05': 'ZS distortion x0.5', 'zs15': 'ZS distortion x1.5', 'kp': 'field profile k = +0.1', 'km': 'k = -0.1',
        'sat': 'saturated events kept', 'h1': 'strip sums X+-1 / Y+-4', 'h4': 'strip sums X+-4 / Y+-12', 'raweta': 'unsmoothed Magboltz eta'}


def f2(x, n=2):
    return '—' if x is None else f'{x:.{n}f}'


def rng(r, n=2, k=1.0):
    return '—' if not r else f'{r[0] * k:.{n}f}–{r[1] * k:.{n}f}'


def fig_stacks(G, g, path):
    per = G['per_stack']; n = len(per); fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.2), squeeze=False)
    for ax, p in zip(axs[0], per):
        y, e = np.array(p['y']), np.array(p['e'])
        ax.fill_between(T, y - e, y + e, color='0.7', lw=0)
        ax.plot(T, y, 'k.', ms=3, label='data')
        ax.plot(T, p['model'], 'C0', lw=1.4, label='fit')
        ax.plot(T, p['model_noair'], 'C3--', lw=1.0, label='same water, no air')
        ax.set_title(f"{p['key'].split('__')[1]} {p['view']} {p['E']:.0f} V/cm", fontsize=8)
        ax.set_xlabel('t [ns]', fontsize=8); ax.tick_params(labelsize=7)
        ax.text(0.98, 0.95, f"χ² {p['chi2']:.0f}/{p['npt']}\nno air {p['chi2_noair']:.0f}", transform=ax.transAxes, ha='right',
                va='top', fontsize=7)
    axs[0][0].legend(fontsize=7, loc='lower left')
    fig.suptitle(f'{g}: water {G["water"]:.2f} %, O₂ {G["o2_ppm"]:.0f} ppm (gap {G["gap"]} mm)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_summary(R, path):
    gs = [g for g in MAIN if g in R['groups']]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, g in enumerate(gs):
        G = R['groups'][g]; c = 'C1' if g.startswith('co2') else 'C0'
        wr = G['water_68_gap'] or [G['water']] * 2; o = G['o2_68_gap'] or [G['o2_ppm']] * 2
        a1.errorbar(i, G['water'], yerr=[[G['water'] - wr[0]], [wr[1] - G['water']]], fmt='o', color=c)
        a2.errorbar(i, G['o2_ppm'], yerr=[[G['o2_ppm'] - o[0]], [o[1] - G['o2_ppm']]], fmt='o', color=c)
    for ax, lab in ((a1, 'water [%]'), (a2, 'O₂ [ppm] (air × 2095)')):
        ax.set_xticks(range(len(gs))); ax.set_xticklabels([f'{g}\n{WHEN[g]}' for g in gs], rotation=60, fontsize=7, ha='right')
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    a1.set_title('water (68 %, gap profiled 27.0–30.0 mm)', fontsize=9); a2.set_title('O₂ (68 %, gap profiled)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_transition(R, path):
    xs, o2, w = {'cf4': [], 'co2': []}, {'cf4': [], 'co2': []}, {'cf4': [], 'co2': []}
    chis = {'cf4': [], 'co2': []}
    for i in range(15):
        for b in ('cf4', 'co2'):
            G = R['groups'].get(f'trans_ov{i:02d}_{b}')
            if G:
                xs[b].append(0.6 + 0.504 * i); o2[b].append(G['o2_ppm']); w[b].append(G['water']); chis[b].append(G['chi2'] / G['ndf'])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for b, c in (('cf4', 'C0'), ('co2', 'C1')):
        axs[0].plot(xs[b], w[b], 'o-', color=c, label=f'{b.upper()} base'); axs[1].plot(xs[b], o2[b], 'o-', color=c)
        axs[2].plot(xs[b], chis[b], 'o-', color=c)
    axs[0].set_ylabel('water [%]'); axs[1].set_ylabel('O₂ [ppm]'); axs[2].set_ylabel('χ²/ndf')
    for ax in axs:
        ax.set_xlabel('h after the CO₂→CF₄ switch'); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8); fig.suptitle('run60 flush: each sub-run in either pure base (indicative only)', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--res', default=os.path.join(HERE, 'results'))
    ap.add_argument('--out', default=os.path.join(HERE, 'report'))
    a = ap.parse_args()
    R = json.load(open(os.path.join(a.res, 'gasfit.json'))); fd = os.path.join(a.out, 'figures'); os.makedirs(fd, exist_ok=True)
    arms = {k: json.load(open(p)) for k in ARMS for p in [os.path.join(a.res, f'gasfit_{k}.json')] if os.path.exists(p)}
    C = json.load(open(os.path.join(a.res, 'closure.json'))) if os.path.exists(os.path.join(a.res, 'closure.json')) else []
    V = json.load(open(os.path.join(a.res, 'verdict.json'))) if os.path.exists(os.path.join(a.res, 'verdict.json')) else {}
    fig_summary(R, os.path.join(fd, 'summary.png'))
    if any(k.startswith('trans_') for k in R['groups']):
        fig_transition(R, os.path.join(fd, 'transition.png'))
    for g in MAIN:
        if g in R['groups']:
            fig_stacks(R['groups'][g], g, os.path.join(fd, f'stacks_{g}.png'))
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>SPS gas fit (H2O + O2), final pass</title>',
         '<style>body{font-family:sans-serif;max-width:1150px;margin:auto;padding:1em}table{border-collapse:collapse;font-size:13px}'
         'td,th{border:1px solid #bbb;padding:3px 7px;text-align:right}th{background:#eee}td:first-child{text-align:left}'
         '.v{background:#f4f8ff;border-left:4px solid #36c;padding:.6em 1em}img{max-width:100%}</style></head><body>',
         '<h1>SPS gas composition (H₂O + O₂) from the final pass</h1>']
    if V.get('verdict'):
        H.append(f'<div class="v"><b>Verdict.</b> {html.escape(V["verdict"])}</div>')
    H.append('<h2>Compositions per epoch (gap profiled over 27.0–30.0 mm)</h2><table><tr><th>group</th><th>when</th><th>stacks</th>'
             '<th>water %</th><th>68 % (gap profiled)</th><th>O₂ ppm</th><th>68 % (gap profiled)</th><th>best gap</th>'
             '<th>gap 68 %</th><th>χ²/ndf</th><th>edge</th></tr>')
    for g in MAIN:
        G = R['groups'].get(g)
        if not G:
            continue
        H.append(f'<tr><td>{g}</td><td>{WHEN[g]}</td><td>{len(G["per_stack"])}</td><td>{f2(G["water"])}</td>'
                 f'<td>{rng(G["water_68_gap"])}</td><td>{G["o2_ppm"]:.0f}</td><td>{rng(G["o2_68_gap"], 0)}</td><td>{G["gap"]}</td>'
                 f'<td>{rng(G["gap_68"], 1)}</td><td>{G["chi2"]:.0f}/{G["ndf"]}</td>'
                 f'<td>{"water " if G["edge"]["water"] else ""}{"air" if G["edge"]["air"] else ""}</td></tr>')
    H.append('</table><p>68 % region: χ² ≤ min + 2.30 × max(1, χ²/ndf), over the (water, air) grid of every gap. '
             'O₂ = air × 2095 ppm (air = N₂/O₂/Ar 78.08/20.95/0.93 replacing argon).</p>')
    if arms:
        H.append('<h2>Systematic arms (best water % / O₂ ppm per group)</h2><table><tr><th>group</th><th>baseline</th>' +
                 ''.join(f'<th>{html.escape(ARMS[k])}</th>' for k in arms) + '</tr>')
        for g in MAIN:
            if g not in R['groups']:
                continue
            b = R['groups'][g]
            H.append(f'<tr><td>{g}</td><td>{b["water"]:.2f} / {b["o2_ppm"]:.0f}</td>' + ''.join(
                f'<td>{A["groups"][g]["water"]:.2f} / {A["groups"][g]["o2_ppm"]:.0f}</td>' if g in A['groups'] else '<td>—</td>'
                for A in arms.values()) + '</tr>')
        H.append('</table>')
    H.append('<h2>Figures</h2><p><img src="figures/summary.png"><br>Water and O₂ per group, in time order. Orange: CO₂ epoch '
             '(Ar/CO₂/iso 95/3/2); blue: CF₄ epoch (Ar/CF₄/iso 88/10/2).</p>')
    for g in MAIN:
        if os.path.exists(os.path.join(fd, f'stacks_{g}.png')):
            H.append(f'<p><img src="figures/stacks_{g}.png"><br>{g}: the head-on strip-sum stacks (grey: jackknife error), the fit, '
                     'and the fit\'s water with no air (red dashed): the late-charge loss the O₂ has to explain.</p>')
    if os.path.exists(os.path.join(fd, 'transition.png')):
        H.append('<p><img src="figures/transition.png"><br>run60 flush, indicative only: the base gas changes, the grid has no mixed '
                 'base.</p>')
    if C:
        H.append('<h2>Closure (toy stacks from the same model)</h2><table><tr><th>set</th><th>gap true / fit</th><th>truth</th>'
                 '<th>fitted water</th><th>fitted air</th><th>χ²/ndf</th></tr>')
        for c in C:
            H.append(f'<tr><td>{c["set"]}</td><td>{c["gap_true"]} / {c["gap_fit"]}</td><td>{c["truth"][0]} / {c["truth"][1]}</td>'
                     f'<td>{c["water"]:.3f}</td><td>{c["air"]:.3f}</td><td>{c["chi2"]:.0f}/{c["ndf"]}</td></tr>')
        H.append('</table>')
    for sec in ('method', 'degeneracies', 'limits'):
        if V.get(sec):
            H.append(f'<h2>{sec.capitalize()}</h2><ul>' + ''.join(f'<li>{html.escape(x)}</li>' for x in V[sec]) + '</ul>')
    H.append('</body></html>')
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, 'report.html'), 'w').write('\n'.join(H))
    print('wrote', os.path.join(a.out, 'report.html'))


if __name__ == '__main__':
    main()
