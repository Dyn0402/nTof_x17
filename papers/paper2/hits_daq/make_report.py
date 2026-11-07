#!/usr/bin/env python3
"""make_report.py [--tag ana1106] [--out DIR] -- the Paper II hits-DAQ section report (report.html + figures/*.png).

Built only from the score JSONs in results/ (regenerate those first, see the commands at the bottom of report.html):
  <tag>_tiers_<row>.json   tiers.py --from-tables         (V families x naive / H0 / H2a, W; det3 Run 1, Run 2)
  <tag>_h2c_<row>.json     score_h2c.py --bins --json     (H2c, W, per |tan| bin)
  <tag>_sps_score.json     sps_score.py                   (SPS det4 known angle, hits vs W on the same planes)
  <tag>_resim_score.json   resim_score.py                 (Arm S: 14 kernel variants, exact truth)
Re-running after the analysis updates numbers, figures and the verdict text together.
"""
import argparse
import html
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']          # categorical, fixed order
INK, MUTED, GRID = '#0b0b0b', '#6b6a64', '#e4e3df'
ROWS = {'sat_det3': 'det3 Run 1', 'p2_det3': 'det3 Run 2'}
BINS = ['0.0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5']
RC = [('rc0', 0), ('rc025', 0.25), ('rc05', 0.5), ('base', 1), ('rc2', 2), ('rc5', 5), ('rc10', 10)]


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def f2(x, p='.2f'):
    return format(x, p) if isinstance(x, (int, float)) else '–'


def det3_tiers(T, H):
    """{row: {view: [(label, s68, steep, in5)]}} for the headline tiers"""
    out = {}
    for row in ROWS:
        if T.get(row) is None or H.get(row) is None:
            continue
        out[row] = {}
        for v in ('x', 'y'):
            V = T[row]['views'][v]
            tt = V['tiers']
            best = min((n for n in tt if tt[n]['H2a']['all']['s68']), key=lambda n: tt[n]['H2a']['all']['s68'])
            h = H[row][f'{row}/{v}']
            L = [('V0 naive', tt['V0_dream']['naive']['all']), ('V0 H0', tt['V0_dream']['H0']['all']),
                 ('V0 H2a', tt['V0_dream']['H2a']['all']), (f'{best.split("_")[0]} H2a (best)', tt[best]['H2a']['all']),
                 ('V0 H2c', h['H2c']), ('W (waveform fit)', V['W']['score']['all'])]
            out[row][v] = [(n, s['s68'], s['steep'], s['frac_in5']) for n, s in L]
    return out


def fig_tiers(D, path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, v in zip(axs, ('x', 'y')):
        for i, (row, lab) in enumerate(ROWS.items()):
            if row not in D:
                continue
            L = D[row][v]
            ys = [k + (0.12 if i else -0.12) for k in range(len(L))]
            ax.plot([s for _, s, _, _ in L], ys, 'o', ms=7, color=C[i], label=lab, mec='white', mew=1.5)
        ax.set_yticks(range(len(L)))
        ax.set_yticklabels([n for n, *_ in L])
        ax.invert_yaxis()
        ax.set_xlim(0, None)
        ax.set_xlabel('angle s68 vs M3 [deg]', color=INK, fontsize=9)
        ax.set_title(f'{v.upper()} view', fontsize=10, color=INK, loc='left')
        style(ax)
    axs[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bins(T, H, path, row='sat_det3'):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, v in zip(axs, ('x', 'y')):
        tt = T[row]['views'][v]['tiers']['V0_dream']
        S = [('V0 naive', tt['naive']), ('V0 H2a', tt['H2a']), ('V0 H2c', H[row][f'{row}/{v}'].get('H2c_bins', {})),
             ('W', T[row]['views'][v]['W']['score'])]
        for i, (lab, sc) in enumerate(S):
            ys = [sc.get(b, {}).get('steep') for b in BINS]
            ax.plot(range(len(BINS)), [y if y is not None else float('nan') for y in ys], '-o', lw=2, ms=7,
                    color=C[i], label=lab, mec='white', mew=1.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS, fontsize=8)
        ax.set_xlabel('|tan θ_ref| bin', color=INK, fontsize=9)
        ax.set_title(f'{v.upper()} view, {ROWS[row]}', fontsize=10, color=INK, loc='left')
        style(ax)
    axs[0].set_ylabel('steep: median signed error [deg]\n(> 0 = reads too steep)', color=INK, fontsize=9)
    axs[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_resim(S, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    V = S['y']['variants']
    xs = range(len(RC))
    for i, (lab, get) in enumerate([('V0 DREAM hits', lambda o: o['fams']['V0_dream']['ratio']),
                                    ('V6 ideal single time', lambda o: o['fams']['V6_ideal']['ratio']),
                                    ('W waveform fit', lambda o: o['W']['ratio'])]):
        ax.plot(xs, [get(V[k]) for k, _ in RC], '-o', lw=2, ms=7, color=C[i], label=lab, mec='white', mew=1.5)
    for k, lab, ls in (('metal', 'metallic limit (no RC, floor footprint), V0', '--'), ('point', "'point' (strip width only), V0", ':')):
        ax.axhline(V[k]['fams']['V0_dream']['ratio'], color=MUTED, lw=1.2, ls=ls)
        ax.text(len(RC) - 1, V[k]['fams']['V0_dream']['ratio'] + 0.03, lab, ha='right', fontsize=8, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'{f:g}×' for _, f in RC])
    ax.set_xlabel('Y resistive spreading rc_D_y (× the measured det3 value)', color=INK, fontsize=9)
    ax.set_ylabel('slope / true slope (Y view)', color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sps(P, path):
    rows = [k for k, r in P['rows'].items() if r['inclined']]
    names = ['V0_dream', 'V0r_dream', 'V1_peak', 'V2_thr', 'V3_tot', 'V4_fd', 'V5_binary', 'V6_ideal', 'V0/H2c_dream']
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, k in enumerate(rows):
        t = P['rows'][k]['views']['y']['tiers']
        ys = [n + (i - 1) * 0.15 for n in range(len(names))]
        ax.plot([t[n]['ratio'] for n in names], ys, 'o', ms=7, color=C[i], mec='white', mew=1.5,
                label=f'{k.replace("__", " ")} (v {P["rows"][k]["v_mm_ns"] * 1e3:.1f} µm/ns)')
    ax.axvline(1, color=MUTED, lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('hits slope / waveform-fit slope, same plane (inclined Y view)', color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(head, rows):
    h = ''.join(f'<th>{html.escape(x)}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


CSS = """body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb;line-height:1.45}
h1{font-size:1.5em}h2{font-size:1.15em;margin-top:1.8em;border-bottom:1px solid #e4e3df}
table{border-collapse:collapse;font-size:.86em;margin:.6em 0}td,th{border:1px solid #e4e3df;padding:3px 8px;text-align:right}
th{background:#f1f0ec}td:first-child,th:first-child{text-align:left}img{max-width:100%}
.verdict{background:#eef4fc;border-left:4px solid #2a78d6;padding:10px 14px}.cap{color:#6b6a64;font-size:.88em}
code{background:#f1f0ec;padding:0 3px}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ana1106')
    ap.add_argument('--out', default=os.path.join(HERE, 'report'))
    a = ap.parse_args()
    R = os.path.join(HERE, 'results')
    T = {r: j(os.path.join(R, f'{a.tag}_tiers_{r}.json')) for r in ROWS}
    H = {r: j(os.path.join(R, f'{a.tag}_h2c_{r}.json')) for r in ROWS}
    P = j(os.path.join(R, f'{a.tag}_sps_score.json'))
    S = j(os.path.join(R, f'{a.tag}_resim_score.json'))
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    D = det3_tiers(T, H)
    fig_tiers(D, os.path.join(a.out, 'figures', 'det3_tiers.png'))
    fig_bins(T, H, os.path.join(a.out, 'figures', 'det3_steep_bins.png'))
    if S:
        fig_resim(S, os.path.join(a.out, 'figures', 'resim_rc_scan.png'))
    if P:
        fig_sps(P, os.path.join(a.out, 'figures', 'sps_ratio.png'))

    r1 = D['sat_det3']
    w = {v: r1[v][-1][1] for v in 'xy'}
    h2c = {v: r1[v][-2][1] for v in 'xy'}
    nv = {v: r1[v][0][1] for v in 'xy'}
    V = S['y']['variants'] if S else None
    parts = [f'<h1>Paper II §6: what a hits-only readout achieves (tag {a.tag})</h1>',
             '<div class="verdict"><b>Verdict.</b> On det3, the best hits-only reconstruction (H2c, a forward fit of the '
             f'emulated hit list) reaches X {f2(h2c["x"])}° / Y {f2(h2c["y"])}° against the waveform fit\'s '
             f'{f2(w["x"])}° / {f2(w["y"])}° (naive hits {f2(nv["x"])}° / {f2(nv["y"])}°). The remaining gap sits in '
             'near-normal tracks. '
             + (f'In the re-simulation the hits compression does not go away without the resistive layer: Y '
                f'{f2(V["base"]["fams"]["V0_dream"]["ratio"], ".2f")} at the measured RC, '
                f'{f2(V["rc0"]["fams"]["V0_dream"]["ratio"], ".2f")} with no RC, {f2(V["metal"]["fams"]["V0_dream"]["ratio"], ".2f")} '
                f'in the metallic limit; the waveform fit reads 0.99–1.00 everywhere.' if V else '')
             + '</div>']

    parts.append('<h2>1. det3, every tier, same planes (Arm D, M3 reference)</h2>')
    parts.append('<img src="figures/det3_tiers.png" alt="det3 s68 per tier"><p class="cap">Angle s68 within ±5° of M3, '
                 'both Paper I runs, every plane (Run 1 6.2k, Run 2 19.2k per view). H0/H2a are reference-trained '
                 '(event-id parity cross-validation); naive and H2c are reference-free.</p>')
    for row, lab in ROWS.items():
        if row in D:
            parts.append(f'<p><b>{lab}</b></p>' + table(['tier', 'X s68', 'X steep', 'X in5', 'Y s68', 'Y steep', 'Y in5'],
                         [[n, f2(sx), f2(tx, '+.2f'), f2(ix, '.3f'), f2(sy), f2(ty, '+.2f'), f2(iy, '.3f')]
                          for (n, sx, tx, ix), (_, sy, ty, iy) in zip(D[row]['x'], D[row]['y'])]))
    parts.append('<img src="figures/det3_steep_bins.png" alt="steep per angle bin"><p class="cap">Signed error per '
                 '|tan θ| bin (Run 1). The hits tiers\' excess is near normal incidence; H2c removes most of it on '
                 'inclined tracks and keeps +0.8 (X) / +1.4 (Y) deg below |tan| 0.05: the far-strip timing of '
                 '<code>far_strip_timing.py</code>.</p>')

    if P:
        parts.append('<h2>2. SPS det4 at a known 25.4° stage angle (no reference track)</h2>')
        parts.append('<img src="figures/sps_ratio.png" alt="SPS ratio"><p class="cap">Hits slope over the waveform '
                     'fit\'s slope on the same plane, inclined Y view, CO₂ rows (v set per row from the stage angle '
                     'through W). The 0° row (run55) and the X view at these slow drift velocities are not scored.</p>')
        rows = []
        for k, r in P['rows'].items():
            if not r['inclined']:
                continue
            t = r['views']['y']['tiers']
            for n in ('V0_dream', 'V0r_dream', 'V4_fd', 'V6_ideal', 'V1_peak', 'V2_thr', 'V5_binary', 'V0/H2c_dream'):
                s = t[n]['score']['all']
                rows.append([k, n, f2(t[n]['ratio'], '.3f'), f2(s['s68']), f2(s['steep'], '+.2f'), f2(s['frac_in5'], '.2f')])
            rows.append([k, 'W', '1 (calibration)', f2(r['views']['y']['W']['all']['s68']), '–',
                         f2(r['views']['y']['W']['all']['frac_in5'], '.2f')])
        parts.append(table(['row', 'tier', 'ratio to W', 's68', 'steep', 'in5'], rows))

    if S:
        parts.append('<h2>3. Arm S: re-simulation on real det3 events, exact truth</h2>')
        parts.append('<img src="figures/resim_rc_scan.png" alt="RC scan"><p class="cap">Each Run 1 plane regenerated '
                     'with its own fitted track and residual through the production model, kernel modified; median '
                     'slope / true slope over |tan| &gt; 0.1. The base kernel reproduces the data\'s compression '
                     '(model X 1.18 / Y 1.34 against P0\'s data X 1.19 / Y 1.37).</p>')
        rows = []
        for var, o in S['y']['variants'].items():
            ox = S['x']['variants'][var]
            rows.append([var, f2(ox['fams']['V0_dream']['ratio'], '.3f'), f2(ox['fams']['V6_ideal']['ratio'], '.3f'),
                         f2(ox['W']['ratio'], '.3f'), f2(o['fams']['V0_dream']['ratio'], '.3f'),
                         f2(o['fams']['V6_ideal']['ratio'], '.3f'), f2(o['W']['ratio'], '.3f')])
        parts.append(table(['variant', 'X V0', 'X V6', 'X W', 'Y V0', 'Y V6', 'Y W'], rows))

    parts.append('<h2>What this does not rule out</h2><ul>'
                 '<li>Arm D emulates the <i>logic</i> of VMM/TIGER/VFAT-like chips at DREAM shaping and 60 ns sampling, '
                 'not their analogue front ends; shaping is not yet an Arm S axis.</li>'
                 '<li>H2c on SPS is not tuned for those rows (its model takes v and t_rise from the bench bundle); its '
                 'SPS s68 (2.0–2.7°) is not a statement about hits at SPS.</li>'
                 '<li>On SPS the waveform fit is the calibration of v, so the SPS numbers are hits <i>relative to W</i>; '
                 'W\'s own known-angle accuracy (0.98 ± 0.03) is Paper I\'s, on CF₄ rows.</li>'
                 '<li>Arm S resolutions are on the model with exact truth: use its ratios and biases, not its absolute '
                 's68.</li>'
                 '<li>The near-normal Y excess (far strips early in data) is not in the model; its cause is a '
                 'hypothesis (prompt capacitive coupling), not shown.</li></ul>')
    parts.append('<h2>Regenerate</h2><pre>'
                 'tiers.py --from-tables tiers_&lt;row&gt;_p*.parquet --out results/&lt;tag&gt;_tiers_&lt;row&gt;.json\n'
                 'score_h2c.py "&lt;row&gt;=h2c_&lt;row&gt;_p*" --bins --json results/&lt;tag&gt;_h2c_&lt;row&gt;.json\n'
                 'sps_score.py --tables &lt;dir&gt; --json results/&lt;tag&gt;_sps_score.json\n'
                 'resim_score.py "&lt;dir&gt;/resim_*.parquet" --json results/&lt;tag&gt;_resim_score.json\n'
                 'make_report.py --tag &lt;tag&gt;</pre>')
    open(os.path.join(a.out, 'report.html'), 'w').write(
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Paper II hits DAQ</title><style>{CSS}</style></head><body>' + '\n'.join(parts) + '</body></html>')
    print('[report] wrote', os.path.join(a.out, 'report.html'))


if __name__ == '__main__':
    main()
