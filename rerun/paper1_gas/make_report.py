#!/usr/bin/env python3
"""make_report.py -- rerun/paper1_gas/report.html (and results/summary.json) from results/gas.json and figures/.

    pg_gas.py <fit dir> && pg_figs.py && make_report.py
"""
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'results', 'gas.json')))
T626 = 600.0
GOLD = {'det2': 'o22_long_det2', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}


def gold():
    out = {}
    for det, key in GOLD.items():
        p = next((p for p in D['points'] if p['key'] == key and p['arm'] == 'in30'), None)
        if p:
            out[det] = {k: p[k] for k in ('v', 'v_err', 'E', 'gap', 'water', 'water_lo', 'water_hi', 'water_gap_lo',
                                         'water_gap_hi', 'n')}
    sc = D['scan']['fixed']
    out['det3'] = dict(water=sc['water'], water_lo=sc['water_68'][0], water_hi=sc['water_68'][1], gap=sc['gap'],
                       chi2=sc['chi2'], ndf=sc['ndf'], source='6-27 drift scan, 500-1100 V, one water')
    return out


def summary():
    A = D['att_626']
    att = {}
    for det in ('det6', 'det7'):
        P = A[det]['points']
        att[det] = dict(k=A[det]['k'], k_err=A[det]['k_err'], k_chi2=A[det]['k_chi2'], k_n=A[det]['k_n'],
                        truncated=[dict(t_h=p['t'] - T626, frac=p['frac'], tau_ns=p['tau_ns'], o2_equiv_ppm=p['o2_equiv_ppm'])
                                   for p in P if 'tau_ns' in p and p['t'] - T626 < 5],
                        water_first=A[det]['water'][0]['water'], water_last=A[det]['water'][-1]['water'])
    return dict(golden_water=gold(), scan=D['scan'], att_626=att, shared_626=A['shared'], grid=D['grid'], v_sys=D['v_sys'])


def main():
    S = summary()
    json.dump(S, open(os.path.join(HERE, 'results', 'summary.json'), 'w'), indent=1, default=float)
    g, sh, sc = S['golden_water'], S['shared_626'], S['scan']
    a6 = S['att_626']['det6']['truncated']
    rows = ''.join(f"<tr><td>{d}</td><td>{x.get('E', float('nan')):.0f}</td><td>{x.get('v', float('nan')):.1f}"
                   f"{' ± %.1f' % x['v_err'] if 'v_err' in x else ''}</td><td><b>{x['water']:.2f}</b> "
                   f"[{x['water_lo']:.2f}, {x['water_hi']:.2f}]</td><td>{x['gap']:.2f}</td>"
                   f"<td>{('%.2f – %.2f' % (min(x['water_gap_lo'], x['water_gap_hi']), max(x['water_gap_lo'], x['water_gap_hi']))) if 'water_gap_lo' in x else '—'}</td></tr>"
                   for d, x in sorted(g.items()))
    att = ''.join(f"<tr><td>{p['t_h']:.1f}</td><td>{p['frac']:.2f}</td><td>{p['tau_ns']:.0f}</td><td>{p['o2_equiv_ppm']:.0f}</td></tr>"
                  for p in a6)
    H = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Bench gas: water and attachment (pass 1)</title>
<style>body{{font-family:sans-serif;max-width:980px;margin:2em auto;line-height:1.45}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:3px 8px;text-align:right}}img{{max-width:100%}}.v{{background:#eef6ee;padding:.6em 1em}}</style>
</head><body><h1>Bench gas against time and field (Paper I §6, pass 1)</h1>
<div class="v"><b>Verdict.</b> The bench drift velocity is the water.
<ul><li><b>Field:</b> one water describes det3's whole 6-27 drift scan, {sc['fixed']['water']:.2f} %
[{sc['fixed']['water_68'][0]:.2f}, {sc['fixed']['water_68'][1]:.2f}], χ² {sc['fixed']['chi2']:.1f}/{sc['fixed']['ndf']}.</li>
<li><b>Time:</b> the v(t) drifts read as water drying out overnight. On 6-26 det6 goes {S['att_626']['det6']['water_first']:.2f} →
{S['att_626']['det6']['water_last']:.2f} % and det7 {S['att_626']['det7']['water_first']:.2f} → {S['att_626']['det7']['water_last']:.2f} %.
The two follow one time shape (r = {sh['corr']:.2f}), and det6 carries {sh['excess_ratio_det6_over_det7']:.1f}× det7's excess.</li>
<li><b>Chambers differ:</b> each chamber has its own water (0.4–1.0 % on the golden rows), so it is per-chamber ingress, not one gas.</li>
<li><b>det6's ladder-vs-column disagreement is attachment.</b> For its first ~4 h on 6-26 the drift column is truncated (visible
fraction {a6[0]['frac']:.2f} → {a6[-1]['frac']:.2f}) by an electronegative contaminant of order {a6[-1]['o2_equiv_ppm']:.0f}–{a6[0]['o2_equiv_ppm']:.0f}
ppm O₂-equivalent (×2 model uncertainty). The M3 ladder does not see it; the timing stack and the column duration do.</li></ul></div>
<h2>Water per chamber (golden rows, timing v = gap / T, interior ≥ 30 mm, masked, pass template)</h2>
<table><tr><th>chamber</th><th>E (V/cm)</th><th>v (µm/ns)</th><th>water % [68 %]</th><th>gap (mm)</th><th>gap ± err band</th></tr>{rows}</table>
<p>det3 is the drift-scan fit (gap 28.19 mm, the S4 re-check interior gap). The gap profile gives {sc['profile']['gap']:.1f} mm
[{sc['profile']['gap_68'][0]:.0f}, {sc['profile']['gap_68'][1]:.0f}]: the drift scan alone does not fix the gap on pass 1.</p>
<h2>v(E): timing and angle-scale v against Magboltz</h2><img src="figures/gas_vE_timing.png">
<p>The angle-scale v (v_drift × kw, M3-calibrated) reaches the low fields, where the column does not fit in the window. The timing v
(gap / T) is a direct transit time. On det3 both sit on one Magboltz curve.</p>
<h2>6-26: two estimators, one gas</h2><img src="figures/v626.png">
<p>The timing stack is divided by the ladder on the rows after 06:00 and in the afternoon HV scan, where nothing truncates the column:
k = {S['att_626']['det6']['k']:.3f} ± {S['att_626']['det6']['k_err']:.3f} (det6, χ² {S['att_626']['det6']['k_chi2']:.1f}/{S['att_626']['det6']['k_n'] - 1}),
{S['att_626']['det7']['k']:.3f} ± {S['att_626']['det7']['k_err']:.3f} (det7).</p>
<img src="figures/water626.png"><img src="figures/attach626.png">
<table><tr><th>det6, h on 6-26</th><th>T / T_true</th><th>τ (ns)</th><th>O₂-eq (ppm)</th></tr>{att}</table>
<h2>Other nights</h2><img src="figures/nights.png">
<p>The same pattern on every night: water falls overnight and steps back up between runs or reconfigurations.
det4 goes 0.7 → 0.2 % over 6-23, then is back at 0.42 % on 6-24.</p>
<h2>What this does not rule out</h2><ul>
<li><b>No gas record.</b> No flow, fill or line record exists for the bench (no new data), so the cause of the water (outgassing after
opening, a permeable line) is not identified. Neither is the order of the chambers on the line.</li>
<li><b>The attachment number is model-dependent.</b> τ ≈ T / ln 2 assumes the box fit ends at the half-height point of an
exponentially decaying column. The stack model has no attachment term. The species is not identified: "O₂-equivalent" means the
rate, not the gas.</li>
<li><b>The gap is the leading systematic on each water</b> (the band column). The bench drift scan constrains it only to 28–30 mm.</li>
<li><b>g_det4's −9 % in-row v drift</b> (ladder and column, rerun/vt_all/CROSSCHECK) is not resolved by the stack thirds
(33.1 / 32.2 / 33.2 ± 1.2–2.9 µm/ns).</li>
<li><b>det2 sits near the velocity maximum at 321 V/cm,</b> where v hardly depends on water. Its water is the weakest of the five, and
one of its thirds has two Magboltz branches.</li></ul>
<p>Grid: Magboltz bench Ar/iso 95/5 + water + air, {D['grid']}; T systematic {D['v_sys'] * 100:.1f} %.</p></body></html>"""
    open(os.path.join(HERE, 'report.html'), 'w').write(H)
    print('report.html, results/summary.json')


if __name__ == '__main__':
    main()
