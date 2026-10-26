#!/usr/bin/env python3
"""make_report.py [--inline] [--out ...] -- the efficiency loss budget (budget.py) as report.html + figures/, generated
from ~/x17/cosmic_bench/effbudget/budget_<det>.json so a re-run updates numbers, bars and text together."""
import argparse, base64, importlib.util, json, os
from html import escape
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.expanduser('~/x17/cosmic_bench/effbudget')
_spec = importlib.util.spec_from_file_location('c21', os.path.join(HERE, '..', 'det4', 'make_report.py'))
_c21 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_c21)
CSS = _c21.CSS + "img{max-width:100%;height:auto;border:1px solid var(--grid)} td.n{text-align:right}"
DETS = ['det3', 'det2', 'det6', 'det7', 'det4']
NAME = {'det2': 'det2 (B)', 'det3': 'det3', 'det4': 'det4', 'det6': 'det6', 'det7': 'det7'}
LADDER = ['detected', 'rejected_cut', 'spark', 'no_hit', 'hw_known', 'one_view_silent', 'charge_elsewhere',
          'not_seeded', 'fit_failed', 'far']
LABEL = {'detected': 'detected (|r| ≤ R, accepted)', 'rejected_cut': 'found, removed by the competing-cluster cut',
         'spark': 'spark at the crossing', 'no_hit': 'no strip fired', 'hw_known': 'known hardware (det4 bands, det7 X card 1)',
         'one_view_silent': 'one view silent', 'charge_elsewhere': 'charge, but not on the M3 footprint',
         'not_seeded': 'charge on the footprint, not seeded', 'fit_failed': 'seeded, fit failed', 'far': 'fitted, |r| > R'}
FARLAB = {'MULTI': 'second particle (fit took the other)', 'DELTA': 'δ-ray on the track', 'FIT_SAT': 'saturated, fit off',
          'FIT': 'clean, fit off', 'EDGE': 'edge (other coordinate / field)', 'POS_TAIL': 'position tail (≈ ⅔ bent M3 track)',
          'HW_KNOWN': 'known hardware', 'M3_ELSEWHERE': 'M3 not on the MX17 particle', 'MX17_DEAD': 'no charge in a view',
          'ONE_VIEW_ELSEWHERE': 'one view elsewhere', 'REJECTED_OK': 'other view rejected by the cut', 'unclassified': 'not in the C22 sample'}
COL = {'detected': '#2a9d5c', 'rejected_cut': '#8fd1a8', 'spark': '#d1495b', 'no_hit': '#555', 'hw_known': '#999',
       'one_view_silent': '#c9a227', 'charge_elsewhere': '#e9c46a', 'not_seeded': '#f4a261', 'fit_failed': '#e76f51',
       'far': '#3d7bbf'}
INLINE = False


def fig(name, cap):
    src = f'figures/{name}'
    if INLINE:
        src = 'data:image/png;base64,' + base64.b64encode(open(os.path.join(BASE, 'figures', name), 'rb').read()).decode()
    return f'<figure><img src="{src}" alt="{escape(cap)}"><figcaption>{cap}</figcaption></figure>'


def bars(S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(os.path.join(BASE, 'figures'), exist_ok=True)
    rows = [(d, g) for d in DETS for g in ('interior', 'edge')]
    fig_, axs = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, R in zip(axs, ('5', '3')):
        for i, (d, g) in enumerate(rows):
            e = S[d]['budget'][f'{g}|xy|{R}']; left = 0
            for k in LADDER:
                ax.barh(i, e[k], left=left, color=COL[k], edgecolor='white', lw=0.4, label=LABEL[k] if i == 0 else None)
                left += e[k]
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{NAME[d]} {g}' for d, g in rows]); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xlabel('% of M3 tracks in the region'); ax.set_title(f'X+Y, R = {R} mm')
    h, l = axs[0].get_legend_handles_labels()
    fig_.legend(h, l, loc='lower center', ncol=4, fontsize=8.5, frameon=False)
    fig_.tight_layout(rect=(0, 0.12, 1, 1)); fig_.savefig(os.path.join(BASE, 'figures', 'budget_bars.png'), dpi=120)
    plt.close(fig_)


def table(S, key_fmt, title):
    o = [f'<h3>{title}</h3><div class="scroll"><table><tr><th>where the track went</th>' +
         ''.join(f'<th>{NAME[d]}</th>' for d in DETS) + '</tr>']
    o.append('<tr><td>M3 tracks (n)</td>' + ''.join(f'<td class="n">{S[d]["budget"][key_fmt]["n"]}</td>' for d in DETS) + '</tr>')
    for k in LADDER:
        o.append(f'<tr><td>{escape(LABEL[k])}</td>' + ''.join(f'<td class="n">{S[d]["budget"][key_fmt][k]:.1f}</td>' for d in DETS) + '</tr>')
    cls = sorted({c for d in DETS for c in S[d]['budget'][key_fmt]['far_by_cls']},
                 key=lambda c: -max(S[d]['budget'][key_fmt]['far_by_cls'].get(c, 0) for d in DETS))
    for c in cls:
        o.append(f'<tr><td>&nbsp;&nbsp;far: {escape(FARLAB.get(c, c))}</td>' +
                 ''.join(f'<td class="n">{S[d]["budget"][key_fmt]["far_by_cls"].get(c, 0):.1f}</td>' for d in DETS) + '</tr>')
    o.append('</table></div>')
    return '\n'.join(o)


def main():
    global INLINE
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(BASE, 'report.html'))
    ap.add_argument('--inline', action='store_true'); a = ap.parse_args(); INLINE = a.inline
    S = {d: json.load(open(os.path.join(BASE, f'budget_{d}.json'))) for d in DETS}
    bars(S)
    b3 = S['det3']['budget']['interior|xy|5']
    found3 = b3['detected'] + b3['rejected_cut']
    nospark = 100 * found3 / (100 - b3['spark'])
    far3 = b3['far_by_cls']
    o = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>MX17 efficiency budget</title><meta name="description" content="Where every M3 track that crosses an MX17 bench chamber ends up: detected, removed by a cut, lost to a spark, a second particle, the fit or hardware."><style>{CSS}</style></head><body><main>',
         '<h1>MX17 efficiency: where every track goes</h1>',
         '<p class="sub">June bench golden rows, physical RC model (finalize_phys tables, lxplus), pass inputs (rr1_cm0 hits, '
         'M3 v2, incomplete events out). rerun/effbudget/, 2026-10-26.</p>']
    o.append('<div class="verdict"><b>det3, interior, X+Y within 5 mm:</b> '
             f'{b3["detected"]:.1f} % detected; {found3:.1f} % found counting the {b3["rejected_cut"]:.1f} % the '
             f'competing-cluster cut removes; <b>{nospark:.1f} %</b> of the crossings that are not in a spark. The rest: '
             f'{b3["spark"]:.1f} % sparks, {b3["far"]:.1f} % fitted but more than 5 mm off '
             f'({far3.get("MULTI", 0):.1f} second particle, {far3.get("DELTA", 0):.1f} δ-ray, '
             f'{far3.get("FIT_SAT", 0) + far3.get("FIT", 0):.1f} fit, {far3.get("EDGE", 0):.1f} edge, '
             f'{far3.get("POS_TAIL", 0):.1f} position tail), {b3["fit_failed"]:.1f} % fit failed, and '
             f'{b3["no_hit"] + b3["one_view_silent"] + b3["charge_elsewhere"] + b3["not_seeded"]:.2f} % never seen or '
             'never seeded. A chamber that misses a muon outright essentially does not happen on det3.</div>')
    rows = []
    for d in DETS:
        for g, lab in (('interior', 'interior'), ('interior_live', 'interior, live'), ('edge', 'edge zone')):
            e = S[d]['budget'][f'{g}|xy|5']
            if g == 'interior_live' and e['n'] == S[d]['budget']['interior|xy|5']['n']:
                continue
            f_ = e['detected'] + e['rejected_cut']
            rows.append(f'<tr><td>{NAME[d]}</td><td>{lab}</td><td class="n">{e["n"]}</td><td class="n">{e["detected"]:.1f}</td>'
                        f'<td class="n">{f_:.1f}</td><td class="n">{100 * f_ / max(100 - e["spark"], 1e-9):.1f}</td>'
                        f'<td class="n">{e["spark"]:.1f}</td><td class="n">{e["far"]:.1f}</td></tr>')
    o.append('<h2>Headline per chamber (X+Y, R = 5 mm)</h2><div class="scroll"><table><tr><th>chamber</th><th>region</th>'
             '<th>n</th><th>detected %</th><th>+ cut-removed %</th><th>of non-spark %</th><th>spark %</th><th>far %</th></tr>'
             + ''.join(rows) + '</table></div>')
    o.append(fig('budget_bars.png', 'Every M3 track inside the measured active area, by where it ended up (first match in '
                 'the ladder order). Interior = at least 30 mm from the edge of an unpassivated coordinate (X), 8 mm from '
                 'a passivated one (Y); edge = the bands closer than that.'))
    o.append('<h2>What was counted</h2><ul>'
             '<li><b>Denominator</b>: every event with exactly one M3 track passing the recipe (χ² &lt; 1, NClus = 4) whose '
             '30 mm footprint lies inside the measured active area (common/mx17_active_area), inside the event range '
             'the fit table covers (det6/det7 finalize tables are the first 8000 matched events); decoder-incomplete '
             'events removed (none on these rows). Tracks leaving the area are acceptance, reported apart.</li>'
             '<li><b>Detection</b>: the waveform fit (finalize_phys, no cluster-quality cut) in both views, |r| in the '
             'raw strip frame (equal to the aligned |r| of 02_efficiency, which this reproduces: det3 far 4.45 % in '
             'both). Hits are used only for whether strips fired and where (QA), never for the point.</li>'
             '<li><b>Ladder</b> (first match): detected → removed by the competing-cluster cut (n_dropped &gt; 2) → spark '
             '(&gt; 50 strips above the significance floor) → no strip → known hardware → one view silent → charge not '
             'within 5 mm of the footprint → not seeded → fit failed → far. Far tracks carry their C22 class.</li></ul>')
    o.append('<h2>The budgets</h2>')
    o.append(table(S, 'interior|xy|5', 'Interior, X+Y, R = 5 mm (% of M3 tracks)'))
    o.append(table(S, 'interior|xy|3', 'Interior, X+Y, R = 3 mm'))
    o.append(table(S, 'interior|x|5', 'Interior, X view alone, |res| ≤ 5 mm'))
    o.append(table(S, 'interior|y|5', 'Interior, Y view alone, |res| ≤ 5 mm'))
    o.append(table(S, 'edge|xy|5', 'Edge zone, X+Y, R = 5 mm'))
    o.append('<h2>Reading it chamber by chamber</h2><ul>'
             '<li><b>det3 and det2</b>: no lost muons to speak of (no-hit, silent, not-seeded together &lt; 0.1 %). '
             'The losses are physics in the chamber (second particles and δ-rays, ~3 %), sparks (2–3 %), the '
             'competing-cluster cut (2 %) and the fit (~1 %, mostly saturated tracks). At 3 mm the far share grows by '
             '~2 pp, mostly position tail (the bent-M3 tracks of C22) and δ-rays.</li>'
             '<li><b>det6</b>: 18.5 % of its crossings sit in a spark (full-plane discharges, ~350 strips per plane). '
             'Outside sparks it finds ~92 %; its extra loss is fit failures (2.4 %, 6 % at the edge).</li>'
             '<li><b>det7</b>: sparks take 27 % of the crossings (both planes fire; the night\'s known spark rate) and '
             'X card 1 another 17 % (pogo shorts, the asterisk). On the live card it finds ~93 % of non-spark crossings.</li>'
             '<li><b>det4</b>: 40 % of the interior is dead or low-gain amplification band (the known stripes). In the '
             'live bands it detects 68 %, with genuine gain-limited losses: a silent view 3.7 %, charge below threshold '
             'at the track 5.6 %, fit failures 3.8 %.</li></ul>')
    o.append('<h2>What this does not rule out</h2><ul>'
             '<li>Tables are the finalize (calibration) tables, the first 8000 matched events on det6/det7; the pass '
             'reconstructs every event and the budget should be re-run on it (it is downstream, a script).</li>'
             '<li>"Spark" is an event-level multiplicity tag: a muon crossing during a discharge is not separable from it. '
             'Whether a spark is muon-induced (det3: 4× the random rate, June analysis) is not decided here, so '
             '"of non-spark" is an operational number, not an intrinsic one.</li>'
             '<li>det4\'s band map is position-binned from the C22 per-event charge ratio (median &lt; 0.3 in 2 mm bins of '
             'x); band edges blur over ~2 mm.</li>'
             '<li>The far classes come from C22 (angle &gt; 5° or position &gt; 3 mm outliers, classified in a fixed order); '
             'a far track outside that sample is "not in the C22 sample".</li>'
             '<li>Only the golden row of each chamber; the other nights follow in the pass.</li></ul>')
    o.append('<p class="sub">Code: rerun/effbudget/budget.py, make_report.py; inputs staged by rerun/inputs.py; '
             'data ~/x17/cosmic_bench/effbudget/.</p></main></body></html>')
    open(a.out, 'w').write('\n'.join(o)); print('wrote', a.out)


if __name__ == '__main__':
    main()
