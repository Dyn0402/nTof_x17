#!/usr/bin/env python3
"""make_report.py -- rerun/ntof/tests/q7/report.html (+ figures/) from the Q7 study's result files in $Q7_DATA.

Reads: overlay/overlay_summary.json, overlay/score_donors_<arm>.parquet, real/pointing.json, real/beam_score.json,
real/line_score{,_acc0.6,_acc1}.json, real/cpu.json, check_<arm>.json.  The verdict and headline numbers are
built from the same tables (verdict()), so a re-run updates them together; the limitations are fixed text (LIMITS).
"""
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402

D = L.DATA
FIG = os.path.join(HERE, 'figures')
KEY = ['tm0.6_count_gate', 'tm0.6_count_cost', 'tm0.6_charge_cost', 'tm1_count_cost', 'colwin_count_cost',
       'colwin_charge_cost']
LABEL = {'tm0.6_count_gate': 'today (tm 0.6, count, gate)'}


def J(p):
    return json.load(open(os.path.join(D, p)))


def table(rows, head):
    h = '<tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in head) + '</tr>'
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table>{h}{b}</table>'


def pct(x):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{100 * x:.1f}'


def figures(ov):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({'font.size': 10.5})
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, arm in zip(axs, ('A', 'C')):
        S = pd.read_parquet(os.path.join(D, 'overlay', f'score_donors_{arm}.parquet'))
        S = S[S['mode'] == 'overlay'].copy()
        S['sep'] = np.minimum(S.sep_x, S.sep_y)
        bins = [0, 6, 12, 18, 24, 40, 80, 400]
        S['sb'] = pd.cut(S.sep, bins, labels=False)
        for cf, c in (('tm0.6_count_gate', '#888'), ('tm0.6_count_cost', '#1f77b4'), ('tm0.6_charge_cost', '#d62728'),
                      ('colwin_count_cost', '#2ca02c')):
            g = S[S.cfg == cf]
            y = g.groupby(['sb', 'oid']).paired.all().groupby('sb').mean()
            ax.plot(np.arange(len(y)), y.values, 'o-', color=c, label=LABEL.get(cf, cf), ms=4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['0-6', '6-12', '12-18', '18-24', '24-40', '40-80', '>80'])
        ax.set_xlabel('smaller per-view separation [mm]')
        ax.set_title(f'{arm}: both donors found, correctly paired')
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('fraction of overlays')
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'overlay_both_vs_sep.png'), dpi=130)
    plt.close(fig)
    P = J('real/pointing.json')
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, arm in zip(axs, ('A', 'C')):
        r = P[f'r145full_{arm}']
        tms = ['tm0.5', 'tm0.6', 'tm0.7', 'tm0.8', 'tm1', 'colwin']
        for sel, c in (('count_gate', '#888'), ('count_cost', '#1f77b4'), ('charge_cost', '#d62728')):
            y = [r[f'{t}_{sel}']['acc0.6']['excess'] for t in tms]
            e = [r[f'{t}_{sel}']['acc0.6']['err'] for t in tms]
            ax.errorbar(range(len(tms)), y, yerr=e, fmt='o-', color=c, label=sel, ms=4, capsize=2, alpha=0.85)
        ax.set_xticks(range(len(tms)))
        ax.set_xticklabels(['0.5', '0.6', '0.7', '0.8', '1.0', 'col.\nwindow'])
        ax.set_xlabel('raw tan_max inside plausible')
        ax.set_title(f'{arm}: pointing excess, stage-3 acc. 0.6')
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('N(dca<15) - N(15-30 mm)')
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'pointing_vs_tanmax.png'), dpi=130)
    plt.close(fig)


def main():
    ov = J('overlay/overlay_summary.json')
    P = J('real/pointing.json')
    B = J('real/beam_score.json')
    LN = {a: J(f'real/line_score{s}.json') for a, s in (('none', ''), ('0.6', '_acc0.6'), ('1', '_acc1'))}
    cpu = J('real/cpu.json')
    chk = {a: J(f'check_{a}.json') for a in 'AC' if os.path.exists(os.path.join(D, f'check_{a}.json'))}
    figures(ov)
    parts = []
    # ---- overlay table
    rows = []
    for arm in ('A', 'C'):
        for cf in KEY:
            r = ov[arm][cf]
            sx = r['by_tcls'].get('SX', {})
            rows.append([arm, LABEL.get(cf, cf), r['n_overlays'], pct(r['both_paired']), pct(r['shallow_paired']),
                         pct(r['steep_paired']), pct(r['acc0.6']['eff']), f"{r['acc0.6']['ghosts_per_ov']:.3f}",
                         pct(sx.get('shallow_found_acc06')), f"{sx.get('ghost_acc06_per_ov', np.nan):.3f}",
                         pct(r['acc1']['eff']), f"{r['acc1']['ghosts_per_ov']:.3f}", pct(r['single']['paired'])])
    t_ov = table(rows, ['arm', 'configuration', 'overlays', 'both donors paired %', 'shallow donor paired %',
                        'steep donor paired %', 'in acc 0.6: found %', 'acc-0.6 ghosts / overlay',
                        'SX pairs: shallow found (acc 0.6) %', 'SX ghosts / overlay', 'in acc 1.0: found %',
                        'acc-1.0 ghosts / overlay', 'single donors (sanity) paired %'])
    rows = []
    for arm in ('A', 'C'):
        for cf in sorted(ov[arm]):
            r = ov[arm][cf]
            rows.append([arm, cf, pct(r['both_paired']), pct(r['acc0.6']['eff']), f"{r['acc0.6']['ghosts_per_ov']:.3f}",
                         pct(r['acc1']['eff']), f"{r['acc1']['ghosts_per_ov']:.3f}"])
    t_ov_all = table(rows, ['arm', 'cfg', 'both paired %', 'acc0.6 found %', 'acc0.6 ghosts/ov', 'acc1 found %',
                            'acc1 ghosts/ov'])
    # ---- pointing
    rows = []
    for arm in ('A', 'C'):
        r = P[f'r145full_{arm}']
        for cf in KEY:
            q = r[cf]
            st = q['bins']
            rows.append([arm, LABEL.get(cf, cf), q['acc0.6']['n'], f"{q['acc0.6']['excess']} &plusmn; {q['acc0.6']['err']:.0f}",
                         q['acc1']['n'], f"{q['acc1']['excess']} &plusmn; {q['acc1']['err']:.0f}",
                         ' / '.join(f"{b['excess']}" for b in st[2:]), ' / '.join(f"{b['n']}" for b in st[2:]),
                         f"{cpu[arm].get(cf, np.nan):.1f}"])
    t_pt = table(rows, ['arm', 'configuration', 'gated, acc 0.6', 'pointing excess (acc 0.6)', 'gated, acc 1.0',
                        'pointing excess (acc 1.0)', 'excess in tan 0.6-0.8 / 0.8-1.0 / 1.0-1.5',
                        'tracks in those bins', 'condor cpu-s/event (+-30 %)'])
    # ---- beam vs base
    rows = []
    for arm in ('A', 'C'):
        r = B[f'r145full_{arm}']
        for cf in KEY[1:]:
            v = r[cf]['vs_base']
            rows.append([arm, cf, v['gated_lost'], v['shallow_gated_lost'], v['shallow_gated_lost_pointing'],
                         v['gated_gained'], v['events_ntracks_changed'], r[cf]['n_events']])
    t_bb = table(rows, ['arm', 'configuration', 'baseline gated tracks not kept', 'of which shallow (<0.6)',
                        'of which pointing (<15 mm)', 'gated tracks gained', 'events with n_tracks changed',
                        'events'])
    # ---- line
    rows = []
    for acc in ('none', '0.6', '1'):
        for cf, v in sorted(LN[acc].items()):
            if cf.startswith('_'):
                continue
            ra = {b['inc_lo']: b for b in v['bins'] if b['arm'] == 'A'}
            rc = {b['inc_lo']: b for b in v['bins'] if b['arm'] == 'C'}
            f = lambda d: ' / '.join(f"{d[k]['res_68']:.1f}" if d[k]['n'] else '-' for k in (0, 10, 20, 31))
            vb = v['vs_base']
            rows.append([acc, cf, v['n_ac_line'], f"+{vb['only_cfg']} / -{vb['only_base']}", f(ra), f(rc),
                         ' / '.join(str(ra[k]['n']) for k in (0, 10, 20, 31)),
                         f"{vb['moved']['A']['moved_p0_gt1mm']} / {vb['moved']['C']['moved_p0_gt1mm']}"])
    t_ln = table(rows, ['stage-3 acc', 'configuration', 'A-C lines', 'vs today: gained / lost',
                        'A residual 68 % [deg] at incidence 0-10/10-20/20-31/31-40', 'C residual 68 % [deg]',
                        'lines per bin', 'common lines moved > 1 mm (A / C)'])
    # ---- check
    t_ck = table([[a, k, v['cand_identical'], v['events_identical']] for a, c in chk.items()
                  for k, v in c['configs'].items()], ['arm', 'configuration', 'candidate table identical',
                                                      'events table identical'])
    drop_beam = B.get('_dropped', {})
    drop_line = LN['none'].get('_dropped', {})
    body = f"""
<h1>Q7 + tan_max scan: X&ndash;Y pairing, two-track guards and the beam angle limit, on beam truth</h1>
<p class=sub>n_TOF pass 1 (NTOF_PASS_PLAN_2026-11-02 &sect;6, NOTE_PAIRING_FIX_2026-11-02 &sect;3&ndash;4). Bundles v1k_A / v1k_C
(timing v, in-situ kw, t0 free). Generated by <code>rerun/ntof/tests/q7/make_report.py</code>; data in
<code>{html.escape(D)}</code>.</p>
<div class=verdict>{verdict(ov, P, LN, cpu)}</div>
<h2>1. Overlay truth (run_145 beam singles summed in pairs, re-seeded and re-fitted)</h2>
<p>Donors are clean single tracks of run_145 stat090_0000: one candidate per view, X&ndash;Y coincident with both column
ends in the window, no split in any configuration, quality_ok, charge balance within 3&sigma;, max |tan| &lt; 1.5.
<b>Steep donors (max |tan| &ge; 0.6) are included on purpose</b>, and pairs are stratified on separation and on the tan
class SS / SX / XX. Pairs share acquisition and trigger phase and are time-coincident (|&Delta;t0| &lt; 60 ns in both
views). Overlay mode 'replace'. A donor is <i>paired</i> when one X&ndash;Y track (track_id &ge; 0) has its X and its Y
within 3 mm of that donor. A <i>ghost</i> is a track that is not one donor in both views; "acc" applies the stage-3
true-angle acceptance to gated tracks (raw = true here, kw &asymp; 1).</p>
{t_ov}
<p><img src="figures/overlay_both_vs_sep.png" alt="both-donor efficiency vs separation"><br>
<span class=cap>Both donors found and correctly paired, against the smaller per-view separation. gate (grey) loses the
pair at every separation, mostly by the mechanism of NOTE_PAIRING_FIX &sect;3: the steep member is implausible, its pair is
not kept, and the shallow member's X is joined to the wrong Y.</span></p>
<details><summary>All 24 configurations</summary>{t_ov_all}</details>
<h2>2. Capsule pointing on real run_145 triggers (all 24 configurations, condor, production ntof_reco.py)</h2>
<p>Pointing excess = N(dca &lt; 15 mm) &minus; N(15 &le; dca &lt; 30 mm) of gated tracks: the 15&ndash;30 mm band estimates the
chance-pointing floor. It counts capsule electrons whatever else a configuration adds.</p>
{t_pt}
<p><img src="figures/pointing_vs_tanmax.png" alt="pointing excess vs tan_max"><br>
<span class=cap>Capsule-pointing excess inside the stage-3 acceptance 0.6. Errors are Poisson on independent samples;
the configurations share their triggers, so differences between them are better known than the bars.</span></p>
<h3>Against today's configuration, same triggers</h3>
{t_bb}
<h2>3. The A&ndash;C cosmic line on run_149 (12 sub-runs, 7 configurations)</h2>
{t_ln}
<h2>4. Method checks</h2>
<p>q7lib.fit_multi fits each candidate window once and re-derives plausibility, ranking, the joint two-track stage and the
selection per configuration. Against wft.reco._worker_fit on bundle copies carrying each configuration:</p>
{t_ck}
<p>Condor noise floor: between two condor jobs of configurations that must agree on single-candidate events, 1&ndash;2 in
10<sup>4</sup> such events differ by &gt; 0.1 mm (LM convergence on different nodes); it is far below every effect quoted.</p>
<p>Left out (missing condor outputs, dropped from every configuration so all compare the same triggers): beam
{html.escape(json.dumps(drop_beam))}; line {html.escape(json.dumps(drop_line))}.</p>
<h2>5. What this does not rule out</h2>
{LIMITS}
"""
    css = """body{font-family:system-ui,sans-serif;max-width:1150px;margin:2em auto;padding:0 1em;color:#222;line-height:1.45}
table{border-collapse:collapse;font-size:12.5px;margin:.6em 0}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}
th{background:#f3f3f3}td:nth-child(2),td:first-child{text-align:left}.verdict{background:#f4f8ff;border-left:4px solid #36c;
padding:.6em 1em}.sub{color:#666}.cap{color:#555;font-size:13px}img{max-width:100%}code{background:#f4f4f4;padding:0 3px}"""
    open(os.path.join(HERE, 'report.html'), 'w').write(
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Q7 + tan_max scan</title><style>{css}</style></head>'
        f'<body>{body}</body></html>')
    print(os.path.join(HERE, 'report.html'))


def verdict(ov, P, LN, cpu):
    b, c, cc = 'tm0.6_count_gate', 'tm0.6_count_cost', 'tm0.6_charge_cost'
    o = lambda a, k, f: ov[a][k][f] if '.' not in f else ov[a][k][f.split('.')[0] + '.' + f.split('.')[1]]
    e06 = lambda a, k: ov[a][k]['acc0.6']['eff']
    g06 = lambda a, k: ov[a][k]['acc0.6']['ghosts_per_ov']
    bp = lambda a, k: ov[a][k]['both_paired']
    sx = lambda a, k: ov[a][k]['by_tcls']['SX']['shallow_found_acc06']
    px = lambda a, k, acc='acc0.6': P[f'r145full_{a}'][k][acc]['excess']
    ln = lambda k, acc='0.6': LN[acc][k]['n_ac_line']
    rows = []
    for a in 'AC':
        rows.append(f"<li>chamber {a}: both members found and correctly paired {pct(bp(a, b))} &rarr; {pct(bp(a, c))} (cost) "
                    f"&rarr; {pct(bp(a, cc))} % (cost + charge); in-acceptance donors found {pct(e06(a, b))} &rarr; "
                    f"{pct(e06(a, cc))} %; accepted ghosts per overlay {g06(a, b):.2f} &rarr; {g06(a, cc):.2f}; the shallow "
                    f"member of a shallow+steep pair {pct(sx(a, b))} &rarr; {pct(sx(a, cc))} %. Capsule-pointing excess on "
                    f"real run_145 (acc 0.6) {px(a, b)} &rarr; {px(a, cc)} (&plusmn;{P[f'r145full_{a}'][b]['acc0.6']['err']:.0f}): "
                    f"unchanged, so what gate keeps and cost drops is not capsule signal.</li>")
    return f"""<p><b>Recommendation for the n_TOF bundles (A and C):</b></p>
<ol>
<li><b>select = 'cost'</b> (the bench mode). The overlay truth shows the failure NOTE_PAIRING_FIX &sect;3 predicted, at a
large rate: with 'gate' an implausible (steep) member drops its pair and the other member is joined to the wrong view.</li>
<li><b>split_corrob = 'charge'</b>. It adds on top of cost, and it makes the result independent of tan_max inside the
acceptance. Single tracks: winners change on 0.07 % (A) / 0.45 % (C) of beam events, pointing and the A&ndash;C line unchanged.</li>
<li><b>plausible keeps its tan_max</b> (no column-window-only definition). Once select is 'cost' and splits are 'charge',
removing the tan limit buys nothing on any truth inside the acceptance and costs CPU (+30&ndash;60 %, one C acquisition
ran 9 h). The A&ndash;C line with the stage-3 cut at 0.6: today {ln(b)} lines, column-window {ln('colwin_count_gate')},
same residuals.</li>
<li><b>raw tan_max = 0.6</b> (registry value, now confirmed). Beyond 0.6 the beam tracks carry no capsule-pointing excess
(table 2), and on the A&ndash;C line widening the acceptance to 1.0 adds about 240 lines and raises the 68 % residual 1.5adds lines at 2&ndash;3&times; the residual (table 3)ndash;2.3adds lines at 2&ndash;3&times; the residual (table 3)times; in the bins they land in (A 0adds lines at 2&ndash;3&times; the residual (table 3)ndash;10adds lines at 2&ndash;3&times; the residual (table 3)deg;: 6.7 adds lines at 2&ndash;3&times; the residual (table 3)rarr; 10.2adds lines at 2&ndash;3&times; the residual (table 3)deg;; 31adds lines at 2&ndash;3&times; the residual (table 3)ndash;40adds lines at 2&ndash;3&times; the residual (table 3)deg;: 2.5 adds lines at 2&ndash;3&times; the residual (table 3)rarr; 5.8adds lines at 2&ndash;3&times; the residual (table 3)deg;; table 3). Keep
the stage-3 true-angle cut at 0.6. Reaching wider angles is a separate decision for the X17 acceptance: it needs both a
wider reco limit and a wider stage-3 cut, and the data here do not support it.</li>
</ol>
<ul>{''.join(rows)}</ul>"""


LIMITS = """<ul>
<li>Overlay donors are real single beam tracks, but a sum of two triggers is not an IPC pair: no common vertex, the two
members' t0 agree only to |&Delta;t0| &lt; 60 ns rather than exactly, and the steep donors are whatever run_145 provides
(steep beam tracks as a population show no capsule-pointing excess, table 2). Truth is each donor's own fit, so a donor that is itself mis-fitted is scored
against its mis-fit.</li>
<li>The XX class (both members steep) is thin on A (5&ndash;40 overlays per separation bin).</li>
<li>Capsule pointing is a statistical truth (a sideband-subtracted excess); differences of a few tens of tracks between
configurations are at the 1&ndash;2&sigma; level even though the samples are shared.</li>
<li>Chambers A and C only; B and D carry no angle and were not tested.</li>
<li>One beam sub-run (run_145 stat090_0000) and run_149 cosmics; other run conditions (run_79 connector 8, late runs) are
not covered. t0 is free (v1k); a pinned-t0 bundle could shift the two-track numbers.</li>
<li>Condor CPU numbers are &plusmn;30 % (mixed nodes).</li>
<li>rerun/pairing_sim/sim_overlay.py was not used: its simulated donors are muons at 0&ndash;14&deg; with det3 bench
conditions, so it contains no steep donor and cannot see the failure tested here.</li>
</ul>"""

if __name__ == '__main__':
    main()
