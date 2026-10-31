#!/usr/bin/env python3
"""make_report.py -- report/report.html for the Paper I efficiency / loss items on pass 1, built from results/
(collect.py rows.csv + summary.json, a1_recat.py a1.json, fine_maps.py map_pools.json and figures). Figures are
referenced by relative links (figures/x.png), so the page works from disk, from the DAQ page or copied elsewhere."""
import html, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); REP = os.path.join(HERE, 'report'); FIG = os.path.join(REP, 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
COL = dict(det2='#1f77b4', det3='#d62728', det4='#2ca02c', det6='#9467bd', det7='#ff7f0e')


def f(x, d=1):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def table(rows, head):
    h = ['<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in head) + '</tr>']
    for r in rows:
        h.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
    return '\n'.join(h + ['</table>'])


def figs(T):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    T = T.copy(); T['resist_V'] = pd.to_numeric(T.resist_V, errors='coerce'); T['drift_V'] = pd.to_numeric(T.drift_V, errors='coerce')
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for d in DETS:
        g = T[(T.det == d) & T.eff_ig.notna() & (T.n_ig >= 200)]
        # one line per night and drift field; skeptical rows (chamber or slot suspect) as grey crosses
        sk = g[g.cls == 'skeptical']
        axs[0].plot(sk.resist_V, sk.eff_ig, 'x', color=COL[d], alpha=.35, ms=4)
        for (run, dv), gg in g[g.cls != 'skeptical'].groupby(['run', 'drift_V']):
            gg = gg.sort_values('resist_V')
            axs[0].errorbar(gg.resist_V, gg.eff_ig, yerr=gg.err_ig, fmt='o-' if len(gg) > 2 else 's', color=COL[d], ms=4,
                            lw=1, label=f'{d} {run.split("_")[-1]} drift {dv:g} V')
        gb = T[(T.det == d) & T.band_any.notna()]
        axs[1].plot(gb.resist_V, gb.band_quiet, 'o', color=COL[d], ms=4, label=d)
        gs = T[(T.det == d) & T.loss_spark_ig.notna() & (T.n_ig >= 200)]
        axs[2].plot(gs.resist_V, gs.loss_spark_ig, 'o', color=COL[d], ms=4, label=d)
    axs[0].set_xlabel('resist (mesh) HV [V]'); axs[0].set_ylabel('detection efficiency, interior_good, R = 5 mm [%]')
    axs[0].set_ylim(0, 100); axs[0].legend(fontsize=6, ncol=2); axs[0].grid(alpha=.3)
    axs[0].set_title('Efficiency vs HV (x = skeptical rows), rows with >= 200 interior rays', fontsize=10)
    axs[1].set_xlabel('resist HV [V]'); axs[1].set_ylabel('triggers with a noise band (no spark) [%]'); axs[1].set_yscale('log')
    axs[1].legend(fontsize=7); axs[1].grid(alpha=.3); axs[1].set_title('Noise-band census, every row', fontsize=10)
    axs[2].set_xlabel('resist HV [V]'); axs[2].set_ylabel('interior_good rays lost to the spark veto [%]')
    axs[2].legend(fontsize=7); axs[2].grid(alpha=.3); axs[2].set_title('Spark-veto loss', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'eff_vs_hv.png'), dpi=90); plt.close(fig)


def verdict(G, A1):
    """Verdict text and limitations, generated from the numbers (so a re-run updates them)."""
    head = '; '.join(f"{d} {f(G[d]['eff_ig'])} &plusmn; {f(G[d]['err_ig'])} %" for d in DETS if d in G)
    keys = {G[d]['key'] for d in G}
    gold = {o['det']: o for k, o in A1.items() if k in keys}
    top = []
    for d in DETS:
        o = gold.get(d)
        if not o:
            continue
        p = {k: v for k, v in o['interior_good']['pct'].items() if k != 'detected'}
        k, v = max(p.items(), key=lambda t: t[1])
        top.append(f'{d} {html.escape(k)} {v:.1f} %')
    rule = '; '.join(f"{d} {G[d]['rule_ig'] - G[d]['eff_ig']:+.2f}" for d in DETS if d in G and G[d].get('rule_ig') is not None)
    credit = '; '.join(f"{d} +{G[d]['any_ig'] - G[d]['eff_ig']:.1f}" for d in DETS if d in G and G[d].get('any_ig') is not None)
    txt = (f'Detection efficiency on the pass-1 golden rows, interior_good, R = 5 mm: {head}. '
           f'The largest single loss class per chamber (A1, interior_good): {"; ".join(top)}. '
           f'The trigger-track rule as a selection changes the golden efficiencies by {rule} points. '
           f'Multi-track credit (reference-assisted, not a selection) would add {credit} points.')
    lim = ['Efficiencies use |r| <= 5 mm at the mesh, which does not depend on v; the interior_good angle s68 in the '
           'tables DO depend on v(t) for g_det4, g_det6_long, g_det7_long and the other VT_PLAN rows.',
           'The M3 reference is not perfect: far:POS_TAIL (mostly tracks bent between the M3 doublets, C22) is counted '
           'as a chamber loss here.',
           "det4's dead / low-gain band map (interior_live) is the pre-pass C22 live-ratio map on the same raw events: "
           'hardware, not reconstruction, but not re-derived on the pass tables.',
           'C22 classes are rebuilt on the pass tables without the M3 cluster trees (m3feat): only the POS_TAIL / '
           'M3_ELSEWHERE explanation text is missing, not the class.',
           'The row-level noise-band census reads at most 4000 triggers per decoded file (about 30 % of a long run); '
           'the golden rows were re-read in full for A1.',
           'Skeptical rows (det3 6-22, det3 6-23, det2 6-17) are listed, not quoted. On det3 6-23 the broken angle scale '
           '(folded v 1.7-3.5 um/ns) moves right-position fits to fit_failed through the |tan| <= 3 sanity cut: the '
           'position-only column is the efficiency there.',
           'The spark veto (> 50 strips above the floor) removes 17-21 % of the det6/det7 golden interior tracks, '
           'almost all classified as discharges from their waveforms: a property of those chambers at 495 V.']
    return dict(verdict=txt, limits=lim)


def main():
    T = pd.read_csv(os.path.join(RES, 'rows.csv'))
    S = json.load(open(os.path.join(RES, 'summary.json')))
    A1 = json.load(open(os.path.join(RES, 'a1.json'))) if os.path.exists(os.path.join(RES, 'a1.json')) else {}
    pools = json.load(open(os.path.join(RES, 'map_pools.json'))) if os.path.exists(os.path.join(RES, 'map_pools.json')) else {}
    V = verdict(S['golden'], A1)
    json.dump(V, open(os.path.join(RES, 'verdict.json'), 'w'), indent=1)
    figs(T)
    G = S['golden']
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Paper I efficiency and losses on pass 1</title><style>'
         'body{font-family:sans-serif;max-width:1250px;margin:auto;padding:1em;line-height:1.4}'
         'table{border-collapse:collapse;font-size:13px;margin:.5em 0}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}'
         'th{background:#eee}td:first-child,th:first-child{text-align:left}.box{background:#f4f8ff;border:1px solid #9bd;padding:.6em 1em}'
         'img{max-width:100%}code{background:#f3f3f3}</style></head><body>']
    H.append('<h1>Paper I: efficiency, loss budget and A1 on the pass-1 bench tables (2026-10-31)</h1>')
    H.append('<div class="box"><b>Verdict.</b> ' + V['verdict'] + '</div>')
    H.append('<h2>Headline: golden rows</h2><p>Detection = both views fitted, |r| &le; R in the raw strip frame against the '
             'M3 ray (&chi;&sup2;&lt;1, NClus = 4, one good ray per event), &le; 2 competing strips per view. '
             '<b>interior_good</b> = the footprint &ge; 30 mm from an unpassivated edge and &ge; 8 mm from a passivated one, '
             'in both views, and for det7 the X footprint on ch 256&ndash;511 (<code>common/mx17_fiducial.py</code>). '
             'Errors are binomial. "rule" = the trigger-track rule picks the candidate in each view; "credit" = any '
             'candidate within R (uses the reference, never a selection).</p>')
    rows = []
    for d in DETS:
        g = G.get(d)
        if not g:
            continue
        rows.append([d, html.escape(g['key']), f"{g['resist_V']}/{g['drift_V']}", g.get('n_ig'), f(g.get('eff_interior')),
                     f"<b>{f(g.get('eff_ig'))}</b> &plusmn; {f(g.get('err_ig'))}", f(g.get('eff_ig_r3')), f(g.get('eff_x_ig')), f(g.get('eff_y_ig')),
                     f(g.get('rule_ig')), f(g.get('any_ig')), f(g.get('band_any')), f(g.get('s68_x_ig'), 2), f(g.get('s68_y_ig'), 2)])
    H.append(table(rows, ['chamber', 'golden row', 'HV resist/drift', 'n interior_good', 'eff interior', 'eff interior_good',
                          'R = 3 mm', 'x view', 'y view', 'rule', 'credit', 'band trig %', 's68 x [deg]', 's68 y [deg]']))
    H.append('<p>The angle s68 on interior_good are <b>v(t)-dependent</b> on the rows <code>rerun/VT_PLAN_2026-10-30.md</code> '
             'flags (g_det4, g_det6_long, g_det7_long, det2 6-22 long, det4 6-23 long, det6/det7 6-26 longer): quote them '
             'only after the v(t) correction. Efficiencies use |r| &le; 5 mm at the mesh, which does not depend on v.</p>')
    # loss ladder
    lad = ['rejected_cut', 'spark', 'no_hit', 'hw_known', 'one_view_silent', 'charge_elsewhere', 'not_seeded', 'fit_failed', 'far']
    rows = [[d] + [f(G[d].get(f'loss_{c}_ig'), 2) for c in lad] for d in DETS if d in G]
    H.append('<h2>Loss ladder (golden rows, interior_good, R = 5 mm, first match wins)</h2>' + table(rows, ['chamber'] + lad))
    # A1
    if A1:
        H.append('<h2>Appendix A1 classes on the pass tables</h2><p>Every interior ray of each golden row in one class '
                 '(a1_recat.py; the satstudy recat order). Columns: % of the interior rays.</p>')
        keys = sorted({k for o in A1.values() for k in o['interior']['pct']}, key=lambda k: (k != 'detected', k))
        rows = []
        for key, o in A1.items():
            for pop in ('interior', 'interior_good', 'interior_live'):
                if pop in o:
                    rows.append([f"{o['det']} {pop}", html.escape(key), o[pop]['n']] +
                                [f(o[pop]['pct'].get(k, 0.0), 2) for k in keys])
        H.append(table(rows, ['chamber / population', 'row', 'n'] + keys))
        H.append('<p>Band coverage (fraction of interior rays whose trigger the band census read): ' +
                 ', '.join(f"{o['det']} {f(100 * o['band_coverage'], 0)} %" for o in A1.values() if o.get('band_coverage') is not None) + '.</p>')
    H.append('<h2>Efficiency, noise bands and sparks against HV, every row</h2><img src="figures/eff_vs_hv.png">')
    H.append('<h2>Fine-kernel efficiency maps</h2><p>Disk radius 8 mm against 25 mm on a 2 mm grid, from the budget\'s '
             'per-ray table (denominator = every measured ray in the active area); right: profile along x with no kernel. '
             'Plateau pool = rows of the chamber at the golden drift field within 3 points of the golden interior efficiency.</p>')
    for d in DETS:
        if os.path.exists(os.path.join(FIG, f'effmap_{d}.png')):
            H.append(f'<h3>{d}</h3><p>pool: {len(pools.get(d, {}).get("pool", []))} rows</p><img src="figures/effmap_{d}.png">')
    # every row table
    cols = ['det', 'subrun', 'resist_V', 'drift_V', 'cls', 'n_ig', 'eff_ig', 'err_ig', 'eff_ig_r3', 'rule_ig', 'any_ig',
            'band_any', 'band_quiet', 'loss_spark_ig', 's68_x_ig', 's68_y_ig']
    U = T.sort_values(['det', 'run', 'subrun'])
    rows = [[html.escape(str(r[c])) if c in ('det', 'subrun', 'cls') else (f(r[c], 2) if isinstance(r[c], float) else r[c]) for c in cols]
            for _, r in U.iterrows()]
    H.append('<h2>Every pass-1 bench row</h2>' + table(rows, cols))
    H.append('<h2>What this does not rule out</h2><ul>' + ''.join(f'<li>{x}</li>' for x in V['limits']) + '</ul>')
    H.append('<p>Code: <code>rerun/effbudget/pass1/</code> (make_dag.py &rarr; condor; collect.py, a1_recat.py, fine_maps.py, '
             'make_report.py on lxplus). Outputs: <code>/eos/user/d/dneff/x17rerun/paper1_eff/</code>.</p></body></html>')
    open(os.path.join(REP, 'report.html'), 'w').write('\n'.join(H))
    print('wrote', os.path.join(REP, 'report.html'))


if __name__ == '__main__':
    main()
