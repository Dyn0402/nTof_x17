#!/usr/bin/env python3
"""make_report.py -- report.html + figures/ for the n_TOF channel layer (P2), generated from make_masks.py's output.

    make_report.py

Reads masks/*.json, cabling_check.json, the census tables in $WORK_DATA/ntof_pass1/channel_layer/ (acq_table.csv,
the control acquisitions' json) and the old run_145 masks (sept26_prelim stage1 hot mask, 102 channels), and writes
report.html with figures/ next to this file (figures are regenerable, not in git).
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sept26_prelim_analysis')]
from census import WORK, POS, FEU_ARM  # noqa: E402

MASKS = os.path.join(HERE, 'masks')
FIG = os.path.join(HERE, 'figures')
OLD145 = '/media/dylan/data/x17/sept26_prelim/stage1/hot_mask_run_145_stat090_0000.csv'
ARM_FEU = {(a, v): f for f, (a, v) in FEU_ARM.items()}
ERAS = ['E1_pre_access', 'E2_post_access']


def old145():
    out = defaultdict(set)
    if os.path.exists(OLD145):
        for r in csv.DictReader(open(OLD145)):
            out[(r['det'][-1], r['plane'])].add(int(r['channel']))
    return out


def fig_occupancy(M):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.use()
    fig, axs = plt.subplots(4, 2, figsize=(9.6, 10.5), sharex=True)
    for i, arm in enumerate('ABCD'):
        for j, v in enumerate('xy'):
            ax = axs[i, j]
            feu = ARM_FEU[(arm, v)]
            for era, col, lw in (('E2_post_access', fs.DET_COLOR[arm], 1.0), ('E1_pre_access', fs.MUTED, 0.8)):
                p = os.path.join(WORK, f'era_occ_{arm}_{v}_{era}.npy')
                if os.path.exists(p):
                    occ = np.load(p)
                    o = np.argsort(POS)
                    ax.plot(POS[o], np.maximum(occ[o], 1e-5), color=col, lw=lw, label=era.split('_', 1)[1].replace('_', '-'))
            m = M.get(f'{arm}_E2_post_access')
            if m:
                c = m['planes'][v]['components']
                for name, mk, col in (('chamber_side', '|', fs.INK), ('ntof_silent', 'v', fs.BAND_DEAD),
                                      ('ntof_flank_dead', 'x', fs.BAND_DEAD), ('ntof_shorts', 's', fs.COPPER),
                                      ('hot', '^', fs.TRACK), ('beam_occupancy_hot', 'o', fs.ACCENT)):
                    ch = c.get(name, [])
                    if ch:
                        ax.plot(POS[ch], np.full(len(ch), 0.6 if name == 'hot' else (0.3 if name == 'beam_occupancy_hot' else
                                                       {'chamber_side': 2e-5, 'ntof_silent': 4e-5, 'ntof_flank_dead': 8e-5,
                                                        'ntof_shorts': 1.6e-4}.get(name, 2e-5))), mk, ms=3.5, color=col,
                                mfc='none' if mk in 'sv' else col, mew=1.0, label=name.replace('_', ' '), lw=0)
            ax.set_yscale('log'); ax.set_ylim(1e-5, 1.0)
            ax.set_title(f'{arm}-{v} (FEU {feu})', loc='left', fontsize=10)
            if j == 0:
                ax.set_ylabel('occupancy')
            if i == 3:
                ax.set_xlabel('strip position [mm]')
            if i == 0 and j == 1:
                ax.legend(fontsize=7, ncol=2, loc='lower right', frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'occupancy_masks.png'), dpi=150)
    plt.close(fig)


def fig_conn8(rows, ctrl):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    import figstyle as fs
    fs.use()
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    sel = [('A', 'x', fs.DET_COLOR['A'], 'o'), ('A', 'y', fs.DET_COLOR['A'], 's'), ('B', 'y', fs.DET_COLOR['B'], 's')]
    for arm, v, col, mk in sel:
        for src, mfc in ((rows, col), (ctrl, 'none')):
            t, y = [], []
            for r in src:
                if r['arm'] == arm and r['view'] == v and int(r.get('n_events', 10 ** 6)) >= 1000:
                    t.append(datetime.strptime(r['t_local'], '%Y-%m-%d %H:%M'))
                    y.append(max(float(r['conn_occ'].split()[7]), 1e-5) / max(float(r['median_occ']), 1e-5))
            if t:
                ax.plot(t, y, mk, ms=3.2 if src is rows else 6, color=col, mfc=mfc, mew=1.0, lw=0,
                        label=f'{arm}-{v}' + (' (controls, out of scope)' if src is ctrl else ''))
    ax.set_yscale('log'); ax.set_ylabel('connector 8 occupancy / view median')
    ax.axhline(0.05, color=fs.BAND_DEAD, lw=0.8, ls='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.legend(fontsize=7, ncol=3, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'connector8_time.png'), dpi=150)
    plt.close(fig)


def fig_hotfreq(M):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.use()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for j, v in enumerate('xy'):
        ax = axs[j]
        for arm in 'ABCD':
            m = M.get(f'{arm}_E2_post_access')
            if not m:
                continue
            hf = m['planes'][v]['hot_frequency']
            ch = np.array([int(c) for c in hf], int); fr = np.array(list(hf.values()), float)
            if len(ch):
                ax.plot(POS[ch], fr, fs.DET_MARKER[arm], ms=3.5, color=fs.DET_COLOR[arm], lw=0, label=arm)
        ax.axhline(json.load(open(os.path.join(MASKS, 'summary.json')))['persist'], color=fs.MUTED, lw=0.8, ls='--')
        ax.set_title(f'{v} views, E2 post-access', loc='left', fontsize=10)
        ax.set_xlabel('strip position [mm]')
        if j == 0:
            ax.set_ylabel('fraction of acquisitions hot')
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'hot_frequency.png'), dpi=150)
    plt.close(fig)


def main():
    M = {os.path.basename(p)[5:-5]: json.load(open(p)) for p in glob.glob(os.path.join(MASKS, 'ntof_*.json'))}
    S = json.load(open(os.path.join(MASKS, 'summary.json')))
    D = json.load(open(os.path.join(MASKS, 'dropouts.json')))
    cab = json.load(open(os.path.join(HERE, 'cabling_check.json')))
    rows = list(csv.DictReader(open(os.path.join(WORK, 'acq_table.csv'))))
    ctrl = []
    for p in sorted(glob.glob(os.path.join(WORK, 'acq', 'ctrl_*.json'))):
        d = json.load(open(p))
        for f, r in d['feu'].items():
            ctrl.append(dict(key=d['key'], t_local=d['t_local'], n_events=10 ** 6, arm=r['arm'], view=r['view'], median_occ=r['median_occ'],
                             conn_occ=' '.join(map(str, r['conn_occ'])), conn_silent=r['conn_silent'], hot=r['hot'],
                             silent=len(r['silent'])))
    fig_occupancy(M); fig_conn8(rows, ctrl); fig_hotfreq(M)
    old = old145()

    def tr(cells, th=False):
        t = 'th' if th else 'td'
        return '<tr>' + ''.join(f'<{t}>{c}</{t}>' for c in cells) + '</tr>'

    # main mask table
    mt = [tr(['arm (bench)', 'era', 'acq.', 'x dead', 'x hot', 'y dead', 'y hot', 'x: chamber-side / bench carried / '
              'n_TOF dead+silent / shorts / seed-hot / beam-occupancy hot', 'y: same'], True)]
    for era in ERAS:
        for arm in 'ABCD':
            m = M.get(f'{arm}_{era}')
            if not m:
                continue
            c = {v: m['planes'][v]['components'] for v in 'xy'}
            comp = {v: f"{len(c[v]['chamber_side'])} / {len(c[v]['bench_carried'])} / "
                       f"{len(set(c[v]['ntof_flank_dead']) | set(c[v]['ntof_silent']))} / {len(c[v]['ntof_shorts'])} / "
                       f"{len(set(c[v]['hot']) | set(c[v]['near_osc']))} / {len(c[v].get('beam_occupancy_hot', []))}" for v in 'xy'}
            mt.append(tr([f"{arm} ({m['bench_chamber']})", era, m['n_acquisitions'], m['planes']['x']['n_dead'],
                          len(m['planes']['x']['hot']), m['planes']['y']['n_dead'], len(m['planes']['y']['hot']),
                          comp['x'], comp['y']]))
    # bench comparison
    bt = [tr(['arm-view', 'bench census dead+shorts', 'pass-1 bundle dead', 'dropped: electronics (bench FEU)',
              'dropped: healthy at n_TOF', 'carried', 'n_TOF E2 mask'], True)]
    for arm in 'ABCD':
        m = M.get(f'{arm}_E2_post_access')
        for v in 'xy':
            p = m['planes'][v]; b = p['bench']
            bt.append(tr([f'{arm}-{v}', len(b['census_dead_shorts']), len(b['pass1_bundle_dead']),
                          ' '.join(map(str, b['dropped_electronics'])) or '-',
                          ' '.join(map(str, b['dropped_healthy_at_ntof'])) or '-',
                          len(p['components']['bench_carried']), p['n_dead']]))
    # old run_145 mask comparison
    ot = [tr(['arm-view', 'old run_145 hot mask (occupancy > 8 %, > 8x median)', 'in new E2 dead', 'in new E2 hot (wildcard)',
              'old in neither'], True)]
    for arm in 'ABCD':
        m = M.get(f'{arm}_E2_post_access')
        for v in 'xy':
            o = old.get((arm, v), set()); new = set(m['planes'][v]['dead']); h = set(m['planes'][v]['hot'])
            ot.append(tr([f'{arm}-{v}', len(o), len(o & new), len(o & h), ' '.join(map(str, sorted(o - new - h))) or '-']))
    # dropouts
    dt = [tr(['arm-view', 'connector (channels)', 'era', 'acquisitions silent', 'runs', 'first', 'last'], True)]
    for x in D:
        dt.append(tr([f"{x['arm']}-{x['view']}", f"{x['connector']} ({x['channels'][0]}-{x['channels'][1]})", x['era'],
                      f"{x['n_acq']} / {x['n_acq_era']}", ' '.join(x['runs'][:6]) + (' ...' if len(x['runs']) > 6 else ''),
                      x['first'], x['last']]))
    ct = [tr(['control acquisition (out of pass-1 scope)', 'time', 'A-y conn 8 silent / 64', 'B-y conn 8 silent / 64',
              'A-x conn 8 silent / 64'], True)]
    byk = defaultdict(dict)
    for r in ctrl:
        byk[(r['key'], r['t_local'])][f"{r['arm']}{r['view']}"] = r['conn_silent'][7]
    for (k, t), d in sorted(byk.items(), key=lambda z: z[0][1]):
        ct.append(tr([k[5:], t, d.get('Ay', '-'), d.get('By', '-'), d.get('Ax', '-')]))
    cab_ok = all(x['n_runs_identical'] == x['n_runs'] for x in cab['arms'].values())
    n_acq = sum(S['n_acq'].values())
    tot = {era: sum(M[f'{a}_{era}']['planes'][v]['n_dead'] for a in 'ABCD' for v in 'xy') for era in ERAS
           if all(f'{a}_{era}' in M for a in 'ABCD')}
    inscope = [x for x in D if not (x['arm'] == 'A' and x['view'] == 'x' and x['connector'] == 8 and x['era'] == 'E1_pre_access')]
    e1c = M['A_E1_pre_access']['planes']['x']['components'].get('dead_connectors', [])
    e2c = M['A_E2_post_access']['planes']['x']['components'].get('dead_connectors', [])
    verdict = (f"Two mask eras are enough for the pass-1 scope. A-x connector 8 (FEU 3 ch 448-511) is "
               f"{'dead as a whole in E1 (run_79-81)' if 8 in e1c else 'NOT found dead in E1'} and "
               f"{'still dead' if 8 in e2c else 'live'} from run_83 on; "
               + ('no other connector drops out in any sampled in-scope acquisition'
                  if not inscope else f'{len(inscope)} other connector dropouts were found (table below)')
               + ". The 9 August A-y connector-8 dropout is confined to run_161 (an HV scan outside the scope); run_160 "
               "and run_162 are live. Chamber D carries most of the mask (its period-5 X shorts and connector-edge hot "
               "strips); A, B and C are light.")
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>n_TOF channel layer (pass 1, P2)</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#1b2430;line-height:1.45}}
table{{border-collapse:collapse;font-size:13px;margin:0.6em 0 1.2em}}td,th{{border:1px solid #c9ced6;padding:3px 7px;text-align:left}}
th{{background:#f2f4f7}}img{{max-width:100%}}.cap{{color:#6a7583;font-size:13px}}code{{background:#f2f4f7;padding:0 3px}}</style></head><body>
<h1>n_TOF channel layer: dead/hot masks per chamber and era</h1>
<p><b>Verdict.</b> {verdict}</p>
<p><b>Headline.</b> {n_acq} acquisitions censused (first and last of every kept sub-run of the pass-1 registry); two mask
eras; censored channels summed over the eight views: {', '.join(f'{k} {v}' for k, v in tot.items())} (of 4096).
Cabling: {'the n_TOF channel to strip map is identical to the bench map for every chamber, both views, all 512 channels, in all '
          + str(cab['arms']['A']['n_runs']) + ' n_TOF run_configs' if cab_ok else 'DIFFERENCES FOUND, see cabling_check.json'},
so bench lists carry by channel number (no remap).</p>
<h2>What was compared</h2>
<p>Each mask is the F12 rule (<code>rerun/model_audit/ws_dead_charge/REPORT.md</code>): chamber-side faults by position
plus this dataset's own faults from its own hits, plus the SPS hot rules (<code>rerun/sps_junk/README.md</code>). Hits
(<code>combined_hits_root</code>) are used for QA only. Rules and constants: <code>rules.py</code>; persistence for hot:
hot in at least {S['persist']:.0%} of the era's sampled acquisitions.</p>
<h2>Masks</h2><table>{''.join(mt)}</table>
<p class="cap">dead = the bundle's censored list (<code>CalibrationBundle.dead</code>: chamber-side, bench carried, n_TOF
dead/silent/shorts, SPS seed-hot and near oscillators, censored as at SPS); hot = the bundle's wildcard list
(<code>CalibrationBundle.hot</code>: never seeded, noise inflated): beam-occupancy hot blocks and SPS coupled channels. Files: <code>masks/ntof_&lt;arm&gt;_&lt;era&gt;.json</code>.</p>
<h2>Against the bench lists</h2><table>{''.join(bt)}</table>
<h2>Against the old run_145 hot mask (102 channels)</h2><table>{''.join(ot)}</table>
<h2>Connector dropouts</h2><table>{''.join(dt) if len(dt) > 1 else '<tr><td>none</td></tr>'}</table>
<p>Controls outside the pass-1 scope (early July, 9 August run_161 HV scan), probed with the same code:</p>
<table>{''.join(ct)}</table>
<h2>Figures</h2>
<img src="figures/occupancy_masks.png"><p class="cap">Era-summed occupancy per channel (fraction of triggers with a hit
&gt; 50 ADC) against strip position; E2 coloured, E1 grey. Markers: mask components of E2 (bottom rows: chamber-side, silent, flank dead, shorts, all censored; top: red triangles SPS seed-hot, censored; purple circles beam-occupancy hot, the wildcard list).</p>
<img src="figures/connector8_time.png"><p class="cap">Connector 8 (ch 448-511) occupancy over the view median, per sampled
acquisition. Dashed: the 5 % silent threshold. Open markers: control acquisitions outside the pass-1 scope.</p>
<img src="figures/hot_frequency.png"><p class="cap">How often each channel is hot in the SPS seed probe, E2. Dashed: the
persistence threshold.</p>
<h2>What this does not rule out</h2><ul>
<li>A channel that is faulty only in sub-runs that were not sampled (two acquisitions per sub-run were). The per-row SPS probe in
the driver would catch it per acquisition.</li>
<li>Hot channels that are hot in fewer than {S['persist']:.0%} of acquisitions are not masked (listed with their frequency).</li>
<li>Flank tests need two bright neighbours: strips at the plate edges and inside long dead runs are judged by the silent rule only.</li>
<li>Coupled channels are an event-by-event effect at SPS; a static list can only approximate it.</li>
<li>Shorts are voted per acquisition: the first sampled acquisition of every run in E2 (73), every sampled acquisition in E1 (46); a short must replicate in at least half of them.</li></ul>
<p class="cap">Generated by <code>rerun/ntof/channel_layer/make_report.py</code>.</p></body></html>"""
    open(os.path.join(HERE, 'report.html'), 'w').write(html)
    print('->', os.path.join(HERE, 'report.html'))


if __name__ == '__main__':
    main()
