"""make_gaps.py -- internal: the drift gap of every bench chamber, every row with a head-on stack, by every method we have.

Methods (all independent of the design value):
  stack    g = v_ang x T, the method of Paper I fig. two_velocities (c): T = head-on column duration of the stack fit
           (rerun/paper1_gas/results/gas.json, arm in30), v_ang = bundle v_drift x mean(kw) (the M3-ladder angle-scale
           velocity of that row; bundles_v.json, read from EOS Analysis_pass1).
  geom     charge extent of inclined tracks, L / tan theta (sps_beam_test_26/analysis/cloud_basics/gaps.py; det6 adopted)
  vq90     v_ladder x median q_u90 of the 14 gate rows, scaled to 28.2 mm on chamber A golden (q_RESULT.md, Table 2).
           Relative only: it measures each chamber against chamber A.
Rows in the first hours after installation are flagged (C before 06:00 on 6-26, D's first row, E's first 6-23 row):
attachment shortens the visible column there (paper1_gas README), so the stack T and the gap from it are low.
Rows whose fitted column reaches the end of the recorded stack (data end 1590 ns after the column start, pg_twov.py) are
flagged too: their T is a lower limit (chamber A at 500 V: T 1592 ns, where v_ang predicts 1800 ns; q_RESULT.md, item 7).

    ../../../../.venv/bin/python make_gaps.py      -> gaps_all.csv, gaps_all.png, report.html (this directory)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics'))
import gaps  # noqa: E402

PUB = {'det2': 'B', 'det3': 'A', 'det4': 'E', 'det6': 'C', 'det7': 'D'}
COL = {'det2': '#4C72B0', 'det3': '#C44E52', 'det4': '#8172B2', 'det6': '#55A868', 'det7': '#CCB974'}
G = json.load(open(os.path.join(REPO, 'rerun/paper1_gas/results/gas.json')))
B = json.load(open(os.path.join(HERE, 'bundles_v.json')))
VQ = os.path.join('/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/q16/vT_rows.csv')


rows = []
for p in G['points']:
    if p['arm'] != 'in30':
        continue
    b = B.get(p['row'])
    if not b or not b.get('v'):
        continue
    kw = 0.5 * (b['kw']['x'] + b['kw']['y'])
    v = b['v'] * kw
    run, sub = p['row'].split('/')[:2]
    early = ('6-26-26' in run) and p.get('dt') is not None
    rows.append(dict(det=p['det'], chamber=PUB[p['det']], run=run, subrun=sub, E=p['E'], V=p['V'], n=p['n'], T=p['T'],
                     T_err=p['T_err'], v_ang=v, g=v * p['T'] / 1e3,
                     g_err=v * p['T'] / 1e3 * np.hypot(p['T_err'] / p['T'], 0.01)))
# golden Run 1 of chamber A (1000 V): never stacked by the gas study; its S4 re-check stack, as in Paper I fig. two_velocities
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_gas'))
import pg_twov  # noqa: E402
_gp = pg_twov.golden_point()
_va = json.load(open(os.path.join(REPO, 'rerun/paper1_gas/results/twov.json')))['ladder']['1000_hv']['v_ang']
rows.append(dict(det='det3', chamber='A', run='mx17_det3_saturday_scan_6-27-26', subrun='long_run_resist_490V_drift_1000V (S4)',
                 E=_gp['E'], V=1000.0, n=_gp['n'], T=_gp['T'], T_err=_gp['T_err'], v_ang=_va, g=_va * _gp['T'] / 1e3,
                 g_err=_va * _gp['T'] / 1e3 * np.hypot(_gp['T_err'] / _gp['T'], 0.01)))
D = pd.DataFrame(rows)
# attachment flag for 6-26: the ladder_626 times (hours on 6-26) per row key
t626 = {}
for q in G.get('ladder_626', []):
    k = q['key'].split('__')
    if len(k) >= 4:
        t626['/'.join(k[1:4])] = q['t626']
D['h626'] = [t626.get(f"{r.run}/{r.subrun}/mx17_{r.det[3:]}") for r in D.itertuples()]
# attachment in the first hours after installation (paper1_gas: C until ~4 h, D's first row 0.86): T reads short.
# E's first 6-23 row (longer_run, 19:30-21:30) shows the same signature: v 27.8 against 35.9 later, T only 5 % longer.
D['attach'] = ((D.h626.notna() & (D.h626 < 6) & (D.det == 'det6'))
               | (D.h626.notna() & (D.h626 < 2.5) & (D.det == 'det7'))
               | ((D.det == 'det4') & (D.run == 'mx17_det3_det4_overnight_6-23-26') & (D.subrun == 'longer_run')))
# the column runs past the end of the recorded stack: T is a lower limit (q_RESULT.md, fourth round, item 7)
STACK_END_NS = 1590.0
D['window'] = (D['T'] >= STACK_END_NS - 60)
D = D.sort_values(['chamber', 'run', 'E']).reset_index(drop=True)
D.to_csv(os.path.join(HERE, 'gaps_all.csv'), index=False)

summ = []
for det, d in D.groupby('det'):
    ok = d[~d.attach & ~d.window]
    w = 1 / ok.g_err ** 2
    m = float((ok.g * w).sum() / w.sum()); e = float(1 / np.sqrt(w.sum()))
    chi2 = float((((ok.g - m) / ok.g_err) ** 2).sum())
    summ.append(dict(chamber=PUB[det], det=det, n_rows=len(ok), stack=m, stack_err=e, chi2=chi2, spread=float(ok.g.std()),
                     geom=gaps.GAP_GEOM[det], geom_err=gaps.GAP_GEOM_ERR[det]))
S = pd.DataFrame(summ).sort_values('chamber')
if os.path.exists(VQ):
    V = pd.read_csv(VQ)
    V['det'] = V.key.str.split(':').str[0]
    V = V[V.rail < 0.12]           # drop rows whose drift time is censored by the depth grid (C's first two)
    S['vq90'] = [float(V[V.det == r.det].g_mm.median()) if (V.det == r.det).any() else np.nan for r in S.itertuples()]
S.to_csv(os.path.join(HERE, 'gaps_summary.csv'), index=False)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for det, d in D.groupby('det'):
    ok, bad = d[~d.attach & ~d.window], d[d.attach | d.window]
    ax.errorbar(ok.E, ok.g, ok.g_err, fmt='o', color=COL[det], ms=4, capsize=2, label=f'{PUB[det]} ({det})')
    if len(bad):
        ax.errorbar(bad.E, bad.g, bad.g_err, fmt='o', mfc='none', color=COL[det], ms=4, capsize=2)
    ax.axhline(gaps.GAP_GEOM[det], color=COL[det], ls=':', lw=1)
ax.axhline(30, color='k', lw=0.8, ls='--')
ax.text(ax.get_xlim()[0], 30.1, ' design 30 mm', fontsize=8, va='bottom')
ax.set_ylim(22, 33); ax.set_xlabel('drift field E [V/cm]'); ax.set_ylabel(r'g = $v_\mathrm{ang}\times T$ [mm]')
ax.set_title('Drift gap per row (filled: used; open: first hours of a night (attachment), or column past the record end). Dotted: geometric gap. C short (15.4 mm) off scale', fontsize=8)
ax.legend(fontsize=8, ncol=5, loc='lower center')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'gaps_all.png'), dpi=140)

tbl = D[['chamber', 'run', 'subrun', 'V', 'E', 'n', 'v_ang', 'T', 'g', 'g_err', 'attach', 'window']].round(2).to_html(index=False)
st = S.round(2).to_html(index=False)
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift gaps, all bench chambers</title>
<style>body{{font-family:sans-serif;max-width:1000px;margin:auto;padding:16px}}table{{border-collapse:collapse;font-size:12px}}
td,th{{border:1px solid #ccc;padding:2px 6px}}</style></head><body>
<h1>Drift gap of every bench chamber (internal)</h1>
<p>Generated by <code>make_gaps.py</code>. The gap is measured as g = v_ang &times; T, the method of Paper I figure
two_velocities (c): T is the duration of the drift column in the head-on stack, and v_ang is the angle-scale velocity of the
same row (M3 ladder). Neither uses the design value. The geometric gap (charge extent of inclined tracks) is an independent
second method. Rows taken in the first hours after a chamber was installed (C before 06:00 on 6-26, D's first row, E's first row on 6-23) are left out of the mean, because attachment shortens the visible column there. Rows whose column runs past the end of the recorded stack
(chamber A at 500 V) are left out too, since their T is only a lower limit.</p>
<h2>Per chamber</h2>{st}
<p>stack: weighted mean of the rows (error from T only, ⊕ 1 % on v); chi2 against that mean; spread = rms over rows.
geom: charge-extent gap. vq90: v &times; median q_u90 of the gate rows with less than 12 % censored tracks, scaled to chamber A golden = 28.2 mm (relative only).</p>
<img src="gaps_all.png" style="max-width:100%">
<h2>Per row</h2>{tbl}
</body></html>"""
open(os.path.join(HERE, 'report.html'), 'w').write(html)
print(S.round(2).to_string())
