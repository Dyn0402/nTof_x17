#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosstalk_feu1_d67.py

FEU1 (M3) cross-talk test for the det6/det7 run, accounting for BOTH detectors'
sparks. "our spark" = det6 OR det7 fires >50 strips. Same method as the det3 test
(crosstalk_feu1.py): decode FEU1 .fdf -> per-event M3 cluster count / self-spark
flag / amplitudes, joined on evn to d67_events.npz.

Extra: dose-response -- neither / det6-only / det7-only / BOTH -- to see whether two
simultaneous discharges inject more noise into the shared M3 readout than one.

Run: ../../.venv/bin/python crosstalk_feu1_d67.py [analyse.root]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.expanduser('~/CLionProjects/cosmic_bench_m3_tracking/root_files/d67_analyse.root'))

e = np.load(os.path.join(HERE, 'd67_events.npz'))
eid = e['eventId'].astype(np.int64)
s6 = e['spark6']; s7 = e['spark7']; seither = e['spark_either']; sboth = e['spark_both']

F = uproot.open(ANA)
t = F[[k for k in F.keys() if k.split(';')[0] == 'T'][0].split(';')[0]]
d = t.arrays(['evn', 'MGv2_NClus', 'MGv2_Spark', 'MGv2_ClusAmpl'], library='np')
m3evn = d['evn'].astype(np.int64)
NClus = np.stack(d['MGv2_NClus']); Ampl = np.stack(d['MGv2_ClusAmpl'])
nclus = NClus.sum(axis=1)
m3spark = (np.stack(d['MGv2_Spark']) > 0).any(axis=1)
row = {int(v): i for i, v in enumerate(m3evn)}
_idx = np.arange(Ampl.shape[2])[None, None, :]

lo, hi = m3evn.min(), m3evn.max()
in_win = (eid >= lo) & (eid <= hi)
print(f'FEU1 analyse: {len(m3evn)} events, evn [{lo},{hi}]')
print(f'our events in window: {in_win.sum()}  (either-spark {int((seither & in_win).sum())})')


def stats(mask):
    ids = eid[mask & in_win]
    present = np.array([i in row for i in ids])
    rows = [row[i] for i in ids if i in row]
    nc = nclus[rows] if rows else np.array([])
    sp = m3spark[rows] if rows else np.array([])
    return dict(n=len(ids), present_pct=100 * present.mean() if len(ids) else 0,
                dropped_pct=100 * (1 - present.mean()) if len(ids) else 0,
                nclus_mean=float(nc.mean()) if len(nc) else 0,
                nclus_med=float(np.median(nc)) if len(nc) else 0,
                selfspark_pct=100 * float(sp.mean()) if len(sp) else 0, nc=nc)


neither = stats(~seither); either = stats(seither)
only6 = stats(s6 & ~s7); only7 = stats(s7 & ~s6); both = stats(sboth)
print('\n== cross-talk: our spark (either) vs neither ==')
for nm, s in [('neither', neither), ('either spark', either)]:
    print(f'  {nm:14s}: N={s["n"]:6d}  present {s["present_pct"]:5.1f}%  '
          f'dropped {s["dropped_pct"]:4.1f}%  M3 selfspark {s["selfspark_pct"]:4.2f}%  '
          f'M3 clus/evt {s["nclus_mean"]:.2f} (med {s["nclus_med"]:.0f})')
print('\n== dose-response (M3 clusters/event) ==')
for nm, s in [('neither', neither), ('det6 only', only6), ('det7 only', only7), ('BOTH', both)]:
    print(f'  {nm:10s}: N={s["n"]:6d}  M3 clus/evt {s["nclus_mean"]:.2f}  '
          f'M3 selfspark {s["selfspark_pct"]:.2f}%  dropped {s["dropped_pct"]:.1f}%')

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
# (a) not knocked out: drops + self-spark for either vs neither
x = np.arange(2)
axes[0].bar(x - 0.2, [either['dropped_pct'], either['selfspark_pct']], 0.4, color='crimson', label='our spark (either)')
axes[0].bar(x + 0.2, [neither['dropped_pct'], neither['selfspark_pct']], 0.4, color='grey', label='neither')
for xi, (a, b) in enumerate([(either['dropped_pct'], neither['dropped_pct']),
                             (either['selfspark_pct'], neither['selfspark_pct'])]):
    axes[0].text(xi - 0.2, max(a, 0.05) + 0.05, f'{a:.1f}%', ha='center', fontsize=8, color='crimson')
    axes[0].text(xi + 0.2, max(b, 0.05) + 0.05, f'{b:.1f}%', ha='center', fontsize=8, color='dimgrey')
axes[0].set_xticks(x); axes[0].set_xticklabels(['FEU1 trigger\nDROPPED', 'M3 fires its\nOWN spark'])
axes[0].set_ylabel('% of events'); axes[0].set_ylim(0, max(5, either['dropped_pct'] * 1.3)); axes[0].legend(fontsize=8)
axes[0].set_title('(a) M3 knocked out?')
# (b) M3 cluster excess either vs neither
bins = np.arange(0, 30)
axes[1].hist(neither['nc'], bins=bins, density=True, alpha=0.6, color='grey', label='neither')
axes[1].hist(either['nc'], bins=bins, density=True, alpha=0.6, color='crimson', label='our spark')
axes[1].set_xlabel('M3 clusters/event (8 layers)'); axes[1].set_ylabel('norm.'); axes[1].legend()
axes[1].set_title('(b) M3 clusters: spark %.1f vs neither %.1f (+%.0f%%)'
                  % (either['nclus_mean'], neither['nclus_mean'],
                     100 * (either['nclus_mean'] / neither['nclus_mean'] - 1)))
# (c) dose-response
labs = ['neither', 'det6\nonly', 'det7\nonly', 'BOTH']
vals = [neither['nclus_mean'], only6['nclus_mean'], only7['nclus_mean'], both['nclus_mean']]
axes[2].bar(range(4), vals, color=['grey', '#4a90d9', '#d94a4a', '#7b2cbf'])
for i, v in enumerate(vals):
    axes[2].text(i, v, f'{v:.1f}', ha='center', va='bottom')
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(labs)
axes[2].set_ylabel('M3 clusters/event')
axes[2].set_title('(c) dose-response\nboth-spark injects the most M3 noise')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_crosstalk_d67.png'), dpi=140, bbox_inches='tight')
print('\nwrote fig_crosstalk_d67.png')

out = dict(evn_window=[int(lo), int(hi)], n_in_window=int(in_win.sum()),
           neither={k: neither[k] for k in ('n', 'nclus_mean', 'selfspark_pct', 'dropped_pct')},
           either={k: either[k] for k in ('n', 'nclus_mean', 'selfspark_pct', 'dropped_pct')},
           det6_only={k: only6[k] for k in ('n', 'nclus_mean', 'selfspark_pct')},
           det7_only={k: only7[k] for k in ('n', 'nclus_mean', 'selfspark_pct')},
           both={k: both[k] for k in ('n', 'nclus_mean', 'selfspark_pct')},
           excess_pct=100 * (either['nclus_mean'] / neither['nclus_mean'] - 1))
json.dump(out, open(os.path.join(HERE, 'crosstalk_d67.json'), 'w'), indent=2)
print(json.dumps(out, indent=2))
