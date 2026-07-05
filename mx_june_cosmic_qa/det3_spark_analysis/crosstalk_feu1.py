#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosstalk_feu1.py

Direct test of the DAQ cross-talk hypothesis: does OUR detector's spark (FEU7/8,
mx17_3) inject excess noise / discharges into the SHARED M3 readout (FEU1)?

Reads the M3/FEU1 cluster tree decoded from the raw .fdf on lxplus
(cosmic_bench_m3_tracking `DataReader ... analyse`), which carries per-event, per
M3-layer:
  evn                event number (== our detector eventId, verified tight)
  MGv2_NClus[8]      # clusters per M3 layer
  MGv2_Spark[8]      M3 producer's OWN spark flag per layer
  MGv2_StripMaxAmpl[8,61]   max strip amplitude per layer

Joined on evn to our events.npz (spark flag on FEU7/8). Three cross-talk signatures:
  (A) dropped triggers  : is a spark event more often ABSENT from the FEU1 tree?
  (B) M3 self-spark     : does M3 raise its own MGv2_Spark when we spark?
  (C) excess M3 activity: elevated M3 NClus / strip amplitude in our spark events?

Run: ../../.venv/bin/python crosstalk_feu1.py [analyse.root]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.expanduser('~/CLionProjects/cosmic_bench_m3_tracking/root_files/p2_627_analyse.root'))

# ---- our detector events ----
e = np.load(os.path.join(HERE, 'events.npz'))
eid = e['eventId'].astype(np.int64); spark = e['spark']; hasray = e['has_ray']; mult = e['mult']
our = {int(x): (bool(spark[i]), bool(hasray[i]), int(mult[i])) for i, x in enumerate(eid)}

# ---- M3 / FEU1 analyse tree ----
F = uproot.open(ANA)
tn = [k for k in F.keys() if k.split(';')[0] == 'T'][0].split(';')[0]
t = F[tn]
d = t.arrays(['evn', 'MGv2_NClus', 'MGv2_Spark', 'MGv2_ClusAmpl', 'MGv2_ClusSize'], library='np')
m3evn = d['evn'].astype(np.int64)
NClus = np.stack(d['MGv2_NClus'])                             # (N,8)
Ampl = np.stack(d['MGv2_ClusAmpl'])                           # (N,8,300)
Size = np.stack(d['MGv2_ClusSize'])
nclus = NClus.sum(axis=1)                                     # per event, over 8 layers
m3spark = (np.stack(d['MGv2_Spark']) > 0).any(axis=1)         # M3 flags a spark in any layer
_idx = np.arange(Ampl.shape[2])[None, None, :]
m3 = {int(v): i for i, v in enumerate(m3evn)}                 # evn -> row


def clusters_of(ids):
    """flatten valid (amplitude, size) over the M3 layers for the given event ids."""
    rows = [m3[i] for i in ids if i in m3]
    if not rows:
        return np.array([]), np.array([])
    sub = np.array(rows)
    val = _idx < NClus[sub][:, :, None]
    return Ampl[sub][val], Size[sub][val]

# ---- restrict to the evn window the FEU1 files actually cover ----
lo, hi = m3evn.min(), m3evn.max()
in_win = (eid >= lo) & (eid <= hi)
print(f'FEU1 analyse: {len(m3evn)} events, evn [{lo},{hi}]')
print(f'our events in that window: {in_win.sum()}  (sparks {int((spark & in_win).sum())})')

our_sp = eid[in_win & spark]
our_ns = eid[in_win & ~spark]


def present(ids):
    return np.array([i in m3 for i in ids])


# ===== (A) dropped triggers =====
pres_sp = present(our_sp); pres_ns = present(our_ns)
dropA = dict(spark_present_pct=100 * pres_sp.mean(), nonspark_present_pct=100 * pres_ns.mean(),
             spark_dropped_pct=100 * (1 - pres_sp.mean()), nonspark_dropped_pct=100 * (1 - pres_ns.mean()))
print('\n(A) FEU1 trigger present?')
print(f'    our spark    : present {dropA["spark_present_pct"]:.1f}%  (dropped {dropA["spark_dropped_pct"]:.1f}%)')
print(f'    our non-spark: present {dropA["nonspark_present_pct"]:.1f}%  (dropped {dropA["nonspark_dropped_pct"]:.1f}%)')

# ===== (B) M3 self-spark, (C) M3 cluster excess for MATCHED events =====
nc_sp = np.array([nclus[m3[i]] for i in our_sp if i in m3])
nc_ns = np.array([nclus[m3[i]] for i in our_ns if i in m3])
sp_sp = np.array([m3spark[m3[i]] for i in our_sp if i in m3])
sp_ns = np.array([m3spark[m3[i]] for i in our_ns if i in m3])
a_sp, s_sp = clusters_of(list(our_sp))
a_ns, s_ns = clusters_of(list(our_ns))
n_sp_ev, n_ns_ev = len(nc_sp), len(nc_ns)
print('\n(B) M3 raises its OWN spark flag (MGv2_Spark):')
print(f'    when we spark {100 * sp_sp.mean():.2f}%   when we DONT {100 * sp_ns.mean():.2f}%')
print('\n(C) M3 cluster excess (matched events):')
print(f'    clusters/event  spark {nc_sp.mean():.2f}  nonspark {nc_ns.mean():.2f}  '
      f'(median {np.median(nc_sp):.0f} vs {np.median(nc_ns):.0f})')
print(f'    excess cluster amplitude  median spark {np.median(a_sp):.0f} vs nonspark {np.median(a_ns):.0f}  '
      f'(smaller = noise-like)')
print(f'    excess cluster size       median spark {np.median(s_sp):.1f} vs nonspark {np.median(s_ns):.1f}')

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
# (a) M3 not knocked out: dropped triggers + self-spark both ~0
x = np.arange(2)
vals_sp = [dropA['spark_dropped_pct'], 100 * sp_sp.mean()]
vals_ns = [dropA['nonspark_dropped_pct'], 100 * sp_ns.mean()]
axes[0].bar(x - 0.2, vals_sp, 0.4, color='crimson', label='our spark')
axes[0].bar(x + 0.2, vals_ns, 0.4, color='grey', label='our non-spark')
for xi, (a, b) in enumerate(zip(vals_sp, vals_ns)):
    axes[0].text(xi - 0.2, 0.12, f'{a:.1f}%', ha='center', fontsize=9, color='crimson')
    axes[0].text(xi + 0.2, 0.12, f'{b:.1f}%', ha='center', fontsize=9, color='dimgrey')
axes[0].set_xticks(x); axes[0].set_xticklabels(['FEU1 trigger\nDROPPED', 'M3 fires its\nOWN spark'])
axes[0].set_ylabel('% of events'); axes[0].set_ylim(0, 5); axes[0].legend()
axes[0].set_title('(a) M3 readout NOT knocked out\n(no drops, M3 never sparks — all 0%)')
# (b) M3 cluster count excess
bins = np.arange(0, 30)
axes[1].hist(nc_ns, bins=bins, density=True, alpha=0.6, color='grey', label='our non-spark')
axes[1].hist(nc_sp, bins=bins, density=True, alpha=0.6, color='crimson', label='our spark')
axes[1].axvline(nc_ns.mean(), color='k', ls='--', lw=0.8); axes[1].axvline(nc_sp.mean(), color='crimson', ls='--', lw=0.8)
axes[1].set_xlabel('M3 clusters/event (8 layers)'); axes[1].set_ylabel('norm.'); axes[1].legend()
axes[1].set_title('(b) EXCESS M3 clusters when we spark\nmean %.1f vs %.1f (+%.0f%%)'
                  % (nc_sp.mean(), nc_ns.mean(), 100 * (nc_sp.mean() / nc_ns.mean() - 1)))
# (c) the excess clusters are low-amplitude (noise-like)
ab = np.logspace(1.5, 4, 45)
axes[2].hist(a_ns, bins=ab, density=True, alpha=0.6, color='grey', label='our non-spark')
axes[2].hist(a_sp, bins=ab, density=True, alpha=0.6, color='crimson', label='our spark')
axes[2].set_xscale('log'); axes[2].set_xlabel('M3 cluster amplitude'); axes[2].set_ylabel('norm.'); axes[2].legend()
axes[2].set_title('(c) excess clusters are LOW-amplitude\nmedian %.0f vs %.0f -> noise pickup, not tracks'
                  % (np.median(a_sp), np.median(a_ns)))
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_crosstalk.png'), dpi=140, bbox_inches='tight')
print('\nwrote fig_crosstalk.png')

out = dict(evn_window=[int(lo), int(hi)], n_our_in_window=int(in_win.sum()),
           n_matched_spark=n_sp_ev, n_matched_nonspark=n_ns_ev,
           dropped=dropA,
           m3_selfspark_when_we_spark_pct=100 * float(sp_sp.mean()),
           m3_selfspark_when_not_pct=100 * float(sp_ns.mean()),
           m3_clusters_per_event_spark=float(nc_sp.mean()), m3_clusters_per_event_nonspark=float(nc_ns.mean()),
           m3_excess_clusters=float(nc_sp.mean() - nc_ns.mean()),
           m3_cluster_ampl_med_spark=float(np.median(a_sp)), m3_cluster_ampl_med_nonspark=float(np.median(a_ns)),
           m3_cluster_size_med_spark=float(np.median(s_sp)), m3_cluster_size_med_nonspark=float(np.median(s_ns)))
json.dump(out, open(os.path.join(HERE, 'crosstalk_feu1.json'), 'w'), indent=2)
print(json.dumps(out, indent=2))
