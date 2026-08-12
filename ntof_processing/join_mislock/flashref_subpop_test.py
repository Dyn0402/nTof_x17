#!/usr/bin/env python3
"""Is the -0.983 ms structure a mis-tagged-flash subpopulation?

On the HEALTHY segment run_79/stat090_0001 x 224572 (truth K, T0 known):

For every physics event, count candidates in a window around the SHIFTED
prediction  t_pred - 0.983 ms  (width +-10 us), and around the TRUE
prediction (+-1 us).  Then ask:

  1. concentration: are the shifted-window counts uniform across bunches
     (rate artifact) or concentrated in a few bunches (mis-tag bursts)?
  2. identity: do the events that populate the shifted window match at
     zero as well (accidental floor) or are they preferentially the
     UNMATCHED events (subpopulation with a broken time base)?
  3. shape: fine histogram of (residual + 0.983 ms) at 200 ns bins --
     does per-event structure exist inside the broad bump?
"""
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path.insert(0, str(REPO))

from ntof_processing.slim_pipeline import config as C          # noqa: E402
from ntof_processing.slim_pipeline import clockfit as cf       # noqa: E402
from ntof_processing.slim_pipeline.slim import (               # noqa: E402
    Segment, _bind_ntof, join_events, bunch_table, pass1_candidates)

OUT = Path(__file__).parent / 'flashref_subpop_results.json'
V12 = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
K_TRUE, T0_TRUE = 1.101174e-04, -254.66
LAG = -0.983e6          # ns
WIDE = 10_000.0         # +-10 us window around the shifted prediction
CORE = 1_000.0          # +-1 us window around the true prediction

seg = Segment('run_79', 'stat090_0001', 224572, ntof_source=V12)
_bind_ntof(seg)
ev = join_events(seg, log=lambda *a: None)
btbl, keep = bunch_table(ev, log=lambda *a: None)
if not keep.all():
    ev = ev[keep].reset_index(drop=True)
phys = ~ev['is_flash'].to_numpy()
ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)[phys]
ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)[phys]
cd, offs, thr = pass1_candidates(seg, np.unique(ev_b),
                                 log=lambda *a: None)
cb, ct = cd['bunch'], cd['t']
print(f'{ev_t.size:,} events, {ct.size:,} candidates')

# per-bunch sorted candidate arrays
order = np.lexsort((ct, cb))
cb_s, ct_s = cb[order], ct[order]
bounds = {b: (lo, hi) for b, lo, hi in zip(
    *(lambda u, idx: (u, idx, np.r_[idx[1:], cb_s.size]))(
        *np.unique(cb_s, return_index=True)))}

pred = ev_t * (1.0 + K_TRUE) + T0_TRUE     # cf.predict without arm terms
n_shift = np.zeros(ev_t.size, np.int32)    # candidates near pred + LAG
n_core = np.zeros(ev_t.size, np.int32)     # candidates near pred
res_fine = []                              # (r - LAG) for the fine histogram

for i in range(ev_t.size):
    b = ev_b[i]
    if b not in bounds:
        continue
    lo, hi = bounds[b]
    t = ct_s[lo:hi]
    p = pred[i]
    a1 = np.searchsorted(t, p + LAG - WIDE)
    a2 = np.searchsorted(t, p + LAG + WIDE)
    n_shift[i] = a2 - a1
    if a2 > a1:
        res_fine.append(t[a1:a2] - p - LAG)
    b1 = np.searchsorted(t, p - CORE)
    b2 = np.searchsorted(t, p + CORE)
    n_core[i] = b2 - b1

matched = n_core > 0
res_fine = np.concatenate(res_fine) if res_fine else np.zeros(0)

# 1. concentration across bunches: shifted-window counts per bunch vs events
import collections
per_bunch_shift = collections.Counter()
per_bunch_ev = collections.Counter()
for b, ns in zip(ev_b, n_shift):
    per_bunch_shift[b] += int(ns)
    per_bunch_ev[b] += 1
bs = np.array(sorted(per_bunch_shift))
s_counts = np.array([per_bunch_shift[b] for b in bs], float)
e_counts = np.array([per_bunch_ev[b] for b in bs], float)
rate = s_counts / np.maximum(e_counts, 1)
mu, sd = rate.mean(), rate.std()
top = np.argsort(rate)[::-1][:10]
print(f'shifted-window rate/event: mean {mu:.2f} sd {sd:.2f} '
      f'(uniform Poisson would give sd~{np.sqrt(mu/e_counts.mean()):.2f})')
print('top bunches by rate:', [(int(bs[i]), round(float(rate[i]), 1))
                               for i in top[:5]])

# 2. identity: shifted-window occupancy for matched vs unmatched events
sh_m = n_shift[matched].mean()
sh_u = n_shift[~matched].mean()
print(f'matched events   ({matched.sum():,}): mean shifted-window count '
      f'{sh_m:.3f}')
print(f'unmatched events ({(~matched).sum():,}): mean shifted-window count '
      f'{sh_u:.3f}')
print(f'ratio unmatched/matched: {sh_u/max(sh_m,1e-9):.2f}')

# 3. shape: fine histogram of residual-minus-LAG
h, edges = np.histogram(res_fine, bins=200, range=(-WIDE, WIDE))
peak_bin = int(h.argmax())
floor = float(np.median(h))
print(f'fine shape at 100ns bins: peak {h.max()} at '
      f'{0.5*(edges[peak_bin]+edges[peak_bin+1]):+.0f} ns, floor {floor:.0f}, '
      f'total {res_fine.size:,}')

json.dump(dict(
    mean_rate=mu, sd_rate=sd, exp_poisson_sd=float(np.sqrt(mu/e_counts.mean())),
    top_bunches=[(int(bs[i]), float(rate[i])) for i in top],
    n_matched=int(matched.sum()), n_unmatched=int((~matched).sum()),
    shift_mean_matched=float(sh_m), shift_mean_unmatched=float(sh_u),
    fine_hist=dict(counts=h.tolist(), lo=float(edges[0]),
                   bin=float(edges[1]-edges[0])),
), open(OUT, 'w'), indent=1)
print('wrote', OUT)
