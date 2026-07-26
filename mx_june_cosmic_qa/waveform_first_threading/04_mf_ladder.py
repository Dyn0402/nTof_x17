#!/usr/bin/env python3
"""Matched-filter per-strip timing → ladder slope vs reference.

For every matched event/plane: template-fit (amplitude, time) per corridor
strip via cross-correlation on a 5 ns grid; robust-fit the time-position
ladder over core strips; compare slope s [ns/mm] with the reference-implied
1/(v*tan_ref) for a range of v. Also record per-strip residual vs the
ref-implied ladder as a function of relative amplitude / position-in-cluster
(to see where sharing distorts).
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
tz = np.load(os.path.join(BASE, 'template.npz'))
GRID, TMPL = tz['grid'], tz['tmpl']
SNS = 60.0
TS = np.arange(32) * SNS
SAT = 3550.0

# template bank: shifted/normalized templates sampled on the 32-sample comb
SHIFTS = np.arange(100.0, 1500.0, 5.0)   # t50-time shifts to scan [ns]
BANK = np.stack([np.interp(TS - s, GRID, TMPL, left=0, right=0) for s in SHIFTS])
BNORM = (BANK * BANK).sum(axis=1)

def mf_fit(w, mask=None):
    """Return (amp, t50, chi2min) from matched filter."""
    if mask is None:
        mask = np.ones(32, bool)
    b = BANK[:, mask]
    bn = (b * b).sum(axis=1)
    dot = b @ w[mask]
    amp = dot / np.maximum(bn, 1e-9)
    # chi2 up to const: ||w||^2 - amp*dot  (for amp>0)
    score = np.where(amp > 0, amp * dot, -np.inf)
    j = int(np.argmax(score))
    return float(amp[j]), float(SHIFTS[j]), j

rows = []
for eid, ev in events.items():
    for plane in ('x', 'y'):
        P = ev[plane]
        W = P['W'].astype(np.float32)
        pos = P['pos'].astype(np.float64)
        noise = P['noise'].astype(np.float32)
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0 = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        amax_all = W.max(axis=1)
        thr = np.maximum(6.0 * noise, 60.0)
        cand = np.where(amax_all > thr)[0]
        if len(cand) < 3:
            continue
        amps, times = {}, {}
        for i in cand:
            w = W[i].copy()
            m = w < SAT
            a, t, _ = mf_fit(w, m)
            if a > thr[i]:
                amps[i] = a; times[i] = t
        if len(amps) < 3:
            continue
        idx = np.array(sorted(amps))
        A = np.array([amps[i] for i in idx])
        T = np.array([times[i] for i in idx])
        Ppos = pos[idx]
        amax = A.max()
        rows.append(dict(eid=eid, plane=plane, tan=tn, p0=p0,
                         pos=Ppos, amp=A, t=T, relamp=A / amax))

print('plane-clusters with >=3 mf strips:', len(rows))
pickle.dump(rows, open(os.path.join(BASE, 'mf_strips.pkl'), 'wb'))

# ---------- ladder slopes: core strips, robust line ----------
def robust_line(x, y, clip=2.5, n_iter=4):
    keep = np.ones(len(x), bool)
    p = None
    for _ in range(n_iter):
        if keep.sum() < 3:
            return None
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r - np.median(r[keep])) < clip * s
    return p

CORE = 0.30
res = {p: {'s': [], 'tan': [], 'n': []} for p in ('x', 'y')}
for r in rows:
    m = r['relamp'] >= CORE
    if m.sum() < 3 or np.ptp(r['pos'][m]) < 1.0:
        continue
    p = robust_line(r['pos'][m], r['t'][m])
    if p is None:
        continue
    res[r['plane']]['s'].append(p[0])
    res[r['plane']]['tan'].append(r['tan'])
    res[r['plane']]['n'].append(m.sum())

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, plane in zip(axs, ('x', 'y')):
    s = np.array(res[plane]['s']); t = np.array(res[plane]['tan'])
    ok = np.abs(t) > 0.05
    # implied velocity per event: v = 1/(s*tan)  [mm/ns] -> um/ns
    v = 1.0 / (s[ok] * t[ok]) * 1000.0
    v = v[(v > 0) & (v < 80)]
    ax.hist(v, bins=60, range=(0, 80))
    med = np.median(v)
    ax.axvline(34, color='r', ls='--', label='v_geom = 34')
    ax.axvline(med, color='k', ls='-', label=f'median = {med:.1f}')
    ax.set_title(f'{plane}: implied v = 1/(s_mf * tan_ref)  [um/ns]')
    ax.legend()
    print(plane, 'n', ok.sum(), 'median implied v', med,
          'q25/q75', np.percentile(v, [25, 75]).round(1))
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'mf_ladder_v.png'), dpi=110)

# ---------- per-strip signed residual vs ref ladder, by relamp ----------
# use v=34 to draw the ref ladder in time; residual dt = t_meas - t_ref(pos)
V = 34.0e-3
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
for ax, plane in zip(axs2, ('x', 'y')):
    dts, rel, du = [], [], []
    for r in rows:
        if r['plane'] != plane or abs(r['tan']) < 0.08:
            continue
        t_ref = (r['pos'] - r['p0']) / (r['tan'] * V)   # u = time since mesh crossing
        okm = (t_ref > -120) & (t_ref < 1100)
        if okm.sum() < 3:
            continue
        # free per-event t0: median offset over core strips
        core = okm & (r['relamp'] > CORE)
        if core.sum() < 3:
            continue
        t0 = np.median(r['t'][core] - t_ref[core])
        dts.append(r['t'][okm] - t_ref[okm] - t0)
        rel.append(r['relamp'][okm])
        du.append(t_ref[okm])
    dts = np.concatenate(dts); rel = np.concatenate(rel); du = np.concatenate(du)
    # profile vs u (depth-time)
    bins = np.arange(0, 1100, 100)
    ib = np.digitize(du, bins) - 1
    med = [np.median(dts[(ib == k) & (rel > CORE)]) for k in range(len(bins) - 1)]
    medlo = [np.median(dts[(ib == k) & (rel <= CORE)]) if ((ib == k) & (rel <= CORE)).sum() > 20 else np.nan
             for k in range(len(bins) - 1)]
    ax.plot(bins[:-1] + 50, med, 'o-', label='core strips (rel>0.3)')
    ax.plot(bins[:-1] + 50, medlo, 's--', label='skirt strips (rel<0.3)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('u = ref drift time since mesh [ns] (v=34)')
    ax.set_ylabel('median (t_mf - t_ref - t0) [ns]')
    ax.set_title(f'{plane}: strip-time residual vs ref ladder')
    ax.legend(); ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(BASE, 'mf_residual_vs_depth.png'), dpi=110)
print('done')
