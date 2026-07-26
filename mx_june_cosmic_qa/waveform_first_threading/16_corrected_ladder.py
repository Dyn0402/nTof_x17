#!/usr/bin/env python3
"""Cheap baseline: matched-filter strip times + S-curve correction + ladder.

The compression correction delta(dt_meas) is derived on the TRAINING split
only, parametrized against the measurable time-into-cluster
dt_meas = t_mf - min(t_core), and applied on the test split. Ladder refit
gives (slope -> w, mesh p0 at u=0). Output in freefit-format for wf15.
"""
import os, sys, pickle, json
import numpy as np

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
rows = pickle.load(open(os.path.join(BASE, 'mf_strips.pkl'), 'rb'))
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
train = set(split['train'])
test = set(split['test'])
hj = json.load(open(os.path.join(BASE, 'hyper_ref.json')))
V = hj['v']
CORE = 0.30

MODE = 'slope-remap'   # per-strip delta(dt_meas) suffers regression dilution
                        # (profile flattens to +-15 ns); remap the fitted slope
                        # instead, exactly one calibration level above raw.

# ---------- derive correction on training ----------
# residual r = t_mf - t_ref_ladder(pos) - t0ref ; x = dt_meas
xs, ys, planes = [], [], []
for r in rows:
    if r['eid'] not in train or abs(r['tan']) < 0.08:
        continue
    t_ref = (r['pos'] - r['p0']) / (r['tan'] * V * 1e-3)
    core = (r['relamp'] >= CORE) & (t_ref > -120) & (t_ref < 1100)
    if core.sum() < 3:
        continue
    t0 = np.median(r['t'][core] - t_ref[core])
    tmin = r['t'][core].min()
    for i in np.where(core)[0]:
        xs.append(r['t'][i] - tmin)
        ys.append(r['t'][i] - t_ref[i] - t0)
        planes.append(r['plane'])
xs, ys = np.array(xs), np.array(ys)
planes = np.array(planes)

BINS = np.arange(0, 1100, 60.0)
corr = {}
for p in ('x', 'y'):
    m = planes == p
    prof = []
    for k in range(len(BINS) - 1):
        s = m & (xs >= BINS[k]) & (xs < BINS[k + 1])
        prof.append(np.median(ys[s]) if s.sum() > 25 else np.nan)
    prof = np.array(prof)
    # fill nans by extension
    ok = np.isfinite(prof)
    prof = np.interp(np.arange(len(prof)), np.where(ok)[0], prof[ok])
    corr[p] = prof
    print(p, 'correction profile [ns]:', np.round(prof, 0))

def apply_corr(p, dt):
    c = corr[p]
    k = np.clip(((dt - BINS[0]) / 60.0).astype(int), 0, len(c) - 1)
    return c[k]

# ---------- apply on test, refit ladders ----------
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

# ---------- fit raw mf ladder per event (train + test) ----------
def ladder(r):
    core = r['relamp'] >= CORE
    if core.sum() < 3 or np.ptp(r['pos'][core]) < 1.0:
        return None
    li = robust_line(r['pos'][core], r['t'][core])
    if li is None or li[0] == 0:
        return None
    w = 1.0 / li[0]
    i0 = np.argmin(r['t'][core])
    p0 = r['pos'][core][i0]              # earliest core strip (prod-style anchor)
    return w, p0

fits = {}
for r in rows:
    L = ladder(r)
    if L is not None:
        fits[(r['eid'], r['plane'])] = (L[0], L[1], r['tan'], r['p0'])

# slope remap w_true ~ alpha*w_meas + beta from training
remap = {}
for p in ('x', 'y'):
    wm, wt = [], []
    for (eid, pl), (w, p0, tan, p0r) in fits.items():
        if pl != p or eid not in train or abs(tan) < 0.05 or abs(w) > 0.05:
            continue
        wm.append(w); wt.append(tan * V * 1e-3)
    wm, wt = np.array(wm), np.array(wt)
    keep = np.ones(len(wm), bool)
    for _ in range(4):
        A = np.vstack([wm[keep], np.ones(keep.sum())]).T
        coef, *_ = np.linalg.lstsq(A, wt[keep], rcond=None)
        rres = wt - (coef[0] * wm + coef[1])
        s = 1.4826 * np.median(np.abs(rres[keep] - np.median(rres[keep]))) + 1e-12
        keep = np.abs(rres - np.median(rres[keep])) < 3 * s
    remap[p] = (float(coef[0]), float(coef[1]))
    print(f'{p}: w remap alpha={coef[0]:.3f} beta={coef[1]*1e3:+.3f} um/ns '
          f'(n={keep.sum()})')

out = {}
for (eid, pl), (w, p0, tan, p0r) in fits.items():
    if eid not in test:
        continue
    a, b = remap[pl]
    wc = a * w + b
    ent = out.setdefault(eid, dict(eid=eid))
    ent[pl] = dict(tan_ref=tan, p0_ref=p0r, p0=float(p0), w=float(wc),
                   t0=np.nan, chi2=np.nan, dof=1, amax=np.nan)
res = list(out.values())
pickle.dump(res, open(os.path.join(BASE, 'freefit_corrladder.pkl'), 'wb'))
print(f'saved freefit_corrladder.pkl ({len(res)} events)')
