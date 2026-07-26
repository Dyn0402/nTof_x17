#!/usr/bin/env python3
"""Empirical single-strip impulse-response template from high-|tan| strips.

Selection: plane |tan| >= TAN_MIN so the direct deposit on one strip spans
<= ~1.5 samples; brightest strip of the cluster, unsaturated, peak in-window.
Aligned on the interpolated 50%-rise crossing, normalized to peak, then a
median template on a fine grid. Validated by predicting the near-vertical
pulse shape as template (x) long-boxcar.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/waveform_first/wfcache.pkl')
OUT = os.path.dirname(CACHE)
d = pickle.load(open(CACHE, 'rb'))
events = d['events']
SNS = 60.0
TS = np.arange(32) * SNS
TAN_MIN = 0.25
SAT_ADC = 3550.0   # ped-subtracted saturation guard (raw 4095 - ped ~330 - margin)

def t50(w):
    ipk = int(np.argmax(w)); a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan

# fine grid template accumulation
GRID = np.arange(-360, 1400, 10.0)
acc, used = [], 0
for eid, ev in events.items():
    for plane in ('x', 'y'):
        tn = abs(ev['tan_x'] if plane == 'x' else ev['tan_y'])
        if tn < TAN_MIN:
            continue
        W = ev[plane]['W'].astype(np.float32)
        amax = W.max(axis=1)
        order = np.argsort(amax)[::-1]
        for i in order[:2]:                      # up to 2 brightest strips
            w = W[i]
            a = w.max(); ipk = int(np.argmax(w))
            if a < 500 or a > SAT_ADC or ipk < 6 or ipk > 20:
                continue
            c = t50(w)
            if not np.isfinite(c):
                continue
            tt = (np.arange(32) - c) * SNS
            acc.append(np.interp(GRID, tt, w / a, left=np.nan, right=np.nan))
            used += 1
acc = np.array(acc)
print('template strips used:', used)
tmpl = np.nanmedian(acc, axis=0)
n_per = np.sum(np.isfinite(acc), axis=0)
tmpl[n_per < 30] = np.nan

# force pre-rise baseline to 0 and store
base = np.nanmedian(tmpl[GRID < -250])
tmpl = tmpl - base
print('pre-rise base', base)

# --- validation: near-vertical shape = tmpl (x) attenuated long boxcar ---
grp = []
for eid, ev in events.items():
    for plane in ('x', 'y'):
        tn = abs(ev['tan_x'] if plane == 'x' else ev['tan_y'])
        if tn > 0.05:
            continue
        W = ev[plane]['W'].astype(np.float32)
        i = int(np.argmax(W.max(axis=1)))
        w = W[i]
        a = w.max(); ipk = int(np.argmax(w))
        if a < 500 or a > SAT_ADC or ipk < 6 or ipk > 22:
            continue
        c = t50(w)
        if np.isfinite(c):
            tt = (np.arange(32) - c) * SNS
            grp.append(np.interp(GRID, tt, w / a, left=np.nan, right=np.nan))
grp = np.array(grp)
vert = np.nanmedian(grp, axis=0)
vert -= np.nanmedian(vert[GRID < -250])

# model: convolve template with exp-attenuated boxcar of duration L ns
def predict(L, lam):
    g = np.zeros_like(GRID)
    tfine = np.arange(0, L, 10.0)
    wgt = np.exp(-lam * tfine)
    ftmpl = np.nan_to_num(tmpl)
    out = np.zeros_like(GRID)
    for t_, w_ in zip(tfine, wgt):
        out += w_ * np.interp(GRID - t_, GRID, ftmpl, left=0, right=0)
    out /= out.max()
    # align to its own t50 like data
    i50 = np.argmax(out >= 0.5)
    t_50 = np.interp(0.5, [out[i50 - 1], out[i50]], [GRID[i50 - 1], GRID[i50]])
    return np.interp(GRID + t_50, GRID, out, left=0, right=0)

best = None
for L in (600, 700, 800, 880, 950, 1050):
    for lam in (0.0, 0.5e-3, 1.0e-3, 1.5e-3, 2.5e-3):
        pred = predict(L, lam)
        m = np.isfinite(vert) & (GRID > -200) & (GRID < 1100)
        chi = np.nansum((pred[m] - vert[m]) ** 2)
        if best is None or chi < best[0]:
            best = (chi, L, lam, pred)
chi, L, lam, pred = best
print(f'best boxcar duration {L} ns, attenuation lambda {lam*1e3:.2f}/us-ish, chi {chi:.3f}')

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(GRID, tmpl, lw=2, label=f'impulse template (n={used})')
q1, q3 = np.nanpercentile(acc, [25, 75], axis=0)
axs[0].fill_between(GRID, q1 - base, q3 - base, alpha=0.25)
axs[0].set_title('empirical impulse response (|tan|>0.25 bright strips)')
axs[0].set_xlabel('t - t50 [ns]'); axs[0].grid(alpha=0.3); axs[0].legend()
axs[1].plot(GRID, vert, lw=2, label='near-vertical median shape (data)')
axs[1].plot(GRID, pred, lw=2, ls='--', label=f'tmpl⊗boxcar L={L}ns, atten={lam*1e3:.1f}e-3/ns')
axs[1].set_title('validation: vertical track = impulse ⊗ gap-long boxcar')
axs[1].set_xlabel('t - t50 [ns]'); axs[1].grid(alpha=0.3); axs[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'template.png'), dpi=110)

np.savez(os.path.join(OUT, 'template.npz'), grid=GRID, tmpl=np.nan_to_num(tmpl),
         boxcar_L=L, atten=lam)
print('saved template.npz')
