#!/usr/bin/env python3
"""
deconv_kernel.py -- measure the sharing kernel's SHAPE, rather than assume it.

At normal incidence every strip sees the same ionisation column, so each
strip's waveform is one common signal C(t) (column arrival folded with the
amplifier) passed through a lateral transfer function:

        W_d(t) = (h_d * C)(t)      =>      W_d(f)/W_0(f) = G_d(f)

Both the column and the electronics cancel in the ratio.  g_d(t) = IFFT(G_d) is
therefore the sharing kernel itself, with no shaper template, no drift model, no
v_drift and no assumed functional form.  The cancellation holds ONLY at normal
incidence -- at an angle each strip sees a different depth slice and C(t) stops
being common, which is why the head-on runs are the ones that can do this.

WHAT THE MODEL ASSUMES.  wft's shipped 'delay' branch builds the copy as the
impulse response translated by tau_s and Gaussian-smeared by sigma_s:

        g_1(t) = c1 * Gauss(t - tau_s, sigma_s)

a SYMMETRIC bump, no prompt term, no tail.  An RC sheet should instead give a
prompt spike plus a one-sided exponential.  This script decides between them.

    ../../../.venv/bin/python deconv_kernel.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
STAGE = '/media/dylan/data/x17/sps_run53_det4_check/staging/run_71/'
ARMS = [('d4_kernel_fit_raw450.npz', 'RAW, 156 V/cm', '#2a78d6'),
        ('d4_kernel_fit_raw275.npz', 'RAW,  95 V/cm', '#eb6834')]
# the frozen production bundle's assumption, for the overlay
TAU_S, SIGMA_S, C1, C2, KY = 145.5, 12.07, 0.0509, 0.0580, 2.875

INK, MUTED, LINE = '#0b0b0b', '#52514e', '#8a8983'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8, 'axes.grid': True,
    'grid.color': '#e6e5e1', 'grid.linewidth': 0.7, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.labelcolor': MUTED, 'text.color': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'legend.frameon': False})


def wiener(Wd, W0, lam_frac, npad=4):
    n = len(W0) * npad
    A, B = np.fft.rfft(W0, n), np.fft.rfft(Wd, n)
    G = B * np.conj(A) / (np.abs(A) ** 2 + lam_frac * np.max(np.abs(A) ** 2))
    g = np.fft.fftshift(np.fft.irfft(G, n))
    return g[n // 2 - len(W0) // 2: n // 2 + len(W0) // 2]


def stats(g, t):
    p = np.clip(g, 0, None)
    A = p.sum()
    if A <= 0:
        return {}
    k = t > t[int(np.argmax(p))]
    kk = k & (g > 0) & (t < t[int(np.argmax(p))] + 900)
    tau = np.nan
    if kk.sum() > 4:
        c = np.polyfit(t[kk], np.log(g[kk]), 1)
        tau = float(-1 / c[0]) if c[0] < 0 else np.nan
    return dict(area=float(A), centroid=float((p * t).sum() / A),
                peak_t=float(t[int(np.argmax(p))]),
                frac_neg=float(p[t < 0].sum() / A), tail_tau=tau)


out = {'meta': dict(source='run_71 RAW per-offset mean waveforms',
                    assumed=dict(tau_s=TAU_S, sigma_s=SIGMA_S, c1=C1, c2=C2,
                                 kY=KY))}

# --- regularisation stability -------------------------------------------
print('regularisation stability (Y view, d=+1, area / tail tau):')
for fn, lab, _ in ARMS:
    z = np.load(STAGE + fn, allow_pickle=True)
    row = []
    for lam in (0.005, 0.01, 0.02, 0.05, 0.10):
        g = wiener(z['y__W1'], z['y__W0'], lam)
        t = (np.arange(len(g)) - len(g) // 2) * 60.0
        s = stats(g, t)
        row.append(f"lam={lam:.3f}: {s['area']:.3f}/{s['tail_tau']:.0f}ns")
    print(f'  {lab}: ' + '  '.join(row))
    out.setdefault('stability', {})[lab] = row

# --- the measurement ------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
for fn, lab, col in ARMS:
    z = np.load(STAGE + fn, allow_pickle=True)
    rec = {}
    for ax, d in zip(axs, (1, 2)):
        gs = [wiener(z[f'y__W{s*d}'], z['y__W0'], 0.02) for s in (1, -1)]
        g = 0.5 * (gs[0] + gs[1])                    # symmetrise +d and -d
        t = (np.arange(len(g)) - len(g) // 2) * 60.0
        rec[f'd{d}'] = stats(g, t)
        rec[f'd{d}']['asym'] = float(
            np.clip(gs[0], 0, None).sum() / max(np.clip(gs[1], 0, None).sum(), 1e-9))
        ax.plot(t, g, '-o', ms=3.2, lw=1.7, color=col, label=lab)
    out[lab] = rec

# the model's assumption, on the same axes
t = (np.arange(61) - 30) * 60.0
for ax, d, amp in zip(axs, (1, 2), (C1 * KY, C2 * KY)):
    gauss = amp * np.exp(-0.5 * ((t - d * TAU_S) / max(SIGMA_S, 1)) ** 2)
    gauss *= 1.0 / max(gauss.sum(), 1e-9) * amp      # unit-area x amplitude
    ax.plot(t, gauss, '--', lw=2.0, color='#9E2B25',
            label=f'the FILM term wft adds\n(Gauss at {d*TAU_S:.0f} ns, '
                  f'$\\sigma$={SIGMA_S:.0f} ns)')
    # wft's OTHER neighbour term is geometric: it multiplies the same impulse
    # response as the centre strip, so in g_d(t) it is a delta at t = 0.  Drawn
    # so the comparison is against the model's FULL neighbour content, not half
    # of it -- the point is that neither term can make a 300 ns tail.
    ax.annotate('', xy=(0, 0.052 if d == 1 else 0.030), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='#9E2B25', lw=2.0))
    ax.text(-30, 0.055 if d == 1 else 0.033,
            r'+ its geometric term:' '\n' r'a $\delta$ at $t=0$',
            ha='right', va='bottom', fontsize=8.5, color='#9E2B25')
    ax.axvline(0, color=LINE, lw=1)
    ax.axhline(0, color=LINE, lw=1)
    ax.set_xlim(-700, 1700)
    ax.set_xlabel('time relative to the centre strip  [ns]')
    ax.set_title(f'sharing kernel to $\\pm${d}', fontsize=11.5)
    ax.legend(fontsize=8.5)
axs[0].set_ylabel('measured $g_d(t)$   (Y view)')
fig.suptitle('The sharing kernel, deconvolved from RAW head-on data — '
             'a prompt spike plus a one-sided tail, not a translated bump',
             fontsize=11.5, y=1.02)
fig.savefig(os.path.join(FIG, 'deconv_kernel.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print('\nwrote', os.path.join(FIG, 'deconv_kernel.png'))

print('\nmeasured Y-view kernel (symmetrised over +d / -d):')
print(f"{'arm':16}{'d':>3}{'area':>8}{'peak t':>9}{'centroid':>10}"
      f"{'tail tau':>10}{'+d/-d':>8}{'wt at t<0':>11}")
for _, lab, _ in ARMS:
    for d in (1, 2):
        s = out[lab][f'd{d}']
        print(f"{lab:16}{d:3d}{s['area']:8.3f}{s['peak_t']:+9.0f}"
              f"{s['centroid']:+10.0f}{s['tail_tau']:10.0f}"
              f"{s['asym']:8.2f}{100*s['frac_neg']:10.1f} %")

with open(os.path.join(HERE, 'deconv_kernel.json'), 'w') as f:
    json.dump(out, f, indent=1)
print('wrote deconv_kernel.json')
