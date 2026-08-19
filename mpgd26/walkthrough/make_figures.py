#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py -- every stage of the det3 forward fit, on one real muon.

    ../../.venv/bin/python make_figures.py [--only f1_raw,f9_scan] [--fast]

Writes figures/*.png and steps.json (the numbers the note quotes, so the note
and the figures cannot drift apart).  --fast skips the 220-event ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

import wt
from wt import OWN, N1, N2, DATA, REF, ACC, GREY

OUT = {}


# --------------------------------------------------------------- 1. the data
def f1_raw(cal, wm, st):
    """What comes in: one plane's window, and the core strip with its
    +-1 and +-2 neighbours drawn explicitly."""
    i0 = wt.core_index(st)
    t, raw, pos = st['t'], st['raw'], st['pos']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw=dict(width_ratios=[1.05, 1.25]))

    ax = axes[0]
    ext = [t[0] - 30, t[-1] + 30, len(pos) - 0.5, -0.5]
    im = ax.imshow(raw, aspect='auto', extent=ext, cmap='magma_r',
                   vmin=0, vmax=np.percentile(raw, 99.7))
    ax.set_yticks(range(len(pos)))
    ax.set_yticklabels([f'{p:.1f}' for p in pos], fontsize=7)
    for d, c in ((0, DATA), (1, N1), (-1, N1), (2, N2), (-2, N2)):
        j = i0 + d
        if 0 <= j < len(pos):
            ax.axhline(j, color=c, lw=1.1, alpha=0.75)
            ax.text(t[-1] + 45, j, f'{d:+d}' if d else ' 0', color=c,
                    va='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('time in the DREAM window [ns]')
    ax.set_ylabel('strip position [mm]')
    ax.set_title(f'The measurement: {len(pos)} strips $\\times$ {st["nsamp"]} '
                 'samples')
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.11, label='ADC (pedestal subtracted)')

    ax = axes[1]
    off = 0.0
    step = 1.15 * float(np.abs(raw).max())
    for d in (2, 1, 0, -1, -2):
        j = i0 + d
        if not (0 <= j < len(pos)):
            continue
        c = DATA if d == 0 else (N1 if abs(d) == 1 else N2)
        ax.plot(t, raw[j] + off, 'o-', ms=3.2, lw=1.4, color=c)
        ax.axhline(off, color=GREY, lw=0.6, alpha=0.4)
        ipk = int(np.argmax(raw[j]))
        ax.plot([t[ipk]], [raw[j][ipk] + off], '|', ms=11, color=c)
        ax.text(t[-1] + 40, off + 0.12 * step,
                f'{d:+d}' if d else ' 0  (core)', color=c, fontsize=9,
                fontweight='bold', va='center')
        ax.text(t[-1] + 40, off - 0.16 * step,
                f'{pos[j]:.2f} mm  ch {st["ch"][j]}', color=GREY, fontsize=7.5,
                va='center')
        off -= step
    ax.set_xlim(t[0] - 40, t[-1] + 430)
    ax.set_yticks([])
    ax.set_xlabel('time in the DREAM window [ns]')
    ax.set_title('The core strip and its $\\pm1$, $\\pm2$ neighbours, raw')
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    OUT['raw'] = dict(
        n_strips=len(pos), nsamp=st['nsamp'],
        core_pos=float(pos[i0]), core_ch=int(st['ch'][i0]),
        peak_adc={str(d): float(raw[i0 + d].max()) for d in (-2, -1, 0, 1, 2)
                  if 0 <= i0 + d < len(pos)},
        peak_ns={str(d): float(t[int(np.argmax(raw[i0 + d]))])
                 for d in (-2, -1, 0, 1, 2) if 0 <= i0 + d < len(pos)},
        noise_med=float(np.median(st['noise'])))
    return wt.save(fig, 'f1_raw')


# ------------------------------------------------------- 2. the three numbers
def f2_track(cal, wm, st):
    """The parameterisation: a straight segment cut into 60 ns arrival slices,
    and the same slices laid over the measurement they have to explain."""
    v, K, DT = cal.v_drift, wm.K, wm.DT
    u = np.asarray(wm.UK)
    p0, w, t0 = st['p0'], st['w'], st['t0']
    gap = 30.0
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    z = v * u * 1e-3                       # depth of each slice [mm]
    p = p0 + w * u                         # transverse position of each slice
    inside = z <= gap
    ax.axhspan(0, gap, color='#f3f4f6', zorder=0)
    ax.axhline(0, color='#111827', lw=2.6)
    ax.axhline(gap, color=GREY, lw=2.0, ls='--')
    ax.text(0.02, 0.055, 'mesh — the strips are here', transform=ax.transAxes,
            fontsize=8.5, color='#111827')
    ax.text(0.02, 0.845, 'cathode, 30 mm above', transform=ax.transAxes,
            fontsize=8.5, color=GREY)
    ax.plot(p, z, '-', color=ACC, lw=2.4, zorder=4)
    ax.scatter(p[inside], z[inside], s=26, color=ACC, zorder=5)
    for k in range(0, K, 3):
        if z[k] > gap:
            continue
        ax.annotate('', xy=(p[k], 0.6), xytext=(p[k], z[k] - 0.4),
                    arrowprops=dict(arrowstyle='->', color=OWN, lw=1.0,
                                    alpha=0.55))
        ax.text(p[k] - 0.25, z[k] + 0.5, f'$u_{{{k}}}$ = {u[k]:.0f} ns',
                fontsize=7.5, color=OWN, ha='right')
    ax.plot([p0], [0], 'o', ms=9, mfc='white', mec=ACC, mew=2.2, zorder=6)
    ax.annotate('$p_0$', xy=(p0, 0), xytext=(p0 + 1.4, -3.4), fontsize=11,
                color=ACC, arrowprops=dict(arrowstyle='->', color=ACC, lw=1.2))
    ax.set_ylim(-5.5, gap + 3)
    ax.set_xlim(min(p[inside]) - 2.5, p0 + 3.5)
    ax.set_xlabel('transverse position along the strips [mm]')
    ax.set_ylabel('drift depth $z = v\\,u$  [mm]')
    ax.set_title(f'$K$ = {K} slices of $\\Delta t$ = {DT:.0f} ns, '
                 'on one straight line')

    ax = axes[1]
    t, raw, pos = st['t'], st['raw'], st['pos']
    ext = [t[0] - 30, t[-1] + 30, pos[-1] + 0.39, pos[0] - 0.39]
    ax.imshow(raw, aspect='auto', extent=ext, cmap='magma_r',
              vmin=0, vmax=np.percentile(raw, 99.7))
    ax.plot(t0 + u, p, 'o-', color='#22d3ee', lw=2.0, ms=5, mec='#0e7490',
            label='the fitted slices, $p_0 + w\\,u_k$ at $t_0 + u_k$')
    ax.axvline(t0, color='#22d3ee', ls=':', lw=1.4)
    ax.text(t0, pos[0] - 0.1, ' $t_0$', color='#0e7490', fontsize=10,
            va='bottom')
    ax.set_xlabel('time in the DREAM window [ns]')
    ax.set_ylabel('strip position [mm]')
    ax.set_title('the same three numbers, on the measurement')
    ax.legend(fontsize=8.5, loc='lower left', labelcolor='#0e7490')
    ax.grid(False)
    fig.tight_layout()
    OUT['track'] = dict(K=int(K), dt_ns=float(DT), v=float(v),
                        p0=p0, w_um_ns=w * 1e3, t0=t0,
                        span_mm=float(abs(w) * u[-1]),
                        drift_full_ns=float(gap * 1e3 / v),
                        tan_raw=float(w * 1e3 / v))
    return wt.save(fig, 'f2_track')


# --------------------------------------------- 3. geometry -> strip fractions
def f3_fractions(cal, wm, st):
    pos, p0, w = st['pos'], st['p0'], st['w']
    h = st['hyper']
    F = wm.strip_fractions(pos, p0, w, h['sigma_p0'], h['Dp'])   # (nstrip, K)
    u = np.asarray(wm.UK)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2),
                             gridspec_kw=dict(width_ratios=[1.15, 1.25, 0.85]))

    ax = axes[0]
    im = ax.imshow(F.T, aspect='auto', cmap='Blues', origin='lower',
                   extent=[pos[0] - 0.39, pos[-1] + 0.39, -0.5, wm.K - 0.5])
    ax.set_xlabel('strip position [mm]')
    ax.set_ylabel('depth slice $k$')
    ax.set_title('$F_{sk}$ — fraction of slice $k$ landing on strip $s$')
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = axes[1]
    ks = [1, 5, 9, 13]
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(ks)))
    for k, c in zip(ks, cols):
        ax.plot(pos, F[:, k], 'o-', ms=4, color=c, lw=1.6,
                label=f'$k$={k}  ($u$={u[k]:.0f} ns, $z$={u[k] * cal.v_drift * 1e-3:.1f} mm)')
    ax.set_xlabel('strip position [mm]')
    ax.set_ylabel('fraction of the slice')
    ax.set_title('Four slices, spread by the strip integral')
    ax.legend(fontsize=8)

    ax = axes[2]
    sig = np.sqrt(h['sigma_p0'] ** 2 + h['Dp'] ** 2 * u)
    ax.plot(u, sig, color=OWN, lw=2.0, label='$\\sqrt{\\sigma_{p0}^2+D_p^2u}$')
    ax.axhline(h['sigma_p0'], color=GREY, ls=':', lw=1.2)
    ax.text(u[-1], h['sigma_p0'] * 0.90,
            f"$\\sigma_{{p0}}$ = {h['sigma_p0']:.3f} mm", fontsize=8.5, color=GREY,
            ha='right')
    ax.axhline(cal.pitch_mm, color=ACC, ls='--', lw=1.2)
    ax.text(u[-1], cal.pitch_mm * 1.02, 'one pitch', fontsize=8.5, color=ACC,
            ha='right')
    ax.set_xlabel('$u$ [ns]')
    ax.set_ylabel('transverse width [mm]')
    ax.set_title('The cloud, slice by slice')
    ax.set_ylim(0, max(cal.pitch_mm * 1.25, sig.max() * 1.15))
    fig.tight_layout()
    OUT['fractions'] = dict(sigma_p0=float(h['sigma_p0']), Dp=float(h['Dp']),
                            pitch=float(cal.pitch_mm),
                            sigma_end=float(sig[-1]),
                            max_frac=float(F.max()))
    return wt.save(fig, 'f3_fractions')


# ------------------------------------------------------------- 4. the kernel
def _kern(cal, wm, plane, t, hyper):
    h = dict(hyper)
    H1, H2 = wm._copy_responses(plane, t, h)
    H0 = np.interp(t, np.asarray(cal.grid, float),
                   np.asarray(cal.tmpl[plane], float), left=0, right=0)
    k = h.get('kY', 1.0) if plane == 'y' else h.get('cX', 1.0)
    c1 = h['c1'] * k
    r = h.get('c2_over_c1')
    c2 = (float(r) * c1) if r is not None else h['c2'] * k
    return H0, c1 * H1, c2 * H2, c1, c2


def f4_kernel(cal, wm, st):
    t = np.linspace(-100, 1400, 900)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    info = {}
    for ax, plane in zip(axes, ('y', 'x')):
        H0, K1, K2, c1, c2 = _kern(cal, wm, plane, t, st['hyper'])
        n = H0.max()
        ax.fill_between(t, 0, H0 / n, color=OWN, alpha=0.20)
        ax.plot(t, H0 / n, color=OWN, lw=2.2, label='own charge')
        ax.plot(t, K1 / n, color=N1, lw=2.2,
                label=f'$\\pm1$ copy   $c_1$={c1:.3f}')
        ax.plot(t, K2 / n, color=N2, lw=2.2,
                label=f'$\\pm2$ copy   $c_2$={c2:.3f}')
        tau = st['hyper']['tau_s']
        ax.annotate('', xy=(t[np.argmax(K1)], 0.30), xytext=(t[np.argmax(H0)], 0.30),
                    arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.1))
        ax.text(0.5 * (t[np.argmax(K1)] + t[np.argmax(H0)]), 0.325,
                f'$\\tau_s$ = {tau:.0f} ns', fontsize=9, color=GREY, ha='center')
        ax.set_xlabel('time since the charge arrived [ns]')
        ax.set_title(f'{plane.upper()} plane' +
                     ('  (strips along $y$ — the resistive direction)'
                      if plane == 'y' else '  (the perpendicular view)'))
        ax.legend(fontsize=9)
        info[plane] = dict(c1=float(c1), c2=float(c2),
                           ratio=float(c2 / c1) if c1 else float('nan'),
                           peak1_over_peak0=float(K1.max() / n),
                           peak2_over_peak0=float(K2.max() / n),
                           lag_ns=float(t[np.argmax(K1)] - t[np.argmax(H0)]))
    axes[0].set_ylabel('response, normalised to the own-charge peak')
    fig.tight_layout()
    OUT['kernel'] = dict(tau_s=float(st['hyper']['tau_s']),
                         sigma_s=float(st['hyper']['sigma_s']),
                         kY=float(st['hyper'].get('kY', 1.0)),
                         share_mode=str(wm.SHARE_MODE), **info)
    return wt.save(fig, 'f4_kernel')


# ----------------------------------------- 5. one slice -> five raw waveforms
def f5_column(cal, wm, st):
    """One column of the design matrix: unit charge in ONE depth slice, and the
    waveform it puts on the strip it landed on and on its +-1, +-2 neighbours.

    Each strip's trace is split by WHERE the charge came from, in the same
    three colours the rest of the note uses: what landed on the strip itself
    (the geometric tail of the cloud), what the +-1 kernel copied in, and what
    the +-2 kernel copied in.
    """
    pos, p0, w, t0, h = st['pos'], st['p0'], st['w'], st['t0'], st['hyper']
    t = st['t']
    # Pick one early and one late slice that are well CENTRED on a strip -- a
    # slice straddling a boundary splits geometrically and hides the point the
    # figure is making (that the outer strips are kernel copies, not charge).
    Fall = wm.strip_fractions(pos, p0, w, h['sigma_p0'], h['Dp'])
    peak = Fall.max(axis=0)
    half = wm.K // 2
    ks = [int(np.argmax(peak[:half])), int(half + np.argmax(peak[half:]))]
    ds = (2, 1, 0, -1, -2)
    fig = plt.figure(figsize=(12.2, 8.6))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.25] + [1] * 5, hspace=0.32,
                          wspace=0.18)
    col = {}
    Mfull = wm.build_matrix(st['plane'], pos, p0, w, t0, h)
    Mown = wm.build_matrix(st['plane'], pos, p0, w, t0, wt._zero_all(h))
    Mo1 = wm.build_matrix(st['plane'], pos, p0, w, t0, wt._zero_c2(h))
    for j, k in enumerate(ks):
        A = Mfull[:, k].reshape(len(pos), wm.NSAMP)
        A0 = Mown[:, k].reshape(len(pos), wm.NSAMP)
        A1 = Mo1[:, k].reshape(len(pos), wm.NSAMP) - A0
        A2 = A - A0 - A1
        F = wm.strip_fractions(pos, p0, w, h['sigma_p0'], h['Dp'])[:, k]
        ic = int(np.argmax(F))

        ax = fig.add_subplot(gs[0, j])
        ax.bar(pos, F, width=0.70, color=OWN, alpha=0.55)
        ax.set_xlim(pos[ic] - 3.2, pos[ic] + 3.2)
        ax.set_ylabel('geometric\nfraction', fontsize=9)
        ax.set_xlabel('strip position [mm]', fontsize=9)
        ax.set_title(f'slice $k$ = {k}:  arrives at $t_0+{wm.UK[k]:.0f}$ ns,  '
                     f'lands at {p0 + w * wm.UK[k]:.2f} mm', fontsize=11)
        ymax = 1.10 * float(A.max())
        for i, d in enumerate(ds):
            s_ = ic + d
            axw = fig.add_subplot(gs[1 + i, j])
            if 0 <= s_ < len(pos):
                axw.stackplot(t, A0[s_], A1[s_], A2[s_], colors=(OWN, N1, N2),
                              alpha=0.9)
                axw.plot(t, A[s_], color='#111827', lw=0.9)
                lab = ('the strip it landed on' if d == 0 else f'{d:+d}')
                axw.text(0.985, 0.80, f'{pos[s_]:.2f} mm   {lab}',
                         transform=axw.transAxes, ha='right', fontsize=8.5,
                         color=(OWN if d == 0 else (N1 if abs(d) == 1 else N2)))
            axw.set_ylim(min(0, 1.15 * float(A.min())), ymax)
            axw.set_yticks([])
            if i < len(ds) - 1:
                axw.set_xticklabels([])
            else:
                axw.set_xlabel('time in the window [ns]')
            axw.spines['left'].set_visible(False)
        col[str(k)] = dict(u=float(wm.UK[k]), p=float(p0 + w * wm.UK[k]),
                           frac_core=float(F[ic]),
                           amp={str(d): float(A[ic + d].max())
                                for d in ds if 0 <= ic + d < len(pos)},
                           own_frac={str(d): float(A0[ic + d].max() /
                                                   max(A[ic + d].max(), 1e-12))
                                     for d in ds if 0 <= ic + d < len(pos)})
    fig.text(0.5, 0.985, 'one unit of charge in one slice  '
             '$\\rightarrow$  a waveform on five strips', ha='center',
             fontsize=12, color='#111827', fontweight='bold')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=OWN, label='landed on this strip'),
                        Patch(fc=N1, label='copied in from $\\pm1$'),
                        Patch(fc=N2, label='copied in from $\\pm2$')],
               loc='upper center', bbox_to_anchor=(0.5, 0.966), ncol=3,
               fontsize=9.5)
    OUT['column'] = col
    return wt.save(fig, 'f5_column')


# ------------------------------------------------ 6. NNLS solves the charges
def f6_nnls(cal, wm, st):
    q, u = st['q'], np.asarray(wm.UK)
    z = u * cal.v_drift * 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

    ax = axes[0]
    M = wm.build_matrix(st['plane'], st['pos'], st['p0'], st['w'], st['t0'],
                        st['hyper'])
    im = ax.imshow(M.T, aspect='auto', cmap='Blues', origin='lower',
                   extent=[0, M.shape[0], -0.5, wm.K - 0.5])
    ax.set_xlabel('the fit vector: strip $\\times$ sample, '
                  f'{len(st["pos"])} $\\times$ {st["nsamp"]} = {M.shape[0]} rows')
    ax.set_ylabel('depth slice $k$')
    ax.set_title('The design matrix $A$ — one column per slice, '
                 'sharing already inside')
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = axes[1]
    ax.bar(z, q, width=z[1] - z[0], color=ACC, alpha=0.8, align='center')
    ax.set_xlabel('drift depth $z=v\\,u$ [mm]')
    ax.set_ylabel('fitted charge $q_k$ [ADC]')
    ax.set_title('$q = \\arg\\min_{q \\geq 0}\\|Aq-y\\|$ — solved, not searched')
    ax.axvline(30.0, color='#111827', lw=1.6)
    ax.text(30.0, ax.get_ylim()[1] * 0.55, ' cathode', fontsize=8.5,
            color='#111827', rotation=90, va='top')
    cum = np.cumsum(q) / max(q.sum(), 1e-9)
    for frac, c, lab in ((0.5, GREY, 'median'), (0.9, GREY, '90 %')):
        zz = float(np.interp(frac, cum, z))
        ax.axvline(zz, color=c, ls='--', lw=1.1)
        ax.text(zz, ax.get_ylim()[1] * 0.95, f' {lab}', fontsize=8, color=c)
    fig.tight_layout()
    OUT['nnls'] = dict(rows=int(M.shape[0]), cols=int(wm.K),
                       q_sum=float(q.sum()), n_zero=int((q <= 0).sum()),
                       z_med=float(np.interp(0.5, cum, z)),
                       z90=float(np.interp(0.9, cum, z)))
    return wt.save(fig, 'f6_nnls')


# ------------------------------------------------------- 7. model vs the data
def f7_modelvsdata(cal, wm, st, name='f7_modelvsdata'):
    i0 = wt.core_index(st)
    t, W = st['t'], st['W']
    ds = [d for d in (-2, -1, 0, 1, 2, 3) if 0 <= i0 + d < len(st['pos'])]
    ncol = 3
    nrow = int(np.ceil(len(ds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.6, 3.3 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    frac = {}
    for ax, d in zip(axes, ds):
        s = i0 + d
        own, s1, s2 = st['own'][s], st['sh1'][s], st['sh2'][s]
        ax.stackplot(t, own, s1, s2, colors=(OWN, N1, N2), alpha=0.85,
                     labels=('own charge', "$\\pm1$ neighbours'",
                             "$\\pm2$ neighbours'"))
        ax.plot(t, W[s], 'o', ms=3.4, color=DATA, label='measured')
        ax.plot(t, st['full'][s], '-', color='#111827', lw=1.0)
        pk = st['full'][s].max()
        f = (s1 + s2).max() / pk if pk > 0 else 0.0
        frac[str(d)] = float(f)
        ax.set_title(f'{st["pos"][s]:.2f} mm   ({d:+d} from the core)'
                     if d else f'{st["pos"][s]:.2f} mm   (core strip)',
                     fontsize=10)
        ax.text(0.97, 0.92, f'neighbours: {100 * f:.0f} % of the peak',
                transform=ax.transAxes, ha='right', fontsize=8.5, color=N1)
    for ax in axes[len(ds):]:
        ax.axis('off')
    axes[0].legend(fontsize=8.5, loc='upper left')
    for ax in axes[-ncol:]:
        ax.set_xlabel('time [ns]')
    for j in range(0, len(ds), ncol):
        axes[j].set_ylabel('ADC (gain corrected)')
    fig.tight_layout()
    OUT['decompose'] = dict(neighbour_frac=frac,
                            chi2_dof=float(st['chi2'] / max(st['dof'], 1)))
    return wt.save(fig, name)


def f8_residual(cal, wm, st):
    W, M, t, pos = st['W'], st['full'], st['t'], st['pos']
    R = (W - M) / st['noise'][:, None]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ext = [t[0], t[-1], pos[-1] + 0.39, pos[0] - 0.39]
    vm = float(np.percentile(np.abs(W), 99.7))
    for ax, A, ttl, cm, kw in (
            (axes[0], W, 'measured', 'magma_r', dict(vmin=0, vmax=vm)),
            (axes[1], M, 'model', 'magma_r', dict(vmin=0, vmax=vm)),
            (axes[2], R, 'residual / noise', 'RdBu_r', dict(vmin=-6, vmax=6))):
        im = ax.imshow(A, aspect='auto', extent=ext, cmap=cm, **kw)
        ax.set_title(ttl)
        ax.set_xlabel('time [ns]')
        ax.grid(False)
        fig.colorbar(im, ax=ax, pad=0.02)
    axes[0].set_ylabel('strip position [mm]')
    fig.tight_layout()
    OUT['residual'] = dict(rms_pull=float(np.sqrt((R ** 2).mean())),
                           worst_pull=float(np.abs(R).max()),
                           rms_adc=float(np.sqrt(((W - M) ** 2).mean())),
                           rms_pct_peak=float(100 * np.sqrt(((W - M) ** 2).mean())
                                              / W.max()),
                           noise_adc=float(np.median(st['noise'])))
    return wt.save(fig, 'f8_residual')


# ------------------------------------------------------------ 9. the search
def f9_scan(cal, wm, st):
    p0, w, t0, v = st['p0'], st['w'], st['t0'], cal.v_drift
    ws = np.linspace(w - 0.006, w + 0.006, 61)
    ps = np.linspace(p0 - 2.4, p0 + 2.4, 61)
    ts = np.linspace(t0 - 90, t0 + 90, 61)
    cw = np.array([wt.chi2_at(wm, st, p0, x, t0) for x in ws])
    cp = np.array([wt.chi2_at(wm, st, x, w, t0) for x in ps])
    ct = np.array([wt.chi2_at(wm, st, p0, w, x) for x in ts])
    c0 = st['chi2']

    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.9))
    ax = axes[0]
    ax.plot(np.degrees(np.arctan(ws * 1e3 / v)), cw / c0, color=OWN, lw=2.0)
    ax.axvline(np.degrees(np.arctan(st['tan'])), color=OWN, ls='--', lw=1.1)
    ax.axvline(np.degrees(np.arctan(st['tan_ref'])), color=REF, ls='-', lw=1.6)
    ax.text(np.degrees(np.arctan(st['tan_ref'])), 0.86 * (cw / c0).max(),
            ' M3 reference', color=REF, fontsize=8.5, ha='left')
    ax.set_xlabel('track angle $\\theta$ [deg]')
    ax.set_ylabel('$\\chi^2$ / $\\chi^2_{min}$')
    ax.set_title('the angle scan')
    ax = axes[1]
    ax.plot(ps, cp / c0, color=ACC, lw=2.0)
    ax.axvline(p0, color=ACC, ls='--', lw=1.1)
    ax.axvline(st['p0_ref'], color=REF, lw=1.6)
    ax.set_xlabel('$p_0$ [mm]')
    ax.set_title('the position scan')
    ax = axes[2]
    ax.plot(ts, ct / c0, color=N2, lw=2.0)
    ax.axvline(t0, color=N2, ls='--', lw=1.1)
    ax.axvline(st['t0_pred'], color=REF, lw=1.6)
    ax.text(st['t0_pred'], 0.86 * (ct / c0).max(), ' trigger prior',
            color=REF, fontsize=8.5)
    ax.set_xlabel('$t_0$ [ns]')
    ax.set_title('the start-time scan')

    ax = axes[3]
    Ps, Ws = np.meshgrid(np.linspace(p0 - 1.6, p0 + 1.6, 41),
                         np.linspace(w - 0.004, w + 0.004, 41))
    Z = np.empty_like(Ps)
    for a in range(Ps.shape[0]):
        for b in range(Ps.shape[1]):
            Z[a, b] = wt.chi2_at(wm, st, Ps[a, b], Ws[a, b], t0)
    cf = ax.contourf(Ps, np.degrees(np.arctan(Ws * 1e3 / v)), np.log10(Z / c0),
                     levels=22, cmap='viridis_r')
    ax.plot([p0], [np.degrees(np.arctan(w * 1e3 / v))], '*', ms=15,
            color='white', mec='#111827')
    ax.set_xlabel('$p_0$ [mm]')
    ax.set_ylabel('$\\theta$ [deg]')
    ax.set_title('$\\log_{10}\\,\\chi^2/\\chi^2_{min}$')
    ax.grid(False)
    fig.colorbar(cf, ax=ax, pad=0.02)
    fig.tight_layout()

    def _hw(x, c):
        m = c <= 2 * c.min()
        return float(x[m].max() - x[m].min())
    OUT['scan'] = dict(
        chi2_min=float(c0), chi2_dof=float(c0 / max(st['dof'], 1)),
        d_theta_deg=float(np.degrees(np.arctan(st['tan'])) -
                          np.degrees(np.arctan(st['tan_ref']))),
        theta_fit=float(np.degrees(np.arctan(st['tan']))),
        theta_ref=float(np.degrees(np.arctan(st['tan_ref']))),
        p0_fit=float(p0), p0_ref=float(st['p0_ref']),
        width_theta=float(_hw(np.degrees(np.arctan(ws * 1e3 / v)), cw)),
        width_p0=_hw(ps, cp), width_t0=_hw(ts, ct))
    return wt.save(fig, 'f9_scan')


# ------------------------------------------------- 10. what was replaced
def f10_ratio(cal, wm, st):
    """The kernel this walkthrough runs on, against the one the FROZEN MPGD26
    production reco actually used -- which has the +-2 copy larger than the
    +-1 copy."""
    calp, wmp = wt.load(wt.BUNDLE_PROD)
    evs = wt.events()
    stp = wt.fit_event(calp, wmp, evs[wt.EID])
    t = np.linspace(-100, 1400, 900)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    for ax, (c, w_, ttl, sub, colr) in zip(axes[:2], (
            (calp, wmp, 'SUPERSEDED',
             'the frozen MPGD26 reco ran with this', '#b91c1c'),
            (cal, wm, 'IN USE HERE',
             'the only ordering a resistive film can give', '#15803d'))):
        H0, K1, K2, c1, c2 = _kern(c, w_, 'y', t, dict(c.hyper))
        n = H0.max()
        ax.fill_between(t, 0, H0 / n, color=OWN, alpha=0.18)
        ax.plot(t, H0 / n, color=OWN, lw=2.0, label='own')
        ax.plot(t, K1 / n, color=N1, lw=2.4, label=f'$\\pm1$   $c_1$={c1:.3f}')
        ax.plot(t, K2 / n, color=N2, lw=2.4, label=f'$\\pm2$   $c_2$={c2:.3f}')
        ax.set_title(f'{ttl}   —   $c_2/c_1$ = {c2 / c1:.2f}', color=colr,
                     fontsize=11.5)
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha='center',
                fontsize=8.8, color=colr)
        ax.set_xlabel('time since arrival [ns]')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.16, 1.05)
    axes[0].set_ylabel('response, normalised to the own peak')

    ax = axes[2]
    s2 = wt.core_index(st) + 2
    ax.plot(st['t'], st['W'][s2], 'o', ms=4.2, color=DATA, label='measured',
            zorder=5)
    for stx, ls, lab, cc in ((stp, '--', 'superseded', '#b91c1c'),
                             (st, '-', 'in use here', '#15803d')):
        sx = wt.core_index(stx) + 2
        ax.plot(stx['t'], stx['full'][sx], ls, color=cc, lw=1.6,
                label=f'model, {lab}')
        ax.plot(stx['t'], stx['sh2'][sx], ls, color=N2, lw=2.0,
                label=f"$\\pm2$ copies, {lab}")
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('ADC (gain corrected)')
    ax.set_title('the $+2$ strip — where the two differ most')
    ax.legend(fontsize=8)
    OUT['ratio'] = dict(
        prod=dict(c1=float(calp.hyper['c1'] * calp.hyper['kY']),
                  c2=float(calp.hyper['c2'] * calp.hyper['kY']),
                  ratio=float(calp.hyper['c2'] / calp.hyper['c1']),
                  theta=float(np.degrees(np.arctan(stp['tan']))),
                  chi2_dof=float(stp['chi2'] / stp['dof']),
                  tau_s=float(calp.hyper['tau_s']),
                  sigma_s=float(calp.hyper['sigma_s']),
                  p0=float(stp['p0']), w=float(stp['w'] * 1e3),
                  t0=float(stp['t0']), t0_pred=float(stp['t0_pred']),
                  bundle=os.path.basename(wt.BUNDLE_PROD)),
        cur=dict(c1=float(cal.hyper['c1'] * cal.hyper['kY']),
                 c2=float(0.6 * cal.hyper['c1'] * cal.hyper['kY']),
                 ratio=0.6,
                 theta=float(np.degrees(np.arctan(st['tan']))),
                 chi2_dof=float(st['chi2'] / st['dof']),
                 tau_s=float(cal.hyper['tau_s']),
                 sigma_s=float(cal.hyper['sigma_s']),
                 p0=float(st['p0']), w=float(st['w'] * 1e3),
                 t0=float(st['t0']), t0_pred=float(st['t0_pred']),
                 bundle=os.path.basename(wt.BUNDLE)),
        theta_ref=float(np.degrees(np.arctan(st['tan_ref']))),
        d_theta=float(np.degrees(np.arctan(st['tan'])) -
                      np.degrees(np.arctan(stp['tan']))))
    wt.load()          # restore this walkthrough's calibration
    return wt.save(fig, 'f10_ratio')


# ------------------------------------------------------------ 11. the ensemble
_EV = None


def _init(cache, bundle):
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))


def _one(payload):
    """Fit one event on one bundle. Returns raw w (um/ns) and the reference, so
    the w -> angle mapping can be applied afterwards."""
    eid, bundle = payload
    from wft.calib import CalibrationBundle
    from wft import model as wm
    cal = CalibrationBundle.load(bundle)
    out = {}
    for plane in ('x', 'y'):
        try:
            st = wt.fit_event(cal, wm, _EV[eid], plane)
        except Exception:
            continue
        out[plane] = (st['tan_ref'], float(st['w'] * 1e3), st['p0_ref'],
                      st['p0'])
    return eid, out


def _run_bundle(bundle, eids):
    from concurrent.futures import ProcessPoolExecutor
    got = {'x': [], 'y': []}
    with ProcessPoolExecutor(max_workers=7, initializer=_init,
                             initargs=(wt.CACHE, bundle)) as pool:
        for _e, o in pool.map(_one, [(e, bundle) for e in eids], chunksize=5):
            for p, tup in o.items():
                got[p].append(tup)
    return {p: np.array(got[p], float) for p in ('x', 'y')}


def _fit_w0kw(arr, v):
    """bench/set_w0.py's recipe: w0 = median(w - v tan_ref) over |tan| < 0.30,
    then kw = median((w - w0) / (v tan_ref)) over 0.10 < |tan| < 0.40."""
    out = {}
    for plane, a in arr.items():
        tanr, w = a[:, 0], a[:, 1]
        s = np.abs(tanr) < 0.30
        w0 = float(np.median(w[s] - v * tanr[s])) if s.sum() else 0.0
        s1 = (np.abs(tanr) > 0.10) & (np.abs(tanr) < 0.40)
        kw = (float(np.median((w[s1] - w0) / (v * tanr[s1])))
              if s1.sum() >= 30 else 1.0)
        out[plane] = (w0, kw)
    return out


def _to_deg(a, v, w0kw, plane):
    w0, kw = w0kw[plane]
    tan = (a[:, 1] - w0) / (kw * v)
    return np.degrees(np.arctan(a[:, 0])), np.degrees(np.arctan(tan))


def f11_ensemble(cal, wm, st, nmax=220):
    """Held-out events, on BOTH calibrations, with each one's w -> angle map
    measured on the training half only (never on the events being scored)."""
    evs = wt.events()
    eids = sorted(evs)
    train, held = eids[:180], eids[180:][:nmax]
    v = cal.v_drift
    arms = {}
    for name, bundle in (('cur', wt.BUNDLE), ('prod', wt.BUNDLE_PROD)):
        tr_ = _run_bundle(bundle, train)
        hd = _run_bundle(bundle, held)
        arms[name] = (hd, _fit_w0kw(tr_, v))

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    res = {}
    cols = dict(x=ACC, y=OWN)
    ax = axes[0, 0]
    tr_deg, tf_deg = _to_deg(arms['cur'][0]['y'], v, arms['cur'][1], 'y')
    ax.plot([-25, 25], [-25, 25], color=GREY, lw=1.0, ls='--')
    ax.plot(tr_deg, tf_deg, '.', ms=5, color=OWN, alpha=0.7)
    ax.set_xlabel('M3 reference angle [deg]')
    ax.set_ylabel('forward-fit angle [deg]')
    ax.set_title(f'Y plane, {len(tr_deg)} held-out events')

    for plane in ('x', 'y'):
        a = arms['cur'][0][plane]
        trd, tfd = _to_deg(a, v, arms['cur'][1], plane)
        d = tfd - trd
        dp = a[:, 3] - a[:, 2]
        s68 = float(np.percentile(np.abs(d - np.median(d)), 68))
        p68 = float(np.percentile(np.abs(dp - np.median(dp)), 68))
        res[plane] = dict(n=len(a), bias=float(np.median(d)), s68=s68,
                          pos_bias=float(np.median(dp)), pos_s68=p68,
                          w0=arms['cur'][1][plane][0],
                          kw=arms['cur'][1][plane][1])
        ap = arms['prod'][0][plane]
        trp, tfp = _to_deg(ap, v, arms['prod'][1], plane)
        dpr = tfp - trp
        res[plane]['s68_prod'] = float(
            np.percentile(np.abs(dpr - np.median(dpr)), 68))
        res[plane]['w0_prod'] = arms['prod'][1][plane][0]
        res[plane]['kw_prod'] = arms['prod'][1][plane][1]
        axes[0, 1].hist(d, bins=np.arange(-6, 6.01, 0.5), histtype='step',
                        lw=2.0, color=cols[plane],
                        label=f'{plane.upper()}: $\\sigma_{{68}}$ = {s68:.2f}$^\\circ$ '
                              f'(superseded {res[plane]["s68_prod"]:.2f}$^\\circ$)')
        axes[0, 1].hist(dpr, bins=np.arange(-6, 6.01, 0.5), histtype='step',
                        lw=1.1, ls='--', color=cols[plane])
        axes[1, 0].hist(dp, bins=np.arange(-3, 3.01, 0.2), histtype='step',
                        lw=2.0, color=cols[plane],
                        label=f'{plane.upper()}: $\\sigma_{{68}}$ = {1e3 * p68:.0f} $\\mu$m, '
                              f'bias {1e3 * np.median(dp):+.0f} $\\mu$m')
    axes[0, 1].set_xlabel('fit $-$ reference angle [deg]')
    axes[0, 1].set_ylabel('events')
    axes[0, 1].set_title('angle residual (dashed = superseded kernel)')
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].set_xlabel('fit $-$ reference position at the mesh [mm]')
    axes[1, 0].set_ylabel('events')
    axes[1, 0].set_title('position residual (reference-limited, see text)')
    axes[1, 0].legend(fontsize=9)

    ax = axes[1, 1]
    bins = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.45)]
    rng = np.random.default_rng(20260818)
    imp = {}
    for plane in ('x', 'y'):
        a = arms['cur'][0][plane]
        xs, ys, es = [], [], []
        for lo, hi in bins:
            m = (np.abs(a[:, 0]) >= lo) & (np.abs(a[:, 0]) < hi)
            if m.sum() < 8:
                continue
            r = a[m, 1] / a[m, 0]
            bs = np.median(r[rng.integers(0, len(r), size=(400, len(r)))],
                           axis=1)
            xs.append(0.5 * (lo + hi))
            ys.append(float(np.median(r)))
            es.append(float(bs.std()))
        ax.errorbar(xs, ys, yerr=es, fmt='o-', color=cols[plane], lw=1.8,
                    capsize=3, label=f'{plane.upper()}')
        imp[plane] = dict(x=xs, v=ys, err=es,
                          spread=float(max(ys) - min(ys)),
                          err_typ=float(np.mean(es)))
    ax.axhline(v, color='#111827', lw=1.4, ls=':')
    ax.text(0.04, 0.90, f'calibration value {v:.1f} $\\mu$m/ns',
            transform=ax.transAxes, fontsize=9, color='#111827')
    ax.set_xlabel(r'$|\tan\theta|$ of the reference track')
    ax.set_ylabel('implied drift velocity [$\\mu$m/ns]')
    ax.set_title('the internal check: $w/\\tan\\theta_{ref}$, 220 held-out events')
    ax.legend(fontsize=9)
    fig.tight_layout()

    ang = json.load(open(wt.ANGLES))
    OUT['implied_v'] = imp
    OUT['ensemble'] = dict(held=res, full_run={
        p: dict(n=int(ang['planes'][p]['n']),
                s68=float(ang['planes'][p]['s68_deg']),
                bias=float(ang['planes'][p]['bias_deg']),
                implied_v_spread=float(ang['planes'][p]['implied_v_spread']))
        for p in ('x', 'y')})
    return wt.save(fig, 'f11_ensemble')


ALL = [f1_raw, f2_track, f3_fractions, f4_kernel, f5_column, f6_nnls,
       f7_modelvsdata, f8_residual, f9_scan, f10_ratio, f11_ensemble]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()
    wt.style()
    cal, wm = wt.load()
    evs = wt.events()
    st = wt.fit_event(cal, wm, evs[wt.EID])
    OUT['event'] = dict(eid=wt.EID, plane=wt.PLANE,
                        bundle=os.path.basename(wt.BUNDLE),
                        run='sat_det3 / long_run_resist_490V_drift_1000V',
                        v_drift=float(cal.v_drift),
                        w0=float(cal.w0.get(wt.PLANE, 0.0)),
                        kw=float(cal.kw.get(wt.PLANE, 1.0)),
                        tan_raw=float(st['w'] * 1e3 / cal.v_drift),
                        tan_corr=float((st['w'] * 1e3 - cal.w0.get(wt.PLANE, 0.0))
                                       / (cal.kw.get(wt.PLANE, 1.0) * cal.v_drift)),
                        tan_ref=float(st['tan_ref']),
                        t0_pred=float(st['t0_pred']),
                        t0_prior_sigma=float(cal.t0_prior_sigma),
                        peak_adc=float(st['W'].max()),
                        chi2_dof=float(st['chi2'] / st['dof']))
    want = [f for f in ALL if not a.only or f.__name__ in a.only.split(',')]
    if a.fast:
        want = [f for f in want if f is not f11_ensemble]
    for f in want:
        print(f.__name__)
        f(cal, wm, st)
    p = os.path.join(wt.HERE, 'steps.json')
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(OUT)
    json.dump(old, open(p, 'w'), indent=1)
    print('wrote steps.json')


if __name__ == '__main__':
    main()
