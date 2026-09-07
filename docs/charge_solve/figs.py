#!/usr/bin/env python3
"""
Figures for "The charge solve" — a deep dive on the design matrix A and the
NNLS step inside the waveform-first fit.

Everything is generated from the live `sat_det3` products: the frozen
calibration bundle (`calib_bundle_r06`, the corrected sharing kernel) and real
waveform windows from the 400-event ref-pinned calibration cache. Nothing is schematic.

    ../../.venv/bin/python figs.py            # -> $CS_FIGDIR (default scratchpad)

Numbers quoted in the note are written to `numbers.json` beside the figures, so
the prose and the plots cannot drift apart.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import nnls

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'docs', 'wft_reference', 'figsrc'))

import wftdoc as K                                        # noqa: E402
from wftdoc import C, CHROME                              # noqa: E402
from wft import model as wm                               # noqa: E402
from wft.calib import effective_c2                        # noqa: E402

FIGDIR = os.environ.get(
    'CS_FIGDIR',
    '/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17-mpgd26/'
    'cf7ef626-6174-476c-b483-f2699f32d221/scratchpad/amat/figs')
EVENT = 1663          # the spine display event (clean, inclined, unsaturated)
SAT_EVENT = 2950      # 33 saturated samples — the censoring demonstration
PLANE = 'x'

N = {}                # numbers harvested for the note


def save(fig, name, pad=0.25):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout(pad=pad)
    p = os.path.join(FIGDIR, name + '.png')
    fig.savefig(p, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  {name}.png  {os.path.getsize(p)/1024:6.0f} kB')
    return p


# --------------------------------------------------------------- the problem
class Case:
    """One plane of one event, fitted, with every intermediate kept."""

    def __init__(self, evs, eid, plane=PLANE, cal=None, fit=True):
        e = evs[eid]
        self.eid, self.plane, self.e = eid, plane, e
        self.P = K.trim_window(e[plane])
        if np.asarray(self.P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(self.P['W']).shape[1])
        self.W, self.noise, self.pos, self.sat = wm.prep_plane(self.P, plane)
        self.tan_ref = float(e[f'tan_{plane}'])
        self.p0_ref = float(e[f'ref_mesh_{plane}'])
        w_ref = self.tan_ref * cal.v_drift * 1e-3
        if fit:
            r = wm.fit_plane_raw(self.P, plane, self.p0_ref, w_ref, 400.0)
        else:
            r = wm.fit_plane_raw(self.P, plane, self.p0_ref, w_ref, 400.0,
                                 fix_p0w=(self.p0_ref, w_ref))
        self.fit = r
        self.p0, self.w, self.t0 = r['p0'], r['w'], r['t0']
        self.n_strip = self.W.shape[0]
        self.build()

    def build(self, p0=None, w=None, t0=None, hyper=None):
        p0 = self.p0 if p0 is None else p0
        w = self.w if w is None else w
        t0 = self.t0 if t0 is None else t0
        hyper = hyper or wm.HYPER
        self.M = wm.build_matrix(self.plane, self.pos, p0, w, t0, hyper)
        self.T = self.M.reshape(self.n_strip, wm.NSAMP, wm.K)
        ok = ~self.sat.reshape(-1)
        self.ok = ok
        self.Wt = np.repeat(1.0 / self.noise, wm.NSAMP)
        self.A = (self.M * self.Wt[:, None])[ok]
        self.y = (self.W / self.noise[:, None]).reshape(-1)[ok]
        self.q, rn = nnls(self.A, self.y, maxiter=50 * wm.K)
        self.chi2 = float(rn * rn)
        self.dof = int(ok.sum())
        return self

    def model(self, q=None):
        q = self.q if q is None else q
        return (self.M @ q).reshape(self.n_strip, wm.NSAMP)

    def pieces(self, p0=None, w=None, t0=None):
        """(F, H0, H1, H2, c1, c2) — the factors build_matrix multiplies."""
        p0 = self.p0 if p0 is None else p0
        w = self.w if w is None else w
        t0 = self.t0 if t0 is None else t0
        h = wm.HYPER
        F = wm.strip_fractions(self.pos, p0, w, h['sigma_p0'], h['Dp'])
        # same branch build_matrix takes: exact interpolation off the 5 ns grid
        t0q = round(t0 / wm.T0_STEP) * wm.T0_STEP
        if abs(t0 - t0q) > 1e-9:
            tmpl, _ = wm._templates(self.plane, h['sigma_s'])
            base = wm.TS[:, None] - (t0 + wm.UK[None, :])
            H0 = np.interp(base, wm.TGRID, tmpl, left=0, right=0)
            H1, H2 = wm._copy_responses(self.plane, base, h)
        else:
            H0, H1, H2 = wm._time_tensors(self.plane, t0q, h)
        kk = h.get('kY', 1.0) if self.plane == 'y' else h.get('cX', 1.0)
        c1, c2 = h['c1'] * kk, h['c2'] * kk
        r = h.get('c2_over_c1')
        if r is not None:
            c2 = float(r) * c1
        return F, H0, H1, H2, c1, c2


def lawson_hanson(A, y, tol=1e-10, maxit=400):
    """Lawson–Hanson NNLS, instrumented. Returns (x, log). Verified equal to
    scipy.optimize.nnls to 2e-12 on the spine event."""
    n = A.shape[1]
    P = np.zeros(n, bool)
    x = np.zeros(n)
    log = []
    g = A.T @ (y - A @ x)
    it = 0
    while (~P).any() and g[~P].max() > tol and it < maxit:
        it += 1
        idx = np.where(~P)[0]
        j = idx[int(np.argmax(g[idx]))]
        grad_in = g.copy()
        P[j] = True
        nb = 0
        while True:
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], y, rcond=None)[0]
            if s[P].min() > 0:
                break
            nb += 1
            neg = P & (s <= 0)
            alpha = (x[neg] / (x[neg] - s[neg])).min()
            x = x + alpha * (s - x)
            P &= (np.abs(x) > 1e-12)
            x[~P] = 0.0
        x = s
        g = A.T @ (y - A @ x)
        log.append(dict(it=it, entered=int(j), passive=np.where(P)[0].copy(),
                        x=x.copy(), chi2=float(((A @ x - y) ** 2).sum()),
                        grad=grad_in, backtracks=nb))
    return x, log


# ---------------------------------------------------------------- 1. the data
def fig_window(cs):
    fig = plt.figure(figsize=(11.2, 3.8))
    gs = GridSpec(1, 3, width_ratios=[1.15, 1.0, 0.95], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cs.W, aspect='auto', origin='lower', cmap='magma',
                   extent=[-30, wm.NSAMP * wm.SNS - 30,
                           cs.pos[0] - 0.39, cs.pos[-1] + 0.39])
    ax.set_xlabel('time within the DAQ window [ns]')
    ax.set_ylabel('strip position [mm]')
    ax.set_title(f'the data: {cs.n_strip} strips × {wm.NSAMP} samples '
                 f'= {cs.n_strip * wm.NSAMP} numbers', loc='left')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('ADC (gain-corrected)', color=CHROME)
    cb.ax.tick_params(colors=CHROME)

    ax = fig.add_subplot(gs[1])
    amp = cs.W.max(axis=1)
    order = np.argsort(amp)[::-1][:6]
    for j, i in enumerate(sorted(order)):
        ax.plot(np.arange(wm.NSAMP) * wm.SNS, cs.W[i],
                color=plt.cm.viridis(j / 5.5), lw=1.4,
                label=f'{cs.pos[i]:.1f} mm')
    ax.axhline(0, color=CHROME, lw=0.7)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('ADC')
    ax.set_title('the six brightest strips: each a single smooth pulse',
                 loc='left')
    ax.legend(ncol=2, fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    ax.bar(cs.pos, amp, width=0.7, color=C['blue'], alpha=0.85)
    ax.axhline(5 * cs.noise.mean(), color=C['red'], lw=1.0, ls='--',
               label='5σ noise')
    ax.set_xlabel('strip position [mm]')
    ax.set_ylabel('peak ADC')
    ax.set_title('peak amplitude per strip', loc='left')
    ax.legend()
    N['n_strip'] = int(cs.n_strip)
    N['n_samp'] = int(wm.NSAMP)
    N['n_row'] = int(cs.n_strip * wm.NSAMP)
    N['peak_adc'] = float(cs.W.max())
    N['noise_adc'] = float(np.median(cs.noise))
    return save(fig, 'f01_window')


# ------------------------------------------------- 2. building one column
def fig_column_build(cs, k=7):
    F, H0, H1, H2, c1, c2 = cs.pieces()
    ts = np.arange(wm.NSAMP) * wm.SNS
    col_own = np.outer(F[:, k], H0[:, k])
    Fs = np.zeros_like(F)
    Fs[1:] = F[:-1]
    Fs[:-1] += F[1:]
    Fs2 = np.zeros_like(F)
    Fs2[2:] = F[:-2]
    Fs2[:-2] += F[2:]
    col_1 = c1 * np.outer(Fs[:, k], H1[:, k])
    col_2 = c2 * np.outer(Fs2[:, k], H2[:, k])
    col = cs.T[:, :, k]
    assert np.allclose(col, col_own + col_1 + col_2, atol=1e-12)

    fig = plt.figure(figsize=(11.4, 6.2))
    gs = GridSpec(2, 4, height_ratios=[0.85, 1.0], hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.barh(cs.pos, F[:, k], height=0.62, color=C['teal'])
    ax.set_ylabel('strip position [mm]')
    ax.set_xlabel('fraction of the bin’s charge')
    ax.set_title(f'① WHERE  ·  F$_{{i,{k}}}$', loc='left')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ts, H0[:, k], color=C['orange'])
    ax.axvline(cs.t0 + wm.UK[k], color=CHROME, ls=':', lw=1.0)
    ax.set_xlabel('sample time [ns]')
    ax.set_ylabel('response')
    ax.set_title(f'② WHEN  ·  h(t − t₀ − u$_{{{k}}}$)', loc='left')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ts, H0[:, k], color=C['orange'], label='own strip  h')
    ax.plot(ts, H1[:, k], color=C['purple'], label='±1 copy  h₁')
    ax.plot(ts, H2[:, k], color=C['pink'], label='±2 copy  h₂')
    ax.set_xlabel('sample time [ns]')
    ax.set_title('③ the neighbours’ copies', loc='left')
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    ax.text(0.0, 0.98,
            'column k  =  outer(F$_{:,k}$, h$_k$)\n'
            '        + c₁ · outer(F$^{±1}_{:,k}$, h₁$_{,k}$)\n'
            '        + c₂ · outer(F$^{±2}_{:,k}$, h₂$_{,k}$)\n\n'
            f'c₁ = {c1:.3f}   c₂ = {c2:.3f}\n'
            f'({cs.plane} plane, det3 frozen bundle)\n\n'
            'Three outer products.\n'
            'Space × time, nothing else.\n\n'
            'The whole column is what ONE unit\n'
            'of charge at depth bin k would\n'
            'have produced in this window.',
            va='top', ha='left', fontsize=9, color=CHROME,
            family='monospace', transform=ax.transAxes)

    ext = [-30, wm.NSAMP * wm.SNS - 30, cs.pos[0] - 0.39, cs.pos[-1] + 0.39]
    vmax = col.max()
    for j, (m, t) in enumerate([
            (col_own, 'own strip only'),
            (col_1, f'±1 copies  (×{c1:.3f})'),
            (col_2, f'±2 copies  (×{c2:.3f})'),
            (col, 'the column: their sum')]):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(m, aspect='auto', origin='lower', cmap='magma',
                  extent=ext, vmin=0, vmax=vmax)
        ax.set_xlabel('time [ns]')
        if j == 0:
            ax.set_ylabel('strip position [mm]')
        ax.set_title(t, loc='left', fontsize=9.5)
    N['c1'] = float(c1)
    N['c2'] = float(c2)
    N['share_frac_x'] = float(1 - np.linalg.norm(col_own) / np.linalg.norm(col))
    return save(fig, 'f02_column_build')


# ---------------------------------------------------------- 3. the atlas
def fig_atlas(cs):
    ext = [-30, wm.NSAMP * wm.SNS - 30, cs.pos[0] - 0.39, cs.pos[-1] + 0.39]
    vmax = cs.T.max()
    fig, axs = plt.subplots(3, 6, figsize=(12.2, 5.4), sharex=True, sharey=True)
    for k in range(wm.K):
        ax = axs.ravel()[k]
        ax.imshow(cs.T[:, :, k], aspect='auto', origin='lower', cmap='magma',
                  extent=ext, vmin=0, vmax=vmax)
        z = wm.UK[k] * 36.6 * 1e-3
        ax.set_title(f'k={k}   u={wm.UK[k]:.0f} ns   z={z:.1f} mm',
                     fontsize=8, loc='left')
        ax.grid(False)
        if k % 6 == 0:
            ax.set_ylabel('pos [mm]', fontsize=8)
        if k >= 12:
            ax.set_xlabel('time [ns]', fontsize=8)
    fig.suptitle('the 18 columns of A — every one a picture of "one unit of '
                 'charge, this deep"', color=CHROME, fontsize=11, y=1.005)
    return save(fig, 'f03_atlas', pad=0.35)


# ------------------------------------------------------- 4. the flattening
def fig_flatten(cs, k=7):
    fig = plt.figure(figsize=(11.4, 4.6))
    gs = GridSpec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1],
                  hspace=0.55, wspace=0.28)

    ext = [-30, wm.NSAMP * wm.SNS - 30, cs.pos[0] - 0.39, cs.pos[-1] + 0.39]
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cs.T[:, :, k], aspect='auto', origin='lower', cmap='magma',
              extent=ext)
    ax.set_title(f'column {k} as the detector sees it  '
                 f'({cs.n_strip} × {wm.NSAMP})', loc='left', fontsize=9.5)
    ax.set_ylabel('strip [mm]')
    ax.set_xlabel('time [ns]')

    ax = fig.add_subplot(gs[1, :])
    v = cs.T[:, :, k].reshape(-1)
    ax.plot(v, color=C['orange'], lw=1.0)
    for i in range(cs.n_strip + 1):
        ax.axvline(i * wm.NSAMP, color=CHROME, lw=0.5, alpha=0.35)
    for i in range(cs.n_strip):
        if v[i * wm.NSAMP:(i + 1) * wm.NSAMP].max() > 0.02 * v.max():
            ax.text(i * wm.NSAMP + wm.NSAMP / 2, v.max() * 1.03,
                    f'{cs.pos[i]:.0f}', ha='center', fontsize=7, color=CHROME)
    ax.set_xlim(0, len(v))
    ax.set_xlabel(f'row index  r = i·{wm.NSAMP} + s     '
                  f'(strip i, sample s)  →  {len(v)} rows')
    ax.set_ylabel('A[r, %d]' % k)
    ax.set_title('…the same column, unrolled into one long vector — '
                 'this is what a column of A literally is', loc='left',
                 fontsize=9.5)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(cs.A, aspect='auto', origin='lower', cmap='magma',
                   interpolation='nearest')
    ax.set_xlabel('depth bin k  (18 columns)')
    ax.set_ylabel('row r  (%d rows)' % cs.A.shape[0])
    ax.set_title('the whole design matrix A, noise-weighted', loc='left',
                 fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02)
    return save(fig, 'f04_flatten')


# ------------------------------------------- 5. the one-dimensional problem
def _sig(A):
    """(condition number, per-column sigma) of a weighted design matrix, via
    the SVD — A'A is the square of an ill-conditioned matrix and inverting it
    directly fails outright at w = 0."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cov = (Vt.T * (1.0 / s ** 2)) @ Vt
    return float(s[0] / s[-1]), np.sqrt(np.diag(cov)), s, cov


def collapse(cs):
    """The same event with the strip index thrown away: sum the model over
    strips and the data over strips. y1(t) = sum_k q_k h1(t - t0 - u_k) — the
    textbook 1-D deconvolution, on this event's real waveforms."""
    T = cs.M.reshape(cs.n_strip, wm.NSAMP, wm.K)
    H = T.sum(0)                                   # (NSAMP, K)
    y = cs.W.sum(0)
    noise = float(np.sqrt((cs.noise ** 2).sum()))  # strips add in quadrature
    return H / noise, y / noise, noise


def fig_oned(cs):
    A1, y1, noise1 = collapse(cs)
    q1, _ = nnls(A1, y1)
    c1d, s1, sv1, cov1 = _sig(A1)
    c2d, s2, sv2, cov2 = _sig(cs.A)
    J = np.ones(wm.K)
    tot1 = float(np.sqrt(J @ cov1 @ J))
    tot2 = float(np.sqrt(J @ cov2 @ J))
    tmpl, _sm = wm._templates(cs.plane, wm.HYPER['sigma_s'])
    hi = np.where(tmpl >= 0.5 * tmpl.max())[0]
    fwhm = float(wm.TGRID[hi[-1]] - wm.TGRID[hi[0]])

    fig = plt.figure(figsize=(11.4, 3.7))
    gs = GridSpec(1, 3, width_ratios=[1.25, 0.95, 1.1], wspace=0.30)

    ax = fig.add_subplot(gs[0])
    ts = wm.TS
    # every column drawn at the same charge, on a common arbitrary scale
    scale = 0.45 * float((y1 * noise1).max()) / float((A1 * noise1).max())
    for k in range(wm.K):
        ax.plot(ts, scale * A1[:, k] * noise1, lw=0.9, color=C['orange'],
                alpha=0.55)
    ax.plot(ts, y1 * noise1, 'o-', ms=3, color=C['blue'],
            label='data, summed over strips')
    ax.plot(ts, (A1 @ q1) * noise1, color=C['green'], lw=1.6,
            label='Σ q̂ₖ hₖ — the 1-D fit')
    ax.plot([], [], color=C['orange'], lw=0.9,
            label='the 18 columns, equal charge')
    ax.set_ylim(top=1.42 * float((y1 * noise1).max()))
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('summed amplitude')
    ax.set_title(f'1-D: eighteen copies of one {fwhm:.0f} ns pulse,\n'
                 'stepped 60 ns apart', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = fig.add_subplot(gs[1])
    ax.semilogy(sv1 / sv1[0], 'o-', ms=4, color=C['red'],
                label=f'1-D  (cond {c1d:,.0f})')
    ax.semilogy(sv2 / sv2[0], 'o-', ms=4, color=C['blue'],
                label=f'strips + slope  (cond {c2d:,.0f})')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('σ / σ₀')
    ax.set_title('what the extra dimension buys', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    kk = np.arange(wm.K)
    ax.bar(kk - 0.19, q1, width=0.38, color=C['red'], alpha=0.85,
           label='1-D solve')
    ax.bar(kk + 0.19, cs.q, width=0.38, color=C['green'], label='full solve')
    ax.errorbar(kk - 0.19, q1, yerr=s1, fmt='none', ecolor=C['red'], lw=0.9,
                capsize=2, alpha=0.75)
    ax.errorbar(kk + 0.19, cs.q, yerr=s2, fmt='none', ecolor=CHROME, lw=1.0,
                capsize=2)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge')
    ax.set_title(f'per-bin error {np.median(s1):,.0f} → {np.median(s2):,.0f};\n'
                 f'total charge {tot1:,.0f} → {tot2:,.0f}', loc='left',
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    N['fwhm_h'] = fwhm
    N['cond_1d'] = c1d
    N['sig_1d_med'] = float(np.median(s1))
    N['sig_2d_med'] = float(np.median(s2))
    N['tot_1d'] = tot1
    N['tot_2d'] = tot2
    N['corr_adj_1d'] = float(np.median(np.diag(
        (A1.T @ A1) / np.outer(np.linalg.norm(A1, axis=0),
                               np.linalg.norm(A1, axis=0)), 1)))
    N['n_1d_row'] = int(wm.NSAMP)
    return save(fig, 'f05_oned')


# ------------------------------------------------ 6. adding the strips back
def fig_buildup(cs, n_w=26):
    """Conditioning as the second dimension is switched on. w is scanned from
    0 (a vertical track: every depth bin lands on the same strips) up to this
    event's fitted value; past that the deep bins walk out of the window and
    the comparison stops being like-for-like."""
    A1, _y1, _nz = collapse(cs)
    c_1d, _s, _sv, _cov = _sig(A1)

    J = np.ones(wm.K)
    ws = np.linspace(0.0, cs.w, n_w)
    conds, sigs, tots = [], [], []
    for w in ws:
        cs.build(w=w)
        c, s, _sv, cov = _sig(cs.A)
        conds.append(c)
        sigs.append(float(np.median(s)))
        tots.append(float(np.sqrt(J @ cov @ J)))
    R = []
    for w in (0.0, cs.w):
        cs.build(w=w)
        G = cs.A.T @ cs.A
        d = np.sqrt(np.diag(G))
        R.append(G / np.outer(d, d))
    cs.build()                                     # restore the fitted matrix

    fig = plt.figure(figsize=(11.4, 3.6))
    gs = GridSpec(1, 3, width_ratios=[0.85, 0.85, 1.3], wspace=0.34)

    for j, (Rm, ttl) in enumerate(zip(
            R, ['w = 0: vertical track', f'w = {cs.w:.4f}: this event'])):
        ax = fig.add_subplot(gs[j])
        im = ax.imshow(Rm, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('depth bin k')
        if j == 0:
            ax.set_ylabel('depth bin k′')
        ax.set_title(ttl, loc='left', fontsize=9)
        if j == 1:
            fig.colorbar(im, ax=ax, pad=0.03)

    ax = fig.add_subplot(gs[2])
    ax.semilogy(ws, conds, 'o-', ms=3.5, color=C['blue'],
                label='cond(A) — left axis')
    ax.axhline(c_1d, color=C['red'], ls='--', lw=1.2,
               label=f'1-D, no strips  ({c_1d:,.0f})')
    ax.set_xlabel('transverse speed w [mm/ns]   (tan θ = w / v)')
    ax.set_ylabel('cond(A)')
    ax.set_title('the strips only help once the track leans', loc='left',
                 fontsize=9.5)
    ax2 = ax.twinx()
    ax2.semilogy(ws, sigs, 's-', ms=3.0, color=C['olive'], alpha=0.9,
                 label='σ of one bin')
    ax2.semilogy(ws, tots, 's-', ms=3.0, color=C['green'], alpha=0.9,
                 label='σ of the total charge')
    ax2.set_ylabel('charge error', color=C['olive'])
    ax2.set_ylim(5, 1e4)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center left')

    N['cond_w0'] = float(conds[0])
    N['cond_wfit'] = float(conds[-1])
    N['sig_w0'] = float(sigs[0])
    N['sig_wfit'] = float(sigs[-1])
    N['tot_w0'] = float(tots[0])
    N['tot_wfit'] = float(tots[-1])
    N['w_fit'] = float(cs.w)
    N['tan_at_w'] = float(cs.w / 36.6e-3)
    return save(fig, 'f06_buildup')


# ------------------------------------------------- 7. the toy worked example
TOY_H = np.array([0.0, 1.0, 0.5, 0.1])          # a 4-sample "template"
TOY_F = np.array([[0.20, 0.05],                  # 3 strips x 2 depth bins
                  [0.60, 0.35],
                  [0.20, 0.60]])
TOY_Q = np.array([100.0, 60.0])


def toy():
    """A 12-row, 2-column version of the same problem, with numbers small
    enough to check by hand. Returns everything the note tabulates."""
    A = np.zeros((3 * 4, 2))
    for k in range(2):
        A[:, k] = np.outer(TOY_F[:, k], TOY_H).reshape(-1)
    rng = np.random.default_rng(7)
    y_true = A @ TOY_Q
    y = y_true + rng.normal(0, 2.0, size=y_true.shape)
    G = A.T @ A
    b = A.T @ y
    q_hat = np.linalg.solve(G, b)
    q_nnls, rn = nnls(A, y)
    return dict(A=A, y=y, y_true=y_true, G=G, b=b, q_hat=q_hat,
                q_nnls=q_nnls, chi2=float(rn ** 2))


def fig_toy():
    t = toy()
    A, y = t['A'], t['y']
    fig = plt.figure(figsize=(11.2, 3.6))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.25, 0.9], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    ax.imshow(A, aspect='auto', origin='lower', cmap='magma',
              interpolation='nearest')
    for r in range(0, 13, 4):
        ax.axhline(r - 0.5, color=CHROME, lw=0.8, alpha=0.6)
    for r in range(12):
        for k in range(2):
            ax.text(k, r, f'{A[r, k]:.2f}', ha='center', va='center',
                    fontsize=7,
                    color='white' if A[r, k] < 0.35 else 'black')
    ax.set_xticks([0, 1])
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('row r = 4·i + s')
    ax.set_title('the toy A: 12 rows × 2 columns', loc='left')

    ax = fig.add_subplot(gs[1])
    r = np.arange(12)
    ax.step(r, y, where='mid', color=C['blue'], label='data y')
    ax.step(r, TOY_Q[0] * A[:, 0], where='mid', color=C['orange'], lw=1.2,
            ls='--', label='100 × column 0')
    ax.step(r, TOY_Q[1] * A[:, 1], where='mid', color=C['purple'], lw=1.2,
            ls='--', label='60 × column 1')
    ax.step(r, A @ t['q_nnls'], where='mid', color=C['green'], lw=1.2,
            label='fitted sum')
    for b_ in (3.5, 7.5):
        ax.axvline(b_, color=CHROME, lw=0.6, alpha=0.5)
    ax.set_xlabel('row r')
    ax.set_ylabel('signal')
    ax.set_title('twelve equations, two unknowns', loc='left')
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    ax.axis('off')
    G, b_, qh = t['G'], t['b'], t['q_hat']
    ax.text(0.0, 1.0,
            'AᵀA  =  [ %6.3f  %6.3f ]\n'
            '        [ %6.3f  %6.3f ]\n\n'
            'Aᵀy  =  [ %7.1f, %7.1f ]\n\n'
            'q̂ = (AᵀA)⁻¹Aᵀy\n'
            '   = [ %6.1f, %6.1f ]\n\n'
            'truth      [ %.0f, %.0f ]\n'
            'NNLS       [ %6.1f, %6.1f ]\n\n'
            'column correlation  %.3f'
            % (G[0, 0], G[0, 1], G[1, 0], G[1, 1], b_[0], b_[1],
               qh[0], qh[1], TOY_Q[0], TOY_Q[1],
               t['q_nnls'][0], t['q_nnls'][1],
               G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])),
            va='top', ha='left', fontsize=9, family='monospace', color=CHROME,
            transform=ax.transAxes)
    ax.set_title('…solved in closed form', loc='left')
    N['toy_q'] = [float(v) for v in t['q_nnls']]
    N['toy_G'] = [[float(v) for v in row] for row in G]
    N['toy_b'] = [float(v) for v in b_]
    N['toy_corr'] = float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]))
    return save(fig, 'f07_toy')


# ------------------------------------------- 8. weighting and censoring
def fig_censor(evs, cal):
    cs = Case(evs, SAT_EVENT, PLANE, cal)
    fig = plt.figure(figsize=(11.4, 3.9))
    gs = GridSpec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.3)
    ext = [-30, wm.NSAMP * wm.SNS - 30, cs.pos[0] - 0.39, cs.pos[-1] + 0.39]

    ax = fig.add_subplot(gs[0])
    ax.imshow(cs.W, aspect='auto', origin='lower', cmap='magma', extent=ext)
    ys, xs = np.where(cs.sat)
    ax.scatter((xs + 0.5) * wm.SNS - 30, cs.pos[ys], s=9, facecolors='none',
               edgecolors=C['green'], lw=0.8,
               label=f'{cs.sat.sum()} clipped samples')
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('strip position [mm]')
    ax.set_title(f'event {SAT_EVENT}: the amplifier clips at '
                 f'{wm.SAT:.0f} ADC', loc='left')
    ax.legend(loc='upper right', fontsize=8)

    ax = fig.add_subplot(gs[1])
    i = int(np.argmax(cs.W.max(axis=1)))
    ts = np.arange(wm.NSAMP) * wm.SNS
    ax.plot(ts, cs.W[i], color=C['blue'], label=f'strip {cs.pos[i]:.1f} mm')
    ax.plot(ts, cs.model()[i], color=C['orange'], label='model')
    ax.axhline(wm.SAT, color=C['green'], ls='--', lw=1.0, label='saturation')
    ax.scatter(ts[cs.sat[i]], cs.W[i][cs.sat[i]], s=18, color=C['green'],
               zorder=5)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('ADC')
    ax.set_title('the model is allowed above the clip, never below',
                 loc='left')
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    keep = (~cs.sat).reshape(-1)
    ax.imshow(keep.reshape(cs.n_strip, wm.NSAMP), aspect='auto',
              origin='lower', cmap='Greys_r', extent=ext, vmin=0, vmax=1)
    ax.set_xlabel('time [ns]')
    ax.set_title(f'the row mask: {keep.sum()} of {len(keep)} rows enter A',
                 loc='left')
    N['sat_n'] = int(cs.sat.sum())
    N['sat_rows'] = int(keep.sum())
    N['sat_tot'] = int(len(keep))
    N['sat_event'] = int(SAT_EVENT)
    return save(fig, 'f08_censor')


# --------------------------------------------- 9. the geometry of the solve
def pick_rows(A2, y, n=3):
    """Three rows that make a legible 3-D picture: rows where the two columns
    are individually large and point in different directions."""
    score = np.abs(A2).sum(1) * (1e-3 + np.abs(A2[:, 0] - A2[:, 1]))
    cand = np.argsort(score)[::-1][:40]
    best, bi = -1, None
    rng = np.random.default_rng(3)
    for _ in range(4000):
        r = rng.choice(cand, n, replace=False)
        v = np.linalg.svd(A2[r], compute_uv=False)
        m = v[-1] / v[0] * np.linalg.norm(y[r])
        if m > best:
            best, bi = m, np.sort(r)
    return bi


def fig_projection(cs, ka=12, kb=13):
    A2 = cs.A[:, [ka, kb]]
    y = cs.y
    q_un = np.linalg.lstsq(A2, y, rcond=None)[0]
    q_nn, _ = nnls(A2, y)

    fig = plt.figure(figsize=(9.4, 3.9))

    # --- panel 2: chi2 landscape in the 2-charge plane ---
    ax = fig.add_subplot(1, 2, 1)
    qa = np.linspace(min(-200, q_un[0] - 400), q_un[0] + 900, 220)
    qb = np.linspace(min(-900, q_un[1] - 400), max(900, q_un[1] + 900), 220)
    QA, QB = np.meshgrid(qa, qb)
    G = A2.T @ A2
    bvec = A2.T @ y
    c0 = float(y @ y)
    CHI = (G[0, 0] * QA ** 2 + 2 * G[0, 1] * QA * QB + G[1, 1] * QB ** 2
           - 2 * (bvec[0] * QA + bvec[1] * QB) + c0)
    chimin = CHI.min()
    lev = chimin + np.array([1, 4, 9, 25, 100, 400, 1600]) * 25
    ax.contour(QA, QB, CHI, levels=lev, colors=CHROME, linewidths=0.7,
               alpha=0.8)
    ax.axhspan(qb[0], 0, color=C['red'], alpha=0.10)
    ax.axhline(0, color=C['red'], lw=1.2)
    ax.axvline(0, color=C['red'], lw=1.2)
    ax.plot(*q_un, 'o', color=C['grey'], ms=7,
            label=f'unconstrained  ({q_un[0]:.0f}, {q_un[1]:.0f})')
    ax.plot(*q_nn, '*', color=C['green'], ms=15,
            label=f'NNLS  ({q_nn[0]:.0f}, {q_nn[1]:.0f})')
    ax.plot([q_un[0], q_nn[0]], [q_un[1], q_nn[1]], color=C['green'], lw=1.0,
            ls=':')
    ax.set_xlabel(f'charge in bin {ka}')
    ax.set_ylabel(f'charge in bin {kb}')
    ax.set_title('χ² in the two-charge plane —\nthe forbidden half is shaded',
                 loc='left', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')

    ax = fig.add_subplot(1, 2, 2)
    tt = np.linspace(0, max(q_nn[0], q_un[0]) * 1.6, 300)
    chi_axis = (G[0, 0] * tt ** 2 - 2 * bvec[0] * tt + c0)
    ax.plot(tt, chi_axis, color=C['green'])
    ax.axvline(q_nn[0], color=C['green'], ls='--', lw=1.0)
    ax.set_xlabel(f'charge in bin {ka}   (bin {kb} held at 0)')
    ax.set_ylabel('χ²')
    ax.set_title('on the boundary the problem is\none-dimensional again',
                 loc='left', fontsize=9)
    N['pair'] = [int(ka), int(kb)]
    N['pair_un'] = [float(v) for v in q_un]
    N['pair_nn'] = [float(v) for v in q_nn]
    N['pair_corr'] = float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]))
    return save(fig, 'f09_projection')


# --------------------------------------------- 10. the Lawson-Hanson walk
def fig_lh(cs):
    x, log = lawson_hanson(cs.A, cs.y)
    xs, _ = nnls(cs.A, cs.y, maxiter=50 * wm.K)
    N['lh_agree'] = float(np.abs(x - xs).max())
    N['lh_iters'] = len(log)
    N['chi2_start'] = float((cs.y ** 2).sum())
    N['chi2_end'] = float(log[-1]['chi2'])
    N['lh_order'] = [int(L['entered']) for L in log]

    fig = plt.figure(figsize=(11.4, 4.0))
    gs = GridSpec(1, 3, width_ratios=[0.95, 1.15, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    g0 = log[0]['grad']
    ax.bar(np.arange(wm.K), g0, color=[C['green'] if k == log[0]['entered']
                                       else C['grey'] for k in range(wm.K)])
    ax.axhline(0, color=CHROME, lw=0.8)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('Aᵀ(y − Aq)   [gradient]')
    ax.set_title('step 1: which bin wants charge most?', loc='left',
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    ch = [float((cs.y ** 2).sum())] + [L['chi2'] for L in log]
    ax.plot(range(len(ch)), ch, 'o-', color=C['blue'], ms=4)
    for L in log:
        ax.annotate(f"+{L['entered']}", (L['it'], L['chi2']),
                    textcoords='offset points', xytext=(4, 6), fontsize=7.5,
                    color=CHROME)
    ax.set_yscale('log')
    ax.set_xlabel('iteration  (one bin admitted per step)')
    ax.set_ylabel('χ²')
    ax.set_title(f"χ²: {ch[0]:,.0f} → {ch[-1]:,.0f} in {len(log)} steps",
                 loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    Q = np.array([L['x'] for L in log])
    im = ax.imshow(Q, aspect='auto', origin='lower', cmap='magma',
                   interpolation='nearest',
                   extent=[-0.5, wm.K - 0.5, 0.5, len(log) + 0.5])
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('iteration')
    ax.set_title('the profile being assembled', loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02, label='charge')
    return save(fig, 'f10_lh')


# ------------------------------------------ 11. unconstrained vs constrained
def fig_uncon(cs):
    qu = np.linalg.lstsq(cs.A, cs.y, rcond=None)[0]
    chi_u = float(((cs.A @ qu - cs.y) ** 2).sum())
    fig = plt.figure(figsize=(11.4, 3.9))
    gs = GridSpec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    kk = np.arange(wm.K)
    ax.bar(kk - 0.2, qu, width=0.4, color=C['grey'], label='unconstrained')
    ax.bar(kk + 0.2, cs.q, width=0.4, color=C['green'], label='NNLS  (q ≥ 0)')
    ax.axhline(0, color=CHROME, lw=0.8)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge')
    ax.set_title(f'{int((qu < 0).sum())} of 18 bins go negative if you let them',
                 loc='left', fontsize=9.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1])
    i = int(np.argmax(cs.W.max(axis=1)))
    ts = np.arange(wm.NSAMP) * wm.SNS
    ax.plot(ts, cs.W[i], color=C['blue'], lw=2.0, label='data')
    ax.plot(ts, (cs.M @ qu).reshape(cs.n_strip, wm.NSAMP)[i], color=C['grey'],
            lw=1.3, label=f'unconstrained  χ²={chi_u:,.0f}')
    ax.plot(ts, cs.model()[i], color=C['green'], lw=1.3, ls='--',
            label=f'NNLS  χ²={cs.chi2:,.0f}')
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('ADC')
    ax.set_title(f'brightest strip: both describe it', loc='left',
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    z = np.array(wm.UK) * 36.6e-3
    ax.plot(z, np.cumsum(qu), 'o-', color=C['grey'], ms=3,
            label='unconstrained')
    ax.plot(z, np.cumsum(cs.q), 'o-', color=C['green'], ms=3, label='NNLS')
    ax.set_xlabel('drift depth z [mm]')
    ax.set_ylabel('cumulative charge')
    ax.set_title('the same total, a different story about where it came from',
                 loc='left', fontsize=9.5)
    ax.legend(fontsize=8)
    N['n_neg'] = int((qu < 0).sum())
    N['n_zero'] = int((cs.q == 0).sum())
    N['chi2_uncon'] = chi_u
    N['chi2_nnls'] = float(cs.chi2)
    N['chi2_gain_pct'] = float(100 * (cs.chi2 - chi_u) / chi_u)
    return save(fig, 'f11_uncon')


# ----------------------------------------------- 12. how independent are they
def fig_gram(cs):
    A = cs.A
    G = A.T @ A
    d = np.sqrt(np.diag(G))
    R = G / np.outer(d, d)
    sv = np.linalg.svd(A, compute_uv=False)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    fig = plt.figure(figsize=(11.4, 3.9))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.1], wspace=0.34)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(R, origin='lower', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('depth bin k′')
    ax.set_title('column correlation', loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = fig.add_subplot(gs[1])
    ax.semilogy(sv / sv[0], 'o-', color=C['blue'], ms=4)
    ax.set_xlabel('singular value index')
    ax.set_ylabel('σ / σ₀')
    ax.set_title(f'condition number {sv[0] / sv[-1]:.0f}', loc='left',
                 fontsize=9.5)

    ax = fig.add_subplot(gs[2])
    for j, col in ((0, C['blue']), (1, C['teal']), (wm.K - 2, C['orange']),
                   (wm.K - 1, C['red'])):
        ax.plot(np.arange(wm.K), Vt[j], 'o-', ms=3, color=col,
                label=f'mode {j}  (σ/σ₀ = {sv[j] / sv[0]:.3f})')
    ax.axhline(0, color=CHROME, lw=0.8)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge pattern')
    ax.set_title('the best- and worst-measured charge patterns', loc='left',
                 fontsize=9.5)
    ax.legend(fontsize=7)
    N['cond'] = float(sv[0] / sv[-1])
    N['corr_adj'] = float(np.median(np.diag(R, 1)))
    N['corr_2'] = float(np.median(np.diag(R, 2)))
    return save(fig, 'f12_gram')


# ------------------------------------------------------ 13. what came out
def fig_result(cs):
    mod = cs.model()
    res = (cs.W - mod) / cs.noise[:, None]
    fig = plt.figure(figsize=(11.6, 6.4))
    gs = GridSpec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.3)
    ext = [-30, wm.NSAMP * wm.SNS - 30, cs.pos[0] - 0.39, cs.pos[-1] + 0.39]

    inner = gs[0, :].subgridspec(3, 6, hspace=0.55, wspace=0.35)
    ts = np.arange(wm.NSAMP) * wm.SNS
    amp = cs.W.max(axis=1)
    show = np.argsort(amp)[::-1][:18]
    show = np.sort(show)
    for j, i in enumerate(show[:18]):
        ax = fig.add_subplot(inner[j // 6, j % 6])
        ax.plot(ts, cs.W[i], color=C['blue'], lw=1.1)
        ax.plot(ts, mod[i], color=C['orange'], lw=1.1)
        ax.set_title(f'{cs.pos[i]:.1f} mm', fontsize=7.5, loc='left')
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.12)
        if j % 6:
            ax.set_yticklabels([])
        if j < 12:
            ax.set_xticklabels([])
        else:
            ax.set_xticks([0, 900, 1800])
    fig.text(0.5, 0.985, 'data (blue) and the fitted model (orange), '
             'strip by strip — one 18-number charge profile makes all of them',
             ha='center', color=CHROME, fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    m = np.abs(res).max()
    im = ax.imshow(res, aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=ext, vmin=-m, vmax=m)
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('strip [mm]')
    ax.set_title(f'residual / σ    χ²/dof = {cs.chi2 / cs.dof:.1f}',
                 loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = fig.add_subplot(gs[1, 1])
    z = np.array(wm.UK) * 36.6e-3
    ax.bar(z, cs.q, width=1.9, color=C['green'])
    ax.set_xlabel('drift depth z = v·u  [mm]')
    ax.set_ylabel('charge [ADC·ns equivalent]')
    ax.set_title('the answer: q — measured, not assumed', loc='left',
                 fontsize=9.5)
    ax.axvline(30.0, color=C['red'], lw=1.0, ls='--')
    ax.text(30.2, cs.q.max() * 0.9, ' cathode', color=C['red'], fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    p = cs.p0 + cs.w * np.array(wm.UK)
    ax.scatter(p, z, s=6 + 90 * cs.q / max(cs.q.max(), 1), color=C['green'],
               alpha=0.85, zorder=3)
    ax.plot(cs.p0 + cs.w * np.array([0, wm.UK[-1]]), [0, z[-1]],
            color=C['orange'], lw=1.2, label='fitted track')
    ax.plot(cs.p0_ref + cs.tan_ref * 36.6e-3 * np.array([0, wm.UK[-1]]),
            [0, z[-1]], color=C['ref'], lw=1.2, ls='--', label='M3 reference')
    ax.set_xlabel('transverse position [mm]')
    ax.set_ylabel('drift depth [mm]')
    ax.set_title('…and the track it implies', loc='left', fontsize=9.5)
    ax.legend(fontsize=8)
    N['chi2_dof'] = float(cs.chi2 / cs.dof)
    N['q_total'] = float(cs.q.sum())
    N['tan_fit'] = float(cs.w / (36.6e-3))
    N['tan_ref'] = float(cs.tan_ref)
    return save(fig, 'f13_result')


# ---------------------------------------------------- 14. errors on the q
def fig_errors(cs):
    free = cs.q > 0
    Af = cs.A[:, free]
    cov = np.linalg.inv(Af.T @ Af)
    sig = np.zeros(wm.K)
    sig[free] = np.sqrt(np.diag(cov))
    Rf = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    J = np.ones(free.sum())
    tot_err = float(np.sqrt(J @ cov @ J))

    fig = plt.figure(figsize=(11.2, 3.7))
    gs = GridSpec(1, 3, width_ratios=[1.2, 0.95, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    kk = np.arange(wm.K)
    ax.bar(kk, cs.q, width=0.7, color=C['green'])
    ax.errorbar(kk[free], cs.q[free], yerr=sig[free], fmt='none',
                ecolor=CHROME, lw=1.2, capsize=2.5)
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge')
    ax.set_title(f'total {cs.q.sum():,.0f} ± {tot_err:,.0f}  '
                 f'({100 * tot_err / cs.q.sum():.2f} %)', loc='left',
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    idx = np.where(free)[0]
    im = ax.imshow(Rf, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, fontsize=6.5)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx, fontsize=6.5)
    ax.set_title('correlation of the surviving bins', loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = fig.add_subplot(gs[2])
    ax.bar(kk[free], 100 * sig[free] / np.maximum(cs.q[free], 1), width=0.7,
           color=C['olive'])
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('σ(q) / q  [%]')
    ax.set_title('a single bin is known to a few per cent at best',
                 loc='left', fontsize=9.5)
    N['q_tot_err'] = tot_err
    N['q_tot_err_pct'] = float(100 * tot_err / cs.q.sum())
    N['sig_q_med'] = float(np.median(sig[free]))
    return save(fig, 'f14_errors')


# ------------------------------------------------------- 15. the profiling
def fig_profile(cs):
    dps = np.linspace(-2.0, 2.0, 81)
    chis, qs = [], []
    for dp in dps:
        c, q = wm.chi2_plane(cs.plane, cs.W, cs.noise, cs.pos, cs.sat,
                             cs.p0 + dp, cs.w, cs.t0, wm.HYPER, snap_t0=False)
        chis.append(c)
        qs.append(q if q is not None else np.zeros(wm.K))
    chis = np.array(chis)
    qs = np.array(qs)

    fig = plt.figure(figsize=(11.4, 3.9))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    ax.plot(cs.p0 + dps, chis, color=C['blue'])
    ax.axvline(cs.p0, color=C['green'], ls='--', lw=1.0)
    ax.set_xlabel('trial p₀ [mm]')
    ax.set_ylabel('χ²  (charges re-solved at every point)')
    ax.set_title('the profile likelihood in p₀', loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(qs.T, aspect='auto', origin='lower', cmap='magma',
                   extent=[cs.p0 + dps[0], cs.p0 + dps[-1], -0.5, wm.K - 0.5])
    ax.axvline(cs.p0, color=C['green'], ls='--', lw=1.0)
    ax.set_xlabel('trial p₀ [mm]')
    ax.set_ylabel('depth bin k')
    ax.set_title('the charge profile is re-derived for every trial',
                 loc='left', fontsize=9.5)
    fig.colorbar(im, ax=ax, pad=0.02)

    ax = fig.add_subplot(gs[2])
    for dp, col in ((-1.2, C['grey']), (0.0, C['green']), (+1.2, C['olive'])):
        j = int(np.argmin(np.abs(dps - dp)))
        ax.step(np.arange(wm.K), qs[j], where='mid', color=col,
                label=f'p₀ {dp:+.1f} mm   χ² {chis[j]:,.0f}')
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge')
    ax.set_title('three of those profiles', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5)
    N['profile_nsolve'] = int(len(dps))
    return save(fig, 'f15_profile')


# ------------------------------------------------------- 16. the 60 ns tooth
def fig_tooth(cs):
    out = []
    for dt in (-60.0, 0.0):
        best = (np.inf, None, None)
        for dp in np.arange(-1.5, 1.51, 0.05):
            c, q = wm.chi2_plane(cs.plane, cs.W, cs.noise, cs.pos, cs.sat,
                                 cs.p0 + dp, cs.w, cs.t0 + dt, wm.HYPER)
            if c < best[0]:
                best = (c, dp, q)
        out.append((dt, *best))

    fig = plt.figure(figsize=(11.2, 3.8))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    for (dt, c, dp, q), col in zip(out, (C['purple'], C['green'])):
        ax.step(np.arange(wm.K), q, where='mid', color=col,
                label=f't₀ {dt:+.0f} ns, p₀ {dp:+.2f} mm   χ² {c:,.0f}')
    ax.set_xlabel('depth bin k')
    ax.set_ylabel('charge')
    ax.set_title('one bin over, and almost as good', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[1])
    i = int(np.argmax(cs.W.max(axis=1)))
    ts = np.arange(wm.NSAMP) * wm.SNS
    ax.plot(ts, cs.W[i], color=C['blue'], lw=2.0, label='data')
    for (dt, c, dp, q), col in zip(out, (C['purple'], C['green'])):
        M = wm.build_matrix(cs.plane, cs.pos, cs.p0 + dp, cs.w, cs.t0 + dt,
                            wm.HYPER)
        ax.plot(ts, (M @ q).reshape(cs.n_strip, wm.NSAMP)[i], color=col,
                lw=1.2, ls='--', label=f't₀ {dt:+.0f} ns')
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('ADC')
    ax.set_title('the two models on the brightest strip', loc='left',
                 fontsize=9.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2])
    grid = np.arange(-185, 61, 5.0)
    cc = []
    for dt in grid:
        b = np.inf
        for dp in np.arange(-2.0, 2.01, 0.05):
            c, _ = wm.chi2_plane(cs.plane, cs.W, cs.noise, cs.pos, cs.sat,
                                 cs.p0 + dp, cs.w, cs.t0 + dt, wm.HYPER)
            b = min(b, c)
        cc.append(b)
    cc = np.array(cc)
    loc_min = [j for j in range(1, len(cc) - 1)
               if cc[j] < cc[j - 1] and cc[j] < cc[j + 1]]
    ax.plot(cs.t0 + grid, cc, color=C['blue'])
    ax.plot((cs.t0 + grid)[loc_min], cc[loc_min], 'v', color=C['red'], ms=6)
    ax.axvline(cs.t0, color=C['green'], ls='--', lw=1.0)
    ax.axvline(cs.t0 - 60, color=C['purple'], ls='--', lw=1.0)
    ax.set_xlabel('trial t₀ [ns]   (p₀ re-optimised at each point)')
    ax.set_ylabel('χ²')
    ax.set_title('secondary minima, one depth bin apart', loc='left',
                 fontsize=9.5)
    N['tooth_minima'] = [float(v) for v in (cs.t0 + grid)[loc_min]]
    N['tooth_minima_chi2'] = [float(v) for v in cc[loc_min]]
    N['tooth_chi2'] = [float(o[1]) for o in out]
    N['tooth_dp'] = [float(o[2]) for o in out]
    N['tooth_dchi2_pct'] = float(100 * (out[0][1] - out[1][1]) / out[1][1])
    return save(fig, 'f16_tooth')


# ------------------------------------------------------- 17. the population
def fig_population(evs, cal, n_scan=250):
    nz, its, bts, prof, chid = [], [], [], [], []
    for eid in sorted(evs)[:n_scan]:
        ev = evs[eid]
        if 'x' not in ev:
            continue
        P = K.trim_window(ev['x'])
        Wsh = np.asarray(P['W']).shape
        if Wsh[1] != wm.NSAMP:
            continue
        W, noise, pos, sat = wm.prep_plane(P, 'x')
        p0 = ev['ref_mesh_x']
        w = ev['tan_x'] * cal.v_drift * 1e-3
        try:
            r = wm.fit_plane_raw(P, 'x', p0, w, 400.0, fix_p0w=(p0, w))
        except Exception:
            continue
        M = wm.build_matrix('x', pos, p0, w, r['t0'], wm.HYPER)
        ok = ~sat.reshape(-1)
        A = (M * np.repeat(1 / noise, wm.NSAMP)[:, None])[ok]
        y = (W / noise[:, None]).reshape(-1)[ok]
        x, log = lawson_hanson(A, y)
        nz.append(int((x > 0).sum()))
        its.append(len(log))
        bts.append(sum(L['backtracks'] for L in log))
        if x.sum() > 0:
            prof.append(x / x.sum())
        chid.append(r['chi2'] / max(r['dof'], 1))
    nz = np.array(nz)
    bts = np.array(bts)
    prof = np.array(prof)

    fig = plt.figure(figsize=(11.4, 3.8))
    gs = GridSpec(1, 3, wspace=0.3)

    ax = fig.add_subplot(gs[0])
    ax.hist(nz, bins=np.arange(0.5, wm.K + 1.5), color=C['green'], alpha=0.85)
    ax.set_xlabel('non-zero depth bins')
    ax.set_ylabel('planes')
    ax.set_title(f'{nz.mean():.1f} of {wm.K} bins survive, on average',
                 loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[1])
    ax.hist(np.array(its), bins=np.arange(0.5, 20.5), color=C['blue'],
            alpha=0.85, label='bins admitted')
    ax.hist(bts, bins=np.arange(-0.5, 20.5), color=C['red'], alpha=0.6,
            label='bins pushed back out')
    ax.set_xlabel('count per plane')
    ax.set_ylabel('planes')
    ax.set_title(f'{100 * np.mean(bts > 0):.0f} % need at least one backtrack',
                 loc='left', fontsize=9.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2])
    z = np.array(wm.UK) * cal.v_drift * 1e-3
    ax.plot(z, prof.mean(0), 'o-', color=C['green'], ms=4, label='mean')
    ax.plot(z, np.median(prof, 0), 'o--', color=C['red'], ms=4,
            label='median  ← truncated')
    ax.axvline(30.0, color=CHROME, ls=':', lw=1.0)
    ax.set_xlabel('drift depth z [mm]')
    ax.set_ylabel('charge fraction per bin')
    ax.set_title(f'{len(prof)} planes averaged: flat in the middle, '
                 f'spike at bin 0, edge at the cathode', loc='left',
                 fontsize=9)
    ax.legend(fontsize=8)
    N['pop_n'] = int(len(nz))
    N['pop_nz'] = float(nz.mean())
    N['pop_nz_frac'] = float(nz.mean() / wm.K)
    N['pop_bt_frac'] = float(np.mean(bts > 0))
    N['pop_bt_max'] = int(bts.max())
    N['pop_chi2dof_med'] = float(np.median(chid))
    return save(fig, 'f17_population')


# ------------------------------------------------- 18. how much is sharing
def fig_sharing(evs, cal):
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.6))
    ev = evs[EVENT]
    stats = {}
    for pl, col in (('x', C['x']), ('y', C['y'])):
        P = K.trim_window(ev[pl])
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            continue
        W, noise, pos, sat = wm.prep_plane(P, pl)
        r = wm.fit_plane_raw(P, pl, ev[f'ref_mesh_{pl}'],
                             ev[f'tan_{pl}'] * cal.v_drift * 1e-3, 400.0)
        h = dict(wm.HYPER)
        M = wm.build_matrix(pl, pos, r['p0'], r['w'], r['t0'], h)
        h0 = dict(h, c1=0.0, c2=0.0)
        M0 = wm.build_matrix(pl, pos, r['p0'], r['w'], r['t0'], h0)
        frac = 1 - np.linalg.norm(M0) / np.linalg.norm(M)
        kk = h.get('kY', 1.0) if pl == 'y' else h.get('cX', 1.0)
        # effective_c2, not h['c2']: on a slaved bundle the stored c2 is 0.0
        # and the ratio carries it -- the trap this note warns about, which
        # this figure walked into on 2026-08-21 before being fixed.
        stats[pl] = dict(frac=float(frac), c1=float(h['c1'] * kk),
                         c2=float(effective_c2(h) * kk))
        k = 7
        prof_full = M.reshape(len(pos), wm.NSAMP, wm.K)[:, :, k].sum(1)
        prof_own = M0.reshape(len(pos), wm.NSAMP, wm.K)[:, :, k].sum(1)
        ax = axs[0] if pl == 'x' else axs[1]
        pc = r['p0'] + r['w'] * wm.UK[k]          # where bin k's charge sits
        ax.bar(pos - pc, prof_own, width=0.6, color=C['grey'],
               label='own strip')
        ax.bar(pos - pc, prof_full - prof_own, width=0.6,
               bottom=prof_own, color=col, label='neighbours’ copies')
        ax.set_xlabel('strip − track position of bin k  [mm]')
        ax.set_ylabel('column-k charge landing on the strip')
        ax.set_title(f'{pl} plane: sharing is {100 * frac:.1f} % of |A|',
                     loc='left', fontsize=9.5)
        ax.legend(fontsize=8)
        ax.set_xlim(-3.2, 3.2)

    ax = axs[2]
    ts = np.arange(wm.NSAMP) * wm.SNS
    for pl, col in (('x', C['x']), ('y', C['y'])):
        h = dict(wm.HYPER)
        base = ts[:, None] - (300.0 + wm.UK[None, :])
        tmpl, _ = wm._templates(pl, h['sigma_s'])
        H0 = np.interp(base, wm.TGRID, tmpl, left=0, right=0)
        H1, _H2 = wm._copy_responses(pl, base, h)
        ax.plot(ts, H0[:, 7] / H0[:, 7].max(), color=col, lw=1.4,
                label=f'{pl}: own')
        ax.plot(ts, H1[:, 7] / H1[:, 7].max(), color=col, lw=1.2, ls='--',
                label=f'{pl}: ±1 copy')
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('peak-normalised')
    ax.set_title('the copy is late as well as small', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5)
    N['share'] = stats
    return save(fig, 'f18_sharing')


def main():
    cal = K.install()
    print('[cs]', cal.summary())
    evs = K.calib_events()
    cs = Case(evs, EVENT, PLANE, cal)
    print(f'[cs] event {EVENT} {PLANE}: chi2/dof '
          f'{cs.chi2 / cs.dof:.1f}  p0 {cs.p0:.2f}  w {cs.w:.5f}  t0 {cs.t0:.0f}')
    fig_window(cs)
    fig_column_build(cs)
    fig_atlas(cs)
    fig_flatten(cs)
    fig_oned(cs)
    fig_buildup(cs)
    fig_toy()
    fig_censor(evs, cal)
    fig_projection(cs)
    fig_lh(cs)
    fig_uncon(cs)
    fig_gram(cs)
    fig_result(cs)
    fig_errors(cs)
    fig_profile(cs)
    fig_tooth(cs)
    fig_sharing(evs, cal)
    fig_population(evs, cal)
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, 'numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, sort_keys=True)
    print('[cs] numbers ->', os.path.join(FIGDIR, 'numbers.json'))


if __name__ == '__main__':
    main()
