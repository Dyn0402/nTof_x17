#!/usr/bin/env python3
"""deflect.py -- remove the post-flash ringing from a DREAM plane before the fit (study, 2026-11-22; README.md).

The ringing after the n_TOF flash is the block common mode with a channel-dependent coupling.  Every 64-channel DREAM
block carries one oscillation (period ~7-8 samples, ~450 ns), and each channel picks it up with its own factor beta.
The plain block median removes beta = 1 everywhere.  A channel with beta = 0 is then left with minus the oscillation,
and a channel with beta = 2 with plus it: the "both signs, all at the same sample" pattern of the audit.

The correction, per event and per block:

  1. c(t) = the block median of the pedestal-subtracted samples (the oscillation, as before).
  2. beta_ch = <x_ch, c> / <c, c>, a least-squares fit of each channel's waveform to c(t), no intercept.
  3. A channel that still has a positive excursion above ZSIG sigma after step 2 may carry ionisation, which would
     pull its beta.  Its beta (and that of its two neighbours) is replaced by the median beta of the unflagged channels
     within +-NB channels of the same block (beta is smooth along the channels).
  4. W_ch(t) = x_ch(t) - beta_ch c(t).

A real pulse is preserved: on late tracks the summed positive charge of a signal channel is unchanged to 1.000
(median; 10th percentile 0.98).  ``gated`` mode applies the correction only to a plane that shows ringing (at least
GATE_NEG channels below -5 sigma after the plain median).  Late planes then stay bit-identical to pass 1.
"""
from __future__ import annotations

import numpy as np

CNS = 64
NB = 4
ZSIG = 5.0
GATE_NEG = 3


def deflect(wfm: np.ndarray, noise: np.ndarray, nb: int = NB, zsig: float = ZSIG) -> np.ndarray:
    """(n_sample, 512) pedestal-subtracted window, BEFORE any common mode -> (n_sample, 512) with the block common
    mode removed with a per-channel coupling (module docstring)."""
    ns = wfm.shape[0]
    nblk = wfm.shape[1] // CNS
    B = wfm.reshape(ns, nblk, CNS)
    c = np.median(B, axis=2)                                    # (ns, nblk)
    cc = np.einsum('sb,sb->b', c, c)
    with np.errstate(invalid='ignore', divide='ignore'):
        beta = np.einsum('sbk,sb->bk', B, c) / cc[:, None]
    beta = np.where(cc[:, None] > 1e-6, beta, 1.0)
    r = B - beta[None] * c[..., None]
    nzb = np.maximum(np.asarray(noise, np.float64), 1.0).reshape(nblk, CNS)
    flag = (r / nzb[None]).max(axis=0) > zsig                   # (nblk, 64)
    fl = flag.copy()
    fl[:, 1:] |= flag[:, :-1]
    fl[:, :-1] |= flag[:, 1:]
    b2 = beta.copy()
    for b in range(nblk):
        idx = np.flatnonzero(fl[b])
        if not len(idx):
            continue
        good = np.flatnonzero(~fl[b])
        if len(good) < 8:
            b2[b, idx] = 1.0
            continue
        for k in idx:
            nbrs = good[np.abs(good - k) <= nb]
            if len(nbrs) < 2:
                nbrs = good[np.argsort(np.abs(good - k))[:4]]
            b2[b, k] = np.median(beta[b, nbrs])
    return (B - b2[None] * c[..., None]).reshape(ns, -1).astype(wfm.dtype, copy=False)


def plain_cm(wfm: np.ndarray) -> np.ndarray:
    ns = wfm.shape[0]
    B = wfm.reshape(ns, -1, CNS)
    return (B - np.median(B, axis=2)[..., None]).reshape(ns, -1)


def ringing(wfm_plain: np.ndarray, noise: np.ndarray) -> int:
    """Channels below -5 sigma in a plain-CM window (n_sample, 512): the ringing gate."""
    return int(((wfm_plain / np.maximum(noise, 1.0)).min(axis=0) < -5.0).sum())


def install(mode: str = 'gated', stats: dict | None = None):
    """Replace wft.io.FeuReader.iter_events so every reader (fit windows, waveform seeding) sees the corrected
    waveforms.  mode 'gated' corrects only planes with >= GATE_NEG ringing channels; 'all' corrects every event.
    Requires CM_MASK = 0 (the pass-1 setting)."""
    from wft import io as wio
    if wio.CM_MASK > 0:
        raise RuntimeError('deflect assumes the plain block median (CM_MASK = 0), as pass 1')
    stats = {} if stats is None else stats

    def iter_events(self, wanted=None, with_raw=False):
        idx = (np.where(np.isin(self.event_ids, np.fromiter(wanted, dtype=np.int64)))[0]
               if wanted is not None else np.arange(self.n_entries))
        for lo in range(0, len(idx), wio.CHUNK):
            block = idx[lo:lo + wio.CHUNK]
            arr = self.tree.arrays(['eventId', 'amplitude', 'ftst'], entry_start=int(block[0]),
                                   entry_stop=int(block[-1]) + 1, library='np')
            base = int(block[0])
            for i in block:
                j = i - base
                a = arr['amplitude'][j]
                if len(a) != self.n_sample * 512:
                    self.incomplete[int(arr['eventId'][j])] = len(a) // 512
                    continue
                raw = a.reshape(-1, 512).astype(np.float32) - self.ped
                w = plain_cm(raw)
                n_neg = ringing(w, self.noise)
                if mode == 'all' or n_neg >= GATE_NEG:
                    w = deflect(raw, self.noise)
                    stats['corrected'] = stats.get('corrected', 0) + 1
                stats['seen'] = stats.get('seen', 0) + 1
                if with_raw:
                    yield (int(arr['eventId'][j]), int(arr['ftst'][j]), w.T,
                           np.asarray(a, dtype=np.int16).reshape(-1, 512).T)
                else:
                    yield int(arr['eventId'][j]), int(arr['ftst'][j]), w.T
    wio.FeuReader.iter_events = iter_events
    return stats


# --------------------------------------------------------------------------------------------------------------------
# deflect2 (open items, 2026-11-23): the same idea with the pulse protected by MASKED SAMPLES instead of borrowed betas,
# and optional extra reference waveforms.
#
#   1. c_b(t) = block median (as before); R0 = x - c_b, the pass-1 plain-CM residual.
#   2. Pulse mask: samples with R0 > PMASK sigma, widened by +-PWID samples, are left out of every fit of that channel.
#      A channel with fewer than MINFREE free samples keeps the plain median (coefficient 1, nothing else).
#   3. Basis per block: c_b(t); with rank 2 also u_b(t), the leading singular vector (over time) of R0 on the block's
#      pulse-free channels -- the part of the waveforms the median does not carry (C's local excursions, D's 2.8 MHz
#      channel group).  With dc also a constant (an offset is safe once the pulse samples are out of the fit).
#   4. Per channel, least squares on its free samples: x = a c_b + g u_b + d; subtracted at every sample.
# --------------------------------------------------------------------------------------------------------------------
PMASK = 3.0
PWID = 2
MINFREE = 8


def _pulse_mask(R0, nz, pmask=PMASK, pwid=PWID):
    """(ns, nc) True where a sample may carry a pulse: R0 > pmask sigma, widened by pwid samples in time."""
    m = R0 > pmask * nz[None, :]
    out = m.copy()
    for k in range(1, pwid + 1):
        out[k:] |= m[:-k]
        out[:-k] |= m[k:]
    return out


def deflect2(wfm: np.ndarray, noise: np.ndarray, rank: int = 1, dc: bool = False, pmask: float = PMASK,
             pwid: int = PWID, minfree: int = MINFREE, return_info: bool = False):
    """(n_sample, 512) pedestal-subtracted window, before any common mode -> corrected (n_sample, 512) (see above)."""
    ns, nc = wfm.shape
    nblk = nc // CNS
    x = wfm.astype(np.float64)
    nz = np.maximum(np.asarray(noise, np.float64), 1.0)
    B = x.reshape(ns, nblk, CNS)
    c = np.median(B, axis=2)                                         # (ns, nblk)
    R0 = (B - c[..., None]).reshape(ns, nc)
    pm = _pulse_mask(R0, nz, pmask, pwid)
    free = ~pm                                                        # (ns, nc)
    nfree = free.sum(axis=0)
    out = np.empty_like(x)
    info = dict(n_masked_ch=int((nfree < ns).sum()), n_fallback=int((nfree < minfree).sum()))
    for b in range(nblk):
        sl = slice(b * CNS, (b + 1) * CNS)
        cols = [c[:, b]]
        if rank >= 2:
            clean = nfree[sl] == ns
            if clean.sum() >= 8:
                M = R0[:, sl][:, clean]
                u, s, _ = np.linalg.svd(M - M.mean(0, keepdims=True) * dc, full_matrices=False)
                cols.append(u[:, 0] * s[0] / np.sqrt(clean.sum()))
        if dc:
            cols.append(np.ones(ns))
        X = np.stack(cols, 1)                                         # (ns, k)
        Y = x[:, sl]
        F = free[:, sl].astype(np.float64)                            # (ns, 64)
        # per-channel normal equations on the free samples
        XtX = np.einsum('tk,tl,tc->ckl', X, X, F) + 1e-9 * np.eye(X.shape[1])[None]
        XtY = np.einsum('tk,tc->ck', X, Y * F)
        coef = np.linalg.solve(XtX, XtY[..., None])[..., 0]            # (64, k)
        fb = nfree[sl] < minfree
        if fb.any():
            coef[fb] = 0.0
            coef[fb, 0] = 1.0
        out[:, sl] = Y - X @ coef.T
    out = out.astype(wfm.dtype, copy=False)
    return (out, info) if return_info else out


# --------------------------------------------------------------------------------------------------------------------
# deflect3 (open items, 2026-11-23): deflect (v1) for the block ringing, then a SECOND, LOCAL component only where a
# block really carries one.
#
#   1. W1 = deflect(x)  (the block common mode with a per-channel coupling, as validated).
#   2. Per block, the channels that look like ionisation ("pulse-like": a peak above 5 sigma and never below -3 sigma)
#      are set aside; the others, oscillating channels included, give the leading time component u_b(t) of W1 (SVD).
#   3. The component is used only if it is coherent and well above the noise: its rms per channel s0 / sqrt(n ns)
#      exceeds SV_NSIG times the block's median noise.  Otherwise the block keeps W1 (late events are unchanged).
#   4. Each channel is fitted to u_b on its free samples (pulse-like channels lose their samples above 3 sigma, +-2
#      samples) and the fit is subtracted.  An offset is never fitted.
# --------------------------------------------------------------------------------------------------------------------
SV_NSIG = 2.0


def deflect3(wfm: np.ndarray, noise: np.ndarray, sv_nsig: float = SV_NSIG, pmask: float = PMASK, pwid: int = PWID,
             minfree: int = MINFREE, return_info: bool = False, w1: np.ndarray | None = None):
    """``w1``: deflect(wfm) when the caller already has it."""
    ns, nc = wfm.shape
    nblk = nc // CNS
    nz = np.maximum(np.asarray(noise, np.float64), 1.0)
    W1 = (deflect(wfm.astype(np.float32), noise) if w1 is None else w1).astype(np.float64)
    z = W1 / nz[None, :]
    pulse_like = (z.max(0) > 5.0) & (z.min(0) > -3.0)
    pm = _pulse_mask(W1, nz, pmask, pwid) & pulse_like[None, :]
    free = ~pm
    out = W1.copy()
    used = 0
    for b in range(nblk):
        sl = slice(b * CNS, (b + 1) * CNS)
        sel = ~pulse_like[sl]
        if sel.sum() < 8:
            continue
        M = W1[:, sl][:, sel]
        u, s, _ = np.linalg.svd(M, full_matrices=False)
        if s[0] / np.sqrt(sel.sum() * ns) < sv_nsig * np.median(nz[sl]):
            continue
        used += 1
        u0 = u[:, 0]
        F = free[:, sl].astype(np.float64)
        Y = W1[:, sl]
        g = (u0[:, None] * Y * F).sum(0) / np.maximum((u0[:, None] ** 2 * F).sum(0), 1e-9)
        g[F.sum(0) < minfree] = 0.0
        out[:, sl] = Y - np.outer(u0, g)
    out = out.astype(wfm.dtype, copy=False)
    return (out, dict(blocks_second=used)) if return_info else out


# --------------------------------------------------------------------------------------------------------------------
# deflect4 (open items, 2026-11-23): iterative sample masking, for both components.
#
# deflect (v1) and deflect3 protect a pulse by flagging its CHANNEL after one fit.  A pulse the first fit absorbs
# (its beta or its g takes the pulse) leaves no 5-sigma residual, is not flagged, and loses charge: an injected late
# pulse keeps only 0.5-0.9 of its charge in 10 % of early events.  Here every fit is repeated NITER times; after each
# fit the samples whose residual is above PMASK sigma (widened by PWID samples) are left out of the next fit of that
# channel.  A real pulse is positive and single-signed, so it ends up outside its own fit; the ringing, which is
# two-signed, keeps most of its samples.
#   step 1: x_ch = beta_ch c_b(t) on free samples (c_b = block median, as v1)        -> W1 = x - beta c_b
#   step 2 (second=True): u_b = leading time component of W1 over the block's channels whose free fraction is
#           complete (no masked sample), used if s0 / sqrt(n ns) > sv_nsig x median noise; x = g_ch u_b on free samples.
# --------------------------------------------------------------------------------------------------------------------
NITER = 3


def _fit_masked(Y, X, nz, niter, pmask, pwid, minfree, fallback):
    """Per channel least squares Y[:, ch] = X @ coef_ch on samples that do not carry a positive residual.
    Y (ns, n), X (ns, k).  Returns residual (ns, n), free mask (ns, n)."""
    ns, n = Y.shape
    F = np.ones((ns, n), bool)
    k = X.shape[1]
    for _ in range(niter):
        Fw = F.astype(np.float64)
        XtX = np.einsum('tk,tl,tc->ckl', X, X, Fw) + 1e-9 * np.eye(k)[None]
        XtY = np.einsum('tk,tc->ck', X, Y * Fw)
        coef = np.linalg.solve(XtX, XtY[..., None])[..., 0]
        bad = F.sum(0) < minfree
        if bad.any():
            coef[bad] = fallback
        r = Y - X @ coef.T
        F_new = ~_pulse_mask(r, nz, pmask, pwid)
        if np.array_equal(F_new, F):
            break
        F = F_new
    return r, F


def deflect4(wfm: np.ndarray, noise: np.ndarray, second: bool = False, sv_nsig: float = SV_NSIG, niter: int = NITER,
             pmask: float = PMASK, pwid: int = PWID, minfree: int = MINFREE, return_info: bool = False):
    ns, nc = wfm.shape
    nblk = nc // CNS
    nz = np.maximum(np.asarray(noise, np.float64), 1.0)
    x = wfm.astype(np.float64)
    B = x.reshape(ns, nblk, CNS)
    c = np.median(B, axis=2)
    out = np.empty_like(x)
    used = 0
    for b in range(nblk):
        sl = slice(b * CNS, (b + 1) * CNS)
        X = c[:, b:b + 1]
        if np.abs(X).sum() < 1e-6:
            out[:, sl] = x[:, sl] - X
            continue
        r, F = _fit_masked(x[:, sl], X, nz[sl], niter, pmask, pwid, minfree, np.array([1.0]))
        if second:
            sel = F.all(0)
            if sel.sum() >= 8:
                u, s, _ = np.linalg.svd(r[:, sel], full_matrices=False)
                if s[0] / np.sqrt(sel.sum() * ns) >= sv_nsig * np.median(nz[sl]):
                    r, _ = _fit_masked(r, u[:, :1], nz[sl], niter, pmask, pwid, minfree, np.array([0.0]))
                    used += 1
        out[:, sl] = r
    out = out.astype(wfm.dtype, copy=False)
    return (out, dict(blocks_second=used)) if return_info else out


# --------------------------------------------------------------------------------------------------------------------
# group_dc (open items, 2026-11-23): after deflect (v1), remove the local baseline excursions -- runs of >= EXC_MINRUN
# neighbouring channels whose whole window sits below -EXC_NSIG sigma (median over the window).  Only those channels
# change: each loses its own offset, the median of its samples that are not part of a pulse (> PMASK sigma above the
# channel's median, +-PWID samples).  Everything else stays as v1.
# --------------------------------------------------------------------------------------------------------------------
EXC_NSIG = 3.0
EXC_MINRUN = 5


def group_dc(wfm: np.ndarray, noise: np.ndarray, w1: np.ndarray | None = None, return_info: bool = False):
    nz = np.maximum(np.asarray(noise, np.float64), 1.0)
    W = (deflect(wfm.astype(np.float32), noise) if w1 is None else w1).astype(np.float64)
    lev = np.median(W, axis=0)
    low = np.flatnonzero(lev < -EXC_NSIG * nz)
    n = 0
    if len(low):
        for r in np.split(low, np.flatnonzero(np.diff(low) > 1) + 1):
            if len(r) < EXC_MINRUN:
                continue
            n += 1
            for ch in r:
                x = W[:, ch]
                free = ~_pulse_mask((x - np.median(x))[:, None], nz[ch:ch + 1], PMASK, PWID)[:, 0]
                W[:, ch] = x - (np.median(x[free]) if free.sum() >= MINFREE else np.median(x))
    out = W.astype(wfm.dtype, copy=False)
    return (out, dict(groups=n)) if return_info else out
