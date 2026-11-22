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
