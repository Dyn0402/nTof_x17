#!/usr/bin/env python3
"""
pulses.py -- find pulses on the raw flash_tail waveforms, flash tail included,
for the full recovery analysis (amplitude against time since the flash).

    python -m ntof_scint_recovery.pulses <flash_<run>_<idx>.root> [...] --out pulses.npz [--hist]

Per channel, per bunch:

* **Baseline.**  Each zero-suppressed block: the median of its first 200 samples
  (the 259 pre-samples before the ZS trigger), constant over the block (~1 us).
  The unsuppressed head block (the flash and its tail): a running median in
  chunks of ``CHUNK_NS`` interpolated between chunk centres -- it follows the
  flash tail on time scales longer than a chunk and lags inside the steep first
  microseconds, which is exactly where "can pulses still be found" is decided.
* **Pulses.**  The baseline-subtracted signal in the pulse direction, smoothed with
  a boxcar of ``SMOOTH`` samples, local maxima above ``THR_MV`` that are the
  largest within +-``SEP_NS``.  Amplitude = the unsmoothed peak height over the
  local baseline; ``width`` = samples above half of it (a pulse-shape check).

Output, per pulse: run, bunch, det, detn, t_us (since the flash at the detector),
amp_mV, width_ns, base_mV (local baseline relative to pre-flash, pulse direction
+), in_head (found inside the unsuppressed flash block).  With ``--hist``, 2D
histograms (log t x log amp, log t x width) per channel and intensity class
instead of the pulse list, for running on the whole dataset in condor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ntof_scint_recovery.flash_summary import C_FALLBACK, DET_NAMES

MV = 0.0306
FAMILY = {d: ('WAL' if d < 4 else 'PSS' if d < 8 else 'LIQ') for d in range(12)}
SMOOTH = {'WAL': 16, 'PSS': 3, 'LIQ': 1}
SEP_NS = {'WAL': 150, 'PSS': 30, 'LIQ': 20}
THR_MV = {'WAL': 4.0, 'PSS': 4.0, 'LIQ': 4.0}
CHUNK_NS = {'WAL': 400, 'PSS': 200, 'LIQ': 200}
NPRE = 200
T_EDGES_US = np.concatenate([np.logspace(np.log10(0.1), 4, 101), [3e4, 3.2e4]])
A_EDGES = np.logspace(np.log10(2), np.log10(2000), 91)
W_EDGES = np.arange(0, 301, 3)


def head_baseline(y: np.ndarray, chunk: int) -> np.ndarray:
    """Running median in chunks, linearly interpolated (pulse-direction frame)."""
    n = len(y) // chunk
    if n < 2:
        return np.full(len(y), np.median(y))
    med = np.median(y[:n * chunk].reshape(n, chunk), axis=1)
    centres = (np.arange(n) + 0.5) * chunk
    return np.interp(np.arange(len(y)), centres, med)


def find(y: np.ndarray, fam: str):
    """Indices and heights of pulses in a baseline-subtracted trace."""
    s = uniform_filter1d(y, SMOOTH[fam]) if SMOOTH[fam] > 1 else y
    sep = SEP_NS[fam]
    mx = maximum_filter1d(s, 2 * sep + 1)
    pk = np.nonzero((s == mx) & (s > THR_MV[fam]))[0]
    if len(pk):
        keep = np.r_[True, np.diff(pk) > sep]       # plateaus: keep the first sample
        pk = pk[keep]
    return pk


def widths(y: np.ndarray, pk: np.ndarray, lim: int = 300) -> np.ndarray:
    """Samples above half height around each peak, vectorised over peaks."""
    if not len(pk):
        return np.zeros(0, int)
    off = np.arange(-lim, lim + 1)
    idx = np.clip(pk[:, None] + off[None, :], 0, len(y) - 1)
    above = y[idx] > (y[pk] / 2)[:, None]
    c = lim
    # extent of the run of True through the centre
    left = np.argmin(above[:, c::-1], axis=1)
    left[above[:, c::-1].all(axis=1)] = lim + 1
    right = np.argmin(above[:, c:], axis=1)
    right[above[:, c:].all(axis=1)] = lim + 1
    return left + right - 1


def width(y: np.ndarray, i: int, amp: float, lim: int = 300) -> int:
    h = amp / 2
    a = i
    while a > 0 and i - a < lim and y[a - 1] > h:
        a -= 1
    b = i
    while b < len(y) - 1 and b - i < lim and y[b + 1] > h:
        b += 1
    return b - a + 1


def pulses_file(path: str, C: dict) -> dict:
    f = uproot.open(path)
    prov = json.loads(Path(path[:-5] + '_provenance.json').read_text())
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    ppp = dict(zip(meta['bunch'].tolist(), meta['PulseIntensity'].tolist()))
    bb = f['blocks']['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    cols = {k: [] for k in ('run', 'bunch', 'det', 'detn', 't_us', 'amp_mV', 'width_ns', 'base_mV', 'in_head', 'ppp',
                            'pk_amp')}
    for e0, e1 in zip(starts, stops):
        a = f['blocks'].arrays(['bunch', 'det', 'detn', 'tof0', 'n', 'samples'], entry_start=e0, entry_stop=e1)
        bunch = int(a['bunch'][0])
        det, detn = np.asarray(a['det']).astype(int), np.asarray(a['detn']).astype(int)
        tof0, n = np.asarray(a['tof0']).astype(np.int64), np.asarray(a['n']).astype(np.int64)
        flat = np.asarray(ak.flatten(a['samples'])).astype(np.float32)
        offs = np.r_[0, np.cumsum(n)[:-1]]
        k = np.nonzero((det == 12) & (tof0 == 0))[0]
        if not len(k):
            continue
        x = flat[offs[k[0]]:offs[k[0]] + n[k[0]]]
        b0 = np.median(x[:5000])
        t_pk = int(np.argmax(x - b0))
        pk_amp = float((x[t_pk] - b0) * MV)
        for d in range(12):
            fam = FAMILY[d]
            pol = 1 if d < 4 else -1
            t_fl = t_pk + C[DET_NAMES[d]]
            for ch in np.unique(detn[det == d]):
                sel = np.nonzero((det == d) & (detn == ch))[0]
                head = sel[tof0[sel] == 0]
                if not len(head):
                    continue
                h = head[0]
                xh = flat[offs[h]:offs[h] + n[h]]
                th = (np.arange(n[h]) - t_fl) / 1e3
                pre = xh[(th > -9) & (th < -1)]
                if len(pre) < 1000:
                    continue
                base0 = float(np.median(pre))
                segs = []
                yh = pol * (xh - base0) * MV
                bh = head_baseline(yh, CHUNK_NS[fam])
                segs.append((0, yh, bh, True))
                for s in sel:
                    if tof0[s] == 0 or n[s] < NPRE + 20:
                        continue
                    y = pol * (flat[offs[s]:offs[s] + n[s]] - base0) * MV
                    segs.append((int(tof0[s]), y, np.full(len(y), np.median(y[:NPRE])), False))
                for t0, y, b, ih in segs:
                    r = y - b
                    pk = find(r, fam)
                    if ih:
                        pk = pk[(t0 + pk - t_fl) > -1000]        # nothing before the flash but a few pre-flash
                    wd = widths(r, pk)
                    for i, wi in zip(pk, wd):
                        amp = float(r[i])
                        cols['run'].append(prov['run'])
                        cols['bunch'].append(bunch)
                        cols['det'].append(d)
                        cols['detn'].append(int(ch))
                        cols['t_us'].append((t0 + i - t_fl) / 1e3)
                        cols['amp_mV'].append(amp)
                        cols['width_ns'].append(int(wi))
                        cols['base_mV'].append(float(b[i]))
                        cols['in_head'].append(ih)
                        cols['ppp'].append(ppp.get(bunch, np.nan))
                        cols['pk_amp'].append(pk_amp)
    return {k: np.array(v) for k, v in cols.items()}


def histogram(P: dict) -> dict:
    """2D histograms per (det, detn, class): log t x log amp, log t x width."""
    cls = np.where(P['ppp'] >= 6.5e12, 1, np.where(P['ppp'] >= 1e11, 0, -1))
    nop = ~np.isfinite(P['ppp'])                    # unprocessed run: PKUP amplitude, as quick.py
    cls[nop] = np.where(P['pk_amp'][nop] >= 650, 1, np.where(P['pk_amp'][nop] >= 130, 0, -1))
    from ntof_scint_recovery.flash_summary import CHANNELS
    nT, nA, nW = len(T_EDGES_US) - 1, len(A_EDGES) - 1, len(W_EDGES) - 1
    HA = np.zeros((2, len(CHANNELS), nT, nA), np.float32)
    HW = np.zeros((2, len(CHANNELS), nT, nW), np.float32)
    ci = {c: i for i, c in enumerate(CHANNELS)}
    it = np.digitize(P['t_us'], T_EDGES_US) - 1
    ia = np.clip(np.digitize(P['amp_mV'], A_EDGES) - 1, 0, nA - 1)
    iw = np.clip(np.digitize(P['width_ns'], W_EDGES) - 1, 0, nW - 1)
    ich = np.array([ci.get((int(d), int(c)), -1) for d, c in zip(P['det'], P['detn'])], dtype=int)
    # a run with no scintillators (1-9 July) has no pulses: empty float arrays cannot index
    it, ia, iw, cls = it.astype(int), ia.astype(int), iw.astype(int), cls.astype(int)
    ok = (it >= 0) & (it < nT) & (cls >= 0) & (ich >= 0)
    np.add.at(HA, (cls[ok], ich[ok], it[ok], ia[ok]), 1)
    np.add.at(HW, (cls[ok], ich[ok], it[ok], iw[ok]), 1)
    nb = {c: len({(r, b) for r, b, k in zip(P['run'], P['bunch'], cls) if k == c}) for c in (0, 1)}
    return dict(HA=HA, HW=HW, t_edges_us=T_EDGES_US, a_edges=A_EDGES, w_edges=W_EDGES,
                channels=np.array(CHANNELS), n_par=nb[0], n_ded=nb[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('products', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hist', action='store_true', help='write histograms, not the pulse list')
    a = ap.parse_args(argv)
    if a.hist:
        # one file at a time: a run's pulse list would not fit a batch slot
        H, n = None, 0
        for p in a.products:
            h = histogram(pulses_file(p, C_FALLBACK))
            n += 1
            if H is None:
                H = h
            else:
                for k in ('HA', 'HW', 'n_par', 'n_ded'):
                    H[k] = H[k] + h[k]
        np.savez_compressed(a.out, **H)
        print(f'{n} files histogrammed ({H["n_par"]} par, {H["n_ded"]} ded bunches) -> {a.out}')
        return 0
    parts = [pulses_file(p, C_FALLBACK) for p in a.products]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if False:
        pass
    else:
        np.savez_compressed(a.out, **out)
        print(f'{len(out["t_us"]):,} pulses -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
