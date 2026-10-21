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
RF_EDGES = np.concatenate([[0], np.logspace(-2, 1, 61)])      # residual rms / amplitude
#: coincidence grids: coarse time, wall-bar dt and ln(A1/A2)
TC_EDGES_US = np.concatenate([np.logspace(-1, 4, 26), [3e4, 3.2e4]])
DT_EDGES = np.arange(-30, 31, 2.0)
LR_EDGES = np.linspace(-3, 3, 61)
SIG_PW = (-25.0, 5.0)        # plastic - wall, ns: the coincidence peak sits at -15..-5
SIDE_PW = (60.0, 210.0)
DT_BAR = 30.0
ACC_SHIFT = 500
#: small-pulse (single-photoelectron candidate) histogram: peaks in the quiet tail of
#: each ZS block (from SPE_SKIP samples on, well after the pulse that opened it)
SPE_EDGES = np.arange(0.5, 10.01, 0.1)
SPE_SKIP = 700
SPE_SMOOTH = {'WAL': 16, 'PSS': 3, 'LIQ': 1}
SPE_SEP = {'WAL': 150, 'PSS': 60, 'LIQ': 60}


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


_TMPL = None


def templates():
    """Per-family template and the least-squares projector for A*T + B*T' + b0 + b1*t
    (the derivative term absorbs a sub-ns time shift, delta = -B/A)."""
    global _TMPL
    if _TMPL is None:
        z = np.load(Path(__file__).with_name('templates.npz'))
        _TMPL = {}
        for fam in ('WAL', 'PSS', 'LIQ'):
            t = z[f't_ns_{fam}'].astype(float)
            T = z[fam].astype(float)
            X = np.stack([T, np.gradient(T), np.ones_like(t), t / 100.0], axis=1)
            _TMPL[fam] = (int(-t[0]), int(t[-1]), X, np.linalg.pinv(X))
    return _TMPL


def fit_pulses(y: np.ndarray, pk: np.ndarray, fam: str, noise: float):
    """Template fit of every peak in a segment: amplitude, chi2/ndf, local baseline
    slope.  ``y`` is the signal relative to the PRE-FLASH baseline (pulse direction
    +), so the local linear baseline is fitted, not assumed.  Peaks whose window
    runs off the segment get NaN."""
    pre, post, X, Xp = templates()[fam]
    amp = np.full(len(pk), np.nan)
    chi = np.full(len(pk), np.nan)
    slope = np.full(len(pk), np.nan)
    rf = np.full(len(pk), np.nan)
    ok = (pk - pre >= 0) & (pk + post < len(y))
    if ok.any():
        idx = pk[ok][:, None] + np.arange(-pre, post + 1)[None, :]
        Y = y[idx]                                   # (npk, L)
        coef = Y @ Xp.T                              # (npk, 4)
        res = Y - coef @ X.T
        amp[ok] = coef[:, 0]
        chi[ok] = (res ** 2).sum(axis=1) / (max(noise, 0.2) ** 2 * (X.shape[0] - X.shape[1]))
        slope[ok] = coef[:, 3]
        rf[ok] = np.sqrt((res ** 2).mean(axis=1)) / np.maximum(np.abs(coef[:, 0]), 1e-3)
    return amp, chi, rf


def pulses_file(path: str, C: dict) -> dict:
    f = uproot.open(path)
    prov = json.loads(Path(path[:-5] + '_provenance.json').read_text())
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    ppp = dict(zip(meta['bunch'].tolist(), meta['PulseIntensity'].tolist()))
    bb = f['blocks']['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    cols = {k: [] for k in ('run', 'bunch', 'det', 'detn', 't_us', 'amp_mV', 'width_ns', 'base_mV', 'in_head', 'ppp',
                            'pk_amp', 'amp_fit', 'chi2', 'resfrac', 'n_neigh')}
    from ntof_scint_recovery.flash_summary import CHANNELS
    ci = {c: i for i, c in enumerate(CHANNELS)}
    SPE = np.zeros((2, len(CHANNELS), len(TC_EDGES_US) - 1, len(SPE_EDGES) - 1), np.float32)
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
        pp = ppp.get(bunch, np.nan)
        bcls = (1 if pp >= 6.5e12 else 0 if pp >= 1e11 else -1) if np.isfinite(pp) else \
            (1 if pk_amp >= 650 else 0 if pk_amp >= 130 else -1)
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
                noise = float(np.std(pre)) * MV
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
                    if not ih and bcls >= 0 and len(r) > SPE_SKIP + 50 and (d, int(ch)) in ci:
                        z = r[SPE_SKIP:]
                        zs = uniform_filter1d(z, SPE_SMOOTH[fam]) if SPE_SMOOTH[fam] > 1 else z
                        mxs = maximum_filter1d(zs, 2 * SPE_SEP[fam] + 1)
                        sp = np.nonzero((zs == mxs) & (zs > SPE_EDGES[0]) & (zs < SPE_EDGES[-1]))[0]
                        if len(sp):
                            tsp = (t0 + SPE_SKIP + sp - t_fl) / 1e3
                            it_ = np.digitize(tsp, TC_EDGES_US) - 1
                            ia_ = np.digitize(zs[sp], SPE_EDGES) - 1
                            g_ = (it_ >= 0) & (it_ < SPE.shape[2])
                            np.add.at(SPE, (bcls, ci[(d, int(ch))], it_[g_], ia_[g_]), 1)
                    pk = find(r, fam)
                    if ih:
                        pk = pk[(t0 + pk - t_fl) > -1000]        # nothing before the flash but a few pre-flash
                    if not len(pk):
                        continue
                    wd = widths(r, pk)
                    af, ch2, rfr = fit_pulses(y, pk, fam, noise)
                    pre_, post_ = templates()[fam][:2]
                    # other peaks inside this peak's fit window: pile-up flag
                    lo = np.searchsorted(pk, pk - pre_)
                    hi = np.searchsorted(pk, pk + post_, side='right')
                    m = len(pk)
                    cols['run'].append(np.full(m, prov['run']))
                    cols['bunch'].append(np.full(m, bunch))
                    cols['det'].append(np.full(m, d))
                    cols['detn'].append(np.full(m, int(ch)))
                    cols['t_us'].append((t0 + pk - t_fl) / 1e3)
                    cols['amp_mV'].append(r[pk].astype(float))
                    cols['width_ns'].append(wd)
                    cols['base_mV'].append(b[pk].astype(float))
                    cols['in_head'].append(np.full(m, ih))
                    cols['ppp'].append(np.full(m, ppp.get(bunch, np.nan)))
                    cols['pk_amp'].append(np.full(m, pk_amp))
                    cols['amp_fit'].append(af)
                    cols['chi2'].append(ch2)
                    cols['resfrac'].append(rfr)
                    cols['n_neigh'].append(hi - lo - 1)
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out['_SPE'] = SPE
    return out


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
    # template-fit amplitude and shape mismatch, isolated pulses only (no neighbour in the fit window)
    iso = ok & (P['n_neigh'] == 0) & np.isfinite(P['amp_fit'])
    iaf = np.clip(np.digitize(P['amp_fit'], A_EDGES) - 1, 0, nA - 1)
    irf = np.clip(np.digitize(P['resfrac'], RF_EDGES) - 1, 0, len(RF_EDGES) - 2)
    HF = np.zeros((2, len(CHANNELS), nT, nA), np.float32)
    HR = np.zeros((2, len(CHANNELS), nT, len(RF_EDGES) - 1), np.float32)
    np.add.at(HF, (cls[iso], ich[iso], it[iso], iaf[iso]), 1)
    np.add.at(HR, (cls[iso], ich[iso], it[iso], irf[iso]), 1)
    nb = {c: len({(r, b) for r, b, k in zip(P['run'], P['bunch'], cls) if k == c}) for c in (0, 1)}
    out = dict(HA=HA, HW=HW, HF=HF, HR=HR, t_edges_us=T_EDGES_US, a_edges=A_EDGES, w_edges=W_EDGES,
               rf_edges=RF_EDGES, channels=np.array(CHANNELS), n_par=nb[0], n_ded=nb[1])
    out.update(coincidences(P, cls))
    if '_SPE' in P:
        out['SPE'] = P['_SPE']
        out['spe_edges'] = SPE_EDGES
    return out


def _amp(P):
    a = P['amp_fit'].copy()
    bad = ~np.isfinite(a)
    a[bad] = P['amp_mV'][bad]
    return a


def coincidences(P: dict, cls: np.ndarray) -> dict:
    """Coincidence histograms on the coarse time grid TC_EDGES_US.

    ``PW_sig/PW_side`` (class, arm, plastic bar, t, amp): plastic pulses with a
      wall pulse of the same arm at dt = t_plastic - t_wall in SIG_PW (signal) or
      SIDE_PW (accidentals; scale by SIG/SIDE width ratio, stored as ``pw_side_scale``).
    ``BAR_r/BAR_g`` (class, wall, bar, t, dt, ln(A1/A2) | geometric mean amp):
      the two ends of a wall bar (detn 2g+1, 2g+2) within +-DT_BAR ns; ``*_acc`` the
      same with end 2 shifted by +ACC_SHIFT ns (accidentals, same window).
    """
    amp = _amp(P)
    key = (P['run'].astype(np.int64) * 10000 + P['bunch'].astype(np.int64)) * 10**8 \
        + np.round(P['t_us'] * 1e3).astype(np.int64)            # ns, unique per (run, bunch)
    itc = np.digitize(P['t_us'], TC_EDGES_US) - 1
    nTC = len(TC_EDGES_US) - 1
    nA = len(A_EDGES) - 1
    good = (cls >= 0) & (itc >= 0) & (itc < nTC)
    PWs = np.zeros((2, 4, 2, nTC, nA), np.float32)
    PWb = np.zeros_like(PWs)
    nDT, nR, nG = len(DT_EDGES) - 1, len(LR_EDGES) - 1, len(A_EDGES) - 1
    BR = np.zeros((2, 4, 4, nTC, nDT, nR), np.float32)
    BG = np.zeros((2, 4, 4, nTC, nDT, nG), np.float32)
    BRa = np.zeros_like(BR)
    BGa = np.zeros_like(BG)

    def any_in(kw, kp, lo, hi):
        a = np.searchsorted(kw, kp - hi)
        b = np.searchsorted(kw, kp - lo, side='right')
        return b > a

    for arm in range(4):
        w = good & (P['det'] == arm) & (amp > 10)
        kw = np.sort(key[w])
        for bar in (1, 2):
            p = good & (P['det'] == 4 + arm) & (P['detn'] == bar) & (amp > 20)
            kp = key[p]
            ia = np.clip(np.digitize(amp[p], A_EDGES) - 1, 0, nA - 1)
            for H, (lo, hi) in ((PWs, SIG_PW), (PWb, SIDE_PW)):
                m = any_in(kw, kp, lo, hi)
                np.add.at(H, (cls[p][m], arm, bar - 1, itc[p][m], ia[m]), 1)
        for g in range(4):
            e1 = good & (P['det'] == arm) & (P['detn'] == 2 * g + 1) & (amp > 10)
            e2 = good & (P['det'] == arm) & (P['detn'] == 2 * g + 2) & (amp > 10)
            if not e1.any() or not e2.any():
                continue
            o = np.argsort(key[e2])
            k2, a2 = key[e2][o], amp[e2][o]
            k1, a1 = key[e1], amp[e1]
            for (HRr, HGg, shift) in ((BR, BG, 0), (BRa, BGa, ACC_SHIFT)):
                kk = k2 + shift
                j = np.clip(np.searchsorted(kk, k1), 1, len(kk) - 1)
                jj = np.where(np.abs(kk[j] - k1) < np.abs(kk[j - 1] - k1), j, j - 1)
                dt = (k1 - kk[jj]).astype(float)
                m = np.abs(dt) <= DT_BAR
                if not m.any():
                    continue
                idt = np.clip(np.digitize(dt[m], DT_EDGES) - 1, 0, nDT - 1)
                lr = np.log(np.maximum(a1[m], 1e-3) / np.maximum(a2[jj][m], 1e-3))
                ir = np.clip(np.digitize(lr, LR_EDGES) - 1, 0, nR - 1)
                gm = np.sqrt(np.maximum(a1[m], 0) * np.maximum(a2[jj][m], 0))
                ig = np.clip(np.digitize(gm, A_EDGES) - 1, 0, nG - 1)
                c1, t1 = cls[e1][m], itc[e1][m]
                np.add.at(HRr, (c1, arm, g, t1, idt, ir), 1)
                np.add.at(HGg, (c1, arm, g, t1, idt, ig), 1)
    return dict(PW_sig=PWs, PW_side=PWb, pw_side_scale=(SIG_PW[1] - SIG_PW[0]) / (SIDE_PW[1] - SIDE_PW[0]),
                BAR_r=BR, BAR_g=BG, BAR_r_acc=BRa, BAR_g_acc=BGa,
                tc_edges_us=TC_EDGES_US, dt_edges=DT_EDGES, lr_edges=LR_EDGES)


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
                for k in ('HA', 'HW', 'HF', 'HR', 'n_par', 'n_ded', 'PW_sig', 'PW_side',
                          'BAR_r', 'BAR_g', 'BAR_r_acc', 'BAR_g_acc', 'SPE'):
                    H[k] = H[k] + h[k]
        np.savez_compressed(a.out, **H)
        print(f'{n} files histogrammed ({H["n_par"]} par, {H["n_ded"]} ded bunches) -> {a.out}')
        return 0
    parts = [pulses_file(p, C_FALLBACK) for p in a.products]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith('_')}
    if False:
        pass
    else:
        np.savez_compressed(a.out, **out)
        print(f'{len(out["t_us"]):,} pulses -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
