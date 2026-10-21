#!/usr/bin/env python3
"""
inject.py -- how well can a pulse be reconstructed on the flash tail?  Inject pulses of
known amplitude and time into the real recorded flash traces, run the reconstruction,
and compare.

    python -m ntof_scint_recovery.inject <flash_<run>_<idx>.root> [...] --out inject.npz [--max-bunches N]

The unsuppressed head block of every channel (the flash, its tail and ~10 us before it;
walls ~30 us, plastics ~200 us, liquids ~20 us after the flash) is the canvas.  Per
bunch, channel and amplitude, ``N_PASS`` passes each add a comb of family-template pulses
(spacing ``SPACING_NS``, random offset, random sub-ns phase) to the trace, clip at the
channel's saturation level, and reconstruct the whole trace exactly as ``pulses.py``
does: running-median baseline (``CHUNK_NS``) -> peak finder -> template fit with a
local linear baseline and a derivative term (sub-ns time shift).  The injected pulse is
found if a peak lies within ``MATCH_NS`` of it.

The pre-flash part of the same trace (-8 to -1.5 us) is the control: same noise, same
code, no flash.  Bias and resolution against time are quoted relative to it.

Truth: amplitude A (the template's peak is 1) and time t0 (ns, continuous).
Reconstructed: A_fit and t_fit = peak sample + delta, delta = -B/A.

Per (class, channel, amplitude, time bin) histograms of
``R`` = A_fit/A, ``D`` = t_fit - t0 (ns), plus the "oracle" ratio ``RO`` (the same fit on
the injected pulse alone + noise-free trace difference, i.e. what a perfect baseline and
no pile-up would give), counts ``N`` (injected) and ``NF`` (found), and ``NP`` (a peak
already sat within the match window in the untouched trace: pile-up on a real pulse).
Every injection is also flagged **clean** or **piled**: piled if a real pulse of the
untouched trace (pile-up-aware finder, prominence >= max(4 mV, 5 % of A)) lies closer than
the template body (above 5 % of peak: WAL ~217, PSS ~42, LIQ ~18 ns).  ``R`` and ``D`` carry that flag as an axis (0 clean, 1 piled), ``NC``/``NFC`` count
the clean ones (injected / found).  Clean pulses measure the flash tail itself (baseline,
shape, clipping); the piled fraction is the beam-induced occupancy, which scales with rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from ntof_scint_recovery.flash_summary import C_FALLBACK, CHANNELS, DET_NAMES
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ntof_scint_recovery.pulses import CHUNK_NS, FAMILY, SMOOTH, THR_MV, find, head_baseline

MV = 0.0306
AMPS = {'WAL': (8.0, 15.0, 30.0, 60.0, 120.0),
        'PSS': (25.0, 50.0, 100.0, 200.0, 400.0, 800.0),
        'LIQ': (25.0, 50.0, 100.0, 200.0, 400.0)}
SPACING_NS = {'WAL': 2000, 'PSS': 600, 'LIQ': 600}
MATCH_NS = {'WAL': 40, 'PSS': 6, 'LIQ': 6}
N_PASS = 12
CTRL_US = (-8.0, -1.5)
#: time bins: [control] + log 0.05 us .. 250 us
T_EDGES_US = np.logspace(np.log10(0.05), np.log10(250.0), 38)
R_EDGES = np.linspace(0.0, 2.5, 251)
D_HALF = {'WAL': 40.0, 'PSS': 8.0, 'LIQ': 8.0}
N_D = 160
FAM_I = {'WAL': 0, 'PSS': 1, 'LIQ': 2}

_T = None


def tmpl():
    """Per family: (pre, post, t_ns, T, X, pinv(X)) with X = [T, T', 1, t/100]."""
    global _T
    if _T is None:
        z = np.load(Path(__file__).with_name('templates.npz'))
        _T = {}
        for fam in ('WAL', 'PSS', 'LIQ'):
            t = z[f't_ns_{fam}'].astype(float)
            T = z[fam].astype(float)
            X = np.stack([T, np.gradient(T), np.ones_like(t), t / 100.0], axis=1)
            _T[fam] = (int(-t[0]), int(t[-1]), t, T, X, np.linalg.pinv(X))
    return _T


def fit_at(y: np.ndarray, pk: np.ndarray, fam: str):
    """Template fit at given peak samples: amplitude, delta (ns)."""
    pre, post, _, _, X, Xp = tmpl()[fam]
    A = np.full(len(pk), np.nan)
    dl = np.full(len(pk), np.nan)
    ok = (pk - pre >= 0) & (pk + post < len(y))
    if ok.any():
        idx = pk[ok][:, None] + np.arange(-pre, post + 1)[None, :]
        coef = y[idx] @ Xp.T
        A[ok] = coef[:, 0]
        dl[ok] = np.clip(-coef[:, 1] / np.where(np.abs(coef[:, 0]) > 1e-6, coef[:, 0], 1e-6), -5, 5)
    return A, dl


def body_width(fam: str) -> float:
    """Width of the template where it is above 5 % of its peak (WAL ~217, PSS ~42, LIQ ~18 ns):
    two pulses closer than this overlap in their bodies."""
    _, _, t, T, _, _ = tmpl()[fam]
    s = t[T > 0.05]
    return float(s[-1] - s[0])


def comb(n: int, fam: str, t0s: np.ndarray, amp: float) -> np.ndarray:
    """Sum of template pulses of amplitude amp at continuous sample positions t0s."""
    pre, post, t, T, _, _ = tmpl()[fam]
    out = np.zeros(n, np.float32)
    for t0 in t0s:
        i0 = int(np.floor(t0))
        frac = t0 - i0
        lo, hi = i0 - pre, i0 + post + 1
        if lo < 0 or hi > n:
            continue
        # sample k sits at time (k - t0); T is tabulated on integer ns about its peak
        out[lo:hi] += amp * np.interp(np.arange(-pre, post + 1) - frac, t, T, left=0.0, right=0.0)
    return out


def reco(y: np.ndarray, fam: str) -> np.ndarray:
    """The pulses.py reconstruction of a head trace: found peak samples."""
    r = y - head_baseline(y, CHUNK_NS[fam])
    return find(r, fam)


#: the pile-up-aware finder: same smoothing and threshold, a shorter dead zone between peaks
SEP_MULTI = {'WAL': 40, 'PSS': 8, 'LIQ': 6}
#: a real pulse counts as pile-up for an injected one of amplitude A if its prominence >= this x A
PILE_FRAC = 0.05


def reco_multi(y: np.ndarray, fam: str, prom: float | None = None) -> np.ndarray:
    """Pile-up-aware finder: local maxima of the smoothed, baseline-subtracted trace
    that rise at least ``prom`` (default the finding threshold) above the deeper of
    their two neighbouring minima (prominence), at least SEP_MULTI apart.  A slow
    tail the running median lags behind makes no peaks; a pulse on another's tail does."""
    r = y - head_baseline(y, CHUNK_NS[fam])
    s = uniform_filter1d(r, SMOOTH[fam]) if SMOOTH[fam] > 1 else r
    p = THR_MV[fam] if prom is None else prom
    pk, _ = find_peaks(s, height=THR_MV[fam], prominence=p, distance=SEP_MULTI[fam],
                       wlen=2 * (tmpl()[fam][0] + tmpl()[fam][1]))
    return pk


def fit_multi(y: np.ndarray, pk_all: np.ndarray, targets: np.ndarray, fam: str):
    """Joint template fit: each target peak with every found peak whose template overlaps
    its window (A_k T + B_k T' each) and one linear baseline over the union window.
    Returns amplitude and delta (ns) of the targets."""
    pre, post, t, T, _, _ = tmpl()[fam]
    Tg = np.gradient(T)
    L = pre + post + 1
    A = np.full(len(targets), np.nan)
    dl = np.full(len(targets), np.nan)
    for i, p in enumerate(targets):
        lo, hi = p - pre, p + post + 1
        if lo < 0 or hi > len(y):
            continue
        nb = pk_all[(pk_all > lo - post) & (pk_all < hi + pre)]
        nb = nb[np.argsort(np.abs(nb - p))][:12]
        if p not in nb:
            nb = np.r_[p, nb[:11]]
        w0, w1 = max(0, min(lo, nb.min() - pre)), min(len(y), max(hi, nb.max() + post + 1))
        n = w1 - w0
        cols = []
        for q in nb:
            c, g = np.zeros(n), np.zeros(n)
            a0, a1 = max(w0, q - pre), min(w1, q + post + 1)
            if a1 <= a0:
                continue
            c[a0 - w0:a1 - w0] = T[a0 - (q - pre):a1 - (q - pre)]
            g[a0 - w0:a1 - w0] = Tg[a0 - (q - pre):a1 - (q - pre)]
            cols += [c, g]
        tt = np.arange(n) / 100.0
        X = np.stack(cols + [np.ones(n), tt], axis=1)
        coef, *_ = np.linalg.lstsq(X, y[w0:w1], rcond=None)
        j = int(np.nonzero(nb == p)[0][0])
        A[i] = coef[2 * j]
        dl[i] = np.clip(-coef[2 * j + 1] / (coef[2 * j] if abs(coef[2 * j]) > 1e-6 else 1e-6), -5, 5)
    return A, dl


def tbin(t_us: np.ndarray) -> np.ndarray:
    """0 = control, 1.. = log bins, -1 = outside."""
    b = np.digitize(t_us, T_EDGES_US)                 # 0 below first edge, len at/above last
    b = np.where((b >= 1) & (b < len(T_EDGES_US)), b, -1)
    b = np.where((t_us > CTRL_US[0]) & (t_us < CTRL_US[1]), 0, b)
    return b


METHODS = ('single', 'multi')


def empty():
    nC, nT = len(CHANNELS), len(T_EDGES_US)            # control + (len-1) log bins
    nA = max(len(v) for v in AMPS.values())
    nM = len(METHODS)
    return dict(N=np.zeros((2, nC, nA, nT), np.float64), NF=np.zeros((nM, 2, nC, nA, nT), np.float64),
                NP=np.zeros((2, nC, nA, nT), np.float64),
                R=np.zeros((nM, 2, nC, nA, nT, 2, len(R_EDGES) - 1), np.float32),
                RO=np.zeros((2, nC, nA, nT, len(R_EDGES) - 1), np.float32),
                D=np.zeros((nM, 2, nC, nA, nT, 2, N_D), np.float32),
                NC=np.zeros((2, nC, nA, nT), np.float64),
                NFC=np.zeros((nM, 2, nC, nA, nT), np.float64),
                n_bunch=np.zeros(2), n_trace=np.zeros(2))


def nearest(pk: np.ndarray, t0s: np.ndarray) -> np.ndarray:
    """Distance-ordered nearest element of sorted pk to each t0 (pk non-empty)."""
    if len(pk) == 1:
        return np.full(len(t0s), pk[0])
    j = np.clip(np.searchsorted(pk, t0s), 1, len(pk) - 1)
    return np.where(np.abs(pk[j] - t0s) < np.abs(pk[j - 1] - t0s), pk[j], pk[j - 1])


def score(H, im, key, yi, y, pk, t0s, ib, piled, fam, A, oracle):
    cls, c, ia = key
    if not len(pk):
        return
    near = nearest(pk, t0s)
    found = np.abs(near - t0s) <= MATCH_NS[fam]
    if not found.any():
        return
    pkf, t0f, ibf, pf = near[found], t0s[found], ib[found], piled[found].astype(int)
    np.add.at(H['NF'], (im, cls, c, ia, ibf), 1)
    np.add.at(H['NFC'], (im, cls, c, ia, ibf[pf == 0]), 1)
    Af, dl = fit_at(yi, pkf, fam) if METHODS[im] == 'single' else fit_multi(yi, pk, pkf, fam)
    ok = np.isfinite(Af)
    ir = np.clip(np.digitize(Af[ok] / A, R_EDGES) - 1, 0, len(R_EDGES) - 2)
    np.add.at(H['R'], (im, cls, c, ia, ibf[ok], pf[ok], ir), 1)
    if oracle:
        # the same fit on the injected pulse alone (no flash, no noise, no pile-up, clipping kept):
        # what the clipping alone does
        Ao, _ = fit_at(yi - y, pkf, fam)
        iro = np.clip(np.digitize(Ao[ok] / A, R_EDGES) - 1, 0, len(R_EDGES) - 2)
        np.add.at(H['RO'], (cls, c, ia, ibf[ok], iro), 1)
    dt = pkf[ok] + dl[ok] - t0f[ok]
    hw = D_HALF[fam]
    idt = np.clip(((dt + hw) / (2 * hw) * N_D).astype(int), 0, N_D - 1)
    np.add.at(H['D'], (im, cls, c, ia, ibf[ok], pf[ok], idt), 1)


def inject_file(path: str, H: dict, rng: np.random.Generator, max_bunches: int | None = None,
                only_det: set | None = None) -> None:
    f = uproot.open(path)
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    ppp = dict(zip(meta['bunch'].tolist(), meta['PulseIntensity'].tolist()))
    T = f['blocks']
    bb = T['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    ci = {c: i for i, c in enumerate(CHANNELS)}
    nb = 0
    for e0, e1 in zip(starts, stops):
        if max_bunches is not None and nb >= max_bunches:
            break
        a = T.arrays(['bunch', 'det', 'detn', 'tof0', 'n', 'samples'], entry_start=e0, entry_stop=e1)
        bunch = int(a['bunch'][0])
        det, detn = np.asarray(a['det']).astype(int), np.asarray(a['detn']).astype(int)
        tof0 = np.asarray(a['tof0']).astype(np.int64)
        head = np.nonzero(tof0 == 0)[0]
        k = head[det[head] == 12]
        if not len(k):
            continue
        x = np.asarray(a['samples'][k[0]]).astype(np.float32)
        b0 = np.median(x[:5000])
        t_pk = int(np.argmax(x - b0))
        pk_amp = float((x[t_pk] - b0) * MV)
        pp = ppp.get(bunch, np.nan)
        cls = (1 if pp >= 6.5e12 else 0 if pp >= 1e11 else -1) if np.isfinite(pp) else \
            (1 if pk_amp >= 650 else 0 if pk_amp >= 130 else -1)
        if cls < 0:
            continue
        nb += 1
        H['n_bunch'][cls] += 1
        for h in head:
            d, ch = int(det[h]), int(detn[h])
            if d >= 12 or (d, ch) not in ci or (only_det is not None and d not in only_det):
                continue
            fam = FAMILY[d]
            pol = 1 if d < 4 else -1
            xh = np.asarray(a['samples'][h]).astype(np.float32)
            t_fl = t_pk + C_FALLBACK[DET_NAMES[d]]
            th = (np.arange(len(xh)) - t_fl) / 1e3
            pre = xh[(th > -9) & (th < -1)]
            if len(pre) < 1000:
                continue
            base0 = float(np.median(pre))
            if np.max(pol * (xh - base0)) * MV < 100:          # dead channel: no flash
                continue
            y = pol * (xh - base0) * MV
            # saturation: ADC limit, or a clipped plateau in this trace (the wall amplifiers ~1 V)
            adc_lim = ((32767 - base0) if pol > 0 else (base0 + 32768)) * MV
            ymax = float(y.max())
            ysat = ymax if (np.sum(y > ymax - 2.0) > 50) else adc_lim
            ysat = min(ysat, adc_lim)
            H['n_trace'][cls] += 1
            c = ci[(d, ch)]
            pk0 = reco(y, fam)
            m = MATCH_NS[fam]
            pre_, post_ = tmpl()[fam][:2]
            sp = SPACING_NS[fam]
            lo_s = max(pre_ + 1, int(t_fl + CTRL_US[0] * 1e3))
            hi_s = len(y) - post_ - 2
            for ia, A in enumerate(AMPS[fam]):
                # real pulses big enough to move this amplitude by >= 5 %: the clean / piled split
                pkm0 = reco_multi(y, fam, max(THR_MV[fam], PILE_FRAC * A))
                for _ in range(N_PASS):
                    t0s = np.arange(lo_s + rng.uniform(0, sp), hi_s, sp) + rng.uniform(0, 1, 1)[0]
                    yi = np.minimum(y + comb(len(y), fam, t0s, A), ysat)
                    ib = tbin((t0s - t_fl) / 1e3)
                    g = ib >= 0
                    t0s, ib = t0s[g], ib[g]
                    if not len(t0s):
                        continue
                    np.add.at(H['N'], (cls, c, ia, ib), 1)
                    if len(pk0):
                        q0 = np.abs(nearest(pk0, t0s) - t0s)
                        np.add.at(H['NP'], (cls, c, ia, ib[q0 <= m]), 1)
                    # piled: a real pulse whose template overlaps this pulse's fit window
                    if len(pkm0):
                        w_ = body_width(fam)
                        lo_ = np.searchsorted(pkm0, t0s - w_)
                        hi_ = np.searchsorted(pkm0, t0s + w_)
                        piled = hi_ > lo_
                    else:
                        piled = np.zeros(len(t0s), bool)
                    np.add.at(H['NC'], (cls, c, ia, ib[~piled]), 1)
                    score(H, 0, (cls, c, ia), yi, y, reco(yi, fam), t0s, ib, piled, fam, A, True)
                    score(H, 1, (cls, c, ia), yi, y, reco_multi(yi, fam), t0s, ib, piled, fam, A, False)


def save(H: dict, out: Path, n_files: int):
    np.savez_compressed(out, **H, n_files=n_files, channels=np.array(CHANNELS), t_edges_us=T_EDGES_US,
                        ctrl_us=np.array(CTRL_US), r_edges=R_EDGES,
                        amps_WAL=np.array(AMPS['WAL']), amps_PSS=np.array(AMPS['PSS']),
                        amps_LIQ=np.array(AMPS['LIQ']), d_half=json.dumps(D_HALF), n_d=N_D,
                        match_ns=json.dumps(MATCH_NS), methods=np.array(METHODS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-bunches', type=int)
    ap.add_argument('--dets', help='comma-separated det indices (test)')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args(argv)
    H = empty()
    n = 0
    if a.merge:
        for p in a.inputs:
            z = np.load(p)
            for k in ('N', 'NF', 'NP', 'NC', 'NFC', 'R', 'RO', 'D', 'n_bunch', 'n_trace'):
                H[k] = H[k] + z[k]
            n += int(z['n_files'])
    else:
        rng = np.random.default_rng(a.seed)
        dets = {int(x) for x in a.dets.split(',')} if a.dets else None
        for p in a.inputs:
            inject_file(p, H, rng, a.max_bunches, dets)
            n += 1
    save(H, a.out, n)
    print(f'{n} files, {H["n_bunch"]} bunches (par, ded), {H["n_trace"]} traces -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
