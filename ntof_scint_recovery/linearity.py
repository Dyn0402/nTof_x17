#!/usr/bin/env python3
"""
linearity.py -- is the detector + electronics still linear after the flash?  Tested on REAL pulses only.

    python -m ntof_scint_recovery.linearity

The injection study (inject.py) adds pulses to the recorded traces, so it assumes linear superposition: it
measures the algorithm and the pile-up with real beam pulses, never a non-linear response of the PMT, SiPM,
amplifier or digitizer to a pulse riding on the flash tail.  Here only real pulses are used.  A response that
is linear up to a gain g turns an amplitude spectrum into the same spectrum with every amplitude times g; a
non-linear response (a compression of large pulses, a baseline-dependent gain) changes its SHAPE.  So per
channel and time bin, the template-fit amplitudes of isolated pulses (pulsehist ``HF``, above ``CUT``) are fitted
as a scaled copy of a reference spectrum, with the cut applied consistently (the model quantile at level p is
g x Q_ref(p | A > CUT / g)), and the shape residuals at q50, q75, q90, q95, q99 are kept.

Two references, each blind to a different confusion:

* ``flash``: dedicated against parasitic bunches at the same time since the flash.  The particles per proton
  are the same, only the flash is twice as large, so a flash-driven non-linearity changes the shape while a
  change of particle mix cannot.  Works from the first bin where both have pulses.  The scale is the ratio of
  the two gains.
* ``time``: the same intensity against its own 20-60 µs spectrum (capture-gamma e+-, gain back).  Valid only
  once the particle mix is the same as in the reference: from ~2-3 µs (Geant4, g4_mix.py); before that the
  fast-neutron recoils change the shape and the test cannot separate mix from non-linearity.

Plus the wall bars, whose two ends see the same light: at a fixed position along the bar (fixed end-to-end time
difference) the end amplitude ratio must not move, and at a fixed amplitude ratio the time difference must keep
its position and width.  This is a differential test of amplitude and timing on real pulses.

Writes out_dir('linearity'): ``shape_tests.csv``, ``bar_ends.csv``, ``linear_from.csv``, ``shape_examples.csv``.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir

P = np.array((0.5, 0.75, 0.9, 0.95, 0.99))
CUT = {'PSS': 20.0, 'LIQ': 20.0, 'WAL': 15.0}
DETS = {'WAL': range(0, 4), 'PSS': range(4, 8), 'LIQ': range(8, 12)}
REF_US = (20.0, 60.0)
T_MAX = 60.0
#: a scaled copy within this (max |residual| over q50..q99, median over channels) counts as linear
TOL = 0.10
MIN_COUNTS = 200
MIN_CH = 3
#: fine pulsehist bins (x1.12) are merged in pairs (x1.26, the injection grid) for statistics
MERGE = 2
G_GRID = np.exp(np.linspace(np.log(0.05), np.log(3.0), 800))


def _cdf(h):
    c = np.r_[0, np.cumsum(np.clip(h, 0, None))]
    return c / c[-1]


def scaled_copy(h, r, edges, cut):
    """Fit h as r scaled by g, both above ``cut``.  Returns (g, residuals q_obs/q_model - 1, q_obs) or None."""
    le = np.log(edges)
    Fh = _cdf(h)
    fo = np.interp(np.log(cut), le, Fh)
    qo = np.exp(np.interp(fo + P * (1 - fo), Fh, le))
    Fr = _cdf(r)
    best = None
    for g in G_GRID:
        fr = np.interp(np.log(cut / g), le, Fr)
        if fr > 0.999:
            continue
        qm = g * np.exp(np.interp(fr + P * (1 - fr), Fr, le))
        d = np.log(qo / qm)
        s = float((d ** 2).sum())
        if best is None or s < best[0]:
            best = (s, g, np.exp(d) - 1, qo)
    return None if best is None else best[1:]


def shape_tests(H) -> pd.DataFrame:
    ae, te0 = H['a_edges'], H['t_edges_us']
    nk = (len(te0) - 1) // MERGE
    te = te0[:nk * MERGE + 1:MERGE]
    HF = H['HF'][:, :, :nk * MERGE].reshape(*H['HF'].shape[:2], nk, MERGE, -1).sum(axis=3)
    tf = np.sqrt(te[1:] * te[:-1])
    chans = [tuple(c) for c in H['channels']]
    ref = (tf > REF_US[0]) & (tf < REF_US[1])
    rows = []
    for fam, dets in DETS.items():
        idx = [i for i, c in enumerate(chans) if c[0] in dets and HF[:, i].sum() > 1e4]
        cut = CUT[fam]
        above = ae[:-1] >= cut
        for test, pairs in (('flash', ((1, 0),)), ('time', ((1, 1), (0, 0)))):
            for ca, cb in pairs:
                cls = 'ded' if ca == 1 else 'par'
                for k in range(len(tf)):
                    if not (0.3 < tf[k] < T_MAX):
                        continue
                    res, gs, qs = [], [], []
                    for i in idx:
                        h = HF[ca, i, k].astype(float)
                        r = (HF[cb, i, k] if test == 'flash' else HF[cb, i][ref].sum(0)).astype(float)
                        if h[above].sum() < MIN_COUNTS or r[above].sum() < MIN_COUNTS:
                            continue
                        b = scaled_copy(h, r, ae, cut)
                        if b is None:
                            continue
                        gs.append(b[0])
                        res.append(b[1])
                        qs.append(b[2])
                    if not res:
                        continue
                    R = np.median(np.array(res), axis=0)
                    Qm = np.median(np.array(qs), axis=0)
                    rows.append(dict(fam=fam, test=test, cls=cls if test == 'time' else 'ded/par', t_us=tf[k],
                                     t_lo_us=te[k], t_hi_us=te[k + 1], n_ch=len(res), n_ch_all=len(idx),
                                     scale=float(np.median(gs)),
                                     **{f'r{int(p * 100)}': v for p, v in zip(P, R)},
                                     max_abs=float(np.abs(R).max()), q50_mV=Qm[0], q99_mV=Qm[-1]))
    return pd.DataFrame(rows)


def bar_ends(H) -> pd.DataFrame:
    """Wall bar ends on real coincident pulses: ln(A1/A2) at a fixed position (|dt - dt_ref| <= 3 ns) and the
    dt position and width at a fixed amplitude ratio (|ln r - ln r_ref| <= 0.15), each against 20-60 µs."""
    tce, dte, le = H['tc_edges_us'], H['dt_edges'], H['lr_edges']
    tcc = np.sqrt(tce[1:] * tce[:-1])
    dc, lc = 0.5 * (dte[1:] + dte[:-1]), 0.5 * (le[1:] + le[:-1])
    ref = (tcc > REF_US[0]) & (tcc < REF_US[1])

    def q(h, e, p):
        c = np.cumsum(np.clip(h, 0, None))
        if c[-1] < 30:
            return np.nan
        return float(np.interp(p, np.r_[0, c / c[-1]], e))

    def s68(h):
        return (q(h, dte, 0.84) - q(h, dte, 0.16)) / 2

    rows = []
    for ci, cls in ((0, 'par'), (1, 'ded')):
        R = (H['BAR_r'] - H['BAR_r_acc'])[ci]                # arm, bar, t, dt, lr
        for k in range(len(tcc)):
            if not (0.3 < tcc[k] < 1e4):
                continue
            out, n = [], 0.0
            for a in range(4):
                for g in range(4):
                    h = R[a, g]
                    if h.sum() < 1e4:
                        continue
                    pl = h[ref].sum(0)
                    sl = np.abs(dc - q(pl.sum(1), dte, 0.5)) <= 3
                    ls = np.abs(lc - q(pl.sum(0), le, 0.5)) <= 0.15
                    n += np.clip(h[k], 0, None).sum()
                    out.append((q(h[k][sl].sum(0), le, 0.5) - q(pl[sl].sum(0), le, 0.5),
                                q(h[k][:, ls].sum(1), dte, 0.5) - q(pl[:, ls].sum(1), dte, 0.5),
                                s68(h[k][:, ls].sum(1)), s68(pl[:, ls].sum(1))))
            if not out:
                continue
            o = np.array(out)
            rows.append(dict(cls=cls, t_us=tcc[k], t_lo_us=tce[k], t_hi_us=tce[k + 1], n_pairs=n, n_bars=len(o),
                             dlnr=np.nanmedian(o[:, 0]), dlnr_spread=np.nanstd(o[:, 0]),
                             ddt_ns=np.nanmedian(o[:, 1]), dt68_ns=np.nanmedian(o[:, 2]), dt68_ref_ns=np.nanmedian(o[:, 3])))
    return pd.DataFrame(rows)


def linear_from(S: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    """First time from which every later bin to T_MAX passes (shape tests: max |residual| <= TOL with at least
    MIN_CH channels; bar ends: |d ln r| <= 0.05, |d dt| <= 1 ns, dt68 <= 1.5 x reference)."""
    def first(t_lo, ok):
        ok = np.asarray(ok)
        bad = np.nonzero(~ok)[0]
        if not len(bad):
            return float(t_lo[0])
        return np.nan if bad[-1] == len(ok) - 1 else float(t_lo[bad[-1] + 1])

    rows = []
    for (fam, test, cls), g in S.groupby(['fam', 'test', 'cls']):
        g = g.sort_values('t_us')
        g = g[g.n_ch >= min(MIN_CH, g.n_ch_all.iloc[0])]
        rows.append(dict(fam=fam, test=test, cls=cls, t_from_us=first(g.t_lo_us.to_numpy(), g.max_abs <= TOL),
                         typical_abs=float(g[g.t_us > 10].max_abs.median())))
    for cls, g in B.groupby('cls'):
        g = g[g.t_us < T_MAX].sort_values('t_us')
        ok = (g.dlnr.abs() <= 0.05) & (g.ddt_ns.abs() <= 1.0) & (g.dt68_ns <= 1.5 * g.dt68_ref_ns)
        rows.append(dict(fam='WAL', test='bar ends', cls=cls, t_from_us=first(g.t_lo_us.to_numpy(), ok),
                         typical_abs=float(g[g.t_us > 10].dlnr.abs().median())))
    return pd.DataFrame(rows)


#: spectra drawn on the slides: (family, time bin centre in µs)
EXAMPLES = (('PSS', 0.56), ('WAL', 1.78), ('WAL', 3.55))


def examples(H) -> pd.DataFrame:
    """Dedicated and parasitic spectra of one channel (the one with the most pulses) in one time bin, above the
    cut, the parasitic one with every amplitude times the fitted scale: a linear response makes them lie on
    each other.  One channel, since the scale differs from channel to channel."""
    ae, te0 = H['a_edges'], H['t_edges_us']
    nk = (len(te0) - 1) // MERGE
    te = te0[:nk * MERGE + 1:MERGE]
    tf = np.sqrt(te[1:] * te[:-1])
    HF = H['HF'][:, :, :nk * MERGE].reshape(*H['HF'].shape[:2], nk, MERGE, -1).sum(axis=3)
    chans = [tuple(c) for c in H['channels']]
    ac = np.sqrt(ae[1:] * ae[:-1])
    rows = []
    for fam, t in EXAMPLES:
        k = int(np.argmin(np.abs(np.log(tf / t))))
        idx = [i for i, c in enumerate(chans) if c[0] in DETS[fam]]
        i = max(idx, key=lambda j: min(HF[1, j, k][ac >= CUT[fam]].sum(), HF[0, j, k][ac >= CUT[fam]].sum()))
        d, q = HF[1, i, k].astype(float), HF[0, i, k].astype(float)
        sc, res, _ = scaled_copy(d, q, ae, CUT[fam])
        m, mp = ac >= CUT[fam], ac * sc >= CUT[fam]
        base = dict(fam=fam, t_us=tf[k], channel=f'{chans[i][0]}:{chans[i][1]}', scale=sc, max_abs=float(np.abs(res).max()),
                    n_ded=d[m].sum(), n_par=q[mp].sum())
        rows += [dict(base, cls='ded', a_mV=a, frac=v) for a, v in zip(ac[m], d[m] / d[m].sum())]
        rows += [dict(base, cls='par', a_mV=a * sc, frac=v) for a, v in zip(ac[mp], q[mp] / q[mp].sum())]
    return pd.DataFrame(rows)


def main() -> int:
    from ntof_scint_recovery.coinc_analysis import load
    H = load(out_dir('pulsehist_v3'))
    S = shape_tests(H)
    B = bar_ends(H)
    L = linear_from(S, B)
    o = out_dir('linearity')
    S.to_csv(o / 'shape_tests.csv', index=False)
    B.to_csv(o / 'bar_ends.csv', index=False)
    L.to_csv(o / 'linear_from.csv', index=False)
    examples(H).to_csv(o / 'shape_examples.csv', index=False)
    pd.set_option('display.width', 250)
    print(L.round(3).to_string(index=False))
    print(B[B.t_us < 60].round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
