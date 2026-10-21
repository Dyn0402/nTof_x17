#!/usr/bin/env python3
"""
mip_eff.py -- how often a minimum-ionizing particle (an X17 lepton) is lost, against time since the flash.

    python -m ntof_scint_recovery.mip_eff

A MIP does not make one pulse size: it makes a Landau distribution of sizes.  So instead of quoting the
efficiency per injected size, the per-size efficiency is folded with the MEASURED MIP spectrum of each counter:

* **MIP spectrum** (data): the 30-32 ms window of every bunch (no beam, cosmic muons), at the late-time gain, in
  mV.  Plastics: pulses with a wall hit of the same arm in coincidence (accidentals subtracted), per bar; the
  through-going peak is fitted (Landau x Gaussian, ``through_going``) and the low shoulder (15-33 % of the
  coincidences: tracks clipping the bar, shower debris) is left out, since an X17 lepton crosses the whole bar.
  Walls: the two ends of a bar in coincidence; the per-end size is the geometric mean of the two ends times
  exp(+-ln(A1/A2)/2), sampled from the measured distributions (all positions along the bar).
* **gain** (data, gain_ref.py / early.gain_model): a MIP at gain g makes g times the pulse.  Plastics before
  2 µs: not measured; bounded by the fine-grid q99 (below) and, for dedicated bunches, by the dedicated/parasitic
  scale of linearity.py (above, since the parasitic gain is <= 1), else 1.
* **per-size efficiency** (injection, early.py): ``found`` (the pulse is seen) and ``reco`` (seen and within
  +-50 %), interpolated in log size; this is the algorithm and the pile-up with real beam pulses, assuming
  linear superposition (tested on real pulses in linearity.py).

Outputs out_dir('early'): ``mip_eff.csv`` (fam, cls, t, efficiencies at the central gain and at the two ends of
its range, and the fraction of MIPs above the 2026 trigger discriminator), ``mip_landau.csv`` (the MIP spectra,
for the slides), ``mip_counters.csv`` (median MIP pulse per counter).  Liquids are not on the lepton path and have no MIP reference: left out.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import PLAS_THR, WALL_THR, out_dir
from ntof_scint_recovery.early import GAIN_FROM, gain_model

N_SAMPLE = 200_000
RNG = np.random.default_rng(17)
X_EDGES = np.linspace(0, 4, 81)


def _sample(h, edges, n):
    """Draw n values from a histogram (log-uniform inside each bin)."""
    h = np.clip(h, 0, None)
    p = h / h.sum()
    i = RNG.choice(len(h), size=n, p=p)
    lo, hi = np.log(edges[i]), np.log(edges[i + 1])
    return np.exp(lo + RNG.random(n) * (hi - lo))


def _median(h, edges):
    c = np.cumsum(np.clip(h, 0, None))
    return float(np.interp(0.5, np.r_[0, c / c[-1]], edges))


def through_going(h, edges):
    """Fit the through-going-muon peak of a plastic coincidence spectrum (bins above 0.6 x the mode) with a
    Landau convolved with a Gaussian; the low shoulder (tracks clipping the bar, shower debris) is left out.
    Returns (amplitude grid in mV, CDF of the fitted spectrum)."""
    from scipy.optimize import least_squares
    from scipy.stats import landau, norm
    ac = np.sqrt(edges[1:] * edges[:-1])
    grid = np.arange(0.5, 3000, 1.0)

    def cdf(p):
        loc, sc, sig = p
        k = norm.pdf(np.arange(-300, 301, 1.0), 0, sig)
        c = np.r_[0, np.cumsum(np.convolve(landau.pdf(grid, loc, sc), k / k.sum(), 'same'))]
        return c / c[-1]

    h = np.clip(h, 0, None)
    mode = ac[np.argmax(h)]
    m = ac >= 0.6 * mode

    def res(p):
        mm = np.diff(np.interp(edges, np.r_[0, grid + 0.5], cdf(p)))
        nn = h[m].sum() / mm[m].sum()
        return (h[m] - nn * mm[m]) / np.sqrt(np.maximum(nn * mm[m], 1))
    r = least_squares(res, [mode, mode * 0.1, mode * 0.1], bounds=([1, 1, 1], [3000, 1000, 500]))
    return np.r_[0, grid + 0.5], cdf(r.x)


def mip_samples(H):
    """Per family: ``a`` the MIP pulse in mV at the late-time gain (one sample per MIP), ``a_mip`` its counter's
    median, ``arm``; walls also ``a_sum`` (both ends summed, what the trigger sees); plastics also ``a_all``
    (every coincidence, shoulder included, for the slides)."""
    ae, le = H['a_edges'], H['lr_edges']
    out = {}
    net = (H['PW_sig'] - H['pw_side_scale'] * H['PW_side'])[:, :, :, -1].sum(axis=0)       # arm, bar, amp
    a_, am, arm_, al, aml = [], [], [], [], []
    for arm in range(4):
        for bar in range(2):
            n = N_SAMPLE // 8
            g, c = through_going(net[arm, bar], ae)
            a = np.interp(RNG.random(n), c, g)
            med = float(np.interp(0.5, c, g))
            a_.append(a)
            am.append(np.full(n, med))
            arm_.append(np.full(n, arm))
            al.append(_sample(net[arm, bar], ae, n))
            aml.append(np.full(n, med))
    out['PSS'] = dict(a=np.concatenate(a_), a_mip=np.concatenate(am), arm=np.concatenate(arm_),
                      a_all=np.concatenate(al), a_mip_all=np.concatenate(aml))
    G = (H['BAR_g'] - H['BAR_g_acc'])[:, :, :, -1].sum(axis=(0, 3))                       # arm, bar, amp
    R = (H['BAR_r'] - H['BAR_r_acc'])[:, :, :, -1].sum(axis=(0, 3))                       # arm, bar, lr
    a_, asum, am, arm_ = [], [], [], []
    for arm in range(4):
        for g in range(4):
            if np.clip(G[arm, g], 0, None).sum() < 300:
                continue
            n = N_SAMPLE // 16
            gm = _sample(G[arm, g], ae, n)
            pr = np.clip(R[arm, g], 0, None)
            lr = RNG.choice(0.5 * (le[1:] + le[:-1]), size=n, p=pr / pr.sum())
            sgn = RNG.choice((-1, 1), size=n)
            a = gm * np.exp(sgn * lr / 2)                                  # one end
            a_.append(a)
            asum.append(gm * (np.exp(lr / 2) + np.exp(-lr / 2)))           # both ends summed (the trigger)
            am.append(np.full(n, np.median(a)))
            arm_.append(np.full(n, arm))
    out['WAL'] = dict(a=np.concatenate(a_), a_sum=np.concatenate(asum), a_mip=np.concatenate(am),
                      arm=np.concatenate(arm_))
    return out


def landau_table(S) -> pd.DataFrame:
    """The MIP spectra in units of each counter's median MIP pulse: walls as measured; plastics the fitted
    through-going spectrum and every coincidence (the shoulder below is not a through-going MIP)."""
    rows = []
    xc = 0.5 * (X_EDGES[1:] + X_EDGES[:-1])
    for fam, s in S.items():
        for kind, a, m in (('mip', s['a'], s['a_mip']),) + ((('all', s['a_all'], s['a_mip_all']),) if fam == 'PSS' else ()):
            h, _ = np.histogram(a / m, X_EDGES)
            h = h / len(a) / np.diff(X_EDGES)
            rows += [dict(fam=fam, kind=kind, x=x, density=d) for x, d in zip(xc, h)]
    return pd.DataFrame(rows)


def eff_curve(T, fam, cls, t, col):
    """(log sizes, efficiencies) of the injection at time bin t, made monotonic in size."""
    g = T[(T.fam == fam) & (T.cls == cls) & (T.t_us == t)].sort_values('amp_mV')
    return np.log(g.amp_mV.to_numpy()), np.maximum.accumulate(g[col].to_numpy())


def fold(la, e, a):
    """Efficiency at sizes a: log-linear interpolation; below the smallest injected size the slope of the two
    smallest is extended (clipped at 0), above the largest the largest is kept."""
    x = np.log(np.maximum(a, 1e-3))
    y = np.interp(x, la, e)
    lo = x < la[0]
    if lo.any():
        k = (e[1] - e[0]) / (la[1] - la[0])
        y[lo] = np.clip(e[0] + k * (x[lo] - la[0]), 0, 1)
    return y


def gain_range(cls, fam, t, g, u, Sc, F):
    """(central, low, high) gain at time t."""
    gt = g(t)
    if np.isfinite(gt):
        ut = u(t)
        return gt, max(gt - ut, 0.05), min(gt + ut, 1.0) if fam == 'PSS' else gt + ut
    if fam == 'WAL':
        return 1.0, 1.0, 1.0
    def q99(c):
        f = F[(F.fam == fam) & (F.cls == c)].sort_values('t_us')
        return float(np.interp(np.log(t), np.log(f.t_us), f.q99))
    # q99 is a lower bound: the fast-neutron recoil spectrum is softer than the capture-gamma reference
    lo, hi = q99(cls), 1.0
    if cls == 'ded':
        # g_ded = scale x g_par (same particles, linearity.py), with q99(par) <= g_par <= 1
        s = Sc.sort_values('t_us')
        sc = float(np.interp(np.log(t), np.log(s.t_us), s.scale))
        hi = min(sc, 1.0)
        lo = max(lo, sc * q99('par'))
    lo = min(lo, hi)
    return np.sqrt(lo * hi), lo, hi


def linear_time(LF, fam, cls):
    """When the real-pulse tests say the response is linear: plastics the dedicated/parasitic shape test, walls
    that and the bar ends (whichever is later)."""
    t = LF[(LF.fam == fam) & (LF.test == 'flash')].t_from_us.max()
    if fam == 'WAL':
        t = max(t, LF[(LF.fam == fam) & (LF.test == 'bar ends') & (LF.cls == cls)].t_from_us.max())
    return float(t)


def first_times(M, LF, t_end=40.0) -> pd.DataFrame:
    """First time from which every bin to t_end passes: MIP seen >= 90 % (at the central gain and at the low end
    of its range), measured within +-50 % >= 90 %; and the linearity time; usable = the later of linear and seen
    (central gain)."""
    rows = []
    for (fam, cls), g in M.groupby(['fam', 'cls']):
        g = g[(g.t_us <= t_end) & (g.t_us > 0.3)].sort_values('t_us')
        out = dict(fam=fam, cls=cls, linear_from_us=linear_time(LF, fam, cls))
        for name, ok in (('seen90', g.found >= 0.9), ('seen90_lo', g.found_lo >= 0.9), ('reco90', g.reco >= 0.9),
                         ('gain_from', g.gain_measured)):
            ok = ok.to_numpy()
            bad = np.nonzero(~ok)[0]
            out[f'{name}_us'] = (float(g.t_lo_us.iloc[0]) if not len(bad) else
                                 np.nan if bad[-1] == len(ok) - 1 else float(g.t_lo_us.iloc[bad[-1] + 1]))
        a, b = out['linear_from_us'], out['seen90_us']
        out['usable_from_us'] = max(a, b) if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> int:
    from ntof_scint_recovery.coinc_analysis import load
    H = load(out_dir('pulsehist_v3'))
    S = mip_samples(H)
    T = pd.read_csv(out_dir('early') / 'early.csv')
    F = pd.read_csv(out_dir('gain') / 'gain_fine.csv')
    L = pd.read_csv(out_dir('linearity') / 'shape_tests.csv')
    Sc = L[(L.fam == 'PSS') & (L.test == 'flash') & (L.n_ch >= 3)]
    rows = []
    for fam in ('PSS', 'WAL'):
        s = S[fam]
        thr = np.array([PLAS_THR['ABCD'[a]] if fam == 'PSS' else WALL_THR['ABCD'[a]] for a in range(4)])[s['arm']]
        for cls in ('ded', 'par'):
            g, u = gain_model(cls, fam)
            for t in sorted(T[(T.fam == fam) & (T.cls == cls)].t_us.unique()):
                gc, glo, ghi = gain_range(cls, fam, t, g, u, Sc, F)
                r = dict(fam=fam, cls=cls, t_us=t, t_lo_us=T[(T.t_us == t)].t_lo_us.iloc[0],
                         t_hi_us=T[(T.t_us == t)].t_hi_us.iloc[0], gain=gc, gain_lo=glo, gain_hi=ghi,
                         gain_measured=bool(np.isfinite(g(t))))
                for col, name in (('eff_found', 'found'), ('eff', 'reco')):
                    la, e = eff_curve(T, fam, cls, t, col)
                    for gg, tag in ((gc, ''), (glo, '_lo'), (ghi, '_hi')):
                        r[f'{name}{tag}'] = float(fold(la, e, s['a'] * gg).mean())
                # amplitude above the 2026 discriminator (plastic: the bar; wall: the sum of the two ends)
                amp = s['a'] if fam == 'PSS' else s['a_sum']
                for gg, tag in ((gc, ''), (glo, '_lo'), (ghi, '_hi')):
                    r[f'trig{tag}'] = float((amp * gg >= thr).mean())
                r['below_grid'] = float((s['a'] * gc < np.exp(la[0])).mean())
                # the median MIP: its pulse is gain x the family's MIP amplitude; resolution, timing and the
                # reconstruction response interpolated in log size to it
                g_ = T[(T.fam == fam) & (T.cls == cls) & (T.t_us == t)].sort_values('amp_mV')
                a_mip = float(np.mean(s['a_mip'])) * gc
                r['mip_mV'] = a_mip
                for c in ('resp', 'sig_res', 'sig_t_ns', 'eff', 'eff_found'):
                    r[f'mip_{c}'] = float(np.interp(np.log(a_mip), np.log(g_.amp_mV), g_[c]))
                # correction uncertainty (gain): the same for every size; plus the gain range where unmeasured
                r['mip_sig_sys'] = float(g_.sig_sys.iloc[0]) if r['gain_measured'] else (ghi - glo) / 2 / gc
                r['mip_sig_E'] = float(np.hypot(r['mip_sig_res'], r['mip_sig_sys']))
                rows.append(r)
    M = pd.DataFrame(rows)
    o = out_dir('early')
    M.to_csv(o / 'mip_eff.csv', index=False)
    first_times(M, pd.read_csv(out_dir('linearity') / 'linear_from.csv')).to_csv(o / 'mip_first.csv', index=False)
    landau_table(S).to_csv(o / 'mip_landau.csv', index=False)
    pd.DataFrame([dict(fam=f, a_mip_mV=v) for f in S for v in np.unique(S[f]['a_mip'])]).to_csv(o / 'mip_counters.csv',
                                                                                              index=False)
    pd.set_option('display.width', 250)
    print(M[(M.t_us < 40)].round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
