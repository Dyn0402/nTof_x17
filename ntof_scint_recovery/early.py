#!/usr/bin/env python3
"""
early.py -- the short-term answer: the earliest time after the γ-flash at which a scintillator
pulse can be reliably reconstructed, the corrections it then needs, and what is left after
correcting: energy and time uncertainty, and the efficiency for small signals.

    python -m ntof_scint_recovery.early
    python -m ntof_scint_recovery.early --examples <a local flash_<run>_<idx>.root>   # traces for the slides

Inputs: the merged injection histograms (out_dir('inject')/inject.npz, multi-pulse fit), the
fine-grid gain (out_dir('gain')/gain_fine.csv) and coarse gain populations (gain_quantiles.csv),
and the energy scale (headline.energy_scale).

Per family, intensity class, injected amplitude and time bin (pooled over live channels):

* ``eff``        the injected pulse is found AND its amplitude is within ±50 % of the pre-flash response:
                 the pulse is "reconstructed" (a detection that a correction can turn into an energy);
* ``resp``       median A_fit/A over its pre-flash value: the reconstruction response to correct, with
                 its statistical error;
* ``gain``       the photodetector gain (data, gain_ref.py), the second response to correct; plastics from
                 the fine-grid q99 of the singles (capped at 1 where the mix lifts it above), with an
                 uncertainty = |coincident - singles| / 2 (coarse grid) ⊕ 3 % floor; walls 1 ± |q99 - 1|
                 (no sag resolved); liquids: no reference (reported as unknown);
* ``sig_res``    σ68(A_fit/A)/median: the resolution, pile-up included (does not shrink with a correction);
* ``sig_sys``    the residual after correcting resp x gain: stat ⊕ gain uncertainty ⊕ half the change of the
                 correction across the time bin;
* ``sig_E``      sig_res ⊕ sig_sys, times the energy: the energy uncertainty after correction;
* ``t_bias``, ``sig_t``  median and σ68 of t_fit - t0 (ns), the median being a correctable offset;
* ``thr50/90``   the amplitude (and energy, gain included) at which eff reaches 50 / 90 %: the smallest
                 signal kept.

The **earliest reliable time** ``t_first`` (per family, class and reference amplitude): the first time bin
from which every later bin up to ``T_HORIZON`` passes ``CRIT``: eff ≥ 0.9, σ68(A)/A ≤ 0.25 and σ68(t) below
1 ns (plastics, liquids) / 10 ns (walls), and the gain known (plastics: t ≥ 2 µs).  No requirement on the size
of the bias: it is corrected.

Writes out_dir('early'): ``early.csv``, ``first.csv``, ``table.csv``.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.headline import WALL_MIP_MEV, energy_scale

FAM_DET = {'WAL': range(0, 4), 'PSS': range(4, 8), 'LIQ': range(8, 12)}
METHOD = 'multi'
T_HORIZON = 50.0
CRIT = dict(eff=0.9, sig_res=0.25, sig_t={'WAL': 10.0, 'PSS': 1.0, 'LIQ': 1.0})
GAIN_FROM = {'PSS': 2.0, 'WAL': 2.0, 'LIQ': np.inf}
#: plots and tables start after the hardware blind time: walls switched off (divert) to ~0.65 µs
T_START = {'WAL': 0.65, 'PSS': 0.1, 'LIQ': 0.1}
TABLE_T = (1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20)


def quant(h, edges, qs):
    c = np.cumsum(h)
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    return np.interp(qs, np.r_[0, c / c[-1]], edges)


def gain_model(cls: str, fam: str):
    """(t -> gain, t -> uncertainty) for the family; NaN where unknown."""
    F = pd.read_csv(out_dir('gain') / 'gain_fine.csv')
    Q = pd.read_csv(out_dir('gain') / 'gain_quantiles.csv')
    f = F[(F.fam == fam) & (F.cls == cls) & np.isfinite(F.q99) & (F.t_us > 0.5) & (F.t_us < 5e3)].sort_values('t_us')
    lt = np.log(f.t_us.to_numpy())
    q = f.q99.to_numpy()
    # light smoothing over 3 fine bins (x1.12 each) against statistical jitter
    qs = np.convolve(np.r_[q[0], q, q[-1]], np.ones(3) / 3, 'valid')
    if fam == 'PSS':
        a = Q[(Q['pop'] == 'plastic|wall') & (Q.cls == cls)].sort_values('t_us')
        b = Q[(Q['pop'] == 'plastic singles') & (Q.cls == cls)].sort_values('t_us')
        dlt = np.log(a.t_us.to_numpy())
        dd = np.abs(a.q99.to_numpy() - b.q99.to_numpy()) / 2
        ok = np.isfinite(dd)

        def g(t):
            if t < GAIN_FROM[fam]:
                return np.nan
            return float(min(np.interp(np.log(t), lt, qs), 1.0))

        def u(t):
            if t < GAIN_FROM[fam]:
                return np.nan
            return float(np.hypot(np.interp(np.log(t), dlt[ok], dd[ok]), 0.03))
        return g, u
    if fam == 'WAL':
        def g(t):
            return 1.0 if t >= GAIN_FROM[fam] else np.nan

        def u(t):
            if t < GAIN_FROM[fam]:
                return np.nan
            return float(max(abs(np.interp(np.log(t), lt, qs) - 1), 0.03))
        return g, u
    return (lambda t: np.nan), (lambda t: np.nan)


def table(z) -> pd.DataFrame:
    ch = z['channels']
    te = z['t_edges_us']
    re = z['r_edges']
    rc = 0.5 * (re[1:] + re[:-1])
    dh = json.loads(str(z['d_half']))
    nd = int(z['n_d'])
    im = list(z['methods']).index(METHOD)
    live = z['N'].sum(axis=(0, 2, 3)) > 0
    ES = energy_scale()
    rows = []
    for fam in ('WAL', 'PSS', 'LIQ'):
        cs = np.array([i for i, (d, _) in enumerate(ch) if d in FAM_DET[fam] and live[i]])
        amps = z[f'amps_{fam}']
        de = np.linspace(-dh[fam], dh[fam], nd + 1)
        scale = ES['pss_mean'] if fam == 'PSS' else (ES['wal_mean'] / WALL_MIP_MEV if fam == 'WAL' else np.nan)
        for ci, cls in enumerate(('par', 'ded')):
            g, u = gain_model(cls, fam)
            for ia, A in enumerate(amps):
                N = z['N'][ci, cs, ia].sum(axis=0)
                R = z['R'][im, ci, cs, ia].sum(axis=(0, 2))               # (t, r)
                D = z['D'][im, ci, cs, ia].sum(axis=(0, 2))
                r16c, r50c, r84c = quant(R[0], re, (0.16, 0.5, 0.84))
                d50c = quant(D[0], de, (0.5,))[0]
                for k in range(1, len(te)):
                    if N[k] < 30:
                        continue
                    t_lo, t_hi = te[k - 1], te[k]
                    t = np.sqrt(t_lo * t_hi)
                    if t < T_START[fam]:
                        continue
                    good = R[k][np.abs(rc / r50c - 1) <= 0.5].sum()
                    r16, r50, r84 = quant(R[k], re, (0.16, 0.5, 0.84))
                    d16, d50, d84 = quant(D[k], de, (0.16, 0.5, 0.84))
                    nfit = R[k].sum()
                    resp = r50 / r50c
                    sig_res = (r84 - r16) / 2 / r50 if r50 > 0 else np.nan
                    stat = 1.2533 * sig_res / np.sqrt(max(nfit, 1))
                    gt, ut = g(t), u(t)
                    dg = abs(g(t * 1.27) - g(t / 1.27)) / 4 if np.isfinite(gt) else np.nan
                    sig_sys = np.sqrt(stat ** 2 + ut ** 2 + (0 if not np.isfinite(dg) else dg) ** 2) if np.isfinite(ut) else np.nan
                    sig_E_rel = np.hypot(sig_res, sig_sys) if np.isfinite(sig_sys) else np.nan
                    E = A / scale if np.isfinite(scale) else np.nan
                    rows.append(dict(fam=fam, cls=cls, amp_mV=A, E_true=E, t_lo_us=t_lo, t_hi_us=t_hi, t_us=t,
                                     n_inj=N[k], eff=good / N[k], eff_found=z['NF'][im, ci, cs, ia, k].sum() / N[k],
                                     resp=resp, resp_stat=stat, gain=gt, gain_unc=ut, gain_interp=dg,
                                     resp_total=resp * gt if np.isfinite(gt) else np.nan,
                                     sig_res=sig_res, sig_sys=sig_sys, sig_E_rel=sig_E_rel,
                                     sig_E=sig_E_rel * E if np.isfinite(E) else np.nan,
                                     t_bias_ns=d50 - d50c, sig_t_ns=(d84 - d16) / 2,
                                     ctrl_sig_res=(r84c - r16c) / 2 / r50c))
    return pd.DataFrame(rows)


def thresholds(T: pd.DataFrame) -> pd.DataFrame:
    """Smallest amplitude kept at 50 / 90 % efficiency, per family, class, time bin (log-interpolated)."""
    ES = energy_scale()
    rows = []
    for (fam, cls, k), g in T.groupby(['fam', 'cls', 't_us']):
        g = g.sort_values('amp_mV')
        a = np.log(g.amp_mV.to_numpy())
        e = np.maximum.accumulate(g.eff.to_numpy())          # efficiency rises with amplitude
        out = {}
        for x in (0.5, 0.9):
            if e[0] >= x:
                v, flag = g.amp_mV.iloc[0], '≤'
            elif e[-1] < x:
                v, flag = g.amp_mV.iloc[-1], '>'
            else:
                j = int(np.argmax(e >= x))
                v = float(np.exp(np.interp(x, [e[j - 1], e[j]], [a[j - 1], a[j]])))
                flag = ''
            out[f'thr{int(x * 100)}_mV'] = v
            out[f'thr{int(x * 100)}_flag'] = flag
        gn = g.gain.iloc[0]
        scale = ES['pss_mean'] if fam == 'PSS' else (ES['wal_mean'] / WALL_MIP_MEV if fam == 'WAL' else np.nan)
        for x in (50, 90):
            # an energy deposit E gives gain x scale x E mV: the energy threshold rises as the gain sags
            out[f'thr{x}_E'] = out[f'thr{x}_mV'] / (scale * (gn if np.isfinite(gn) else 1.0)) if np.isfinite(scale) else np.nan
        rows.append(dict(fam=fam, cls=cls, t_us=k, t_lo_us=g.t_lo_us.iloc[0], t_hi_us=g.t_hi_us.iloc[0], gain=gn, **out))
    return pd.DataFrame(rows)


def first_times(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, cls, A), g in T.groupby(['fam', 'cls', 'amp_mV']):
        g = g[g.t_hi_us <= T_HORIZON * 1.3].sort_values('t_us')
        if not len(g):
            continue
        gain_ok = (g.t_lo_us >= GAIN_FROM[fam]) if fam == 'PSS' else pd.Series(True, index=g.index)
        tests = {
            'reco': (g.eff >= CRIT['eff']) & (g.sig_res <= CRIT['sig_res']) & (g.sig_t_ns <= CRIT['sig_t'][fam]),
            'found90': g.eff >= CRIT['eff'],
            'found50': g.eff >= 0.5,
        }
        tests['reco+gain'] = tests['reco'] & gain_ok
        for name, ok in tests.items():
            ok = ok.to_numpy()
            bad = np.nonzero(~ok)[0]
            if not len(bad):
                t = float(g.t_lo_us.iloc[0])
            elif bad[-1] == len(ok) - 1:
                t = np.nan
            else:
                t = float(g.t_lo_us.iloc[bad[-1] + 1])
            rows.append(dict(fam=fam, cls=cls, amp_mV=A, E_true=g.E_true.iloc[0], test=name, t_first_us=t))
    return pd.DataFrame(rows)


def at(T, fam, cls, A, t, col):
    g = T[(T.fam == fam) & (T.cls == cls) & (T.amp_mV == A) & (T.t_lo_us <= t) & (T.t_hi_us > t)]
    return float(g[col].iloc[0]) if len(g) else np.nan


def main() -> int:
    z = dict(np.load(out_dir('inject') / 'inject.npz'))
    T = table(z)
    TH = thresholds(T)
    FT = first_times(T)
    o = out_dir('early')
    T.to_csv(o / 'early.csv', index=False)
    TH.to_csv(o / 'thresholds.csv', index=False)
    FT.to_csv(o / 'first.csv', index=False)
    pd.set_option('display.width', 250)
    print(FT.pivot_table(index=['fam', 'amp_mV'], columns=['cls', 'test'], values='t_first_us').round(2).to_string())
    return 0




def examples(path: str, bunch_index: int = 2) -> None:
    """Example traces for the slides: one channel per family, a dedicated bunch, ~1-MIP pulses injected
    every µs from 0.5 µs, reconstructed with the multi-pulse fit.  Writes out_dir('early')/examples.npz
    (trace decimated to min/max per 8 ns, injected times, fitted amplitudes)."""
    import awkward as ak  # noqa: F401
    import uproot
    from ntof_scint_recovery.flash_summary import C_FALLBACK, DET_NAMES
    from ntof_scint_recovery import inject as I
    MV = 0.0306
    f = uproot.open(path)
    T = f['blocks']
    meta = f['bunches'].arrays(['bunch', 'PulseIntensity'], library='np')
    bb = T['bunch'].array(library='np')
    starts = np.r_[0, np.nonzero(np.diff(bb))[0] + 1]
    stops = np.r_[starts[1:], len(bb)]
    ded = [i for i, p in enumerate(meta['PulseIntensity']) if p > 6.5e12]
    e0, e1 = starts[ded[bunch_index]], stops[ded[bunch_index]]
    a = T.arrays(['det', 'detn', 'tof0', 'samples'], entry_start=e0, entry_stop=e1)
    det, detn, tof0 = (np.asarray(a[k]).astype(int) for k in ('det', 'detn', 'tof0'))
    k = np.nonzero((det == 12) & (tof0 == 0))[0][0]
    x = np.asarray(a['samples'][k]).astype(float)
    t_pk = int(np.argmax(x - np.median(x[:5000])))
    out = {}
    for fam, d, ch, A in (('WAL', 0, 3, 30.0), ('PSS', 4, 1, 200.0), ('LIQ', 8, 1, 100.0)):
        s = np.nonzero((det == d) & (detn == ch) & (tof0 == 0))[0][0]
        xh = np.asarray(a['samples'][s]).astype(np.float32)
        pol = 1 if d < 4 else -1
        t_fl = t_pk + C_FALLBACK[DET_NAMES[d]]
        th = (np.arange(len(xh)) - t_fl) / 1e3
        y = pol * (xh - np.median(xh[(th > -9) & (th < -1)])) * MV
        adc = ((32767 - np.median(xh[(th > -9) & (th < -1)])) if pol > 0 else (np.median(xh[(th > -9) & (th < -1)]) + 32768)) * MV
        ymax = float(y.max())
        ysat = min(ymax if np.sum(y > ymax - 2) > 50 else adc, adc)
        t0s = t_fl + np.arange(0.5, 10.0, 1.0) * 1e3 + 0.37
        yi = np.minimum(y + I.comb(len(y), fam, t0s, A), ysat)
        pk = I.reco_multi(yi, fam)
        near = np.array([pk[np.argmin(np.abs(pk - q))] if len(pk) else -1 for q in t0s])
        found = np.abs(near - t0s) <= I.MATCH_NS[fam]
        Af, _ = I.fit_multi(yi, pk, near[found], fam)
        amp = np.full(len(t0s), np.nan)
        amp[found] = Af
        m = (th > -0.5) & (th < 10.5)
        idx = np.nonzero(m)[0]
        n8 = len(idx) // 8 * 8
        yy, yo = yi[idx[:n8]].reshape(-1, 8), y[idx[:n8]].reshape(-1, 8)
        out[f'{fam}_t'] = th[idx[:n8]].reshape(-1, 8)[:, 0]
        out[f'{fam}_max'] = yy.max(axis=1)
        out[f'{fam}_min'] = yy.min(axis=1)
        out[f'{fam}_orig_max'] = yo.max(axis=1)
        out[f'{fam}_inj_t'] = (t0s - t_fl) / 1e3
        out[f'{fam}_inj_A'] = np.full(len(t0s), A)
        out[f'{fam}_fit_A'] = amp
        out[f'{fam}_peak_y'] = np.array([yi[int(q)] for q in t0s])
        out[f'{fam}_channel'] = np.array(f'{DET_NAMES[d]}{ch}')
    np.savez_compressed(out_dir('early') / 'examples.npz', **out)
    print('examples ->', out_dir('early') / 'examples.npz')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--examples':
        examples(sys.argv[2])
        sys.exit(0)
    sys.exit(main())
