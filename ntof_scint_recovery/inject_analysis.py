#!/usr/bin/env python3
"""
inject_analysis.py -- turn the injection histograms (inject.py) into the headline:
from when after the flash can a pulse be reconstructed, and with what amplitude bias
and amplitude / time resolution, per family, intensity class and pulse amplitude.

    python -m ntof_scint_recovery.inject_analysis [inject.npz] [--out DIR]

Per (method, family, class, amplitude, time bin), pooled over the family's live channels,
for all injected pulses and for the CLEAN ones (no real pulse overlapping the fit window):
  eff(_clean)        found / injected
  f_piled            fraction of injections that land on a real pulse (beam occupancy)
  bias(_clean)       median A_fit/A over its control value, - 1
  r_s68(_clean)      half the 16-84 % range of A_fit/A over its median (relative resolution);
  res_excess         in quadrature above the control's
  d_bias_ns, d_s68   median and half 16-84 % of t_fit - t0, relative to / above the control
  good5, good10      fraction of injected pulses found within 5 / 10 % of the control amplitude
  ro_med             the oracle ratio (the injected pulse alone): the clipping only

"Reconstructable" (``t_rec``): the first time bin from which every later bin passes, on
clean pulses: efficiency, |bias|, resolution excess and time-resolution excess under the
``CRITERIA``; and with every pulse (occupancy included): good5 or good10 >= 0.9.

Writes ``inject_summary.csv`` (one row per family/class/amplitude/time bin) and
``inject_trec.csv`` (t_rec per family/class/amplitude, for several criteria).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir

FAMS = ('WAL', 'PSS', 'LIQ')
FAM_DET = {'WAL': range(0, 4), 'PSS': range(4, 8), 'LIQ': range(8, 12)}


def quant(h: np.ndarray, edges: np.ndarray, qs=(0.16, 0.5, 0.84)) -> np.ndarray:
    """Quantiles of a binned distribution (linear inside bins)."""
    c = np.cumsum(h)
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    c = c / c[-1]
    return np.interp(qs, np.r_[0, c], edges)


def _stats(R, D, re, de):
    r16, r50, r84 = quant(R, re)
    d16, d50, d84 = quant(D, de)
    return dict(r_med=r50, r_s68=(r84 - r16) / 2 / r50 if r50 > 0 else np.nan, n_fit=R.sum(),
                d_med=d50, d_s68=(d84 - d16) / 2)


def summarise(z) -> pd.DataFrame:
    ch = z['channels']
    te = z['t_edges_us']
    re = z['r_edges']
    rc = 0.5 * (re[1:] + re[:-1])
    dh = json.loads(str(z['d_half']))
    nd = int(z['n_d'])
    live = z['N'].sum(axis=(0, 2, 3)) > 0
    rows = []
    for fam in FAMS:
        cs = np.array([i for i, (d, _) in enumerate(ch) if d in FAM_DET[fam] and live[i]])
        if not len(cs):
            continue
        amps = z[f'amps_{fam}']
        de = np.linspace(-dh[fam], dh[fam], nd + 1)
        for im, meth in enumerate(z['methods']):
            for cls in (0, 1):
                for ia, A in enumerate(amps):
                    rctrl = None
                    for it in range(len(te)):
                        N = z['N'][cls, cs, ia, it].sum()
                        if N < 20:
                            continue
                        NC = z['NC'][cls, cs, ia, it].sum()
                        NF = z['NF'][im, cls, cs, ia, it].sum()
                        NFC = z['NFC'][im, cls, cs, ia, it].sum()
                        Rp = z['R'][im, cls, cs, ia, it].sum(axis=0)          # (pile, r)
                        Dp = z['D'][im, cls, cs, ia, it].sum(axis=0)
                        RO = z['RO'][cls, cs, ia, it].sum(axis=0)
                        allst = _stats(Rp.sum(0), Dp.sum(0), re, de)
                        clst = _stats(Rp[0], Dp[0], re, de) if NC >= 10 else \
                            dict(r_med=np.nan, r_s68=np.nan, n_fit=0, d_med=np.nan, d_s68=np.nan)
                        if it == 0:
                            rctrl = allst['r_med']
                            t_lo, t_hi, t_mid = z['ctrl_us'][0], z['ctrl_us'][1], np.nan
                        else:
                            t_lo, t_hi = te[it - 1], te[it]
                            t_mid = np.sqrt(t_lo * t_hi)
                        if rctrl is None:
                            continue
                        good = lambda h, tol: float(h[np.abs(rc / rctrl - 1) <= tol].sum())  # noqa: E731
                        rows.append(dict(
                            method=str(meth), fam=fam, cls='ded' if cls else 'par', amp_mV=A, tbin=it,
                            t_lo_us=t_lo, t_hi_us=t_hi, t_us=t_mid, n_inj=N, n_clean=NC,
                            f_piled=1 - NC / N, f_hit_real=z['NP'][cls, cs, ia, it].sum() / N,
                            eff=NF / N, eff_clean=NFC / NC if NC else np.nan,
                            good5=good(Rp.sum(0), 0.05) / N, good10=good(Rp.sum(0), 0.10) / N,
                            good5_clean=good(Rp[0], 0.05) / NC if NC else np.nan,
                            ro_med=quant(RO, re)[1],
                            **{k: v for k, v in allst.items()},
                            **{f'{k}_clean': v for k, v in clst.items()}))
    S = pd.DataFrame(rows)
    out = []
    for _, g in S.groupby(['method', 'fam', 'cls', 'amp_mV'], sort=False):
        c = g[g.tbin == 0].iloc[0]
        g = g.copy()
        for sfx in ('', '_clean'):
            g['bias' + sfx] = g['r_med' + sfx] / c.r_med - 1
            g['res_excess' + sfx] = np.sqrt(np.clip(g['r_s68' + sfx] ** 2 - c.r_s68 ** 2, 0, None))
            g['d_bias_ns' + sfx] = g['d_med' + sfx] - c.d_med
            g['d_res_excess_ns' + sfx] = np.sqrt(np.clip(g['d_s68' + sfx] ** 2 - c.d_s68 ** 2, 0, None))
        g['res_ctrl'] = c.r_s68
        g['d_res_ctrl'] = c.d_s68
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


#: name: (clean efficiency min, |amplitude bias| max, amplitude resolution excess max (in quadrature),
#:        time resolution excess max ns (plastics/liquids; x4 for the walls))
CRITERIA = {
    'strict': (0.98, 0.01, 0.01, 0.25),
    'nominal': (0.95, 0.03, 0.03, 0.5),
    'loose': (0.90, 0.10, 0.10, 1.0),
}


def t_rec(S: pd.DataFrame) -> pd.DataFrame:
    """First time from which every later bin passes, on CLEAN pulses (the tail itself);
    and the same with every pulse (the occupancy included) on good5 >= 0.9."""
    rows = []
    for (meth, fam, cls, A), g in S[S.tbin > 0].groupby(['method', 'fam', 'cls', 'amp_mV'], sort=False):
        g = g.sort_values('tbin')
        k = 4.0 if fam == 'WAL' else 1.0
        tests = {name: ((g.eff_clean >= e) & (g.bias_clean.abs() <= b) & (g.res_excess_clean <= r)
                        & (g.d_res_excess_ns_clean <= k * d) & (g.d_bias_ns_clean.abs() <= k * d))
                 for name, (e, b, r, d) in CRITERIA.items()}
        tests['all_good5_0.9'] = g.good5 >= 0.9
        tests['all_good10_0.9'] = g.good10 >= 0.9
        for name, ok in tests.items():
            ok = ok.fillna(False).to_numpy()
            bad = np.nonzero(~ok)[0]
            if not len(bad):
                t, note = g.t_lo_us.iloc[0], 'from first bin'
            elif bad[-1] == len(ok) - 1:
                t, note = np.nan, 'not within the head'
            else:
                t, note = g.t_lo_us.iloc[bad[-1] + 1], ''
            rows.append(dict(method=meth, fam=fam, cls=cls, amp_mV=A, criterion=name, t_rec_us=t,
                             note=note, t_max_us=g.t_hi_us.iloc[-1]))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inject', nargs='?', default=str(out_dir('inject') / 'inject.npz'))
    ap.add_argument('--out', type=Path, default=out_dir('inject'))
    a = ap.parse_args(argv)
    z = dict(np.load(a.inject))                       # decompress once, not per access
    S = summarise(z)
    a.out.mkdir(parents=True, exist_ok=True)
    S.to_csv(a.out / 'inject_summary.csv', index=False)
    TR = t_rec(S)
    TR.to_csv(a.out / 'inject_trec.csv', index=False)
    pd.set_option('display.width', 200)
    print(TR.pivot_table(index=['fam', 'cls', 'amp_mV'], columns=['method', 'criterion'], values='t_rec_us').round(2).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
