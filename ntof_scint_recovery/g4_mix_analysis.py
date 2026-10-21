#!/usr/bin/env python3
"""
g4_mix_analysis.py -- what Geant4 (g4_mix.py) says arrives after the flash, against the data.

    python -m ntof_scint_recovery.g4_mix_analysis [--g4 g4mix_all.npz] [--hists DIR]

Writes to ``out_dir('g4mix')``:

``g4_rates.csv``   per family and coarse time bin: Geant4 clusters per counter per µs per bunch
                   above ``L_THR`` MeVee (dedicated intensity), the class fractions, and the
                   data pulse rate per channel above 4 mV (production, dedicated) for scale;
``g4_spectra.csv`` per family and time bin: light quantiles (50/90/95 %) above ``L_THR``,
                   each over its 1-10 ms value, by class and all;
``g4_pw_edge.csv`` the plastic-behind-a-wall-hit upper edge (95 %) over its 1-10 ms value,
                   Geant4 (no gain change by construction) against the data (median over bars).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_scint_recovery.common import out_dir

L_THR = 0.1                 # MeVee: ~4-7 mV on a plastic, ~5 mV on a wall end
PPP_DED, PPP_NOM = 8.5e12, 7e12
#: coarse time edges (µs) for Geant4 statistics
TC = np.array([0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 20, 30, 50, 100, 200, 300, 500,
               1000, 2000, 3000, 5000, 1e4])
PLATEAU = (1e3, 1e4)
#: counters per arm in the simulation: wall = the whole arm, 2 plastic bars, 2 liquid cells
N_COUNT = {0: 1, 1: 2, 2: 2}
FAMS = ('wall', 'plastic', 'liquid')
DFAM = {'wall': 'WAL', 'plastic': 'PSS', 'liquid': 'LIQ'}


def rebin_t(H: np.ndarray, te: np.ndarray, axis: int) -> np.ndarray:
    """Sum the fine log-time bins into TC (by bin centre)."""
    c = np.sqrt(te[:-1] * te[1:])
    idx = np.digitize(c, TC) - 1
    out = []
    for k in range(len(TC) - 1):
        out.append(np.take(H, np.nonzero(idx == k)[0], axis=axis).sum(axis=axis))
    return np.stack(out, axis=axis)


def quant(h, edges, q):
    c = np.cumsum(h)
    if c[-1] < 20:
        return np.nan
    return float(np.interp(q, np.r_[0, c / c[-1]], edges))


def data_rates(hdir: Path) -> pd.DataFrame:
    from ntof_scint_recovery.coinc_analysis import load
    H = load(hdir)
    te, ae = H['t_edges_us'], H['a_edges']
    chans = [tuple(c) for c in H['channels']]
    HA = H['HA']                                  # (cls, ch, t, amp)
    nb = H['nb'][1]
    rows = []
    for fam, fd in DFAM.items():
        dets = {'WAL': range(4), 'PSS': range(4, 8), 'LIQ': range(8, 12)}[fd]
        cis = [i for i, c in enumerate(chans) if c[0] in dets]
        live = [i for i in cis if HA[1, i].sum() > 0]
        h = HA[1, live][:, :, ae[:-1] >= 4.0].sum(axis=(0, 2))       # per fine t bin
        hc = rebin_t(h[:100], te[:101], 0)
        for k in range(len(TC) - 1):
            rows.append(dict(fam=fam, t_lo_us=TC[k], t_hi_us=TC[k + 1],
                             data_rate_per_us=hc[k] / nb / len(live) / (TC[k + 1] - TC[k])))
    return pd.DataFrame(rows)


def data_pw_edge(hdir: Path) -> pd.DataFrame:
    from ntof_scint_recovery.coinc_analysis import load
    H = load(hdir)
    tce, ae = H['tc_edges_us'], H['a_edges']
    net = H['PW_sig'] - H['pw_side_scale'] * H['PW_side']            # (cls, arm, bar, t, amp)
    tcc = np.sqrt(tce[:-1] * tce[1:])
    pl = (tcc > PLATEAU[0]) & (tcc < PLATEAU[1])
    curves = []
    for arm in range(4):
        for bar in range(2):
            n = np.clip(net[1, arm, bar], 0, None)
            if n.sum() < 1000:
                continue
            e = np.array([quant(n[k], ae, 0.95) for k in range(len(tcc))])
            ref = quant(n[pl].sum(axis=0), ae, 0.95)
            curves.append(e / ref)
    C = np.array(curves)
    return pd.DataFrame(dict(t_us=tcc, data_edge_rel=np.nanmedian(C, axis=0),
                             data_edge_rms=np.nanstd(C, axis=0), n_bars=np.isfinite(C).sum(axis=0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--g4', type=Path, default=out_dir('g4mix') / 'g4mix_all.npz')
    ap.add_argument('--hists', type=Path, default=out_dir('pulsehist_v3'))
    a = ap.parse_args(argv)
    z = np.load(a.g4)
    te, le = z['t_edges_us'], z['l_edges']
    n_pulse = float(z['n_events']) / float(z['n_per_pulse'])          # beam pulses simulated (7e12)
    scale = PPP_DED / PPP_NOM / n_pulse                                # per dedicated bunch
    H = z['H'][:, :, :120]                                             # (fam, arm, t, L, cls)
    Hc = rebin_t(H, te[:121], 2)                                       # (fam, arm, tc, L, cls)
    lc = np.sqrt(le[:-1] * le[1:])
    above = lc >= L_THR
    rows, srows = [], []
    for f, fam in enumerate(FAMS):
        h = Hc[f].sum(axis=0)                                          # (tc, L, cls)
        n_counters = 4 * N_COUNT[f]
        pl = (TC[:-1] >= PLATEAU[0]) & (TC[1:] <= PLATEAU[1])
        ref = {}
        for cname, sl in (('all', slice(None)),):
            hp = h[pl][:, :, sl].sum(axis=(0, 2))
            ref = {q: quant(hp * above, le, q) for q in (0.5, 0.9, 0.95)}
        for k in range(len(TC) - 1):
            hk = h[k] * above[:, None]
            tot = hk.sum()
            fr = hk.sum(axis=0) / max(tot, 1)
            big = h[k] * (lc >= 1.0)[:, None]
            frb = big.sum(axis=0) / max(big.sum(), 1)
            rows.append(dict(fam=fam, t_lo_us=TC[k], t_hi_us=TC[k + 1], g4_n=tot,
                             g4_rate_per_us=tot * scale / n_counters / (TC[k + 1] - TC[k]),
                             f_em=fr[0], f_proton=fr[1], f_ion=fr[2], f_other=fr[3],
                             n_ge1=big.sum(), f1_em=frb[0], f1_proton=frb[1], f1_ion=frb[2]))
            hs = hk.sum(axis=1)
            srows.append(dict(fam=fam, t_lo_us=TC[k], t_hi_us=TC[k + 1], n=tot,
                              **{f'q{int(q * 100)}_rel': quant(hs, le, q) / ref[q] for q in (0.5, 0.9, 0.95)},
                              **{f'q{int(q * 100)}_MeVee': quant(hs, le, q) for q in (0.5, 0.9, 0.95)},
                              **{f'q95_{c}_MeVee': quant(hk[:, i], le, 0.95) for i, c in enumerate(('em', 'p', 'ion'))}))
    R = pd.DataFrame(rows)
    try:
        R = R.merge(data_rates(a.hists), on=['fam', 't_lo_us', 't_hi_us'], how='left')
    except Exception as e:                                             # noqa: BLE001
        print(f'no data rates: {e!r}')
    o = out_dir('g4mix')
    R.to_csv(o / 'g4_rates.csv', index=False)
    S = pd.DataFrame(srows)
    S.to_csv(o / 'g4_spectra.csv', index=False)
    # plastic behind a wall hit
    PW = rebin_t(z['PW'][:, :120], te[:121], 1).sum(axis=0)           # (tc, L, cls)
    pl = (TC[:-1] >= PLATEAU[0]) & (TC[1:] <= PLATEAU[1])
    hp = PW[pl].sum(axis=(0, 2))
    ref = quant(hp, le, 0.95)
    E = pd.DataFrame(dict(t_lo_us=TC[:-1], t_hi_us=TC[1:], t_us=np.sqrt(TC[:-1] * TC[1:]),
                          g4_n=PW.sum(axis=(1, 2)),
                          g4_edge_MeVee=[quant(PW[k].sum(axis=1), le, 0.95) for k in range(len(TC) - 1)]))
    E['g4_edge_rel'] = E.g4_edge_MeVee / ref
    E['g4_f_proton'] = PW[:, :, 1].sum(axis=1) / np.maximum(PW.sum(axis=(1, 2)), 1)
    E.to_csv(o / 'g4_pw_edge.csv', index=False)
    try:
        D = data_pw_edge(a.hists)
        D.to_csv(o / 'data_pw_edge.csv', index=False)
    except Exception as e:                                             # noqa: BLE001
        print(f'no data edge: {e!r}')
    pd.set_option('display.width', 250)
    print(f'{n_pulse:.1f} beam pulses simulated (7e12 ppp)')
    print(R.round(4).to_string(index=False))
    print(S.round(3).to_string(index=False))
    print(E.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
