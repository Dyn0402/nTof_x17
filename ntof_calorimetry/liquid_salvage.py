#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liquid_salvage.py -- PLAN.md C4: the liquids' efficiency for particles that
certainly reach them, separated from punch-through.

The scint-stack report could only say "the liquid answered for x % of the
tracks pointing at it", where most beam tracks are few-MeV electrons that stop
in the plastic.  Here every particle is minimum-ionising and is CONFIRMED to
have crossed the plastic in front of the liquid: the predicted plastic bar
fired with a MIP-like deposit (0.6-2.5 x that bar's own MIP peak, `scint_ecal`)
and both ends of the predicted wall group fired.  A muon that did that crosses
the 18 mm liquid cell behind it if its line is on the cell's face, so the
liquid's answer is its MIP efficiency, cell by cell.

Expected MIP deposit in the cell: ~2.6 MeV (`landau.mpv`, LAB 18 mm), i.e.
~110-130 mV on the source scale -- far above any sensible threshold.

Net of accidentals: the same-width window before the trigger (``lf_off``).

    python -m ntof_calorimetry.liquid_salvage        # -> OUT/c4/*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_calorimetry.scint_ecal import mip_selection, samples  # noqa: E402
from ntof_scint_stack import ana as SA  # noqa: E402

C4 = OUT / 'c4'
CELL = 50.0
EDGE_L = 20.0
MIP_LO, MIP_HI = 0.6, 2.5


def tagged(D: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    """MIP crossing the plastic in front of the liquid, pointing at the cell."""
    ref = R[R['sample'] == 'both'].set_index(['arm', 'bar']).mpv.to_dict()
    mref = np.array([ref.get((a, b), np.nan) for a, b in zip(D.arm, D.bar)])
    r = D.e_mv.to_numpy() / mref
    return (mip_selection(D) & (r > MIP_LO) & (r < MIP_HI) & D.on_l & (D.d_edge_l > EDGE_L))


def net(on: np.ndarray, off: np.ndarray) -> tuple:
    n = len(on)
    if n == 0:
        return np.nan, np.nan
    p_on, p_off = on.mean(), off.mean()
    e = (p_on - p_off) / max(1 - p_off, 1e-9)
    err = np.sqrt(p_on * (1 - p_on) / n + p_off * (1 - p_off) / n) / max(1 - p_off, 1e-9)
    return float(e), float(err)


def overall(D: pd.DataFrame, T: pd.Series, ecal: dict) -> pd.DataFrame:
    rows = []
    for samp in ('cosmic', 'beam', 'both'):
        ms = np.ones(len(D), bool) if samp == 'both' else (D['sample'] == samp).to_numpy()
        for arm in 'ABCD':
            m = T.to_numpy() & ms & (D.arm == arm).to_numpy()
            x = D[m]
            e, err = net(x.lf_on.to_numpy(), x.lf_off.to_numpy())
            k = ecal.get(f'LIQ{arm}1', (np.nan, 0.0))[0]
            la = x.la[x.lf_on & ~x.lsat]
            f = LD.fit(la.to_numpy()) if len(la) >= 40 else dict(mpv_landau=np.nan, mpv_err=np.nan)
            rows.append(dict(sample=samp, arm=arm, n=int(m.sum()), eff=e, err=err,
                             p_off=float(x.lf_off.mean()) if len(x) else np.nan,
                             n_fired=int(x.lf_on.sum()),
                             amp_med_mv=float(la.median()) if len(la) else np.nan,
                             amp_mpv_mv=f['mpv_landau'], amp_mpv_err=f['mpv_err'],
                             mpv_mev=f['mpv_landau'] / k / 1000 if np.isfinite(k) else np.nan,
                             exp_mev=LD.mpv(18.0 / float(x.cos.median()) if len(x) else 18.0, 30, 'LAB'),
                             area_over_amp=float(np.median(x.larea[x.lf_on] / x.la[x.lf_on]))
                             if x.lf_on.any() else np.nan,
                             frac_sat=float(x.lsat[x.lf_on].mean()) if x.lf_on.any() else np.nan))
    return pd.DataFrame(rows)


def maps(D: pd.DataFrame, T: pd.Series) -> pd.DataFrame:
    """50 mm cells on the liquid face, liquid-centred u, v, per sample."""
    rows = []
    for samp in ('cosmic', 'beam'):
        rows += _maps(D[T & (D['sample'] == samp)], samp)
    return pd.DataFrame(rows)


def _maps(x: pd.DataFrame, samp: str) -> list:
    rows = []
    for arm in 'ABCD':
        a = x[x.arm == arm]
        iu = np.floor((a.u_l + SA.LS_HALF_U) / CELL).astype(int)
        iv = np.floor((a.v_l + SA.LS_HALF_V) / CELL).astype(int)
        for (i, j), g in a.groupby([iu, iv]):
            e, err = net(g.lf_on.to_numpy(), g.lf_off.to_numpy())
            la = g.la[g.lf_on & ~g.lsat]
            rows.append(dict(sample=samp, arm=arm, iu=i, iv=j, u=-SA.LS_HALF_U + (i + 0.5) * CELL,
                             v=-SA.LS_HALF_V + (j + 0.5) * CELL, n=len(g), eff=e, err=err,
                             amp_med=float(la.median()) if len(la) >= 5 else np.nan))
    return rows


def profiles(D: pd.DataFrame, T: pd.Series) -> pd.DataFrame:
    """Efficiency along u and along v (each integrated over the other), the
    PMT-orientation test: A/D horizontal with the PMT at +u, B/C vertical."""
    rows = []
    x = D[T & (D['sample'] == 'cosmic')]
    edges = np.arange(-225, 226, 50)
    for arm in 'ABCD':
        a = x[x.arm == arm]
        for ax in ('u', 'v'):
            c = a[f'{ax}_l'].to_numpy()
            for lo, hi in zip(edges[:-1], edges[1:]):
                g = a[(c >= lo) & (c < hi)]
                e, err = net(g.lf_on.to_numpy(), g.lf_off.to_numpy())
                la = g.la[g.lf_on & ~g.lsat]
                rows.append(dict(arm=arm, axis=ax, lo=lo, hi=hi, mid=0.5 * (lo + hi), n=len(g),
                                 eff=e, err=err,
                                 amp_med=float(la.median()) if len(la) >= 5 else np.nan))
    return pd.DataFrame(rows)


#: n_TOF zero-suppression threshold of each liquid, mV (DAQsettings of the
#: run_149 n_TOF runs 224678-87: -16.0/-17.0/-18.0/-17.0).  A pulse below it
#: opens no waveform segment, so it is lost unless something else opened one.
LIQ_ZS = {'A': 16.0, 'B': 17.0, 'C': 18.0, 'D': 17.0}


def mip_scale(D: pd.DataFrame, T: pd.Series) -> pd.DataFrame:
    """The liquid MIP peak on cosmics, fitted with the ZS threshold as the
    truncation, and the MIP-based mV/MeV against the source scale."""
    from ntof_scint_stack.ana import energy_scale
    ecal = energy_scale()
    rows = []
    x = D[T & (D['sample'] == 'cosmic') & D.lf_on & ~D.lsat]
    for arm in 'ABCD':
        a = x[x.arm == arm]
        if len(a) < 40:
            continue
        f = LD.fit_trunc(a.la.to_numpy(), np.full(len(a), LIQ_ZS[arm]), 200.0,
                         xi_ratio=0.051, res_prior=(0.35, 0.15))
        exp = LD.mpv(18.0 / float(a.cos.median()), 30, "LAB")
        k_src = ecal.get(f'LIQ{arm}1', (np.nan, 0))[0] * 1000
        rows.append(dict(arm=arm, n=len(a), mpv_mv=f['mpv'], mpv_err=f['mpv_err'], sigma_mv=f['sigma'],
                         exp_mev=exp, mv_per_mev_mip=f['mpv'] / exp, mv_per_mev_source=k_src,
                         mip_over_source=f['mpv'] / exp / k_src, zs_mv=LIQ_ZS[arm],
                         zs_mev_mip=LIQ_ZS[arm] / (f['mpv'] / exp)))
    return pd.DataFrame(rows)


def penetration(D: pd.DataFrame, T: pd.Series, M: pd.DataFrame) -> pd.DataFrame:
    """Do in-beam 'through-goers' reach the liquid?  Each beam event's
    expected liquid probability is the COSMIC efficiency of its cell (cells
    with >= 15 cosmic tags); observed / expected = 1 for penetrating MIPs."""
    rows = []
    x = D[T & (D['sample'] == 'beam')].copy()
    Mc = M[(M['sample'] == 'cosmic') & (M.n >= 15)]
    for arm in 'ACD':
        a = x[x.arm == arm]
        iu = np.floor((a.u_l + SA.LS_HALF_U) / CELL).astype(int)
        iv = np.floor((a.v_l + SA.LS_HALF_V) / CELL).astype(int)
        e = pd.Series(list(zip(iu, iv)), index=a.index).map(
            Mc[Mc.arm == arm].set_index(['iu', 'iv']).eff.to_dict())
        ok = e.notna()
        if ok.sum() < 20:
            continue
        exp_ = float(e[ok].clip(0, 1).sum())
        obs = float((a.lf_on[ok].astype(float) - a.lf_off[ok].astype(float)).sum())
        rows.append(dict(arm=arm, n=int(ok.sum()), expected=exp_, observed=obs,
                         ratio=obs / exp_, ratio_err=np.sqrt(max(obs, 1)) / exp_))
    return pd.DataFrame(rows)


def main() -> int:
    C4.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(OUT / 'c1' / 'mip_per_bar.csv')
    D = samples()
    T = tagged(D, R)
    ecal = SA.energy_scale()
    O = overall(D, T, ecal)
    M = maps(D, T)
    P = profiles(D, T)
    S = mip_scale(D, T)
    PN = penetration(D, T, M)
    S.to_csv(C4 / 'mip_scale.csv', index=False)
    PN.to_csv(C4 / 'penetration.csv', index=False)
    O.to_csv(C4 / 'overall.csv', index=False)
    M.to_csv(C4 / 'maps.csv', index=False)
    P.to_csv(C4 / 'profiles.csv', index=False)
    keep = ['sample', 'arm', 'u_l', 'v_l', 'lf_on', 'lf_off', 'la', 'larea', 'lsat', 'l_dt',
            'e_mv', 'bar', 'cos']
    D[T][keep].to_parquet(C4 / 'liquid_tagged.parquet', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(O.round(3).to_string(index=False))
        print(S.round(3).to_string(index=False))
        print(PN.round(3).to_string(index=False))
        print(P[P.n >= 20].round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
