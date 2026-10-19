#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leg_penetration.py -- how far into the stack an aluminium-capture lepton gets,
against an X17 lepton, and where the single tracks come from.  Geant4 truth
only, for the 2026-10-15 status talk (make_status_oct15_deck.py).

Inputs
  Al / singles   MX17_Full_Geant same_arm_pairs.py per-lepton truth over
                 neutrons_thermal_trig_2cm_nose (10^9 thermal n, nose-first
                 capsule): ~/x17_local/g4_same_arm/parts/*.parquet.  One row per
                 lepton entering a drift gap, with its OWN wall and plastic
                 deposits (wall_eV, plas_eV), its origin and capture site.
  X17            the C3 pair sim pairs_thermal_trig_2cm_nose (X17 + IPC, 10^7),
                 reduced per event x arm (ntof_calorimetry.pair_sensitivity.load),
                 deposits per arm incl. the liquid.  X17 = event_type 0, legs
                 into two different arms, both arms' gas > 0.5 keV.
  liquid, Al     not in the per-lepton truth; folded from the C2 single-electron
                 response (calorimetry c2/response.csv, P(liquid) vs T at the
                 capsule) at T = gap-entry energy + 0.6 MeV (half the measured
                 upstream loss of a conversion pair).  Marked as an estimate.

Ladder, per lepton that crosses >= 20 mm of a drift gap:
  gap -> wall (>= 0.3 MeV) -> plastic (>= 0.3 MeV) -> plastic >= 2.5 MeV
  (the trigger's measured 2.1-2.9 MeVee) -> liquid.

Outputs (paths.out('status_oct15')):
  penetration.csv     ladder per population, per lepton and per pair
  leg_spectra.csv     gap-entry (Al) / creation (X17) kinetic energy histograms
  singles_origin.csv  triggered track-like singles in A and C by origin

    python -m sept26_prelim_analysis.leg_penetration
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths

PARTS = Path.home() / 'x17_local' / 'g4_same_arm' / 'parts'
C2 = Path('/media/dylan/data/x17/calorimetry/c2/response.csv')
OUT = paths.out('status_oct15')
TRACK_W_MM = 20.0
WALL_MIN, PLAS_MIN, PLAS_TRIG = 0.3, 0.3, 2.5     # MeV
LIQ_MIN = 0.5
UP_LOSS_TO_GAP = 0.6                               # MeV per leg, capsule -> gap
SIM_ARM = {'A': 2, 'C': 3}
E_BINS = np.arange(0, 16.01, 0.5)


def load_g4() -> pd.DataFrame:
    L = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob('*.parquet'))], ignore_index=True)
    L['w'] = L.w_max - L.w_min
    return L[L.w >= TRACK_W_MM].copy()


def p_liq_curve():
    R = pd.read_csv(C2)
    R = R[R.phi == 0].groupby('T').p_liq.mean()          # e+ and e- alike at these energies
    return lambda T: np.interp(T, R.index.to_numpy(), R.to_numpy(), left=0.0, right=float(R.iloc[-1]))


def c2_ladder() -> pd.DataFrame:
    """Energy-only penetration: the C2 single-lepton guns (capsule centre, aimed at a
    plastic bar, theta 90, phi 0), conditional on the lepton reaching the gap."""
    rows = []
    for f in sorted((C2.parent / 'raw').glob('*_th90_ph0.csv.gz')):
        part, T = f.name.split('_')[0], float(f.name.split('_')[1].replace('MeV', ''))
        h = pd.read_csv(f)
        h = h[h.armID == 0]
        h = h[h.e_gas > 0]
        if len(h) < 100:
            continue
        pl = h.e_plas_L + h.e_plas_R
        rows.append(dict(particle=part, T=T, n=len(h), wall=float(np.mean(h.e_wall >= WALL_MIN)),
                         plastic=float(np.mean(pl >= PLAS_MIN)), plastic_trig=float(np.mean(pl >= PLAS_TRIG)),
                         liquid=float(np.mean(h.e_liq >= LIQ_MIN))))
    R = pd.DataFrame(rows)
    return R.groupby('T')[['wall', 'plastic', 'plastic_trig', 'liquid']].mean().reset_index()


def fold(name, T, C):
    out = dict(population=name, kind='per lepton, energy only (aimed at a bar)', n=len(T), gap=1.0)
    for c in ('wall', 'plastic', 'plastic_trig', 'liquid'):
        out[c] = float(np.mean(np.interp(T, C['T'], C[c], left=0.0, right=float(C[c].iloc[-1]))))
    return out


def fold_pair(name, T1, T2, C):
    out = dict(population=name, kind='both legs, energy only (aimed at a bar)', n=len(T1), gap=1.0)
    for c in ('wall', 'plastic', 'plastic_trig', 'liquid'):
        f = lambda T: np.interp(T, C['T'], C[c], left=0.0, right=float(C[c].iloc[-1]))
        out[c] = float(np.mean(f(T1) * f(T2)))
    return out


def ladder(name, kind, wall, plas, liq, n_events=None):
    n = len(wall)
    return dict(population=name, kind=kind, n=n,
                gap=1.0, wall=float(np.mean(wall >= WALL_MIN)),
                plastic=float(np.mean(plas >= PLAS_MIN)),
                plastic_trig=float(np.mean(plas >= PLAS_TRIG)),
                liquid=float(np.mean(liq)) if liq.dtype != bool else float(np.mean(liq)))


def al(L: pd.DataFrame, fliq):
    rows = []
    conv = L[(L.op == 'conv') & (L.ov == 'He3Cap_Al') & (L.cap == 'ncapture_wall_al')]
    T = conv.ke0.to_numpy() + UP_LOSS_TO_GAP
    rows.append(ladder('Al conversion lepton', 'per lepton', conv.wall_eV.to_numpy() / 1e6,
                       conv.plas_eV.to_numpy() / 1e6, fliq(T)))
    # pairs: both legs of one photon in the same arm's gap
    g = conv.groupby(['file', 'event', 'parent', 'arm'])
    k = g.size()
    two = k[k == 2].index
    P = conv.set_index(['file', 'event', 'parent', 'arm']).loc[two].reset_index()
    agg = P.groupby(['file', 'event', 'parent', 'arm']).agg(
        wmin=('wall_eV', 'min'), pmin=('plas_eV', 'min'), pmax=('plas_eV', 'max'), kmax=('ke0', 'max'),
        kmin=('ke0', 'min'))
    pl = fliq(agg.kmin.to_numpy() + UP_LOSS_TO_GAP) * fliq(agg.kmax.to_numpy() + UP_LOSS_TO_GAP)
    rows.append(dict(population='Al e⁺e⁻ pair (same chamber)', kind='both legs', n=len(agg), gap=1.0,
                     wall=float(np.mean(agg.wmin / 1e6 >= WALL_MIN)),
                     plastic=float(np.mean(agg.pmin / 1e6 >= PLAS_MIN)),
                     plastic_trig=float(np.mean(agg.pmin / 1e6 >= PLAS_TRIG)),
                     liquid=float(np.mean(pl))))
    # a same-chamber pair shares one plastic: what the trigger sees is the sum
    rows.append(dict(population='Al e⁺e⁻ pair (same chamber)', kind='arm can trigger (sum of legs)', n=len(agg),
                     gap=1.0, wall=np.nan, plastic=np.nan,
                     plastic_trig=float(np.mean((P.groupby(['file', 'event', 'parent', 'arm']).plas_eV.sum()
                                                 / 1e6) >= PLAS_TRIG)), liquid=np.nan))
    spec = np.histogram(conv.ke0, E_BINS)[0]
    return rows, spec, conv, agg


def x17():
    from ntof_calorimetry import pair_sensitivity as PS
    E, H = PS.load()
    E = E[E.event_type == 0].copy()
    H = H[H.ev.isin(E.ev)]
    E['arm_m'] = PS.leg_arm(E.em_px, E.em_py, E.em_pz)
    E['arm_p'] = PS.leg_arm(E.ep_px, E.ep_py, E.ep_pz)
    H = H.assign(e_plas=H.e_plas_L + H.e_plas_R)
    idx = H.set_index(['ev', 'armID'])
    for leg in ('m', 'p'):
        k = pd.MultiIndex.from_arrays([E.ev, E[f'arm_{leg}']])
        for c in ('e_gas', 'e_plas', 'e_wall', 'e_liq'):
            E[f'{c}_{leg}'] = idx[c].reindex(k).fillna(0.0).to_numpy()
    S = E[(E.arm_m >= 0) & (E.arm_p >= 0) & (E.arm_m != E.arm_p)
          & (E.e_gas_m > PS.GAS_MIN) & (E.e_gas_p > PS.GAS_MIN)]
    wall = np.r_[S.e_wall_m, S.e_wall_p]
    plas = np.r_[S.e_plas_m, S.e_plas_p]
    liq = np.r_[S.e_liq_m, S.e_liq_p] >= LIQ_MIN
    rows = [ladder('X17 lepton', 'per lepton', wall, plas, liq),
            dict(population='X17 pair (opposite chambers)', kind='both legs', n=len(S), gap=1.0,
                 wall=float(np.mean((S.e_wall_m >= WALL_MIN) & (S.e_wall_p >= WALL_MIN))),
                 plastic=float(np.mean((S.e_plas_m >= PLAS_MIN) & (S.e_plas_p >= PLAS_MIN))),
                 plastic_trig=float(np.mean((S.e_plas_m >= PLAS_TRIG) & (S.e_plas_p >= PLAS_TRIG))),
                 liquid=float(np.mean((S.e_liq_m >= LIQ_MIN) & (S.e_liq_p >= LIQ_MIN)))),
            dict(population='X17 pair (opposite chambers)', kind='either arm can trigger', n=len(S), gap=1.0,
                 wall=np.nan, plastic=np.nan,
                 plastic_trig=float(np.mean(((S.e_plas_m >= PLAS_TRIG) & (S.e_wall_m >= WALL_MIN))
                                            | ((S.e_plas_p >= PLAS_TRIG) & (S.e_wall_p >= WALL_MIN)))),
                 liquid=np.nan)]
    spec = np.histogram(np.r_[S.em_ke, S.ep_ke], E_BINS)[0]
    return rows, spec, S[['em_ke', 'ep_ke', 'openingAngle']]


def singles(L: pd.DataFrame) -> pd.DataFrame:
    """Triggered, track-like leptons in A and C, by what made them."""
    S = L[L.arm.isin(SIM_ARM.values()) & (L.trig_arms > 0)].copy()
    capsule = S.ov.isin(['He3Cap_Al', 'He3Cap_CFRP', 'He3Gas'])
    chamber = S.ov.str.contains('Drift|Mesh|PCB|Resistive|GasWindow|Cathode', regex=True)
    al_cap = S.cap == 'ncapture_wall_al'
    proc = S.op.map({'compt': 'Compton electron', 'conv': 'pair lepton (e⁺e⁻)', 'phot': 'photo-electron',
                     'eIoni': 'delta ray'}).fillna('other')
    where = np.select([capsule, chamber, S.ov == 'World'], ['made in the capsule', 'made in the chamber',
                                                            'made in the air'], 'made in other material')
    capt = np.where(al_cap, 'capture on the capsule Al',
                    np.where(S.cap == 'ncapture_wall_cfrp', 'capture on the capsule CFRP',
                             'capture elsewhere (scintillators, structure)'))
    S = S.assign(process=proc, made=where, capture=capt)
    out = []
    for col in ('process', 'made', 'capture'):
        c = S[col].value_counts()
        for k, v in c.items():
            out.append(dict(split=col, category=k, n=int(v), share=float(v / len(S))))
    g = S.gamma_E
    al_line = (g > 7.6) & (g < 7.8)
    out.append(dict(split='gamma', category='from the 7.72 MeV ²⁷Al line', n=int(al_line.sum()),
                    share=float(al_line.mean())))
    out.append(dict(split='summary', category='n_singles', n=len(S), share=1.0))
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fliq = p_liq_curve()
    L = load_g4()
    ra, sa, conv, agg = al(L, fliq)
    rx, sx, S = x17()
    C = c2_ladder()
    C.to_csv(OUT / 'c2_ladder.csv', index=False)
    rf = [fold('Al conversion lepton', conv.ke0.to_numpy() + UP_LOSS_TO_GAP, C),
          fold('X17 lepton', np.r_[S.em_ke, S.ep_ke], C),
          fold_pair('Al e⁺e⁻ pair (same chamber)', agg.kmin.to_numpy() + UP_LOSS_TO_GAP,
                    agg.kmax.to_numpy() + UP_LOSS_TO_GAP, C),
          fold_pair('X17 pair (opposite chambers)', S.em_ke.to_numpy(), S.ep_ke.to_numpy(), C)]
    pd.DataFrame(ra + rx + rf).to_csv(OUT / 'penetration.csv', index=False)
    pd.DataFrame(dict(lo=E_BINS[:-1], hi=E_BINS[1:], al_gap=sa, x17_birth=sx)).to_csv(
        OUT / 'leg_spectra.csv', index=False)
    so = singles(L)
    so.to_csv(OUT / 'singles_origin.csv', index=False)
    print(pd.DataFrame(ra + rx + rf).round(3).to_string())
    print(so.round(3).to_string())
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
