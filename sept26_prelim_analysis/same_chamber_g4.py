#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
same_chamber_g4.py -- what Geant4 says the same-chamber pairs should be, and how
many of them this analysis should have seen.

Input: the per-lepton truth tables of MX17_Full_Geant scripts/same_arm_pairs.py
(one row per lepton entering a drift gap, prompt, with its ancestry, its first
gap step and its own scintillator deposits; per-file trigger counts in the JSON),
over neutrons_thermal_trig_2cm_nose (10^9 thermal neutrons, nose-first capsule).

What it does, per arm (A = sim arm 2, C = 3):

1.  Track-like leptons: >= TRACK_W_MM of drift-gap depth crossed (a delta ray
    born in the gas makes a blob, not a track).
2.  Pairs = every two track-like leptons of one event in one arm, classified by
    origin: e+e- from one gamma, lepton + its own delta ray, two Compton or photo
    electrons of different gammas, ...
3.  The detector response, on the TRUE gap-entry positions and directions --
    so the multiple scattering between the vertex and the gap is in it:
    the measured tan resolution (cosmic A-C truth), |tan| < 0.6, the measured
    per-view two-track efficiency, and det_a_intra's vertex test (lines within
    30 mm, midpoint within 20 mm of the beam axis).
4.  Normalised PER TRIGGER (legacy leg in any arm, as the DAQ's global trigger),
    so the DAQ livetime and acceptance comb cancel; the per-track efficiency is
    calibrated by the single-track rate per trigger in data.

Output: tables + one JSON, which make_same_chamber_report picks up.

    python -m sept26_prelim_analysis.same_chamber_g4 --parts <dir with *.parquet/*.json>
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths
from sept26_prelim_analysis import same_chamber_physics as SP

SIM_ARM = {'A': 2, 'C': 3}
TRACK_W_MM = 20.0
WALL_HALF_MIP_EV = 0.5 * 0.4751e6
TAN_MAX = 0.6
VERTEX_R_MM, VERTEX_DCA_MM = 20.0, 30.0
OPEN_BINS = np.arange(0.0, 90.1, 10.0)
N_SMEAR = 20           # smearing replicas per pair


def load(parts: Path):
    L = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('*.parquet'))],
                  ignore_index=True)
    C = {}
    for p in sorted(parts.glob('*.json')):
        for k, v in json.load(open(p))['C'].items():
            C[k] = C.get(k, 0) + v
    return L, C


def frame(L: pd.DataFrame, arm: int):
    """Affine map global -> arm-local (u, v, w), fitted on the first gap steps."""
    g = L[L.arm == arm]
    X = np.c_[g.gx0, g.gy0, g.gz0, np.ones(len(g))].astype(float)
    Y = np.c_[g.u0, g.v0, g.w0].astype(float)
    M, *_ = np.linalg.lstsq(X, Y, rcond=None)
    R, t = M[:3].T, M[3]
    resid = float(np.abs(X @ M - Y).max())
    return R, t, resid


def classify(x, y) -> str:
    if x.op == 'conv' and y.op == 'conv' and x.parent == y.parent:
        return 'e+e- (one gamma)'
    if (x.op == 'eIoni' and x.parent == y.track) or (y.op == 'eIoni' and y.parent == x.track):
        return 'lepton + own delta ray'
    ops = {x.op, y.op}
    if ops <= {'compt', 'phot'}:
        return 'two Compton (one gamma)' if x.parent == y.parent else 'two Compton (two gammas)'
    if 'eIoni' in ops:
        return 'with a delta ray (other)'
    if ops == {'conv'}:
        return 'two conversions'
    return 'conversion + Compton'


def pairs(L: pd.DataFrame, arm: int, R, t) -> pd.DataFrame:
    T = L[(L.arm == arm) & ((L.w_max - L.w_min) >= TRACK_W_MM)]
    rows = []
    for (f, e), g in T.groupby(['file', 'event'], sort=False):
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(range(len(g)), 2):
            x, y = g.iloc[i], g.iloc[j]
            rows.append(dict(file=f, event=e, cls=classify(x, y),
                             trig=int(x.trig_arms) != 0, cap=x.cap,
                             gammaE=x.gamma_E if x.op == 'conv' else np.nan,
                             **{f'{k}{n}': getattr(z, k) for n, z in ((1, x), (2, y))
                                for k in ('u0', 'v0', 'w0', 'px0', 'py0', 'pz0', 'ke0',
                                          'wall_eV', 'ox', 'oy', 'oz', 'op')}))
    P = pd.DataFrame(rows)
    if P.empty:
        return P
    for n in (1, 2):
        d = np.c_[P[f'px0{n}'], P[f'py0{n}'], P[f'pz0{n}']] @ R.T
        P[f'tu{n}'] = d[:, 0] / d[:, 2]
        P[f'tv{n}'] = d[:, 1] / d[:, 2]
        P[f'dw{n}'] = np.sign(d[:, 2])
    c = (P.px01 * P.px02 + P.py01 * P.py02 + P.pz01 * P.pz02).clip(-1, 1)
    P['open_true'] = np.degrees(np.arccos(c))
    return P


def singles(L: pd.DataFrame, arm: int, R) -> pd.DataFrame:
    T = L[(L.arm == arm) & ((L.w_max - L.w_min) >= TRACK_W_MM)].copy()
    d = np.c_[T.px0, T.py0, T.pz0] @ R.T
    T['tu'], T['tv'] = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
    T['trig'] = T.trig_arms != 0
    return T


def response(P: pd.DataFrame, arm_letter: str, R, t, rng) -> pd.DataFrame:
    """Smeared reconstruction of each pair, N_SMEAR replicas: acceptance, the
    two-track weight, reco opening angle and the vertex test."""
    res = SP._resolution(arm_letter)
    sep = SP._sep_eff(arm_letter)
    # the beam axis in this arm's frame: the global line (0, y, 0)
    a0 = t
    adir = R @ np.array([0.0, 1.0, 0.0])
    adir = adir / np.linalg.norm(adir)
    out = []
    for r in range(N_SMEAR):
        q = {}
        for n in (1, 2):
            tu = SP._smear(P[f'tu{n}'].to_numpy(float), 'x', res, rng)
            tv = SP._smear(P[f'tv{n}'].to_numpy(float), 'y', res, rng)
            p = np.c_[P[f'u0{n}'] + rng.normal(0, 1.0, len(P)),
                      P[f'v0{n}'] + rng.normal(0, 1.0, len(P)), P[f'w0{n}']]
            q[n] = (tu, tv, p, np.c_[tu, tv, np.ones(len(P))])
        acc = ((np.abs(q[1][0]) < TAN_MAX) & (np.abs(q[1][1]) < TAN_MAX)
               & (np.abs(q[2][0]) < TAN_MAX) & (np.abs(q[2][1]) < TAN_MAX)
               & (P.dw1.to_numpy() > 0) & (P.dw2.to_numpy() > 0))
        # separation at a common depth (mid-gap)
        def at(n, w):
            tu, tv, p, _ = q[n]
            return p[:, 0] + tu * (w - p[:, 2]), p[:, 1] + tv * (w - p[:, 2])
        u1, v1 = at(1, 15.0)
        u2, v2 = at(2, 15.0)
        wsep = np.nan_to_num(sep(u1 - u2, v1 - v2))
        p1, r1 = q[1][2], q[1][3]
        p2, r2 = q[2][2], q[2][3]
        w0 = p1 - p2
        aa, bb, cc = (r1 * r1).sum(1), (r1 * r2).sum(1), (r2 * r2).sum(1)
        dd, ee = (r1 * w0).sum(1), (r2 * w0).sum(1)
        den = np.where(aa * cc - bb * bb > 1e-12, aa * cc - bb * bb, np.nan)
        sc, tc = (bb * ee - cc * dd) / den, (aa * ee - bb * dd) / den
        c1, c2 = p1 + sc[:, None] * r1, p2 + tc[:, None] * r2
        dca = np.linalg.norm(c1 - c2, axis=1)
        mid = 0.5 * (c1 + c2)
        rel = mid - a0
        vr = np.linalg.norm(rel - (rel @ adir)[:, None] * adir[None, :], axis=1)
        vy = rel @ adir
        vtx = (dca < VERTEX_DCA_MM) & (vr < VERTEX_R_MM)
        th = np.degrees(np.arccos(np.clip(bb / np.sqrt(aa * cc), -1, 1)))
        out.append(pd.DataFrame(dict(i=np.arange(len(P)), rep=r, acc=acc, w=wsep * acc,
                                     vtx=vtx & acc, open_reco=th, vy=vy)))
    S = pd.concat(out, ignore_index=True)
    g = S.groupby('i')
    return pd.DataFrame(dict(p_acc=g.w.mean(), p_vtx=(S.w * S.vtx).groupby(S.i).mean(),
                             open_reco=g.open_reco.median(),
                             vy_med=S[S.vtx].groupby('i').vy.median()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--parts', default=str(Path.home() / 'x17_local' / 'g4_same_arm' / 'parts'))
    ap.add_argument('--data', default=str(Path.home() / 'x17_local' / 'same_chamber'))
    ap.add_argument('--out', default=str(paths.out('same_chamber')))
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    L, C = load(Path(a.parts))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    n_trig = C['trig.events']
    summary, allP, allS = {}, [], []
    for letter, arm in SIM_ARM.items():
        R, t, resid = frame(L, arm)
        P = pairs(L, arm, R, t)
        S = singles(L, arm, R)
        # singles: per trigger, track-like, |tan| < 0.6 with smearing (one replica)
        res = SP._resolution(letter)
        tu = SP._smear(S.tu.to_numpy(float), 'x', res, rng)
        tv = SP._smear(S.tv.to_numpy(float), 'y', res, rng)
        S['acc'] = (np.abs(tu) < TAN_MAX) & (np.abs(tv) < TAN_MAX)
        n_single_trig = int((S.trig & S.acc).sum())
        Rsp = response(P, letter, R, t, rng) if len(P) else pd.DataFrame()
        P = P.join(Rsp)
        P['arm'] = letter
        S['armL'] = letter
        allP.append(P)
        allS.append(S[['file', 'event', 'op', 'trig', 'acc', 'ke0', 'wall_eV', 'tu', 'tv']]
                    .assign(arm=letter))
        tp = P[P.trig]
        by = tp.groupby('cls').agg(n=('cls', 'size'), acc=('p_acc', 'sum'), vtx=('p_vtx', 'sum'),
                                   open_true_med=('open_true', 'median'),
                                   open_reco_med=('open_reco', 'median'))
        summary[letter] = dict(frame_resid_mm=resid, n_trig=n_trig,
                               singles_trig_acc=n_single_trig,
                               singles_per_trig=n_single_trig / n_trig,
                               pairs_trig=by.reset_index().to_dict('records'),
                               pairs_all_events=P.groupby('cls').size().to_dict())
        print(f'\n{letter}: frame residual {resid:.3f} mm; triggers {n_trig:,}; '
              f'triggered track-like singles in acceptance {n_single_trig:,} '
              f'({n_single_trig / n_trig:.3f}/trigger)')
        print(by.round(2).to_string())
    P = pd.concat(allP, ignore_index=True)
    S = pd.concat(allS, ignore_index=True)
    P.to_parquet(od / 'g4_pairs.parquet', index=False)
    S.to_parquet(od / 'g4_singles.parquet', index=False)
    json.dump(dict(counts=C, summary=summary, track_w_mm=TRACK_W_MM, n_smear=N_SMEAR,
                   parts=a.parts), open(od / 'g4_same_arm.json', 'w'), indent=1, default=float)
    print(f'\nwrote {od}/g4_pairs.parquet, g4_singles.parquet, g4_same_arm.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
