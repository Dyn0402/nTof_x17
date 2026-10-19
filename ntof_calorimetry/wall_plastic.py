#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wall_plastic.py -- does the scintillator stack (SiPM wall + plastic) measure
energy?  A proof-of-concept check for LNL and other facilities, reopened
2026-10-14 after PLAN.md C1-C4 closed the n_TOF physics use.

The idea: calibrate each layer ONCE, on cosmic muons (a MIP in 3 mm and in
20 mm of PVT), and then ask whether the two calibrated layers describe a
different population -- the beam's Compton and pair electrons (1-7 MeV) -- the
way Geant4 says they should.  Neither layer is tuned on the beam.

W1  WALL MIP SCALE.  run_149 beam-off muons (`mip_sample` cosmic), single
    track pointing inside a wall group (both ends fired) and a plastic bar
    that fired.  Estimator: the two ends' SUM (what the hardware discriminates)
    x cos.  Unbinned truncated Landau (x) Gauss fit (`landau.fit_trunc`), the
    Landau width held at xi/Delta_p for 3 mm.  mV/MeV = MPV / Bichsel Delta_p(3 mm).
    Checks: response along the bar (attenuation of the sum), per group, and
    linearity in path.  The plastic scale is C1's (`c1/calib_plastic_e.json`).
W2  BEAM ELECTRONS vs GEANT4.  Data: every gated single track >= 12 ms after
    the flash whose OWN arm fired the trigger (A, C).  E_plas = the larger
    calibrated bar, E_wall = the largest group sum.  Geant4: the 10^9-neutron
    thermal campaign reduced by MX17_Full_Geant scripts/stack_edep.py (TOTAL
    prompt deposits per strip and bar, every particle), one track-like
    lepton in the arm's gap, the same trigger in MeV, each layer smeared by its
    measured resolution.  Compared: the plastic spectrum (shape above the
    trigger), the wall spectrum, and <E_wall | E_plas>; and the scale factor
    that would make Geant4 match the data, per layer.

    python -m ntof_calorimetry.wall_plastic          # -> OUT/wp/*.csv, summary.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry import mip_sample as MS  # noqa: E402
from ntof_calorimetry.mip_sample import OUT  # noqa: E402
from ntof_scint_stack.extract import WALL_THR, PLAS_THR  # noqa: E402
from sept26_prelim_analysis import paths  # noqa: E402

WP = OUT / 'wp'
WALL_MM = 3.0
BG = 30.0
MIP_W = LD.mpv(WALL_MM, BG)                       # MeV, Bichsel Delta_p
XI_W = 0.5 * LD.K * LD.MATERIALS['PVT'][1] * LD.MATERIALS['PVT'][0] * WALL_MM / 10
XI_OVER_MPV_W = XI_W / MIP_W
MIP_P = LD.mpv(20.0, BG)
#: Geant4 muon guns (1 GeV mu-, capsule -> arm 0, normal and tilted; condor
#: 4454427), reduced per arm by the calorimetry reduce_edep.py.  Every MeV
#: scale in the data-vs-Geant4 comparison is anchored to THESE muons, fitted
#: with the data's own fitter after smearing to each counter's measured muon
#: width (`anchors`), so data and simulation share one MIP definition.  (The
#: 0.4751 / 3.3494 MeV of MX17_Full_Geant analysis/mip_2cm are a different
#: definition: the same muons give 0.462 / 3.398 with this fitter, unsmeared.)
G4_MUONS = Path('/media/dylan/data/x17/calorimetry/wp_g4/muons')
MIP_W0, MIP_P0 = 0.462, 3.40     # MeV: only the scale of the sqrt(E) smearing
MUON_COS = {'th90_ph0': 1.0, 'th90_ph30': np.cos(np.radians(30)), 'th60_ph0': np.cos(np.radians(30))}
EDGE_W = 15.0
LATE_MS = 12.0
ARMS = ('A', 'C')
SIM_ARM = {'A': 2, 'C': 3}
G4_PARTS = Path('/media/dylan/data/x17/calorimetry/wp_g4/parts')
WALL_EDGES = np.array([-225.0, -125.0, -25.0, 75.0, 175.0])
EP_BINS = np.arange(0.0, 10.01, 0.25)
EW_BINS = np.arange(0.0, 2.51, 0.05)
EP_FIT = (3.0, 8.0)
LATE_PLATEAU_MS = 40.0
#: which Geant4 bar (BackScintL) is data bar 1, per arm.  Pinned on A by the
#: bar-1 / bar-2 ratio of the electron spectra (data 1.063 at every quantile,
#: Geant4 L/R 1.06-1.07); C faces A, so its left and right mirror (Geant4's L
#: is C's softer bar, as data bar 2 is).
G4_BAR_IS_L = {('A', 1): True, ('A', 2): False, ('C', 1): False, ('C', 2): True}       # the plastic gain recovers from the flash-rate sag by ~40 ms          # MeV: above every bar's trigger after smearing
EW_FIT = (0.2, 1.6)


def plastic_cal() -> dict:
    return json.loads((OUT / 'c1' / 'calib_plastic_e.json').read_text())['bars']


# --------------------------------------------------------------------------- #
# W1
# --------------------------------------------------------------------------- #
def wall_sample() -> pd.DataFrame:
    D = MS.load('cosmic')
    m = ((D.n_trk == 1) & (D.grp >= 0) & (D.d_edge_w > EDGE_W) & D.wboth_on & D.pm_on & (D.bar > 0))
    D = D[m].copy()
    g = D.grp.to_numpy().astype(int)
    a1 = np.choose(g, [D[f'w{2 * k + 1}_amp_on'].to_numpy() for k in range(4)])
    a2 = np.choose(g, [D[f'w{2 * k + 2}_amp_on'].to_numpy() for k in range(4)])
    D['a1'], D['a2'] = a1, a2
    D['s_mv'] = (a1 + a2) * D.cos
    D['path_mm'] = WALL_MM / D.cos
    # the floor: where this arm's trigger needed its wall (cosmic triggers in run_149
    # are the arm's own; `scint_ecal.trigger_floor` logic for the wall sum)
    own = D.ntof_arm.astype(str) == D.arm
    D['t_floor'] = np.where(own, D.arm.map(WALL_THR).to_numpy(float) * D.cos, 0.0)
    return D


def _fit_wall(x: pd.DataFrame) -> dict:
    rough = float(np.median(x.s_mv))
    t = np.maximum(x.t_floor.to_numpy(), 0.45 * rough)
    return LD.fit_trunc(x.s_mv.to_numpy(), t, 2.6 * rough, xi_ratio=XI_OVER_MPV_W, n_boot=30)


def wall_mip(D: pd.DataFrame) -> tuple:
    rows, prof, path = [], [], []
    for arm in ('A', 'B', 'C', 'D'):
        x = D[D.arm == arm]
        if len(x) < 80:
            continue
        f = _fit_wall(x)
        rows.append(dict(arm=arm, grp=-1, n_sel=len(x), **f, mv_per_mev=f['mpv'] / MIP_W,
                         thr_mev=WALL_THR[arm] / (f['mpv'] / MIP_W), cos_med=float(x.cos.median())))
        for g in range(4):
            xg = x[x.grp == g]
            if len(xg) >= 60:
                fg = _fit_wall(xg)
                rows.append(dict(arm=arm, grp=g, n_sel=len(xg), **fg, mv_per_mev=fg['mpv'] / MIP_W))
        # along the bar: median of the sum and of each end, relative
        for lo in np.arange(-250, 250, 50.0):
            xv = x[(x.v_w >= lo) & (x.v_w < lo + 50)]
            if len(xv) >= 25:
                prof.append(dict(arm=arm, v=lo + 25, n=len(xv), sum_rel=float(np.median(xv.s_mv) / f['mpv']),
                                 e1_rel=float(np.median(xv.a1 * xv.cos) / (0.5 * f['mpv'])),
                                 e2_rel=float(np.median(xv.a2 * xv.cos) / (0.5 * f['mpv']))))
        # in path: raw (no cos) MPV vs path, against Bichsel
        xx = x.assign(ab=pd.qcut(x.path_mm, 3, duplicates='drop'))
        for _, gq in xx.groupby('ab', observed=True):
            gr = gq.assign(s_mv=gq.a1 + gq.a2, t_floor=gq.t_floor / gq.cos)
            fq = _fit_wall(gr)
            pm = float(gq.path_mm.median())
            path.append(dict(arm=arm, path_mm=pm, n=len(gq), mpv_mv=fq['mpv'], err=fq['mpv_err'],
                             mpv_rel=fq['mpv'] / f['mpv'], exp_rel=LD.mpv(pm, BG) / MIP_W))
    return pd.DataFrame(rows), pd.DataFrame(prof), pd.DataFrame(path)


def g4_muons() -> tuple:
    """Geant4 muon deposits x cos: (wall, all guns), (plastic, normal-incidence
    guns only -- the tilted guns miss the bars)."""
    w, p = [], []
    for f in sorted(G4_MUONS.glob('mu_*.csv.gz')):
        if f.name.endswith('_events.csv.gz'):
            continue
        c = MUON_COS[next(k for k in MUON_COS if k in f.name)]
        h = pd.read_csv(f)
        h = h[h.armID == 0]
        w.append(h.e_wall.to_numpy(float) * c)
        pl = (h.e_plas_L + h.e_plas_R).to_numpy(float)
        if 'th90_ph0' in f.name:
            p.append(pl)
    w, p = np.concatenate(w), np.concatenate(p)
    return w[w > 0.05], p[p > 0.3]


def matched_anchor(g4: np.ndarray, target: float, mip0: float, fit, n: int = 20000, seed: int = 1) -> dict:
    """The MIP of a counter, defined exactly as on the data: Geant4's muons,
    smeared (sigma = r sqrt(E mip0)) until the SAME fit returns the data's
    sigma/MPV, then fitted.  With the Landau width fixed, the fitted MPV moves
    with the smearing (+2-3 % from Geant4's bare 0.06 to the data's 0.16-0.22
    on the wall), so the anchor must carry the data's width."""
    rng = np.random.default_rng(seed)
    x0 = g4[:n]
    z = rng.normal(0, 1, len(x0))
    best = None
    for r in np.arange(0.0, 0.32, 0.02):
        x = np.clip(x0 + z * r * np.sqrt(np.clip(x0, 0, None) * mip0), 0, None)
        f = fit(x)
        d = abs(f['sigma'] / f['mpv'] - target)
        if best is None or d < best[0]:
            best = (d, r, f)
    d, r, f = best
    return dict(anchor=f['mpv'], r=float(r), sig_over_mpv=f['sigma'] / f['mpv'], target=target)


def _fit_wall_g4(x):
    rw = float(np.median(x))
    return LD.fit_trunc(x, np.full(len(x), 0.45 * rw), 2.6 * rw, xi_ratio=XI_OVER_MPV_W, n_boot=3)


def _fit_plas_g4(x):
    from ntof_calorimetry.scint_ecal import FLOOR, HI, RES_PRIOR
    rp = float(np.median(x))
    return LD.fit_trunc(x, np.full(len(x), FLOOR * rp), HI * rp, n_boot=3, res_prior=RES_PRIOR)


def anchors(Wf: pd.DataFrame) -> pd.DataFrame:
    """Per wall arm and per plastic bar: the matched Geant4 anchor (MeV)."""
    gw, gp = g4_muons()
    rows = []
    for r in Wf[Wf.grp == -1].itertuples():
        a = matched_anchor(gw, r.sigma / r.mpv, MIP_W0, _fit_wall_g4)
        rows.append(dict(layer='wall', ch=r.arm, **a, g4_bare=float(_fit_wall_g4(gw[:20000])['mpv'])))
    pc = plastic_cal()
    bare_p = float(_fit_plas_g4(gp[:20000])['mpv'])
    for ch, c in pc.items():
        a = matched_anchor(gp, c['resolution_sigma_over_mpv'], MIP_P0, _fit_plas_g4)
        rows.append(dict(layer='plastic', ch=ch, **a, g4_bare=bare_p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# W2
# --------------------------------------------------------------------------- #
STACK_COLS = (['run', 'event_id', 'arm', 'n_trk', 't_since_flash_ns', 'hw_A', 'hw_C', 'u_mm', 'v_mm',
               'tan_raw_x', 'tan_v_raw', 'k_arm', 'p1_amp_on', 'p2_amp_on', 'p1_sat_on', 'p2_sat_on']
              + [f'w{i}_amp_on' for i in range(1, 9)])


def beam_data(cal_w: dict, anc_p: dict) -> pd.DataFrame:
    """``cal_w``: wall mV/MeV per arm; ``anc_p``: plastic anchor (MeV) per bar --
    C1's mV/MeVee was set with Bichsel's MPV, re-anchored here to Geant4's."""
    pc = plastic_cal()
    parts = []
    for f in sorted(glob.glob(str(paths.spell('scint', 'tracks', 'stack_run_*.parquet')))):
        t = pd.read_parquet(f, columns=STACK_COLS)
        t = t[t.arm.isin(ARMS) & (t.n_trk == 1) & (t.t_since_flash_ns >= LATE_MS * 1e6)]
        t = t[np.where(t.arm == 'A', t.hw_A, t.hw_C).astype(bool)]
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T['arm'] = T.arm.astype(str)
    mvpm = {f'{a}{b}': pc[f'PSS{a}{b}']['mip_mpv_mv'] / anc_p[f'PSS{a}{b}'] for a in ARMS for b in (1, 2)}
    e1 = np.nan_to_num(T.p1_amp_on.to_numpy()) / T.arm.map(lambda a: mvpm[f'{a}1']).to_numpy()
    e2 = np.nan_to_num(T.p2_amp_on.to_numpy()) / T.arm.map(lambda a: mvpm[f'{a}2']).to_numpy()
    T['e_plas'] = np.fmax(e1, e2)
    T['e_plas_raw'] = T.e_plas
    T['bar'] = np.where(e1 >= e2, 1, 2)
    T['p_sat'] = np.where(T.bar == 1, np.nan_to_num(T.p1_sat_on), np.nan_to_num(T.p2_sat_on)) > 0
    sums = np.stack([np.nan_to_num(T[f'w{2 * g + 1}_amp_on'].to_numpy())
                     + np.nan_to_num(T[f'w{2 * g + 2}_amp_on'].to_numpy()) for g in range(4)], 1)
    T['e_wall'] = sums.max(1) / T.arm.map(cal_w).to_numpy()
    T['wgrp'] = sums.argmax(1)
    # predicted plastic crossing (the scint-stack's calibrated pointing predictor)
    from ntof_scint_stack import ana as SA
    cal = json.loads(paths.spell('scint', 'ana', 'meta.json').read_text())['cal']
    T['u_p'], T['v_p'] = np.nan, np.nan
    for arm in ARMS:
        m = (T.arm == arm).to_numpy()
        t = T[m].assign(k_eff=T.loc[m, 'k_arm'].fillna(T.loc[m, 'k_arm'].median()))
        g, c = SA.geometry(arm), cal[arm]
        ax, mx, ay, my = SA.pointing(t, g, arm)
        T.loc[m, 'u_p'] = t.u_mm.to_numpy() + g['L_plas'] * (c['alpha'] * ax + c['lam'] * mx) - c['off_p']
        T.loc[m, 'v_p'] = t.v_mm.to_numpy() + g['L_plas'] * c['lam_v'] * my
    return T.drop(columns=[c for c in T.columns if c.endswith('_amp_on')])


FACE_U = np.arange(-250.0, 250.01, 25.0)
FACE_V = np.arange(-150.0, 150.01, 50.0)


def face_correct(T: pd.DataFrame) -> tuple:
    """The plastic bars' response across their face, and the beam electrons
    put on the muon-calibrated scale where they land.

    Each bar has one PMT, and its light collection varies across the face.
    The muon calibration is the bar AVERAGE over where muons cross; the beam
    electrons land mostly on the inner halves (toward the capsule).  So:
      R(cell)  = the electrons' median plastic energy in a 25 x 50 mm cell /
                 the bar's median (relative response; assumes the electron
                 spectrum does not change across the face -- A's flat bars
                 bound that at the few-% level);
      R_mu     = R averaged over the cosmic muons' crossings of that bar
                 (the position mix the calibration was made on);
      e_corr   = e_plas x R_mu / R(cell).
    Returns (T with e_plas_raw / e_plas corrected, the map)."""
    D = MS.load('cosmic')
    from ntof_calorimetry.scint_ecal import mip_selection
    D = D[mip_selection(D)]
    T = T.copy()
    T['e_plas_raw'] = T.e_plas
    iu = np.digitize(T.u_p, FACE_U) - 1
    iv = np.digitize(T.v_p, FACE_V) - 1
    T['cell'] = np.where((iu >= 0) & (iu < len(FACE_U) - 1) & (iv >= 0) & (iv < len(FACE_V) - 1),
                         iu * 100 + iv, -1)
    rows = []
    corr = np.ones(len(T))
    for arm in ARMS:
        for bar in (1, 2):
            m = ((T.arm == arm) & (T.bar == bar) & (T.cell >= 0) & ~T.p_sat).to_numpy()
            med_bar = float(np.median(T.e_plas[m]))
            g = T[m].groupby('cell').e_plas.agg(['median', 'size'])
            g = g[g['size'] >= 300]
            R = (g['median'] / med_bar).to_dict()
            dm = D[(D.arm == arm) & (D.bar == bar)]
            mu_iu = np.digitize(dm.u_p, FACE_U) - 1
            mu_iv = np.digitize(dm.v_p, FACE_V) - 1
            rmu = [R.get(a * 100 + b, np.nan) for a, b in zip(mu_iu, mu_iv)]
            rmu = np.array(rmu, float)
            R_mu = float(np.nanmean(rmu))
            cover = float(np.isfinite(rmu).mean())
            mm = (T.arm == arm).to_numpy() & (T.bar == bar).to_numpy()
            rc = T.cell[mm].map(R).to_numpy(float)
            corr[mm] = np.where(np.isfinite(rc), R_mu / rc, np.nan)
            for c, r in R.items():
                rows.append(dict(arm=arm, bar=bar, u=FACE_U[c // 100] + 12.5, v=FACE_V[c % 100] + 25.0, R=r,
                                 n=int(g.loc[c, 'size'])))
            rows.append(dict(arm=arm, bar=bar, u=np.nan, v=np.nan, R=R_mu, n=len(dm), muon_cover=cover))
    T['e_plas'] = T.e_plas_raw * corr
    return T[np.isfinite(T.e_plas)], pd.DataFrame(rows)


def g4_sample(cal_w_res: dict, rng, anc_p: dict) -> pd.DataFrame:
    """Geant4 rows in the data's selection: arm A or C, one track-like lepton
    in the gap, the arm's own trigger on smeared totals in MeV."""
    pc = plastic_cal()
    fs = sorted(G4_PARTS.glob('*.parquet'))
    R = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    inv = {v: k for k, v in SIM_ARM.items()}
    R = R[R.arm.isin(SIM_ARM.values()) & (R.n_lep == 1) & (R.lead_w_span >= 20.0)].copy()
    R['armL'] = R.arm.map(inv)
    W = R[[f'w{c:02d}' for c in range(20)]].to_numpy(float)
    ucen = -250.0 + 25.0 * (np.arange(20) + 0.5)
    gi = np.digitize(ucen, WALL_EDGES) - 1
    G = np.stack([W[:, gi == g].sum(1) for g in range(4)], 1)
    out = []
    for arm, x in R.groupby('armL'):
        Gx = G[R.armL.to_numpy() == arm]
        rw = cal_w_res[arm]
        sw = Gx + rng.normal(0, 1, Gx.shape) * rw * np.sqrt(np.clip(Gx, 0, None) * MIP_W0)
        sw = np.clip(sw, 0, None)
        pl = []
        for b, col in ((1, 'pL'), (2, 'pR')):
            e = x[col].to_numpy(float)
            r = pc[f'PSS{arm}{b}']['resolution_sigma_over_mpv']
            pl.append(np.clip(e + rng.normal(0, 1, len(e)) * r * np.sqrt(np.clip(e, 0, None) * MIP_P0), 0, None))
        thr_w = WALL_THR[arm] / cal_w_res[f'{arm}_mvpm']
        tp = [PLAS_THR[arm] / (pc[f'PSS{arm}{b}']['mip_mpv_mv'] / anc_p[f'PSS{arm}{b}']) for b in (1, 2)]
        trig = (sw.max(1) >= thr_w) & ((pl[0] >= tp[0]) | (pl[1] >= tp[1]))
        y = x.assign(e_plas=np.fmax(pl[0], pl[1]), e_wall=sw.max(1), e_plas_true=np.fmax(x.pL, x.pR),
                     e_wall_true=Gx.max(1), trig=trig)
        out.append(y[y.trig])
    return pd.concat(out, ignore_index=True)


def hist(v, bins, lo, hi):
    n, _ = np.histogram(v, bins)
    c = 0.5 * (bins[1:] + bins[:-1])
    m = (c >= lo) & (c < hi)
    return n, c, m


def _match(data: np.ndarray, sim: np.ndarray, lo: float, hi: float) -> float:
    """Scale s with mean(s E_sim | s E_sim in [lo, hi)) = mean(E_data | in [lo, hi))."""
    md = float(np.mean(data[(data >= lo) & (data < hi)]))
    a, b = 0.6, 1.6
    for _ in range(40):
        s = 0.5 * (a + b)
        x = s * sim
        ms = float(np.mean(x[(x >= lo) & (x < hi)]))
        a, b = (s, b) if ms < md else (a, s)
    return 0.5 * (a + b)


def scale_fit(data: np.ndarray, sim: np.ndarray, bins, lo, hi, n_boot: int = 200, seed: int = 3):
    """Scale s with Geant4 x s matching the data in [lo, hi) (truncated means,
    bootstrap error on BOTH samples), and a KS test of the shapes at s = 1 and
    at the fitted s."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    data, sim = np.asarray(data, float), np.asarray(sim, float)
    s0 = _match(data, sim, lo, hi)
    dsub = data if len(data) <= 200_000 else rng.choice(data, 200_000, replace=False)
    bs = [_match(rng.choice(dsub, len(dsub)), rng.choice(sim, len(sim)), lo, hi) for _ in range(n_boot)]
    w = lambda x: x[(x >= lo) & (x < hi)]
    ks1 = ks_2samp(w(dsub), w(sim))
    kss = ks_2samp(w(dsub), w(sim * s0))
    return dict(scale=s0, err=float(np.std(bs)), ks_d_at_1=float(ks1.statistic), ks_p_at_1=float(ks1.pvalue),
                ks_d_fit=float(kss.statistic), ks_p_fit=float(kss.pvalue), n_g4_win=int(len(w(sim))))


def res_scale_fit(data: np.ndarray, true: np.ndarray, mip: float, lo: float, hi: float,
                  seed: int = 11) -> pd.DataFrame:
    """Scan the extra Gaussian smearing r (sigma = r sqrt(E mip)) applied to the
    UNSMEARED Geant4 deposit; at each r the scale from `_match`, and the KS
    distance of the shapes.  The muon fit's Gaussian width cannot be used for
    the wall: for a 3 mm layer it also absorbs straggling beyond the Landau
    (fixed xi) that Geant4 already simulates, so applying it double counts."""
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, len(true))
    dsub = data if len(data) <= 200_000 else rng.choice(data, 200_000, replace=False)
    rows = []
    for r in np.arange(0.0, 0.205, 0.01):
        x = np.clip(true + z * r * np.sqrt(np.clip(true, 0, None) * mip), 0, None)
        s = _match(data, x, lo, hi)
        w = lambda v: v[(v >= lo) & (v < hi)]
        k = ks_2samp(w(dsub), w(x * s))
        rows.append(dict(r=float(r), scale=s, ks_d=float(k.statistic), ks_p=float(k.pvalue)))
    return pd.DataFrame(rows)


def compare(T: pd.DataFrame, S: pd.DataFrame) -> tuple:
    """Spectra, scale fits and <E_wall | E_plas>, data vs Geant4.  For the
    wall, Geant4 is shown with the extra smearing FITTED on the beam data
    (``e_wall_fit``, see `res_scale_fit`); ``e_wall`` (the muon fit's
    Gaussian applied) is kept as the reference."""
    scans = []
    S = S.copy()
    for arm in ARMS:
        d = T[(T.arm == arm) & ~T.p_sat & (T.e_plas >= EP_FIT[0])]
        m = (S.armL == arm).to_numpy()
        tr = S.loc[m, 'e_wall_true'].to_numpy(float)
        sc = res_scale_fit(d.e_wall.to_numpy(), tr[S.loc[m, 'e_plas'].to_numpy() >= EP_FIT[0]], MIP_W0, *EW_FIT)
        scans.append(sc.assign(arm=arm))
        bst = sc.loc[sc.ks_d.idxmin()]
        z = np.random.default_rng(13).normal(0, 1, len(tr))
        S.loc[m, 'e_wall_fit'] = np.clip(tr + z * bst.r * np.sqrt(np.clip(tr, 0, None) * MIP_W0), 0, None)
        S.loc[m, 'r_fit'] = bst.r
    spec, prof, fits = [], [], []
    for arm in ARMS:
        d = T[(T.arm == arm) & ~T.p_sat]
        s = S[S.armL == arm]
        hi_p = d.e_plas >= EP_FIT[0]
        hs_p = s.e_plas >= EP_FIT[0]
        for var, svar, bins, dsel, ssel in (('e_plas', 'e_plas', EP_BINS, None, None),
                                            ('e_wall', 'e_wall', EW_BINS, hi_p, hs_p),
                                            ('e_wall_fit', 'e_wall_fit', EW_BINS, hi_p, hs_p)):
            dv = d[var if var.startswith('e_plas') else 'e_wall']
            sv = s[svar]
            if dsel is not None:
                dv, sv = dv[dsel], sv[ssel]
            nd, _ = np.histogram(dv, bins)
            ns, _ = np.histogram(sv, bins)
            for lo, hi, a, b in zip(bins[:-1], bins[1:], nd, ns):
                spec.append(dict(arm=arm, var=var, lo=lo, hi=hi, data=int(a), g4=int(b)))
            lo, hi = EP_FIT if var.startswith('e_plas') else EW_FIT
            f = scale_fit(dv.to_numpy(), sv.to_numpy(float), bins, lo, hi)
            fits.append(dict(arm=arm, var=var, n_data=int(len(dv)), n_g4=int(len(sv)),
                             r_fit=float(s.r_fit.iloc[0]) if var == 'e_wall_fit' else np.nan, **f))
        # subsets of the plastic: per bar (against Geant4's two bars pooled: which
        # Geant4 bar is which data bar is not pinned), and the rate-free plateau
        late = d.t_since_flash_ns >= LATE_PLATEAU_MS * 1e6
        sL = (s.pL >= s.pR).to_numpy()
        g4bar = {b: (sL if G4_BAR_IS_L[(arm, b)] else ~sL) for b in (1, 2)}
        for sub, dm, sm in (('bar 1', d.bar == 1, g4bar[1]), ('bar 2', d.bar == 2, g4bar[2]),
                            (f'>= {LATE_PLATEAU_MS:g} ms', late, np.ones(len(s), bool)),
                            (f'bar 1, >= {LATE_PLATEAU_MS:g} ms', (d.bar == 1) & late, g4bar[1]),
                            (f'bar 2, >= {LATE_PLATEAU_MS:g} ms', (d.bar == 2) & late, g4bar[2])):
            f = scale_fit(d.e_plas[dm].to_numpy(), s.e_plas[sm].to_numpy(float), EP_BINS, *EP_FIT, n_boot=100)
            fits.append(dict(arm=arm, var='e_plas', subset=sub, n_data=int(dm.sum()), n_g4=int(sm.sum()), **f))
        f = scale_fit(d.e_wall[hi_p & late].to_numpy(), s.e_wall_fit[hs_p].to_numpy(float), EW_BINS, *EW_FIT, n_boot=100)
        fits.append(dict(arm=arm, var='e_wall_fit', subset=f'>= {LATE_PLATEAU_MS:g} ms', n_data=int((hi_p & late).sum()),
                         n_g4=int(hs_p.sum()), **f))
        sc = [f for f in fits if f['arm'] == arm and f['var'] == 'e_wall_fit'][0]['scale']
        pb = np.arange(2.0, 8.01, 0.5)
        for lo, hi in zip(pb[:-1], pb[1:]):
            md = d[(d.e_plas >= lo) & (d.e_plas < hi) & (d.e_wall > 0)]
            ms = s[(s.e_plas >= lo) & (s.e_plas < hi) & (s.e_wall_fit > 0)]
            if len(md) >= 30 and len(ms) >= 10:
                w = ms.e_wall_fit * sc
                prof.append(dict(arm=arm, lo=lo, hi=hi, n_data=len(md), n_g4=len(ms),
                                 wall_med_data=float(md.e_wall.median()), wall_med_g4=float(w.median()),
                                 wall_q16_data=float(md.e_wall.quantile(0.16)), wall_q84_data=float(md.e_wall.quantile(0.84)),
                                 wall_q16_g4=float(w.quantile(0.16)), wall_q84_g4=float(w.quantile(0.84))))
    return pd.DataFrame(spec), pd.DataFrame(prof), pd.DataFrame(fits), pd.concat(scans, ignore_index=True)


def main() -> int:
    WP.mkdir(parents=True, exist_ok=True)
    D = wall_sample()
    Wf, Wprof, Wpath = wall_mip(D)
    Wf.to_csv(WP / 'wall_mip.csv', index=False)
    Wprof.to_csv(WP / 'wall_along_bar.csv', index=False)
    Wpath.to_csv(WP / 'wall_path.csv', index=False)
    D[['arm', 'grp', 'cos', 's_mv', 't_floor', 'v_w', 'pa_on']].to_parquet(WP / 'wall_cosmic.parquet')
    print(Wf.round(3).to_string())
    print(Wpath.round(3).to_string())
    A = anchors(Wf)
    A.to_csv(WP / 'anchors.csv', index=False)
    print(A.round(4).to_string())
    aw = A[A.layer == 'wall'].set_index('ch').anchor.to_dict()
    ap = A[A.layer == 'plastic'].set_index('ch').anchor.to_dict()
    arm_rows = Wf[Wf.grp == -1].set_index('arm')
    cal_w = {a: float(arm_rows.loc[a, 'mpv'] / aw[a]) for a in arm_rows.index}
    res = {a: float(arm_rows.loc[a, 'sigma'] / arm_rows.loc[a, 'mpv']) for a in arm_rows.index}
    T = beam_data(cal_w, ap)
    _, FM = face_correct(T)            # diagnostic only: flat to ~5 % (crossings smeared ~50 mm)
    FM.to_csv(WP / 'face_map.csv', index=False)
    rb = [12, 15, 20, 25, 30, 40, 50, 60, 80]
    rp = (T[~T.p_sat].assign(mb=pd.cut(T.t_since_flash_ns / 1e6, rb))
          .groupby(['arm', 'bar', 'mb'], observed=True)
          .agg(n=('e_plas_raw', 'size'), plas_med=('e_plas_raw', 'median'), wall_med=('e_wall', 'median')).reset_index())
    rp['lo'] = rp.mb.map(lambda i: i.left).astype(float)
    rp['hi'] = rp.mb.map(lambda i: i.right).astype(float)
    rp.drop(columns='mb').to_csv(WP / 'rate_profile.csv', index=False)
    T.to_parquet(WP / 'beam_data.parquet')
    summ = dict(mip_wall_bichsel=MIP_W, mip_plas_bichsel=MIP_P, xi_over_mpv_wall=XI_OVER_MPV_W,
                anchor_wall=aw, anchor_plas=ap, wall_mv_per_mev=cal_w, wall_res_sigma_over_mpv=res,
                n_beam={a: int((T.arm == a).sum()) for a in ARMS})
    if G4_PARTS.exists() and any(G4_PARTS.glob('*.parquet')):
        rng = np.random.default_rng(7)
        S = g4_sample({**res, **{f'{a}_mvpm': cal_w[a] for a in cal_w}}, rng, ap)
        S.drop(columns=[f'w{c:02d}' for c in range(20)]).to_parquet(WP / 'g4_sample.parquet')
        spec, prof, fits, scans = compare(T, S)
        scans.to_csv(WP / 'wall_res_scan.csv', index=False)
        spec.to_csv(WP / 'spectra.csv', index=False)
        prof.to_csv(WP / 'wall_vs_plastic.csv', index=False)
        fits.to_csv(WP / 'scale_fits.csv', index=False)
        summ['n_g4'] = {a: int((S.armL == a).sum()) for a in ARMS}
        print(fits.round(3).to_string())
        print(prof.round(3).to_string())
    json.dump(summ, open(WP / 'summary.json', 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
