#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_edep_combo.py -- MM gap deposit together with the wall and plastic
(MM_EDEP.md, stage 4): what the measured MM deposit adds to the scintillators'
picture of a track, predicted from Geant4 and set against run_145.

SIMULATION.  The Geant4 neutron-beam events (MX17_Full_Geant neutron_bg_job*)
in two reductions of the SAME files, joined per (file, event, arm):
  wp_g4/parts         wall strips, plastic bars, the leading gap lepton's
                      kinetic energy at the gap (`stack_edep.py`)
  g4_angle/neutrons   the gap deposit of ALL particles (edep_gap), the
                      dominant track's share, the gap slopes
The gap deposit is taken two ways: Geant4's own (opt4, Ar/iso 95/5, scattering
and every particle in the gap), and Heed's at the leading lepton's energy
(the better thin-gas straggling, straight track, one particle).  Then the
measured MM resolution (stage 2-3, lognormal) and the counters' measured
resolutions (`wall_plastic` anchors) are applied, and each event is put in the
category the data uses: plastic fired / wall only / neither.

DATA.  run_145 >= 12 ms, chamber A in-window tracks (`mm_edep_data`), the
same categories from the scint-stack pulses (WALL_THR / PLAS_THR mV).

ENERGY.  How well the track's kinetic energy at the gap is estimated from
(wall, plastic) with and without the MM deposit: k-nearest-neighbour
regression, 5-fold cross-validated, on the Geant4 sample -- and the same on
the single-electron runs (C2) for the X17 range, with the liquid.
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

from ntof_calorimetry.mip_sample import OUT  # noqa: E402

EDEP = OUT / 'mm_edep'
HEED = OUT / 'heed'
PARTS = OUT / 'wp_g4' / 'parts'
RED = Path('/media/dylan/data/x17/ntof_cosmics/g4_angle/neutrons_nose')
SIM_ARM = {'A': 2, 'C': 3}
WALL_EDGES = np.array([-225.0, -125.0, -25.0, 75.0, 175.0])
MIP_W0, MIP_P0 = 0.462, 3.40
CATS = ('plastic', 'wall only', 'none')


# ------------------------------------------------------------------ inputs
def scint_cal() -> dict:
    """Wall mV/MeV and resolution per arm, plastic MeV threshold and
    resolution per bar, from the wall_plastic / C1 products."""
    from ntof_scint_stack.extract import PLAS_THR, WALL_THR
    s = json.loads((OUT / 'wp' / 'summary.json').read_text())
    A = pd.read_csv(OUT / 'wp' / 'anchors.csv')
    pc = json.loads((OUT / 'c1' / 'calib_plastic_e.json').read_text())['bars']
    cal = {}
    for arm in SIM_ARM:
        rw = float(A[(A.layer == 'wall') & (A.ch == arm)].r.iloc[0])
        bars = {}
        for b in (1, 2):
            k = f'PSS{arm}{b}'
            mvpm = pc[k]['mip_mpv_mv'] / s['anchor_plas'][k]
            bars[b] = dict(thr=PLAS_THR[arm] / mvpm, r=float(A[(A.layer == 'plastic') & (A.ch == k)].r.iloc[0]))
        cal[arm] = dict(wall_thr=WALL_THR[arm] / s['wall_mv_per_mev'][arm], wall_r=rw, bars=bars)
    return cal


def g4_sample() -> pd.DataFrame:
    W, R = [], []
    for f in sorted(PARTS.glob('*.parquet')):
        w = pd.read_parquet(f)
        W.append(w[w.arm.isin(SIM_ARM.values())])
    for f in sorted(RED.glob('r*.parquet')):
        r = pd.read_parquet(f, columns=['eventID', 'arm', 'edep_gap', 'w_lo', 'w_hi', 'tan_gap_u', 'tan_gap_v',
                                        'dom_share', 'n_tracks_gap', 'dom_ke_gap'])
        r['file'] = 'neutron_bg_job' + f.stem[1:]
        R.append(r[r.arm.isin(SIM_ARM.values())])
    G = pd.concat(W).merge(pd.concat(R), left_on=['file', 'event', 'arm'], right_on=['file', 'eventID', 'arm'])
    # a track crossing the gap, as the data's reconstructed tracks do
    G = G[(G.w_hi - G.w_lo > 20) & (G.lead_w_span >= 20)].copy()
    G['armL'] = G.arm.map({v: k for k, v in SIM_ARM.items()})
    G['sec'] = np.sqrt(1 + G.tan_gap_u ** 2 + G.tan_gap_v ** 2)
    return G.reset_index(drop=True)


class HeedPools:
    """Gap ionisation (electrons) for an electron of kinetic energy T, normal
    incidence, 30 mm: drawn from the Heed pool at the nearest T (log)."""

    def __init__(self):
        self.T, self.n = [], []
        for f in sorted(HEED.glob('e-_*_th0.npz')):
            z = np.load(f)
            self.T.append(float(z['T']))
            self.n.append(z['n_e'].astype(float))
        o = np.argsort(self.T)
        self.T = np.array(self.T)[o]
        self.n = [self.n[i] for i in o]
        self.lT = np.log(self.T)

    def draw(self, T: np.ndarray, rng) -> np.ndarray:
        k = np.abs(np.log(np.clip(T, 0.05, 50))[:, None] - self.lT[None, :]).argmin(1)
        out = np.empty(len(T))
        for j in np.unique(k):
            m = k == j
            out[m] = rng.choice(self.n[j], m.sum())
        return out


def smear_scint(G: pd.DataFrame, cal: dict, rng) -> pd.DataFrame:
    Wst = G[[f'w{c:02d}' for c in range(20)]].to_numpy(float)
    ucen = -250.0 + 25.0 * (np.arange(20) + 0.5)
    gi = np.digitize(ucen, WALL_EDGES) - 1
    grp = np.stack([Wst[:, gi == g].sum(1) for g in range(4)], 1)
    ew, ep, cat = np.zeros(len(G)), np.zeros(len(G)), np.empty(len(G), object)
    for arm in SIM_ARM:
        m = (G.armL == arm).to_numpy()
        c = cal[arm]
        g = grp[m]
        sw = np.clip(g + rng.normal(0, 1, g.shape) * c['wall_r'] * np.sqrt(np.clip(g, 0, None) * MIP_W0), 0, None)
        pl, fired = [], np.zeros(m.sum(), bool)
        for b, col in ((1, 'pL'), (2, 'pR')):
            e = G.loc[m, col].to_numpy(float)
            e = np.clip(e + rng.normal(0, 1, len(e)) * c['bars'][b]['r'] * np.sqrt(np.clip(e, 0, None) * MIP_P0),
                        0, None)
            fired |= e > c['bars'][b]['thr']
            pl.append(e)
        ew[m], ep[m] = sw.max(1), np.fmax(*pl)
        cat[m] = np.where(fired, 'plastic', np.where(sw.max(1) > c['wall_thr'], 'wall only', 'none'))
    return G.assign(e_wall=ew, e_plas=ep, cat=cat)


def mm_deposits(G: pd.DataFrame, sigma_meas: float, rng, pools: HeedPools) -> pd.DataFrame:
    """Measured MM deposit per mm of depth, in units of the 1-MIP median, two
    ways.  Geant4: all particles' gap deposit / sec.  Heed: the leading
    lepton only.  Both x lognormal(sigma_meas)."""
    q_g4 = G.edep_gap.to_numpy(float) / G.sec.to_numpy()
    q_h = pools.draw(G.lead_ke0.to_numpy(float), rng)
    sm = lambda v: v * np.exp(rng.normal(0, sigma_meas, len(v)))  # noqa: E731
    return G.assign(q_g4=sm(q_g4), q_heed=sm(q_h))


# ------------------------------------------------------------- categories
def category_table(G: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for arm, g in G.groupby('armL'):
        m = g[g.cat == 'plastic'][col].median()
        for c in CATS:
            v = g[g.cat == c][col].to_numpy() / m
            x = g[g.cat == c]
            rows.append(dict(arm=arm, cat=c, n=len(v), med=np.median(v), q25=np.quantile(v, .25),
                             q75=np.quantile(v, .75), frac_gt15=float((v > 1.5).mean()),
                             frac_multi=float((x.dom_share < 0.8).mean()),
                             med_single=float(np.median(x[x.dom_share >= 0.8][col]) / m),
                             frac_slow=float((x.lead_ke0 < 1.0).mean())))
    return pd.DataFrame(rows)


def data_categories(D: pd.DataFrame) -> pd.DataFrame:
    """D: run_145 tracks with q (per mm of depth) and cat."""
    rows = []
    for arm, g in D.groupby('arm'):
        m = g[g.cat == 'plastic'].q.median()
        for c in CATS:
            v = g[g.cat == c].q.to_numpy() / m
            if not len(v):
                continue
            # bootstrap error on the median ratio
            rng = np.random.default_rng(5)
            b = [np.median(rng.choice(v, len(v))) for _ in range(300)]
            rows.append(dict(arm=arm, cat=c, n=len(v), med=np.median(v), med_err=float(np.std(b)),
                             q25=np.quantile(v, .25), q75=np.quantile(v, .75), frac_gt15=float((v > 1.5).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ energy
def knn_cv(X: np.ndarray, y: np.ndarray, k: int = 40, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Cross-validated kNN regression (mean of the neighbours' y), on
    standardised features."""
    from scipy.spatial import cKDTree
    Xs = (X - X.mean(0)) / X.std(0)
    rng = np.random.default_rng(seed)
    f = rng.integers(0, folds, len(y))
    out = np.empty(len(y))
    for i in range(folds):
        tr, te = f != i, f == i
        t = cKDTree(Xs[tr])
        _, nn = t.query(Xs[te], k=k)
        out[te] = np.mean(y[tr][nn], axis=1)
    return out


def width68(v: np.ndarray) -> float:
    return float(0.5 * np.subtract(*np.quantile(v, [.84, .16])))


def energy_beam(G: pd.DataFrame, col: str = 'q_g4') -> pd.DataFrame:
    """Kinetic energy at the gap from (wall, plastic) vs (wall, plastic, MM)."""
    rows = []
    y = G.lead_ke0.to_numpy(float)
    base = np.c_[np.log1p(G.e_wall), np.log1p(G.e_plas)]
    full = np.c_[base, np.log(np.clip(G[col], 1e-3, None))]
    pb, pf = knn_cv(base, y), knn_cv(full, y)
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 20]
    for c in ('all',) + CATS:
        m = np.ones(len(y), bool) if c == 'all' else (G.cat == c).to_numpy()
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (y >= lo) & (y < hi)
            if mm.sum() < 50:
                continue
            rows.append(dict(cat=c, T_lo=lo, T_hi=hi, n=int(mm.sum()),
                             w68_scint=width68(pb[mm] - y[mm]), w68_scint_mm=width68(pf[mm] - y[mm]),
                             bias_scint=float(np.median(pb[mm] - y[mm])),
                             bias_scint_mm=float(np.median(pf[mm] - y[mm]))))
        mm = m
        rows.append(dict(cat=c, T_lo=0, T_hi=99, n=int(mm.sum()),
                         w68_scint=width68(pb[mm] - y[mm]), w68_scint_mm=width68(pf[mm] - y[mm]),
                         bias_scint=float(np.median(pb[mm] - y[mm])), bias_scint_mm=float(np.median(pf[mm] - y[mm]))))
    return pd.DataFrame(rows)


def slow_tag(G: pd.DataFrame, col: str = 'q_g4', T_cut: float = 1.0) -> pd.DataFrame:
    """Among tracks with no plastic: separate T < T_cut at the gap from the
    rest, by the wall alone, the MM alone, and both (kNN probability)."""
    from scipy.stats import mannwhitneyu
    g = G[G.cat != 'plastic']
    y = (g.lead_ke0 < T_cut).to_numpy()
    rows = []
    for name, X in (('wall', np.c_[np.log1p(g.e_wall)]),
                    ('MM', np.c_[np.log(np.clip(g[col], 1e-3, None))]),
                    ('wall+MM', np.c_[np.log1p(g.e_wall), np.log(np.clip(g[col], 1e-3, None))])):
        p = knn_cv(X, y.astype(float), k=60)
        u = mannwhitneyu(p[y], p[~y]).statistic / (y.sum() * (~y).sum())
        rows.append(dict(features=name, n=len(y), frac_slow=float(y.mean()), auc=float(u)))
    return pd.DataFrame(rows)


def energy_c2(sigma_meas: float, rng) -> pd.DataFrame:
    """Single electrons from the capsule (C2), the X17 range: vertex T from
    (wall, plastic, liquid) vs + MM gas.  Counters smeared as on the data."""
    from ntof_calorimetry import g4_response as GR
    d = GR.load()
    d = d[(d.particle == 'e-') & (d.e_gas > 0)].copy()
    rw, rp = 0.14, 0.20
    d['w'] = np.clip(d.e_wall + rng.normal(0, 1, len(d)) * rw * np.sqrt(d.e_wall.clip(0) * MIP_W0), 0, None)
    pl = d.e_plas_L + d.e_plas_R
    d['p'] = np.clip(pl + rng.normal(0, 1, len(d)) * rp * np.sqrt(pl.clip(0) * MIP_P0), 0, None)
    d['l'] = np.clip(d.e_liq + rng.normal(0, 1, len(d)) * 0.25 * np.sqrt(d.e_liq.clip(0) * 3.0), 0, None)
    d['g'] = d.e_gas * np.exp(rng.normal(0, sigma_meas, len(d)))
    y = d['T'].to_numpy(float)
    base = np.c_[np.log1p(d.w), np.log1p(d.p), np.log1p(d.l)]
    full = np.c_[base, np.log(d.g)]
    pb, pf = knn_cv(base, y), knn_cv(full, y)
    rows = []
    for T in sorted(d['T'].unique()):
        m = y == T
        rows.append(dict(T=T, n=int(m.sum()), w68_scint=width68(pb[m] - T), w68_scint_mm=width68(pf[m] - T),
                         bias_scint=float(np.median(pb[m] - T)), bias_scint_mm=float(np.median(pf[m] - T))))
    return pd.DataFrame(rows)


def run(sigma_meas: float, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    G = g4_sample()
    G = smear_scint(G, scint_cal(), rng)
    G = mm_deposits(G, sigma_meas, rng, HeedPools())
    out = dict(
        cat_g4=category_table(G, 'q_g4').assign(model='Geant4 gap (all particles)'),
        cat_heed=category_table(G, 'q_heed').assign(model='Heed (leading lepton)'),
        energy_beam=energy_beam(G, 'q_g4'),
        slow_tag=slow_tag(G, 'q_g4'),
        energy_c2=energy_c2(sigma_meas, rng))
    G[['armL', 'cat', 'lead_ke0', 'e_wall', 'e_plas', 'q_g4', 'q_heed', 'dom_share', 'n_lep', 'sec']].to_parquet(
        EDEP / 'combo_g4.parquet')
    return out


if __name__ == '__main__':
    r = run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.15)
    with pd.option_context('display.width', 200):
        for k, v in r.items():
            print('==', k)
            print(v.round(3).to_string(index=False))
