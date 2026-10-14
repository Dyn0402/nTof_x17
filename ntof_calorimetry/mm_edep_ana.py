#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_edep_ana.py -- MM_EDEP.md stages 2-3: the per-track estimate of the
energy a track deposits in the 30 mm gap, its accuracy in simulation, and
whether the data agree.

ESTIMATOR.  E_hat = Q_road / (k * c(f)), Q_road = Qx + Qy (`mm_charge`),
k = ADC per keV of ionisation (W = 27.5 eV/pair, Heed), and c(f) the share of
the charge the 20-sample window holds, as a function of the observed drift
fraction f = (1200 ns - t0) / (30 mm / v).  c is taken from the simulation,
where both the charge and the truth are known; on chamber A f > 1 for ~90 %
of tracks, on C (v ~ 26-28 um/ns) for ~2-7 % -- C's estimate is mostly an
extrapolation of the first ~80 % of the drift.

TRUTH.  E_true = n_e x W: the energy that went into ionisation in the gap.

ACCURACY (simulation).  r = E_hat / E_true per track: robust sigma of ln r
(1.4826 MAD), the 68 % half-width, the tail beyond |ln r| > 0.4, and the
chain split stage by stage (ionisation statistics, avalanche, window/road
containment, noise + common mode + the reco's road).

DATA.  A forward fold: data ln(E_hat/sec) against simulated
ln(E_hat/sec) + ln s + N(0, sigma_x), fitted on the core of the
distribution.  s is the data/sim gain, so k_data = s k_sim; sigma_x is what
the real chamber adds that the simulation does not have.  Run on cosmic
muons (all, penetrating A-C, centre) and on late beam electrons, with and
without a position gain map (built on one half of the data, tested on the
other; or built on cosmics and applied to the beam).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry.mip_sample import OUT  # noqa: E402

EDEP = OUT / 'mm_edep'
W_EV = 27.5
WIN_NS = 1200.0
GAP_UM = 30000.0
LATE_NS = 12e6
CELL = 50.0
CENTRE = 120.0
FIT_LO, FIT_HI = -0.7, 0.45       # ln(E/peak) range of the forward fold (the core)


def rsd(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def w68(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(0.5 * np.subtract(*np.quantile(v, [.84, .16])))


# ------------------------------------------------------------------ loading
def _common(D: pd.DataFrame) -> pd.DataFrame:
    D['sec'] = np.sqrt(1 + D.tanx ** 2 + D.tany ** 2)
    D['Q'] = D.x_Q + D.y_Q
    D['f_obs'] = (WIN_NS - D.t0) / (GAP_UM / D.v)
    D['sat'] = (D.x_raw_max > 3700) | (D.y_raw_max > 3700)
    D['centre'] = (D.x_local.abs() < CENTRE) & (D.y_local.abs() < CENTRE)
    D['cell'] = (np.floor((D.x_local + 200) / CELL).astype(int) * 100
                 + np.floor((D.y_local + 200) / CELL).astype(int))
    return D


def load_sim(mode: str, arm: str) -> pd.DataFrame:
    S = pd.read_parquet(EDEP / f'sim_{mode}_{arm}.parquet')
    S = S.rename(columns={'reco_x_t0': 'rx_t0', 'reco_y_t0': 'ry_t0'})
    S['t0'] = S[['rx_t0', 'ry_t0']].mean(1)
    S['E_true'] = S.n_e * W_EV / 1e3
    S = S.dropna(subset=['x_Q', 'y_Q', 't0'])
    return _common(S)


def load_cosmic(arm: str) -> pd.DataFrame:
    C = pd.read_pickle(OUT / 'm2' / 'charge_run149.pkl')
    C = C[C.arm == arm].dropna(subset=['x_Q', 'y_Q', 'tanx', 'tany', 'x_t0', 'y_t0']).copy()
    C['t0'] = C[['x_t0', 'y_t0']].mean(1)
    C['v'] = C.v_drift_um_ns
    C['both'] = False
    A = pd.read_pickle(OUT / 'm2' / 'charge_run149.pkl')
    n = A.dropna(subset=['x_Q']).groupby(['subrun', 'event_id']).arm.nunique()
    both = set(n[n == 2].index)
    C['both'] = [k in both for k in zip(C.subrun, C.event_id)]
    return _common(C.reset_index(drop=True))


def load_beam(arm: str, late: bool = True) -> pd.DataFrame:
    B = pd.read_pickle(EDEP / 'beam_run_145.pkl')
    B = B[B.arm == arm].dropna(subset=['x_Q', 'y_Q', 'tanx', 'tany', 'x_t0', 'y_t0']).copy()
    if late:
        B = B[B.t_since_flash_ns >= LATE_NS]
    B['t0'] = B[['x_t0', 'y_t0']].mean(1)
    B['v'] = B.v_drift_um_ns
    return _common(B.reset_index(drop=True))


# ------------------------------------------------------- calibration (sim)
def window_curve(S: pd.DataFrame) -> pd.DataFrame:
    """Median Q / E_true against the observed drift fraction, normalised to
    the fully contained tracks (f > 1.35)."""
    r = S.Q / S.E_true
    k = float(np.median(r[S.f_obs > 1.35]))
    edges = np.r_[0.3, np.arange(0.55, 1.401, 0.05), 3.0]
    b = pd.cut(S.f_obs, edges)
    c = (r / k).groupby(b, observed=True).median()
    n = r.groupby(b, observed=True).size()
    out = pd.DataFrame(dict(f=[iv.mid for iv in c.index], c=c.to_numpy(), n=n.to_numpy()))
    out.attrs['k'] = k
    return out[out.n >= 30]


def apply_window(D: pd.DataFrame, W: pd.DataFrame, k: float) -> pd.Series:
    c = np.interp(D.f_obs.clip(W.f.min(), W.f.max()), W.f, W.c)
    return D.Q / (k * c)


def accuracy(S: pd.DataFrame, W: pd.DataFrame) -> dict:
    k = W.attrs['k']
    S = S.assign(E_hat=apply_window(S, W, k))
    lr = np.log(S.E_hat / S.E_true)
    g = S[S.f_obs > 1.35]
    sw = S.x_S_road + S.y_S_road
    st = dict(
        ionisation=rsd(np.log(S.n_e * W_EV / S.e_dep)),
        avalanche=rsd(np.log(S.G_sum / S.n_e)),
        containment=rsd(np.log(sw / S.G_sum)),
        noise_cm_road=rsd(np.log((S.Q / sw).clip(1e-3))))
    cum = dict(after_avalanche=rsd(np.log(S.G_sum / S.e_dep)),
               after_containment=rsd(np.log(sw / S.e_dep)),
               measured=rsd(np.log(S.Q / S.e_dep)))
    rows = []
    for lo, hi in ((0.3, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.35), (1.35, 3)):
        m = (S.f_obs >= lo) & (S.f_obs < hi)
        if m.sum() >= 50:
            rows.append(dict(f_lo=lo, f_hi=hi, n=int(m.sum()), rsd=rsd(lr[m]), w68=w68(lr[m]),
                             bias=float(np.median(lr[m])), tail=float((lr[m].abs() > 0.4).mean())))
    return dict(k=k, n=len(S), rsd=rsd(lr), w68=w68(lr), bias=float(np.median(lr)),
                tail=float((lr.abs() > 0.4).mean()), rsd_contained=rsd(lr[S.f_obs > 1.35]),
                frac_contained=float((S.f_obs > 1.35).mean()), stages=st, cumulative=cum,
                by_f=pd.DataFrame(rows), xy_rsd=rsd(np.log((S.x_Q / S.y_Q).clip(1e-3))),
                xy_rsd_contained=rsd(np.log((g.x_Q / g.y_Q).clip(1e-3))),
                ctrl_rsd=rsd((S.x_Q_ctrl + S.y_Q_ctrl) / S.Q), E_hat=S.E_hat)


# --------------------------------------------------------------- gain map
def gain_map(D: pd.DataFrame, col: str = 'q', shrink: float = 30.0) -> pd.Series:
    """Per 50 mm cell, ln(median q / overall median), shrunk toward 0 by
    n / (n + shrink) so thinly populated cells do not over-correct."""
    m = np.log(D[col].median())
    g = D.groupby('cell')[col].agg(['median', 'size'])
    return (np.log(g['median']) - m) * g['size'] / (g['size'] + shrink)


def corrected(D: pd.DataFrame, gm: pd.Series, col: str = 'q') -> pd.Series:
    return D[col] / np.exp(D.cell.map(gm).fillna(0.0))


def crossval_map(D: pd.DataFrame, col: str = 'q', seed: int = 2) -> pd.Series:
    """Each half corrected with the map of the other half."""
    h = np.random.default_rng(seed).integers(0, 2, len(D))
    out = pd.Series(np.nan, index=D.index)
    for i in (0, 1):
        gm = gain_map(D[h != i], col)
        out[h == i] = corrected(D[h == i], gm, col)
    return out


# ------------------------------------------------------------ forward fold
def fold_fit(data: np.ndarray, sim: np.ndarray, n_boot: int = 0, seed: int = 4) -> dict:
    """ln data = ln sim + ln s + N(0, sigma_x): binned Poisson deviance on the
    core (ln(v / median) in [FIT_LO, FIT_HI]); grid over ln s and sigma_x."""
    rng = np.random.default_rng(seed)
    ld = np.log(data[data > 0])
    ls0 = np.log(sim[sim > 0])
    med = np.median(ld)
    edges = med + np.arange(FIT_LO - 0.3, FIT_HI + 0.3 + 1e-9, 0.04)
    core = (edges[:-1] >= med + FIT_LO) & (edges[1:] <= med + FIT_HI)
    z = rng.normal(0, 1, (4, len(ls0)))

    def dev(lsc, sx, ld_):
        h_d = np.histogram(ld_, edges)[0][core].astype(float)
        hs = sum(np.histogram(ls0 + lsc + sx * z[i], edges)[0] for i in range(4))[core].astype(float)
        hs = hs / hs.sum() * h_d.sum()
        hs = np.clip(hs, 1e-3, None)
        return 2 * np.sum(hs - h_d + np.where(h_d > 0, h_d * np.log(np.clip(h_d, 1e-9, None) / hs), 0))

    def best(ld_):
        g0 = np.median(ld_) - np.median(ls0)
        grid_s = g0 + np.arange(-0.15, 0.1501, 0.01)
        grid_x = np.arange(0.0, 0.501, 0.01)
        D = np.array([[dev(a, b, ld_) for b in grid_x] for a in grid_s])
        i, j = np.unravel_index(D.argmin(), D.shape)
        return grid_s[i], grid_x[j], D[i, j], int(core.sum())

    s, sx, d, nb = best(ld)
    out = dict(scale=float(np.exp(s)), sigma_x=float(sx), deviance=float(d), nbins=nb, n=int(len(ld)))
    if n_boot:
        bs = [best(rng.choice(ld, len(ld))) for _ in range(n_boot)]
        out['scale_err'] = float(np.std([np.exp(b[0]) for b in bs]))
        out['sigma_x_err'] = float(np.std([b[1] for b in bs]))
    # tails: data fraction above +0.8 in ln vs the folded simulation's
    zf = ls0 + s + sx * z[0]
    out['tail_hi_data'] = float((ld - med > 0.8).mean())
    out['tail_hi_sim'] = float((zf - med > 0.8).mean())
    out['tail_lo_data'] = float((ld - med < -0.8).mean())
    out['tail_lo_sim'] = float((zf - med < -0.8).mean())
    return out


def total_resolution(acc_rsd: float, sigma_x: float) -> float:
    return float(np.hypot(acc_rsd, sigma_x))


# ------------------------------------------------- 1D gain profile along x
#: the gain structure is one-dimensional: tracks at the same x_local share
#: their charge residual (r = 0.26 in BOTH views for |dx| < 4 mm, |dy| > 80
#: mm) while tracks at the same y do not (r = 0.01) -- an amplification
#: structure along x (resistive strips / mesh), not strip electronics, which
#: would move one view only.  Correlation length ~20-30 mm.
XBIN = 4.0


def x_profile(D: pd.DataFrame, col: str = 'q', smooth_mm: float = 6.0, shrink: float = 20.0) -> pd.DataFrame:
    """ln(median q / overall median) in XBIN bins of x_local, shrunk by
    n / (n + shrink), then Gaussian-smoothed over smooth_mm."""
    m = np.log(D[col].median())
    b = np.floor((D.x_local + 200) / XBIN).astype(int)
    g = D.groupby(b)[col].agg(['median', 'size'])
    full = pd.DataFrame(index=np.arange(int(400 / XBIN) + 1))
    full = full.join(g)
    v = ((np.log(full['median']) - m) * full['size'] / (full['size'] + shrink)).fillna(0.0).to_numpy()
    if smooth_mm > 0:
        k = np.arange(-3, 4) * XBIN
        w = np.exp(-0.5 * (k / smooth_mm) ** 2)
        num = np.convolve(v, w, 'same')
        den = np.convolve(np.ones_like(v), w, 'same')
        v = num / den
    return pd.DataFrame(dict(x=-200 + (full.index + 0.5) * XBIN, lng=v))


def apply_x_profile(D: pd.DataFrame, P: pd.DataFrame, col: str = 'q') -> pd.Series:
    return D[col] / np.exp(np.interp(D.x_local, P.x, P.lng))


def crossval_x(D: pd.DataFrame, col: str = 'q', seed: int = 2, with_2d: bool = True) -> pd.Series:
    h = np.random.default_rng(seed).integers(0, 2, len(D))
    out = pd.Series(np.nan, index=D.index)
    for i in (0, 1):
        tr = D[h != i]
        P = x_profile(tr, col)
        c_tr = tr.assign(_c=apply_x_profile(tr, P, col))
        te = D[h == i]
        c = apply_x_profile(te, P, col)
        if with_2d:
            gm = gain_map(c_tr, '_c')
            c = c / np.exp(te.cell.map(gm).fillna(0.0))
        out[h == i] = c
    return out


# ------------------------------------------------------ plateau estimator
#: for tracks whose drift does not fit the window (chamber C, v ~ 26-28
#: um/ns): each 60 ns sample inside the drift holds the charge of v x 60 ns
#: of depth, so the MEAN road charge per sample there, / (v x 60 ns), is the
#: charge per mm of depth; x 30 mm is the gap-equivalent charge.  The first
#: RISE_NS after t0 (the shaping rise) are skipped.  An energy estimator, so
#: the plain mean (no truncation).
RISE_NS = 180.0
MIN_PLAT = 4


def plateau_Q(D: pd.DataFrame) -> np.ndarray:
    out = np.full(len(D), np.nan)
    for i, r in enumerate(D[['xp', 'yp', 't0', 'v']].itertuples(index=False)):
        if not (hasattr(r.xp, '__len__') and hasattr(r.yp, '__len__')) or not np.isfinite(r.t0):
            continue
        p = np.asarray(r.xp, float) + np.asarray(r.yp, float)
        t = np.arange(len(p)) * 60.0
        m = (t >= r.t0 + RISE_NS) & (t <= r.t0 + GAP_UM / r.v)
        if m.sum() < MIN_PLAT:
            continue
        out[i] = p[m].mean() / (r.v * 60.0 / 1000.0) * 30.0
    return out


def load_sim_prof(mode: str, arm: str) -> pd.DataFrame:
    S = pd.read_parquet(EDEP / f'sim_{mode}_{arm}.parquet')
    P = pd.read_pickle(EDEP / f'sim_{mode}_{arm}_prof.pkl')
    S['xp'], S['yp'] = P.x_prof.values, P.y_prof.values
    S['t0'] = S[['reco_x_t0', 'reco_y_t0']].mean(1)
    S['E_true'] = S.n_e * W_EV / 1e3
    S = S.dropna(subset=['x_Q', 'y_Q', 't0']).reset_index(drop=True)
    S = _common(S)
    S['Q_plat'] = plateau_Q(S)
    return S


def with_plateau(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D['xp'], D['yp'] = D.x_prof, D.y_prof
    D['Q_plat'] = plateau_Q(D)
    return D


# --------------------------------------------- local (kNN) position map
K_NN = 25


def knn_map(ref: pd.DataFrame, target: pd.DataFrame, col: str = 'q', k: int = K_NN) -> np.ndarray:
    """ln gain at each target track: median ln(q / median) of the k nearest
    REFERENCE tracks in (x_local, y_local)."""
    from scipy.spatial import cKDTree
    lr = np.log(ref[col].to_numpy())
    lr = lr - np.median(lr)
    t = cKDTree(np.c_[ref.x_local, ref.y_local])
    _, nn = t.query(np.c_[target.x_local, target.y_local], k=k)
    return np.median(lr[nn], axis=1)


def knn_crossval(D: pd.DataFrame, col: str = 'q', seed: int = 3) -> np.ndarray:
    h = np.random.default_rng(seed).integers(0, 2, len(D))
    out = np.empty(len(D))
    for i in (0, 1):
        out[h == i] = knn_map(D[h != i], D[h == i], col)
    return out


def spatial_corr(D: pd.DataFrame, lnres: np.ndarray, edges=(0, 4, 10, 20, 35, 60, 100, 150)) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    from scipy.stats import norm, rankdata
    z = norm.ppf((rankdata(lnres) - 0.5) / len(lnres))
    P = np.c_[D.x_local, D.y_local]
    pr = cKDTree(P).query_pairs(edges[-1], output_type='ndarray')
    d = np.hypot(*(P[pr[:, 0]] - P[pr[:, 1]]).T)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        rows.append(dict(d_lo=lo, d_hi=hi, n=int(m.sum()), r=float(np.corrcoef(z[pr[m, 0]], z[pr[m, 1]])[0, 1])))
    return pd.DataFrame(rows)


def halves(D: pd.DataFrame, split: float = 0.5) -> tuple:
    """Road charge in the first and second part of the drift (contained
    tracks only, t0 >= -30 ns and drift + 300 ns inside the window)."""
    h1, h2 = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for i, r in enumerate(D[['xp', 'yp', 't0', 'v']].itertuples(index=False)):
        if not (hasattr(r.xp, '__len__') and hasattr(r.yp, '__len__')) or not np.isfinite(r.t0):
            continue
        td = GAP_UM / r.v
        if r.t0 + td + 300 > WIN_NS or r.t0 < -30:
            continue
        p = np.asarray(r.xp, float) + np.asarray(r.yp, float)
        t = np.arange(len(p)) * 60.0
        c = r.t0 + td * split + 100
        h1[i], h2[i] = p[(t >= r.t0 - 60) & (t < c)].sum(), p[t >= c].sum()
    return h1, h2


def common_cov(h1: np.ndarray, h2: np.ndarray, corr: np.ndarray | None = None) -> tuple:
    """Robust cov(ln h1, ln h2) = (rsd(sum)^2 - rsd(diff)^2) / 4 and its
    bootstrap error; ``corr`` (ln) is subtracted from both halves."""
    m = np.isfinite(h1) & np.isfinite(h2) & (h1 > 0) & (h2 > 0)
    a, b = np.log(h1[m]), np.log(h2[m])
    if corr is not None:
        a, b = a - corr[m], b - corr[m]
    f = lambda a, b: (rsd(a + b) ** 2 - rsd(a - b) ** 2) / 4  # noqa: E731
    rng = np.random.default_rng(8)
    bs = []
    for _ in range(100):
        j = rng.integers(0, len(a), len(a))
        bs.append(f(a[j], b[j]))
    return float(f(a, b)), float(np.std(bs)), int(m.sum())


# ===================================================================== main
RES = EDEP / 'results'


def physics() -> pd.DataFrame:
    """Heed truth per (particle, T, theta): ionisation per track in the gap."""
    rows = []
    for f in sorted((OUT / 'heed').glob('*_th[0-9]*.npz')):
        z = np.load(f)
        n = z['n_e'].astype(float)
        E = n * W_EV / 1e3
        h, b = np.histogram(E, bins=np.arange(0, np.quantile(E, .9), 0.15))
        hs = np.convolve(h, np.ones(5) / 5, 'same')
        mpv = 0.5 * (b[hs.argmax()] + b[hs.argmax() + 1])
        half = hs.max() / 2
        k = hs.argmax()
        lo, hi = b[:k][hs[:k] < half], b[k:-1][hs[k:] < half]
        rows.append(dict(particle=str(z['particle']), T=float(z['T']), theta=float(z['theta']),
                         n=len(n), mean_keV=E.mean(), median_keV=np.median(E), mpv_keV=mpv,
                         fwhm_over_mpv=(hi[0] - lo[-1]) / mpv if len(lo) and len(hi) else np.nan,
                         q16=np.quantile(E, .16), q84=np.quantile(E, .84), q25=np.quantile(E, .25),
                         q75=np.quantile(E, .75), ne_median=float(np.median(n)),
                         W_eV=float(z['e_dep'].sum() / n.sum())))
    return pd.DataFrame(rows).sort_values(['particle', 'theta', 'T'])


def _q(D: pd.DataFrame, W: pd.DataFrame, k: float, contained_only: bool) -> pd.DataFrame:
    D = D.copy()
    D['E_hat'] = apply_window(D, W, k)
    D['q'] = D.E_hat / D.sec
    D = D[np.isfinite(D.q) & (D.q > 0)]
    if contained_only:
        D = D[D.f_obs > 1.35]
    return D.reset_index(drop=True)


def chamber(arm: str) -> dict:
    """Cosmics then beam for one chamber."""
    out = {}
    contained = arm == 'A'           # A: use the tracks whose drift fits; C: all
    Sc = load_sim_prof('cosmic', arm)
    W = window_curve(Sc)
    k = W.attrs['k']
    acc = accuracy(Sc, W)
    W.to_csv(RES / f'window_{arm}.csv', index=False)
    acc['by_f'].to_csv(RES / f'acc_by_f_cosmic_{arm}.csv', index=False)
    Sq = _q(Sc, W, k, contained)
    out['sim_cosmic'] = {kk: v for kk, v in acc.items() if kk not in ('by_f', 'E_hat')}
    # sim accuracy on the SAME subset the data use
    lr = np.log(Sq.E_hat / Sq.E_true)
    out['sim_cosmic']['rsd_used'] = rsd(lr)
    out['sim_cosmic']['w68_used'] = w68(lr)
    out['sim_cosmic']['tail_used'] = float((np.abs(lr) > 0.4).mean())
    pd.DataFrame(dict(E_true=Sq.E_true, E_hat=Sq.E_hat, sec=Sq.sec)).to_parquet(RES / f'sim_pairs_cosmic_{arm}.parquet')

    # ---- cosmic data
    C = with_plateau(load_cosmic(arm))
    C = _q(C, W, k, contained)
    C['g'] = knn_crossval(C, 'q')
    C['q_c'] = C.q / np.exp(C.g)
    sc_raw = spatial_corr(C, np.log(C.q.to_numpy())).assign(which='raw')
    sc_cor = spatial_corr(C, np.log(C.q_c.to_numpy())).assign(which='corrected')
    pd.concat([sc_raw, sc_cor]).to_csv(RES / f'spatial_cosmic_{arm}.csv', index=False)
    folds = []
    for lab, m in (('all', np.ones(len(C), bool)), ('centre', C.centre.to_numpy()),
                   ('penetrating', C.both.to_numpy())):
        for col in ('q', 'q_c'):
            r = fold_fit(C[col][m].to_numpy(), Sq.q.to_numpy(), n_boot=30)
            folds.append(dict(sample='cosmic', subset=lab, map='none' if col == 'q' else 'cosmic self (cv)', **r))
    # x vs y and the noise road
    out['cosmic_data'] = dict(n=len(C), xy_rsd=rsd(np.log((C.x_Q / C.y_Q).clip(1e-3))),
                              ctrl_rsd=rsd((C.x_Q_ctrl + C.y_Q_ctrl) / C.Q),
                              sim_xy_rsd=rsd(np.log((Sq.x_Q / Sq.y_Q).clip(1e-3))),
                              sim_ctrl_rsd=rsd((Sq.x_Q_ctrl + Sq.y_Q_ctrl) / Sq.Q),
                              map_sd=float(np.std(C.g)), kev_per_adc_cosmic=1.0 / k)
    # 1D (x) vs 2D structure
    out['cosmic_data']['struct'] = structure_1d(C)
    # halves (contained only)
    hs = halves(Sc)
    cs = common_cov(*hs)
    h1, h2 = halves(C)
    hd = common_cov(h1, h2)
    hc = common_cov(h1, h2, C.g.to_numpy())
    hcc = common_cov(h1[C.centre.to_numpy()], h2[C.centre.to_numpy()], C.g.to_numpy()[C.centre.to_numpy()])
    out['halves_cosmic'] = dict(sim=cs, data_raw=hd, data_corrected=hc, data_corrected_centre=hcc,
                                sigma_g_raw=float(np.sqrt(max(hd[0] - cs[0], 0))),
                                sigma_g_corrected=float(np.sqrt(max(hc[0] - cs[0], 0))),
                                sigma_g_corrected_centre=float(np.sqrt(max(hcc[0] - cs[0], 0))))
    C[['q', 'q_c', 'x_local', 'y_local', 'centre', 'both', 'f_obs', 'sec']].to_parquet(RES / f'cosmic_{arm}.parquet')
    Sq[['q', 'E_true', 'E_hat', 'sec', 'f_obs']].to_parquet(RES / f'sim_cosmic_{arm}_q.parquet')

    # ---- beam
    beam_sim = EDEP / f'sim_beam_{arm}.parquet'
    Ball = with_plateau(load_beam(arm, late=False))
    Ball['E_hat'] = apply_window(Ball, W, k)
    Ball['q'] = Ball.E_hat / Ball.sec
    out['flash'] = flash_curve(Ball)
    out['flash'].to_csv(RES / f'flash_{arm}.csv', index=False)
    B = _q(Ball[Ball.t_since_flash_ns >= LATE_NS], W, k, contained)
    B['g_self'] = knn_crossval(B, 'q')
    B['g_cos'] = knn_map(C, B, 'q')
    out['transfer'] = dict(corr_maps=float(np.corrcoef(B.g_self, B.g_cos)[0, 1]),
                           sd_self=float(B.g_self.std()), sd_cos=float(B.g_cos.std()),
                           beam_over_cosmic=float(B.q.median() / C.q.median()))
    for lab, g in (('none', 0.0), ('cosmic map', B.g_cos), ('beam self (cv)', B.g_self)):
        lq = np.log(B.q.to_numpy()) - np.asarray(g)
        out['transfer'][lab] = dict(rsd=rsd(lq), spatial=spatial_corr(B, lq, (0, 4, 10, 20, 35, 60)).r.tolist())
    B['q_c'] = B.q / np.exp(B.g_self)
    if beam_sim.exists():
        Sb = load_sim_prof('beam', arm)
        Sbq = _q(Sb, W, k, contained)
        lr = np.log(Sbq.E_hat / Sbq.E_true)
        out['sim_beam'] = dict(n=len(Sbq), rsd=rsd(lr), w68=w68(lr), bias=float(np.median(lr)),
                               tail=float((np.abs(lr) > 0.4).mean()),
                               k_ratio=float(np.median(Sbq.E_hat / Sbq.E_true)))
        for lab, m in (('all', np.ones(len(B), bool)), ('centre', B.centre.to_numpy())):
            for col, mp in (('q', 'none'), ('q_c', 'beam self (cv)')):
                r = fold_fit(B[col][m].to_numpy(), Sbq.q.to_numpy(), n_boot=30)
                folds.append(dict(sample='beam', subset=lab, map=mp, **r))
        hb = common_cov(*halves(Sb))
        h1, h2 = halves(B)
        hbd = common_cov(h1, h2, B.g_self.to_numpy())
        hbr = common_cov(h1, h2)
        out['halves_beam'] = dict(sim=hb, data_raw=hbr, data_corrected=hbd,
                                  sigma_g_raw=float(np.sqrt(max(hbr[0] - hb[0], 0))),
                                  sigma_g_corrected=float(np.sqrt(max(hbd[0] - hb[0], 0))))
        Sbq[['q', 'E_true', 'E_hat', 'sec', 'f_obs', 'T']].to_parquet(RES / f'sim_beam_{arm}_q.parquet')
    B[['subrun', 'event_id', 'track_id', 'q', 'q_c', 'E_hat', 'x_local', 'y_local', 'centre', 'f_obs', 'sec',
       't_since_flash_ns']].to_parquet(RES / f'beam_{arm}.parquet')
    pd.DataFrame(folds).to_csv(RES / f'folds_{arm}.csv', index=False)
    out['folds'] = folds
    return out


def structure_1d(C: pd.DataFrame, n_pairs: int = 4_000_000) -> dict:
    """Is the position structure one-dimensional?  Residual correlation of
    pairs at the same x far apart in y, and vice versa, in each view."""
    from scipy.stats import norm, rankdata
    D = C[(C.x_Q > 0) & (C.y_Q > 0)].reset_index(drop=True)
    gz = lambda v: norm.ppf((rankdata(np.log(v)) - 0.5) / len(v))  # noqa: E731
    zx, zy = gz(D.x_Q / D.sec), gz(D.y_Q / D.sec)
    x, y = D.x_local.to_numpy(), D.y_local.to_numpy()
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(D), (2, n_pairs))
    m = i != j
    i, j = i[m], j[m]
    dx, dy = np.abs(x[i] - x[j]), np.abs(y[i] - y[j])
    out = {}
    for lab, mm in (('same_xy', (dx < 4) & (dy < 4)), ('same_x_far_y', (dx < 4) & (dy > 80)),
                    ('same_y_far_x', (dy < 4) & (dx > 80)), ('far', (dx > 80) & (dy > 80))):
        out[lab] = dict(n=int(mm.sum()), r_x=float(np.corrcoef(zx[i[mm]], zx[j[mm]])[0, 1]),
                        r_y=float(np.corrcoef(zy[i[mm]], zy[j[mm]])[0, 1]))
    return out


def flash_curve(B: pd.DataFrame) -> pd.DataFrame:
    """Gain (median q) and the x/y agreement against time since the flash,
    normalised to >= 12 ms."""
    late = B[B.t_since_flash_ns >= LATE_NS]
    m0 = late.q.median()
    edges = [0, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 20, 30, 45, 80]
    b = pd.cut(B.t_since_flash_ns / 1e6, edges)
    rows = []
    for iv, g in B.groupby(b, observed=True):
        q = g.q[np.isfinite(g.q)]
        rows.append(dict(ms_lo=iv.left, ms_hi=iv.right, n=len(g), gain=float(np.median(q) / m0),
                         q25=float(np.quantile(q, .25) / m0), q75=float(np.quantile(q, .75) / m0),
                         xy=float(np.median(g.x_Q / g.y_Q)),
                         xy_rsd=rsd(np.log((g.x_Q / g.y_Q).clip(1e-3)))))
    return pd.DataFrame(rows)


def main() -> int:
    import json
    RES.mkdir(parents=True, exist_ok=True)
    P = physics()
    P.to_csv(RES / 'physics.csv', index=False)
    summ = {}
    for arm in ('A', 'C'):
        r = chamber(arm)
        r.pop('flash', None)
        summ[arm] = r
        print(arm, json.dumps({k: v for k, v in r.items() if k != 'folds'}, indent=1, default=str)[:4000])
    (RES / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
