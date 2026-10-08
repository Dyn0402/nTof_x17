#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_dedx_cosmics.py -- PLAN.md M2 (+ the statistical part of M3): what the
raw road charge of `mm_charge` can do as dE/dx, on run_149 cosmic muons in
chambers A and C.

Selection: one gated track in the arm; both planes' roads inside the strip
map (5 mm margin); the drift fits the 20-sample window (t0 + 30 mm / v +
SHAPING_NS <= window); no raw sample on the 12-bit rail.  Path = 30 mm x
sqrt(1 + tanx^2 + tany^2) with the calibrated (k) slopes.

Estimators, each normalised to its own MPV:
  whole    Q = Qx + Qy over the road and window, per mm of path;
  trunc_t  truncated mean over the samples inside the drift window (road
           sums per 60 ns sample = ~2 mm of depth), lowest 65 % kept, x and
           y averaged.  Shaping and the resistive kernel's ~47 ns delayed
           copies correlate neighbouring samples -- the test is whether it
           helps anyway;
  trunc_s  the same over strips (per-strip road sums), lowest 65 %.
  plat     the PLATEAU estimator, which needs no complete drift: inside the
           drift (from t0 + RISE_NS to the drift end or the window end,
           whichever is first) each 60 ns sample holds the charge of one
           v x 60 ns depth slice, so the truncated mean of those samples /
           (v x 60 ns x sec) is a charge per mm of path.  Arm C drifts at
           ~28 um/ns (July water), so 30 mm takes ~1.1 us and its deep charge
           falls off the 1.2 us window: whole-gap Q is truncated on most C
           tracks, and only this estimator covers C.

Saturated tracks (a road sample on the 12-bit rail, 5-10 %) are KEPT --
cutting them removes the high tail the 2-MIP test needs -- and flagged.

2-MIP: a two-muon sample made by adding the charges of random pairs of real
tracks (same estimator, same path normalisation).  This ignores what the
overlap does to the reconstruction and to ZS, so it is the BEST case for
M3; M3's waveform overlay gives the real one.

    python -m ntof_calorimetry.mm_dedx_cosmics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ntof_calorimetry import landau as LD  # noqa: E402
from ntof_calorimetry.mm_charge import GAP_MM, M2, N_SAMPLE, SAMPLE_NS, SAT_RAW  # noqa: E402

SHAPING_NS = 240.0
RISE_NS = 180.0
MIN_PLAT = 5
EDGE_MM = 5.0
STRIP_MAX = 398.58
KEEP = 0.65
CELL = 40.0


def load() -> pd.DataFrame:
    D = pd.read_pickle(M2 / 'charge_run149.pkl')
    D = D.dropna(subset=['x_Q', 'y_Q', 'tanx', 'tany']).reset_index(drop=True)
    D['path_mm'] = GAP_MM * np.sqrt(1 + D.tanx ** 2 + D.tany ** 2)
    D['sec'] = D.path_mm / GAP_MM
    t0 = D[['x_t0', 'y_t0']].mean(axis=1)
    D['t_end'] = t0 + GAP_MM * 1000 / D.v_drift_um_ns + SHAPING_NS
    D['in_window'] = D.t_end <= N_SAMPLE * SAMPLE_NS
    ok = np.ones(len(D), bool)
    for p in 'xy':
        p0, tn = D[f'{p}_p0'].to_numpy(), D[f'{p}_tan_theta'].to_numpy()
        a, b = p0 - 3.0 * tn, p0 + 33.0 * tn
        ok &= (np.minimum(a, b) > EDGE_MM) & (np.maximum(a, b) < STRIP_MAX - EDGE_MM)
    D['in_map'] = ok
    D['sat'] = (D.x_raw_max > SAT_RAW) | (D.y_raw_max > SAT_RAW)
    D['sel'] = D.in_map
    D['q_whole'] = np.where(D.in_window, (D.x_Q + D.y_Q) / D.path_mm, np.nan)
    D['q_trunc_t'] = np.where(D.in_window, _trunc_time(D) / D.path_mm, np.nan)
    D['q_trunc_s'] = np.where(D.in_window, _trunc_strip(D) / D.path_mm, np.nan)
    D['q_plat'], D['n_plat'] = _plateau(D)
    return D


def _plateau(D: pd.DataFrame) -> tuple:
    out = np.full(len(D), np.nan)
    npl = np.zeros(len(D), int)
    for i, r in enumerate(D.itertuples(index=False)):
        v, n = [], []
        for p in 'xy':
            prof = getattr(r, f'{p}_prof')
            t0 = getattr(r, f'{p}_t0')
            t = np.arange(len(prof)) * SAMPLE_NS
            m = (t >= t0 + RISE_NS) & (t <= t0 + GAP_MM * 1000 / r.v_drift_um_ns)
            if m.sum() < MIN_PLAT:
                continue
            s = np.sort(prof[m])
            k = max(1, int(round(KEEP * len(s))))
            v.append(s[:k].mean())
            n.append(m.sum())
        if len(v) == 2:
            # two planes, each sample = v * 60 ns of depth; per mm of path
            out[i] = np.sum(v) / (r.v_drift_um_ns * SAMPLE_NS / 1000) / r.sec
            npl[i] = min(n)
    return out, npl


def _trunc_time(D: pd.DataFrame) -> np.ndarray:
    out = np.full(len(D), np.nan)
    for i, r in enumerate(D.itertuples(index=False)):
        v = []
        for p in 'xy':
            prof = getattr(r, f'{p}_prof')
            t0 = getattr(r, f'{p}_t0')
            k0 = int(np.clip(np.floor(t0 / SAMPLE_NS), 0, N_SAMPLE - 1))
            k1 = int(np.clip(np.ceil((t0 + GAP_MM * 1000 / r.v_drift_um_ns + SHAPING_NS) / SAMPLE_NS),
                             k0 + 3, len(prof)))
            s = np.sort(prof[k0:k1])
            n = max(1, int(round(KEEP * len(s))))
            # mean of the kept samples x the number in the window = a charge
            v.append(s[:n].mean() * len(s))
        out[i] = np.sum(v)
    return out


def _trunc_strip(D: pd.DataFrame) -> np.ndarray:
    out = np.full(len(D), np.nan)
    for i, r in enumerate(D.itertuples(index=False)):
        v = []
        for p in 'xy':
            s = np.sort(getattr(r, f'{p}_strip_q'))
            n = max(1, int(round(KEEP * len(s))))
            v.append(s[:n].mean() * len(s))
        out[i] = np.sum(v)
    return out


def width(v: np.ndarray, mpv: float) -> dict:
    """FWHM / MPV from a histogram of v / mpv, and the tail fraction."""
    x = v / mpv
    x = x[np.isfinite(x)]
    n, e = np.histogram(x, bins=np.linspace(0, 4, 161))
    c = 0.5 * (e[1:] + e[:-1])
    ns = np.convolve(n, np.ones(5) / 5, mode='same')
    k = ns.argmax()
    half = ns[k] / 2
    lo = c[:k][ns[:k] < half]
    hi = c[k:][ns[k:] < half]
    fwhm = (hi[0] if len(hi) else np.nan) - (lo[-1] if len(lo) else np.nan)
    return dict(fwhm_over_mpv=float(fwhm), sigma_eq=float(fwhm / 2.355),
                tail_gt2=float((x > 2).mean()), q16=float(np.quantile(x, 0.16)),
                q84=float(np.quantile(x, 0.84)))


def resolution(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in 'AC':
        x = D[D.sel & (D.arm == arm)]
        for est in ('q_whole', 'q_trunc_t', 'q_trunc_s', 'q_plat'):
            v = x[est].dropna().to_numpy()
            # C's whole-gap estimators exist only on the ~6 % of its tracks
            # whose drift fits the window -- a biased, too-small subset
            if len(v) < 1000:
                continue
            f = LD.fit(v, lo_q=0.05, hi_mult=2.5, nbins=50)
            rows.append(dict(arm=arm, estimator=est, n=len(v), mpv=f['mpv_landau'],
                             mpv_err=f['mpv_err'], sigma_fit=f['sigma'] / f['mpv_landau'],
                             **width(v, f['mpv_landau'])))
    return pd.DataFrame(rows)


def two_mip(D: pd.DataFrame, R: pd.DataFrame, n_pairs: int = 20000) -> pd.DataFrame:
    """Efficiency to flag a 2-MIP track at 10 % (and 5 %) 1-MIP mis-tag."""
    rng = np.random.default_rng(3)
    rows = []
    for arm in 'AC':
        x = D[D.sel & (D.arm == arm)]
        for est in ('q_whole', 'q_trunc_t', 'q_trunc_s', 'q_plat'):
            rr = R[(R.arm == arm) & (R.estimator == est)]
            if not len(rr):
                continue
            m = float(rr.mpv.iloc[0])
            y = x[x[est].notna()]
            v = y[est].to_numpy() / m
            Q = y[est].to_numpy() * y.path_mm.to_numpy()
            p = y.path_mm.to_numpy()
            i, j = rng.integers(0, len(v), (2, n_pairs))
            # two muons through the same path: add charges, keep path i
            two = (Q[i] + Q[j] * p[i] / p[j]) / p[i] / m
            for fr in (0.10, 0.05):
                cut = np.quantile(v, 1 - fr)
                rows.append(dict(arm=arm, estimator=est, mistag=fr, cut=float(cut),
                                 eff_2mip=float((two > cut).mean()),
                                 sep_sigma=float((np.median(two) - np.median(v)) /
                                                 (0.5 * (np.quantile(v, .84) - np.quantile(v, .16))))))
    return pd.DataFrame(rows)


def vs_path(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in 'AC':
        x = D[D.sel & (D.arm == arm) & D.q_plat.notna()].copy()
        x['b'] = pd.qcut(x.sec, 5)
        for b, g in x.groupby('b', observed=True):
            # plateau charge per mm of DEPTH (x sec undone): should grow as sec
            Qd = (g.q_plat * g.sec).to_numpy()
            f = LD.fit(Qd, lo_q=0.05, hi_mult=2.5)
            rows.append(dict(arm=arm, sec=float(g.sec.median()), n=len(g), mpv_Q=f['mpv_landau'],
                             err=f['mpv_err']))
    P = pd.DataFrame(rows)
    for arm in 'AC':
        m = P.arm == arm
        P.loc[m, 'mpv_rel'] = P.mpv_Q[m] / P.mpv_Q[m].iloc[0] * P.sec[m].iloc[0]
    return P


def depth_profile(D: pd.DataFrame) -> pd.DataFrame:
    """Mean ABSOLUTE road charge per sample (/ sec) against depth = v x
    (sample time - t0), every unsaturated track contributing to each depth
    its window covers.  NOT normalised per track and NOT restricted to tracks
    whose drift fits the window: on C those are a minority with early t0,
    and normalising that subset per track manufactures a fall with depth
    (an earlier version of this function did exactly that)."""
    rows = []
    for arm in 'AC':
        x = D[D.sel & (D.arm == arm) & ~D.sat]
        acc = {}
        for r in x.itertuples(index=False):
            for p in 'xy':
                prof = getattr(r, f'{p}_prof')
                t = np.arange(len(prof)) * SAMPLE_NS - getattr(r, f'{p}_t0')
                z = t * r.v_drift_um_ns / 1000
                for bi, q in zip(np.floor(z / 2.0).astype(int), prof / r.sec):
                    acc.setdefault(bi, []).append(q)
        for bi, qs in sorted(acc.items()):
            if len(qs) > 300:
                rows.append(dict(arm=arm, depth_mm=2.0 * bi + 1.0, n=len(qs),
                                 frac=float(np.mean(qs)), frac_err=float(np.std(qs) / np.sqrt(len(qs)))))
    return pd.DataFrame(rows)


def gain_map(D: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in 'AC':
        x = D[D.sel & (D.arm == arm)]
        x = x[x.q_plat.notna()]
        m = float(R[(R.arm == arm) & (R.estimator == 'q_plat')].mpv.iloc[0])
        iu = np.floor((x.x_local + 200) / CELL).astype(int)
        iv = np.floor((x.y_local + 200) / CELL).astype(int)
        for (i, j), g in x.groupby([iu, iv]):
            if len(g) < 40:
                continue
            f = LD.fit(g.q_plat.to_numpy(), lo_q=0.05, hi_mult=2.5, nbins=25)
            if not (f['mpv_landau'] > 0 and f['mpv_err'] < 0.15 * f['mpv_landau']):
                continue
            rows.append(dict(arm=arm, iu=i, iv=j, x=-200 + (i + .5) * CELL, y=-200 + (j + .5) * CELL,
                             n=len(g), gain=f['mpv_landau'] / m, err=f['mpv_err'] / m))
    return pd.DataFrame(rows)


def same_muon(D: pd.DataFrame, G: pd.DataFrame) -> dict:
    """A and C on the same muon: correlation of the gain-corrected q."""
    x = D[D.sel].copy()
    gm = {(r.arm, r.iu, r.iv): r.gain for r in G.itertuples()}
    iu = np.floor((x.x_local + 200) / CELL).astype(int)
    iv = np.floor((x.y_local + 200) / CELL).astype(int)
    x['g'] = [gm.get(k, np.nan) for k in zip(x.arm, iu, iv)]
    x['qc'] = x.q_plat / x.g
    p = x.pivot_table(index=['subrun', 'event_id'], columns='arm', values=['qc', 'q_plat'])
    p = p.dropna()
    if len(p) < 30:
        return dict(n=int(len(p)))
    a, c = p[('qc', 'A')].to_numpy(), p[('qc', 'C')].to_numpy()
    from scipy.stats import spearmanr
    return dict(n=int(len(p)), spearman=float(spearmanr(a, c).correlation),
                spearman_raw=float(spearmanr(p[('q_plat', 'A')], p[('q_plat', 'C')]).correlation))


def main() -> int:
    D = load()
    print('selection:', D.groupby('arm').sel.agg(['size', 'sum']).to_dict(),
          'in_window', D.in_window.mean().round(3), 'in_map', D.in_map.mean().round(3),
          'sat', D.sat.mean().round(4))
    ctrl = D[D.sel].groupby('arm').apply(lambda g: pd.Series(dict(
        ctrl_med=np.median((g.x_Q_ctrl + g.y_Q_ctrl) / (g.x_Q + g.y_Q)),
        ctrl_iqr=np.subtract(*np.quantile((g.x_Q_ctrl + g.y_Q_ctrl) / (g.x_Q + g.y_Q), [.75, .25])))))
    R = resolution(D)
    T = two_mip(D, R)
    P = vs_path(D)
    Z = depth_profile(D)
    G = gain_map(D, R)
    S = same_muon(D, G)
    xy = D[D.sel].groupby('arm').apply(lambda g: pd.Series(dict(
        qx_over_qy=float(np.median(g.x_Q / g.y_Q)))))
    for name, df in (('resolution', R), ('two_mip', T), ('vs_path', P), ('depth', Z), ('gain_map', G)):
        df.to_csv(M2 / f'{name}.csv', index=False)
    D[D.sel][['arm', 'subrun', 'event_id', 'sec', 'path_mm', 'q_whole', 'q_trunc_t', 'q_trunc_s', 'q_plat', 'sat', 'in_window',
              'x_Q', 'y_Q', 'x_local', 'y_local', 'q_per_len']].to_parquet(M2 / 'dedx_selected.parquet')
    summ = dict(n_sel=D.groupby('arm').sel.sum().to_dict(), frac_in_window=float(D.in_window.mean()),
                frac_sat=float(D.sat.mean()), ctrl=ctrl.to_dict(orient='index'), same_muon=S,
                xy=xy.to_dict(orient='index'),
                gain_map_spread={a: float(1.4826 * np.median(np.abs(G[G.arm == a].gain - G[G.arm == a].gain.median())))
                                 for a in 'AC'},
                gain_map_range={a: [float(G[G.arm == a].gain.quantile(q)) for q in (0.05, 0.95)] for a in 'AC'})
    (M2 / 'summary.json').write_text(json.dumps(summ, indent=1))
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(R.round(3).to_string(index=False))
        print(T.round(3).to_string(index=False))
        print(P.round(3).to_string(index=False))
        print(Z.round(4).to_string(index=False))
        print(json.dumps(summ, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
