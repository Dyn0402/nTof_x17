#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_map.py -- an in-situ angle correction for chambers A and C from the A-C cosmic lines, validated on held-out
runs and, independently, on the beam capsule pointing.

Truth: the joined A-C line from the two positions (line_residuals.py build: every beam-off sub-run, one gated track
per chamber, lines within 60 mm).  Residual r = (global slope d_p/d_z of the chamber's track) - (joined slope j).
"Clean" lines: the other chamber's same view and this chamber's other view agree with the line within CLEAN (so a
shower / two-particle event, whose tails are shared by all four views, does not train the map), |j| < 0.45 and
|r| < 0.15.

Model per chamber and view: r = a(x, y) + s(x, y) * j, with a (tilt, tan units) and s (scale) per 50 mm cell of the
track's local position (x_local, y_local), bilinearly interpolated between cell centres; cells with fewer than N_MIN
clean lines take the chamber-wide fit.  Corrected slope = (slope - a) / (1 + s).

    python -m rerun.ntof.improve.angle_map fit        -> <out>/angle_map.json, held-out table (train even / test odd
                                                         runs and the reverse)
    python -m rerun.ntof.improve.angle_map beam       -> capsule pointing on the late (>= 20 ms) beam tracks with
                                                         and without the map: core width, excess, sharpest focus s
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.improve.track_quality import out_dir, load_tracks, source_x, pointing  # noqa: E402

CLEAN = 0.10
EDGES = np.arange(-200.0, 201.0, 50.0)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
N_MIN = 80
OTHER = {'a': 'c', 'c': 'a'}
ARM = {'a': 'A', 'c': 'C'}


def clean_lines(M: pd.DataFrame, s: str, p: str) -> pd.DataFrame:
    o, q = OTHER[s], ('y' if p == 'x' else 'x')
    j, jq = M[f'j{p}'], M[f'j{q}']
    slope = M[f'd_{p}_{s}'] / M[f'd_z_{s}']
    r = slope - j
    ok = (((M[f'd_{p}_{o}'] / M[f'd_z_{o}'] - j).abs() < CLEAN)
          & ((M[f'd_{q}_{s}'] / M[f'd_z_{s}'] - jq).abs() < CLEAN)
          & (j.abs() < 0.45) & (r.abs() < 0.15))
    return pd.DataFrame({'x': M.loc[ok, f'x_local_{s}'], 'y': M.loc[ok, f'y_local_{s}'], 'j': j[ok], 'r': r[ok],
                         'run': M.loc[ok, 'run']})


def robust_line(j, r, n_iter=3):
    keep = np.ones(len(j), bool)
    a = s = 0.0
    for _ in range(n_iter):
        if keep.sum() < 20:
            break
        s, a = np.polyfit(j[keep], r[keep], 1)
        res = r - (a + s * j)
        sig = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < 3 * sig
    return float(a), float(s), int(keep.sum())


def fit_map(G: pd.DataFrame) -> dict:
    a0, s0, _ = robust_line(G.j.to_numpy(), G.r.to_numpy())
    nc = len(CENTRES)
    A = np.full((nc, nc), a0)
    S = np.full((nc, nc), s0)
    N = np.zeros((nc, nc), int)
    ix = np.clip(np.digitize(G.x, EDGES) - 1, 0, nc - 1)
    iy = np.clip(np.digitize(G.y, EDGES) - 1, 0, nc - 1)
    for i in range(nc):
        for k in range(nc):
            m = (ix == i) & (iy == k)
            N[i, k] = m.sum()
            if m.sum() >= N_MIN:
                A[i, k], S[i, k], _ = robust_line(G.j.to_numpy()[m], G.r.to_numpy()[m])
    return dict(a=A.tolist(), s=S.tolist(), n=N.tolist(), a0=a0, s0=s0, centres=CENTRES.tolist())


def interp(mp: dict, x, y):
    """Bilinear between cell centres (clamped at the outer centres)."""
    c = np.asarray(mp['centres'])
    A, S = np.asarray(mp['a']), np.asarray(mp['s'])
    x = np.clip(np.asarray(x, float), c[0], c[-1])
    y = np.clip(np.asarray(y, float), c[0], c[-1])
    i = np.clip(np.searchsorted(c, x) - 1, 0, len(c) - 2)
    k = np.clip(np.searchsorted(c, y) - 1, 0, len(c) - 2)
    tx = (x - c[i]) / (c[i + 1] - c[i])
    ty = (y - c[k]) / (c[k + 1] - c[k])

    def bl(Z):
        return ((1 - tx) * (1 - ty) * Z[i, k] + tx * (1 - ty) * Z[i + 1, k]
                + (1 - tx) * ty * Z[i, k + 1] + tx * ty * Z[i + 1, k + 1])
    a, s = bl(A), bl(S)
    a = np.where(np.isfinite(x) & np.isfinite(y), a, mp['a0'])
    s = np.where(np.isfinite(x) & np.isfinite(y), s, mp['s0'])
    return a, s


def correct(slope, x, y, mp):
    a, s = interp(mp, x, y)
    return (slope - a) / (1.0 + s)


def mad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def fit() -> dict:
    M = pd.read_parquet(out_dir() / 'lines.parquet')
    rn = M.run.str.replace('run_', '').astype(int)
    maps, rows = {}, []
    for s in 'ac':
        for p in 'xy':
            G = clean_lines(M, s, p)
            maps[f'{ARM[s]}_{p}'] = fit_map(G)
            # held-out: train on one run parity, test on the other, on ALL lines (not only clean) and on clean ones
            for tr in (0, 1):
                mp = fit_map(G[(G.run.str.replace('run_', '').astype(int) % 2) == tr])
                te = (rn % 2) != tr
                j = M.loc[te, f'j{p}'].to_numpy()
                slope = (M.loc[te, f'd_{p}_{s}'] / M.loc[te, f'd_z_{s}']).to_numpy()
                x, y = M.loc[te, f'x_local_{s}'].to_numpy(), M.loc[te, f'y_local_{s}'].to_numpy()
                sc = correct(slope, x, y, mp)
                g = (np.abs(j) < 0.45)
                Gt = G[(G.run.str.replace('run_', '').astype(int) % 2) != tr]
                cs = correct(Gt.r.to_numpy() + Gt.j.to_numpy(), Gt.x.to_numpy(), Gt.y.to_numpy(), mp)
                edge = (np.abs(Gt.x) > 125) | (np.abs(Gt.y) > 125)
                rows.append(dict(arm=ARM[s], view=p, train=tr,
                                 sig_all_before=mad((slope - j)[g]), sig_all_after=mad((sc - j)[g]),
                                 tail_before=float(np.mean(np.abs(slope - j)[g] > 0.15)),
                                 tail_after=float(np.mean(np.abs(sc - j)[g] > 0.15)),
                                 sig_clean_before=mad(Gt.r), sig_clean_after=mad(cs - Gt.j.to_numpy()),
                                 sig_edge_before=mad(Gt.r[edge]), sig_edge_after=mad((cs - Gt.j.to_numpy())[edge]),
                                 bias_edge_rms_before=float(np.sqrt(np.mean(np.square(
                                     pd.Series(Gt.r.to_numpy()[edge]).groupby(
                                         (np.digitize(Gt.x[edge], EDGES) * 10 + np.digitize(Gt.y[edge], EDGES))).median())))),
                                 n_test_clean=len(Gt)))
    meta = dict(built_from='rerun/ntof/improve/line_residuals.py lines.parquet (all beam-off sub-runs, pass 1 view)',
                model='r = a(x,y) + s(x,y) j per 50 mm cell, bilinear; corrected = (slope - a)/(1 + s), global d_p/d_z',
                clean=CLEAN, n_min=N_MIN, edges=EDGES.tolist())
    json.dump(dict(meta=meta, maps=maps), open(out_dir() / 'angle_map.json', 'w'), indent=1)
    T = pd.DataFrame(rows)
    T.to_csv(out_dir() / 'angle_map_heldout.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.round(4).to_string())
    return maps


def beam():
    maps = json.load(open(out_dir() / 'angle_map.json'))['maps']
    d = load_tracks()
    xs = source_x(d)
    d = d[(d.ms >= 20) & (d.tanx.abs() >= 0.1)]
    rng = np.random.default_rng(7)
    rows = []
    for arm, g in d.groupby('arm'):
        sx = (g.d_x / g.d_z).to_numpy()
        sx_c = correct(sx, g.x_local.to_numpy(), g.y_local.to_numpy(), maps[f'{arm}_x'])
        pz, pos = g.p0_z.to_numpy(), g.pos.to_numpy()
        for lab, sl in (('before', sx), ('after', sx_c)):
            disp = -pz * sl
            miss = pos + disp - xs[arm]
            sh = rng.permutation(len(disp))
            miss0 = pos + disp[sh] - xs[arm]
            r = pointing(miss, miss0)
            # sharpest focus scale on the nominal axis (focus_split's estimator, median |miss| perpendicular)
            S = np.round(np.arange(0.9, 1.1001, 0.005), 3)
            med = [np.median(np.abs((pos + s_ * disp - xs[arm]) / np.hypot(s_ * sl, 1.0))) for s_ in S]
            k = int(np.argmin(med))
            # edge region of the chamber
            e = (g.x_local.abs() > 125).to_numpy()
            re = pointing(miss[e], miss0[e])
            rows.append(dict(arm=arm, map=lab, n=r['n'], exc15=r['exc1'], exc45=r['exc2'], tail=r['tail'],
                             w68=r['w68'], focus_s=S[k], med_miss=med[k], edge_exc15=re.get('exc1'),
                             edge_w68=re.get('w68'), edge_n=re['n']))
    T = pd.DataFrame(rows)
    T.to_csv(out_dir() / 'angle_map_beam.csv', index=False)
    print(T.round(4).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['fit', 'beam'])
    a = ap.parse_args()
    fit() if a.step == 'fit' else beam()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
