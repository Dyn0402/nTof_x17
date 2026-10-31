#!/usr/bin/env python3
"""position_ms.py [--out results/position_ms.json] -- the bench position resolution against M3 with its multiple-
scattering term made explicit, to set it beside the SPS 190-210 µm (README.md §3).

The bench reference is a 4-plane M3 line (two doublets, z = 24 ... 1302 mm, the DUT near 700 mm).  Its MS-free pointing
P (m3_self_resolution, per-plane sigma, 0.21-0.24 mm) is what res_rows subtracts.  A muon that scatters between the
doublets leaves the M3 line off its true position at the DUT by about kink x lever; the 4-plane chi2 < 1 cut bounds the
kink but does not remove it.  So the position residual should grow with the M3 chi2 of the matching coordinate, and its
chi2 -> 0 limit is the MS-free bench residual.  Per row and view (interior_good, the mesh residual p0 - mesh in the raw
strip frame, re-aligned for offset + rotation + scale as xasym_sys.realign):
  core sigma in M3-chi2 quintiles of each M3 coordinate; sigma^2 linear in chi2 -> intercept (chi2 -> 0) and the mean
  ms term   q(core all, core at chi2 -> 0): what the reference's own scattering adds
  chamber   q(core at chi2 -> 0, P): the bench chamber resolution once the reference is taken out, near-normal
            (|theta| < 5 deg) as well as all angles, to compare with the SPS head-on 190-210 µm (det4, run63)
"""
import argparse
import json
import os
import sys

import numpy as np

import floorlib as C
from xasym_sys import core_sigma, realign

sys.path.insert(0, os.path.join(C.REPO, 'rerun', 'paper1_res'))


def pointing_um(det):
    """the row's M3 pointing per view as rerun/paper1_res quoted it (summary.json), µm."""
    s = json.load(open(os.path.join(C.REPO, 'rerun', 'paper1_res', 'results', 'summary.json')))
    rows = s.get('rows') if isinstance(s, dict) else s
    return rows


def intercept(res, chi2, nboot=60, seed=5):
    """sigma^2 linear in the M3 chi2 over quintiles -> (chi2 -> 0 core sigma, its bootstrap error) [mm]."""
    def fit(r, c):
        ed = np.nanpercentile(c, [0, 20, 40, 60, 80, 100])
        ed[-1] += 1e-9
        xs, ys = [], []
        for lo, hi in zip(ed[:-1], ed[1:]):
            m = (c >= lo) & (c < hi) & (np.abs(r) < 15)
            xs.append(float(np.mean(c[m])))
            ys.append(core_sigma(r[m])[0] ** 2)
        sl, ic = np.polyfit(xs, ys, 1)
        return float(np.sqrt(max(ic, 0)))
    rng = np.random.default_rng(seed)
    v = fit(res, chi2)
    b = []
    for _ in range(nboot):
        i = rng.integers(0, len(res), len(res))
        b.append(fit(res[i], chi2[i]))
    return v, float(np.nanstd(b))


def one(key, det, P):
    F = C.load(key)
    out = {}
    for p in ('x', 'y'):
        g = F[(F.view == p) & F.ig & np.isfinite(F.p0) & np.isfinite(F.r_mesh_x) & np.isfinite(F.r_mesh_y)].copy()
        mx, my = g.r_mesh_x.to_numpy(), g.r_mesh_y.to_numpy()
        mv = mx if p == 'x' else my
        res = realign(g.p0.to_numpy() - mv, mx, my)
        g['res'] = res
        th = np.abs(np.degrees(np.arctan(g.tan_ref.to_numpy())))
        o = {}
        for c in ('m3_chi2x', 'm3_chi2y'):
            ed = np.nanpercentile(g[c], [0, 20, 40, 60, 80, 100])
            ed[-1] += 1e-9
            xs, ys, bins = [], [], []
            for lo, hi in zip(ed[:-1], ed[1:]):
                m = (g[c] >= lo) & (g[c] < hi) & (np.abs(g.res) < 15)
                cs, s68 = core_sigma(g.res[m].to_numpy())
                xs.append(float(g[c][m].mean()))
                ys.append(cs ** 2)
                bins.append((round(xs[-1], 3), round(1e3 * cs, 1)))
            sl, ic = np.polyfit(xs, ys, 1)
            o[c] = dict(bins_um=bins, slope=float(sl), intercept_um=float(1e3 * np.sqrt(max(ic, 0))))
        call, _ = core_sigma(g.res[np.abs(g.res) < 15].to_numpy())
        cn, _ = core_sigma(g.res[(np.abs(g.res) < 15) & (th < 5)].to_numpy())
        # the coordinate whose chi2 matters is the one with the larger slope
        cc = max(('m3_chi2x', 'm3_chi2y'), key=lambda c: o[c]['slope'])
        r_all, c_all = g.res.to_numpy(), g[cc].to_numpy()
        i0, e0 = intercept(r_all, c_all)
        nn = th < 5
        i5, e5 = intercept(r_all[nn], c_all[nn])
        Pv = P.get(p) if P else None

        def cham(i, e):
            if Pv is None:
                return None, None
            v = C.q(i, Pv)
            return 1e3 * v, (1e3 * i * e / v if v > 0 else None)
        ch, che = cham(i0, e0)
        cn5, cn5e = cham(i5, e5)
        o.update(n=int(len(g)), n_near_normal=int(nn.sum()), core_all_um=1e3 * call, core_near_normal_um=1e3 * cn,
                 matched=cc, chi2_0_um=1e3 * i0, chi2_0_err_um=1e3 * e0, chi2_0_near_normal_um=1e3 * i5,
                 chi2_0_near_normal_err_um=1e3 * e5, ms_um=1e3 * C.q(call, i0), P_um=None if Pv is None else 1e3 * Pv,
                 chamber_um=ch, chamber_err_um=che, chamber_near_normal_um=cn5, chamber_near_normal_err_um=cn5e)
        out[p] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'position_ms.json'))
    a = ap.parse_args()
    import pandas as pd
    R = pd.read_csv(os.path.join(C.REPO, 'rerun', 'paper1_res', 'results', 'rows.csv'))
    Pk = {}
    for k, v, pm in R[['key', 'view', 'P_mm']].itertuples(index=False):
        Pk.setdefault(k, {})[v] = float(pm)
    out = {}
    for det, key in list(C.CALIB.items()) + [(d, k) for d, k in C.HEADLINE.items()]:
        role = 'headline' if key in C.HEADLINE.values() else 'calib'
        r = one(key, det, Pk.get(key))
        out[f'{det}|{role}'] = r
        for p, o in r.items():
            f = lambda v: 'n/a' if v is None else f'{v:.0f}'
            print(f"{det} {role} {p}: core all {o['core_all_um']:.0f} | {o['matched']} chi2->0 "
                  f"{o['chi2_0_um']:.0f}±{o['chi2_0_err_um']:.0f} | MS {o['ms_um']:.0f} | P {f(o['P_um'])} | chamber "
                  f"{f(o['chamber_um'])}±{f(o['chamber_err_um'])} | near-normal (n {o['n_near_normal']}) chi2->0 "
                  f"{o['chi2_0_near_normal_um']:.0f}±{o['chi2_0_near_normal_err_um']:.0f} chamber "
                  f"{f(o['chamber_near_normal_um'])}±{f(o['chamber_near_normal_err_um'])}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
