#!/usr/bin/env python3
"""hits_arm.py [--out results/fleet.json] [--fig figures/fleet.pdf] -- H0 (the production hits chain) against W (the
pass-1 waveform fit) on the SAME events, five chambers: the Paper II fleet figure (papers/paper2/figures.toml fig.fleet).

H0 is the rr1 hits chain (Analysis_rr1/<row>/cache/event_results_veto50.pkl: the decode the pass seeds from, sigrel 0.1,
spark veto 50): per view a straight line t(x) through the cluster's per-strip hit times (slope dt/dx [ns/mm]) and the
mesh position (strip of the earliest hit).  It is given every advantage W has:
  angle     its angle scale is calibrated against M3 on the row itself (tan_ref = a + k dx/dt, 3-MAD trimmed), exactly
            what kw/w0 do for W; both arms then get FD's convention (global d(tan_ref) line removed, the cross-validated
            2D 40 mm tilt map removed, s68 over |d| <= 5 deg).  k gives the hits' implied v = 1/k, set against W's v.
  position  both arms re-aligned on the row (offset + rotation + scale, xasym_sys.realign); core sigma of the mesh
            residual on the common events
  efficiency  on interior_good rays (common.mx17_fiducial, rays_<det>.parquet), both views measured and the radial
            residual of the re-aligned pick within 5 mm (one definition for both arms; effbudget's det_prod_5 is the
            Paper I number for W and is quoted beside it)
Events: 03_angles' interior_good sample (the frames), W fit ok and an H0 fit in the view.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

import floorlib as C
from xasym_sys import core_sigma, realign

sys.path[:0] = [C.REPO, os.path.join(C.REPO, 'cosmic_bench_analysis')]
HITS = os.environ.get('FLOOR_HITS') or os.path.join(C.HERE, 'hits')
RAYS = os.environ.get('FLOOR_RAYS') or os.path.join(C.HERE, 'rays')
DETS = {'det2': 'mx17_2', 'det3': 'mx17_3', 'det4': 'mx17_4', 'det6': 'mx17_6', 'det7': 'mx17_7'}
V_W = {}


def hits_table(key):
    L = pickle.load(open(os.path.join(HITS, f'{key}.pkl'), 'rb'))
    rows = []
    for e in L:
        r = dict(event_id=int(e.event_id))
        for p in ('x', 'y'):
            f = getattr(e, f'{p}_fit', None)
            ok = f is not None and np.isfinite(f.slope_ns_per_mm) and abs(f.slope_ns_per_mm) > 1e-9
            r[f'h{p}_u'] = 1.0 / f.slope_ns_per_mm if ok else np.nan          # dx/dt [mm/ns]
            r[f'h{p}_pos'] = f.mesh_position_mm if f is not None else np.nan
            r[f'h{p}_n'] = f.n_strips if f is not None else 0
        rows.append(r)
    H = pd.DataFrame(rows)
    return H[~H.event_id.duplicated(keep=False)].set_index('event_id')


def robust_line(x, y, it=4):
    k = np.isfinite(x) & np.isfinite(y)
    c = np.polyfit(x[k], y[k], 1)
    for _ in range(it):
        r = y - np.polyval(c, x)
        mad = np.nanmedian(np.abs(r[k] - np.nanmedian(r[k])))
        k = np.isfinite(x) & np.isfinite(y) & (np.abs(r) < 3 * 1.4826 * mad)
        c = np.polyfit(x[k], y[k], 1)
    return c


def fd_s68(d, tref, p0, o_p0, eid):
    d = np.asarray(d, float)
    k = np.abs(d) < 5
    ds = d - np.polyval(np.polyfit(tref[k], d[k], 1), tref)
    half = (eid % 2).astype(int)
    dt = C.xval(ds, (np.floor(p0 / C.CELL_MM).astype(int), np.floor(o_p0 / C.CELL_MM).astype(int)), half)
    return C.boot_s68(ds, n=100), C.boot_s68(dt, n=100)


def one(det, key):
    F = C.load(key)
    H = hits_table(key)
    out = dict(key=key, views={})
    for p in ('x', 'y'):
        g = C.sample(F, p)
        u = H[f'h{p}_u'].reindex(g.event_id).to_numpy()
        hp = H[f'h{p}_pos'].reindex(g.event_id).to_numpy()
        m = np.isfinite(u) & np.isfinite(hp)
        g, u, hp = g[m].reset_index(drop=True), u[m], hp[m]
        tref = g.tan_ref.to_numpy()
        # u = dx/dt = v tan: the binned-median line u(tan_ref) (20 quantiles of tan_ref) is stable against the huge u
        # of near-flat hit slopes, which a regression of tan_ref on u is not
        qb = pd.qcut(tref, 20, labels=False, duplicates='drop')
        bm = pd.DataFrame(dict(t=tref, u=u, b=qb)).groupby('b').agg(t=('t', 'median'), u=('u', 'median'), n=('u', 'size'))
        vh, ah = np.polyfit(bm.t, bm.u, 1, w=np.sqrt(bm.n))
        c = (1.0 / vh, -ah / vh)
        th = c[1] + c[0] * u
        d_h = np.degrees(np.arctan(th) - np.arctan(tref))
        eid = g.event_id.to_numpy()
        (hs, hse), (ht, hte) = fd_s68(d_h, tref, g.p0.to_numpy(), g.o_p0.to_numpy(), eid)
        res = {}
        for arm in ('d_vt', 'd_pass'):
            (ws, wse), (wt, wte) = fd_s68(g[arm].to_numpy(), tref, g.p0.to_numpy(), g.o_p0.to_numpy(), eid)
            res[arm] = dict(s68_slope=ws, s68_slope_err=wse, s68=wt, s68_err=wte)
        mx, my = g.r_mesh_x.to_numpy(), g.r_mesh_y.to_numpy()
        mv = mx if p == 'x' else my
        ok = np.isfinite(mx) & np.isfinite(my)
        rw = realign(g.p0.to_numpy()[ok] - mv[ok], mx[ok], my[ok])
        rh = realign(hp[ok] - mv[ok], mx[ok], my[ok])
        cw, _ = core_sigma(rw[np.abs(rw) < 15])
        ch, _ = core_sigma(rh[np.abs(rh) < 15])
        out['views'][p] = dict(n_common=int(len(g)), W=res, H0=dict(s68_slope=hs, s68_slope_err=hse, s68=ht, s68_err=hte),
                               hits_k_mm_per_ns_inv=float(c[0]), hits_v_implied_um_ns=float(1e3 / abs(c[0])),
                               pos_core_um=dict(W=1e3 * cw, H0=1e3 * ch))
    # efficiency on interior_good rays, one definition for both arms
    from common.mx17_fiducial import fiducial_mask
    R = pd.read_parquet(os.path.join(RAYS, f'{key}.parquet')).drop_duplicates('event_id', keep=False)
    ig = fiducial_mask('interior_good', R.mesh_x.to_numpy(), R.mesh_y.to_numpy(), R.tan_x.to_numpy(),
                       R.tan_y.to_numpy(), DETS[det])
    R = R[ig].set_index('event_id')
    hx, hy = H.hx_pos.reindex(R.index).to_numpy(), H.hy_pos.reindex(R.index).to_numpy()
    mx, my = R.mesh_x.to_numpy(), R.mesh_y.to_numpy()
    eff = {}
    for arm, (px, py) in (('W', (R.x_p0.to_numpy(), R.y_p0.to_numpy())), ('H0', (hx, hy))):
        okp = np.isfinite(px) & np.isfinite(py)
        rx = np.full(len(R), np.nan)
        ry = np.full(len(R), np.nan)
        rx[okp] = realign(px[okp] - mx[okp], mx[okp], my[okp])
        ry[okp] = realign(py[okp] - my[okp], mx[okp], my[okp])
        eff[arm] = float(np.mean(np.hypot(rx, ry) < 5))
    eff['W_effbudget_det_prod_5'] = float(R.det_prod_5.mean()) if 'det_prod_5' in R else None
    out['eff_5mm'] = eff
    out['n_rays_ig'] = int(len(R))
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dets = list(res)
    x = np.arange(len(dets))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for i, p in enumerate(('x', 'y')):
        w = [res[d]['views'][p]['W']['d_vt']['s68'] for d in dets]
        h = [res[d]['views'][p]['H0']['s68'] for d in dets]
        off = -0.18 + 0.36 * i
        ax[0].bar(x + off - 0.08, h, 0.16, color='0.65' if p == 'x' else '0.8', label=f'H0 hits, {p.upper()}')
        ax[0].bar(x + off + 0.08, w, 0.16, color='C0' if p == 'x' else 'C9', label=f'W waveforms, {p.upper()}')
    ax[0].set_ylabel('angle s68 [deg] (same events)')
    ax[0].legend(fontsize=7, ncol=2)
    for i, p in enumerate(('x', 'y')):
        w = [res[d]['views'][p]['pos_core_um']['W'] for d in dets]
        h = [res[d]['views'][p]['pos_core_um']['H0'] for d in dets]
        off = -0.18 + 0.36 * i
        ax[1].bar(x + off - 0.08, h, 0.16, color='0.65' if p == 'x' else '0.8')
        ax[1].bar(x + off + 0.08, w, 0.16, color='C0' if p == 'x' else 'C9')
    ax[1].set_ylabel('position core sigma vs M3 [um]\n(M3 pointing + scattering included)')
    ax[2].bar(x - 0.12, [100 * res[d]['eff_5mm']['H0'] for d in dets], 0.24, color='0.65', label='H0 hits')
    ax[2].bar(x + 0.12, [100 * res[d]['eff_5mm']['W'] for d in dets], 0.24, color='C0', label='W waveforms')
    ax[2].set_ylabel('efficiency, within 5 mm [%]')
    ax[2].legend(fontsize=7)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(dets)
        a.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'fleet.json'))
    ap.add_argument('--fig', default=os.path.join(C.HERE, 'figures', 'fleet.pdf'))
    a = ap.parse_args()
    res = {}
    for det, key in C.CALIB.items():
        res[det] = one(det, key)
        r = res[det]
        print(det, {p: (v['n_common'], round(v['H0']['s68'], 3), round(v['W']['d_vt']['s68'], 3),
                        round(v['hits_v_implied_um_ns'], 1), round(v['pos_core_um']['H0']), round(v['pos_core_um']['W']))
                    for p, v in r['views'].items()}, r['eff_5mm'])
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    figure(res, a.fig)


if __name__ == '__main__':
    main()
