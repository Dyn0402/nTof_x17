#!/usr/bin/env python3
"""m3feat.py <det> -- M3 cluster-level features for every event of the joined table.

Input: the DataReader cluster trees and rays rebuilt by m3clus/job.sh (production v2 chain, bit-identical rays,
plus the per-layer measured positions of commit b71ffc9), ~/x17/cosmic_bench/outliers/m3clus/.
Per event (one row), M3 frame unless stated:
  m3_ncl_<plane>      clusters on each of the 8 planes (X0..X3, Y0..Y3); m3_extra = sum over planes of (n - 1)+
  m3_kink_<v>         angle between the upper doublet (z 1302/1185) and the lower doublet (z 144/24) lines [mrad]
  m3_dupdn_<v>        their separation at the DUT [mm] (raw detector frame, see below)
  m3_nalt_<v>         number of OTHER 4-layer cluster combinations with chi2 < 1 mm^2 (the production recipe)
  m3_alt_<v>          JSON list (<= 6) of alternative combinations with chi2 < 5: [chi2, position at the DUT, tan],
                      position and tan already in the RAW DETECTOR frame of that view, so they compare directly with
                      the MX17 strip coordinate (p0) and the raw-frame tangents
  m3_up_<v>/m3_dn_<v> the upper / lower doublet line at the DUT, raw frame (position)
Raw frame: x_M3 * ref_x_sign at z_x / z_y, then the inverse alignment (cosmic_micro_tpc_analysis
attach_reference_positions); tangents rotated as _rotate_ref_tangents.
Strip -> mm: pos = a * ClusPos + b + c * (other coordinate at that layer), fitted on single-cluster planes against
the tracker's own measured positions (residual < 0.07 mm on every night; c is the stale angle_z on layers 1/3).
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'cosmic_bench_analysis')]
from olib import BASE  # noqa: E402

RUNS = {'det3': 'mx17_det3_saturday_scan_6-27-26__long_run_resist_490V_drift_1000V',
        'det4': 'mx17_det4_day_6-24-26__long_run', 'det7': 'mx17_det6_det7_overnight_6-26-26__long_run',
        'det6': 'mx17_det6_det7_overnight_6-26-26__long_run', 'det2': 'mx17_det2_det3_overnight_6-22-26__longer_run'}
ALIGN = {'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
         'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7',
         'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6', 'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'}
# plane index in MGv2_* -> (view, layer); z per layer
PLANES = {1: ('x', 0), 3: ('x', 1), 5: ('x', 2), 6: ('x', 3), 0: ('y', 0), 2: ('y', 1), 4: ('y', 2), 7: ('y', 3)}
ZL = [1302.0, 1185.0, 144.0, 24.0]
MAXCL = 6


def calib(an, ry):
    """(a, b, c) per plane from single-cluster planes vs the tracker's measured positions."""
    ie = {int(e): i for i, e in enumerate(an['evn'])}
    out = {}
    for p, (v, l) in PLANES.items():
        o = 'y' if v == 'x' else 'x'
        A, Y = [], []
        for _, r in ry.iterrows():
            i = ie.get(int(r.evn))
            if i is None or an['MGv2_NClus'][i][p] != 1:
                continue
            m, mo = r[f'm{v.upper()}{l}'], r[f'm{o.upper()}{l}']
            if np.isfinite(m) and np.isfinite(mo):
                A.append((an['MGv2_ClusPos'][i, p, 0], 1.0, mo)); Y.append(m)
        c, *_ = np.linalg.lstsq(np.array(A), np.array(Y), rcond=None)
        out[p] = c
    return out


def load_run(det):
    import uproot, awkward as ak
    an_all, ry_all = [], []
    for f in sorted(glob.glob(os.path.join(BASE, 'm3clus', f'{RUNS[det]}__*_analyse.root'))):
        a = uproot.open(f)['T'].arrays(['evn', 'MGv2_NClus', 'MGv2_ClusPos', 'MGv2_ClusAmpl', 'MGv2_ClusSize',
                                        'MGv2_Spark'], library='np')
        r = uproot.open(f.replace('_analyse', '_rays'))['T'].arrays(
            ['evn', 'Chi2X', 'Chi2Y'] + [f'm{c}{l}' for c in 'XY' for l in range(4)], library='ak')
        d = {'evn': ak.to_numpy(r.evn)}
        for k in r.fields[1:]:
            d[k] = ak.to_numpy(ak.fill_none(ak.firsts(r[k]), np.nan))
        an_all.append(a); ry_all.append(pd.DataFrame(d))
    return an_all, ry_all


def fit_line(z, x):
    A = np.c_[z, np.ones_like(z)]; s, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return s, b, float(((x - (s * z + b)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('det'); a = ap.parse_args(); det = a.det
    al = json.load(open(os.path.join(BASE, 'final', ALIGN[det], 'wft/final_phys/alignment/alignment.json')))
    th = np.deg2rad(al['theta_deg']); ct, st = np.cos(th), np.sin(th); cx, cy = al['centre_x'], al['centre_y']
    sx = al['ref_x_sign']; zx, zy = al['z_x'], al['z_y']

    def to_raw(xm, ym):                              # M3-frame point (already sign-corrected) -> raw det frame
        u, w = xm - cx - al['x_offset'], ym - cy - al['y_offset']
        return cx + ct * u + st * w, cy - st * u + ct * w

    def tan_raw(tx, ty):
        return ct * tx + st * ty, -st * tx + ct * ty

    J = pd.read_parquet(os.path.join(BASE, f'joined_{det}.parquet'), columns=['event_id', 'ref_mesh_x', 'ref_mesh_y'])
    want = set(J.event_id.astype(int))
    an_all, ry_all = load_run(det)
    rows = []
    for an, ry in zip(an_all, ry_all):
        C = calib(an, ry)
        ri = {int(e): i for i, e in enumerate(ry.evn)}
        for i, e in enumerate(an['evn']):
            e = int(e)
            if e not in want or e not in ri:
                continue
            r = ry.iloc[ri[e]]; out = {'event_id': e}
            ncl = an['MGv2_NClus'][i]
            out['m3_spark'] = int(np.sum(an['MGv2_Spark'][i] > 0))
            for p, (v, l) in PLANES.items():
                out[f'm3_ncl_{v}{l}'] = int(ncl[p])
            out['m3_extra'] = int(sum(max(int(n) - 1, 0) for n in ncl))
            # chosen layer positions
            ch = {v: np.array([r[f'm{v.upper()}{l}'] for l in range(4)]) for v in 'xy'}
            # candidate cluster positions per view/layer (mm, M3 frame)
            cand = {'x': [[] for _ in range(4)], 'y': [[] for _ in range(4)]}
            for p, (v, l) in PLANES.items():
                o = 'y' if v == 'x' else 'x'
                oth = ch[o][l] if np.isfinite(ch[o][l]) else 0.0
                n = min(int(ncl[p]), 300)
                amp = an['MGv2_ClusAmpl'][i, p, :n]; idx = np.argsort(-amp)[:MAXCL]
                cand[v][l] = [C[p][0] * an['MGv2_ClusPos'][i, p, k] + C[p][1] + C[p][2] * oth for k in idx]
            lines = {}
            for v in 'xy':
                m = ch[v]
                if np.all(np.isfinite(m)):
                    su = (m[0] - m[1]) / (ZL[0] - ZL[1]); sd = (m[2] - m[3]) / (ZL[2] - ZL[3])
                    out[f'm3_kink_{v}'] = 1e3 * (np.arctan(su) - np.arctan(sd))
                    zd = zx if v == 'x' else zy
                    lines[v] = dict(up=m[1] + su * (zd - ZL[1]), dn=m[2] + sd * (zd - ZL[2]))
                # alternative 4-layer combinations
                alts, nalt = [], 0
                if all(len(c) for c in cand[v]):
                    z = np.array(ZL)
                    for combo in itertools.product(*cand[v]):
                        xx = np.array(combo)
                        if np.all(np.isfinite(m)) and np.allclose(xx, m, atol=0.05):
                            continue                                  # the chosen ray
                        s, b, c2 = fit_line(z, xx)
                        if c2 < 1.0:
                            nalt += 1
                        if c2 < 5.0:
                            alts.append((c2, s, b))
                alts.sort()
                out[f'm3_nalt_{v}'] = nalt
                out[f'_alts_{v}'] = alts[:6]
            # transform alternatives / doublet lines to the raw detector frame. A view's alternative is paired
            # with the CHOSEN ray of the other view (the two views are tracked independently by M3).
            for v in 'xy':
                o = 'y' if v == 'x' else 'x'
                mo = ch[o]
                if not np.all(np.isfinite(mo)):
                    out[f'm3_alt_{v}'] = '[]'; continue
                so, bo, _ = fit_line(np.array(ZL), mo)
                res = []
                for c2, s, b in out.pop(f'_alts_{v}'):
                    if v == 'x':
                        X, Y = sx * (s * zx + b), so * zy + bo; TX, TY = sx * s, so
                    else:
                        X, Y = sx * (so * zx + bo), s * zy + b; TX, TY = sx * so, s
                    px, py = to_raw(X, Y); tx, ty = tan_raw(TX, TY)
                    # the raw view that this M3 view mainly feeds (alignment ~90 deg swaps them)
                    res.append([round(c2, 3), round(float(px), 2), round(float(py), 2), round(float(tx), 4), round(float(ty), 4)])
                out[f'm3_alt_{v}'] = json.dumps(res)
                if v in lines:
                    for k in ('up', 'dn'):
                        if v == 'x':
                            X, Y = sx * lines['x'][k], so * zy + bo
                        else:
                            X, Y = sx * (so * zx + bo), lines['y'][k]
                        px, py = to_raw(X, Y)
                        out[f'm3_{k}_{v}'] = json.dumps([round(float(px), 2), round(float(py), 2)])
            rows.append(out)
    M = pd.DataFrame(rows)
    M.to_parquet(os.path.join(BASE, f'm3feat_{det}.parquet'))
    print(det, len(M), 'events;', 'kink rms x/y [mrad]',
          np.nanstd(M.get('m3_kink_x', np.nan)), np.nanstd(M.get('m3_kink_y', np.nan)))


if __name__ == '__main__':
    main()
