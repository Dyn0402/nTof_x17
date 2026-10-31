#!/usr/bin/env python3
"""res_rows.py --out-dir D [--only k1,k2] [--dets mx17_2,...] -- Paper I bench resolutions per pass-1 row, two arms
(PASS1_RESOLUTION_2026-10-31.md).  Table level only: nothing is refitted, READ-ONLY on Analysis_pass1.

Per non-excluded registry row (rerun/pass_registry.csv), from
  the pass table + meta + full-coverage angles        Analysis_pass1/<rel>/wft/{events.parquet, events.meta.json,
                                                       angles/per_event.npz}
  the effbudget's per-ray table                        paper1_eff/budget/<rel>/rays_<det>.parquet (M3 ray in the
                                                       chamber's raw strip frame at the alignment z, fiducial region,
                                                       spark multiplicity, the row's p0 per view)
  the v(t) correction (rows of vt_corrected_rows.csv)  vt_correction/<rel>/events_vt.parquet (v_rel per event)

arms
  pass   the pass table as it is (tan_theta of events.parquet)
  vt     v -> v r(t) in the angle mapping, then w0/kw re-folded on the corrected mapping over the row's full-coverage
         events (set_w0kw's ladder: CROSSCHECK_2026-10-31 §2 'vt+fold'), taken as a difference: w0_vt = w0_pass +
         (w0(r) - w0(1)), kw_vt = kw_pass kw(r) / kw(1), so the ladder's own re-measurement scatter cancels and only
         what v(t) changes is applied.  Identical to pass on a row with no v(t) map.  The fold of kw into v
         (set_w0kw --fold-kw: v := v kw_mean, kw := kw / kw_mean) is angle-neutral, so it is not repeated.
  The angle s68 uses 03_angles' 68th percentile (closure); the positions the 68.27th.

selection (one per quantity, both arms identical)
  angle      03_angles' definition on interior_good (common.mx17_fiducial): every ray whose footprint is interior in
             both views and, on det7, on X ch 256-511; the view's fit ok; finite tangents.  Closure: s68 reproduces
             effbudget's s68_<p>_ig (03_angles --fiducial interior_good) to the printed digit.
  position   02_efficiency's accounting on interior_good: also the spark veto (multiplicity > wft.seed.SPARK_VETO_HITS
             leaves), and |residual| < 15 mm.  Residual per view in the RAW strip frame (the effbudget's res_x, res_y):
             mesh   p0 - M3 at the alignment z (the pass's position)
             mid    (p0 + 15 tan_fit) - (M3 + 15 tan_ref): the track at mid-gap (depends on the angle, hence on the arm)

statistics per (row, arm, view, selection)
  bias (median), s68 (68.27th percentile of |d - median|), core sigma (Gaussian fitted to the histogram in a FIXED
  window median +- 2 s68: no iterative clip, which latches onto tails), n.  Angle bins of |theta_ref|: 0-5, 5-10, 10-15,
  15-20 deg (the M3 telescope's acceptance ends near 20 deg: interior_good holds < 25 events above it per row).
  Position: the M3 pointing P(z) per M3 coordinate (m3_self_resolution/results.json per-plane sigma, a 4-point line
  fit interpolated to the row's z), rotated into the view (P_view^2 = cos^2 P_X'^2 + sin^2 P_Y'^2 with the alignment
  rotation), subtracted in quadrature from the CORE sigma (multiple scattering lives in the tails and is not
  subtracted: what is quoted is an upper bound on the chamber, see the report).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('RES_CODE') or os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'vt_within_rows'), os.path.join(REPO, 'rerun', 'vt_all')]
from wft import reco as wr  # noqa: E402
from wft.seed import SPARK_VETO_HITS  # noqa: E402
from common.mx17_fiducial import fiducial_mask  # noqa: E402
from vt_common import ladder, bow_aS  # noqa: E402
import vt_bench as vb  # noqa: E402

P1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EFF = '/eos/user/d/dneff/x17rerun/paper1_eff'
VTC = '/eos/user/d/dneff/x17rerun/vt_correction'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
ANG_BINS = (0.0, 5.0, 10.0, 15.0, 20.0)                # |theta_ref| [deg]; the M3 acceptance ends near 20 deg
RES_MAX_MM = 15.0
MID_MM = 15.0                                         # mid-gap height above the mesh
FIG_MIN = 50                                          # fewest events for a statistic
ANG_Q = 68.0                                          # 03_angles' s68 percentile (positions: 68.27)


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


def core_sigma(d, s68):
    """Gaussian fitted to the histogram of d in the fixed window median +- 2 s68 (truncation handled by the fit)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < FIG_MIN or not np.isfinite(s68) or s68 <= 0:
        return np.nan, np.nan
    med = np.median(d)
    lo, hi = med - 2 * s68, med + 2 * s68
    k = (d > lo) & (d < hi)
    nb = int(np.clip(np.sqrt(k.sum()), 12, 60))
    h, e = np.histogram(d[k], bins=nb, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    try:
        from scipy.optimize import curve_fit
        f = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
        p, cov = curve_fit(f, c, h, p0=(h.max(), med, s68), sigma=np.sqrt(np.maximum(h, 1)), maxfev=5000)
        return float(abs(p[2])), float(np.sqrt(cov[2, 2]))
    except Exception:
        return np.nan, np.nan


def stats(d, q=68.27):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d)
    if n < FIG_MIN:
        return dict(n=n, bias=np.nan, s68=np.nan, core=np.nan, core_err=np.nan, s68_err=np.nan)
    med = float(np.median(d))
    s = float(np.percentile(np.abs(d - med), q))
    cs, ce = core_sigma(d, s)
    return dict(n=n, bias=med, s68=s, s68_err=float(s / np.sqrt(2 * n)), core=cs, core_err=ce)


def hat_coeffs(z_used, z_eval):
    A = np.vstack([np.ones_like(z_used), z_used]).T
    M = np.linalg.inv(A.T @ A)
    return np.array([1.0, z_eval]) @ M @ A.T


M3R = json.load(open(os.path.join(REPO, 'mx_june_cosmic_qa', 'm3_self_resolution', 'results.json')))


def pointing(z):
    """M3 4-plane pointing at z per M3 coordinate [mm] (measured per-plane sigma; MS-free by construction)."""
    Z = np.array(M3R['z_layers'])
    out = {}
    for c in ('X', 'Y'):
        sk = np.array([L['sigma_geomean_um'] / 1000.0 for L in M3R['coords'][c]['layers']])
        g = hat_coeffs(Z, z)
        out[c] = float(np.sqrt(np.sum(g ** 2 * sk ** 2)))
    return out


def view_pointing(al):
    """per detector view: the M3 pointing projected into the raw strip frame (alignment rotation theta)."""
    th = np.deg2rad(al['theta_deg']); c, s = np.cos(th), np.sin(th)
    px, py = pointing(al['z_x']), pointing(al['z_y'])
    # mesh_x = cx + c u + s v (u ~ M3 X, v ~ M3 Y): the view-x residual carries c^2 P_X^2 + s^2 P_Y^2
    return dict(x=float(np.sqrt(c ** 2 * px['X'] ** 2 + s ** 2 * px['Y'] ** 2)),
                y=float(np.sqrt(s ** 2 * py['X'] ** 2 + c ** 2 * py['Y'] ** 2)),
                z_x=al['z_x'], z_y=al['z_y'])


def load_calib(wdir):
    m = json.load(open(os.path.join(wdir, 'events.meta.json')))
    ac = m['angle_constants']
    from types import SimpleNamespace
    cal = SimpleNamespace(v_drift=float(m['bundle']['v_drift']), w0=dict(ac['w0']), kw=dict(ac['kw']), bow={})
    if (m.get('bow') or {}).get('applied'):
        cal.bow = vb.golden_bow(P1, m['run']['detector'], m['bow']['source'])
    return m, cal


def vt_arm(wdir, df, cal, vrel):
    """per view: eid -> tan in the vt arm (full-coverage events), the re-measured w0/kw, closure of the const map."""
    tcx = (df.x_w * 1e3 - cal.w0['x']) / (cal.kw['x'] * cal.v_drift)
    tcy = (df.y_w * 1e3 - cal.w0['y']) / (cal.kw['y'] * cal.v_drift)
    U = ((df.x_p0 + wr.BOW_MID_MM * tcx) - 200.0) / 100.0
    W = ((df.y_p0 + wr.BOW_MID_MM * tcy) - 200.0) / 100.0
    U = U.where(np.isfinite(U) & (U.abs() <= wr.BOW_UW_MAX) & df.x_ok, 0.0).fillna(0.0).to_numpy()
    W = W.where(np.isfinite(W) & (W.abs() <= wr.BOW_UW_MAX) & df.y_ok, 0.0).fillna(0.0).to_numpy()
    z = np.load(os.path.join(wdir, 'angles', 'per_event.npz'))
    out = {}
    for p in ('x', 'y'):
        e = z[f'{p}_eid'].astype(int)
        k = np.isin(e, df.index.to_numpy())
        e = e[k]
        pos = df.index.get_indexer(e)
        a, S = bow_aS(cal, p, U[pos], W[pos])
        w = z[f'{p}_w'][k] * 1e3
        tp = (z[f'{p}_tan_ref'][k] + a) * S
        r = vrel.reindex(e).to_numpy() if vrel is not None else np.ones(len(e))
        cover = float(np.isfinite(r).mean()) if len(r) else np.nan
        r = np.where(np.isfinite(r), r, 1.0)
        const = (w - cal.w0[p]) / (cal.kw[p] * cal.v_drift * S) - a
        tab = df[f'{p}_tan_theta'].to_numpy()[pos]
        sane = np.isfinite(tab) & (np.abs(tab) <= 3)
        clos = float(np.mean(np.abs(const[sane] - tab[sane]) < 1e-6)) if sane.any() else np.nan
        if vrel is None:
            w0n, kwn, nkw, kwe = cal.w0[p], cal.kw[p], 0, 0.0
        else:
            # the re-fold as a DIFFERENCE: w0/kw re-measured on the corrected mapping minus the same measurement with
            # r = 1, added to the pass's constants (the estimator's own re-measurement scatter cancels)
            w01, kw1, _, _ = ladder(cal, w, tp)
            wc = cal.w0[p] + (w - cal.w0[p]) / r
            w0r, kwr, nkw, kwe = ladder(cal, wc, tp)
            w0n, kwn = cal.w0[p] + (w0r - w01), cal.kw[p] * kwr / kw1
        tan = (w - w0n) / (kwn * cal.v_drift * r * S) - a
        out[p] = dict(tan=pd.Series(tan, index=e), w0=w0n, kw=kwn, kw_err=kwe, n_kw=nkw, w0_pass=cal.w0[p],
                      kw_pass=cal.kw[p], cover=cover, closure=clos)
    return out


def do_row(r, by, vtrows):
    d = SHORT[r['det']]
    wdir = os.path.join(P1, rel(r), 'wft')
    ak = r.get('align_key') or r['calib_key']
    al = json.load(open(os.path.join(P1, rel(by[ak]), 'wft', 'alignment', 'alignment.json')))
    rays = pd.read_parquet(os.path.join(EFF, 'budget', rel(r), f'rays_{d}.parquet'))
    m, cal = load_calib(wdir)
    df = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    dup = df.event_id.duplicated(keep=False)
    df = df[~dup].set_index('event_id')
    vrel = None
    vt_path = None
    if r['key'] in vtrows:
        vt_path = vtrows[r['key']]['table']
        vrel = pd.read_parquet(vt_path).drop_duplicates('event_id').set_index('event_id')['v_rel']
    V = vt_arm(wdir, df, cal, vrel)
    rays = rays[~rays.event_id.isin(df.index[df.index.duplicated()])].drop_duplicates('event_id', keep=False)
    ig = fiducial_mask('interior_good', rays.mesh_x.to_numpy(), rays.mesh_y.to_numpy(), rays.tan_x.to_numpy(),
                       rays.tan_y.to_numpy(), r['det'])
    R = rays[ig].set_index('event_id')
    P = view_pointing(al)
    out = dict(key=r['key'], det=d, run=r['run'], subrun=r['subrun'], cls=r['cls'], resist_V=float(r['resist_V']),
               drift_V=float(r['drift_V']), E_drift_V_cm=float(r['drift_V']) / 3.0, v_drift=cal.v_drift,
               vt=dict(applied=vrel is not None, table=vt_path,
                       model=(vtrows.get(r['key']) or {}).get('model'),
                       verdict=(vtrows.get(r['key']) or {}).get('survey_verdict')),
               n_rays_ig=int(len(R)), pointing=P, views={})
    ANG = np.load(os.path.join(EFF, 'angles', rel(r), 'per_event.npz'))      # 03_angles --fiducial interior_good
    for p in ('x', 'y'):
        v = V[p]
        # angles: exactly 03_angles' interior_good sample (closure with effbudget's s68_<p>_ig)
        ea = ANG[f'{p}_eid'].astype(int)
        ka = ~pd.Index(ea).duplicated(keep=False)
        ea = ea[ka]
        tra = ANG[f'{p}_tan_ref'][ka]
        tha = np.degrees(np.arctan(tra))
        tpa = ANG[f'{p}_tan_fit'][ka]
        tva = v['tan'].reindex(ea).to_numpy()
        # positions: the rays' interior_good sample with the view's fit ok (02_efficiency accounting)
        ok = df[f'{p}_ok'].reindex(R.index).astype('boolean').fillna(False).astype(bool)
        e_ok = R.index[ok.to_numpy()]
        tref = R.loc[e_ok, f'tan_{p}'].to_numpy()
        th_ref = np.degrees(np.arctan(tref))
        tan_pass = df.loc[e_ok, f'{p}_tan_theta'].to_numpy()
        tan_vt = v['tan'].reindex(e_ok).to_numpy()
        res = {}
        for arm, tfa, tf in (('pass', tpa, tan_pass),
                             ('vt', np.where(np.isfinite(tva), tva, tpa), np.where(np.isfinite(tan_vt), tan_vt, tan_pass))):
            dth = np.degrees(np.arctan(tfa)) - tha
            A = dict(all=stats(dth, ANG_Q))
            A['by_angle'] = []
            for lo, hi in zip(ANG_BINS[:-1], ANG_BINS[1:]):
                k = (np.abs(tha) >= lo) & (np.abs(tha) < hi)
                A['by_angle'].append(dict(lo=lo, hi=hi, **stats(dth[k], ANG_Q)))
            # position: 02_efficiency accounting on top
            spark = (R.loc[e_ok, 'mult'].to_numpy() > SPARK_VETO_HITS)
            dm = R.loc[e_ok, f'res_{p}'].to_numpy()
            dmid = dm + MID_MM * (tf - tref)
            keep = (~spark) & np.isfinite(dm) & (np.abs(dm) < RES_MAX_MM)
            pos = {}
            for nm, dd in (('mesh', dm), ('mid', dmid)):
                s = stats(dd[keep])
                c = s['core']
                s['sigma_dut_core'] = float(np.sqrt(max(c ** 2 - P[p] ** 2, 0.0))) if np.isfinite(c) else np.nan
                s['ref_var_frac'] = float((P[p] / c) ** 2) if np.isfinite(c) and c > 0 else np.nan
                s['by_angle'] = []
                for lo, hi in zip(ANG_BINS[:-1], ANG_BINS[1:]):
                    k = keep & (np.abs(th_ref) >= lo) & (np.abs(th_ref) < hi)
                    sb = stats(dd[k])
                    cb = sb['core']
                    sb['sigma_dut_core'] = float(np.sqrt(max(cb ** 2 - P[p] ** 2, 0.0))) if np.isfinite(cb) else np.nan
                    s['by_angle'].append(dict(lo=lo, hi=hi, **sb))
                pos[nm] = s
            A['position'] = pos
            res[arm] = A
        out['views'][p] = dict(n_ok=int(len(e_ok)), n_ang=int(len(ea)), vt_missing_ang=int((~np.isfinite(tva)).sum()), w0_pass=v['w0_pass'], kw_pass=v['kw_pass'], w0_vt=v['w0'],
                               kw_vt=v['kw'], kw_vt_err=v['kw_err'], vt_cover=v['cover'], const_closure=v['closure'],
                               arms=res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--only', default=None)
    ap.add_argument('--dets', default=None)
    ap.add_argument('--registry', default=os.path.join(REPO, 'rerun', 'pass_registry.csv'))
    ap.add_argument('--vt-rows', default=os.path.join(REPO, 'rerun', 'vt_all', 'results', 'vt_corrected_rows.csv'))
    a = ap.parse_args()
    reg = list(csv.DictReader(open(a.registry)))
    by = {r['key']: r for r in reg}
    vtrows = {r['key']: r for r in csv.DictReader(open(a.vt_rows))}
    only = set(a.only.split(',')) if a.only else None
    dets = set(a.dets.split(',')) if a.dets else None
    os.makedirs(a.out_dir, exist_ok=True)
    for r in reg:
        if r['cls'] == 'excluded' or r['det'] not in SHORT:
            continue
        if only and r['key'] not in only:
            continue
        if dets and r['det'] not in dets:
            continue
        f = os.path.join(a.out_dir, f"{r['key']}.json")
        try:
            o = do_row(r, by, vtrows)
        except FileNotFoundError as e:
            print(f"{r['key']}: missing input {e}", flush=True)
            continue
        json.dump(o, open(f, 'w'), indent=1, default=float)
        v = o['views']
        print(r['key'], {p: (round(v[p]['arms']['pass']['all']['s68'], 3), round(v[p]['arms']['vt']['all']['s68'], 3),
                             round(v[p]['arms']['pass']['position']['mesh']['core'], 3)) for p in 'xy'}, flush=True)


if __name__ == '__main__':
    main()
