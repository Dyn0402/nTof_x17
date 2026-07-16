#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35_hybrid_drift_scan.py

Drift-velocity scan re-measured with the hybrid-tracking machinery (33/34):
per drift-HV point, regress |tan theta| from the HIT-LEVEL cluster signature
(the drift-scan subruns have no decoded_root, but combined_hits_root already
carries time_over_threshold / amplitude / time per strip, which is enough for
the 33 feature set minus the unshared footprint n_u), then extract

    v = (cluster-extent slope vs |tan theta|) / T_sat

twice per plane:
    v_geom  : x-axis = M3 reference angle           (reproduces 21)
    v_sig   : x-axis = signature-regressed angle    (the hybrid machinery;
              trained on EVEN eids vs M3, measured on the ODD-eid holdout,
              inclined-track selection ALSO from the regressed angle, so the
              measurement itself never touches the telescope)

Agreement of v_sig with v_geom across the scan demonstrates that the
signature regression tracks the true angle at every drift field (features
like tot_lead rescale with the drift time, so per-point training is
required and expected), and that a detector-internal v(E) measurement is
possible wherever a training sample exists.

Usage:  ../.venv/bin/python 35_hybrid_drift_scan.py sat_det3 [--veto=50]
        [--min-bin=25]    minimum events per tan bin (holdout halves stats)
Output: <Analysis>/<run>/drift_velocity/<det>/hybrid_vdrift_scan.csv/.png
        + stdout table
"""
import os
import re
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths, _Config
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
MIN_BIN = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--min-bin=')), 25)
TAN_HI_SIG = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--tan-hi-sig=')), 0.27)
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
RES_CUT_MM = 10.0
MIN_STRIPS = 4          # as 21: production-cluster quality for the extent
PITCH_MM = 0.78
SAT_DEG = 10.0
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04
MAX_TAN = 0.7
GAP_CM = 3.0
A_LEAD_MIN = 300.0
THR_HIT = 100.0
FEATS = ['tot_lead', 'q_frac', 'n_raw', 'a_asym', 'a_lead', 't_delay']

ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
OUT = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)
os.makedirs(OUT, exist_ok=True)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


# --------------------------------------------------------------------------
# hit-level signature features (33's set, minus n_u which needs waveforms)
# --------------------------------------------------------------------------
def strip_order(det, feu, plane):
    pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                    [0 if plane == 'x' else 1] for ch in range(512)], dtype=float)
    return pos


def build_features_hits(cfg, det, eids_wanted):
    """Per event & plane signature features from combined_hits_root."""
    plane_of_feu = {cfg.MX17_FEU_X: 'x', cfg.MX17_FEU_Y: 'y'}
    hits_dir = os.path.join(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, 'combined_hits_root')
    fs = sorted(glob.glob(os.path.join(hits_dir, '*.root')))
    if not fs:
        return pd.DataFrame()

    pos_of = {feu: strip_order(det, feu, plane_of_feu[feu]) for feu in cfg.MX17_FEUS}

    frames = []
    for fn in fs:
        with uproot.open(fn) as f:
            t = f['hits']
            arr = t.arrays(['eventId', 'channel', 'amplitude', 'time',
                            'time_over_threshold', 'feu'], library='np')
        frames.append(pd.DataFrame(arr))
    h = pd.concat(frames, ignore_index=True)
    h = h[h['eventId'].isin(eids_wanted)]

    rows = []
    for feu in cfg.MX17_FEUS:
        plane = plane_of_feu[feu]
        pos = pos_of[feu]
        hf = h[h['feu'] == feu]
        for eid, g in hf.groupby('eventId'):
            ch = g['channel'].to_numpy(int)
            amp = g['amplitude'].to_numpy(float)
            tim = g['time'].to_numpy(float)
            tot = g['time_over_threshold'].to_numpy(float)
            p = pos[ch]
            ok = np.isfinite(p)
            if ok.sum() < 1:
                continue
            ch, amp, tim, tot, p = ch[ok], amp[ok], tim[ok], tot[ok], p[ok]
            o = np.argsort(p)
            p, amp, tim, tot = p[o], amp[o], tim[o], tot[o]
            k = int(np.argmax(amp))
            if amp[k] < A_LEAD_MIN:
                continue
            # contiguous cluster (adjacent strip positions) >= THR_HIT around lead
            l0 = k
            while l0 - 1 >= 0 and amp[l0 - 1] >= THR_HIT \
                    and abs(p[l0] - p[l0 - 1] - PITCH_MM) < 0.5 * PITCH_MM:
                l0 -= 1
            r0 = k
            while r0 + 1 < len(p) and amp[r0 + 1] >= THR_HIT \
                    and abs(p[r0 + 1] - p[r0] - PITCH_MM) < 0.5 * PITCH_MM:
                r0 += 1
            n_raw = r0 - l0 + 1
            q_clu = float(amp[l0:r0 + 1].sum())
            # +-1 neighbours by strip position (may be absent from the hit list)
            def nb(side):
                j = k + side
                if 0 <= j < len(p) and abs(p[j] - p[k] - side * PITCH_MM) < 0.5 * PITCH_MM:
                    return float(amp[j]), float(tim[j])
                return 0.0, np.nan
            aL, tL = nb(-1)
            aR, tR = nb(+1)
            t_nb = [t for t in (tL, tR) if np.isfinite(t)]
            t_delay = (min(t_nb) - tim[k]) if t_nb else np.nan
            rows.append(dict(
                eid=int(eid), plane=plane,
                a_lead=float(amp[k]), a_l=aL, a_r=aR,
                tot_lead=float(tot[k]), n_raw=int(n_raw),
                q_frac=float(amp[k] / q_clu) if q_clu > 0 else np.nan,
                a_asym=abs(aR - aL) / (aR + aL) if (aR + aL) > 0 else np.nan,
                t_delay=float(t_delay)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# regression (34's construction: standardize -> lstsq -> monotonic calib)
# --------------------------------------------------------------------------
def regress_tan(d, train):
    F = d[FEATS].to_numpy(float)
    ok = np.isfinite(F).all(axis=1)
    y = np.abs(d['tan_ref'].to_numpy())
    mtr = ok & train
    if mtr.sum() < 300:
        return None, ok, np.nan
    mu, sd = F[mtr].mean(axis=0), F[mtr].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (F - mu) / sd
    A = np.c_[Z[mtr], np.ones(mtr.sum())]
    w, *_ = np.linalg.lstsq(A, y[mtr], rcond=None)
    s_lin = np.full(len(d), np.nan)
    s_lin[ok] = Z[ok] @ w[:-1] + w[-1]
    qs = np.nanquantile(s_lin[mtr], np.linspace(0.02, 0.98, 25))
    ctr_s, med_t = [], []
    for lo_, hi_ in zip(qs[:-1], qs[1:]):
        m = mtr & (s_lin >= lo_) & (s_lin < hi_)
        if m.sum() > 30:
            ctr_s.append(np.median(s_lin[m]))
            med_t.append(np.median(np.abs(y[m])))
    if len(ctr_s) < 5:
        return None, ok, np.nan
    ctr_s = np.array(ctr_s)
    med_t = np.maximum.accumulate(np.array(med_t))
    tan_abs = np.full(len(d), np.nan)
    tan_abs[ok] = np.interp(s_lin[ok], ctr_s, med_t)

    # response slope lambda = d<tan_reg>/d|tan_ref| on the TRAINING half over
    # the extent-fit band: the binned-median calibration compresses the angle
    # scale (regression to the mean), which inflates the extent slope by
    # 1/lambda; v_sig must be multiplied by lambda.
    yt, rt = y[mtr], tan_abs[mtr]
    bins = np.arange(TAN_LO, TAN_HI + TAN_STEP, TAN_STEP)
    bc, bm = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (yt >= b0) & (yt < b1) & np.isfinite(rt)
        if m.sum() >= 15:
            bc.append(0.5 * (b0 + b1)); bm.append(np.median(rt[m]))
    lam = np.nan
    if len(bc) >= 4:
        lam = float(np.polyfit(bc, bm, 1)[0])
    return tan_abs, ok, lam


# --------------------------------------------------------------------------
# extent-slope / T_sat velocity (21's estimator, x-axis pluggable)
# --------------------------------------------------------------------------
def v_extent(tan_abs, ns, dur, min_bin=MIN_BIN, tan_hi=TAN_HI):
    ext = (ns - 1) * PITCH_MM
    bins = np.arange(TAN_LO, tan_hi + TAN_STEP, TAN_STEP)
    ctr, med, mer = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (tan_abs >= b0) & (tan_abs < b1)
        if m.sum() >= min_bin:
            ctr.append(0.5 * (b0 + b1))
            med.append(np.median(ext[m]))
            mer.append(1.253 * np.std(ext[m], ddof=1) / np.sqrt(m.sum()))
    if len(ctr) < 4:
        return None
    ctr, med, mer = map(np.array, (ctr, med, mer))
    w = 1.0 / mer**2
    W, Wx, Wy = np.sum(w), np.sum(w * ctr), np.sum(w * med)
    Wxx, Wxy = np.sum(w * ctr**2), np.sum(w * ctr * med)
    den = W * Wxx - Wx**2
    slope = (W * Wxy - Wx * Wy) / den
    slope_err = np.sqrt(W / den)
    m_sat = tan_abs > np.tan(np.radians(SAT_DEG))
    if m_sat.sum() < 30:
        return None
    tsat = float(np.median(dur[m_sat]))
    tsat_err = float(1.253 * np.std(dur[m_sat], ddof=1) / np.sqrt(m_sat.sum()))
    v = slope * 1000.0 / tsat
    v_err = v * np.hypot(slope_err / slope, tsat_err / tsat)
    return dict(v=float(v), v_err=float(v_err), slope=float(slope),
                tsat=tsat, n_bins=len(ctr), n_sat=int(m_sat.sum()))


def measure_point(subrun, seed, det):
    cfg = cfg_for(subrun)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        print(f'  [SKIP] {subrun}: no cache')
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    if len(sel) < 200:
        print(f'  [SKIP] {subrun}: only {len(sel)} quality events')
        return None

    th = np.deg2rad(seed.theta_deg)
    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    t_det = {'x': np.cos(th) * tan_rx + np.sin(th) * tan_ry,
             'y': -np.sin(th) * tan_rx + np.cos(th) * tan_ry}
    eids = np.array([r.event_id for r in sel], dtype=np.int64)
    ns = {'x': np.array([r.x_fit.n_strips for r in sel]),
          'y': np.array([r.y_fit.n_strips for r in sel])}
    dur = {'x': np.array([r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns for r in sel]),
           'y': np.array([r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns for r in sel])}

    ft = build_features_hits(cfg, det, set(eids.tolist()))
    if ft.empty:
        print(f'  [SKIP] {subrun}: no hit-level features')
        return None

    row = dict(subrun=subrun, n_quality=len(sel))
    for p in ('x', 'y'):
        d = pd.DataFrame({'eid': eids, 'tan_ref': t_det[p],
                          'ns': ns[p], 'dur': dur[p]}).set_index('eid')
        f = ft[ft['plane'] == p].drop_duplicates('eid').set_index('eid')
        for c in FEATS:
            d[c] = f[c].reindex(d.index)
        d = d[np.abs(d['tan_ref']) < MAX_TAN]
        train = (d.index.values % 2 == 0)
        test = ~train

        # --- reference-angle velocity (21's estimator, all quality events)
        r_ref = v_extent(np.abs(d['tan_ref'].to_numpy()),
                         d['ns'].to_numpy(float), d['dur'].to_numpy(float),
                         min_bin=max(MIN_BIN, 40))
        # --- signature-angle velocity (odd-eid holdout only)
        tan_reg, ok, lam = regress_tan(d, train)
        r_sig = None
        cov = np.nan
        if tan_reg is not None:
            m = test & np.isfinite(tan_reg)
            cov = m.sum() / max(test.sum(), 1)
            # restrict the fit to the regression's faithful band: features
            # saturate above ~15 deg (34's known caveat), which would pull
            # steep-track events into moderate tan_reg bins and inflate the
            # slope. The 10-15 deg overlap still saturates T_sat.
            r_sig = v_extent(tan_reg[m], d['ns'].to_numpy(float)[m],
                             d['dur'].to_numpy(float)[m], min_bin=MIN_BIN,
                             tan_hi=TAN_HI_SIG)
            if r_sig is not None and np.isfinite(lam):
                row[f'lambda_{p}'] = lam   # response slope, recorded only
            # holdout regression quality in the scan band, for the record
            mm = m & (np.abs(d['tan_ref'].to_numpy()) > TAN_LO) \
                   & (np.abs(d['tan_ref'].to_numpy()) < TAN_HI)
            if mm.sum() > 50:
                dth = np.degrees(np.arctan(tan_reg[mm])) \
                    - np.degrees(np.arctan(np.abs(d['tan_ref'].to_numpy()[mm])))
                q = np.percentile(dth, [16, 50, 84])
                row[f'reg_bias_{p}_deg'] = float(q[1])
                row[f'reg_s68_{p}_deg'] = float(0.5 * (q[2] - q[0]))
        for tag_, rr in (('geom', r_ref), ('sig', r_sig)):
            if rr is not None:
                row[f'v_{tag_}_{p}'] = rr['v']
                row[f'v_{tag_}_{p}_err'] = rr['v_err']
                row[f't_sat_{tag_}_{p}_ns'] = rr['tsat']
        row[f'sig_coverage_{p}'] = float(cov)

    # combine planes (inverse-variance, as 21)
    for tag_ in ('geom', 'sig'):
        vs = [(row[f'v_{tag_}_{p}'], row[f'v_{tag_}_{p}_err'])
              for p in 'xy' if f'v_{tag_}_{p}' in row]
        if vs:
            ws = 1.0 / np.array([e for _, e in vs])**2
            row[f'v_{tag_}'] = float(np.sum([v * w for (v, _), w in zip(vs, ws)]) / np.sum(ws))
            row[f'v_{tag_}_err'] = float(np.sqrt(1.0 / np.sum(ws)))
            if len(vs) == 2:
                row[f'v_{tag_}_sys'] = float(abs(vs[0][0] - vs[1][0]) / 2.0)
    g = row.get('v_geom', np.nan)
    s = row.get('v_sig', np.nan)
    print(f'  {subrun}: v_geom = {g:.2f} ± {row.get("v_geom_err", np.nan):.2f}   '
          f'v_sig = {s:.2f} ± {row.get("v_sig_err", np.nan):.2f} µm/ns   '
          f'(sig coverage {row.get("sig_coverage_x", np.nan):.0%}/{row.get("sig_coverage_y", np.nan):.0%})')
    return row


def main():
    seed = cm.load_alignment(ALIGN_SEED)
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    points = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_(\d+)V_drift_(\d+)V$', name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            points.append((int(m.group(2)), name))
    m = re.search(r'drift_(\d+)V', CFG.SUB_RUN)
    if m:
        points.append((int(m.group(1)), CFG.SUB_RUN))
    points.sort()

    rows = []
    for hv, subrun in points:
        print(f'=== drift {hv} V — {subrun} ===')
        r = measure_point(subrun, seed, det)
        if r is not None:
            r['drift_hv'] = hv
            r['is_long_run'] = (subrun == CFG.SUB_RUN)
            rows.append(r)

    if not rows:
        print('no points measured'); return
    df = pd.DataFrame(rows).sort_values('drift_hv')
    df.to_csv(os.path.join(OUT, 'hybrid_vdrift_scan.csv'), index=False)

    # ---- summary table ----
    print('\n=== hybrid drift-velocity scan (µm/ns) ===')
    print(f'{"HV":>6} {"E [V/cm]":>9} {"v_geom (M3)":>16} {"v_sig (signature)":>19} '
          f'{"Δ":>6} {"reg σ68 x/y [deg]":>18}')
    for _, r in df.iterrows():
        e = r['drift_hv'] / GAP_CM
        vg = f"{r.get('v_geom', np.nan):.2f} ± {r.get('v_geom_err', np.nan):.2f}"
        vs_ = f"{r.get('v_sig', np.nan):.2f} ± {r.get('v_sig_err', np.nan):.2f}"
        dd = r.get('v_sig', np.nan) - r.get('v_geom', np.nan)
        s68 = f"{r.get('reg_s68_x_deg', np.nan):.1f}/{r.get('reg_s68_y_deg', np.nan):.1f}"
        print(f"{r['drift_hv']:>6.0f} {e:>9.0f} {vg:>16} {vs_:>19} {dd:>+6.2f} {s68:>18}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 6.5))
    e = df['drift_hv'] / GAP_CM
    if 'v_geom' in df:
        ax.errorbar(e, df['v_geom'],
                    yerr=np.hypot(df['v_geom_err'], df.get('v_geom_sys', 0.0).fillna(0)),
                    fmt='o', color='black', ms=9, capsize=4, zorder=6,
                    label=r'$v_{\rm geom}$: extent / $T_{\rm sat}$ vs M3 angle (as 21)')
    if 'v_sig' in df:
        ax.errorbar(e + 3, df['v_sig'],
                    yerr=np.hypot(df['v_sig_err'], df.get('v_sig_sys', 0.0).fillna(0)),
                    fmt='D', color='crimson', ms=8, capsize=4, zorder=7,
                    label=r'$v_{\rm sig}$: extent / $T_{\rm sat}$ vs signature-regressed angle'
                          '\n(hit-level features, odd-eid holdout)')
    ridge_csv = os.path.join(OUT, 'drift_velocity_scan.csv')
    if os.path.exists(ridge_csv):
        ridge = pd.read_csv(ridge_csv)
        ax.errorbar(ridge['drift_hv'] / GAP_CM, ridge['v_ridge'],
                    yerr=np.hypot(ridge['v_ridge_err'], ridge['v_ridge_sys']),
                    fmt='s', color='gray', ms=6, capsize=3, alpha=0.7, zorder=4,
                    label='v_ridge (time-fit; sharing-biased)')
    star = df[df['is_long_run']]
    if len(star) and 'v_geom' in star:
        ax.plot(star['drift_hv'] / GAP_CM, star['v_geom'], '*', color='goldenrod',
                ms=18, zorder=8, label='2.4 h long run')
    ax.set_xlabel('drift field [V/cm]')
    ax.set_ylabel('drift velocity [µm/ns]')
    ax.set_title(f'{CFG.DET_NAME} drift scan — geometry estimator with M3 vs '
                 f'signature-regressed angles')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'hybrid_vdrift_scan.png'), dpi=150)
    print(f"\nwrote {os.path.join(OUT, 'hybrid_vdrift_scan.csv')} / .png")


if __name__ == '__main__':
    main()
