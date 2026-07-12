#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_transfer_bench.py — train per-detector HITS-LEVEL angle models on
the June cosmic-bench data ("factory calibration") and measure how well they
TRANSFER across operating conditions with the telescope removed.

This is the dress rehearsal for nTOF: at the beam there is no M3, so day-1
angles come from a frozen bench model (optionally restandardized in situ).
Here we quantify exactly what that costs, using conditions the bench DID
measure with M3 truth (truth used for SCORING ONLY in transfer modes):

  train      sat_det3 @ drift 1000 V  ->  models/mx17_3_hits6.json   (det A)
             o22_long_det2 @ 1000 V   ->  models/mx17_2_hits6.json   (det B)
             g_det6_long  @ 700 V     ->  models/mx17_6_hits6.json   (det C)
             g_det7_long  @ 700 V     ->  models/mx17_7_hits6.json   (det D)
  transfer   det3 drift-scan points 500/700/900/1100 V   (condition change)
             det2 / det6 / det7 long runs                 (detector change)
    modes:   self       = trained on this condition (even eids) — ceiling
             frozen     = det3 sat model, mu/sd as trained      — floor
             frozen_rs  = det3 sat model, mu/sd restandardized on THIS
                          data (unsupervised; the beam-realistic mode)

  Also per condition: v_sig from the frozen_rs regressed angles vs v_geom
  (M3 abscissa) — the telescope-free in-situ drift-velocity monitor.

Run:  cd mx_june_cosmic_qa && ../.venv/bin/python ../ntof_tracking/validate_transfer_bench.py
Output: ntof_tracking/models/*.json, ntof_tracking/validation/transfer_validation.csv/.png
"""
import os
import re
import sys
import glob
import json
import pickle
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'mx_june_cosmic_qa'))

import qa_config
from qa_config import get_config, _Config
qa_config.setup_paths()
import uproot  # noqa: E402
import cosmic_micro_tpc_analysis as cm  # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles  # noqa: E402
from common.Mx17StripMap import RunConfig  # noqa: E402

from ntof_tracking import microtpc_lib as mt  # noqa: E402
from ntof_tracking import bench_constants as bc  # noqa: E402

VETO = 50
CHI2_CUT = 5.0
RES_CUT_MM = 10.0
MIN_STRIPS = 4          # quality selection, as scripts 21/35
FEATS = list(bc.FEATS_HITS6)
SIGN_BAND = (np.tan(np.radians(2.0)), np.tan(np.radians(10.0)))

MODELS_DIR = os.path.join(HERE, 'models')
VAL_DIR = os.path.join(HERE, 'validation')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
def strip_order(det, feu, plane):
    pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                    [0 if plane == 'x' else 1] for ch in range(512)], dtype=float)
    return pos


def build_features(cfg, det, eids_wanted):
    """Per (event, plane) signature features from combined_hits_root via
    mt.hit_features (bit-faithful to 35's builder, plus signed asyms)."""
    plane_of_feu = {cfg.MX17_FEU_X: 'x', cfg.MX17_FEU_Y: 'y'}
    hits_dir = os.path.join(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, 'combined_hits_root')
    fs = sorted(glob.glob(os.path.join(hits_dir, '*.root')))
    fs = [f for f in fs if '_pedestals_' not in os.path.basename(f)]
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
            ft = mt.hit_features(pos[g['channel'].to_numpy(int)],
                                 g['amplitude'].to_numpy(float),
                                 g['time'].to_numpy(float),
                                 g['time_over_threshold'].to_numpy(float))
            if ft is None:
                continue
            ft['eid'] = int(eid)
            ft['plane'] = plane
            rows.append(ft)
    return pd.DataFrame(rows)


def load_condition(key=None, run=None, subrun=None, seed_key=None):
    """Load one bench condition -> per-plane DataFrames with truth + features.
    Truth (M3) is attached for training/scoring; the transfer modes never use
    it before the scoring step."""
    if key is not None:
        cfg = get_config(key)
    else:
        base = get_config(seed_key)
        cfg = _Config(f'{seed_key}_{subrun}', base.RUN, subrun, feus=base.MX17_FEUS,
                      det_z=base.DET_PLANE_Z, det_name=base.DET_NAME,
                      base_path=base.BASE_PATH, zero_suppressed=base.ZERO_SUPPRESSED)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache):
        print(f'  [SKIP] {cfg.KEY}: no cache {cache}')
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)

    own_align = os.path.join(cfg.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
    if os.path.exists(own_align):
        params = cm.load_alignment(own_align)
    else:
        seed = cm.load_alignment(os.path.join(get_config(seed_key).OUT_BASE,
                                              f'alignment_tpc_veto{VETO}', 'alignment.json'))
        params = cm.translation_alignment(results, rays, seed)
    xang = params.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, params, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    if len(sel) < 200:
        print(f'  [SKIP] {cfg.KEY}: only {len(sel)} quality events')
        return None

    th = np.deg2rad(params.theta_deg)
    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])
    t_det = {'x': np.cos(th) * tan_rx + np.sin(th) * tan_ry,
             'y': -np.sin(th) * tan_rx + np.cos(th) * tan_ry}
    eids = np.array([r.event_id for r in sel], dtype=np.int64)
    fit = {'x': [r.x_fit for r in sel], 'y': [r.y_fit for r in sel]}

    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH)
    det = rc.get_detector(cfg.DET_NAME)
    ft = build_features(cfg, det, set(eids.tolist()))
    if ft.empty:
        print(f'  [SKIP] {cfg.KEY}: no features')
        return None

    planes = {}
    for p in ('x', 'y'):
        d = pd.DataFrame({
            'eid': eids, 'tan_ref': t_det[p],
            'ns': [f_.n_strips for f_ in fit[p]],
            'dur': [f_.latest_time_ns - f_.earliest_time_ns for f_ in fit[p]],
            'S_prod': [1000.0 * f_.slope_mm_per_ns for f_ in fit[p]],
        }).set_index('eid')
        f = ft[ft['plane'] == p].drop_duplicates('eid').set_index('eid')
        for c in FEATS + ['a_asym_sgn', 't_asym_sgn']:
            d[c] = f[c].reindex(d.index)
        d = d[np.abs(d['tan_ref']) < bc.MAX_TAN]
        planes[p] = d.reset_index()
    return dict(cfg=cfg, planes=planes, n_quality=len(sel))


# ---------------------------------------------------------------------------
def band_metrics(tan_abs, tan_ref, mask):
    """Regression quality in the faithful band (as 35): bias / s68 in deg of
    atan(tan_reg) - atan(|tan_ref|), plus coverage of the mask sample."""
    tr = np.abs(np.asarray(tan_ref, float))
    m = mask & np.isfinite(tan_abs) & (tr > bc.TAN_LO) & (tr < bc.TAN_HI)
    cov = np.isfinite(tan_abs[mask]).sum() / max(mask.sum(), 1)
    if m.sum() < 50:
        return dict(bias=np.nan, s68=np.nan, cov=cov, n=int(m.sum()))
    dth = np.degrees(np.arctan(tan_abs[m])) - np.degrees(np.arctan(tr[m]))
    q = np.percentile(dth, [16, 50, 84])
    return dict(bias=float(q[1]), s68=float(0.5 * (q[2] - q[0])),
                cov=float(cov), n=int(m.sum()))


def low_band_metrics(tan_signed, tan_ref, mask, lim_deg=5.0):
    """Signed angle performance in |theta_ref| < 5 deg (the head-on band the
    hybrid exists for): s68 and bias of theta_est - theta_ref in deg."""
    tr = np.asarray(tan_ref, float)
    m = mask & np.isfinite(tan_signed) & (np.abs(np.degrees(np.arctan(tr))) < lim_deg)
    if m.sum() < 50:
        return dict(bias=np.nan, s68=np.nan, n=int(m.sum()))
    dth = np.degrees(np.arctan(tan_signed[m])) - np.degrees(np.arctan(tr[m]))
    q = np.percentile(dth, [16, 50, 84])
    return dict(bias=float(q[1]), s68=float(0.5 * (q[2] - q[0])), n=int(m.sum()))


def sign_accuracy(wg, d, mask):
    tr = d['tan_ref'].to_numpy()
    ta = np.abs(tr)
    m = mask & (ta > SIGN_BAND[0]) & (ta < SIGN_BAND[1])
    if m.sum() < 50:
        return np.nan
    s = mt.apply_sign(wg, d['a_asym_sgn'].to_numpy()[m], d['t_asym_sgn'].to_numpy()[m],
                      fallback_sign=d['S_prod'].to_numpy()[m])
    return float((s == np.sign(tr[m])).mean())


# ---------------------------------------------------------------------------
def train_model_for(key, drift_hv, gas_note):
    cond = load_condition(key=key)
    if cond is None:
        return None, None
    det_num = int(cond['cfg'].DET_NAME.split('_')[-1])
    planes_out = {}
    for p in ('x', 'y'):
        d = cond['planes'][p]
        train = (d['eid'].to_numpy() % 2 == 0)
        test = ~train
        F = d[FEATS].to_numpy(float)
        y = np.abs(d['tan_ref'].to_numpy())
        model = mt.train_tan_regression(F[train], y[train], feats=FEATS)
        if model is None:
            print(f'  [{key}/{p}] regression untrainable')
            continue
        tan_abs, _ = mt.apply_tan_regression(model, F)
        hold = band_metrics(tan_abs, d['tan_ref'].to_numpy(), test)
        wg = mt.train_sign_fisher(d['a_asym_sgn'].to_numpy()[train],
                                  d['t_asym_sgn'].to_numpy()[train],
                                  np.sign(d['tan_ref'].to_numpy())[train],
                                  np.abs(d['tan_ref'].to_numpy())[train])
        sacc = sign_accuracy(wg, d, test) if wg is not None else np.nan
        m = test & np.isfinite(tan_abs)
        vres = mt.v_extent(tan_abs[m], d['ns'].to_numpy(float)[m],
                           d['dur'].to_numpy(float)[m], tan_hi=bc.TAN_HI_SIG)
        planes_out[p] = dict(**model, wg=wg,
                             holdout=dict(**hold, sign_acc=sacc),
                             v_sig=(vres or {}).get('v'),
                             t_sat_ns=(vres or {}).get('tsat'))
        print(f'  [{key}/{p}] holdout bias {hold["bias"]:+.2f} deg, s68 {hold["s68"]:.2f} deg, '
              f'cov {hold["cov"]:.0%}, sign acc {sacc if sacc == sacc else float("nan"):.0%}, '
              f'v_sig {(vres or {}).get("v", float("nan")):.2f} um/ns')
    if not planes_out:
        return None, None
    prov = dict(run=cond['cfg'].RUN, subrun=cond['cfg'].SUB_RUN,
                trained=str(datetime.date.today()), drift_hv=drift_hv,
                gas=gas_note, sample_ns=bc.BENCH_SAMPLE_NS,
                n_quality=cond['n_quality'], train_split='even eids')
    path = os.path.join(MODELS_DIR, f'mx17_{det_num}_hits6.json')
    mt.save_model(path, cond['cfg'].DET_NAME, prov, planes_out)
    print(f'  saved {path}')
    return cond, planes_out


def transfer_eval(cond, det3_model, label, drift_hv, rows):
    """Evaluate self / frozen / frozen_rs on one condition (M3 = scoring only)."""
    for p in ('x', 'y'):
        d = cond['planes'][p]
        if p not in det3_model['planes']:
            continue
        mplane = det3_model['planes'][p]
        train = (d['eid'].to_numpy() % 2 == 0)
        test = ~train
        F = d[FEATS].to_numpy(float)
        y_ref = d['tan_ref'].to_numpy()

        variants = {}
        selfm = mt.train_tan_regression(F[train], np.abs(y_ref)[train], feats=FEATS)
        if selfm is not None:
            variants['self'] = mt.apply_tan_regression(selfm, F)[0]
        variants['frozen'] = mt.apply_tan_regression(mplane, F)[0]
        variants['frozen_rs'] = mt.apply_tan_regression(mplane, F, restandardize=True)[0]

        vg = mt.v_extent(np.abs(y_ref), d['ns'].to_numpy(float),
                         d['dur'].to_numpy(float), min_bin=40)
        for mode, tan_abs in variants.items():
            met = band_metrics(tan_abs, y_ref, test)
            row = dict(condition=label, drift_hv=drift_hv, plane=p, mode=mode,
                       **met, v_geom=(vg or {}).get('v'))
            # signed low-angle performance with the frozen sign model
            wg = mplane.get('wg')
            if wg is not None:
                sgn = mt.apply_sign(wg, d['a_asym_sgn'].to_numpy(),
                                    d['t_asym_sgn'].to_numpy(),
                                    fallback_sign=d['S_prod'].to_numpy())
                lo = low_band_metrics(sgn * tan_abs, y_ref, test)
                row.update(lo_bias=lo['bias'], lo_s68=lo['s68'])
                row['sign_acc'] = sign_accuracy(wg, d, test)
            m = test & np.isfinite(tan_abs)
            vs = mt.v_extent(tan_abs[m], d['ns'].to_numpy(float)[m],
                             d['dur'].to_numpy(float)[m], tan_hi=bc.TAN_HI_SIG)
            if vs is not None:
                row['v_sig'] = vs['v']
                row['v_sig_err'] = vs['v_err']
            rows.append(row)


def main():
    print('=== training per-detector hits6 models (factory calibration) ===')
    cond3, _ = train_model_for('sat_det3', 1000, 'Ar/iso 95/5 +~1% H2O +~1% air')
    train_model_for('o22_long_det2', 1000, 'Ar/iso 95/5 +~1% H2O (det2 line)')
    train_model_for('g_det6_long', 700, 'Ar/iso 95/5 (~1.2% H2O, high O2)')
    train_model_for('g_det7_long', 700, 'Ar/iso 95/5 (~0.7% H2O, high O2)')

    det3_model = mt.load_model(os.path.join(MODELS_DIR, 'mx17_3_hits6.json'))

    print('\n=== transfer validation (frozen det3 model; M3 for scoring only) ===')
    rows = []
    if cond3 is not None:
        transfer_eval(cond3, det3_model, 'det3 @1000V (train run)', 1000, rows)

    base = get_config('sat_det3')
    run_dir = os.path.join(base.BASE_PATH, base.RUN)
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_\d+V_drift_(\d+)V$', name)
        if not m:
            continue
        hv = int(m.group(1))
        if hv < 500:
            continue      # <=300 V: no lever arm (bench-established)
        print(f'--- det3 drift scan {hv} V ---')
        cond = load_condition(subrun=name, seed_key='sat_det3')
        if cond is not None:
            transfer_eval(cond, det3_model, f'det3 @{hv}V (scan)', hv, rows)

    for key, hv, lbl in (('o22_long_det2', 1000, 'det2 @1000V'),
                         ('g_det6_long', 700, 'det6 @700V'),
                         ('g_det7_long', 700, 'det7 @700V')):
        print(f'--- {lbl} ---')
        cond = load_condition(key=key)
        if cond is not None:
            transfer_eval(cond, det3_model, lbl, hv, rows)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(VAL_DIR, 'transfer_validation.csv')
    df.to_csv(out_csv, index=False)

    print('\n=== summary: regression s68 [deg] in the 3.4-24 deg band (odd-eid holdout) ===')
    piv = df.pivot_table(index=['condition', 'plane'], columns='mode',
                         values='s68', aggfunc='first')
    print(piv.round(2).to_string())
    print('\n=== bias [deg] ===')
    print(df.pivot_table(index=['condition', 'plane'], columns='mode',
                         values='bias', aggfunc='first').round(2).to_string())
    print('\n=== v: telescope-free v_sig (frozen_rs) vs v_geom (M3) [um/ns] ===')
    vv = df[df['mode'] == 'frozen_rs'].groupby('condition').agg(
        v_geom=('v_geom', 'mean'), v_sig=('v_sig', 'mean'))
    vv['dv_pct'] = 100 * (vv['v_sig'] - vv['v_geom']) / vv['v_geom']
    print(vv.round(2).to_string())
    print(f'\nwrote {out_csv}')

    # ---- figure: s68 by condition x mode ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, p in zip(axes, ('x', 'y')):
        sub = df[df['plane'] == p]
        conds = sub['condition'].unique()
        modes = ['self', 'frozen_rs', 'frozen']
        xpos = np.arange(len(conds))
        for k, mode in enumerate(modes):
            vals = [sub[(sub['condition'] == c) & (sub['mode'] == mode)]['s68'].mean()
                    for c in conds]
            ax.bar(xpos + (k - 1) * 0.25, vals, 0.25, label=mode)
        ax.set_xticks(xpos)
        ax.set_xticklabels(conds, rotation=30, ha='right', fontsize=8)
        ax.set_title(f'{p} plane')
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(2.1, color='k', ls=':', lw=1,
                   label='bench regression-only plateau (2.1 deg)' if p == 'x' else None)
    axes[0].set_ylabel('regression s68 [deg] (3.4-24 deg band, holdout)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen det3 hits-level model transfer across conditions — nTOF day-1 dress rehearsal')
    fig.tight_layout()
    out_png = os.path.join(VAL_DIR, 'transfer_validation.png')
    fig.savefig(out_png, dpi=150)
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main()
