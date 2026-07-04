#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_waveform_investigation.py

Full waveform-level investigation of the micro-TPC timing story, using the
decoded (non-zero-suppressed) DREAM data pulled from lxplus:
512 channels x 32 samples/event/FEU, sample-major vectors.

Questions (from 20-23_*.py):
 - The geometry estimators give v(1000V) = 34-36 um/ns; ALL time-fit
   estimators on the hits-level `time` give 28-30. Are the hit times
   corrupted by resistive-strip RC spreading as the pulse-bar displays
   suggest (mesh-end skirts late, deep-end strips early)?
 - Can a waveform-level time estimator (threshold / CFD / rise-fit /
   derivative-max) recover the geometric velocity?
 - What do skirt waveforms actually look like (delayed copies? slow rise?),
   and is there time distortion inside the core?

Outputs (bias_study dir):
  wf_displays.png      true waveform heatmaps for ~0/15/30 deg tracks
  wf_shapes.png        average normalized pulse shapes: core vs skirt,
                       early-end vs late-end, aligned to the core-line time
  wf_timing_ridge.png  ridge v per waveform time estimator vs v_geom
  wf_strip_times.csv   per-strip re-extracted times (cached for reuse)

Usage: ../.venv/bin/python 24_waveform_investigation.py sat_det3 [--veto=50] [--refit]
"""
import os
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
REFIT = '--refit' in sys.argv
SAMPLE_NS = 60.0
MIN_STRIPS = 4
RES_CUT_MM = 10.0
CHI2_CUT = 20.0
GAP_MM_CLUSTER = getattr(cm, 'GAP_THRESHOLD_MM', 2.0)
CORE_FRAC = 0.30
THR_ADC = 150.0
N_PED_EVENTS = 300
CHUNK = 400
V_GEOM = 33.9

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
TIMES_CSV = os.path.join(OUT, 'wf_strip_times.csv')


def robust_line(x, y, n_iter=4, clip=3.0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 10:
            return np.nan, np.nan, np.nan, 0
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    n = int(keep.sum())
    xk = x[keep]
    resid = y[keep] - np.polyval(p, xk)
    se = np.sqrt(np.sum(resid**2) / max(n - 2, 1) / np.sum((xk - xk.mean())**2))
    return float(p[0]), float(p[1]), float(se), n


def largest_cluster(pos):
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > GAP_MM_CLUSTER)[0]
    return max(np.split(o, breaks + 1), key=len)


# ---------------- waveform time estimators (w: 32 ped/cns-subtracted) ----------------
TS = np.arange(32) * SAMPLE_NS


def _interp_cross(w, level, i_end):
    """Time of first upward crossing of `level` at/before sample i_end."""
    for i in range(1, i_end + 1):
        if w[i] >= level > w[i - 1]:
            return SAMPLE_NS * (i - 1 + (level - w[i - 1]) / (w[i] - w[i - 1]))
    return np.nan


def wf_times(w):
    """Return dict of per-strip times [ns] from one waveform."""
    ipk = int(np.argmax(w))
    a = w[ipk]
    out = dict(amp_wf=float(a), t_peak=float(TS[ipk]))
    if a < THR_ADC or ipk == 0:
        out.update(t_thr=np.nan, t_cfd=np.nan, t_rise=np.nan, t_dmax=np.nan)
        return out
    out['t_thr'] = _interp_cross(w, THR_ADC, ipk)
    out['t_cfd'] = _interp_cross(w, 0.5 * a, ipk)
    # rise fit: samples on rising edge between 20% and 85% of peak
    m = np.zeros(32, bool)
    m[:ipk + 1] = (w[:ipk + 1] >= 0.2 * a) & (w[:ipk + 1] <= 0.85 * a)
    if m.sum() >= 2:
        sl, b = np.polyfit(TS[m], w[m], 1)
        out['t_rise'] = float(-b / sl) if sl > 0 else np.nan
    else:
        out['t_rise'] = np.nan
    d = np.diff(w[:ipk + 1])
    out['t_dmax'] = float(SAMPLE_NS * (np.argmax(d) + 0.5)) if len(d) else np.nan
    return out


def build_strip_times(ref, det):
    """Loop decoded files, extract per-strip waveform times for matched events."""
    hits_fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                     if f.endswith('.root') and '_datrun_' in f)
    hf = uproot.concatenate(
        [f'{CFG.combined_hits_dir}{f}:hits' for f in hits_fs],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'time'], library='pd')
    hf = hf[hf['feu'].isin(CFG.MX17_FEUS)]
    hpe = hf.groupby('eventId')['eventId'].transform('size')
    hf = hf[(hpe <= VETO) & hf['eventId'].isin(ref)].copy()
    hf = cm._map_strip_positions(hf, det)
    print(f'{hf["eventId"].nunique():,} matched+veto events in hits')

    # per-event/plane core-cluster channel lists from the hits tree
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    clusters = {}   # (eventId, plane) -> (channels, positions, hit_amps, hit_times)
    for (eid, feu), g in hf.groupby(['eventId', 'feu']):
        p = plane_of_feu[feu]
        pcol = f'{p}_position_mm'
        gp = g[g[pcol].notna()]
        if len(gp) < MIN_STRIPS:
            continue
        pos_all = gp[pcol].to_numpy()
        idx = largest_cluster(pos_all)
        if len(idx) < MIN_STRIPS:
            continue
        gp = gp.iloc[idx]
        clusters[(eid, p)] = (gp['channel'].to_numpy().astype(int),
                              gp[pcol].to_numpy(),
                              gp['amplitude'].to_numpy(),
                              gp['time'].to_numpy())

    rows = []
    for feu in CFG.MX17_FEUS:
        p = plane_of_feu[feu]
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
        print(f'FEU {feu} ({p} plane): {len(fs)} decoded files')
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            # pedestal from the first N events of this file
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            stack = np.stack([a.reshape(32, 512) for a in a0]).astype(np.float32)
            ped = np.median(stack, axis=(0, 1))          # (512,)
            for lo in range(0, t.num_entries, CHUNK):
                hi = min(lo + CHUNK, t.num_entries)
                want = [i for i in range(lo, hi) if (int(eids_all[i]), p) in clusters]
                if not want:
                    continue
                arr = t.arrays(['eventId', 'amplitude'], entry_start=lo,
                               entry_stop=hi, library='np')
                for i in want:
                    j = i - lo
                    eid = int(arr['eventId'][j])
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    # common-noise per chip (64 ch) per sample
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    chans, poss, hamps, htimes = clusters[(eid, p)]
                    amax = hamps.max()
                    for ch, po, ha, ht in zip(chans, poss, hamps, htimes):
                        d = wf_times(wfm[:, ch])
                        rows.append(dict(eventId=eid, plane=p, channel=int(ch),
                                         pos_mm=float(po), hit_amp=float(ha),
                                         hit_time=float(ht),
                                         relamp=float(ha / amax), **d))
            print(f'  {os.path.basename(fn)}: cumulative {len(rows):,} strips')
    df = pd.DataFrame(rows)
    df.to_csv(TIMES_CSV, index=False)
    return df


def main():
    cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = {'x': tx, 'y': ty}

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    if os.path.exists(TIMES_CSV) and not REFIT:
        df = pd.read_csv(TIMES_CSV)
        print(f'loaded cached {len(df):,} strip times')
    else:
        df = build_strip_times(ref, det)

    df['tan'] = [ref.get(e, {}).get(p, np.nan) for e, p in zip(df['eventId'], df['plane'])]

    # ---------- per-event core line fits per timing method ----------
    METHODS = ['hit_time', 't_thr', 't_cfd', 't_rise', 't_dmax', 't_peak']
    print('\n== ridge v per timing method (core strips, OLS, detector frame) ==')
    fitrows = {m: {'x': ([], []), 'y': ([], [])} for m in METHODS}
    for (eid, p), g in df.groupby(['eventId', 'plane']):
        tanv = g['tan'].iloc[0]
        if not np.isfinite(tanv):
            continue
        gc = g[g['relamp'] >= CORE_FRAC]
        if len(gc) < 3 or np.ptp(gc['pos_mm'].to_numpy()) == 0:
            continue
        pos = gc['pos_mm'].to_numpy()
        for m in METHODS:
            tv = gc[m].to_numpy()
            ok = np.isfinite(tv)
            if ok.sum() < 3 or np.ptp(pos[ok]) == 0:
                continue
            sl = np.polyfit(pos[ok], tv[ok], 1)[0]
            if sl == 0:
                continue
            S, T = fitrows[m][p]
            S.append(1000.0 / sl)   # µm/ns
            T.append(tanv)

    labels, vv, ve = [], [], []
    for m in METHODS:
        vs = []
        for p in ('x', 'y'):
            S, T = fitrows[m][p]
            S, T = np.array(S), np.array(T)
            for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
                msk = (T > lo) & (T < hi) & np.isfinite(S)
                v, b, se, n = robust_line(T[msk], S[msk])
                if np.isfinite(v) and n > 100:
                    vs.append(v)
        vm, vsd = np.mean(vs), np.std(vs, ddof=1) / np.sqrt(len(vs))
        labels.append(m); vv.append(vm); ve.append(vsd)
        print(f'  {m:10s}: v = {vm:6.2f} ± {vsd:4.2f} µm/ns  ({len(vs)} fits)')

    fig, ax = plt.subplots(figsize=(8, 5))
    yp = np.arange(len(labels))
    ax.barh(yp, vv, xerr=ve, color='steelblue', alpha=0.85)
    ax.axvline(V_GEOM, color='crimson', lw=2, label=f'v_geom = {V_GEOM}')
    ax.axvline(28.1, color='gray', ls='--', lw=1.5, label='hits-level ridge 28.1')
    ax.set_yticks(yp); ax.set_yticklabels(labels)
    ax.set_xlabel('ridge v [µm/ns]'); ax.set_xlim(20, 42)
    ax.legend(); ax.grid(alpha=0.3, axis='x')
    ax.set_title('waveform-level time estimators (core strips)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'wf_timing_ridge.png'), dpi=160)
    plt.close(fig)

    # ---------- average pulse shapes: core vs skirt, aligned to core line ----------
    # per event/plane: core line from t_cfd of core strips; accumulate normalized
    # waveform-time histograms via (t_cfd - t_line(pos)) per relamp class
    classes = [('core high (>=0.7)', 0.7, 1.01, 'tab:green'),
               ('core low (0.3-0.7)', 0.3, 0.7, 'tab:olive'),
               ('skirt (0.15-0.3)', 0.15, 0.3, 'tab:orange'),
               ('deep skirt (<0.15)', 0.0, 0.15, 'tab:red')]
    resid = {c[0]: [] for c in classes}
    rise = {c[0]: [] for c in classes}
    for (eid, p), g in df.groupby(['eventId', 'plane']):
        tanv = g['tan'].iloc[0]
        if not (np.isfinite(tanv) and 0.15 < abs(tanv) < 0.55):
            continue
        gc = g[(g['relamp'] >= CORE_FRAC) & np.isfinite(g['t_cfd'])]
        if len(gc) < 4:
            continue
        pc = np.polyfit(gc['pos_mm'], gc['t_cfd'], 1)
        pred = np.polyval(pc, g['pos_mm'])
        dres = g['t_cfd'].to_numpy() - pred
        drise = (g['t_peak'] - g['t_cfd']).to_numpy()
        for name, lo, hi, _ in classes:
            m = (g['relamp'].to_numpy() >= lo) & (g['relamp'].to_numpy() < hi) \
                & np.isfinite(dres)
            resid[name].extend(dres[m])
            rise[name].extend(drise[m][np.isfinite(drise[m])])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bins = np.linspace(-400, 700, 75)
    for name, lo, hi, c in classes:
        r = np.array(resid[name])
        if len(r) < 50:
            continue
        ax.hist(r, bins=bins, histtype='step', lw=2, color=c, density=True,
                label=f'{name}: med={np.median(r):+.0f} ns (n={len(r):,})')
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('t_CFD − core-line prediction [ns]')
    ax.set_ylabel('normalized')
    ax.set_title('strip CFD-time residual by relative amplitude')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for name, lo, hi, c in classes:
        r = np.array(rise[name])
        if len(r) < 50:
            continue
        ax.hist(r, bins=np.linspace(0, 800, 60), histtype='step', lw=2, color=c,
                density=True, label=f'{name}: med={np.median(r):.0f} ns')
    ax.set_xlabel('t_peak − t_CFD  (rise-scale) [ns]')
    ax.set_ylabel('normalized')
    ax.set_title('pulse rise scale by relative amplitude')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'{CFG.RUN} — waveform shape systematics (inclined tracks)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'wf_shapes.png'), dpi=160)
    plt.close(fig)

    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
