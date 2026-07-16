#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_position_estimators.py

HIT-MODE POSITION accuracy: can we beat the production "earliest-hit strip"
anchor? The production position is ONE strip position (quantised at the
0.78 mm pitch, selected by noisy timing); several estimators should do
better:

  prod       production earliest-hit strip (cache baseline)
  lead_u     earliest CFD strip on the UNSHARED waveform (same idea,
             cleaner timing)
  cog_raw    amplitude centroid of the raw cluster (all charge; displaced
             along the track by ~half the recorded column for inclined
             tracks -> its effective plane is DEEPER than the mesh)
  cog_u      centroid of the unshared cluster (direct footprint)
  early_raw  centroid of the charge arriving in the FIRST K samples after
             cluster start: that charge drifted from near the MESH, and the
             resistive sharing spreads it over neighbours = a built-in
             sub-pitch interpolator anchored at the impact point
  early_u    same on the unshared waveform (no sharing interpolation)
  fit_t0     TRACK-POINTING: amplitude-weighted straight line pos(t) on the
             unshared core cluster, evaluated at the cluster start time =
             the track's impact point on the mesh. (Fitted as pos vs t so
             it stays finite for vertical tracks, where m -> 0 and the
             intercept degrades gracefully to a weighted mean.)

Every estimator is evaluated identically: per-plane raw positions -> joint
rotation to the aligned frame (both planes required) -> residual vs the M3
v2 ray at the alignment plane, per-estimator median offset removed. Report
sigma68 + Gaussian core, overall and per angle band, and a z-SCAN of the
reference plane: sigma68 vs assumed ray depth, whose minimum locates each
estimator's effective measurement plane (mesh-anchored estimators should
minimise at the alignment z; the full centroid ~half a column deeper).

Usage: ../.venv/bin/python 36_position_estimators.py sat_det3 [--veto=50]
       [--rebuild]
Output: <alignment_tpc_vetoN>/position/ position_estimates.csv (cache),
        position_benchmark.png, position_summary.csv + stdout
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
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
REBUILD = '--rebuild' in sys.argv
SAMPLE_NS = 60.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
RES_CUT_MM = 10.0
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
EARLY_K = 2            # samples in the early-charge window
N_PED_EVENTS = 300
CHUNK = 400
ALPHA = 0.5
MAX_RES_MM = 8.0       # residual window for sigma evaluation

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'position')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
EST_CSV = os.path.join(OUT, 'position_estimates.csv')
CSHARE = {7: (0.449, 0.052), 8: (0.516, 0.151), 6: (0.449, 0.052),
          3: (0.449, 0.052), 4: (0.516, 0.151)}
N_SWITCH = 9           # combo: early-charge centroid for n_strips <= N_SWITCH
                       # (footprint ~ 3mm + 23mm*tan -> 9 strips ~ 10 deg)
ESTIMATORS = ['prod', 'lead_u', 'cog_raw', 'cog_u', 'early_raw', 'early_u',
              'fit_t0', 'combo']
EST_LABEL = {
    'prod': 'production earliest strip',
    'lead_u': 'earliest strip (unshared)',
    'cog_raw': 'cluster centroid (raw)',
    'cog_u': 'cluster centroid (unshared)',
    'early_raw': f'early-charge centroid (raw, {EARLY_K} samp)',
    'early_u': f'early-charge centroid (unshared, {EARLY_K} samp)',
    'fit_t0': 'track-fit impact point (pos(t) @ t0)',
    'combo': f'COMBO (early-charge if n<={N_SWITCH}, else prod)',
}


def cfd_time(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a < THR_WF or ipk == 0:
        return np.nan
    lvl = 0.5 * a
    for i in range(1, ipk + 1):
        if w[i] >= lvl > w[i - 1]:
            return SAMPLE_NS * (i - 1 + (lvl - w[i - 1]) / (w[i] - w[i - 1]))
    return np.nan


def nsum(x, k):
    out = np.zeros_like(x)
    out[k:] += x[:-k]
    out[:-k] += x[k:]
    return out


def unshare(wb, c1, c2, alpha=ALPHA):
    n, ns = wb.shape
    if n < 3:
        return wb
    ab = np.zeros((5, n))
    ab[0, 2:] = alpha * c2
    ab[1, 1:] = alpha * c1
    ab[2, :] = 1.0
    ab[3, :-1] = alpha * c1
    ab[4, :-2] = alpha * c2
    X = np.zeros_like(wb)
    for s in range(ns):
        rhs = wb[:, s].copy()
        if s >= 1:
            rhs -= (1 - alpha) * c1 * nsum(X[:, s - 1], 1)
        if s >= 2:
            rhs -= (1 - alpha) * c2 * nsum(X[:, s - 2], 2)
        X[:, s] = solve_banded((2, 2), ab, rhs)
    return X


def build_estimates(ref, det):
    """One waveform pass -> per event & plane, all position estimates [mm]."""
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    blocks, pos_of = {}, {}
    for feu in CFG.MX17_FEUS:
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                        [0 if plane_of_feu[feu] == 'x' else 1]
                        for ch in range(512)], dtype=float)
        pos_of[feu] = pos
        ok = np.where(np.isfinite(pos))[0]
        o = ok[np.argsort(pos[ok])]
        brk = np.where(np.diff(pos[o]) > 1.5 * PITCH_MM)[0]
        blocks[feu] = np.split(o, brk + 1)

    rows = []
    for feu in CFG.MX17_FEUS:
        c1, c2 = CSHARE[feu]
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(32, 512) for a in a0
                                      if a.size == 32 * 512]), axis=(0, 1))
            for lo in range(0, t.num_entries, CHUNK):
                hi = min(lo + CHUNK, t.num_entries)
                want = [i for i in range(lo, hi) if int(eids_all[i]) in ref]
                if not want:
                    continue
                arr = t.arrays(['eventId', 'amplitude'], entry_start=lo,
                               entry_stop=hi, library='np')
                for i in want:
                    j = i - lo
                    eid = int(arr['eventId'][j])
                    if arr['amplitude'][j].size != 32 * 512:
                        continue                         # malformed multi-frame event
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    wfm = wfm.T
                    # raw cluster: contiguous >= THR_HIT around the max strip
                    best = None
                    for blk in blocks[feu]:
                        wb = wfm[blk]
                        amax = wb.max(axis=1)
                        k = int(np.argmax(amax))
                        if best is None or amax[k] > best[0]:
                            best = (amax[k], k, blk, wb)
                    a_pk, k, blk, wb = best
                    if a_pk < 1.5 * THR_HIT:
                        continue
                    amax = wb.max(axis=1)
                    l0 = k
                    while l0 - 1 >= 0 and amax[l0 - 1] >= THR_HIT:
                        l0 -= 1
                    r0 = k
                    while r0 + 1 < len(blk) and amax[r0 + 1] >= THR_HIT:
                        r0 += 1
                    sl_ = slice(l0, r0 + 1)
                    pos = pos_of[feu][blk[sl_]]
                    wraw = wb[sl_]
                    a_raw = wraw.max(axis=1)
                    wu = unshare(wb[max(0, l0 - 3):min(len(blk), r0 + 4)], c1, c2)
                    off = l0 - max(0, l0 - 3)
                    wu = wu[off:off + (r0 - l0 + 1)]
                    a_u = wu.max(axis=1)

                    row = dict(eid=eid, plane=plane_of_feu[feu],
                               n_strips=int(r0 - l0 + 1))
                    # centroids
                    row['cog_raw'] = float(np.sum(a_raw * pos) / a_raw.sum())
                    m_u = a_u >= THR_HIT
                    row['cog_u'] = float(np.sum(a_u[m_u] * pos[m_u]) / a_u[m_u].sum()) \
                        if m_u.sum() >= 1 else np.nan
                    # times (raw + unshared)
                    tt_u = np.array([cfd_time(w) for w in wu])
                    ok_u = np.isfinite(tt_u)
                    if ok_u.sum() >= 1:
                        i_lead = int(np.nanargmin(np.where(ok_u, tt_u, np.inf)))
                        row['lead_u'] = float(pos[i_lead])
                        t0 = float(np.nanmin(tt_u[ok_u]))
                    else:
                        row['lead_u'], t0 = np.nan, np.nan
                    # early-charge centroids (window from cluster start)
                    if np.isfinite(t0):
                        s0 = max(0, int(t0 // SAMPLE_NS))
                        qe_raw = np.clip(wraw[:, s0:s0 + EARLY_K], 0, None).sum(axis=1)
                        qe_u = np.clip(wu[:, s0:s0 + EARLY_K], 0, None).sum(axis=1)
                        row['early_raw'] = float(np.sum(qe_raw * pos) / qe_raw.sum()) \
                            if qe_raw.sum() > THR_HIT else np.nan
                        row['early_u'] = float(np.sum(qe_u * pos) / qe_u.sum()) \
                            if qe_u.sum() > THR_HIT else np.nan
                    else:
                        row['early_raw'], row['early_u'] = np.nan, np.nan
                    # track-fit impact point: weighted pos(t) on the core,
                    # evaluated at the cluster start time t0
                    mcore = (a_u >= CORE_FRAC * a_u.max()) & ok_u
                    if mcore.sum() >= 2 and np.isfinite(t0) \
                            and np.ptp(tt_u[mcore]) > 0:
                        wgt = a_u[mcore]
                        tc, pc = tt_u[mcore], pos[mcore]
                        W = wgt.sum()
                        tm, pm = np.sum(wgt * tc) / W, np.sum(wgt * pc) / W
                        vart = np.sum(wgt * (tc - tm) ** 2)
                        m = np.sum(wgt * (tc - tm) * (pc - pm)) / vart \
                            if vart > 0 else 0.0
                        row['fit_t0'] = float(pm + m * (t0 - tm))
                    else:
                        row['fit_t0'] = np.nan
                    rows.append(row)
            print(f'  {os.path.basename(fn)} done ({len(rows):,})')
    return pd.DataFrame(rows)


def sigma68(v):
    q = np.percentile(v, [16, 84])
    return 0.5 * (q[1] - q[0])


def main():
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (r.ref_x_mm, r.ref_y_mm,
                           float(np.degrees(np.arctan(np.hypot(tx, ty)))),
                           r.det_x_mm, r.det_y_mm)
    print(f'{len(ref):,} matched events')

    if os.path.exists(EST_CSV) and not REBUILD:
        est = pd.read_csv(EST_CSV)
        print(f'Loaded cached estimates: {len(est):,} rows')
    else:
        est = build_estimates(ref, det)
        est.to_csv(EST_CSV, index=False)
        print(f'Estimates cached: {len(est):,} rows -> {EST_CSV}')

    # ---- assemble per-event raw (x, y) per estimator ----
    ex = est[est['plane'] == 'x'].drop_duplicates('eid').set_index('eid')
    ey = est[est['plane'] == 'y'].drop_duplicates('eid').set_index('eid')
    eids = np.array(sorted(set(ex.index) & set(ey.index) & set(ref)))
    ref_x = np.array([ref[e][0] for e in eids])
    ref_y = np.array([ref[e][1] for e in eids])
    th_sp = np.array([ref[e][2] for e in eids])

    # reference positions as a function of z (for the effective-depth scan):
    # rays are straight lines -> evaluate at two planes and interpolate
    zs = np.linspace(best.z_mean - 24, best.z_mean + 24, 17)
    xz0, yz0, ev0 = get_xy_positions(rays.ray_data, zs[0])
    xz1, yz1, ev1 = get_xy_positions(rays.ray_data, zs[-1])
    bx0 = dict(zip(ev0, best.ref_x_sign * np.array(xz0)))
    by0 = dict(zip(ev0, yz0))
    bx1 = dict(zip(ev1, best.ref_x_sign * np.array(xz1)))
    by1 = dict(zip(ev1, yz1))
    have_z = np.array([e in bx0 and e in bx1 for e in eids])
    rx0 = np.array([bx0.get(e, np.nan) for e in eids])
    ry0 = np.array([by0.get(e, np.nan) for e in eids])
    rx1 = np.array([bx1.get(e, np.nan) for e in eids])
    ry1 = np.array([by1.get(e, np.nan) for e in eids])

    cos_t, sin_t = np.cos(th), np.sin(th)

    def to_aligned(xr, yr):
        u, v = xr - best.centre_x, yr - best.centre_y
        return (cos_t * u - sin_t * v + best.centre_x + best.x_offset,
                sin_t * u + cos_t * v + best.centre_y + best.y_offset)

    prod_x = np.array([ref[e][3] for e in eids])
    prod_y = np.array([ref[e][4] for e in eids])
    ns_x = ex['n_strips'].reindex(eids).to_numpy()
    ns_y = ey['n_strips'].reindex(eids).to_numpy()

    def raw_positions(name):
        if name == 'prod':
            return prod_x, prod_y
        if name == 'combo':
            # per-plane switch on that plane's raw footprint; fall back to
            # the production anchor whenever the early centroid is missing
            e_x = ex['early_raw'].reindex(eids).to_numpy()
            e_y = ey['early_raw'].reindex(eids).to_numpy()
            xr = np.where((ns_x <= N_SWITCH) & np.isfinite(e_x), e_x, prod_x)
            yr = np.where((ns_y <= N_SWITCH) & np.isfinite(e_y), e_y, prod_y)
            return xr, yr
        return (ex[name].reindex(eids).to_numpy(),
                ey[name].reindex(eids).to_numpy())

    def residuals(name, zfrac=None):
        """Aligned-frame residuals for estimator `name`. zfrac in [0,1]
        interpolates the reference between zs[0] and zs[-1]; None = the
        alignment plane (attach_reference_positions convention)."""
        xr, yr = raw_positions(name)
        xa, ya = to_aligned(xr, yr)
        if zfrac is None:
            rxx, ryy = ref_x, ref_y
        else:
            rxx = rx0 + zfrac * (rx1 - rx0)
            ryy = ry0 + zfrac * (ry1 - ry0)
        dx, dy = rxx - xa, ryy - ya
        m = np.isfinite(dx) & np.isfinite(dy)
        if zfrac is not None:
            m &= have_z
        if name == 'combo':
            # remove offsets per BRANCH (the two mixed estimators carry
            # different systematic offsets; one global median would leave a
            # relative shift that fakes a broadening)
            e_x = ex['early_raw'].reindex(eids).to_numpy()
            e_y = ey['early_raw'].reindex(eids).to_numpy()
            bx = (ns_x <= N_SWITCH) & np.isfinite(e_x)
            by = (ns_y <= N_SWITCH) & np.isfinite(e_y)
            for arr, br in ((dx, bx), (dy, by)):
                for bm in (br, ~br):
                    mm_ = m & bm
                    if mm_.sum() > 50:
                        arr[bm] -= np.median(arr[mm_])
        else:
            dx = dx - np.median(dx[m])
            dy = dy - np.median(dy[m])
        m &= (np.abs(dx) < MAX_RES_MM) & (np.abs(dy) < MAX_RES_MM)
        return dx, dy, m

    # ---- headline table ----
    bands = [('all', np.ones(len(eids), bool)),
             ('th<5', th_sp < 5), ('5-15', (th_sp >= 5) & (th_sp < 15)),
             ('th>15', th_sp >= 15)]
    print(f'\n== sigma68 [mm] per plane (x / y), offset-removed, '
          f'|res|<{MAX_RES_MM:g} mm ==')
    print(f'{"estimator":38s} {"cov":>5s} ' +
          ' '.join(f'{b:>11s}' for b, _ in bands))
    summ = []
    for nm in ESTIMATORS:
        dx, dy, m = residuals(nm)
        cov = m.mean()
        cells = []
        for bname, bm in bands:
            mm_ = m & bm
            if mm_.sum() > 100:
                sx, sy = sigma68(dx[mm_]), sigma68(dy[mm_])
                cells.append(f'{sx:4.2f}/{sy:4.2f}')
                summ.append(dict(estimator=nm, band=bname, n=int(mm_.sum()),
                                 s68_x=sx, s68_y=sy, coverage=cov))
            else:
                cells.append('    -    ')
        print(f'{EST_LABEL[nm]:38s} {100*cov:4.0f}% ' +
              ' '.join(f'{c:>11s}' for c in cells))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, 'position_summary.csv'), index=False)

    # ---- z-scan: effective measurement plane ----
    zscan = {}
    for nm in ESTIMATORS:
        sig = []
        for i, z in enumerate(zs):
            f = i / (len(zs) - 1)
            dx, dy, m = residuals(nm, zfrac=f)
            sig.append(0.5 * (sigma68(dx[m]) + sigma68(dy[m])) if m.sum() > 300 else np.nan)
        zscan[nm] = np.array(sig)

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))
    colors = dict(zip(ESTIMATORS, ['gray', 'tab:blue', 'tab:green', 'olive',
                                   'crimson', 'tab:orange', 'tab:purple', 'k']))

    ax = axes[0, 0]
    for nm in ['prod', 'cog_raw', 'early_raw', 'fit_t0', 'combo']:
        dx, dy, m = residuals(nm)
        ax.hist(dx[m], bins=np.arange(-4, 4.02, 0.08), histtype='step', lw=1.8,
                density=True, color=colors[nm],
                label=f'{EST_LABEL[nm]} ({sigma68(dx[m]):.2f})')
    ax.set_xlabel('x residual [mm]'); ax.set_ylabel('normalised')
    ax.set_title('x-plane residuals (σ68 in legend)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    edges_th = np.arange(0, 30, 2.5)
    for nm in ['prod', 'cog_raw', 'early_raw', 'fit_t0', 'combo']:
        dx, dy, m = residuals(nm)
        ctr, sig = [], []
        for b0, b1 in zip(edges_th[:-1], edges_th[1:]):
            mm_ = m & (th_sp >= b0) & (th_sp < b1)
            if mm_.sum() > 150:
                ctr.append(0.5 * (b0 + b1))
                sig.append(0.5 * (sigma68(dx[mm_]) + sigma68(dy[mm_])))
        ax.plot(ctr, sig, 'o-', ms=4, color=colors[nm], label=EST_LABEL[nm])
    ax.set_xlabel('ray space angle θ [deg]'); ax.set_ylabel('mean σ68(x,y) [mm]')
    ax.set_title('resolution vs track angle'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_ylim(0, 2.2)

    ax = axes[0, 2]
    for nm in ESTIMATORS:
        ax.plot(zs, zscan[nm], 'o-', ms=3, color=colors[nm],
                label=EST_LABEL[nm].split(' (')[0])
    ax.axvline(best.z_mean, color='k', ls='--', lw=1, label='alignment plane')
    ax.set_xlabel('assumed reference z [mm]'); ax.set_ylabel('mean σ68 [mm]')
    ax.set_title('z-scan: effective measurement plane per estimator')
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    for nm in ESTIMATORS:
        dx, dy, m = residuals(nm)
        r2 = np.hypot(dx[m], dy[m])
        xs = np.sort(r2)
        ax.plot(xs, np.linspace(0, 1, len(xs)), '-', color=colors[nm],
                label=f'{EST_LABEL[nm].split(" (")[0]} '
                      f'(68%: {np.percentile(r2, 68):.2f} mm)')
    ax.axhline(0.68, color='k', ls=':', lw=1)
    ax.set_xlim(0, 4)
    ax.set_xlabel('2D residual r [mm]'); ax.set_ylabel('cumulative fraction')
    ax.set_title('2D residual CDF'); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    # best-vs-baseline per angle band bar chart (mean of x/y)
    width = 0.11
    for k, nm in enumerate(ESTIMATORS):
        vals = []
        for bname, bm in bands[1:]:
            rowv = [s for s in summ if s['estimator'] == nm and s['band'] == bname]
            vals.append(0.5 * (rowv[0]['s68_x'] + rowv[0]['s68_y']) if rowv else np.nan)
        ax.bar(np.arange(3) + (k - 3) * width, vals, width, color=colors[nm],
               label=EST_LABEL[nm].split(' (')[0])
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([b for b, _ in bands[1:]])
    ax.set_ylabel('mean σ68(x,y) [mm]'); ax.set_title('by angle band')
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=7.5)

    ax = axes[1, 2]
    ax.axis('off')
    lines = ['POSITION ESTIMATORS — SUMMARY', '',
             f'{"estimator":24s} {"σx":>5s} {"σy":>5s} {"r68":>5s}']
    for nm in ESTIMATORS:
        dx, dy, m = residuals(nm)
        r68 = np.percentile(np.hypot(dx[m], dy[m]), 68)
        lines.append(f'{EST_LABEL[nm].split(" (")[0]:24s} '
                     f'{sigma68(dx[m]):5.2f} {sigma68(dy[m]):5.2f} {r68:5.2f}')
    lines += ['', f'sample: {len(eids):,} matched events',
              f'offsets removed per estimator;',
              f'residual window ±{MAX_RES_MM:g} mm',
              '', 'z-scan minima locate each estimator\'s',
              'effective depth: mesh-anchored ones at',
              'the alignment plane, full centroid',
              'deeper by ~half the recorded column.']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top',
            fontsize=10.5, family='monospace')

    fig.suptitle(f'{CFG.RUN} — hit-mode position estimators vs M3 v2 '
                 f'(track-pointing, centroids, early-charge)', fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'position_benchmark.png'), dpi=155)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
