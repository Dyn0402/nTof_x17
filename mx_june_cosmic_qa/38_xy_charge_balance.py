#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_xy_charge_balance.py

PAPER POINT 4 — charge balance between the X strip layer and the Y strip
layer through the pixelated top layer (the one paper topic with NO prior
measurement anywhere in the repo).

The avalanche charge lands on the pixelated top layer; the pixels route it
capacitively/resistively to the X strips and, below them, the Y strips.  If
the routing works the balance fraction

    f = q_X / (q_X + q_Y)

is NARROW and POSITION-INDEPENDENT.  Its central value need not be 0.5 (Y
sits below X and may systematically see less charge) -- the *constancy* is
the design-success metric; the central value is a design number worth
quoting.  A wide or position-dependent f means uneven pixel coupling; the
event-by-event spread of f at fixed total charge measures the sharing
fluctuation.

Two independent charge levels, cross-checked (they should agree in the
overlap; the difference bounds noise-strip contamination of the hits sums):

  HITS level   combined_hits `hits` tree, one row per strip pulse.  Per event
               & plane q = Sum(amplitude) over that plane's strips with
               amplitude >= THR_HIT.  Full coverage (~all firing events).
  SEG level    microtpc_segments.csv `amp_sum` (already clustered, unshared
               footprint, spark-free) -- the ~51 % of tracks with a fit.

The combined_hits reco fits a saturation-CORRECTED `amplitude` (saturated
strips have median amplitude ~4750 ADC, above the ~4150 12-bit clip seen in
the raw `local_max`), so a saturated peak strip is NOT a reason to drop the
event -- doing so would keep only low-charge tracks and bias f.  Instead the
core sample drops only genuine fit FAILURES (amplitude above a sanity
ceiling), and the per-hit `saturated` flag is used to SPLIT the core
(≥1 saturated strip vs none) as a systematic: if the saturation correction
is unbiased the two f distributions coincide.  `integral` (pulse area = true
collected charge, NOT saturation-corrected) is a second charge proxy,
evaluated on the unsaturated subset where it is unbiased, so the headline f
is shown not to be an artefact of using peak amplitude.

Selection: matched single M3 v2 ray, both planes fired, ray inside the
active area with a 5 mm fiducial margin, radial residual < 10 mm.  Events
come from the veto50 results cache, so the >50-strip spark veto is already
applied.

Usage: ../.venv/bin/python 38_xy_charge_balance.py sat_det3 [--veto=50]
       (det2 twin: o22_long_det2 -- run it FIRST so the det3 figure can
        overlay its f distribution)
Output: <alignment_tpc_vetoN>/charge_balance/
        xy_charge_balance.png, xy_charge_balance.csv, f_hits_core.npy
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
from matplotlib.colors import LogNorm

from qa_config import config_from_argv, get_config, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from common.mx17_active_area import draw_outlines
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
RES_CUT_MM = 10.0              # clean single-track match
FID_MARGIN_MM = 5.0            # fiducial inset from the active-area edges
THR_HIT = 100.0                # per-strip amplitude floor [ADC]
AMP_CEIL = 5.0e4               # per-strip amplitude sanity ceiling: real pulses
                               # (even saturation-corrected) reach ~2e4; anything
                               # above is a fit divergence -> flag the event bad
MAX_REF_TAN = 0.9              # analysis angle range

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'charge_balance')
# det3 <-> det2 twin, for the cross-detector f overlay
TWIN = {'sat_det3': 'o22_long_det2', 'o22_long_det2': 'sat_det3'}.get(CFG.KEY)


def twin_f_path(key):
    """Path to the twin run's saved core-f array (may not exist yet)."""
    if key is None:
        return None
    c = get_config(key)
    return os.path.join(c.OUT_BASE, f'alignment_tpc{tag}', 'charge_balance',
                        'f_hits_core.npy')


def active_bounds(det):
    """Active-area bounding box from the strip map itself (map_hit)."""
    lims = {}
    for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))[axis]
                        for ch in range(512)], dtype=float)
        pos = pos[np.isfinite(pos)]
        lims[axis] = (float(pos.min()), float(pos.max()))
    return lims[0], lims[1]


def hits_charge_by_event(ref):
    """Stream combined_hits; per event in `ref`, sum per-plane charge.

    Returns DataFrame(eid, qx, qy, nx, ny, qx_int, qy_int, sat, bad) where
    q* are (saturation-corrected) amplitude sums (>= THR_HIT), q*_int the
    matching integral sums, `sat` True if any strip in either plane is flagged
    saturated (a systematic split, kept in the core), and `bad` True if any
    strip exceeds the amplitude sanity ceiling (a fit divergence -> dropped)."""
    fx, fy = CFG.MX17_FEU_X, CFG.MX17_FEU_Y
    acc = {}   # eid -> [qx, qy, nx, ny, qx_int, qy_int, sat, bad]
    files = sorted(glob.glob(os.path.join(CFG.combined_hits_dir, '*.root')))
    for fn in files:
        arr = uproot.open(fn)['hits'].arrays(
            ['eventId', 'feu', 'channel', 'amplitude', 'integral', 'saturated'],
            library='np')
        eid = arr['eventId']
        feu = arr['feu']
        amp = arr['amplitude'].astype(np.float64)
        itg = arr['integral'].astype(np.float64)
        sat = arr['saturated'].astype(bool)
        keep = np.isin(feu, (fx, fy)) & (amp >= THR_HIT)
        for e, fu, a, ig, sa in zip(eid[keep], feu[keep], amp[keep],
                                    itg[keep], sat[keep]):
            e = int(e)
            if e not in ref:
                continue
            r = acc.get(e)
            if r is None:
                r = acc[e] = [0.0, 0.0, 0, 0, 0.0, 0.0, False, False]
            if fu == fx:
                r[0] += a; r[2] += 1; r[4] += ig
            else:
                r[1] += a; r[3] += 1; r[5] += ig
            if sa:
                r[6] = True
            if a > AMP_CEIL:
                r[7] = True
        print(f'  {os.path.basename(fn)} done ({len(acc):,} events)')
    rows = [dict(eid=e, qx=v[0], qy=v[1], nx=v[2], ny=v[3],
                 qx_int=v[4], qy_int=v[5], sat=v[6], bad=v[7])
            for e, v in acc.items() if v[2] > 0 and v[3] > 0]
    return pd.DataFrame(rows)


def sigma68(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    q = np.percentile(v, [16, 84])
    return 0.5 * (q[1] - q[0])


def profile(x, y, edges, min_n=60, stat='med'):
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        n = int(m.sum())
        if n < min_n:
            continue
        ctr.append(0.5 * (b0 + b1))
        if stat == 'med':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(q[1]); err.append(0.5 * (q[2] - q[0]) / np.sqrt(n))
        elif stat == 's68':
            q = np.percentile(y[m], [16, 84])
            val.append(0.5 * (q[1] - q[0]))
            err.append(0.5 * (q[1] - q[0]) / np.sqrt(2 * n))
    return np.array(ctr), np.array(val), np.array(err)


def main():
    cache_res = os.path.join(CFG.OUT_BASE, 'cache', f'event_results{tag}.pkl')
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
    (xmn, xmx), (ymn, ymx) = active_bounds(det)
    print(f'active area: x [{xmn:.0f}, {xmx:.0f}]  y [{ymn:.0f}, {ymx:.0f}] mm')

    # ---- matched, fiducial single-track reference sample ----
    # NB position/fiducial use the production detector hit position
    # (det_{x,y}_mm, strip-map frame [0, active]) -- that is where the charge
    # landed and shares the frame with active_bounds.  The M3 ref_{x,y}_mm live
    # in the aligned/telescope frame (different origin) and are used only via
    # the residual match; the ray ANGLE comes from ref_tan_theta.
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y):
            continue
        if not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > RES_CUT_MM:
            continue
        if not (np.isfinite(r.det_x_mm) and np.isfinite(r.det_y_mm)):
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_tan_theta_y):
            continue
        if not (xmn + FID_MARGIN_MM < r.det_x_mm < xmx - FID_MARGIN_MM
                and ymn + FID_MARGIN_MM < r.det_y_mm < ymx - FID_MARGIN_MM):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        tan_sp = float(np.hypot(tx, ty))
        if tan_sp > MAX_REF_TAN:
            continue
        ref[r.event_id] = (r.det_x_mm, r.det_y_mm, tan_sp)
    print(f'{len(ref):,} matched fiducial single-track events '
          f'(r<{RES_CUT_MM:g} mm, {FID_MARGIN_MM:g} mm margin, det frame)')

    # ---- hits-level per-event charge ----
    hf = hits_charge_by_event(ref)
    hf['rx'] = [ref[e][0] for e in hf['eid']]
    hf['ry'] = [ref[e][1] for e in hf['eid']]
    hf['tan_sp'] = [ref[e][2] for e in hf['eid']]
    hf['qtot'] = hf['qx'] + hf['qy']
    hf['f'] = hf['qx'] / hf['qtot']
    hf['logr'] = np.log(hf['qx'] / hf['qy'])
    hf['f_int'] = hf['qx_int'] / (hf['qx_int'] + hf['qy_int'])
    # core = everything except genuine fit failures; saturated-strip events are
    # KEPT (amplitude is saturation-corrected) and split out only as a systematic
    core = hf[~hf['bad'] & (hf['qtot'] > 0)].copy()
    nsat = int(core['sat'].sum())
    unsat = core[~core['sat']]                # integral is unbiased only here
    print(f'hits-level: {len(hf):,} both-plane events; '
          f'{int(hf["bad"].sum()):,} fit-failure dropped -> {len(core):,} core '
          f'({nsat:,} with ≥1 saturated strip, {len(unsat):,} clean)')
    # cache the full per-event core table for downstream report figures
    core.to_csv(os.path.join(OUT, 'charge_balance_events.csv'), index=False)

    # ---- segment-level cross-check (amp_sum, already clustered) ----
    seg_csv = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}',
                           'microtpc_metrics', 'microtpc_segments.csv')
    seg_f = None
    if os.path.exists(seg_csv):
        seg = pd.read_csv(seg_csv)
        sx = seg[seg['plane'] == 'x'].drop_duplicates('eid').set_index('eid')['amp_sum']
        sy = seg[seg['plane'] == 'y'].drop_duplicates('eid').set_index('eid')['amp_sum']
        common = sx.index.intersection(sy.index).intersection(list(ref))
        if len(common):
            qsx = sx.reindex(common).to_numpy()
            qsy = sy.reindex(common).to_numpy()
            seg_f = qsx / (qsx + qsy)
            print(f'segment-level: {len(common):,} dual-plane fitted events')

    # ---- core statistics ----
    f = core['f'].to_numpy()
    fmed, fs68 = float(np.median(f)), sigma68(f)
    lr = core['logr'].to_numpy()
    lrmed, lrs68 = float(np.median(lr)), sigma68(lr)
    pear = float(np.corrcoef(core['qx'], core['qy'])[0, 1])
    fint_med = float(np.median(unsat['f_int'])) if len(unsat) else np.nan
    seg_med = float(np.median(seg_f)) if seg_f is not None else np.nan
    # saturation systematic: median f with vs without a saturated strip
    f_sat = core[core['sat']]['f'].to_numpy()
    f_clean = core[~core['sat']]['f'].to_numpy()
    dsat = (float(np.median(f_sat)) - float(np.median(f_clean))) \
        if len(f_sat) and len(f_clean) else np.nan
    # f vs |tan| slope (robust линейный fit on the profile is overkill; use
    # a straight polyfit of the medians)
    ct, mt, _ = profile(core['tan_sp'].to_numpy(), f,
                        np.arange(0, MAX_REF_TAN + 0.01, 0.1), 80, 'med')
    slope_tan = float(np.polyfit(ct, mt, 1)[0]) if len(ct) >= 3 else np.nan
    print(f'\n== CORE charge balance (hits level, amplitude) ==')
    print(f'  f = qX/(qX+qY): median {fmed:.3f}, sigma68 {fs68:.3f}')
    print(f'  ln(qX/qY):      median {lrmed:+.3f}, sigma68 {lrs68:.3f}')
    print(f'  Pearson r(qX,qY) = {pear:.3f}')
    print(f'  f(integral) median = {fint_med:.3f}  (unsat subset, peak-amp check)')
    print(f'  f(segment)  median = {seg_med:.3f}  (clustered cross-check)')
    print(f'  saturation systematic Δf(sat−clean) = {dsat:+.3f}')
    print(f'  slope df/d|tan| = {slope_tan:+.3f}')

    # ---- f-map over the active area (median f per 2D bin) ----
    nbx, nby = 14, 14
    xe = np.linspace(xmn, xmx, nbx + 1)
    ye = np.linspace(ymn, ymx, nby + 1)
    MIN_BIN = 40
    fmap = np.full((nby, nbx), np.nan)
    rxv, ryv = core['rx'].to_numpy(), core['ry'].to_numpy()
    ix = np.clip(np.digitize(rxv, xe) - 1, 0, nbx - 1)
    iy = np.clip(np.digitize(ryv, ye) - 1, 0, nby - 1)
    exp_scat = []                    # expected per-bin median scatter from stats
    for a in range(nbx):
        for b in range(nby):
            m = (ix == a) & (iy == b)
            n = int(m.sum())
            if n >= MIN_BIN:
                fmap[b, a] = np.median(f[m])
                if 1 <= a <= nbx - 2 and 1 <= b <= nby - 2:
                    exp_scat.append(1.2533 * fs68 / np.sqrt(n))
    # flatness excluding the known low-X fringe / edge (1-bin frame). The
    # bin-to-bin STD is the design flatness metric (max|Δf| is tail-sensitive);
    # compare it to the scatter EXPECTED from finite per-bin statistics.
    inner = fmap[1:-1, 1:-1]
    fin = inner[np.isfinite(inner)]
    dstd = float(np.std(fin)) if len(fin) else np.nan
    dmax = float(fin.max() - fin.min()) if len(fin) else np.nan
    exp_std = float(np.sqrt(np.mean(np.square(exp_scat)))) if exp_scat else np.nan
    print(f'  f-map inner flatness: std(f) = {dstd:.3f} '
          f'(expected from stats {exp_std:.3f}), max|Δf| = {dmax:.3f} '
          f'over {len(fin)} bins')

    # ---- save core-f for the twin overlay + summary CSV ----
    np.save(os.path.join(OUT, 'f_hits_core.npy'), f)
    summ = []
    summ.append(dict(level='hits', det=CFG.DET_NAME, charge='amplitude',
                     n=len(core), r_qxqy=pear, f_median=fmed, f_sigma68=fs68,
                     logr_median=lrmed, logr_sigma68=lrs68,
                     slope_f_vs_tan=slope_tan, fmap_std=dstd,
                     fmap_exp_std=exp_std, fmap_dmax=dmax))
    summ.append(dict(level='hits', det=CFG.DET_NAME, charge='integral_unsat',
                     n=len(unsat), r_qxqy=float(np.corrcoef(unsat['qx_int'],
                     unsat['qy_int'])[0, 1]) if len(unsat) > 5 else np.nan,
                     f_median=fint_med, f_sigma68=sigma68(unsat['f_int']),
                     logr_median=np.nan, logr_sigma68=np.nan,
                     slope_f_vs_tan=np.nan, fmap_dmax=np.nan))
    if seg_f is not None:
        summ.append(dict(level='segment', det=CFG.DET_NAME, charge='amp_sum',
                         n=len(seg_f), r_qxqy=np.nan, f_median=seg_med,
                         f_sigma68=sigma68(seg_f), logr_median=np.nan,
                         logr_sigma68=np.nan, slope_f_vs_tan=np.nan,
                         fmap_dmax=np.nan))
    # saturation systematic: core split by presence of a saturated strip
    for lab, fv in [('amplitude_satstrip', f_sat), ('amplitude_cleanstrip', f_clean)]:
        summ.append(dict(level='hits', det=CFG.DET_NAME, charge=lab,
                         n=len(fv), r_qxqy=np.nan,
                         f_median=float(np.median(fv)) if len(fv) else np.nan,
                         f_sigma68=sigma68(fv), logr_median=np.nan,
                         logr_sigma68=np.nan, slope_f_vs_tan=dsat,
                         fmap_dmax=np.nan))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, 'xy_charge_balance.csv'), index=False)

    # ================= FIGURE =================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    # (0,0) qX vs qY 2D
    ax = axes[0, 0]
    qm = np.percentile(core['qtot'], 99) * 0.7
    ax.hist2d(core['qx'], core['qy'], bins=[80, 80],
              range=[[0, qm], [0, qm]], norm=LogNorm(), cmap='viridis')
    ax.plot([0, qm], [0, qm], 'r--', lw=1, label='qX = qY')
    ax.plot([0, qm], [0, qm * (1 - fmed) / fmed], color='w', lw=1.2,
            label=f'median f = {fmed:.3f}')
    ax.set_xlabel('q_X  (Σ amplitude, X strips) [ADC]')
    ax.set_ylabel('q_Y  (Σ amplitude, Y strips) [ADC]')
    ax.set_title(f'X vs Y layer charge (r = {pear:.3f})')
    ax.legend(fontsize=9, loc='upper right')

    # (0,1) f histogram + twin overlay + integral + segment
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 61)
    ax.hist(f, bins=bins, density=True, histtype='stepfilled', color='tab:blue',
            alpha=0.55, label=f'{CFG.DET_NAME} hits (med {fmed:.3f}, σ {fs68:.3f})')
    ax.hist(unsat['f_int'], bins=bins, density=True, histtype='step', lw=1.8,
            color='tab:green', label=f'integral charge (med {fint_med:.3f})')
    if seg_f is not None:
        ax.hist(seg_f, bins=bins, density=True, histtype='step', lw=1.8,
                color='crimson', label=f'segment amp_sum (med {seg_med:.3f})')
    tf = twin_f_path(TWIN)
    if tf and os.path.exists(tf):
        ftw = np.load(tf)
        ax.hist(ftw, bins=bins, density=True, histtype='step', lw=1.8,
                color='k', ls='--',
                label=f'{get_config(TWIN).DET_NAME} hits (med {np.median(ftw):.3f})')
    ax.axvline(0.5, color='gray', ls=':', lw=1)
    ax.set_xlabel('f = q_X / (q_X + q_Y)'); ax.set_ylabel('normalised')
    ax.set_title('balance fraction — narrow = good routing')
    ax.legend(fontsize=8)

    # (0,2) f vs |tan theta|
    ax = axes[0, 2]
    ctr, med, err = profile(core['tan_sp'].to_numpy(), f,
                            np.arange(0, MAX_REF_TAN + 0.01, 0.1), 80, 'med')
    ax.errorbar(ctr, med, yerr=err, fmt='o-', color='tab:blue', ms=5)
    ax.axhline(fmed, color='gray', ls='--', lw=1, label=f'overall {fmed:.3f}')
    ax.set_xlabel('|tan θ_ref|  (track inclination)')
    ax.set_ylabel('median f')
    ax.set_title(f'f vs angle (slope {slope_tan:+.3f} / unit tan)')
    ax.set_ylim(fmed - 0.12, fmed + 0.12); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,0) f-map over the detector
    ax = axes[1, 0]
    im = ax.imshow(fmap, origin='lower', extent=[xmn, xmx, ymn, ymx],
                   aspect='auto', cmap='RdBu_r', vmin=fmed - 0.1, vmax=fmed + 0.1)
    draw_outlines(ax, det_name=CFG.DET_NAME)  # detector-local strip frame, no transform needed
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.set_xlabel('detector x [mm]'); ax.set_ylabel('detector y [mm]')
    ax.set_title(f'median-f map (inner std {dstd:.3f}, stat {exp_std:.3f})')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='median f')

    # (1,1) f vs q_tot, split by saturated-strip presence (systematic)
    ax = axes[1, 1]
    qedge = np.percentile(core['qtot'], np.linspace(0, 98, 16))
    cln = core[~core['sat']]
    stp = core[core['sat']]
    ctr, med, err = profile(cln['qtot'].to_numpy(), cln['f'].to_numpy(),
                            qedge, 40, 'med')
    ax.errorbar(ctr, med, yerr=err, fmt='o-', color='tab:blue', ms=5,
                label=f'no saturated strip (n={len(cln):,})')
    if len(stp) > 60:
        cs, ms_, es = profile(stp['qtot'].to_numpy(), stp['f'].to_numpy(),
                              np.percentile(stp['qtot'], np.linspace(0, 98, 12)),
                              40, 'med')
        ax.errorbar(cs, ms_, yerr=es, fmt='s--', color='crimson', ms=5,
                    label=f'≥1 saturated strip (n={len(stp):,})')
    ax.axhline(fmed, color='gray', ls=':', lw=1)
    ax.set_xlabel('total charge q_X + q_Y [ADC]'); ax.set_ylabel('median f')
    ax.set_title(f'f vs total charge (sat systematic Δf={dsat:+.3f})')
    ax.set_ylim(fmed - 0.14, fmed + 0.14); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # (1,2) sigma68(f) vs q_tot  +  text summary
    ax = axes[1, 2]
    ctr, s68, err = profile(core['qtot'].to_numpy(), f, qedge, 60, 's68')
    ax.errorbar(ctr, s68, yerr=err, fmt='o-', color='tab:purple', ms=5)
    ax.set_xlabel('total charge q_X + q_Y [ADC]')
    ax.set_ylabel('σ68(f)  (sharing fluctuation)')
    ax.set_title('event-by-event spread vs total charge')
    ax.grid(alpha=0.3)
    txt = [
        f'CHARGE BALANCE — {CFG.DET_NAME}',
        f'{CFG.RUN}',
        '',
        f'core events      {len(core):,}',
        f'  w/ sat strip   {nsat:,} ({100*nsat/max(len(core),1):.0f} %)',
        f'r(qX,qY)         {pear:.3f}',
        f'median f         {fmed:.3f}',
        f'σ68(f)           {fs68:.3f}',
        f'f (integral)     {fint_med:.3f}',
        f'f (segment)      {seg_med:.3f}',
        f'Δf(sat−clean)    {dsat:+.3f}',
        f'ln(qX/qY) med    {lrmed:+.3f}',
        f'df/d|tan|        {slope_tan:+.3f}',
        f'f-map std        {dstd:.3f} (stat {exp_std:.3f})',
    ]
    ax.text(0.98, 0.97, '\n'.join(txt), transform=ax.transAxes, va='top',
            ha='right', fontsize=9.5, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))

    fig.suptitle(f'{CFG.RUN} — X/Y layer charge balance through the pixel top '
                 f'layer ({CFG.DET_NAME}, M3 v2)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'xy_charge_balance.png'), dpi=155)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
