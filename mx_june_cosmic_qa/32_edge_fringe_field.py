#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_edge_fringe_field.py

Edge / fringe-field study of the det3 drift volume.

The drift gap (30 mm) is graded at its perimeter by a 3-step copper-ring
degrader; some field non-uniformity is still expected near the edges. A
fringe field with a transverse component E_t tilts the drift lines, which
in micro-TPC mode shows up as
  * an OUTWARD (or inward) apparent-angle offset for tracks near an edge,
  * a position-residual displacement toward/away from the edge,
  * locally longer/shorter drift-time spans (local v_drift change),
  * a shorter recorded column (strip extent) where drift lines exit the
    active volume.

CONFOUNDER, handled explicitly: resistive-strip charge sharing already
produces a sign-following angle offset (tan_det = tan_ref +/- w/z_rec,
scripts 13/27). That bias follows the ANGLE sign, not the position, so we
remove it by subtracting the whole-detector median response curve
baseline(tan_ref) per plane; the remainder delta = Dtan - baseline is the
distortion-sensitive observable, profiled against edge distance with an
OUTWARD sign convention (+ = apparent angle tilts away from detector
centre).

All positions are the M3 ray in RAW detector coordinates (ref_mesh_x/y),
so "edge" means the physical strip-plane edge where the degrader rings sit.

Uses the unshared segment table cached by 31_microtpc_metrics.py when
available (cleaner angles); otherwise falls back to the production hits
fits. T-span / extent / efficiency profiles always come from the hits-level
cache (waveform-independent).

Usage: ../.venv/bin/python 32_edge_fringe_field.py sat_det3 [--veto=50]
Output: <alignment_tpc_vetoN>/edge_fringe/ edge_fringe_field.png,
        edge_zone_table.csv + stdout
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
CHI2_CUT = 5.0   # M3 v2 recipe
RES_CUT_MM = 10.0
MIN_STRIPS = 4
PITCH_MM = 0.78
SAT_DEG = 10.0            # |theta| above which the time span saturates
TAN_LO, TAN_HI = 0.06, 0.55
EDGE_ZONES = ((0.0, 25.0, 'edge'), (25.0, 60.0, 'mid'), (60.0, 1e9, 'core'))
OTHER_AXIS_GUARD_MM = 40.0   # isolate one axis: stay away from the other axis' edges

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'edge_fringe')
SEG_CSV = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'microtpc_metrics',
                       'microtpc_segments.csv')


def robust_line(x, y, n_iter=4, clip=3.0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 10:
            return np.nan
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    return float(p[0])


def vetoed_event_ids():
    """Spark-vetoed eids (> VETO hit-rows) + the eid range where BOTH planes
    were alive, re-derived from combined_hits. Zero-hit events inside the
    live range count as inefficiency; events outside it (e.g. the Saturday
    run's file-003 tail, where FEU 8 died: eff 89% -> 45% -> 0%) leave the
    denominator -- same dead-file dilution the p2-run QA guard fixed."""
    files = [f for f in os.listdir(CFG.combined_hits_dir)
             if f.endswith('.root') and '_datrun_' in f]
    counts, feu_rng = {}, {f: [None, None] for f in CFG.MX17_FEUS}
    for f in files:
        arr = uproot.open(os.path.join(CFG.combined_hits_dir, f))['hits'].arrays(
            ['eventId', 'feu'], library='np')
        m = np.isin(arr['feu'], CFG.MX17_FEUS)
        eid, cnt = np.unique(arr['eventId'][m], return_counts=True)
        for e, c in zip(eid.tolist(), cnt.tolist()):
            counts[e] = counts.get(e, 0) + c
        for feu in CFG.MX17_FEUS:
            ef = arr['eventId'][arr['feu'] == feu]
            if len(ef):
                lo, hi = feu_rng[feu]
                feu_rng[feu] = [min(int(ef.min()), lo) if lo is not None else int(ef.min()),
                                max(int(ef.max()), hi) if hi is not None else int(ef.max())]
    vetoed = {e for e, c in counts.items() if c > VETO}
    eid_lo = max(r[0] for r in feu_rng.values())
    eid_hi = min(r[1] for r in feu_rng.values())
    print(f'per-FEU live eid ranges: {feu_rng} -> both-planes-alive [{eid_lo}, {eid_hi}]')
    return vetoed, (eid_lo, eid_hi)


def profile(x, y, edges, min_n=50, stat='med'):
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1) & np.isfinite(y)
        n = m.sum()
        if n < min_n:
            continue
        ctr.append(0.5 * (b0 + b1))
        if stat == 'med':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(q[1]); err.append(0.5 * (q[2] - q[0]) / np.sqrt(n))
        elif stat == 'eff':
            k = float(np.sum(y[m]))
            val.append(k / n); err.append(np.sqrt(max(k, 0.25) * (1 - k / n)) / n)
    return np.array(ctr), np.array(val), np.array(err)


def baseline_curve(tan_ref, dtan, nbins=28, lim=0.7):
    """Whole-detector median response Dtan(tan_ref): the charge-sharing bias
    curve, to be subtracted before looking for position dependence."""
    edges = np.linspace(-lim, lim, nbins + 1)
    ctr, med, _ = profile(tan_ref, dtan, edges, min_n=60)
    def f(t):
        return np.interp(t, ctr, med, left=np.nan, right=np.nan)
    return f


def active_bounds(det):
    """Active-area bounding box from the strip map itself (det.map_hit),
    the same lookup the waveform scripts use -- cm.get_active_det_bounds
    returns (0,0) for this run's config."""
    lims = {}
    for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))[axis]
                        for ch in range(512)], dtype=float)
        pos = pos[np.isfinite(pos)]
        lims[axis] = (float(pos.min()), float(pos.max()))
    return lims[0], lims[1]


def zone_of(d):
    for lo, hi, name in EDGE_ZONES:
        if lo <= d < hi:
            return name
    return 'core'


def main():
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    (xmn, xmx), (ymn, ymx) = active_bounds(det)
    cx0, cy0 = 0.5 * (xmn + xmx), 0.5 * (ymn + ymx)
    print(f'active area: x [{xmn:.0f}, {xmx:.0f}]  y [{ymn:.0f}, {ymx:.0f}] mm')

    # ---- per-event table from the hits-level cache ----
    rows = []
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x) \
                or np.isnan(r.ref_mesh_x_mm):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        rows.append(dict(
            eid=r.event_id, mx=r.ref_mesh_x_mm, my=r.ref_mesh_y_mm,
            tan_rx=tx, tan_ry=ty,
            S_x=r.x_fit.slope_mm_per_ns * 1000.0, S_y=r.y_fit.slope_mm_per_ns * 1000.0,
            n_x=r.x_fit.n_strips, n_y=r.y_fit.n_strips,
            dur_x=r.x_fit.cluster_duration_ns, dur_y=r.y_fit.cluster_duration_ns,
            ok_x=r.x_fit.n_strips >= MIN_STRIPS, ok_y=r.y_fit.n_strips >= MIN_STRIPS,
            res_x=r.residual_x_mm, res_y=r.residual_y_mm))
    ev = pd.DataFrame(rows)
    print(f'{len(ev):,} matched events (hits-level)')

    # ---- angles: prefer the unshared segment table from 31 ----
    src = 'production hits fits'
    if os.path.exists(SEG_CSV):
        seg = pd.read_csv(SEG_CSV)
        for p in 'xy':
            d = seg[seg['plane'] == p][['eid', 'S_um_ns']].rename(
                columns={'S_um_ns': f'S_u_{p}'})
            ev = ev.merge(d, on='eid', how='left')
        src = 'unshared waveform segments (31 cache)'
    else:
        ev['S_u_x'], ev['S_u_y'] = np.nan, np.nan
    use_unshared = np.isfinite(ev['S_u_x']).sum() > 2000
    print(f'angle source: {src if use_unshared else "production hits fits"}')

    # per-plane velocity scale (robust ridge fit on the chosen S) then Dtan
    def ridge_v(S, T):
        vs = []
        for lo, hi in [(TAN_LO, TAN_HI), (-TAN_HI, -TAN_LO)]:
            m = (T > lo) & (T < hi) & np.isfinite(S)
            if m.sum() < 200:
                continue
            v = robust_line(T[m], S[m])
            if np.isfinite(v):
                vs.append(v)
        return float(np.mean(vs)) if vs else np.nan

    # edge distances and outward signs (position-based)
    ev['d_x'] = np.minimum(ev['mx'] - xmn, xmx - ev['mx'])
    ev['d_y'] = np.minimum(ev['my'] - ymn, ymx - ev['my'])
    ev['d_edge'] = np.minimum(ev['d_x'], ev['d_y'])
    ev['o_x'] = np.where(ev['mx'] > cx0, 1.0, -1.0)   # outward = +x on the +x side
    ev['o_y'] = np.where(ev['my'] > cy0, 1.0, -1.0)

    dtan, base = {}, {}
    core_m = (ev['d_edge'] > EDGE_ZONES[1][1]).to_numpy()   # baseline from CORE only
    for p in 'xy':
        S = ev[f'S_u_{p}'].to_numpy() if use_unshared else ev[f'S_{p}'].to_numpy()
        T = ev[f'tan_r{p}'].to_numpy()
        v = ridge_v(S[core_m], T[core_m])
        ev[f'tan_det_{p}'] = S / v
        dtan[p] = ev[f'tan_det_{p}'].to_numpy() - T
        base[p] = baseline_curve(T[core_m], dtan[p][core_m])
        ev[f'delta_{p}'] = dtan[p] - base[p](T)   # bias-subtracted residual
        print(f'plane {p}: v_scale = {v:.1f} um/ns '
              f'(core-only baseline curve subtracted before edge profiling)')

    edges_d = np.arange(0, 130, 10.0)
    fig, axes = plt.subplots(3, 3, figsize=(19, 15))

    # ---- (0,0)/(0,1): 2D maps of the bias-subtracted angle residual ----
    for k, p in enumerate('xy'):
        ax = axes[0, k]
        nb = 16
        H, xe, ye = np.histogram2d(ev['mx'], ev['my'], bins=nb,
                                   range=[[xmn, xmx], [ymn, ymx]])
        num = np.full((nb, nb), np.nan)
        d = ev[f'delta_{p}'].to_numpy()
        ix = np.clip(((ev['mx'] - xmn) / (xmx - xmn) * nb).astype(int), 0, nb - 1)
        iy = np.clip(((ev['my'] - ymn) / (ymx - ymn) * nb).astype(int), 0, nb - 1)
        for i in range(nb):
            for j in range(nb):
                m = (ix == i) & (iy == j) & np.isfinite(d)
                if m.sum() >= 25:
                    num[i, j] = np.median(d[m])
        im = ax.imshow(num.T, origin='lower', extent=[xmn, xmx, ymn, ymx],
                       cmap='RdBu_r', vmin=-0.06, vmax=0.06, aspect='auto')
        plt.colorbar(im, ax=ax, label=f'median δtanθ_{p}')
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_title(f'δ{p} = Δtanθ_{p} − baseline(tanθ_ref): map')

    # ---- (0,2): hit + TPC efficiency vs edge distance ----
    x_ref, _, evx = rays.get_xy_positions(best.z_x)
    _, y_ref, evy = rays.get_xy_positions(best.z_y)
    x_by = dict(zip(evx, best.ref_x_sign * np.array(x_ref)))
    y_by = dict(zip(evy, y_ref))
    cos_t, sin_t = np.cos(th), np.sin(th)
    vetoed, (eid_lo, eid_hi) = vetoed_event_ids()
    d_ray, in_hit, in_tpc = [], [], []
    hit_ok = {r.event_id for r in results
              if r.has_both and np.isfinite(r.radial_residual_mm)
              and r.radial_residual_mm < RES_CUT_MM}
    tpc_ok = set(ev[ev['ok_x'] & ev['ok_y']]['eid'])
    for e in set(evx) & set(evy):
        u = x_by[e] - best.centre_x - best.x_offset
        v = y_by[e] - best.centre_y - best.y_offset
        mx = best.centre_x + cos_t * u + sin_t * v
        my = best.centre_y - sin_t * u + cos_t * v
        if not (xmn - 10 < mx < xmx + 10 and ymn - 10 < my < ymx + 10):
            continue
        if not (eid_lo <= e <= eid_hi) or e in vetoed:
            continue
        d_ray.append(min(mx - xmn, xmx - mx, my - ymn, ymx - my))
        in_hit.append(e in hit_ok)
        in_tpc.append(e in tpc_ok)
    d_ray = np.array(d_ray)
    ax = axes[0, 2]
    for arr, lab, c in [(np.array(in_hit, float), 'hit-mode (X+Y, r<10mm)', 'tab:green'),
                        (np.array(in_tpc, float), f'micro-TPC (≥{MIN_STRIPS} strips X+Y)', 'tab:blue')]:
        ctr, eff, err = profile(d_ray, arr, np.arange(-10, 130, 5.0), 80, 'eff')
        ax.errorbar(ctr, 100 * eff, yerr=100 * err, fmt='o-', ms=4, color=c, label=lab)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('ray distance to nearest edge [mm]'); ax.set_ylabel('efficiency [%]')
    ax.set_title('efficiency vs edge distance'); ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right'); ax.set_ylim(0, 100)

    # ---- (1,0)/(1,1): outward angle residual vs own-axis edge distance ----
    for k, p in enumerate('xy'):
        ax = axes[1, k]
        other = 'y' if p == 'x' else 'x'
        guard = ev[f'd_{other}'] > OTHER_AXIS_GUARD_MM
        d = ev[f'd_{p}'].to_numpy()
        o = ev[f'o_{p}'].to_numpy()
        val = (o * ev[f'delta_{p}']).to_numpy()
        for m_side, lab, c in [((ev[f'o_{p}'] > 0) & guard, f'+{p} side', 'tab:red'),
                               ((ev[f'o_{p}'] < 0) & guard, f'−{p} side', 'tab:blue')]:
            ctr, med, err = profile(d[m_side], val[m_side], edges_d, 50)
            ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=lab)
        ctr, med, err = profile(d[guard], val[guard], edges_d, 100)
        ax.errorbar(ctr, med, yerr=err, fmt='k--', lw=2, label='both sides')
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel(f'distance to {p}-edge [mm]')
        ax.set_ylabel(f'outward δtanθ_{p}')
        ax.set_title(f'{p}-plane: outward angle residual '
                     f'(guard: d_{other} > {OTHER_AXIS_GUARD_MM:.0f} mm)')
        ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(-0.08, 0.08)

    # ---- (1,2): outward position residual vs edge distance ----
    ax = axes[1, 2]
    for p, c in [('x', 'tab:blue'), ('y', 'tab:orange')]:
        other = 'y' if p == 'x' else 'x'
        guard = ev[f'd_{other}'] > OTHER_AXIS_GUARD_MM
        # residual = ref - det; outward-positive means det pulled INWARD
        val = (ev[f'o_{p}'] * ev[f'res_{p}']).to_numpy()
        ctr, med, err = profile(ev[f'd_{p}'].to_numpy()[guard], val[guard], edges_d, 50)
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c,
                    label=f'{p} plane (outward ref−det)')
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('distance to own-axis edge [mm]')
    ax.set_ylabel('outward (ref − det) [mm]')
    ax.set_title('position residual vs edge distance')
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(-1.5, 1.5)

    # ---- (2,0): time span of inclined tracks vs edge distance ----
    ax = axes[2, 0]
    tan_sat = np.tan(np.radians(SAT_DEG))
    for p, c in [('x', 'tab:blue'), ('y', 'tab:orange')]:
        m = (np.abs(ev[f'tan_r{p}']) > tan_sat) & ev[f'ok_{p}']
        ctr, med, err = profile(ev['d_edge'].to_numpy()[m],
                                ev[f'dur_{p}'].to_numpy()[m], edges_d, 50)
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=f'{p} plane')
    ax.set_xlabel('distance to nearest edge [mm]')
    ax.set_ylabel(f'median cluster Δt [ns]  (|θ_ref|>{SAT_DEG:.0f}°)')
    ax.set_title('drift-time span (T_sat proxy ∝ z_rec/v) vs edge')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (2,1): strip extent of inclined tracks vs edge distance ----
    ax = axes[2, 1]
    for p, c in [('x', 'tab:blue'), ('y', 'tab:orange')]:
        tanp = np.abs(ev[f'tan_r{p}'])
        m = (tanp > 0.18) & (tanp < 0.45) & ev[f'ok_{p}']
        ext = (ev[f'n_{p}'] - 1) * PITCH_MM / tanp    # extent/|tan| = column proxy [mm]
        ctr, med, err = profile(ev['d_edge'].to_numpy()[m], ext.to_numpy()[m],
                                edges_d, 50)
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=f'{p} plane')
    ax.set_xlabel('distance to nearest edge [mm]')
    ax.set_ylabel('extent / |tanθ| [mm]  (0.18<|tan|<0.45)')
    ax.set_title('recorded column proxy vs edge')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # ---- (2,2): zone table — local v_geom per zone ----
    ax = axes[2, 2]
    ax.axis('off')
    lines = [f'ZONES (dist to nearest edge):',
             '  edge  0–25 mm | mid 25–60 | core >60', '',
             f'{"zone":6s} {"n_ev":>6s} {"v_x":>6s} {"v_y":>6s} '
             f'{"T_x":>6s} {"T_y":>6s} {"δx":>7s} {"δy":>7s}']
    zone_rows = []
    ev['zone'] = [zone_of(d) for d in ev['d_edge']]
    for _, _, zn in EDGE_ZONES:
        d = ev[ev['zone'] == zn]
        row = dict(zone=zn, n_ev=len(d))
        for p in 'xy':
            # zone-local geometry velocity: extent-slope / T_sat
            tanp = np.abs(d[f'tan_r{p}'])
            msk = d[f'ok_{p}'] & (tanp > TAN_LO) & (tanp < TAN_HI)
            if msk.sum() > 300:
                ext = ((d[f'n_{p}'] - 1) * PITCH_MM)[msk]
                tt = tanp[msk]
                p1 = np.polyfit(tt, ext, 1)
                m_sat = d[f'ok_{p}'] & (tanp > tan_sat)
                tsat = float(np.median(d[f'dur_{p}'][m_sat])) if m_sat.sum() > 50 else np.nan
                row[f'v_{p}'] = p1[0] * 1000.0 / tsat if np.isfinite(tsat) else np.nan
                row[f'T_{p}'] = tsat
            else:
                row[f'v_{p}'], row[f'T_{p}'] = np.nan, np.nan
            row[f'delta_{p}'] = (float(np.nanmedian(d[f'o_{p}'] * d[f'delta_{p}']))
                                 if len(d) else np.nan)
        zone_rows.append(row)
        lines.append(f'{zn:6s} {row["n_ev"]:6d} {row.get("v_x", np.nan):6.1f} '
                     f'{row.get("v_y", np.nan):6.1f} {row.get("T_x", np.nan):6.0f} '
                     f'{row.get("T_y", np.nan):6.0f} {row["delta_x"]:+7.3f} '
                     f'{row["delta_y"]:+7.3f}')
    lines += ['', 'v in µm/ns, T in ns; δ = outward', 'angle residual (bias-subtracted)',
              '', f'angle source: {"unshared" if use_unshared else "production"}']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top',
            fontsize=10.5, family='monospace')
    pd.DataFrame(zone_rows).to_csv(os.path.join(OUT, 'edge_zone_table.csv'), index=False)
    print('\n== zone table ==')
    print('\n'.join(lines))

    fig.suptitle(f'{CFG.RUN} — {CFG.DET_NAME} edge / fringe-field study '
                 f'(3-ring degrader; M3 v2 rays)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, 'edge_fringe_field.png'), dpi=155)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
