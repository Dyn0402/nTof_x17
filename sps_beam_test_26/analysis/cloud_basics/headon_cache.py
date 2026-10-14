#!/usr/bin/env python3
"""headon_cache.py -- every near-head-on reference event of a bench run, as a waveform cache.

The big caches (plane_ratio/build_big_cache.py) hold the first 3000 reference events of a run, of which
~400 are head-on in a view: enough for one stack per chamber, not for a split by position or a second
dataset.  This is wft.calibrate.build_cache with the angle cut moved in front of the waveform read
(|tan| < TAN_MAX in either view) and no event cap, so a whole run's head-on sample fits in ~100 MB.
Same event format, so rc_diffusion.stack and bench_attach.predict take it unchanged.

    headon_cache.py <run_key> [--tan-max 0.06] [--out DIR]
        -> DIR/headon_<run_key>.pkl  (default DIR ~/x17/cosmic_bench/cloud_basics/xwidth/caches)
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
OUT = '/home/dylan/x17/cosmic_bench/cloud_basics/xwidth/caches'


def ref_only(cfg, best, m3_dir, tan_max):
    """Reference events straight from the M3 rays, through another sub-run's alignment (same run, no
    move): what attach_reference_positions does, without the hits chain -- so no has_x/has_y and no
    hit-residual cut; an empty window just fails the stack's threshold crossing."""
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions
    rays = M3RefTracking(m3_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, yang, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    xp, _, nx = get_xy_positions(rays.ray_data, best.z_x)
    _, yp, ny = get_xy_positions(rays.ray_data, best.z_y)
    xr = {e: best.ref_x_sign * float(x) for e, x in zip(nx, xp)}
    yr = {e: float(y) for e, y in zip(ny, yp)}
    th = np.deg2rad(best.theta_deg); c, s = np.cos(th), np.sin(th)
    cx, cy = best.centre_x, best.centre_y
    events = {}
    for e, ax, ay in zip(anum, xang, yang):
        if e not in xr or e not in yr:
            continue
        t1, t2 = np.tan(ax), np.tan(ay)
        tx, ty = c * t1 + s * t2, -s * t1 + c * t2
        if min(abs(tx), abs(ty)) >= tan_max:
            continue
        u, v = xr[e] - cx - best.x_offset, yr[e] - cy - best.y_offset
        events[int(e)] = dict(eid=int(e), tan_x=float(tx), tan_y=float(ty),
                              ref_mesh_x=float(cx + c * u + s * v), ref_mesh_y=float(cy - s * u + c * v))
    print(f'[headon] ref-only: {len(anum):,} M3 rays, {len(events):,} with |tan| < {tan_max} in a view', flush=True)
    return events


def build(cfg, tan_max, pad_mm=5.0, z_lo=-3.0, z_hi=33.0, res_cut_mm=10.0, veto=50, refonly=None):
    """refonly = (alignment cfg, m3 tracking dir): take the reference from the rays alone (ref_only)."""
    from wft import io as wio
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    import cosmic_micro_tpc_analysis as cm
    from M3RefTracking import M3RefTracking, get_xy_angles

    if refonly is not None:
        acfg, m3_dir = refonly
        best = cm.load_alignment(os.path.join(acfg.OUT_BASE, f'alignment_tpc_veto{veto}', 'alignment.json'))
        return read_windows(cfg, ref_only(cfg, best, m3_dir, tan_max), pad_mm, z_lo, z_hi)

    results = pickle.load(open(os.path.join(cfg.out_dir('cache'), f'event_results_veto{veto}.pkl'), 'rb'))
    best = cm.load_alignment(os.path.join(cfg.OUT_BASE, f'alignment_tpc_veto{veto}', 'alignment.json'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _y, anum = get_xy_angles(rays.ray_data)
    cm.attach_reference_positions(results, rays, best, best.ref_x_sign * np.array(xang), anum)

    events, nref = {}, 0
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > res_cut_mm:
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        nref += 1
        tx, ty = cm._rotate_ref_tangents(r, best)
        if min(abs(tx), abs(ty)) >= tan_max:
            continue
        events[int(r.event_id)] = dict(eid=int(r.event_id), tan_x=float(tx), tan_y=float(ty),
                                       ref_mesh_x=float(r.ref_mesh_x_mm), ref_mesh_y=float(r.ref_mesh_y_mm))
    print(f'[headon] {nref:,} reference events, {len(events):,} with |tan| < {tan_max} in a view', flush=True)
    return read_windows(cfg, events, pad_mm, z_lo, z_hi)


def read_windows(cfg, events, pad_mm, z_lo, z_hi):
    from wft import io as wio
    pos_maps = wio.strip_position_map(cfg)
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        pm = pos_maps[feu]
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            rdr = wio.FeuReader(f)
            want = set(events) & set(rdr.event_ids.tolist())
            for eid, ftst, wfm in rdr.iter_events(want):
                ev = events[eid]
                p0, tn = ev[f'ref_mesh_{plane}'], ev[f'tan_{plane}']
                a, b = p0 + z_lo * tn, p0 + z_hi * tn
                ch = np.where((pm >= min(a, b) - pad_mm) & (pm <= max(a, b) + pad_mm))[0]
                ch = ch[np.argsort(pm[ch])]
                if len(ch) < 4:
                    continue
                ev[plane] = dict(ch=ch.astype(np.int16), pos=pm[ch].astype(np.float32), W=wfm[ch].astype(np.float32),
                                 noise=np.maximum(rdr.noise[ch], 3.0).astype(np.float32))
                ev[f'ftst_{plane}'] = ftst
    events = {k: v for k, v in events.items() if 'x' in v and 'y' in v}
    print(f'[headon] {len(events):,} events with both waveform windows')
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--tan-max', type=float, default=0.06)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--subrun', help='ref-only mode: this sub-run of the key\'s run, through the key\'s alignment')
    ap.add_argument('--base', help='ref-only mode: BASE_PATH holding <RUN>/<subrun>/{decoded_root,m3_tracking_root}')
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(a.run_key)
    name = a.run_key
    if a.subrun:
        import copy
        acfg, cfg = cfg, copy.copy(cfg)
        cfg.SUB_RUN = a.subrun
        if a.base:
            cfg.BASE_PATH = a.base
        ev = build(cfg, a.tan_max, refonly=(acfg, cfg.m3_tracking_dir))
        name = f'{a.run_key}__{a.subrun}'
    else:
        ev = build(cfg, a.tan_max)
    os.makedirs(a.out, exist_ok=True)
    path = f'{a.out}/headon_{name}.pkl'
    pickle.dump(ev, open(path, 'wb'), protocol=4)
    print('wrote', path)


if __name__ == '__main__':
    main()
