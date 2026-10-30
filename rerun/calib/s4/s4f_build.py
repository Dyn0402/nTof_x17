#!/usr/bin/env python3
"""s4f_build.py <key> --alignment A --out DIR [--prod-results R --prod-alignment PA] [--rr1-results R1 --rr1-alignment A1]
-- S4 follow-up: the head-on cache of s4_stacks.py (ref-only, through the pass alignment) on whatever decode stage_job
staged (run it with --gen prod for the pre-rerun decode), plus the event lists the pre-pass input used.

The pre-pass inputs (unsat_width.py, 10-16) read headon_cache.build without refonly: the hits chain's events
(has_x and has_y, radial residual < 10 mm, event_results_veto50.pkl) through alignment_tpc_veto50.  S4 reads every M3
ray through the pass alignment.  This writes, next to headon.pkl, the hits-chain selection of the prod chain and of
the rr1 chain, so the fit can take the old selection out of the new cache.

    -> DIR/headon.pkl, DIR/selections.json {name: [eid, ...]}
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_stacks  # noqa: E402  (sets sys.path for cloud_basics and the bench libraries)


def hits_selection(cfg, results_path, alignment_path, tan_max=s4_stacks.TAN_HEADON, res_cut_mm=10.0):
    """headon_cache.build's event selection, without reading any waveform."""
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    import cosmic_micro_tpc_analysis as cm
    from M3RefTracking import M3RefTracking, get_xy_angles
    results = pickle.load(open(results_path, 'rb'))
    best = cm.load_alignment(alignment_path)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _y, anum = get_xy_angles(rays.ray_data)
    cm.attach_reference_positions(results, rays, best, best.ref_x_sign * np.array(xang), anum)
    out = []
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > res_cut_mm:
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, best)
        if min(abs(tx), abs(ty)) < tan_max:
            out.append(int(r.event_id))
    print(f'[s4f] hits selection {results_path}: {len(out)} events', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--alignment', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--prod-results'); ap.add_argument('--prod-alignment')
    ap.add_argument('--rr1-results'); ap.add_argument('--rr1-alignment')
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    import cosmic_micro_tpc_analysis as cm
    import headon_cache as HC
    setup_paths()
    cfg = get_config(a.key)
    os.makedirs(a.out, exist_ok=True)
    sel = {}
    for name, rp, ap_ in (('hits_prod', a.prod_results, a.prod_alignment), ('hits_rr1', a.rr1_results, a.rr1_alignment)):
        if rp:
            sel[name] = hits_selection(cfg, rp, ap_)
    json.dump(sel, open(os.path.join(a.out, 'selections.json'), 'w'))
    best = cm.load_alignment(a.alignment)
    ev = HC.ref_only(cfg, best, cfg.m3_tracking_dir, s4_stacks.TAN_HEADON)
    head = HC.read_windows(cfg, ev, 5.0, -3.0, 33.0)
    pickle.dump(head, open(os.path.join(a.out, 'headon.pkl'), 'wb'), protocol=4)
    print(f'[s4f] {a.key}: {len(head)} head-on events', flush=True)


if __name__ == '__main__':
    main()
