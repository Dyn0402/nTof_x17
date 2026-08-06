#!/usr/bin/env python3
"""
build_cache.py — one-time benchmark cache for fast A/B testing of fit variants.

Stores, per M3-matched event, exactly what the production reconstruction
feeds the fitter (candidate windows + seed info + ftst diff), plus the M3
truth in the raw detector frame (reference position at z_mean, rotated
tangents) so variants can be scored without re-running the analysis chain.

The truth is used ONLY for scoring — never as a fit input. The windows are
produced by the same code path as production (`wft.reco._stream_windows`),
so a variant scored here is exactly what production would do.

    ../../.venv/bin/python mx_june_wft/bench/build_cache.py sat_det3
Output: <OUT_BASE>/wft/bench_cache.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import reco as wr                                 # noqa: E402
from wft import seed as wseed                              # noqa: E402
from wft import io as wio                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--truth-only', action='store_true',
                    help='write only the M3 truth table (no waveform windows)')
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    out = args.out or os.path.join(cfg.OUT_BASE, 'wft', 'bench_cache.pkl')
    align_path = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
    params = cm.load_alignment(align_path)

    # ---- truth: reference position (raw frame, z_mean) + rotated tangents
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    ya = np.array(ya)
    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)

    theta = np.deg2rad(params.theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    cx, cy = params.centre_x, params.centre_y
    # inverse of the alignment transform: aligned/M3 frame -> raw strip frame
    u = px - cx - params.x_offset
    v = py - cy - params.y_offset
    ref_x_raw = cx + ct * u + st * v
    ref_y_raw = cy - st * u + ct * v

    tan_by_id = {int(e): (float(ct * np.tan(tx) + st * np.tan(ty)),
                          float(-st * np.tan(tx) + ct * np.tan(ty)))
                 for e, tx, ty in zip(an, xa, ya)}
    truth = {int(e): dict(ref_x=float(rx), ref_y=float(ry),
                          tan_x=tan_by_id.get(int(e), (np.nan, np.nan))[0],
                          tan_y=tan_by_id.get(int(e), (np.nan, np.nan))[1])
             for e, rx, ry in zip(evn, ref_x_raw, ref_y_raw)}
    print(f'{len(truth):,} events with M3 truth')

    if args.truth_only:
        out = args.out or os.path.join(cfg.OUT_BASE, 'wft', 'bench_truth.pkl')
        with open(out, 'wb') as f:
            pickle.dump(dict(meta=dict(run_key=args.run_key,
                                       alignment=align_path,
                                       z_mean=float(params.z_mean),
                                       theta_deg=float(params.theta_deg)),
                             truth=truth), f, protocol=4)
        print(f'wrote {out} (truth only)')
        return

    # ---- production seeding + windows
    pos_maps = wio.strip_position_map(cfg)
    hits = wr._load_hits(cfg)
    seeds = wseed.seeds_from_hits(hits, pos_maps, cfg.MX17_FEU_X, cfg.MX17_FEU_Y)
    del hits
    wanted = set(seeds) & set(truth)
    wanted = {e for e in wanted
              if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    if args.limit:
        wanted = set(sorted(wanted)[:args.limit])
    print(f'{len(wanted):,} events to cache')

    events = {}
    for payloads in wr._stream_windows(cfg, pos_maps, seeds, wanted, pad_strips=3):
        for (eid, wins, sd, n_hits, spark, fd) in payloads:
            sinfo = {p: [dict(n_strips=s.n_strips, n_dropped=s.n_dropped,
                              amp_sum=s.amp_sum) for s in sd.get(p, [])]
                     for p in ('x', 'y')}
            events[eid] = dict(wins=wins, seeds=sinfo, n_hits=n_hits,
                               spark=spark, ftst_diff=fd, truth=truth[eid])

    # active box from the existing production reco (fixed for all variants)
    box = None
    tab = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    if os.path.exists(tab):
        import pandas as pd
        df = pd.read_parquet(tab)
        # box in raw frame from the truth refs of events production reconstructed
        ok = df['x_ok'] & df['y_ok']
        ids = set(df.loc[ok, 'event_id'].astype(int))
        rx = np.array([truth[e]['ref_x'] for e in events if e in ids and e in truth])
        ry = np.array([truth[e]['ref_y'] for e in events if e in ids and e in truth])
        box = dict(x=list(np.percentile(rx, [0.5, 99.5])),
                   y=list(np.percentile(ry, [0.5, 99.5])))

    meta = dict(run_key=args.run_key, alignment=align_path,
                bundle=os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle'),
                z_mean=float(params.z_mean), theta_deg=float(params.theta_deg),
                n_events=len(events), box=box)
    with open(out, 'wb') as f:
        pickle.dump(dict(meta=meta, events=events), f, protocol=4)
    sz = os.path.getsize(out) / 1e6
    print(f'wrote {out} ({len(events):,} events, {sz:.0f} MB)')


if __name__ == '__main__':
    main()
