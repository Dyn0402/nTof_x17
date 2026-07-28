#!/usr/bin/env python3
"""
wft command line.

    python -m wft.cli bundle  <run_key> [--from-legacy DIR] [--out DIR]
    python -m wft.cli reco    <run_key> [--jobs N] [--limit N] [--matched-only]
    python -m wft.cli info    <table.parquet>

Outputs live under the run's Analysis tree: ``<OUT_BASE>/wft/``.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'mx_june_cosmic_qa'))
sys.path.insert(0, os.path.join(REPO, 'cosmic_bench_analysis'))

from wft.calib import CalibrationBundle       # noqa: E402
from wft.reco import reconstruct_run          # noqa: E402


def _cfg(run_key):
    from qa_config import get_config, setup_paths
    setup_paths()
    return get_config(run_key)


def wft_dir(cfg, *parts):
    return cfg.out_dir('wft', *parts)


def matched_event_ids(cfg):
    """Event ids with an M3 reference ray passing the standard recipe. Used only
    to decide WHICH events to reconstruct — never as a fit input."""
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking, get_xy_angles
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    _xang, _yang, evn = get_xy_angles(rays.ray_data)
    return set(int(e) for e in evn)


def cmd_bundle(args):
    cfg = _cfg(args.run_key)
    legacy = args.from_legacy or os.path.join(cfg.OUT_BASE, 'waveform_first')
    cal = CalibrationBundle.from_legacy(
        legacy, detector=cfg.DET_NAME, run_key=args.run_key,
        conditions=dict(run=cfg.RUN, sub_run=cfg.SUB_RUN))
    out = args.out or wft_dir(cfg, 'calib_bundle')
    cal.save(out, note=f'imported from {legacy}')
    print(cal.summary())
    print('wrote', out)


def cmd_reco(args):
    cfg = _cfg(args.run_key)
    bundle_path = args.bundle or wft_dir(cfg, 'calib_bundle')
    cal = CalibrationBundle.load(bundle_path)
    filt = matched_event_ids(cfg) if args.matched_only else None
    if filt is not None:
        print(f'[wft] {len(filt):,} events have an M3 ray passing the recipe')
    out = args.out or os.path.join(wft_dir(cfg), 'events.parquet')
    reconstruct_run(cfg, cal, out, event_filter=filt, jobs=args.jobs,
                    limit=args.limit, bundle_path=bundle_path)


def cmd_info(args):
    import pandas as pd
    df = pd.read_parquet(args.table)
    print(f'{len(df):,} events')
    for p in ('x', 'y'):
        ok = df[f'{p}_ok']
        print(f'  {p}: fitted {ok.sum():,} ({100*ok.mean():.1f} %), '
              f'slope_reliable {df[f"{p}_slope_reliable"].mean()*100:.1f} %, '
              f'quality_ok {df[f"{p}_quality_ok"].mean()*100:.1f} %, '
              f'median chi2/dof '
              f'{(df[f"{p}_chi2"]/df[f"{p}_dof"].clip(lower=1)).median():.1f}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wft')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('bundle', help='make/import a calibration bundle')
    b.add_argument('run_key')
    b.add_argument('--from-legacy', default=None)
    b.add_argument('--out', default=None)
    b.set_defaults(func=cmd_bundle)

    r = sub.add_parser('reco', help='reconstruct a run into a parquet table')
    r.add_argument('run_key')
    r.add_argument('--bundle', default=None)
    r.add_argument('--out', default=None)
    r.add_argument('--jobs', type=int, default=12)
    r.add_argument('--limit', type=int, default=None)
    r.add_argument('--matched-only', action='store_true',
                   help='only events with an M3 reference ray')
    r.set_defaults(func=cmd_reco)

    i = sub.add_parser('info', help='summarise a reco table')
    i.add_argument('table')
    i.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
