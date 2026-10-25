#!/usr/bin/env python3
"""reco_subset.py <run_key> --bundle B --events ids.csv --out events.parquet [--jobs 8]

The production reconstruction (wft.reco.reconstruct_run, as `wft.cli reco`) on a chosen set of events, so a switch
(e.g. WFT_TWO_TRACK_FIT=1) can be compared event by event with the finalize table. Run with the finalize stage's
environment (WFT_PIVOT_START=1 WFT_COL_GUARD=0.01) and X17_BENCH_ROOT pointing at a tree with the same inputs
(decoded_root, combined_hits_root = rr1_cm0, m3_tracking_root_v2).
"""
import argparse, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('--bundle', required=True); ap.add_argument('--events', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    cfg = get_config(a.run_key)
    ids = set(pd.read_csv(a.events).event_id.astype(int))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    reconstruct_run(cfg, CalibrationBundle.load(a.bundle), a.out, event_filter=ids, jobs=a.jobs, bundle_path=a.bundle)


if __name__ == '__main__':
    main()
