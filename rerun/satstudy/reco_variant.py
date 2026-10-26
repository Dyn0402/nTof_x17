#!/usr/bin/env python3
"""reco_variant.py --events ids.csv --out DIR/events.parquet [--no-spark-veto] [--min-strips N] [--jobs 2]

The production reconstruction (wft.reco.reconstruct_run, as rerun/outliers/reco_subset.py) on a chosen set of det3
golden-row events with one seeding rule changed, to test whether a budget category is recoverable without the
reference:
  --no-spark-veto   no 50-hit event veto (seeds_from_hits spark_veto=None): the candidate fit decides
  --min-strips N    a cluster needs N strips after the 10 % significance floor instead of 3
Both are default arguments bound at import in wft.seed, so they are passed by wrapping the functions, not by setting
the module constants. Run with the finalize environment (WFT_PIVOT_START=1 WFT_COL_GUARD=0.01) and
X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage (rr1_cm0 hits, decoded_root, M3 v2 of the golden row).
"""
import argparse, functools, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
BUNDLE = os.path.expanduser('~/x17/cosmic_bench/satstudy/bundle_det3_phys')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--no-spark-veto', action='store_true'); ap.add_argument('--min-strips', type=int)
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import seed as wseed
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    if a.no_spark_veto:
        wseed.seeds_from_hits = functools.partial(wseed.seeds_from_hits, spark_veto=None)
    if a.min_strips:
        wseed.seed_candidates = functools.partial(wseed.seed_candidates, min_strips=a.min_strips)
    cfg = get_config('sat_det3')
    ids = set(pd.read_csv(a.events).event_id.astype(int))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    reconstruct_run(cfg, CalibrationBundle.load(BUNDLE), a.out, event_filter=ids, jobs=a.jobs, bundle_path=BUNDLE)


if __name__ == '__main__':
    main()
