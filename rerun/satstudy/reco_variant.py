#!/usr/bin/env python3
"""reco_variant.py --events ids.csv --out DIR/events.parquet [--no-spark-veto] [--min-strips N] [--jobs 2]

The production reconstruction (wft.reco.reconstruct_run, as rerun/outliers/reco_subset.py) on a chosen set of det3
golden-row events with one seeding rule changed, to test whether a budget category is recoverable without the
reference:
  --no-spark-veto   no 50-hit event veto (seeds_from_hits spark_veto=None): the candidate fit decides
  --min-strips N    a cluster needs N strips after the 10 % significance floor instead of 3
  --drop-flat       drop every hit that is above threshold for the whole window (trunc_left AND trunc_right) before
                    seeding: a particle's signal lasts at most a drift column, so such a strip is a baseline offset
                    (the "noise bands", bands.py), not charge. It then neither clusters nor counts for the spark veto
  --headon-sig T    adaptive head-on rule: MIN_STRIPS stays 3, but a 2-strip cluster is also admitted when its weaker
                    strip has hit significance >= T. A head-on track puts its whole drift column on 1-2 strips, so its
                    third strip falls under the plane-relative 10 % floor although it is far above noise
Both are default arguments bound at import in wft.seed, so they are passed by wrapping the functions, not by setting
the module constants. Run with the finalize environment (WFT_PIVOT_START=1 WFT_COL_GUARD=0.01) and
X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage (rr1_cm0 hits, decoded_root, M3 v2 of the golden row).
"""
import argparse, functools, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
BUNDLE = os.path.expanduser('~/x17/cosmic_bench/satstudy/bundle_det3_phys')


def headon(wseed, thr):
    """Seed on significance instead of ADC amplitude (Seed.amp_sum is not used downstream) and admit a 2-strip
    cluster only when both strips have significance >= thr."""
    orig_sc, orig_sfh = wseed.seed_candidates, wseed.seeds_from_hits

    def sc(pos, channels, amps, *args, **kw):
        kw['min_strips'] = 2
        sig = dict(zip(np.asarray(channels, int).tolist(), np.asarray(amps, float).tolist()))
        return [s for s in orig_sc(pos, channels, amps, *args, **kw)
                if s.n_strips >= wseed.MIN_STRIPS or min(sig.get(int(c), 0.0) for c in s.channels) >= thr]

    def sfh(df, *args, **kw):
        df = df.assign(amplitude=df['significance'])
        return orig_sfh(df, *args, **kw)
    wseed.seed_candidates, wseed.seeds_from_hits = sc, sfh


def drop_flat():
    import uproot
    from wft import reco as wr

    def load(cfg):
        files = [f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f]
        df = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in files],
                                expressions=['eventId', 'feu', 'channel', 'amplitude', 'significance',
                                             'trunc_left', 'trunc_right'], library='pd')
        df = df[df['feu'].isin(cfg.MX17_FEUS)]
        df = df[~(df.trunc_left & df.trunc_right)]
        return df[['eventId', 'feu', 'channel', 'amplitude', 'significance']]
    wr._load_hits = load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--no-spark-veto', action='store_true'); ap.add_argument('--min-strips', type=int)
    ap.add_argument('--headon-sig', type=float)
    ap.add_argument('--drop-flat', action='store_true')
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
    if a.headon_sig:
        headon(wseed, a.headon_sig)
    if a.drop_flat:
        drop_flat()
    cfg = get_config('sat_det3')
    ids = set(pd.read_csv(a.events).event_id.astype(int))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    reconstruct_run(cfg, CalibrationBundle.load(BUNDLE), a.out, event_filter=ids, jobs=a.jobs, bundle_path=BUNDLE)


if __name__ == '__main__':
    main()
