#!/usr/bin/env python3
"""merged_rereco.py <run_key> --bench ROOT --bundle B --events ids.txt --out DIR [--variants ...] [--jobs N]
-- re-reconstruct a set of bench events from waveforms under several two-track / pairing variants
(MERGED_PROJECTION_2026-11-02.md).  Same harness as pairing_rereco.py: the production chain (wft.reco.reconstruct_run)
from staged inputs (X17_BENCH_ROOT layout), the variant applied through the bundle exactly as a re-pass would.

Variants (bundle xy_pairing edits):
  p1      the bundle as it is (pass 1: select 'gate', split_corrob 'count')
  cost    select 'cost' (the pairing fix), split_corrob 'count'      -- the bench baseline after PAIRING_FIX
  charge  select 'cost', split_corrob 'charge'                      -- this change
  charge_gate  select 'gate', split_corrob 'charge'
Writes <out>/<variant>/events.parquet (+ .candidates/.splits/.meta).
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
VARIANTS = {'p1': {}, 'cost': dict(select='cost', split_corrob='count'),
            'charge': dict(select='cost', split_corrob='charge'),
            'charge_gate': dict(select='gate', split_corrob='charge')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bench', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--variants', default='p1,cost,charge')
    a = ap.parse_args()
    os.environ['X17_BENCH_ROOT'] = a.bench
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    cfg = get_config(a.run_key)
    ids = {int(x) for x in open(a.events).read().split()}
    for v in a.variants.split(','):
        cal = CalibrationBundle.load(a.bundle)
        cal.xy_pairing = dict(cal.xy_pairing, **VARIANTS[v])
        d = os.path.join(a.out, v)
        bp = os.path.join(d, 'calib_bundle')
        cal.save(bp, note=f'merged_rereco.py: variant {v} {VARIANTS[v]}')
        reconstruct_run(cfg, CalibrationBundle.load(bp), os.path.join(d, 'events.parquet'), event_filter=ids,
                        jobs=a.jobs, bundle_path=bp)


if __name__ == '__main__':
    main()
