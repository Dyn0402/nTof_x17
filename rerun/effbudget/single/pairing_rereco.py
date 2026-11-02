#!/usr/bin/env python3
"""pairing_rereco.py <run_key> --bench ROOT --bundle B --events ids.txt --out DIR [--jobs N]
-- re-reconstruct a set of events twice, select_tracks mode 'gate' and 'cost' (PAIRING_FIX_2026-11-02.md §3).

The production reconstruction (wft.reco.reconstruct_run, as `wft.cli reco --matched-only` in the pass) on the listed
events, from staged inputs (X17_BENCH_ROOT layout: <ROOT>/det3/<run>/<subrun>/{decoded_root, combined_hits_root,
m3_tracking_root_v2}, the rr1 generation).  The bundle is copied twice next to the outputs, once as it is and once with
xy_pairing['select'] = 'cost', so the switch is exercised exactly as a re-pass would (through the bundle, read by
_worker_init); writes <out>/{gate,cost}/events.parquet (+ .candidates/.splits/.meta).
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bench', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--modes', default='gate,cost')
    a = ap.parse_args()
    os.environ['X17_BENCH_ROOT'] = a.bench
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    cfg = get_config(a.run_key)
    ids = {int(x) for x in open(a.events).read().split()}
    for mode in a.modes.split(','):
        cal = CalibrationBundle.load(a.bundle)
        cal.xy_pairing = dict(cal.xy_pairing, select=mode)
        d = os.path.join(a.out, mode)
        bp = os.path.join(d, 'calib_bundle')
        cal.save(bp, note=f'pairing_rereco.py: xy_pairing select = {mode}')
        reconstruct_run(cfg, CalibrationBundle.load(bp), os.path.join(d, 'events.parquet'), event_filter=ids,
                        jobs=a.jobs, bundle_path=bp)


if __name__ == '__main__':
    main()
