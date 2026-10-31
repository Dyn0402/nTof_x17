#!/usr/bin/env python3
"""bench_ycache.py <key> --alignment A --out DIR -- bench events inclined in the Y view and flat in X, for bench_gap.py --view y.

det4's dead amplification stripes are x intervals, so an inclined X column crosses them and reads short; a track
inclined in y-z at one x (|tan_x| < TANX_MAX) keeps its whole Y column live (or dead) together.  S4's inclined cache
(rerun/calib/s4/s4_stacks.py build) was selected on |tan_x| and holds ~85 such events on g_det4; this builds them on
purpose, the same way: M3 rays through the PASS alignment (headon_cache.ref_only), pass-1 decode (headon_cache.read_windows),
windows z in (-8, 40) mm along the track.  Run through rerun/stage_job.py (it stages the row's inputs); -> DIR/ycache.pkl.
"""
import argparse
import os
import pickle
import sys

#: the code tree: this file's repo, or the frozen archive the job runs on (ycache_job.sh exports RERUN_CODE_DIR)
REPO = os.environ.get('RERUN_CODE_DIR') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics'), REPO,
                os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

TANY = (0.15, 0.65)
TANX_MAX = 0.06


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--alignment', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    import cosmic_micro_tpc_analysis as cm
    setup_paths()
    import headon_cache as HC
    cfg = get_config(a.key)
    best = cm.load_alignment(a.alignment)
    sel = lambda tx, ty: TANY[0] <= abs(ty) <= TANY[1] and abs(tx) < TANX_MAX
    ev = HC.ref_only(cfg, best, cfg.m3_tracking_dir, None, select=sel)
    ev = HC.read_windows(cfg, ev, 5.0, -8.0, 40.0)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump(ev, open(os.path.join(a.out, 'ycache.pkl'), 'wb'), protocol=4)
    print(f'[ycache] {a.key}: {len(ev)} Y-inclined, X-flat events', flush=True)


if __name__ == '__main__':
    main()
