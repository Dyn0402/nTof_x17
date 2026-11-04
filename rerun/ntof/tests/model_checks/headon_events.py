#!/usr/bin/env python3
"""headon_events.py -- candidate head-on X planes on n_TOF, from the seeding only (no fit), to keep the reco small.

    headon_events.py --run 145 --subrun stat090_0000 --arm A --bundle DIR --out ev.txt [--tags N] [--max-strips 6]

The pass seeder (ntof_tracking.wft_beam.seeds_from_hits_beam, min strips 3, the bundle's dead/hot lists) runs on the
hits of each acquisition; an event is listed when its X plane offers exactly one candidate cluster of 3..max-strips
strips (a near-normal column puts its charge on 3-5 strips; an inclined one on more).  Only 'which events to fit'
comes from here: the head-on selection proper is made on the fitted tan afterwards (template_resid.py, dp_stack.py).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'rerun', 'ntof'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tags', type=int, default=None)
    ap.add_argument('--max-strips', type=int, default=6)
    ap.add_argument('--base', default='/media/ucla/work/ntof_pass1/runs/')
    ap.add_argument('--hits-dir', default='combined_hits_root')
    a = ap.parse_args()
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle
    from ntof_reco import FEUS, MIN_STRIPS
    cal = CalibrationBundle.load(a.bundle)
    fx, fy = FEUS[a.arm]
    cfg = wb.BeamConfig(KEY=f'run_{a.run}_{a.subrun}_mx17_{a.arm}', RUN=f'run_{a.run}', SUB_RUN=a.subrun,
                        DET_NAME=f'mx17_{a.arm}', MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=a.base,
                        OUT_BASE='/tmp', HITS_DIR=a.hits_dir)
    pos_maps = wio.strip_position_map(cfg)
    tags = wb.subrun_tags(cfg)[:a.tags] if a.tags else wb.subrun_tags(cfg)
    ids, n_seeded = [], 0
    for tag in tags:
        hp = wb.hits_file_for_tag(cfg, tag)
        if hp is None:
            continue
        seeds = wb.seeds_from_hits_beam(wb.read_hits_tag(hp, (fx, fy)), pos_maps, fx, fy, hot=cal.hot,
                                        min_strips=MIN_STRIPS, dead=cal.dead)
        n_seeded += len(seeds)
        sel = [e for e, r in seeds.items() if len(r['x']) == 1 and r['x'][0].n_strips <= a.max_strips]
        ids += sel
        print(f'{tag}: {len(seeds)} seeded, {len(sel)} single narrow X cluster', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write('\n'.join(map(str, sorted(set(ids)))) + '\n')
    print(f'{len(set(ids))} of {n_seeded} seeded events -> {a.out}')


if __name__ == '__main__':
    main()
