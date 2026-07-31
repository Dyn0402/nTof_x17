#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_candidates.py -- reconstruct the n_TOF sector SINGLES for a whole sub-run
and cache them, so every downstream question is cheap.

One pass over the wall and plastic trees of the candidate processing rebuilds the
N1081B chain per arm (top+bottom analogue sum over the wall threshold, ANDed with
a plastic bar over its threshold inside the 20 ns logic pulse), for both legs:

    leg 'wp'   wall AND plastic   -- the hardware SINGLES that triggered DREAM
    leg 'w'    wall only          -- the same without the plastic requirement,
                                     which is what isolates the plastic leg's cost

The result is a few hundred MB of flat arrays, and after that the window scan,
the purity estimate and the per-arm breakdown are all seconds rather than hours.

Bunches are processed in chunks because a whole sub-run of wall hits does not
want to be in memory at once (~25 M hits per arm over 2061 bunches).

USAGE
    python build_candidates.py [sub-run] [--nb N] [--chunk 250]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from study_common import (use_variant, dream_events, DATA, NTOF_RUN, DREAM_RUN,
                          SUBRUNS)

use_variant()

from ntof_dream_merge import dream_trigger as dt          # noqa: E402
from ntof_dream_merge import fast_singles as fs           # noqa: E402

# The top/bottom cable offsets are instrumental constants, so they are measured
# once on a sample rather than per chunk. OFFSET_BUNCHES is generous: the
# estimator is a modal value over every late hit of those bunches, ~1e5 pairs.
OFFSET_BUNCHES = 150

KEEP = dict(bunch=np.int32, t=np.float64, wall_mv=np.float32, seg=np.int8,
            pss_dt=np.float32, pss_mv=np.float32, arm=np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('sub', nargs='?', default='stat090_0000', choices=SUBRUNS)
    ap.add_argument('--nb', type=int, default=None,
                    help='use only the first N bunches (default: all)')
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--repair-tflash', action='store_true',
                    help="use the laptop-side tflash repair instead of the "
                         "candidate file's own stored tflash")
    ap.add_argument('--tag', default='',
                    help='suffix for the output files, to keep variants apart')
    args = ap.parse_args()
    fs.REPAIR_TFLASH = args.repair_tflash
    tag = args.tag or ('_rep' if args.repair_tflash else '')
    base = ('laptop tflash repair ON' if args.repair_tflash
            else "the candidate file's own stored tflash (repair OFF)")
    print(f'time base: {base}')

    ev, bunches = dream_events(args.sub, nb=args.nb)
    thr, adc = dt.load_thresholds(DREAM_RUN, args.sub), dt.load_adc_mv()
    print(f'{args.sub}: {len(bunches)} bunches ({bunches.min()}-{bunches.max()}), '
          f'{len(ev):,} non-flash DREAM events')
    print(f'thresholds (polled {thr["polled_at"]}): '
          + '  '.join(f'{a} wall {thr["wall"][a]:.0f} / pss {thr["plastic"][a]:.0f} mV'
                      for a in dt.ARMS))

    t0 = time.time()
    offs = {a: fs.measure_tb_offsets(NTOF_RUN, bunches[:OFFSET_BUNCHES], a)
            for a in dt.ARMS}
    print(f'\ntop/bottom cable offsets (ns), measured on {OFFSET_BUNCHES} bunches '
          f'in {time.time() - t0:.0f} s:')
    for a in dt.ARMS:
        print(f'  wall {a}: ' + '  '.join(f'{offs[a][g]:+6.1f}' for g in range(4)))

    chunks = [bunches[i:i + args.chunk] for i in range(0, len(bunches), args.chunk)]
    for leg, req in (('wp', True), ('w', False)):
        acc, t0 = [], time.time()
        for i, ch in enumerate(chunks):
            acc.append(fs.all_arms(NTOF_RUN, ch, thr, adc, offsets=offs,
                                   require_plastic=req))
            n = sum(a['t'].size for a in acc)
            print(f'  [{leg}] chunk {i + 1}/{len(chunks)} '
                  f'(bunches {ch[0]}-{ch[-1]}): {n:,} candidates, '
                  f'{time.time() - t0:5.0f} s', flush=True)
        out = {k: np.concatenate([a[k] for a in acc]).astype(v)
               for k, v in KEEP.items()}
        o = np.lexsort((out['t'], out['bunch']))
        out = {k: v[o] for k, v in out.items()}
        out['bunches'] = bunches.astype(np.int32)
        p = DATA / f'cand_{args.sub}_{leg}{tag}.npz'
        np.savez_compressed(p, **out)
        print(f'  [{leg}] {out["t"].size:,} candidates '
              f'({out["t"].size / len(bunches):.0f}/bunch) -> {p.name} '
              f'[{time.time() - t0:.0f} s]')

    np.savez(DATA / f'events_{args.sub}{tag}.npz',
             eventId=ev['eventId'].to_numpy().astype(np.int64),
             bunch=ev['BunchNumber'].to_numpy().astype(np.int32),
             t=ev['t_since_flash_ns'].to_numpy().astype(np.float64),
             bunches=bunches.astype(np.int32),
             tb_offsets=np.array([[offs[a][g] for g in range(4)] for a in dt.ARMS]))
    print(f'events -> events_{args.sub}{tag}.npz')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
