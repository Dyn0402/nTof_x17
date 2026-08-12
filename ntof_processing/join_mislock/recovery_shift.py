#!/usr/bin/env python3
"""Recovery demo: fix the join's bunch offset and run the standard chain.

The wide-span shift scan (2026-08-12) proved run_96/stat090_0001 x 224597
failed because the burst->pulse join locked 26 pulses off (S/N 1273 at
shift +26, zero at shift 0). This applies the constant shift and runs the
UNMODIFIED standard chain: bootstrap, fit_global, fit_perbunch, efficiency.
If the result matches the fleet's ~95-96% at +-25 ns, the data is fully
recovered and the fix belongs in the join, not the fit.

Usage: recovery_shift.py <dream_run> <dream_subrun> <ntof_run> <shift>
           --ntof-source DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ntof_processing.slim_pipeline import config as C           # noqa: E402
from ntof_processing.slim_pipeline import clockfit as cf        # noqa: E402
from ntof_processing.slim_pipeline.slim import (                # noqa: E402
    Segment, _bind_ntof, join_events, bunch_table, pass1_candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('shift', type=int)
    ap.add_argument('--ntof-source', default=None)
    args = ap.parse_args()

    seg = Segment(args.dream_run, args.dream_subrun, args.ntof_run,
                  ntof_source=Path(args.ntof_source) if args.ntof_source
                  else None)
    _bind_ntof(seg)
    ev = join_events(seg)
    btbl, keep = bunch_table(ev)
    if not keep.all():
        ev = ev[keep].reset_index(drop=True)
    phys = ~ev['is_flash'].to_numpy()
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)[phys] + args.shift
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)[phys]
    print(f'applied bunch shift {args.shift:+d}: bunches now '
          f'{ev_b.min()}..{ev_b.max()}')

    cd, offs, thr = pass1_candidates(seg, np.unique(ev_b))
    cb, ct, ca = cd['bunch'], cd['t'], cd['arm']
    print(f'{ev_t.size:,} events, {ct.size:,} candidates')

    K, T0, arm_off, gi = cf.fit_global(ev_b, ev_t, cb, ct, ca)
    ci, cv, pb = cf.fit_perbunch(ev_b, ev_t, cb, ct, ca, K, T0, arm_off)
    qa = cf.efficiency(ev_b, ev_t, cb, ct, ca, K, T0, arm_off, ci,
                       C.ACCEPT_NS)
    qacv = cf.efficiency(ev_b, ev_t, cb, ct, ca, K, T0, arm_off, cv,
                         C.ACCEPT_NS)
    boot = gi.get('bootstrap', {})
    print(f'RECOVERED: bootstrap S/N {boot.get("snr", float("nan")):.0f}, '
          f'K={K:.6e} T0={T0:+.2f} ns, arm {np.round(arm_off,2)}')
    print(f'efficiency {qa["efficiency"]:.4%} (cv {qacv["efficiency"]:.4%}) '
          f'accidental {qa["accidental"]:.4%} purity {qa["purity"]:.4%}')
    json.dump(dict(shift=args.shift, K=K, T0_ns=T0,
                   arm_off=list(arm_off), boot_snr=boot.get('snr'),
                   eff=qa['efficiency'], eff_cv=qacv['efficiency'],
                   accidental=qa['accidental']),
              open(f'recovery_shift_{args.dream_run}_{args.dream_subrun}_'
                   f'{args.ntof_run}.json', 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
