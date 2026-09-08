#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixed_point.py -- the acceptance test for a target-pinned calibration.

WHY A CALIBRATION FROM ``beam_cache`` NEEDS ONE.  Its truth is
``tan = (u - foot_x)/d_perp``, and ``u`` comes from the existing
reconstruction's ``p0`` -- produced with the very bundle the fit is replacing.
That is a loop, and the honest way to close it is not to argue that the loop is
weak but to show that it converges:

    fit hypers on the cache  ->  re-reconstruct the training events with the
    new hypers  ->  rebuild the truth from the new positions  ->  fit again

If the hypers stop moving, the answer is a property of the data rather than of
the bundle you started from.  If they keep moving, the loop is driving them and
the result must not be installed.  ``k_arm`` ran exactly this argument for the
angle scale and passed: re-deriving the sample with k applied moved 8-21 % of
the rows and changed k by less than a grid step.

WHAT "STOPPED MOVING" MEANS HERE.  Per hyper, the fractional change between
successive iterations, and the test is on the WORST one -- an average would let
a single runaway parameter hide behind six stable ones.  ``TOL`` is 5 %, chosen
against the spread the calibration already tolerates elsewhere: ``k_arm``
certifies a chamber when its three independent estimators agree to 25 %, so a
self-consistency loop that is still moving by more than 5 % per iteration is
contributing a systematic comparable to the measurement itself.

DIVERGENCE IS A RESULT, NOT A FAILURE.  If the iteration does not converge that
says the target constraint cannot pin this chamber's kernel, which is worth
knowing and is reported as such rather than retried with different settings.

    python -m sept26_prelim_analysis.fixed_point --arm B --iters 3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

#: Worst-hyper fractional change between iterations that counts as converged.
TOL = 0.05
#: Hypers the test watches. kY is excluded because beam_cache cannot pin it --
#: it stays at its transferred value, so it cannot move and would dilute a
#: worst-of test with a guaranteed zero.
WATCH = ('c1', 'c2', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')


def hyper_delta(a: dict, b: dict) -> dict:
    """Fractional change per watched hyper, and the worst of them."""
    out = {}
    for k in WATCH:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        out[k] = abs(y - x) / abs(x) if x else (0.0 if not y else float('inf'))
    out['_worst'] = max((v for k, v in out.items() if not k.startswith('_')),
                        default=float('nan'))
    return out


def reconstruct_training(arm: str, run: str, subrun: str, bundle: str,
                         event_ids, jobs: int = 4):
    """Re-fit just the training events with a candidate bundle.

    Only the training events, not the sub-run: the loop asks whether THESE
    positions move, and reconstructing 50 000 events to answer that would cost
    hours per iteration for no extra information.
    """
    from ntof_tracking import wft_beam as WB
    cfg = WB.beam_config(arm, run, subrun)
    out = os.path.join(paths.out('kcal', 'fixed_point'),
                       f'{arm}_iter_events.parquet')
    WB.reconstruct_subrun(cfg, bundle, out, jobs=jobs,
                          allow_events={t: set(int(e) for e in event_ids)
                                        for t in WB.subrun_tags(cfg)})
    import pandas as pd
    return pd.read_parquet(out)


def run(arm: str, run_: str, subrun: str, iters: int, jobs: int,
        events: int, train: int) -> dict:
    from sept26_prelim_analysis import beam_cache as BC
    from sept26_prelim_analysis import refit_B_beam as RB
    from ntof_tracking.wft_beam import beam_config

    cfg = beam_config(arm, run_, subrun)
    work = cfg.out_dir('wft', 'calib_work')
    cache = os.path.join(work, 'calib_cache.pkl')
    merged = str(paths.out('fullpass') / run_)

    hist, deltas = [], []
    for it in range(iters):
        print(f'\n===== iteration {it}')
        # Each iteration must rebuild the cache from the CURRENT positions;
        # reusing it would make the loop a no-op and pass trivially.
        if os.path.exists(cache):
            os.remove(cache)
        BC.build(run_, subrun, arm, merged, events, out_path=cache)
        out = RB.fit(arm, run_, subrun, jobs, events, train,
                     'delay', None, use_beam_cache=True)
        h = json.load(open(os.path.join(out, 'bundle.json')))['hyper']
        hist.append({k: h.get(k) for k in WATCH})
        print('  ' + '  '.join(f'{k}={h.get(k):.4g}' for k in WATCH
                               if h.get(k) is not None))
        if it:
            d = hyper_delta(hist[-2], hist[-1])
            deltas.append(d)
            print(f'  worst change vs previous: {100 * d["_worst"]:.1f} %  '
                  f'({"CONVERGED" if d["_worst"] < TOL else "still moving"})')
            if d['_worst'] < TOL:
                break
        # feed the new bundle back in: re-reconstruct, so the next cache is
        # built from positions this bundle produced
        iter_dir = os.path.join(paths.out('kcal', 'fixed_point'),
                                f'{arm}_iter{it}')
        os.makedirs(iter_dir, exist_ok=True)
        shutil.copytree(out, os.path.join(iter_dir, 'bundle'),
                        dirs_exist_ok=True)

    ok = bool(deltas) and deltas[-1]['_worst'] < TOL
    res = dict(arm=arm, run=run_, subrun=subrun, iterations=len(hist),
               tol=TOL, hypers=hist, deltas=deltas, converged=ok,
               verdict=('FIXED POINT -- the result is a property of the data'
                        if ok else
                        'NOT CONVERGED -- the loop is driving the hypers; do '
                        'not install this bundle'))
    p = paths.out('kcal', 'fixed_point') / f'fixed_point_{arm}.json'
    json.dump(res, open(p, 'w'), indent=1, default=float)
    print(f'\n{res["verdict"]}\nwrote {p}')
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arm', default='B')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--events', type=int, default=400)
    ap.add_argument('--train', type=int, default=180)
    a = ap.parse_args()
    r = run(a.arm, a.run, a.subrun, a.iters, a.jobs, a.events, a.train)
    return 0 if r['converged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
