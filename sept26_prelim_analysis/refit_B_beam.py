#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refit_B_beam.py -- fit chamber B's sharing kernel on BEAM data.

**THIS CANNOT RUN YET, AND THE REASON IS THE POINT.**  It is kept as the
statement of what is needed, not as a working tool.

Chamber B is the one chamber with no usable angle scale, and five suspects have
been eliminated: statistics, the scan range, the charge window, the wall
readout order and the geometry (see STATUS.md "Chamber B").  What is left is
that B's individual tracks carry less angle information than A's or D's, and
the leading cause is that B's bundle was *transferred*: its kernel (kY 5.40,
sigma_s 172 ns) was fitted on bench cosmics on det2, and only v, sat_adc and
the sample grid were replaced for the beam.

The obvious next step is to fit B on beam data instead.  It is not available:

    wft.calibrate.build_cache is REF-PINNED.  It selects its training events
    along the M3 reference corridor -- it imports M3RefTracking, requires the
    hits-chain alignment, and uses per-event reference track parameters
    (ref_tan_theta_x, ref_mesh_x_mm) as the truth the model is fitted against.
    The beam has no reference telescope, so there is nothing to pin to and the
    import fails at `from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS`.

This also settles a question about the existing bundles: **all four are bench
transfers.**  Arm C's provenance reads `"fitted": "wft.calibrate"`, but that
describes the parent BENCH fit -- the same record carries
`"transferred": "template + sharing kernel + w0/kw (bench)"` and
`"replaced": "v_drift -> 42.6 ..."`.  No bundle has ever been fitted on beam
data.

WHAT WOULD MAKE IT POSSIBLE.  A ref-free training-event selector.  The beam
does have a constraint the bench does not: the source is a point, 234.6 mm
away, so position and angle are not independent -- which is exactly the
relation `k_arm.py` already exploits to measure the angle scale.  A calibration
could be pinned on the target the same way, fitting the kernel against
tan = (u - foot)/d_perp instead of against a reference ray.  That is a piece of
development, not a re-run, and it is the honest size of "refit B on the beam".

The fit/evaluate harness below is left in place so that work has somewhere to
land; `--evaluate` already works against any candidate bundle that appears.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

CANDIDATE = 'calib_bundle_{arm}beam_candidate'


def fit(arm: str, run: str, subrun: str, jobs: int, events: int, train: int,
        share_mode: str, seed: str | None, use_beam_cache: bool = False) -> str:
    from ntof_tracking.wft_beam import beam_config, BEAM_DETS
    from wft import calibrate as C

    if use_beam_cache:
        from sept26_prelim_analysis import beam_cache as BC
        cfg = beam_config(arm, run, subrun)
        work = cfg.out_dir('wft', 'calib_work')
        cache = os.path.join(work, 'calib_cache.pkl')
        if not os.path.exists(cache):
            BC.build(run, subrun, arm,
                     str(paths.out('fullpass') / run), events, out_path=cache)
        else:
            print(f'reusing training cache {cache}')
        print(f'  hypers this cache can pin: '
              f'{", ".join(BC.hypers_to_fit())}  '
              f'(kY stays transferred -- no y truth)')
    try:
        import qa_config  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            'wft.calibrate.build_cache is ref-pinned: it selects training '
            'events along the M3 reference corridor and needs qa_config, '
            'M3RefTracking and the hits-chain alignment. The beam has no '
            'reference telescope, so this cannot run as written -- see this '
            'module\'s docstring for what a beam calibration would need.')
    cfg = beam_config(arm, run, subrun)
    out = cfg.out_dir('wft', CANDIDATE.format(arm=arm))
    seed = seed or BEAM_DETS[arm]['bundle']
    print(f'fitting arm {arm} on {run}/{subrun}')
    print(f'  seed bundle : {seed}')
    print(f'  out         : {out}')
    print(f'  {events} events, {train} training, {jobs} jobs, '
          f'share_mode={share_mode}')
    # v is FIXED at the bundle prior on purpose. This fit is about the sharing
    # kernel, and letting v float here would confound the two things the whole
    # investigation is trying to separate: the kernel and the angle scale. k is
    # measured afterwards, from the target image, exactly as for every other
    # chamber.
    C.calibrate(cfg, cfg.KEY, n_events=events, n_train=train, jobs=jobs,
                out=out, seed_bundle=seed, share_mode=share_mode,
                v_fixed=42.6)
    return out


def evaluate(arm: str, run: str, subruns, out_dir: str | None = None) -> dict:
    """Compare the candidate against the transferred bundle, on one sample."""
    import numpy as np
    from ntof_tracking.wft_beam import beam_config, BEAM_DETS
    from wft import calib as wcal

    cfg = beam_config(arm, run, subruns[0])
    cand = out_dir or cfg.out_dir('wft', CANDIDATE.format(arm=arm))
    have = os.path.exists(os.path.join(cand, 'bundle.json'))
    print(f'candidate: {cand}  {"" if have else "-- NOT FITTED YET"}')
    if not have:
        return {}

    new = json.load(open(os.path.join(cand, 'bundle.json')))
    old_dir = BEAM_DETS[arm]['bundle']
    old = json.load(open(os.path.join(old_dir, 'bundle.json')))

    print(f'\n{"hyper":>12} {"transferred":>13} {"beam-fitted":>13} {"ratio":>8}')
    for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp'):
        o, n = old['hyper'].get(k), new['hyper'].get(k)
        if o is None or n is None:
            continue
        r = (n / o) if o else float('nan')
        print(f'{k:>12} {o:>13.4f} {n:>13.4f} {r:>8.2f}')
    c1n, c2n = float(new['hyper']['c1']), wcal.effective_c2(new['hyper'])
    gate = 'OK  (c2 < c1)' if c2n < c1n else 'FAILS the c2 < c1 gate'
    print(f'\n  effective c2/c1 = {c2n / c1n:.3f}   {gate}')
    return dict(candidate=cand, hyper_new=new['hyper'], hyper_old=old['hyper'],
                c2_over_c1=c2n / c1n, gate_ok=bool(c2n < c1n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--arm', default='B')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--jobs', type=int, default=3,
                    help='keep this small: the campaign census is usually '
                         'using the machine')
    ap.add_argument('--events', type=int, default=400)
    ap.add_argument('--train', type=int, default=180)
    ap.add_argument('--share-mode', default='delay', choices=('delay', 'lp'))
    ap.add_argument('--seed-bundle', default=None)
    ap.add_argument('--evaluate', action='store_true',
                    help='do not fit; report the existing candidate')
    ap.add_argument('--use-beam-cache', action='store_true',
                    help='build a target-pinned training cache with '
                         'beam_cache.py and place it where calibrate() looks, '
                         'instead of the ref-pinned bench one. This is what '
                         'makes the fit possible at all -- see the docstring '
                         'for what it still cannot constrain.')
    a = ap.parse_args()

    if a.evaluate:
        r = evaluate(a.arm, a.run, [a.subrun])
        return 0 if r else 1
    out = fit(a.arm, a.run, a.subrun, a.jobs, a.events, a.train,
              a.share_mode, a.seed_bundle, use_beam_cache=a.use_beam_cache)
    print(f'\nfitted -> {out}')
    print('NOT installed. Evaluate it:')
    print(f'  python -m sept26_prelim_analysis.refit_B_beam --arm {a.arm} --evaluate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
