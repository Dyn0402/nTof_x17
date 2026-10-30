#!/usr/bin/env python3
"""conv_iter.py <run_key> --calib <row>/wft/calib --final <row>/wft/calib_bundle --it 4 --out CALIB_CONVERGENCE_it4.json

The extra calibration iteration (Dylan 10-29: a chamber whose CALIB_CONVERGENCE fails gets one more CAL4 iteration):
RECO<it> made with the final calib_bundle -> ALIGN<it> -> ANG<it> -> W0KW<it> (c<it>/bundle).  Both alignments now
see bow-corrected positions, so r<it-1> vs r<it> is the like-for-like test the first CONV could not make (ALIGN2 had
no bow, ALIGN3 had it: the plane moves by the mean mesh height, 1-2 mm).  Criteria as convergence.py:
  alignment  x/y offset |d| <= 0.1 mm, theta |d| <= 0.05 deg, z |d| <= 1 mm (one grid step)
  w0 / kw    final calib_bundle vs c<it>/bundle: |dw0| <= 0.02 um/ns, |dkw|/kw <= 0.5 %
The final calib_bundle stays the one W0KW3 wrote; c<it>/bundle is the check.
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convergence import align, bundles_2f  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('--calib', required=True); ap.add_argument('--final', required=True)
    ap.add_argument('--it', type=int, default=4); ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from wft.calib import CalibrationBundle
    J = lambda *p: json.load(open(os.path.join(a.calib, *p)))
    groups = {f'align_{a.it - 1}_to_{a.it}': align(J(f'r{a.it - 1}', 'alignment', 'alignment.json'),
                                                   J(f'r{a.it}', 'alignment', 'alignment.json')),
              f'final_to_c{a.it}': bundles_2f(CalibrationBundle.load(a.final),
                                              CalibrationBundle.load(os.path.join(a.calib, f'c{a.it}', 'bundle')))}
    res = dict(run_key=a.run_key, iteration=a.it, converged=all(c['ok'] for g in groups.values() for c in g),
               failed=[c['name'] for g in groups.values() for c in g if not c['ok']], groups=groups,
               rule='extra iteration (Dylan 10-29); like-for-like alignment (both with the bow)')
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"[conv_iter] {a.run_key} it{a.it}: {'CONVERGED' if res['converged'] else 'NOT converged: ' + ', '.join(res['failed'])}")
    for g, cs in groups.items():
        print(f'  {g}: ' + ', '.join(f"{c['name']} {c['value']:+.4g}{'' if c['ok'] else ' FAIL'}" for c in cs))


if __name__ == '__main__':
    main()
