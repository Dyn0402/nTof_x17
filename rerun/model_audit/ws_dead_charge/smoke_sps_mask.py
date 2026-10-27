"""Smoke test: every sps_mask mode on the promoted bench det4 bundle."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, '..', '..', '..')), os.path.dirname(HERE)]
from wft.calib import CalibrationBundle   # noqa: E402
import sps_mask                           # noqa: E402
for mode in sps_mask.MODES:
    cal = CalibrationBundle.load(os.path.expanduser('~/x17/cosmic_bench/model_audit/det4/calib_bundle'))
    r = sps_mask.apply(cal, mode)
    print(mode, {p: len(cal.dead[p]) for p in 'xy'}, {k: v for k, v in r.items() if k != 'remap'})
