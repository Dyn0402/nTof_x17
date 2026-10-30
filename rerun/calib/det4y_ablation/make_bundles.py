#!/usr/bin/env python3
"""make_bundles.py <fe bundle> <pass c2 bundle> <out dir> -- the det4 Y ablation's bundles (DET4Y_ABLATION_2026-10-30.md).

    bundle_a  the audit's calib_bundle_fe as is
    bundle_b  bundle_a with the pass's templates (tmpl_x, tmpl_y of Analysis_pass1 .../wft/calib/c2/bundle)
    bundle_d  the pass's c2/bundle (the bundle the G1 reconstruction calib/r3 ran with)

The bow arms (c = a + pass bow, e = b + pass bow) are made on condor by rerun/w12/set_w0kw.py --bow-json, from the
reconstruction of a (resp. b), as the calib phase's W0KW2b node does.
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from wft.calib import CalibrationBundle  # noqa: E402


def main():
    fe_dir, c2_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    for name, src in (('bundle_a', fe_dir), ('bundle_d', c2_dir)):
        shutil.copytree(src, os.path.join(out, name), dirs_exist_ok=True)
    fe, c2 = CalibrationBundle.load(fe_dir), CalibrationBundle.load(c2_dir)
    for p in 'xy':
        a, b = fe.tmpl[p], c2.tmpl[p]
        print(f'tmpl_{p}: rms diff {np.sqrt(np.mean((a - b) ** 2)) / a.max():.4f} of peak, max {np.abs(a - b).max() / a.max():.4f}')
    fe.tmpl = {p: c2.tmpl[p].copy() for p in 'xy'}
    fe.provenance = dict(fe.provenance, ablation='det4 Y ablation arm b: audit calib_bundle_fe + the pass c2/bundle '
                                                 'templates (tmpl_x, tmpl_y)')
    fe.save(os.path.join(out, 'bundle_b'))


if __name__ == '__main__':
    main()
