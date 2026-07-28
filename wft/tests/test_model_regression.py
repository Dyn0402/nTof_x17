#!/usr/bin/env python3
"""
Regression test: wft.model must reproduce the R&D forward_model2/3 numerics
exactly, on real det3 events.

The R&D code (mx_june_cosmic_qa/waveform_first_threading/forward_model2.py,
forward_model3.py) is what every number in WAVEFORM_FIRST_THREADING.md was
produced with. Packaging it must not change a digit.

    ../../.venv/bin/python wft/tests/test_model_regression.py [n_events]
"""
import os
import sys
import pickle

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RND = os.path.join(REPO, 'mx_june_cosmic_qa', 'waveform_first_threading')
sys.path.insert(0, REPO)
sys.path.insert(0, RND)

from wft.calib import CalibrationBundle              # noqa: E402
from wft import model as wm                          # noqa: E402
import forward_model2 as fm2                         # noqa: E402
import forward_model3 as fm3                         # noqa: E402

WF_DIR = fm2.BASE


def main(n_events=25):
    cal = CalibrationBundle.from_legacy(WF_DIR, detector='mx17_3',
                                        run_key='sat_det3')
    wm.use_calibration(cal)
    hyper = dict(cal.hyper)

    d = pickle.load(open(os.path.join(WF_DIR, 'wfcache.pkl'), 'rb'))
    events = d['events']
    eids = sorted(events)[:n_events]

    worst = dict(chi2=0.0, p0=0.0, w=0.0, t0=0.0, matrix=0.0, frac=0.0)
    n = 0
    for eid in eids:
        ev = events[eid]
        for plane in ('x', 'y'):
            P = ev[plane]
            tan_ref = ev[f'tan_{plane}']
            p0_ref = ev[f'ref_mesh_{plane}']

            # --- design matrix and chi2 at a fixed point
            pos = np.asarray(P['pos'], float)
            M_ref = fm3.build_matrix_fast(plane, pos, p0_ref, tan_ref * 0.0366,
                                          400.0, hyper)
            M_new = wm.build_matrix(plane, pos, p0_ref, tan_ref * 0.0366,
                                    400.0, hyper)
            worst['matrix'] = max(worst['matrix'],
                                  float(np.max(np.abs(M_ref - M_new))))

            F_ref = fm3.strip_fractions_vec(pos, p0_ref, 0.005,
                                            hyper['sigma_p0'], hyper['Dp'])
            F_new = wm.strip_fractions(pos, p0_ref, 0.005,
                                       hyper['sigma_p0'], hyper['Dp'])
            worst['frac'] = max(worst['frac'],
                                float(np.max(np.abs(F_ref - F_new))))

            # --- full fit, same seeds
            g_ref = fm2.init_guess(P, plane, tan_ref, p0_ref, cal.v_drift * 1e-3)
            g_new = wm.init_guess(P, plane, tan_ref, p0_ref, cal.v_drift)
            assert np.allclose(g_ref, g_new), (g_ref, g_new)

            r_ref = fm3.fit_plane(P, plane, *g_ref, hyper=hyper)
            r_new = wm.fit_plane_raw(P, plane, *g_new, hyper=hyper)
            for k in ('chi2', 'p0', 'w', 't0'):
                worst[k] = max(worst[k], abs(r_ref[k] - r_new[k]))
            n += 1

    print(f'{n} plane-fits compared against forward_model3')
    for k, v in worst.items():
        print(f'  max |delta {k}| = {v:.3e}')
    ok = (worst['matrix'] < 1e-12 and worst['frac'] < 1e-12 and
          worst['chi2'] < 1e-6 and worst['p0'] < 1e-9 and
          worst['w'] < 1e-12 and worst['t0'] < 1e-9)
    print('REGRESSION', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 25))
