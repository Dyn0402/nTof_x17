#!/usr/bin/env python3
"""t0i_cost.py <cache.pkl> <bundle> [n] -- wall time and t0_err definedness of fit_plane with and without the RC t0
interpolation (F2).  Runs both arms in this one process, interleaved per event, so machine load cancels.
"""
import os, pickle, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
from wft.calib import CalibrationBundle   # noqa: E402
from wft import model as wm, reco as wr   # noqa: E402


def main():
    cache, bundle = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    C = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    cal.t0_prior_sigma = 0.0
    wm.use_calibration(cal); wm.set_depth_bins(24); wm.COL_GUARD = 0.01
    wr.PIVOT_START = True
    T = {0: 0.0, 1: 0.0}; nan = {0: 0, 1: 0}; nf = {0: 0, 1: 0}; nfev = {0: [], 1: []}
    for eid in sorted(C)[180:180 + n]:
        for plane in 'xy':
            P = C[eid].get(plane)
            if P is None:
                continue
            for arm in (0, 1):
                wm.RC_T0_INTERP = bool(arm)
                t = time.perf_counter()
                f = wr.fit_plane(P, plane, cal)
                T[arm] += time.perf_counter() - t
                if f is not None:
                    nf[arm] += 1
                    nan[arm] += int(not np.isfinite(f.t0_err))
    for arm in (0, 1):
        print(f'RC_T0_INTERP={arm}: {nf[arm]} plane fits, {T[arm]:.1f} s ({T[arm] / max(nf[arm], 1):.2f} s/fit), '
              f't0_err NaN {nan[arm]} ({100 * nan[arm] / max(nf[arm], 1):.1f} %)')


if __name__ == '__main__':
    main()
