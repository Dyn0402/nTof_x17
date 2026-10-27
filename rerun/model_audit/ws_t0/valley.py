#!/usr/bin/env python3
"""valley.py <det> [n] -- chi2 along the t0 valley for a few held-out events (workstream D figure input).

For each plane: the free fit (p0, w, t0); then chi2 at t0 + d for d in [-240, 240] ns with
  (a) p0 slid along the track, p0 + w d  (the valley: same line in (time, position), the 'mesh' moves along it)
  (b) p0 held                           (what the 1-D curvature error t0_err measures)
the charge profile re-solved (NNLS) at every point, w held.  Output ws_t0/valley_<det>.npz.
"""
import os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
from wft.calib import CalibrationBundle   # noqa: E402
from wft import model as wm, reco as wr   # noqa: E402

M = '/home/dylan/x17/cosmic_bench/model_audit'


def main():
    det = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    C = pickle.load(open(f'{M}/ws_t0/caches/{det}.pkl', 'rb'))
    cal = CalibrationBundle.load(f'{M}/{det}/calib_bundle'); cal.t0_prior_sigma = 0.0
    wm.use_calibration(cal); wm.set_depth_bins(30); wm.COL_GUARD = 0.01; wm.RC_T0_INTERP = True
    wr.PIVOT_START = True
    d = np.arange(-240, 241, 5.0)
    ids = [e for e in sorted(C)[180:] if abs(C[e]['tan_x']) > 0.12 and abs(C[e]['tan_y']) > 0.12][:n]
    rows = []
    for eid in ids:
        for vi, plane in enumerate('xy'):
            P = C[eid][plane]
            f = wr.fit_plane(P, plane, cal)
            if f is None:
                continue
            W, noise, pos, sat = wm.prep_plane(P, plane)
            ca = [wm.chi2_plane(plane, W, noise, pos, sat, f.p0 + f.w * dd, f.w, f.t0 + dd, cal.hyper, snap_t0=False)[0] for dd in d]
            cb = [wm.chi2_plane(plane, W, noise, pos, sat, f.p0, f.w, f.t0 + dd, cal.hyper, snap_t0=False)[0] for dd in d]
            rows.append(dict(eid=eid, view=vi, tan_ref=float(C[eid][f'tan_{plane}']), w=f.w, p0=f.p0, t0=f.t0,
                             ftst=int(C[eid][f'ftst_{plane}']), dof=f.dof, chi_valley=np.array(ca), chi_fixed=np.array(cb)))
            print(eid, plane, f'tan {C[eid][f"tan_{plane}"]:+.2f} t0 {f.t0:.1f} chi2/dof {f.chi2 / f.dof:.1f} '
                  f'valley min at {d[int(np.argmin(ca))]:+.0f}  +-60: {ca[list(d).index(-60)] - min(ca):.0f} / {ca[list(d).index(60)] - min(ca):.0f}')
    np.savez(f'{M}/ws_t0/valley_{det}.npz', d=d, rows=np.array(rows, dtype=object))


if __name__ == '__main__':
    main()
