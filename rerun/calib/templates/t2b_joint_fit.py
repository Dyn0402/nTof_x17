"""SUPERSEDED by t4_closure.py (2026-10-28): the box width here is rounded to whole bins, so the fit has no gradient in L and stalls at its start values. Kept for the record; do not use its output.

T2b: one physical impulse response h fitted to every |tan| bin's measured template at once.

template_b(t) = unit-peak, t50-aligned [ h (x) box(L / (v |tan_b|)) ](t),  h = DREAM(t_peak, beta) (x) [(1-f) delta + f rect(T)]
(box over each bin's own |tan| distribution).  Free: t_peak, beta, f, T, L.  The fitted h is the response with the
strip's drift segment removed; f, T say how much ion current the data allow.  Usage: python -I t2b_joint_fit.py [det]"""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_tan_amp import CACHE, TAN_BINS, pulses, GRID, HERE
from t2_shaper_model import response
V = dict(det2=39.3, det3=35.8, det4=33.0, det6=24.2, det7=36.8)
FIT = (GRID >= -300) & (GRID <= 1300)


def align(t, h):
    h = h / h.max(); i = np.argmax(h >= 0.5)
    t50 = t[i - 1] + (0.5 - h[i - 1]) / (h[i] - h[i - 1]) * (t[i] - t[i - 1])
    return np.interp(GRID, t - t50, h, left=0.0, right=np.nan)


def model_bins(p, tans_per_bin, v):
    tp, beta, f, T, L = p
    t, h = response(tp, beta, max(T, 1.0), f, dt=2.0, tmax=4000.0)
    out = []
    for tans in tans_per_bin:
        acc = np.zeros_like(h)
        for tn in tans:                      # mixture of boxes over the bin's own |tan|
            w = max(int(round(L * 1000 / (v * tn) / 2.0)), 1)
            acc += np.convolve(h, np.ones(w) / w)[:len(h)]
        out.append(align(t, acc))
    return out


def main(det):
    ev = pickle.load(open(CACHE.format(det), 'rb')); v = V[det]; res = {}
    for plane in 'xy':
        P = pulses(ev, plane)
        tn = np.array([q[0] for q in P]); S = np.array([q[2] for q in P])
        data, tans = [], []
        for lo, hi in TAN_BINS[2:]:             # >= 0.15: below that the column, not the segment, fills the strip
            m = (tn >= lo) & (tn < hi)
            if m.sum() < 25:
                continue
            d = np.nanmedian(S[m], axis=0); d -= np.nanmedian(d[GRID < -250]); d /= np.nanmax(d)
            data.append(d); tans.append(np.quantile(tn[m], np.linspace(0.05, 0.95, 7)))

        def resid(p):
            M = model_bins(p, tans, v)
            return np.concatenate([np.nan_to_num((m - d)[FIT]) for m, d in zip(M, data)])
        best = None
        for f0, T0 in ((0.9, 340.0), (0.3, 340.0), (0.3, 150.0)):
            r = least_squares(resid, [170.0, 0.3, f0, T0, 1.0],
                              bounds=([80, 0, 0, 20, 0.2], [300, 1, 1, 800, 4]), x_scale=[20, .2, .2, 100, .3])
            if best is None or r.cost < best.cost:
                best = r
        p = best.x; rms = float(np.sqrt(np.mean(best.fun ** 2)))
        # the pinned sim physics (f 0.908, T 340), only t_peak, beta, L free
        rs = least_squares(lambda q: resid([q[0], q[1], 0.908, 340.0, q[2]]), [170, .3, 1.0],
                           bounds=([80, 0, .2], [300, 1, 4]), x_scale=[20, .2, .3])
        rms_sim = float(np.sqrt(np.mean(rs.fun ** 2)))
        res[plane] = dict(t_peak=p[0], beta=p[1], f_ion=p[2], T_ion=p[3], L_mm=p[4], rms=rms, n_bins=len(data),
                          sim_pinned=dict(t_peak=rs.x[0], beta=rs.x[1], L_mm=rs.x[2], rms=rms_sim))
        print(f'{det} {plane}: free  t_peak {p[0]:5.1f} beta {p[1]:.2f} f_ion {p[2]:.2f} T {p[3]:4.0f} ns L {p[4]:.2f} mm '
              f'rms {rms:.4f} ({len(data)} bins)\n         sim f=0.908 T=340: t_peak {rs.x[0]:5.1f} beta {rs.x[1]:.2f} '
              f'L {rs.x[2]:.2f} rms {rms_sim:.4f}')
    json.dump(res, open(f'{HERE}/results/t2b_{det}.json', 'w'), indent=1)


if __name__ == '__main__':
    for d in sys.argv[1:] or list(V):
        main(d)
