#!/usr/bin/env python3
"""set_w0kw.py <run_key> --bundle <src bundle> --angles <03_angles out dir> --out <bundle dir>   (W6)

Measure the angle-mapping constants of a bundle from a reconstruction made WITH that bundle, and write a copy of the
bundle carrying them.  Same estimator as mx_june_wft/bench/set_w0.py, but read from 03_angles' per-event table
(fitted w and the M3 angle through the waveform alignment):

    w0 = median(w * 1e3 - v * tan_ref)                     over |tan_ref| < 0.30
    kw = median((w * 1e3 - w0) / (v * tan_ref))            over 0.10 < |tan_ref| < 0.40

Under a kernel calibrated with the right v and reference, w0 ~ 0 and kw ~ 1; a large kw is a calibration failure,
not a constant (RERUN_PLAN W6).  The w0/kw an RC bundle inherits from its r06 source bundle are another kernel's.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--angles', default=None, help='03_angles output dir (required unless --pairing-only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-depth-bins', type=int, default=None,
                    help="write the bundle with this depth grid K (vrefit --keep-v: the group's registry K, W2)")
    ap.add_argument('--table', default=None,
                    help='the reconstruction table (events.parquet) made with --bundle: also re-measure dt_xy (median '
                         't0x - t0y per ftst class on clean single tracks) and the xy_pairing block (lq, u50, u90, t0 '
                         'medians and robust widths).  Model audit 2026-10-27: the dt_xy inherited from the old-kernel '
                         'calibration is stale by ~50-60 ns on det2/det3 under the phys kernel.')
    ap.add_argument('--latency-cache', default=None,
                    help='a waveform cache of this row (w12 it1 cache.pkl): measure the trigger latency per plane from '
                         'the raw leading edge (wft.calibrate.measure_latency, model-free) and write t0_abs = C - 10 ns x '
                         'ftst.  Model audit 2026-10-27 (ws_t0): the t0_abs the bundles carried were free-fit medians of '
                         'another kernel, toothed by the 60 ns depth bin.')
    ap.add_argument('--pin-t0', action='store_true',
                    help='with --latency-cache: also set t0_prior_sigma = T0_PIN_SIGMA (t0 pinned at the trigger latency '
                         'instead of fitted).  Bench / scintillator-triggered data only: on n_TOF the trigger is not the '
                         'particle.')
    ap.add_argument('--fold-kw', action='store_true',
                    help='vrefit --keep-v: fold the mean kw into v_drift (v := v * kw_mean, kw := kw / kw_mean), so the '
                         'bundle carries this row\'s ladder v instead of the golden row\'s; tan is unchanged')
    ap.add_argument('--pairing-only', action='store_true',
                    help='with --table: re-measure dt_xy and xy_pairing only, keep the bundle\'s w0/kw (a gate arm that '
                         'must not move the angle mapping)')
    a = ap.parse_args()
    if a.pairing_only and not a.table:
        ap.error('--pairing-only needs --table')
    if not a.pairing_only and not a.angles:
        ap.error('--angles is required')
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    v = float(cal.v_drift)
    w0, kw, rep = dict(cal.w0), dict(cal.kw), {}
    z = None if a.pairing_only else np.load(os.path.join(a.angles, 'per_event.npz'))
    for p in (() if a.pairing_only else ('x', 'y')):
        w, t = z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref']
        ok = np.isfinite(w) & np.isfinite(t)
        s = ok & (np.abs(t) < 0.30)
        d = w[s] - v * t[s]
        w0[p] = float(np.median(d))
        s1 = ok & (np.abs(t) > 0.10) & (np.abs(t) < 0.40)
        kw[p] = float(np.median((w[s1] - w0[p]) / (v * t[s1]))) if s1.sum() >= 30 else 1.0
        rep[p] = dict(w0=w0[p], kw=kw[p], n_w0=int(s.sum()), n_kw=int(s1.sum()),
                      angle_bias_if_uncorrected_deg=float(np.degrees(w0[p] / v)),
                      old_w0=float(cal.w0.get(p, 0.0)), old_kw=float(cal.kw.get(p, 1.0)))
        # the per-event angle error floor (ws_floor, F3): this view's measured core resolution in tan, robust width of
        # tan_fit - tan_ref over the same |tan_ref| < 0.30 sample, M3 included (the "model floor", reference-limited)
        tf = (w[s] - w0[p]) / (kw[p] * v) - t[s]
        cal.tan_floor[p] = float(1.4826 * np.median(np.abs(tf - np.median(tf))))
        rep[p]['tan_floor'] = cal.tan_floor[p]
        print(f'{p}: w0 {w0[p]:+.3f} um/ns ({np.degrees(w0[p] / v):+.2f} deg), kw {kw[p]:.4f}  '
              f'[was w0 {rep[p]["old_w0"]:+.3f}, kw {rep[p]["old_kw"]:.4f}]  n={int(s.sum())}/{int(s1.sum())}')
    if a.fold_kw:
        # tan = (w - w0) / (kw v) is invariant; v becomes the reconstruction (ladder) v of this row (C18), which is
        # what z = v t should use rather than the golden row's v (a 500 V row has kw ~ 0.6 against a 1000 V golden v)
        km = float(np.mean([kw['x'], kw['y']]))
        prov_fold = dict(v_was=float(cal.v_drift), kw_mean=km)
        cal.v_drift = float(cal.v_drift) * km
        kw = {p: kw[p] / km for p in kw}
        for p in rep:
            rep[p]['kw_folded'] = kw[p]
        print(f'folded kw: v {prov_fold["v_was"]:.2f} -> {cal.v_drift:.2f} um/ns, kw x {kw["x"]:.4f} y {kw["y"]:.4f}')
    cal.w0, cal.kw = w0, kw
    if a.table:
        import pandas as pd
        df = pd.read_parquet(a.table)
        both = df[df.x_ok & df.y_ok & (df.x_n_dropped <= 2) & (df.y_n_dropped <= 2)]
        fd = (both.x_ftst - both.y_ftst).astype(int).to_numpy(); d = (both.x_t0 - both.y_t0).to_numpy()
        rs = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x))))
        dt_new = {int(k): float(np.median(d[fd == k])) for k in np.unique(fd) if (fd == k).sum() >= 30}
        pred = np.array([dt_new.get(int(k), np.nan) for k in fd]); t0r = d - pred; okr = np.isfinite(t0r)
        feats = dict(lq=np.log(both.x_q_sum / both.y_q_sum).to_numpy(), u50=(both.x_q_u50 - both.y_q_u50).to_numpy(),
                     u90=(both.x_q_u90 - both.y_q_u90).to_numpy(), t0=t0r[okr])
        pairing = dict(features=['lq', 'u50', 'u90', 't0'], median={k: float(np.median(v)) for k, v in feats.items()},
                       rsig={k: rs(v) for k, v in feats.items()}, n=int(len(both)), source=a.table)
        rep['dt_xy'] = dict(old={str(k): float(v) for k, v in cal.dt_xy.items()}, new={str(k): v for k, v in dt_new.items()},
                            n={str(k): int((fd == k).sum()) for k in dt_new})
        rep['xy_pairing'] = pairing
        print(f'dt_xy: {rep["dt_xy"]["old"]} -> { {k: round(v, 1) for k, v in dt_new.items()} }  (n {rep["dt_xy"]["n"]})')
        print(f'xy_pairing rsig: { {k: round(v, 3) for k, v in pairing["rsig"].items()} }')
        cal.dt_xy = dt_new
        cal.xy_pairing = pairing
    if a.latency_cache:
        import pickle
        from wft.calibrate import measure_latency, t0_abs_from_latency, T0_PIN_SIGMA
        lat = measure_latency(pickle.load(open(a.latency_cache, 'rb')))
        if set(lat) == {'x', 'y'}:
            rep['latency'] = dict(lat, source=a.latency_cache, t0_abs_old={p: {str(k): float(v) for k, v in d.items()}
                                                                          for p, d in cal.t0_abs.items()})
            cal.t0_abs = t0_abs_from_latency(lat)
            if a.pin_t0:
                rep['latency']['t0_prior_sigma_old'] = float(cal.t0_prior_sigma)
                cal.t0_prior_sigma = T0_PIN_SIGMA
            print(f't0_abs <- latency C x {lat["x"]["C"]:.1f} / y {lat["y"]["C"]:.1f} ns; t0_prior_sigma {cal.t0_prior_sigma}')
        else:
            print(f'latency: too few planes measured ({sorted(lat)}); t0_abs left unchanged')
    prov = dict(cal.provenance)
    if a.pairing_only:
        prov.update(xy_pairing_note=f'dt_xy + xy_pairing measured by rerun/w12/set_w0kw.py --pairing-only from {a.table}',
                    xy_pairing_measurement=rep)
    else:
        prov.update(w0_kw_stale=False, w0_kw_note=f'w0/kw measured by rerun/w12/set_w0kw.py from {a.angles} '
                    f'(a reconstruction with this bundle, {a.run_key})', w0_kw_measurement=rep)
    if a.n_depth_bins and a.n_depth_bins != cal.n_depth_bins:
        prov['n_depth_bins_was'] = int(cal.n_depth_bins)
        cal.n_depth_bins = int(a.n_depth_bins)
    if a.fold_kw:
        prov['kw_folded_into_v'] = prov_fold
    cal.provenance = prov
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    cal.save(a.out, note=f'{"dt_xy/xy_pairing" if a.pairing_only else "w0/kw"} measured ({a.run_key})')
    json.dump(rep, open(os.path.join(a.out, 'w0kw.json'), 'w'), indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
