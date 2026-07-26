#!/usr/bin/env python3
"""
wft_reco.py — production entry point for the waveform-first forward-fit
reconstruction (det3 June cosmics R&D, 2026-07-25/26; see
WAVEFORM_FIRST_THREADING.md §15).

Usage
-----
    from wft_reco import WFTReco
    reco = WFTReco(calib_dir)          # dir with templates_perplane.npz,
                                       # gainmap.npz, hyper_v2.json, dt_xy.json
    fit = reco.fit_plane(W, pos, noise, ch, plane='x')
    ev  = reco.fit_event(event)        # event dict as in wfcache.pkl

Outputs per plane: p0 [mm] (track position at first-arrival depth), w [mm/ns]
(transverse speed; tan_theta = w / v_drift), t0 [ns], q (charge profile in
60 ns bins), chi2/dof, parameter errors from the chi2 curvature, and
slope_reliable (False when |tan| < TAN_MIN_SLOPE — timing carries no slope
information on 1-2 strips; use the position only, or the joint fit).

Calibration products are produced by scripts 03/11/12/13 in this directory
(impulse templates per plane, per-channel gain map, 8-hyper ref-pinned fit,
FEU t0 offsets). v_drift is part of the hyper json.

Physics floor (§12): per-event angle sigma ~1 deg is diffusion/charge
granularity, not fit noise — do not expect better from any per-event method.
"""
import os
import json
import numpy as np

import forward_model2 as fm2
import forward_model3 as fm3

TAN_MIN_SLOPE = 0.08
# physics floor (report §12: 0.30 mm/bin charge-centroid jitter): added in
# quadrature to the (tiny) statistical curvature errors.
FLOOR_TAN = 0.018          # ~1.0 deg
FLOOR_P0_MM = 0.33
CHI2DOF_BAD = 300.0        # quality flag threshold (showers / multi-track)


class PlaneFit(dict):
    """dict with attribute access."""
    __getattr__ = dict.__getitem__


class WFTReco:
    def __init__(self, calib_dir=None):
        # forward_model2 loads templates/gains/dt_xy from its BASE at import;
        # allow overriding with another calibration directory.
        if calib_dir is not None and os.path.abspath(calib_dir) != fm2.BASE:
            tz = np.load(os.path.join(calib_dir, 'templates_perplane.npz'))
            fm2.TGRID = tz['grid']
            fm2.TMPL = {'x': tz['tmpl_x'], 'y': tz['tmpl_y']}
            gz = np.load(os.path.join(calib_dir, 'gainmap.npz'))
            fm2.GAIN = {'x': gz['gain_x'], 'y': gz['gain_y']}
            fm2.DT_XY = {int(k): v for k, v in json.load(
                open(os.path.join(calib_dir, 'dt_xy.json'))).items()}
            fm2._smear_cache.clear()
            fm3._TT_CACHE.clear()
            self.calib_dir = calib_dir
        else:
            self.calib_dir = fm2.BASE
        hj = json.load(open(os.path.join(self.calib_dir, 'hyper_v2.json')))
        self.hyper = {k: hj[k] for k in
                      ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
        self.v_drift = hj['v']          # um/ns

    # ------------------------------------------------------------------
    def fit_plane(self, W, pos, noise, ch, plane, tan_seed=0.0, p0_seed=None):
        """Fit one plane. W (nstrip,32) ped/CNS-subtracted ADC; pos [mm];
        noise per strip; ch channel numbers (for the gain map)."""
        P = dict(W=np.asarray(W, np.float16), pos=np.asarray(pos, np.float32),
                 noise=np.asarray(noise, np.float32),
                 ch=np.asarray(ch, np.int16))
        if p0_seed is None:
            amax = np.asarray(W).max(axis=1)
            p0_seed = float(pos[int(np.argmax(amax))])
        g = fm2.init_guess(P, plane, tan_seed, p0_seed, self.v_drift * 1e-3)
        r = fm3.fit_plane(P, plane, *g, hyper=self.hyper)   # fast fitter
        # (fm3 = vectorized fm2 with coarse-grid start: ~5x faster and finds
        #  a lower chi2 minimum in ~20% of events)
        tan = r['w'] * 1e3 / self.v_drift
        err = self._errors(P, plane, r)
        tan_stat = err[1] * 1e3 / self.v_drift
        return PlaneFit(
            p0=r['p0'], w=r['w'], t0=r['t0'], q=r['q'],
            tan_theta=tan, theta_deg=float(np.degrees(np.arctan(tan))),
            chi2=r['chi2'], dof=r['dof'],
            p0_err=float(np.hypot(err[0], FLOOR_P0_MM)),
            w_err=err[1],
            tan_err=float(np.hypot(tan_stat, FLOOR_TAN)),
            slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
            quality_ok=bool(r['chi2'] / max(r['dof'], 1) < CHI2DOF_BAD))

    def fit_event(self, ev, joint_if_vertical=True):
        """Fit both planes of a cache-style event dict. If a plane's slope is
        unreliable and the other's is not, optionally re-fit jointly (shared
        charge profile + tied t0) to stabilise it."""
        out = {}
        for plane in ('x', 'y'):
            P = ev[plane]
            out[plane] = self.fit_plane(P['W'], P['pos'], P['noise'], P['ch'],
                                        plane,
                                        tan_seed=ev.get(f'tan_{plane}', 0.0),
                                        p0_seed=None)
        if joint_if_vertical and 'ftst_x' in ev and \
                out['x'].slope_reliable != out['y'].slope_reliable:
            try:
                j = fm2.fit_joint(ev, out['x'].p0, out['x'].w,
                                  out['y'].p0, out['y'].w, out['x'].t0,
                                  hyper=self.hyper)
                for plane, p0k, wk in (('x', 'p0x', 'wx'), ('y', 'p0y', 'wy')):
                    out[plane]['joint_p0'] = j[p0k]
                    out[plane]['joint_w'] = j[wk]
                out['joint_chi2'] = j['chi2']
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    def _errors(self, P, plane, r, dp=0.05, dw=2e-4):
        """1-sigma errors on (p0, w) from the chi2 curvature (t0, q profiled
        implicitly via refit at displaced values; scaled by sqrt(chi2/dof) to
        absorb model systematics)."""
        W, noise, pos, sat = fm2.prep_plane(P, plane)

        def chi(p0v, wv):
            c, _ = fm2.chi2_plane(plane, W, noise, pos, sat, p0v, wv,
                                  r['t0'], self.hyper)
            return c
        c0 = r['chi2']
        try:
            d2p = (chi(r['p0'] + dp, r['w']) - 2 * c0 +
                   chi(r['p0'] - dp, r['w'])) / dp ** 2
            d2w = (chi(r['p0'], r['w'] + dw) - 2 * c0 +
                   chi(r['p0'], r['w'] - dw)) / dw ** 2
            scale = max(r['chi2'] / max(r['dof'], 1), 1.0)
            ep = float(np.sqrt(2 * scale / d2p)) if d2p > 0 else np.nan
            ew = float(np.sqrt(2 * scale / d2w)) if d2w > 0 else np.nan
            return ep, ew
        except Exception:
            return np.nan, np.nan
