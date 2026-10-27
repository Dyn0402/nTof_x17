#!/usr/bin/env python3
"""bow_test.py — is the position ramp of w0 a BOWED CHAMBER (the readout plane, and with it the drift direction,
curved) or a drift-field lens inside a flat chamber?  Positions alone decide, against M3.

If the mesh surface sits at height h(p) above/below the plane the alignment assumes, a track of slope tan crosses it
displaced by tan·h(p): the position residual p0 − ref_mesh has slope dr/dtan = h(p) in each position bin.  A drift
direction normal to a curved surface is tilted by α(p) = −h'(p) (sign: see below), so a bow predicts
    w0(p) − w0(c) = ∓ v · h'(p)       i.e. h(p) = ∓ ∫ (w0(p) − w0(c))/v dp,
a parabola for a linear w0 ramp.  A lens (only the field lines bend, the plane is flat) predicts h(p) = const.
We measure h(p) from positions (ws_v_w0 per-event tables: p0, ref_mesh, tan_ref) and compare its curvature with the
two signs of the w0-ramp prediction; the sign of the relation is not assumed, the measured h(p) chooses it.
Writes ws_widthtilt/bow_test.json and fig_bow_test.png.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curvature_fit import robust

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
EDGES = np.linspace(20, 380, 10)
RNG = np.random.default_rng(3)


def main():
    W0 = json.load(open(f'{ROOT}/ws_widthtilt/w0_profile.json'))
    out = {}
    fig, axs = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
    for i, det in enumerate(DETS):
        df = pd.read_parquet(f'{ROOT}/ws_v_w0/{det}.parquet')
        b = json.load(open(f'{ROOT}/{det}/calib_bundle/bundle.json')); v = b['v_drift']
        for k, p in enumerate('xy'):
            t = df[f'tan_ref_{p}'].to_numpy(); r = (df[f'{p}_p0'] - df[f'ref_mesh_{p}']).to_numpy()
            pm = df[f'ref_mesh_{p}'].to_numpy() + 15.0 * t
            ok = np.isfinite(t) & np.isfinite(r) & (np.abs(r - np.nanmedian(r)) < 3) & (np.abs(t) < 0.45)
            ctr, h, he = [], [], []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = ok & (pm >= lo) & (pm < hi)
                if m.sum() < 60:
                    continue
                X = np.c_[np.ones(m.sum()), t[m]]
                bb = robust(X, r[m], 0.5)
                bs = [robust(X[j], r[m][j], 0.5)[1] for j in (RNG.integers(0, m.sum(), m.sum()) for _ in range(40))]
                ctr.append(0.5 * (lo + hi)); h.append(float(bb[1])); he.append(float(np.std(bs)))
            ctr, h, he = map(np.array, (ctr, h, he))
            # prediction from the w0 ramp: integrate (w0(p) - w0 at the centre)/v over p, both signs
            S = np.array(W0[f'{det}_{p}']['same'], float)
            g = np.isfinite(S[:, 1]) & (S[:, 0] > 30) & (S[:, 0] < 370)
            pc, w0c = S[g, 0], S[g, 1]
            slope = np.polyfit(pc, w0c, 1)[0]                     # um/ns per mm
            # bow: alpha(p) = slope (p - c) / v  ->  h(p) = -/+ slope (p - c)^2 / (2 v)  (+ const)
            c = 200.0
            curv_pred = slope / v                                 # |h''| predicted [1/mm]
            # measured curvature of h(p): quadratic fit
            if len(ctr) >= 4:
                A = np.c_[np.ones_like(ctr), ctr - c, (ctr - c) ** 2]
                wts = 1 / np.maximum(he, 0.05)
                coef, *_ = np.linalg.lstsq(A * wts[:, None], h * wts, rcond=None)
                cov = np.linalg.inv((A * wts[:, None]).T @ (A * wts[:, None]))
                h2, h2e = 2 * coef[2], 2 * np.sqrt(cov[2, 2])
            else:
                h2, h2e = np.nan, np.nan
            out[f'{det}_{p}'] = dict(p=ctr.tolist(), h=h.tolist(), h_err=he.tolist(), w0_slope=float(slope),
                                     curv_pred_abs=float(curv_pred), curv_meas=float(h2), curv_meas_err=float(h2e))
            print(f'{det} {p}: w0 slope {slope * 1e3:+.2f} um/ns/m  -> |h\'\'| pred {curv_pred * 1e6:.1f} /km ; '
                  f'measured h\'\' {h2 * 1e6:+.1f} ± {h2e * 1e6:.1f} /km   h(p) ' + ' '.join(f'{x:+.2f}' for x in h))
            ax = axs[k, i]
            ax.errorbar(ctr, h - np.median(h), yerr=he, fmt='o', color='k', label='measured (M3 positions)')
            pp = np.linspace(20, 380, 50)
            for sg, ls in ((1, '--'), (-1, ':')):
                ax.plot(pp, sg * 0.5 * curv_pred * ((pp - c) ** 2 - np.median((ctr - c) ** 2)), ls, color='C3',
                        label=f'bow predicted from w0 ramp ({"+" if sg > 0 else "−"})')
            ax.axhline(0, color='gray', lw=0.6); ax.set_title(f'{det} {p.upper()}'); ax.set_ylim(-3, 3)
            if i == 0:
                ax.set_ylabel('mesh height offset h [mm]\n(slope of residual vs tan)')
            if k == 1:
                ax.set_xlabel('position at mid-gap [mm]')
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('Is the chamber bowed? Mesh height from positions vs the bow the w0 ramp would need', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{ROOT}/ws_widthtilt/fig_bow_test.png', dpi=110)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/bow_test.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
