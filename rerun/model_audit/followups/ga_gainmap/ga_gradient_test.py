#!/usr/bin/env python3
"""ga_gradient_test.py -- GA: does the measured footprint-scale gain gradient bias the reconstructed angle on data?

Mechanism under test (REPORT, "What a gain non-uniformity does to the fit"): with an amplification gain
g(x) = exp(l(x)), the electrons of depth z land at x(z) = x_m + z tan with a lateral spread s^2(z) = Dp^2 z / v; the
gain-weighted landing centroid moves by delta(z) = l'(x(z)) * s^2(z) (exact for a locally linear l).  The resistive
spread and the strip sampling come AFTER the avalanche and do not weight.  The fit sees a straight cloud of the right
width with a shifted slope: chi2-blind.  Predicted error of the fitted tan = least-squares slope of delta(z) over the gap
(z in [0, G]), divided by kw (the bundle's angle scale).  A per-strip readout gain would instead weight the full
footprint: delta(z) = l'(x(z)) * (sigma0^2 + pitch^2/12 + Dp^2 z / v)  (variant 'strip').

Test on the golden finalize tables (ws_v_w0/<det>.parquet), every view, |tan_ref| < 0.45, fitted, no spark:
  l'(x)  from the head-on charge profile (ga_chargemap: 1 mm medians, 1.5 mm smoothing) of the OTHER parity half of
         events (no event predicts itself), d/dx numerically
  r      = fitted tan - tan_ref (M3), minus its 10 mm running median along x (removes the bow / tilt surfaces, which
         vary on a 100 mm scale, and anything else smooth)
  slope k of r on the prediction (Huber, bootstrap); k = 1: the mechanism explains the gain-related angle error
  fully; k = 0: no effect.  Also the rms of the prediction (the size of the effect) and how much of r's variance it
  removes.

    ga_gradient_test.py -> results/gradient_test.json, figures/gradient_test.png
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ga_chargemap as gcm  # noqa: E402

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = gcm.DETS
G = 30.0
PITCH = 0.78


def huber(x, y, it=20, c=1.345):
    X = np.c_[np.ones_like(x), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = y - X @ b
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        w = np.where(np.abs(r) <= c * s, 1.0, c * s / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b


def run_median(x, y, half=5.0):
    o = np.argsort(x); xs, ys = x[o], y[o]
    out = np.empty_like(y)
    lo = np.searchsorted(xs, xs - half); hi = np.searchsorted(xs, xs + half)
    med = np.array([np.median(ys[l:h]) for l, h in zip(lo, hi)])
    out[o] = med
    return out


def profiles(x, l, par):
    edges = np.arange(np.floor(np.percentile(x, 0.5)), np.ceil(np.percentile(x, 99.5)) + gcm.BIN, gcm.BIN)
    c = 0.5 * (edges[1:] + edges[:-1])
    D = {}
    for h in (0, 1):
        Gp, _ = gcm.profile(x[par == h], l[par == h], edges)
        S = gcm.gsmooth(c, Gp, gcm.SMALL_MM)
        D[h] = (c, np.gradient(S, c))
    return D


def predict(c, dl, xm, tan, Dp, v, kw, s0=0.0, mode='amp'):
    """slope of delta(z) over z in [0, G] for each track (mesh position xm, slope tan) / kw."""
    z = np.linspace(0, G, 31)
    X = xm[:, None] + z[None, :] * tan[:, None]
    okc = np.isfinite(dl)
    a = np.interp(X, c[okc], dl[okc], left=np.nan, right=np.nan)
    s2 = Dp ** 2 * (z / v)[None, :]
    if mode == 'strip':
        s2 = s2 + s0 ** 2 + PITCH ** 2 / 12
    d = a * s2
    zc = z - z.mean()
    slope = ((d - d.mean(1, keepdims=True)) * zc[None, :]).sum(1) / (zc ** 2).sum()
    inter = d.mean(1) - slope * z.mean()
    return slope / kw, inter


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = {}
    fig, axs = plt.subplots(2, 5, figsize=(20, 7))
    for i, det in enumerate(DETS):
        b = json.load(open(os.path.join(ROOT, det, 'calib_bundle', 'bundle.json')))
        Dp, v = float(b['hyper']['Dp']), float(b['v_drift']) * 1e-3          # mm/sqrt(ns), mm/ns
        d = pd.read_parquet(os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet'))
        d = d[d.x_ok.fillna(False).astype(bool) & d.y_ok.fillna(False).astype(bool) & ~d.spark.astype(bool)]
        par = np.arange(len(d)) % 2
        res[det] = {}
        for j, (p, o) in enumerate((('x', 'y'), ('y', 'x'))):
            kw = float(b['kw'][p]) if isinstance(b.get('kw'), dict) else 1.0
            s0 = float(b['hyper'].get(f'sigma_p0_{p}', b['hyper']['sigma_p0']))
            t, to = d[f'tan_ref_{p}'].to_numpy(float), d[f'tan_ref_{o}'].to_numpy(float)
            xm = d[f'ref_mesh_{p}'].to_numpy(float)
            xmid = xm + gcm.MID * t
            q = d[f'{p}_q_sum'].to_numpy(float)
            tf = d[f'{p}_tan_theta'].to_numpy(float)
            base = np.isfinite(t) & np.isfinite(xm) & np.isfinite(q) & (q > 0) & np.isfinite(to)
            mk = base & (np.abs(t) < gcm.TMAX)
            l = np.full(len(d), np.nan)
            l[mk] = np.log(q[mk] / np.sqrt(1 + t[mk] ** 2 + to[mk] ** 2)); l[mk] -= np.median(l[mk])
            P = profiles(xmid[mk], l[mk], par[mk])
            sel = base & np.isfinite(tf) & (np.abs(t) < 0.45) & (np.abs(tf - t) < 0.15)
            pred = {m: np.full(len(d), np.nan) for m in ('amp', 'strip')}
            for h in (0, 1):
                k = sel & (par == h)
                c, dl = P[1 - h]                      # the other half's profile
                for m in pred:
                    pred[m][k] = predict(c, dl, xm[k], t[k], Dp, v, kw, s0, m)[0]
            r = tf - t
            ok = sel & np.isfinite(pred['amp'])
            rh = r[ok] - run_median(xmid[ok], r[ok], 5.0)
            out = dict(n=int(ok.sum()), Dp=Dp, v=v, kw=kw)
            rs = np.random.default_rng(5)
            for m in pred:
                pr = pred[m][ok]
                kk = huber(pr, rh)[1]
                bs = [huber(pr[ii], rh[ii])[1] for ii in (rs.integers(0, len(pr), len(pr)) for _ in range(100))]
                s68 = lambda a: float(np.percentile(np.abs(a - np.median(a)), 68))
                out[m] = dict(k=float(kk), k_err=float(np.std(bs)), pred_rms_deg=float(np.degrees(np.std(pr))),
                              pred_p95_deg=float(np.degrees(np.percentile(np.abs(pr), 95))),
                              resid_s68_deg=float(np.degrees(s68(rh))),
                              resid_s68_corrected_deg=float(np.degrees(s68(rh - pr))))
            res[det][p] = out
            ax = axs[j, i]
            pr = pred['amp'][ok]
            e = np.percentile(pr, [1, 99])
            bins = np.linspace(e[0], e[1], 13)
            idx = np.digitize(pr, bins)
            bc = [np.degrees(np.median(pr[idx == q_])) for q_ in range(1, len(bins)) if (idx == q_).sum() > 20]
            bm = [np.degrees(np.median(rh[idx == q_])) for q_ in range(1, len(bins)) if (idx == q_).sum() > 20]
            ax.plot(bc, bm, 'o-', color='C3')
            lim = max(0.05, np.max(np.abs(bc)) if bc else 0.05)
            ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
            ax.axhline(0, color='0.7', lw=0.5)
            ax.set_title(f"{det} {p.upper()}: k = {out['amp']['k']:.2f} ± {out['amp']['k_err']:.2f}, "
                         f"pred rms {out['amp']['pred_rms_deg']:.3f}°", fontsize=9)
            ax.set_xlabel('predicted tan error [deg] (gradient × Dp²/v)'); ax.set_ylabel('fit − M3, high-passed [deg]')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'gradient_test.png'), dpi=110)
    json.dump(res, open(os.path.join(HERE, 'results', 'gradient_test.json'), 'w'), indent=1)
    for det in DETS:
        for p in 'xy':
            o = res[det][p]
            print(det, p, o['n'], ' '.join(f"{m}: k={o[m]['k']:+.2f}±{o[m]['k_err']:.2f} pred_rms={o[m]['pred_rms_deg']:.3f}° "
                                          f"p95={o[m]['pred_p95_deg']:.3f}° s68 {o[m]['resid_s68_deg']:.3f}->{o[m]['resid_s68_corrected_deg']:.3f}"
                                          for m in ('amp', 'strip')))


if __name__ == '__main__':
    main()
