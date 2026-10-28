#!/usr/bin/env python3
"""ga_chargemap.py -- GA task 1: how much does the gain really vary, and at which spatial scale, from the per-event
track charge of the golden finalize tables (ws_v_w0/<det>.parquet: finalize reco + M3 + alignment), split in two
independent halves so that the noise of the map (Landau ~0.6 in ln q per event) is separated from real structure.

Per chamber and view p (X: position across X strips = M3 u at mid-gap; Y: w), head-on-ish tracks (|tan_ref| < TMAX in
that view, so the charge column covers +-2 strips + 30 mm * TMAX), both views fitted, no spark:
  l = ln(q_p * cos theta) - median          (q_p = fitted depth-profile sum; path-normalised)
  profile G(x): running median of l in BIN-mm bins, for event halves A (even) and B (odd) and for all events.
Scale decomposition (on the profiles, bins with >= NMIN events in both halves):
  large  : Gaussian smooth, sigma LARGE_MM (the part a track-charge normalisation absorbs completely)
  mid    : smooth(SMALL_MM) - smooth(LARGE_MM)
  fine   : profile - smooth(SMALL_MM)  (strip scale; cannot be resolved by the charge of a 5-strip cluster anyway)
For each component: rms of the all-event profile, the SIGNAL rms = sqrt(cov(A, B)) (noise cancels), the NOISE rms =
std(A - B)/2... and the gradient d G / dx of the SMALL-smoothed profile (what biases the fit; see REPORT), also as
signal sqrt(cov).  Along-strip check (X view): the profile in the lower and upper half of the chamber in y, their
signal correlation -> is a per-channel map (constant along the strip) a complete description?

Selection artefacts (see REPORT): only detected tracks enter (low-gain regions are biased up: efficiency compresses
the map); saturated tracks cap the bright end (compresses); dead strips are censored in the fit (q extrapolated).
So these numbers are LOWER bounds on the large-scale gain variation, not a calibration.

    ga_chargemap.py   -> results/chargemap.json, figures/chargemap_<det>.png
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
TMAX = 0.15
BIN = 1.0
NMIN = 4
SMALL_MM, LARGE_MM = 1.5, 8.0
MID = 15.0


def gsmooth(x, y, s, w=None):
    ok = np.isfinite(y)
    w = np.ones_like(y) if w is None else w
    out = np.full_like(y, np.nan)
    for i in range(len(x)):
        k = np.exp(-0.5 * ((x[ok] - x[i]) / s) ** 2) * w[ok]
        if k.sum() > 1e-6 and ok[i]:
            out[i] = (k * y[ok]).sum() / k.sum()
    return out


def profile(pos, l, edges):
    idx = np.digitize(pos, edges) - 1
    G = np.full(len(edges) - 1, np.nan); N = np.zeros(len(edges) - 1, int)
    for i in range(len(edges) - 1):
        v = l[idx == i]
        N[i] = len(v)
        if len(v) >= NMIN:
            G[i] = np.median(v)
    return G, N


def sig(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return dict(signal=np.nan, noise=np.nan, n=int(ok.sum()))
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    c = float(np.mean(a * b))
    return dict(signal=float(np.sign(c) * np.sqrt(abs(c))), noise=float(np.std(a - b) / 2), corr=float(np.corrcoef(a, b)[0, 1]),
                n=int(ok.sum()))


def analyse(det):
    d = pd.read_parquet(os.path.join(ROOT, 'ws_v_w0', f'{det}.parquet'))
    d = d[d.x_ok.fillna(False).astype(bool) & d.y_ok.fillna(False).astype(bool) & ~d.spark.astype(bool)]
    out = {}
    for p, other in (('x', 'y'), ('y', 'x')):
        t, to = d[f'tan_ref_{p}'].to_numpy(float), d[f'tan_ref_{other}'].to_numpy(float)
        pos = (d[f'ref_mesh_{p}'] + MID * d[f'tan_ref_{p}']).to_numpy(float)
        opos = (d[f'ref_mesh_{other}'] + MID * d[f'tan_ref_{other}']).to_numpy(float)
        q = d[f'{p}_q_sum'].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(pos) & np.isfinite(q) & (q > 0) & (np.abs(t) < TMAX) & np.isfinite(to)
        l = np.log(q[ok] / np.sqrt(1 + t[ok] ** 2 + to[ok] ** 2)); l -= np.median(l)
        x, ox = pos[ok], opos[ok]
        par = np.arange(ok.sum()) % 2
        edges = np.arange(np.floor(np.percentile(x, 0.5)), np.ceil(np.percentile(x, 99.5)) + BIN, BIN)
        c = 0.5 * (edges[1:] + edges[:-1])
        G, N = profile(x, l, edges)
        GA, _ = profile(x[par == 0], l[par == 0], edges)
        GB, _ = profile(x[par == 1], l[par == 1], edges)
        both = np.isfinite(GA) & np.isfinite(GB)
        GA[~both] = np.nan; GB[~both] = np.nan
        comp = {}
        S = {k: gsmooth(c, g, SMALL_MM) for k, g in (('A', GA), ('B', GB), ('all', G))}
        L = {k: gsmooth(c, g, LARGE_MM) for k, g in (('A', GA), ('B', GB), ('all', G))}
        comp['total'] = sig(GA, GB)
        comp['large'] = sig(L['A'], L['B'])
        comp['mid'] = sig(S['A'] - L['A'], S['B'] - L['B'])
        comp['fine'] = sig(GA - S['A'], GB - S['B'])
        gA, gB = np.gradient(S['A'], c), np.gradient(S['B'], c)
        comp['grad_small'] = sig(gA, gB)          # d ln g / dx at the footprint scale [1/mm]
        gLA, gLB = np.gradient(L['A'], c), np.gradient(L['B'], c)
        comp['grad_large'] = sig(gLA, gLB)
        # along-strip: lower vs upper half of the other coordinate
        med = np.median(ox)
        GL, _ = profile(x[ox < med], l[ox < med], edges)
        GU, _ = profile(x[ox >= med], l[ox >= med], edges)
        along = sig(gsmooth(c, GL, SMALL_MM), gsmooth(c, GU, SMALL_MM))
        out[p] = dict(n_events=int(ok.sum()), rms_l_event=float(np.std(l)), centers=c.tolist(),
                      G=[None if not np.isfinite(v) else round(float(v), 4) for v in G], N=N.tolist(),
                      S=[None if not np.isfinite(v) else round(float(v), 4) for v in S['all']],
                      Lg=[None if not np.isfinite(v) else round(float(v), 4) for v in L['all']],
                      comp=comp, along_strip=along,
                      frac_bins_low=float(np.nanmean(G < np.log(0.5))), p90_p10=float(np.exp(np.nanpercentile(G, 90) - np.nanpercentile(G, 10))))
    return out


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True); os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    R = {}
    fig, axs = plt.subplots(5, 2, figsize=(14, 15), sharey=True)
    for i, det in enumerate(DETS):
        R[det] = analyse(det)
        for j, p in enumerate('xy'):
            r = R[det][p]; ax = axs[i, j]
            c = np.array(r['centers']); G = np.array(r['G'], float); S = np.array(r['S'], float); L = np.array(r['Lg'], float)
            ax.plot(c, G, '.', ms=2, color='0.6', label='1 mm median ln q')
            ax.plot(c, S, '-', lw=1, color='C0', label=f'smooth {SMALL_MM} mm')
            ax.plot(c, L, '-', lw=1.5, color='C3', label=f'smooth {LARGE_MM} mm')
            cm = r['comp']
            ax.set_title(f"{det} {p.upper()}: signal rms large {cm['large']['signal']:.2f}, mid {cm['mid']['signal']:.2f}, "
                         f"fine {cm['fine']['signal']:.2f}; grad {cm['grad_small']['signal']:.3f}/mm", fontsize=9)
            ax.set_ylim(-2.5, 1.5); ax.axhline(0, color='k', lw=0.5)
            if i == 4:
                ax.set_xlabel(f'M3 position at mid-gap across {p.upper()} strips [mm]')
            if j == 0:
                ax.set_ylabel('ln(q cos θ) − median')
        axs[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'chargemap_profiles.png'), dpi=110)
    json.dump(dict(TMAX=TMAX, BIN=BIN, SMALL_MM=SMALL_MM, LARGE_MM=LARGE_MM, res=R), open(os.path.join(HERE, 'results', 'chargemap.json'), 'w'))
    print(f"{'det':5s} {'v':2s} {'n':>5s} {'p90/p10':>7s} | signal rms (noise): total  large  mid  fine | grad_small [1/mm] | along-strip corr")
    for det in DETS:
        for p in 'xy':
            r = R[det][p]; c = r['comp']
            f = lambda k: f"{c[k]['signal']:+.3f}({c[k]['noise']:.3f})"
            print(f"{det:5s} {p:2s} {r['n_events']:5d} {r['p90_p10']:7.2f} | {f('total')} {f('large')} {f('mid')} {f('fine')} | "
                  f"{f('grad_small')} | {r['along_strip']['corr']:.2f}")


if __name__ == '__main__':
    main()
