#!/usr/bin/env python3
"""w0_profile.py — the lateral drift offset w0 and the angle scale as functions of where the track crosses the chamber.

Per chamber/view, in 12 bins of the track position at mid-gap along the view's own axis (from the M3 track), a robust
(Huber) line w = w0_j + s_j · v·kw · tan_ref over 0.04 < |tan_ref| < 0.40.  w0_j is a lateral drift speed [µm/ns]
the chamber adds to every track crossing there (a field-line bend); s_j the local angle scale.  Errors: per-bin
bootstrap (50).  Also the same against the OTHER axis (does it depend on the orthogonal coordinate?).
Writes ws_widthtilt/w0_profile.json and fig_w0_profile.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT
from curvature_fit import robust

RNG = np.random.default_rng(5)
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
COL = {'det2': '#1f77b4', 'det3': '#2ca02c', 'det4': '#d62728', 'det6': '#9467bd', 'det7': '#ff7f0e'}


def line(t, w, vk, scale):
    X = np.c_[np.ones_like(t), vk * t]
    return robust(X, w, scale)


def profile(pos, t, w, vk, edges, scale):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pos >= lo) & (pos < hi)
        if m.sum() < 40 or (t[m] > 0.04).sum() < 10 or (t[m] < -0.04).sum() < 10:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, np.nan, np.nan, int(m.sum()))); continue
        b = line(t[m], w[m], vk, scale)
        bs = []
        idx = np.where(m)[0]
        for _ in range(50):
            i = RNG.choice(idx, len(idx))
            bs.append(line(t[i], w[i], vk, scale))
        bs = np.array(bs)
        rows.append((0.5 * (lo + hi), float(b[0]), float(np.std(bs[:, 0])), float(b[1]), float(np.std(bs[:, 1])), int(m.sum())))
    return rows


def main():
    out = {}
    edges = np.linspace(0, 400, 13)
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for det in DETS:
        b, T = load(det); v = b['v_drift']
        X = T['x'].set_index('eid'); Y = T['y'].set_index('eid')
        for k, p in enumerate('xy'):
            df = T[p].copy(); oth = Y if p == 'x' else X
            df['pm'] = df.p0 + 15.0 * df.t
            df['po'] = (oth.p0 + 15.0 * oth.t).reindex(df.eid).to_numpy()
            vk = v * b['kw'][p]
            df = df[(np.abs(df.t) > 0.04) & (np.abs(df.t) < 0.40) & (np.abs(df.w - vk * df.t) < 0.15 * v)]
            t, w = df.t.to_numpy(), df.w.to_numpy()
            same = profile(df.pm.to_numpy(), t, w, vk, edges, 0.03 * v)
            ok = np.isfinite(df.po.to_numpy())
            other = profile(df.po.to_numpy()[ok], t[ok], w[ok], vk, edges, 0.03 * v)
            out[f'{det}_{p}'] = dict(same=same, other=other, w0_bundle=b['w0'][p], v=v, kw=b['kw'][p])
            R = np.array(same, float)
            axs[0, k].errorbar(R[:, 0], R[:, 1], yerr=R[:, 2], fmt='o-', color=COL[det], capsize=2, label=det)
            axs[1, k].errorbar(R[:, 0], R[:, 3], yerr=R[:, 4], fmt='o-', color=COL[det], capsize=2, label=det)
            print(det, p, 'w0(p):', ' '.join(f'{x:+.2f}' if np.isfinite(x) else '  nan' for x in R[:, 1]))
            print('       s(p): ', ' '.join(f'{x:.3f}' if np.isfinite(x) else '  nan' for x in R[:, 3]))
            Ro = np.array(other, float)
            print('     w0(other):', ' '.join(f'{x:+.2f}' if np.isfinite(x) else '  nan' for x in Ro[:, 1]))
    for k, p in enumerate('xy'):
        axs[0, k].set_title(f'{p.upper()} view'); axs[0, k].axhline(0, color='k', lw=0.7)
        axs[0, k].set_ylabel('lateral drift offset w0 [µm/ns]'); axs[1, k].set_ylabel('local angle scale')
        axs[1, k].axhline(1, color='k', lw=0.7); axs[1, k].set_xlabel(f'track {p} at mid-gap [mm]')
        axs[1, k].set_ylim(0.8, 1.2)
        for a in axs[:, k]:
            a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Where the chamber bends the drift: offset and scale of the micro-TPC slope vs position', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{ROOT}/ws_widthtilt/fig_w0_profile.png', dpi=120)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/w0_profile.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
