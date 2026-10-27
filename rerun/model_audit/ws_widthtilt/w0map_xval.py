#!/usr/bin/env python3
"""w0map_xval.py — does a position-dependent angle calibration (lateral drift offset w0(p) and scale s(p) along the
view's own axis, p = track position at mid-gap) remove the angle nonlinearity and improve the resolution?

Two-fold cross-validation on the finalize tables: the map is measured on even event ids and applied to odd ones, and
vice versa (so no event calibrates itself).  The map is the 12-bin robust profile of w0_profile.py, linearly
interpolated in p, with p taken from the FIT (p0 + 15 mm · tan_fit — production has no M3) — the train side uses the
same fit-based p so the map is what production would apply.  Compared, on the held-out halves:
  const : tan = (w − w0)/(kw·v)              (production, bundle constants)
  map   : tan = (w − w0(p))/(s(p)·kw·v)
Metrics: s68 of Δθ = θ_fit − θ_ref (all, inclined |t| ≥ 0.1, steep |t| ≥ 0.25), far (> 5°) %, the symmetric scale S(|t|)
and even term E(|t|).  Writes ws_widthtilt/w0map_xval.json and fig_w0map_xval.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT
from curvature_fit import robust

DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
EDGES = np.linspace(0, 400, 13)
SBINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def build_map(p, t, w, vk, scale):
    ctr, a, s = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (p >= lo) & (p < hi) & (np.abs(t) > 0.04) & (np.abs(t) < 0.40) & (np.abs(w - vk * t) < 0.15 * vk)
        if m.sum() < 40 or (t[m] > 0.04).sum() < 10 or (t[m] < -0.04).sum() < 10:
            continue
        b = robust(np.c_[np.ones(m.sum()), vk * t[m]], w[m], scale)
        ctr.append(0.5 * (lo + hi)); a.append(b[0]); s.append(b[1])
    return np.array(ctr), np.array(a), np.array(s)


def metrics(t_ref, t_fit):
    d = np.degrees(np.arctan(t_fit) - np.arctan(t_ref)); far = np.abs(d) > 5
    out = {'far_pct': float(100 * far.mean())}
    for nm, m in (('all', np.abs(t_ref) < 9), ('inclined', np.abs(t_ref) >= 0.1), ('steep', np.abs(t_ref) >= 0.25)):
        mm = m & ~far
        x = d[mm]
        out[nm] = float(np.percentile(np.abs(x - np.median(x)), 68)) if mm.sum() > 10 else np.nan
    S, E = [], []
    for lo, hi in SBINS:
        bp = (t_ref >= lo) & (t_ref < hi) & ~far; bn = (t_ref <= -lo) & (t_ref > -hi) & ~far
        if bp.sum() < 8 or bn.sum() < 8:
            S.append(np.nan); E.append(np.nan); continue
        rp, rn = np.median(t_fit[bp] / t_ref[bp]), np.median(t_fit[bn] / t_ref[bn])
        S.append(float(0.5 * (rp + rn))); E.append(float(0.5 * (np.median(t_fit[bp] - t_ref[bp]) + np.median(t_fit[bn] - t_ref[bn]))))
    out['S'] = S; out['E_tan'] = E
    return out


def main():
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    col = {'det2': '#1f77b4', 'det3': '#2ca02c', 'det4': '#d62728', 'det6': '#9467bd', 'det7': '#ff7f0e'}
    for det in DETS:
        b, T = load(det); v = b['v_drift']
        for k, pl in enumerate('xy'):
            df = T[pl]; vk = v * b['kw'][pl]
            t, w = df.t.to_numpy(), df.w.to_numpy(); eid = df.eid.to_numpy()
            tconst = (w - b['w0'][pl]) / vk
            p = df.p0.to_numpy() + 15.0 * tconst
            tmap = np.full_like(t, np.nan)
            for fold in (0, 1):
                tr = (eid % 2) == fold; te = ~tr
                c, a, s = build_map(p[tr], t[tr], w[tr], vk, 0.03 * v)
                a_i = np.interp(p[te], c, a); s_i = np.interp(p[te], c, s)
                tmap[te] = (w[te] - a_i) / (s_i * vk)
            ok = np.isfinite(tmap) & np.isfinite(tconst)
            M0, M1 = metrics(t[ok], tconst[ok]), metrics(t[ok], tmap[ok])
            out[f'{det}_{pl}'] = dict(const=M0, map=M1, n=int(ok.sum()))
            print(f'{det} {pl}: s68 all {M0["all"]:.3f}->{M1["all"]:.3f}  incl {M0["inclined"]:.3f}->{M1["inclined"]:.3f}  '
                  f'steep {M0["steep"]:.3f}->{M1["steep"]:.3f}  far {M0["far_pct"]:.2f}->{M1["far_pct"]:.2f}  '
                  f'S {" ".join(f"{x:.3f}" for x in M0["S"])} -> {" ".join(f"{x:.3f}" for x in M1["S"])}', flush=True)
            ctr = [0.5 * (lo + hi) for lo, hi in SBINS]
            axs[k].plot(ctr, M0['S'], '--', color=col[det], alpha=0.6)
            axs[k].plot(ctr, M1['S'], 'o-', color=col[det], lw=2, label=det)
    for k, pl in enumerate('xy'):
        axs[k].axhline(1, color='k', lw=0.8); axs[k].set_title(f'{pl.upper()} view'); axs[k].grid(alpha=0.3)
        axs[k].set_xlabel('|tan θ| (M3)')
    axs[0].set_ylabel('measured / true slope'); axs[0].set_ylim(0.88, 1.08)
    axs[0].legend(title='solid: position map (cross-validated)\ndashed: constant w0, kw', fontsize=8, title_fontsize=8)
    fig.suptitle('Angle response after a position-dependent calibration w0(p), s(p)', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{ROOT}/ws_widthtilt/fig_w0map_xval.png', dpi=130)
    json.dump(out, open(f'{ROOT}/ws_widthtilt/w0map_xval.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
