#!/usr/bin/env python3
"""width_summary.py [dir] — read width_probe_<det>[_toy].npz and answer, per chamber/view:

  trade   : d ln(w_free) / d ln(width) — how much the fitted slope moves when the assumed lateral width changes
            (sigma_p0 at fixed Dp, and Dp at fixed sigma_p0), per |tan| bin
  prefer  : which (sigma_p0, Dp) grid point the data prefer at the FREE angle (event-summed chi2, profile off)
  at M3   : median (chi2 at the M3 angle − chi2 at the free angle, same p0 grid)/dof per grid point, central tracks only
            (|p_mid − 200| < 70 mm, where the bow ramp is < 0.3 µm/ns); does any width make the M3 angle preferred?
Writes width_summary.json next to the npz directory and fig_width_trade.png.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_widthtilt/eos')
TB = [(0.08, 0.15), (0.15, 0.25), (0.25, 0.6)]


def main():
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for f in sorted(glob.glob(f'{D}/width_probe_*.npz')):
        tag = os.path.basename(f)[12:-4]
        z = np.load(f); R = z['per']; S0 = list(z['s0']); DPF = list(z['dpf'])
        if R.ndim != 2 or R.shape[1] < 15:
            continue
        res = {}
        for vi, p in enumerate('xy'):
            m = R[:, 1] == vi
            if m.sum() == 0:
                continue
            A = R[m]
            keys = np.unique(A[:, 0])
            # per event, grid arrays
            ev = {}
            for row in A:
                ev.setdefault(row[0], {})[(round(row[4], 3), round(row[5], 3))] = row
            full = [e for e in ev.values() if len(e) == len(S0) * len(DPF)]
            g0 = (round(S0[1] if p == 'x' else S0[2], 3), 1.0)        # nearest grid point to the bundle (0.238 / 0.343)
            trade = {}
            for lo, hi in TB:
                E = [e for e in full if lo <= abs(next(iter(e.values()))[2]) < hi]
                if len(E) < 8:
                    continue
                ws = np.array([[e[(round(s, 3), round(d, 3))][6] for s in S0 for d in DPF] for e in E])
                w_ref = np.array([next(iter(e.values()))[3] for e in E])
                i0 = [(round(s, 3), round(d, 3)) for s in S0 for d in DPF].index(g0)
                rel = np.median(ws / ws[:, [i0]], axis=0).reshape(len(S0), len(DPF))
                trade[f'{lo}-{hi}'] = dict(n=len(E), rel=rel.tolist(), ratio_to_ref=float(np.median(ws[:, i0] / w_ref)))
            # preferred width at the free angle
            chi = np.array([[e[(round(s, 3), round(d, 3))][7] for s in S0 for d in DPF] for e in full])
            dchi = (chi - chi.min(axis=1, keepdims=True)).sum(0).reshape(len(S0), len(DPF))
            best = np.unravel_index(np.argmin(dchi), dchi.shape)
            # M3 angle vs free angle, central tracks
            cen = [e for e in full if abs(next(iter(e.values()))[13] + 15 * next(iter(e.values()))[2] - 200) < 70
                   and abs(next(iter(e.values()))[2]) >= 0.15]
            if cen:
                dd = np.array([[(e[(round(s, 3), round(d, 3))][8] - e[(round(s, 3), round(d, 3))][11]) / e[(round(s, 3), round(d, 3))][9]
                                for s in S0 for d in DPF] for e in cen])
                d_med = np.median(dd, axis=0).reshape(len(S0), len(DPF)); f_pos = np.mean(dd > 0, axis=0).reshape(len(S0), len(DPF))
            else:
                d_med = f_pos = np.full((len(S0), len(DPF)), np.nan)
            res[p] = dict(n=len(full), trade=trade, prefer=dict(sigma_p0=S0[best[0]], dp_factor=DPF[best[1]], dchi_sum=dchi.tolist()),
                          m3=dict(n=len(cen), dchi_per_dof_median=d_med.tolist(), frac_model_prefers_fit=f_pos.tolist()))
            print(f'{tag} {p}: n {len(full)}  preferred sigma_p0 {S0[best[0]]} Dp x{DPF[best[1]]}  '
                  f'| M3-vs-fit dchi2/dof (central, n {len(cen)}) at bundle {d_med[S0.index(g0[0]), DPF.index(1.0)]:+.2f}, '
                  f'min over grid {np.nanmin(d_med):+.2f}')
            for k, tr in trade.items():
                rel = np.array(tr['rel'])
                print(f'      |tan| {k}: n {tr["n"]}  w(free)/w_ref {tr["ratio_to_ref"]:.3f}  '
                      f'w vs sigma_p0 (Dp x1) ' + ' '.join(f'{x:.3f}' for x in rel[:, DPF.index(1.0)]) +
                      '   w vs Dp (s0 bundle) ' + ' '.join(f'{x:.3f}' for x in rel[S0.index(g0[0])]))
            if not tag.endswith('toy') and '0.25-0.6' in trade:
                rel = np.array(trade['0.25-0.6']['rel'])
                axs[0 if p == 'x' else 1].plot(S0, rel[:, DPF.index(1.0)], 'o-', label=tag)
        out[tag] = res
    for k, p in enumerate('xy'):
        axs[k].axhline(1, color='gray', lw=0.7); axs[k].set_xlabel('assumed prompt footprint σ0 [mm]')
        axs[k].set_title(f'{p.upper()} view, steep tracks |tan| ≥ 0.25 (data)'); axs[k].grid(alpha=0.3)
    axs[0].set_ylabel('fitted slope / slope at the bundle σ0'); axs[0].legend(fontsize=8)
    fig.suptitle('Width–tilt trade on data: how the fitted slope moves with the assumed footprint', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{D}/../fig_width_trade.png', dpi=130)
    json.dump(out, open(f'{D}/../width_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
