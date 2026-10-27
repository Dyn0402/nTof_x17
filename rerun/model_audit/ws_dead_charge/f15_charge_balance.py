#!/usr/bin/env python3
"""f15_charge_balance.py [--root ~/x17/cosmic_bench/model_audit] -- what sets the width of ln(qx/qy) per chamber (F15).

Inputs per golden row: final_physmask/events.parquet (fitted total charge per view, x_q_sum / y_q_sum, p0, flagged =
dead-censored strips in the window), angles/per_event.npz (M3-matched events, tan_ref) and the promoted bundle.

Steps, each a subset or a correction of the previous one (robust sigma = 1.4826 MAD):
  all       both views fitted (the F15 number)
  m3        both views M3-matched and within 5 deg of the M3 angle (both views are the same, real track)
  live      ... and no censored (dead/short) strip in either window
  unsat     ... and no saturated sample in either view (dof == n_strips x n_samples)
  conn      ... ln(qx/qy) corrected by a measured per-connector X and Y gain: ln r = c + a[conn_x] - b[conn_y],
            fitted by iterated medians on the m3 sample with the TRAIN half of the events, scored on the TEST half
Also: ln r against total charge (noise), against the window's censored-strip count, and the 2D map in (x_p0, y_p0).
Amplification gain is common to both views (one avalanche feeds both readout layers), so it cancels in qx/qy; what
survives is readout-side: per-connector (= one DREAM chip) gain, dead/shorted strips, pickup, window truncation.

Writes ws_dead_charge/f15.json, figures/f15_*.png, and xy_gain_<det>.json (the per-connector table).
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DETS = ['det3', 'det2', 'det4', 'det6', 'det7']
NSAMP = 32
CONN_MM = 49.92


def rsig(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) > 5 else np.nan


def conn_of(p):
    return np.clip(np.floor(np.asarray(p, float) / CONN_MM), 0, 7).astype(int)


def fit_conn(lr, cx, cy, n_iter=20, nmin=15):
    """ln r = c + a[cx] - b[cy] by alternating medians; a, b zero-median over populated connectors."""
    a = np.zeros(8); b = np.zeros(8); c = float(np.median(lr))
    for _ in range(n_iter):
        r = lr - c + b[cy]
        for k in range(8):
            m = cx == k
            a[k] = np.median(r[m]) if m.sum() >= nmin else 0.0
        r = -(lr - c - a[cx])
        for k in range(8):
            m = cy == k
            b[k] = np.median(r[m]) if m.sum() >= nmin else 0.0
        c = float(np.median(lr - a[cx] + b[cy]))
    return c, a, b


def load(root, det):
    d = os.path.join(root, det)
    df = pd.read_parquet(f'{d}/final_physmask/events.parquet').drop_duplicates('event_id', keep=False).set_index('event_id')
    z = np.load(f'{d}/final_physmask/angles/per_event.npz')
    b = json.load(open(f'{d}/calib_bundle/bundle.json'))
    ref = {}
    for p in 'xy':
        ref[p] = pd.Series(z[f'{p}_tan_ref'], index=z[f'{p}_eid'])
        ref[p] = ref[p][~ref[p].index.duplicated(keep=False)]
    return df, ref, b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.path.expanduser('~/x17/cosmic_bench/model_audit'))
    a = ap.parse_args()
    out = {}
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    fig2, axs2 = plt.subplots(2, 5, figsize=(20, 7.5))
    for i, det in enumerate(DETS):
        df, ref, b = load(a.root, det)
        both = df[df.x_ok & df.y_ok & (df.x_q_sum > 0) & (df.y_q_sum > 0)].copy()
        both['lr'] = np.log(both.x_q_sum / both.y_q_sum)
        R = dict(n_all=len(both), all=rsig(both.lr), median_all=float(np.median(both.lr)))
        idx = both.index.intersection(ref['x'].index).intersection(ref['y'].index)
        m3 = both.loc[idx].copy()
        for p in 'xy':
            dth = np.degrees(np.arctan(m3[f'{p}_tan_theta']) - np.arctan(ref[p].loc[idx]))
            m3[f'{p}_dth'] = dth.to_numpy()
        m3 = m3[(m3.x_dth.abs() < 5) & (m3.y_dth.abs() < 5)]
        R.update(n_m3=len(m3), m3=rsig(m3.lr))
        flag = (m3.x_n_flagged_strips > 0) | (m3.y_n_flagged_strips > 0)
        R.update(n_flagged=int(flag.sum()), frac_flagged=float(flag.mean()), flagged=rsig(m3.lr[flag]),
                 median_shift_flagged=float(np.median(m3.lr[flag]) - np.median(m3.lr[~flag])) if flag.sum() > 5 else np.nan)
        # per-view flagged: does X losing a strip lower ln r (charge lost) or not (censoring restores it)?
        for p, s in (('x', +1), ('y', -1)):
            fp = (m3[f'{p}_n_flagged_strips'] > 0) & ~((m3[f'{"y" if p == "x" else "x"}_n_flagged_strips']) > 0)
            R[f'shift_flagged_only_{p}'] = float(np.median(m3.lr[fp]) - np.median(m3.lr[~flag])) if fp.sum() > 10 else np.nan
            R[f'n_flagged_only_{p}'] = int(fp.sum())
        live = m3[~flag]
        R.update(n_live=len(live), live=rsig(live.lr))
        sat = np.zeros(len(live), bool)
        for p in 'xy':
            sat |= live[f'{p}_dof'].to_numpy() < 0.98 * live[f'{p}_n_strips'].to_numpy() * NSAMP
        R.update(frac_sat=float(sat.mean()), sat=rsig(live.lr[sat]), unsat=rsig(live.lr[~sat]),
                 median_shift_sat=float(np.median(live.lr[sat]) - np.median(live.lr[~sat])) if sat.sum() > 5 else np.nan)
        us = live[~sat].copy()
        # per-connector gains, train on even events, test on odd (fit on the m3 sample incl. flagged: gains are
        # a property of the chip, the flagged windows only add scatter)
        tr = (us.index.to_numpy() % 2) == 0
        cx, cy = conn_of(us.x_p0), conn_of(us.y_p0)
        lr = us.lr.to_numpy()
        c, ga, gb = fit_conn(lr[tr], cx[tr], cy[tr])
        corr = lr - ga[cx] + gb[cy]
        R.update(unsat_test=rsig(lr[~tr]), conn_test=rsig(corr[~tr]), conn_a=ga.round(3).tolist(), conn_b=gb.round(3).tolist(),
                 n_per_conn_x=np.bincount(cx, minlength=8).tolist(), n_per_conn_y=np.bincount(cy, minlength=8).tolist())
        c_all, ga_all, gb_all = fit_conn(lr, cx, cy)
        json.dump(dict(det=det, run_key=b['run_key'], note='ln(qx/qy) = c + a[conn_x] - b[conn_y]; connector = floor(p0 / 49.92 mm) '
                       '= one DREAM chip; gain_x per connector = exp(a), gain_y = exp(b) (divide the view charge by it)',
                       c=c_all, a=ga_all.tolist(), b=gb_all.tolist(), n_x=np.bincount(cx, minlength=8).tolist(),
                       n_y=np.bincount(cy, minlength=8).tolist(), sample='M3-matched single tracks, no censored strip, unsaturated'),
                  open(os.path.join(HERE, f'xy_gain_{det}.json'), 'w'), indent=1)
        # residual vs charge (noise): split in total-charge terciles
        qt = np.log(np.sqrt(us.x_q_sum * us.y_q_sum)).to_numpy()
        e = np.quantile(qt, [0, 1 / 3, 2 / 3, 1])
        R['conn_by_charge_tercile'] = [rsig(corr[(qt >= e[k]) & (qt <= e[k + 1])]) for k in range(3)]
        # position structure that the connector model leaves: 25 mm map residual spread
        bx = np.clip((us.x_p0.to_numpy() // 25).astype(int), 0, 15); by = np.clip((us.y_p0.to_numpy() // 25).astype(int), 0, 15)
        M = np.full((16, 16), np.nan); Mc = np.full((16, 16), np.nan)
        for ix in range(16):
            for iy in range(16):
                m = (bx == ix) & (by == iy)
                if m.sum() >= 12:
                    M[iy, ix] = np.median(lr[m]); Mc[iy, ix] = np.median(corr[m])
        R['map25_spread_raw'] = rsig(M[np.isfinite(M)].ravel()); R['map25_spread_conn'] = rsig(Mc[np.isfinite(Mc)].ravel())
        out[det] = R
        # figure 1: the cascade
        steps = [('all fitted', R['all']), ('M3 track\nboth views', R['m3']), ('no dead strip\nin window', R['live']),
                 ('unsaturated', R['unsat']), ('connector\ngains (test)', R['conn_test'])]
        ax = axs[i]
        ax.bar(range(len(steps)), [s[1] for s in steps], color=['#888', '#6a9fd4', '#4f86c6', '#2f6db0', '#d0703c'])
        ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=8)
        for k, s in enumerate(steps):
            ax.text(k, s[1] + 0.005, f'{s[1]:.3f}', ha='center', fontsize=8)
        ax.set_title(f'{det} ({b["run_key"]})'); ax.grid(axis='y', alpha=.3)
        if i == 0:
            ax.set_ylabel('robust width of ln(q$_x$/q$_y$)')
        # figure 2: maps
        vm = 0.4
        for row, (MM, lab) in enumerate(((M, 'raw'), (Mc, 'after connector gains'))):
            ax2 = axs2[row, i]
            im = ax2.imshow(MM - np.nanmedian(MM), origin='lower', extent=(0, 400, 0, 400), cmap='RdBu_r', vmin=-vm, vmax=vm)
            for k in range(1, 8):
                ax2.axvline(k * CONN_MM, color='k', lw=.4, alpha=.4); ax2.axhline(k * CONN_MM, color='k', lw=.4, alpha=.4)
            ax2.set_title(f'{det}: median ln(qx/qy), {lab}', fontsize=9)
            ax2.set_xlabel('x (X-strip position) [mm]'); ax2.set_ylabel('y [mm]')
        print(det, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.items() if not isinstance(v, list)}))
    fig.suptitle('Where the x–y charge-balance width comes from (each bar a subset / correction of the previous)')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'figures', 'f15_cascade.png'), dpi=130)
    fig2.colorbar(im, ax=axs2, shrink=.6, label='ln(qx/qy) − median')
    fig2.savefig(os.path.join(HERE, 'figures', 'f15_maps.png'), dpi=110, bbox_inches='tight')
    json.dump(out, open(os.path.join(HERE, 'f15.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
