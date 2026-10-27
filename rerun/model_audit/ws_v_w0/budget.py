#!/usr/bin/env python3
"""budget.py -- read v_uncertainty.json, separate three different questions, and draw the figures (F8).

  1. How well is the AVERAGE ladder v of this chamber in this run known?   stat (max of event and time-block
     bootstrap) + estimator choice + angle window + M3 reference scale/rotation + the excess scatter of splits that
     must not matter at all (ftst class, M3 chi2).
  2. How much does the LOCAL v differ from that average?   the excess scatter over time (gas, temperature over the
     run), position (quadrants: gap / field non-uniformity) and charge.  This is what you are exposed to when one v is
     applied to a subset of tracks -- a different region, a different hour, a beam that illuminates one corner.
  3. Is the response symmetric in the sign of the angle?   half the difference of the + and - tan ladders (with the
     head-on offset from the whole sample).  Workstream A (the width-tilt trade) owns the cause; listed here because
     it is not an uncertainty on v but a non-linearity a single v cannot describe.

Plus the det3 X-Y question: the same events in both views, so v (a property of the gas) is common and the
difference X-Y is a view-dependent angle scale.  Paired (same-event) bootstrap, per quadrant and time quintile.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v_uncertainty import est, ROOT, DETS

WS = os.path.join(ROOT, 'ws_v_w0')
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
RNG = np.random.default_rng(5)


def sign_split(det, p, nb=150):
    """+tan and -tan ladders with the offset from the robust line intercept (w0_study.json).  The head-on median
    used by the split in v_uncertainty (median w over |tan| < 0.05) is biased by v * median(tan) of that band
    (0.004 x 36 = 0.15 um/ns on det3 X), which alone makes a 2-7 % false sign asymmetry."""
    W = json.load(open(f'{WS}/w0_study.json'))
    a = W['tilt'][det][f'offset_{p}']['a']
    t = pd.read_parquet(f'{WS}/{det}.parquet')
    ok = t[f'{p}_ok'].astype(bool).to_numpy()
    w = t[f'{p}_w'].to_numpy()[ok] * 1e3; tr = t[f'tan_ref_{p}'].to_numpy()[ok]
    out = {}
    for name, m in (('v_pos', tr > 0), ('v_neg', tr < 0)):
        i = np.flatnonzero(m)
        out[name] = est(w[i], tr[i], c=a)
        out[name + '_err'] = float(np.nanstd([est(w[j], tr[j], c=a) for j in (RNG.choice(i, len(i)) for _ in range(nb))]))
    out['half_diff'] = 0.5 * (out['v_pos'] - out['v_neg'])
    out['err'] = 0.5 * float(np.hypot(out['v_pos_err'], out['v_neg_err']))
    out['offset_used'] = a
    return out


def split_budget(R):
    ex = {k: v['sigma_excess'] for k, v in R['splits'].items()}
    b = R['budget']
    avg = dict(stat=b['stat'], estimator=b['estimator'], window=b['window'], reference=b['reference'],
               selection=float(np.hypot(ex.get('ftst', 0), ex.get('m3_chi2', 0))))
    avg['total'] = float(np.sqrt(sum(x ** 2 for x in avg.values())))
    nonu = dict(time=ex.get('time', 0), position=ex.get('quadrant', 0), charge=ex.get('charge', 0))
    nonu['total'] = float(np.sqrt(sum(x ** 2 for x in nonu.values())))
    return dict(average=avg, nonuniformity=nonu)


def paired_xy(det, nb=200):
    t = pd.read_parquet(f'{WS}/{det}.parquet')
    ok = (t.x_ok.astype(bool) & t.y_ok.astype(bool) & np.isfinite(t.x_w) & np.isfinite(t.y_w)).to_numpy()
    t = t[ok].sort_values('evttime')
    wx, wy = t.x_w.to_numpy() * 1e3, t.y_w.to_numpy() * 1e3
    tx, ty = t.tan_ref_x.to_numpy(), t.tan_ref_y.to_numpy()

    def diff(i):
        vx, vy = est(wx[i], tx[i]), est(wy[i], ty[i])
        return 100 * (vx - vy) / vy

    def bt(i):
        return float(np.nanstd([diff(RNG.choice(i, len(i))) for _ in range(nb)]))
    allidx = np.arange(len(t))
    out = dict(all=(diff(allidx), bt(allidx)), quadrant={}, time={})
    px, py = t.x_p0.to_numpy(), t.y_p0.to_numpy()
    q = (px > np.median(px)).astype(int) * 2 + (py > np.median(py)).astype(int)
    for k in range(4):
        i = np.flatnonzero(q == k); out['quadrant'][k] = (diff(i), bt(i))
    tq = np.minimum((np.arange(len(t)) / len(t) * 5).astype(int), 4)
    for k in range(5):
        i = np.flatnonzero(tq == k); out['time'][k] = (diff(i), bt(i))
    return out


def main():
    U = json.load(open(f'{WS}/v_uncertainty.json'))
    B = {det: {p: dict(split_budget(U[det][p]), sign=sign_split(det, p)) for p in 'xy'} for det in DETS}
    XY = {det: paired_xy(det) for det in DETS}
    for det in DETS:
        for p in 'xy':
            b = B[det][p]; R = U[det][p]
            print(f"{det} {p.upper()}: v {R['v']:.2f}  ledger ±{R['err_naive']:.2f} | average ±{b['average']['total']:.2f} "
                  f"({', '.join(f'{k} {v:.2f}' for k, v in b['average'].items() if k != 'total')}) | non-uniformity "
                  f"±{b['nonuniformity']['total']:.2f} ({100 * b['nonuniformity']['total'] / R['v']:.1f} %; "
                  f"{', '.join(f'{k} {v:.2f}' for k, v in b['nonuniformity'].items() if k != 'total')}) | sign ±"
                  f"{b['sign']['half_diff']:+.2f}±{b['sign']['err']:.2f} (v+ {b['sign']['v_pos']:.2f}, v- {b['sign']['v_neg']:.2f})")
        d, e = XY[det]['all']
        print(f"   X-Y paired: {d:+.2f} ± {e:.2f} %   quadrants " +
              ' '.join(f'{a:+.1f}±{b:.1f}' for a, b in XY[det]['quadrant'].values()) + '   time ' +
              ' '.join(f'{a:+.1f}±{b:.1f}' for a, b in XY[det]['time'].values()))
    json.dump(dict(budget=B, xy_paired=XY), open(f'{WS}/budget.json', 'w'), indent=1)
    figures(U, B, XY)


def figures(U, B, XY):
    os.makedirs(FIG, exist_ok=True)
    labels = [f'{d} {p.upper()}' for d in DETS for p in 'xy']
    x = np.arange(len(labels))
    # 1: three numbers per chamber-view, relative
    fig, ax = plt.subplots(figsize=(11.5, 5))
    led = [100 * U[d][p]['err_naive'] / U[d][p]['v'] for d in DETS for p in 'xy']
    avg = [100 * B[d][p]['average']['total'] / U[d][p]['v'] for d in DETS for p in 'xy']
    nonu = [100 * B[d][p]['nonuniformity']['total'] / U[d][p]['v'] for d in DETS for p in 'xy']
    sgn = [100 * abs(B[d][p]['sign']['half_diff']) / U[d][p]['v'] for d in DETS for p in 'xy']
    ax.bar(x - 0.3, led, 0.2, color='0.6', label='quoted in the ledger (statistics only)')
    ax.bar(x - 0.1, avg, 0.2, color='#4C72B0', label='average v: real uncertainty')
    ax.bar(x + 0.1, nonu, 0.2, color='#DD8452', label='local v varies (time + position) by')
    ax.bar(x + 0.3, sgn, 0.2, color='#C44E52', label='+tan vs −tan ladder differ by (½)')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('relative to v  [%]')
    ax.set_title('Ladder v: the quoted ±0.2 % is the statistics; the chamber is not uniform at the 2–9 % level')
    ax.legend(frameon=False, fontsize=9); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'v_budget.png'), dpi=140); plt.close(fig)

    # 2: where the variation is -- time and position, all chambers
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for d in DETS:
        for p, ls in (('x', '-'), ('y', '--')):
            R = U[d][p]
            v = np.array(R['splits']['time']['values']); e = np.array(R['splits']['time']['errors'])
            axs[0].errorbar(np.arange(len(v)) + (0.04 if p == 'y' else 0), 100 * (v / R['v'] - 1), yerr=100 * e / R['v'],
                            fmt='o' + ls, ms=4, capsize=2, label=f'{d} {p.upper()}')
            v = np.array(R['splits']['quadrant']['values']); e = np.array(R['splits']['quadrant']['errors'])
            axs[1].errorbar(np.arange(len(v)) + (0.04 if p == 'y' else 0), 100 * (v / R['v'] - 1), yerr=100 * e / R['v'],
                            fmt='o' + ls, ms=4, capsize=2)
    axs[0].set_xticks(range(5)); axs[0].set_xticklabels(['first fifth', '2', '3', '4', 'last fifth'])
    axs[0].set_title('v over the run (det6 and det7 share one night)'); axs[0].set_ylabel('v / run average − 1  [%]')
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(['x low, y low', 'x low, y high', 'x high, y low', 'x high, y high'])
    axs[1].set_title('v across the chamber (quadrants)')
    for a in axs:
        a.axhline(0, color='0.5', lw=1); a.grid(alpha=.3)
    axs[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'v_time_position.png'), dpi=140); plt.close(fig)

    # 3: X - Y, paired
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    for k, d in enumerate(DETS):
        dd, e = XY[d]['all']
        X, Y = U[d]['x'], U[d]['y']
        e_led = 100 * np.hypot(X['err_naive'], Y['err_naive']) / Y['v']
        ax.errorbar(k, dd, yerr=e, fmt='o', color='k', capsize=4, label='paired bootstrap' if k == 0 else None)
        ax.errorbar(k + 0.15, dd, yerr=e_led, fmt='none', color='#C44E52', capsize=3, label='ledger error' if k == 0 else None)
    ax.axhline(0, color='0.5', lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(DETS)
    ax.set_ylabel('(v_X − v_Y) / v_Y  [%]'); ax.set_title('X vs Y on the SAME events'); ax.legend(frameon=False)
    ax = axs[1]
    for k, d in enumerate(DETS):
        vals = list(XY[d]['quadrant'].values()) + [(np.nan, np.nan)] + list(XY[d]['time'].values())
        xs = np.arange(len(vals)) + (k - 2) * 0.08
        ax.errorbar(xs, [a for a, _ in vals], yerr=[b for _, b in vals], fmt='o', ms=4, capsize=2, label=d)
    ax.set_xticks(range(10)); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', '', 't1', 't2', 't3', 't4', 't5'])
    ax.axhline(0, color='0.5', lw=1); ax.set_ylabel('(v_X − v_Y) / v_Y  [%]')
    ax.set_title('X − Y per quadrant and per fifth of the run'); ax.legend(frameon=False, ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'v_x_minus_y.png'), dpi=140); plt.close(fig)

    # 4: is the change over the run real?  The ladder (angle route) against the column duration (timing route, 1/u90):
    # the two share no estimator, so a common trend is the gas, not the fit
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for d in DETS:
        t = pd.read_parquet(f'{WS}/{d}.parquet').sort_values('evttime')
        q = np.minimum((np.arange(len(t)) / len(t) * 5).astype(int), 4)
        for p, mk in (('x', 'o'), ('y', 's')):
            ok = t[f'{p}_ok'].astype(bool).to_numpy(); u90 = t[f'{p}_q_u90'].to_numpy()
            T = np.array([np.nanmedian(u90[ok & (q == k)]) for k in range(5)])
            lv = np.array(U[d][p]['splits']['time']['values']); le = np.array(U[d][p]['splits']['time']['errors'])
            ax.errorbar(100 * (T[0] / T - 1), 100 * (lv / lv[0] - 1), yerr=100 * le / lv[0], fmt=mk + '-', ms=4,
                        capsize=2, label=f'{d} {p.upper()}', alpha=.85)
    ax.plot([-12, 14], [-12, 14], 'k--', lw=1, label='same change')
    ax.axhline(0, color='0.8', lw=.8); ax.axvline(0, color='0.8', lw=.8)
    ax.set_xlabel('change of 1 / column duration (timing route)  [%]')
    ax.set_ylabel('change of ladder v (angle route)  [%]')
    ax.set_title('Over each run, both routes see v move: the gas changed')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'v_time_two_routes.png'), dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
