#!/usr/bin/env python3
"""make_figures.py -- the FA slides: figures/*.png from results/*.json (vgas.py, xy_scale.py, tilt.py, tilt_summary.py)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fa_common import RES, FIG, DETS, COL  # noqa: E402

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=140); plt.close(fig); print('wrote', name)


def fig_vt():
    V = {d: J(f'vgas_{d}.json') for d in DETS}
    V = {d: v for d, v in V.items() if v}
    if not V:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for d, R in V.items():
        h = np.array(R['vt']['h']); vr = 100 * (np.array(R['vt']['v_rel']) - 1)
        e = 100 * np.array(R['vt']['T_err']) / np.array(R['vt']['T'])
        ax[0].errorbar(h, vr, yerr=e, fmt='-', color=COL[d], lw=1.4, alpha=.9, label=f"{d} ({R['block_minutes']:.0f}-min blocks)")
        lb = R['ladder_blocks']
        hc = [0.5 * (b['h_lo'] + b['h_hi']) for b in lb]
        vl = np.array([0.5 * (b['v_x'] + b['v_y']) for b in lb]); el = np.array([0.5 * np.hypot(b['v_x_err'], b['v_y_err']) for b in lb])
        ax[0].errorbar(hc, 100 * (vl / np.median(vl) - 1), yerr=100 * el / np.median(vl), fmt='o', color=COL[d], mfc='white', ms=5, capsize=2)
        invT = np.array([1 / b['T'] for b in lb]); eT = np.array([b['T_err'] / b['T'] for b in lb])
        ax[1].errorbar(100 * (invT / invT.mean() - 1), 100 * (vl / vl.mean() - 1), xerr=100 * eT, yerr=100 * el / vl.mean(),
                       fmt='o', color=COL[d], capsize=2, label=f"{d}: slope {R['ladder_vs_T_slope']:.2f}")
    ax[0].axhline(0, color='0.6', lw=.8)
    ax[0].set_xlabel('hours into the run'); ax[0].set_ylabel('v(t) / run mean − 1  [%]')
    ax[0].set_title('v(t): line = column duration (every trigger, no M3)\nopen circles = ladder (M3)', fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)
    lim = max(abs(np.array(ax[1].get_xlim())).max(), abs(np.array(ax[1].get_ylim())).max())
    ax[1].plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='same change')
    ax[1].set_xlabel('change of 1 / T (column duration)  [%]'); ax[1].set_ylabel('change of ladder v  [%]')
    ax[1].set_title('Same 8 time blocks: the two routes move together', fontsize=11); ax[1].legend(frameon=False, fontsize=8)
    save(fig, 'fa_vt.png')


def fig_allan():
    V = {d: J(f'vgas_{d}.json') for d in DETS}
    V = {d: v for d, v in V.items() if v}
    if not V:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for d, R in V.items():
        a = np.array(R['allan'])
        mins = 60 * a[:, 0] / R['rate_per_h']
        ax.loglog(mins, a[:, 1], 'o-', color=COL[d], ms=3, label=f"{d}: {R['block_minutes']:.0f} min ({R['block_N']} triggers)")
        ax.plot(R['block_minutes'], R['block_allan_pct'], '*', color=COL[d], ms=14, mec='k')
    ax.set_xlabel('time block [minutes]'); ax.set_ylabel('Allan deviation of block-median T  [%]')
    ax.set_title('Block size from the data: statistics fall, gas drift rises', fontsize=11)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3, which='both')
    save(fig, 'fa_allan.png')


def fig_maps():
    V = {d: J(f'vgas_{d}.json') for d in DETS}
    V = {d: v for d, v in V.items() if v}
    if not V:
        return
    fig, axs = plt.subplots(3, len(V), figsize=(3.4 * len(V), 9.6), squeeze=False)
    for k, (d, R) in enumerate(V.items()):
        e = np.array(R['Tmap']['e']); TM = 100 * (np.array(R['Tmap']['T_rel'], float) - 1)
        im = axs[0, k].pcolormesh(e, e, TM.T, cmap='RdBu_r', vmin=-12, vmax=12)
        axs[0, k].set_title(f"{d}: column T(x,y) − 1 [%]\nrms {R['Tmap']['rms_pct']:.1f} %", fontsize=10)
        eL = np.array(R['vmap']['e']); VM = 100 * (np.array(R['vmap']['vXY'], float) - 1)
        axs[1, k].pcolormesh(eL, eL, VM.T, cmap='RdBu_r', vmin=-12, vmax=12)
        axs[1, k].set_title(f"ladder v(x,y) − 1 [%]\nrms {R['vmap']['rms_pct']:.1f} %", fontsize=10)
        tm = R['two_map']
        if tm:
            G = np.array(tm['dlng_pct'], float)
            axs[2, k].pcolormesh(eL, eL, G.T, cmap='RdBu_r', vmin=-12, vmax=12)
            axs[2, k].set_title(f"gap = v·T − 1 [%]\nrms {tm['rms_gap_pct']:.1f} %", fontsize=10)
        for a in axs[:, k]:
            a.set_aspect('equal'); a.set_xlabel('x [mm]', fontsize=9); a.set_ylabel('y [mm]', fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.5, label='% from chamber median')
    fig.savefig(os.path.join(FIG, 'fa_maps.png'), dpi=130); plt.close(fig); print('wrote fa_maps.png')


def fig_twomap():
    V = {d: J(f'vgas_{d}.json') for d in DETS}
    V = {d: v for d, v in V.items() if v and v.get('two_map')}
    S = J('vgas_summary.json')
    if not V:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for d, R in V.items():
        tm = R['two_map']
        p = np.array(tm['pred_dlnv_pct'], float).ravel(); m = np.array(tm['meas_dlnv_pct'], float).ravel()
        e = np.array(tm['meas_err_pct'], float).ravel()
        ax[0].errorbar(p, m, yerr=e, fmt='o', color=COL[d], ms=4, capsize=1.5,
                       label=f"{d}: r = {tm['corr_meas_pred']:.2f}, χ² {tm['chi2_meas_vs_pred']:.0f} (flat {tm['chi2_meas_vs_flat']:.0f}) / {tm['ndf']}")
    lim = 8
    ax[0].plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax[0].set_xlabel('v(x,y) predicted from the column-duration map via the field  [%]')
    ax[0].set_ylabel('v(x,y) measured by the ladder (M3)  [%]')
    ax[0].set_title(f"Is v(x,y) the bow's field effect?  β = dln v/dln E = {S['beta']:.2f}" if S and S.get('beta') else 'v(x,y)', fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)
    if S and S.get('scan'):
        sc = np.array(S['scan'], float)
        E = sc[:, 0] / 3.0; v = 30.0 / (sc[:, 1] * 1e-3)
        ax[1].errorbar(E, v, yerr=v * sc[:, 2] / sc[:, 1], fmt='o', color=COL['det3'], capsize=3)
        for x, y, n in zip(E, v, sc[:, 3]):
            ax[1].annotate(f'{int(n)}', (x, y), textcoords='offset points', xytext=(4, -10), fontsize=8, color='0.4')
        ax[1].set_xlabel('drift field  [V/cm]  (HV / 30 mm)'); ax[1].set_ylabel('30 mm / T  [µm/ns]')
        ax[1].set_title('det3 drift scan, column duration only (no M3, no angle)', fontsize=11)
    save(fig, 'fa_twomap.png')


def fig_gate():
    V = {d: J(f'vgas_{d}.json') for d in DETS}
    V = {d: v for d, v in V.items() if v}
    if not V:
        return
    names = ['+vt', '+vt+vxy', '+vt+vxyTk', '+vt+w0xy', '+vt+vxy+w0xy', 'angle_map']
    lab = ['v(t) (column duration)', '+ v(x,y) ladder', '+ v(x,y) = T(x,y)^k (no M3 map)', '+ w0(x,y) only (bow tilt)',
           '+ v(x,y) + w0(x,y)', 'ws_widthtilt 1-D map (NOT held out)']
    metr = [(0, 'Δ s68 (all)  [deg]'), (2, 'Δ angle-scale spread over time  [pp]'), (3, 'Δ angle-scale spread over cells  [pp]')]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    labels = [f'{d} {p.upper()}' for d in V for p in 'xy']
    x = np.arange(len(labels)); wdt = 0.8 / len(names)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(names)))
    for a, (mi, title) in zip(axs, metr):
        for k, n in enumerate(names):
            vals = [V[d]['gate'][p].get(n, {}).get('delta_vs_prod', [np.nan] * 4)[mi] for d in V for p in 'xy']
            errs = [V[d]['gate'][p].get(n, {}).get('delta_err', [np.nan] * 4)[mi] for d in V for p in 'xy']
            a.bar(x + (k - len(names) / 2 + .5) * wdt, vals, wdt, yerr=errs, color=cols[k], label=lab[k], capsize=1)
        a.axhline(0, color='k', lw=.8); a.set_xticks(x); a.set_xticklabels(labels, rotation=45, fontsize=9); a.set_title(title, fontsize=11)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Held-out gate (maps from even events, metrics on odd): change against the production conversion; below 0 = better', fontsize=11)
    save(fig, 'fa_gate.png')


def fig_xy():
    X = J('xy_scale.json')
    if not X:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = ax[0]
    tests = [('golden', 'ladder (full row)'), ('golden_w0map', 'after the w0(p) bow map')]
    arms = ['base', 's0c', 'pin', 'pins0c']; armlab = {'base': 'arm: today\'s code', 's0c': 'arm: common σ0, no choice',
                                                     'pin': 'arm: t0 pinned', 'pins0c': 'arm: pinned + common σ0'}
    k = 0; xt = []
    for d in DETS:
        R = X[d]; xs = []
        for j, (key, lab) in enumerate(tests):
            v, e, n = R[key]['all']; a.errorbar(k + j * 0.12, v, yerr=e, fmt='os'[j], color=COL[d], capsize=2, label=lab if d == 'det2' else None)
        for j, arm in enumerate(arms):
            if arm in R.get('condor_arms', {}):
                v, e, n = R['condor_arms'][arm]['all']
                a.errorbar(k + 0.24 + j * 0.1, v, yerr=e, fmt='^v<>'[j], color=COL[d], mfc='white', capsize=2,
                           label=armlab[arm] if d == 'det3' else None)
        if R.get('colsum'):
            c = R['colsum']['unsaturated']
            a.errorbar(k + 0.7, c['TX_over_TY_pct'], yerr=c['err'], fmt='D', color='k', ms=6, capsize=2,
                       label='column duration T_X/T_Y − 1 (no angle)' if d == 'det2' else None)
        xt.append(k); k += 1
    a.axhline(0, color='0.5', lw=.8); a.set_xticks(np.array(xt) + .3); a.set_xticklabels(DETS)
    a.set_ylabel('X / Y − 1  [%]  (same events)'); a.set_title('Ladder X/Y vs column duration X/Y', fontsize=11)
    a.legend(frameon=False, fontsize=7.5, loc='lower left')
    a = ax[1]
    # rows whose ladder does not follow M3 (tilt.py angle_valid: 6-22 / 6-23 bottom slot) are left out
    T = {(r['run'], r['subrun']): r.get('angle_valid', True) for r in (J('tilt_rows.json') or []) if 'failed' not in r}
    rows = [r for r in X.get('rows', []) if r['det'] == 'mx17_3' and T.get((r['run'], r['subrun']), True)]
    short = {'det3_d700': 'sat drift 700 V', 'det3_d500': 'sat drift 500 V', 'det3_day625': '6-25 (bottom slot)'}
    lab = []
    for i, r in enumerate(rows):
        a.errorbar(i, r['r'][0], yerr=r['r'][1], fmt='o', color=COL['det3'], capsize=2)
        s = (r['subrun'].replace('drift_scan_resist_490V_drift_', 'sat drift ').replace('hv_scan2_resist_', 'sat mesh ')
             .replace('hv_scan_resist_', 'sat mesh ').replace('_drift_1000V', '').replace('long_run_p2_det1_sanity_check', '6-27/28 night'))
        lab.append(short.get(r['run'], s))
    a.errorbar(len(rows), X['det3']['golden']['all'][0], yerr=X['det3']['golden']['all'][1], fmt='*', ms=12, color='k', capsize=2)
    lab.append('golden: sat long run')
    a.axhline(0, color='0.5', lw=.8); a.set_xticks(range(len(lab))); a.set_xticklabels(lab, rotation=75, fontsize=7)
    a.set_ylabel('det3 ladder X / Y − 1  [%]'); a.set_title('det3: every run and HV row', fontsize=11)
    save(fig, 'fa_xy_scale.png')


def fig_tilt():
    S = J('tilt_summary.json'); R = J('tilt_rows.json')
    if not R:
        return
    R = [r for r in R if 'failed' not in r and r['n_events'] >= 150]
    for r in R:                          # no working ladder -> no angle route (tilt.py angle_valid)
        if not r.get('angle_valid', True):
            for p in 'xy':
                for k in ('angle_deg', 'angle_err', 'curv_angle_per_km', 'curv_angle_err'):
                    r[f'tilt_{p}'][k] = np.nan
    fig, axs = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True)
    order = sorted(R, key=lambda r: (DETS.index(r['det']), tuple(int(x) for x in r['date'].split('-')[:2]), r['subrun']))
    labels = []
    prev = None
    for i, r in enumerate(order):
        for a, p in zip(axs, 'xy'):
            T = r[f'tilt_{p}']
            a.errorbar(i - .12, T['angle_deg'], yerr=T['angle_err'], fmt='o', color=COL[r['det']], capsize=1.5)
            a.errorbar(i + .12, T['pos_deg'], yerr=T['pos_err'], fmt='s', mfc='white', color=COL[r['det']], capsize=1.5)
        key = (r['det'], r['run'])
        if prev and key != prev:
            for a in axs:
                a.axvline(i - .5, color='0.85' if key[0] == prev[0] else '0.4', lw=1 if key[0] == prev[0] else 1.6)
        prev = key
        slot = {232.0: 'B', 702.0: 'T'}.get(r['slot_z'], '?')
        sub = r['subrun'].replace('drift_scan_resist_490V_', '').replace('hv_scan2_resist_', 'hv2 ').replace('hv_scan_resist_', 'hv ')
        labels.append(f"{r['det']} {r['date']} {slot} {sub}"[:40])
    for a, p in zip(axs, 'xy'):
        a.axhline(0, color='0.5', lw=.8); a.set_ylabel(f'{p.upper()} tilt at centre [deg]')
    axs[0].plot([], [], 'o', color='k', label='angle route (w at tan = 0)'); axs[0].plot([], [], 's', mfc='white', color='k', label='position route (residual ∝ tan·p)')
    axs[0].legend(frameon=False, fontsize=9, ncol=2)
    axs[1].set_xticks(range(len(labels))); axs[1].set_xticklabels(labels, rotation=80, fontsize=7)
    axs[0].set_title('Tilt against M3 per bench row (thin lines: new run of the same chamber; thick: next chamber; T/B = top/bottom slot)', fontsize=11)
    save(fig, 'fa_tilt_rows.png')
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for a, p in zip(axs, 'xy'):
        for i, r in enumerate(order):
            T = r[f'tilt_{p}']
            a.errorbar(i - .12, T['curv_angle_per_km'], yerr=T['curv_angle_err'], fmt='o', color=COL[r['det']], capsize=1.5)
            a.errorbar(i + .12, T['curv_pos_per_km'], yerr=T['curv_pos_err'], fmt='s', mfc='white', color=COL[r['det']], capsize=1.5)
        a.axhline(0, color='0.5', lw=.8); a.set_ylabel(f'{p.upper()} curvature k [1/km]'); a.set_xticks([])
        a.set_title(f'Bow curvature along {p.upper()} per row (o angle, □ position)', fontsize=11)
    save(fig, 'fa_bow_rows.png')


if __name__ == '__main__':
    for f in (fig_vt, fig_allan, fig_maps, fig_twomap, fig_gate, fig_xy, fig_tilt):
        try:
            f()
        except Exception as e:
            print('FAILED', f.__name__, repr(e))
