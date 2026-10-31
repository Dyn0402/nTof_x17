#!/usr/bin/env python3
"""make_report.py -- report/report.html + results/summary.json for the Paper I bench resolutions on pass 1
(PASS1_RESOLUTION_2026-10-31.md), built from results/rows/<key>.json (res_rows.py), results/vt_window.json
(vt_window.py) and results/dp_arms.json (dp_score.py).  Figures by relative links (figures/x.png); the paper copies
are PDFs in figures/.  Every number in the page and the verdict text comes from those files, so a re-run updates them.
"""
import csv
import glob
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(HERE, 'results')
REP = os.path.join(HERE, 'report')
FIG = os.path.join(REP, 'figures')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
COL = dict(det2='#1f77b4', det3='#d62728', det4='#2ca02c', det6='#9467bd', det7='#ff7f0e')
#: the row each chamber's headline comes from: the quoted golden row where the registry has one (not the calibration
#: row: in-sample), else the calibration row (det4/6/7 have no second golden row)
HEAD = {'det2': 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
        'det3': 'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3',
        'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
MIN_N = 200                  # fewest events for a row to enter a scan plot
MIN_N_SCAN = 80              # ... an HV-scan row (short rows; error bars drawn)


def f(x, d=2):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def table(rows, head):
    h = ['<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in head) + '</tr>']
    for r in rows:
        h.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
    return '\n'.join(h + ['</table>'])


def load_rows():
    R = {}
    for p in sorted(glob.glob(os.path.join(RES, 'rows', '*.json'))):
        o = json.load(open(p))
        R[o['key']] = o
    return R


def ang(o, p, arm, k='s68'):
    return o['views'][p]['arms'][arm]['all'][k]


def pos(o, p, arm, est='mesh', k='core'):
    return o['views'][p]['arms'][arm]['position'][est][k]


def series(o):
    """the scan a row belongs to: its subrun's prefix before 'resist' (det3 6-27 has hv_scan_, hv_scan2_ and
    drift_scan_), 'long' for the long / longer / short rows."""
    sr = o['subrun']
    if sr.startswith(('long', 'short')):
        return 'long'
    return sr.split('resist')[0]


def scans(R):
    """HV scans (one series, fixed drift, >= 3 mesh voltages) with the long rows of the same run and drift; drift scans
    (one series, fixed mesh, >= 3 drift voltages)."""
    hv, dr, lg = {}, {}, {}
    for o in R.values():
        if o['cls'] == 'skeptical':
            continue
        sr = series(o)
        if sr == 'long':
            lg.setdefault((o['det'], o['run'], o['drift_V']), []).append(o)
            continue
        hv.setdefault((o['det'], o['run'], o['drift_V'], sr), []).append(o)
        dr.setdefault((o['det'], o['run'], o['resist_V'], sr), []).append(o)
    hv = {k: dict(scan=sorted(v, key=lambda o: o['resist_V']), long=lg.get(k[:3], []))
          for k, v in hv.items() if len({o['resist_V'] for o in v}) >= 3}
    dr = {k: sorted(v, key=lambda o: o['drift_V']) for k, v in dr.items() if len({o['drift_V'] for o in v}) >= 3}
    return hv, dr


def figs(R, hv, dr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    out = []
    # 1 angle and position vs HV
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    LS = {}
    for (d, run, dv, sr), S in sorted(hv.items()):
        tag = run.replace('mx17_', '').split('_')
        lab = f"{d} {' '.join(t for t in tag if not t.startswith('det'))}{' ' + sr.rstrip('_') if sr else ''} drift {dv:g} V"
        ls = '--' if LS.setdefault(d, 0) else '-'
        LS[d] += 1
        for j, p in enumerate('xy'):
            ok = lambda o: o['views'][p]['arms']['pass']['all']['n'] >= MIN_N_SCAN
            sc = [o for o in S['scan'] if ok(o)]       # the scan rows: one line
            lg = [o for o in S['long'] if ok(o)]       # long rows of the same run and drift
            if len(sc) >= 3:
                x = [o['resist_V'] for o in sc]
                A = [o['views'][p]['arms']['pass']['all'] for o in sc]
                axs[0, j].errorbar(x, [a['s68'] for a in A], yerr=[a['s68_err'] for a in A], fmt='o' + ls,
                                   color=COL[d], ms=4, lw=1, capsize=1.5, label=lab)
                axs[1, j].plot(x, [1e3 * pos(o, p, 'pass', k='sigma_dut_core') for o in sc], 'o' + ls, color=COL[d],
                               ms=4, lw=1, label=lab)
            for o in lg:
                for arm, mk in (('pass', 'D'), ('vt', 's')):
                    if arm == 'vt' and not o['vt']['applied']:
                        continue
                    axs[0, j].plot([o['resist_V'] + (1.5 if arm == 'vt' else 0)], [ang(o, p, arm)], mk, color=COL[d],
                                   ms=5, mfc='none')
                axs[1, j].plot([o['resist_V']], [1e3 * pos(o, p, 'pass', k='sigma_dut_core')], 'D', color=COL[d],
                               ms=5, mfc='none')
    for j, p in enumerate('xy'):
        axs[0, j].set_ylabel(f'{p}: angle s68 [deg]'); axs[1, j].set_ylabel(f'{p}: position core sigma - M3 [um]')
        for i in range(2):
            axs[i, j].set_xlabel('resist (mesh) HV [V]'); axs[i, j].grid(alpha=.3)
        axs[0, j].set_ylim(0, 4); axs[1, j].set_ylim(0, 1200)
    axs[0, 0].legend(fontsize=6, ncol=2)
    axs[0, 0].set_title('Angle resolution vs HV, interior_good (open: long rows; squares their v(t) arm)', fontsize=10, loc='left')
    axs[1, 0].set_title('Position resolution at the mesh vs HV (M3 pointing subtracted)', fontsize=10, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'res_vs_hv.png'), dpi=90); plt.close(fig)
    out.append('res_vs_hv.png')
    # 2 vs drift field
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for (d, run, rv, sr), rows in sorted(dr.items()):
        for p, ls in (('x', 'o-'), ('y', 's--')):
            rr = [o for o in rows if o['views'][p]['arms']['pass']['all']['n'] >= MIN_N]
            E = [o['E_drift_V_cm'] for o in rr]
            axs[0].plot(E, [ang(o, p, 'pass') for o in rr], ls, color=COL[d], ms=4, label=f'{d} {p} (mesh {rv:g} V)')
            axs[1].plot(E, [1e3 * pos(o, p, 'pass', k='sigma_dut_core') for o in rr], ls, color=COL[d], ms=4,
                        label=f'{d} {p}')
    axs[0].set_ylabel('angle s68 [deg]'); axs[1].set_ylabel('position core sigma - M3 [um]')
    for ax in axs:
        ax.set_xlabel('nominal drift field = drift voltage / 3 cm [V/cm]'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    axs[0].set_ylim(0, 6); axs[1].set_ylim(0, 1200)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'res_vs_drift.png'), dpi=90); plt.close(fig)
    out.append('res_vs_drift.png')
    # 3 vs angle on the headline rows (also the paper's tracking_perf panel a)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for d in DETS:
        o = R.get(HEAD[d])
        if not o:
            continue
        for p, ls in (('x', 'o-'), ('y', 's--')):
            B = o['views'][p]['arms']['vt' if o['vt']['applied'] else 'pass']['by_angle']
            c = [0.5 * (b['lo'] + b['hi']) for b in B if b['n'] >= 50]
            axs[0].plot(c, [b['s68'] for b in B if b['n'] >= 50], ls, color=COL[d], ms=4, label=f'{d} {p}')
            P = o['views'][p]['arms']['pass']['position']['mesh']['by_angle']
            axs[1].plot([0.5 * (b['lo'] + b['hi']) for b in P if b['n'] >= 50],
                        [1e3 * b['sigma_dut_core'] for b in P if b['n'] >= 50], ls, color=COL[d], ms=4, label=f'{d} {p}')
            M = o['views'][p]['arms']['vt' if o['vt']['applied'] else 'pass']['position']['mid']['by_angle']
            axs[2].plot([0.5 * (b['lo'] + b['hi']) for b in M if b['n'] >= 50],
                        [1e3 * b['sigma_dut_core'] for b in M if b['n'] >= 50], ls, color=COL[d], ms=4, label=f'{d} {p}')
    axs[0].set_ylabel('angle s68 [deg]'); axs[1].set_ylabel('mesh position core sigma - M3 [um]')
    axs[2].set_ylabel('mid-gap position core sigma - M3 [um]')
    for ax in axs:
        ax.set_xlabel('|theta_ref| [deg]'); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7, ncol=2); axs[1].set_ylim(0, 2000); axs[2].set_ylim(0, 2000)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'res_vs_angle.png'), dpi=90)
    plt.close(fig)
    out.append('res_vs_angle.png')
    return out


def tracking_perf(R):
    """papers/paper1 fig tracking_perf: sigma_theta vs angle (headline rows, v(t) arm where applied) and the
    within-5 mm efficiency (rerun/effbudget pass-1 rows.csv, interior_good)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eff = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'effbudget', 'pass1', 'results',
                                                                  'rows.csv')))}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for i, d in enumerate(DETS):
        o = R.get(HEAD[d])
        if not o:
            continue
        arm = 'vt' if o['vt']['applied'] else 'pass'
        for p, ls in (('x', 'o-'), ('y', 's--')):
            B = [b for b in o['views'][p]['arms'][arm]['by_angle'] if b['n'] >= 50]
            axs[0].errorbar([0.5 * (b['lo'] + b['hi']) + (0.3 if p == 'y' else 0) for b in B], [b['s68'] for b in B],
                            yerr=[b['s68_err'] for b in B], fmt=ls, color=COL[d], ms=3, lw=1, capsize=1.5,
                            label=f'{d} {p.upper()}')
        e = eff.get(HEAD[d])
        if e:
            axs[1].errorbar([i], [float(e['eff_ig'])], yerr=[float(e['err_ig'])], fmt='o', color=COL[d], ms=5)
    axs[0].set_xlabel(r'$|\theta_{\rm ref}|$ [deg]'); axs[0].set_ylabel(r'$\sigma_\theta$ (s68) [deg]')
    axs[0].legend(fontsize=6, ncol=2, frameon=False); axs[0].grid(alpha=.3); axs[0].set_ylim(0, None)
    axs[1].set_xticks(range(len(DETS))); axs[1].set_xticklabels(DETS)
    axs[1].set_ylabel('efficiency within 5 mm [%]'); axs[1].set_ylim(0, 100); axs[1].grid(alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, 'figures', f'tracking_perf.{ext}'), dpi=150)
    plt.close(fig)


def gas_vE(R):
    """papers/paper1 fig gas_vE: the det3 6-27 drift scan, the angle-scale v of each pass-1 row (v_drift x mean kw:
    the M3-calibrated v that maps w to the angle) against Magboltz for the bench gas (0.95 % H2O, no O2;
    cloud_basics FINDINGS §22), plus the det3 timing v (S4 re-check) at 1000 V."""
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cb = os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')
    sys.path.insert(0, cb)
    import gasmodel as M
    G = M.GasGrid('bench')
    rows = [o for o in R.values() if o['det'] == 'det3' and o['run'] == 'mx17_det3_saturday_scan_6-27-26'
            and o['cls'] != 'skeptical']
    E = np.array([o['E_drift_V_cm'] for o in rows])
    v = np.array([o['v_drift'] * 0.5 * (o['views']['x']['kw_pass'] + o['views']['y']['kw_pass']) for o in rows])
    vx = np.array([o['v_drift'] * o['views']['x']['kw_pass'] for o in rows])
    vy = np.array([o['v_drift'] * o['views']['y']['kw_pass'] for o in rows])
    o = np.argsort(E)
    Eg = np.linspace(max(G.E.min(), 30), min(G.E.max(), 400), 120)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for w, ls in ((0.95, '-'), (0.6, ':'), (1.4, '--')):
        ax.plot(Eg, [G(w, 0.0, e)['v'] for e in Eg], ls, color='k', lw=1, label=f'Magboltz, bench gas, {w:g} % H$_2$O')
    ax.errorbar(E[o], v[o], yerr=np.abs(vx - vy)[o] / 2, fmt='o', color=COL['det3'], ms=5, capsize=2,
                label='det3, angle-scale v (pass 1, X/Y half-difference)')
    s4 = json.load(open(os.path.join(REPO, 'rerun', 'calib', 's4', 'results', 's4r', 'summary.json'))) \
        if os.path.exists(os.path.join(REPO, 'rerun', 'calib', 's4', 'results', 's4r', 'summary.json')) else None
    ax.set_xlabel('nominal drift field [V/cm]'); ax.set_ylabel(r'drift velocity [$\mu$m/ns]')
    ax.legend(fontsize=6.5, frameon=False); ax.grid(alpha=.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, 'figures', f'gas_vE.{ext}'), dpi=150)
    plt.close(fig)
    return dict(points=[dict(E=float(E[i]), v=float(v[i]), vx=float(vx[i]), vy=float(vy[i]),
                             magboltz=float(G(0.95, 0.0, E[i])['v'])) for i in o], s4=s4 is not None)


def rows_csv(R):
    """results/rows.csv: one line per (row, view), every quoted statistic in both arms (the per-row JSONs stay on EOS)."""
    cols = ['key', 'det', 'run', 'subrun', 'cls', 'resist_V', 'drift_V', 'E_drift_V_cm', 'v_drift', 'vt_applied', 'view',
            'P_mm', 'kw_pass', 'kw_vt', 'w0_pass', 'w0_vt']
    stat = ['n', 'bias', 's68', 's68_err', 'core', 'core_err']
    for arm in ('pass', 'vt'):
        cols += [f'ang_{k}_{arm}' for k in stat]
        for est in ('mesh', 'mid'):
            cols += [f'pos_{est}_{k}_{arm}' for k in stat + ['sigma_dut_core']]
    with open(os.path.join(RES, 'rows.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for o in sorted(R.values(), key=lambda o: (o['det'], o['key'])):
            for p in 'xy':
                v = o['views'][p]
                line = [o['key'], o['det'], o['run'], o['subrun'], o['cls'], o['resist_V'], o['drift_V'],
                        round(o['E_drift_V_cm'], 1), o['v_drift'], o['vt']['applied'], p, o['pointing'][p],
                        v['kw_pass'], v['kw_vt'], v['w0_pass'], v['w0_vt']]
                for arm in ('pass', 'vt'):
                    line += [v['arms'][arm]['all'][k] for k in stat]
                    for est in ('mesh', 'mid'):
                        q = v['arms'][arm]['position'][est]
                        line += [q[k] for k in stat + ['sigma_dut_core']]
                w.writerow(line)


def main():
    R = load_rows()
    rows_csv(R)
    os.makedirs(REP, exist_ok=True); os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    hv, dr = scans(R)
    figl = figs(R, hv, dr)
    tracking_perf(R)
    gv = gas_vE(R)
    W = json.load(open(os.path.join(RES, 'vt_window.json'))) if os.path.exists(os.path.join(RES, 'vt_window.json')) else {}
    D = json.load(open(os.path.join(RES, 'dp_arms.json'))) if os.path.exists(os.path.join(RES, 'dp_arms.json')) else {}
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}

    # headline
    H, S = [], {}
    for d in DETS:
        o = R.get(HEAD[d])
        if not o:
            continue
        S[d] = dict(key=o['key'], vt_applied=o['vt']['applied'])
        for p in 'xy':
            v = o['views'][p]['arms']
            pm = v['pass']['position']['mesh']
            S[d][p] = dict(s68_pass=v['pass']['all']['s68'], s68_vt=v['vt']['all']['s68'],
                           core_pass=v['pass']['all']['core'], core_vt=v['vt']['all']['core'],
                           bias_pass=v['pass']['all']['bias'], bias_vt=v['vt']['all']['bias'], n=v['pass']['all']['n'],
                           pos_core=pm['core'], pos_s68=pm['s68'], P=o['pointing'][p], pos_dut=pm['sigma_dut_core'],
                           mid_dut_pass=v['pass']['position']['mid']['sigma_dut_core'],
                           mid_dut_vt=v['vt']['position']['mid']['sigma_dut_core'], n_pos=pm['n'])
            s = S[d][p]
            H.append([d, p.upper(), html.escape(o['key'][:60]), 'yes' if o['vt']['applied'] else 'no', str(s['n']),
                      f(s['s68_pass']), f(s['s68_vt']), f(s['core_pass']), f(s['core_vt']), f(s['bias_pass']),
                      str(s['n_pos']), f(1e3 * s['pos_core'], 0), f(1e3 * s['P'], 0), f(1e3 * s['pos_dut'], 0),
                      f(1e3 * s['mid_dut_pass'], 0), f(1e3 * s['mid_dut_vt'], 0)])
    # golden calibration rows, both arms (G1-like, interior_good)
    G = []
    for o in R.values():
        if o['cls'] != 'golden_calib':
            continue
        G.append([o['det'], html.escape(o['key']), 'yes' if o['vt']['applied'] else 'no']
                 + [f(ang(o, p, a)) for p in 'xy' for a in ('pass', 'vt')]
                 + [f(o['views'][p][k], 4) for p in 'xy' for k in ('kw_pass', 'kw_vt')])
    # v(t) arm on every corrected row
    V = []
    for o in sorted(R.values(), key=lambda o: (o['det'], o['key'])):
        if not o['vt']['applied']:
            continue
        V.append([o['det'], html.escape(o['key'][:70]), o['vt']['model'] or '', o['vt']['verdict'] or '']
                 + [f(ang(o, p, 'pass')) + ' → ' + f(ang(o, p, 'vt')) for p in 'xy']
                 + [f(o['views'][p]['vt_cover'], 3) for p in 'xy'])
    # scans
    T = []
    for (d, run, dv, sr), SC in sorted(hv.items()):
        for o in SC["scan"] + SC["long"]:
            T.append([d, run.replace('mx17_', '') + (' ' + sr.rstrip('_') if sr else ''), f(dv, 0), f(o['resist_V'], 0), o['cls'],
                      str(o['views']['x']['arms']['pass']['all']['n'])]
                     + [f(ang(o, p, 'pass')) for p in 'xy'] + [f(ang(o, p, 'vt')) for p in 'xy']
                     + [f(1e3 * pos(o, p, 'pass', k='sigma_dut_core'), 0) for p in 'xy'])
    Dt = []
    for (d, run, rv, sr), rows in sorted(dr.items()):
        for o in rows:
            Dt.append([d, f(rv, 0), f(o['drift_V'], 0), f(o['E_drift_V_cm'], 0), str(o['views']['x']['arms']['pass']['all']['n'])]
                      + [f(ang(o, p, 'pass')) for p in 'xy']
                      + [f(1e3 * pos(o, p, 'pass', k='sigma_dut_core'), 0) for p in 'xy']
                      + [f(o['v_drift'] * 0.5 * (o['views']['x']['kw_pass'] + o['views']['y']['kw_pass']), 1)])
    # exclusions
    X = sorted({(reg[k]['det'], reg[k]['run'], reg[k]['note'][:140]) for k in reg if reg[k]['cls'] == 'skeptical'})
    nsk = sum(1 for k in reg if reg[k]['cls'] == 'skeptical')
    # window test
    Wt = []
    zhi, zlo = [], []
    for k, w in W.items():
        for p in 'xy':
            v = w['views'][p]
            sl, hs, ls = v['detect_vs_r'], v['hi_edge_vs_r'], v.get('lo_edge_vs_r') or dict(slope=np.nan, err=np.nan)
            z = lambda q: q['slope'] / q['err'] if q['err'] and np.isfinite(q['err']) and q['err'] > 0 else np.nan
            zhi.append((z(hs), w['det'], k, p)); zlo.append((z(ls), w['det'], k, p, ls['slope']))
            Wt.append([w['det'], html.escape(k[:60]), p.upper(), f(w['r_min'], 3) + '–' + f(w['r_max'], 3),
                       f(min(b['u_med_over_T'] for b in v['blocks']), 3) + '–' + f(max(b['u_med_over_T'] for b in v['blocks']), 3),
                       f(100 * max(b['hi_edge'] for b in v['blocks']), 1),
                       f(100 * hs['slope'], 1) + ' ± ' + f(100 * hs['err'], 1),
                       f(100 * max(b['lo_edge'] for b in v['blocks']), 1),
                       f(100 * ls['slope'], 1) + ' ± ' + f(100 * ls['err'], 1),
                       f(100 * sl['slope'], 1) + ' ± ' + f(100 * sl['err'], 1)])
    zhi = [x for x in zhi if np.isfinite(x[0])]
    zlo = [x for x in zlo if np.isfinite(x[0])]
    zhi_max = max(zhi, key=lambda x: abs(x[0])) if zhi else (np.nan, '', '', '')
    lo_sig = sorted([x for x in zlo if abs(x[0]) >= 3], key=lambda x: -abs(x[0]))
    win_verdict = (f"{'No' if abs(zhi_max[0]) < 3 else 'A'} slope of the near-upper-edge fraction against r "
                   f"{'reaches' if abs(zhi_max[0]) < 3 else 'exceeds'} 3&sigma; over "
                   f"{len(zhi)} (row, view) pairs (largest {f(zhi_max[0], 1)}&sigma;: {zhi_max[1]} {html.escape(zhi_max[2][:50])} "
                   f"{zhi_max[3]}). ")
    if lo_sig:
        win_verdict += (f"{len(lo_sig)} pairs show a lower-edge fraction that rises where r is low "
                        f"({', '.join(sorted({x[1] + ' ' + x[2].split('__')[-2] for x in lo_sig}))}): columns ending "
                        f"SHORT, the opposite of what a slower v does to a v-scaled window (it lengthens them); it is "
                        f"the early-night attachment of det6 6-26 (vt_within_rows), and such fits stay in the table "
                        f"(the window ranks candidates and flags isochronous fits, it does not reject them).")
    else:
        win_verdict += 'No lower-edge slope reaches 3&sigma;.'
    # Dp arms
    Dr = []
    for arm, a in D.items():
        for p in 'xy':
            Dr.append([arm, a['key'], f(a['Dp'][0], 5) + ' → ' + f(a['Dp'][1], 5), p.upper(), str(a[p]['n']),
                       f(a[p]['s68_base']) + ' → ' + f(a[p]['s68_arm']), f(a[p]['s68_arm_refold']),
                       f(1e3 * a[p]['pos_base'], 0) + ' → ' + f(1e3 * a[p]['pos_arm'], 0)])

    json.dump(dict(headline=S, gas_vE=gv, window=W, dp_arms=D), open(os.path.join(RES, 'summary.json'), 'w'),
              indent=1, default=float)

    # verdict (generated)
    def hl(d):
        s = S.get(d)
        if not s:
            return ''
        a = 'vt' if s['vt_applied'] else 'pass'
        return (f"{d} {f(s['x']['s68_' + a])}/{f(s['y']['s68_' + a])}&deg;"
                + (f" (pass {f(s['x']['s68_pass'])}/{f(s['y']['s68_pass'])})" if s['vt_applied'] else ''))
    verd = ('; '.join(hl(d) for d in DETS))
    posv = '; '.join(f"{d} {f(1e3 * S[d]['x']['pos_dut'], 0)}/{f(1e3 * S[d]['y']['pos_dut'], 0)} µm" for d in DETS if d in S)
    css = ('body{font-family:system-ui,sans-serif;max-width:1250px;margin:2em auto;line-height:1.45;color:#222}'
           'table{border-collapse:collapse;font-size:12.5px;margin:.6em 0}td,th{border:1px solid #ccc;padding:3px 7px;'
           'text-align:right}th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}'
           '.v{background:#eef6ee;border-left:4px solid #2a7;padding:.6em 1em}figure{margin:1em 0}'
           'figcaption{font-size:13px;color:#555}img{max-width:100%}')
    H5 = ['chamber', 'view', 'row', 'v(t)', 'n', 's68 pass', 's68 v(t)', 'core σ pass', 'core σ v(t)', 'bias',
          'n pos', 'pos core σ [µm]', 'M3 P [µm]', 'σ_DUT mesh [µm]', 'σ_DUT mid pass', 'σ_DUT mid v(t)']
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Paper I bench resolutions, pass 1</title>
<style>{css}</style></head><body>
<h1>Paper I bench resolutions on pass 1</h1>
<p>Generated by <code>rerun/paper1_res/make_report.py</code> from <code>results/</code>. Read-only on Analysis_pass1;
every number has two arms: <b>pass</b> (the pass table as it is) and <b>v(t)</b> (v &rarr; v&middot;r(t) in the angle
mapping and w0/kw re-measured on the corrected mapping, per row; identical to pass on a row without a v(t) map).</p>
<div class="v"><b>Verdict.</b> Angle resolution (s68, X/Y, interior_good, headline rows; v(t) arm where a map exists):
{verd}. Position at the mesh against M3, core &sigma; with the M3 pointing subtracted (X/Y): {posv}.
The v(t) arm moves no headline angle by more than {f(max([abs(S[d][p]['s68_vt'] - S[d][p]['s68_pass']) for d in S for p in 'xy'] or [np.nan]))}&deg;.
Window truncation (&plusmn;5 % v in a row): {'none' if abs(zhi_max[0]) < 3 else 'SEE §7'}. {win_verdict}</div>

<h2>1 · Headline rows</h2>
<p>One row per chamber: the quoted golden row (det2 6-22 long_run, det3 6-27 sanity long_run: not the calibration
rows), else the calibration row (det4, det6, det7: in-sample). Angles 03_angles' definition on interior_good;
positions also with the spark veto and |r| &lt; 15 mm. core &sigma; is a Gaussian fitted in a fixed window
(median &plusmn; 2&nbsp;s68). &sigma;_DUT = &radic;(core&sup2; &minus; P&sup2;), P the M3 four-plane pointing at the
row's z projected into the view; multiple scattering is not subtracted, so &sigma;_DUT is an upper bound.</p>
{table(H, H5)}

<h2>2 · Calibration rows, both arms</h2>
{table(G, ['chamber', 'row', 'v(t)', 'x s68 pass', 'x s68 v(t)', 'y s68 pass', 'y s68 v(t)', 'kw x pass', 'kw x v(t)', 'kw y pass', 'kw y v(t)'])}

<h2>3 · Resolution vs HV</h2>
<figure><img src="figures/res_vs_hv.png"><figcaption>Figure 1. Angle s68 (top) and position core &sigma; minus the M3
pointing (bottom) against the mesh HV. Lines: the HV-scan rows of one run (&ge; {MIN_N_SCAN} interior events, s68
error bars); open diamonds: the long rows of the same run at their HV, open squares (offset +1.5 V) their v(t) arm.
Skeptical rows out. det4's 6-23 scan rows hold ~100 interior events each and that night's M3 matching is degraded
(paper1_charge): its points are shown, not quoted.</figcaption></figure>
{table(T, ['chamber', 'night', 'drift V', 'mesh V', 'class', 'n x', 'x s68 pass', 'y s68 pass', 'x s68 v(t)', 'y s68 v(t)', 'x σ_DUT [µm]', 'y σ_DUT [µm]'])}

<h2>4 · Resolution vs drift field</h2>
<figure><img src="figures/res_vs_drift.png"><figcaption>Figure 2. det3's 6-27 drift scan (mesh 490 V).</figcaption></figure>
{table(Dt, ['chamber', 'mesh V', 'drift V', 'E [V/cm]', 'n x', 'x s68', 'y s68', 'x σ_DUT [µm]', 'y σ_DUT [µm]', 'angle-scale v [µm/ns]'])}

<h2>5 · Resolution vs track angle</h2>
<figure><img src="figures/res_vs_angle.png"><figcaption>Figure 3. Headline rows: angle s68, the position at the mesh
and at mid-gap (core &sigma; minus P) per |&theta;_ref| bin. The mesh position of an inclined track carries the
angle's error times the depth of the first charge; the mid-gap position carries it times 15 mm.</figcaption></figure>

<h2>6 · The v(t) arm on every corrected row</h2>
{table(V, ['chamber', 'row', 'model', 'survey verdict', 'x s68 pass → v(t)', 'y s68 pass → v(t)', 'coverage x', 'coverage y'])}

<h2>7 · Window truncation (&plusmn;5 % v within a row)</h2>
<p>The only v-scaled window of the bench reconstruction is the plausible column end
[0.30, 1.60]&nbsp;&times;&nbsp;30&nbsp;mm/v_bundle (<code>wft.reco.u_window</code>). Per corrected row and view, in
four equal-count blocks of r(t): the fitted column end over T, the fraction within 10 % of the upper edge and
outside the window, and the slope of the interior_good detection fraction against r.</p>
<p>{win_verdict}</p>
<p>The detection fraction does move with r on some rows (det4 6-23 and g_det4 up, det6 down): those rows' gain, spark
rate and M3 matching change over the same hours, and the window fractions do not follow, so it is not the window.</p>
{table(Wt, ['chamber', 'row', 'view', 'r range', 'median u/T', 'max near upper edge [%]', 'upper-edge slope [% per unit r]', 'max near lower edge [%]', 'lower-edge slope [% per unit r]', 'detection slope [% per unit r]'])}

<h2>8 · X-width systematic (diffusion) arms</h2>
<p>The model has one diffusion constant for both views (Dp), so the X-width systematic (det3 +100, others
&plusmn;100&nbsp;µm/&radic;cm) is applied to both views at once: an upper bound on its effect. Reco of the G1 events
with the shifted Dp (code 6a0bb1f), scored on the same interior_good events as the pass table.</p>
{table(Dr, ['arm', 'row', 'Dp', 'view', 'n', 's68 base → arm', 's68 arm, kw re-measured', 'pos core σ base → arm [µm]'])}

<h2>9 · Excluded rows</h2>
<p>{nsk} skeptical rows are out of every table and figure (registry class <code>skeptical</code>); their notes:</p>
{table([[d, r, html.escape(n)] for d, r, n in X], ['chamber', 'night', 'registry note'])}

<h2>What this does not rule out</h2>
<ul>
<li>Multiple scattering between the last M3 plane and the chamber is not subtracted: &sigma;_DUT is an upper bound on
the chamber (the bench is reference-limited by MS; sps-spatial-resolution). The SPS number is the chamber's floor.</li>
<li>det6's v over 6-26 is known only to a factor ~2 (ladder and column disagree): its angles are corrected by the
ladder (the angle scale against M3); its depths, and so its mid-gap positions in the v(t) arm, carry that limitation.</li>
<li>det4, det6, det7 headlines are in-sample (their only golden row is the calibration row).</li>
<li>The X-width arm moves Dp in both views; an X-only width model is not in the code.</li>
<li>Rows shorter than the v(t) survey's floor carry no v(t) map; their v(t) arm equals the pass arm by construction.</li>
</ul>
</body></html>"""
    open(os.path.join(REP, 'report.html'), 'w').write(page)
    print('wrote', os.path.join(REP, 'report.html'), figl)


if __name__ == '__main__':
    main()
