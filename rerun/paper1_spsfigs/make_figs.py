#!/usr/bin/env python3
"""make_figs.py [--only NAME ...] -- the SPS figures of Paper I/II on the final SPS chain (2026-11-01).

    sps_resolution   det4 intrinsic resolution at H4: the 3-zone telescope decomposition (rerun/sps_junk/results/final_rule.json)
    kernel_forms     the head-on sharing kernel from RAW run71 (results/kernel_<key>.npz, kernel_job.py)
    angled_kernel    run63 flat700 against run63_rot25 d425/d325 on the same builder (results/kernel_<key>.npz)
    known_angle      measured |tan| of the inclined view against the stage angle (results/known_angle.json, tan_known.py)
    attachment       SPS late-charge loss vs the gas fit (rerun/sps_gasfit/results, when it is pushed)

Writes figures/<name>.pdf (+ .png) and results/figs_summary.json (every number a caption or the numbers sheet quotes).
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FIG = os.path.join(HERE, 'figures'); RES = os.path.join(HERE, 'results')
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

XC, YC = '#2a78d6', '#eb6834'
EST_C = {'p0': '#4a4a4a', 'p_mid': '#2a78d6', 'c_pos': '#1a7f37', 'hits': '#b0341d'}
EST_L = {'p0': 'fit p0 (mesh)', 'p_mid': 'fit at 7.5 mm', 'c_pos': 'charge centroid', 'hits': 'hits centroid'}
INK2, MUTED = '#52514e', '#8a8983'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.7, 'axes.axisbelow': True, 'legend.frameon': False,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
BINARY_UM = 780.0 / np.sqrt(12)
SUMMARY = {}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(FIG, name + '.pdf'))


def zfit(zones):
    """sigma_res^2 = sigma_det4^2 + b p^2 on the three telescope zones (as rerun/sps_audit/analyze.py decompose)."""
    P = np.array([z['pitch'] for z in zones]); y = np.array([z['sigma_res'] ** 2 for z in zones])
    ye = np.array([2 * z['sigma_res'] * z['err'] for z in zones])
    X = np.column_stack([np.ones(len(P)), P ** 2]); Ci = np.diag(1 / ye ** 2)
    cov = np.linalg.inv(X.T @ Ci @ X); bb = cov @ (X.T @ Ci @ y)
    return bb, cov


def sps_resolution():
    R = json.load(open(os.path.join(REPO, 'rerun', 'sps_junk', 'results', 'final_rule.json')))
    r63, r53 = R['run63 flat700']['est'], R['run53']['est']
    fig = plt.figure(figsize=(10.5, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)
    pp = np.linspace(0, 1.6, 50)
    for k, v in enumerate('XY'):
        ax = fig.add_subplot(gs[k])
        for e in ('p0', 'p_mid', 'c_pos'):
            V = r63[e][v]; zs = V['zones']
            P = np.array([z['pitch'] for z in zs]); s = np.array([z['sigma_res'] for z in zs]) * 1e3
            se = np.array([z['err'] for z in zs]) * 1e3
            ax.errorbar(P ** 2, s ** 2 / 1e3, yerr=2 * s * se / 1e3, fmt='o', ms=4, color=EST_C[e], capsize=2,
                        label=f"{EST_L[e]}: {V['fit']['sigma_det4']*1e3:.0f} ± {V['fit']['sigma_det4_err']*1e3:.0f} µm"
                              f"  (χ²/1 {V['fit']['chi2']:.1f})")
            bb, _ = zfit(zs)
            ax.plot(pp ** 2, (bb[0] + bb[1] * pp ** 2) * 1e3, '-', color=EST_C[e], lw=1)
        ax.set_xlabel('back-uRWELL pitch² [mm²]'); ax.set_xlim(0, 2.4)
        ax.set_ylabel('σ_res² [10³ µm²]'); ax.set_ylim(0, None)
        ax.set_title(f'run63 flat700, {v} view: intercept = det4', fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
    ax = fig.add_subplot(gs[2])
    cols = [('run63', 'X', r63), ('run63', 'Y', r63), ('run53', 'Y', r53)]
    ests = ['p0', 'p_mid', 'c_pos', 'hits']
    summ = {}
    for i, (run, v, E) in enumerate(cols):
        for j, e in enumerate(ests):
            if e not in E or not E[e].get(v) or not E[e][v].get('fit'):
                continue
            f = E[e][v]['fit']; x = i + (j - 1.5) * 0.17
            ok = f['chi2'] < 4.0                       # 1 dof: the 3-zone model describes the zones (95 % CL)
            ax.errorbar(x, f['sigma_det4'] * 1e3, yerr=f['sigma_det4_err'] * 1e3, fmt='o', ms=5, color=EST_C[e], capsize=2,
                        mfc=EST_C[e] if ok else 'white', label=EST_L[e])
            summ[f'{run} {v} {e}'] = dict(sigma_det4_um=round(f['sigma_det4'] * 1e3, 1), err_um=round(f['sigma_det4_err'] * 1e3, 1),
                                          f_back=round(f['f_back'], 3), chi2_1dof=round(f['chi2'], 2),
                                          zones_um={str(z['pitch']): round(z['sigma_det4'] * 1e3) for z in E[e][v]['zones']})
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker='o', ls='', color=EST_C[e]) for e in ests], [EST_L[e] for e in ests],
              fontsize=7, loc='upper right')
    ax.axhline(BINARY_UM, ls='--', color=MUTED, lw=1); ax.text(2.45, BINARY_UM + 4, 'pitch/√12', fontsize=7, color=MUTED, ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels(['run63 X\nCF₄', 'run63 Y\nCF₄', 'run53 Y\nAr/CO₂'], fontsize=8)
    ax.set_ylabel('det4 intrinsic σ [µm]'); ax.set_ylim(0, 380); ax.set_xlim(-0.5, 2.5)
    ax.set_title('det4 alone (open: zone fit χ²/1 > 4)', fontsize=9)
    save(fig, 'sps_resolution')
    SUMMARY['sps_resolution'] = summ


def _load_kernel(key):
    p = os.path.join(RES, f'kernel_{key}.npz')
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def _boot(P, cols, n=200, seed=1):
    rng = np.random.default_rng(seed); m = np.nanmean(P[:, cols], 0)
    b = np.array([np.nanmean(P[rng.integers(0, len(P), len(P))][:, cols], 0) for _ in range(n)])
    return m, b.std(0)


def kernel_forms():
    keys = [('run71_raw__raw700', '243 V/cm'), ('run71_raw__raw450', '156 V/cm'), ('run71_raw__raw275', '96 V/cm')]
    K = [(k, lab, _load_kernel(k)) for k, lab in keys]
    K = [t for t in K if t[2] is not None]
    if not K:
        print('kernel_forms: no kernel_*.npz yet'); return
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw=dict(wspace=0.32))
    summ = {}
    shades = ['#0b3d91', '#2a78d6', '#8fb8ef']
    for i, (k, lab, Z) in enumerate(K):
        for v, ax, base in (('x', axs[0], XC), ('y', axs[1], YC)):
            P = Z[f'{v}_prof']
            m, s = _boot(P, list(range(7)))
            off = Z['offs']
            ax.errorbar(off + (i - 1) * 0.08, m, yerr=s, fmt='o-', ms=3.5, lw=1, capsize=2,
                        color=shades[i] if v == 'x' else ['#8a3b0f', '#eb6834', '#f5b48a'][i], label=f'{lab} (n={len(P)})')
            r1 = np.nanmean(np.concatenate([P[:, 2], P[:, 4]])); r2 = np.nanmean(np.concatenate([P[:, 1], P[:, 5]]))
            summ[f'{k} {v}'] = dict(n=int(len(P)), r_pm1=round(float(r1), 3), r_pm2=round(float(r2), 3),
                                    r2_over_r1=round(float(r2 / r1), 3),
                                    asym_pm1=round(float(np.nanmean(P[:, 4]) - np.nanmean(P[:, 2])), 3))
    for ax, v in ((axs[0], 'X'), (axs[1], 'Y')):
        ax.set_yscale('log'); ax.set_ylim(5e-3, 1.5); ax.set_xlabel('strip offset from the lead strip')
        ax.set_ylabel('peak amplitude / lead'); ax.set_title(f'{v} view, run71 RAW (no zero suppression)', fontsize=9)
        ax.legend(fontsize=7, loc='lower center')
    # time forms of the 243 V/cm row
    Z = K[0][2]; t = Z['t_rel_ns']; order = list(Z['tr_order'])
    ax = axs[2]
    for v, c in (('x', XC), ('y', YC)):
        T = Z[f'{v}_tr']
        for d, ls in ((0, '-'), (1, '--'), (2, ':')):
            sel = [order.index(d)] + ([order.index(-d)] if d else [])
            tr = np.nanmean(np.concatenate([T[:, s] for s in sel]), 0)
            ax.plot(t, tr if d == 0 else tr / np.nanmax(tr) * np.nanmax(tr), ls, color=c, lw=1.2,
                    label=f'{v.upper()} d=±{d}' if d else f'{v.upper()} lead')
    ax.set_xlabel('time from the lead peak [ns]'); ax.set_ylabel('amplitude / lead peak')
    ax.set_title(f'time forms, {K[0][1]}', fontsize=9); ax.legend(fontsize=7, ncol=2)
    save(fig, 'kernel_forms')
    SUMMARY['kernel_forms'] = summ


def angled_kernel():
    keys = [('run63_flat__flat700', 'flat, 243 V/cm'), ('run63_rot25__d425', '25.4°, 148 V/cm'),
            ('run63_rot25__d325', '25.4°, 113 V/cm')]
    K = [(k, lab, _load_kernel(k)) for k, lab in keys]
    K = [t for t in K if t[2] is not None]
    if not K:
        print('angled_kernel: no kernel_*.npz yet'); return
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw=dict(wspace=0.3))
    summ = {}
    for i, (k, lab, Z) in enumerate(K):
        for v, ax in (('x', axs[0]), ('y', axs[1])):
            P = Z[f'{v}_prof']; m, s = _boot(P, list(range(7)))
            ax.errorbar(Z['offs'] + (i - 1) * 0.08, m, yerr=s, fmt='o-', ms=3.5, lw=1, capsize=2,
                        color=['#4a4a4a', XC if v == 'x' else YC, '#9b59b6'][i], label=f'{lab} (n={len(P)})')
            summ[f'{k} {v}'] = dict(n=int(len(P)), r_m1=round(float(np.nanmean(P[:, 2])), 3),
                                    r_p1=round(float(np.nanmean(P[:, 4])), 3), r_m2=round(float(np.nanmean(P[:, 1])), 3),
                                    r_p2=round(float(np.nanmean(P[:, 5])), 3))
    for ax, v, note in ((axs[0], 'X', 'normal incidence in both mounts'), (axs[1], 'Y', 'carries the rotation')):
        ax.set_yscale('log'); ax.set_ylim(5e-3, 1.5); ax.set_xlabel('strip offset from the lead strip')
        ax.set_ylabel('peak amplitude / lead'); ax.set_title(f'{v} view: {note}', fontsize=9); ax.legend(fontsize=7, loc='lower center')
    save(fig, 'angled_kernel')
    SUMMARY['angled_kernel'] = summ


def known_angle():
    p = os.path.join(RES, 'known_angle.json')
    if not os.path.exists(p):
        print('known_angle: no results/known_angle.json yet'); return
    D = json.load(open(p))
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1]))
    ax = axs[0]
    marks = {'registry': ('o', '#4a4a4a', 'own v (registry)'), 'column': ('s', '#1a7f37', 'v from the column duration'),
             'bench': ('x', '#b0341d', 'no v (bench 33 µm/ns)')}
    summ = {}
    for k, r in sorted(D['rows'].items()):
        for src in ('column', 'registry', 'bench'):
            t = r.get(f'tan_y_{src}')
            if t is None:
                continue
            mk, c, lab = marks[src]
            ax.plot(r['field_Vcm'], abs(t), mk, color=c, ms=4.5 if mk != 'x' else 5,
                    mfc=c if src != 'bench' else 'none', label=lab, alpha=0.9 if r['mount'] != '15.465 deg' else 0.6)
        summ[k] = {kk: r[kk] for kk in r if kk.startswith('tan_') or kk in ('field_Vcm', 'mount', 'v_col', 'v_reg')}
    for ang, c in ((25.4, MUTED), (15.465, '#bbbbbb')):
        ax.axhline(np.tan(np.radians(ang)), ls='--', color=c, lw=1)
        ax.text(250, np.tan(np.radians(ang)) + 0.01, f'tan {ang}°', fontsize=7, color=c, ha='right')
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
    ax.legend(u.values(), u.keys(), fontsize=7, loc='lower right')
    ax.set_xlabel('drift field [V/cm]'); ax.set_ylabel('|median tan θ_y| (inclined view)'); ax.set_ylim(0, 0.6)
    ax.set_title('det4 at a known stage angle', fontsize=9)
    ax = axs[1]
    for k, r in sorted(D['rows'].items()):
        t = r.get('tan_y_column') or r.get('tan_y_registry')
        if t is None:
            continue
        ax.plot(abs(r['tan_x']), abs(t) / np.tan(np.radians(r['stage_deg'])), 'o', ms=4,
                color='#1a7f37' if r.get('tan_y_column') else '#4a4a4a')
    ax.axhline(1, ls='--', color=MUTED, lw=1)
    ax.set_xlabel('|median tan θ_x| (normal view)'); ax.set_ylabel('measured / stage tan (inclined view)')
    ax.set_title('ratio to the stage angle', fontsize=9)
    save(fig, 'known_angle')
    SUMMARY['known_angle'] = dict(rows=summ, headline=D.get('headline'))


def attachment():
    p = os.path.join(REPO, 'rerun', 'sps_gasfit', 'results', 'verdict.json')
    if not os.path.exists(p):
        print('attachment: rerun/sps_gasfit/results/verdict.json not on this branch yet'); return
    from attach_fig import draw          # written once the gas fit's output format is known
    SUMMARY['attachment'] = draw(p, save)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    for name in ('sps_resolution', 'kernel_forms', 'angled_kernel', 'known_angle', 'attachment'):
        if not a.only or name in a.only:
            globals()[name]()
    p = os.path.join(RES, 'figs_summary.json')
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(SUMMARY)
    json.dump(old, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
