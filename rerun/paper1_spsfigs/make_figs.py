#!/usr/bin/env python3
"""make_figs.py [--only NAME ...] -- the SPS figures of Paper I/II on the final SPS chain (2026-11-01).

    sps_resolution   det4 intrinsic resolution at H4 in the small working region R (decisions A + B, 2026-11-01;
                     rerun/sps_region/results/region.json): residuals, telescope subtraction against f, upper bounds
    sps_resolution_zones  SUPERSEDED: the full-beam 3-zone decomposition (rerun/sps_junk/results/final_rule.json)
    kernel_forms     the head-on sharing kernel from RAW run71 (results/kernel_<key>.npz, kernel_job.py)
    angled_kernel    run63 flat700 against run63_rot25 d425/d325 on the same builder (results/kernel_<key>.npz)
    known_angle      measured |tan| of the inclined view against the stage angle (results/known_angle.json, tan_known.py)
    attachment       SPS water and O2 from the gas fit (rerun/sps_gasfit/results/verdict.json; attach_fig.py)

Writes figures/<name>.pdf (+ .png) and results/figs_summary.json (every number a caption or the numbers sheet quotes).
"""
import argparse, json, os, sys
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


Z_DET4, Z_BACK = 1120.0, 1370.0     # rerun/sps_audit/analyze.py: det4 and the telescope back plane along the beam (mm)


def pointing_mm(f, pitch=1.0):
    """sps_audit.analyze.pointing: the telescope pointing at det4 for back-plane resolution f x pitch and a binary
    front plane (1 mm / sqrt 12)."""
    return float(np.sqrt((Z_DET4 / Z_BACK) ** 2 * (f * pitch) ** 2 + (1 - Z_DET4 / Z_BACK) ** 2 * (1.0 / np.sqrt(12)) ** 2))


def sps_resolution():
    """Paper I SPS cross-check (decisions A + B, 2026-11-01): det4 in the small working region R of run 63 flat700
    (rerun/sps_region/results/region.json).  (a)/(b) the X (p_mid) and Y (fit p0) residuals in R, all single-cluster
    telescope tracks and the subsample with a 2-strip back-plane cluster (the model-free upper bound), each with its
    Gaussian core; (c) det4 = sqrt(sigma_res^2 - P(f)^2) against the telescope back-plane f, the f range shaded, and the
    upper bounds.  The superseded full-beam zone fits are sps_resolution_zones()."""
    sys.path.insert(0, os.path.join(REPO, 'papers'))
    import paper_style as ps
    import pandas as pd
    from scipy.optimize import curve_fit
    ps.use()
    G = json.load(open(os.path.join(REPO, 'rerun', 'sps_region', 'results', 'region.json')))
    r63 = G['resolution']['run63']
    fc, frng = G['f_common']['f'], G['f_common']['range']
    V = {'X': (r63['R|p_mid']['X'], 'fit at mid-gap', 'res_b2f2'), 'Y': (r63['R|p0']['Y'], 'fit at the mesh', 'res_b2')}
    for v, (o, _, _) in V.items():
        assert abs(pointing_mm(fc) - o['pointing']) < 1e-4, 'pointing model does not reproduce region.json'
    edges = np.arange(-1.5, 1.5001, 0.05); xc = 0.5 * (edges[1:] + edges[:-1])
    c4 = ps.color('det4'); ink = ps.INK
    # checklist pass 2026-11-02: full width, no panel titles, (a)-(c) labels, one legend above for (a)/(b); the sigma
    # values, the estimators (X at mid-gap, Y at the mesh), f and the upper-bound numbers are in the caption
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axs = ps.fig_for('full', height=2.45, ncols=3, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    rows, summ = [], {}
    g = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
    for ax, v in zip(axs[:2], 'XY'):
        o, est, ub = V[v]
        for key, s_key, mk, lab in (('hist', 'sigma_res', 'o', 'all tracks'), ('hist_b2', 'res_b2', 'o', '2-strip back cluster')):
            h = np.array(o[key], float); hn = h / h.sum() / 0.05
            s = o[s_key] if key == 'hist' else o['res_b2'][0]
            core = np.abs(xc) < 1.2
            (a, m), _ = curve_fit(lambda x, a, m: g(x, a, m, s), xc[core], hn[core], p0=(hn.max(), 0.0))
            full = key == 'hist'
            col = ink if full else ps.ACCENT
            ax.errorbar(xc, hn, yerr=np.sqrt(np.maximum(h, 1)) / h.sum() / 0.05, fmt=mk, ms=2.4, lw=0.5, color=col,
                        mfc=col if full else 'white', mec=col, zorder=4 if full else 3)
            xx = np.linspace(-1.2, 1.2, 200)
            ax.plot(xx, g(xx, a, m, s), '-' if full else '--', color=ps.MODEL, lw=0.9, zorder=5)
            for xi, yi, hi in zip(xc, hn, h):
                rows.append(dict(panel='a' if v == 'X' else 'b', view=v, sample=key, residual_mm=xi, density=yi, count=hi))
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 3.25)
        ax.set_xlabel(f'{v} residual $\\Delta {v.lower()}$ (mm)')
    axs[0].set_ylabel('density (mm$^{-1}$)')
    ps.legend_above(fig, [Line2D([], [], ls='', marker='o', ms=3, color=ink),
                          Line2D([], [], ls='', marker='o', ms=3, color=ps.ACCENT, mfc='white'),
                          Line2D([], [], color=ps.MODEL, lw=0.9), Patch(fc=ps.ACCENT, alpha=0.18, lw=0)],
                    ['all tracks', '2-strip back-plane cluster', 'Gaussian core', 'upper bound (2-strip)'],
                    ncol=4, handlelength=1.6)
    ax = axs[2]
    ff = np.linspace(0.20, 0.34, 141)
    ax.axvspan(frng[0], frng[1], color=ps.GRID, lw=0, zorder=0)
    ax.axvline(fc, color=ps.MUTED, lw=0.7, zorder=1)
    for v in 'XY':
        o, est, ub = V[v]
        s = o['sigma_res']
        d4 = np.array([np.sqrt(max(s ** 2 - pointing_mm(f) ** 2, 0)) for f in ff]) * 1e3
        kw = ps.det('det4', view=v.lower())
        ax.plot(ff, d4, ls=kw['ls'], color=ink, lw=1.2, zorder=3)
        ax.errorbar([fc], [o['sigma_det4'] * 1e3], yerr=[o['stat'] * 1e3], fmt='o', ms=4.5, color=ink,
                    mfc=ink if v == 'X' else 'white', mec=ink, zorder=5)
        b, be = o[ub][0] * 1e3, o[ub][1] * 1e3
        ax.axhspan(b - be, b + be, color=ps.ACCENT, alpha=0.18 if v == 'X' else 0.10, lw=0, zorder=1)
        ax.axhline(b, color=ps.ACCENT, lw=0.9, ls=kw['ls'], zorder=2)
        for fi, di in zip(ff[::10], d4[::10]):
            rows.append(dict(panel='c', view=v, f=fi, sigma_det4_um=di, upper_bound_um=b, upper_bound_err_um=be))
        summ[v] = dict(estimator=est, sigma_res_um=round(s * 1e3, 1), pointing_um=round(o['pointing'] * 1e3, 1),
                       sigma_det4_um=round(o['sigma_det4'] * 1e3, 1), stat_um=round(o['stat'] * 1e3, 1),
                       f_range_um=[round(x * 1e3, 1) for x in o['sys_range']], upper_bound_um=[round(b, 1), round(be, 1)],
                       upper_bound_sample=ub, n_core=o['n_core'])
    ax.set_xlim(0.20, 0.34); ax.set_ylim(0, 300)
    ax.set_xlabel(r"back-plane $f$ ($\sigma$ / pitch)"); ax.set_ylabel('det4 resolution $\\sigma$ (µm)')
    ax.legend(ps.view_handles(color=ink), ['X', 'Y'], loc='upper right')
    ps.panel_labels(axs)
    ps.save(fig, os.path.join(FIG, 'sps_resolution'), data=pd.DataFrame(rows))
    print('wrote', os.path.join(FIG, 'sps_resolution.pdf'))
    summ['f'] = dict(f=round(fc, 4), range=[round(x, 4) for x in frng])
    summ['region'] = G['region']
    SUMMARY['sps_resolution'] = summ


def sps_resolution_zones():
    """SUPERSEDED (2026-11-01, decisions A + B): the run63 full-beam 3-zone decomposition.  Kept for the record."""
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
    save(fig, 'sps_resolution_zones')
    SUMMARY['sps_resolution_zones'] = summ


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
             'bench': ('x', '#b0341d', 'no v (bench 33 µm/ns): not an angle')}
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
    ax.axhspan(0.97, 1.03, color='#e6e5e1', lw=0)
    for k, r in sorted(D['rows'].items()):
        if r['stage_deg'] != 25.4:
            continue
        for src, mk, c in (('registry', 'o', '#4a4a4a'), ('column', 's', '#1a7f37')):
            t = r.get(f'tan_y_{src}')
            if t is not None:
                ax.plot(r['field_Vcm'] + (2 if src == 'column' else -2), abs(t) / np.tan(np.radians(25.4)), mk, ms=4.5, color=c)
    ax.axhline(1, ls='--', color=MUTED, lw=1)
    ax.set_xlabel('drift field [V/cm]'); ax.set_ylabel('measured / tan 25.4° (inclined view)'); ax.set_ylim(0.5, 1.1)
    ax.set_title('25.4° mount: the column-duration v closes it', fontsize=9)
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
    for name in ('sps_resolution', 'sps_resolution_zones', 'kernel_forms', 'angled_kernel', 'known_angle', 'attachment'):
        if not a.only or name in a.only:
            with plt.rc_context():          # paper_style's ps.use() inside a producer must not leak into the next one
                globals()[name]()
    p = os.path.join(RES, 'figs_summary.json')
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(SUMMARY)
    json.dump(old, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
