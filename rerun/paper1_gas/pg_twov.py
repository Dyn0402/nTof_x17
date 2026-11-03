#!/usr/bin/env python3
"""pg_twov.py -- Paper I §6 "two drift velocities and the gap" (Dylan's outline, 2026-11-02).

The angle uses only v: a track of slope tan(theta) = dx/dz drifts its charge across the strips at w = dx/dt = v tan(theta),
so the M3 ladder (fitted w against the M3 slope, no gap anywhere) measures v.  The gap enters only through the column
duration of a head-on track, T = g / v.  The two are independent, so their product v_ladder x T measures the gap.

    pg_twov.py extract <data dir>   -> results/twov.json   (small; reads the products listed below)
    pg_twov.py fig                  -> figures/two_velocities.{pdf,png,csv}

Inputs under <data dir> (EOS paths with the cosmic_bench/ prefix dropped; copied with xrootd, read only):
  x17rerun/paper1_gas/Analysis/mx17_det3_saturday_scan_6-27-26/<subrun>/mx17_3/s4/headon.pkl   the head-on caches
  x17rerun/paper1_gas/fit_v1/fit_T*.json                                                     the stack fits (arm in30)
  Analysis_pass1/mx17_det3_saturday_scan_6-27-26/<subrun>/mx17_3/wft/{angles/per_event.npz, events.*, calib_bundle/}
The stack (rc_diffusion.stack, |tan_x| < 0.03, the in30 arm's mask / edge / saturation cut) and the model
(x_width.predict_x at the stored sigma_0, Dd, T, dt) are the ones s4f_fit.py fitted; only the amplitude A is not stored,
so it is refitted here (linear).  The ladder is vt_common.ladder (rerun/w12/set_w0kw.py's estimator) on the row's own
03_angles table with the bow (t' = (tan_ref + a) S); it reproduces the applied kw and gives its error, which rows.csv
does not store.  Nothing is refitted except A.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.path.join(HERE, 'results', 'twov.json')
OUT = os.path.join(HERE, 'figures')
RUN = 'mx17_det3_saturday_scan_6-27-26'
STACK_V = (500, 700, 900, 1100)                 # the fields with a timing fit (300 V: the column overruns the window)
LADDER_SUBS = ['drift_scan_resist_490V_drift_%dV' % v for v in (100, 300, 500, 700, 900, 1100)] + \
              ['hv_scan2_resist_490V_drift_1000V']
TBIN = np.arange(-0.60, 0.6001, 0.05)
GAP_REF = 28.19                                  # det3 interior gap, S4 re-check (the E and timing-v gap)
GAP_REF_ERR = 0.19
GAP_DET4 = (28.6, 0.5)                           # det4 geometric gap (DET4GAP)


def _local(path, root):
    return os.path.join(root, path.split('/cosmic_bench/')[1])


def extract(root):
    sys.path[:0] = [os.path.join(REPO, 'rerun', 'calib', 's4'), os.path.join(REPO, 'rerun', 'vt_within_rows')]
    import s4_stacks                    # noqa: E402  (puts cloud_basics on sys.path)
    import s4f_fit                      # noqa: E402
    import x_width as xw                # noqa: E402
    from rc_diffusion import stack, GRID   # noqa: E402
    import vt_common as V               # noqa: E402
    V.NAME.setdefault('mx17_3', 'det3')
    gas = json.load(open(os.path.join(HERE, 'results', 'gas.json')))

    fits = {}
    for f in glob.glob(os.path.join(root, 'x17rerun', 'paper1_gas', 'fit_v1', 'fit_T*.json')):
        for k, r in json.load(open(f)).items():
            if k.endswith('|in30') and RUN in k and 'T' in r:
                fits[k] = r
    out = dict(grid_ns=GRID.tolist(), stacks={}, ladder={}, fields={})
    rng = np.random.default_rng(3)
    for Vd in STACK_V:
        key = next(k for k in fits if f'drift_{Vd}V__' in k)
        arm = fits[key]
        ev = s4f_fit.load(_local(arm['cache'], root))
        ev = s4f_fit.mask(ev, set(arm['mask_x']))
        (x0, x1), (y0, y1), d, side = arm['edge'][0]['x'], arm['edge'][0]['y'], arm['edge'][1], arm['edge'][2]
        ev = {k: e for k, e in ev.items()
              if (min(e['ref_mesh_x'] - x0, x1 - e['ref_mesh_x'], e['ref_mesh_y'] - y0, y1 - e['ref_mesh_y']) >= d)}
        ev = {k: e for k, e in ev.items() if e['x']['W'].max() < arm['sat']}
        n, M = stack(ev, 'x', tan_max=xw.TAN)
        h = s4_stacks.pass_template(os.path.dirname(_local(arm['template'], root)))
        P = xw.predict_x(h, arm['sig0'], arm['Dd'], arm['T'], 1.0, arm['dt'], len(GRID))
        sel = np.all(np.isfinite(M), axis=0)
        A = float((P[:, sel] * M[:, sel]).sum() / (P[:, sel] ** 2).sum())
        data = np.nansum(M, axis=0); data[~sel] = np.nan
        boots = []                          # event bootstrap of the strip-sum stack (error bars)
        ks = list(ev)
        for b in range(30):
            e2 = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
            _, Mb = stack(e2, 'x', tan_max=xw.TAN)
            sb = np.nansum(Mb, axis=0) / np.nanmax(np.nansum(M, axis=0))
            boots.append(sb)
        err = np.nanstd(np.array(boots), axis=0)
        out['stacks'][str(Vd)] = dict(key=key, n=int(n), T=arm['T'], T_err=arm['T_err'], dt=arm['dt'],
                                      sig0=arm['sig0'], Dd=arm['Dd'], A=A, data=data.tolist(), err=err.tolist(),
                                      model=(A * P.sum(axis=0)).tolist())
        print(f'stack {Vd} V: n {n}, T {arm["T"]:.0f} +- {arm["T_err"]:.0f} ns, dt {arm["dt"]:.0f}, A {A:.3f}')

    aroot = os.path.join(root, 'Analysis_pass1')
    for s in LADDER_SUBS:
        R = V.load_row(os.path.join(aroot, RUN, s, 'mx17_3', 'wft'), aroot)
        cal = R['cal']
        Vd = int(s.rsplit('drift_', 1)[1].rstrip('V'))
        o = dict(subrun=s, drift_V=Vd, v_drift=cal.v_drift, kw_applied=dict(cal.kw), w0_applied=dict(cal.w0), views={})
        for p in ('x', 'y'):
            vw = R['views'][p]
            tp = (vw['tan_ref'] + vw['a']) * vw['S']
            w0, kw, nkw, kwe = V.ladder(cal, vw['w'], tp)
            ok = np.isfinite(vw['w']) & np.isfinite(tp)
            dw = vw['w'] - cal.w0[p]
            med, emed, xc = [], [], []
            for lo, hi in zip(TBIN[:-1], TBIN[1:]):
                m = ok & (tp >= lo) & (tp < hi)
                if m.sum() >= 8:
                    q = dw[m]
                    xc.append(float(np.median(tp[m])))
                    med.append(float(np.median(q)))
                    emed.append(float(1.2533 * 1.4826 * np.median(np.abs(q - np.median(q))) / np.sqrt(m.sum())))
            o['views'][p] = dict(w0_est=w0, kw_est=kw, kw_err=kwe, n_kw=nkw, t=xc, dw_med=med, dw_err=emed)
            if Vd == 700:
                o['views'][p]['points'] = dict(t=tp[ok].tolist(), dw=dw[ok].tolist())
        kwm = 0.5 * (cal.kw['x'] + cal.kw['y'])
        rel = 0.5 * np.hypot(o['views']['x']['kw_err'], o['views']['y']['kw_err']) / kwm
        o['v_ang'] = cal.v_drift * kwm
        o['v_ang_err'] = o['v_ang'] * rel
        out['ladder'][str(Vd) + ('' if 'drift_scan' in s else '_hv')] = o
        print(f'ladder {s}: v_ang {o["v_ang"]:.2f} +- {o["v_ang_err"]:.2f}')

    pts = {int(p['V']): p for p in gas['scan']['points']}
    for Vd in STACK_V:
        p = pts[Vd]
        T = GAP_REF / p['v'] * 1e3                  # ns (the fit's T; v = gap / T)
        T_err = T * p['v_err'] / p['v']             # bootstrap T (+) the 1.5 % v_sys pg_gas adds
        L = out['ladder'][str(Vd)]
        g = L['v_ang'] * T * 1e-3
        ge = g * np.hypot(L['v_ang_err'] / L['v_ang'], T_err / T)
        out['fields'][str(Vd)] = dict(E=p['E'], v_timing=p['v'], v_timing_err=p['v_err'], T=T, T_err=T_err,
                                      v_ang=L['v_ang'], v_ang_err=L['v_ang_err'], gap=g, gap_err=ge,
                                      ratio=L['v_ang'] / p['v'])
    G = np.array([f['gap'] for f in out['fields'].values()]); Ge = np.array([f['gap_err'] for f in out['fields'].values()])
    wt = 1 / Ge ** 2
    gm = float((G * wt).sum() / wt.sum()); gme = float(1 / np.sqrt(wt.sum()))
    out['gap_mean'] = dict(gap=gm, err=gme, chi2=float(((G - gm) ** 2 * wt).sum()), ndf=len(G) - 1,
                           profile=gas['scan']['profile'], gap_ref=[GAP_REF, GAP_REF_ERR], gap_det4=list(GAP_DET4))
    print(f'gap from v_ladder x T: {gm:.2f} +- {gme:.2f} mm, chi2 {out["gap_mean"]["chi2"]:.1f}/{len(G) - 1}')
    json.dump(out, open(RES, 'w'), indent=1, default=float)
    print('wrote', RES)


def fig():
    sys.path.insert(0, os.path.join(REPO, 'papers'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import paper_style as ps
    ps.use()
    D = json.load(open(RES))
    c3 = ps.DET_COLOR['det3']
    # four fields of the one chamber: a sequential ramp of det3's blue (light = low field), tints of the palette colour
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D

    def tint(c, f):                    # f = 0: white, 1: the colour
        return tuple(1 - f * (1 - x) for x in to_rgb(c))
    # Dylan 2026-11-03: field colours are a sequential map that is no chamber's colour (viridis, truncated so the
    # lightest still shows on white); dark = low field
    cmap = plt.get_cmap('viridis')
    shade = dict(zip(map(str, STACK_V), [cmap(f) for f in (0.0, 0.3, 0.58, 0.82)]))
    # checklist pass 2026-11-02: full width, no panel titles, (a)-(c) labels, one field legend above the panels
    # Dylan 2026-11-03: (a) on its own full-width row to show its extent; (b), (c) side by side below
    fig = plt.figure(figsize=(6.1, 4.3 * 0.8), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    rows = []

    ax = axs[0]
    t = np.array(D['grid_ns'])
    for Vd in map(str, STACK_V):
        S = D['stacks'][Vd]
        F = D['fields'][Vd]
        y, e, m = np.array(S['data'], float), np.array(S['err'], float), np.array(S['model'], float)
        k = np.isfinite(y)
        ax.errorbar(t[k][::2], y[k][::2], yerr=e[k][::2], fmt='o', ms=2.0, color=shade[Vd], lw=0.6, zorder=3)
        ax.plot(t, m, '-', color=shade[Vd], lw=1.0, zorder=4)
        t0, t1 = S['dt'], S['dt'] + S['T']
        ax.plot([t0], [-0.06], marker='^', color=shade[Vd], ms=4, clip_on=False, zorder=5)
        ax.plot([t1], [-0.06], marker='v', color=shade[Vd], ms=4, clip_on=False, zorder=5)
        for i in np.where(k)[0]:
            rows.append(dict(panel='a', drift_V=int(Vd), E=F['E'], t_ns=t[i], stack=y[i], err=e[i], model=m[i],
                             col_start_ns=t0, col_end_ns=t1))
    ax.set_xlim(-250, 2000); ax.set_ylim(-0.1, 1.12)   # 2026-11-03: data end at 1590 ns (the window), the 500 V model tail reaches 0 at ~2000 ns
    ax.set_xlabel(r'time after leading edge $t$ (ns)'); ax.set_ylabel('strip-sum signal (normalised)')

    ax = axs[1]
    for Vd in map(str, STACK_V):
        L = D['ladder'][Vd]
        for p in ('x', 'y'):
            vw = L['views'][p]
            ax.errorbar(vw['t'], vw['dw_med'], yerr=vw['dw_err'], fmt=ps.DET_MARKER['det3'], ms=2.6, lw=0.6,
                        color=shade[Vd], mfc=shade[Vd] if p == 'x' else 'white', mec=shade[Vd], zorder=3)
            for tt, mm, ee in zip(vw['t'], vw['dw_med'], vw['dw_err']):
                rows.append(dict(panel='b', drift_V=int(Vd), view=p, tan_M3=tt, w_minus_w0=mm, err=ee))
        tt = np.linspace(-0.4, 0.4, 3)
        ax.plot(tt, L['v_ang'] * tt, '-', color=shade[Vd], lw=1.0, zorder=4)
    ax.set_xlim(-0.42, 0.42)
    ax.set_xlabel(r'reference slope $\tan\theta_{\mathrm{ref}}$'); ax.set_ylabel(r'fitted $w - w_0$ (µm/ns)')
    ax.legend(ps.view_handles(color=ps.MUTED), ['X', 'Y'], loc='lower right')

    ax = axs[2]
    G = D['gap_mean']
    E = [D['fields'][v]['E'] for v in map(str, STACK_V)]
    g = [D['fields'][v]['gap'] for v in map(str, STACK_V)]
    ge = [D['fields'][v]['gap_err'] for v in map(str, STACK_V)]
    hb = ax.axhspan(G['gap'] - G['err'], G['gap'] + G['err'], color=ps.INK, alpha=0.15, lw=0, zorder=1)
    ax.axhline(G['gap'], color=ps.INK, lw=1.0, zorder=2)
    hh = ax.axhspan(GAP_REF - GAP_REF_ERR, GAP_REF + GAP_REF_ERR, fc='none', hatch='////', ec=ps.ACCENT, lw=0, zorder=2)
    for x, y, e, Vd in zip(E, g, ge, map(str, STACK_V)):
        ax.errorbar(x, y, yerr=e, fmt='o', ms=4, color=shade[Vd], zorder=5)
        rows.append(dict(panel='c', drift_V=int(Vd), E=x, gap_mm=y, err=e, v_ang=D['fields'][Vd]['v_ang'],
                         v_timing=D['fields'][Vd]['v_timing'], T_ns=D['fields'][Vd]['T']))
    ax.legend([hb, hh], ['mean of the four', r'geometric gap $L/\tan\theta$'], loc='upper left', handlelength=1.4)
    ax.set_xlim(150, 420); ax.set_ylim(21.5, 34)
    ax.set_xlabel(r'drift field $E$ (V/cm)'); ax.set_ylabel(r'gap $g = v\,T$ (mm)')
    hs = [Line2D([], [], color=shade[Vd], marker='o', ms=3.5,
                 label=f"{D['fields'][Vd]['E']:.0f} V/cm: $v$ = {D['ladder'][Vd]['v_ang']:.1f}")
          for Vd in map(str, STACK_V)]
    ps.legend_above(fig, hs, ncol=4, handlelength=1.4, title=r"field: angle-scale $v$ (µm/ns)", title_fontsize=ps.BASE_PT * 0.85)
    ps.panel_labels(axs)
    os.makedirs(OUT, exist_ok=True)
    ps.save(fig, os.path.join(OUT, 'two_velocities'), data=pd.DataFrame(rows))
    print('wrote', os.path.join(OUT, 'two_velocities.pdf'))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'extract':
        extract(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.environ.get('WORK_DATA', '/tmp'), 'paper1_gas_twov'))
    else:
        fig()
