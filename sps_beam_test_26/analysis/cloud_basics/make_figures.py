#!/usr/bin/env python3
"""make_figures.py -- figures for the late-charge (attachment) study, from results/*.json only.

    make_figures.py [--out DIR]
Writes DIR/figures/*.png and DIR/fits.json (forward fits used by make_report.py).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from scipy.optimize import least_squares                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = '/home/dylan/x17/cosmic_bench/cloud_basics/attachment'

# reference categorical palette, slots 1-3 (validated all-pairs); neutrals for references
FIELD_C = {243: '#2a78d6', 150: '#eb6834', 92: '#1baf7a'}
INK, INK2, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#e4e3df', '#8f8d86'
PLAT = (('raw700', 243), ('raw450', 150), ('raw275', 92))
V_WET = {243: 12.76, 150: 7.38, 92: 4.41}      # Magboltz Ar/CF4/iso + 1.7 % H2O, um/ns


def J(name):
    return json.load(open(os.path.join(RES, name)))


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


# ---------------------------------------------------------------- forward model
TM = J('template_det3_x.json')
TG, TMP = np.array(TM['grid']), np.array(TM['tmpl_x'])
DT = TG[1] - TG[0]
FINE = np.arange(-400.0, 64 * 60 + 2000, DT)


def model(t, amp, t0, T, r):
    """box(t0, t0+T) * exp(-r (t-t0)) (x) template, averaged over the 60 ns sampling phase."""
    u = FINE - t0
    # fractional overlap of each fine bin with [0, T): keeps the model continuous in t0 and T
    w = np.clip(np.minimum(u + DT, T) - np.maximum(u, 0.0), 0.0, DT) / DT
    cur = w * np.exp(-r * np.clip(u, 0, None))
    sig = np.convolve(cur, TMP)[:len(FINE)] * DT
    sig = np.convolve(sig, np.ones(6) / 6, mode='same')      # 60 ns phase average on the 10 ns grid
    return amp * np.interp(t, FINE + TG[0], sig)


def fit_stack(t, y, err, T_free):
    m = (t >= 500) & (t <= 3800)

    def f(p):
        amp, t0, T, r = p if T_free else (p[0], p[1], 9000.0, p[2])
        return (model(t, amp, t0, T, r)[m] - y[m]) / err[m]
    x0 = [1 / 300, 650, 2100, 1e-4] if T_free else [1 / 300, 650, 1e-4]
    sol = least_squares(f, x0, x_scale=[1e-3, 50, 100, 1e-4][:len(x0)] if T_free else [1e-3, 50, 1e-4])
    p = list(sol.x) if T_free else [sol.x[0], sol.x[1], 9000.0, sol.x[2]]
    cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    er = float(np.sqrt(cov[-1, -1]) * np.sqrt(max(sol.cost * 2 / max(m.sum() - len(x0), 1), 1)))
    # r = 0 refit (same freedom otherwise) for the no-attachment expectation
    def f0(q):
        return (model(t, q[0], q[1], q[2] if T_free else 9000.0, 0.0)[m] - y[m]) / err[m]
    s0 = least_squares(f0, x0[:-1], x_scale=[1e-3, 50, 100][:len(x0) - 1])
    p0 = [s0.x[0], s0.x[1], s0.x[2] if T_free else 9000.0, 0.0]
    return p, er, p0, float(np.sum(sol.fun ** 2)), float(np.sum(s0.fun ** 2)), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    fd = os.path.join(a.out, 'figures'); os.makedirs(fd, exist_ok=True)
    H = J('headon_masked_k12.json'); t = np.array(H['t'])
    fits = {}

    def curve(lab, v, h):
        S = np.array(H[lab][v]['sum'][str(h)])
        ref = S[(t >= 1080) & (t <= 1260)].mean()
        band = np.array(H[lab][v]['band'][str(h)])
        return S / ref, np.maximum(band, 1e-3)

    # ---------------- F1: the observable, both views, three fields, vs the r = 0 expectation
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (lab, E) in zip(axs, PLAT):
        c = FIELD_C[E]
        yx, ex = curve(lab, 'x', 2); yy, ey = curve(lab, 'y', 8)
        p, er, p0, chi, chi0, nd = fit_stack(t, yx, ex, T_free=(E == 243))
        fits[lab] = dict(E=E, amp=p[0], t0=p[1], T=p[2], r=p[3], r_err=er, chi2=chi, chi2_r0=chi0, ndf=nd,
                         p_r0=p0)
        tt = np.linspace(0, 3840, 400)
        noloss = model(tt, p[0], p[1], p[2], 0.0)
        ax.fill_between(tt / 1e3, model(tt, *p), noloss, color=NEUTRAL, alpha=0.18, lw=0,
                        label='charge lost (same fit, r → 0)')
        ax.plot(tt / 1e3, noloss, color=NEUTRAL, lw=1.4, ls=':')
        ax.fill_between(t / 1e3, yx - ex, yx + ex, color=c, alpha=0.18, lw=0)
        ax.plot(t / 1e3, yx, color=c, lw=2, label='X, ±2 strips')
        ax.fill_between(t / 1e3, yy - ey, yy + ey, color=c, alpha=0.10, lw=0)
        ax.plot(t / 1e3, yy, color=c, lw=2, ls='--', label='Y, ±8 strips')
        ax.plot(tt / 1e3, model(tt, *p), color=INK, lw=1, label=f'fit: r = {p[3] * 1e4:.2f} e-4/ns')
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f'run_71 RAW, {E} V/cm', fontsize=11, color=INK, loc='left')
        ax.set_xlabel('time after trigger [µs]')
        ax.set_xlim(0.4, 3.84); ax.set_ylim(-0.15, 1.15)
        style(ax)
    axs[0].set_ylabel('signal / its 1.08–1.26 µs level')
    axs[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Late charge is missing in both views: head-on beam tracks, Ar/CF$_4$/iso, '
                 'trigger-placed stacks (no data-driven alignment)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f1_headon_xy.png'), dpi=140); plt.close(fig)

    # ---------------- F2: time vs depth
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for lab, E in PLAT:
        y, e = curve(lab, 'x', 2)
        t0 = fits[lab]['t0']
        m = (t >= t0 + 300) & (t <= (t0 + fits[lab]['T'] - 250 if E == 243 else 3840))
        for ax, x in ((axs[0], (t - t0) / 1e3), (axs[1], V_WET[E] * (t - t0) / 1e3)):
            ax.errorbar(x[m], y[m], yerr=e[m], color=FIELD_C[E], lw=1.8, marker='o', ms=4,
                        capsize=0, label=f'{E} V/cm')
    # what a loss per unit DEPTH (fitted on 243) would give at the other fields, in time
    r243 = fits['raw700']['r']; lam = V_WET[243] / r243 / 1e3      # mm
    for lab, E in PLAT[1:]:
        tt = np.linspace(0.3, 3.1, 50)
        y0 = np.exp(-(V_WET[E] * tt) / lam) / np.exp(-(V_WET[E] * 0.48) / lam)
        axs[0].plot(tt, y0, color=FIELD_C[E], ls=':', lw=1.4)
    axs[0].text(2.1, 0.93, 'dotted: a per-depth loss\n(fixed λ from 243 V/cm)', fontsize=8, color=INK2)
    axs[0].set_xlabel('drift time since onset [µs]'); axs[1].set_xlabel('drift depth = v·t [mm]  (v: Magboltz, 1.7 % H$_2$O)')
    axs[0].set_ylabel('X ±2 / its 1.08–1.26 µs level')
    axs[0].set_title('vs time: the three fields coincide', fontsize=11, loc='left', color=INK)
    axs[1].set_title('vs depth: they do not', fontsize=11, loc='left', color=INK)
    axs[1].legend(frameon=False, fontsize=9)
    for ax in axs:
        style(ax); ax.set_ylim(0.5, 1.12)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f2_time_not_depth.png'), dpi=140); plt.close(fig)

    # ---------------- F3: the undershoot test at 243 V/cm
    lab = 'raw700'; y, e = curve(lab, 'x', 2); yy, ey = curve(lab, 'y', 8)
    p = fits[lab]
    tt = np.linspace(0, 3840, 769)
    att = model(tt, p['amp'], p['t0'], p['T'], p['r'])
    # an LTI high-pass with the time constant that makes the same plateau sag
    flat = model(tt, p['amp'], p['t0'], p['T'], 0.0)
    tau = 1.0 / p['r']; dtt = tt[1] - tt[0]
    hp = flat.copy(); acc = 0.0
    for i in range(1, len(tt)):
        acc = acc * np.exp(-dtt / tau) + flat[i - 1] * dtt / tau
        hp[i] = flat[i] - acc
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(tt / 1e3, hp, color=NEUTRAL, lw=1.6, ls='--', label='readout high-pass, same sag (τ = 1/r)')
    ax.plot(tt / 1e3, att, color=INK, lw=1, label='charge lost in the gas (attachment fit)')
    ax.fill_between(t / 1e3, y - e, y + e, color=FIELD_C[243], alpha=0.2, lw=0)
    ax.plot(t / 1e3, y, color=FIELD_C[243], lw=2, marker='o', ms=3, label='data X ±2')
    ax.plot(t / 1e3, yy, color=FIELD_C[243], lw=1.6, ls='--', label='data Y ±8')
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(0.4, 3.84); ax.set_ylim(-0.35, 1.12)
    ax.set_xlabel('time after trigger [µs]'); ax.set_ylabel('signal / its 1.08–1.26 µs level')
    ax.set_title('243 V/cm: the drift ends inside the window and nothing swings negative',
                 fontsize=11, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='lower left'); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f3_undershoot.png'), dpi=140); plt.close(fig)
    late = (t >= 3000)
    fits['undershoot'] = dict(data_x=float(y[late].mean()), data_y=float(yy[late].mean()),
                              highpass=float(np.interp(t[late], tt, hp).mean()))

    # ---------------- F4: run_63 rotated mount: X head-on vs Y all strips, same events
    R = J('zs_timestack.json'); tr = np.array(R['t'])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (lab, Ev) in zip(axs, (('r63_d425', 142), ('r63_d325', 108))):
        for v, ls, nm in (('x', '-', 'X head-on, ±2'), ('y', '--', 'Y ladder, all strips')):
            yv = np.array(R[lab][v + '_mid']); bv = np.array(R[lab][v + '_mid_band'])
            ax.fill_between(tr / 1e3, yv - bv, yv + bv, color=INK2, alpha=0.12, lw=0)
            ax.plot(tr / 1e3, yv, color=INK if v == 'x' else INK2, lw=2, ls=ls, label=nm)
        ax.set_xlim(0.6, 3.84); ax.set_ylim(0.3, 1.3)
        ax.set_title(f'run_63, 25.64°, {Ev} V/cm (ZS 4σ), middle 60 % by charge', fontsize=10.5, loc='left', color=INK)
        ax.set_xlabel('time after trigger [µs]'); style(ax)
    axs[0].set_ylabel('signal / its 1.08–1.26 µs level'); axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f4_rot_xy_time.png'), dpi=140); plt.close(fig)

    # ---------------- F5: template-free ladder, charge vs depth: beam (Y) and bench (X, Y)
    L = J('ladder_profile.json'); BL = J('bench_ladder.json')
    tz = 1 / np.tan(np.radians(25.64))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    lad_c = {'rot_d425': ('#2a78d6', '142 V/cm'), 'rot_d325': ('#eb6834', '108 V/cm'),
             'rot_d225': ('#1baf7a', '75 V/cm')}
    for arm, (c, nm) in lad_c.items():
        rows = L[arm]['rows']; f = L[arm]['fit']
        u = np.array([r['u'] for r in rows]); ma = np.array([r['A_q70'] for r in rows])
        sel = (u >= f['u_range'][0] - 1e-6) & (u <= f['u_range'][1] + 1e-6)
        z = (f['u_range'][1] - u[sel]) * tz + 0.8 * tz
        ax = axs[0]
        ax.plot(z, ma[sel] / ma[sel][-1], color=c, marker='o', ms=4, lw=1.8,
                label=f'{nm}: fit (mean, unfired = 0) {f["per_mm"][0]:.3f}±{f["per_mm"][1]:.3f} /mm')
    axs[0].set_title('beam, run_63 Y ladder (25.64°): per-strip charge, 70th pct', fontsize=11, loc='left', color=INK)
    zb = np.array(BL['z'])
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        for v, ls in (('x', '-'), ('y', '--')):
            if det == 'det4' and v == 'x':
                continue          # det4's amplification stripes run across X: per-strip charge is the stripe map
            if v in BL.get(det, {}):
                q = np.array(BL[det][v]['Q']); q = q / np.nanmean(q[2:4])
                axs[1].plot(zb, q, color=NEUTRAL, lw=1.2, ls=ls, alpha=0.9)
    axs[1].plot([], [], color=NEUTRAL, ls='-', label='bench X (det2/3/6/7)')
    axs[1].plot([], [], color=NEUTRAL, ls='--', label='bench Y (5 chambers)')
    axs[1].axvspan(24, 30, color=GRID, alpha=0.6, lw=0)
    axs[1].text(24.4, 0.42, 'gap end /\nM3 smearing', fontsize=8, color=INK2)
    axs[1].set_title('bench cosmics, inclined tracks, Ar/iso: charge per strip', fontsize=11, loc='left', color=INK)
    for ax in axs:
        ax.set_xlabel('drift depth [mm]'); style(ax); ax.set_ylim(0.35, 1.3); ax.set_xlim(0, 30)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
    axs[0].set_ylabel('charge / its mesh-end value')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f5_ladder_depth.png'), dpi=140); plt.close(fig)

    # ---------------- F6: every dataset, both views: R = level(2.4-2.7) / level(1.08-1.26)
    Zs = J('zs_headon.json'); tz_ = np.array(Zs['t']); Em = J('zs_emulate.json')
    def Rz(S):
        S = np.array(S); return S[(tz_ >= 2400) & (tz_ < 2700)].mean() / S[(tz_ >= 1080) & (tz_ <= 1260)].mean()
    rows = []
    for lab, E in PLAT:
        rows.append((f'run_71 RAW, CF$_4$, {E} V/cm', H[lab]['x']['metrics']['2']['r2400'],
                     H[lab]['x']['metrics_err']['2']['r2400'], H[lab]['y']['metrics']['8']['r2400'],
                     H[lab]['y']['metrics_err']['8']['r2400'], 'RAW'))
    rows.append(('run_71 RAW → emulated ZS 4σ, 243 V/cm', Em['raw700']['x']['zs4'], np.nan, np.nan, np.nan, 'ZSemu'))
    Zt = J('zs_timestack.json')
    for k, nm in (('r63_flat700', 'run_63 flat, CF$_4$, 243 V/cm (ZS 4σ)'),
                  ('r63_d425', 'run_63 25.64°, CF$_4$, 142 V/cm (ZS 4σ)'),
                  ('r63_d325', 'run_63 25.64°, CF$_4$, 108 V/cm (ZS 4σ)'),
                  ('r56_625V', 'run_56 flat, CO$_2$, resist 625 V (ZS 5σ)'),
                  ('r56_590V', 'run_56 flat, CO$_2$, resist 590 V (ZS 5σ)')):
        ra, rm = Zt[k]['R_x_all'][0], Zt[k]['R_x_mid'][0]
        ya, ym = Zt[k]['R_y_all'][0], Zt[k]['R_y_mid'][0]
        rows.append((nm, 0.5 * (ra + rm), 0.5 * abs(ra - rm), 0.5 * (ya + ym), 0.5 * abs(ya - ym), 'ZS'))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    yv = np.arange(len(rows))[::-1]
    for yy_, (nm, rx, ex_, ry, ey_, kind) in zip(yv, rows):
        ax.errorbar(rx, yy_ + 0.12, xerr=None if np.isnan(ex_) else ex_, fmt='o', color=INK, ms=7, capsize=0)
        if not np.isnan(ry):
            ax.errorbar(ry, yy_ - 0.12, xerr=None if np.isnan(ey_) else ey_, fmt='s', color=INK2, ms=7,
                        mfc='white', capsize=0)
    ax.plot([], [], 'o', color=INK, label='X'); ax.plot([], [], 's', color=INK2, mfc='white', label='Y (ZS Y is censored low)')
    ax.axvline(1.0, color=INK2, lw=0.8, ls=':')
    ax.text(0.995, len(rows) - 0.4, 'no loss', ha='right', fontsize=8, color=INK2)
    ax.set_yticks(yv); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel('R = level(2.4–2.7 µs) / level(1.08–1.26 µs);  ZS bars: event-selection range'); ax.set_xlim(0.3, 1.05)
    ax.legend(frameon=False, fontsize=8, loc='lower right'); style(ax)
    ax.set_title('Every beam dataset loses late charge, in both views', fontsize=11, loc='left', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f6_all_datasets.png'), dpi=140); plt.close(fig)
    fits['dataset_rows'] = [dict(name=r[0], Rx=r[1], Rx_err=r[2], Ry=r[3], Ry_err=r[4], kind=r[5]) for r in rows]

    # ---------------- F7: discriminators
    G = J('gain_vs_loss.json'); S = J('headon_split.json'); T = J('split_toy.json')
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for lab, E in PLAT:
        for key, mk in (('by_x', 'o'), ('by_y', 's')):
            rws = G[lab][key]['rows']; g = np.array([r['gain'] for r in rws])
            axs[0].errorbar(g / g.mean(), [r['Rx'][0] for r in rws], yerr=[r['Rx'][1] for r in rws],
                            fmt=mk, color=FIELD_C[E], ms=5, capsize=0, mfc='white' if key == 'by_y' else FIELD_C[E])
    for E in (243, 150, 92):
        axs[0].plot([], [], 'o', color=FIELD_C[E], label=f'{E} V/cm')
    gg = np.array([0.5, 2.1]); axs[0].plot(gg, 1 - 0.23 * gg, color=NEUTRAL, ls='--', lw=1.2)
    axs[0].text(1.3, 0.62, 'charging that made the\nwhole loss: R = 1 − 0.23·g', fontsize=8, color=INK2)
    axs[0].set_xlabel('local gain / mean (position bins, filled: by x, open: by y)')
    axs[0].set_ylabel('R (X ±2)'); axs[0].set_ylim(0.4, 1.0); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title('gain map: no charging trend', fontsize=11, loc='left', color=INK)
    for k, (key, nm) in enumerate((('rate', 'beam rate'), ('spill', 'spill phase'))):
        for lab, E in PLAT:
            rws = S[lab][key]
            axs[1].errorbar(np.arange(3) + 4 * k + {243: -0.2, 150: 0, 92: 0.2}[E],
                            [r['x'][0] for r in rws], yerr=[r['x'][1] for r in rws], fmt='o',
                            color=FIELD_C[E], ms=5, capsize=0)
    axs[1].set_xticks([0, 1, 2, 4, 5, 6]); axs[1].set_xticklabels(['low', 'mid', 'high', 'early', 'mid', 'late'], fontsize=8)
    axs[1].text(0, 0.62, 'beam rate', fontsize=9, color=INK2); axs[1].text(4, 0.62, 'spill phase', fontsize=9, color=INK2)
    axs[1].set_ylim(0.6, 0.95); axs[1].set_ylabel('R (X ±2)')
    axs[1].set_title('rate and spill: flat (no space charge)', fontsize=11, loc='left', color=INK)
    rws = S['raw700']['charge']
    axs[2].errorbar([0, 1, 2], [r['x'][0] for r in rws], yerr=[r['x'][1] for r in rws], fmt='o',
                    color=FIELD_C[243], ms=7, capsize=0, label='data, 243 V/cm')
    for rk, ls in (('0.0', ':'), ('0.00019', '-')):
        axs[2].plot([0, 1, 2], T[rk]['terciles'], color=NEUTRAL if rk == '0.0' else INK, ls=ls, marker='.',
                    label='toy, no attachment' if rk == '0.0' else 'toy, r = 1.9e-4/ns')
    axs[2].set_xticks([0, 1, 2]); axs[2].set_xticklabels(['faint', 'mid', 'bright'])
    axs[2].set_ylim(0.6, 1.05); axs[2].set_ylabel('R (X ±2)'); axs[2].legend(frameon=False, fontsize=8)
    axs[2].set_title('split by event charge vs toy', fontsize=11, loc='left', color=INK)
    for ax in axs:
        style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f7_discriminators.png'), dpi=140); plt.close(fig)

    # ---------------- F8: the air model (air_fit.py), if it has been run
    afp = os.path.join(RES, 'air_fit_beam.json')
    if os.path.exists(afp):
        import air_fit as AF
        Af = J('air_fit_beam.json'); b = Af['best']
        Gr = AF.load_grid('beam'); Wg = np.array(sorted({k[0] for k in Gr})); Ag = np.array(sorted({k[1] for k in Gr}))
        Es = np.linspace(60, 260, 81)
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
        for (w, x, ls, nm) in ((b['water_pct'], b['air_pct'], '-', f'best: {b["water_pct"]:.2f} % H$_2$O, {b["air_pct"]:.3f} % air'),
                               (b['water_pct'], 0.0, ':', 'same water, no air'),
                               (0.0, b['air_pct'], '--', 'same air, dry')):
            pr = np.array([AF.interp(Gr, Wg, Ag, w, x, E) for E in Es])
            axs[0].plot(Es, pr[:, 0], color=INK if ls == '-' else NEUTRAL, ls=ls, lw=1.6, label=nm)
            if x > 0:
                axs[1].plot(Es, pr[:, 1] * 1e4, color=INK if ls == '-' else NEUTRAL, ls=ls, lw=1.6, label=nm)
        mv = {int(k): x for k, x in Af['measured_v'].items()}; mr = {int(k): x for k, x in Af['measured_r'].items()}
        axs[0].errorbar(list(mv), [x[0] for x in mv.values()], yerr=[0.03 * x[0] for x in mv.values()], fmt='o',
                        color=FIELD_C[243], ms=7, capsize=0, label='measured (ladder / drift end)')
        axs[1].errorbar(list(mr), [x[0] * 1e4 for x in mr.values()], yerr=[0.1 * x[0] * 1e4 for x in mr.values()],
                        fmt='o', color=FIELD_C[150], ms=7, capsize=0, label='measured (run_71 RAW)')
        axs[0].set_ylim(0, 25); axs[0].set_xlabel('drift field [V/cm]'); axs[0].set_ylabel('drift velocity [µm/ns]')
        axs[0].set_title('v(E): water sets it', fontsize=11, loc='left', color=INK)
        axs[1].set_ylim(0, 4); axs[1].set_xlabel('drift field [V/cm]'); axs[1].set_ylabel('loss rate η·v [10$^{-4}$/ns]')
        axs[1].set_title('loss rate: air sets it; field shape in tension', fontsize=11, loc='left', color=INK)
        ch = np.array(Af['chi2']); wsg = np.array(Af['grid_water']); asg = np.array(Af['grid_air'])
        cs = axs[2].contour(asg * 0.2095e4, wsg, ch - ch.min(), levels=[2.30, 6.18, 11.8], colors=[INK, INK2, NEUTRAL])
        axs[2].clabel(cs, fmt={2.30: '1σ', 6.18: '2σ', 11.8: '3σ'}, fontsize=8)
        axs[2].plot(b['o2_ppm'], b['water_pct'], '+', color=INK, ms=12)
        axs[2].set_xlim(0, 400); axs[2].set_ylim(1.2, 1.9)
        axs[2].set_xlabel('O$_2$ from air [ppm]'); axs[2].set_ylabel('water [%]')
        axs[2].set_title(f'χ² = {b["chi2"]:.1f} / {b["ndf"]}', fontsize=11, loc='left', color=INK)
        for ax in axs[:2]:
            ax.legend(frameon=False, fontsize=7.5)
        for ax in axs:
            style(ax)
        fig.tight_layout(); fig.savefig(os.path.join(fd, 'f8_air_model.png'), dpi=140); plt.close(fig)

    json.dump(fits, open(os.path.join(a.out, 'fits.json'), 'w'), indent=1)
    for lab, E in PLAT:
        f = fits[lab]
        print(f'{lab}: r = {f["r"] * 1e4:.2f} ± {f["r_err"] * 1e4:.2f} e-4/ns  T = {f["T"]:.0f}  t0 = {f["t0"]:.0f}  '
              f'chi2 {f["chi2"]:.0f} vs r=0 {f["chi2_r0"]:.0f} (ndf {f["ndf"]})')
    print('undershoot', fits['undershoot'])
    print('wrote', fd)


if __name__ == '__main__':
    main()
