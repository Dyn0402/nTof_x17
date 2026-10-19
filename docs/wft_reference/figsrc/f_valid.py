#!/usr/bin/env python3
"""
Part VII figures — does it work? Everything here is measured on the full
`sat_det3` reconstruction (7,093 events, the promoted lp bundle) against the
M3 reference, which never entered any fit.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import wftdoc as K
from wftdoc import C, save

import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles
from wft import compat, reco as wr

ANGLE_BINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def rsig(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def load():
    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    cfg = get_config(K.RUN_KEY)
    table = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    align = os.path.join(cfg.OUT_BASE, 'wft', 'alignment', 'alignment.json')
    params = cm.load_alignment(align)
    df = compat.load_table(table)
    results = compat.as_event_results(df)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    ref = {}
    for r in results:
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, params)
        ref[int(r.event_id)] = dict(tan_x=tx, tan_y=ty,
                                    mesh_x=r.ref_mesh_x_mm,
                                    mesh_y=r.ref_mesh_y_mm,
                                    radial=r.radial_residual_mm)
    print(f'[valid] {len(df):,} reconstructed events, {len(ref):,} with an '
          f'M3 reference')
    return cfg, df, ref


def arrays(df, ref, plane):
    idx = df.set_index('event_id')
    eids = [e for e in idx.index if e in ref and idx.loc[e, f'{plane}_ok']]
    d = idx.loc[eids]
    tan_ref = np.array([ref[e][f'tan_{plane}'] for e in eids])
    return dict(
        eid=np.array(eids),
        tan_ref=tan_ref,
        tan_fit=d[f'{plane}_tan_theta'].to_numpy(),
        tan_err=d[f'{plane}_tan_err'].to_numpy(),
        p0=d[f'{plane}_p0'].to_numpy(),
        p0_err=d[f'{plane}_p0_err'].to_numpy(),
        w=d[f'{plane}_w'].to_numpy(),
        chi2dof=(d[f'{plane}_chi2'] / d[f'{plane}_dof'].clip(lower=1)).to_numpy(),
        rel=d[f'{plane}_slope_reliable'].to_numpy().astype(bool),
        nstr=d[f'{plane}_n_strips'].to_numpy(),
        ndrop=d[f'{plane}_n_dropped'].to_numpy(),
        ncand=d[f'{plane}_n_candidates'].to_numpy(),
        qsum=d[f'{plane}_q_sum'].to_numpy(),
        uend=d[f'{plane}_q_uend'].to_numpy(),
        u50=d[f'{plane}_q_u50'].to_numpy(),
    )


# ------------------------------------------------------------ angle residual
def fig_angles(df, ref, v):
    fig = plt.figure(figsize=(13, 3.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    summ = {}
    ax = fig.add_subplot(gs[0])
    for plane, col in (('x', C['x']), ('y', C['y'])):
        a = arrays(df, ref, plane)
        m = a['rel'] & np.isfinite(a['tan_fit']) & np.isfinite(a['tan_ref'])
        d = np.degrees(np.arctan(a['tan_fit'][m])) - \
            np.degrees(np.arctan(a['tan_ref'][m]))
        s, b = rsig(d), float(np.median(d))
        summ[plane] = (s, b, int(m.sum()), a)
        ax.hist(d, bins=np.linspace(-8, 8, 110), histtype='step', lw=1.8,
                color=col, label=f'{plane}: {b:+.2f}° ± {s:.2f}°  (n={m.sum():,})')
        print(f'[valid] {plane}: angle bias {b:+.3f} deg, sigma {s:.3f} deg, '
              f'n={m.sum()}')
    ax.axvline(0, color=K.CHROME, lw=0.8)
    ax.set_xlabel('reconstructed − reference angle [deg]')
    ax.set_ylabel('planes')
    ax.set_title('per-event angle residual', loc='left')
    ax.legend(fontsize=8)

    # resolution vs angle
    ax = fig.add_subplot(gs[1])
    for plane, col in (('x', C['x']), ('y', C['y'])):
        a = summ[plane][3]
        at = np.abs(a['tan_ref'])
        ctr, sig = [], []
        for lo, hi in ANGLE_BINS:
            m = a['rel'] & (at >= lo) & (at < hi)
            d = np.degrees(np.arctan(a['tan_fit'][m])) - \
                np.degrees(np.arctan(a['tan_ref'][m]))
            ctr.append(0.5 * (lo + hi)); sig.append(rsig(d))
        ax.plot(ctr, sig, 'o-', color=col, label=plane)
    ax.axhline(np.degrees(np.arctan(wr.FLOOR_TAN)), color=C['red'], ls='--',
               label='measured physics floor (toy closure)')
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(r'|tan$\theta$| reference')
    ax.set_ylabel(r'robust $\sigma$ of the angle residual [deg]')
    ax.set_title('flat across the angle range, and close to the floor',
                 loc='left')
    ax.legend(fontsize=8)

    # implied v
    ax = fig.add_subplot(gs[2])
    for plane, col in (('x', C['x']), ('y', C['y'])):
        a = summ[plane][3]
        at = np.abs(a['tan_ref'])
        vimp = a['w'] * 1e3 / a['tan_ref']
        ctr, med, err = [], [], []
        for lo, hi in ANGLE_BINS:
            m = a['rel'] & (at >= lo) & (at < hi)
            ctr.append(0.5 * (lo + hi))
            med.append(float(np.nanmedian(vimp[m])))
            err.append(rsig(vimp[m]) / max(np.sqrt(m.sum()), 1))
        ax.errorbar(ctr, med, yerr=err, fmt='o-', color=col, capsize=3,
                    label=f'{plane}  (spread {np.nanmax(med)-np.nanmin(med):.1f})')
        print(f'[valid] {plane}: implied v {np.round(med,2)}')
    ax.axhline(v, color=K.CHROME, ls='--', lw=1, label=f'calibration v = {v:.1f}')
    ax.set_xlabel(r'|tan$\theta$| reference')
    ax.set_ylabel(r'median $w/\tan\theta$ [µm/ns]')
    ax.set_title('implied velocity — flat is the pass criterion', loc='left')
    ax.legend(fontsize=8)
    save(fig, 'angles')
    return summ


# ---------------------------------------------------------------- position
def fig_position(df, ref):
    idx = df.set_index('event_id')
    eids = [e for e in idx.index if e in ref and idx.loc[e, 'x_ok']
            and idx.loc[e, 'y_ok']]
    d = idx.loc[eids]
    dx = d['x_p0'].to_numpy() - np.array([ref[e]['mesh_x'] for e in eids])
    dy = d['y_p0'].to_numpy() - np.array([ref[e]['mesh_y'] for e in eids])
    r = np.hypot(dx, dy)
    core = rsig(r[r < 3])
    print(f'[valid] position: sigma_x {rsig(dx):.3f} mm, sigma_y {rsig(dy):.3f} mm, '
          f'within 5 mm {100*np.mean(r<5):.2f} %, core sigma {core:.3f} mm')

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    ax = axs[0]
    for a, lab, col in ((dx, 'x', C['x']), (dy, 'y', C['y'])):
        ax.hist(a, bins=np.linspace(-4, 4, 130), histtype='step', lw=1.8,
                color=col, label=f'{lab}: $\\sigma$ = {rsig(a):.2f} mm')
    ax.set_xlabel('fitted mesh position − reference [mm]')
    ax.set_ylabel('events')
    ax.set_title('position at the mesh', loc='left')
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.hist(r, bins=np.linspace(0, 12, 120), color=C['blue'], alpha=0.85)
    ax.axvline(5, color=C['red'], ls='--',
               label=f'5 mm: {100*np.mean(r<5):.1f} % of reconstructed events')
    ax.set_yscale('log')
    ax.set_xlabel('radial residual |r| [mm]')
    ax.set_ylabel('events')
    ax.set_title('the tail is real: wrong-cluster seeds', loc='left')
    ax.legend(fontsize=8)

    ax = axs[2]
    ax.hexbin(dx, dy, gridsize=60, extent=(-4, 4, -4, 4), cmap='magma',
              mincnt=1)
    ax.set_xlabel('Δx [mm]'); ax.set_ylabel('Δy [mm]')
    ax.set_title('no structure — the alignment is centred', loc='left')
    ax.grid(False)
    save(fig, 'position')


# ------------------------------------------------------------------- pulls
def fig_pulls(summ):
    """Are the reported errors honest? A pull distribution of width 1 says the
    quoted sigma means what it says."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axs[0]
    for plane, col in (('x', C['x']), ('y', C['y'])):
        s, b, n, a = summ[plane]
        m = a['rel'] & np.isfinite(a['tan_err']) & (a['tan_err'] > 0)
        pull = (a['tan_fit'][m] - a['tan_ref'][m]) / a['tan_err'][m]
        ax.hist(pull, bins=np.linspace(-5, 5, 110), histtype='step', lw=1.8,
                color=col, label=f'{plane}: width {rsig(pull):.2f}')
        print(f'[valid] {plane}: angle pull width {rsig(pull):.3f}')
    ax.axvline(0, color=K.CHROME, lw=0.8)
    ax.set_xlabel(r'(tan$\theta_{\rm fit}$ − tan$\theta_{\rm ref}$) / reported error')
    ax.set_ylabel('planes')
    ax.set_title('angle pull — width 1 would be perfect calibration of the '
                 'error', loc='left')
    ax.legend(fontsize=8)

    ax = axs[1]
    for plane, col in (('x', C['x']), ('y', C['y'])):
        s, b, n, a = summ[plane]
        m = a['rel']
        q = np.linspace(2, 98, 40)
        cs = np.percentile(a['chi2dof'][m], q)
        d = np.abs(np.degrees(np.arctan(a['tan_fit'][m]))
                   - np.degrees(np.arctan(a['tan_ref'][m])))
        sig = [rsig(d[(a['chi2dof'][m] >= lo) & (a['chi2dof'][m] < hi)])
               for lo, hi in zip(cs[:-1], cs[1:])]
        ax.plot(0.5 * (cs[:-1] + cs[1:]), sig, 'o-', color=col, ms=4,
                label=plane)
    ax.axvline(wr.CHI2DOF_BAD, color=C['red'], ls='--',
               label=f'quality_ok cut, {wr.CHI2DOF_BAD:.0f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\chi^2/\mathrm{dof}$ of the plane fit')
    ax.set_ylabel(r'angle $\sigma$ [deg]')
    ax.set_title(r'$\chi^2$/dof does predict which fits to distrust',
                 loc='left')
    ax.legend(fontsize=8)
    save(fig, 'pulls')


# ------------------------------------------------------- quality and flags
def fig_quality(df, summ):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    ax = axs[0]
    for plane, col in (('x', C['x']), ('y', C['y'])):
        c = (df.loc[df[f'{plane}_ok'], f'{plane}_chi2'] /
             df.loc[df[f'{plane}_ok'], f'{plane}_dof'].clip(lower=1))
        ax.hist(c, bins=np.logspace(0, 4, 90), histtype='step', lw=1.8,
                color=col, label=f'{plane}: median {np.median(c):.0f}')
        print(f'[valid] {plane}: median chi2/dof {np.median(c):.1f}')
    ax.axvline(wr.CHI2DOF_BAD, color=C['red'], ls='--', label='quality_ok cut')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\chi^2/\mathrm{dof}$')
    ax.set_ylabel('planes')
    ax.set_title(r'$\chi^2$/dof is large by construction —'
                 '\nevery sample counts, the model is imperfect', loc='left')
    ax.legend(fontsize=8)

    ax = axs[1]
    for plane, col in (('x', C['x']), ('y', C['y'])):
        t = np.abs(df.loc[df[f'{plane}_ok'], f'{plane}_tan_theta'])
        ax.hist(t, bins=np.linspace(0, 0.7, 90), histtype='step', lw=1.8,
                color=col, label=plane)
    ax.axvline(wr.TAN_MIN_SLOPE, color=C['red'], ls='--',
               label=f'slope_reliable, |tan| ≥ {wr.TAN_MIN_SLOPE}')
    ax.set_xlabel(r'fitted |tan$\theta$|')
    ax.set_ylabel('planes')
    ax.set_title('the near-vertical population, where timing\n'
                 'carries no slope information', loc='left')
    ax.legend(fontsize=8)

    ax = axs[2]
    for plane, col in (('x', C['x']), ('y', C['y'])):
        n = df.loc[df[f'{plane}_ok'], f'{plane}_n_dropped']
        ax.hist(np.clip(n, 0, 20), bins=np.arange(-0.5, 21, 1),
                histtype='step', lw=1.8, color=col, label=plane)
    ax.axvline(compat.MAX_DROPPED + 0.5, color=C['red'], ls='--',
               label=f'cluster-quality cut, n_dropped ≤ {compat.MAX_DROPPED}')
    ax.set_yscale('log')
    ax.set_xlabel('strips in competing clusters (n_dropped)')
    ax.set_ylabel('planes')
    ax.set_title('how often another cluster competes for the seed', loc='left')
    ax.legend(fontsize=8)
    save(fig, 'quality')


# --------------------------------------------------------- charge column
def fig_column(df, ref):
    """The deconvolved charge profile's endpoint, mapped across the chamber:
    this is how the cathode tilt was measured without opening anything."""
    idx = df.set_index('event_id')
    eids = [e for e in idx.index if e in ref and idx.loc[e, 'x_ok']
            and idx.loc[e, 'y_ok']]
    d = idx.loc[eids]
    px = d['x_p0'].to_numpy(); py = d['y_p0'].to_numpy()
    u50 = 0.5 * (d['x_q_u50'].to_numpy() + d['y_q_u50'].to_numpy())
    ok = np.isfinite(u50) & (d['x_chi2'].to_numpy() /
                             d['x_dof'].clip(lower=1).to_numpy() < 400)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    ax = axs[0]
    for p, col in (('x', C['x']), ('y', C['y'])):
        a = d[f'{p}_q_uend'].to_numpy()
        a = a[np.isfinite(a)]
        ax.hist(a, bins=np.linspace(0, 1200, 90), histtype='step', lw=1.8,
                color=col, label=f'{p}: median {np.median(a):.0f} ns')
    ax.set_xlabel('column end $u_{\\rm end}$ [ns]')
    ax.set_ylabel('planes')
    ax.set_title('where the charge stops arriving', loc='left')
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.hist(u50[ok], bins=np.linspace(200, 900, 90), color=C['blue'],
            alpha=0.85)
    ax.set_xlabel('$u_{50}$, median charge arrival [ns]')
    ax.set_ylabel('events')
    ax.set_title(f'$u_{{50}}$ median {np.nanmedian(u50[ok]):.0f} ns', loc='left')

    ax = axs[2]
    from scipy.stats import binned_statistic_2d
    xs = np.linspace(np.nanpercentile(px, 2), np.nanpercentile(px, 98), 7)
    ys = np.linspace(np.nanpercentile(py, 2), np.nanpercentile(py, 98), 7)
    st, _xe, _ye, _n = binned_statistic_2d(px[ok], py[ok], u50[ok],
                                           statistic='median', bins=[xs, ys])
    im = ax.imshow(st.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax.set_xlabel('x at the mesh [mm]'); ax.set_ylabel('y at the mesh [mm]')
    ax.set_title('$u_{50}$ across the chamber — the pattern that\n'
                 'became the cathode-flatness measurement', loc='left')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('$u_{50}$ [ns]', color=K.CHROME)
    cb.ax.tick_params(colors=K.CHROME)
    cb.outline.set_edgecolor(K.CHROME)
    print(f'[valid] u50 map spans {np.nanmin(st):.0f}-{np.nanmax(st):.0f} ns '
          f'= {(np.nanmax(st)-np.nanmin(st))*36.6e-3:.1f} mm of column')
    save(fig, 'column')


def fig_efficiency(cfg):
    p = os.path.join(cfg.OUT_BASE, 'wft', 'efficiency',
                     'efficiency_breakdown.json')
    if not os.path.exists(p):
        print('[valid] no efficiency json'); return
    with open(p) as f:
        j = json.load(f)
    print('[valid] efficiency:', j)
    cats = [('no_hit', 'no strip fired'), ('spark_cat', 'spark (> 50 strips)'),
            ('hit_no_reco', 'fired, no reconstruction'),
            ('reco_far', 'reconstructed, |r| > 5 mm'),
            ('within_R', 'reconstructed, |r| ≤ 5 mm')]
    vals, labs = [], []
    for k, lab in cats:
        if k in j:
            vals.append(float(j[k]))
            labs.append(lab)
    if not vals:
        return
    tot = sum(vals)
    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    left = 0
    cols = [C['grey'], C['purple'], C['orange'], C['red'], C['green']]
    for v_, lab, col in zip(vals, labs, cols):
        ax.barh([0], [100 * v_ / tot], left=left, color=col, height=0.5)
        if 100 * v_ / tot > 3:
            ax.text(left + 50 * v_ / tot, 0, f'{100*v_/tot:.1f} %',
                    ha='center', va='center', fontsize=9, color='w')
        left += 100 * v_ / tot
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of muons the M3 reference says crossed the chamber')
    ax.set_title('efficiency breakdown — detection stays hits-defined,\n'
                 'the fit only decides where the point goes', loc='left')
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cols[:len(labs)]],
              labs, fontsize=7.5, ncol=3, loc='lower left',
              bbox_to_anchor=(0, -1.0))
    save(fig, 'efficiency')


def main():
    cfg, df, ref = load()
    cal = K.bundle()
    summ = fig_angles(df, ref, cal.v_drift)
    fig_position(df, ref)
    fig_pulls(summ)
    fig_quality(df, summ)
    fig_column(df, ref)
    fig_efficiency(cfg)


if __name__ == '__main__':
    main()
