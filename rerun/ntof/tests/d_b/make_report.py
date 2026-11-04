#!/usr/bin/env python3
"""make_report.py -- build /media/ucla/work/ntof_pass1/d_b/report.html (and figures/) from the D/B study tables.

Generated, not hand-written: re-run after dbeam.py, ringing.py, line_kw.py, wall_kw.py and bd_cosmic.py and the
numbers, figures and verdict text follow.  Figures are referenced with relative links (figures/x.png).

    .venv/bin/python rerun/ntof/tests/d_b/make_report.py
"""
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]

D = '/media/ucla/work/ntof_pass1/d_b'
FIG = f'{D}/figures'
KW_V1 = {'D': json.load(open('/media/ucla/work/ntof_pass1/bundles/v1_D/bundle.json'))['kw']}
V1_D = 31.0


def rd(p, **kw):
    p = f'{D}/{p}'
    if not os.path.exists(p):
        return None
    return pd.read_json(p, **kw) if p.endswith('.json') and kw else (pd.read_csv(p) if p.endswith('.csv') else
                                                                     pd.read_parquet(p))


def jl(p):
    p = f'{D}/{p}'
    return json.load(open(p)) if os.path.exists(p) else None


def table(df, cols=None, fmt=None, cls=''):
    if df is None or not len(df):
        return '<p><i>(no data)</i></p>'
    cols = cols or list(df.columns)
    fmt = fmt or {}
    h = ''.join(f'<th>{html.escape(str(c))}</th>' for c in cols)
    b = ''
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt.get(c)
            if f and isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
                s = f.format(v)
            elif isinstance(v, (float, np.floating)) and not np.isfinite(v):
                s = '–'
            else:
                s = str(v)
            cells.append(f'<td>{html.escape(s)}</td>')
        b += '<tr>' + ''.join(cells) + '</tr>'
    return f'<table class="{cls}"><tr>{h}</tr>{b}</table>'


# --------------------------------------------------------------------------- figures
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sept26_prelim_analysis import figstyle as FS
    FS.use()
    os.makedirs(FIG, exist_ok=True)
    out = {}
    t = np.arange(20) * 60

    st = jl('dbeam/stacks.json')
    if st:
        fig, axs = plt.subplots(1, 2, figsize=FS.WIDE, sharey=True)
        rows = []
        for ax, (run, lab) in zip(axs, (('145', 'beam, run_145'), ('149', 'cosmics, run_149'))):
            FS.strip(ax)
            for sel, col, ls in (('all clean', FS.MUTED, '-'), ('clean, own trigger', FS.DET_COLOR['D'], '-'),
                                 ('clean, other arm trigger', FS.COPPER, '--')):
                k = f'{run}_Dx_{sel}'
                if k in st:
                    ax.plot(t, st[k], color=col, ls=ls, label=f'D x, {sel}')
                    rows += [dict(run=run, selection=sel, t_ns=tt, S=s) for tt, s in zip(t, st[k])]
            k = f'{run}_Ax_clean, own trigger'
            if k in st:
                ax.plot(t, st[k], color=FS.DET_COLOR['A'], lw=1.0, label='A x, own trigger')
                rows += [dict(run=run, selection='A own', t_ns=tt, S=s) for tt, s in zip(t, st[k])]
            ax.set_xlabel('time in the 20-sample window [ns]')
            ax.text(0.02, 0.95, lab, transform=ax.transAxes, va='top', color=FS.INK)
        axs[0].set_ylabel('median normalised strip sum')
        axs[1].legend(loc='lower center', fontsize=8.5)
        FS.fig_title(fig, "D's column is a box only on its own trigger: on beam the rest is not the triggering particle",
                     'stacked strip sum of the main cluster (>= 5 strips, one cluster, unsaturated)')
        out['stacks'] = FS.save(fig, f'{FIG}/d_stacks', data=pd.DataFrame(rows))

    vt = rd('ringing/ringing_vs_time.csv')
    if vt is not None:
        fig, ax = FS.figure(FS.FIG)
        for arm in 'ACD':
            for p, ls in (('x', '-'), ('y', ':')):
                g = vt[(vt.arm == arm) & (vt.plane == p)]
                ax.plot(0.5 * (g.t_lo_ms + g.t_hi_ms), 100 * g.frac_ringing, ls=ls,
                        **{k: v for k, v in FS.det_style(arm).items() if k != 'label'},
                        label=f'{arm} {p}')
        ax.set_xlabel('time since the gamma flash [ms]')
        ax.set_ylabel('triggers with >= 4 ringing strips [%]')
        ax.set_ylim(-3, 105)
        ax.legend(ncol=3, loc='center right')
        FS.title(ax, 'The beam-on oscillation dies out in A and C; in D it stays for the whole window',
                 'run_145 stat090_0000 _000, events with a track in the scintillator stack')
        out['ringing_time'] = FS.save(fig, f'{FIG}/ringing_vs_time', data=vt)

    if os.path.exists(f'{D}/ringing/examples.npz'):
        z = np.load(f'{D}/ringing/examples.npz')
        labs = [('D_beam_ring', 'D x, beam > 10 ms,\nother arm trigger'), ('D_beam_own', 'D x, beam > 10 ms,\nD own trigger'),
                ('A_beam_own', 'A x, beam > 30 ms,\nA own trigger'), ('D_cosmic_BDline', 'D x, cosmic,\nB-D line muon')]
        labs = [l for l in labs if l[0] in z]
        fig, axs = plt.subplots(1, len(labs) + 1, figsize=FS.WIDE,
                                gridspec_kw=dict(width_ratios=[1] * len(labs) + [0.06]))
        rows = []
        for ax, (k, lab) in zip(axs[:-1], labs):
            Wf = z[k]
            im = ax.imshow(Wf.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-300, vmax=300,
                           extent=(-0.5, 511.5, -30, 1170), interpolation='nearest')
            ax.set_title(f'{lab}\nevent {int(z[k + "_eid"])}, {int(z[k + "_nring"])} ringing strips', fontsize=8.5)
            ax.set_xlabel('channel')
            ax.grid(False)
            rows.append(dict(panel=lab.replace(chr(10), ' '), event=int(z[k + '_eid']), n_ringing=int(z[k + '_nring'])))
        axs[0].set_ylabel('time in window [ns]')
        for ax in axs[1:-1]:
            ax.set_yticklabels([])
        fig.colorbar(im, cax=axs[-1], label='ADC')
        FS.fig_title(fig, 'Beam-on D: tens of strips with three to four humps ~360 ns apart; a drift signal has one',
                     'pedestal- and common-mode-subtracted waveforms, all 512 channels of the x FEU')
        out['ringing_examples'] = FS.save(fig, f'{FIG}/ringing_examples', data=pd.DataFrame(rows))

    L = rd('line_kw/lines_D_v1.parquet')
    if L is not None and len(L):
        L = L[L.miss_mm < 60]
        fig, axs = plt.subplots(1, 2, figsize=FS.WIDE, sharey=True)
        rows = []
        for ax, v in zip(axs, 'xy'):
            FS.strip(ax)
            for part, mk, col in (('A', 'o', FS.DET_COLOR['A']), ('B', 's', FS.DET_COLOR['B'])):
                g = L[L.partner == part]
                tt = g[f'true_tan{v}'].to_numpy()
                r = np.corrcoef(g[f'tan{v}'], tt)[0, 1] if len(g) > 5 else 1.0
                tt = tt * np.sign(r)
                ax.plot(tt, g[f'tan{v}'], mk, color=col, ms=3.5, alpha=0.8, label=f'{part}-D line ({len(g)})')
                rows += [dict(partner=part, view=v, true=a, fit=b) for a, b in zip(tt, g[f'tan{v}'])]
            xx = np.linspace(-0.8, 0.8, 3)
            ax.plot(xx, xx, color=FS.MUTED, lw=1, ls='--', label='fitted = true')
            ax.axvspan(-0.45, -0.15, color=FS.GRID, lw=0, zorder=0)
            ax.axvspan(0.15, 0.45, color=FS.GRID, lw=0, zorder=0)
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)
            ax.set_xlabel(f'line tan {v} in D (positions only)')
        axs[0].set_ylabel('D fitted tan (v1_D: v 31.0, kw 1.00)')
        axs[0].legend(loc='upper left', fontsize=8.5)
        FS.fig_title(fig, "D's fitted angles against lines through another chamber's cluster centroid",
                     'run_149 cosmics; miss < 60 mm; grey = the ratio window |tan| 0.15-0.45')
        out['lines'] = FS.save(fig, f'{FIG}/d_lines', data=pd.DataFrame(rows))

    P = rd('wall_kw/profile_D_v1.csv')
    if P is not None:
        S = jl('wall_kw/summary_D_v1.json')
        fig, ax = FS.figure(FS.FIG)
        ax.plot(P.s, P.dnll, color=FS.INK)
        ax.set_xlabel('wall scale s  (true tan = s x D fitted tan, v1_D)')
        ax.set_ylabel('-log L (relative)')
        ax.set_ylim(0, min(P.dnll.max(), 20))
        FS.title(ax, f"D's own wall: s = {S['s_best']:.3f} on {S['n_single_x']} single x tracks",
                 'edge likelihood of the three internal group boundaries (cosmic_wall_scale.py estimator)')
        out['wall'] = FS.save(fig, f'{FIG}/d_wall_profile', data=P)
    return out


# --------------------------------------------------------------------------- numbers
def pick_lines():
    """the D angle-scale numbers from the centroid lines.  Primary: partner B (parallel, opposite: its centroid is a
    full column at mid-gap), strict partner, miss < 60 mm.  Cross-check: partner A with the depth taken from time;
    the A mid-gap vs depth-from-time difference is that method's systematic.  Short A-D lines (< 250 mm) read 0.71-0.83
    and depend on where A's partial column is placed (mid vs time differ by up to 0.09); from 250 mm they agree with
    B-D."""
    R = rd('line_kw/measure_D_v1.csv')
    if R is None:
        return None, None
    base = R[(R.point == 'D fitted p0') & (R.miss_max == 60.0) & R.partner_strict]
    out = []
    for v in 'xy':
        b = base[(base.partner == 'B') & (base.view == v) & (base.lever_min == 0.0)].iloc[0]
        # A-D: only lines >= 250 mm long, where the placement of A's partial (steep) column no longer matters
        at = base[(base.partner == 'A') & (base.depth == 'time') & (base.view == v) & (base.lever_min == 250.0)].iloc[0]
        am = base[(base.partner == 'A') & (base.depth == 'mid') & (base.view == v) & (base.lever_min == 250.0)].iloc[0]
        out.append(dict(view=v, n_BD=int(b.n), ratio_BD=b.ratio, err_BD=b.err, n_AD=int(at.n), ratio_AD=at.ratio,
                        err_AD=at.err, syst_AD=abs(at.ratio - am.ratio), kw_D=KW_V1['D'][v] * b.ratio,
                        kw_D_err=KW_V1['D'][v] * b.err, kwv=KW_V1['D'][v] * b.ratio * V1_D))
    return base, pd.DataFrame(out)


def verdict_html(sb, comb, wall, plas, dv, eff, bpos, rg, dbr=None):
    P = []
    own = sb[(sb.run == 'run_145') & (sb.arm == 'D') & (sb.selection == 'clean, own trigger')].set_index('view')
    allc = sb[(sb.run == 'run_145') & (sb.arm == 'D') & (sb.selection == 'all clean')].set_index('view')
    act = sb[(sb.run == 'run_145') & (sb.selection == 'ACTIVITY')].set_index(['arm', 'view'])
    rgt = pd.DataFrame(rg)
    rb = rgt[(rgt.run == 'run_145') & (rgt.arm == 'D')]
    P.append(f"<b>1. D on beam.</b> The box fit fails because the clusters it is given are mostly not the triggering "
             f"particle. Only {100 * act.loc[('D', 'x'), 'own_frac_of_clean']:.0f} % of D's clean beam clusters are on "
             f"D's own trigger (A {100 * act.loc[('A', 'x'), 'own_frac_of_clean']:.0f} %, C "
             f"{100 * act.loc[('C', 'x'), 'own_frac_of_clean']:.0f} %). On its own trigger D's box is where A's and C's "
             f"are (start {own.loc['x', 'stack_t0']:.0f} / {own.loc['y', 'stack_t0']:.0f} ns, x / y), so this is "
             f"not a window, latency, trigger-timing or column offset; the withdrawn window idea stays withdrawn. "
             f"The rest is a beam-gated oscillation, period ~{rb.period_ns_med.median():.0f} ns (2.8 MHz), on tens of D "
             f"strips: {100 * rb.frac_ringing.min():.0f}–{100 * rb.frac_ringing.max():.0f} % of D beam triggers, "
             f"every trigger after ~10 ms; A and C show the same oscillation only in the first 10–30 ms after the "
             f"flash. D's own-trigger beam columns are not truncated (box end ≈ "
             f"{own.loc['x', 'stack_t0'] + own.loc['x', 'stack_T']:.0f} ns of 1200), but they read "
             f"{own.loc['x', 'stack_T']:.0f} / {own.loc['y', 'stack_T']:.0f} ns against 920 / 874 ns on cosmics "
             f"(A and C: +3–5 % beam vs cosmics, D +9 / +18 %).")
    if dbr is not None and len(dbr):
        b = dbr.set_index(['arm', 'sample'])
        if ('D', 'own, > 10 ms') in b.index and ('A', 'own, > 10 ms') in b.index:
            d, a = b.loc[('D', 'own, > 10 ms')], b.loc[('A', 'own, > 10 ms')]
            P[-1] += (f" The fit sees it: on its own trigger after 10 ms D's beam fits have median chi2/dof "
                      f"{d.x_chi2dof_p50:.0f} / {d.y_chi2dof_p50:.0f} (A {a.x_chi2dof_p50:.1f} / {a.y_chi2dof_p50:.1f}) and a "
                      f"t0 width of {d.x_t0_w68:.0f} / {d.y_t0_w68:.0f} ns (A {a.x_t0_w68:.0f} / {a.y_t0_w68:.0f}).")
    P[-1] += (" <b>D's v cannot be measured on beam, and D's beam angles are limited by the oscillation: a stated "
              "limitation, unless the oscillation is removed from D's beam waveforms (see the proposals).</b>")
    if comb is not None and len(comb):
        c = comb.set_index('view')
        txt = (f"<b>2. kw_D.</b> D has an angle truth after all: cosmic lines through another chamber's cluster "
               f"centroid (positions only). B–D lines (B parallel and opposite; its positions need no field cage): "
               f"fitted/true = {c.loc['x', 'ratio_BD']:.3f} ± {c.loc['x', 'err_BD']:.3f} (x, n {c.loc['x', 'n_BD']}), "
               f"{c.loc['y', 'ratio_BD']:.3f} ± {c.loc['y', 'err_BD']:.3f} (y, n {c.loc['y', 'n_BD']}), so with the v1 "
               f"v = 31.0 <b>kw_D = {c.loc['x', 'kw_D']:.3f} ± {c.loc['x', 'kw_D_err']:.3f} (x), "
               f"{c.loc['y', 'kw_D']:.3f} ± {c.loc['y', 'kw_D_err']:.3f} (y)</b>, i.e. kw·v ≈ "
               f"{c.loc['x', 'kwv']:.1f} / {c.loc['y', 'kwv']:.1f} µm/ns. A–D lines agree within their method "
               f"systematic ({c.loc['x', 'ratio_AD']:.2f} / {c.loc['y', 'ratio_AD']:.2f}, ± "
               f"{c.loc['x', 'syst_AD']:.2f} / {c.loc['y', 'syst_AD']:.2f} from where A's partial column is placed; lines ≥ 250 mm, where that placement stops mattering; shorter A–D lines read lower and move with it).")
        if wall:
            fc = wall['frame_check']
            txt += (f" D's own wall (x): direct s = {wall['s_best']:.2f} ({wall['s_prof_lo']:.2f}–"
                    f"{wall['s_prof_hi']:.2f}); the old chain's 1.525 carried onto the new fits gives s = "
                    f"{fc.get('s_chain', np.nan):.3f} ± {fc.get('s_chain_err', np.nan):.3f}, i.e. fitted/true "
                    f"{1 / fc.get('s_chain', np.nan):.3f}; corrected for the wall's +3.6 % on A, "
                    f"{1.036 / fc.get('s_chain', np.nan):.3f}.")
        chk = jl('wall_kw/check_A.json')
        if plas:
            txt += (f" The plastic L/R gap (192.6 mm lever) gives s = {plas['s']:.2f} ({plas['boot_lo']:.2f}–"
                    f"{plas['boot_hi']:.2f}), but it is not a validated estimator: its edge sits at D's dead-strip "
                    f"boundary, and on A (where the line fixes s = 1)"
                    + (f" it returns {chk['plastic_s']:.2f} on {chk['n_single_x']} tracks (wall: "
                       f"{chk['wall_prof_best']:.2f}, {chk['wall_prof_lo']:.2f}–{chk['wall_prof_hi']:.2f})" if chk else '')
                    + "; it is not used.")
        P.append(txt)
    if dv is not None and len(dv):
        d = dv.set_index(['partner', 'view'])
        if ('all', 'x') in d.index:
            P.append(f"<b>D's timing v</b> depends on which muons are stacked: "
                     f"{d.loc[('all', 'x'), 'v']:.1f} ± {d.loc[('all', 'x'), 'v_err']:.1f} / "
                     f"{d.loc[('all', 'y'), 'v']:.1f} ± {d.loc[('all', 'y'), 'v_err']:.1f} µm/ns (x / y) on all line "
                     f"muons, {d.loc[('B', 'x'), 'v']:.1f} / {d.loc[('B', 'y'), 'v']:.1f} on the B–D ones, "
                     f"{d.loc[('A', 'x'), 'v']:.1f} / {d.loc[('A', 'y'), 'v']:.1f} on the A–D ones (D's A-side edge); "
                     f"32.0 / 33.7 on D's own cosmic triggers, 29.9–32.0 in the v1 measurement (gap 29.44 ± 1.12 mm). "
                     f"The angle scale is flat across D (B–D ratio 0.83–0.90 in every quarter of D and of B) and says "
                     f"kw·v ≈ {comb.kwv.mean():.1f}. So D's timing v and its angle scale disagree by "
                     f"{100 * (1 - comb.kwv.mean() / 31.0):.0f} %, where A and C agree to ~1 %. Only kw·v enters the "
                     f"angle: the bundle keeps the timing v = 31.0 (as for A and C) and carries the difference in kw.")
    if eff is not None:
        e = eff.set_index('partner')
        P.append("<b>3. B.</b> Confirmed: no field-shaping ring (zero drift current is the missing divider; STATUS "
                 "2026-09-08), no angle, no v measurable (its column is not a box). Cosmic efficiency: " + ', '.join(
                     f"{100 * r.eff:.0f} ± {100 * r.eff_err:.0f} % on {r.tracks.split(' (')[0]}-predicted crossings "
                     f"(n {r.n_pred})" for r in eff[eff.partner == 'B'].itertuples()) +
                 " (C on A-predicted: " + ', '.join(f"{100 * r.eff:.0f} %" for r in eff[(eff.partner == 'C') & (eff.n_pred > 20)].itertuples()) +
                 "). B's v enters no position: p0 is the fit's mesh crossing (v-free); only the bow height term "
                 "p0 − tan·h and the plausibility gate (which candidate wins) see it.")
    if bpos is not None and len(bpos):
        P.append("B fitted at v 42.6 and 30.0 on the same events: when the same candidate wins, p0 is identical "
                 "(|Δp0| = " + ', '.join(f"{r.dp0_95:.3f}" for r in bpos.itertuples()) + " mm at 95 %): v only "
                 "rescales tan after the fit. It reaches positions through the bow height term p0 − tan·h alone: "
                 + ', '.join(f"{r.sample} {r.view} {r.bow_term_p50:.2f} / {r.bow_term_p95:.2f} mm" for r in bpos.itertuples())
                 + " (median / 95 %) for a 30 % change in v. It does change which candidate wins ("
                 + ', '.join(f"{100 * r.frac_other_winner:.0f} %" for r in bpos.itertuples())
                 + ") and the paired-track yield (" + ', '.join(f"{r.sample}: {r.n_paired_42} vs {r.n_paired_30}" for r in bpos[bpos.view == 'x'].itertuples())
                 + "), because the plausibility gate is a cut in tan. <b>B's v is a limitation for B's efficiency, "
                 "not for its positions.</b>")
    return ''.join(f'<p>{p}</p>' for p in P)


def main():
    figs = figures()
    sb = rd('dbeam/stack_fits.csv')
    rg = jl('ringing/ringing.json')
    val = rd('line_kw/validate.csv')
    meas = rd('line_kw/measure_D_v1.csv')
    sel, comb = pick_lines()
    wall = jl('wall_kw/summary_D_v1.json')
    walls = rd('wall_kw/scales_D_v1.csv')
    dv = rd('bd_cosmic/dv_lines.csv')
    eff = rd('bd_cosmic/eff.csv')
    bpos = rd('bd_cosmic/bpos.csv')
    dbr = rd('dbeam/d_beam_reco.csv')

    def fimg(k, cap):
        if k not in figs:
            return ''
        return f'<figure><img src="figures/{os.path.basename(str(figs[k]))}"><figcaption>{cap}</figcaption></figure>'

    def sbv(run, arm, view, selname):
        if sb is None:
            return None
        g = sb[(sb.run == run) & (sb.arm == arm) & (sb.view == view) & (sb.selection == selname)]
        return g.iloc[0] if len(g) else None

    d_own = [sbv('run_145', 'D', v, 'clean, own trigger') for v in 'xy']
    d_all = [sbv('run_145', 'D', v, 'all clean') for v in 'xy']
    d_cos = [sbv('run_149', 'D', v, 'clean, own trigger') for v in 'xy']
    rg_t = pd.DataFrame(rg) if rg else None

    head = []
    if d_own[0] is not None:
        head.append(f"<li><b>D beam box fit:</b> on all clean clusters the box start sits at the grid edge "
                    f"(t0 {d_all[0].stack_t0:.0f} ns, T {d_all[0].stack_T:.0f} ns, x). On D's <i>own</i> trigger "
                    f"(its wall and plastic fired) it is where A's and C's are: t0 {d_own[0].stack_t0:.0f} / "
                    f"{d_own[1].stack_t0:.0f} ns, T {d_own[0].stack_T:.0f} ± {d_own[0].stack_T_err:.0f} / "
                    f"{d_own[1].stack_T:.0f} ± {d_own[1].stack_T_err:.0f} ns (x / y). Only "
                    f"{100 * sb[(sb.run == 'run_145') & (sb.arm == 'D') & (sb.view == 'x') & (sb.selection == 'ACTIVITY')].own_frac_of_clean.iloc[0]:.0f} % "
                    f"of D's clean beam clusters are on its own trigger (A 68 %).</li>")
    if rg_t is not None:
        r = rg_t[(rg_t.run == 'run_145') & (rg_t.arm == 'D')]
        rc = rg_t[(rg_t.run == 'run_149') & (rg_t.arm == 'D')]
        ra = rg_t[(rg_t.run == 'run_145') & (rg_t.arm.isin(['A', 'C']))]
        head.append(f"<li><b>Why:</b> a ~{r.period_ns_med.median():.0f} ns (≈ 2.8 MHz) oscillation on tens of D "
                    f"strips: {100 * r.frac_ringing.min():.0f}–{100 * r.frac_ringing.max():.0f} % of beam triggers "
                    f"(cosmics {100 * rc.frac_ringing.min():.0f}–{100 * rc.frac_ringing.max():.0f} %; A, C on beam "
                    f"{100 * ra.frac_ringing.min():.0f}–{100 * ra.frac_ringing.max():.0f} %, and only in the first "
                    f"10–30 ms).</li>")
    if comb is not None and len(comb):
        head.append('<li><b>kw_D (B–D centroid lines, v1_D v = 31.0):</b> ' + '; '.join(
            f"{r.view}: fitted/true {r.ratio_BD:.3f} ± {r.err_BD:.3f} → kw {r.kw_D:.3f} ± {r.kw_D_err:.3f} "
            f"(kw·v {r.kwv:.1f} µm/ns, n {r.n_BD}); A–D {r.ratio_AD:.3f} ± {r.err_AD:.3f} ± {r.syst_AD:.3f} syst"
            for r in comb.itertuples()) + '</li>')
    if wall:
        head.append(f"<li><b>D wall (x):</b> s = {wall['s_best']:.3f} (profile {wall['s_prof_lo']:.3f}–"
                    f"{wall['s_prof_hi']:.3f}, sub-run bootstrap {wall['boot_lo']:.3f}–{wall['boot_hi']:.3f}) on "
                    f"{wall['n_single_x']} tracks → kw_x {wall['kw_x_from_wall']:.3f}; method systematic +3.6 % "
                    f"(A on the old chain).</li>")
    if eff is not None:
        e = eff.set_index(['tracks', 'partner'])
        head.append('<li><b>B:</b> cosmic efficiency on D-predicted crossings ' + ', '.join(
            f"{p} {100 * r.eff:.0f} ± {100 * r.eff_err:.0f} % ({r.tracks.split(' ')[0]} tracks)"
            for r in eff.itertuples() for p in [r.partner]) + '.</li>')

    plas = jl('wall_kw/plastic_D_v1.json')
    vhtml = verdict_html(sb, comb, wall, plas, dv, eff, bpos, rg, dbr)

    sbt = sb[sb.selection != 'ACTIVITY'] if sb is not None else None
    act = sb[sb.selection == 'ACTIVITY'] if sb is not None else None
    css = """body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;color:#1b2430}
table{border-collapse:collapse;margin:.8em 0;font-size:13px}td,th{border-bottom:1px solid #e6e9ee;padding:2px 8px;
text-align:right}th{color:#6a7583;font-weight:600}td:first-child,th:first-child{text-align:left}img{max-width:100%}
.verdict{border-left:4px solid #8a3f8f;padding:.4em 1em;background:#f7f4f8}figcaption{color:#6a7583;font-size:13px}
code{font-size:13px}"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>n_TOF pass 1: chambers D and B before the
bundles freeze</title><style>{css}</style></head><body>
<h1>n_TOF pass 1: chambers D and B before the bundles freeze</h1>
<div class="verdict">{vhtml}</div>
<h2>Headline numbers</h2><ul>{''.join(head)}</ul>

<h2>1. D on beam: why the column-duration box fit fails</h2>
<p>The fit (<code>colsum.GridFit</code>, pass-1 X template) is applied to the median stacked strip sum of the main
cluster, split by who made the trigger (scintillator stack: <code>hw_&lt;arm&gt;</code> = that arm's wall sum and
plastic fired). Same latency (27), window (20 × 60 ns) and DREAM for all four chambers; on cosmics D's box starts with
the others, so this is not a window or latency offset (the plan's withdrawn idea stays withdrawn).</p>
{fimg('stacks', "D's x strip-sum stack, beam (left) and cosmics (right). On its own trigger D shows the same box as A; on another arm's trigger the stack is flat with a 6-sample periodic structure.")}
{table(sbt, ['run', 'arm', 'view', 'selection', 'n', 'stack_t0', 'stack_T', 'stack_T_err', 'v', 'v_err', 'frac_prompt', 'frac_late', 'frac_end_past_window'],
       dict(stack_t0='{:.0f}', stack_T='{:.0f}', stack_T_err='{:.0f}', v='{:.1f}', v_err='{:.1f}', frac_prompt='{:.2f}', frac_late='{:.2f}', frac_end_past_window='{:.2f}'))}
<p>Activity per plane (fraction of triggers with a cluster that are busy, i.e. more than one cluster; fraction that
are clean; share of the clean ones on the arm's own trigger):</p>
{table(act, ['run', 'arm', 'view', 'n', 'frac_busy', 'frac_clean', 'own_frac_of_clean', 'nsig_p50'],
       dict(frac_busy='{:.2f}', frac_clean='{:.2f}', own_frac_of_clean='{:.2f}', nsig_p50='{:.0f}'))}
{fimg('ringing_examples', 'Strip × time images of the main pulsed region. The beam-on oscillation puts three to four in-phase humps ~360 ns apart on tens of strips; the block-median common mode does not remove it because it is not on the whole 64-channel block.')}
{table(rg_t, ['run', 'arm', 'plane', 'n_events', 'frac_ge3_pulsed', 'frac_ringing', 'period_ns_med', 'ringing_channels_per_event_med', 'ringing_odd_channel_frac'],
       dict(frac_ge3_pulsed='{:.2f}', frac_ringing='{:.3f}', period_ns_med='{:.0f}', ringing_channels_per_event_med='{:.0f}', ringing_odd_channel_frac='{:.2f}'))}
{fimg('ringing_time', 'Ringing vs time since the gamma flash. A and C ring after the flash and are quiet by 15-30 ms; D x starts ringing at ~5 ms and rings on every trigger to 80 ms.')}
<h3>D's own beam fits (v1_D, run_145 stat090_0000 _000)</h3>
{table(dbr)}

<h2>2. D's angle scale without an A–C line</h2>
<p><b>Centroid lines.</b> A partner chamber contributes only the charge centroid of its main cluster (no angle, no v),
and D its fitted p0 at the mesh. The line through the two is the truth; D's fitted tan is compared on
|true tan| 0.15–0.45 (the A–C rule). Partners: B (parallel, opposite, 468 mm; its positions need no field cage) and A
(perpendicular: the line is steep in A, which lights only part of its column above threshold, so A's centroid is placed
at the depth of its visible charge from its mean time, <code>depth=time</code>). C–D lines do not exist: a line shallow
in D meets D on its dead x half (137 dead channels at local x &gt; +40 mm).</p>
<p><b>Validation on A and C</b> (run_149 cos_0001, v1k bundles): the centroid line reproduces ac_line.py's fitted-line
ratio to ~1 %.</p>
{table(val[(val.miss_max.isna()) | (val.miss_max == 60.0)] if val is not None else None, ['method', 'partner_strict', 'arm', 'view', 'n', 'ratio', 'err'], dict(ratio='{:.3f}', err='{:.3f}'))}
<p><b>D</b> (all configurations; the headline uses partner A with depth from time and partner B at mid-gap, strict
partner, miss &lt; 60 mm):</p>
{fimg('lines', "D's fitted tan against the centroid-line tan. A fitted/true ratio below 1 means D's kw·v is too large.")}
{table(comb, None, dict(ratio_BD='{:.3f}', err_BD='{:.3f}', ratio_AD='{:.3f}', err_AD='{:.3f}', syst_AD='{:.3f}', kw_D='{:.3f}', kw_D_err='{:.3f}', kwv='{:.1f}'))}
{table(meas[(meas.miss_max == 60.0)] if meas is not None else None, ['partner', 'partner_strict', 'depth', 'point', 'lever_min', 'view', 'n', 'ratio', 'err', 'res_tan_68'], dict(ratio='{:.3f}', err='{:.3f}', res_tan_68='{:.3f}'))}
<p><b>D's own wall</b> (x only; the edge likelihood is valid on cosmics, not on beam):</p>
{fimg('wall', 'Profile of the wall edge likelihood in the scale s.')}
{table(walls, ['selection', 'sample', 'n', 's', 's_err', 'sigma'], dict(s='{:.3f}', s_err='{:.3f}', sigma='{:.1f}'))}
<p><b>D's timing v on the line muons</b> (box fit to D's stacked strip sum on the centroid-line events):</p>
{table(dv, None, dict(t0='{:.0f}', T='{:.0f}', T_err='{:.0f}', v='{:.1f}', v_err='{:.1f}', end_ns='{:.0f}'))}

<h3>3. B</h3>
{table(eff, None, dict(eff='{:.3f}', eff_err='{:.3f}', n_acc='{:.1f}'))}
<p>B fitted at v 42.6 and 30.0 on the same events: the p0 shift between the two (positions must not depend on v), and
the winner changes:</p>
{table(bpos, None, dict(frac_other_winner='{:.3f}', dp0_med='{:.3f}', dp0_68='{:.3f}', dp0_95='{:.3f}', tan_ratio_42_over_30='{:.3f}'))}

<h2>What this does not rule out</h2>
<ul>
<li>The oscillation's source is not identified from the data (it is beam-gated and strongest on D's FEUs 1/2); nothing
here says whether it also biases D's own-trigger beam fits beyond the column-duration lengthening seen.</li>
<li>The centroid line with a steep partner (A–D) relies on the depth-from-time placement of A's visible charge; it is
validated on A–C only where that correction is small.</li>
<li>Statistics: 12 staged run_149 sub-runs (of 87); D is gated in few cosmic events (|tan| &lt; 0.6 in both views, half
its x plane dead).</li>
<li>B's field-cage distortion of positions is not measured (no B truth); its positions are used as positions only.</li>
</ul>
<p><small>Generated by <code>rerun/ntof/tests/d_b/make_report.py</code> from the tables in
<code>{D}</code>.</small></p></body></html>"""
    open(f'{D}/report.html', 'w').write(doc)
    print(f'wrote {D}/report.html')


if __name__ == '__main__':
    main()
