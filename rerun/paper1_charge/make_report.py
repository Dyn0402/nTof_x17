#!/usr/bin/env python3
"""make_report.py --in D [--out O] -- Paper I charge on the pass-1 bench tables: numbers, figures, report.html.

Input: the output of charge_rows.py (D/rows/*.json, D/events/*.npz; on EOS /eos/user/d/dneff/x17rerun/paper1_charge/v1).
Output (default: this directory): results/summary.json, results/rows.csv, figures/*.png|pdf, report.html, and the
Paper I figure papers/paper1 charge_balance source (figures/charge_balance.pdf).

Estimators of a row's charge (q = (q_x + q_y) cos(theta), unsaturated by the 3550 rule, interior >= 30 mm):
  median_unsat   the median of the unsaturated events (PASS_PLAN §4: charge from unsaturated events only)
  cens_q         the q-th quantile of ALL selected events with every saturated event ranked above every
                 unsaturated one (a saturated event's charge is only known to be large).  Exact when every saturated
                 event lies above that quantile; valid only for f_sat < 1 - q.  The unsaturated events still carry
                 every number: a saturated event contributes its rank, never its fitted charge.
The gain curve (d ln Q / dV) is fitted to cens_25 (f_sat < 0.6) and checked against median_unsat (f_sat < 0.15).
"""
import argparse
import csv
import glob
import html
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DETS = ['mx17_2', 'mx17_3', 'mx17_4', 'mx17_6', 'mx17_7']
NAME = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
# reference categorical palette (dataviz references/palette.md, light mode), slots 1-5 in fixed order
COL = {'mx17_2': '#2a78d6', 'mx17_3': '#eb6834', 'mx17_4': '#1baf7a', 'mx17_6': '#eda100', 'mx17_7': '#e87ba4'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
GAP_MM = {'mx17_2': 31.12, 'mx17_3': 28.08, 'mx17_4': 30.48, 'mx17_6': 28.0, 'mx17_7': 29.44}  # physics_inputs.json
N_MIN = 100                       # interior events for a row to enter a curve
FSAT_CENS = 0.6                   # cens_25 is used up to this saturated fraction (needs f_sat < 0.75)
FSAT_UNSAT = 0.15                 # median_unsat is used for the check only below this
# Series excluded from the gain fits, with the reason (written into the report)
EXCLUDE = {
    'mx17_det3_det4_overnight_6-23-26': 'M3 reference degraded on this night (~4 % clean tracks, qa_config); the '
                                        'charge is flat over 465-525 V in BOTH chambers although the HV readback '
                                        'follows the scan (hv_monitor.csv), so the M3-matched sample is not the '
                                        'muon sample. Sub-runs were taken in descending HV over the night, so a '
                                        'gain drift with time could also cancel the scan. Not separable here.',
    'mx17_det1_det2_overnight_6-17-26': 'det2 6-17: half-dead X view, 1-115 interior events per row.',
}
N_BOOT = 300
RNG = np.random.default_rng(7)


def cens_quantile(q, sat, p):
    """p-quantile with saturated events ranked above all unsaturated ones; nan if it falls among the saturated."""
    n = len(q)
    if n == 0:
        return np.nan
    k = int(np.floor(p * (n - 1)))
    qu = np.sort(q[~sat])
    return float(qu[k]) if k < len(qu) else np.nan


def boot(fn, q, sat, n=N_BOOT):
    if len(q) < 20:
        return np.nan
    idx = RNG.integers(0, len(q), size=(n, len(q)))
    v = np.array([fn(q[i], sat[i]) for i in idx])
    v = v[np.isfinite(v)]
    return float(np.std(v)) if len(v) > n // 2 else np.nan


def row_numbers(o, z):
    q = z['qx'] + z['qy']
    sat = z['sat_x'] | z['sat_y']
    k = np.isfinite(q)
    q, sat = q[k], sat[k]
    out = dict(o)
    med_u = (lambda a, s: float(np.median(a[~s])) if (~s).sum() else np.nan)
    out['est'] = dict(
        n=int(len(q)), f_sat=float(sat.mean()) if len(q) else np.nan,
        median_unsat=med_u(q, sat), median_unsat_err=boot(med_u, q, sat),
        cens_25=cens_quantile(q, sat, 0.25), cens_25_err=boot(lambda a, s: cens_quantile(a, s, 0.25), q, sat),
        cens_50=cens_quantile(q, sat, 0.50), cens_50_err=boot(lambda a, s: cens_quantile(a, s, 0.50), q, sat),
        median_all_fit=float(np.median(q)) if len(q) else np.nan)
    # how often a saturated event's own fitted charge falls BELOW cens_25: the assumption behind the estimator
    c25 = out['est']['cens_25']
    out['est']['sat_below_cens25'] = float(np.mean(q[sat] < c25)) if sat.sum() and np.isfinite(c25) else np.nan
    u = ~sat
    if u.sum() > 20:
        lr = np.log(z['qx'][k][u] / z['qy'][k][u])
        lr = lr[np.isfinite(lr)]
        med = float(np.median(lr))
        out['est']['xy_ln_median'] = med
        out['est']['xy_ln_core_sigma'] = float(1.4826 * np.median(np.abs(lr - med)))
        out['est']['xy_ln_s68'] = float(0.5 * np.diff(np.percentile(lr, [16, 84]))[0])
        out['est']['xy_tail_frac'] = float(np.mean(np.abs(lr - med) > 0.5))
        fx = z['qx'][k][u] / (z['qx'][k][u] + z['qy'][k][u])                  # Paper I's f = q_X / (q_X + q_Y)
        fx = fx[np.isfinite(fx)]
        out['est']['f_median'] = float(np.median(fx))
        out['est']['f_s68'] = float(0.5 * np.diff(np.percentile(fx, [16, 84]))[0])
    return out


def series_key(r):
    return (r['det'], r['run'], r['drift_V'])


def fit_line(V, y, e):
    w = 1.0 / np.maximum(e, 1e-9) ** 2
    A = np.column_stack([np.ones_like(V), V - V.mean()])
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ c) ** 2))
    return dict(a=float(c[0]), b=float(c[1]), b_err=float(np.sqrt(cov[1, 1])), V0=float(V.mean()),
                chi2=chi2, dof=int(len(V) - 2))


def gain_fits(rows):
    by = {}
    for r in rows:
        if r['subrun'].startswith(('long', 'short', 'longer', 'quick')) and 'resist' not in r['subrun']:
            continue                                     # only the scan sub-runs define a curve
        by.setdefault(series_key(r), []).append(r)
    fits = []
    for (det, run, drift), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r['resist_V'])
        if len({r['resist_V'] for r in rs}) < 4:
            continue
        ent = dict(det=det, run=run, drift_V=drift, excluded=EXCLUDE.get(run), points=[])
        for r in rs:
            e = r['est']
            ent['points'].append(dict(V=r['resist_V'], n=e['n'], f_sat=e['f_sat'], cens_25=e['cens_25'],
                                      cens_25_err=e['cens_25_err'], median_unsat=e['median_unsat'],
                                      median_unsat_err=e['median_unsat_err'], subrun=r['subrun']))
        for est, flim in (('cens_25', FSAT_CENS), ('median_unsat', FSAT_UNSAT)):
            P = [p for p in ent['points'] if p['n'] >= N_MIN and p['f_sat'] < flim and np.isfinite(p[est])
                 and np.isfinite(p[est + '_err']) and p[est] > 0]
            if len(P) < 3:
                ent[est] = None
                continue
            V = np.array([p['V'] for p in P])
            y = np.log([p[est] for p in P])
            e = np.array([p[est + '_err'] / p[est] for p in P])
            f = fit_line(V, y, e)
            f.update(V_min=float(V.min()), V_max=float(V.max()), n_points=len(P),
                     per10V=10 * f['b'], per10V_err=10 * f['b_err'], doubling_V=float(np.log(2) / f['b']))
            if len(P) >= 5:                              # curvature: local slopes of the low and high halves
                h = len(P) // 2
                lo, hi = fit_line(V[:h + 1], y[:h + 1], e[:h + 1]), fit_line(V[h:], y[h:], e[h:])
                f['per10V_low'], f['per10V_high'] = 10 * lo['b'], 10 * hi['b']
                f['per10V_low_err'], f['per10V_high_err'] = 10 * lo['b_err'], 10 * hi['b_err']
            ent[est] = f
        fits.append(ent)
    return fits


def drift_scans(rows):
    """series at fixed mesh HV with >= 4 drift voltages (det3 6-27 scan + its 1000 V long run)."""
    by = {}
    for r in rows:
        by.setdefault((r['det'], r['run'], r['resist_V']), []).append(r)
    out = []
    for (det, run, V), rs in by.items():
        if len({r['drift_V'] for r in rs}) < 4:
            continue
        pts = []
        for r in sorted(rs, key=lambda r: r['drift_V']):
            e = r['est']
            pts.append(dict(drift_V=r['drift_V'], E_V_cm=r['drift_V'] / (GAP_MM[det] / 10), n=e['n'], f_sat=e['f_sat'],
                            cens_50=e['cens_50'], cens_50_err=e['cens_50_err'], cens_25=e['cens_25'],
                            cens_25_err=e['cens_25_err'], median_unsat=e['median_unsat'], subrun=r['subrun']))
        ref = [p for p in pts if p['drift_V'] == 1000.0 and 'scan' in p['subrun']]    # a scan sub-run, not the long run
        refv = ref[0]['cens_50'] if ref else max(p['cens_50'] for p in pts if np.isfinite(p['cens_50']))
        for p in pts:
            p['rel_cens_50'] = p['cens_50'] / refv
            p['rel_cens_50_err'] = p['cens_50_err'] / refv
        out.append(dict(det=det, run=run, resist_V=V, points=pts))
    return out


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def fig_gain(fits, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    ax = axes[0]
    for f in fits:
        if f['excluded']:
            continue
        det = f['det']
        P = [p for p in f['points'] if p['n'] >= N_MIN and np.isfinite(p['cens_25'])]
        if not P:
            continue
        V = np.array([p['V'] for p in P])
        y = np.array([p['cens_25'] for p in P])
        e = np.array([p['cens_25_err'] for p in P])
        use = np.array([p['f_sat'] < FSAT_CENS for p in P])
        mk = 'o' if '6-27' in f['run'] or det != 'mx17_3' else 's'
        lab = f"{NAME[det]} {f['run'][-7:-3]}, drift {f['drift_V']:.0f} V" + (' (squares)' if mk == 's' else '')
        ax.errorbar(V[use], y[use], yerr=e[use], fmt=mk, ms=5, color=COL[det], mec='white', mew=0.8, lw=1,
                    label=lab, zorder=3)
        ax.plot(V[~use], y[~use], mk, ms=5, mfc='none', mec=COL[det], zorder=2)
        g = f['cens_25']
        if g:
            vv = np.linspace(g['V_min'], g['V_max'], 20)
            ax.plot(vv, np.exp(g['a'] + g['b'] * (vv - g['V0'])), color=COL[det], lw=1.5, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('mesh / resistive-strip HV [V]', color=INK)
    ax.set_ylabel('charge, 25th percentile [ADC, fit units]', color=INK)
    ax.set_title('Gain curves (unsaturated rank estimator)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    ax = axes[1]
    for f in fits:
        if not f['excluded']:
            continue
        P = [p for p in f['points'] if np.isfinite(p['cens_25'])]
        ax.errorbar([p['V'] for p in P], [p['cens_25'] for p in P], yerr=[p['cens_25_err'] for p in P], fmt='o',
                    ms=5, color=COL[f['det']], mec='white', mew=0.8, lw=1,
                    label=f"{NAME[f['det']]} {f['run'][-7:]}")
    ax.set_yscale('log')
    ax.set_xlabel('mesh / resistive-strip HV [V]', color=INK)
    ax.set_title('Excluded nights (see text)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'gain_curves.{ext}'), dpi=150)
    plt.close(fig)


def fig_bias(rows, figdir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for det in DETS:
        R = [r for r in rows if r['det'] == det and r['est']['n'] >= N_MIN and np.isfinite(r['est']['cens_50'])
             and r['run'] not in EXCLUDE]
        x = [r['est']['f_sat'] for r in R]
        y = [r['est']['median_unsat'] / r['est']['cens_50'] for r in R]
        ax.plot(x, y, 'o', ms=5, color=COL[det], mec='white', mew=0.8, label=NAME[det])
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xlabel('saturated fraction of the interior events', color=INK)
    ax.set_ylabel('median(unsaturated) / rank-censored median', color=INK)
    ax.set_title('What "unsaturated only" does to a median', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'unsat_bias.{ext}'), dpi=150)
    plt.close(fig)


def fig_drift(scans, figdir):
    if not scans:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for s in scans:
        P = [p for p in s['points'] if np.isfinite(p['rel_cens_50']) and 'scan' in p['subrun']]
        L = [p for p in s['points'] if np.isfinite(p['rel_cens_50']) and 'scan' not in p['subrun']]
        ax.errorbar([p['E_V_cm'] for p in P], [p['rel_cens_50'] for p in P], yerr=[p['rel_cens_50_err'] for p in P],
                    fmt='o-', ms=5, color=COL[s['det']], mec='white', mew=0.8, lw=1.5,
                    label=f"{NAME[s['det']]} {s['run'][-7:]}, mesh {s['resist_V']:.0f} V, scan sub-runs")
        if L:
            ax.errorbar([p['E_V_cm'] for p in L], [p['rel_cens_50'] for p in L],
                        yerr=[p['rel_cens_50_err'] for p in L], fmt='D', ms=6, mfc='none', color=COL[s['det']],
                        lw=1, label='same, the long run (other time of day)')
    ax.set_xlabel('nominal drift field = drift voltage / gap [V/cm]', color=INK)
    ax.set_ylabel('charge / charge at 1000 V drift', color=INK)
    ax.set_title('Charge vs drift field (rank-censored median)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'drift_field.{ext}'), dpi=150)
    plt.close(fig)


def fig_balance(golden, ev, figdir, paper_out=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(-0.6, 0.6, 61)
    for r in golden:
        z = ev[r['key']]
        u = ~(z['sat_x'] | z['sat_y'])
        lr = np.log(z['qx'][u] / z['qy'][u])
        lr = lr[np.isfinite(lr)]
        e = r['est']
        ax.hist(lr, bins=bins, histtype='step', lw=1.6, color=COL[r['det']], density=True,
                label=f"{NAME[r['det']]}: median {e['xy_ln_median']:+.3f}, core σ {e['xy_ln_core_sigma']:.3f}"
                      f" (n={len(lr)})")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel('ln(Q_x / Q_y), unsaturated events, golden rows', color=INK)
    ax.set_ylabel('events (normalised)', color=INK)
    ax.set_title('X/Y charge balance (pass 1, waveform fit)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'charge_balance.{ext}'), dpi=150)
    plt.close(fig)


def fig_spectra(golden, ev, figdir):
    fig, axes = plt.subplots(1, len(golden), figsize=(3.0 * len(golden), 3.2), sharey=False)
    for ax, r in zip(np.atleast_1d(axes), golden):
        z = ev[r['key']]
        q = z['qx'] + z['qy']
        sat = z['sat_x'] | z['sat_y']
        hi = np.percentile(q, 99)
        b = np.linspace(0, hi, 50)
        ax.hist(q, bins=b, histtype='stepfilled', color=GRID, label='all (fitted)')
        ax.hist(q[~sat], bins=b, histtype='step', lw=1.6, color=COL[r['det']], label='unsaturated')
        ax.set_title(f"{NAME[r['det']]}  f_sat {r['est']['f_sat']:.2f}", color=INK, fontsize=10, loc='left')
        ax.set_xlabel('Q cosθ [ADC]', color=INK)
        style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(figdir, f'golden_spectra.{ext}'), dpi=150)
    plt.close(fig)


def fmt(x, p=3):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{p}f}'


def table(head, rows):
    h = ''.join(f'<th>{html.escape(c)}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    figdir = os.path.join(a.out, 'figures')
    resdir = os.path.join(a.out, 'results')
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(resdir, exist_ok=True)
    rows, ev = [], {}
    for f in sorted(glob.glob(os.path.join(a.inp, 'rows', '*.json'))):
        o = json.load(open(f))
        z = dict(np.load(os.path.join(a.inp, 'events', o['key'] + '.npz'), allow_pickle=True))  # v1 stored acq as object
        ev[o['key']] = z
        rows.append(row_numbers(o, z))
    fits = gain_fits(rows)
    scans = drift_scans(rows)
    golden = sorted([r for r in rows if r['cls'] in ('golden_calib',)], key=lambda r: DETS.index(r['det']))
    # sign check over every row with enough planes
    sign_rows = []
    for r in rows:
        for p in ('x', 'y'):
            s = r['sign'][p]
            if s['n'] >= 200 and s['corr_w_tanref'] is not None:
                sign_rows.append(dict(key=r['key'], det=r['det'], view=p, n=s['n'], corr=s['corr_w_tanref'],
                                      slope=s['slope_w_vs_tanref'], kw=s['kw'], same_sign=s['frac_same_sign'],
                                      excluded=r['run'] in EXCLUDE))
    sgn = dict(n_planes_rows=len(sign_rows), n_corr_pos=sum(s['corr'] > 0 for s in sign_rows),
               n_kw_pos=sum(s['kw'] > 0 for s in sign_rows), n_slope_pos=sum(s['slope'] > 0 for s in sign_rows),
               corr_min=min(s['corr'] for s in sign_rows), worst=min(sign_rows, key=lambda s: s['corr']))
    # golden-row operating point
    op = []
    for r in golden:
        e = r['est']
        f = [g for g in fits if g['det'] == r['det'] and not g['excluded'] and g['cens_25']]
        ext = None
        if f:
            g = max(f, key=lambda g: g['cens_25']['n_points'])['cens_25']
            if r['resist_V'] > g['V_max']:
                ext = float(np.exp(g['a'] + g['b'] * (r['resist_V'] - g['V0'])))
        op.append(dict(det=r['det'], key=r['key'], V=r['resist_V'], drift=r['drift_V'], n=e['n'], f_sat=e['f_sat'],
                       cens_25=e['cens_25'], cens_25_err=e['cens_25_err'], cens_50=e['cens_50'],
                       median_unsat=e['median_unsat'], cens25_extrap=ext,
                       xy_ln_median=e.get('xy_ln_median'), xy_ln_core_sigma=e.get('xy_ln_core_sigma'),
                       xy_tail_frac=e.get('xy_tail_frac'), f_median=e.get('f_median'), f_s68=e.get('f_s68')))
    # cross-day reproducibility: det3 at 490-495 V / 1000 V drift on different nights
    repro = [dict(key=r['key'], label=f"{r['run'][5:]}/{r['subrun']}", V=r['resist_V'], f_sat=r['est']['f_sat'], cens_25=r['est']['cens_25'],
                  cens_25_err=r['est']['cens_25_err']) for r in rows
             if r['det'] == 'mx17_3' and r['drift_V'] == 1000.0 and r['resist_V'] in (490.0,) and r['est']['n'] >= N_MIN]
    summary = dict(about=__doc__.split('\n')[0], selection=dict(edge_mm=30, det7_x_min_mm=204.68, n_min=N_MIN,
                                                                 fsat_cens=FSAT_CENS, fsat_unsat=FSAT_UNSAT),
                   gain_fits=fits, drift_scans=scans, golden=op, sign=sgn, repro_det3_490=repro, excluded=EXCLUDE)
    json.dump(summary, open(os.path.join(resdir, 'summary.json'), 'w'), indent=1, default=float)
    cols = ['key', 'det', 'run', 'subrun', 'cls', 'resist_V', 'drift_V']
    ecols = ['n', 'f_sat', 'median_unsat', 'median_unsat_err', 'cens_25', 'cens_25_err', 'cens_50', 'cens_50_err',
             'median_all_fit', 'sat_below_cens25', 'xy_ln_median', 'xy_ln_core_sigma', 'xy_s68', 'xy_tail_frac']
    with open(os.path.join(resdir, 'rows.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols + ecols + ['kw_x', 'kw_y', 'corr_w_tanref_x', 'corr_w_tanref_y'])
        for r in sorted(rows, key=lambda r: (r['det'], r['run'], r['drift_V'], r['resist_V'])):
            w.writerow([r[c] for c in cols] + [r['est'].get(c, r['est'].get('xy_ln_s68') if c == 'xy_s68' else None)
                                               for c in ecols] +
                       [r['sign']['x']['kw'], r['sign']['y']['kw'], r['sign']['x']['corr_w_tanref'],
                        r['sign']['y']['corr_w_tanref']])
    fig_gain(fits, figdir)
    fig_bias(rows, figdir)
    fig_drift(scans, figdir)
    fig_balance(golden, ev, figdir)
    fig_spectra(golden, ev, figdir)
    write_html(a.out, summary, rows)
    print(json.dumps(dict(gain=[(NAME[f['det']], f['run'][-7:], f['cens_25'] and round(f['cens_25']['per10V'], 3),
                                 f['median_unsat'] and round(f['median_unsat']['per10V'], 3)) for f in fits],
                          sign={k: v for k, v in sgn.items() if k != 'worst'}), indent=0))


def write_html(out, S, rows):
    fits = [f for f in S['gain_fits'] if not f['excluded'] and f['cens_25']]
    det3 = [f for f in fits if f['det'] == 'mx17_3']
    g_rows = []
    for f in S['gain_fits']:
        c, m = f['cens_25'], f['median_unsat']
        g_rows.append([NAME[f['det']], html.escape(f['run']), f"{f['drift_V']:.0f}",
                       'excluded' if f['excluded'] else 'used',
                       f"{fmt(c['per10V'])} ± {fmt(c['per10V_err'])}" if c else '–',
                       f"{c['V_min']:.0f}–{c['V_max']:.0f} ({c['n_points']})" if c else '–',
                       f"{fmt(c.get('per10V_low'))} / {fmt(c.get('per10V_high'))}" if c else '–',
                       f"{c['doubling_V']:.1f}" if c else '–', f"{c['chi2']:.1f}/{c['dof']}" if c else '–',
                       f"{fmt(m['per10V'])} ± {fmt(m['per10V_err'])} ({m['V_min']:.0f}–{m['V_max']:.0f})" if m else '–'])
    op_rows = [[NAME[o['det']], f"{o['V']:.0f}/{o['drift']:.0f}", str(o['n']), fmt(o['f_sat'], 2),
                f"{fmt(o['cens_25'], 0)} ± {fmt(o['cens_25_err'], 0)}", fmt(o['cens_50'], 0),
                fmt(o['median_unsat'], 0), fmt(o['cens25_extrap'], 0), fmt(o['xy_ln_median']),
                fmt(o['xy_ln_core_sigma']), fmt(o['xy_tail_frac'], 3), fmt(o['f_median']), fmt(o['f_s68'])] for o in S['golden']]
    d_rows = []
    for s in S['drift_scans']:
        for p in s['points']:
            d_rows.append([NAME[s['det']], f"{s['resist_V']:.0f}", f"{p['drift_V']:.0f}", f"{p['E_V_cm']:.0f}",
                           str(p['n']), fmt(p['f_sat'], 2), fmt(p['cens_50'], 0),
                           f"{fmt(p['rel_cens_50'])} ± {fmt(p['rel_cens_50_err'])}", fmt(p['median_unsat'], 0)])
    sg = S['sign']
    w = sg['worst']
    rep = S['repro_det3_490']
    verdict = (
        f"<p><b>Gain.</b> On every usable scan the charge rises exponentially with the mesh HV. Fitted to the "
        f"25th percentile of the interior charge, with saturated events counted by rank only: "
        + '; '.join(f"{NAME[f['det']]} {f['run'][-7:]}: <b>{f['cens_25']['per10V']:.3f} ± "
                    f"{f['cens_25']['per10V_err']:.3f}</b> in ln Q per 10 V (×2 every {f['cens_25']['doubling_V']:.1f} V, "
                    f"{f['cens_25']['V_min']:.0f}–{f['cens_25']['V_max']:.0f} V)" for f in fits)
        + ". The ladder is curved (the local slope rises with HV, columns low/high in the table), as found on r06; "
          "do not carry one slope outside its range.</p>"
        f"<p><b>Unsaturated only is not enough by itself.</b> A median over the unsaturated events stops rising "
        f"once a third of the events saturate (figure 2): above ~500 V on det2/det3 and at the det6/det7 operating "
        f"points (f_sat 0.86–0.92) it measures the 3550-ADC rule, not the chamber. Every number here therefore "
        f"comes from the unsaturated events but treats a saturated event as 'large' (rank-censored quantiles), "
        f"which is exact as long as the saturated events lie above the quantile; their own fitted charge falls "
        f"below the 25th percentile in "
        f"{np.nanmedian([r['est']['sat_below_cens25'] for r in rows if r['est']['n'] >= N_MIN]) * 100:.1f} % "
        f"of cases (median over rows), so the assumption holds to a few per cent.</p>"
        f"<p><b>Reproducible across nights.</b> det3 at 490 V / 1000 V drift: "
        + ', '.join(f"{html.escape(r['label'])} {r['cens_25']:.0f} ± {r['cens_25_err']:.0f}"
                    for r in rep) + ".</p>"
        f"<p><b>Bench tan sign.</b> On {sg['n_corr_pos']}/{sg['n_planes_rows']} (row, view) pairs with ≥ 200 "
        f"M3-matched planes the fitted transverse speed w rises with the M3 tangent (corr &gt; 0; lowest "
        f"{w['corr']:.2f}, {html.escape(w['key'])} {w['view']}); kw &gt; 0 on {sg['n_kw_pos']}/{sg['n_planes_rows']}. "
        f"With M3's z pointing up (ray files Z_Up = 1302 mm, Z_Down = 24 mm: the top plane is L0, cosmic muons go "
        f"down), charge that arrives later comes from higher up on every chamber and both slots: the readout board is "
        f"down, electrons drift down onto the mesh, and the pass's tan = +d(strip coordinate)/dz_up in both views.</p>")
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Paper I charge on pass 1</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#1f1f1e;line-height:1.45}}
table{{border-collapse:collapse;font-size:13px;margin:0.6em 0 1.4em}}td,th{{border-bottom:1px solid #e4e3dd;padding:3px 8px;text-align:right}}
th{{color:#6b6a64;font-weight:600}}td:first-child,th:first-child{{text-align:left}}figure{{margin:1.2em 0}}
figcaption{{color:#6b6a64;font-size:13px}}img{{max-width:100%}}code{{font-size:12px}}</style></head><body>
<h1>Paper I charge on the pass-1 bench tables</h1>
<p style="color:#6b6a64">Generated by <code>rerun/paper1_charge/make_report.py</code> from <code>charge_rows.py</code> on
EOS <code>Analysis_pass1</code> (code 6a0bb1f/b5a29af). Charge = fitted view charge q_sum (ADC units of the
deconvolved fit, no fC calibration), × cos θ from the M3 tangents, X + Y. Interior ≥ 30 mm, det7 X ch 256–511 only,
single-track, spark-free, M3-matched in both views, no χ² gate.</p>
<h2>Verdict</h2>{verdict}
<h2>Gain curves</h2>
{table(['chamber', 'night', 'drift V', 'status', 'd lnQ / 10 V (cens 25 %)', 'range (points)',
        'local low / high half', 'doubling V', 'χ²/dof', 'check: median unsat, f_sat < 0.15'], g_rows)}
<p>Excluded nights: {'; '.join(f"<b>{html.escape(k)}</b>: {html.escape(v)}" for k, v in S['excluded'].items())}</p>
<figure><img src="figures/gain_curves.png"><figcaption>Figure 1. Charge (25th percentile, rank-censored) vs mesh HV.
Filled: used in the fit (f_sat &lt; {FSAT_CENS}); open: more saturated. Right: the excluded 6-23 night, flat in both
chambers.</figcaption></figure>
<figure><img src="figures/unsat_bias.png"><figcaption>Figure 2. The median of the unsaturated events, divided by the
rank-censored median, against the saturated fraction. Below 1 = the unsaturated-only median is biased low.</figcaption></figure>
<h2>Golden rows (operating points)</h2>
{table(['chamber', 'mesh/drift V', 'n', 'f_sat', 'cens 25 %', 'cens 50 %', 'median unsat', 'cens 25 % from the HV curve',
        'ln(Qx/Qy) median', 'core σ', 'tail |Δ| > 0.5', 'f = Qx/(Qx+Qy) median', 'f s68'], op_rows)}
<figure><img src="figures/golden_spectra.png"><figcaption>Figure 3. Golden-row charge spectra: all interior events
(grey, fitted charge) and the unsaturated ones.</figcaption></figure>
<figure><img src="figures/charge_balance.png"><figcaption>Figure 4. X/Y charge balance on the golden rows (unsaturated
events). This is the Paper I <code>charge_balance</code> figure, now from the waveform fit.</figcaption></figure>
<h2>Drift field</h2>
{table(['chamber', 'mesh V', 'drift V', 'E [V/cm]', 'n', 'f_sat', 'cens 50 %', 'relative to 1000 V', 'median unsat'], d_rows)}
<figure><img src="figures/drift_field.png"><figcaption>Figure 5. Charge vs nominal drift field (drift voltage / the
chamber's gap from physics_inputs.json).</figcaption></figure>
<h2>What this does not rule out</h2><ul>
<li>The charge is in fit ADC units. There is no electronics charge calibration of the bench DREAM gain setting in
this analysis, so no absolute gain (electrons) is quoted.</li>
<li>The 3550 rule misses 5–46 % of truly clipped samples (K4); those events count as unsaturated with a charge biased
low. Strongest on det6 (46 %) and det4 (37 %).</li>
<li>Rank censoring assumes saturated events are above the quantile; a head-on track concentrates charge and can
saturate at a lower total, so a small low-side leak remains (the 'sat below' column of rows.csv).</li>
<li>At the lowest drift fields the column can exceed the readout window and drift slowly through the gas; the
charge drop there mixes window truncation, attachment and transparency, and is not decomposed.</li>
<li>det4 has no usable bench HV scan (the 6-23 night is excluded); its gain curve must come from SPS.</li>
<li>det6's gain rose ×10 through the 6-26 night (short_run → long_run at the same 495 V; the v(t) study's early
contaminant). Only the 6-26 afternoon HV scan, after it settled, enters the curve.</li>
<li>Time dependence of the gain inside a row is not corrected.</li></ul>
</body></html>"""
    open(os.path.join(out, 'report.html'), 'w').write(body)


if __name__ == '__main__':
    main()
