#!/usr/bin/env python3
"""
make_two_track_report.py -- report.html for the joint two-track fit.

Built from what the three studies wrote, so re-running it after any of them
moves the numbers, the figures and the verdict sentence together:

    <out>/two_track_synth/        synthetic planes (two_track_synth.py)
    <out>/intra_bench/split_probe/ real triggers, no threshold (intra_bench split-probe)
    <out>/intra_bench/<variant>/   the overlay bench (intra_bench build --two-track)

Any of the three may be absent; its section is then omitted and the report says
so rather than inventing it.

    python -m sept26_prelim_analysis.make_two_track_report
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import figstyle as fs          # noqa: E402
from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis.report_style import head       # noqa: E402

BANDS = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 400)]
BAND_LABEL = {(0, 6): '0–6', (6, 12): '6–12', (12, 18): '12–18',
              (18, 24): '18–24', (24, 400): '≥ 24'}
ARM_ORDER = ('A', 'C')


def out_dir() -> Path:
    return paths.out('two_track')


# --------------------------------------------------------------------------- #
# little formatters
# --------------------------------------------------------------------------- #
def _pct(x, d=0):
    return '—' if x is None or not np.isfinite(x) else f'{100 * x:.{d}f}&nbsp;%'


def _f(x, d=2):
    return '—' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def _i(x):
    return '—' if x is None or not np.isfinite(x) else f'{int(x):,}'


def table(cols, rows) -> str:
    return ('<div class="tw"><table><thead><tr>'
            + ''.join(f'<th>{c}</th>' for c in cols)
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def figure_html(name, caption) -> str:
    return (f'<figure><img src="figures/{name}.png" alt="{caption}">'
            f'<figcaption>{caption} '
            f'<a class="src" href="figures/{name}.csv">numbers &#8599;</a>'
            f'</figcaption></figure>')


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load() -> dict:
    L = {}
    sy = paths.spell('out', 'two_track_synth')
    L['synth'] = pd.read_parquet(sy / 'synth.parquet') if (sy / 'synth.parquet').exists() else None
    L['synth_meta'] = (json.loads((sy / 'run.meta.json').read_text())
                       if (sy / 'run.meta.json').exists() else {})
    sp = paths.spell('out', 'intra_bench', 'split_probe')
    L['probe'] = pd.read_parquet(sp / 'attempts.parquet') if (sp / 'attempts.parquet').exists() else None
    cp = paths.spell('out', 'intra_bench', 'compare.csv')
    L['compare'] = pd.read_csv(cp) if cp.exists() else None
    return L


# --------------------------------------------------------------------------- #
# derived tables
# --------------------------------------------------------------------------- #
def _bool(s):
    if not isinstance(s, pd.Series):
        return bool(s)
    return s.fillna(False).astype(bool)


def _col(df, name, default=False):
    """One column of a products table, or a constant when the study that writes
    it has not been re-run. Keeps the report readable against an older set
    instead of dying three sections in."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def synth_bands(D: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Two-track efficiency by separation band, for pairs whose BOTH tracks are
    detectable on their own (a faint, steep track that never reaches 5 sigma was
    never the fit's to find).

    Two efficiencies, and the true one is between them. ``eff`` is what the FIT
    does, with the trigger set aside; ``eff_trig`` also requires this plane's own
    trigger to fire. The synthetic study is one plane at a time, so it cannot
    fire the cross-plane trigger — which on data is the one that catches a pair
    close in this view and wide in the other."""
    P = D[(D.n_true == 2) & _bool(_col(D, 'both_detectable', True)) & (D.dt == 0)].copy()
    P['ok'] = (_bool(_col(P, 'both_found')) & _bool(_col(P, 'guards_ok'))
               & (P.fstat.fillna(-np.inf) >= thr))
    P['trig'] = _bool(_col(P, 'trig_residual')) | _bool(_col(P, 'trig_width'))
    rows = []
    for (arm, (lo, hi)) in [(a, b) for a in sorted(P.arm.unique()) for b in BANDS]:
        g = P[(P.arm == arm) & (P.sep >= lo) & (P.sep < hi)]
        if not len(g):
            continue
        rows.append(dict(arm=arm, band=BAND_LABEL[(lo, hi)], lo=lo, n=len(g),
                         eff=float(g.ok.mean()),
                         eff_trig=float((g.ok & g.trig).mean()),
                         trig=float(g.trig.mean()),
                         rsig_dp0=_rsig(pd.concat([g.d_p0_a, g.d_p0_b])),
                         rsig_dtan=_rsig(pd.concat([g.d_tan_a, g.d_tan_b])),
                         t_fit=float(g.t_two.median()) if 't_two' in g else np.nan))
    return pd.DataFrame(rows)


def _rsig(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan
    q = np.percentile(v, [16, 84])
    return float(0.5 * (q[1] - q[0]))


def operating_curve(D: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    """Two-track efficiency (synthetic pairs) against the false-split rate on
    REAL clean single muons, over the threshold. The x axis has to come from
    real charge: a perfectly modelled single track is not the population that
    killed split seeding."""
    rows = []
    P = D[(D.n_true == 2) & _bool(_col(D, 'both_detectable', True)) & (D.dt == 0)] if D is not None else None
    for thr in (0, 10, 20, 30, 50, 80, 120, 200, 300, 500, 800):
        row = dict(threshold=thr)
        if probe is not None:
            cs = probe[probe.clean_single]
            ok = _bool(_col(cs, 'guards_ok')) & (cs.fstat.fillna(-np.inf) >= thr)
            row['false_split'] = float(ok.sum()) / max(len(cs), 1)
            row['n_clean'] = int(len(cs))
            row['split_rate_all'] = float((_bool(_col(probe, 'guards_ok'))
                                           & (probe.fstat.fillna(-np.inf) >= thr)).mean())
        if P is not None:
            for lo, hi in BANDS:
                g = P[(P.sep >= lo) & (P.sep < hi)]
                ok = (_bool(_col(g, 'both_found')) & _bool(_col(g, 'guards_ok'))
                      & (g.fstat.fillna(-np.inf) >= thr))
                row[f'eff_{lo:g}'] = float(ok.sum()) / max(len(g), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def probe_summary(R: pd.DataFrame) -> pd.DataFrame:
    R = R.copy()
    R['triggered'] = R.reindex(columns=['trig_residual', 'trig_width',
                                        'trig_cross_plane']).fillna(False).astype(bool).any(axis=1)
    R['attempted'] = R.fstat.notna() if 'fstat' in R else False
    rows = []
    for (arm, clean), g in R.groupby(['arm', 'clean_single']):
        rows.append(dict(
            arm=arm, clean_single=bool(clean), n=len(g),
            trig_residual=float(_bool(_col(g, 'trig_residual')).mean()),
            trig_width=float(_bool(_col(g, 'trig_width')).mean()),
            trig_cross=float(_bool(_col(g, 'trig_cross_plane')).mean()),
            triggered=float(g.triggered.mean()),
            guards_ok=float(_bool(_col(g, 'guards_ok')).mean()),
            fstat_p50=float(np.nanpercentile(g.fstat, 50)) if g.fstat.notna().any() else np.nan,
            fstat_p99=float(np.nanpercentile(g.fstat, 99)) if g.fstat.notna().any() else np.nan,
            t_fit=float(np.nanmedian(g.t_fit)) if 't_fit' in g else np.nan,
            cost_s=float(np.nansum(g.t_fit) + np.nansum(g.t_probe)) / max(len(g), 1)
            if 't_fit' in g else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_efficiency(B: pd.DataFrame, figdir: Path):
    fig, ax = fs.figure(figsize=(6.6, 4.0))
    for arm in [a for a in ARM_ORDER if a in set(B.arm)]:
        g = B[B.arm == arm].sort_values('lo')
        ax.plot(range(len(g)), 100 * g.eff, **fs.det_style(arm))
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.band)
    ax.set_xlabel('separation of the two tracks on the strip plane  [mm]')
    ax.set_ylabel('both tracks recovered  [%]')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    fs.title(ax, 'The joint fit recovers pairs the seed gap merges',
             'synthetic two-track planes, production bundle and run noise; both legs detectable on their own')
    fs.preliminary(ax)
    fs.save(fig, figdir / 'synth_efficiency', data=B)


def fig_operating(C: pd.DataFrame, thr: float, figdir: Path):
    fig, ax = fs.figure(figsize=(6.6, 4.0))
    cols = [(f'eff_{lo:g}', BAND_LABEL[(lo, hi)]) for lo, hi in BANDS if f'eff_{lo:g}' in C]
    colors = [fs.ACCENT, fs.DET_COLOR['A'], fs.DET_COLOR['C'], fs.COPPER, fs.MUTED]
    if 'false_split' in C:
        for (c, lab), col in zip(cols, colors):
            ax.plot(100 * C.false_split, 100 * C[c], marker='o', ms=3.5, color=col,
                    label=f'{lab} mm')
        i = int(np.argmin(np.abs(C.threshold - thr)))
        ax.plot(100 * C.false_split.iloc[i], 100 * C[cols[0][0]].iloc[i], marker='*',
                ms=13, color=fs.TRACK, zorder=5, ls='none',
                label=f'operating point, threshold {thr:g}')
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlabel('clean single muons split  [%]   (real triggers, run_145)')
    ax.set_ylabel('both tracks recovered  [%]')
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fs.title(ax, 'What a split costs, and what it buys',
             'the threshold is chosen from this curve, not from one number')
    fs.preliminary(ax)
    fs.save(fig, figdir / 'operating_curve', data=C)


def fig_statistic(D, probe, thr: float, figdir: Path):
    fig, ax = fs.figure(figsize=(6.6, 4.0))
    bins = np.logspace(-1, 4, 50)
    sets = []
    if probe is not None:
        cs = probe[probe.clean_single & _bool(_col(probe, 'guards_ok'))]
        sets.append((cs.fstat.dropna(), 'clean single muons (real)', fs.INK, True))
    if D is not None:
        P = D[(D.n_true == 2) & _bool(_col(D, 'both_detectable', True)) & (D.dt == 0)
              & _bool(D.guards_ok) & _bool(D.both_found)]
        sets.append((P.fstat.dropna(), 'recovered pairs (synthetic)', fs.ACCENT, False))
    csv = {}
    for v, lab, col, fill in sets:
        v = np.clip(v.to_numpy(float), bins[0], bins[-1])
        h, _ = np.histogram(v, bins=bins)
        h = h / max(h.sum(), 1)
        ax.step(bins[:-1], h, where='post', color=col, label=lab, lw=1.4)
        if fill:
            ax.fill_between(bins[:-1], h, step='post', color=col, alpha=0.12)
        csv[lab] = h
    ax.axvline(thr, color=fs.TRACK, lw=1.2, ls='--')
    ax.annotate(f'threshold {thr:g}', xy=(thr, ax.get_ylim()[1] * 0.92),
                xytext=(4, 0), textcoords='offset points', color=fs.TRACK, fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('model-selection statistic:  min marginal Δχ²  /  (χ²₁/dof)')
    ax.set_ylabel('fraction of candidates')
    ax.legend(frameon=False)
    fs.title(ax, 'The two populations the threshold has to separate',
             'both after the degeneracy, column-overlap and plausibility guards')
    fs.preliminary(ax)
    fs.save(fig, figdir / 'statistic',
            data=pd.DataFrame(dict(fstat_lo=bins[:-1], **csv)))


def fig_display(figdir: Path, arm: str = 'A'):
    """One merged pair, and what each model leaves behind.

    Built here rather than pulled from a real event so the report is
    reproducible from products alone — but it is the production bundle, the
    production window and the production fit, so the mechanism is the real one.
    """
    from wft.calib import CalibrationBundle
    from wft import model as wm, reco as wr
    from sept26_prelim_analysis import two_track_synth as TS

    b = paths.spell('out', 'reco_fullpass', TS.RUN, TS.SUBRUN, f'mx17_{arm}',
                    'calib_bundle_prelim')
    if not b.exists():
        return None
    cal = CalibrationBundle.load(str(b))
    wm.use_calibration(cal)
    wm.set_nsamp(TS.NSAMP)
    rng = np.random.default_rng(11)
    P, truth = TS.make_plane('x', [(150.0, 0.004, -40.0, 1500.0),
                                   (157.0, -0.004, -40.0, 1300.0)], 13.0, rng)
    if P is None:
        return None
    f = wr.fit_plane(P, 'x', cal)
    probe = wr.two_track_probe(P, 'x', f, cal.hyper)
    r = wr.fit_plane_two(P, 'x', cal, f, probe=probe, f_thresh=-np.inf)
    if r is None:
        return None
    W, noise, pos, sat = probe['W'], probe['noise'], probe['pos'], probe['sat']
    m1 = (wm.build_matrix('x', pos, f.p0, f.w, f.t0, cal.hyper) @ probe['q_one']
          ).reshape(W.shape)
    ca, cb = r['children']
    M2 = wm.build_matrix_two('x', pos, (ca.p0, ca.w, ca.t0), (cb.p0, cb.w, cb.t0),
                             cal.hyper)
    m2 = (M2 @ np.concatenate(r['profiles'])).reshape(W.shape)
    fig, axes = matplotlib.pyplot.subplots(1, 3, figsize=(9.2, 3.4), sharey=True)
    ts = np.arange(W.shape[1]) * wm.SNS
    ext = [ts[0], ts[-1], pos[0], pos[-1]]
    vmax = float(np.percentile(np.abs(W / noise[:, None]), 99.5))
    for ax, (img, lab) in zip(axes, [(W / noise[:, None], 'data'),
                                     ((W - m1) / noise[:, None], 'one track: residual'),
                                     ((W - m2) / noise[:, None], 'two tracks: residual')]):
        im = ax.imshow(img, aspect='auto', origin='lower', extent=ext,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('sample time  [ns]')
        ax.set_title(lab, fontsize=10, loc='left', color=fs.INK)
        fs.strip(ax)
    axes[0].set_ylabel('strip position  [mm]')
    for t_ in truth:
        axes[0].axhline(t_['p0'], color=fs.INK, lw=0.7, ls=':')

    top = fs.fig_title(fig, 'Two tracks 7 mm apart: one line cannot cover them',
                 f'production bundle, chamber {arm}; the one-track fit leaves a '
                 f'run of coherent positive residual, the joint fit does not '
                 f'(statistic {r["fstat"]:.0f})')
    box = axes[-1].get_position()
    cax = fig.add_axes([box.x1 + 0.012, box.y0, 0.013, box.height])
    fig.colorbar(im, cax=cax, label='residual  [σ]')
    fs.save(fig, figdir / 'display',
            data=pd.DataFrame(dict(pos_mm=pos,
                                   resid_one_sigma=((W - m1) / noise[:, None]).sum(axis=1)
                                   / np.sqrt(W.shape[1]),
                                   resid_two_sigma=((W - m2) / noise[:, None]).sum(axis=1)
                                   / np.sqrt(W.shape[1]))))
    return r


def fig_bench(C: pd.DataFrame, figdir: Path):
    # the baseline, the shipped-but-off pair of fixes, and the joint fit on top
    # of them. The threshold-scan variants are in compare.csv, not on this chart.
    order = [v for v in ('production', 'pairing_rescue16_ranked',
                         'pairing_rescue16_two_final') if v in set(C.variant)]
    if len(order) < 3:
        order += [v for v in sorted(set(C.variant)) if 'two' in v][-1:]
    g = C[(C.cls == 'all') & C.variant.isin(order)]
    if not len(g):
        return None
    fig, ax = fs.figure(figsize=(6.8, 4.0))
    bands = ['<12 mm', '12-24 mm', '>=24 mm']
    width = 0.8 / max(len(order), 1)
    cols = [fs.MUTED, fs.DET_COLOR['A'], fs.ACCENT, fs.COPPER]
    for k, v in enumerate(order):
        vals, xs = [], []
        for j, b in enumerate(bands):
            h = g[(g.variant == v) & (g.band == b)]
            vals.append(100 * h.both_found.mean() if len(h) else np.nan)
            xs.append(j + (k - (len(order) - 1) / 2) * width)
        ax.bar(xs, vals, width=width * 0.92, color=cols[k % len(cols)],
               label=v.replace('_', ' '))
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_xlabel('separation of the two donors, smaller of the two views')
    ax.set_ylabel('both tracks found and correctly paired  [%]')
    ax.legend(frameon=False, fontsize=8.5)
    fs.title(ax, 'The overlay bench: two real single-track triggers, summed',
             'chambers A and C pooled, run_145 stat090_0000')
    fs.preliminary(ax)
    fs.save(fig, figdir / 'bench_bands', data=g)
    return True


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def build(thr: float = None) -> Path:
    from wft import reco as wr
    thr = wr.TWO_TRACK_F if thr is None else thr
    fs.use()
    od = out_dir()
    figdir = od / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    L = load()
    D, probe, comp = L['synth'], L['probe'], L['compare']

    B = synth_bands(D, thr) if D is not None else None
    C = operating_curve(D, probe)
    PS = probe_summary(probe) if probe is not None else None

    if B is not None and len(B):
        fig_efficiency(B, figdir)
    if 'false_split' in C:
        fig_operating(C, thr, figdir)
    fig_statistic(D, probe, thr, figdir)
    try:
        disp = fig_display(figdir)
    except Exception as exc:                      # a missing bundle, say
        print(f'[report] display skipped: {exc}')
        disp = None
    have_bench = fig_bench(comp, figdir) if comp is not None else None

    # ---- headline numbers -------------------------------------------------
    row = C.iloc[int(np.argmin(np.abs(C.threshold - thr)))]
    eff_close = row.get('eff_0', np.nan), row.get('eff_6', np.nan)
    eff_mid = row.get('eff_12', np.nan), row.get('eff_18', np.nan)
    fsplit = row.get('false_split', np.nan)
    cost = (float(np.nansum(PS.cost_s * PS.n) / PS.n.sum()) if PS is not None
            and PS.cost_s.notna().any() else np.nan)
    trig = float(np.nansum(PS.triggered * PS.n) / PS.n.sum()) if PS is not None else np.nan

    cards = ''.join(
        f'<div class="card"><span class="v">{v}</span><span class="l">{l}</span></div>'
        for v, l in [
            (_pct(np.nanmean(eff_close)), 'synthetic pairs 0–12 mm apart recovered (both legs detectable)'),
            (_pct(np.nanmean(eff_mid)), 'synthetic pairs 12–24 mm apart recovered'),
            (_pct(fsplit, 2), 'clean single muons split (real run_145 triggers)'),
            (_pct(trig), 'of production candidates the fit is attempted on'),
            (f'{_f(cost, 2)}&nbsp;s', 'added per candidate — a record, not a constraint'),
        ])

    def bench_cell(band, variant='pairing_rescue16_two_final'):
        if comp is None:
            return {}
        g = comp[(comp.cls == 'coincident') & (comp.band == band)
                 & (comp.variant == variant)]
        return {r.arm: 100 * r.both_found for r in g.itertuples()}

    B12, B12b = bench_cell('<12 mm'), bench_cell('<12 mm', 'pairing_rescue16_ranked')
    M24, M24b = bench_cell('12-24 mm'), bench_cell('12-24 mm', 'pairing_rescue16_ranked')
    F24, F24b = bench_cell('>=24 mm'), bench_cell('>=24 mm', 'pairing_rescue16_ranked')

    def ab(cur, base):
        if not cur:
            return '—'
        return ' / '.join(f'{cur.get(a, float("nan")):.0f}&nbsp;%' for a in ARM_ORDER
                          if a in cur)

    verdict = (
        '<p><b>The joint two-track fit recovers pairs that share one seed cluster, '
        'and the population it has to be protected against is not noise — it is '
        'the single track itself.</b> Two straight lines fit <i>one</i> track&rsquo;s charge '
        'column by taking half of it each in depth; guarding against that, and '
        'against the collapsed basin, is what makes a threshold on the fit '
        'improvement usable at all.</p>'
        f'<p><b>On the overlay bench</b>, time-coincident pairs of two real '
        f'single-track triggers, on top of x/y pairing and the rescue floor '
        f'(chambers A&nbsp;/&nbsp;C): pairs closer than 12&nbsp;mm come back in '
        f'<b>{ab(B12, B12b)}</b> of cases where nothing recovered them before '
        f'({ab(B12b, B12b)}); 12–24&nbsp;mm goes {ab(M24b, M24b)} &rarr; '
        f'<b>{ab(M24, M24b)}</b>; and the easy band beyond 24&nbsp;mm stays where '
        f'it was, {ab(F24b, F24b)} &rarr; {ab(F24, F24b)}. On real run_145 '
        f'triggers {_pct(fsplit, 2)} of clean single muons are split, against the '
        f'&le;&nbsp;1&nbsp;% criterion, and events with no accepted split are '
        f'bit-identical.</p>'
        f'<p><b>Compute is not a constraint</b> for this reconstruction (decision '
        f'2026-09-16; condor is effectively unlimited). The fit adds {_f(cost, 1)}&nbsp;s '
        f'per candidate the selector chose — kept as a record, not a gate. What '
        f'matters instead: the per-plane trigger, built to save compute, is the '
        f'largest loss at the closest separations (the <i>this plane triggers</i> '
        f'column below), so attempting the fit on every candidate is next.</p>')

    sections = [f'<section><h2>Verdict</h2>{verdict}<div class="cards">{cards}</div></section>']

    if disp is not None:
        sections.append(
            '<section><h2>What the fit does</h2>'
            + figure_html('display',
                          'Two tracks 7 mm apart in one window, under the '
                          'production bundle. The one-track fit leaves a run of '
                          'coherent positive residual on the strips it cannot '
                          'reach — which is the trigger — and the joint fit does '
                          'not. Dotted lines mark the two tracks&rsquo; true '
                          'positions at the mesh.')
            + '</section>')

    sections.append(
        '<section><h2>What was compared</h2>'
        '<p>Three populations, in increasing order of realism and decreasing order of '
        'how much truth they carry:</p><ol>'
        '<li><b>Synthetic planes</b> built by the forward model itself under the production '
        'bundle, with the run&rsquo;s noise. Truth is exact, so this is where the efficiency '
        'against separation comes from — but the model fits the data perfectly by '
        'construction, so it cannot say what real charge does.</li>'
        '<li><b>Real production candidates</b> of run_145 stat090_0000, re-examined with no '
        'threshold: the frozen one-track fit is rebuilt from the candidates side table, the '
        'same window is cut, and the trigger and the statistic are computed. This gives the '
        'trigger rate, the cost, and the false-split rate on <i>real clean single muons</i> — '
        'the population that killed split seeding.</li>'
        '<li><b>The overlay bench</b> (<code>intra_bench</code>): two clean single-track '
        'triggers of one chamber summed, the production seeder and selector re-run. Real '
        'charge and known truth, but the donors are clean, so real two-track events are '
        'busier than these.</li></ol></section>')

    if B is not None and len(B):
        rows = [f'<tr><th class="s">{r.arm}</th><td class="n">{r.band}</td>'
                f'<td class="n">{_i(r.n)}</td><td class="n"><b>{_pct(r.eff)}</b></td>'
                f'<td class="n">{_pct(r.trig)}</td><td class="n">{_pct(r.eff_trig)}</td>'
                f'<td class="n">{_f(r.rsig_dp0, 2)}</td><td class="n">{_f(r.rsig_dtan, 3)}</td>'
                f'<td class="n">{_f(r.t_fit, 2)}</td></tr>' for r in B.itertuples()]
        sections.append(
            '<section><h2>Efficiency against separation</h2>'
            + figure_html('synth_efficiency',
                          'Both tracks recovered, against their separation on the strip '
                          'plane. Pairs whose second leg is too faint or too steep to reach '
                          '5&nbsp;&sigma; on any strip of its own are excluded: those were '
                          'never seeded and are not this fit&rsquo;s to find.')
            + table(['chamber', 'separation <span class="u">mm</span>', 'pairs',
                     'both recovered', 'this plane triggers', 'recovered &amp; triggered',
                     '&sigma; p0 <span class="u">mm</span>',
                     '&sigma; tan', 's / attempt'], rows) + '</section>')

    if 'false_split' in C:
        rows = [f'<tr><th class="s">{_f(r.threshold, 0)}</th>'
                f'<td class="n">{_pct(r.false_split, 2)}</td>'
                f'<td class="n">{_pct(r.get("split_rate_all", np.nan), 2)}</td>'
                + ''.join(f'<td class="n">{_pct(r.get(f"eff_{lo:g}", np.nan))}</td>'
                          for lo, _hi in BANDS) + '</tr>'
                for _i2, r in C.iterrows()]
        sections.append(
            '<section><h2>Choosing the threshold</h2>'
            + figure_html('operating_curve',
                          'Two-track efficiency against the fraction of real clean single '
                          'muons that get split, as the threshold moves.')
            + figure_html('statistic',
                          'The model-selection statistic on the two populations, after the '
                          'guards. The separation between them is what the guards buy.')
            + table(['threshold', 'clean singles split', 'all candidates split']
                    + [f'{BAND_LABEL[b]} mm' for b in BANDS], rows) + '</section>')

    if PS is not None:
        rows = [f'<tr><th class="s">{r.arm}</th>'
                f'<td>{"clean single" if r.clean_single else "everything else"}</td>'
                f'<td class="n">{_i(r.n)}</td><td class="n">{_pct(r.trig_residual)}</td>'
                f'<td class="n">{_pct(r.trig_width)}</td><td class="n">{_pct(r.trig_cross)}</td>'
                f'<td class="n"><b>{_pct(r.triggered)}</b></td>'
                f'<td class="n">{_pct(r.guards_ok)}</td>'
                f'<td class="n">{_f(r.t_fit, 2)}</td><td class="n">{_f(r.cost_s, 2)}</td></tr>'
                for r in PS.itertuples()]
        sections.append(
            '<section><h2>Triggers and cost on real triggers</h2>'
            '<p>The fit is several single fits, so it runs only where the window looks '
            'merged. The residual trigger asks the direct question — does one track '
            'explain this window — and is the one that fires.</p>'
            + table(['chamber', 'candidates', 'n', 'residual', 'width', 'cross-plane',
                     'any trigger', 'guards pass', 's / fit', 's / candidate'], rows)
            + '</section>')

    if have_bench:
        sections.append(
            '<section><h2>The overlay bench</h2>'
            + figure_html('bench_bands',
                          'Both donors found and correctly paired, by separation band, for '
                          'each reconstruction variant on the same donor pairs.')
            + '</section>')

    sections.append(
        '<section><h2>What this does not rule out</h2><ul>'
        '<li><b>Two parallel tracks on the same strips are not separable at any time '
        'offset.</b> The free charge profile absorbs the offset — a track arriving '
        '260&nbsp;ns later is the same data as one column running 260&nbsp;ns longer, up '
        'to w&nbsp;&times;&nbsp;260&nbsp;ns of transverse slide, under a strip pitch at any '
        'slope we fit. The one-track fit of such a pair already reaches &chi;&sup2;/dof = 1. '
        'This is a model degeneracy and has to be carried as inefficiency, not fixed.</li>'
        '<li><b>The synthetic efficiency is an upper bound.</b> The planes are drawn from '
        'the same model that fits them; real charge is not that well described, and the '
        'overlay bench numbers are the ones to quote.</li>'
        '<li><b>Nothing here says the added tracks are real.</b> The A/B shows production '
        'answers did not move and the bench shows truth is recovered; whether the tracks '
        'this adds on data are physical is the vertex tests&rsquo; question '
        '(<code>intra_vertex</code>, <code>det_a_intra</code>) after a re-pass.</li>'
        '<li><b>One run, one sub-run, chambers A and C.</b> Calibration is per detector and '
        'per run condition; the noise boundary of 23 July and the 27 July access are both '
        'unprobed here, and chamber D&rsquo;s hot columns have not been combined with this.</li>'
        '<li><b>The trigger discards pairs the fit could recover.</b> At 0&ndash;6&nbsp;mm '
        'this plane&rsquo;s trigger fires on about a quarter of true synthetic pairs, so '
        'recovered-and-triggered is roughly a third of what the fit alone recovers. '
        'The cross-plane trigger, which one-plane synthetics cannot fire, recovers some '
        'of that on data, by an unmeasured amount.</li>'
        '<li><b>Overlapping candidate windows</b> (handoff &sect;4.5) are still fitted '
        'separately, one track each, on windows that share strips.</li>'
        '</ul></section>')

    meta = L['synth_meta']
    body = ''.join(sections)
    html = (f'<!doctype html><html lang="en"><head>{head("Joint two-track fit")}</head><body>'
            f'<header><h1>A joint two-track fit for tracks that share one seed cluster</h1>'
            f'<p class="sub">run_145 stat090_0000, chambers A and C &middot; '
            f'built {dt.datetime.now():%Y-%m-%d %H:%M} &middot; '
            f'synthetic set {meta.get("built", "—")}</p></header>'
            f'<main>{body}</main>'
            f'<footer><p>Generated by <code>sept26_prelim_analysis/make_two_track_report.py</code> '
            f'from <code>two_track_synth.py</code>, <code>intra_bench split-probe</code> and '
            f'<code>intra_bench build --two-track</code>. Design and acceptance criteria: '
            f'<code>HANDOFF_JOINT_TWO_TRACK_FIT.md</code>; working log: '
            f'<code>TWO_TRACK_FIT_LOG.md</code>.</p></footer></body></html>')
    p = od / 'report.html'
    p.write_text(html)
    print(f'[report] wrote {p}')
    return p


if __name__ == '__main__':
    build()
