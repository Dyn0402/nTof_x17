#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the run_67 sliding-window (boxcar) tracking-efficiency analysis.

Driver for slide.py. Deliverables, all with the operator's boxcar
(W = 6 ms, step 1 ms, linear) over the MEASURED dt acceptance, and with drift
kept as a full 4th axis (never pooled):

  A. recovery/    eff vs time-since-flash. One figure per (mip, drift):
                  4 det panels, one curve per resist. THE core plot.
  B. thresh/      the same curves re-sliced to overlay the three plastic
                  thresholds at fixed (drift, resist) -- Det A, panels = resist.
  C. drift/       overlay the three drift settings at fixed (mip, resist)
                  -- Det A, panels = resist. (The axis kept, made visible.)
  D. maps/        eff heat-map over (dt x resist) per (mip, drift, det).
  E. slices/      eff vs resist at a few fixed dt, with the boxcar error bars.
  F. slide_curves.csv  the full tidy table behind every figure.

READ THE CORRELATION WARNING in slide.py: neighbouring points share ~5/6 of
their events, so only features wider than ~6 ms in dt are real.

Run: .venv/bin/python ntof_july_analysis/run67_scan/slide_plots.py
     [--width 6] [--step 1] [--metric p_pair|p_trk]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import scan_lib as L  # noqa: E402
import stats as ST  # noqa: E402
import slide as SL  # noqa: E402

OUT_DIR = os.path.join(L.OUT_BASE, 'slide')

METRIC_LABEL = {'p_pair': 'P(3D x/y pair) per trigger',
                'p_trk': 'P(track segment) per trigger'}
ERR_OF = {'p_pair': 'e_pair', 'p_trk': 'e_trk'}


def _outdir(*parts):
    d = os.path.join(OUT_DIR, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def resist_colors(resists):
    """Low resist (low gain) -> dark, high resist (high gain) -> bright."""
    cm = plt.get_cmap('viridis')
    rs = sorted(resists)
    return {r: cm(i / max(len(rs) - 1, 1)) for i, r in enumerate(rs)}


def _band(ax, x, y, e, **kw):
    ax.plot(x, y, **kw)
    ax.fill_between(x, y - e, y + e, alpha=0.18, lw=0,
                    color=kw.get('color'))


def _finish(ax, metric, logy=False, xlabel=True):
    # Only the bottom row of a sharex grid carries tick numbers, so only it
    # gets the axis label — an x-label under an unlabelled axis reads as a
    # broken plot.
    if xlabel:
        ax.set_xlabel('time since gamma flash [ms]')
    ax.set_ylabel(METRIC_LABEL[metric])
    if logy:
        ax.set_yscale('log')
    ax.grid(alpha=0.3)


def _boxcar_note(fig, cur):
    w = cur.width_ms.iloc[0]
    s = cur.step_ms.iloc[0]
    fig.text(0.995, 0.005,
             f'boxcar W={w:g} ms, step={s:g} ms — adjacent points share '
             f'~{1 - s / w:.0%} of their events; features narrower than '
             f'{w:g} ms are the kernel, not physics',
             ha='right', va='bottom', fontsize=7, style='italic', color='0.35')


# ---------------------------------------------------------------- A. recovery
def fig_recovery(cur, metric='p_pair'):
    """eff vs dt, one figure per (mip, drift); 4 det panels, curve per resist."""
    d = _outdir('recovery')
    err = ERR_OF[metric]
    for (mip, dr), g in cur.groupby(['mip', 'drift']):
        cols = resist_colors(g.resist.unique())
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
        for i, (ax, Ld) in enumerate(zip(axes.ravel(), 'ABCD')):
            gd = g[g.det == Ld]
            for r, gr in gd.groupby('resist'):
                gr = gr.sort_values('dt_ms')
                _band(ax, gr.dt_ms, gr[metric], gr[err],
                      color=cols[r], lw=1.6, label=f'{r:.0f} V')
            ax.set_title(f'Det {Ld}' + ('  (clean M1 — reference)' if Ld == 'A'
                                        else '  (bad M1 — noise-inflated)'),
                         fontsize=10)
            _finish(ax, metric, xlabel=(i >= 2))
        axes[0, 0].legend(title='resist HV', fontsize=8, ncol=2,
                          title_fontsize=8)
        fig.suptitle(f'run_67 post-flash tracking efficiency — '
                     f'{L.MIP_LABEL[mip]} plastic threshold, drift {dr} V',
                     fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        p = os.path.join(d, f'recovery_m{mip}_dr{dr}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


# ------------------------------------------------------------- B. thresholds
def fig_threshold(cur, metric='p_pair', det='A'):
    """Overlay the three plastic thresholds; panels = resist, one fig/drift."""
    d = _outdir('thresh')
    err = ERR_OF[metric]
    c = cur[cur.det == det]
    for dr, g in c.groupby('drift'):
        rs = sorted(g.resist.unique())
        ncol = min(4, len(rs))
        nrow = int(np.ceil(len(rs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        bottom = {i for i in range(len(rs)) if i + ncol >= len(rs)}
        for i, (ax, r) in enumerate(zip(axes.ravel(), rs)):
            for mip, gm in g[g.resist == r].groupby('mip'):
                gm = gm.sort_values('dt_ms')
                _band(ax, gm.dt_ms, gm[metric], gm[err],
                      color=L.MIP_COLOR[mip], lw=1.6, label=L.MIP_LABEL[mip])
            ax.set_title(f'resist {r:.0f} V', fontsize=10)
            _finish(ax, metric, xlabel=(i in bottom))
        for ax in axes.ravel()[len(rs):]:
            ax.axis('off')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'run_67 Det {det} — plastic-threshold comparison vs '
                     f'time since flash, drift {dr} V', fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        p = os.path.join(d, f'thresh_det{det}_dr{dr}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


# ------------------------------------------------------------------ C. drift
def fig_drift(cur, metric='p_pair', det='A'):
    """Overlay the drift settings; panels = resist, one fig per threshold."""
    d = _outdir('drift')
    err = ERR_OF[metric]
    c = cur[cur.det == det]
    dcol = {dv: col for dv, col in zip(sorted(c.drift.unique()),
                                       ['#1b7837', '#762a83', '#b35806',
                                        '#2166ac'])}
    for mip, g in c.groupby('mip'):
        rs = sorted(g.resist.unique())
        ncol = min(4, len(rs))
        nrow = int(np.ceil(len(rs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        bottom = {i for i in range(len(rs)) if i + ncol >= len(rs)}
        for i, (ax, r) in enumerate(zip(axes.ravel(), rs)):
            for dv, gd in g[g.resist == r].groupby('drift'):
                gd = gd.sort_values('dt_ms')
                _band(ax, gd.dt_ms, gd[metric], gd[err],
                      color=dcol[dv], lw=1.6, label=f'drift {dv} V')
            ax.set_title(f'resist {r:.0f} V', fontsize=10)
            _finish(ax, metric, xlabel=(i in bottom))
        for ax in axes.ravel()[len(rs):]:
            ax.axis('off')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'run_67 Det {det} — drift comparison vs time since '
                     f'flash, {L.MIP_LABEL[mip]} threshold', fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        p = os.path.join(d, f'drift_det{det}_m{mip}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


# ------------------------------------------------------------------- D. maps
def fig_maps(cur, metric='p_pair'):
    """eff heat-map over (dt x resist), one figure per (mip, drift)."""
    d = _outdir('maps')
    for (mip, dr), g in cur.groupby(['mip', 'drift']):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharey=True)
        vmax = np.nanpercentile(g[metric], 99) or 1e-3
        for ax, Ld in zip(axes, 'ABCD'):
            gd = g[g.det == Ld]
            if gd.empty:
                ax.axis('off')
                continue
            piv = gd.pivot_table(index='resist', columns='dt_ms',
                                 values=metric)
            im = ax.pcolormesh(piv.columns.to_numpy(), piv.index.to_numpy(),
                               piv.to_numpy(), cmap='magma',
                               vmin=0, vmax=vmax, shading='nearest')
            ax.set_title(f'Det {Ld}', fontsize=11)
            ax.set_xlabel('time since flash [ms]')
        axes[0].set_ylabel('resist HV [V]')
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01,
                     label=METRIC_LABEL[metric])
        fig.suptitle(f'run_67 efficiency map (boxcar-smoothed in dt) — '
                     f'{L.MIP_LABEL[mip]}, drift {dr} V', fontsize=13)
        _boxcar_note(fig, cur)
        p = os.path.join(d, f'map_m{mip}_dr{dr}.png')
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print('  wrote', p)


# ----------------------------------------------------------------- E. slices
def fig_slices(cur, metric='p_pair', dts=(6.0, 12.0, 25.0, 50.0, 70.0)):
    """eff vs resist at fixed dt (nearest boxcar centre), per (mip, det)."""
    d = _outdir('slices')
    err = ERR_OF[metric]
    have = np.sort(cur.dt_ms.unique())
    picks = sorted({float(have[np.argmin(np.abs(have - t))]) for t in dts})
    cm = plt.get_cmap('plasma')
    tcol = {t: cm(i / max(len(picks) - 1, 1)) for i, t in enumerate(picks)}
    for mip, g in cur.groupby('mip'):
        drs = sorted(g.drift.unique())
        fig, axes = plt.subplots(len(drs), 4,
                                 figsize=(17, 3.3 * len(drs)),
                                 sharex=True, squeeze=False)
        for i, dv in enumerate(drs):
            for j, Ld in enumerate('ABCD'):
                ax = axes[i, j]
                gd = g[(g.drift == dv) & (g.det == Ld)]
                for t in picks:
                    s = gd[gd.dt_ms == t].sort_values('resist_eff')
                    if s.empty:
                        continue
                    ax.errorbar(s.resist_eff, s[metric], yerr=s[err],
                                marker='o', ms=4, lw=1.3, capsize=2,
                                color=tcol[t], label=f'{t:g} ms')
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.set_title(f'Det {Ld}', fontsize=11)
                if j == 0:
                    ax.set_ylabel(f'drift {dv} V\n{METRIC_LABEL[metric]}',
                                  fontsize=9)
                if i == len(drs) - 1:
                    ax.set_xlabel('resist HV [V]')
        axes[0, 0].legend(title='dt since flash', fontsize=7,
                          title_fontsize=7, ncol=2)
        fig.suptitle(f'run_67 efficiency vs resist HV at fixed time since '
                     f'flash — {L.MIP_LABEL[mip]}', fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        p = os.path.join(d, f'slices_m{mip}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


# -------------------------------------------------------------- F. blindness
def fig_blind(cur):
    """The other half of the story: fraction of triggers the det produced NO
    hits for (post-flash blindness), same boxcar. Kept separate from the
    efficiency because it is an observable, not a cut."""
    d = _outdir('recovery')
    for (mip, dr), g in cur.groupby(['mip', 'drift']):
        cols = resist_colors(g.resist.unique())
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True,
                                 sharey=True)
        for ax, Ld in zip(axes.ravel(), 'ABCD'):
            for r, gr in g[g.det == Ld].groupby('resist'):
                gr = gr.sort_values('dt_ms')
                ax.plot(gr.dt_ms, gr.blind_frac, color=cols[r], lw=1.6,
                        label=f'{r:.0f} V')
            ax.set_title(f'Det {Ld}', fontsize=10)
            ax.set_xlabel('time since gamma flash [ms]')
            ax.set_ylabel('blind fraction (no hits produced)')
            ax.grid(alpha=0.3)
        axes[0, 0].legend(title='resist HV', fontsize=8, ncol=2,
                          title_fontsize=8)
        fig.suptitle(f'run_67 post-flash BLINDNESS — {L.MIP_LABEL[mip]}, '
                     f'drift {dr} V', fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        p = os.path.join(d, f'blind_m{mip}_dr{dr}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


# -------------------------------------------------------------- G. intensity
def fig_intensity(cur, metric='p_pair', det='A'):
    """LOW vs HIGH beam-pulse intensity, overlaid vs dt; panels = resist.

    One figure per (mip, drift). The LOW band is ~8x rarer than HIGH, so its
    curve carries much larger errors at the same boxcar width — read `n` in
    slide_curves.csv before believing a LOW-vs-HIGH difference.
    """
    import intensity as IN
    d = _outdir('intensity')
    err = ERR_OF[metric]
    c = cur[cur.det == det]
    for (mip, dr), g in c.groupby(['mip', 'drift']):
        rs = sorted(g.resist.unique())
        ncol = min(4, len(rs))
        nrow = int(np.ceil(len(rs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        bottom = {i for i in range(len(rs)) if i + ncol >= len(rs)}
        for i, (ax, r) in enumerate(zip(axes.ravel(), rs)):
            for band, gb in g[g.resist == r].groupby('iband'):
                gb = gb.sort_values('dt_ms')
                _band(ax, gb.dt_ms, gb[metric], gb[err],
                      color=IN.BAND_COLOR.get(band, 'k'), lw=1.6,
                      label=IN.BAND_LABEL.get(band, band))
            ax.set_title(f'resist {r:.0f} V', fontsize=10)
            _finish(ax, metric, xlabel=(i in bottom))
        for ax in axes.ravel()[len(rs):]:
            ax.axis('off')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'run_67 Det {det} — beam-pulse INTENSITY split vs time '
                     f'since flash, {L.MIP_LABEL[mip]}, drift {dr} V',
                     fontsize=13)
        _boxcar_note(fig, cur)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        p = os.path.join(d, f'intensity_det{det}_m{mip}_dr{dr}.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print('  wrote', p)


def fig_intensity_ratio(cur, metric='p_pair'):
    """HIGH/LOW efficiency ratio vs dt — is tracking rate beam-loading limited?

    A ratio flat at 1 means the per-trigger efficiency does not care about how
    much beam came in that pulse. A ratio below 1 at small dt means the higher
    intensity pulse leaves the chamber less efficient — space charge / recovery
    that scales with delivered flux, not just with time.
    """
    d = _outdir('intensity')
    piv_all = []
    for (mip, dr, r, det), g in cur.groupby(['mip', 'drift', 'resist', 'det']):
        p = g.pivot_table(index='dt_ms', columns='iband', values=metric)
        n = g.pivot_table(index='dt_ms', columns='iband', values='n')
        if not {'low', 'high'} <= set(p.columns):
            continue
        # p_low can be exactly 0 in a sparse LOW cell -> inf. Keep those rows
        # out of the median/IQR rather than letting them poison the aggregate.
        ratio = (p['high'] / p['low']).to_numpy()
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        q = pd.DataFrame({
            'dt_ms': p.index, 'ratio': ratio,
            'n_low': n['low'].to_numpy(), 'n_high': n['high'].to_numpy(),
            'mip': mip, 'drift': dr, 'resist': r, 'det': det})
        piv_all.append(q.dropna(subset=['ratio']))
    if not piv_all:
        print('  intensity ratio: no cell has BOTH bands — skipped')
        return None
    rat = pd.concat(piv_all, ignore_index=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    for ax, Ld in zip(axes, 'ABCD'):
        gd = rat[rat.det == Ld]
        if gd.empty:
            ax.axis('off')
            continue
        # pool over HV cells: median ratio and its spread, per dt
        s = gd.groupby('dt_ms')['ratio'].agg(['median', 'count',
                                             lambda x: x.quantile(0.25),
                                             lambda x: x.quantile(0.75)])
        s.columns = ['med', 'n', 'q25', 'q75']
        ax.plot(s.index, s['med'], color='crimson', lw=1.8, label='median')
        ax.fill_between(s.index, s['q25'], s['q75'], color='crimson',
                        alpha=0.18, lw=0, label='IQR over HV cells')
        ax.axhline(1.0, color='k', ls='--', lw=1, label='no dependence')
        ax.set_xlabel('time since gamma flash [ms]')
        ax.set_title(f'Det {Ld}', fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f'{METRIC_LABEL[metric]}\nHIGH / LOW intensity')
    axes[0].legend(fontsize=8)
    fig.suptitle('run_67 — HIGH/LOW beam-intensity efficiency ratio vs time '
                 'since flash (pooled over HV cells)', fontsize=13)
    _boxcar_note(fig, cur)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    p = os.path.join(d, 'intensity_ratio_vs_dt.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print('  wrote', p)
    return rat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=SL.WIDTH_MS)
    ap.add_argument('--step', type=float, default=SL.STEP_MS)
    ap.add_argument('--metric', default='p_pair', choices=['p_pair', 'p_trk'])
    ap.add_argument('--min-n', type=int, default=SL.MIN_N_PER_POINT)
    ap.add_argument('--int-width', type=float, default=None,
                    help='boxcar width for the intensity split (default 2x '
                         '--width: the LOW band holds only ~12 %% of events)')
    ap.add_argument('--no-intensity', action='store_true',
                    help='skip the beam-pulse intensity split')
    # run_all.py calls main([]) — it has its own flags (--force-feu) that
    # argparse would otherwise reject here. argv=None keeps normal CLI use.
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    ev, _ = ST.load()
    cur = SL.build(ev, width=args.width, step=args.step, min_n=args.min_n)

    csv = os.path.join(OUT_DIR, 'slide_curves.csv')
    cur.to_csv(csv, index=False)
    print('  wrote', csv, len(cur), 'rows')

    fig_recovery(cur, args.metric)
    fig_blind(cur)
    fig_threshold(cur, args.metric)
    fig_drift(cur, args.metric)
    fig_maps(cur, args.metric)
    fig_slices(cur, args.metric)

    # ---- beam-pulse intensity split (separate build: extra cell key) ----
    if not args.no_intensity:
        print('\n--- beam-pulse intensity split ---')
        import intensity as IN
        evi = IN.attach(ev)
        # LOW holds only ~12 % of events, so the intensity curves need a wider
        # box than the pooled ones to reach comparable errors.
        iw = args.int_width or max(args.width, 2.0 * args.width)
        curi = SL.build(evi, width=iw, step=args.step, min_n=args.min_n,
                        group_extra=('iband',))
        icsv = os.path.join(OUT_DIR, 'slide_curves_intensity.csv')
        curi.to_csv(icsv, index=False)
        print('  wrote', icsv, len(curi), 'rows')
        for det in 'ABCD':
            fig_intensity(curi, args.metric, det=det)
        rat = fig_intensity_ratio(curi, args.metric)
        if rat is not None:
            rat.to_csv(os.path.join(OUT_DIR, 'intensity_ratio.csv'),
                       index=False)
    print('done ->', OUT_DIR)


if __name__ == '__main__':
    main()
