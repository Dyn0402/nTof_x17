#!/usr/bin/env python3
"""
A1 — run_58 singles 2-D scan: track yield vs time-since-flash and HV, per det.

The singles DAQ accepts in a rigid deadtime comb -> three probe batches per
30 ms gate: early (~flash, saturated), mid (~13 ms), late (~28 ms, fully
recovered). 'late' is the clean efficiency proxy; the early/mid ladder is the
post-flash recovery.

Efficiency metric: P(3D x/y pair) per recorded trigger is the trustworthy one
(the X/Y coincidence kills the residual common-mode fake 'tracks' that inflate
the single-plane track-segment yield on the noisy B/C/D M1 cards). Both are
reported; Det A (clean M1) is the reference detector.

Run: .venv/bin/python ntof_july_analysis/run58_scan/analyze_tracks.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run58_scan/tracks/
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'tracks')
DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}
CLASSES = ['early', 'mid', 'late']
CLS_X = {'early': 0, 'mid': 1, 'late': 2}
CLS_LBL = ['early\n~0 ms\n(flash)', 'mid\n~13 ms', 'late\n~28 ms\n(recovered)']


def binom_err(k, n):
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))


def load():
    ev, segs, _ = L.load_all()
    if ev.empty:
        sys.exit('no cached events yet — run process.py first')
    ev = L.add_probe_class(ev)
    ev = ev[ev.flash_ok].copy()
    if not segs.empty:
        segs = L.add_probe_class(segs)
        segs = segs[segs.flash_ok].copy()
    print(f'loaded {len(ev)} flash-ok events, {len(segs)} track segs from '
          f'{ev.subrun.nunique()} sub-runs '
          f'({ev.drift.nunique()} drift x {ev.resist.nunique()} resist pts)')
    return ev, segs


def per_cell_stats(ev):
    """Per (drift, resist, det, probe_class): yields, liveness, gain."""
    rows = []
    for (dr, r), grp in ev.groupby(['drift', 'resist']):
        for Ld in 'ABCD':
            for cls in CLASSES:
                p = grp[grp.probe_class == cls]
                n = len(p)
                k_seg = int((p[f'n_trkseg_{Ld}'] > 0).sum())
                k_pair = int((p[f'n_pair_{Ld}'] > 0).sum())
                p_seg, e_seg = binom_err(k_seg, n)
                p_pair, e_pair = binom_err(k_pair, n)
                rows.append({
                    'drift': dr, 'resist': r, 'resist_eff': L.resist_for_det(r, Ld),
                    'det': Ld, 'probe_class': cls, 'n': n,
                    'p_trk': p_seg, 'e_trk': e_seg,
                    'p_pair': p_pair, 'e_pair': e_pair,
                    'busy_frac': ((p[f'n_clean_strips_{Ld}'] > L.BUSY_CLEAN_STRIPS).mean()
                                  if n else np.nan),
                    'nhits_med': p[f'n_hits_{Ld}'].median() if n else np.nan,
                    'q_med': p[f'seg_q_{Ld}'].median() if n else np.nan,
                })
    return pd.DataFrame(rows)


def _agg_yield(st, cls, xcol, metric='p_pair'):
    """Marginalise the yield over the OTHER HV axis (pooled binomial)."""
    s = st[st.probe_class == cls].copy()
    s['k'] = np.round(s[metric] * s['n']).astype('Int64')
    g = s.groupby(['det', xcol]).agg(k=('k', 'sum'), n=('n', 'sum')).reset_index()
    p, e = binom_err(g.k.to_numpy(float), g.n.to_numpy(float))
    g['p'], g['e'] = p, e
    return g


def fig_time_recovery(st):
    """P(pair) vs probe class (recovery from flash), curves per resist,
    drift-pooled, faceted by det."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    resists = sorted(st.resist.unique())
    cmap = plt.cm.viridis
    for ax, Ld in zip(axes.flat, 'ABCD'):
        for i, r in enumerate(resists):
            xs, ps, es = [], [], []
            for cls in CLASSES:
                cell = st[(st.det == Ld) & (st.resist == r)
                          & (st.probe_class == cls)]
                k = np.round(cell.p_pair * cell.n).sum()
                n = cell.n.sum()
                if n < 50:
                    continue
                p, e = binom_err(k, n)
                xs.append(CLS_X[cls]); ps.append(p); es.append(e)
            c = cmap(i / max(1, len(resists) - 1))
            ax.errorbar(xs, ps, es, color=c, marker='o', ms=4, capsize=2,
                        label=f'{r} V' if Ld == 'A' else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLS_LBL, fontsize=8)
        ax.set_title(f'Det {Ld}' + (' (clean M1)' if Ld == 'A' else ''),
                     fontsize=10, color=DET_COL[Ld])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(3D x/y pair) per trigger')
    fig.legend(loc='center right', fontsize=8, title='resist HV\n(A/B/C)')
    fig.suptitle('run_58 singles — post-flash recovery of track yield '
                 '(drift-pooled, per resist)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return _save(fig, 'time_recovery.png')


def fig_yield_vs_hv(st):
    """LATE-probe (recovered) yield vs resist (left) and vs drift (right),
    per det — the operating-point local maximum."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (xcol, xlab) in zip(axes[0], [('resist_eff', 'resist HV [V] (D = -10)'),
                                          ('drift', 'drift HV [V]')]):
        base = 'resist' if xcol == 'resist_eff' else 'drift'
        for Ld in 'ABCD':
            g = _agg_yield(st, 'late', base, 'p_pair')
            g = g[g.det == Ld].sort_values(base)
            if base == 'resist':
                x = L.resist_for_det(g[base].to_numpy(), Ld)
            else:
                x = g[base].to_numpy()
            ax.errorbar(x, g.p, g.e, color=DET_COL[Ld], marker='o', ms=5,
                        capsize=2, label=f'Det {Ld}')
        ax.set_xlabel(xlab); ax.set_ylabel('P(3D x/y pair | late probe)')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # bottom row: single-plane track-seg yield (noise-inflated on B/C/D)
    for ax, (xcol, xlab, base) in zip(
            axes[1], [('resist_eff', 'resist HV [V] (D = -10)', 'resist'),
                      ('drift', 'drift HV [V]', 'drift')]):
        for Ld in 'ABCD':
            g = _agg_yield(st, 'late', base, 'p_trk')
            g = g[g.det == Ld].sort_values(base)
            x = (L.resist_for_det(g[base].to_numpy(), Ld) if base == 'resist'
                 else g[base].to_numpy())
            ax.errorbar(x, g.p, g.e, color=DET_COL[Ld], marker='s', ms=4,
                        capsize=2, ls='--', label=f'Det {Ld}')
        ax.set_xlabel(xlab)
        ax.set_ylabel('P(track segment | late)  [noise-inflated B/C/D]')
        ax.grid(alpha=0.3)
    axes[0, 0].set_title('3D-pair yield vs resist', fontsize=10)
    axes[0, 1].set_title('3D-pair yield vs drift', fontsize=10)
    fig.suptitle('run_58 — recovered-window (late) track yield vs HV '
                 '(top: clean 3D pairs; bottom: single-plane segs)', fontsize=12)
    fig.tight_layout()
    return _save(fig, 'yield_vs_hv.png')


def fig_gain_vs_hv(st):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (base, xlab) in zip(axes, [('resist', 'resist HV [V] (D = -10)'),
                                       ('drift', 'drift HV [V]')]):
        for Ld in 'ABCD':
            s = st[(st.det == Ld) & (st.probe_class == 'late')]
            g = s.groupby(base).apply(
                lambda d: np.average(d.q_med, weights=d.n.where(d.n > 0, np.nan))
                if d.q_med.notna().any() else np.nan, include_groups=False)
            x = (L.resist_for_det(g.index.to_numpy(), Ld) if base == 'resist'
                 else g.index.to_numpy())
            ax.plot(x, g.values, color=DET_COL[Ld], marker='o', ms=5,
                    label=f'Det {Ld}')
        ax.set_xlabel(xlab); ax.set_yscale('log'); ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('best track-seg charge q_sum [ADC] (n-wtd median)')
    axes[0].legend(fontsize=8)
    fig.suptitle('run_58 — gain proxy (late-probe track charge) vs HV', fontsize=12)
    fig.tight_layout()
    return _save(fig, 'gain_vs_hv.png')


def fig_liveness(st):
    """Median raw hits/event (front-end liveness) by probe class & resist."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    resists = sorted(st.resist.unique())
    cmap = plt.cm.viridis
    for ax, Ld in zip(axes.flat, 'ABCD'):
        for i, r in enumerate(resists):
            ys = []
            for cls in CLASSES:
                cell = st[(st.det == Ld) & (st.resist == r)
                          & (st.probe_class == cls)]
                ys.append(np.average(cell.nhits_med, weights=cell.n.where(cell.n > 0, np.nan))
                          if cell.nhits_med.notna().any() else np.nan)
            ax.plot(range(3), ys, color=cmap(i / max(1, len(resists) - 1)),
                    marker='o', ms=4, label=f'{r} V' if Ld == 'A' else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLS_LBL, fontsize=8)
        ax.set_title(f'Det {Ld}', fontsize=10, color=DET_COL[Ld])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('median raw hits/event')
    fig.legend(loc='center right', fontsize=8, title='resist HV')
    fig.suptitle('run_58 — front-end liveness vs probe class (drift-pooled)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return _save(fig, 'liveness.png')


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, segs = load()
    st = per_cell_stats(ev)
    st.to_csv(os.path.join(OUT, 'per_cell_stats.csv'), index=False)
    for f in (fig_time_recovery(st), fig_yield_vs_hv(st),
              fig_gain_vs_hv(st), fig_liveness(st)):
        print('  ->', f)
    # console summary: best resist / drift per det on the late 3D-pair yield
    print('\nbest late-probe 3D-pair operating point per det (drift-pooled):')
    for Ld in 'ABCD':
        g = _agg_yield(st, 'late', 'resist', 'p_pair')
        g = g[g.det == Ld]
        if g.p.notna().any():
            b = g.loc[g.p.idxmax()]
            print(f'  Det {Ld}: resist {L.resist_for_det(b.resist, Ld):.0f} V '
                  f'-> P(pair|late)={b.p:.3f} +- {b.e:.3f}')


if __name__ == '__main__':
    main()
