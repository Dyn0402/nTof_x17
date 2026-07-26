#!/usr/bin/env python3
"""
A1 — run_61 singles 2-D scan: track yield vs time-since-flash and HV, per det.

The singles DAQ accepts in a rigid deadtime comb. Measured on run_61 (see
scan_lib): the flash trigger at dt=0, then 4 events at ~4.1 ms (front-end
BLIND), 2 at ~13.5 ms (partially recovered), and 2 each at ~27/41/55/69 ms
(recovered) -> ~14.9 events/spill. Probe classes:
  early = the ~4.1 ms batch   (first reconstructable events after the flash)
  mid   = the ~13.5 ms pair
  late  = >=27 ms, all 4 teeth (~8 events/spill)   <- the efficiency probe
'late' is the clean efficiency proxy; the early/mid ladder is the post-flash
recovery. Pooling all four recovered teeth gives ~4x the late statistics
run_58's narrower (20,33) ms window could reach (~1300 vs ~390 events/cell).

Efficiency metric: P(3D x/y pair) per recorded trigger is the trustworthy one
(the X/Y coincidence kills the residual common-mode fake 'tracks' that inflate
the single-plane track-segment yield on the noisy B/C/D M1 cards). Both are
reported; Det A (clean M1) is the reference detector.

Run: .venv/bin/python ntof_july_analysis/run61_scan/analyze_tracks.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/run61_scan/tracks/
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
import feu_presence as FP  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'tracks')
DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}
CLASSES = ['early', 'mid', 'late']
CLS_X = {'early': 0, 'mid': 1, 'late': 2}
CLS_LBL = ['early\n~4.1 ms\n(blind)', 'mid\n~13.5 ms', 'late\n>=27 ms\n(recovered)']
# per-spill accept slots: 0 = flash trigger (not reco'd), 1-4 = early,
# 5-6 = mid, 7+ = late. Finer recovery ladder than the 3 classes.
IDX_MAX = 15
IDX_DT_MS = {0: 0.0, 1: 4.1, 2: 4.1, 3: 4.1, 4: 4.2, 5: 13.4, 6: 13.6,
             7: 27.2, 8: 27.9, 9: 41.0, 10: 41.6, 11: 55.3, 12: 55.9,
             13: 69.1, 14: 69.6, 15: 76.6}


def binom_err(k, n):
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))


def load():
    ev, segs, _ = L.load_all()
    if ev.empty:
        sys.exit('no cached events yet — run process.py first')
    ev = L.add_probe_class(ev)
    ev = ev[ev.flash_ok].copy()
    # run_61-specific: attach per-event FEU-presence flags. Every per-detector
    # rate below MUST be evaluated on that detector's live events only —
    # see feu_presence.py for why (run_58 needs none of this; it is clean).
    ev = FP.attach(ev)
    if not segs.empty:
        segs = L.add_probe_class(segs)
        segs = segs[segs.flash_ok].copy()
    print(f'loaded {len(ev)} flash-ok events, {len(segs)} track segs from '
          f'{ev.subrun.nunique()} sub-runs '
          f'({ev.drift.nunique()} drift x {ev.resist.nunique()} resist pts)')
    print('  FEU readout fraction (denominator cut): '
          + ', '.join(f'{d}={ev[f"readout_{d}"].mean():.3f}' for d in 'ABCD'))
    print('  hit-producing ("live") fraction, an OBSERVABLE not a cut: '
          + ', '.join(f'{d}={ev[f"live_{d}"].mean():.3f}' for d in 'ABCD'))
    return ev, segs


def per_cell_stats(ev):
    """Per (drift, resist, det, probe_class): yields, liveness, gain."""
    rows = []
    for (dr, r), grp in ev.groupby(['drift', 'resist']):
        for Ld in 'ABCD':
            # Denominator = events this detector was READ OUT for (file-level).
            # NOT live_{Ld}: that flags "produced hits", which is 0 when the
            # detector is blind from the flash — the very inefficiency being
            # measured. See feu_presence.attach().
            live = grp[grp[f'readout_{Ld}']]
            for cls in CLASSES:
                p = live[live.probe_class == cls]
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
                    # busy = discharge/pile-up OR pathological (reco-skipped);
                    # both count as inefficiency, neither leaves the denominator
                    'busy_frac': (((p[f'n_clean_strips_{Ld}'] > L.BUSY_CLEAN_STRIPS)
                                   | p['reco_skipped']).mean() if n else np.nan),
                    # blind = read out but produced no hits (post-flash
                    # blindness) — an OBSERVABLE, already inside the denominator
                    'blind_frac': (1.0 - p[f'live_{Ld}'].mean()) if n else np.nan,
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
    fig.suptitle('run_61 singles — post-flash recovery of track yield '
                 '(drift-pooled, per resist)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return _save(fig, 'time_recovery.png')


# the three time groups the operator asked to see separately, in spill order
CLS_TITLE = {
    'early': ('4 ms group — the 4 triggers at ~4.1 ms after the gamma flash '
              '(PRIMARY)'),
    'mid':   '13 ms group — the 2 triggers at ~13.5 ms',
    'late':  'all later triggers integrated — 27 / 41 / 55 / 69 ms',
}


def fig_yield_vs_hv(st, cls='late'):
    """Track yield vs resist (left) and vs drift (right), per det, for ONE
    time-since-flash group.

    One figure per group (early / mid / late) rather than a single pooled plot:
    the groups sit at very different points of the post-flash recovery, so
    their HV dependence is not the same measurement. The 4 ms group is the
    operationally important one — it is the earliest the front end can be read
    at all, so its efficiency is what sets how much of the post-flash window is
    actually usable.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (xcol, xlab) in zip(axes[0], [('resist_eff', 'resist HV [V] (D = -10)'),
                                          ('drift', 'drift HV [V]')]):
        base = 'resist' if xcol == 'resist_eff' else 'drift'
        for Ld in 'ABCD':
            g = _agg_yield(st, cls, base, 'p_pair')
            g = g[g.det == Ld].sort_values(base)
            if base == 'resist':
                x = L.resist_for_det(g[base].to_numpy(), Ld)
            else:
                x = g[base].to_numpy()
            ax.errorbar(x, g.p, g.e, color=DET_COL[Ld], marker='o', ms=5,
                        capsize=2, label=f'Det {Ld}')
        ax.set_xlabel(xlab); ax.set_ylabel(f'P(3D x/y pair | {cls})')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # bottom row: single-plane track-seg yield (noise-inflated on B/C/D)
    for ax, (xcol, xlab, base) in zip(
            axes[1], [('resist_eff', 'resist HV [V] (D = -10)', 'resist'),
                      ('drift', 'drift HV [V]', 'drift')]):
        for Ld in 'ABCD':
            g = _agg_yield(st, cls, base, 'p_trk')
            g = g[g.det == Ld].sort_values(base)
            x = (L.resist_for_det(g[base].to_numpy(), Ld) if base == 'resist'
                 else g[base].to_numpy())
            ax.errorbar(x, g.p, g.e, color=DET_COL[Ld], marker='s', ms=4,
                        capsize=2, ls='--', label=f'Det {Ld}')
        ax.set_xlabel(xlab)
        ax.set_ylabel(f'P(track segment | {cls})  [noise-inflated B/C/D]')
        ax.grid(alpha=0.3)
    axes[0, 0].set_title('3D-pair yield vs resist', fontsize=10)
    axes[0, 1].set_title('3D-pair yield vs drift', fontsize=10)
    fig.suptitle(f'run_61 — {CLS_TITLE[cls]}\ntrack yield vs HV '
                 '(top: clean 3D pairs; bottom: single-plane segs)', fontsize=12)
    fig.tight_layout()
    return _save(fig, f'yield_vs_hv_{cls}.png')


def fig_slot_recovery(ev):
    """P(3D pair) vs per-spill accept SLOT, curves per resist, drift-pooled.

    The 3-class ladder collapses 15 accept slots into 3 points; run_61's comb
    actually samples the post-flash recovery at ~4, 13.5, 27, 41, 55 and 69 ms,
    so plot it at full resolution. Slot 0 (the flash trigger) is not reco'd and
    is omitted.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    resists = sorted(ev.resist.unique())
    slots = [i for i in range(1, IDX_MAX + 1)]
    cmap = plt.cm.viridis
    for ax, Ld in zip(axes.flat, 'ABCD'):
        for i, r in enumerate(resists):
            xs, ps, es = [], [], []
            for s in slots:
                g = ev[(ev.resist == r) & (ev.idx_in_burst == s)
                       & ev[f'readout_{Ld}']]
                n = len(g)
                if n < 50:
                    continue
                p, e = binom_err(int((g[f'n_pair_{Ld}'] > 0).sum()), n)
                xs.append(s); ps.append(p); es.append(e)
            ax.errorbar(xs, ps, es, color=cmap(i / max(1, len(resists) - 1)),
                        marker='o', ms=3, capsize=2, lw=1,
                        label=f'{r} V' if Ld == 'A' else None)
        ax.set_xticks(slots)
        ax.set_xticklabels([f'{s}\n{IDX_DT_MS.get(s, np.nan):.0f}' for s in slots],
                           fontsize=7)
        ax.set_title(f'Det {Ld}' + (' (clean M1)' if Ld == 'A' else ''),
                     fontsize=10, color=DET_COL[Ld])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(3D x/y pair) per trigger')
    for ax in axes[1]:
        ax.set_xlabel('accept slot in spill  /  approx. ms since flash')
    fig.legend(loc='center right', fontsize=8, title='resist HV\n(A/B/C)')
    fig.suptitle('run_61 — post-flash recovery at full comb resolution '
                 '(drift-pooled, per resist)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return _save(fig, 'slot_recovery.png')


# sibling package's A1 output (…/July_HV_Scan/run58_scan/tracks/)
RUN58_STATS = os.path.join(os.path.dirname(L.OUT_BASE), 'run58_scan', 'tracks',
                           'per_cell_stats.csv')


def fig_vs_run58(st):
    """Overlay the late-probe resist curves on run_58's, at the shared drifts.

    Cross-check, not a combined fit. Same trigger recipe and same DAQ config,
    but: (a) the resist windows overlap only on 560..540 V (run_58 580->540,
    run_61 560->515); (b) run_58's 'late' was the single ~28 ms batch while
    run_61's pools 27-69 ms, so run_61 sits slightly HIGHER wherever recovery is
    still completing past 28 ms. Shape agreement in the overlap is the thing to
    read, not the absolute offset.
    """
    if not os.path.exists(RUN58_STATS):
        print(f'  (skipping run_58 overlay — {RUN58_STATS} not found)')
        return None
    s58 = pd.read_csv(RUN58_STATS)
    shared = sorted(set(st.drift.unique()) & set(s58.drift.unique()))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    for ax, Ld in zip(axes.flat, 'ABCD'):
        for src, s, col, ls, mk in (('run_58', s58, 'gray', '--', 's'),
                                    ('run_61', st, DET_COL[Ld], '-', 'o')):
            g = s[(s.det == Ld) & (s.probe_class == 'late') & s.drift.isin(shared)]
            if g.empty:
                continue
            # pool over the shared drifts (binomial) -> one curve per run
            g = g.assign(k=np.round(g.p_pair * g.n))
            gg = (g.groupby('resist_eff').agg(k=('k', 'sum'), n=('n', 'sum'))
                  .reset_index().sort_values('resist_eff'))
            p, e = binom_err(gg.k.to_numpy(float), gg.n.to_numpy(float))
            ax.errorbar(gg.resist_eff, p, e, color=col, marker=mk, ms=4,
                        capsize=2, ls=ls, label=src if Ld == 'A' else None)
        ax.set_title(f'Det {Ld}' + (' (clean M1)' if Ld == 'A' else ''),
                     fontsize=10, color=DET_COL[Ld])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(3D x/y pair | late)')
    for ax in axes[1]:
        ax.set_xlabel('resist HV [V]  (det D = setpoint - 10)')
    fig.legend(loc='center right', fontsize=8)
    fig.suptitle(f'run_61 vs run_58 — recovered-window track yield vs resist '
                 f'(drifts {shared} pooled; overlap 560-540 V)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return _save(fig, 'vs_run58.png')


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
    fig.suptitle('run_61 — gain proxy (late-probe track charge) vs HV', fontsize=12)
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
    fig.suptitle('run_61 — front-end liveness vs probe class (drift-pooled)',
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
    figs = [fig_time_recovery(st), fig_slot_recovery(ev)]
    figs += [fig_yield_vs_hv(st, c) for c in CLASSES]   # 4 ms / 13 ms / rest
    figs += [fig_gain_vs_hv(st), fig_liveness(st), fig_vs_run58(st)]
    for f in figs:
        if f:
            print('  ->', f)

    # console summary: best resist / drift per det, per time group
    for cls in CLASSES:
        print(f'\n=== {cls}: {CLS_TITLE[cls]} ===')
        for base, lbl in (('resist', 'resist'), ('drift', 'drift')):
            g = _agg_yield(st, cls, base, 'p_pair')
            line = []
            for Ld in 'ABCD':
                gd = g[g.det == Ld]
                if not gd.p.notna().any():
                    continue
                b = gd.loc[gd.p.idxmax()]
                v = (L.resist_for_det(b[base], Ld) if base == 'resist'
                     else b[base])
                line.append(f'{Ld}: {lbl} {v:.0f} V -> {b.p:.4f}+-{b.e:.4f}')
            print('  best ' + lbl + ' | ' + ' | '.join(line))
        n_tot = st[st.probe_class == cls].n.sum()
        print(f'  ({int(n_tot)} det-events total in this group)')


if __name__ == '__main__':
    main()
