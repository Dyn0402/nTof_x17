#!/usr/bin/env python3
"""
Aggregate analysis of the run_53/run_55 doubles-trigger recursive HV scan:
track-based DAQ recovery + efficiency vs resist HV, per detector.

Probe classes (see scan_lib docstring): early ~0.02-0.1 ms, mid 8-12 ms,
late 17-23 ms after the gamma flash. All yields are per recorded trigger in
flash-confirmed bursts.

Run: .venv/bin/python ntof_july_analysis/hv_track_scan/analyze.py
Output -> <ANALYSIS_DIR>/July_HV_Scan/hv_track_scan/
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
import scan_lib  # noqa: E402

OUT = scan_lib.OUT_BASE
DET_COL = {'A': 'crimson', 'B': 'royalblue', 'C': 'seagreen', 'D': 'darkorange'}
HV_CMAP = plt.cm.viridis

# fine dt bins inside the two physical probe batches
FINE_BINS = [(8, 9), (9, 10), (10, 11), (11, 12), (17, 19), (19, 21), (21, 23)]


def binom_err(k, n):
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    return p, np.sqrt(np.maximum(p * (1 - p), 1e-12) / np.maximum(n, 1))


def load():
    ev, segs = scan_lib.load_all()
    print(f'loaded {len(ev)} events, {len(segs)} track segs from '
          f'{ev.subrun.nunique()} sub-runs')
    ev = ev[ev.flash_ok].copy()
    segs = segs[segs.flash_ok].copy()
    return ev, segs


def per_hv_stats(ev):
    """Per (resist, det): probe counts + track/pair probabilities per class."""
    rows = []
    for (r,), grp in ev.groupby(['resist']):
        for L in 'ABCD':
            row = {'resist': r, 'det': L}
            for cls in ('early', 'mid', 'late'):
                p = grp[grp.probe_class == cls]
                n = len(p)
                k_seg = (p[f'n_trkseg_{L}'] > 0).sum()
                k_pair = (p[f'n_pair_{L}'] > 0).sum()
                row[f'n_{cls}'] = n
                row[f'p_trk_{cls}'], row[f'e_trk_{cls}'] = binom_err(k_seg, n)
                row[f'p_pair_{cls}'], row[f'e_pair_{cls}'] = binom_err(k_pair, n)
                row[f'busy_frac_{cls}'] = (
                    (p[f'n_clean_strips_{L}'] > scan_lib.BUSY_CLEAN_STRIPS).mean()
                    if n else np.nan)
                row[f'nhits_med_{cls}'] = p[f'n_hits_{L}'].median() if n else np.nan
                row[f'q_med_{cls}'] = p[f'seg_q_{L}'].median() if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(['det', 'resist'])


def fig_eff_vs_hv(st):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for L in 'ABCD':
        s = st[st.det == L]
        axes[0].errorbar(s.resist, s.p_trk_late, s.e_trk_late, color=DET_COL[L],
                         marker='o', ms=5, capsize=2, label=f'Det {L}')
        axes[1].errorbar(s.resist, s.p_pair_late, s.e_pair_late, color=DET_COL[L],
                         marker='o', ms=5, capsize=2, label=f'Det {L}')
        ref = s.p_trk_late.iloc[-1]
        if ref > 0:
            axes[2].errorbar(s.resist, s.p_trk_late / ref, s.e_trk_late / ref,
                             color=DET_COL[L], marker='o', ms=5, capsize=2,
                             label=f'Det {L}')
    for ax, t in zip(axes, ['P(track segment | late probe 17-23 ms)',
                            'P(3D x/y pair | late probe)',
                            'track yield relative to 560 V']):
        ax.set_xlabel('resist HV [V]')
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[2].axhline(1.0, color='gray', lw=0.8, ls='--')
    axes[0].set_ylabel('probability per recorded trigger')
    axes[0].legend()
    fig.suptitle('run_53/55 (90/10) doubles HV scan — MM track yield vs resist HV '
                 '(fully-recovered late probes)', fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, 'eff_vs_hv.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_gain_vs_hv(segs):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    late = segs[segs.probe_class == 'late']
    for L in 'ABCD':
        s = late[late.det == L]
        med = s.groupby('resist')['q_sum'].median()
        q1 = s.groupby('resist')['q_sum'].quantile(0.25)
        q3 = s.groupby('resist')['q_sum'].quantile(0.75)
        ax.plot(med.index, med.values, color=DET_COL[L], marker='o', ms=5,
                label=f'Det {L}')
        ax.fill_between(med.index, q1.values, q3.values, color=DET_COL[L],
                        alpha=0.12, lw=0)
    ax.set_yscale('log')
    ax.set_xlabel('resist HV [V]')
    ax.set_ylabel('track-segment cluster charge q_sum [ADC] (median, IQR band)')
    ax.set_title('gain curves — late probes (17-23 ms)', fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT, 'gain_vs_hv.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_recovery_vs_hv(st):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for L in 'ABCD':
        s = st[st.det == L]
        ratio = s.p_trk_mid / s.p_trk_late
        err = ratio * np.sqrt((s.e_trk_mid / s.p_trk_mid) ** 2 +
                              (s.e_trk_late / s.p_trk_late) ** 2)
        axes[0].errorbar(s.resist, ratio, err, color=DET_COL[L], marker='o',
                         ms=5, capsize=2, label=f'Det {L}')
        axes[1].plot(s.resist, s.q_med_mid / s.q_med_late, color=DET_COL[L],
                     marker='o', ms=5, label=f'Det {L}')
    for ax, t in zip(axes, ['track yield ratio mid (8-12 ms) / late (17-23 ms)',
                            'best-segment charge ratio mid / late']):
        ax.axhline(1.0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('resist HV [V]')
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('mid / late')
    axes[0].legend()
    fig.suptitle('DAQ/detector recovery at 8-12 ms relative to 17-23 ms', fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, 'recovery_vs_hv.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_recovery_fine(ev):
    resists = sorted(ev.resist.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    for ax, L in zip(axes.flat, 'ABCD'):
        for i, r in enumerate(resists):
            grp = ev[(ev.resist == r) & ev.probe_class.isin(['mid', 'late'])]
            xs, ps, es = [], [], []
            for lo, hi in FINE_BINS:
                m = (grp.dt_ms >= lo) & (grp.dt_ms < hi)
                n = m.sum()
                if n < 30:
                    continue
                pcur, e = binom_err((grp[m][f'n_trkseg_{L}'] > 0).sum(), n)
                xs.append((lo + hi) / 2)
                ps.append(pcur)
                es.append(e)
            c = HV_CMAP(i / max(1, len(resists) - 1))
            ax.errorbar(xs, ps, es, color=c, marker='o', ms=4, capsize=2,
                        label=f'{r} V' if L == 'A' else None)
        ax.axvspan(12, 17, color='gray', alpha=0.10, lw=0)
        ax.set_title(f'Det {L}', fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('dt since gamma flash [ms]')
    for ax in axes[:, 0]:
        ax.set_ylabel('P(track segment) per trigger')
    fig.legend(loc='center right', fontsize=8, title='resist HV')
    fig.suptitle('track yield vs time since flash (fine bins inside the two '
                 'accept batches; grey = unsampled)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    p = os.path.join(OUT, 'recovery_fine.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_liveness(ev):
    """Median per-det raw hit count (noise-carpet liveness proxy) by dt class."""
    resists = sorted(ev.resist.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    classes = ['early', 'mid', 'late']
    xpos = {c: i for i, c in enumerate(classes)}
    for ax, L in zip(axes.flat, 'ABCD'):
        for i, r in enumerate(resists):
            grp = ev[ev.resist == r]
            ys = [grp[grp.probe_class == c][f'n_hits_{L}'].median()
                  for c in classes]
            c = HV_CMAP(i / max(1, len(resists) - 1))
            ax.plot([xpos[c_] for c_ in classes], ys, color=c, marker='o',
                    ms=4, label=f'{r} V' if L == 'A' else None)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(['early\n~0.05 ms', 'mid\n8-12 ms', 'late\n17-23 ms'])
        ax.set_title(f'Det {L}', fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('median raw hits/event (carpet = alive)')
    fig.legend(loc='center right', fontsize=8, title='resist HV')
    fig.suptitle('front-end liveness proxy vs probe class', fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    p = os.path.join(OUT, 'liveness.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_qa(ev):
    per = ev.groupby(['run', 'subrun', 'resist', 'cycle']).agg(
        n_ev=('eventId', 'size'), n_burst=('burst', 'nunique')).reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    for run, mk in [('run_53', 'o'), ('run_55', 's')]:
        s = per[per.run == run]
        sc = ax.scatter(s.cycle + (s.resist - 540) / 60.0, s.n_burst,
                        c=s.resist, cmap=HV_CMAP, marker=mk, s=28,
                        label=run)
    ax.set_xlabel('cycle (x-offset by resist HV)')
    ax.set_ylabel('flash-confirmed bursts / sub-run')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.colorbar(sc, ax=ax, label='resist HV [V]')
    ax.set_title('beam availability QA: bursts per 10-min sub-run', fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, 'qa_bursts.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fom_table(st):
    """Figure of merit: late-probe efficiency x usable window fraction.

    t_rec bracketing from the data:
      mid yield consistent with late (ratio > 0.8) -> recovered by <=8 ms
      mid suppressed but nonzero -> recovery inside 8-12 ms (use 10 ms)
      mid ~zero, late ok -> recovery in 12-17 ms (use 14.5 ms)
      late also suppressed vs its own max across HV -> >23 ms
    Usable window for X17 = (max(3, t_rec), 30) ms out of (3, 30).
    """
    rows = []
    for L in 'ABCD':
        s = st[st.det == L].sort_values('resist')
        p_late_max = s.p_trk_late.max()
        for _, r in s.iterrows():
            ratio = r.p_trk_mid / r.p_trk_late if r.p_trk_late > 0 else np.nan
            late_ok = r.p_trk_late > 0.8 * p_late_max
            if not late_ok:
                t_rec = 23.0
            elif ratio > 0.8:
                t_rec = 8.0
            elif ratio > 0.2:
                t_rec = 10.0
            else:
                t_rec = 14.5
            live_frac = (30.0 - max(3.0, t_rec)) / 27.0
            rows.append({'det': L, 'resist': r.resist,
                         'p_trk_late': r.p_trk_late, 'ratio_mid_late': ratio,
                         't_rec_bracket_ms': t_rec, 'live_frac_3_30': live_frac,
                         'fom': r.p_trk_late * live_frac})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'fom.csv'), index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for L in 'ABCD':
        s = df[df.det == L]
        ax.plot(s.resist, s.fom, color=DET_COL[L], marker='o', ms=5,
                label=f'Det {L}')
        best = s.loc[s.fom.idxmax()]
        ax.annotate(f'{best.resist:.0f}', (best.resist, best.fom),
                    textcoords='offset points', xytext=(0, 7),
                    color=DET_COL[L], fontsize=9, ha='center')
    ax.set_xlabel('resist HV [V]')
    ax.set_ylabel('track yield x live fraction of 3-30 ms window')
    ax.set_title('operating-point figure of merit', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT, 'fom.png')
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return df, p


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, segs = load()
    st = per_hv_stats(ev)
    st.to_csv(os.path.join(OUT, 'per_hv_stats.csv'), index=False)
    for f in (fig_eff_vs_hv(st), fig_gain_vs_hv(segs), fig_recovery_vs_hv(st),
              fig_recovery_fine(ev), fig_liveness(ev), fig_qa(ev)):
        print('  ->', f)
    df, p = fom_table(st)
    print('  ->', p)
    # console summary
    for L in 'ABCD':
        s = df[df.det == L]
        best = s.loc[s.fom.idxmax()]
        print(f'Det {L}: best FOM at {best.resist:.0f} V '
              f'(p_trk_late={best.p_trk_late:.3f}, t_rec<={best.t_rec_bracket_ms:.0f} ms)')


if __name__ == '__main__':
    main()
