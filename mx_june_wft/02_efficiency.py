#!/usr/bin/env python3
"""
02_efficiency.py — efficiency breakdown on the waveform-first reconstruction.

Same categories and definitions as the hits chain's 09_efficiency_breakdown.py,
so the numbers are directly comparable:

    no_hit      detector fired no strip
    spark       > 50 strips fired (full-detector discharge, not a muon)
    hit_no_reco fired strips, no valid X+Y reconstruction
    reco_far    reconstructed, |r| > R
    reco_near   reconstructed, |r| <= R      <- the headline efficiency

**Detection stays hits-defined on purpose.** Whether the detector saw the muon
is a property of the analyzer's trigger, not of the fit; keeping that decision
on hits is what makes this efficiency comparable with the old chain. What the
waveform fit changes here is only *where* the reconstructed point is — which
moves reco_near/reco_far, not has_any.

    ../.venv/bin/python mx_june_wft/02_efficiency.py <run_key> [--r 5]
Outputs: <OUT_BASE>/wft/efficiency/{efficiency_breakdown.txt,.json,.png}

The JSON also carries a small reduction for downstream figures — `eff_vs_R`
(efficiency at a range of match radii, same denominator as `within_R`) and
`r_hist_edges`/`r_hist_counts` (the |r| distribution to 30 mm in 0.25 mm bins).
`mpgd26/make_efficiency_breakdown.py` builds the conference figures from those
fields alone, so no slide number can drift from this accounting.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS  # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions  # noqa: E402
from wft import compat                                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--table', default=None)
    ap.add_argument('--alignment', default=None)
    ap.add_argument('--r', type=float, default=5.0)
    ap.add_argument('--source', choices=('wft', 'hits'), default='wft',
                    help='"hits" runs the OLD chain through this exact '
                         'accounting, which is the only apples-to-apples '
                         'comparison (the old 09 script uses its own box and '
                         'event list)')
    ap.add_argument('--max-dropped', type=int, default=compat.MAX_DROPPED,
                    help='cluster-quality cut; -1 disables')
    args = ap.parse_args()

    cfg = get_config(args.run_key)
    table = args.table or os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
    out_dir = cfg.out_dir('wft', 'efficiency')
    R = args.r

    if args.source == 'hits':
        import pickle
        align_path = args.alignment or os.path.join(
            cfg.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json')
        params = cm.load_alignment(align_path)
        # 09_efficiency_breakdown.py reads the UN-vetoed cache (sparks are
        # tagged by multiplicity in the accounting, not removed upstream), so
        # match that or the hits chain is scored on fewer events than it had.
        cache_path = os.path.join(cfg.OUT_BASE, 'cache', 'event_results.pkl')
        if not os.path.exists(cache_path):
            cache_path = os.path.join(cfg.OUT_BASE, 'cache',
                                      'event_results_veto50.pkl')
        results = pickle.load(open(cache_path, 'rb'))
        # No event-list restriction: each chain reconstructs whatever it can,
        # and both are scored against the same M3 rays in the same active box.
        # (Restricting the hits chain to the wft table's events would count its
        # successes on events wft never attempted as 'hit, no reco' — unfair.)
        print(f'HITS chain through the wft accounting: {len(results):,} events')
    else:
        align_path = args.alignment or os.path.join(cfg.OUT_BASE, 'wft',
                                                    'alignment', 'alignment.json')
        params = cm.load_alignment(align_path)
        md = None if args.max_dropped < 0 else args.max_dropped
        df = compat.load_table(table, max_dropped=md)
        results = compat.as_event_results(df)

    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xa, ya, an = get_xy_angles(rays.ray_data)
    xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(results, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm)
            for r in results if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}
    print(f'{len(reco):,} events with a waveform-first X+Y point')

    # --- detection bookkeeping from hits (see docstring) ---
    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'significance'], library='pd')
    det_raw = raw[raw['feu'].isin(cfg.MX17_FEUS)]
    fired = set(int(e) for e in det_raw['eventId'].unique())
    det_lo, det_hi = int(det_raw['eventId'].min()), int(det_raw['eventId'].max())
    mult_raw = det_raw.groupby('eventId').size()
    mult = (cm.apply_significance_floor(det_raw, rel=SIG_REL_FLOOR)
            .groupby('eventId').size().reindex(mult_raw.index).fillna(0).astype(int))
    n_firing = int(len(mult))
    spark_frac = 100.0 * int((mult > SPARK_VETO_HITS).sum()) / n_firing if n_firing else np.nan
    mult_by_ev = mult.to_dict()

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr)
    py = np.array(yr)
    evn = [int(v) for v in evn]

    # Active box = 0.5-99.5 percentiles of THIS source's reconstructed
    # positions, which is what 09_efficiency_breakdown.py does. Note the
    # consequence when comparing two sources: each gets its own box, so even
    # has_any can differ slightly between them (det6: 98.0 % vs 98.6 %) purely
    # from which rays fall inside. It is a second-order effect on the headline
    # numbers, but do not read a sub-percent has_any difference as physics.
    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])

    cat = {k: 0 for k in ('no_hit', 'hit_no_reco', 'spark', 'reco_far', 'reco_near')}
    rlist = []
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1
                and ay0 <= y <= ay1):
            continue
        if mult_by_ev.get(e, 0) > SPARK_VETO_HITS:
            cat['spark'] += 1
            continue
        if e in reco:
            r = float(np.hypot(x - reco[e][0], y - reco[e][1]))
            rlist.append(r)
            cat['reco_near' if r <= R else 'reco_far'] += 1
        elif e in fired:
            cat['hit_no_reco'] += 1
        else:
            cat['no_hit'] += 1

    n = sum(cat.values())
    rlist = np.array(rlist)
    pct = {k: 100.0 * v / n for k, v in cat.items()}
    has_any = 100.0 * (n - cat['no_hit']) / n
    reco_all = 100.0 * (cat['reco_near'] + cat['reco_far']) / n
    core = rstd(rlist[rlist < 15]) if len(rlist) else np.nan
    med = float(np.median(rlist)) if len(rlist) else np.nan

    # Reduction for downstream figures (mpgd26/make_efficiency_breakdown.py):
    # the |r| tail histogram and the efficiency-vs-match-radius curve, both on
    # the SAME denominator n as within_R, so a plot built from this JSON alone
    # cannot disagree with the breakdown above. eff_vs_R[str(R)] == within_R.
    r_edges = np.arange(0.0, 30.0 + 1e-9, 0.25)
    r_counts = (np.histogram(rlist, bins=r_edges)[0] if len(rlist)
                else np.zeros(len(r_edges) - 1, int))
    radii = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30)
    eff_vs_R = {str(float(rr)): (100.0 * float((rlist <= rr).sum()) / n
                                 if len(rlist) else 0.0) for rr in radii}

    summary = dict(run_key=args.run_key, source=args.source,
                   max_dropped=(None if args.source == 'hits' or args.max_dropped < 0
                                else args.max_dropped),
                   n_rays=n, R_mm=R,
                   within_R=pct['reco_near'], reco_at_all=reco_all,
                   reco_far=pct['reco_far'], hit_no_reco=pct['hit_no_reco'],
                   no_hit=pct['no_hit'], spark_cat=pct['spark'],
                   has_any=has_any, spark_frac=spark_frac,
                   core_sigma_mm=core, median_r_mm=med,
                   n_reco=len(reco),
                   basis=('hits chain, wft accounting' if args.source == 'hits'
                          else 'waveform-first (wft)'),
                   eff_vs_R=eff_vs_R,
                   r_hist_edges=[float(v) for v in r_edges],
                   r_hist_counts=[int(v) for v in r_counts],
                   n_r_overflow=int((rlist > r_edges[-1]).sum()) if len(rlist) else 0,
                   table=table, alignment=align_path)
    # headline file (no suffix) = wft with no cluster cut, so the digest and
    # the hits comparison are both scored without an extra selection
    if args.source == 'hits':
        tag = '_hits'
    elif args.max_dropped is not None and args.max_dropped >= 0:
        tag = '_cut'
    else:
        tag = ''
    with open(os.path.join(out_dir, f'efficiency_breakdown{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    lines = [f'{k:12s} {cat[k]:7d}  {pct[k]:6.2f} %' for k in cat]
    lines += [f'{"has_any":12s} {"":7s}  {has_any:6.2f} %',
              f'{"reco_at_all":12s} {"":7s}  {reco_all:6.2f} %',
              f'{"within " + str(R) + "mm":12s} {"":7s}  {pct["reco_near"]:6.2f} %',
              f'core sigma |r| {core:.3f} mm, median |r| {med:.3f} mm',
              f'spark_frac (all firing events) {spark_frac:.2f} %']
    txt = '\n'.join(lines)
    with open(os.path.join(out_dir, f'efficiency_breakdown{tag}.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].bar(list(cat), [pct[k] for k in cat],
               color=['#999', '#e69f00', '#d55e00', '#56b4e9', '#009e73'])
    axs[0].set_ylabel('% of M3 rays')
    axs[0].set_title(f'{args.run_key} — waveform-first breakdown (n={n:,})')
    axs[0].tick_params(axis='x', rotation=20)
    if len(rlist):
        axs[1].hist(rlist, bins=np.linspace(0, 15, 120), histtype='step', lw=2)
        axs[1].axvline(R, color='gray', ls=':')
        axs[1].set_xlabel('|r| detector - reference [mm]')
        axs[1].set_title(f'core sigma {core:.2f} mm, median {med:.2f} mm')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'efficiency_breakdown{tag}.png'), dpi=110)
    print(f'\nwrote {out_dir}')


if __name__ == '__main__':
    main()
