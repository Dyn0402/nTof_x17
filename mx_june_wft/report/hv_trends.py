#!/usr/bin/env python3
"""
hv_trends.py — aggregate the campaign's off-conditions (HV-scan) staged
results into per-detector trend tables and figures for the fleet report.

The 127 `*__offcond` dirs in the campaign staging tree hold reco parquets
only (no accounting), and they are TREND-GRADE by construction, for two
independent reasons recorded in FREEZE_MPGD26_2026-08-12.md: the frozen
calibration bundle is used outside its HV conditions, and their M3 reference
is v1-only ([chi2<1], no NClus>=4 clause — a looser cut than the golden
rows'). So nothing here is absolute geometry; what survives off-conditions
are *shapes*: reconstructed fraction vs resist V, relative gain (median
fitted charge) vs resist V, spark fraction vs resist V.

Metrics are computed from each parquet alone (denominator = the row's
M3-matched event count), never from ray positions — a within-R efficiency
would need per-row alignment, which off-conditions rows don't have.

    ../../.venv/bin/python mx_june_wft/report/hv_trends.py \
        [--results /home/dylan/x17/cosmic_bench/condor_campaign/results] \
        [--out /home/dylan/x17/cosmic_bench/Analysis/fleet_report]

Writes <out>/hv_trends.json and <out>/figures/hv_<det>_<metric>.png.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS = '/home/dylan/x17/cosmic_bench/condor_campaign/results'
DEFAULT_OUT = '/home/dylan/x17/cosmic_bench/Analysis/fleet_report'

# Golden rows give each detector one on-conditions anchor point, computed
# through the same parquet-only metrics so the anchor is comparable to the
# trend points (NOT the headline within-5mm numbers, which need alignment).
GOLDEN = {'mx17_2': 'o22_long_det2', 'mx17_3': 'sat_det3', 'mx17_4': 'g_det4',
          'mx17_6': 'g_det6_long', 'mx17_7': 'g_det7_long'}

DET_LABEL = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4',
             'mx17_6': 'det6', 'mx17_7': 'det7'}


def row_metrics(parquet):
    df = pd.read_parquet(parquet, columns=[
        'event_id', 'spark', 'x_ok', 'y_ok', 'x_q_sum', 'y_q_sum',
        'x_chi2', 'x_dof', 'y_chi2', 'y_dof',
        'x_quality_ok', 'y_quality_ok'])
    n = len(df)
    if n == 0:
        return None
    both = df.x_ok & df.y_ok
    ok = df[both]
    # Gain-normalized shape chi2: the fit's chi2 is weighted by PEDESTAL
    # noise, so chi2/dof grows as amplitude^2 (measured exponent 2.0 on
    # sat_det3, INVESTIGATION_2026-08-12.md) and the absolute-threshold
    # x/y_quality_ok flag is an amplitude cut in disguise — do not plot it.
    # (chi2/dof)/(q_sum/1e3)^2 is flat vs HV within a run, so a rise flags a
    # genuine shape breakdown (saturation, wrong-v off-conditions) instead
    # of just gain.
    def cnorm(d, p):
        q = d[f'{p}_q_sum']
        c = d[f'{p}_chi2'] / d[f'{p}_dof'].clip(lower=1)
        v = (c / (q / 1e3) ** 2).replace([np.inf, -np.inf], np.nan)
        return float(v.median()) if len(v) else np.nan
    return dict(
        n_events=int(n),
        frac_reco=float(both.mean()),
        frac_spark=float(df.spark.mean()),
        frac_quality=float((df.x_quality_ok & df.y_quality_ok).mean()),
        cnorm_x=cnorm(ok, 'x') if len(ok) else np.nan,
        med_qsum_x=float(ok.x_q_sum.median()) if len(ok) else np.nan,
        med_qsum_y=float(ok.y_q_sum.median()) if len(ok) else np.nan,
    )


def collect(results_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(results_dir, '*__offcond'))):
        rowf = os.path.join(d, 'job_row.json')
        pq = os.path.join(d, 'events.parquet')
        if not (os.path.exists(rowf) and os.path.exists(pq)):
            continue
        row = json.load(open(rowf))
        m = row_metrics(pq)
        if m is None:
            continue
        rows.append(dict(
            det=row['det'], run=row['run'], subrun=row['subrun'],
            resist_V=float(row['resist_V']), drift_V=float(row['drift_V']),
            gas=row['gas'], bundle_used=row['bundle_used'],
            on_conditions=False, **m))
    return rows


def golden_anchors():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    sys.path[:0] = [repo, os.path.join(repo, 'mx_june_cosmic_qa')]
    from qa_config import get_config, setup_paths
    setup_paths()
    out = []
    for det, key in GOLDEN.items():
        cfg = get_config(key)
        pq = os.path.join(cfg.OUT_BASE, 'wft', 'events.parquet')
        if not os.path.exists(pq):
            continue
        m = row_metrics(pq)
        out.append(dict(det=det, run=cfg.RUN, subrun=cfg.SUB_RUN,
                        resist_V=np.nan, drift_V=np.nan, gas='',
                        bundle_used='(golden, on-conditions)',
                        on_conditions=True, **m))
    return out


def fill_golden_hv(rows, manifest):
    """The golden anchors' HV comes from the manifest (tier-A golden rows)."""
    if not os.path.exists(manifest):
        return
    man = pd.read_csv(manifest, dtype=str)
    for r in rows:
        if not r['on_conditions']:
            continue
        hit = man[(man.det == r['det']) & (man.run == r['run'])
                  & (man.subrun == r['subrun'])]
        if len(hit):
            try:
                r['resist_V'] = float(hit.iloc[0].resist_V)
                r['drift_V'] = float(hit.iloc[0].drift_V)
                r['gas'] = hit.iloc[0].gas
            except (TypeError, ValueError):
                pass


def figures(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    written = []
    # NOTE the parquet's `spark` column is inert (always False, even on
    # golden rows whose accounting shows 8-37 % spark) — spark categorization
    # lives in the hits-chain accounting, not the reco table. Do not plot it.
    # NOTE the x/y_quality_ok flag is an absolute chi2/dof cut on a
    # pedestal-weighted chi2 → an amplitude cut in disguise; plot the
    # gain-normalized shape chi2 instead (see row_metrics).
    metrics = [('frac_reco', 'fraction of M3-matched events reconstructed '
                '(both planes)', 'reco'),
               ('med_qsum_x', 'median fitted charge, X plane [ADC·samples]',
                'gain'),
               ('cnorm_x', 'shape χ² per (q/1000)², X plane — '
                'gain-normalized', 'shape')]
    for det, g in df.groupby('det'):
        lab = DET_LABEL.get(det, det)
        for col, ylab, tag in metrics:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            # one series per (run, drift_V): different runs are different
            # mounts/epochs and must not be silently merged
            for (run, dv), s in g.groupby(['run', 'drift_V']):
                s = s.sort_values('resist_V')
                on = s[s.on_conditions]
                off = s[~s.on_conditions]
                rlab = run.replace('mx17_', '').replace('_scan', '')
                line = ax.plot(off.resist_V, off[col], 'o-', ms=4,
                               label=f'{rlab} · drift {dv:.0f} V')
                if len(on):
                    ax.plot(on.resist_V, on[col], '*', ms=15,
                            color=line[0].get_color(), zorder=5)
            if (df.det == det).any() and (g.on_conditions).any():
                ax.plot([], [], 'k*', ms=12,
                        label='golden point (bundle conditions)')
            ax.set_xlabel('resistive-strip HV [V]')
            ax.set_ylabel(ylab)
            if tag in ('gain', 'shape'):
                ax.set_yscale('log')
            short = {'reco': 'reconstructed fraction',
                     'gain': 'relative gain (median fitted charge)',
                     'shape': 'gain-normalized shape χ²'}[tag]
            ax.set_title(f'{lab} — {short} vs resist HV (trend-grade)',
                         fontsize=11)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            name = f'hv_{lab}_{tag}.png'
            fig.savefig(os.path.join(out_dir, name), dpi=110)
            plt.close(fig)
            written.append(name)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(os.path.dirname(here), 'condor',
                            'campaign_manifest.csv')

    rows = collect(args.results)
    print(f'{len(rows)} off-conditions rows aggregated')
    anchors = golden_anchors()
    fill_golden_hv(anchors, manifest)
    rows += anchors

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'hv_trends.json'), 'w') as f:
        json.dump(dict(rows=rows,
                       caveat='TREND-GRADE: frozen bundle off-conditions + '
                              'v1-only M3 reference on scan rows'), f,
                  indent=1)
    figs = figures(rows, os.path.join(args.out, 'figures'))
    print(f'wrote hv_trends.json + {len(figs)} figures in {args.out}')


if __name__ == '__main__':
    main()
