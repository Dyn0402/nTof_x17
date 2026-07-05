#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlate_sparks.py

In the mx17_det6_det7_overnight_6-26-26 / long_run, det6 (FEU3/4) and det7 (FEU6/8)
are two detectors read out through the SAME DAQ. Do their sparks correlate event by
event? If a spark in one nearly always coincides with a spark in the other, the
discharges share a common cause (DAQ/HV glitch or inter-detector coupling); if
independent, each sparks on its own.

Per event: mult6 = FEU3+4 strips, mult7 = FEU6+8 strips; spark = mult > 50.
Outputs a contingency table, the conditional spark rates, odds ratio / phi, a figure,
and a combined per-event table (eventId, spark6, spark7) for the FEU1 cross-talk test.

Run: ../../.venv/bin/python correlate_sparks.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
import uproot

SPARK = 50
HERE = os.path.dirname(os.path.abspath(__file__))
FEU6, FEU7 = [3, 4], [6, 8]           # det6 = FEU3/4, det7 = FEU6/8


def main():
    c = get_config('g_det7_long')
    fs = sorted(f for f in os.listdir(c.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{c.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    raw['eventId'] = raw['eventId'].astype(np.int64)
    d6 = raw[raw['feu'].isin(FEU6)].groupby('eventId').size()
    d7 = raw[raw['feu'].isin(FEU7)].groupby('eventId').size()
    ev = np.array(sorted(set(d6.index) | set(d7.index)), dtype=np.int64)
    m6 = np.array([int(d6.get(e, 0)) for e in ev])
    m7 = np.array([int(d7.get(e, 0)) for e in ev])
    s6 = m6 > SPARK; s7 = m7 > SPARK
    N = len(ev)
    print(f'events (either detector fired): {N}')
    print(f'det6 spark rate {100*s6.mean():.2f}%   det7 spark rate {100*s7.mean():.2f}%')

    # contingency
    n11 = int((s6 & s7).sum()); n10 = int((s6 & ~s7).sum())
    n01 = int((~s6 & s7).sum()); n00 = int((~s6 & ~s7).sum())
    print(f'\ncontingency:  both {n11}  det6-only {n10}  det7-only {n01}  neither {n00}')

    p7_g6 = 100 * n11 / max(1, n11 + n10)          # P(det7 spark | det6 spark)
    p7_ng6 = 100 * n01 / max(1, n01 + n00)         # P(det7 spark | det6 NOT spark)
    p6_g7 = 100 * n11 / max(1, n11 + n01)
    p6_ng7 = 100 * n10 / max(1, n10 + n00)
    print(f'\nP(det7 spark | det6 spark)     = {p7_g6:.2f}%')
    print(f'P(det7 spark | det6 NOT spark) = {p7_ng6:.2f}%   -> enrichment {p7_g6/max(1e-9,p7_ng6):.1f}x')
    print(f'P(det6 spark | det7 spark)     = {p6_g7:.2f}%')
    print(f'P(det6 spark | det7 NOT spark) = {p6_ng7:.2f}%   -> enrichment {p6_g7/max(1e-9,p6_ng7):.1f}x')

    # odds ratio + phi coefficient + expected-if-independent
    OR = (n11 * n00) / max(1, n10 * n01)
    phi = (n11 * n00 - n10 * n01) / np.sqrt(
        max(1.0, (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    exp_both = (n11 + n10) * (n11 + n01) / N        # expected coincidences if independent
    print(f'\nodds ratio = {OR:.1f}   phi = {phi:.3f}')
    print(f'both-spark: observed {n11} vs expected-if-independent {exp_both:.0f}  '
          f'({n11/max(1,exp_both):.1f}x)')

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # (a) contingency heatmap
    M = np.array([[n00, n01], [n10, n11]])
    im = axes[0].imshow(M, cmap='Blues', norm=matplotlib.colors.LogNorm())
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{M[i,j]}', ha='center', va='center',
                         color='white' if M[i, j] > M.max() / 3 else 'black', fontsize=12)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['det7 no', 'det7 spark'])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['det6 no', 'det6 spark'])
    axes[0].set_title('(a) event contingency (log colour)')
    # (b) conditional rates
    axes[1].bar([0, 1], [p7_ng6, p7_g6], color=['grey', 'crimson'])
    for i, v in enumerate([p7_ng6, p7_g6]):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['det6 quiet', 'det6 SPARK'])
    axes[1].set_ylabel('det7 spark rate [%]')
    axes[1].set_title('(b) det7 spark rate vs det6 state\nenrichment %.1f×' % (p7_g6 / max(1e-9, p7_ng6)))
    # (c) observed vs independent coincidences
    axes[2].bar([0, 1], [exp_both, n11], color=['lightgrey', 'crimson'])
    for i, v in enumerate([exp_both, n11]):
        axes[2].text(i, v, f'{v:.0f}', ha='center', va='bottom')
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(['expected\n(independent)', 'observed'])
    axes[2].set_ylabel('both-spark events')
    axes[2].set_title('(c) coincident sparks\n%.1f× above chance' % (n11 / max(1, exp_both)))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_correlation.png'), dpi=140, bbox_inches='tight')
    plt.close(fig); print('\nwrote fig_correlation.png')

    # save combined per-event spark table for the FEU1 test
    np.savez(os.path.join(HERE, 'd67_events.npz'),
             eventId=ev, mult6=m6, mult7=m7, spark6=s6, spark7=s7,
             spark_either=(s6 | s7), spark_both=(s6 & s7))
    out = dict(N=N, det6_rate_pct=100 * float(s6.mean()), det7_rate_pct=100 * float(s7.mean()),
               contingency=dict(both=n11, det6_only=n10, det7_only=n01, neither=n00),
               P_det7_given_det6=p7_g6, P_det7_given_not_det6=p7_ng6,
               P_det6_given_det7=p6_g7, P_det6_given_not_det7=p6_ng7,
               odds_ratio=float(OR), phi=float(phi),
               both_observed=n11, both_expected_indep=float(exp_both),
               coincidence_factor=float(n11 / max(1, exp_both)))
    json.dump(out, open(os.path.join(HERE, 'correlation.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
