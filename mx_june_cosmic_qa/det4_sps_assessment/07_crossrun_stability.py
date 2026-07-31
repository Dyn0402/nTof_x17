#!/usr/bin/env python3
"""
07_crossrun_stability.py — is det4's stripe pattern the same on two different days?

A pattern that moves between runs would point at something transient (a gas
bubble, a charging-up transient, a loose contact). A pattern that sits still is
structural. Compares the X-plane per-strip occupancy of the 6-24 dedicated det4
day run against the 6-23 overnight scan point at the same resist voltage (495 V).

Caveat that this script handles: FEU 6 connectors 7 and 8 (local X > 300 mm)
recorded nothing in the 6-23 run — they are the *highest*-occupancy connectors
on 6-24, so that is a readout state of the 6-23 run and those channels must be
excluded from the comparison rather than counted as chamber.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/07_crossrun_stability.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

from qa_config import setup_paths                          # noqa: E402
setup_paths()
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
import uproot                                              # noqa: E402
from scipy.stats import spearmanr                          # noqa: E402
import cosmic_micro_tpc_analysis as cm                     # noqa: E402
from wft.seed import SIG_REL_FLOOR, SPARK_VETO_HITS        # noqa: E402
from common.Mx17StripMap import Mx17StripMap               # noqa: E402

RUNS = {
    '6-24 day run (resist 495 V)':
        '/home/dylan/x17/cosmic_bench/det4_day/mx17_det4_day_6-24-26/long_run',
    '6-23 scan point (resist 495 V)':
        '/home/dylan/x17/cosmic_bench/det3_det4/mx17_det3_det4_overnight_6-23-26/'
        'resist_495V_drift_600V',
}
FEU_X = 6
FEUS = (6, 8)


def occupancy(path):
    fs = sorted(glob.glob(os.path.join(path, 'combined_hits_root', '*.root')))
    raw = uproot.concatenate([f'{f}:hits' for f in fs],
                             expressions=['eventId', 'feu', 'channel',
                                          'significance'], library='pd')
    n_ev = int(raw['eventId'].nunique())
    # the discharge veto counts both planes, as everywhere else in this package
    det = cm.apply_significance_floor(raw[raw['feu'].isin(FEUS)], rel=SIG_REL_FLOOR)
    mult = det.groupby('eventId').size()
    det = det[~det['eventId'].isin(set(mult[mult > SPARK_VETO_HITS].index))]
    det = det[det['feu'] == FEU_X]
    occ = np.zeros(512)
    for ch, sub in det.groupby('channel'):
        if 0 <= int(ch) < 512:
            occ[int(ch)] = len(sub) / max(n_ev, 1)
    return occ, n_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()
    sm = Mx17StripMap(os.path.join(REPO, 'mx17_m1_map.csv'))
    pos = np.array([sm.lookup('x', *Mx17StripMap.feu_channel_to_connector(c))[0]
                    for c in range(512)])

    occ, nev = {}, {}
    for label, path in RUNS.items():
        occ[label], nev[label] = occupancy(path)
        print(f'{label}: {nev[label]:,} events')
    labels = list(RUNS)
    a, b = occ[labels[0]], occ[labels[1]]
    conn_tot = np.array([b[k * 64:(k + 1) * 64].sum() for k in range(8)])
    common = np.repeat(conn_tot > 0.01 * conn_tot.max(), 64)
    print(f'connectors read out in both runs: '
          f'{[k + 1 for k in range(8) if conn_tot[k] > 0.01 * conn_tot.max()]}')

    m = common & (a > 0) & (b > 0)
    rep = dict(
        n_strips_common=int(common.sum()), n_strips_firing_both=int(m.sum()),
        pearson_log_occ=float(np.corrcoef(np.log10(a[m]), np.log10(b[m]))[0, 1]),
        spearman_common=float(spearmanr(a[common], b[common]).statistic),
        live_frac=dict((lab, float((occ[lab][common]
                                    > 0.25 * np.percentile(occ[lab][common], 90)).mean()))
                       for lab in labels),
        n_events=nev)
    with open(os.path.join(args.out, 'crossrun_stability.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))

    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, lab, col in zip(axs, labels, ('k', '#0072b2')):
        o = np.where(common, occ[lab], np.nan)
        ax.semilogy(pos, np.clip(o, 1e-5, None), lw=.9, color=col)
        ax.set_ylabel('X-plane hits/event/strip')
        ax.set_title(f'{lab} — {nev[lab]:,} events')
        ax.grid(alpha=.3, which='both')
    axs[1].set_xlabel('detector-local X [mm]')
    fig.suptitle(f'det4 stripe pattern on two different days '
                 f'(Spearman {rep["spearman_common"]:.2f}); '
                 f'X > 300 mm not read out on 6-23')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'crossrun_occ.png'), dpi=115)


if __name__ == '__main__':
    main()
