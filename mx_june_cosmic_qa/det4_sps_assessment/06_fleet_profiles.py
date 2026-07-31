#!/usr/bin/env python3
"""
06_fleet_profiles.py — is det4's stripe pattern a common construction feature?

Overlays the reference-free charge-vs-local-X profiles of the whole June fleet
(each normalised to its own median, so only the *shape* is compared). All five
chambers are read out through identical strip maps, so detector-local X is a
common frame: if the live stripes sat at the same X on every chamber it would be
a design/tooling feature that only became fatal on det4. If they do not, det4's
pattern belongs to det4's bulk.

Run 03_charge_vs_position.py for every key first; this script only reads its npz.

    ../../.venv/bin/python mx_june_cosmic_qa/det4_sps_assessment/06_fleet_profiles.py
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

KEYS = ['g_det4', 'sat_det3', 'o22_long_det2', 'g_det6_long', 'g_det7_long']
LABEL = {'g_det4': 'det4', 'sat_det3': 'det3', 'o22_long_det2': 'det2',
         'g_det6_long': 'det6', 'g_det7_long': 'det7'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=HERE)
    args = ap.parse_args()

    prof = {}
    for k in KEYS:
        p = os.path.join(args.out, f'charge_{k}.npz')
        if not os.path.exists(p):
            print(f'missing {p}')
            continue
        d = np.load(p)
        c = 0.5 * (d['ex'][:-1] + d['ex'][1:])
        q = d['qxp'] + d['qyp']
        prof[k] = (c, q / np.nanmedian(q))

    fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for k, (c, q) in prof.items():
        axs[0].semilogy(c, q, lw=1.4 if k == 'g_det4' else 1.0,
                        color='k' if k == 'g_det4' else None, label=LABEL[k])
    axs[0].set_ylabel('median charge / that chamber\'s median')
    axs[0].legend(fontsize=9, ncol=5)
    axs[0].grid(alpha=.3, which='both')
    axs[0].set_title('collected charge vs detector-local X, June fleet '
                     '(each normalised to itself)')

    ref = prof.get('g_det4')
    corr = {}
    if ref is not None:
        for k, (c, q) in prof.items():
            if k == 'g_det4':
                continue
            m = np.isfinite(q) & np.isfinite(ref[1])
            corr[LABEL[k]] = float(np.corrcoef(np.log10(np.clip(ref[1][m], 1e-3, None)),
                                               np.log10(np.clip(q[m], 1e-3, None)))[0, 1])
        axs[1].bar(list(corr), list(corr.values()), color='#0072b2')
        axs[1].axhline(0, color='k', lw=.8)
        axs[1].set_ylabel('corr(log charge profile) with det4')
        axs[1].set_ylim(-1, 1)
        axs[1].grid(alpha=.3)
        axs[1].set_xlabel('detector-local X [mm]  (top panel)')
        axs[1].set_title('shape correlation against det4 — near zero means the '
                         'stripes are det4\'s own')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fleet_profiles.png'), dpi=115)
    stats = {LABEL[k]: dict(rel_rms=float(np.nanstd(q) / np.nanmean(q)),
                            p90_over_p10=float(np.nanpercentile(q, 90)
                                               / max(np.nanpercentile(q, 10), 1e-9)))
             for k, (c, q) in prof.items()}
    out = dict(corr_with_det4=corr, profile_stats=stats)
    with open(os.path.join(args.out, 'fleet_profiles.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
