#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_imon_figure.py -- the MPGD2026 figure for the imon impulse-response
measurement (HANDOFF_FLASH_CHARGE_2026-08-09.md sec 4 / sec 8).

    ../../.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py

Reads only the committed reduction that `imon_response.py` writes --
`results/imon_response_run_79.json` and `results/imon_fold_run_79_*.csv` -- and
writes `mpgd26/slides/assets/img/status_imon_response.png`, in the deck's house
style (`mpgd26/plotstyle.py`, whose categorical palette is the already-validated
one; this figure adds no new hues).

It deliberately lives here rather than in `mpgd26/make_status_plots.py` so that
the deck's figure script is not touched while other people are editing it; it can
be folded in later as one more entry in that file's FIGURES dict.

Left panel  -- the monitor's measured response to one beam pulse, and the same
               fold on the raw 1 s timestamps, so the reader can see what the
               timestamp reconstruction bought.
Right panel -- the same response on three chambers, each divided by its own
               area, plus the drift-cathode channel as the in-situ null: the
               shape belongs to the MONITOR, not to a chamber.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'mpgd26'))

import plotstyle as P  # noqa: E402

RESULTS = os.path.join(HERE, 'results')
OUT = os.path.join(REPO, 'mpgd26', 'slides', 'assets', 'img')
RUN = 'run_79'


def load_fold(run: str, det: str, tag: str):
    p = os.path.join(RESULTS, f'imon_fold_{run}_{det}_{tag}.csv')
    if not os.path.exists(p):
        return None
    lo, hi, n, m, e = [], [], [], [], []
    for r in csv.DictReader(open(p)):
        lo.append(float(r['tau_lo_s']))
        hi.append(float(r['tau_hi_s']))
        n.append(int(r['n']))
        m.append(float(r['mean_excess_ua']))
        e.append(float(r['err_ua']))
    a = [np.array(x, dtype=float) for x in (lo, hi, m, e)]
    return 0.5 * (a[0] + a[1]), a[2], a[3], np.array(n)


def make(out_dir: str = OUT, run: str = RUN) -> str:
    js = os.path.join(RESULTS, f'imon_response_{run}.json')
    if not os.path.exists(js):
        raise SystemExit(f'missing {js} -- run imon_response.py first')
    R = json.load(open(js))
    P.use()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.3),
                                   gridspec_kw=dict(width_ratios=[1.32, 1.0],
                                                    wspace=0.26))

    # ------------------------------------------------------------------ left
    C = R['dets']['C']
    tau, m, e, n = load_fold(run, 'C', 'isolated')
    tl, ml, el, _ = load_fold(run, 'C', 'isolated_labels')
    ok = np.isfinite(m) & (n > 20)

    axL.axhline(0.0, color=P.LINE, lw=1.0, zorder=1)
    axL.fill_between(tau[ok], 0, m[ok], color=P.DET_COLOR['C'], alpha=0.16,
                     lw=0, zorder=2)
    okl = np.isfinite(ml)
    axL.step(np.r_[tl[okl] - 0.2, tl[okl][-1] + 0.2], np.r_[ml[okl], ml[okl][-1]],
             where='post', color=P.MUTED, lw=2.0, ls=(0, (3.5, 2.2)),
             zorder=5, alpha=0.95)
    axL.errorbar(tau[ok], m[ok], yerr=e[ok], color=P.DET_COLOR['C'],
                 marker=P.DET_MARKER['C'], ms=6.5, lw=2.0, elinewidth=1.0,
                 capsize=0, markeredgecolor=P.SURFACE, markeredgewidth=0.8,
                 zorder=4)

    iso = C['isolated']
    lab = C['isolated_on_labels']
    pk = iso['peak_ua']
    axL.set_xlim(-0.85, 2.45)
    axL.set_ylim(-0.014, pk * 1.66)

    # what the alternative hypothesis would have looked like
    axL.annotate('', xy=(0.03, pk * 1.55), xytext=(0.03, 0.0),
                 arrowprops=dict(arrowstyle='-|>', color=P.BAND_DEAD, lw=2.0,
                                 shrinkA=0, shrinkB=0))
    axL.text(0.14, pk * 1.50,
             'a monitor that read instantaneously would put\n'
             f'the same charge into a 10 ms spike at '
             f'{C["instant_dimax_expected_ua"]:.1f} µA,\n'
             f'in 1 sample in {1 / C["instant_frac_expected"]:.0f} — '
             f'{C["instant_dimax_expected_ua"] / C["di_max_ua"]:.0f}× the biggest\n'
             f'excess this channel ever records',
             color=P.BAND_DEAD, fontsize=10.2, va='top', ha='left',
             linespacing=1.35)

    axL.annotate(f'peak {pk * 1e3:.0f} nA', xy=(iso['peak_tau_s'], pk),
                 xytext=(iso['peak_tau_s'] + 0.30, pk * 1.06),
                 color=P.DET_COLOR['C'], fontsize=11, fontweight='bold')
    axL.text(iso['centroid_s'], pk * 0.30,
             f'area = {iso["q_nc"]:.0f} nC\nper pulse',
             color=P.DET_COLOR['C'], fontsize=11.5, fontweight='bold',
             ha='center', va='center', linespacing=1.3)
    axL.text(-0.80, pk * 0.88,
             'dashed: the same fold on the raw\n1 s labels, and it can only be\n'
             'binned this coarsely. Note it is\nalready rising BEFORE the pulse.',
             color=P.MUTED, fontsize=10, ha='left', va='center', linespacing=1.35)
    axL.text(-0.80, -0.008, 'before the pulse', color=P.MUTED, fontsize=9.5,
             ha='left', va='bottom')

    axL.set_xlabel('time since the beam pulse  [s]')
    axL.set_ylabel('supply-current excess above leakage  [µA]')
    P.strip(axL)
    P.title(axL, 'The readback is a ~1 s averager, not a snapshot',
            f'det C · run_79 production setpoint · {iso["n_pulses"]} isolated '
            f'beam pulses')

    # ----------------------------------------------------------------- right
    axR.axhline(0.0, color=P.LINE, lw=1.0, zorder=1)
    for det in ('A', 'C', 'D'):
        if det not in R['dets']:
            continue
        tau, m, e, n = load_fold(run, det, 'isolated')
        q = R['dets'][det]['isolated']['q_nc'] * 1e-3      # nC -> µA*s
        ok = np.isfinite(m) & (n > 20) & (tau < 2.4)
        axR.plot(tau[ok], m[ok] / q, color=P.DET_COLOR[det],
                 marker=P.DET_MARKER[det], ms=4.6, lw=1.7,
                 markeredgecolor=P.SURFACE, markeredgewidth=0.6,
                 label=f'det {det}', zorder=3)
    tau, m, e, n = load_fold(run, 'NULL', 'isolated')
    ok = np.isfinite(m) & (n > 20) & (tau < 2.4)
    axR.plot(tau[ok], m[ok], color=P.MUTED, lw=1.8, ls=':', zorder=2,
             label='drift cathode (null)')
    axR.text(-0.78, 0.33, 'drift-cathode channel:\nflat to the last digit\n'
             '(same crate, same logger,\nno avalanche current)',
             color=P.MUTED, fontsize=9.8, ha='left', va='center', linespacing=1.3)

    axR.set_xlim(-0.85, 2.45)
    axR.set_xlabel('time since the beam pulse  [s]')
    axR.set_ylabel('response ÷ its own area  [s$^{-1}$]')
    P.strip(axR)
    P.title(axR, 'One shape, three chambers',
            'each divided by its own area — the shape belongs to the monitor')
    axR.set_ylim(-0.08, None)
    axR.legend(loc='upper left', ncol=1, handletextpad=0.6,
               bbox_to_anchor=(0.0, 1.0))

    tb, ck = R['timebase'], R['clock']
    P.note(fig,
           f'HV-monitor current readback ({R["n_samples"]} samples, '
           f'{R["n_pulses"]} logged beam pulses at '
           f'{R["pulse_rate_hz"]:.3f} Hz) phase-folded against the beam-intensity '
           f'log. The logger writes whole-second timestamps but runs at '
           f'{R["cadence"]["loop_period_s"]:.4f} s, so the sub-second phase drifts '
           f'and a raw-label fold is smeared by a 1 s box (289 ms rms) — the '
           f'dashed curve, which is broader and, impossibly, already rising '
           f'before the pulse. The true sample times are recovered from that drift '
           f'({tb["frac_good"] * 100:.0f}% of samples, {tb["median_unc_ms"]:.0f} ms '
           f'median), and the residual host-clock offset to the intensity log is '
           f'bounded at ~{abs(ck["best_lag_s"])} s by a ±1 h lag scan — an offset '
           f'shifts this curve, it cannot widen it. Randomising the pulse times '
           f'flattens the fold (chi2/ndf {R["null_random_times_chi2_flat"]:.2f} vs '
           f'{R["real_chi2_flat"]:.0f}). Verdict: the readback averages over '
           f'>= {C["w_min_s"]:.2f} s, so mean - median IS the time-average '
           f'current and the charge numbers are measurements, not lower bounds.',
           y=-0.05)

    path = os.path.join(out_dir, 'status_imon_response.png')
    P.save(fig, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--run', default=RUN)
    a = ap.parse_args()
    make(a.out, a.run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
