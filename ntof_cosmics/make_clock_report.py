#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build results/clock_match/report.html from clock_match.py's outputs.

Generated, not hand-written: re-run clock_match.py, then this, and the
numbers, figures and verdict move together.  Figures ship their CSVs
(figstyle.save) and are linked relatively.

    .venv/bin/python ntof_cosmics/make_clock_report.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'sept26_prelim_analysis'))
import figstyle as fs  # noqa: E402
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

RES = HERE / 'results' / 'clock_match'
FIG = RES / 'figures'


def figures(stem: str, S: dict) -> list[tuple[str, str]]:
    fs.use()
    out = []

    # 1 -- coarse scan
    h = S['coarse']['hist_1ms']
    full = np.array(h['full_100ms'])
    x = (h['full_lo_ns'] + 1e8 * (np.arange(full.size) + 0.5)) / 1e9
    z = np.array(h['zoom'])
    xz = (h['zoom_lo_ns'] + 1e6 * (np.arange(z.size) + 0.5)) / 1e9
    fig, (a1, a2) = fs.plt.subplots(1, 2, figsize=fs.WIDE,
                                    gridspec_kw=dict(width_ratios=[2, 1]))
    a1.step(x, full, where='mid', color=fs.INK, lw=0.8)
    a1.set_xlabel('n_TOF − DREAM − log anchor [s]')
    a1.set_ylabel('pairs per 100 ms')
    a2.step(xz * 1e3 - S['coarse']['S_ns'] / 1e6, z, where='mid', color=fs.ACCENT)
    a2.set_xlabel('offset from peak [ms]')
    a2.set_ylabel('pairs per 1 ms')
    fs.fig_title(fig, f"One translation stands out of ±60 s: S = "
                      f"{S['coarse']['S_ns'] / 1e9:+.4f} s",
                 f"first {S['coarse']['n_dream_slice']} DREAM triggers (60 s) "
                 f"against every n_TOF singles candidate")
    fs.save(fig, FIG / 'coarse', data=pd.DataFrame(dict(x_s=x, pairs=full)))
    out.append(('coarse.png', 'Stage 1. Every n_TOF wall×plastic singles '
                'candidate minus every DREAM trigger in the first 60 s, after '
                'anchoring DREAM on the DAQ-log start line. One 1 ms bin holds '
                f"{S['coarse']['steps'][0]['peak']} pairs against a median of "
                f"{S['coarse']['steps'][0]['median_bin']:.0f}; the flat floor is "
                'the 0.5 s n_TOF comb against uncorrelated triggers.'))

    # 2 -- drift
    d = pd.read_csv(RES / f'drift_pairs_{stem}.csv')
    k = S['drift']['k']
    dS = S['drift']['S_ns'] - S['coarse']['S_ns']
    fig, ax = fs.figure(fs.FIG)
    ax.plot(d.t_dream_s, d.r_us / 1e3, '.', ms=1.5, color=fs.MUTED, alpha=0.5)
    tt = np.linspace(0, d.t_dream_s.max(), 50)
    ax.plot(tt, (dS + k * tt * 1e9) / 1e6, color=fs.ACCENT, lw=1.2,
            label=f'S + k·t, k = {k * 1e6:+.2f} ppm')
    ax.set_ylim(-5, 5)
    ax.set_xlabel('DREAM time since go [s]')
    ax.set_ylabel('n_TOF − DREAM − S [ms]')
    ax.legend(frameon=False, loc='upper right')
    fs.title(ax, 'The DREAM clock runs 3.8 ppm fast against psTime',
             'every pair within ±5 ms of the coarse translation, whole sub-run')
    fs.save(fig, FIG / 'drift', data=d)
    out.append(('drift.png', 'Stage 2. With the translation fixed, a ±5 ms '
                'window around every trigger follows the drift: a straight line, '
                f"{S['drift']['k'] * 1e6:+.2f} ppm, about 3.5 ms over 15 min. "
                'Tightening the window to ±200 µs stops improving the core at '
                f"σ ≈ {S['drift']['steps'][-1]['core_sigma_ns'] / 1e3:.0f} µs: "
                'that floor is the per-bunch jitter of figure 3, not the rate.'))

    # 3 -- delta_b
    db = pd.read_csv(RES / f'delta_b_{stem}.csv')
    fig, ax = fs.figure(fs.FIG)
    ax.plot(db.bunch_t_s - db.bunch_t_s.min(), db.delta_b_ns / 1e3, '.', ms=2,
            color=fs.DET_COLOR['A'])
    ax.set_xlabel('bunch time [s]')
    ax.set_ylabel('δ_b: window start − psTime [µs]')
    fs.title(ax, 'A slow drift the straight line misses, plus ~30 µs of jitter',
             f"slow part spans {S['delta_b_slow_range_ns'] / 1e3:.0f} µs; fast "
             f"scatter about a {S['smooth_s']:.0f} s running median: MAD "
             f"{S['delta_b_fast_mad_ns'] / 1e3:.0f} µs")
    fs.save(fig, FIG / 'delta_b', data=db)
    out.append(('delta_b.png', 'Stage 4. The offset of each n_TOF window from its '
                'own psTime, from the bunch\'s matched triggers. Two parts: a '
                'slow wander of a few hundred µs — curvature of the oscillator '
                'drift that the straight stage-2 line does not follow — and a '
                f"fast scatter of {S['delta_b_fast_mad_ns'] / 1e3:.0f} µs MAD "
                'that is uncorrelated bunch to bunch (the step between '
                f"neighbours, {S['delta_b_step_mad_ns'] / 1e3:.0f} µs, is √2 "
                'times it). The slow part can be followed; the fast part cannot, '
                'so a trigger alone in its bunch is placed only to ~30 µs.'))

    # 4 -- LOO residual
    p = pd.read_csv(RES / f'pairs_{stem}.csv')
    r = p.res.dropna()
    W = S['window_ns']
    fig, ax = fs.figure(fs.FIG)
    bins = np.arange(-200, 202, 5)
    hh, _ = np.histogram(r, bins=bins)
    ax.step(0.5 * (bins[1:] + bins[:-1]), hh, where='mid', color=fs.INK)
    ax.axvspan(-W, W, color=fs.ACCENT, alpha=0.08, lw=0)
    ax.set_xlabel('leave-one-out residual [ns]')
    ax.set_ylabel('triggers per 5 ns')
    fs.title(ax, f"With δ_b and κ the match is {S['res_core_mad_ns']:.0f} ns MAD",
             f"δ_b from the bunch's OTHER triggers; ±{W:g} ns shaded")
    fs.save(fig, FIG / 'residual', data=pd.DataFrame(
        dict(lo_ns=bins[:-1], n=hh)))
    out.append(('residual.png', 'Stage 4. Each trigger is predicted from the '
                'other triggers of its bunch only, with the in-bunch rate κ '
                f"({S['kappa']['kappa'] * 1e6:.1f} ppm) from stage 3 — nothing "
                'validates itself. The core is about as narrow as the 6 ns of '
                'the beam-on calibration.'))
    return out


def main() -> None:
    summaries = sorted(RES.glob('summary_*.json'))
    if not summaries:
        raise SystemExit('run clock_match.py first')
    sp = summaries[0]
    stem = sp.stem.removeprefix('summary_')
    S = json.loads(sp.read_text())
    figs = figures(stem, S)
    k3 = S['kappa']
    rows = [
        ('1 coarse', f"S = {S['coarse']['S_ns'] / 1e9:+.6f} s",
         'peak / median of the 1 ms scan',
         f"{S['coarse']['steps'][0]['peak']} / {S['coarse']['steps'][0]['median_bin']:.0f}"),
        ('2 drift', f"S = {S['drift']['S_ns'] / 1e9:+.6f} s, k = "
         f"{S['drift']['k'] * 1e6:+.3f} ppm", 'core σ (jitter floor)',
         f"{S['drift']['steps'][-1]['core_sigma_ns'] / 1e3:.0f} µs"),
        ('3 κ', f"{k3['kappa'] * 1e6:.1f} ppm (beam-on run_79: 110.4)",
         'σ of the two-trigger difference', f"{k3['sigma_pair_ns']:.1f} ns "
         f"({k3['n_bunches']} bunches)"),
        ('4 δ_b', f"slow {S['delta_b_slow_range_ns'] / 1e3:.0f} µs span + fast "
         f"{S['delta_b_fast_mad_ns'] / 1e3:.0f} µs MAD",
         'leave-one-out core MAD', f"{S['res_core_mad_ns']:.1f} ns"),
    ]
    tab = ''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>'
                  for a, b, c, d in rows)
    fig_html = ''.join(
        f'<figure><img src="figures/{f}" alt=""><figcaption>{html.escape(c)}'
        f'</figcaption></figure>' for f, c in figs)

    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Beam-off clock match</title>{HEAD}</head><body><main>
<p class="eyebrow">n_TOF EAR2 · X17 · ntof_cosmics/clock_match</p>
<h1>Beam-off DREAM triggers on the n_TOF clock</h1>

<div class="verdict"><p><b>It works at the nanosecond level.</b> On
{S['run']}/{S['subrun']} against n_TOF {S['ntof']}, with no beam and no flash,
<b>{S['matched_loo']} DREAM cosmic triggers match an n_TOF wall×plastic
singles within ±{S['window_ns']:g} ns — {S['efficiency_loo'] * 100:.1f} % of the
{S['efficiency_denominator']} triggers that fall inside an n_TOF window</b>
(beam-on: 96 %), with a {S['res_core_mad_ns']:.0f} ns core and an accidental
rate too small to measure ({S['sideband']} trigger in the 1–300 µs sideband).
The beam-on map carries over with the flash replaced by psTime, plus one
thing beam-on never had to deal with: each free-running n_TOF window starts
off its psTime by a slow wander ({S['delta_b_slow_range_ns'] / 1e3:.0f} µs over the
sub-run) plus {S['delta_b_fast_mad_ns'] / 1e3:.0f} µs (MAD) of bunch-to-bunch
jitter, which the bunch's own triggers have to measure.</p></div>

<h2>The map</h2>
<pre>t_DREAM_on_nTOF = t_log + S + td·(1 + k)
off             = t_DREAM_on_nTOF − psTime_b = δ_b + tof·(1 − κ)</pre>
<table><thead><tr><th>stage</th><th>result</th><th>quality measure</th><th>value</th></tr></thead>
<tbody>{tab}</tbody></table>

<h2>What was matched</h2>
<ul>
<li>DREAM: {S['n_dream']} triggers (scintillator singles, ~25 Hz), every
trigger from FEU 1's decoded tree — not combined_hits, which only holds events
with Micromegas activity.</li>
<li>n_TOF {S['ntof']}: free-running at 0.5009 s with an 80 ms window;
{S['n_bunches']} bunches with a psTime ({S['n_bunches_no_pstime']} without, dropped
for now); {S['n_singles']} wall×plastic singles from
<code>fast_singles</code> on raw tof, with the plastics moved
{', '.join(f'{k} {v:.0f}' for k, v in S['pss_shift_ns'].items())} ns earlier to
restore the beam-on wall–plastic timing (raw tof has no common zero across
trees once tflash is meaningless).</li>
<li>Translation prior: the DAQ-log start line. Beam-on sub-runs put DREAM's go
6–11 s after it on the NXCALS clock; S = {S['drift']['S_ns'] / 1e9:+.2f} s here
sits inside that.</li>
</ul>

<h2>Figures</h2>
{fig_html}

<h2>What this does not establish</h2>
<ul>
<li><b>One sub-run, one n_TOF run.</b> S is per sub-run (the DAQ start latency
varies by seconds), so stage 1 has to run on every one; k and κ should transfer
but are not yet shown to.</li>
<li><b>{S['singletons']} triggers are alone in their bunch</b> and are placed
only to the fast δ_b scatter (~{S['delta_b_fast_mad_ns'] / 1e3:.0f} µs MAD) once the
slow part is followed. Their arm and time are
unvalidated at the ns level, and they are left out of the efficiency.</li>
<li>The efficiency denominator only includes bunches where at least one other
trigger matched (δ_b must be known); a bunch where n_TOF missed every trigger
does not count against it.</li>
<li>{S['n_bunches_no_pstime']} bunches without a psTime are dropped. The free-
running period is regular to ~0.1 ms, so they can be recovered from their
neighbours and a wider stage-4 window.</li>
<li><b>Stage 2 is a straight line and the drift is not.</b> The slow δ_b
component is that curvature; a smooth (spline / running) drift model would
absorb it and let stage 4's ±300 µs association window shrink.</li>
<li>κ = {k3['kappa'] * 1e6:.1f} ppm differs from the beam-on 110.4 ppm by
{k3['kappa'] * 1e6 - 110.4:+.1f} ppm; per-bunch δk (≈1 ppm with beam) is not
fitted, which is why the two-trigger residual widens from
{k3['mad_by_sep']['0-10ms']:.0f} to {k3['mad_by_sep']['30-80ms']:.0f} ns MAD
with separation.</li>
</ul>
</main></body></html>
"""
    (RES / 'report.html').write_text(out)
    print(f'wrote {RES / "report.html"}')


if __name__ == '__main__':
    main()
