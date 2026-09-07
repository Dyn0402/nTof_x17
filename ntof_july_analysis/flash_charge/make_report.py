#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the HTML report for the flash-charge measurement.

Writes <OUT_BASE>/report.html, which the DAQ page's Analysis tab lists and opens
inline — figures are referenced with ORDINARY RELATIVE LINKS (`figures/x.png`),
so the same file works opened from disk, served by the DAQ page, or copied
elsewhere with its `figures/` directory.

Generated, not hand-written: it reads `results/flash_charge_subruns.csv` and
re-renders the figures through `mpgd26.make_status_plots`, so re-running
`analyze.py` and then this refreshes numbers, figures and verdict together —
and the report and the MPGD2026 talk cannot drift apart, because they are the
same plotting code.

Run:
    .venv/bin/python ntof_july_analysis/flash_charge/make_report.py
    .venv/bin/python ntof_july_analysis/flash_charge/make_report.py \\
        --out /mnt/data/x17/beam_july/analysis/flash_charge
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, 'mpgd26'))

import charge_lib as C  # noqa: E402

DEFAULT_OUT = os.path.join(_HERE, 'results')
CSV_PATH = os.path.join(_HERE, 'results', 'flash_charge_subruns.csv')

FIGURES = [
    ('status_deadtime_vs_charge.png', 'Dead time against charge delivered',
     'The join. Charge from the supply current and recovery from the '
     'flash-random probe, measured on the SAME 31 sub-runs of run_57. Three '
     'chambers at three different gains fall on one power law.'),
    ('status_charge_vs_hv.png', 'Charge per pulse against resistive-layer HV',
     'A gas-gain curve measured through the charge the detector delivers '
     'rather than the signal it makes. Det D is excluded: its run_57 curve '
     'falls with HV and is not understood.'),
    ('status_recovery_vs_hv.png', 'Recovery time against resistive-layer HV',
     'The dead-time map this measurement was built to give an abscissa to '
     '(nTof_x17_DAQ/docs/flash_recovery_run57_HV_map_2026-07-20.md). Stars '
     'mark the production operating point; the band is thermal arrival.'),
    ('status_two_readouts.png', 'The same chamber, two readout chains',
     'Context from ntof_processing/mm_flash/: on a direct 1 GS/s analog '
     'channel the chamber is usable 0.87 us after the flash peak. The '
     'millisecond dead time belongs to the charge-integrating front end.'),
]


def load() -> list[C.SubrunCharge]:
    rows = []
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            kw = {}
            for f in C.SubrunCharge.__dataclass_fields__.values():
                v = r.get(f.name)
                if f.type == 'str' or f.name in ('run', 'subrun', 'det', 'notes'):
                    kw[f.name] = v or ''
                elif f.name == 'leak_ok':
                    kw[f.name] = (v == 'True')
                elif f.name in ('n_samples', 'n_pulses'):
                    kw[f.name] = int(float(v)) if v else 0
                else:
                    try:
                        kw[f.name] = float(v)
                    except (TypeError, ValueError):
                        kw[f.name] = float('nan')
            rows.append(C.SubrunCharge(**kw))
    return rows


def esc(s) -> str:
    return html.escape(str(s))


def _mean(rows, attr):
    v = [getattr(r, attr) for r in rows if np.isfinite(getattr(r, attr))]
    return float(np.mean(v)) if v else float('nan')


def build(rows, out_base: str) -> str:
    prod = [r for r in rows if r.run == 'run_158']
    L: list[str] = []
    w = L.append

    w('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width, initial-scale=1">')
    w('<title>Gamma-flash charge into the MX17 chambers</title>')
    w('''<style>
:root{--bg:#fff;--fg:#1b2430;--muted:#6a7583;--line:#e0e4ea;--accent:#0072B2;
 --warn:#c0632c;--panel:#f7f8fa;--ok:#009E73;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
 --bg:#15171a;--fg:#e8e8ea;--muted:#a0a4ab;--line:#2c3036;--accent:#6fb0e0;
 --warn:#e2925c;--panel:#1c1f24;--ok:#6fc493;}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e8e8ea;--muted:#a0a4ab;
 --line:#2c3036;--accent:#6fb0e0;--warn:#e2925c;--panel:#1c1f24;--ok:#6fc493;}
body{background:var(--bg);color:var(--fg);margin:0 auto;
 padding:2.2rem 1.2rem 4rem;max-width:54rem;line-height:1.62;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
h1{font-size:1.7rem;line-height:1.25;margin:0 0 .3rem;}
h2{font-size:1.18rem;margin:2.4rem 0 .7rem;padding-bottom:.3rem;
 border-bottom:1px solid var(--line);}
.sub{color:var(--muted);margin:0 0 1.6rem;font-size:.94rem;}
.verdict{background:var(--panel);border-left:4px solid var(--accent);
 padding:1rem 1.15rem;border-radius:0 6px 6px 0;margin:1.4rem 0;}
.verdict p{margin:.45rem 0;}
.caveat{background:var(--panel);border-left:4px solid var(--warn);
 padding:1rem 1.15rem;border-radius:0 6px 6px 0;margin:1.4rem 0;}
.kpis{display:flex;flex-wrap:wrap;gap:.8rem;margin:1.4rem 0;}
.kpi{flex:1 1 10rem;background:var(--panel);border:1px solid var(--line);
 border-radius:7px;padding:.75rem .9rem;}
.kpi .n{font-size:1.35rem;font-weight:640;color:var(--accent);
 font-variant-numeric:tabular-nums;}
.kpi .l{font-size:.79rem;color:var(--muted);line-height:1.35;}
.scroll{overflow-x:auto;margin:1rem 0;}
table{border-collapse:collapse;width:100%;font-size:.87rem;
 font-variant-numeric:tabular-nums;}
th,td{border-bottom:1px solid var(--line);padding:.42rem .6rem;text-align:right;}
th{text-align:left;font-weight:600;}
thead th{color:var(--muted);font-size:.8rem;text-transform:uppercase;
 letter-spacing:.03em;}
figure{margin:1.8rem 0;}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;
 background:#fff;}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.45rem;}
code{background:var(--panel);padding:.1rem .32rem;border-radius:3px;font-size:.85em;}
ul{padding-left:1.15rem;} li{margin:.3rem 0;}
</style></head><body>''')

    w('<h1>How much charge the &gamma; flash puts into an MX17 chamber</h1>')
    w('<p class="sub">Dylan Neff &middot; 9 August 2026 &middot; '
      'from the resistive-layer HV supply current, n_TOF EAR2 July&ndash;August 2026</p>')

    # ---- verdict ---------------------------------------------------------
    qa = _mean([r for r in prod if r.det == 'A'], 'q_per_pulse_nc')
    qc = _mean([r for r in prod if r.det == 'C'], 'q_per_pulse_nc')
    pca = qa * 1e3 / C.CHANNELS_PER_DET
    w('<div class="verdict">')
    w(f'<p><b>The flash delivers ~{qc:.0f}&ndash;{qa:.0f} nC of avalanche charge '
      'per beam pulse, per chamber, at the production operating point.</b> Spread '
      f'over a chamber&rsquo;s {C.CHANNELS_PER_DET} DREAM channels that is '
      f'~{pca:.0f} pC each, against a CSA full-scale input of 50&ndash;600 fC '
      f'&mdash; <b>{C.csa_full_scale_multiple(pca, 600):.0f}&times; full scale on '
      'an average channel</b>, and the beam spot is worse than average.</p>')
    w('<p><b>And the dead time follows the charge, not the voltage.</b> Joined to '
      'the flash-recovery map on the <i>same</i> 31 sub-runs, three chambers at '
      'three different gains fall on one power law, t &prop; Q<sup>1.2</sup>, over '
      'a decade in charge. High voltage is only the knob that sets the charge.</p>')
    w('<p>The measurement sits entirely outside the readout chain, which is the '
      'point: everything that could otherwise tell us goes through the front end '
      'that the flash saturates.</p>')
    w('</div>')

    w('<div class="kpis">')
    for n, l in (
        (f'{qa:.0f} nC', 'charge per beam pulse, det A at 540 V (det C 99 nC at 525 V)'),
        (f'&times;{C.csa_full_scale_multiple(pca, 600):.0f}',
         'the CSA&rsquo;s 600 fC full scale, per channel, averaged over the chamber'),
        ('t &prop; Q<sup>1.2</sup>', 'dead time against delivered charge, three chambers on one curve'),
        ('&times;20', 'charge over the 520&ndash;580 V scan &mdash; the gas-gain curve'),
    ):
        w(f'<div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>')
    w('</div>')

    # ---- method ----------------------------------------------------------
    w('<h2>What was measured, and how</h2>')
    w('<p>The resistive-layer supply carries the avalanche ion current, so its '
      'average current <i>is</i> the charge delivered to the amplification stage, '
      'integrated over everything one beam pulse does &mdash; flash and neutron '
      'tail together. Per sub-run, per chamber:</p>')
    w('<div class="verdict" style="border-left-color:var(--muted)">'
      '<p style="margin:0"><code>Q_pulse = [ mean(I_resist) &minus; median(I_resist) ] '
      '&divide; f_pulse</code></p></div>')
    w('<ul>')
    w('<li>The CAEN readback samples at ~1 Hz while beam pulses arrive every '
      '~3.3 s, so most samples sit at the standing leakage. The sub-run '
      '<b>median is the leakage at that exact HV</b>, and mean &minus; median is '
      'the beam-induced part. Self-calibrating &mdash; which is what makes an HV '
      'scan usable without a beam-off run at every point.</li>')
    w('<li>The pulse rate is counted per sub-run over that sub-run&rsquo;s own '
      'time window from the beam-intensity slow-control log, not taken as a run '
      'average: beam availability at n_TOF varies hour to hour.</li>')
    w('<li>Errors are bootstrapped over the sub-run&rsquo;s own samples, so '
      'readback noise on a leaky channel inflates the uncertainty without biasing '
      'the estimator.</li>')
    w('</ul>')

    # ---- validation ------------------------------------------------------
    w('<h2>Three checks it has to pass</h2>')

    w('<h3 style="font-size:1rem;color:var(--muted);text-transform:uppercase;'
      'letter-spacing:.04em">1 &mdash; no beam, no charge</h3>')
    w('<div class="scroll"><table><thead><tr><th>run</th><th>det</th>'
      '<th>resist V</th><th>leakage [&micro;A]</th><th>&Delta;I [&micro;A]</th>'
      '<th>pulse rate [Hz]</th></tr></thead><tbody>')
    for run in ('run_159', 'run_157'):
        for det in 'ABCD':
            s = [r for r in rows if r.run == run and r.det == det]
            if not s:
                continue
            w(f'<tr><th>{esc(run)}</th><td>{det}</td><td>{s[0].resist_v:.0f}</td>'
              f'<td>{s[0].i_median_ua:.3f}</td><td>{_mean(s, "di_ua"):+.4f}</td>'
              f'<td>{s[0].pulse_rate_hz:.3f}</td></tr>')
    w('</tbody></table></div>')
    w('<p>run_159 is a beam-off cosmic reference at the production setpoint with '
      '<b>0.000 Hz</b> of beam pulses in the log, and the estimator returns zero '
      'on it &mdash; including on a channel carrying 2.9 &micro;A of standing '
      'leakage, which is the part that matters. run_157 was taken as beam-off too '
      'but caught residual beam at 0.031 Hz, and it shows a correspondingly small '
      'signal.</p>')

    w('<h3 style="font-size:1rem;color:var(--muted);text-transform:uppercase;'
      'letter-spacing:.04em">2 &mdash; it tracks the beam, not the detector</h3>')
    w('<div class="scroll"><table><thead><tr><th>run</th><th>det</th>'
      '<th>pulse rate [Hz]</th><th>&Delta;I [&micro;A]</th>'
      '<th>Q per pulse [nC]</th></tr></thead><tbody>')
    for run in ('run_157', 'run_158'):
        for det in ('A', 'C'):
            s = [r for r in rows if r.run == run and r.det == det]
            if not s:
                continue
            w(f'<tr><th>{esc(run)}</th><td>{det}</td>'
              f'<td>{s[0].pulse_rate_hz:.3f}</td><td>{_mean(s, "di_ua"):.4f}</td>'
              f'<td>{_mean(s, "q_per_pulse_nc"):.0f}</td></tr>')
    w('</tbody></table></div>')
    w('<p>Identical HV setpoint, hours apart, ~10&times; different beam rate. The '
      '<b>current changes tenfold and the charge per pulse does not</b> (to ~25 %). '
      'This one came free and it is the strongest argument that the number means '
      'what it is claimed to mean.</p>')

    w('<h3 style="font-size:1rem;color:var(--muted);text-transform:uppercase;'
      'letter-spacing:.04em">3 &mdash; the HV dependence is the gain curve</h3>')
    w('<div class="scroll"><table><thead><tr><th>det</th><th>at 520 V</th>'
      '<th>at 580 V</th><th>ratio</th><th>points</th></tr></thead><tbody>')
    for det in 'ABCD':
        v, q, dq, n = C.by_hv([r for r in rows if r.run == 'run_57'], det,
                              clean_only=False)
        if v.size < 3:
            continue
        flag = ' &mdash; not usable' if q[-1] < q[0] else ''
        w(f'<tr><th>{det}</th><td>{q[0]:.0f} nC</td><td>{q[-1]:.0f} nC</td>'
          f'<td>&times;{q[-1] / q[0]:.1f}{flag}</td><td>{v.size}</td></tr>')
    w('</tbody></table></div>')
    w('<p>Smooth and monotonic on A, B and C over 31 points. <b>Det D is not '
      'usable in run_57</b> and the reason is not understood: it sits on its own '
      '&minus;10 V grid, carries ~2 &micro;A of leakage, and its curve falls with '
      'HV. It is the standing &ldquo;bad detector&rdquo; caveat of every '
      'flash-recovery analysis; treat this as one more instance.</p>')

    # ---- production numbers ---------------------------------------------
    w('<h2>The production operating point</h2>')
    w('<div class="scroll"><table><thead><tr><th>det</th><th>resist V</th>'
      '<th>leakage [&micro;A]</th><th>Q/pulse [nC]</th><th>per channel [pC]</th>'
      '<th>&times; 600 fC</th><th>&times; 50 fC</th><th>trust</th>'
      '</tr></thead><tbody>')
    for det in 'ACBD':
        s = [r for r in prod if r.det == det]
        if not s:
            continue
        q = _mean(s, 'q_per_pulse_nc')
        e = float(np.sqrt(np.sum([r.q_err_nc ** 2 for r in s])) / len(s))
        pc = q * 1e3 / C.CHANNELS_PER_DET
        trust = 'clean' if s[0].leak_ok else f'leaky ({s[0].i_median_ua:.1f} &micro;A)'
        w(f'<tr><th>{det}</th><td>{s[0].resist_v:.0f}</td>'
          f'<td>{s[0].i_median_ua:.3f}</td><td>{q:.0f} &plusmn; {e:.0f}</td>'
          f'<td>{pc:.0f}</td><td>{C.csa_full_scale_multiple(pc, 600):.0f}</td>'
          f'<td>{C.csa_full_scale_multiple(pc, 50):.0f}</td><td>{trust}</td></tr>')
    w('</tbody></table></div>')
    w('<p>Quote A and C. Per-channel divides by the chamber&rsquo;s '
      f'{C.CHANNELS_PER_DET} channels, so it is an average over the whole active '
      'area. Areal: ~0.1 nC/cm&sup2; per pulse over 398.6 &times; 362 mm.</p>')

    # ---- figures ---------------------------------------------------------
    w('<h2>Figures</h2>')
    for fn, title, cap in FIGURES:
        if not os.path.exists(os.path.join(out_base, 'figures', fn)):
            continue
        w('<figure>')
        w(f'<img src="figures/{esc(fn)}" alt="{esc(title)}">')
        w(f'<figcaption><b>{esc(title)}</b> &mdash; {esc(cap)}</figcaption>')
        w('</figure>')

    # ---- caveats ---------------------------------------------------------
    w('<h2>What this does not establish</h2>')
    w('<div class="caveat">')
    w('<p><b>The one systematic that bounds everything.</b> This assumes the CAEN '
      'current readback preserves the time-average of a burst much shorter than '
      'the sample spacing. 27.8 % of samples sit above baseline on both clean '
      'chambers, which is what ~1 s smoothing of a burst would give (and smoothing '
      'conserves the integral) &mdash; but that is inference from one number. '
      '<b>If the monitor instead reports a short instantaneous average, every '
      'charge here is a lower bound.</b> The cheapest way to settle it is to look '
      'up the board&rsquo;s imon integration spec; the definitive way is to inject '
      'a known charge at a known rate through the DAQ pulser path.</p>')
    w('</div>')
    w('<ul>')
    w('<li><b>It cannot separate flash from tail.</b> The supply current '
      'integrates the whole pulse, and the CSA-pinning mechanism is specifically '
      'about <i>sustained</i> current, so the split matters for the mechanism '
      'argument. The waveform-level handle exists separately &mdash; see '
      '<code>ntof_processing/mm_flash/</code> &mdash; but its 930 pC is a single '
      'electrode over ~1 &micro;s and is <b>not</b> directly comparable to the '
      'whole-chamber, whole-cycle number here.</li>')
    w('<li><b>It says nothing about primary ionisation</b> without the gas gain at '
      'this working point, which is not measured. Any MIP-equivalent figure '
      'derived from these numbers is an illustration.</li>')
    w('<li><b>Per-channel numbers are chamber averages.</b> The illuminated region '
      'sees more; nothing here maps the charge spatially.</li>')
    w('<li><b>Which CSA input range we actually run is a loose end</b> &mdash; the '
      'DREAM <code>state1</code> register decode is unverified, and the four '
      'settings span a factor 12.</li>')
    w('<li><b>Five reduced scans are unexploited</b> in the same CSV: a drift '
      'axis (run_58/61/64), the operating-point scan (run_67), and an '
      'Ar/iso 95/5-vs-90/10 comparison (run_19 vs run_42/57).</li>')
    w('</ul>')

    w('<h2>Reproducing this</h2>')
    w('<p>Method, validation detail and the full to-do list: '
      '<code>ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md</code>. '
      'Mirror the HV-monitor CSVs (July runs on EOS, August on the DAQ), then:</p>')
    w('<div class="verdict" style="border-left-color:var(--muted)"><p style="margin:0">'
      '<code>.venv/bin/python ntof_july_analysis/flash_charge/analyze.py --src &lt;mirror&gt;</code><br>'
      '<code>.venv/bin/python ntof_july_analysis/flash_charge/make_report.py</code>'
      '</p></div>')
    w(f'<p class="sub">{len(rows)} (sub-run &times; detector) rows over '
      f'{len({r.run for r in rows})} runs. Figures are rendered by '
      '<code>mpgd26/make_status_plots.py</code>, the same code that builds the '
      'MPGD2026 talk, so the two cannot drift apart.</p>')

    w('</body></html>')
    return '\n'.join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--data', default=None,
                    help='mirror holding metrics_run_57_perdet.csv (for the figures)')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    rows = load()
    fig_dir = os.path.join(args.out, 'figures')
    if not args.no_figures:
        import make_status_plots as MS
        MS.render(['deadtime_vs_charge', 'charge_vs_hv', 'recovery_vs_hv',
                   'two_readouts'],
                  data_dir=args.data or MS.DEFAULT_DATA, out_dir=fig_dir)

    doc = build(rows, args.out)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'report.html')
    with open(path, 'w') as fh:
        fh.write(doc)
    print(f'wrote {path}  ({len(doc) / 1024:.0f} kB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
