#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build results/report.html from inventory.py's two CSVs.

Generated, not hand-written: re-run `inventory.py` then this, and the numbers,
tables and verdict move together.  No figures yet -- this is an inventory.

    .venv/bin/python ntof_cosmics/make_report.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sept26_prelim_analysis.report_style import HEAD  # noqa: E402

RES = HERE / 'results'


def table(df: pd.DataFrame) -> str:
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in r) + '</tr>'
        for r in df.itertuples(index=False))
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def main() -> None:
    run = pd.read_csv(RES / 'cosmic_runs.csv')
    sub = pd.read_csv(RES / 'cosmic_subruns.csv')
    prod = run[run.era == 'production_point']
    p_h, p_ev = prod.usable_hours.sum(), prod.usable_events.sum()
    c_h = prod.loc[prod.category == 'production point, clean', 'hours'].sum()
    q_h = run.ntof_quiet_h.sum()
    rate = prod.events.sum() / max(prod.hours.sum(), 1e-9) / 3600
    interior = run[run.ntof_beam_interior > 0].run.tolist()
    big = prod.sort_values('hours', ascending=False).head(5)
    quiet_runs = run[run.ntof_quiet_h > 0].sort_values('ntof_quiet_h',
                                                       ascending=False)

    cat = (run.groupby('category')
           .agg(runs=('run', 'size'), subruns=('subruns', 'sum'),
                hours=('hours', 'sum'), events=('events', 'sum'),
                usable_hours=('usable_hours', 'sum'),
                ntof_quiet_h=('ntof_quiet_h', 'sum'))
           .round(2).reset_index())

    cols = ['run', 'start', 'category', 'subruns', 'hours', 'events', 'rate_hz',
            'noise', 'a_x_conn8_dead', 'ntof_beam', 'ntof_beam_interior',
            'usable_hours', 'ntof_quiet_h']

    out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>n_TOF beam-off cosmic runs</title>{HEAD}</head><body><main>
<p class="eyebrow">n_TOF EAR2 · X17 · ntof_cosmics</p>
<h1>The beam-off cosmic runs: what we have</h1>

<div class="verdict"><p><b>{len(prod)} cosmic runs at the production operating
point hold {p_h:.1f} h and {p_ev/1e6:.2f} M scintillator-singles triggers</b>,
all decoded on EOS, all under the post-23-July noise condition, and all with the
run_79 HV and readout — only the trigger differs. {c_h:.1f} h of it never saw an
n_TOF pulse. Everywhere else the beam sits only at run edges (beam loss and beam
return around the auto-substituted cosmic run), so an event-level veto around
the pulse times leaves the whole sample usable. Beam inside the cosmic running:
{', '.join(f'run_{r}' for r in interior) or 'none'}.
<b>For {q_h:.1f} h, n_TOF was recording PS-timed triggers with no protons
on target</b>, which gives a random-trigger view of the scintillators during the
cosmic running. Most of it is in run_149 and run_103.</p></div>

<h2>Headline</h2>
<ul>
<li>Trigger: scintillator singles (per-sector wall×plastic), veto open, no PS
gate. Rate {rate:.1f} Hz at the production point, stable run to run.
<b>One arm is enough to trigger</b>, so a particle through two arms is a
subset still to be counted, not the sample.</li>
<li><b>run_80 still has chamber A x connector 8 dead</b> (channels 448–511 of
FEU 3, live again from run_83): mask it, or drop run_80 ({run.loc[run.run == 80, 'hours'].sum():.1f} h)
from any A-x study.</li>
<li>Readout identical to run_79: no zero suppression, latency 27, 20 samples ×
60 ns. HV resist A540/B540/C525/D520, drift 700 on all four.</li>
<li>Pre-production (run_54–74): {run[run.era == 'pre_production'].hours.sum():.1f} h at
other HV and timing (latency 35, 32 samples), mostly resist ladders. Useful for
gain studies, not as a background reference for the production data.</li>
<li>run_148 is empty.</li>
</ul>

<h2>By category</h2>
{table(cat)}

<h2>Which runs to use</h2>
<p><b>Background reference at the production point:</b> the five longest,
which make up {big.hours.sum():.1f} of the {prod.hours.sum():.1f} h —
{', '.join(f'run_{r} ({h:.1f} h)' for r, h in zip(big.run, big.hours))}.
PLAN §S4 named run_83 and run_146; they are 0.2 h and 0.35 h and both have
beam at an edge, so they are a poor choice.</p>
<p><b>With an n_TOF random-trigger companion:</b>
{', '.join(f'run_{r} ({h:.2f} h)' for r, h in zip(quiet_runs.run, quiet_runs.ntof_quiet_h))}.
These are the only places a cosmic scintillator time (and hence
arm-to-arm time of flight) could exist. That depends on two things not yet
checked: that cosmic triggers land inside n_TOF acquisition windows, and that the
DREAM↔n_TOF clock join can be carried across a run with no flash to lock on.</p>

<h2>Per run</h2>
{table(run[cols])}
<p>Per sub-run, with HV and the n_TOF runs overlapping each one:
<code>cosmic_subruns.csv</code> ({len(sub)} rows).</p>

<h2>What this does not establish</h2>
<ul>
<li><b>Beam flags are minute-resolution.</b> They come from the per-minute
<code>beam_class</code> log (n_TOF-destined pulses only). A sub-run shorter than
about 6 min has almost no "interior", so <code>edge</code> on a short sub-run is
weak evidence. run_157's 22 pulses in 7 min (0.031 Hz residual beam, found by the
flash-charge study) are exactly that case. An analysis must veto by pulse
time, not by this flag.</li>
<li><b>Configured HV, not measured.</b> The <code>hv_monitor.csv</code> per
sub-run is on EOS and was not read; a trip would not show here.</li>
<li><b>"n_TOF quiet" is inferred from file size</b> (bytes per bunch, a gap of
two decades between beam and no-beam runs), confirmed against
<code>beam_state</code> only for 224706. Whether n_TOF's windows contain any of
our cosmic triggers is not checked.</li>
<li>No tracking has been run on any of this; nothing here says how many
triggers carry a track, let alone a through-going one.</li>
</ul>
</main></body></html>
"""
    (RES / 'report.html').write_text(out)
    print(f'wrote {RES / "report.html"}')


if __name__ == '__main__':
    main()
