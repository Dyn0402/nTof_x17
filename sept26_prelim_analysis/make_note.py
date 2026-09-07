#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_note.py -- build the preliminary-analysis progress page.

One self-contained HTML document carrying the plan, the live status and the
deferred register, for <https://dylan-neff.web.cern.ch/notes/>.  Written as a
generator rather than by hand so that re-running it after the analysis moves
updates the numbers, the stage table and the verdict text together -- the same
convention as ntof_july_analysis/leadshield_compare/make_report.py and
ntof_run_report/make_report.py.

The prose lives here.  There is no template file to keep in sync.

    python make_note.py                 # -> report.html beside this file
    python make_note.py --out X.html

Publish with the publish-note skill:

    py <site>/scripts/add-note.py sept26_prelim_analysis/report.html \\
        --slug x17-prelim-plan --tags X17,analysis --force --deploy

Tables that are still empty are rendered as an explicit "not measured yet" row
rather than omitted, because a missing row reads as an oversight and an empty
one reads as honest.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TITLE = "X17 preliminary analysis — plan and status"
SUMMARY = ("The plan of record for the first end-to-end pass over the 2026 n_TOF "
           "campaign: the chain stage by stage, what it will not establish, and "
           "what is deferred to October.")
BOARD = "https://dylan-neff.web.cern.ch/x17/analysis.html"

# --------------------------------------------------------------------------
# The state tables.  These are the things that change week to week; everything
# below them is prose.  Replace the literals with reads from sample.csv, the
# class census and the benchmark log as those land.
# --------------------------------------------------------------------------

STAGES = [
    ("Freeze the analysis sample", "active",
     "sample.csv — one row per (run, sub-run, tag), with the reason each was kept or cut"),
    ("Time base and the flash veto", "todo",
     "per-trigger time-since-flash and E<sub>n</sub>, and one stated veto window with its cost"),
    ("The candidate filter", "todo",
     "candidates.parquet — one row per DREAM trigger (~31 M), one terminal class each"),
    ("Reconstruct the candidates (WFT)", "todo",
     "per-(arm, tag) track tables + the multi-track sidecar, and a measured cost per 1,000 events"),
    ("The track database", "todo",
     "tracks.parquet and scint.parquet"),
    ("Scintillator positions", "todo",
     "fitted attenuation length and Δt scale per bar, a quoted resolution, the position column"),
    ("Pair finding — intra, inter, implied", "todo",
     "pairs.parquet and a cutflow per topology"),
    ("The opening-angle spectrum", "todo",
     "the raw spectra with statistical errors, and the numbers behind them"),
    ("Acceptance and efficiency", "todo",
     "a correction curve versus opening angle, per topology"),
    ("Backgrounds", "todo",
     "one normalised shape per component, and their sum against the raw spectrum"),
    ("The preliminary picture", "todo",
     "the summary figures, sized for a projected slide, and this page finished"),
]

# (what, value, measured on).  value None -> rendered as "not measured yet".
BENCHMARKS = [
    ("WFT reconstruction throughput", None, None),
    ("Candidate-filter throughput", None, None),
    ("Class census (fraction per class)", None, None),
    ("Campaign reconstruction cost", None, None),
    ("Link speed to CERN from this machine", None, None),
]

BENCH_NOTE = ("The only number in hand is historical: run_79, 12,534 events on six "
              "cores in about 1 h 50 m, i.e. <b>~1,100 events per core-hour per "
              "chamber</b> on 20-sample windows. Treat it as an order of magnitude "
              "until it is re-measured.")

INHERITED = [
    ("DREAM ↔ n_TOF match",
     "279,938 of 281,488 beam pulses joined since run_79 = <b>99.45 %</b>. 1,516 pulses "
     "had no n_TOF running — irrecoverable, and outside the denominator. Per-trigger match "
     "efficiency 95.8 % (wall AND plastic), accidental rate 0.049 %, resolution 6 ns.",
     "qa-match.html"),
    ("n_TOF reprocessing",
     "445 of 445 runs have a complete product at the campaign recipe.", None),
    ("Reconstruction",
     "Waveform-first, per-chamber bundles seeded from the June cosmic bench, on the corrected "
     "sharing kernel. Returns every candidate track, not just the winner.", None),
    ("Geometry",
     "Chambers are pinwheeled; the in-plane sign and the pointing lever are both corrected. "
     "Active volumes ported from the as-built Geant4 model.", None),
    ("Pointing",
     "Chambers A and C, calibrated independently and looking at each other from opposite sides, "
     "back-project to the same spot — <b>−9.6 and −8.9 mm</b>, inside the 10 mm capsule bore.", None),
]

CONDITIONS = [
    ("The noise floor doubled on 23 July and never came back",
     "The DREAM readout clock divider went 6 → 4 and the per-channel residual noise rose ×2.0 on "
     "all eight front-ends at once. run_67 and earlier are the quiet configuration; run_69 onward "
     "the noisy one; run_68 sits inside the bracket and is not placed by the pedestals. "
     "<b>The entire production period is on the noisy side.</b>"),
    ("Chamber A's x-view connector 8 was dead through run_79",
     "Channels 448–511 were electrically disconnected from 22 July to the 27 July access; 41 of "
     "those 64 recorded no hits at all. Every sub-run of run_79 needs them masked. Live again "
     "from run_83."),
    ("The angle scale is not physical on chambers B and D",
     "In-situ k of 1.25 (A), 1.58 (C), <b>1.99 (B), 1.70 (D)</b> — drift velocities from 34 down "
     "to 21 µm/ns against a Magboltz prior of 42.6. B additionally truncates 46–56 % of its "
     "columns; D has a dead region and a vertical stripe. A is credible, C is quotable, B and D "
     "are not."),
    ("No absolute position better than about a centimetre",
     "The residual ~9 mm offset common to A and C is either a real target/beam offset or a survey "
     "error, and nothing in hand separates them."),
]

CLASSES = [
    ("INTER", "track-like clusters in exactly <b>two</b> chambers", "inter-chamber pairs — the signal"),
    ("INTRA", "≥ 2 separated track-like clusters in <b>one</b> chamber", "intra-chamber pairs — the control"),
    ("IMPLIED", "clusters in one chamber, but an n_TOF coincidence in two arms", "recovering a B or D miss"),
    ("SINGLE", "one track-like cluster, one chamber", "QA, efficiency, scintillator calibration (prescaled)"),
    ("BUSY", "≥ 3 chambers active, or &gt; 120 clean strips in ≥ 3", "pile-up: vetoed, but counted"),
    ("NONE", "nothing track-like", "denominator only"),
]

NOT_ESTABLISHED = [
    ("No resolution measurement.",
     "There is no reference telescope at n_TOF. Every width quoted is an upper limit."),
    ("No absolute position better than ~1 cm.",
     "The A/C common ~9 mm offset is unresolved between a real target offset and a survey error."),
    ("Chamber B and D angles are not quotable.",
     "They enter as tagging chambers — was there a track at all? Their angles carry a flag, and any "
     "pair using one is reported separately from an A×C pair."),
    ("No invariant mass.",
     "It needs the energy sharing, which needs a scintillator energy calibration that does not exist yet."),
    ("No efficiency from first principles.",
     "Acceptance is geometry folded with a measured reconstruction efficiency. Both preliminary."),
    ("Merged double tracks are lost.",
     "Two tracks closer than 12 mm fit as one bad-χ² compromise. They are flagged and counted, not recovered."),
]

NEXT_STEPS = [
    ("Land on the machine",
     "pull both repositories, check for unpushed commits on each first, and confirm "
     "<code>kinit</code> and <code>ssh lxplus</code>."),
    ("Deploy what is already written",
     "one command, and it has been waiting — it puts this page and the board live."),
    ("Verify the five CERN assumptions",
     "is the re-slim complete and which sub-runs have a product; which runs already carry a "
     "reconstruction and at which code commit; whether run_79's products get rebuilt or patched "
     "at read time; the link speed home; condor throughput and quota. Each answer written down, "
     "because an unanswered row is more useful than a forgotten one."),
    ("Stage the run_145 development bundle",
     "cheapest-and-most-useful first: the existing track tables are megabytes and unblock the "
     "database stage immediately; the waveforms are only needed to prove the reconstruction path."),
    ("Scaffold the package",
     "the shared presentation figure style, and machine-aware data roots so that no script "
     "hard-codes a path — both before any analysis code, because everything imports them."),
    ("Start the chain",
     "freeze the sample, then run the candidate filter on run_145."),
]

DEFERRED = [
    ("D1", "Merged-cluster double tracks",
     "they are flagged and counted, so the loss is measured rather than silent",
     "a two-column NNLS design matrix plus a 1-vs-2-track model-selection penalty"),
    ("D2", "Simulating the double-track finding efficiency",
     "it does not block a preliminary spectrum; it blocks calling one a measurement",
     "Geant4 pairs through the response sim and the full chain, binned by opening angle and "
     "track separation. The most valuable single item here"),
    ("D3", "Breaking the X↔Y pairing ambiguity",
     "this week writes both hypotheses and quotes the spread between them as a systematic",
     "two handles already in the schema and unused — whether a pairing points back at the ³He "
     "sample, and whether the two projections carry the same charge profile"),
    ("D4", "Calibrating chambers B and D",
     "they enter as tagging chambers only, and no angle of theirs is quoted alone",
     "a dedicated per-chamber study. The biggest single limit on the signal region, since its "
     "topology needs one of them in most events"),
    ("D5", "dE/dx in the Micromegas",
     "gain variation across a chamber is expected to dominate, so a number now would measure the "
     "gain map rather than the ionisation",
     "a per-region gain map from the pulser and source runs"),
    ("D6", "Scintillator energy calibration and calorimetry",
     "without it the deliverable is the opening angle alone, which is what this week promises",
     "an energy scale for the plastics, the SiPM wall and the liquids, each verified before use. "
     "This is what would unlock the invariant mass"),
    ("D7", "Liquid scintillator gain across the surface",
     "no liquid amplitude is used as energy this week",
     "measuring how far the gain moves with distance from the PMT and across the surface, with "
     "Micromegas tracks defining the impact point"),
    ("D8", "Absolute alignment",
     "positions are quoted to about a centimetre, which is enough for an angle",
     "survey information, or an independent beam-spot measurement — not more tracks"),
    ("D9", "In-situ drift velocity and diffusion",
     "a wrong drift velocity scales angles; it does not invent a peak",
     "porting the stacked-NNLS erfc-endpoint gap study to the beam runs. It costs no refit"),
    ("D10", "Per-channel gain and dead-strip maps",
     "one gain per chamber changes resolution, not the presence of a peak",
     "a pass over the pulser and source runs, per run condition"),
    ("D11", "The neutron-energy axis",
     "time-since-flash is already written into every row, so nothing is lost by not analysing it",
     "the spectrum resolved in E<sub>n</sub>, once there are enough pairs to bin twice"),
    ("D12", "External pair conversion in the material",
     "this week it is a geometry argument rather than a measured shape",
     "a simulated conversion background normalised to the capsule and frame material"),
    ("D13", "Measuring the candidate filter's efficiency properly",
     "the prescaled control sample gives a first number",
     "a dedicated unfiltered reconstruction pass over one sub-run"),
    ("D14", "run_67 and run_68",
     "they are the quiet-configuration runs and cannot be pooled with the production period",
     "either their own calibration, or recording the loss"),
    ("D15", "Pile-up within a bunch",
     "the BUSY class is vetoed wholesale and counted, so it is a known exclusion",
     "looking inside it — about 112 triggers share each proton pulse"),
]


# --------------------------------------------------------------------------
def esc(s):
    return html.escape(str(s), quote=False)


def stage_table():
    rows = []
    for i, (name, status, produces) in enumerate(STAGES, 1):
        rows.append(
            f'<tr><td class="num">{i}</td><td>{name}</td>'
            f'<td><span class="pill {status}">{status}</span></td>'
            f'<td class="soft">{produces}</td></tr>')
    return "\n".join(rows)


def bench_table():
    rows = []
    for what, value, where in BENCHMARKS:
        if value is None:
            rows.append(f'<tr><td>{what}</td><td class="empty">not measured yet</td>'
                        f'<td class="empty">—</td></tr>')
        else:
            rows.append(f'<tr><td>{what}</td><td><b>{value}</b></td><td class="soft">{where}</td></tr>')
    return "\n".join(rows)


def build(out_path):
    today = _dt.date.today().isoformat()

    inherited = "\n".join(
        f'<tr><td>{k}</td><td>{v}'
        + (f' <a href="../x17/{link}">→</a>' if link else '')
        + '</td></tr>'
        for k, v, link in INHERITED)

    conditions = "\n".join(
        f'<div class="cond"><h4>{k}</h4><p>{v}</p></div>' for k, v in CONDITIONS)

    classes = "\n".join(
        f'<tr><td><code>{k}</code></td><td>{d}</td><td class="soft">{f}</td></tr>'
        for k, d, f in CLASSES)

    notest = "\n".join(
        f'<li><b>{k}</b> {v}</li>' for k, v in NOT_ESTABLISHED)

    next_steps = "\n".join(
        f'<li><b>{k}</b> — {v}</li>' for k, v in NEXT_STEPS)

    deferred = "\n".join(
        f'<div class="defer"><h4><span class="did">{i}</span> {t}</h4>'
        f'<p><b>Safe because</b> {why}. <b>Unblocked by</b> {unb}.</p></div>'
        for i, t, why, unb in DEFERRED)

    doc = f"""<!--note
title: {TITLE}
summary: {SUMMARY}
date: {today}
tags: X17,analysis,n_TOF
-->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(TITLE)}</title>
<meta name="description" content="{esc(SUMMARY)}">
<style>
  :root {{
    --bg:#fbfaf8; --fg:#1a1a1a; --soft:#5a5a5a; --rule:#e2ded8;
    --card:#ffffff; --accent:#8a3324; --accent2:#2c5f7c; --code:#f2efe9;
    --ok:#2f6b3f; --active:#8a6d1f; --todo:#6a6a6a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#15161a; --fg:#e8e6e1; --soft:#a5a29b; --rule:#2e3038;
      --card:#1c1e24; --accent:#e07a63; --accent2:#7fb5d4; --code:#22252c;
      --ok:#7fc08e; --active:#d6b45a; --todo:#8d8d8d;
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:17px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:48px 22px 90px; }}
  h1 {{ font-size:2.05rem; line-height:1.2; margin:0 0 6px; letter-spacing:-.01em; }}
  h2 {{ font-size:1.4rem; margin:56px 0 6px; padding-top:22px; border-top:1px solid var(--rule);
        letter-spacing:-.01em; }}
  h3 {{ font-size:1.1rem; margin:32px 0 6px; }}
  h4 {{ font-size:1rem; margin:0 0 4px; }}
  p {{ margin:.7em 0; }}
  .kicker {{ color:var(--accent); font-weight:600; text-transform:uppercase;
    letter-spacing:.09em; font-size:.75rem; margin:0 0 10px; }}
  .dateline {{ color:var(--soft); font-size:.9rem; margin:0 0 26px; }}
  .lede {{ font-size:1.13rem; }}
  .verdict {{ background:var(--card); border:1px solid var(--rule);
    border-left:4px solid var(--accent); border-radius:6px; padding:18px 22px; margin:26px 0; }}
  .verdict p:first-child {{ margin-top:0; }} .verdict p:last-child {{ margin-bottom:0; }}
  table {{ width:100%; border-collapse:collapse; margin:18px 0; font-size:.95rem; }}
  th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--rule);
           vertical-align:top; }}
  th {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; color:var(--soft);
        font-weight:600; }}
  td.num {{ color:var(--soft); font-variant-numeric:tabular-nums; width:2.4em; }}
  td.soft, .soft {{ color:var(--soft); }}
  td.empty {{ color:var(--soft); font-style:italic; }}
  .scroll {{ overflow-x:auto; }}
  code {{ background:var(--code); padding:1px 5px; border-radius:3px; font-size:.9em; }}
  pre {{ background:var(--code); padding:16px 18px; border-radius:6px; overflow-x:auto;
    font-size:.86rem; line-height:1.5; }}
  a {{ color:var(--accent2); }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:.74rem;
    font-weight:600; text-transform:uppercase; letter-spacing:.05em;
    border:1px solid currentColor; }}
  .pill.todo {{ color:var(--todo); }} .pill.active {{ color:var(--active); }}
  .pill.done {{ color:var(--ok); }} .pill.blocked {{ color:var(--accent); }}
  .cond, .defer {{ background:var(--card); border:1px solid var(--rule); border-radius:6px;
    padding:14px 18px; margin:12px 0; }}
  .cond h4 {{ color:var(--accent); }}
  .cond p, .defer p {{ margin:4px 0 0; font-size:.95rem; }}
  .did {{ display:inline-block; background:var(--code); color:var(--soft); border-radius:4px;
    padding:1px 7px; font-size:.8rem; font-weight:700; margin-right:6px; }}
  ul.tight li {{ margin:.5em 0; }}
  .blocker {{ background:var(--card); border:1px solid var(--accent); border-radius:6px;
    padding:16px 20px; margin:22px 0; }}
  .blocker h4 {{ color:var(--accent); }}
  .foot {{ margin-top:64px; padding-top:20px; border-top:1px solid var(--rule);
    color:var(--soft); font-size:.86rem; }}
</style>
</head>
<body>
<div class="wrap">

<p class="kicker">X17 at n_TOF · internal</p>
<h1>{esc(TITLE)}</h1>
<p class="dateline">Written 2026-09-07 · working week 8–15 September ·
  collaboration meeting 30 September ·
  <a href="{BOARD}">the analysis board</a></p>

<p class="lede">The goal for the week is one distribution: <b>the opening angle
of e<sup>+</sup>e<sup>&minus;</sup> pair candidates</b> from the 2026 n_TOF EAR2
campaign, preliminary. This page is the plan of record, the live status, and the
list of what is deliberately being left for October.</p>

<div class="verdict">
<p><b>The one thing to know before reading on.</b> An X17 at 16.8 MeV produced at
20.58 MeV has a minimum opening angle of <b>109&deg;</b>, and the pair piles up
between 110 and 140&deg;. One chamber, 400 mm square at 235 mm from the target,
subtends about &plusmn;40&deg;. <b>A signal pair therefore cannot fit inside one
chamber</b> — it lands in two, and in this geometry that usually pairs one of the
good chambers (A, C, which face each other) with one of the two that are not
calibrated (B, D, which sit between them).</p>
<p>So the interesting topology is the hard one, and the two chambers on the
critical path are the two we have least confidence in. The single-chamber
topology spans 0–90&deg;: it is the internal-pair-creation continuum below the
X17 threshold, which makes it the control and the normalisation, not the
signal.</p>
</div>

<h2>Where the analysis stands</h2>

<div class="scroll"><table>
<thead><tr><th></th><th>Stage</th><th>Status</th><th>Produces</th></tr></thead>
<tbody>
{stage_table()}
</tbody></table></div>

<div class="blocker">
<h4>Everything above is written; nothing above has been run</h4>
<p>The plan was written on a machine with no Kerberos ticket and no accepted
key, so lxplus was unreachable and nothing at CERN could be verified or moved.
The analysis therefore runs on the Ubuntu laptop, which has a working ticket,
the data on a local disk, and every path the repository's documentation already
assumes. Nothing needs redoing after the move.</p>
</div>

<h3>Next, in order</h3>
<ol class="tight">
{next_steps}
</ol>

<h2>What this is standing on</h2>
<p>Inherited from the campaign work, and not to be rebuilt:</p>
<div class="scroll"><table><tbody>
{inherited}
</tbody></table></div>

<h3>Four conditions that would silently bias a number</h3>
{conditions}

<h2>The chain</h2>
<pre>[0] sample        which runs, sub-runs and tags enter, and what each cut costs
[1] time base     time-since-flash, E_n, and where the usable window starts
[2] candidates    hits-level filter over ALL triggers -> one class per trigger
[3] reco          full waveform fit, only on the classes worth it
[4] database      X/Y pairing -> 3D segments -> global frame -> predictions
[5] scint         where a particle crossed the n_TOF scintillators
[6] pairs         intra-chamber / inter-chamber / implied, plus backgrounds
[7] spectrum      opening angle per topology, raw and acceptance-corrected</pre>

<p>Stages 0–2 are cheap and run over everything. Stage 3 is expensive and runs
only on what stage 2 selected. <b>That asymmetry is the whole design:</b>
reconstructing 31.2 million triggers across four chambers blind is of order
10<sup>5</sup> core-hours, which is not a week.</p>

<h3>The candidate classes</h3>
<p>Every trigger gets exactly one terminal class, so the ledger is a partition
and every downstream efficiency has an honest denominator. The classification
runs on hit tables alone — no waveform fitting — crossed with which n_TOF arms
saw a wall-and-plastic coincidence.</p>

<div class="scroll"><table>
<thead><tr><th>Class</th><th>Definition</th><th>Feeds</th></tr></thead>
<tbody>
{classes}
</tbody></table></div>

<p class="soft">The filter's own efficiency is unmeasured going in. A prescaled
control sample drawn from <code>SINGLE</code>, <code>BUSY</code> and
<code>NONE</code> is reconstructed alongside the candidates for exactly that
reason — without it the spectrum has no acceptance.</p>

<h3>The track database</h3>
<p>The artefact the week has to leave behind, because every later analysis reads
it instead of re-running anything. One row per 3D track segment, carrying its
geometry as a line rather than a point, its quality, its charge profile, its
timing, where it points, which scintillator volumes it should have crossed —
and which calibration bundle produced it. Beside it, one row per n_TOF hit with
the track predicted to have crossed it, which is what turns "characterise the
liquid scintillators with the other detectors defining the track" from a project
into a query.</p>
<p>Two rules, both learned the hard way upstream: <b>never collapse an ambiguity
at write time</b> — a two-track event's X↔Y pairing is genuinely degenerate, so
both hypotheses are written with their scores — and <b>carry the calibration
provenance in the row</b>, because a bundle used outside its conditions is a
silent error.</p>

<h3>Scintillator positions</h3>
<p>Across the SiPM wall, the segment that fired gives 100 mm bins. Along the bar
— the beam direction, since EAR2's beam is vertical — there are two independent
estimators, the top-to-bottom transit time and the logarithm of the amplitude
ratio, which is linear in position with the attenuation length. Both were
characterised on n_TOF data alone. What is new this week is calibrating them
against Micromegas tracks, which supply the impact point the earlier study did
not have.</p>

<h2>Benchmarks</h2>
<p>Filled in as they are measured. Empty is honest; a guess is not.</p>
<div class="scroll"><table>
<thead><tr><th>What</th><th>Value</th><th>Measured on</th></tr></thead>
<tbody>
{bench_table()}
</tbody></table></div>
<p class="soft">{BENCH_NOTE}</p>

<h2>What this will not establish</h2>
<p>Stated up front so that no single figure has to carry the disclaimer alone.</p>
<ul class="tight">
{notest}
</ul>

<h2>Deferred to October</h2>
<p>Each item says <b>why it is safe to defer</b> and <b>what would unblock
it</b>, so that nothing here can later be read as an oversight.</p>
{deferred}

<p class="foot">Generated by <code>sept26_prelim_analysis/make_note.py</code> on
{today}. Source of record: <code>sept26_prelim_analysis/PLAN.md</code> and
<code>STATUS.md</code> in <code>nTof_x17</code>. Status is tracked live on
<a href="{BOARD}">the analysis board</a>.</p>

</div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=os.path.join(HERE, "report.html"))
    a = ap.parse_args()
    p = build(a.out)
    print(f"wrote {p}  ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()
