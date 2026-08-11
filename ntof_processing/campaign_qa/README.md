# campaign_qa — are the runs we processed as good as n_TOF's?

Acceptance checking for the self-processing campaign
(`../process_missing_runs.sh`, `../SELF_PROCESSING_RUNBOOK.md`). Answer as of
2026-08-11: **yes** — see [`results/report.html`](results/report.html).

## Two kinds of comparison, because there are two kinds of run

**On 08-11 n_TOF merged 27 previously-unmerged runs**, two of which
(224573, 224577) we had processed ourselves. So eight runs now exist in **both**
processings, and for those the check is direct: `compare_identity.py` compares
every hit, column by column, on the same bunches. Result: with the same
UserInput we reproduce n_TOF **bit for bit**. See
[`../FINDINGS_2026-08-11_official_ledger.md`](../FINDINGS_2026-08-11_official_ledger.md).

**For 224688-224718 there is still nothing to diff against** — that block is
exactly the block n_TOF has never processed. There the check stays an
equivalence argument in two parts:

* **configuration**, compared exactly — every product records the UserInput it
  was made with in its own `history` object;
* **behaviour**, compared statistically against the nearest official runs in
  time, with everything normalised to delivered protons.

## Two traps that make a healthy run look broken

* **The official runs next door have no beam.** 224678-224687 sit at zero
  PulseIntensity and zero PKUP amplitude. Use one as the control and our output
  looks 400x too busy.
* **Empty PS pulses inside our own runs.** Those bunches have no flash, so
  `tflash` is 0 and every flash check flags them. The first partial of 224692 is
  75 % empty; the run as a whole is 98 % beam and clean.

So: **run `beam_state.py` first, and gate every comparison on protons.**

## The scripts

| script | what it answers |
|---|---|
| `official_ledger.py` | per run: what n_TOF has, what we have, and which UserInput each was made with — the bookkeeping |
| `compare_identity.py` | hit for hit, ours against official on the same bunches, for the runs that exist in both |
| `beam_state.py` | whole-run beam state, one open per run, from the `index` tree |
| `verify_transferred.py` | structure: contiguity, `ceil(raw/4)`, history, and a real read of **every** partial |
| `history_diff.py` | which UserInput a product was actually made with, ours vs official, path prefixes dropped |
| `compare_campaign.py` | beam-gated physics: rates per 1e12 p, modal `tflash`, off-flash fraction, per-arm offsets, amplitudes |
| `run_profile.py` | ungated per-run profile incl. DAQ zero-suppression settings; use for a first look |
| `inspect_pair.py` | trees, branches and entries of two files side by side |
| `make_report.py` | builds `results/report.html` from the logs above |

## Running it

Everything except the report runs on lxplus, where the files are.

```bash
ssh -K lxplus
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
cd /afs/cern.ch/work/d/dneff/x17_reproc/campaign_qa      # rsync this directory there

O=/eos/experiment/ntof/data/x17/reproc/prod_v12
F=/eos/experiment/ntof/processing/official/completed

python3 -u official_ledger.py --csv=ledger.csv --json=ledger.json   # ~15 min, 445 runs
python3 -u compare_identity.py --run=224572 \
        --ours=/eos/experiment/ntof/data/x17/reproc/v12_liqpileup/completed/224572 \
        --official=$F/224572 --json=identity_224572.json
python3 -u beam_state.py --json=beam_state.json $O/*/completed/*
python3 -u verify_transferred.py 224688 224689 ...        # slow: opens every partial
python3 -u compare_campaign.py --partials=1 --json=compare.json \
        ours=$O/224691/completed/224691,... official=$F/224672,...
python3 ../quality_metrics.py ours=<f>,<f> official=<f>,<f>
```

**Use `python3 -u`.** Output to a log on AFS is block-buffered otherwise and a
half-hour job shows nothing until it exits.

Then, locally, with the logs rsynced into `results/`:

```bash
.venv/bin/python ntof_processing/campaign_qa/make_report.py
```

`make_report.py` reads whatever is in `results/` and marks missing sections as
pending, so it can be run while a check is still going.

## What is committed

The scripts, `results/report.html`, and the small text/JSON records the report is
built from. **Figures are not tracked** — the repo stopped tracking `*.png` in
August 2026 (they were 250 MB of a 272 MB pack); `make_report.py` regenerates
`results/figures/` from `results/compare.json`. The logs under `results/` are
force-added past the global `*.log` ignore because the report quotes them and they
are a few KB each.
