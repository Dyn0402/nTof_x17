# ntof_run_report — the end-of-run report for the 2026 n_TOF EAR2 physics run

One document above the per-topic analysis packages: what was installed, what
the γ flash does to the read-out, what the run actually measured, and what a
post-LS3 measurement would have to solve.

```bash
.venv/bin/python -m ntof_run_report.make_report                 # report.html + figures/
.venv/bin/python -m ntof_run_report.make_report --inline OUT.html   # single self-contained file
```

Published (unlisted) at
<https://dylan-neff.web.cern.ch/notes/x17-ntof-end-of-run-2026.html>.
`--inline` is what goes there: the notes site is an offline-first PWA, so the
note has to carry its own images.

## Layout

| | |
|---|---|
| `make_report.py` | the prose, the tables and the page. Everything is here; there is no template file to keep in sync. |
| `assets.py` | the figure inventory — one source path per figure, in the analysis package that produced it. A moved figure fails loudly at build time instead of vanishing from the page. |
| `figures_local.py` | the two figures that exist nowhere else: beam availability and the event census. |
| `count_events.py` | **runs on lxplus, not here.** Scans `decoded_root` on EOS for every sub-run's event count. ~15 min; the result is committed as `data/events_per_subrun.csv` so the report builds without EOS. |
| `data/events_per_subrun.csv` | one row per (sub-run, file tag): events and the DAQ's own timestamp. |

## Conventions this report holds itself to

- **Nothing is re-reduced here.** Every number is quoted from a committed result
  elsewhere in the repo and carries its source in the text.
- **Preliminary is marked, everywhere.** The reconstruction chain has an in-situ
  calibration for one arm on two runs; every figure built on it carries a red
  badge, and the banner at the top says so in full. If that stops being true,
  the badges come off — not before.
- **The beam record quotes the total, never the blame.** The DAQ's logger splits
  downtime into PS and n_TOF; that split is unreliable and the report says so.

## Re-running the event census

```bash
scp ntof_run_report/count_events.py lxplus:x17count/
ssh lxplus "source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh; \
            cd ~/x17count && python3 count_events.py"
scp lxplus:x17count/events_per_subrun.csv ntof_run_report/data/
```

Needs LCG_105 for uproot (the system python on lxplus has none). Only ROOT
headers are read, so this is a metadata scan — it does not move the 18 TB.
