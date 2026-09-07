# slim_study — the measurements behind the slim

Findings: [`../SLIM_FEASIBILITY_2026-08-08.md`](../SLIM_FEASIBILITY_2026-08-08.md).
The pipeline these measurements justify is
[`../slim_pipeline/`](../slim_pipeline/README.md).

This directory is the *evidence*, not the product. Nothing here runs in
production; each script answers one question and writes its answer next to
itself, so the findings document can cite a number rather than a recollection.

## The scripts

| script | question | runtime |
|---|---|---|
| `window_envelope.py` | how wide is the match residual with only the GLOBAL clock? | seconds |
| `window_yield.py` | how many hits survive a ±W window? | ~4 min |
| `slim_closure.py` | does the full ±25 ns match still work after a ±W slim? | ~3 min |
| `slim_prototype.py` | what does a real slim file weigh, in bytes? | ~1 min |
| `coverage_map.py` | which DREAM sub-runs can be slimmed today, and what blocks the rest? | instant |
| `why_skipped.py` | is there anything different about the runs n_TOF's pass skipped? | instant |
| `make_handoff.py` | the request to n_TOF — markdown, HTML and CSV from one pass | instant |
| `handoff_html.py` | the standalone page `make_handoff.py` renders | — |

`window_yield.py --perbunch` repeats the scan on the fully-fitted clock (the
window the pipeline actually cuts on); adding `--shift-ns 100000` measures the
+100 µs accidental control.

## Outputs kept

```
window_yield_{narrow,wide,final,control}.json   the ±W scans
coverage_map_2026-08-08.txt                     per-DREAM-run coverage
missing_runs_2026-08-08.csv                     the 41 runs n_TOF still owes us
ntof_reprocessing_request.html                  the request, as a page
../NTOF_REPROCESSING_REQUEST_2026-08-08.md      the request, as markdown
coverage_inputs/                                the cached listings, below
```

The request is published at
<https://dylan-neff.web.cern.ch/notes/ntof-reprocessing-request.html>.

## coverage_inputs/

Five text listings, all cheap to regenerate and all cached so the analysis runs
offline. Exact commands are in `coverage_map.refresh_inputs()` and the
`why_skipped` / `make_handoff` docstrings.

| file | what | from |
|---|---|---|
| `ntof_index_times.txt` | first/last bunch wall-clock per processed run | `index` tree of every file in `processing/official/done/` |
| `ntof_raw_times.txt` | first/last mtime per staged run | `*/stream1/*_s1.raw*` under the DAQ tree |
| `raw_sizes.txt`, `out_sizes.txt` | bytes in / bytes out per run | `du` / `ls` |
| `dream_{eos,daq}_subruns.txt` | DREAM sub-run start + file count | the `datrun_YYMMDD_HHhMM` stamp in the first decoded file |
| `needed_runs_raw.txt` | file count and bytes for the requested runs | `ls` / `du` |

**Regenerate them after n_TOF processes anything**, then re-run `coverage_map.py`
and `make_handoff.py`.

## Two traps recorded in code, not in memory

- **The `index` tree's `Date`/`Time` are LOCAL (UTC+2), not UTC.**
  `ntof_io._index_epoch` builds an epoch from them as though they were UTC —
  harmless where it is used (relative matching inside one run), a flat +7200 s
  error the moment it is compared to anything else.
  `coverage_map.INDEX_LOCAL_SHIFT_S` corrects it, and the value is confirmed
  against the raw mtimes, which are true UTC: `raw_start − index_start` is a flat
  −7127 s over the 109 runs that have both. Before this was found, 13.2 % of beam
  time looked unrecoverable; it is 1.7 %.
- **Never hand-copy a run list.** Two were wrong in this study — the raw-gone
  gaps (two runs short) and the class of runs with no measurable window at all,
  which hid 224649/224650 from the request entirely. Everything is derived now.

## Reproduce

```bash
cd ntof_processing/slim_study
python window_envelope.py
python window_yield.py
python window_yield.py --out window_yield_wide.json \
       --windows 250 1000 2000 5000 20000
python window_yield.py --perbunch --out window_yield_final.json \
       --windows 25 50 75 100 150 250
python window_yield.py --perbunch --shift-ns 100000 \
       --out window_yield_control.json --windows 25 100 250
python slim_closure.py
python slim_prototype.py
python coverage_map.py [--verbose]
python why_skipped.py
python make_handoff.py
```

The first four read n_TOF 224572 `v12_liqpileup` through
`ntof_dream_merge/match_study/scripts/study_common.use_variant()` and take the
DREAM→n_TOF map from `ntof_dream_merge/calibration.py`. They do **not** re-derive
it — that map is per (DREAM run, n_TOF processing) pair and nothing transfers;
see `../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`. The pipeline, unlike
these scripts, re-fits it for every segment.
