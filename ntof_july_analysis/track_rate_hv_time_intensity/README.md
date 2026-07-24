# track_rate_hv_time_intensity

Reusable scripts for the Micromegas per-trigger 2D-track rate, sliced by **HV scan
point**, **time since the gamma flash**, and **beam-pulse intensity**. First applied to
run_67 (2026-07-24).

Moved here from `~/beam_july/analysis/July_HV_Scan/run67_track_hv_time/` on 2026-07-24 so
the code is version-controlled; **analysis outputs (cache/, figures/) stay under
`~/beam_july/analysis/`** and are pointed at with `--cache`/`--out`.

## Scripts

```bash
V=~/PycharmProjects/nTof_x17/.venv/bin/python
C=~/beam_july/analysis/July_HV_Scan/<run>/cache
F=~/beam_july/analysis/July_HV_Scan/<run>/figures

$V build_cache.py     --run run_XX --cache $C          # tracking + denominator (incremental)
$V plots.py           --cache $C --out $F              # HV x time figures + best-HV table
$V intensity_split.py --run run_XX --cache $C --out $F # beam-intensity split (auto-refreshes)
```

Defaults (no flags) are `./cache` and `./figures` next to the scripts.

## Full method + rationale

See **`nTof_x17_DAQ/docs/METHOD_track_rate_vs_hv_time_intensity.md`** (the DAQ repo). It
covers the CNS prerequisite (get this wrong and every number is ~1000× off), the cut
definitions, the decoded-list denominator, the pulse_match beam-intensity split and its
burst-order alignment, per-detector reliability, and how to port to another run via
`SUB_PATTERNS` (the only run-specific edit).

## Dependencies

- `beam_track_finding.py`, `common.Mx17StripMap` — repo root (absolute path in the scripts).
- `pulse_match.py` — sibling `ntof_july_analysis/` (absolute path).
- `flash_timing_lib.py` — **external**, at `~/beam_july/analysis/flash_timing_threshold/`
  (added to `sys.path`). Not in this repo; keep that dir in place.
- Reads run data from `/mnt/data/x17/beam_july/runs/` and the beam-intensity CSVs under
  `/mnt/data/x17/beam_july/slow_control/beam_intensity/`.

## Results produced so far

- run_67: `~/beam_july/analysis/July_HV_Scan/run67_track_hv_time/` (README + figures)
- run_71 (other session): `~/beam_july/analysis/July_HV_Scan/run_71/` — uses `build_cache`
  via a thin `run71_build.py` driver (cycle-major order, Det A+C) + a `mesh_compare.py`.
