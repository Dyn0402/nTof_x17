# nTof_x17 — working notes for Claude

## Reconstruction basis — read this before writing any analysis

**Never reconstruct position, angle or drift depth from `combined_hits` times.**

On these resistive-strip detectors a per-strip hit time is an *aggregate* of the
strip's own charge and delayed, dispersed copies of its neighbours' (~29 % at
τ ≈ 47 ns to ±1 strip). It compresses the drift-time ladder by 20–30 %, reads
~4° too steep, and makes the reconstructed cluster fan away from the true track
with depth. This is estimator-independent — rising edge, CFD and matched filter
all show it — so no threshold change fixes it. Measured, with displays, in
`RECONSTRUCTION_BASIS.md`; decided 2026-07-28.

Hits are still the right input for **candidate finding** (which events/strips to
look at) and **QA** (rates, amplitudes, occupancy, detection efficiency).
Geometry comes from the waveforms (`decoded_root`) through the forward-model
reconstruction in `wft/`.

If an analysis you are about to write needs a position, an angle or a depth,
check the migration status table in `RECONSTRUCTION_BASIS.md` first.

## Layout

- `cosmic_bench_analysis/` — the original (hits-based) June cosmic analysis
  library. Still runs; being superseded topic by topic. `M3RefTracking.py` and
  the alignment/plot helpers are reference- and position-side and remain valid.
- `mx_june_cosmic_qa/` — June cosmic-bench QA scripts (01–47) and the run
  registry `qa_config.py`. Entry point: `MICROTPC_RUNBOOK.md`.
- `mx_june_cosmic_qa/waveform_first_threading/` — the waveform-first study that
  established the basis above; `WAVEFORM_FIRST_THREADING.md` is the report.
- `mx_july_beam_qa/`, `ntof_*` — July beam and n_TOF work.
- `common/` — strip maps, active area, shared config.
- Bench data: `/media/dylan/data/x17/cosmic_bench` (mirror at
  `~/x17/cosmic_bench`); waveforms in `<run>/<subrun>/decoded_root`.
- venv is `.venv` (`../../.venv/bin/python` from a subdirectory).

## Conventions

- The M3 reference recipe lives in `qa_config.py` (χ² < 1.0 and NClus = 4) —
  pass `M3_MIN_NCLUS` explicitly to every `M3RefTracking(...)` call; the class
  default is shared with other packages.
- Detector-local frame: x/y from the strip maps, z = drift depth from the mesh.
- Anything calibrated (kernel, template, v, gap map) is per detector **and** per
  run condition. A bundle used outside its conditions is a silent error.
