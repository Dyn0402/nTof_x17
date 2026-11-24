# Audit of rerun/paper1_gap — common brief for every test agent (2026-11-24)

You run ONE test of an independent audit. The orchestrator (another model) combines the tests. Read this file fully,
then your own task.

## Setup
- Repo: `/home/dylan/PycharmProjects/nTof_x17`. Python: `/home/dylan/PycharmProjects/nTof_x17/.venv/bin/python`
  (numpy, scipy, pandas, pyarrow, matplotlib). 16 cores, 62 GB RAM, shared with ~6 other agents: use <= 4 processes.
- The analysis under audit: `rerun/paper1_gap/` — read `README.md`, `gapmap.py`, `cellobs.py`, `joint.py`,
  `track_test.py`, `track_grid.py` FIRST (about 900 lines). Reuse their selections and helpers by importing them
  (`sys.path.insert(0, '<repo>/rerun/paper1_gap')`; `import gapmap as GM, cellobs as CO, joint as JT, track_test as TT`).
  Importing is safe; do not call their `main()` (they write into the repo).
- Data: `WORK=/media/ucla/work/claude-jobs/e1a7a26e/gap`
  - `$WORK/pass1/<run>/<subrun>/bow_gate_table.parquet` + `calib_bundle/bundle.json`: the two golden runs of chamber A
    (det3). `GM.RUNS` maps `run1` (6-27 Saturday, 7,143 rows) and `run2` (6-28 p2 overnight, 22,300 rows) to `rel`.
    `GM.load_table(WORK, rel)` applies the standard selection (single track, M3 chi2 < 1, 4 clusters, interior >= 30 mm
    from the active-area edge) and adds `xm, ym` (reference track at mid-gap).
  - `$WORK/pass1_all/<run>/<subrun>/<det>/bow_gate_table.parquet` (+ `calib_bundle/bundle.json`,
    `alignment/alignment.json`): the same table for all 14 gate rows of all five chambers (may still be downloading when
    you start; `$WORK/pass1_tables_listing.tsv` lists them).
  - `$WORK/cellobs.json`, `$WORK/gapmap.json`: the per-cell observables of the audited analysis (4x4 cells).
  - `$WORK/colsum/{sat,p2}_colsum.npz`: per-trigger column duration (see `GM.colsum_T`).
  - `$WORK/build_sat_det3/out/sat_det3/{incl,headon}.pkl`, `$WORK/build_g_det3_wknd/out/g_det3_wknd/{incl,headon}.pkl`:
    cached waveform windows of inclined (|tan| 0.2-0.8) and head-on tracks (see
    `sps_beam_test_26/analysis/cloud_basics/gap_extent.py` for how they are read).
- **Your outputs**: scripts ONLY as new files under `rerun/paper1_gap/audit/` named `<yourtag>_*.py`; everything else
  (json, png, logs, `RESULT.md`) under `$WORK/audit/<yourtag>/`. Never edit or delete an existing file. No git
  commands that change anything (no add/commit/checkout/stash). No ssh. Nothing outside those two places.
- The dataset is final. Never propose a new run, scan or hardware check. If something cannot be answered from the
  existing data, say so.

## The detector and the columns
- Chamber A is a 400 x 400 mm Micromegas micro-TPC, horizontal on a cosmic bench inside the M3 reference telescope.
  X strips measure x, Y strips measure y (pitch 0.78 mm). Drift gap about 28 mm, drift velocity about 37 um/ns, column
  duration about 750 ns. The readout board (with the mesh) is below, the cathode (a 50 um copper-clad kapton foil) is
  above. z is up.
- Per event and per view the waveform fit gives `x_p0` (track position where it crosses the mesh, mm), `x_w` (transverse
  rate in mm/ns: how fast the landing position of the charge moves with arrival time), `x_t0`, and the pass-1 angle
  `x_tan_theta` (which already contains a quadratic "bow block" correction). Same with `y_`.
- Reference: `tan_ref_x`, `tan_ref_y` (M3 slope dx/dz, dy/dz in the chamber frame), `ref_mesh_x`, `ref_mesh_y` (M3 track
  at the nominal mesh plane), `Chi2X/Y`, `NClusX/Y`. The chamber is rotated about 90 deg in the telescope (chamber x is
  along M3 y).
- For a straight track `1e3 * x_w = w0 + v * tan_ref_x` (um/ns): the "ladder". `v` (about 37) is the angle-scale
  velocity, `w0` the offset. Coordinates: `U = (x - 200)/100`, `W = (y - 200)/100` (units of 100 mm).

## The claim under audit
- Mesh surface h(x, y) and cathode surface c(x, y), heights with z up; gap g = c - h. Both quadratic in the audited fit.
- Drift lines leave the mesh normal to it and end normal to the cathode. In the long-wavelength limit the slope of a
  drift line at fractional height s (0 at the mesh, 1 at the cathode) is grad h + s (grad c - grad h). Averaged over the
  column the tilt is tau = (grad h + grad c)/2, and the fit sees it as an offset of the transverse rate:
  `w0_x = const + v tau_x`, `w0_y = const + v tau_y`.
- The position route: slope of (p0 - ref_mesh) against tan_ref in a cell = local mesh height h.
- Gap: geometric (charge extent of inclined tracks), column duration T ~ g^(1+beta), velocity v ~ g^(-beta).
- Tracking claim: a 2-D map of w0 (and v) in ~42 mm cells, made on one run and applied to the other, improves the angle
  resolution (s68 of theta - theta_ref) by 16-21 % over pass 1 (X 0.905 -> 0.736 deg, Y 0.796 -> 0.627 deg).

## How to work
- Be skeptical and quantitative. Give every headline number an uncertainty (bootstrap over events, or run 1 against
  run 2). Do not tune anything towards the expected answer. If a result contradicts the expectation in your task, say so
  plainly and show the numbers: a contradiction is as valuable as a confirmation. Report failed or inconclusive checks.
- Keep the two runs separate wherever a noise-independent comparison is possible (run 1 x run 2 cross-products are
  free of noise bias; within-run products are not).
- Figures: matplotlib (Agg), ordinary document style (about 3:2, 10 pt), PNG, axis labels with units.
- Finish with `$WORK/audit/<yourtag>/RESULT.md` (what was done, tables of numbers, verdict per question, caveats) and
  return a compact final message (at most ~60 lines): key numbers with errors, verdict per question, anything
  surprising, and the paths of RESULT.md, the scripts and the figures. No preamble, no restating of the task.
