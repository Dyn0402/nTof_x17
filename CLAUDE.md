# nTof_x17 — working notes for Claude

## THE DATASET IS FINAL — we no longer have access to the detectors

**No more data can be taken: not on the cosmic bench, not at SPS, not at n_TOF.** The chambers are
gone (Dylan, 2026-10-16, after asking several times). Every open question must be answered with the
data we already have, plus simulation (Magboltz, Garfield, Geant4) and the literature. **Never suggest
a new run, a dry-gas run, a new scan, a re-measurement or a hardware check**, not in chat, not in a
handoff, not in a report's "next steps". If a question cannot be answered from existing data, say
so and write it down as a stated limitation.

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

## Coordinates: +y is up (stage-3 y was mirrored 2026-09-07 → 10-13; fixed and regenerated 10-14)

**+y is the beam direction, up.** The y view keeps its position and flips its tan (`COORDINATES.md`). Every live
stage-3 set was regenerated on 2026-10-14 and carries `y_convention: "up"` in its sidecar; **the superseded
`stage3_is2_v1_pilot` was not and is still mirrored**, as is any table without the flag. Read stage 3 through
`build_tracks.y_up(df, path)`, which handles both. What moved: `Y_FLIP_AUDIT.md`. The capsule source (band
crossing) is at −31 (A) / −35 (C) mm, the nose, where Geant4's capsule puts it (−28 by the same estimator).

**Never decide an axis sign from a self-consistency test.** corr(position, tan), A–C lines and band slopes are all
blind to a global mirror. Pin every orientation to an external fact, such as cosmic muons going down or a PMT
position, and write that fact next to the constant. Frame, conventions and the evidence: `COORDINATES.md`.

## Layout

- `cosmic_bench_analysis/` — the original (hits-based) June cosmic analysis
  library. Still runs; being superseded topic by topic. `M3RefTracking.py` and
  the alignment/plot helpers are reference- and position-side and remain valid.
- `mx_june_cosmic_qa/` — June cosmic-bench QA scripts (01–47) and the run
  registry `qa_config.py`. Entry point: `MICROTPC_RUNBOOK.md`.
- `mx_june_cosmic_qa/waveform_first_threading/` — the waveform-first study that
  established the basis above; `WAVEFORM_FIRST_THREADING.md` is the report.
- `mx_july_beam_qa/`, `ntof_*` — July beam and n_TOF work. **Our n_TOF data
  taking ended the morning of 2026-08-10** — the dataset is final (last run with
  beam 224715). n_TOF keeps running *other* experiments, so new runs on the DAQ
  page and under `/eos/experiment/ntof/` are not ours: identify ours by **run
  title**, never by recency or run-number proximity.
- `ntof_cosmics/` — the 47 beam-off cosmic runs (inventory) and the
  through-going-particle background; `clock_match.py` puts beam-off DREAM
  triggers on the n_TOF clock (~10 ns) with no flash. Entry point: its README.
- **Same-chamber pair reconstruction**: two-track separation (T1) and single-track
  angle truth / the seeder (T2), both on main since 2026-10-12. Map of both, how they
  constrain each other, and the latest results on the is2_v2 re-pass:
  `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`.
- `papers/` — the two MX17 companion papers (LaTeX), synced to Overleaf by `papers/overleaf.sh`;
  figures come from `papers/<paper>/figures.toml` via `papers/make_figures.py`. Its README is the workflow.
- `ntof_scint_stack/` — every SiPM wall, plastic and liquid mapped (efficiency,
  response) by the MM tracks that point at them; output in
  `/media/dylan/data/x17/scint_stack/`. Its README has the track-extrapolation
  calibration (note: stage-3 `tanx = k · tan_raw`, not `/ k`).
- `ntof_pedestal_qa/` — the DREAM pedestal history of the n_TOF campaign, per
  channel and per chip. Two of its findings are campaign-wide and are listed
  under "Two n_TOF conditions" below; read them before comparing anything
  across 23 July or using chamber A in run_79.
- `common/` — strip maps, active area, shared config.
- Bench data: `/media/dylan/data/x17/cosmic_bench` (mirror at
  `~/x17/cosmic_bench`); waveforms in `<run>/<subrun>/decoded_root`.
- venv is `.venv` (`../../.venv/bin/python` from a subdirectory).

## Two n_TOF conditions that will silently bias an analysis

Both were measured from the pedestal runs and confirmed against the data;
`ntof_pedestal_qa/README.md` has the evidence and the caveats.

**The noise floor doubled on 23 July and never came back.** The DREAM readout
clock divider went 6 → 4 (`RdClk_Div`, sampling clock unchanged) and the
per-channel residual noise rose **×2.0 on all eight FEUs at once** — so did the
5σ threshold the DAQ loaded. This splits the campaign in two, and **the entire
production period is on the noisy side**. The bracketing pedestals are 23 July
10:05 (old) and 16:47 (new), so:

| | |
|---|---|
| **run_67 and earlier** | old, quiet configuration (run_67 ends 09:52) |
| **run_68** | 14:54–16:08, *inside* the bracket — not placed by the pedestals. It loaded the 10:04 pedestal, which suggests the old configuration, but that is an inference, not a measurement. Exclude it or check it directly. |
| **run_69 onward** | new, noisy configuration (run_69 starts 17:27) |

Never pool amplitude, threshold, efficiency or hit-rate numbers across that
boundary without saying so. The coherent common mode did *not* change, so a
single overall gain factor does not undo it.

**Chamber A x-view connector 8 was dead through run_79.** Channels 448–511 of
FEU 3 were electrically disconnected from 22 July to the 27 July access — every
sub-run of run_79, the first long production run. 41 of those 64 channels
recorded no hits at all. Any A-x occupancy, efficiency or cluster study on
run_79 must mask them; the strips are live again from run_83.

## Reporting results — write an HTML report, not just PNGs

Every analysis should ship a `report.html` in its output directory alongside
the figures. The DAQ web page serves them: its **Analysis tab lists any
`.html`** in an analysis directory and opens it inline, so results are readable
from the control-room browser with no file-system access.

- **Generate it, don't hand-write it** — a `make_report.py` in the analysis
  package that builds the HTML from the tables, so re-running after the
  analysis updates numbers, charts and verdict text together. Reference model:
  `ntof_july_analysis/leadshield_compare/make_report.py`.
- Reference figures with **relative links** (`figures/x.png`) — the DAQ's
  `/analysis_file/<relpath>` route is path-based, so the same file works from
  disk, from the web page, or copied elsewhere.
- Lead with the answer: verdict → headline numbers → what was compared → the
  tables → figures with captions → **what the result does not rule out**.

## Conventions

- The M3 reference recipe lives in `qa_config.py` (χ² < 1.0 and NClus = 4) —
  pass `M3_MIN_NCLUS` explicitly to every `M3RefTracking(...)` call; the class
  default is shared with other packages.
- Detector-local frame: x/y from the strip maps, z = drift depth from the mesh.
- **The expected IPC spectrum comes from `sept26_prelim_analysis/ipc_born.py`,
  not from `pair_physics.py` and not from `IPC/capture = 2.1e-3`.** That
  2.1e-3 (the rate tables, the INTC proposal, `mpgd26/make_x17_rate.py`) is
  Viviani et al.'s ratio at E_n = 0.17–2 MeV, where a p-wave 1⁻ resonance
  dominates. Below ~2 eV — which is every neutron after the 1 ms flash veto —
  that resonance is gone and only two s-wave channels are open: 1⁺(³S₁)→0⁺ M1,
  which is the measured 55 μb (n,γ), and 0⁺(¹S₀)→0⁺ E0, which **emits no real
  photon at all** and so is not bounded by the (n,γ) cross section. The E0
  continuum is far broader in opening angle (11 % above 109° against M1's
  4.6 %), so the mix changes the X17-region background, not just its
  normalisation. Full argument and validation:
  <https://dylan-neff.web.cern.ch/x17/ipc-continuum/>.
- **Take positions from pointing, not from the shape of a distribution.** A fit
  to the shape of a track-position distribution (an acceptance ray trace, a
  profile fit) is dominated by each chamber's own efficiency along that axis,
  dead strips and noisy columns; a pointing estimator (the band crossing: the
  correlation of angle with position) is not. Measured 2026-09-14 on the
  capsule height: per 100 % efficiency tilt, 0.3–3.7 mm for the band crossing
  against 32–65 mm for the ray trace. So chamber-to-chamber differences in a
  shape fit are not alignment, and when two such estimators are combined only
  the part common to all chambers means anything. Evidence and the mistake it
  came from: `capsule_y.py --sensitivity` and the Outcome box of
  `sept26_prelim_analysis/HANDOFF_CAPSULE_Y.md`.
- Anything calibrated (kernel, template, v, gap map) is per detector **and** per
  run condition. A bundle used outside its conditions is a silent error.
- **Dead channels: one treatment everywhere** (closed 2026-10-26). A bundle's `dead` list (census dead + shorts,
  `rerun/channel_layer/`) is censored in the fit (unknown charge, not zero), and the seeder bridges dead strips and pads
  windows by live strips (`WFT_DEAD_SEED`, default on, bench and beam). All five bench chambers carry their mask. Don't
  re-gate a mask on χ²/dof: censoring lowers the dof more than the χ². Evidence: `rerun/channel_layer/README.md`.
- **The sharing kernel must have c2 < c1**, always: the ±2 strip is reached only
  through the ±1. Every bundle shipped before 2026-08-21 had it backwards (det3
  1.14, det2 1.53, det7 1.75, det4 2.12) because the ref-pinned cosmic χ² is
  flat in that direction. `wft.calib.check_kernel_ordering` now refuses such a
  bundle at load, save and install, and every product built on one was retired
  and re-reconstructed — record `mx_june_wft/RETIRE_C2GTC1_2026-08-21.md`. The
  copy-amplitude kernels (r06 and before) are themselves superseded: the pass
  uses the physical RC kernel (`build_matrix_rc`, no fitted copy amplitudes;
  `rerun/RERUN_PLAN.md` §0, the `fe` recipe in `rerun/REPROCESSING_TODO.md`).
  On an old r06 bundle the stored `c2` is literally `0.0` and the ratio lives in
  `c2_over_c1`: read it with `wft.calib.effective_c2`, never `hyper['c2']`.
- **History of the analysis and what each lesson requires of the chain**:
  `rerun/history/LESSONS.md` (generated from `rerun/history/timeline_data.py`).
  Check it before re-deciding anything. The pass plan is `rerun/PASS_PLAN_2026-10-28.md`.
- **The from-scratch pass is done (2026-10-30).** All bench and SPS products are in EOS `Analysis_pass1` (code `6a0bb1f`),
  calibrated from nothing; all five chambers pass G1 against the audit. Analysis reads those tables, not Analysis_rr1 or the
  audit bundles. Status, findings and exclusions: `rerun/PASS1_STATUS_2026-10-30.md`; per-row ledger `rerun/PASS1_LEDGER.md`.
