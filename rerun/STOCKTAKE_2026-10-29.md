# Where the reconstruction stands before the from-scratch pass (stock-take, 2026-10-29 night)

The pass = recalibrate every chamber from nothing in a DAG (`PASS_PLAN_2026-10-28.md`), then reprocess all bench rows
(159) and SPS rows (71). The checklist is `REPROCESSING_TODO.md`; today's record is `calib/SESSION_2026-10-29.md`.

## 1 · The algorithm: settled (in the code on main)
- **Fit:** forward waveform model (physical RC kernel, X template for both views, σ0 0.30 mm, Dp, D_rc,y), variable-
  projection LM fitter, pinned t0 on the bench (t0 = C − 10·ftst, no δ), PIVOT start, censored dead/band strips,
  two-track fit (4-pitch minimum), spark veto > 50 seeding hits (det4 confirmed today).
- **Calibration content:** templates `measure_templates` (no stretch, no extra shape parameter: T15/T16), D_rc,y by
  `recal.py --t0-pin --unsat-only` (det4 2.2e-4 from H4), w0/kw per row, own v per row with ≥ 500 clean triggers.
- **Bow:** one tan-form block per chamber from its golden row, no per-mount/per-run bow; angles through `tan_bow`,
  positions through `wft.compat.mesh_position` (p0 − tan·h_view). Validated on every dataset of every chamber.
- **Selection/QA:** χ²/dof flag 5 × median; x–y pairing by charge + column time per event class; no gain map.

## 2 · The error-model recipe: gated and NOT adopted; the simple model LOCKED (Dylan, 2026-10-29 evening)
**Decision:** no cap, no per-event width, no shape term, no variance model, no rails. Cost ≤ 0.05° on det3, nothing
in charge or efficiency; nine known simplifications carried as stated limitations:
`calib/saturation/DECISION_2026-10-29.md`. §7 items 1 and 3 are closed by it (rails off; mean model after the pass,
if ever, with a physical mechanism). The gate record that led to it:
The full-reconstruction gate (8000 events × five golden rows) and det3's other datasets failed it
(`calib/saturation/ERROR_MODEL.md`, Decision):
- **The cap costs efficiency on det6 (−0.7…−0.8 pt) and det7 (−1.4…−1.6 pt).** With every bright sample only a lower
  bound, a heavily clipped near-vertical column fits as well as a steep line, and the fit runs away (det6: 85 planes
  from |tan| < 0.6 to > 1). The held-out harness could not see it (windows follow M3, t0 pinned). Two-track splits
  fall, far +0.1…+0.5 pt everywhere (the relative χ² flag then passes them).
- **The strip nuisance changes sign with drift field on det3** (500 V: +0.20° alone, recipe +0.08°).
- The cap alone wins on all 14 held-out datasets, so the angle gain is real; the next form to try keeps the uncapped
  fit's solution and lets the cap only refine it (cannot run away). Not tested.
- **Per-sample rails** are now carried by `wft.io` (opt-in, bit-identical when off). In full reco they are efficiency-
  and angle-neutral and tighten det6 x-y charge pairing (power 0.816 → 0.865). Neutral on det3 → a correctness fix,
  Dylan's call.
So the pass runs with the current treatment unless one of these is decided before the archive.

## 3 · Engineering still to build (blocking, ~4–5 h + dry run)
| # | item | state |
|---|---|---|
| E1 | `rerun/calib/make_bundle.py` + `physics_inputs.json` (σ0, Dp, det4 D_rc,y, gaps, timing v, provenance) | not started |
| E2 | bow node `rerun/calib/bow_node.py` | **done** (reproduces the study block) |
| E3 | `make_pass_dag.py calib` phase CAL0–CAL4 into `Analysis_pass1`: TMPL, DRC (`--t0-pin --unsat-only`, disjoint training), MAKE, RECO/ALIGN/ANG, W0KW (`--table` everywhere), BOW2 + W0KW2b, convergence → `CALIB_CONVERGENCE.json`; the bow transfer gate (`calib/bow/bow_finalize.py --manifest`) after the pass reco | not started (spec in PASS_PLAN §2) |
| E4 | SPS phase (new det4 bundle, bench bow/v_map/w0/t0 dropped, quality_flags + timestamp in `sps_reco`, K +15) | partly (drops done 10-28) |
| — | if the recipe passes: `SAT_RAIL` needs pedestal + CM carried into the windows (`wft.io` / cache builder), or adopt without it | with the gate session |

## 4 · Pre-flight (PASS_PLAN F7–F12)
- F7 the 14 GA channels: flank test + waveform display each (**blocks E1**, the masks go into the bundles).
- F8 H4 ch 372 (SPS only). F9 10 rows without gas-time entries. F10 det4 6-23 M3 clean fraction. F11 g_det4 seeding
  on FEU 6 conn 7–8. F12 (optional) decode the five never-decoded det6/det7 6-26 HV points (S0b).
- Then: `pytest wft/tests`, dry run `calib` det3 `--limit 300` + one vrefit + one pass row, `lxstore status`
  (offload old `code_*`), code archive, `make_registry.py`.

## 5 · Submission (PASS_PLAN §3)
S1 `calib` (~5 × 20 nodes, 4–6 h) → **G1** (convergence + held-out vs the audit numbers; also re-checks the LM
post-gate fixes) → S1b `vrefit --keep-v` (117 rows, ~2 h) → S2+S3 `pass` + `sps` (overnight) → completion ledger.

## 6 · What the algorithm still does not do (limitations or after the pass)
- **Resolution gap.** det3 ~1.0° (bench 1.0–1.8° fleet) against a 0.30–0.35° floor for a perfect model. Explained
  so far: the bow (−0.07…−0.56°, in), bright samples (recipe, pending). Remaining: the mean model at bright samples
  (centre strip under-filled 5–11 %; once right, a Poisson variance should gain another 7–10 %, toy), and a
  χ²-blind 0.7–1.4° that the audit (FD) attributes to fine-scale geometry. Neither blocks the pass; both are model
  work that would need a second pass.
- F4 residual angle nonlinearity (~2 %, det4 Y 7 %); F21 bright tracks read 5–10 % shallow on det6/det7 (worth
  checking on the recipe's gate output: it acts on bright samples).
- Y RC branch cannot carry footprint tails (code); det6 Y RC model under watch.
- 60 ns charge grid biases steep tracks (−0.5° at 40°): negligible on the bench (≤ 24°), the main term for n_TOF pairs
  at 30–45° (out of the pass's scope; 20 ns needs a pinned t0, which n_TOF lacks).
- Two-track fit still Nelder–Mead (speed only). n_TOF: χ²-flag purity unmeasured, free t0.
- Downstream after the pass: v(t) per time block, SPS gas fit, loss budget, re-quoted resolutions, figures.

## 7 · Decisions for Dylan
1. ~~Rails / cap refinement~~ **decided 10-29 evening: neither; simple model locked (§2).**
2. S0b: decode the five never-decoded det6/det7 HV points before the pass, or list them as excluded. **Open.**
3. ~~Bright-sample mean model before or after~~ **decided: not before the pass; a second pass only with a physical
   mechanism (DECISION_2026-10-29.md, last section).**
