# LM variable-projection fitter: gated and adopted (2026-10-28)

## Fleet gate `lm_r1` (lxplus, 15:09–17:29): PASS -> `FITTER = 'lm'`, `LM_HOPS = 0` is the algorithm

`fe_nm` (current main, grid + Nelder–Mead) vs `fe_lm` (`WFT_DIAG=model.FITTER=lm`), the same pinned-t0
`calib_bundle_fe` on the five golden rows, 8000 matched events each, code `code_5d4f5e4` on EOS, DAG
`~/x17rerun/fe/lm_r1.dag` (30/30 nodes, exit 0). Tables: `results/lm_gate.md` (gate_compare) and
`results/lm_gate_pairs.md` (per event, `lm_gate_pairs.py`).

| chamber | Δ detected 5 mm (pt) | Δ far 5 mm (pt) | s68 X / Y nm → lm (°) | same w X / Y | t0 moved > 30 ns | reco wall lm / nm |
|---|---|---|---|---|---|---|
| det2 | −0.09 | +0.09 | 1.161 / 1.214 → 1.161 / 1.217 | 98.7 / 98.5 % | ≤ 0.1 % | 0.70 |
| det3 | −0.06 | +0.06 | 1.082 / 0.966 → 1.084 / 0.967 | 98.8 / 98.3 % | ≤ 0.1 % | 0.42 |
| det4 | +0.13 | −0.04 | 2.509 / 2.601 → 2.509 / 2.593 | 93.1 / 93.6 % | ≤ 0.6 % | 0.86 |
| det6 | −0.05 | 0.00 | 1.469 / 1.619 → 1.471 / 1.625 | 97.2 / 97.3 % | ≤ 0.3 % | 0.88 |
| det7 | +0.16 | +0.01 | 2.499 / 1.580 → 2.508 / 1.588 | 91.1 / 95.3 % | ≤ 0.4 % | 0.81 |

- **Every aggregate is unchanged within noise.** Detection moves by ≤ 0.16 pt, far by ≤ 0.09 pt, s68 by ≤ 0.009°,
  and the bias by ≤ 0.009°. The median χ² ratio is 1.0000 on every view.
- **Per event**, 1.4–1.7 % of det2/det3 planes end in a different well (dθ > 1°). The figure is 3–5 % on det6 and
  det7 Y, and 7–11 % on det4 and det7 X, the chambers with the known missing-charge faults (det4 dead stripes, det7 X
  M1 seating), where the χ² surface has several near-equal wells. These differences are symmetric: they move neither
  s68 nor far. For scale, switching band censoring off (`fe` vs `fe_nb`) moves 0.2–0.9 %.
- **Cost on condor: 0.77× the reconstruction wall time** summed over the five rows (4 h 15 min vs 5 h 31 min). That is
  less than the fitter-only 0.45×, because seeding, I/O, candidate selection and the two-track fit are unchanged.
- The covariance w_err is ×1.8–2.3; tan_err is ×1.001–1.004 (the floors dominate).

**Changes after the gate** (in the merged code, found by the full test suite with LM as the default; the gate ran
the version before them):
1. A non-finite LM step or Jacobian is rejected. This was a NaN t0 reaching `build_matrix` on the delay kernel in
   the two-track tests.
2. t0 stays within ±`LM_T0_RANGE` = 180 ns of the start. With free t0, an unbounded LM followed the
   profile-absorbs-a-time-offset valley 250 ns out on a synthetic pair.
3. The polish is tighter (xatol 1e-4, fatol 0.02, ≤ 80 iterations instead of 5e-4 / 0.15 / 40). LM then lands on
   NM's minimum to the printed digit on that pair, and the two-track children are identical.

All three move LM toward NM, and (2) is inactive under the bench pin. They were checked locally on the pinned FE
bundles, all five chambers × 100 events (`/media/dylan/data/x17/cosmic_bench/model_audit/lm_ab/polish2/`): same w on
96–100 % of planes, s68 and far within noise, and 0.59–0.68× the solves. They were **not** re-gated on condor.

---

## Earlier: implemented and A/B-tested (2026-10-28, branch `lm-varpro`)

## Update, same day: with t0 pinned (the FE bench bundles) LM is a drop-in, at 0.45x the cost

The first pass below used the vladder bundles, which carry the old σ = 5 ns prior and a stale `t0_abs`, so t0 was
effectively free. The bench production configuration is the FE one (FOLLOWUPS item 7): `set_w0kw --pin-t0` sets
`t0_abs` from the raw-waveform trigger latency and σ = 0.05 ns, so the prior is a pin. Rerun on the FE bundles
(`fe_unify2/bundles/<det>/calib_bundle_fe`, same runs as the vladder caches, 200 events per chamber):

| chamber·view | same w (nohop) | s68 nm → lm_nohop (°) | far5 nm → nohop (%) | position s68 vs M3 nm → nohop (mm) | t0 moved | solves / time vs nm |
|---|---|---|---|---|---|---|
| det2 X | 98.5 % | 1.255 → 1.260 | 6.0 → 5.5 | 0.586 → 0.583 | 0 % | 0.44 / 0.44 |
| det2 Y | 99.5 % | 1.082 → 1.074 | 2.5 → 2.5 | 0.535 → 0.538 | 0 % | 0.45 / 0.45 |
| det3 X | 99.5 % | 1.075 → 1.082 | 4.0 → 4.0 | 0.631 → 0.631 | 0 % | 0.43 / 0.41 |
| det3 Y | 98.5 % | 0.898 → 0.898 | 5.5 → 5.5 | 0.540 → 0.540 | 0 % | 0.47 / 0.44 |
| det4 X | 99.0 % | 2.460 → 2.513 | 16.0 → 14.5 | 0.768 → 0.768 | 0 % | 0.45 / 0.47 |
| det4 Y | 96.5 % | 2.297 → 2.290 | 13.0 → 13.0 | 0.672 → 0.672 | 0 % | 0.45 / 0.50 |
| det6 X | 97.5 % | 1.437 → 1.462 | 8.0 → 7.0 | 0.528 → 0.531 | 0 % | 0.43 / 0.46 |
| det6 Y | 98.5 % | 1.568 → 1.563 | 4.5 → 4.0 | 0.492 → 0.492 | 0 % | 0.43 / 0.45 |
| det7 X | 94.0 % | 1.847 → 1.834 | 14.6 → 14.6 | 0.916 → 0.918 | 0 % | 0.44 / 0.44 |
| det7 Y | 97.0 % | 1.396 → 1.379 | 8.0 → 7.5 | 0.754 → 0.763 | 0 % | 0.44 / 0.46 |

- **No t0 relabelling on any plane, and positions are unchanged** (≤ 0.01 mm in s68; p0 differs by > 0.3 mm on 1–10
  of 200 planes). Angle s68 is within ±2.2 % on every row, and the far fraction within ±1.5 pp. The median |dθ| is
  ≤ 0.001°.
- **Cost: 0.43–0.47× the NNLS solves and 0.41–0.50× the wall time**, i.e. a ~2.2× faster bench reco.
- **Use `LM_HOPS = 0` with the pin.** Hops find nothing once t0 is pinned, cost 20 %, and on det7 X they read worse
  (s68 1.847 → 1.893, far 14.6 → 16.1 %).
- The planes where the two fitters end in different wells (dθ > 1°: 1–3.5 %, det7 6–7 %) cancel in s68 and far.
  det7 X is the chamber with the known readout faults.
- Errors: the covariance gives w_err ×2.1–2.7. t0_err is unchanged (the pin sets it), and tan_err/p0_err are unchanged.

**Revised recommendation.** On the bench with pinned-t0 bundles, `FITTER = 'lm'`, `LM_HOPS = 0` reproduces the
current reconstruction within statistics and is ~2.2× faster. It is a candidate for the pass once the fleet gate
(`make_gate_dag.py`, golden rows, promoted bundles) confirms it. **n_TOF is different**: t0 is never pinned there, and
the free-t0 result below still holds. Any outer solver, LM or NM, picks the 60 ns basin by its search path on 13–33 %
of det3 planes, with the same angle but p0 moved by 60·w. A single algorithm for both would make the beam t0/p0 labels
differ from today's NM on that fraction, without either being shown to be better. Whether that matters depends on what
the beam analysis takes from t0/p0 (the x–y pairing and the column-time gate use them). It is Dylan's call; LM
everywhere is defensible, since nothing at n_TOF has been quoted yet.

The sections below are the first (free / inert-prior t0) pass and stand as the n_TOF-like result.

---


The algo_review recommendations (REPORT.md §"Recommendations" 2–3; HANDOFF "Fitter: LM variable-projection") were
built and tested on all five bench chambers. Everything is **opt-in**. With the defaults the reconstruction is
bit-identical: all 104 existing `wft/tests` pass unchanged.

## Verdict (first pass: vladder bundles, t0 effectively free)

- **Angles: a safe drop-in.** On every chamber and view, w agrees within 1 µm/ns on 93–100 % of planes. s68 against M3
  moves by ≤ 3.5 % (inside the 3–5 % statistical error of 200–300 events), and the far fraction is unchanged within
  1.5 pp.
- **Cost: 0.55–0.65× the NNLS solves and ~0.6× the wall time** (ladder start + hops), or 0.75× when started from
  the NM grid. That is not the 5× of the review: the review counted only the refinement, but the pivot scan,
  the global start and the polish are shared costs.
- **t0 / p0: NOT minimal-change, and no LM variant can be.** The χ² surface has a basin at every
  (p0 + 60 w, w, t0 + 60 ns) lattice step. Which basin a fit ends in is decided by the search path. Any change of
  outer solver relabels t0 by one depth bin, and p0 by 60·w along the same line, on **6–48 % of planes** (more on
  det6/det7 and on X). This holds even when LM starts from NM's own grid point, because NM's simplex (20 ns t0 step)
  itself hops basins during refinement. The angle is the same in every basin.
- **The deeper-χ² basin is not the more physical one.** With lattice hops, LM ends at lower χ² than NM on 14–52 %
  of planes and higher on 0–10 %. Yet where it moves t0, it is closer to the trigger-predicted t0 only 18–56 % of the
  time (constant offset removed). It is closer to the M3 mesh point 29–61 % of the time, and position s68 against M3
  gets worse on det6/det7/det2-Y (det7-X 1.30 → 1.49 mm). Where the grid-start variant differs from NM, NM is
  usually the one nearer M3.
- **The bench t0 prior does not pin the basin.** At σ = 5 ns a one-bin hop costs 144 χ² units, against data χ² of
  1e4–4e5 (χ²/dof 13–330). NM's own fits sit a median 40–60 ns from the bundle's trigger prediction, and > 30 ns away
  on 60–73 % of planes, on these vladder bundles. This is the real lever, and it belongs to the model, not the
  optimiser.

**Recommendation (first pass, superseded on the bench by the update above):** keep `FITTER = 'nm'` for the B1 pass. The LM code stays on the branch as a study option.
Adopting it would cost re-validating positions for a ~40 % speed-up. The 60 ns basin choice is what actually limits
t0/p0, and it needs a t0 constraint that carries weight (e.g. the prior scaled by χ²/dof, or the bench t0 pinned
outright). That is a model decision for Dylan, out of scope here. **`COV_ERRORS`** (below) is the one piece that
changes nothing in the fit and could go in as is, if anything downstream ever uses w_err or t0_err.

## What was built (`wft/model.py`, `wft/reco.py`)

| switch (wft.diag) | default | what |
|---|---|---|
| `model.FITTER` | `'nm'` | `'lm'`: Levenberg–Marquardt (Marquardt scaling) on the NNLS-profiled residual. Kaufman Jacobian: d(Mq)/dθ at fixed q, projected off the free weighted columns over the live + active-clip rows. The SAT_SOLVE / penalty / ZS_BOUND / t0-prior rows are all in the residual, and r·r reproduces `_solve_nnls` χ² exactly. Finished with a 40-iteration Nelder–Mead polish. Returns `cov`. |
| `model.LM_START` | `'ladder'` | `'ladder'`: 7 t0 points × {start p0, pivot line} (13 solves). `'grid'`: NM's 105-point grid, point for point. |
| `model.LM_HOPS` | `3` | after LM, restart one bin over (p0 ± 60w, t0 ± 60) and keep hopping while χ² falls, up to 3 bins each way |
| `reco.COV_ERRORS` | `False` | errors from (JᵀJ)⁻¹·max(χ²/dof, 1) at the minimum of *any* fitter (`model.vp_covariance`). The fit is bit-identical. |

The derivatives are analytic on the RC kernel (`_rc_dmq`): Gaussian, pseudo-Voigt, one-sided copy, RC loop and
drain, plus the t0 derivative of the 5 ns tensor interpolation. They match central differences to 2e-8 on all
branches. The delay kernel falls back to finite differences. `fit_plane_raw(..., pivot=(x_c, t_c))` is a new
optional argument that `_pivot_fit` now passes, and NM ignores it. `fix_p0w` (calibration) and the two-track fit are
untouched.

## Measurements (vladder bench caches with M3 reference; det3 `bundle_phys_mask`, others `bundle_phys`; t0 prior from each bundle where it has `t0_abs`, i.e. det2/3/4; det6/7 free)

Full tables: `results/lm_ab_tables.md` (nm / lm / lm_nohop; det3 also with free t0),
`results/lm_ab_tables_grid.md` (nm / lm_grid) and `results/lm_ab_tables_fe.md` (pinned-t0 FE bundles, the update).

| chamber·view | same w (lm / grid) | s68 nm → lm → grid (°) | t0 relabelled > 30 ns (lm / grid) | solves lm / grid vs nm | position s68 vs M3 nm → lm → grid (mm) |
|---|---|---|---|---|---|
| det2 X | 97.0 / 97.5 % | 1.354 → 1.398 → 1.338 | 28 / 15 % | 0.54 / 0.78 | 0.67 → 0.62 → 0.68 |
| det2 Y | 99.0 / 100 % | 1.248 → 1.247 → 1.248 | 14 / 12 % | 0.64 / 0.70 | 0.55 → 0.58 → 0.56 |
| det3 X | 99.0 / 100 % | 1.017 → 1.032 · 1.033 → 1.055 | 33 / 17 % | 0.52 / 0.77 | 0.76 → 0.67 · 0.80 → 0.85 |
| det3 Y | 98.3 / 99.5 % | 0.996 → 0.994 · 0.978 → 0.980 | 12 / 7 % | 0.63 / 0.75 | 0.55 → 0.54 · 0.59 → 0.57 |
| det4 X | 93.5 / 99.0 % | 2.547 → 2.610 → 2.551 | 27 / 10 % | 0.52 / 0.78 | 0.90 → 0.90 → 0.90 |
| det4 Y | 94.5 / 98.5 % | 2.121 → 2.109 → 2.125 | 20 / 14 % | 0.51 / 0.80 | 0.81 → 0.77 → 0.78 |
| det6 X | 97.5 / 98.5 % | 1.763 → 1.789 → 1.808 | 43 / 22 % | 0.54 / 0.77 | 0.56 → 0.60 → 0.56 |
| det6 Y | 98.0 / 99.0 % | 2.067 → 2.068 → 2.057 | 38 / 25 % | 0.52 / 0.79 | 0.59 → 0.67 → 0.68 |
| det7 X | 94.0 / 98.5 % | 2.096 → 2.089 → 2.093 | 48 / 30 % | 0.57 / 0.76 | 1.30 → 1.49 → 1.33 |
| det7 Y | 97.0 / 98.5 % | 1.541 → 1.512 → 1.500 | 40 / 29 % | 0.56 / 0.76 | 0.91 → 1.07 → 1.03 |

(det3: the lm columns come from the 300-event run, the grid columns from a separate 200-event run, hence two NM
baselines.) Free t0 on det3, the beam situation: identical picture, s68 X 1.029 → 1.033, Y 0.997 → 0.992, t0
relabelled on 33 / 13 %.

**Errors.** The covariance gives w_err and t0_err 1.9–2.9× the 1-D curvature values on every chamber (det3 X: w
0.015 → 0.037 µm/ns, t0 1.4 → 3.3 ns). p0_err (0.330 → 0.331 mm) and tan_err (0.0180 → 0.0181) barely move,
because `FLOOR_P0_MM` and `tan_floor` dominate them.

## Limits

These are the vladder bundles, not the promoted `final_phys` ones (on lxplus EOS). Each chamber had 200–300 events
on a loaded laptop, so wall times are relative only. No fleet gate (`make_gate_dag.py`) has been run. The M3
position referee uses a linear ref→p0 map fitted per plane on the NM arm. The trigger referee uses the bundle's
`t0_abs` with the median offset removed. det6/det7 have no `t0_abs`, so they have no trigger referee and were
fitted with free t0.

## Files and commands

- `lm_ab.py` — the A/B driver: `--fe` uses the pinned-t0 FE bundles, `--free` adds the no-prior arms, `--grid` runs nm vs lm_grid only. Row pickles are in
  `/media/dylan/data/x17/cosmic_bench/model_audit/lm_ab/`.
- `lm_ab_summary.py <rows.pkl>...` → `results/lm_ab_summary.json` + markdown tables.
- `wft/tests/test_lm_fitter.py` — Jacobian vs FD, residual = χ², COV_ERRORS leaves the fit unchanged, LM vs NM.
- `.venv/bin/python rerun/model_audit/algo_review/lm_ab.py --dets det3 det2 det4 det6 det7 --n 200 --jobs 4 [--grid]`
