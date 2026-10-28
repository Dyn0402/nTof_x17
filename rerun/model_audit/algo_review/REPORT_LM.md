# LM variable-projection fitter: implemented and A/B-tested, not promoted (2026-10-28, branch `lm-varpro`)

The algo_review recommendations (REPORT.md §"Recommendations" 2–3; HANDOFF "Fitter: LM variable-projection") were
built and tested on all five bench chambers. Everything is **opt-in**. With the defaults the reconstruction is
bit-identical: all 104 existing `wft/tests` pass unchanged.

## Verdict

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

**Recommendation:** keep `FITTER = 'nm'` for the B1 pass. The LM code stays on the branch as a study option.
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

Full tables: `results/lm_ab_tables.md` (nm / lm / lm_nohop; det3 also with free t0) and
`results/lm_ab_tables_grid.md` (nm / lm_grid).

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

- `lm_ab.py` — the A/B driver: `--free` adds the no-prior arms, `--grid` runs nm vs lm_grid only. Row pickles are in
  `/media/dylan/data/x17/cosmic_bench/model_audit/lm_ab/`.
- `lm_ab_summary.py <rows.pkl>...` → `results/lm_ab_summary.json` + markdown tables.
- `wft/tests/test_lm_fitter.py` — Jacobian vs FD, residual = χ², COV_ERRORS leaves the fit unchanged, LM vs NM.
- `.venv/bin/python rerun/model_audit/algo_review/lm_ab.py --dets det3 det2 det4 det6 det7 --n 200 --jobs 4 [--grid]`
