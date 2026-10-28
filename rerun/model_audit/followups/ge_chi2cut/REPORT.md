# GE — Has the fixed χ²/dof cut been useful? (FOLLOWUPS round 3, item 8, 2026-10-28)

## Verdict: replace the fixed number with one relative rule
**Rule:** `quality_ok = χ²/dof ≤ 5 × median(χ²/dof)` of the same view, over the same reco table (one chamber × one run condition).

- **χ²/dof as a ranking is trustworthy** where bad reconstructions add charge (det2, det3, det6, det7 Y): AUC 0.78–0.96
  for bad vs good views. That beats brightness alone (0.67–0.90) and every charge- or model-error-normalised χ² tried
  (0.60–0.86).
- **The fixed 300 is what can't be trusted.** Model error grows with signal (χ²/dof ∝ q^1.1–2.1), so the median χ²/dof
  ranges from 4 to 741 across tables:

| table | median χ²/dof | what 300 does |
|---|---|---|
| det4 bench | 10–19 | |
| det3 bench | 68–71 | 4.4 × median, the chamber 300 was tuned on |
| det6 bench | 509–741 | removes 70–81 % of good muons |
| n_TOF A, run_100 | 4–6 | 50–70 × median, removes 0.6–0.7 % |
| n_TOF C, run_100 | 11 | removes 0.8–1.6 % |
| n_TOF D | ~250 | |

- **The relative rule replaces a per-chamber `CHI2DOF_BAD`** (HANDOFF next step 4).
- **det4 and det7 X:** χ² is blind there (AUC 0.38–0.49). Their bad views are *missing* charge: det4's dead stripes, the
  badly seated M1 connector on det7 X. No χ² rule fixes this; it is a hardware limit and is quoted as one.

## What the cut does downstream today: almost nothing
- **`wft/reco.py`:** writes `quality_ok`, but nothing in reco reads it (ranking, track gate, two-track, head-on rescue and
  pairing don't).
- **Bench gates** (`effbudget/budget.py`, `mx_june_wft/03_angles.py`, `gate_compare.py`): no quality cut.
  `compat.load_table` defaults to `quality_only=False`.
- **n_TOF stage 3:** `build_tracks.py` gates on `track_gated & acceptance`, without `quality_ok`. Its docstring
  (lines 12–15) is stale.
- **Analyses that do cut on it:**
  - n_TOF: capsule_tof/pose, run79 prelim figures and merge, column_bias_check, displays, scint_acceptance, yield_gate,
    repass_readiness, cosmic_wall_scale. They lose 0.6–3.4 % of views.
  - bench and sim: 07_t0_prior_gate, column_ledger, ws_t0, t14_compare, t14_angle_trend. These apply a data-tuned
    absolute number to sim; whether that biases the comparison was not checked.

## Sample
- **Bench:** the `fe` gate rows (8000 M3 muons per golden row), interior views, 3195–6295 per chamber × view.
- **"Bad" view:** |Δθ vs M3| > 5° or |position residual| > 5 mm, measured about the median.

## What χ²/dof > 300 removes
"Bad among kept → removed" uses far5 (angle only); "removed / kept q" is the ratio of median charges.

| view | median χ²/dof | removed % | bad among kept → removed % | removed / kept q | saturated, kept → removed % |
|---|---|---|---|---|---|
| det2 x / y | 151 / 162 | 27.9 / 32.8 | 0.3 → 10.6 / 0.3 → 10.7 | 2.2 / 2.3 | 40 → 94 / 30 → 89 |
| det3 x / y | 68 / 71 | 11.9 / 15.3 | 0.3 → 25.8 / 0.2 → 20.7 | 2.3 / 2.3 | 23 → 91 / 22 → 89 |
| det4 x / y | 19 / 10 | 17.5 / 14.7 | 15.8 → 25.3 / 19.2 → 29.3 | 10 / 15 | 9 → 93 / 9 → 94 |
| det6 x / y | 509 / 741 | 69.6 / 81.1 | 3.1 → 5.5 / 0.5 → 5.6 | 1.6 / 1.8 | 69 → 96 / 53 → 94 |
| det7 x / y | 124 / 284 | 32.8 / 48.6 | 16.2 → 15.4 / 2.5 → 12.7 | 2.7 / 3.3 | 64 → 99 / 60 → 99 |

It removes both bright good tracks and real second deposits:
- **Bright good tracks:** 2–2.5 × the median charge, normal width and hit multiplicity, angle s68 1.2–1.4° against
  0.92–0.96°.
- **Second deposits on det2/det3 (δ electron or shower):**
  - 3–4 × the median charge;
  - 18–19 strips wide against 12;
  - twice the hit multiplicity;
  - high in the other view too (78–94 %);
  - 57–65 % are > 5 mm from the M3 ray;
  - almost never split into two tracks (n_tracks ≥ 2 in ≤ 1 %).
- **Not steep tracks:** at fixed charge, χ²/dof does not depend on the angle (`figures/chi2_vs_angle.png`).
- **At fixed charge, bad views sit 2–10 × higher in χ² than good ones** (`figures/chi2_vs_charge.png`): χ² carries
  information beyond brightness.

## Alternatives compared
Sources: `results/rules.md`, `results/scores.md`, `figures/purity_eff.png`.

| view | far5 with no cut % | fixed 300: removed / bad removed / far5 kept | 5 × median: removed / bad removed / far5 kept | AUC χ²/dof | AUC q | AUC χ² at fixed q, sat, angle | AUC χ²/q^a |
|---|---|---|---|---|---|---|---|
| det2 x | 3.16 | 27.9 / 93.3 / 0.26 | 9.8 / 80.8 / 0.62 | 0.93 | 0.83 | 0.77 | 0.86 |
| det2 y | 3.73 | 32.8 / 92.0 / 0.33 | 12.0 / 81.6 / 0.71 | 0.90 | 0.81 | 0.74 | 0.80 |
| det3 x | 3.32 | 11.9 / 86.8 / 0.29 | 10.1 / 85.0 / 0.34 | 0.94 | 0.90 | 0.74 | 0.77 |
| det3 y | 3.33 | 15.3 / 92.9 / 0.19 | 13.2 / 92.9 / 0.19 | 0.96 | 0.90 | 0.71 | 0.73 |
| det4 x | 17.49 | 17.5 / 23.7 / 15.82 | 29.4 / 33.4 / 16.02 | 0.49 | 0.43 | 0.64 | 0.73 |
| det4 y | 20.68 | 14.7 / 18.2 / 19.20 | 31.0 / 28.9 / 20.17 | 0.43 | 0.34 | 0.60 | 0.76 |
| det6 x | 4.79 | 69.6 / 79.5 / 3.08 | 9.7 / 48.8 / 2.66 | 0.71 | 0.67 | 0.60 | 0.69 |
| det6 y | 4.64 | 81.1 / 97.5 / 0.53 | 6.8 / 45.2 / 2.71 | 0.86 | 0.77 | 0.66 | 0.74 |
| det7 x | 15.95 | 32.8 / 30.8 / 16.20 | 21.2 / 25.4 / 14.97 | 0.38 | 0.49 | 0.48 | 0.43 |
| det7 y | 7.45 | 48.6 / 82.4 / 2.50 | 19.9 / 60.7 / 3.65 | 0.78 | 0.71 | 0.67 | 0.70 |

- **Per-chamber quantile cut:** within a chamber it ranks views exactly as χ²/dof does, and it fixes the inefficiency
  by construction. Rejected in favour of a multiple of the median.
- **χ² normalised by charge, or model-error-aware (proxies):** worse where χ² works (0.71–0.77 against 0.90–0.96 on
  det2/det3). This agrees with F17/F19's rejection of `MODEL_FRAC`.
- **Diagnostic features:** none beats χ²/dof (split statistic 0.56–0.62, width excess 0.28–0.65, both views high
  0.41–0.93, charge > 3 × median). `tan_err` is useless as a cut: the floor carries it.
- **Why 5 ×:**
  - It is the knee: 3 × removes 18–21 % for 90–95 % of bad views; 10 × removes 5–6 % for 58–78 %.
  - It reproduces 300 on det3, where 300 was tuned.
  - It returns 18–23 pp of efficiency on det2, 60–74 pp on det6 and 29 pp on det7 Y. The cost on det6: Y far5 among
    kept views rises from 0.5 % to 2.7 %.
- **SPS det4 (run_63, inclined Y, truth = the beam angle):**
  - fixed 300 removes 4–6 % of Y views, of which 68–76 % are far (whole sample: 25–30 % far);
  - 5 × median removes 16–18 % at 49–56 % far;
  - AUC 0.70–0.74 on Y, 0.50–0.57 on X.
- **n_TOF run_100 (arms A and C):** 5 × median removes 8.9–11.2 %, against 0.6–1.6 % for the fixed cut. Per-file
  medians are stable to ±10 %. Purity is **not measured**: there is no truth.

## Code it would change (not applied)
- **`wft/reco.py`:** drop `CHI2DOF_BAD` (l.30) and the per-fit `quality_ok` (l.492, 989, the PlaneFit field l.156,
  the empty row l.1651). Add `quality_flags(df, meta)`, called when events and candidates are written; candidates use
  the events table's medians. Write `chi2dof_median_{x,y}` and `quality_rule` to `events.meta.json`.
- **`compat.load_table` and `build_tracks`:** apply the same function when the meta has no `quality_rule`, so old
  tables are re-flagged from their stored χ² without a reco pass. Fix the docstrings (compat.py:115,
  build_tracks.py:12–15).
- **n_TOF:** compute the median per run × arm over all sub-runs.
- **Tests:** a test that the flag is relative.

## What this does not rule out
- **M3 truth is wrong ~0.8 %** of the time (bent tracks). Showers that M3 accepts count as bad, which is right for
  tail purity.
- **n_TOF purity is not validated** (different particles, low gain, the post-23-July noisy configuration). Arm D is
  where 300 would have bitten. An n_TOF analysis should check the flag on its own sample, with x–y charge balance or
  inter-chamber pointing.
- **det4 and det7 X:** neither rule helps, and the relative rule costs 21–31 % there.
- **A true post-fit χ² with a fractional error term** (`wm.chi2_plane` with `MODEL_FRAC`, no refit) was not run; only
  the proxies were.
- **The medians assume mostly single tracks.** A shower- or flash-dominated table would move its own threshold.
- **SPS:** det4 only, with the truth taken as the beam median.

## Files
All under `rerun/model_audit/followups/ge_chi2cut/`:
- `collect.py` builds `~/x17/cosmic_bench/model_audit/ge_chi2cut/bench_views_fe.parquet`;
- `scores.py` and `rules.py` write `results/{scores,rules}.{md,json}`;
- `figures.py` writes `figures/{chi2_vs_charge,chi2_vs_angle,purity_eff}.png`.

The SPS and n_TOF numbers came from inline runs and are not scripted. Inputs:
`fe_unify2/sps/*_h4.npz`, `/media/dylan/data/x17/is2_v2_stage/fullpass/run_100/stat090_000[0-3]`, and
`stage3_fullpass/tracks_run_145_stat090_0000.parquet`.
