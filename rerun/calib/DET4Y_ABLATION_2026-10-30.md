# det4 Y −0.25° ablation (G1 follow-up, HANDOFF step 1) — 2026-10-30

**Verdict: the template accounts for it, not the bow.** G1 found that det4 Y s68 improved by −0.245° (2.348° pass against 2.593° audit `fe_lm`).
On the consistent path, where each bundle's w0/kw is measured under its own template and bow, the shift breaks down as follows:

| step | Δ s68 y |
|---|---|
| code 5d4f5e4 → 6a0bb1f (arm a − audit fe_lm) | +0.010° |
| bow: audit → pass, on the fe template (c − a) | −0.013° |
| **template: audit → pass, under the pass bow (e − c)** | **−0.233°** |
| rest: kw/w0 source, xy_pairing, X dead list, generic σ_p0 (d − e) | −0.009° |
| **total (d − fe_lm)** | **−0.245°** |

- **The suspect is cleared.** Swapping the bow block is worth ≤ 0.013° on det4 Y.
  - The −0.43° the bow gate quotes compares bow against **no** bow. Both bundles here carry a bow; only the pass's is in tan form.
- **The pass's template is better on det4.** Its X template drives both views (`rc_tmpl = 1`). Compared with the audit's:
  - its peak is 10 ns earlier (140 against 150 ns);
  - its FWHM is 30 ns narrower (310 against 340 ns);
  - it has twice the undershoot (−4.6 % against −2.4 % of the area);
  - the RMS difference is 3.5 % of the peak.
- **The improvement holds across the board:**
  - It is the same under both fixed alignments.
  - It also shows in X (−0.02 to −0.08°).
  - Detection is unchanged or slightly better (+0.1 to +0.2 pt at 5 mm), and far is unchanged.
- **So det4 Y 2.35° can be quoted as a pass number.** The gain over the audit comes from the from-scratch template.

## Arms

Setup:
- **Row:** g_det4, with the G1 events (`wft.cli reco --matched-only --limit 8000`, 8014 rows with splits).
- **Code:** one archive for every arm, `code_6a0bb1f` (the pass's).
- **Scoring:** each arm is scored by `03_angles` under two fixed alignments:
  - **pass-al:** the pass `calib/r3` alignment G1 used;
  - **prod-al:** the audit `prod_alignment` G1 used for `fe_lm`.
- **Budget:** `effbudget` on pass-al.

| arm | bundle | s68 y pass-al | s68 y prod-al | s68 x | bias y | n y | kw y | w0 y | det 5 mm | far 5 mm |
|---|---|---|---|---|---|---|---|---|---|---|
| a | audit `calib_bundle_fe` as is | 2.603 | 2.603 | 2.509 | +0.005 | 7010 | 0.9833 | −0.2424 | 40.82 | 1.63 |
| b | fe + pass template | 2.269 | 2.269 | 2.434 | +0.025 | 7024 | 0.9833 | −0.2424 | 41.02 | 1.57 |
| c | fe + pass bow (`set_w0kw --bow-json`, w0/kw re-measured on a) | 2.590 | 2.590 | 2.527 | +0.002 | 7011 | 0.9858 | −0.2680 | 40.84 | 1.63 |
| e | fe + pass template + pass bow (w0/kw re-measured on b) | 2.357 | 2.357 | 2.487 | −0.013 | 7024 | 0.9610 | −0.2556 | 41.02 | 1.57 |
| d | pass `calib/c2/bundle` (the G1 r3 bundle) | 2.348 | 2.348 | 2.488 | −0.010 | 7010 | 0.9606 | −0.2579 | 40.94 | 1.57 |
| G1 pass | `calib/r3` as published | 2.348 | – | 2.488 | −0.010 | 7010 | | | 40.94 | 1.57 |
| G1 audit | `fe_lm` as published (code 5d4f5e4) | – | 2.593 | 2.509 | +0.006 | 7021 | | | 40.96 | 1.57 |

Checks:
- Arm d reproduces G1's pass value to the last digit, so the reco is deterministic.
- Arm a is 0.010° from `fe_lm`, which is the code drift between the two archives.
- Arm e is a 2×2 complement that the handoff did not list. It closes the decomposition: template and bow together, with nothing else changed.

## Template and bow are not additive

- **Template first, as the handoff ordered the arms:** the template is worth −0.334° (b − a), and the pass bow then gives back +0.088° (e − b).
- **Bow first:** the template is worth −0.233° (e − c).
- **Arm b is not a consistent bundle.** It keeps the fe w0/kw, which were measured under the fe template. Under the pass template plus a bow, w0/kw re-measure to kw_y 0.961 instead of 0.983 (arm e).
  - A kw too large by 2.3 % shrinks the fitted-noise term of tan by 2.3 % (about −0.05° of s68) and adds a scale error instead.
  - So b's 2.269° is partly a scale artefact. **Quote e and d, not b.**

## What this does not rule out

- **This is in-sample.** Both templates, the pass's (c1 cache, 1500 events) and the audit's, were measured on this same g_det4 row and are scored on it. A held-out check would mean scoring both templates on another det4 row. The pass rows that use the det4 golden bundle can do that, but there is no audit fe reconstruction of those rows to compare against. Not done.
- **The other G1 chambers were not ablated.** Their Δ s68 ≤ 0.06° is inside the gate. This says nothing about whether their templates differ by as much.
- **The small terms are not resolved individually.** The −0.009° remainder (d − e) lumps together w0/kw source, xy_pairing, the five extra X dead strips, and the generic σ_p0 (0.30 against 0.44 mm, used only by the two-track width probe). The same goes for the ±0.01° code drift. Each is at the LM gate's arm-to-arm reproducibility (±0.009°).

## Files

- **Scripts:** `rerun/calib/det4y_ablation/`
  - `make_bundles.py`: bundles a, b, d;
  - `make_dag.py`: the 22-node DAG, a copy in `ablation.dag`;
  - `score.py`: this table; `score.json` holds its output.
- **lxplus:** `~/x17rerun/det4y_ablation/`.
- **Products:** `/eos/user/d/dneff/x17rerun/det4y_ablation/Analysis/mx17_det4_day_6-24-26/long_run/mx17_4/wft/`
  - `bundle_{a..e}`;
  - `arm_{a..e}/{events.parquet, angles_pass, angles_prod, budget}`;
  - `align_{pass,prod}`;
  - `bow_block_pass.json`.
- **Nothing in Analysis_pass1 or Analysis_fe was written.** Two status-126 EOS failures were retried by the DAG, and it exited with status 0.
