# FE — FOLLOWUPS items 7, 11, 12, 13: the single reconstruction, round 2, 2026-10-27/28

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Final commit 79bb78d; tests `wft/tests` 104/104. Nothing promoted over production bundles.)

## Recommendation: the `fe` arm is the single reconstruction
`fe` = t0 pinned at the measured trigger latency + one σ0 = 0.30 mm (FC) + dead lists under one rule + FB's bow block +
column-time x–y gate + charge pairing per event class + band censoring and band re-seed. It improves angle width on every
view and detects more on four of five chambers. The bow block carries the angle gain (`fe_nobow`: neutral to slightly
worse); band handling carries the detection gain (`fe_nb`: same angles, no detection gain). FA's v(x,y) (`fefa`) =
`fe` on det2/det3 within noise; FA's gain is on det6 — add once det4/6/7 `fefa` finish. 20 ns grid **not adopted**
(≤ 0.02° on det2/3/6, det4 0.04–0.07° worse, −0.2…−1.1 pt detection, 3× CPU; needs pinned t0); slot kept.

## Algorithm changes
| item | before | now |
|---|---|---|
| 7 t0 | free | bench bundles pin t0 = C − 10 ns·ftst (C 300–339 ns); n_TOF refuses a bundle with an active pin |
| 7 x–y gate | t0x − t0y vs ftst table | column time t0 + u50, phase-corrected, zero offset (medians −3.6…+1.1 ns); tolerance p98 from data, 197–746 ns |
| 11 pairing | one ln(qx/qy) width per chamber | width per class: clean / censored (dead or band strip) / saturated |
| 12 σ0 | 0.238 X / 0.343 Y + per-event wide/narrow refit | one σ0 = 0.30 mm; per-event choice removed from the code |
| 12 bow (FB) | angle_map unused | `bow` bundle block, tilt surfaces A(U,W) from both views, one iteration; writes mesh height `bow_h` |
| 12 v(x,y) (FA) | — | 4×4 bilinear map common to both views; replaces the bow's linear scale term |
| 12 v(t) (FA) | — | slot only — event record has no DREAM timestamp |
| 12 20 ns grid | — | `depth_subdiv` slot, gated, not adopted |
| 12 dead lists | census (det4 over-, det6/det7 under-masked; none at n_TOF) | chamber-side by position + dataset's own faults − census strips the flank test shows firing: det4 X 77→51, det6 X 24→40, det7 X 114→137; n_TOF lists per arm/epoch in `masks/` |
| 13 bands | flat hits dropped from seeding, bench only | band strips detected per event on the full waveform, censored like dead strips; events with hits on band strips (sparks incl.) re-seeded without them. Windows ≥ 1800 ns only (bench, SPS); off at n_TOF |
| n_TOF transfer | bench content went across | `make_bundle` drops bow, v map, bench w0; `wft_beam` refuses a bundle carrying any |

## Gate (five golden rows, 8000 M3 muons, vs gate3 `unified`; `results/gate_fe.md`)
| chamber | Δ detected pt | Δ far pp | s68 X / Y unified → fe (°) | fe_nobow Δs68 X / Y |
|---|---|---|---|---|
| det2 | +0.15 | +0.12 | 1.294 / 1.488 → 1.161 / 1.214 | −0.01 / +0.06 |
| det3 | +0.37 | +0.15 | 1.156 / 1.088 → 1.082 / 0.966 | +0.04 / +0.03 |
| det4 | −0.12 | +0.04 | 2.574 / 2.679 → 2.509 / 2.601 | +0.09 / +0.23 |
| det6 | +1.05 | −0.16 | 2.031 / 2.083 → 1.469 / 1.619 | +0.01 / +0.06 |
| det7 | +3.30 | +0.10 | 2.591 / 1.791 → 2.499 / 1.580 | pending |
χ²/dof under the pin: median +3…+37 % (X +16…+37, Y +3…+8) = model error the free t0 absorbed; detection doesn't read χ².
Two-track splits fall on four chambers (det2 28→22, det4 47→32), rise on det7 (43→60). `CHI2DOF_BAD` = 300 now fails
0.1–7 pp more (det6 already 64–74 %) → needs a per-chamber value before use as a cut.

## Pairing per class (toy from real singles, `fe`)
| chamber | clean | censored | saturated | width clean / cens / sat |
|---|---|---|---|---|
| det2 | 98.1 | 96.9 | 98.3 | 0.064 / 0.115 / 0.079 |
| det3 | 98.0 | 97.3 | 99.0 | 0.058 / 0.076 / 0.075 |
| det4 | 98.2 | 97.9 | 98.0 | 0.095 / 0.123 / 0.165 |
| det6 | 96.8 | 92.1 | 93.6 | 0.102 / 0.210 / 0.173 |
| det7 | 97.6 | 84.4 | 85.4 | 0.075 / 0.388 / 0.334 |
Per-class widths move pairing ≤ ±1 pt vs one width, except det7 clean +1.7, censored +2.5. Charge alone 80–96 %. On
det6/det7 the limit is saturation itself.

## Bands (det3 sample: 252 band events, 400 controls)
Production 145 (57.5 %); flat-hit drop (10-26) 196; censoring alone 197 (losses happen before the fit: competing band
clusters, spark vetoes); **censoring + band re-seed 221 (87.7 %)**, controls 400/400; non-band det3 92.6 %.

## SPS gate with the H4 mask (run_63 25.64°, corrected caches, 1000 events/plateau)
Mask changes 19–22 % of X fits, < 1 % of Y. Inclined Y angle unchanged: −23.4/−22.2/−19.3 ± 1.4–1.7°, 19–26 % far.
Flat X width +0.1° with the mask. χ²/dof 7–16 (vs 100–150 on the offset caches). Keep `h4`; check its X strips with the
flank test.

## n_TOF close pairs (wft_beam, run_145 overlay bench, two-track 4-pitch minimum vs off)
| arm | < 12 mm | 12–24 mm | ≥ 24 mm |
|---|---|---|---|
| A donor tracks found | 15.4 % (vs 0) | 32.1 % (vs 15.4) | 36.7 % (same) |
| A events with ≥ 2 tracks | 31.9 % | 49.4 % | 51.1 % |
| C donor tracks found | 9.4 % (vs 0) | 26.9 % (vs 14.8) | 31.3 % (same) |
Absolute levels depressed everywhere (truth = old is2_v2 fit of each donor, refitted differently): compare columns.

## Pending
`fefa` det4/det6/det7 and `fe_nobow` det7 (lxplus auth expired; outputs land in
`/eos/user/d/dneff/x17rerun/followups/fe/Analysis_fe/` → `pull_fe.sh fefa fe_nobow`, rerun `gate_compare.py`). FA v(t)
needs the DREAM timestamp + acquisition tag in the event record (and FA warns it worsens det2/det3 X). FD gain map: slot
exists, values 1.0. FB finer bow cells. FC systematics to quote: −1.2 % compression, −4 % det6/det7 bright, −4 % det4 Y.

## Promotion (prepared, not applied)
`promote_fe.sh` (dry run unless `--apply`): moves each golden row's `calib_bundle` → `calib_bundle_pre_fe`, installs
`calib_bundle_fe` (or `_fefa` once gated). Then `promote_phys.py --fill-conditions --apply` (det7 empty conditions), then
`vrefit` + `set_w0kw.py --table` for every non-golden row. Downstream changes: angles (s68 −0.07…−0.56°, w0 bias shifts
≤ 0.3°), χ² +3…+37 %, `dt_xy` no longer read, new columns `bow_h`, `n_band_strips`, `n_sat_strips`, dead lists. n_TOF
dead lists via `set_w0kw.py --keep-w0kw --dead-json masks/ntof_<arm>_<run>.json` — none applied.

## Risks
`fe` finalised from tables reconstructed with σ0 0.29/0.33 (kw moves a few 0.1 %); column-time tolerance very wide on
det4/6/7 (p98 543/468/746 ns, heavy tails); band re-seed now fits previously vetoed spark events (det7 +359; far fraction
unchanged; not individually validated); n_TOF own-fault lists on thin statistics (0–131 channels, run_55 + run_145); H4
mask slightly widens SPS X; quality flag needs per-chamber values.

## Files
`wft/` (model.py, reco.py, calib.py, tests/test_fe_unify2.py), `ntof_tracking/wft_beam.py`, `rerun/w12/set_w0kw.py`
(`--sigma0`, `--dead-json`, `--depth-subdiv`, `--keep-w0kw`, `--bow-json`, `--vmap-json`); scripts/results here
(finalize_fe.sh, promote_fe.sh, pull_fe.sh, results/{gate_fe,chi2_pin,pairing_eff,sps_gate,pairs}.*, masks/). EOS
`/eos/user/d/dneff/x17rerun/followups/fe/`, mirror `~/x17/cosmic_bench/model_audit/fe_unify2/`.
