# Paper I bench resolutions on pass 1 (2026-10-31)

PASS_PLAN_2026-10-28 §4 "re-quote every resolution", on `Analysis_pass1` (code 6a0bb1f, post-fixes b5a29af/a72b889),
after the v(t) cross-check (`rerun/vt_all/CROSSCHECK_2026-10-31.md`). Read-only on Analysis_pass1; nothing was
refitted except the three X-width arms (§4). Code `rerun/paper1_res/` (archive `code_res_18efcd0d` on EOS: a pre-squash commit; res_rows, vt_window, dp_arms, dp_score and bow_pass1 are byte-identical to this commit, make_report.py runs locally from the repo), per-row
outputs `/eos/user/d/dneff/x17rerun/paper1_res/results_18efcd0d/`, flattened in `results/rows.csv`; generated report
`report/report.html`.

## Verdict

1. **Angle resolution (s68, interior_good, headline row per chamber, X / Y):**

   | chamber | row | pass | v(t) arm | core σ (v(t) arm) |
   |---|---|---|---|---|
   | det2 | 6-22 long_run (quoted golden) | 1.045 / 1.125° | 1.038 / 1.119° | 0.96 / 1.01° |
   | det3 | 6-27 sanity long_run (quoted golden) | 0.991 / 0.901° | (no map) | 0.92 / 0.84° |
   | det4 | g_det4 (calibration, in-sample) | 2.412 / 2.076° | 2.409 / **2.013°** | 1.71 / 1.23° |
   | det6 | g_det6_long (calibration, in-sample) | 1.256 / 1.506° | **1.211 / 1.454°** | 1.06 / 1.31° |
   | det7 | g_det7_long (calibration, in-sample; X ch 256–511) | 1.232 / 1.461° | 1.225 / 1.449° | 1.03 / 1.20° |

   The v(t) arm moves no headline by more than 0.063° (det4 Y), always down. It agrees with the cross-check's G1
   delta on all events (det4 Y 2.348 → 2.290°). Which arm to quote is Dylan's decision; every number here has both.
2. **Position resolution against M3 (per view, at the mesh, core σ with the M3 pointing subtracted, X / Y):**
   det2 456 / 444 µm, det3 447 / 443, det4 663 / 526, det6 348 / 382, det7 434 / 488. The M3 pointing P (0.21–0.28 mm)
   is 12–40 % of the variance. **Multiple scattering is not subtracted, so these are upper bounds on the chamber**
   (the bench is reference-limited by MS: `sps-spatial-resolution`). The chamber's floor is the SPS number. The
   mid-gap position reads within 45 µm of it, and the v(t) arm moves it by ≤ 16 µm.
3. **vs HV:** the angle resolution is best near the low end of every scan and worsens by 30–50 % towards the top (det2
   0.87° at 460 V → 1.30° at 525 V; det3 0.81–0.91° at 435–470 V → 1.13–1.31° at 525 V; det7 0.78° at 430 V
   → 1.65° at 500 V), as the det3 HV study found (charge saturation). Position σ stays at 350–500 µm across HV
   (reference-limited). det4's 6-23 scan rows hold ~100 interior events each on a night with degraded M3 matching:
   shown, not quoted.
4. **vs drift field (det3 6-27, mesh 490 V):** s68 0.92–0.97° (X) and 0.87–1.00° (Y) from 166 to 366 V/cm, 2.0° at
   100 V/cm, 7–9° at 33 V/cm (the column outlasts the readout window, and the drift is slow).
5. **vs angle:** the M3 telescope's acceptance ends near 20°, where interior_good holds < 25 events per row. Up to
   20°, s68 is flat on det2/det3/det7 (within 0.15°) and grows with angle on det4 X (1.9 → 2.8°) and det6 X (1.1 → 1.4°).
   The mesh position degrades from 350–500 µm at 0–5° to 600–750 µm at 15–20°, because the angle's error is carried
   to the first charge's depth.
6. **Window truncation (±5 % v within a row): none.**
   - **Test, table level:** 76 (row, view) pairs.
   - **Upper edge:** the fraction of column ends near the v-scaled window's upper edge never follows r(t) (no slope
     reaches 3σ).
   - **Lower edge:** it rises where r is low on det6's early 6-26 night only. Those columns end short, the opposite of
     a slower v, so this is the attachment epoch vt_within_rows saw.
   - **Not a gate:** the window ranks candidates and flags isochronous fits; it does not reject them.
7. **X-width systematic (diffusion ±100 µm/√cm, both views, re-folded): ≤ 0.04° on the angle s68 and ≤ 24 µm on
   the position core.** A wider diffusion model reads the angle 0.005–0.013° better; a narrower one 0.03° worse. The
   kw it implies moves ∓2 % and is absorbed by the re-fold (§4). Quote it as a systematic: +0.04 / −0.01° (angle),
   +24 µm (position, X).

## 1 · Method (both arms)

- **pass arm.** The pass table as it is.
- **v(t) arm.** v → v·r(t) in the angle mapping (r = `v_rel` of the row's `vt_correction` table; the 51 rows of
  `rerun/vt_all/results/vt_corrected_rows.csv`). Then w0/kw are re-folded on the corrected mapping by set_w0kw's
  ladder, taken as a difference to the same measurement with r = 1, so the ladder's own scatter cancels. This is
  CROSSCHECK §2's "vt+fold". It is identical to the pass on a row with no map.
- **Angles.** Exactly 03_angles' interior_good sample (the effbudget `ANG` outputs); s68 is its 68th percentile.
  Closure: g_det4 2.412 / 2.076°, sat_det3 0.975 / 0.896° = effbudget `s68_<p>_ig` to the digit. Core σ is a Gaussian
  fitted in the fixed window median ± 2 s68.
- **Positions.**
  - **Accounting:** 02_efficiency's, on the effbudget rays (interior_good, view fit ok, spark veto, |r| < 15 mm).
  - **Frame:** per view in the raw strip frame.
  - **Pointing:** P is the M3 4-plane pointing at the row's alignment z (measured per-plane σ, `m3_self_resolution`),
    rotated into the view. It is 0.21 / 0.24 mm on the top slot and 0.25–0.28 on the bottom slot (det6).
- **Excluded.** The 45 skeptical rows (registry class; notes listed in the report), e.g. det2 6-17 (half-dead X) and
  det3 6-23 (broken angle scale).

## 2 · Calibration rows, both arms (interior_good)

g_det4 Y 2.076 → 2.013°, g_det6_long 1.256 / 1.506 → 1.211 / 1.454°, g_det7_long 1.232 / 1.461 → 1.225 / 1.449°.
Re-folded kw moves ≤ 0.6 % (CROSSCHECK §2). Table in the report §2.

## 3 · Figures rebuilt for the papers

- **tracking_perf:** σθ vs angle on the headline rows, plus the within-5 mm efficiency.
- **gas_vE:** det3's drift scan, angle-scale v vs Magboltz bench gas.
  - The angle-scale v is v_drift × mean kw, so it carries kw: it is not a timing v.
  - The 333 V/cm cluster is the HV-scan rows at 1000 V.
- **bow_profiles, bow_closure:** fb_bow_maps' estimator unchanged, on pass-1 golden tables (`bow_pass1.py`).
  - The v(t) arm is what the paper shows. The pass arm is in `figures/*_pass.pdf`.
  - The mesh curvature moves ≤ 11 /km between arms, and the w0 gradient ≤ 0.24 µm/ns per m (det4 X).
- `papers/FIGURE_PROVENANCE_2026-10-31.md` is updated.
- **Still placeholders:** `tiers_benchmark` and `fleet` (Paper II). They need the hits tiers reconstructed on the
  pass-1 events, which are not a pass product, so v(t) no longer blocks them.

## 4 · X-width systematic arms

PASS_PLAN §4 says det3 +100 and the others ±100 µm/√cm. The model has one diffusion constant Dp for both views
(Dp = D_T √v), so the arm shifts D_T in both views: an upper bound on an X-only effect.

The arms reconstruct the G1 events on code 6a0bb1f (`~/x17rerun/paper1_res/dp/`, condor DAG exit 0). They are
scored table-level by `dp_score.py` on exactly the pass's interior_good events and M3 rays; the baseline reproduces
res_rows to the digit. Angle s68 (X / Y) is shown as base → arm → arm with w0/kw re-folded; positions are core σ
before the M3 subtraction.

| arm | row | Dp (D_T µm/√cm) | angle s68 X | angle s68 Y | kw ratio X / Y | pos core X | pos core Y |
|---|---|---|---|---|---|---|---|
| det3 + | sat_det3 | 0.02300 → 0.02898 (384 → 484) | 0.975 → 0.964 → 0.969 | 0.896 → 0.876 → 0.886 | 0.983 / 0.983 | 504 → 521 µm | 472 → 473 µm |
| det2 + | o22_long_det2 | 0.02500 → 0.03127 (399 → 499) | 1.025 → 1.019 → 1.014 | 1.084 → 1.069 → 1.071 | 0.984 / 0.982 | 487 → 511 µm | 476 → 482 µm |
| det2 − | o22_long_det2 | 0.02500 → 0.01873 (399 → 299) | 1.025 → 1.074 → 1.057 | 1.084 → 1.154 → 1.116 | 1.019 / 1.021 | 487 → 479 µm | 476 → 475 µm |

- **Angle.** A wider diffusion reads 0.005–0.013° better after the re-fold, a narrower one 0.03° worse. The ~2 % kw
  shift each implies is exactly what the per-row w0/kw fold absorbs.
- **Position.** A wider diffusion costs up to 24 µm in X (17 on det3) and nothing in Y; a narrower one nothing.
- **The X-only effect is smaller still,** because these arms move both views.

## What this does not rule out

- **Multiple scattering** between the last M3 plane and the chamber is in the position widths: σ_DUT is an upper
  bound. The bench cannot give the chamber's floor; SPS does.
- **det4, det6 and det7 headlines are in-sample.** Their only golden row is the calibration row.
- **det6's drift velocity over 6-26 is known only to a factor ~2** (ladder against column, CROSSCHECK §1). Its
  angles are corrected by the ladder, the angle scale against M3. Its mid-gap positions in the v(t) arm carry that
  limitation, but they move ≤ 16 µm.
- **Rows below the v(t) survey's floor** (mostly 10–20 min scan rows) carry no map, so their v(t) arm equals the pass.
- **The 03_angles sample and the position sample differ slightly.** 03_angles builds its own reference rays; the
  positions use the effbudget's rays. n differs by ≤ 2 %.
- **The X-width arm** moves both views; an X-only width model is not in the code.
