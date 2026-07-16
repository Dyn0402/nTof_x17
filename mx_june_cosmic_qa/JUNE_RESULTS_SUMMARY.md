# June 2026 cosmic-bench QA — combined results

> **Changelog 2026-07-14 — stricter M3 reference recipe (χ²<1.0 & NClus=4), fleet-wide.**
> Per `det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`: the position residual on the
> old χ²<5 & NClus≥3 recipe was **reference-limited**, not chamber-limited (no plateau
> inside χ²<5). Recipe centralised in `qa_config.py` (`M3_CHI2_CUT=1.0`, `M3_MIN_NCLUS=4`)
> and applied to every golden-chain `M3RefTracking(...)` call; alignment (`03`) and
> efficiency/residual (`08`/`09`/`12`) re-run fleet-wide. **All 7 alignments converged
> cleanly** (θ, z unchanged within fit error) — no detector needed the NClus≥3 fallback,
> including the weaker-M3-yield det4/6/7. Fleet-wide **efficiency (≤5mm) / core σ**, old →
> new:
> - **det3 (A, headline `g_det3_wknd`):** 88.8→**92.9 %**, 0.63→**0.47 mm** (reco_far 3.1 %)
> - **det3 (6-22 B-slot, `g_det3`/long_run):** →**90.0 %**, →**0.46 mm** (reco_far 3.1 %)
> - **det2 (B, `g_det2`/long_run):** 87.0→**91.3 %**, 0.64→**0.46 mm** (reco_far 3.6 %)
> - **det6 (C):** 55.7→**57.8 %**, 0.59→**0.45 mm** (still spark-limited, 23.0 % spark)
> - **det7 (D):** 36.4→**43.1 %**, 0.86→**0.59 mm** (still spark-limited, 33.4 % spark)
> - **det4 (E):** 20.0→**20.7 %**, 0.89→**0.67 mm** (still gain-limited)
>
> Every detector improves in the same direction and rough magnitude the note predicted
> (σ down ~25–30 %, efficiency up a few points as reference-error near-misses leave the
> sample) — this is expected/correct, not a regression; re-quote headline numbers
> accordingly. **Bug found + fixed in passing:** `sat_det3`'s no-veto `event_results.pkl`
> cache predated the 7-06 file-003 dead-y-plane-tail recovery (cached 6-29, data fixed
> 7-06); refreshed via `03 --no-veto --refit` — unrelated to the M3 cut, but was silently
> giving a wrong (73–77 %) efficiency for that run before the fix.
> Active-area outlines (nominal 40×40 cm dashed + measured true-active solid red,
> `common/mx17_active_area.py`) are now drawn on every 2-D position map (08/09/12, the
> shared `plot_efficiency_map`/`plot_resolution_map`/`plot_resolution_map_sliding` used by
> `03`, occupancy `04`, edge/fringe maps `32`, charge-balance `38`, spark position maps).
> Old χ²<5 numbers below are superseded; kept in §1 footnote for reference.

> **Changelog 2026-07-14 (continued) — waveform-dependent chain (26–42) re-run fleet-wide.**
> Unsharing (26), refinement/kernel (27), calibration (28), micro-TPC metrics (31), head-on
> (33), hybrid tracking (34), position estimators (36), charge balance (38), time
> resolution (42) all re-run on the new χ²<1.0+NClus=4 recipe for all 5 detectors
> (`sat_det3`, `o22_long_det2`, `g_det4`, `g_det6_long`, `g_det7_long`). **Hybrid angular
> σ68 barely moved** (fleet-wide ≤0.15°, several detectors slightly improved) — confirms
> angle resolution is only weakly coupled to the M3 position cut, exactly as predicted.
> Measured sharing constants (c1/c2) and calibration constants (v, b) all consistent with
> the pre-existing design-reference/2026-07-12 values within statistics — see
> `REPROCESSING_CONSTANTS.md`. **Two real bugs found + fixed:** (1) `34_hybrid_tracking.py`
> crashed on det4 (empty monotonic-calibration bin array under the lower stats the
> stricter cut leaves) — fixed with an adaptive min-per-bin threshold + raw-score
> fallback. (2) `36_position_estimators.py` was missing the malformed-multi-frame-event
> guard already present in 26/27/28/31/33 (det4's FEU8 has ~0.3% malformed frames) — 36
> had simply never been run on det4 before this pass, so the gap was never hit. Both
> fixed in the scripts, not worked around. HV scans (`10_hv_scan_efficiency.py`) re-run
> for det2/det3: peak efficiency 89.9→**94.9 %** @490V (det2, was 480V), 90.8→**93.2 %**
> @495V (det3, was 480V) — same clean improvement pattern. `build_final_pdf.py` and
> `build_hv_scan_pdf.py` (det2/det3 only) rebuilt with the fresh numbers.

> **Changelog 2026-07-14 (continued 2) — det6/det7 HV scans + engineer package rebuild.**
> All 4 remaining HV-scan runs completed: `g_det6_hv`/`g_det7_hv` (dedicated 400–500V
> scans, 16 pts each) and `g_det6_long`/`g_det7_long` (overnight 505–530V/480–505V
> points, 6 pts each). Peak efficiency: det6 76.2→**78.8 %** @480V, det7
> 63.1→**64.1 %** @440V — same clean pattern as det2/det3, confirming the fleet-wide
> consistency (7/7 detector-HV-scan re-measurements now agree in direction and
> magnitude). `build_hv_scan_pdf.py` rebuilt with all 5 detectors (`june_hv_scans.pdf`,
> 5 pages). The engineer-package LaTeX report (`report/main.tex`, 23 pages) was fully
> rebuilt: all 34 M3-recipe-dependent figures (10–18, 20–26, 30–34, 44) regenerated or
> freshly copied, and every headline number reconciled by hand across the abstract,
> keyboxes, fleet table, and quotable-numbers table — including correctly distinguishing
> the **operating** efficiency (92.9 %, includes in-spark coincidence) from the
> **spark-free** efficiency (96.6 %), which coincidentally share a similar-looking digit
> to the old operating number (88.8→92.9 %) but are different quantities; conflating them
> would have been a real error. Compiles cleanly (0 missing figures, `pdftotext`-verified).
> `make_slide_deck.py` (18 slides) also rebuilt and verified (python-pptx read-back
> confirms no stale "88.8" text remains).

> **Changelog 2026-07-14 (continued 3) — final two open items closed.** (1) det3-frozen-
> model transfer cross-check re-verified on the new recipe (fresh model saved on
> `sat_det3`, applied untouched to det2/4/6/7): every detector's transfer degradation/
> improvement matches the pre-recipe-change pattern exactly, confirming the near-
> identical regression weights are a genuine design property (full table:
> `REPROCESSING_CONSTANTS.md`). (2) Measured the true-active-area Y passivation on
> det2/4/6/7 (generalised `edge_chi2_extract.py`/`edge_chi2_plots.py` from det3-only to
> any run key) — **17.9–20.4 mm across all 5 detectors**, confirming it's a common
> construction feature. X passivation stays nominal everywhere (det4/6/7's X profiles
> are corrupted by genuine hardware pathologies, not real edges — see
> `M3_CUT_AND_ACTIVE_AREA_NOTE.md`). `common/mx17_active_area.py` extended with a
> `TRUE_ACTIVE_BY_DET` fleet table; every plot that draws the active-area outline
> (08/09/12/04/32/38, spark maps, shared `plot_*_map` functions — ~20 call sites) now
> uses each detector's OWN measured outline. 08/09/12 re-run fleet-wide to pick up the
> corrected outlines; engineer report + slide deck recompiled clean. This closes every
> item from the original note — nothing outstanding.
> **chi2<0.5 check (det3, both runs, +NClus=4):** efficiency is flat (92.9→93.0 %,
> 93.4→93.5 %) while core σ keeps falling with no plateau (0.47→0.43 mm both runs, ~9 %
> further) for a stats cost of ~43 %→~30 % kept. Use chi2<1.0+NClus=4 for headline
> efficiency/position (current); chi2<0.5 is a good choice for a dedicated
> spatial-resolution-only quote. Details:
> `det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`.

> **Changelog 2026-07-12 — fleet-wide hybrid angular reprocessing (REPROCESSING_PLAN.md).**
> The "Angular σ" column is now the golden **hybrid tracker** (script 34), fleet-wide, replacing
> the superseded script-03 time-fit numbers. New fleet **lt5 σ68 (self, holdout)**:
> **det3 1.75°, det2 2.47°, det4 2.70°, det7 3.64°, det6 4.14°** (plateau: det3 1.86°, det2 2.17°,
> det4 4.09°, det7 4.90°, det6 broken). Two findings: (1) **det6's X-plane (FEU 3) is dead for
> micro-TPC** — no drift-time development (v→119 µm/ns, plateau σ68≈16°, ψ 83°), board-C mesh
> defect; only its |θ|<5° signature-regression band is quotable. (2) **det4 is measurable** (gate
> passed, contrary to the "likely gain-limited" expectation) and has the cleanest micro-TPC angles
> of the three new detectors — its limit is efficiency, not tracking. det6/det7 both show X-plane
> c1≈0.25 (~½ det3), a measured board-C/D property. **Deviation from clean-detector expectation:**
> det6/det7 unshared-time-fit v disagree with v_geom by >3 % (degraded 700 V lever arm) — reported
> as degraded, *not* shipped as clean. Efficiency/spatial/spark numbers are unchanged (their v2
> alignment was re-verified bit-identical). Archived 44 superseded items →
> `Analysis/_archive_superseded_2026-07-12/` (script-03 angle JSON/PNGs, pre-v2 backups, no-veto
> alignment trees); overview PDF + engineer report/deck regenerated (0 MISSING). Per-detector
> constants: `REPROCESSING_CONSTANTS.md`. Log: `Analysis/_grand_logs/2026-07-12_*`.

> **Updated 2026-07-06 — reprocessed on M3 v2 reference tracking.** All numbers below
> are recomputed with the v2 M3 tracker (layer-drop rescue, per-plane cluster-count
> `NClus` branches, charge-weighted centroid, refreshed 2026 plane offsets) and the
> recommended reference-track recipe **`NClusX≥3 & NClusY≥3 & χ²<5`** (was `χ²<20` with
> no NClus cut). The cleaner reference lifts efficiency several points and tightens the
> spatial resolution across the fleet; a new **micro-TPC angular resolution** row is
> added. Spark rate is now quoted as the **crossing-based** fraction (the efficiency-
> breakdown `spark` category), so headline and breakdown-bar agree.

Saclay cosmic-bench characterisation of the MX17 micro-TPC Micromegas detectors
(**2, 3, 4, 6, 7**; no 5), June 2026. Efficiency is measured against M3 reference
tracks: for every clean M3 single muon track (v2 recipe above), project to the aligned
detector plane and ask whether the detector has a reconstructed X+Y hit within **5 mm**
(a track with no DREAM readout is a genuine miss, kept in the denominator). Resolution
is the Gaussian core of the residual distribution. The micro-TPC angular resolution is
the **hybrid tracker** σ68 of (θ_hybrid − θ_ref) — time-fit above ~5°, head-on signature
regression below — reported per band with coverage (script 34, odd-eid holdout). All runs Ar/Iso 95/5, non-zero-suppressed,
`det_orientation.z = 90`; slot map bottom = FEU 3(X)/4(Y) z≈232 mm, top = FEU 6(X)/8(Y)
z≈702 mm.

Two compiled PDFs (under `~/x17/cosmic_bench/Analysis/`):
- **`june_detectors_overview.pdf`** — one page per detector (headline efficiency +
  resolution stat cards, sliding-window within-5 mm efficiency map, binned map,
  breakdown, sliding resolution map, residuals, pulse-height-vs-strip, hit/miss scatter).
- **`june_hv_scans.pdf`** — resist-HV scans: summary overlay + per-detector
  efficiency-vs-HV and resolution-vs-HV.

**Spark separation.** The efficiency breakdown (bottom bar of every overview page) counts
**`spark`** (event fires >50 strips = a full-detector discharge) as its own category,
pulled out *before* the reco/hit split, so sparks no longer masquerade as
`reco_far`/`reco_near`. The **spark rate quoted here and on the overview headline is the
crossing-based fraction** — sparks as a % of active-area M3 crossings (the breakdown-bar
`spark` category), the same denominator as efficiency. (A second, firing-event–based
number, `spark_frac` = discharges / firing events, is ~2–3× larger and still printed in
`efficiency_breakdown.txt`; it is *not* the headline value.)

---

## 1. Per-detector overview (best high-stats subrun, M3 v2)

**M3 reference recipe (2026-07-14): χ²<1.0 & NClus=4 on both planes** (was χ²<5 &
NClus≥3 — see the 2026-07-14 changelog above). Every column below, including the hybrid
angular σ68, is re-measured on the new recipe (26→28, 31, 33, 34 re-run per detector
2026-07-14, `sat_det3`/`o22_long_det2`/`g_det4`/`g_det6_long`/`g_det7_long`).

| Det | Run / subrun used | Clean M3 rays (active area) | Efficiency (≤5 mm) | Core σ | Hybrid σ68 <5° | Spark | Verdict |
|----:|---|---:|---:|---:|---:|---:|---|
| **3 (A)** | 6-27/28 weekend / long_run_p2 (top slot) | 22.3k | **92.9 %** | **0.47 mm** | **1.66°** @98% | 3.9 % | Best performer |
| **2 (B)** | 6-22 overnight / long_run | 19.3k | **91.3 %** | **0.46 mm** | **2.34°** @96% | 5.0 % | Healthy |
| **6 (C)** | 6-26 overnight / long_run | 10.4k | **57.8 %** | **0.45 mm** | **3.88°** @98% ᶜ | 23.0 % | Spark-limited |
| **7 (D)** | 6-26 overnight / long_run | 9.6k | **43.1 %** | **0.59 mm** | **3.36°** @90% | 33.4 % | Spark-limited |
| **4 (E)** | 6-24 daytime / long_run | 12.3k | **20.7 %** | **0.67 mm** | **2.56°** @79% | 3.2 % | Gain-limited |

**Superseded (χ²<5 & NClus≥3, pre-2026-07-14):** det3 88.8 %/0.63 mm/1.75°@97%, det2
87.0 %/0.64 mm/2.47°@96%, det6 55.7 %/0.59 mm/4.14°@98%, det7 36.4 %/0.86 mm/3.64°@90%,
det4 20.0 %/0.89 mm/2.70°@81% — do not quote; the stricter cut removes reference-error
near-misses the old recipe let through (see changelog). "Clean M3 rays" also dropped (the
recipe keeps ~18–43 % of tracks depending on detector) — this is the expected cost of the
stricter cut, not a stats problem; tens of k crossings remain. The hybrid angular σ68
barely moved (fleet-wide ≤0.15° shift, most detectors slightly IMPROVED) — confirms
angular resolution is only weakly coupled to the M3 position cut, as expected (the hybrid
estimator is trained/scored against θ_ref from the SAME M3 track, so a cleaner reference
sharpens the target slightly but doesn't change the detector-side physics).

Efficiency and spark are % of active-area crossings; core σ is the residual Gaussian
core. **The angular column is the hybrid micro-TPC tracker (script 34): the
self-trained σ68 of (θ_hybrid − θ_ref) in the |θ_ref|<5° band, odd-eid holdout, with
coverage.** This supersedes the pre-hybrid script-03 production time-fit angles
(det3 2.04°, det2 2.15°, det6 3.15°, det7 2.50°, det4 2.49° — *do not quote*; that
estimator is alignment-QA-only now). Plateau (|θ_ref|>8°) σ68 (2026-07-14, new recipe):
det3 1.84°@97%, det2 2.11°@97%, det4 3.84°@81%, det7 4.97°@83%; det6 still BROKEN
(19.1°@98%, unchanged hardware defect — X-plane FEU3 develops no drift-time structure,
see below). det3-frozen-model transfer cross-check (overfit test, re-verified
2026-07-14 with a freshly-saved det3 model on the new recipe): det2 2.65°, det4 2.14°,
det7 4.49°, det6 3.35° — every detector's transfer behaviour (degrades/improves, and by
roughly how much) matches the 2026-07-12 pattern exactly, confirming the near-identical
weights are a genuine design property, not overfitting, and that this conclusion is
stable across the M3 recipe change (full table: `REPROCESSING_CONSTANTS.md`).
Alignment converged sub-mm with θ≈89.2–90.15° and z≈713–714 mm for every detector on the
new recipe too (all re-verified 2026-07-14; consistent with the χ²<5 geometry within fit
error — the cut only removes bad tracks, it doesn't change the detector's physical
alignment).

ᶜ **det6 (C) low-angle only.** Its plateau time-fit is unusable — the X-plane (FEU 3)
does not develop micro-TPC drift-time structure (v rails to ~119 µm/ns, plateau σ68 ≈ 16°,
3D opening-angle median 83°), a consequence of the board-C mesh defect noted in run_config.
The |θ|<5° band IS measured, via the charge-topology **signature regression** (Fisher LDA
AUC 0.867), which needs no time-fit lever arm. det6/det7 both show X-plane charge-sharing
c1≈0.25 (~half det3's 0.449) — a consistent board-C/D property (measured, not assumed).

**Notes on run choice / v2 vs pre-v2:**
- **det3 (A)** headlines the **6-27/28 weekend** run (top slot, FEU 7/8, started Sun
  06-28 01:33): **88.8 %** efficiency with a clean, linear micro-TPC angle (v_drift≈33
  µm/ns, σ≈2°) — the best of the fleet. NB the subrun is labelled `p2_det1_sanity_check`,
  but that sanity check was for the *other* (P2/det1) detectors; det3 is a full data run.
  An earlier pass read only **80.6 %** here because det3's `combined_hits` was **missing
  its file 000** (never reconstructed): M3 covers eventId 1–153405 but det3's hits started
  at ~12976, so events from the first file were miscounted as "silent" (9.4 %). **Fixed by
  reconstructing that file** (`process_run.py` on the raw FEU 7/8 fdf) → complete hits,
  52.0k rays, 0.2 % silent, on both local and EOS. A stray `01H29` false-start (1277
  events, colliding eventIds) was moved to `_false_start_01H29/`. The 6-22 bottom-slot run
  reaches 87.2 % but its micro-TPC angle is unusable (r≈0.62), so the weekend run is the
  headline on every metric.
- **Detector-data range guard** (`08`/`09`): efficiency counts only M3 rays whose eventId
  falls within the detector's `combined_hits` span, so an unreconstructed raw file cannot
  masquerade as detector inefficiency. No-op now that the June dataset's hits are complete.
- **det4 (E) recovered (7-06 evening)** — the 6-24 run had the same decode-interruption
  bug as the saturday det3 run: combined file 000 was missing FEU 8 (det4's y-plane) and
  combined file 007 was **empty** (built from FEU 03 alone, which had no hits). The raw
  fdfs were intact on EOS; files 000/007 were re-decoded + re-combined (all FEUs) and
  synced back. Full chain re-run (`03 --refit`, `03 --no-veto --refit`, `08/09/12`):
  det4 measures **20.0 % within 5 mm on 30.0k active-area rays** (has_any 69.7 %,
  reco-at-all 23.7 %, core σ 0.89 mm, 84.3 % of reconstructed within 5 mm) — roughly
  double the pre-recovery 10.3 % / 2.8k-ray quote (that tiny denominator was itself a
  symptom of the broken files). Still gain-limited (hit_no_reco 42.6 %, silent 30.3 %),
  but meaningfully better than previously reported.
- v2 vs pre-v2 (same runs): efficiency up (**det2 75.4→87.0 %**, det6 51.2→55.7 %, det7
  31.9→36.4 %, det4 8.2→10.3 %) and core σ tighter (det2 0.83→0.64, det7 1.18→0.86 mm) —
  the cleaner v2 reference removes spurious "misses" and sharpens the alignment.
- **det6 (C) angular resolution** — on the golden hybrid chain (script 34, 2026-07-12) the
  plateau time-fit is unusable: the X-plane (FEU 3) develops no micro-TPC drift-time
  structure (unshared v rails to ~119 µm/ns, plateau σ68 ≈ 16°, 3D ψ median 83°). Only the
  |θ|<5° band is measurable, **4.14° @ 98% via signature regression**. The earlier "3.15°,
  r≈0.86" was the pre-hybrid script-03 combined X+Y correlation — the Y-plane carried that
  correlation while the X-plane time-fit was already biased; the hybrid, which quotes an
  honest per-band σ68, exposes the X-plane limitation. This is the board-C mesh defect
  showing up in the micro-TPC angle. (Efficiency/spatial numbers above are unaffected.)

### Notes per detector
- **det2 / det3** — healthy detectors: high, spatially-uniform efficiency at sub-mm
  resolution. det3 is the best of the batch. The headline det3 is now the **6-28 weekend
  run** (top slot, FEU7/8, z702; 53.0k rays, 79.7 %, silent only 2.4 %); the earlier 6-22
  det3 (bottom slot) gave a consistent **80.7 %** — det3 is reliably ~80 % in either slot.
- **det4** — gain-limited (numbers post-recovery, see note above). Fires on ~70 % of
  muons; the loss is dominated by `hit_no_reco` (42.6 %) + silent (30.3 %): clusters
  rarely reach the ≥3 strips needed to reconstruct. The pre-recovery "silent 49.4 % /
  fires on 51 %" figure was inflated by the missing y-plane files. Same pathology the
  old det1 had → an HV/threshold/gas (gain) issue, not a dead detector.
- **det6** — good core (0.68 mm) but **spark-limited**: 23.7 % of crossings are
  full-detector discharges (drift 700 V, resist likely past optimum). With sparks removed
  the reco_far tail is small (7.5 %); efficiency 51.2 %. Prefer long_run (short_run is
  low-stats / unsettled).
- **det7** — reconstructs with a good core (σ≈0.9–1.1 mm per axis) but is the most
  **spark-limited** of the batch: **31 %** of crossings are discharges. Once sparks are
  separated the outlier `reco_far` tail collapses from ≈40 % to **11.2 %** — i.e. the tail
  reported earlier was *mostly sparks*, not a distinct saturation pathology. The residual
  Y-plane (FEU 8) saturation band still contributes to the remaining tail. A tighter
  discharge/saturation veto should recover a truer efficiency. **Open follow-up.**

### What is the reco_far tail? (det3 deep-dive)

Full characterisation in `det3_recofar_analysis/main.pdf` (run `g_det3_wknd`, 7271
reco_far events). The tail is **two overlapping populations**, not one thing:
- **~40 % near-miss shoulder** (5–10 mm, just past the cut) — the ordinary
  resolution/angle tail; widening the cut to ~7 mm absorbs it.
- **~38 % genuine mis-reco** (>20 mm; 23 % land >50 mm) driven by **elevated-multiplicity
  "sub-veto" discharge activity** (median 20 strips vs 16 for good events, extending up to
  the 50-strip veto; long multi-pulse cluster tails), spatially **concentrated on the
  low-X / left edge** of the chamber (reco_far rate ~10 % in the bulk, 25–70 % on the left
  edge; bad reco points pile up at low X).

Cleanly **ruled out**: multi-ray / ray-mismatch (0.00 % of reco_far have >1 M3 ray) and a
single-plane readout fault (X-only 36 %, Y-only 24 %, both-planes-wrong 41 %). Conclusion:
reco_far is a localised detector/HV (edge sparking) + competing-cluster-selection effect,
not a tracking artefact. Follow-ups: tighten the discharge veto toward ~30–40 strips (or
add a duration/multi-pulse veto); inspect the low-X edge strips.

---

## 2. Resist-HV scans

Efficiency vs resist HV, integrated over a fixed per-detector active box; alignment
seeded from each run's long_run subrun and re-translated per HV point.

| Det | Scan(s) (drift) | Peak efficiency | at HV | Behaviour |
|----:|---|---:|---:|---|
| **2** | 6-22 (1000 V), 450–525 V | **89.9 %** | **480 V** | plateau, sparks-off above ~510 V |
| **3** | 6-22 (1000 V), 450–525 V | **90.8 %** | **480 V** | best plateau, σ≈0.7 mm at optimum |
| **6** | 6-26 hv_scan 400–500 V + overnight 505–530 V (700 V) | **76.2 %** | **480 V** | full turn-on → plateau → falloff |
| **7** | 6-26 hv_scan 400–500 V + overnight 480–505 V (700 V) | **63.1 %** | **440 V** | full turn-on → plateau → falloff |

(v2 peaks; pre-v2 were det2 77.8 %, det3 81.1 %, det6 71.0 %, det7 54.7 %.)

In the 6-22 scan det2 (resist ch 3:4) and det3 (3:3) were stepped **together**. det6/det7
have **two** scans each: the dedicated 6-26 `hv_scan` run (stepped together, 400–500 V)
and the earlier 6-26 overnight points (higher V); overlaid they give the complete curve.

**Reading:** `any_hit` stays ~flat near 100 % while reco-efficiency falls at high HV →
the high-voltage losses are sparking-induced reconstruction failures, not the detector
going silent. This is now shown directly: each HV-scan page overlays a **spark fraction**
(events with >50 strips firing, right axis) — it stays <10 % through the plateau then
climbs steeply (to 40–57 %) exactly where efficiency rolls off. Resolution mirrors it:
best near the efficiency optimum, degrading into the sparking regime.

**Takeaways (operating points):**
- det2 / det3 optimum ≈ **485–490 V** (drift 1000 V).
- **det6 optimum ≈ 480 V (~71 %)**, det7 ≈ **440 V (~55 %)** (drift 700 V). The earlier
  overnight scan had started past these optima; the dedicated low-V re-scan recovered the
  turn-on and plateau. (5 mid-range points 455–490 V were not decoded at analysis time —
  a small gap; the shape is unaffected.)

---

## 3. Excluded / not measured

- **6-23 overnight (det3 + det4)** — both the long-run alignment and the HV scan are
  **blocked by a degraded M3 reference** (~3.9 % clean tracks; alignment rails to
  z=569/411 mm vs nominal 232). No reliable track reference → no trustworthy efficiency.
  See `TODO_m3_reference_6-23.md`. det3 and det4 are characterised instead from the
  6-22 and 6-24 runs respectively.
- **6-25 det3 long-run** and other raw-only subruns were not decoded at analysis time.
  The **6-27/6-28 weekend det3** runs (top slot) HAVE since been analysed and are now the
  headline det3 (79.7 %, above); the 6-27 saturday + 6-28 p2 long runs pool to 76.1 %.
  **RESOLVED (7-06 evening): the 6-27 saturday "dead y-plane tail" was a decode
  interruption, not a DAQ failure — and it has been recovered.** The raw
  `..._003_08.fdf` exists on EOS (byte-identical size to its FEU 1/7 siblings; the DAQ
  log confirms 47,452 events in all 3 FEUs) — only the online decode was killed after
  003_07 when the run ended. File 003 was re-decoded + re-combined with both FEUs
  (`process_run.py`, same recipe as the p2 file-000 recovery) and synced back to EOS;
  the FEU7-only combined file is parked in `_backup_feu7only_003/`. On complete
  statistics the saturday run measures **92.9 ± 0.2 % in a 5 mm fiducial** (16.3k rays,
  live-range guard now a no-op) **and ~96 % in the core (>25 mm from the edge)**; the
  0–25 mm degrader edge band has a 0→96 % turn-on and holds most of the remaining
  inefficiency (see `32_edge_fringe_field.py` + `DET3_WEEKEND_ANALYSIS.md` §7–8).
  The earlier pooled 76.1 % and 79.7 % full-area numbers predate the recovery and
  understate the detector (dead tail + edge band inside the denominator).

---

## 4. Provenance & reproduction

- Run registry: `qa_config.py` keys `g_det2 g_det3_wknd g_det4 g_det6_long g_det7_long`
  (+ `g_det3` = 6-22 det3, `g_det6 g_det7` short_run, and per-subrun variants).
  `g_det3_wknd` (6-27 weekend, top slot) is the det3 headline — clean micro-TPC angle.
- v2 rerun (per key): `03_alignment_and_tpc.py --full` (alignment refit + maps +
  micro-TPC angle → `angular_resolution.json`), then `08`, `09`, `12`; all M3 loads use
  the centralised recipe `qa_config.M3_CHI2_CUT=1.0` / `M3_MIN_NCLUS=4` (2026-07-14; was
  `chi2_cut=5`/`NClus≥3` default). Caches (`event_results*.pkl`) are det-hit-only and
  reused — **check their mtime against `combined_hits_root` before trusting a quote**
  (sat_det3's no-veto cache was found stale by a week; see 2026-07-14 changelog). venv is
  **`.venv`** (repo root).
- Per-detector overview: `.venv/bin/python build_final_pdf.py` (default keys headline the
  weekend det3; page 1 is the fleet summary). HV scans: `10_hv_scan_efficiency.py` per
  detector → `build_hv_scan_pdf.py g_det2 g_det3 g_det6_hv g_det6_long g_det7_hv g_det7_long`.
- Analysis outputs live under `~/x17/cosmic_bench/Analysis/<run>/...` (not in the repo);
  logs under `Analysis/_grand_logs/`.
- **Hybrid angle chain (2026-07-12 fleet reprocessing):** waveform scripts `26→27→28`,
  `31`, `33`, `34` (`--veto=50`) on M3 v2 rays + decoded_root pulled from lxplus. Quoted
  angle = `34`'s self-trained lt5/plateau σ68 (`--dump-events` persists per-event
  predictions). Measured charge-sharing constants c1/c2 = A(±1)/A(0), A(±2)/A(0) from
  vertical tracks (script 26), per FEU (X/Y), edited into the `CSHARE` dicts of 27/28/31/33
  before each detector's run (full record: `REPROCESSING_CONSTANTS.md`):
  - det3 (design): X (FEU7) 0.449/0.052, Y (FEU8) 0.516/0.151; calib b_x/b_y +0.033/+0.029
  - det2: X 0.43, Y 0.52 (≈det3, design property)
  - det4 (E): X (FEU6) 0.370/0.047, Y (FEU8) 0.432/0.112; b_x/b_y +0.064/+0.061
  - det6 (C): X (FEU3) **0.250**/0.048, Y (FEU4) 0.451/0.204 — X-plane sharing ~½ det3
  - det7 (D): X (FEU6) **0.263**/0.058, Y (FEU8) 0.513/0.220 — X-plane sharing ~½ det3
  The low X-plane c1 on boards C/D is measured and consistent (not averaged into det3's).
  det4's decoded FEU8 has a ~0.3% tail of malformed multi-frame events (non-32-sample);
  the waveform scripts now skip them (size-guard on `reshape(32,512)`).

## 5. Open follow-ups
1. **det7 saturation veto** — re-measure efficiency with saturated hits removed.
2. **det4 gain** — raise gain (HV / threshold / gas) so clusters reach ≥3 strips.
3. **6-23 M3 reference** — diagnose the degradation (`TODO_m3_reference_6-23.md`).
4. **det6/det7 HV gap** — re-run the HV PDF once the 5 stalled mid-range points
   (455–490 V) of the 6-26 `hv_scan` run finish decoding (cheap rerun).

*Done:* det6/det7 re-scanned at lower resist HV (6-26 `hv_scan` run) → optima
480 V / 440 V found.
