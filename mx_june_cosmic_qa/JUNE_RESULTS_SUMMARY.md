# June 2026 cosmic-bench QA — combined results

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
the σ of (θ_ref − θ_det) at the best drift velocity, reported only where the θ_det–θ_ref
correlation is strong (r ≥ 0.7). All runs Ar/Iso 95/5, non-zero-suppressed,
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

| Det | Run / subrun used | Clean M3 rays | Efficiency (≤5 mm) | Core σ | Angular σ | Spark | Verdict |
|----:|---|---:|---:|---:|---:|---:|---|
| **3 (A)** | 6-27/28 weekend / long_run_p2 (top slot) | 52.0k | **88.8 %** | **0.63 mm** | **2.04°** | 4.4 % | Best performer |
| **2 (B)** | 6-22 overnight / long_run | 52.9k | **87.0 %** | **0.64 mm** | **2.15°** | 5.5 % | Healthy |
| **6 (C)** | 6-26 overnight / long_run | 15.1k | **55.7 %** | **0.59 mm** | **3.15°** | 24.0 % | Spark-limited |
| **7 (D)** | 6-26 overnight / long_run | 20.5k | **36.4 %** | **0.86 mm** | **2.50°** | 31.9 % | Spark-limited |
| **4 (E)** | 6-24 daytime / long_run | 2.8k | **10.3 %** | **0.91 mm** | **2.49°** | 2.6 % | Gain-limited |

Efficiency and spark are % of active-area crossings; core σ is the residual Gaussian
core; angular σ is the micro-TPC θ resolution (reported where r ≥ 0.7). Alignment
converged sub-mm with θ≈89.3–89.8° and z≈710–714 mm for every detector.

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
- v2 vs pre-v2 (same runs): efficiency up (**det2 75.4→87.0 %**, det6 51.2→55.7 %, det7
  31.9→36.4 %, det4 8.2→10.3 %) and core σ tighter (det2 0.83→0.64, det7 1.18→0.86 mm) —
  the cleaner v2 reference removes spurious "misses" and sharpens the alignment.
- **det6 (C) angular resolution** (3.15°) is now measurable: its θ_det–θ_ref correlation
  is real (r≈0.86), even though the micro-TPC time-fit v_drift rails low (~25 µm/ns, the
  known ~20 % low bias of the time estimator); the resolution σ is unaffected by that bias.

### Notes per detector
- **det2 / det3** — healthy detectors: high, spatially-uniform efficiency at sub-mm
  resolution. det3 is the best of the batch. The headline det3 is now the **6-28 weekend
  run** (top slot, FEU7/8, z702; 53.0k rays, 79.7 %, silent only 2.4 %); the earlier 6-22
  det3 (bottom slot) gave a consistent **80.7 %** — det3 is reliably ~80 % in either slot.
- **det4** — gain-limited. Fires on ~51 % of muons but the loss is dominated by
  `hit_no_reco` (38.8 %) + silent (49.4 %): clusters rarely reach the ≥3 strips needed
  to reconstruct. Same pathology the old det1 had → an HV/threshold/gas (gain) issue,
  not a dead detector.
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
  `chi2_cut=5` (NClus≥3 is the M3RefTracking default). Caches (`event_results*.pkl`) are
  det-hit-only and reused. venv is **`.venv`** (repo root).
- Per-detector overview: `.venv/bin/python build_final_pdf.py` (default keys headline the
  weekend det3; page 1 is the fleet summary). HV scans: `10_hv_scan_efficiency.py` per
  detector → `build_hv_scan_pdf.py g_det2 g_det3 g_det6_hv g_det6_long g_det7_hv g_det7_long`.
- Analysis outputs live under `~/x17/cosmic_bench/Analysis/<run>/...` (not in the repo);
  logs under `Analysis/_grand_logs/`.

## 5. Open follow-ups
1. **det7 saturation veto** — re-measure efficiency with saturated hits removed.
2. **det4 gain** — raise gain (HV / threshold / gas) so clusters reach ≥3 strips.
3. **6-23 M3 reference** — diagnose the degradation (`TODO_m3_reference_6-23.md`).
4. **det6/det7 HV gap** — re-run the HV PDF once the 5 stalled mid-range points
   (455–490 V) of the 6-26 `hv_scan` run finish decoding (cheap rerun).

*Done:* det6/det7 re-scanned at lower resist HV (6-26 `hv_scan` run) → optima
480 V / 440 V found.
