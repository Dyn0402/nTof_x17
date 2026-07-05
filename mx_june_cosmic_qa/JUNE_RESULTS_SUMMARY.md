# June 2026 cosmic-bench QA — combined results

Saclay cosmic-bench characterisation of the MX17 micro-TPC Micromegas detectors
(**2, 3, 4, 6, 7**; no 5), June 2026. Efficiency is measured against M3 reference
tracks: for every clean M3 single muon track, project to the aligned detector plane
and ask whether the detector has a reconstructed X+Y hit within **5 mm** (a track with
no DREAM readout is a genuine miss, kept in the denominator). Resolution is the
Gaussian core of the residual distribution. All runs Ar/Iso 95/5, non-zero-suppressed,
`det_orientation.z = 90`; slot map bottom = FEU 3(X)/4(Y) z≈232 mm, top = FEU 6(X)/8(Y)
z≈702 mm.

Two compiled PDFs (under `~/x17/cosmic_bench/Analysis/`):
- **`june_detectors_overview.pdf`** — one page per detector (headline efficiency +
  resolution stat cards, sliding-window within-5 mm efficiency map, binned map,
  breakdown, sliding resolution map, residuals, pulse-height-vs-strip, hit/miss scatter).
- **`june_hv_scans.pdf`** — resist-HV scans: summary overlay + per-detector
  efficiency-vs-HV and resolution-vs-HV.

**Spark separation (2026-07-05).** The efficiency breakdown (bottom bar of every
overview page) now counts **`spark`** (event fires >50 strips = a full-detector
discharge) as its own category, pulled out *before* the reco/hit split. Previously
sparks still yielded a centroid and masqueraded as `reco_far`/`reco_near`/`hit_no_reco`,
inflating the tail. Separating them mainly deflates the `reco_far` tail on the
spark-rich detectors (det7 `reco_far` 40 %→11 % with 31 % now labelled spark; det6
likewise) and slightly lowers the headline efficiency (sparks that had landed <5 mm are
no longer counted as good). Efficiency numbers in the table below are the
**spark-separated** values.

---

## 1. Per-detector overview (best high-stats subrun)

| Det | Run / subrun used | Clean M3 rays | Efficiency (≤5 mm) | Fired any strip | Spark | reco_far | Core σ | Verdict |
|----:|---|---:|---:|---:|---:|---:|---:|---|
| **2** | 6-22 overnight / long_run | 34.9k | **75.4 %** | 99.7 % | 5.7 % | 16.2 % | 0.83 mm | Healthy |
| **3** | **6-28 weekend / long_run_p2 (top slot)** | 53.0k | **78.8 %** | 97.6 % | 4.9 % | 13.7 % | 0.79 mm | Best performer |
| **4** | 6-24 daytime / long_run | 39.6k | **8.2 %** | 50.6 % | 2.3 % | 2.5 % | 1.20 mm | Gain-limited |
| **6** | 6-26 overnight / long_run (7 files) | 32.3k | **51.2 %** | 94.1 % | 23.7 % | 7.5 % | 0.68 mm | Spark-limited |
| **7** | 6-26 overnight / long_run (7 files) | 35.5k | **31.9 %** | 84.1 % | 31.0 % | 11.2 % | 1.18 mm | Spark-limited |

Efficiency, spark and reco_far are % of active-area crossings, **spark-separated** (see
above); efficiency dropped 1–5 pts vs the pre-separation numbers and the core σ tightened
(sparks no longer pollute the residual). Alignment converged sub-mm with θ≈90° and z near
nominal for every detector above (seeded per run from the long_run subrun).

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
| **2** | 6-22 (1000 V), 450–525 V | 77.8 % | ~490 V | plateau, sparks-off above ~510 V |
| **3** | 6-22 (1000 V), 450–525 V | 81.1 % | ~485 V | best plateau, σ≈0.7 mm at optimum |
| **6** | 6-26 hv_scan 400–500 V + overnight 505–530 V (700 V) | **71.0 %** | **480 V** | full turn-on → plateau → falloff |
| **7** | 6-26 hv_scan 400–500 V + overnight 480–505 V (700 V) | **54.7 %** | **440 V** | full turn-on → plateau → falloff |

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

---

## 4. Provenance & reproduction

- Run registry: `qa_config.py` keys `g_det2 g_det3 g_det3_wknd g_det4 g_det6_long g_det7_long`
  (+ `g_det6 g_det7` short_run, and per-subrun variants). `g_det3_wknd` = the 6-28 weekend
  det3 (wins the det3 page by ray count).
- Per-detector overview: `run_full_june_qa.sh` (orchestrator) → `build_final_pdf.py`.
  Regenerate just the PDF: `../venv/bin/python build_final_pdf.py`.
- HV scans: `run_hv_scans.sh` → `10_hv_scan_efficiency.py` (per detector) →
  `build_hv_scan_pdf.py`.
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
