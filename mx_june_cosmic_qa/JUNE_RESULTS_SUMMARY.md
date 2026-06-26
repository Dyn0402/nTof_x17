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

---

## 1. Per-detector overview (best high-stats subrun)

| Det | Run / subrun used | Clean M3 rays | Efficiency (≤5 mm) | Fired any strip | Core σ | Verdict |
|----:|---|---:|---:|---:|---:|---|
| **2** | 6-22 overnight / long_run | 34.9k | **76.4 %** | 99.7 % | 0.84 mm | Healthy |
| **3** | 6-22 overnight / long_run | 31.1k | **80.7 %** | 98.1 % | 0.69 mm | Best performer |
| **4** | 6-24 daytime / long_run | 39.6k | **8.4 %** | 50.6 % | 1.24 mm | Gain-limited |
| **6** | 6-26 overnight / long_run (2 files) | 9.7k | **63.8 %** | 93.7 % | 0.69 mm | Healthy |
| **7** | 6-26 overnight / long_run (2 files) | 10.6k | **41.1 %** | 93.9 % | 1.30 mm | Saturation tail |

Alignment converged sub-mm with θ≈90° and z near nominal for every detector above
(seeded per run from the long_run subrun).

### Notes per detector
- **det2 / det3** — healthy detectors: high, spatially-uniform efficiency at sub-mm
  resolution. det3 is the best of the batch.
- **det4** — gain-limited. Fires on ~51 % of muons but the loss is dominated by
  `hit_no_reco` (38.8 %) + silent (49.4 %): clusters rarely reach the ≥3 strips needed
  to reconstruct. Same pathology the old det1 had → an HV/threshold/gas (gain) issue,
  not a dead detector.
- **det6** — healthy (63.8 %, 0.69 mm). NB its `short_run` gave a misleading 9.7 %
  (low-stats / unsettled); the long_run is the real picture. Always prefer long_run here.
- **det7** — reconstructs with a good core (σ≈0.9–1.1 mm per axis) but has a heavy
  **outlier tail**: median radial residual 1.84 mm vs **mean 23.9 mm**, `reco_far` ≈40 %.
  Alignment is healthy (z=714 mid-window, θ=90.1°) — the tail is a *reconstruction*
  problem concentrated in the **Y plane (FEU 8)**: a band of **saturated hits (~4000 ADC,
  near the 4095 ceiling) across all strips** forms fake clusters the micro-TPC fit
  sometimes selects. This is a detector/readout condition (HV / sparking / common-mode),
  not an analysis bug. A saturation veto (`local_max ≳ 3900`) should recover a truer,
  higher efficiency. **Open follow-up.**

---

## 2. Resist-HV scans

Efficiency vs resist HV, integrated over a fixed per-detector active box; alignment
seeded from each run's long_run subrun and re-translated per HV point.

| Det | Scan (drift) | Peak efficiency | at HV | Behaviour |
|----:|---|---:|---:|---|
| **2** | 6-22 (1000 V), 450–525 V | 77.8 % | ~490 V | plateau, sparks-off above ~510 V |
| **3** | 6-22 (1000 V), 450–525 V | 81.1 % | ~485 V | best plateau, σ≈0.7 mm at optimum |
| **6** | 6-26 (700 V), 505–530 V | 56.7 % | 505 V (lowest) | monotonic falloff → optimum **below** range |
| **7** | 6-26 (700 V), 480–505 V | 42.6 % | 480 V (lowest) | monotonic falloff → run **lower** |

In the 6-22 scan, det2 (resist ch 3:4) and det3 (3:3) were stepped **together**. In the
6-26 scan det6/det7 were stepped at **different** voltages (encoded in the subrun name).

**Reading:** `any_hit` stays ~flat near 100 % while reco-efficiency falls at high HV →
the high-voltage losses are sparking-induced reconstruction failures, not the detector
going silent. Resolution mirrors it: best near the efficiency optimum, degrading into
the sparking regime.

**Takeaways:**
- det2 / det3 optimum ≈ **485–490 V** (drift 1000 V).
- det6 / det7 were **scanned too high** — both fall monotonically across their range;
  they should be re-scanned at lower resist HV to find the plateau.

---

## 3. Excluded / not measured

- **6-23 overnight (det3 + det4)** — both the long-run alignment and the HV scan are
  **blocked by a degraded M3 reference** (~3.9 % clean tracks; alignment rails to
  z=569/411 mm vs nominal 232). No reliable track reference → no trustworthy efficiency.
  See `TODO_m3_reference_6-23.md`. det3 and det4 are characterised instead from the
  6-22 and 6-24 runs respectively.
- **6-25 det3 long-run** and other raw-only subruns were not decoded at analysis time.

---

## 4. Provenance & reproduction

- Run registry: `qa_config.py` keys `g_det2 g_det3 g_det4 g_det6_long g_det7_long`
  (+ `g_det6 g_det7` short_run, and per-subrun variants).
- Per-detector overview: `run_full_june_qa.sh` (orchestrator) → `build_final_pdf.py`.
  Regenerate just the PDF: `../venv/bin/python build_final_pdf.py`.
- HV scans: `run_hv_scans.sh` → `10_hv_scan_efficiency.py` (per detector) →
  `build_hv_scan_pdf.py`.
- Analysis outputs live under `~/x17/cosmic_bench/Analysis/<run>/...` (not in the repo);
  logs under `Analysis/_grand_logs/`.

## 5. Open follow-ups
1. **det7 saturation veto** — re-measure efficiency with saturated hits removed.
2. **Re-scan det6 / det7 at lower resist HV** (next beam session) to bracket their plateau.
3. **det4 gain** — raise gain (HV / threshold / gas) so clusters reach ≥3 strips.
4. **6-23 M3 reference** — diagnose the degradation (`TODO_m3_reference_6-23.md`).
