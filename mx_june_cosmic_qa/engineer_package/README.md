# MX17 detector performance package — for the construction conference talk

Material prepared from the June 2026 Saclay cosmic-bench campaign for the
engineer's ~20-minute talk on the construction of the MX17 Micromegas
detectors. Everything here is self-contained: you do not need access to the
analysis code or data to use it.

**Status: preliminary** (mature analysis, pre-publication). Please keep the
caveats at the bottom of this file attached to any quoted number.

## What's in this directory

| Item | What it is |
|---|---|
| `report/main.pdf` | Comprehensive plain-language report: how the detectors were tested, how they work in micro-TPC mode, and every performance result with figures. **Start here.** |
| `figures/` | Every figure as standalone PNG **and** PDF, named by topic (`10-det3-…`, `20-hv-…`). `figures/FIGURE_GUIDE.md` has a one-line description + suggested slide caption for each. |
| `event_displays/` | Fresh presentation-quality single-muon event displays (PNG + PDF) rendered from the raw detector 3 data, plus a spark event for contrast. `DISPLAYS.md` lists captions. |
| `event_displays_3d/` | 3-D single-event displays: the drift-time charge cloud with the independent M3 reference track drawn as a line (static PNG + rotating GIF per event). `DISPLAYS_3D.md` lists them. The polished figures used in the report/deck are `figures/07` (hero) and `figures/08` (gallery); regenerate all via `make_event_displays_3d.py --figures`. |
| `slides/mx17_detector_performance.pptx` | Starter slide deck (~19 slides) you can pull from or present directly; every slide has speaker notes. |
| `source_reports/` | The detailed technical write-ups behind the headline numbers (time resolution, X/Y charge balance, spark studies, per-detector overview, HV scans). For depth, not for slides. |

## The detectors, as tested

- Resistive-strip Micromegas, 2-D strip readout: 512 X-strips + 512 Y-strips,
  **0.78 mm pitch**, ~**40 × 40 cm²** active area; pixelated top layer routes
  avalanche charge to both strip layers.
- **30 mm** conversion/drift gap, operated as a **micro-TPC** (electron drift
  time = depth), drift field ≈ 330 V/cm (1000 V), amplification 440–490 V.
- Gas: Ar/isobutane 95/5, atmospheric, flushed.
- Readout: DREAM front-end, 60 ns sampling, 1.92 µs window (full waveforms).
- Bench reference: multiplexed-Micromegas tracking telescope ("M3") +
  scintillator-paddle trigger. All performance numbers are defined against
  independently reconstructed reference muon tracks.

## Quotable performance numbers

Detector letters A–E are the experiment labels; mx17 numbers are the chamber
serials. Detector **A (mx17_3)** is the reference/best chamber and is the one
most results are quoted on.

### Headline (Detector A at operating point, 490 V / 1000 V)

| Quantity | Value | Notes |
|---|---|---|
| Muon detection efficiency (active area) | **88.8 ± 0.2 %** | reconstructed within 5 mm of the reference track, **operating** number (in-spark coincidences folded in). See the loss budget below. |
| — intrinsic (spark-free events) | **92.9 ± 0.2 %** | the 88.8 % with the 4.4 % in-spark coincidence removed: 92.9 × (1 − 0.044) = 88.8 |
| — core (>25 mm from frame) | **~96 %** | edge band is a fringe-field effect, ~25 mm wide |
| — at a 10 mm match window | **~95 %** | the >5 mm residue is a near-miss position tail, almost all inside 10 mm |
| Muons leaving no signal at all | **0.2 %** | the chamber is essentially never blind (produces a signal for 99.8 %) |
| Position resolution | **σ ≈ 0.6–0.65 mm** | **sub-pitch** (pitch 0.78 mm) thanks to resistive charge sharing; includes telescope error → conservative |
| Track-angle resolution (single chamber) | **≈1.8° (σ68)** | hybrid tracking: 1.75° near-vertical band / 1.86° plateau, at ALL angles incl. head-on; bias 0.19°; 97–99 % of tracks usable |
| Time resolution (detector) | **33 ns ≈ 1 mm of drift** | telescope-free (two orthogonal layers timing the same electrons); 29 ns after walk correction |
| Absolute event time vs trigger | **σ68 = 37.7 ns** | detector-dominated budget |
| Drift velocity (1000 V) | **34 ± 1.5 µm/ns** | measured; matches Magboltz for Ar/iso 95/5 + ~1 % H₂O |
| Spark rate at optimum | **4.4 % of crossings** (0.33 Hz) | Poisson (random), muon-induced (~4×), edge-seeded |
| Dead time after a spark | **none measurable** | efficiency and gain flat vs time-since-spark; discharges self-quench within the 2 µs window |
| X/Y charge balance (pixel layer) | **f = 0.49–0.53, σ68 ≈ 0.07** | flat across position & angle; X–Y charge correlation r ≈ 0.9 |
| Charge sharing to neighbour strip | **45–52 %** (+ ~70 ns delay) | same on A and B → design property; enables sub-pitch resolution |

### Where the efficiency loss goes (Detector A, 52,006 telescope muons)

This is the loss budget behind the 88.8 % (figure `21-det3A-efficiency-breakdown`).
The point of it: the ~11 % off 100 % is **almost never the chamber failing to
see the muon** — it is a spark coincidence plus an edge/near-miss *position* tail.

| Where the muon went | % | What it is |
|---|---:|---|
| Reconstructed within 5 mm | **88.8 %** | the efficiency |
| Detected, point >5 mm off track | 6.5 % | valid X+Y point formed, but off the telescope track — near-misses (5–10 mm), competing clusters, edge distortion, sub-veto discharges. **Not blindness.** |
| Sparked during the crossing | 4.4 % | full-detector discharge coincided with the muon; self-quenching, **zero dead time afterward** |
| Fired, no valid point | 0.1 % | strips fired but no X+Y point could be built |
| Silent (no signal at all) | 0.2 % | genuine blindness — essentially never |

Reads as: **the chamber produces a signal for 99.8 % of muons and a track point
for 95.3 %.** Of the muons it reconstructs, 93 % land within 5 mm; loosening the
match to 10 mm pulls almost all of the >5 mm tail back in (~95 %). The only
irreducible detection loss is the 0.2 % silent fraction. The whole-area 88.8 %
rises to ~96 % in the core because the >5 mm tail concentrates in the outer
~25 mm fringe-field band.

### Fleet comparison (identical method, best June runs)

| Det | Serial | Efficiency (≤5 mm) | Position σ | Angle σ68 (hybrid) | Spark % of crossings | Verdict |
|---|---|---|---|---|---|---|
| A | mx17_3 | 88.8 % | 0.63 mm | 1.8° | 4.4 % | Best performer |
| B | mx17_2 | 87.0 % | 0.64 mm | 2.5° | 5.5 % | Healthy |
| C | mx17_6 | 55.7 % | 0.59 mm | 4.1° † | 24.0 % | Spark-limited (operating point) |
| D | mx17_7 | 36.4 % | 0.86 mm | 3.6° | 31.9 % | Spark-limited (operating point) |
| E | mx17_4 | 20.0 % | 0.89 mm | 2.7° | 3.4 % | Gain-limited (fires on ~70 % of muons but clusters too small to reconstruct) |

Angle σ68 is from the current-best **hybrid tracking** (|θ|<5° band, self-trained,
odd-eid holdout), now run **fleet-wide** (2026-07-12). A/B (1000 V) reach ≈1.8/2.5°;
the 700 V spark-limited chambers C/D and the gain-limited chamber E are degraded but
measured: **C 4.1°, D 3.6°, E 2.7°** (coverage 98/90/81 %). Plateau (|θ|>8°) σ68:
A 1.9°, B 2.2°, E 4.1°, D 4.9°.
† **C is low-angle-only:** its X-plane (FEU 3) develops no micro-TPC drift-time
structure (the board-C mesh defect), so the time-fit plateau is unusable and only the
head-on signature-regression band is quoted. E, despite the lowest efficiency, has the
cleanest micro-TPC angles of C/D/E (both planes work).

HV-scan optima (amplification): A 480–490 V, B 480–490 V (drift 1000 V);
C 480 V, D 440 V (drift 700 V). Peak scan efficiencies: A 90.8 %, B 89.9 %,
C 76.2 %, D 63.1 %. C/D's low table numbers are an operating-point effect
(high spark fraction at their June settings), not construction defects — when
they fire, they reconstruct with the same sub-mm quality as A/B.

### Construction-relevant findings worth highlighting

1. **The resistive spark protection works, verified at the waveform level**:
   a discharge is a fast, global, self-quenching baseline step, over within
   2 µs, localized (edge-seeded), non-propagating — and causes **zero measured
   dead time**.
2. **The pixelated charge-routing layer works**: X/Y charge split is flat and
   near 50/50 everywhere (per-chamber offset ±0.04 = assembly-level effect;
   the *constancy* is the design metric).
3. **Charge sharing is a design property, not a defect**: identical on the two
   chambers measured, and it is exactly what makes the position resolution
   beat the strip pitch.
4. **Sub-pitch position, ~2° angles, and ~1 mm-equivalent timing from a single
   chamber** — the 30 mm drift gap turns each chamber into a self-contained
   3-D tracker (micro-TPC), which the experiment exploits.
5. **Gas quality is observable in the data**: water content sets the drift
   speed (A visibly dried from >3 % to ~1 % H₂O over a week), oxygen from air
   eats the drifting charge (attachment length ~15 mm at ~1 % air). Gas
   tightness and flushing matter, and the chambers monitor their own gas
   in situ.
6. **Edges behave as expected for a bounded drift volume**: fringe-field band
   of ~25–40 mm (efficiency turn-on 0→96 % over 25 mm, ~3° angle tilt,
   spark seeding); everything inboard is uniform.

## Caveats (attach to any quoted number)

1. **Preliminary** — June 2026 bench campaign, pre-publication.
2. Position resolutions are **detector ⊕ reference telescope** (not yet
   deconvolved) → quoted values are conservative upper limits.
3. Results are for **cosmic muons** at the specific gas/HV listed.
4. C, D, E numbers reflect their June operating points; HV scans show better
   is available (C/D) and E's issue is gas gain, not a dead chamber.

## Provenance

Produced from `mx_june_cosmic_qa/` in the nTof_x17 analysis repository
(July 2026): `JUNE_RESULTS_SUMMARY.md`, `PAPER_STATUS.md`,
`MICROTPC_RUNBOOK.md`, `DET3_WEEKEND_ANALYSIS.md`, and the standalone reports
copied into `source_reports/`. Reference tracking: M3 v2 reprocessing;
reference-track recipe NClusX/Y = 4 & χ² < 1.0 (since 2026-07-14; was NClusX/Y ≥ 3 &
χ² < 5 -- see `../det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`); efficiency
counted within 5 mm.

**2026-07-14 status:** `source_reports/june-detectors-overview.pdf` and
`source_reports/june-hv-scans.pdf` (all 5 detectors now, including det6/det7's two
scan segments each) are regenerated on the new recipe. `report/main.tex` is rebuilt:
all 34 numbered figures that depend on the M3 recipe (10-18, 20-26, 30-34, 44) were
regenerated/copied fresh and every headline number in the prose, keyboxes, fleet
table, and quotable-numbers table was reconciled against the new measurements
(figures 40-43, 50-52, 60-61 unaffected by design or already fresh; 70-73 and 80
predate the recipe change and are flagged as such in the report). `make_slide_deck.py`
below is also rebuilt (18 slides) with every headline number reconciled the same way,
including the operating-vs-spark-free efficiency distinction. Not re-run: the
det3-frozen-model transfer cross-check and the det2/4/6/7 measured-passivation outline.
Event displays rendered from run `mx17_det3_saturday_scan_6-27-26 /
long_run_resist_490V_drift_1000V` raw waveforms.
