# REPROCESSING PLAN — unify the June analysis on the current-best chain and purge stale results

> **2026-07-14: M3 reference recipe changed to χ²<1.0 & NClus=4** (was χ²<5 & NClus≥3,
> row 1 of the table below) — the old recipe was reference-limited, not chamber-limited
> (`det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`). Centralised in `qa_config.py`
> (`M3_CHI2_CUT`, `M3_MIN_NCLUS`); every table row below that touches M3 tracks inherits
> the new recipe. Alignment + efficiency/residual already re-run fleet-wide (`JUNE_RESULTS_
> SUMMARY.md` 2026-07-14 changelog); the waveform-dependent rows (unsharing, hybrid angle
> tracking, time resolution) are being re-run under the new recipe now.

*Written 2026-07-12 for a follow-up session/model. Mission: the June cosmic
campaign accumulated several generations of analysis (pre-M3-v2, pre-unsharing,
pre-hybrid, pre-walk-correction). The best chain is mature and validated, but
old-generation numbers and figures are still mixed into summaries, overview
PDFs, and per-detector trees. Your job: (1) reprocess every detector with the
best chain end-to-end, (2) re-quote every headline number from that chain
only, (3) archive/remove the superseded outputs so nothing stale can be picked
up again, (4) update every document and generated artifact to be internally
consistent. Data campaign is FROZEN — no new runs, no re-decoding beyond
pulling existing decoded files from lxplus.*

**Read first, in this order:** `MICROTPC_RUNBOOK.md` (how to run everything;
constants; gotchas), `JUNE_RESULTS_SUMMARY.md` (current fleet quotes and run
choices), `PAPER_STATUS.md` (per-topic status + verified figure inventory),
`DET3_WEEKEND_ANALYSIS.md` (det3 deep-dive), `paper_plans/PLAN_37/40/41`
(still-open work), and the auto-memory notes. Environment: venv is
**`nTof_x17/.venv`**; run all scripts FROM `mx_june_cosmic_qa/` as
`../.venv/bin/python <script>.py <run_key> --veto=50`.

---

## 1. The golden chain (the ONLY analysis allowed in quoted results)

Every headline number must come from this chain. Anything else is a
cross-check or an intermediate.

| Quantity | Golden method | Script(s) | Needs waveforms? |
|---|---|---|---|
| Reference tracks | M3 **v2** (`m3_tracking_root_v2/`), recipe **NClusX=4 & NClusY=4 & χ²<1.0** (`qa_config.M3_CHI2_CUT`/`M3_MIN_NCLUS`, since 2026-07-14; was χ²<5 & NClus≥3) | loader default | no |
| Alignment + cache | `03_alignment_and_tpc.py <key> --veto=50` (veto50 cache everywhere) | 03 | no |
| Efficiency + breakdown + maps | within-5 mm vs M3 v2, **live-range guard** on, **spark separated** (crossing-based spark fraction is the quotable one) | 08, 09, 12 | no |
| Position resolution | residual Gaussian core (03/08 outputs) for the standard quote **plus** the estimator benchmark (early-charge centroid / COMBO) for the best-value quote | 36 | **yes** |
| Charge sharing constants | measured per detector from vertical tracks (c1, c2, Δt) | 26 | **yes** |
| Unsharing + kernel | α=0.5 kernel applied; before/after ladders | 27 | **yes** |
| Angle calibration | additive tan-space constants b per plane, per detector/run-era | 28 | **yes** |
| micro-TPC scoreboard | segment ladder, σ68, coverage | 31 | **yes** |
| Head-on features | waveform signature CSV (feeds hybrid) | 33 | **yes** |
| **Angular resolution (QUOTED)** | **hybrid tracking** (time-fit above ~5°, signature regression below), odd-eid holdout; per-detector self-trained + det3-frozen transfer as cross-check | 34 | no (uses 31+33 CSVs) |
| Drift velocity | v_geom (extent-slope/T_sat) per drift point; hybrid drift scan as telescope-free cross-check | 21/23, 35 | no |
| Gas ID / attachment | Magboltz closure (v→H₂O, λ→O₂); fleet survey | 15, 17–19, 30 | no |
| Time resolution | ftst-verified absolute + inter-plane method, **walk-corrected** | 42 | **yes** |
| X/Y charge balance | f = qX/(qX+qY), three charge proxies | 38 | no |
| Sparks | Poisson/muon-induced/edge (spark analysis dirs), dead-time null (39), waveform anatomy (40) | 39, 40 | 40: yes |

**Never quote as results:** script 03's `angular_resolution.{json,png}` and
`angle_correlation_*.png` (production time-fit, no unsharing — superseded by
34); script 14's ridge v_drift (documented ~10–20 % low bias — keep only as
the cautionary-tale reference); anything from a no-veto (`alignment_tpc/`)
tree; anything from `*_prev2_backup/`; the un-vetoed `event_results.pkl`
numbers.

**Definitions that must not drift:** efficiency = X+Y reco hit within 5 mm of
the projected M3 v2 ray, full-active-area denominator including silent events,
live-range guard on; spark % = crossing-based; angular σ = σ68 of
(θ_hybrid − θ_ref) with stated coverage; position σ still includes the M3
pointing term until PLAN_37 (deconvolution) is executed — every position quote
carries that caveat.

## 2. What already IS golden (do not redo, just verify)

- **det3 / A** — the reference detector. sat_det3
  (`mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V`) has the
  full chain 26→28, 31–36, 38, 40, 42 done on M3 v2; g_det3_wknd
  (`mx17_det3_p2_det1_overnight_6-27-26/...`) holds the headline efficiency
  (88.8 %) + spark dead-time null. Verify outputs exist and are post-7-06
  (M3 v2) — then leave alone.
  - **Efficiency is ORTHOGONAL to the hybrid tracker — do not "reprocess it
    with hybrid".** Efficiency (08/09/12) is a hit-within-5 mm count against the
    M3 ray; it does not use angle reconstruction at all. Hybrid tracking (34)
    only fixed the *angular* resolution on near-vertical tracks. The efficiency
    maps/breakdowns are already correct on the M3 v2 basis. When regenerating
    docs, keep the three efficiency definitions straight and labelled: **88.8 %**
    operating (in-spark coincidence folded in; the headline), **92.9 %** intrinsic
    (spark-free crossings), **~96 %** core (>25 mm from the frame). The >5 mm
    residue is `reco_far` — a near-miss/edge *position* tail (recovers to ~95 %
    at a 10 mm match), NOT blindness (silent is 0.2 %). The engineer package's
    `make_efficiency_breakdown.py` renders this budget in plain language.
- **det2 / B** — 29 (chain validation), hybrid self+transfer (34), time
  resolution (42), charge balance (38) done on `o22_long_det2`
  (`mx17_det2_det3_overnight_6-22-26/longer_run`). Decoded waveforms are
  LOCAL for this run and for sat_det3.
- HV scans (10 + build_hv_scan_pdf), efficiency chain (08/09/12) for all five
  detectors, fleet gas survey (30) — all already v2. Efficiency numbers in
  `JUNE_RESULTS_SUMMARY.md` §1 are current.

## 3. The actual gap: det4, det6, det7 have NO hybrid-generation angle analysis

Their quoted angular resolutions (det6 3.15°, det7 2.50°, det4 2.49° in
`JUNE_RESULTS_SUMMARY.md`) are **script-03 production time-fit numbers** —
the method the hybrid work superseded. The engineer package now quotes "—"
for C/D/E angles pending this work. Per detector:

### 3.1 Data staging (decoded waveforms from lxplus)

Scripts 26/27/28/33 (and 36/40/42 if you extend those too) need
`decoded_root`. Locally present ONLY for sat_det3 and det2 `longer_run`. For
det4/6/7 pull from lxplus (user dneff):

```bash
# pattern (see MICROTPC_RUNBOOK.md §0b); ~260 MB per file per FEU — pull only needed FEUs
rsync -av lxplus:~dneff/x17/cosmic_bench/june_tests/<run>/<subrun>/decoded_root/'*_<FEU>.root' \
      ~/x17/cosmic_bench/<bench_dir>/<run>/<subrun>/decoded_root/
```

Runs/subruns (registry keys in `qa_config.py` — confirm before use):
- det6: `mx17_det6_det7_overnight_6-26-26/long_run` (key ~`g_det6_long`)
- det7: same run, `mx17_7` (key ~`g_det7_long`)
- det4: `mx17_det4_day_6-24-26/long_run` (key ~`g_det4`)

Find each detector's FEU pair from the run's `run_config.json` /
`autodetect_feus` (FEU↔detector mapping changes between runs — NEVER assume).
Remember drift HV differs: det6/7 ran at 700 V, det4 read from
`hv_monitor.csv` (channel map in runbook §2) — decode it, never assume 1000 V.

### 3.2 Per-detector chain (template = `29_det2_validation.py`)

For each of det6, det7, det4, in order:

1. Confirm veto50 cache + alignment exist (03 already run for all).
2. `26_unsharing_analysis.py <key> --veto=50` → measures c1/c2/Δt.
   **Record the constants.** Needs ≥ a few hundred vertical-track leads.
   - det4 caveat: gain-limited (clusters rarely ≥3 strips) — 26 may not find
     enough leads. If so, STOP for det4 and record "hybrid not measurable at
     June operating point (insufficient gain)" — that is the honest result;
     do NOT silently fall back to det3's constants.
   - det6/7 caveat: 24–32 % of crossings are sparks; the veto50 cache
     excludes them, but expect reduced usable statistics; sub-veto discharges
     (30–50 strips) can pollute vertical-track selection — check 26's lead
     QA plots before trusting constants.
3. Edit the hardcoded `CSHARE` dict in **27 and 28** for this detector's FEUs
   (see runbook §5 — keys are FEU numbers). Run 27 (kernel/ladders), then 28
   (calibration constants b per plane; gas-quality dependent, so per run).
4. `31_microtpc_metrics.py <key> --veto=50` → segments CSV.
5. `33_headon_tracks.py <key> --veto=50` → headon features CSV.
6. `34_hybrid_tracking.py <key> --veto=50` → the quotable angular numbers
   (low-angle band σ68 + coverage, plateau σ68 + coverage), self-trained with
   odd-eid holdout. ALSO run with `--model=` pointing at the frozen det3
   model (transfer) as the overfit cross-check — report both, quote
   self-trained.
7. Sanity: compare 34's implied velocity behaviour and 31's ladder quality
   against det2's write-up (29). Disagreement between unshared time-fit v and
   v_geom beyond ~3 % means something is wrong (fit windows, FEU map, HV) —
   investigate, don't ship.

Persist per-event hybrid predictions (34 currently only writes the summary
CSV + dashboard: extend it with a `--dump-events` CSV output rather than
recomputing ad hoc; keep the change backward-compatible).

### 3.3 Optional same-pass extensions (cheap once decoded_root is staged)

While the waveforms are local per detector, also run: 36 (position estimator
benchmark), 42 (time resolution) — currently det3/det2-only results that the
paper quotes as "replicated on det2"; extending to the full fleet makes the
fleet table uniformly golden. Do this if time permits; the angle gap
(§3.2) is the priority.

## 4. Re-quote + regenerate every derived artifact

Single source of truth: update `JUNE_RESULTS_SUMMARY.md` §1 table first, then
propagate. Order:

1. **`JUNE_RESULTS_SUMMARY.md`** — replace the "Angular σ" column with hybrid
   σ68 (+ coverage) for all detectors measured; move the old time-fit values
   into a clearly-marked "superseded (pre-hybrid method)" footnote or drop
   them; state per-detector c1/c2 and calibration constants in the provenance
   section.
2. **`MICROTPC_RUNBOOK.md` §0 scoreboard** — add fleet hybrid rows; mark the
   03-based angle outputs as alignment-QA-only.
3. **`PAPER_STATUS.md`** — topic 3 (hybrid) becomes "fleet-wide"; refresh the
   figure inventory paths.
4. **`build_final_pdf.py`** (overview PDF) — the per-detector stat cards
   currently print the script-03 angular resolution. Change the angular stat
   to the hybrid number (read 34's summary CSV; print "—" where absent), add
   a "method: hybrid" tag, regenerate `june_detectors_overview.pdf`.
   Check `build_hv_scan_pdf.py` and `build_june_summary_pdf.py` for the same
   stale stat and fix identically.
5. **`engineer_package/`** — fleet tables in `README.md`, `report/main.tex`
   (§ fleet keybox + caveat 5), and `make_slide_deck.py` (slide 15 table)
   currently show "—" for C/D/E angles: fill in the new hybrid numbers,
   delete the "await re-analysis" footnotes, re-run `make_slide_deck.py`,
   recompile `report/main.tex` (pdflatex ×2 from `report/`; note
   `\pkgfig` tests explicit `../figures/` paths because `\IfFileExists`
   ignores `\graphicspath`). If 36/42 were extended fleet-wide, refresh those
   rows too. Update `figures/FIGURE_GUIDE.md` if any figure is regenerated.

Numeric-consistency sweep at the end: grep the repo docs for the superseded
values so they can't survive anywhere:
`grep -rn "2\.04\|2\.15\|3\.15\|2\.49\|2\.50°\|14\.9" *.md engineer_package/ --include=*.md --include=*.tex`
(also re-check `80.6`, `79.7`, `76.1` — pre-recovery det3 efficiencies that
should only appear in historical notes, clearly marked).

## 5. Archive / purge stale outputs (do this LAST, after §4 verifies)

Policy: **archive, don't delete**, except for exact-duplicate backups. Create
`~/x17/cosmic_bench/Analysis/_archive_superseded_<date>/` and `mv` preserving
relative paths. Never touch: raw data, `combined_hits_root`, `decoded_root`,
`m3_tracking_root*`, `cache/` pickles, `hv_monitor.csv`, run configs.

Move to archive:
1. Every `*_prev2_backup/` directory under the Analysis tree (pre-M3-v2).
2. Every no-veto `alignment_tpc/` directory where an `alignment_tpc_veto50/`
   sibling exists (03 can regenerate them; they are a stale-number hazard).
3. Script-03 angle artifacts in each `alignment_tpc_veto50/`:
   `angular_resolution.json`, `angular_resolution.png`,
   `angle_correlation_raw.png`, `angle_correlation_corrected*.png` — AFTER
   confirming nothing in §4 still reads them (build_final_pdf.py reads
   `angular_resolution.json` today — that is exactly the dependency §4.4
   removes). If a script still needs the file for alignment QA, leave the
   JSON and archive only the PNGs.
4. `report_det3_weekend/` compiled PDF (rev 4) — two generations stale as a
   *results* document. Keep the source + `mk_diagrams.py` (the schematics are
   current and used by the engineer package); add a top-of-README note
   "results superseded; see JUNE_RESULTS_SUMMARY.md" or rebuild it if you
   have time (not required).
5. Stray analysis-era leftovers flagged in the docs:
   `_false_start_01H29/` (already parked), `_backup_feu7only_003/` (already
   parked — leave), any `efficiency*/` outputs that predate the live-range
   guard IF the guard rerun produced a sibling (compare timestamps; the 7-06
   reruns are authoritative).

Write an `_archive_superseded_<date>/MANIFEST.md` listing every moved path
and why. Log the whole pass under `Analysis/_grand_logs/`.

## 6. Verification checklist (all must pass before you call it done)

- [ ] For each of det2/3/6/7 (+ det4 if measurable): hybrid summary CSV
      exists, self-trained σ68 + coverage recorded, det3-transfer cross-check
      recorded, unshared-time-fit v vs v_geom agree within ~3 %.
- [ ] c1/c2 per detector recorded; consistent with the "design property"
      expectation (det2≈det3); any outlier investigated, not averaged away.
- [ ] `JUNE_RESULTS_SUMMARY.md`, `MICROTPC_RUNBOOK.md`, `PAPER_STATUS.md`,
      overview/HV PDFs, and `engineer_package/` all quote identical numbers
      (one grep per headline number returns only consistent hits).
- [ ] `build_final_pdf.py` regenerated overview has NO stat sourced from
      script-03 angles or ridge velocities.
- [ ] engineer package: `report/main.pdf` recompiles with zero
      `MISSING FIGURE` boxes (`pdftotext main.pdf - | grep -c "MISSING"`),
      deck rebuilds, fleet table complete.
- [ ] Archive manifest written; a fresh grep for superseded values (§4 list)
      hits only clearly-marked historical notes.
- [ ] Nothing under raw/caches/reference trees was moved or deleted.

## 7. Gotchas that have each cost real time before (respect them)

- Run everything from `mx_june_cosmic_qa/` with `../.venv/bin/python`.
- Waveform work REQUIRES pedestal + per-chip CNS (pattern in 24); production
  hits have no CNS. FEU 6/8 have huge common mode (raw σ~115).
- FEU↔detector map changes per run; det slots two positions (z≈232/702);
  alignment θ converges near 89–90° — that is correct, not a bug.
- det3's gas dried during the week (>3 %→~1 % H₂O before/after ~6-24): never
  average det3 across those eras; the 6-23 run is broken (degraded M3) and
  stays excluded.
- MIN_STRIPS: 4 before unsharing, 3 after. λ_att fit window z∈[8,22] mm.
- The sat run's FEU 8 dies after eid ~38,926 (file-003 tail) — live-range
  guard must stay on for any whole-run quote.
- eventIds are continuous across datrun files; small id gaps are normal.
- Runtimes: hits-level minutes; waveform scripts 5–15 min each per detector
  (stream in 400-event chunks, ~1 GB RAM).
- Do not touch `ntof_tracking/` (July-beam package, separately validated) —
  but if hybrid constants change for det2/det3, flag it there
  (`TRACK_PLAN_*`) since its frozen models derive from this chain.

## 8. Suggested execution order

1. §2 verification pass (hours: 0.5) — confirm golden outputs in place.
2. §3.1 staging det6+det7 (network-bound), then §3.2 det6 → det7 (each
   ~1–2 h including checks). det4 last (§3.2 step-2 gate decides if it
   proceeds).
3. §3.3 optional extensions if the above went smoothly.
4. §4 re-quote + regenerate (1–2 h, mostly careful editing).
5. §5 archive pass + manifest (0.5 h).
6. §6 checklist, end to end.

Deliverable at the end: a short summary note (append to
`JUNE_RESULTS_SUMMARY.md` changelog header) stating the new fleet hybrid
numbers, det4's outcome, what was archived, and any deviations from this plan.
