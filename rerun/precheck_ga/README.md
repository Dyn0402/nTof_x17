# F7 · the 14 channels the GA gain map added to the masks (pre-flight, 2026-10-29)

PASS_PLAN F7 / REPROCESSING_TODO B2. GA (10-28, `model_audit/followups/ga_gainmap`) added 14 channels to the golden-row
masks from its in-fit gain map: det4 X 94 125 196 376 400 435 439 455 472 498 → dead, det4 X 274 → hot, det3 X 62 →
hot, det3 Y 41 / 510 → dead. Dylan asked for each to be double-checked on the hits, as the other masks were, with a waveform
display, and for any that fail to be dropped. Done on the rr1 inputs of the golden rows (sat_det3, g_det4), first 4
acquisitions (43k / 52k events on det4 X).

## Tests
- `ga_check.py`: **flank** (both ±1 neighbours > 300 ADC: fired fraction > 50 ADC, the `flank_census` rule) and the
  **two-sided ratio** A_c / mean(A_L, A_R) on those events, on rr1_cm0 hits and on fa80c06 waveform peaks
  (pedestal + plain block CM, as `wft.io`). Divided by the local median over every judged channel within ±32 strips → q.
  Plus a waveform display of the channel and ±2 neighbours on six flank events.
- `ga_mirror.py`: **mirror ratio** for channels with a dead or rarely bright neighbour. With a live neighbour n as the
  cluster peak, Σ A_c / Σ A_m, where m is n's other neighbour → q per side. Noisy: healthy controls read 0.24–1.36 on one side.
- `ga_display1.py`: one-sided displays for 94 / 400 / 435 / 439.

Rule: **healthy → dropped from the mask**. Confirmed weak, or inconclusive → **kept censored**. Censoring a healthy
strip costs a little information (the fit treats its charge as unknown). A weak strip fitted as live biases the fit.

## Result
| chamber view ch | GA | flank fired | q two-sided (hits / wf) | q mirror (wf, −/+) | display | verdict |
|---|---|---|---|---|---|---|
| det3 X 62 | hot | 1.00 | 1.11 / 1.06 | – / 0.82 | normal | **healthy: hot dropped** |
| det3 Y 41 | dead | 1.00 | 0.98 / 0.98 | 1.00 / 1.34 | normal | **healthy: dropped** |
| det3 Y 510 | dead | 0.57 | 0.00 / 0.04 | 0.04 / 0.23 | flat beside bright neighbours; −4000 ADC excursions | **dead: kept** |
| det4 X 94 | dead | (5 events) | – | 0.42 / – | (1 event) | weak (occupancy 0.2× median): **kept** |
| det4 X 125 | dead | 0.96 | 0.72 / 0.91 | 0.79 / 0.43 | normal | **healthy: dropped** |
| det4 X 196 | dead | 1.00 | 1.01 / 1.00 | 0.65 / 0.91 | normal | **healthy: dropped** |
| det4 X 376 | dead | 0.98 | 0.86 / 0.86 | 0.53 / 1.12 | normal | **healthy: dropped** |
| det4 X 400 | dead | (13 events) | – | 0.21 / 1.29 | ~50 ADC where the mirror strip gets 200–500 | weak: **kept** |
| det4 X 435 | dead | 1.00 (20) | – / 1.01 | 0.49 / – | ~40 ADC beside a 0.5–1k peak | weak: **kept** |
| det4 X 439 | dead | 1.00 (20) | 0.59 / 0.80 | 0.20 / 0.77 | mostly coherent-noise events; one normal | inconclusive: **kept** |
| det4 X 455 | dead | 1.00 | 0.77 / 0.77 | 0.90 / 0.82 | normal | **healthy: dropped** |
| det4 X 472 | dead | 0.99 | 0.55 / 0.59 | 0.65 / 0.62 | near 0 beside 1–4k neighbours | weak: **kept** |
| det4 X 498 | dead | 1.00 | 0.95 / 0.95 | 0.62 / 0.69 | normal | **healthy: dropped** |
| det4 X 274 | hot | 0.93 | 0.97 / 1.01 | 0.73 / – | normal | **healthy: hot dropped** (not a short partner of 271–273) |

**8 dropped, 6 kept.** The masks were written by `apply_f7.py`
(`model_audit/followups/fe_unify2/masks/bench_det{3,4}.json`: `ga_added.f7_dropped` / `f7_kept`). det3 X and det4 X now
carry no hot channel.

Inputs and results: `results/ga_check.json`, `results/ga_mirror.json`, `results/figures/ga_*.png` (EOS copy
`/eos/user/d/dneff/x17rerun/precheck_ga/`).

---

# S0b, F8–F11 (pre-flight, 2026-10-29)

## S0b · the five det6/det7 6-26 HV points that were never decoded: no data
455 / 460 / 470 / 475 / 490 V (`june_tests/mx17_det6_det7_hv_scan_6-26-26/resist_det6_<V>V_det7_<V>V_drift_700V`):
**every FEU's datrun .fdf is 0 bytes**, and each RunCtrl log ends in a FEU configuration failure
(`FeuConfig: FEU Run Control WrRd_Missmatch for poke 0x200000`). The DAQ never took data, and no other copy exists
(the laptop mirrors don't have these subruns). They were never in the campaign manifest, so nothing reads them. They are
**excluded: no data (DAQ configuration failure)**. The HV scan has 16 points per chamber, not 21.

## F8 · H4 ch 372 (det4 at SPS, Y view, FEU 03 numbering): confirmed, masked
run_71 fa80c06 decode, two acquisitions (`f8_ch372.py`). The median per-event rms is **1311–1372 ADC against 312–332 for the
Y view** (4×). Its neighbours look like the rest of the view. This matches the 08-04 reanalysis (±400–900 ADC quasi-periodic swings,
`sps_beam_test_26/analysis/RAW_RUN71_REANALYSIS_2026-08-04.md`). It was added to `rerun/channel_layer/det4_sps_h4.json` Y
`dead` and `dead_stripes`, censored like 510 (the waveform is not charge), with an `oscillating` record. `make_sps_mask.py`
carries it (`OSCILLATING`), so a regenerated mask keeps it.

## F9 · rows without a gas-time entry: inheritance checked, two groups can't fit v
The 10 rows GC could not time (`gc_gas_time/results/bench_rows.json`, `missing`) have these clean-trigger counts:
- det2 6-17: 0–52 across the eight rows. This is the half-dead X: X is present in 18 % of the longer_run triggers.
- det3 6-23 short_run: 40. It inherits long_run, which has 16k clean triggers. Correct.
- det3 6-26 quick 530 V / 1200 V: 0. The combined rr1 hits hold 24 → 44k entries per acquisition (production hits are the
  same). The resist channel draws 1.3–2.7 µA, decaying through the run. The chamber was charging up, not amplifying.

The registry had the group reference of det2 6-17 (longer_run, 14 clean) and det3 quick (0) fitting their **own v**,
and every other row of those groups inheriting it. `make_registry.py` now has a floor, `N_MIN_V = 100`: a group whose
reference row is below it takes the chamber's golden bundle, cls `skeptical`, note "angle scale NOT calibrated
(positions/efficiency only)". The det3 100 V drift point (475 clean, a unique field) keeps its own v. Diff: 11 rows,
all det2 6-17 plus det3 quick.

## F10 · det4 6-23 M3: clean fraction acceptable
Clean M3 rays (χ² < 1 both coordinates, NClus ≥ 4) per trigger, first three ray files (`f10_m3clean.py`):
- det4 6-23 rows: **0.074–0.115**, which is 47–73 % of g_det4 6-24 (0.158).
- 6-22 det2/det3 long_run: 0.103 (accepted and quoted); 6-27 det3: 0.179.

The cut defines ray quality, so a lower fraction is statistics, not bias. The worst row, 6-23 long_run (0.074), still has
about 2.9k clean rays for its own v. **No reclassification.**

## F11 · g_det4 FEU 6 connectors 7–8: confirmed, but not unique, so no change (stated limitation)
- The analyzer's pedestal noise there is **0.47–0.55×** the data run's noise (connectors 1–6: 0.82–1.12). Significance is
  inflated about 2×, near-threshold hits are up 2.7× and isolated hits 3.5× (`f11_connectors.py`).
- Effect on seeding (wft.seed on two acquisitions, hits below the corrected 5σ removed): **14.5 % of events change
  their top X seed**. 2653 of 24369 lose a seed made only of sub-threshold hits on connectors 7–8. Spark-vetoed events
  go from 2439 to 2084.
- A guard was written (`wft.reco.pedestal_guard`, tested). It compares the hit noise with the run's own waveform noise per
  connector and rescales below `PED_MISMATCH_REL`. The **fleet scan** (`ped_guard_fleet.py`, every pass row,
  `results/ped_guard_fleet.json`) shows a continuum, not one broken pedestal: median 0.89, 5th percentile 0.74,
  70 connectors < 0.7, mostly det6/det7 at high HV (0.31–0.6) and det3 at 515–525 V. With the data noise taken from
  the pre-signal samples only (`f11_early_noise.py`), **the golden det7 row still reads 0.50**, the same as g_det4.
  The data-run noise with HV on is above the pedestal run's across much of the fleet. The hits, and so the
  established seeding and efficiency semantics, have always carried this.
- **Decision (run-through rule, locked-model principle):** the guard changes seeding fleet-wide, so it would need the
  full-reconstruction gate first. It stays in the code **off** (`PED_MISMATCH_REL = 0`, bit-identical), opt-in like
  the rails. **Stated limitation:** where the data noise exceeds the pedestal run's (g_det4 conn 7–8, det6/det7 at high
  HV, det3 ≥ 515 V), sub-threshold noise hits form seeds in events with no real cluster on that view. The relative 10 %
  floor removes them on events with a real track.
