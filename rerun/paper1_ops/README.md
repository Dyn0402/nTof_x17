# Paper I §4 on pass 1: HV, gain and discharges (2026-10-31)

Paper I §4 ("Operation: high voltage, gain and sparks") re-derived on the pass-1 products. All numbers are in
`report.html`, which `make_report.py` generates. Per-row tables are in `results/`.

EOS: `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_ops/` (`spark/`, `tel_gain/`, `sps_gain/`, `hits_gain/`,
`code_9527d489/`). Condor dirs on lxplus: `~/x17rerun/paper1_ops/`. `Analysis_pass1` was only read.

## Verdict

1. **The text's operating points (480/480/440 V) are what the HV scans support. The golden long runs sat 10–65 V above them,
   and det6/det7 paid for it in discharges.**

   | chamber (scan) | best HV | efficiency there | plateau (within 1 pt) | discharges/trigger there | golden row: HV, efficiency, tracks lost to discharges |
   |---|---|---|---|---|---|
   | det2 (6-22) | 480 V | 92.2 % | 475–490 V | 0.008 | 495 V: 90.3–90.9 %, 1.5–1.9 % |
   | det3 (6-27) | 470 V | 93.8 % | 460–490 V | 0.006 | 490 V: 92.2–92.6 %, 0.9–1.2 % |
   | det6 (6-26 scan) | 480 V | 85.8 % | 480 V | 0.026 | 495 V: 74.4 %, 16.8 % |
   | det7 (6-26 scan) | 430 V | 89.8 % | 430–435 V | 0.027 | 495 V: 68.0 %, 24.7 % |

   - Efficiency is the loss budget's interior_good (`rerun/effbudget/pass1`, R = 5 mm).
   - **Paper consequence:** the quoted det6/det7 efficiencies (74 / 68 %) are discharge-limited operating points, not the
     chambers' best. At their scan optima they read 86 / 90 %; quote both.
   - **The angle resolution is no worse at the optimum:** det7 0.78/1.05° at 430 V against 1.22/1.45° at 495 V; det6
     1.01/1.28° against 1.21/1.45° (s68, v(t) arm). The scan rows are small (150–700 rays), so these are indicative.
   - Read the text's "480/480/440" as det3/det6 at 480 and det7 at 440: it matches det3 470 / det6 480 / det7 430–435. If it
     meant det2/det3/det7 it also fits (det2 480). **Check which chambers it names.**
2. **Discharges, measured on every pass-1 row.**
   - **Definition:** a trigger the pass vetoes (> 50 strips over the floor) with a DREAM chip of ≥ 8 strips over 3000 ADC.
   - **Rate:** it rises ×10 every 26 V (det2), 27 V (det3), 21 V (det6) and 42 V (det7), against the gain's ×10 every
     ~56–63 V. So the discharge rate goes roughly as gain^2–2.7 (det7 gain^1.4).
   - **At the golden HV:** 0.13–0.17 discharges/s on det2/det3, 0.69 on det6, 1.14 on det7, at 4–6 Hz of triggers.
3. **No dead time after a discharge.** Efficiency against the time since the previous discharge is flat from 1 ms to
   1 s. The lost time per discharge, integrated below 0.1 s:

   | chamber | discharges used | dead time per discharge | 95 % UL |
   |---|---|---|---|
   | det2 | 17 208 | −1.7 ± 1.3 ms | 2.1 ms |
   | det3 | 45 500 | 0.2 ± 0.6 ms | 1.2 ms |
   | det4 | 4 303 | −2.4 ± 9.4 ms | 15.5 ms |
   | det6 | 24 126 | 0.8 ± 1.1 ms | 2.6 ms |
   | det7 | 42 827 | 0.4 ± 1.1 ms | 2.1 ms |

   - **No gain sag either:** the median unsaturated charge after a discharge is 0.96–1.03 of the late value on det2/det3.
   - On det6/det7 it reads 1.04–1.17 at 0.1–1 s. That is a time confound, not recovery: discharges cluster in the
     high-gain stretches of a row whose gain drifts (det6's ×10 night).
   - **So the only discharge loss is the coincidence itself:** a muon crossing during a discharge.
4. **The pass's spark veto also removes events that are not discharges, and at low HV that costs efficiency.**
   - **det6 at 400–445 V:** 8–12 % of triggers are > 50-strip bursts with no strip near the rail, i.e. coherent chip
     events (known on det6/det7). The veto throws away 7–10 % of interior tracks there; at 495 V it is 1.5 %.
   - **g_det4:** bursts take 6.3 % of interior tracks; det2/det3 below 1 %.
   - It explains det6's low efficiency at the bottom of its scan (62–81 %): that is not a gain turn-on.
   - The new `hv_scan` figure shows bursts separately (dotted).
5. **det4's gain at SPS barely depends on its HV.**
   - **The trap:** the pass table's own events cannot measure it. Only events over det4's zero-suppression threshold
     reach the table, so their charge quantile moves 7 % over 580–780 V (run66) while the event yield halves.
   - **The method:** det4's response on EVERY clean uRWELL telescope track in its live band (missed crossings ranked
     lowest).
   - **run66 (Ar/CF₄/iso, one 30-min sub-run, 580→780 V):** the median charge goes 2140 → 3538 ADC (×1.65) and the
     detection fraction 85 → 98 %.
   - **runs 55/56 (Ar/CO₂/iso 95/3/2, 550–625 V):** flat (×0.94); detection 99 %.
   - **Comparison:** the bench chambers double every 16–19 V. A bench-like slope would move det4's lead strips to the
     rail (f_sat stays ≈ 1 %, median lead peak 388 → 460 ADC).
   - **Bench scale (same DREAM registers 6/7 = 0xAAAA, 200 fC, at both sites):** det4's median MIP-like lead strip is
     ≈ 1040 ADC on the bench at 495 V (g_det4, cosmics), but 350–460 ADC at SPS at 550–780 V (normal incidence). The
     gas and the site differ, so this is not an absolute-gain statement.
   - **Not explained from the data.** See the open questions.
6. **The 6-23 night's flat gain curve is the night, not M3 and not det4.**
   - **Method:** a gain proxy with no M3 and no fit (median lead-strip amplitude of MIP-like triggers, `hits_gain.py`).
   - **It reproduces every other scan's fitted slope:** det2 0.43, det3 0.40–0.45, det6 0.36, det7 0.36–0.39 per 10 V,
     against rerun/paper1_charge's 0.418 / 0.414 / 0.364 / 0.388.
   - **On 6-23:** det3 reads 0.10–0.12 and det4 0.06–0.09 per 10 V.
   - So the 6-23 exclusion stands. Its reason is now "both chambers' charge barely followed the HV that night (HV and
     time confounded: the sub-runs ran in descending HV)", not "M3 degraded".
   - det4 still has no usable bench gain curve.

## What was compared, and how

- **Discharges / dead time** (`spark_timeline.py`, condor `condor/spark.sub`, all 156 bench rows):
  - per trigger from the pass's rr1 hits: the floor multiplicity (the pass's veto), the chip test, and trigger
    timestamps;
  - joined to the loss budget's per-ray table (`paper1_eff/budget/.../rays_<det>.parquet`): interior_good rays, their
    category at R = 5 mm, and the time since the previous discharge epoch (discharges < 10 ms apart merged);
  - charge from the pass table (unsaturated events).
  - **Check:** the in-spark loss reproduces the budget's `loss_spark_ig` (det7 495 V: 26 % against 24–26 %).
- **Trade-off** (`make_report.tradeoff`): the budget's efficiency and loss ladder, `rerun/paper1_res` angle/position
  resolution (v(t) arm), `rerun/paper1_charge` charge, joined per row.
  - The scans used: det2 6-22, det3 6-27, det6/det7 6-26 `hv_scan`.
  - Not used: the 6-26 overnight resist steps, which ran while det6's gain rose ×10 (HV and time mixed); they stay in
    `results/tradeoff.csv`.
- **det4 SPS gain** (`tel_gain.py`, condor `condor/tel_gain.sub`; `sps_gain.py` for the pass-table comparison):
  - the telescope from banco's combined_hits (`rerun/sps_audit/tel_job.telescope_zs`), strict (one cluster in each of
    the four plane-views);
  - det4 from the feu-3 decoded file: banco's combined_hits carry no det4 hits after run_55;
  - the plateaus from the registry windows on the pass's wall clock;
  - the live band from det4's own detection map along the back uRWELL's y at the highest-HV plateau of each run.
- **Hits proxy** (`hits_gain.py`, condor `condor/hits_gain.sub`, the 139 scan and golden rows). 133 rows came back; the
  6 missing are det2 6-17, the skeptical half-dead-X night, already excluded.

## Figures

- `figures/hv_scan.pdf` → Paper I `hv_scan` (rebuilt; `papers/paper1/figures.toml`): efficiency, tracks lost to
  discharges vs bursts, and discharges per trigger against HV, det2/det3/det6/det7, golden rows starred.
- `figures/spark_deadtime.pdf` → Paper I `spark_deadtime` (rebuilt): efficiency and charge against the time since the
  previous discharge, all five chambers.
- `figures/tradeoff.pdf`: efficiency, discharge ceiling and angle s68 against HV per chamber (for the discussion; not in
  the manifest).
- `figures/det4_gain_sps.pdf`: det4 charge and detection against HV at SPS (candidate for §9 / §4).
- `figures/hits_gain.pdf`: the M3-free gain proxy per scan, 6-23 dashed.

## What this does not rule out / limitations

- **The scan rows are short** (150–700 interior rays per HV point; det7 151–341), so the best-HV choice is ±5 V. The
  plateau edges matter more than the single best point.
- **The time ordering of each scan is not separated from HV.** It is mild on 6-22/6-27/6-26 (the proxy slopes match
  across nights) and fatal on 6-23.
- **The discharge definition is a choice:** chip ≥ 8 strips over 3000 ADC, the band census test.
  - A partial discharge that stays below 3000 ADC on every strip counts as a burst.
  - At the golden HV bursts are ≤ 1.5 % of tracks on det2/3/6/7, so the split moves the discharge loss by at most that.
- **"Muon-induced (4–6×)" and "concentrated at the edges" (the text) are not re-derived here.** They need the
  discharge positions and a random-trigger comparison, which the bench data do not have. They stay as the June hits
  results or are dropped.
- **No dead time is seen down to the first bin (0–3 ms).** A recovery faster than a few ms is not constrained, but it
  would cost < 0.3 % at the measured rates.
- **det4 at SPS:**
  - the live-band selection uses det4's own detection at the top plateau, which biases the lowest-HV efficiency up by
    construction; the charge trend is the result;
  - run55 and run56 are different runs (≈ 1.5 h apart, different beam spots);
  - absolute gain is not measured anywhere: charge is in ADC, with no fC calibration.

## Open questions for Dylan

1. **Which chambers does "480/480/440 V" name?** And should Paper I quote both efficiencies for det6/det7 (the golden
   row at 495 V, discharge-limited, and the scan optimum at 480/430 V)?
2. **det4's weak HV dependence at SPS (×1.65 over 200 V):**
   - Is the CAEN `12:2` resist channel det4's amplification HV directly, or through a filter / series resistor with a
     leakage path? `resist imon` is ~0.7 µA at 770 V.
   - With no new data, this can only be a stated observation plus Stephan's input (§4 already has an `askStephan`).
   - It also bears on the "efficiency still climbing at 670 V, 175 V above the bench point" puzzle (det4-h4-efficiency).
3. **The pass's spark veto on bursts:**
   - It is a pass-level choice: the multiplicity on a relative floor (`wft.seed.SPARK_VETO_HITS`).
   - It removes 6 % of g_det4's interior tracks and 8–10 % of det6's at low HV without a discharge.
   - The efficiency session counts these as "spark" losses. Re-label them, or leave the veto as is and state it?
     Re-running the reco without the veto is a pass change (not done).
