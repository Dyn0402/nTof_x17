# det4 (mx17_E) at SPS H4: final pass, 2026-10-30

This is the final SPS pass that Dylan approved. Every SPS row was re-run on one code version, `3ac0e79`, which includes
the seed fix found in the audit (Part 2 below). Every head-on position estimator is stored per event. The resolution,
footprint and timing are measured with a minimal, justified selection.

- Report: `report/report_final.html` (`make_report_final.py`). The audit page is `report/report.html`.
- Products: `Analysis_pass1/sps/<key>/`, replaced after a verified backup.
- Telescope tables: `/eos/user/d/dneff/x17rerun/sps_audit/tel/F*`. Footprint tables: `.../footprint/`.

## 1. Answer: which run the paper's position resolution should come from

**Quote run 63 flat700 (Ar/CF4/iso 88/10/2, ZS 64 samples, flat). Use run 53 as the Ar/CO2/iso cross-check. Do not
use run 71 (RAW) for a resolution.**

- **Run 63 gives the most tracks with the telescope calibrated in the same run.**
  - It has 87 683 telescope tracks with a det4 record, and 59–63 k per view after the fiducial cut.
  - All three back-plane pitch zones are populated in both coordinates. So the free 3-zone fit measures the telescope
    (f) and det4 together, in run 63 itself.
  - It does not borrow the run-53 telescope model. The fitted f is 0.25–0.29 here, against 0.29–0.30 on run 53 with the
    same estimator.
- **Run 53 can be put through the fixed chain, and it reproduces the original number when read the original way.**
  - Its FEU 3 data were re-decoded with the pinned decoder fa80c06. The first 20 000 events are identical, byte for
    byte, to banco's decode.
  - Through the fixed chain with no pass cut it gives 91 832 telescope tracks with det4.
  - On the same events, the run-53 hits estimator gives **det4 Y = 177 ± 10 µm** (3-zone fit, f = 0.293). That is the
    2026-08-13 176 ± 10 µm again.
  - Its limits:
    - only Y has the 0.5 mm zone; X has two zones, so X is pointing-model dependent;
    - the gas (Ar/CO2/iso 95/3/2) and gain (resist 505–535 V) are not those of the CF4 data;
    - it is ZS with 32 samples, and the pass's angle scale is wrong there (no v), which spoils the wft fit (see 2).
- **Run 71 (RAW) cannot give the resolution.**
  - Only 25 % of its telescope tracks pass the cross-view fiducial, against 62–75 % on ZS. The 3.84 µs RAW window
    collects beam pile-up, and the self-seeded window often picks another particle.
  - The 0.5 mm zone is empty.
  - In the 1.0 mm zone the residual (230–340 µm) is at or below the telescope pointing, so det4 is not separable.
  - It remains a cross-check: X 1.0-zone residual 269 ± 10 µm (p_mid), Y 232 ± 7 µm.
- **The telescope model therefore does not hinge on run 53.** f is measured independently in run 63 and run 66 (all
  zones, every estimator).
  - Its spread across estimators and runs (0.25–0.34) is the main systematic: about ±40 µm on det4.
  - This is because f and det4 are anticorrelated in the 3-zone fit.

## 2. Position resolution (µm, det4 intrinsic from the free 3-zone fit; per-zone values in `results/final.json`)

| run (n per view after fiducial) | view | fit p0 | p0 read 7.5 mm into gap (p_mid) | charge centroid | hits (run-53 method) |
|---|---|---|---|---|---|
| run 63 flat700, CF4 ZS64 (59–63 k) | X | 314 ± 8 (f 0.280) | **209 ± 11** (f 0.280) | 280 ± 6 (f 0.269) | n/a (no det4 hits in combined_hits) |
| | Y | **190 ± 10** (f 0.276) | 217 ± 12 (f 0.285) | 199 ± 10 (f 0.252) | n/a |
| run 66 r580–r730, CF4 ZS32 (27–33 k) | X | 335 ± 11 | 270 ± 14 | 240 ± 7 | n/a |
| | Y | 197 ± 13 | 165 ± 15 | 173 ± 9 | n/a |
| run 53, Ar/CO2/iso ZS32 (35–39 k) | X | 2 zones: 1.0 → 395, 1.5 → 375 | 312 / 308 | 170 / 157 | 189 / 133 |
| | Y | 262 ± 12 | 286 ± 17 | **154 ± 8** (f 0.254) | **177 ± 10** (f 0.293) |
| run 71 raw700, CF4 RAW64 (1.5–1.9 k) | X | 1.0 zone residual 338 ± 13 | 269 ± 10 | 340 ± 10 | — |
| | Y | 1.0 zone residual 247 ± 8 | 232 ± 7 | 239 ± 7 | — |

Read the table as follows:
- **The estimator matters by up to 1.6×, and differently per view.**
  - CF4: p_mid is the best X estimator (run 63: 209 against 280 and 314 µm). p0 and the centroid are the best Y
    estimators (190 and 199 against 217 µm).
  - Ar/CO2 run 53: the centroid and hits beat the wft fit in both views. The fit has no v there, so the bench v
    (33 µm/ns) sets its angle scale, and its 32-sample window cuts the column.
  - Dylan picks the estimator; all four are stored per event: `x/y_p0`, `x/y_p_mid`, `x/y_c_pos`, and hits in the
    telescope tables.
- **det4 X is consistently worse than Y.** In CF4: X 209–335 µm against Y 165–217 µm.
  - It is not an edge effect of the dead stripes: keeping tracks ≥ 1 mm inside a live band changes X by ≤ 4 %
    (`band_edge.py`, `results/band_edge.json`).
  - It is not a wider footprint either: without zero suppression (run71 RAW) the neighbour/lead ratio is 0.31 (X)
    against 0.34 (Y). The X/Y difference stays unexplained.
- **The wft fit's p0 is the position at the mesh.** At normal incidence its angle noise leaks into it (X tan s68 ≈ 0.1
  on ZS). p_mid = p0 + 7.5 mm × tan reads the track where the p0–angle correlation vanishes. The 7.5 mm was chosen on
  run 63 and applied unchanged elsewhere. It depends on v, so it is meaningless on rows without v (run 53, run 55).

## 3. Selection: what is cut and why (paper methods)

These are the only cuts on the resolution sample. Each is justified by a number from the data.

1. **Telescope track: exactly one cluster in each of the four uRWELL plane-views** (front and back, x and y; clusters
   of hit strips in position, gap 3 mm).
   - 82 % of four-view events on run 53, 71 % on run 55.
   - In the rejected multi-cluster events, the leading back cluster agrees with the front plane within 3 mm in only
     40–43 % of cases, against 87–93 % for single-cluster events. The 68 % width of back − front is 19–24 mm against
     0.8–1.1 mm, so the leading cluster there is often another particle (`tel_cut.py`, `results/tel_cut.json`).
2. **det4 signal in both views.** The brightest strip of each view, excluding the oscillating/hot channels, is at least
   50 ADC (5 × the 10 ADC strip noise).
   - This is the noise floor. Below it a "seed" is a noise sample.
   - Run 53: 91 832 of 398 397 single-cluster telescope tracks pass (23 %). The rest cross the dead amplification
     stripes or are outside det4's window.
3. **Estimator defined.** The fit converged (p0, p_mid), a cluster above 50 ADC exists (centroid), or exactly one hit
   cluster per view inside the drift-time gate (hits, run-53 method).
   - For the comparison table, every estimator is required on the same events.
4. **Cross-view fiducial: the other view's residual within ±1.5 mm.** The residual has a narrow core on a flat floor
   (`figures/final_fiducial.png`).
   - The floor is tracks crossing a dead X stripe: det4 then reads another particle in a live band. The failure
     fraction is 0.15–0.3 inside live bands and ~1.0 inside dead ones, against the telescope position.
   - Passing: 62–76 % of tracks on ZS, 25 % on RAW.
   - It is independent of the view being measured.
5. **Plateau and beam.** Events within 20 s of a drift/resist plateau edge are dropped (the pass rule). run60 ov07 is
   not used.

The resolution is the σ of a Gaussian fitted to the residual core in a fixed ±1.2 mm window (not an iterative clip).
The alignment is a robust 2-D affine.

**Cuts tested and NOT applied, because they do not change the core width.** Core σ in µm, wft p0, passing / failing /
all (`results/final.json` 'tests'):

| candidate cut | passes | effect |
|---|---|---|
| the pass's χ²/dof ≤ 5 × median | 81–94 % | X and Y change ≤ 6 %, with no consistent sign (run63 X 441/415/439, Y 331/332/332) |
| the pass's seed cut (peak ≥ 300 ADC and ≥ 4 strips, both views) | 5–27 % | ±10 % with no consistent sign: better in run63/66/71 X (−7…−14 %), worse in run53 Y (+13 %) and run55 d500; it would discard 73–95 % of the tracks |
| cluster ≤ 6 strips (pile-up/delta) | 95–99 % | no change (≤ 1 %) |
| track ≥ 1 mm inside a live X band | 70–80 % | X changes ≤ 4 % |

## 4. Timing: not measurable at the level of interest (limitation)

- The trigger at H4 is asynchronous to the 60 ns DREAM clock. The fine timestamp ftst gives its phase in 6 classes.
- **The wft fit's t0 is not a time at normal incidence.** With w = 0 every depth lands on the same strips, and the
  profiled depth distribution absorbs a t0 shift. Across 1.2–34 k events per dataset, t0_X − t0_Y has a robust σ of
  180–260 ns (ZS) and 260–390 ns (RAW).
- The cluster's own leading edge (`footprint.py`: 50 % or 20 % of the summed waveform's maximum, or a 100 ADC column
  start) does no better:
  - σ(X − Y) = 170–210 ns (CF4 ZS), 132 ns at best (run71 RAW raw700, no zero suppression) and 280–410 ns (Ar/CO2);
  - against the ftst-corrected trigger: 400–440 ns (CF4) and 940–1000 ns (Ar/CO2);
  - the ftst correction does not sharpen it (the best sign changes from row to row).
- The column of a head-on track spans the whole 30 mm gap (~2 µs). Its shape is set by where the primary clusters sit,
  so the threshold crossing moves by a large fraction of the column.
- A ns-level time resolution would need the column start of a single primary cluster and a precise trigger reference.
  Neither is in this data. **det4's time resolution is not measured. The numbers above are an upper bound on what these
  estimators give, not the chamber's resolution.**

## 5. Footprint at normal incidence (`results/footprint.csv`, `figures/final_footprint.png`)

Sample: every flat row, 3000 events each, no pass cut. Seed ≥ 50 ADC; censored strips are bridged. The cluster is the
strips above 50 ADC within ±4 samples of the lead's peak.

- **Resist scan (run66, CF4, 243 V/cm, 580 → 780 V).** Mean cluster size grows 2.23 → 2.38 strips (X) and 2.37 → 2.72
  (Y). The lead-strip fraction falls 0.64 → 0.61 (Y) and stays at 0.61 (X). The median summed peak rises only
  300 → 370 ADC (X).
- **Drift field.**
  - run70 (CF4, 770 V, 35 → 243 V/cm): 2.08 → 2.31 strips (X), 2.15 → 2.62 (Y); lead fraction 0.65 → 0.62.
  - run55 (Ar/CO2/iso, 550 V, 139 → 243 V/cm): 1.77 → 1.89 (X), 1.84 → 2.42 (Y). Y is mostly single-strip at low field
    (median lead fraction 1.0).
- **Threshold-free footprint (run71 RAW, no zero suppression, 243 V/cm).** The cluster has 2.25 (X) and 3.02 (Y)
  strips above 50 ADC. The lead strip carries 0.65 (X) / 0.58 (Y) of the cluster's peak sum, and each neighbour
  0.31 (X) / 0.34 (Y) of the lead.
  - At 156 and 96 V/cm: 2.11/2.85 and 1.87/2.72 strips.
  - On ZS rows the X neighbour ratio reads 0.50–0.55 against Y 0.38–0.43. That is the ZS threshold (a suppressed
    neighbour is a zero), not the chamber.
- **Limitation.** The ZS threshold sets the visible footprint on ZS rows, so the strip counts are above-threshold
  multiplicities, not the charge-sharing kernel. The kernel itself is the pass's calibration (`sharing_kernel/`, bundle
  hyper).

## 6. Re-run done (seed fix), ledger, and what changed in the products

- **Products.** The 71 SPS rows were re-run twice, all on one code version each time (`make_pass_dag.py sps`, staged,
  then `condor/promote_sps.sh`, which md5-verifies a backup before copying).
  1. **`c3e6a04c`, the seed fix and the head-on columns.** Staged in `.../sps_final_stage/sps/`. The pass-1 products
     are backed up in `/eos/user/d/dneff/x17rerun/pass1_backup_2026-10-30/sps_before_seedfix/`.
  2. **`3ac0e79`, final.** The oscillating/hot strips' samples are now also blanked inside the window. They were already
     censored in the fit, but the fit's start (`wft.model.init_guess`, the brightest strip of the window) could still
     land on them.
     - The pile-up at 156–160.5 mm left after the seed fix: run55 d700 7.9 → 3.6 %, run57 d650 3.2 %, run60
       ov01/03/05/09 2.4–3.2 %, run59 2.8 %.
     - Staged in `.../sps_final2_stage/sps/`. The `c3e6a04c` products are backed up in `.../sps_before_blanking/`.
- **Ledger.** `pass1_ledger.py` now requires on every SPS row:
  - code `3ac0e79`;
  - the meta key `hot_seed_channels`, and a seeding rule that names `RAW_CM_BAD`;
  - the head-on columns `x/y_p_mid`, `x/y_c_pos`, `x/y_c_n_strips`, `ftst`.

  Result: pending (the 3ac0e79 DAG; see the next commit).
- **Census of what changed** (`results/seed_census.csv`):
  - **run71 raw275/450/700: 81–83 % of events had a pre-fix seed on ch 372 or 510.** The RAW Y windows were mostly on
    the oscillation.
  - 15 ZS rows by 2–46 % (ov07 46 %, run55 d700 31 %, run57 d650/d700 16 %, ...);
  - 42 rows by ≤ 0.4 % (a quiet ch 510);
  - 11 rows not at all.
  - Hot channels found by the per-plateau probe (excluded from the seed and censored):
    - run55 d700: 173, 373; d600: 172, 173, 373; d500: 172, 173;
    - run60 ov00: 75;
    - run60 ov01/03/05/09/14: 373;
    - run60 ov07: 70, 74, 128, 311;
    - run61 d070/d140/d280: 373;
    - every row also excludes 372 and 510.
- **New per-event columns in every SPS table:**
  - `x/y_p_mid`;
  - `x/y_c_pos`, `_c_n_strips`, `_c_q_sum`, `_c_q_lead`, `_c_lead_frac`, `_c_nb_lo`, `_c_nb_hi`, `_c_t_peak_ns`;
  - `ftst`.
- **The pass rows keep the pass selection** (peak ≥ 300 ADC, ≥ 4 strips, 3000 events per plateau). The resolution
  sample above is the separate telescope-first sample without that cut.

## 7. Limitations

- **Telescope pointing model.** It is self-calibrated only where all three back-plane zones are populated: run 53 (Y),
  run 63 and run 66 (both views). The fitted f ranges 0.25–0.34 across estimators and runs, and det4 moves with it
  (±40 µm).
- **run 55 and run 71 have no 0.5 mm zone.** Their per-zone det4 values use f = 0.301 and fall to 0 where the residual
  is below that pointing.
- **The hits estimator exists only where banco's combined_hits carry det4 (runs 53, 55).** Runs 63 and 66 have uRWELL
  hits only.
- **Unfinished telescope groups.** None: all 59 telescope groups (run53 ×4, run55 ×6, run63 ×9, run66 ×4, run71 ×36) are in.
- **Time resolution:** not measured (section 4).
- **The SPS gases (Ar/CO2/iso 95/3/2, Ar/CF4/iso 88/10/2) are not n_TOF's.** These numbers characterise the
  detector technology at normal incidence, not the n_TOF conditions.

---

# Part 2: the audit that found the seed bug (earlier the same day)


Question: on 2026-08-13 det4 measured about 176 µm at normal incidence in both views (run 53, hits, against the uRWELL
telescope). Today's follow-up (`../sps_followup/README.md`, commit a07f9464) reported, on RAW run71 through pass 1,
X = 193 ± 36 µm, Y "dead under the telescope", and 44 % of the Y fits piled up at 155–160 mm. Which of these hold?

Report: `report/report.html` (generated by `make_report.py`; figures in `report/figures/`, gitignored). Outputs and the
telescope tables are on EOS in `/eos/user/d/dneff/x17rerun/sps_audit/`. Condor files are in `condor/`, and on lxplus in
`~/x17rerun/sps_audit/`.

## Verdict

1. **det4 Y is not dead. The Y pile-up is a chain bug, now fixed.** `sps_build_cache` seeds each view's ±16-strip
   window on the brightest strip. A few Y channels oscillate by ±400–900 ADC in some sub-runs: ch 372 (158.34 mm),
   its neighbour ch 373 (157.56 mm) and ch 510 (250.38 mm). They win the seed, so the window sits on the oscillation
   instead of the track. This was already known on 2026-08-04
   (`sps_beam_test_26/analysis/RAW_RUN71_REANALYSIS_2026-08-04.md` §1, which says to exclude ch 510/372 as central
   strips), but the fix never reached the pass's cache builder. Effect on the pass-1 tables:
   - run71 raw275/450/700: 41–44 % of the Y fits sit on ch 372 and 9–12 % on ch 510;
   - also run60 ov07 (ch 510: 37 %), run55 d700 (ch 373: 24 %), run57 d650/d700 (ch 372: 16 %), and smaller
     fractions in others (table below).
   With the fix, det4 Y follows the telescope over its whole footprint (Y ≈ 145–255 mm): `report/figures/scatter.png`.
2. **The follow-up's telescope missed 90–97 % of the tracks.** It required an in-time 3-strip sum above 100 ADC with a
   single peak in every plane-view. In RAW the uRWELL lead strip is mostly 40–100 ADC (noise 3.9 ADC), so only 0.054 %
   of triggers kept a correlated four-view track. On ZS run66, two hours earlier with the same HV and mount, the rate is
   3.7 %. A ZS-like telescope (`urw_raw_scan.py`: single strip above 40 ADC, samples 9–17, clusters in position, as on
   ZS) keeps 1.55 % on group 000, 29× more. Neither the uRWELL HV (`hv_monitor.csv`: 8:6/8:7 at 620 V, 12:0/12:1 at
   420 V, the same as run66) nor the event matching (below) explains the low rate.
3. **run71 RAW, fixed chain, ZS-like telescope, both views.** n = 1297 telescope tracks with det4 in raw700, 2-D affine
   det(A) = 0.996, roll −89.8°. The residual is quoted on the cross-view fiducial sample (the other view within 1.5 mm
   of the telescope), with the core taken from a fixed-window Gaussian:

   | det4 view | residual, all zones | back 1.0 mm zone | telescope pointing in that zone (run-53 f = 0.301) | det4 left over |
   |---|---|---|---|---|
   | X, wft p0 | 370 ± 17 µm (n 441) | 283 ± 18 µm (n 169) | 252 µm | ~130 µm (≤ 190 at 2σ) |
   | Y, wft p0 | 283 ± 12 µm (n 432) | 254 ± 16 µm (n 164) | 252 µm | ~30 µm (≤ 135 at 2σ) |
   | X, p0 read 7.5 mm into the gap | 254 ± 15 µm | 233 ± 14 µm | 252 µm | not resolved |
   | Y, same | 256 ± 16 µm | 249 ± 18 µm | 252 µm | not resolved |

   **Neither view's intrinsic resolution can be extracted from run71.** The residual is telescope-dominated, and the
   pointing model (run 53, ZS hits) does not strictly apply to RAW 1-strip uRWELL clusters. What holds model-free: in the
   1.0 mm zone the det4 residual is 233–283 µm in **both** views, so det4 is ≤ ~250 µm in both, and consistent with run
   53's 176 µm. The follow-up's X = 193 ± 36 µm comes out the same way within its error (wft p0, zone-mixed pointing:
   210 ± 29 µm here). Its error bar does not include the pointing model, and its size is set by that model.
4. **Pass 1 cannot reproduce 176 µm on run 53, and that is the selection, not the reconstruction.**
   - The audit's machinery reproduces the 2026-08-13 result exactly on the same events: n = 44 198, det(A) = 0.9945,
     back plane f = 0.301, det4 = 176 ± 10 µm (`results/baseline.json`). This is the uRW-x coordinate, so the 176 µm is
     **det4 Y**; det4 X reads 184–212 µm per zone.
   - The pass's per-view selection (peak ≥ 300 ADC and at least 4 strips with samples in the window) was built for
     high-gain CF4 data. On run 53 (Ar/CO2/iso, resist 505–535 V, mean cluster 2.4/1.8 strips) it keeps only the
     high-charge tail.
   - On exactly those events, run 53's own hits estimator does no better than the pass:

     | estimator | X | Y |
     |---|---|---|
     | wft p0 | 525 ± 19 µm | 385 ± 16 µm |
     | hits | 521 ± 20 µm | 428 ± 17 µm |
     | charge centroid | 505 ± 16 µm | 413 ± 15 µm |

     The events that fail the pass's ≥ 4-strip cut have a median of 3 X / 4 Y strips and half the peak (`hits_baseline.py`).
   - The intended test, the pass on run-53-like data giving ~176 µm, cannot be run: on run-53-like data the pass
     reconstructs a different, worse population.
5. **At normal incidence, wft p0 is a noisier position than it needs to be, in X.**
   - p0 is the position at the mesh, and the fitted angle's noise leaks into it. On head-on ZS run63, X tan has an
     s68 of 0.107; Y's is 0.037.
   - On run63 flat700 (13.8 k tracks), in the back 1.0 mm zone, the X residual is 358 µm with p0 against 294 µm with
     the charge centroid. Reading the track 7.5 mm into the gap gives 287 µm. Y: 276 / 302 / 265 µm.
   - This is a definition, not a bug. A position resolution quoted from p0 at normal incidence is inflated in X.

## Bugs found, fixed on this branch

| where | what | fix |
|---|---|---|
| `rerun/model_audit/sps_build_cache.py` (`_fill`, the per-view seed; now `event_record`) | the brightest strip seeds the window, oscillating channels included | `SEED_BAD` (ch 372, 510) never seeds; each plateau is probed first (`HOT_SCAN` = 2000 events) and a channel that seeds > 3 % of events and > 10× the median of its ±4 neighbours is also kept out (`hot_channels`, `HOT_LAST`); the own-centroid ref leaves them out |
| `rerun/model_audit/sps_reco.py` (`_init`, `main`) | a hot channel that is not in the H4 dead list (ch 373 on run55) would be fitted as charge | the plateau's hot channels are censored like dead ones; `hot_seed_channels` is written to `events.meta.json` |

Tests: `wft/tests/test_sps_seed_bad.py` covers the seed exclusion, a bad channel kept in the window for censoring, the
event with only an oscillation rejected, and the hot-channel detector on an oscillator, a broad beam and one 12 mm live
band. The full `wft/tests` suite passes (149). Smoke test on lxplus: `sps_reco.py run55_flatdrift__d700 --per-plateau 600`
flags ch 373 (703 of 2000 seeds) and the 156–160.5 mm bin falls from 25 % to 2.7 % (1.2 % in the next bin).

Not changed:
- **The census-dead channels** (`det4_sps_h4.json` 'dead') still may seed. Some of them carry as much charge as their
  neighbours (X ch 2 and ch 64 on run66), and they are censored in the fit anyway. An intermediate version that excluded
  them moved ~1 % of the X windows of every row by one strip.
- **`common/Mx17StripMap.py` `Detector.map_hit()`** looks up `feu_connector` instead of the detector-side connector.
  The bug is real but sits outside the SPS chain: the cache uses `det4_sps_map.py` (inverted plugs, windows sorted by
  position, and the model builds its sharing from `pos`, so physical neighbours are fit neighbours).
  `wft/io.py` and two audits use `map_hit` on the bench. Left alone; it needs a bench decision.

## Re-run: which SPS rows, and what it costs

`seed_census.py` rebuilds each pass-1 row's 3000-event cache with the pre-fix seed, without fitting, and counts the
events whose seed the fix excludes. Those events get a new window and every other event is bit-identical (except that
a hot channel is now censored where it sits inside a window). Table: `results/seed_census.csv`.

**Must re-run: the Y (and for some rows X) windows of 2–46 % of events change.**

| row | events changed (of 3000) | hot channels found by the probe | pre-fix seeds on excluded channels |
|---|---|---|---|
| run71_raw__raw700 | ~51 % (pile-up excess in the pass table; census still running) | (372, 510 in `SEED_BAD`) | ch 372 ~42 %, ch 510 ~9 % |
| run71_raw__raw450 | ~50 % (pile-up excess in the pass table; census still running) | (372, 510 in `SEED_BAD`) | ch 372 ~40 %, ch 510 ~9 % |
| run71_raw__raw275 | ~55 % (pile-up excess in the pass table; census still running) | (372, 510 in `SEED_BAD`) | ch 372 ~44 %, ch 510 ~12 % |
| run60_flush__ov07 | 1376 (45.9 %) | 70 74 128 510 | 70:500 74:502 128:343 510:1335 |
| run55_flatdrift__d700 | 926 (30.9 %) | 173 373 | 173:135 373:819 510:2 |
| run57_rot25_co2__d700 | 488 (16.3 %) | 372 | 372:485 510:3 |
| run57_rot25_co2__d650 | 469 (15.6 %) | 372 | 372:467 510:2 |
| run55_flatdrift__d500 | 311 (10.4 %) | 172 173 | 172:101 173:160 510:56 |
| run60_flush__ov01 | 309 (10.3 %) | 373 | 373:305 510:4 |
| run60_flush__ov09 | 304 (10.1 %) | 373 | 373:292 510:12 |
| run60_flush__ov03 | 282 (9.4 %) | 373 | 373:281 510:1 |
| run60_flush__ov05 | 252 (8.4 %) | 373 | 373:250 510:2 |
| run55_flatdrift__d600 | 227 (7.6 %) | 172 173 | 172:86 173:140 510:1 |
| run61_rot15_ladder__d070 | 133 (4.9 %) | 373 | 373:129 510:4 |
| run61_rot15_ladder__d140 | 131 (4.4 %) | 373 | 373:129 510:2 |
| run61_rot15_ladder__d280 | 95 (3.2 %) | 373 | 373:93 510:2 |
| run60_flush__ov00 | 91 (3.0 %) | 75 | 75:89 510:2 |
| run60_flush__ov13 | 68 (2.3 %) | 373 | 373:58 510:10 |

**Should re-run for consistency, but small:** 42 rows, each with 1–56 events seeded on ch 510, ≤ 0.4 % except run55 d500, which is in the table above.
In these rows ch 510 is quiet, so this is a one-strip window shift on a strip the fit already censors: run55_flatdrift d400, run56_m70V 590V, run56_m70V 625V, run57_rot25_co2 d600, run59_co2 co2, run60_flush ov02, run60_flush ov06, run60_flush ov10, run60_flush ov11, run60_flush ov12, run60_flush ov14, run61_rot15_ladder d210, run61_rot15_ladder d420, run61_rot15_ladder d560, run61_rot15_ladder d630, run61_rot15_ladder d700, run62_rot25_ladder d200, run62_rot25_ladder d400, run62_rot25_ladder d500, run62_rot25_ladder d600, run62_rot25_ladder d700, run63_flat flat700, run63_rot25 d325, run63_rot25 d425, run63_rot25 d475, run63_rot25 d525, run63_rot25 d575, run63_rot25 d625, run63_rot25 d675, run66_flat_resist r630, run66_flat_resist r680, run66_flat_resist r705, run66_flat_resist r730, run66_flat_resist r780, run70_flat_drift d100, run70_flat_drift d200, run70_flat_drift d300, run70_flat_drift d400, run70_flat_drift d500, run70_flat_drift d600, run70_flat_drift d700, run70_flat_drift d700b.

**Unchanged (11):** run57_rot25_co2 d550, run60_flush ov04, run60_flush ov08, run61_op25 op25, run61_rot15_ladder d350, run61_rot15_ladder d490, run62_rot25_ladder d300, run66_flat_resist r580, run66_flat_resist r605, run66_flat_resist r655, run66_flat_resist r755.

Cost of `make_pass_dag.py sps` restricted to these rows (8 cores per node, from the pass-1 node logs):
- RAW run71 rows: 2.2–3.2 h each;
- ZS rows: 4–25 min each.

- The 18 must-re-run rows (3 RAW, 15 ZS): about 3 × 2.7 h + 15 × ~0.2 h on 8 cores = **~10 node-hours, ~80 core-hours**. Run in parallel, the wall clock is set by the RAW rows: **~3.2 h**.
- Adding the 42 small rows costs another ~42 × 4–25 min (≤ 18 node-hours) but no extra wall-clock time.
- Downstream of the re-run:
  - the run71 resolution here, and `../sps_followup` (i);
  - any S4 or ws_sps input built from these rows (`model_audit/followups`, `ws_spsv`).
  - Then the ledger check (`pass1_ledger.py`).
- The hot channels found per row are written to each re-run row's `events.meta.json` (`hot_seed_channels`); compare them with the table above.

Re-run the run71 rows first: they are the only RAW rows, and the only rows where most Y fits change. Not launched here.

## What was checked, and how

- **Strip map and channel order.**
  - Pass chain: `sps_build_cache` → `det4_sps_map.POSITION_MM`, with every plug inverted:
    `pos = (conn−1)·49.92 + (63−local)·0.78`, Dreams 0–3 = X, 4–7 = Y.
  - Windows are taken in position, not channel number, and sorted by position.
  - `wft.model` builds every sharing/footprint matrix from `pos`, so neighbouring strips are neighbours in the fit.
  - The dead list (`det4_sps_h4.json`) is in FEU-03 numbering, as `prep_plane` compares.
  - X live bands show as diagonal segments in X-vs-telescope (X ≈ 150–162, 178–192, 200–215, 228–233 mm), so X is mapped
    correctly. A connector-reversal sawtooth would break these segments every 49.92 mm; they are continuous.
- **Dead regions against mapping artefacts.**
  - X has real dead amplification stripes. Tracks through them leave det4 reading another particle in a live band: the
    horizontal bands in the scatter. Hence the cross-view fiducial.
  - Y has no dead band under the telescope. The "dead Y band 144–160 mm" was ch 372/373.
- **Mount and alignment.**
  - All flat rows (run53, 55, 63, 66, 71) give det(A) = 0.991–0.997 and roll −89.8 to −90.3°.
  - Telescope x ↔ det4 Y (slope +1) and telescope y ↔ det4 X (slope −1).
  - z-scan minima: 1100–1140 mm (run63, 13.8 k tracks: 1100/1120), against 1120 for run 53.
  - run71 is flat (after the 2026-08-03 00:40 access), the same mount as run63_flat and run66.
- **Event matching.** FEU 1 and FEU 3 carry the same `eventId` with identical `timestamp` for every event (groups 000
  and 023: 16 960 and 17 696 events, Δt = 0). The fa80c06 eventIds run continuously across groups, so the `% 1e7` of the
  pass key is exact. The ~0.5 % overlap was a symptom, but of the telescope threshold (verdict 2), not of the matching.
- **The common-mode fix (a72b889).** The code was read and its test passes. After the seed fix, the RAW Y positions of
  telescope events show no residual pile-up at 158 mm: 0.49 % in the bin, against 0.37–0.53 % around it.
- **Estimators on the same events.** The analysis uses `analyze.py` (`seeded_fit`: robust to the 40–60 % of tracks
  through dead stripes; `FID_MM` = 1.5 cross-view fiducial). That machinery also reproduces run 53 (176 ± 10 µm,
  `baseline.json` 'all_seeded').

## Limitations (no new data can be taken)

- **Pointing.** The telescope model is measured once, on run 53 ZS (f = 0.301). The CF4 and RAW runs give no
  3-zone fit: the 0.5 mm zone holds too few fiducial tracks. Their per-zone det4 numbers carry the pointing model's
  uncertainty, of order the result itself; some zones even fall below the model's pointing. Only the residuals
  are model-free.
- **Statistics.** run71 measures only in the 1.0 mm back zone (≥ 150 events needed per zone). raw450 and raw275 have
  50 fiducial tracks each: not measurable. Group 010 of run71 is missing (its reco step wrote nothing; 1 of 36).
- **Telescope tables made before the final seed rule.** The `tel_job.py` tables were made with an intermediate seed
  mask: census-dead channels excluded, no hot probe. So:
  - ~1 % of X windows sit one strip off the final rule;
  - the run55 telescope sample still has ch 373 seeding Y, so its run55 numbers are not used in the verdict;
  - run71's ch 372/510 were excluded in both versions.
- **The hot probe is a threshold rule (3 % of seeds, 10× the median of the ±4 neighbours).**
  - It catches every oscillator seen in the pass tables except run61 d210, where ch 373 seeds 4.2 % of the row
    overall but under 3 % of the first 2000 events; that row's pile-up excess is 1.3 %.
  - It also flags some X channels: 172/173 in run55 d500–d700, 75 in ov00, and 70/74/128 in ov07 (beam-off
    triggers). Each seeds 10–40× its neighbours, which charge sharing cannot do, so they are judged hot. They are
    excluded from the seed and censored for those rows only.
- **pass_mid.** The 7.5 mm depth was chosen on run63 and applied unchanged elsewhere. It does not help rows without a
  measured v (run53, run55: the angle scale is the bench v).
- **Unexplained telescope artefacts.** The robust fit drops them, but they were not chased: a steep segment at
  telescope x 15–25 mm in run66, and a vertical stripe at telescope y ≈ 12 mm in run71.

## Files

| file | what |
|---|---|
| `tel_job.py` | per FEU group: telescope (ZS combined_hits, or a RAW scan), det4 through the pass code with the old and new seed, hits |
| `urw_raw_scan.py` | ZS-like uRWELL telescope for RAW run71 |
| `hits_baseline.py`, `analyze_baseline.py` | run-53 hits estimator on every telescope event; reproduces 176 µm |
| `analyze.py` | resolution per dataset and estimator → `results/audit.json`, `results/audit_resid.npz` |
| `pile_rows.py`, `pass_hists.py` | pass-1 Y pile-up per row; histograms for the report |
| `seed_diff.py`, `collect_seed_diff.py` | first event-level count with the intermediate mask (superseded by the census) |
| `seed_census.py`, `collect_census.py` | which events the committed fix changes, per row → `results/seed_census.csv` |
| `make_report.py` | `report/report.html` |
| `analyze_final.py`, `make_report_final.py` | the final numbers → `results/final.json`, `results/footprint.csv`; `report/report_final.html` |
| `footprint.py` | head-on footprint and leading-edge times of every flat row, no pass cut (condor) |
| `tel_cut.py`, `band_edge.py` | the telescope-cut justification; the live-band edge test |
| `condor/make_jobs_final.py`, `condor/dec53_job.sh`, `condor/promote_sps.sh` | final telescope job lists; run 53 fa80c06 decode; verified backup + promotion |
| `results/ledger_2026-10-31.md` | the ledger after the final promotion |
| `condor/` | job scripts, submit files, job lists |
