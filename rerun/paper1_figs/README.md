# Paper I §5, §9 and A1 on pass 1: figures, the edge zone, det4's stripes, the loss-class audit (2026-10-31)

All inputs are pass-1 products, and `Analysis_pass1` was only read. Outputs are on EOS at
`/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_figs/`.

## Verdict

1. **The edge zone is explained: a drift-field fringe at the two sides of the plate that have no passivation (x = 0 and
   x = 398.6 mm), over about 20–30 mm, plus a 0–5 mm strip on every side where the track's column leaves the strips.**
   Three independent signals see it with the same depth:
   - **the X-view angle bias.** The fitted X tangent minus M3 reaches 0.1–0.36 within 5–20 mm of an x side and falls
     below 0.01 by 30 mm. On det3, det4 (x = 0) and det6 the sign is opposite on the two sides (det2 reads negative on
     both, and det4's x = 398.6 side carries almost none), so drift lines converge inward near the side:
     charge released near the cathode lands further inside the plate than it started;
   - **the column duration** (FA colsum, every trigger, no M3, no track model). It is longer within 5–20 mm of an x side
     (det2 +170 / +480 ns, det3 up to +1100 ns at 5–10 mm), and those triggers carry more charge and more pulse strips:
     the edge strips collect drift electrons from outside the strip area, over longer paths;
   - **the efficiency turn-on.** It reaches the plateau by ~5 mm on the passivated y sides and by ~10–20 mm on the
     x sides.

   The passivated y sides are sharp for efficiency, but not for angles. A 10–30 mm y band keeps full efficiency (equal or
   better than the interior on all five chambers), while its Y-angle s68 is 1.3–1.9× the interior's on det2 and det3. So
   **the ≥ 30 mm interior is physically justified for angles and positions. For efficiency alone, 10 mm from the y sides
   would do (+9–10 % tracks).** That is Dylan's choice; nothing was changed.

   **The bow does not explain the edge T.** The dish predicts the edge column *shorter* (−45 to −64 ns, mesh nearer the
   cathode at the rim). The measured edge shift is −4 / +26 / +78 / +27 / −115 ns (det2/3/4/6/7). The fringe adds the
   opposite sign. In the interior, the column duration follows the bow surface with c = 0.73 ± 0.02 (det2),
   0.41 ± 0.06 (det3), 0.21 ± 0.14 (det4) of the full dish. det6 (2075 triggers) and det7 (X faults) don't constrain it.
   This agrees with the skeptic review's "column duration: inconclusive, sag mostly on the mesh".
2. **det4's stripes: bench and SPS agree on where they are.** In the SPS beam spot (x 152–231 mm), the live/dead
   classification of 1 mm bins agrees on 82 % (85 % with a 0.5 mm shift) between bench g_det4 (pass 1, M3) and run63
   flat700 (CF4, uRWELL telescope). The mirrored map agrees on only 52 %, so the orientation is fixed.
   - Live bands at SPS (run63): 152–165, 178–195, 200–215, 225–231 mm. Bench: 145–168, 176–217 (with a 30–60 % dip at
     182–195), 226–235.
   - Run53 (Ar/CO2/iso, lower gain) loses the marginal bands (162–182, 192–200 mm dead). **The stripes are a gain
     continuum, not a binary map:** which areas fall below threshold depends on the gas and the gain.
   - On pass 1, 52 % of the bench x range (interior) is below half the X plateau, and the two-view detection over the
     interior area is 41 %. The paper's "62 % never amplifies" came from a different definition (hits, gain) and
     should be re-worded or re-derived.
3. **A1 loss classes: the det3-tuned rules hold on the other four chambers, and no rule changed.** Six random events
   per class per chamber were displayed (both views, M3 and the pass fit) and checked by eye; contact sheets are
   `paper1_figs/a1/<det>/<class>.png` on EOS.
   - **Discharges, det6 17 % and det7 24 %** (the biggest losses in the fleet): real discharges, not random coherent
     noise. A median of 300–360 of 512 strips are above SNR 6 and 230–260 ring below −10σ, in both views, and the
     chip-wide blocks start at the muon's crossing.
   - **Discharge vs burst** (decision H; the §4 session's definition: a vetoed trigger is a discharge if a chip has
     ≥ 8 strips over 3000 ADC, otherwise a burst; `a1_burst_split.py`, `results/a1_burst_split.json`). The pass's veto is
     unchanged. Interior_good, % of tracks:

     | | det2 | det3 | det4 | det6 | det7 |
     |---|---|---|---|---|---|
     | one spark class (discharge + muon+other + shower) | 2.50 | 1.56 | 6.48 | 18.13 | 24.95 |
     | of which true discharge | 1.80 | 0.88 | 1.82 | 16.73 | 24.18 |
     | of which burst | 0.71 | 0.68 | 4.66 | 1.40 | 0.77 |

     det6/det7's spark losses are 92–97 % true discharges. det4's are 72 % bursts: its A1 "discharge" class (strip
     count) is 1.66 % true + 2.05 % burst. The A1 strip-count rule and the ADC-chip rule are different definitions; the
     paper should use one, and the ADC-chip one is the physical one. (det4 here is all of interior_good, n = 11 482, not
     interior_live.)
   - **The muon track is still visible in both views in most det6 discharge displays.** A fit with the ringing chip
     blocks censored might recover part of that loss. That's a reconstruction change, so it's not done here.
   - **The band, multi, delta, FIT_SAT, POS_TAIL and EDGE classes look as defined on every chamber.**
   - **`far:*` mixes two things.** About 22 % of the far losses on every chamber (32 % on det7), 0.3–0.7 % of tracks,
     are fits 5–8 mm from M3: the resolution or reference tail meeting the 5 mm detection radius, not a second deposit.
     On det7, `far:MULTI` therefore includes clean single tracks fitted 5–8 mm off. This is a property of the radius,
     not of the class rules; quote it next to the R = 3 / 5 mm efficiencies.
   - **On pass 1 the old deck's band and head-on events are detected**: the pass recovers them. The head-on class is
     0.02 %.
4. **Figures rebuilt on pass 1:** `efficiency_map` (all five chambers, 8 mm), `edge_turnon` (det3, three panels),
   `det4_stripes` (bench vs SPS profiles), and all 14 `loss_*` displays (det3, the pass fit, rr1 waveforms). 8 of the
   11 deck events keep their class and are reused; the rest were re-picked by eye from the sheets. `loss_headon` now
   shows the one `fit_failed` event. With the §4 figures rebuilt by the paper1_ops session, no Paper I figure is stale
   (`papers/FIGURE_PROVENANCE_2026-10-31.md`).

## Numbers

Edge study, golden rows (`results/edge_study.json`, `results/edge_angles.json`):

| | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| clean triggers (colsum) | 43 697 | 15 134 | 12 390 | 2 075 | 3 204 |
| column T, interior ≥ 30 mm [ns] | 829 | 803 | 729 | 1148 | 921 |
| T(edge < 30 mm) − T(interior) [ns] | −4 | +26 | +78 | +27 | −115 |
| bow prediction for that (whole dish in the gap) [ns] | −52 | −48 | −45 | −59 | −64 |
| bow fraction c in the gap (interior fit) | 0.73 ± 0.02 | 0.41 ± 0.06 | 0.21 ± 0.14 | (−2.4, not constrained) | (−1.3, X faults) |
| max \|tan bias\| x = 0 side, 0–30 mm | 0.22 | 0.20 | 0.11 | 0.13 | (no data) |
| max \|tan bias\| x = 398.6 side, 0–30 mm | 0.36 | 0.17 | 0.02 | 0.17 | 0.11 |
| tan bias at 30–45 mm | ≤ 0.009 | ≤ 0.004 | ≤ 0.007 | ≤ 0.016 | (X faults) |

Fiducial variant, golden rows: 10–30 mm from a y side, ≥ 30 mm from the x sides, against the ≥ 30 mm interior.
Each cell is that band's value, with the interior value in brackets:

| | tracks | efficiency | Y-angle s68 (tan) |
|---|---|---|---|
| det2 | +10 % | 0.930 (0.907) | 0.0345 (0.0186) |
| det3 | +9 % | 0.936 (0.926) | 0.0204 (0.0154) |
| det4 | +9 % | 0.452 (0.407) | 0.0398 (0.0361) |
| det6 | +10 % | 0.749 (0.745) | 0.0236 (0.0270) |
| det7 | +10 % | 0.640 (0.626) | 0.0223 (0.0245) |

det4 stripes (`results/det4_stripes.json`):

| | bench g_det4 | SPS run63 flat700 | SPS run53 |
|---|---|---|---|
| X plateau | 0.90 | 0.87 | 0.96 |
| tracks | 14 786 M3 rays | 87 683 telescope tracks | 91 832 |
| live/dead agreement with the bench (identity / mirror / best shift) | — | 0.82 / 0.52 / 0.85 (+0.5 mm) | 0.65 / 0.56 / 0.65 |

A1 fractions, golden rows (interior_good; det4 interior_live), from `rerun/effbudget/pass1/results/a1.json`, in % of
interior M3 tracks:

| class | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| detected | 90.31 | 92.64 | 71.30 | 74.43 | 68.04 |
| second particle (m3_picked / other_picked / merged) | 1.65 / 1.68 / 0.53 | 1.02 / 1.43 / 0.57 | 2.86 / 0.29 / 1.50 | 1.52 / 0.79 / 0.50 | 1.22 / 0.71 / 0.28 |
| spark: discharge / muon+other / shower | 1.41 / 0.38 / 0.71 | 0.50 / 0.40 / 0.67 | 5.04 / 0.92 / 0.58 | 17.15 / 0.26 / 0.72 | 24.29 / 0.23 / 0.43 |
| δ-ray / saturated δ blob / fit off | 1.00 / 0.88 / 0.12 | 0.90 / 0.84 / 0.23 | 0.70 / 0.48 / 0.75 | 0.34 / 0.32 / 0.20 | 0.56 / 0.38 / 0.03 |
| noise band | 0.32 | 0.22 | 1.04 | 0.35 | 2.12 |
| second deposit / edge / position tail | 0.47 / 0.12 / 0.18 | 0.19 / 0.15 / 0.20 | 0.46 / 0.31 / 0.15 | 1.05 / 0.16 / 0.01 | 0.99 / 0.08 / 0.10 |
| fit failed / cut other | 0.15 / 0.09 | 0.02 / 0.02 | 0.70 / 0.99 | 1.04 / 0.55 | 0.10 / 0.28 |
| tracks | 3 395 | 6 454 | 4 129 | 8 358 | 3 920 |

## Method

- **Edge, column duration** (`edge_study.py`). FA's `colsum.npz` of each golden row (s4_inputs / vt_check, made with the
  pass bundle). Clean = both views ≥ 5 pulse strips, one cluster, unsaturated. Position = each view's charge centroid.
  T = mean of the two views. Trimmed means (20 %) with bootstrap errors. Bow: the golden bundle's `bow.h` surface
  (wft.reco convention, M3 z up, h > 0 = mesh nearer the cathode), fitted as T = T_c (1 − c h / 30 mm).
- **Edge, M3** (`edge_angles.py`). The loss budget's rays (`paper1_eff/budget`, the `detected` definition) and the pass's
  own 03_angles `per_event.npz` (no fiducial cut), joined on event_id. Distance = the M3 ray at mid-gap to the nearest
  side of `TRUE_ACTIVE_BY_DET`.
- **Efficiency map** (`make_effmap_figure.py`). `rerun/effbudget/pass1/fine_maps.py` (load, emap), same pools.
- **det4 stripes** (`det4_stripes.py`).
  - Bench: X-view `cat_x_5 == detected` against x at mid-gap, y interior, 1 mm bins, with no dead-stripe map used.
  - SPS: the final telescope tables (`sps_audit/tel/G*`), one cluster in each uRWELL plane, the track at det4's z.
    A robust affine map telescope → det4 centroid, then X detected = centroid within 3 mm of the prediction.
  - Live = above half the 90th-percentile plateau.

- **A1 displays** (`a1_displays.py`, condor `condor/a1d.sub`, code archives `paper1_figs/code_02e9ab72` on EOS).
  - Per golden row and class: 6 random events (seed 17), from a1_recat.py's classes, the budget's rays (M3) and the
    pass's events.parquet (fit, acquisition).
  - Waveforms: rr1 `decoded_root_fa80c06`, data files only (pedestal files reuse event ids), each event from its own
    acquisition. Spark-vetoed events that the pass table doesn't hold were searched in every data file; event ids are
    unique within these rows.
  - Read with wft.io.FeuReader, with the strip map from the row's EOS run_config.

## Limitations

- **The A1 audit is by eye on 6 events per class per chamber** (≈ 400 displays), not a measured purity.

- **One row per chamber.** The edge study uses the golden rows only. The fringe depth may depend on the drift field;
  the 6-27 drift scan could test that, but it wasn't run.
- **colsum positions are charge centroids**, not M3: at the very edge, a centroid is pulled inward by truncation.
  The M3 profiles (angles, efficiency) don't have that problem and show the same depth.
- **The mechanism is inferred, not simulated.** The inward focusing is read from the bias sign and the extra charge.
  A Garfield field map of the drift cage near an unpassivated side would confirm it. No new data is needed for that.
- **det2's sign:** its X-view bias has the same sign on both x sides, and that isn't understood. A residual alignment
  rotation would add an antisymmetric term, but not a symmetric one.
- **SPS covers 80 mm of det4's x range only** (the beam spot); the rest of the stripe map is bench-only.

## Open questions for Dylan

1. Fiducial: keep ≥ 30 mm on all sides, or ≥ 30 mm on x and ≥ 10 mm on y for efficiency-only numbers (+9–10 % tracks)?
2. §9: replace "62 % of the area never amplifies" with the pass-1 numbers (52 % of the x range below half the plateau;
   41 % two-view detection), and say the stripes are a gain continuum (run53 vs run63)?
3. Should §5 show the edge figure for det3 (the paper's chamber) or det2 (10× the triggers, cleaner curves)?
   Both are in `figures/`.
4. A1: do det2/det4/det6/det7 get columns (table above)? `loss_headon` now shows a `fit_failed` event (the head-on
   class is recovered on pass 1): keep the figure with that caption, or drop it and say so in the text?
5. Is it worth trying a discharge-tolerant fit (censor the ringing chip blocks) on det6/det7, where the muon is
   usually still visible? How much it recovers is not measured (the discharge class is 17 / 24 % of tracks, an upper
   bound); it changes the reconstruction.
