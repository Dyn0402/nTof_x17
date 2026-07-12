# PLAN 38 — X/Y charge balance through the pixelated top layer

**Paper point 4. Priority 2. The only topic on the paper list with NO existing
measurement anywhere in the repo.**

## Goal

Measure whether the pixelated top layer distributes an event's charge effectively
between the X strip layer and the Y strip layer underneath (Y is at the bottom).
Deliver: the q_X vs q_Y correlation, the balance-fraction distribution
f = q_X/(q_X+q_Y) (median, width), and its dependence on track angle, position on the
detector, and total charge. Compare det3 vs det2.

## Physics framing (what the numbers mean)

The avalanche charge lands on the pixelated top layer; pixels route it capacitively/
resistively to the X strips and (below them) the Y strips. If the routing works, f is
narrow and position-independent; its central value need NOT be 0.5 (Y sits below X and
may systematically see less charge) — the *constancy* is the design success metric, the
central value is a design number worth quoting. A wide or position-dependent f would
mean uneven pixel coupling. Event-by-event spread of f at fixed total charge measures
the sharing fluctuation.

## Inputs (verified)

Two levels — do both, they cross-check:

1. **Hits-level (full coverage, ~100 % of firing events).**
   `CFG.combined_hits_dir` root files (tree `hits`, one row per strip pulse:
   `eventId, feu, channel, amplitude, time, ...`). det3 sat run: FEU **7 = X**,
   FEU **8 = Y** (verify against `qa_config.py` — never assume for other runs).
2. **Segment-level (cleaner, ~51 % of tracks).**
   `.../alignment_tpc_veto50/microtpc_metrics/microtpc_segments.csv`
   (columns verified: `eid, plane, S_um_ns, n_strips, amp_sum, pos_lead_mm, t_span_ns`;
   one row per plane per event — pivot on `eid` to pair the planes; check the actual
   values in the `plane` column first, likely 'x'/'y').

Reference/selection inputs:
- M3 rays (`M3RefTracking`, standard v2 recipe) for angle θ_ref and ray position;
  alignment from `alignment.json` (follow the load pattern at the top of
  `09_efficiency_breakdown.py` or `31_microtpc_metrics.py`).
- Spark veto: drop events with >50 hit rows total (SPARK_THRESH=50 convention).
- det2 twin: run key `o22_long_det2`, outputs under
  `.../mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/alignment_tpc_veto50/`
  (det2 FEUs are 6/8 — read the map from qa_config).

## Method

1. Per event, per plane: q_plane = Σ amplitude over that plane's strips with
   amplitude ≥ THR_HIT = 100 ADC. Tag events where any strip is ADC-saturated
   (amplitude at/near the 12-bit clip; baseline ~220, so clip ≈ 4095−220 ≈ 3875 —
   inspect the amplitude histogram endpoint and pick the clip value empirically).
2. Match to M3 (eventId ↔ evn, single clean ray), require the ray inside the active
   area with a 5 mm margin, both planes fired.
3. Core sample: spark-vetoed, unsaturated, |θ_ref| available. Compute:
   - scatter/2D-hist q_X vs q_Y + Pearson r (expect strongly correlated — both sample
     the same avalanche);
   - f = q_X/(q_X+q_Y): histogram, median, σ68; report also on the log-ratio
     ln(q_X/q_Y) which is better behaved;
   - f vs |tanθ_ref| (profile) — inclined tracks spread over more strips/pixels; a
     flat profile = routing independent of topology;
   - f vs ray position: 2D map of median f over the active area (this is the "pixel
     layer uniformity" map — the paper figure). Watch the known low-X edge (bad strips)
     and mask the 0–25 mm fringe band or show it separately;
   - f vs total charge q_X+q_Y (profile) — nonlinearity/saturation check; overlay the
     saturated-event population separately;
   - width of f at fixed total charge (bin by q_tot, plot σ68(f)) = event-by-event
     sharing fluctuation.
4. Repeat 1–3 on the segment-level `amp_sum` (already clustered, spark-free, unshared
   footprint) — numbers should agree in the overlap; differences quantify how much the
   hits-level sums are contaminated by noise strips.
5. Repeat for det2 (`o22_long_det2`). Same f central value across detectors = design
   property (like the sharing constants); different = per-chamber assembly effect.
   Note det2 ran resist 525 V (different gain) — compare f, not absolute q.

## Outputs

- Script: `38_xy_charge_balance.py <run_key>` (default `sat_det3`).
- `.../alignment_tpc_veto50/charge_balance/xy_charge_balance.png` — panels:
  q_X vs q_Y 2D, f histogram (det3 + det2 overlay), f vs |tanθ|, f-map over the
  detector, f vs q_tot with saturation overlay, σ68(f) vs q_tot.
- `.../charge_balance/xy_charge_balance.csv` — one row per selection variant
  (hits-level / segment-level / det) with n_events, r, median f, σ68 f, slope of f vs
  tanθ, max |Δf| across the position map (excluding masked edge).
- Appended results section in this file.

## Acceptance checks

- Event counts: det3 sat run has ~15.3k matched events (before the both-planes and
  saturation cuts) — if you get far fewer, the M3 match or FEU selection is wrong.
- q_X vs q_Y Pearson r should be high (≥0.8) for clean events; if near zero, you are
  pairing wrong eventIds (check the eid pivot / evn matching).
- Hits-level and segment-level medians of f agree within a few %.
- The f map should be flat in the core; structure at the low-X edge is expected
  (known bad strips / spark region) — mask, don't average over it.

## Gotchas

- **Y-plane (FEU 8) saturation**: det7's Y saturation band is documented; det3 less so,
  but always separate saturated events rather than mixing them in.
- Amplitudes have no CNS; hits-level sums include any noise strips above 100 ADC —
  that is exactly what the segment-level cross-check (step 4) bounds.
- One row per STRIP in the hits tree — group by (eventId, feu) then sum.
- The sat-run FEU 8 file-003 recovery: hits are complete NOW, but if you filter by
  eventId ranges copy the per-FEU live-range guard pattern from 31/32 anyway
  (defensive; it is a no-op on complete data).
- det2's f uses det2's OWN alignment/active box — don't reuse det3's.

---

## RESULTS (2026-07-11) — DONE

Script `38_xy_charge_balance.py <run_key>` (run FROM `mx_june_cosmic_qa/`;
`o22_long_det2` first so the det3 figure overlays it). Outputs per run in
`<alignment_tpc_veto50>/charge_balance/` — `xy_charge_balance.png` (6 panels),
`xy_charge_balance.csv` (one row per level/charge/det variant), `f_hits_core.npy`
(for the twin overlay).

**Headline: the pixel top layer splits the avalanche charge EVENLY and STABLY
between the X and Y strip layers.** f = q_X/(q_X+q_Y) is narrow (σ68 ≈ 0.07),
essentially position- and angle-independent, and its central value is close to
0.5 with a small per-chamber offset.

| quantity | det3 (`sat_det3`, 490 V) | det2 (`o22_long_det2`, 525 V) |
|---|---|---|
| core events (matched, fiducial) | 15,129 | 10,589 |
| median f (hits, amplitude) | **0.487** | **0.531** |
| σ68(f) | 0.067 | 0.071 |
| f (integral, unsat subset) | 0.480 | 0.527 |
| f (segment `amp_sum`) | 0.488 | 0.509 |
| Pearson r(q_X, q_Y) | 0.881 | 0.846 |
| ln(q_X/q_Y) median | −0.052 | +0.122 |
| slope df/d\|tanθ\| | +0.021 | −0.003 |
| f-map inner std (stat expectation) | 0.020 (0.009) | 0.021 (0.011) |
| saturation systematic Δf(sat−clean) | −0.007 | −0.037 |

**Reading the numbers.**
- *Balance.* det3 sits essentially at 0.5 (Y marginally favoured); det2 at 0.53
  (X marginally favoured). The ~0.04 chamber-to-chamber offset is a per-assembly
  effect (like the sharing constants), NOT a universal design number — but BOTH
  chambers are narrow and flat, so the routing works in both. The constancy, not
  the central value, is the design-success metric (as framed in the plan).
- *Three independent charge proxies agree* within ≤0.008 (det3) / ≤0.022 (det2):
  hits-level saturation-corrected amplitude, hits-level `integral` (pulse area,
  measured on the unsaturated subset where it is unbiased), and the clustered
  unshared `amp_sum` from `microtpc_segments.csv`. So f is not an artefact of the
  peak-amplitude estimator. (integral's σ68 is much tighter, 0.011–0.026 — area
  averages out peak noise — but its *median* matches, which is the point.)
- *Position.* The median-f map is flat: inner-frame bin-to-bin std 0.020–0.021
  vs 0.009–0.011 expected from finite per-bin statistics → real position
  variation only ~0.018 (≈2 % of the central value). Structure sits at the top-y
  and low-x edges (known bad-strip / fringe band), which the 1-bin inner frame
  excludes.
- *Angle.* f vs |tanθ| is flat (slope ≈ +0.02 det3, ≈0 det2) — routing is
  independent of track topology, as expected if the pixel layer couples charge
  before it reaches the strips.
- *Saturation handling (important deviation from the plan).* The combined_hits
  `amplitude` branch is already saturation-CORRECTED (saturated strips have
  median amplitude ~4750 ADC, above the ~4150 raw `local_max` clip). The plan
  assumed `amplitude` was the raw clipped value and said to separate saturated
  events out; doing that here would drop ~40 % of tracks (any event whose peak
  strip clips) and bias f toward low charge. Instead the script keeps saturated
  events (using the corrected amplitude) and uses the per-hit `saturated` flag
  only to SPLIT the core as a systematic. det3 Δf(sat−clean) = −0.007 is
  negligible; det2's −0.037 is larger (higher gain → correction matters more) and
  is quoted as the systematic. Only genuine fit divergences (per-strip amplitude
  > 5×10⁴ ADC; 1–4 events per run) are dropped.

**Acceptance checks (plan §Acceptance): all pass.** det3 matched count 15,133
≈ the ~15.3k anchor; r(q_X,q_Y) = 0.88 ≥ 0.8; hits vs segment medians agree to
0.001 (det3) / 0.022 (det2); f-map flat in the core with edge structure only.

**Note on coordinate frame (trap for re-runs):** fiducial and the f-map use the
production detector hit position `det_{x,y}_mm` (strip-map frame [0, ~399], the
frame `active_bounds` returns). The M3 `ref_{x,y}_mm` live in the aligned/
telescope frame (origin offset, range ≈ [−220, 195]) and must NOT be fed to a
[0,399] fiducial box — doing so silently keeps a biased corner (3.2k of 15.1k
events). The ray ANGLE still comes from `ref_tan_theta`.

**Standalone LaTeX report (7-12):** `report_charge_balance/main.pdf` (6 pp) —
physics-schematic explanation of the routing + metrics, then paper-/talk-ready
figures. Built from `38b_charge_balance_report_figs.py` (reads the per-event
`charge_balance_events.csv` cache that `38_xy_charge_balance.py` now writes, plus
the summary CSV; emits `report_charge_balance/figs/`: `schematic_routing.png`,
`schematic_metric.png`, `qx_vs_qy.png`, `f_distribution.png`, `f_maps.png`,
`f_stability.png`). Rebuild: run 38 for `o22_long_det2` + `sat_det3`, then 38b,
then `pdflatex main.tex` (×2) in `report_charge_balance/`.
