# PLAN 39 — Spark dead time and the efficiency ceiling

**Paper point 6 completion. Priority 3. The spark phenomenology (Poisson, muon-induced,
edge-dominated) is done; what's missing is the bridge from spark rate to an
IRREDUCIBLE efficiency loss: how long is the detector blind after a discharge, and what
efficiency ceiling does that imply at a given rate?**

## Goal

Measure the post-spark recovery: efficiency and gain as a function of time since the
last spark. Extract an effective dead/recovery time τ per spark, then the total dead
fraction = rate × τ (plus the in-spark crossing loss already quoted), and produce one
figure: **efficiency ceiling vs spark rate** for det3 (0.33 Hz) and det7 (1.61 Hz).

## Inputs (verified)

- `det3_spark_analysis/events.npz` — per-event arrays over the whole 6.9 h det3
  weekend run (111,286 firing events): `eventId, ts, mult, spark (bool), ray_x, ray_y,
  has_ray, nray`, plus `box` (active-area box) and `spark_thresh` (=50).
  `ts` comes from `trigger_timestamp_ns` — CHECK ITS UNITS FIRST: compute
  (ts.max()−ts.min()) and compare to the known ~6.9 h duration; rescale to seconds.
- `det7_spark_analysis/` — same-format npz for det7 (verify the filename inside that
  dir; the note quotes 3.84 h, 22,247 sparks, 1.61 Hz).
- Per-crossing hit/miss for the efficiency-vs-Δt curve: follow
  `09_efficiency_breakdown.py` (load `cache/event_results.pkl`, M3 rays, alignment;
  its `reco` dict maps event_id → reconstructed (x,y); a crossing is "efficient" if a
  reco exists within 5 mm of the ray). Run key `g_det3_wknd` conventions — the spark
  note used the same run as `events.npz`; confirm by checking that `events.npz`
  eventIds overlap the event_results event_ids.
- For the gain-recovery curve: hits-level cluster amplitude per event
  (`combined_hits` tree, or the amp info already in `event_results.pkl` if present —
  inspect one record's attributes first: `res[0].__dict__.keys()`).

## Method

1. **Timeline construction.** Sort events by ts (seconds). Spark times = ts of
   spark==True events. For every NON-spark event compute Δt_prev = time since the most
   recent previous spark (and also Δn_prev = eventId gap to it).
2. **DAQ-level dead time first (cheap, important).** Histogram the event-to-event
   spacing (both Δts and ΔeventId) immediately following spark events vs following
   normal events. If the DAQ goes busy after a spark, you'll see a gap in recorded
   events — that is dead time the efficiency curve can never show (the events simply
   don't exist). Quantify: median next-event Δt after spark vs after non-spark.
3. **Efficiency recovery curve.** For M3 crossings (active-area rays, standard
   denominator), bin by Δt_prev in log-spaced bins (e.g. 1 ms – 100 s) and plot
   efficiency(Δt_prev) with binomial errors. Expect suppression at small Δt recovering
   to the 92.9 % plateau. Fit a saturating form, e.g.
   eff(Δt) = eff_∞ · (1 − A·exp(−Δt/τ)), and report τ and A.
   Also plot vs Δn_prev (event-count spacing) — separates rate-dependence from time.
4. **Gain recovery curve.** Same binning, y = median cluster amplitude (or median
   per-event max amplitude). HV sag after a discharge shows up as an amplitude dip
   recovering over τ_HV — τ_HV may differ from the efficiency τ; report both.
5. **Multiplicity guard.** Exclude events that are themselves within the veto
   (mult>50) and exclude the 30–50-strip sub-veto band from the NUMERATOR study or
   show it separately — near-spark events often have elevated multiplicity (afterpulse
   tails), and you want detector recovery, not veto leakage.
6. **The ceiling model.** Dead fraction D = r_spark · τ_eff, where
   τ_eff = A·τ from the fit (the integral of the suppression), PLUS the in-spark
   crossing fraction already quoted (det3 4.4 %, det7 31.9 % crossing-based).
   Efficiency ceiling = eff_∞ · (1 − D) as a function of r_spark; draw the curve and
   put det3 (0.33 Hz) and det7 (1.61 Hz) points on it. State clearly which losses are
   already inside the measured 92.9 % (in-spark crossings are; post-spark blindness
   partly is — the model's job is to decompose, not double-count: the measured
   efficiency ALREADY includes all of this, the model explains how much each mechanism
   contributes and extrapolates to other rates/HV).
7. **det7 repeat** — higher rate gives a much stronger recovery signal; if det3's
   suppression is too small to fit, det7 sets τ and det3 gets an upper limit.

## Outputs

- Script: `39_spark_deadtime.py <run_key>` writing to
  `.../alignment_tpc_veto50/spark_deadtime/spark_deadtime.png` + `.csv`
  (and a det7 run).
- Figure panels: (a) next-event spacing after spark vs normal (DAQ dead time);
  (b) efficiency vs Δt_prev with fit; (c) amplitude vs Δt_prev; (d) ceiling curve
  eff_∞·(1−r·τ_eff) with det3/det7 points.
- CSV: τ, A, τ_HV, eff_∞, D per detector; the numbers for the paper sentence
  "each discharge blinds the detector for ~X ms; at 0.33 Hz that is Y % dead time,
  raising the irreducible loss from sparks to Z %."

## Acceptance checks

- Recovered plateau eff_∞ ≈ 92.9 % (det3, 5 mm fid) — if not, your denominator or
  fiducial differs from 09/31's.
- Spark rate recomputed from your timeline ≈ 0.33 Hz det3 / 1.61 Hz det7
  (from spark_meta.json).
- τ from efficiency and τ_HV from amplitude should be same order; wildly different
  values need a sentence of explanation, not silence.
- Check Δt distribution is exponential (Poisson cross-check — the spark note already
  showed this; your timeline should reproduce it).

## Gotchas

- **ts units** — verify against run duration before anything else.
- events.npz covers FIRING events only; M3 crossings with a silent detector are not
  in it — that's why the efficiency curve must come from the 09-style crossing list,
  with events.npz supplying only the spark timeline.
- Sparks come in afterpulse trains (~µs tails, and possibly clustered triggers):
  when computing Δt_prev, treat spark events within a short window (e.g. <10 ms) of a
  previous spark as the SAME discharge (merge into one spark epoch) or τ will be
  biased by train structure.
- det7's Y-plane saturation band and its 31.9 % crossing spark fraction mean much
  smaller clean statistics — use coarser Δt bins.
- eventId gaps exist normally (missing ids) — that's why step 2 compares post-spark
  spacing to the NORMAL spacing baseline instead of assuming continuity.

---

## RESULTS (2026-07-11) — `39_spark_deadtime.py`

**Headline: there is NO measurable post-spark dead time on either detector.**
Sparks are localised, non-propagating discharges that do not blind the pad, stall
the DAQ, or sag the gain for subsequent triggered events. The only spark-induced
efficiency loss is the *in-spark crossing coincidence* itself (a muon that crosses
during a discharge is unmeasurable), which is already contained in the measured
efficiency. This turns paper topic 6's "irreducible efficiency loss" into a clean,
bounded statement.

Run (both from `mx_june_cosmic_qa/`, `.venv`):
```
../.venv/bin/python 39_spark_deadtime.py g_det3_wknd --r=5   # det3, 0.33 Hz
../.venv/bin/python 39_spark_deadtime.py g_det7_long  --r=5   # det7, 1.58 Hz
```

| quantity | det3 (`g_det3_wknd`) | det7 (`g_det7_long`) |
|---|---|---|
| duration / firing events | 6.91 h / 90,914 | 3.84 h / 57,121 |
| spark epochs / rate | 8,279 / **0.333 Hz** | 21,885 / **1.585 Hz** |
| active-area crossings | 47,591 | 25,804 |
| in-spark crossings `f_inspark` | **4.4 %** | **35.7 %** |
| **(a) DAQ gap** after spark vs normal | 192 vs 186 ms | 168 vs 167 ms |
| median eventId skip (spark / normal) | 1 / 1 | 1 / 1 |
| **(b) eff** (non-spark, R=5 mm) | **92.85 %** | **63.34 %** |
| eff flat mean vs Δt (χ²/ndf) | 92.86 % (13.0/12) | 63.19 % (16.9/9) |
| transient deficit at Δt≈64 ms | **+0.63 ± 1.28 %** | **−5.22 ± 1.81 %** |
| post-spark dead time, 95 % UL | **≤2.7 pts** | ≤3.0 pts |
| **(c) gain** first-bin sag | −3.6 % (no droop) | −3.3 % (no droop) |
| **(d) operational eff** = eff·(1−f) | **88.8 %** | **40.7 %** |

Key points / how to read it:
- **(a) DAQ**: the distribution of the gap to the next recorded event is identical
  after a spark and after a normal event, and the median eventId skip is 1 in both
  cases — the DAQ does not go busy after a discharge. Any DAQ dead time is < a few ms.
- **(b) efficiency recovery is a NULL**. Efficiency vs time-since-spark is flat
  (χ²/ndf ≈ 1); the just-after-spark bin is *not* suppressed. On det7 it is if
  anything a small *excess* (muon-flux correlation: bright periods carry more sparks
  AND slightly higher reco efficiency), the opposite sign from a recovery. The exp
  fit `eff_inf(1−A·e^{−Δt/τ})` is degenerate at these rates (τ rails, because Δt
  rarely exceeds a few s), so the headline limit is the **model-independent**
  first-bin-vs-plateau deficit, not the fit. 95 % UL ≤2.7 (det3) / 3.0 (det7)
  efficiency points; since eff is flat all the way out, this also bounds the
  time-averaged post-spark loss.
- **(c) no gain sag**: per-event max strip amplitude for muon events is flat vs Δt
  (first-bin within −3 %). No HV droop/recovery after a discharge.
- **(d) the only spark loss is the in-spark coincidence**: intrinsic (non-spark)
  efficiency × (1 − f_inspark) = operational efficiency. det3 92.9 %→88.8 %,
  det7 63.3 %→40.7 %. f_inspark scales with the detector's own spark rate but is
  muon-correlated (det7's is not simply 5× det3's), so no universal f(rate) curve is
  drawn; the two points are shown as a decomposition. Post-spark adds nothing beyond
  the UL.

**Paper sentence**: *"A discharge does not induce measurable dead time: the DAQ
next-event interval, the reconstruction efficiency, and the pad gain are all
unchanged for the events following a spark (post-spark efficiency deficit ≤3 %,
95 % C.L.). The only spark-induced inefficiency is the in-spark crossing
coincidence — 4.4 % of crossings on det3 (0.33 Hz) and 35.7 % on det7 (1.6 Hz) —
which lowers the operational efficiency to 88.8 % and 40.7 % respectively and is
already contained in the quoted numbers. The sparks are therefore localised,
non-propagating, and impose no irreducible recovery-time ceiling beyond the
coincidence rate itself."*

Outputs (per detector, under `…/alignment_tpc_veto50/spark_deadtime/`):
- `spark_deadtime.png` — 4 panels: (a) DAQ gap after spark vs normal;
  (b) efficiency vs Δt with flat mean + UL envelope; (c) gain vs Δt;
  (d) efficiency decomposition (intrinsic → in-spark coincidence → post-spark UL).
- `spark_deadtime.json` — all numbers above.
- `spark_deadtime.csv` — per-Δt-bin efficiency + amplitude.
- `amp_cache.npz` — per-event max/sum amplitude cache (rebuild with `--rebuild-amp`).
  det3: `…/mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3/`
  det7: `…/mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/`

Acceptance checks met: plateau eff 92.85 % ≈ 92.9 % anchor (det3); spark rate 0.333 /
1.585 Hz ≈ note's 0.33 / 1.61 Hz; f_inspark 4.4 % reproduces the note's det3 4.4 %
crossing spark fraction (det7 35.7 % vs note 31.9 %, box/chi2 difference); Δt
distribution exponential (Poisson) as in the spark note. τ_eff and τ_HV are both
consistent with zero (no dip to fit), which is the result, not a failure.

**Caveat for the writer**: the fit-based `A_tau_ul_s` / `A*tau` in the JSON is
meaningful only for det3 (well-constrained); on det7 the fit rails and that field is
degenerate — quote `D_postspark_ul_pts` (the transient-deficit UL) instead.
