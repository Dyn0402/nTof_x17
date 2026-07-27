# det3 reconstruction recovered on the matched-filter hits

**Answers `HANDOFF_det3_reco_matched_filter_2026-07-25.md`.**
Written 2026-07-25, same day, on the same `a1cce79` hits — no reprocessing.

---

## 1. Result

det3 is back, and the position accuracy is now **better than pre-rework** while
keeping the low-amplitude sensitivity the analyzer rework bought.

| metric | pre-rework | broken (7-25 rerun) | **now** | target |
|---|---|---|---|---|
| sat_det3 within 5 mm | 93.4 % | 84.1 % | **93.1 %** | ≥ 93 % ✅ |
| sat_det3 core σ\|r\| | 0.48 mm | 0.64 mm | **0.45 mm** | ≤ 0.50 mm ✅ |
| sat_det3 median \|r\| | 0.80 mm | 1.01 mm | **0.76 mm** | ≤ 0.85 mm ✅ |
| sat_det3 sliding-map within | 94.4 % | 85.9 % | **94.1 %** | ≥ 93 % ✅ |
| sat_det3 σ_θ X / Y | 2.42 / 2.60° | 2.42 / 2.60° | **1.99 / 1.87°** | no worse ✅ |
| reco-at-all | 96.4 % | 90.2 % | **97.0 %** | — |
| has_any | 100.0 % | 99.4 % | **100.0 %** | — |
| spark_frac | 9.1 % | 17.3 % | **8.2 %** | — |
| 03 X / Y plane residual | — | 0.63 / 0.86 mm | **0.59 / 0.64 mm** | — |

Guard rails from handoff §5, all measured, none broken:

- **hybrid σ68** — lt5 HYBRID **1.626°** (was 1.63°), gt8 unshared+cal time-fit
  **1.636°** (was 1.64°). No regression. See §5 for the coverage caveat.
- **26 charge sharing still measurable** — FEU 7 c1 = 0.456 (was 0.450),
  FEU 8 c1 = 0.519 (was 0.522). Stable to ~1 %.
- **det4 keeps its gain** — see §3.

---

## 2. The fix

The handoff's mechanism (§2.3) is correct and I confirmed it in code: the
reported DUT position is the single earliest-time strip of the largest cluster
(`cosmic_micro_tpc_analysis.py:660`), with no quality requirement on that strip.
But the anchor rule is not the right place to fix it. Two findings changed the
approach:

**(a) The fix belongs before clustering, not at anchor selection.** Anchor-side
rules — require significance ≥ max(10, 0.3 × cluster max), earliest-among-top-N,
largest-cluster-by-charge, tighter `GAP_THRESHOLD_MM` — were all tried offline
and top out at **87.1 %** (from 84.2 %). Filtering the same hits *before*
clustering reaches **92.7 %** with the anchor rule untouched. Marginal strips
don't only steal the anchor, they bridge and reshape the clusters the anchor is
chosen from. Tightening the gap threshold does essentially nothing (84.2 → 84.3
at 4 mm), so this is not a clustering-parameter problem.

**(b) The spark veto was misfiring, and this is half the loss.** Handoff §2.1
correctly rules out the spark *threshold* — raising it doesn't help. But the
problem is the *observable*, not the threshold: `09` counts every strip that
passed the 5σ gate. The matched-filter analyzer admits residual coherent noise,
which inflates raw multiplicity, so ordinary muons cross a veto meant for
discharges. On identical data det3's spark_frac went 9.1 % → 17.3 %. Counting
only the strips the fit actually uses, with **the threshold left at 50**,
restores it to 8.2 %.

That the two are linked is what makes it one fix rather than two knobs: the same
noise strips wreck the position *and* fake the discharge tag.

### What was changed

A per-plane **relative** significance floor: keep strips with
`significance ≥ 0.10 × (that plane's strongest strip in that event)`.

Relative, not absolute, for two reasons — both measured:

- Gains differ ~2× across the fleet. An absolute floor is exactly the wrong knob
  the handoff warned about: `abs 50` takes det4 from 34.8 % → 25.8 %, i.e. it
  claws det3 back by killing det4. The relative floor takes det4 *up* to 40.1 %.
- The X and Y planes of one detector collect different fractions of the charge
  (26's measured c1 differs by up to 2× within a detector), so a per-*event*
  maximum over-cuts the weaker plane. This is not cosmetic: on det2 the
  per-event version gives 85.9 % and the per-plane version 91.4 %.

| file | change |
|---|---|
| `cosmic_bench_analysis/cosmic_micro_tpc_analysis.py` | new `apply_significance_floor()`; `SIG_REL_FLOOR = 0.10` |
| `mx_june_cosmic_qa/03_alignment_and_tpc.py` | apply the floor in `_load_hits()` before the veto; `--sigrel=` flag (`0` disables); cache sidecar (below) |
| `mx_june_cosmic_qa/09_efficiency_breakdown.py` | spark multiplicity counted on floor-filtered hits; `--sigrel=` flag |

`08` and `12` needed no change — `08` takes `has_any` from raw hits (correct: the
detector did fire) and `12` consumes `08`'s output.

**Cache safety.** The per-event cache filename is fixed (`08`/`09`/`12` read it
by name), so a cache built under a different floor would be silently wrong rather
than merely stale. `03` now writes `event_results*.meta.json` beside the pickle
and forces a refit when the hit-selection parameters differ.

---

## 3. Fleet: measured offline on all six detectors

Scored with the same `09` definitions against the stored alignments. Every
detector improves, and det4/det6/det7 land **well above** their pre-rework values
— the low-amplitude sensitivity is not just preserved, it now pays off.

| key | pre-rework | broken | relmax 0.10 per-plane | core σ pre → now |
|---|---|---|---|---|
| sat_det3 | 93.4 | 84.1 | **93.1** (full pipeline; 92.5 offline) | 0.48 → 0.45 |
| g_det4 | 20.7 | 35.3 | **40.7** (full pipeline; 40.1 offline) | 0.67 → 0.66 |
| o22_long_det2 | 91.1 | 80.3 | **91.4** (offline) | 0.44 → 0.44 |
| g_det6_long | 57.8 | 42.8 | **70.9** (offline) | 0.45 → 0.39 |
| g_det7_long | 43.1 | 16.7 | **51.2** (offline) | 0.59 → 0.59 |

The offline numbers run slightly low because the scan reuses the stored
alignment; refitting recovers the rest (det3 92.5 → 93.1, det4 40.1 → 40.7).

**det4 guard rail, measured in the full pipeline:** `has_any` = **95.6 %**
(≥ 95 ✅, unchanged — it is computed from raw hits and the floor cannot affect
it), `within5` = **40.7 %** (≥ 35 ✅, up from 35.3 % and nearly double the
pre-rework 20.7 %). Core σ 0.88 → **0.66 mm**, marginally better than the
pre-rework 0.67. Plane residuals 0.78/1.00 → **0.69/0.83 mm**, spark_frac
14.9 → 9.8 %. det4's median \|r\| is 1.14 mm vs 1.06 pre-rework — the one number
that does not beat pre-rework, and expected, since it now reconstructs roughly
twice as many crossings and the added ones are the harder low-amplitude
population.

The per-detector optimum of `rel` spans 0.08–0.20, but the curve is flat over
0.08–0.15 for every detector, so **0.10 is a single fleet-wide value**, not a
per-detector tune.

---

## 4. Why 0.10 and why threshold 50 — neither is fitted to the target

Both were set from the data before comparing to the acceptance criteria.

**Spark threshold.** Binning rays by surviving-strip multiplicity, det3
reconstruction collapses sharply: within-5mm is 99.2 % up to 20 strips, 82.2 %
at 20–25, then **43.7 %** at 25–30 and 25.9 % at 30–40. The same binning on
*raw* multiplicity degrades gradually (70.9 % / 55.0 % / 50.4 %) — i.e. the
filtered count is a genuine discriminator and the raw count is not. Independently,
keeping the threshold at its existing value of 50 on filtered strips reproduces
the pre-rework discharge rate (7.8 % vs 9.1 %), so the number did not have to
move at all.

**Floor value.** `within5` rises monotonically as the veto is loosened, so it
cannot be used to pick the floor. 0.10 sits mid-plateau for all six detectors and
minimises core σ on det3 (0.45–0.46 mm over rel 0.08–0.12).

Direct evidence for the coherent-noise interpretation, rather than "these are
sparks": det3 rays with raw multiplicity 100–150 reconstruct within 5 mm only
22 % of the time under the current pipeline, but **74 %** once the floor is
applied. They are muons buried in noise, not discharges — consistent with the
session-notes §3 observation that full-board events carry *lower* median
significance than clean ones.

### This is a CNS *residual*, and that is why it bites

Checked, because the mechanism depends on it: **CNS is ON** for these hits.
`WaveformAnalyzer.h:85` has `commonNoiseSubtraction = true` (cosmic-bench
default), `reprocess_cosmic_bench.py` never passes `--cns`, and the analyzer only
force-disables CNS on zero-suppressed data (median distinct channels < 256) —
June cosmics are RAW ~512-channel frames. Cross-check:
`Cosmic_Bench_DAQ_Control/processor_config.py` also has it `True`.

So the noise the floor removes is what *survives* CNS, and the reason a small
residual matters is the calibration asymmetry: the threshold is set on the
post-CNS noise floor (FEU 6/8 σ~10 ADC vs raw σ~115), so a residual worth only a
few percent of the original common mode is still a multi-σ excursion appearing on
many strips at once. CNS also subtracts a per-sample median across each
**64-channel block**, so it removes only what is common *within* a block and is
robust only while under half the block is signal-like.

Note `nTof_x17_DAQ/processor_config.py` (the beam DAQ) sets
`common_noise_subtraction: False`, deliberately, in commit `ca7baed` on 7-02
during beam prep. That is a different repo from the cosmic bench and largely moot
for ZS beam frames (the analyzer forces CNS off there anyway) — but any RAW-frame
beam run is a CNS-off hit generation and is **not** comparable to this data.

---

## 5. Caveats and what I did not do

- ~~Estimator coverage drops ~7 points in `34`.~~ **Withdrawn.** That came from an
  ad-hoc `34` run that omitted `--veto=50`. Re-run with the driver's flags,
  coverage is *identical* to the pre-fix baseline (lt5 0.9781, gt8 0.9722) and
  σ68 matches to the 4th digit. See `DET3_FULL_CHAIN_2026-07-25.md` — `34`, `31`
  and `33` turn out to be insensitive to the reco fix altogether, because they
  work from waveforms rather than the hits tree.
- **`v_drift_y` from `03` moved 38.0 → 33.5 µm/ns** on det3 (X: 32.5 → 31.5).
  The angle correlation also improved (corr_x 0.808 → 0.857, corr_y 0.826 →
  0.898). This is a physics-facing number and the shift is *toward* the
  established ~34 value, but it should be looked at before propagating into the
  paper. `26`'s unsharing-based v is unchanged in character.
- **The σ_θ anomaly (handoff §6.2) is gone but not explained.** sat_det3's
  bit-identical 2.417884172795585 / n_events 6306 is now 1.99 / 6197. Whatever
  froze it in the 7-25 rerun did not survive a clean refit; I did not find the
  cause, so treat it as unexplained rather than fixed.
- **Only det3 and det4 were run through the real pipeline.** det2/det6/det7
  numbers in §3 are offline (same scoring code as `09`, stored alignments). They
  should be re-run properly before any of it is propagated. Both detectors that
  *were* run came out above their offline estimate, so the offline numbers are
  more likely conservative than optimistic — but that is two points, not a rule.
- `cache/cshare.json` is written by `rerun_june_analysis.sh` parsing `26`'s
  stdout, not by `26` itself, so it still holds the previous run's values
  (0.450/0.522 vs the new 0.456/0.519, a ~1 % difference). Regenerate it via the
  driver if `27`/`28` are re-run.
- Nothing was committed or pushed. Previous sat_det3 / g_det4 outputs and caches
  are backed up under the session scratchpad.

---

## 6. Reproducing

```bash
cd mx_june_cosmic_qa
../.venv/bin/python 03_alignment_and_tpc.py sat_det3 --refit --full
../.venv/bin/python 03_alignment_and_tpc.py sat_det3 --refit --no-veto
../.venv/bin/python 08_efficiency_maps.py sat_det3
../.venv/bin/python 09_efficiency_breakdown.py sat_det3
../.venv/bin/python 12_efficiency_map_sliding.py sat_det3 --kernel=25 --grid=120
```

`--sigrel=0` on `03` and `09` reproduces the broken behaviour exactly, for A/B.

The offline scanner used for §3/§4 (seconds per configuration instead of a
10-minute refit, replicating 03's hit preparation and 09's scoring) is in the
session scratchpad: `anchor_scan.py`, `prefilter_scan.py`, `perplane_scan.py`,
`multdiag.py`, `spark_thresh.py`. Worth keeping if more knobs get swept.
