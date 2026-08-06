# Reconstruction benchmark — critical review of the wft fit, 2026-07-29

Systematic A/B benchmark of the waveform-first fit on det3 (`sat_det3`), with
cross-detector consistency checks. Harness: `bench/build_cache.py` (production
windows + seeds + M3 truth, validated bit-identical to `events.parquet`) and
`bench/run_bench.py` (variant runner, 1,500-event fixed subset unless noted).
Baselines and all variant logs: `/tmp/wft_logs/bench_*.log`, JSON summaries in
`<det3>/wft/bench/`.

**Status: benchmarking in flight — numbers below are final where stated,
sections marked (pending) fill in as runs finish.**

## What the autopsy of the current production chain found

**1. The candidate seeder already solved multi-cluster seeding.** Of 155
reco_far events (2.24 % of rays), *zero* have more than one candidate cluster —
the 2026-07-29 seeder fixed the population it was built for. Selection
variants (pair coincidence, duration scoring) have nothing left to fix on det3.

**2. The remaining far tail is giant-charge events.** The 155 far events have
median fitted charge ~160 x a MIP, chi2/dof ~2000 (good fits: ~100), fitted
|tan| railed at 0.74 vs true 0.12, and the M3 reference is *inside* the fit
window for 90 % of them. Delta-ray blobs / small showers: the model cannot
describe the charge, so the chi2 fit slides off; the hits chain's crude
centroid was robust here (on common events the hits chain is +1.1 points
within-5mm). They are recoverable, and are what the `mf5` weighting and the
robust refit target.

**3. The chi2 was model-error dominated, and fixing the error model is a
broad win (`mf5`).** chi2/dof ~110 (X) / ~180 (Y) means percent-level template
mismatch on bright samples dominated the fit, not noise. Adding a 5 %
fractional model-error term to the per-sample sigma
(`wft.model.MODEL_FRAC = 0.05`):

| metric (1500-event subset) | baseline | **mf5** |
|---|---|---|
| within 5 mm (n_dropped<=2 cut) | 94.21 % | **94.96 %** |
| reco_far | 1.91 % | **1.16 %** |
| core sigma | 0.481 mm | **0.475 mm** |
| sigma_theta X / Y | 1.14 / 1.12 | 1.17 / **1.05** |
| implied-v spread X / Y | 2.6 / 2.8 | 2.8 / **1.0** |
| cost | 0.81 s/fit | 1.12 s/fit |

**4. The fleet-wide negative Y angle bias is a per-plane transverse-speed
offset `w0`, and it is a calibration constant, not a fit change.** Vertical
reference tracks fit with nonzero w. Measured (um/ns), with the angle bias it
predicts vs the bias observed in the fleet digest:

| det | w0 X | w0 Y | predicted bias X / Y | observed bias X / Y |
|---|---|---|---|---|
| det3 | -0.01 | -0.18 | -0.02 / -0.28 | -0.04 / -0.29 |
| det2 | -0.25 | -0.20 | -0.36 / -0.29 | -0.38 / -0.38 |
| det4 | -0.10 | -0.16 | -0.17 / -0.26 | -0.21 / -0.30 |
| det6 | +0.05 | -0.45 | +0.11 / -0.96 | +0.03 / -0.95 |
| det7 | +0.03 | -0.22 | +0.06 / -0.48 | (-0.10 / -0.31 degenerate cal) |

Every observed bias is explained. Correction: `tan = (w*1e3 - w0) / v_drift`
(`CalibrationBundle.w0`, applied in `reco.fit_plane`; measured by
`wft.calibrate.measure_w0` from free fits of reference tracks, or retrofitted
with `bench/set_w0.py`). Split-sample validation on det3-Y: bias -0.289 ->
+0.017 deg.

*Attribution caveat:* the Y offsets are ~-0.2 um/ns on four chambers whose
strip maps are all ~90 deg rotated against M3, so a single +0.3 deg systematic
in M3's tan_X would produce the common part. Detector-specific parts (det6-Y,
det2-X) exist on top. Operationally the correction is right either way — every
angle here is scored against M3 — but the *physical* detector tilt is not
separable from an M3 angle systematic with this data alone.

**5. The angle-bias metric itself was partly a selection artifact.**
`slope_reliable` selects on the *fitted* |tan| >= 0.08, which biases
near-threshold bins away from zero (up to +-0.5-0.8 deg in the central bins).
The benchmark adds a reference-selected metric (`comp14`: sign-folded angle
residual at |tan_ref| >= 0.14) that is unbiased by construction. On it, X is
clean (+0.00 deg) and Y carries a real symmetric ~-0.3 deg magnitude
compression at moderate angles — the last genuine Y-model defect (kernel-shape
diagnostics pending).

**6. Position is essentially at the physics + reference floor.** Per-axis
robust sigma 0.63/0.61 mm = (M3 pointing 0.40) + (detector ~0.45-0.49) vs the
toy-closure floor ~0.35 at mid-jitter; no pitch locking; resolution grows
0.53 -> 0.94 mm from flat to steep tracks as diffusion predicts. The
0.02 mm core-sigma gap to the hits chain on det3 is event-mix + the far tail,
not a core-precision deficit (hits core 0.465 vs wft 0.472 on common events).

## det7 — the calibration story resolved into a physics question

- Template thinness was real (Y template = median of 4 waveforms) but is NOT
  the root cause: with loosened cuts (`--tmpl-tan-min 0.10 --tmpl-min-amp
  250`, Y n=74) the v-pinned refit still validates at sigma 5.7/4.4 deg.
- Head-to-head validation on 220 held-out corridor events
  (`bench/val_calib.py`):

| det7 bundle | sigma X / Y (deg) | note |
|---|---|---|
| degenerate (free v -> 36.7, c1 ~ 0) | 1.24 / 1.30 | angles valid, kernel unphysical |
| production (v pinned 26.4) | 5.92 / 3.51 | broken |
| loose templates (v pinned 26.4) | 5.68 / 4.43 | still broken |
| **v pinned 36.6, det3 kernel seed, loose templates** | **1.50 / 1.19** | **physical kernel: c1=0.297, c2=0.084, kY=2.48, tau=63, sigma_s=80, sigma_p0=0.080; w0 = -0.01/-0.22 auto-measured** |

- The data insist on v ~ 36.6 um/ns although the HV monitor + run_config
  confirm det7's drift channel (0:7) at 700.25 V, where det3's drift scan
  gives 26.4 and det6 (same run, channel 0:6 at 699.5 V) freely calibrates to
  26.7. The fitted charge column is ~24-25 mm at v=36.6. A smaller-gap
  explanation is self-inconsistent (a 20 mm gap cannot hold a 24.5 mm
  column); the consistent story is a **normal gap with much faster gas in
  det7** — humidity/contamination raises v strongly at low field (the same
  physics as the June v-tension), and det7's 37 % spark fraction points the
  same way. Chamber-specific gas fault (leak?) — hardware follow-up.
- Operationally: **use `calib_bundle_v36` for det7** — physical kernel,
  validated angles, w0 measured. Do not quote det7 absolute v or depth until
  the gas question is settled.

## Variant results (1500-event det3 subset) — (pending: full table)

| variant | within5 | far | core | sX / sY | note |
|---|---|---|---|---|---|
| baseline | 94.21 | 1.91 | 0.481 | 1.14 / 1.12 | production 2026-07-29 |
| robust (no accept rule) | 94.55 | 1.57 | 0.506 | 1.17 / 1.15 | recovers tail, hurts core — superseded by min-move rule |
| mf5 | 94.96 | 1.16 | 0.475 | 1.17 / 1.05 | **wins nearly everywhere** |
| mf5_robust | 94.96 | 1.16 | 0.475 | 1.17 / 1.05 | identical to mf5 — under mf5 the chi2 scale collapses (chi2/dof ~ O(1)), so the chi2/dof>300 trigger never fires; ROBUST_REFIT is unnecessary with mf5, and CHI2DOF_BAD needs re-deriving |
| **mf3** | **94.96** | **1.16** | 0.481 | **1.10 / 1.05** | same tail recovery as mf5, better angles (s14 0.96/1.06), best median r (0.734), cost ~= baseline (0.84 s/fit) — current best single change |
| nmx (NM restart) | 94.21 | 1.91 | 0.478 | 1.15 / 1.11 | identical: the local optimizer is converged; closes the wrong-basin question (+70 % cost for nothing) |
| pair (time-coincidence) | 94.21 | 1.91 | 0.481 | 1.15 / 1.12 | identical: no multi-candidate failures left to fix on det3 |
| durw (duration-scored candidates) | 94.21 | 1.91 | 0.481 | 1.14 / 1.12 | identical, same reason |
| k22 (deeper charge basis) | 94.28 | 1.84 | 0.485 | 1.14 / 1.14 | the Y q_uend pile-up at the basis end does NOT distort the fit |
| candA: mf5 + kernel recalibrated under mf5 (v pinned, w0) | 95.16 | **0.95** | 0.487 | 1.33 / 1.11 | best tail but the halved c1 (0.137) under-models sharing and **breaks X** (cmp14_x +0.33): rejected |
| candB: mf5 + production kernel + w0 | 94.96 | 1.16 | 0.475 | 1.17 / 1.05 | **bias eliminated** (bX +0.02, bY −0.05) with all of mf5's gains — the composition works |
| fast (coarse pre-scan) | 94.28 | 1.84 | 0.486 | 1.14 / 1.10 | accuracy parity, ~30-40 % faster: adopt |
| mf10 | 94.41 | 1.70 | 0.539 | 1.32 / 1.25 | past the optimum — the MODEL_FRAC sweep peaks at 3-5 % |
| iter2 (2nd t0 scan) | 94.21 | 1.91 | 0.471 | 1.16 / 1.12 | noise-level: not needed |
| c1p10 / kyp10 / ayp15 / aym15 / ktau120 (800-ev diagnostics) | — | — | — | — | cmp14_y stays −0.17…−0.26 under every kernel perturbation: the residual Y compression (~2.3 % of slope) is not a single-knob kernel error |

**Recalibrating the kernel under the mf-weighted likelihood is NOT recommended:**
both free and v-pinned refits slide to c1 ~ 0.14 (half the validated sharing)
and the free fit also runs v to 39.4 against the drift scan's 36.6 — the
mf-weighted likelihood flattens the c1<->v valley rather than sharpening it.
Keep the kernels calibrated under the unweighted likelihood; apply the
weighting only at reconstruction time.

**Residual Y slope compression (~-0.25 deg at |tan|>=0.14, ~2.3 % of slope):**
stable under every kernel perturbation tried; X shows none. The physical
context (added 2026-07-29 evening): the resistive strips run in the Y
direction, so the Y readout sees diffusive RC spreading along the resistive
line — which is why every Y-specific term exists (kY = 1.4-2.5, the 4x deeper
Y undershoot, the Y w0 offsets, and this compression). The specific
RC-diffusion refinement of putting the +-2 copy at 4*tau instead of 2*tau
('rc4') was tested and is a wash (cmp14_y -0.38 -> -0.33, within noise — the
+-2 amplitude is too small for its timing to matter); the compression lives
deeper in the continuum line response. Until that is modelled, the empirical
per-plane slope constant `kw` (measured like w0 from reference tracks;
det3: kw_y = 0.967, kw_x = 1.007) removes it: split-validated
cmp14_y -0.47 -> -0.12. `kw` is in the bundle schema and applied at reco.
det4 shows the same signature at -1.0 deg (its legacy calibration needs the
det6/det7-style redo).

## Production recommendation (validated composition)

1. **`MODEL_FRAC = 0.03`** at reconstruction (env `WFT_MODEL_FRAC=0.03`).
   Re-derive `CHI2DOF_BAD` for the new chi2 scale (the old 300 never fires).
2. **`w0` in every calibration bundle** (`bench/set_w0.py --write` retrofit,
   or the new `measure_w0` stage on fresh calibrations).
3. **`PRESCAN_COARSE = True`** (K=9 x 120 ns global scan) — parity, ~1.5x
   faster.
4. **det7: `calib_bundle_v36`**; det4: recalibrate; det6: retrofit w0
   (its -0.95 deg Y bias is pure w0).
5. NOT adopted: robust refit (redundant under mf), pair/duration candidate
   scoring (nothing left to fix), NM restarts / iterated scans (converged),
   K=22 (no effect), mf-weighted kernel recalibration (degenerate).

## Full-set validation (all 7,093 events, both accountings)

| metric | production 2026-07-29 | **prod candidate (mf3 + w0 + fast)** |
|---|---|---|
| within 5 mm (n_dropped<=2) | 93.36 % | **93.87 %** |
| reco_far (cut) | 2.24 % | **1.73 %** |
| within 5 mm (no cut) | 95.61 % | **95.86 %** |
| core sigma | 0.472 mm | 0.476 mm |
| median r | 0.754 mm | **0.730 mm** |
| sigma_theta X / Y | 1.18 / 1.15 deg | **1.10 / 1.08 deg** |
| angle bias X / Y | -0.04 / -0.28 deg | **+0.00 / -0.02 deg** |
| s14 (ref-selected) X / Y | 1.04 / 1.04 | **0.98 / 0.98** |
| cost | 0.81 s/plane-fit | **0.51 s/plane-fit** |

In the 02_efficiency headline scale this is 93.47 -> ~94.0 within 5 mm, with
angles ~7 % narrower, the bias gone, and reconstruction 1.6x faster. The one
metric not improved is cmp14_y (-0.31 -> -0.41): the residual Y slope
compression, addressable by the optional `kw` slope constant.

## det3 golden-run, full chain on the v2 config (2026-07-29 evening)

Per user direction, the full chain (reco -> alignment -> efficiency -> angles
-> maps -> digest) was re-run on `sat_det3` only, with the final config:
`WFT_MODEL_FRAC=0.03`, `WFT_PRESCAN=1`, `WFT_CHI2DOF_BAD=250`, bundle
`calib_bundle_w0` (w0 = -0.01/-0.178, kw = 1.007/0.967). Previous outputs
preserved in `wft/prev_20260729_unweighted/`. Runtime: reco 19 min at 6 jobs
(was ~1 h), chain +5 min.

Everything through the standard 02_efficiency accounting:

| metric | hits chain | wft 2026-07-29 AM | **wft v2** |
|---|---|---|---|
| within 5 mm (headline) | 93.13 % | 93.47 % | **93.71 %** |
| reco_far | 3.89 % | 3.8 % | **3.58 %** |
| core sigma / median r | **0.448** / 0.764 mm | 0.470 / 0.763 mm | 0.467 / **0.739 mm** |
| reco-at-all | 97.02 % | 97.29 % | 97.29 % |
| has_any / spark_frac | 99.99 / 8.22 % | 100.0 / 8.22 % | 100.0 / 8.22 % (unchanged, as designed) |
| sigma_theta X / Y | 2.42 / 2.60 deg | 1.20 / 1.15 deg | **1.10 / 1.07 deg** |
| angle bias X / Y | — | -0.04 / -0.29 deg | **-0.01 / -0.00 deg** |
| implied-v spread X / Y | ~17 um/ns | 2.3 / 2.4 | 2.5 / **1.2** |
| cmp14 (slope-magnitude honesty) X / Y | — | +0.03 / -0.31 deg | **-0.02 / -0.07 deg** |

Cut accounting (n_dropped<=2): within 5 mm 91.76 %, reco_far **1.59 %**, core
0.465 mm, median 0.723 mm. The new alignment moved z_x 713 -> 714 mm and
theta 89.45 -> 89.40 deg (response to the changed positions; sub-pitch scale).
sigma_theta 1.10/1.07 sits essentially at the ~1.05 deg toy-closure floor.

**Open avenues (curiosity list, not blockers):** settle the M3-vs-detector
attribution of the common Y w0 (needs either a second reference or
cross-detector coincidence data); the 0.02 mm core-sigma gap to the hits
chain remains the measured cost of reference-free seeding.

## The continuum RC line investigation (2026-07-29 night)

Three measurements (`bench/rc_line_step{1,2,3}.py`), then a model upgrade:

1. **The Y-vs-X template difference is a uniform resistive leak, not
   transport.** T_Y = T_X (x) [delta + d/dt exp(-t/tau_g)] with tau_g ~ 7 us
   fits the measured Y template 88x better than X-as-Y; position-binned
   templates show tau_g flat along the strip (8.6/8.5/6.5/8.0 us) and X
   drain-free everywhere -> vertical leak through the resistive layer, not
   evacuation along it. Fleet-consistent: tau_g = 7.3 (det3), 5.8 (det2),
   5.4 (det6), 5.3 us (det7). A real detector characterisation number.
2. **Sharing is NOT line diffusion.** Pure RC diffusion consistent with the
   direct-strip shape predicts a +-1 amplitude of 0.01 vs the measured 0.40
   — falsified. The sharing is prompt lateral RC coupling.
3. **But the copy SHAPE was wrong in the model.** Measured directly on
   near-vertical tracks (dim neighbour of a bright strip, n ~ 1000/plane):
   the +-1 copy is a LOW-PASSED template — peak 90-130 ns later than the
   calibrated shift(47 ns)+smear(90 ns) copy, with a long late tail. A
   single-stage RC low-pass (template (x) exp kernel, tau = 230 ns X /
   410 ns Y) fits the measured copy 14x (X) / 54x (Y) better.

**Model upgrade `share_lp` (RC-ladder kernel):** +-1 copy = template (x)
exp(tau_s), +-2 = two cascaded stages; per-plane tau via kTauY. Recalibrated
on det3 (`calib_bundle_lp`, v pinned 36.6). *(Correction: an earlier draft
claimed a 24 % total-chi2 drop — that compared a warm-t0 evaluation against
a cold one; cold-vs-cold the lp kernel wins by ~1.3 % (1.4025e8 vs
1.4209e8), warm-vs-warm measurement below.)* The acid test does pass —
**the auto-measured slope scales come out kw = 0.994 (X) / 1.011 (Y)**: the
-3 % Y compression the empirical kw was patching is explained and removed by
the copy shape. w0 also shrinks (Y -0.18 -> -0.12).

Reconstruction-level (1500-event bench, prod config): lp bundle at parity —
within5 94.75 vs 94.96, core 0.487 vs 0.475, sX 1.11 vs 1.17 (better),
sY 1.11 vs 1.05 (worse), median r 0.726 (best), cmp14 +0.02/-0.08 with NO
empirical patch. Its hyper fit sits in the c1<->sigma_p0 valley
(c1 = 0.077, sigma_p0 = 0.373) and stopped at maxiter — likely tunable
(fit kTauY, longer convergence).

**Convergence runs** (`calib_bundle_lp2`, continuation; `calib_bundle_lp_hi`,
high-sharing seed): both land at the same optimum — c1 at/near the 0.05
floor, sigma_p0 ~ 0.4 mm, kY 2.3-2.9, tau 122-146 ns, kw = 1.010 — so under
the corrected copy shape the likelihood genuinely reattributes most of the
old "discrete sharing" to transverse charge spread. Not a seed artifact.
(Chi2-comparison caveat: warm-t0 vs cold-t0 evaluations differ by ~20 %, so
total-chi2 kernel comparisons are unreliable; the copy-shape measurement and
kw -> 1 are the evidence.)

Converged lp2 bench (1500 events, prod config) vs candB:

| | candB (old kernel + w0/kw) | **lp2 (RC-ladder)** |
|---|---|---|
| within5 / far | 94.96 / 1.16 | 94.89 / 1.23 |
| core sigma / median | 0.475 / 0.764 | **0.468 / 0.709** |
| sX / sY | 1.17 / **1.05** | **1.08** / 1.19 |
| cmp14 X / Y | +0.05 / -0.37 (pre-kw) | +0.09 / **+0.03** (no patch) |
| s/fit | 0.58 | **0.32** |

**The sY trade-off dissolved under scrutiny (2026-07-30).** A Y-knob scan
(kTauY, sigma_sY, sigma_p0Y — per-plane overrides added to the model) showed
kTauY trades sY against cmp14_y monotonically with the measured 1.78 already
optimal, and no knob helps — because the "regression" is a METRIC artifact:
`slope_reliable` selects on the fitted |tan| > 0.08. Reference-selected,
per-|tan_ref|-bin, the lp kernel's Y resolution aggregates to **1.02 vs
candB's 1.05 deg** over 0.08-0.45, and at near-vertical it is **1.50 vs
2.67 deg** — nearly 2x better where candB's Y scatter blows up.

**det3 golden chain v3 (RC-ladder `calib_bundle_lp2`, prod env).** One
procedural finding: w0/kw measured on the 150-event calibration corridor
differ from the production-run values by ~0.06 um/ns / 1.5 % (corridor
windows bias the constants) — the production retrofit (`set_w0.py --bundle
... --write` + re-reco) is the correct second pass and zeroes the biases.
Final, standard accounting:

| | hits chain | v2 (old kernel + w0/kw) | **v3 (RC-ladder)** |
|---|---|---|---|
| within 5 mm | 93.13 % | **93.71 %** | 93.54 % |
| core sigma / median r | 0.448 / 0.764 | 0.467 / 0.739 | **0.460 / 0.708** |
| sigma_theta X / Y (slope_reliable) | 2.42 / 2.60 | 1.10 / 1.07 | 1.08 / 1.11 |
| Y sigma, ref-selected 0.08-0.45 | — | 1.05 | **1.02** |
| Y sigma, near-vertical (<0.08) | — | 2.67 | **1.50** |
| angle bias X / Y | — | -0.01 / -0.00 | -0.03 / -0.01 |
| reco wall time (8 jobs) | — | 19 min | **13 min** |

**Fleet recommendation: adopt `share_lp` (RC-ladder) as the fleet
configuration.** It is at or above parity on every honest metric, physically
grounded (its constants — tau_RC per plane, tau_g, w0 — are measurable
detector properties that transfer), fastest, and free of shape patches; v2
remains as a validated fallback. Per-detector procedure: `wft.calibrate
<key> --share-lp --fix-v <drift-scan v> [--tmpl-tan-min 0.10 --tmpl-min-amp
250 on low-gain chambers]`, reco with `WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1
WFT_CHI2DOF_BAD=250`, then the `set_w0.py` production retrofit + re-reco.

## Recommended production changes — (pending final composition)

1. `MODEL_FRAC = 0.05` (fractional model-error weighting).
2. `w0` per plane in every bundle (`set_w0.py` retrofit or recalibration).
3. Robust refit with basin-change acceptance for chi2/dof > 300 planes
   (pending chain-2 numbers).
4. Coarse-basis global pre-scan for speed (pending parity check).
5. det7: adopt-degenerate-for-angles + resolve the field/gap question.
