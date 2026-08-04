# The H4 beam as a calibration constraint on the June cosmic fleet

**2026-08-05 (overnight).** First re-analysis of the June cosmic data in the
light of the det4 SPS beam campaign (`sps_beam_test_26/`), on the reprocessed
(2026-07-24 matched-filter) hits now authoritative on EOS. Three questions:

1. Do the beam-measured kernel observables *reproduce* on the bench? (They
   must, if the kernel belongs to the resistive layer — the beam's own gain-
   and drift-invariance results say it does.)
2. Can the beam kernel *rescue* the det4 cosmic calibration, whose free fit
   falls into the v ↔ sharing degeneracy?
3. What does the comparison teach the rest of the fleet, which will never see
   a beam?

Everything below ran on the new machine from EOS + the repo; det4's legacy R&D
calibration (lost with the campaign desktop) was **not** used.

## 0. Inputs and their provenance

- `mx17_det4_day_6-24-26/long_run` re-pulled from EOS: reprocessed
  `combined_hits_root` (07-30 upload), `m3_tracking_root_v2`, `decoded_root`
  (unchanged since decode). The local stale pre-reprocessing hits were
  replaced; the hits-chain alignment + event cache were rebuilt from scratch
  (`03_alignment_and_tpc.py g_det4 --refit --veto=50`): z = 714/715 mm,
  θ = 90.10°, residual σ_X/σ_Y = 0.69/0.83 mm, 6,294 M3-matched events —
  in family with the 07-29 fleet run.
- Beam side: the run_71 RAW clean library
  (`robust_library_run71_raw.npz`, `RAW_RUN71_REANALYSIS_2026-08-04.md`) and
  the run_56 `share_lp` kernel fit (`M70V_FLAT_ANALYSIS.md`).
- Bench conditions for `g_det4`: Ar/iso 95/5, drift 600 V over the 28.8 mm
  column (208 V/cm), resist 495 V. Beam: Ar/CF₄/iso 88/10/2 (wet), drift
  233/150/92 V/cm. Different gas, different fields — which is the point: only
  layer properties may transfer.

## 1. The beam kernel observables reproduce on the bench (Y view)

`05_beam_kernel_xcheck.py` applies the beam's model-independent estimator
(robust_waveforms.py: leading strip, per-event ±d traces, peak-aligned
stacks) to near-normal cosmic tracks (|tan| < 0.06, central peak 400–3000
ADC) from the det4 wft calibration cache. Beam numbers are recomputed over
the *matched* relative window (−300…+1140 ns), since the bench window is
32 × 60 ns against the beam's 64.

| observable (Y view) | bench cosmics | beam run_71 (700/450/275 V) |
|---|---:|---:|
| event-wise ±1 peak shift, median | **+60 ns** (parabolic +54) | **+60 ns** (doc: +54–61) |
| ±1 / central, matched-window area | **0.63 / 0.66** | **0.65 / 0.63–0.66** |
| ±2 / central, matched-window area | 0.24–0.26 | 0.27–0.28 |
| ±1 / central, aligned-stack peak | 0.36–0.37 | 0.30–0.31 |
| ±2 / central, aligned-stack peak | 0.075–0.085 | 0.057–0.062 |

Read from the top: the **delay and the dispersed charge budget transfer
exactly** — across gas (Ar/iso ↔ wet CF₄ mix), amplification point, drift
field (92–233 V/cm beam, 208 V/cm bench), readout mode (ZS bench decode vs
RAW), and particle delivery (cosmics vs beam). This closes the loop the beam
campaign opened: the kernel is a property of det4's resistive layer, and its
two robust observables agree between completely independent datasets.

The stack-*peak* ratios sit ~20 % above the beam on the bench. That residual
lives exactly where the *gas* enters: prompt transverse diffusion is larger
in Ar/iso 95/5 than in the strongly-cooled wet CF₄ mixture, and prompt charge
adds at the peak while the dispersed copy adds in the tail. (The beam's own
α(±1) ≈ 0.19 prompt component was gas-specific for the same reason.)
Amplitude-matching the samples (q0 400–1000, the beam's spectrum) does not
move the bench ratios — it is not a gain effect.

Two side findings:

- **The X-view asymmetry has the same sign on the bench.** Bench stack-pk
  +1/−1 = 0.24/0.34; beam 0.19–0.21/0.36–0.38. The H4 analysis attributed
  det4's X asymmetry to the ~0.9° mount tilt; the bench (different mount,
  different continent) shows the same-sign asymmetry, so part of it is
  plausibly *internal* to det4 (strip plane vs mesh, or anisotropic layer
  coupling). The beam tilt number `tan θ_X = −0.015 ± 0.002` may carry a
  det4-internal component; worth remembering before using it as a pure
  mount constant.
- The estimator needs nothing but decoded waveforms + a handful of
  near-normal reference tracks. **It is now a bench-portable kernel
  measurement** — see §4.

## 2. The free det4 cosmic calibration falls into the degeneracy; the beam kernel is the cure

`wft.calibrate g_det4` (free 8-parameter fit, reprocessed data, no legacy
seed) converges to

    c1 = 0.067, c2 = 0.095, kY = 3.20, tau_s = 134 ns, sigma_s = 345 ns,
    sigma_p0 = 0.47 mm, Dp = 0.0136, v = 39.6 um/ns     (chi2 1.660e8)

— the det7 pathology (`FLEET_2026-07-29.md`): X-plane sharing driven to the
floor, kY blown up to compensate, σ_p0 inflated to absorb the missing
charge spread. The beam *measured* c1 ≈ 0.28 with a +60 ns delay on this
very detector; c1 = 0.067 is not a possible property of det4's layer. This
is the v ↔ sharing degeneracy again, and cosmics alone cannot break it.

The beam-pinned recalibration (`wft.calibrate g_det4 --fix-hyper
"c1=0.25,c2=0.10,kY=1.12,tau_s=60"`, i.e. Y-view sharing 0.28 = beam c1,
X = Y/1.12 from the beam's indicative kY, delay = the drift-invariant
event-wise peak shift) fits the remaining gas/electronics parameters:

    sigma_s = 268 ns, sigma_p0 = 0.42 mm, Dp = 0.0164, v = 40.05 um/ns
    (chi2 1.708e8, +2.9% over the free fit)

`--fix-hyper` is new in `wft/calibrate.py` (this session), alongside a fix
to `measure_dt_xy` for det4's mixed 32/37-sample windows.

### Reconstruction A/B/C/D

| metric | free (degenerate) | beam kernel (kY = 1.12) | + σ_p0 = 0.10 (kY free) | **+ v = 34 pinned** | fleet 07-29 (legacy calib, lost) |
|---|---:|---:|---:|---:|---:|
| has_any | 95.76 % | 95.76 % | 95.76 % | 95.76 % | 95.8 % |
| within 5 mm | 41.93 % | 41.69 % | 41.90 % | **41.96 %** | 41.65 % |
| core σ | 0.68 mm | 0.71 mm | 0.70 mm | 0.70 mm | 0.69 mm |
| σ_θ X / Y | 4.20 / 3.36° | 4.25 / 3.59° | 4.06 / 3.44° | **2.63 / 2.44°** | 2.73 / 2.58° |
| bias X / Y | −0.17 / −0.06° | −0.18 / −0.09° | −0.14 / −0.04° | −0.20 / −0.25° | −0.21 / −0.30° |
| implied-v spread X / Y | 5.2 / 5.7 | 5.6 / 5.7 | 6.1 / 6.4 | **4.1 / 4.5** | 4.9 / 5.0 µm/ns |
| n(reliable slope) X / Y | 3,163 / 3,046 | 3,117 / 3,183 | 3,454 / 3,268 | **3,753 / 3,736** | — |
| fitted v | 39.6 | 40.05 | 39.88 | 34.0 (pinned) | **34.2** µm/ns |

**The last column wins on every angle metric** — better σ_θ, smaller
implied-v spread, ~15 % more slope-reliable events than the legacy
calibration it replaces, at position parity. `calib_bundle_beamv34` has been
promoted to the canonical `calib_bundle` / `events.parquet` for `g_det4`
(the degenerate free fit is kept as `calib_bundle_free_degenerate`; all four
variants and their outputs sit side by side under `wft/`).

The validated recipe, ingredient by ingredient (each arm isolates one):

    kernel (c1, c2, tau_s)  <- measured (beam / bench estimator), pinned
    sigma_p0                <- physical initial cloud (~0.10 mm), pinned
    v                       <- from the data: implied-v plateau / drift scan, pinned
    kY, sigma_s, Dp         <- fitted (plane asymmetry, dispersion, diffusion)

Lessons, one per arm:

1. **Position is at parity everywhere** — within-5-mm and core σ do not care
   which calibration is used (41.7–41.9 % vs the fleet's 41.65 %). The June
   position/efficiency results are robust against all of this.
2. **Every free-v fit runs v to ≈ 40 µm/ns, and the data says that is
   wrong.** The reconstruction's own implied-v (median `w / tan_ref`) reads
   30–36 µm/ns in every angle bin, for every bundle — and the lost legacy
   calibration carried v = 34.2. `tan θ = w / v`, so a ~20 % high v maps
   directly onto the σ_θ gap. The ref-pinned calibration χ² *prefers* the
   wrong v (it trades against the kernel shape, §3, at only ~1 % in χ²);
   the *reconstruction* measures the right one. Re-pinning v = 34 is what
   converts a 4.1°/3.4° calibration into a 2.6°/2.4° one — the entire gap
   was v. This is the det7 lesson (`FLEET_2026-07-29.md`: "a fitted
   per-detector v is a nuisance parameter unless checked against an
   independent measurement") demonstrated on det4 with the kernel *known*.
3. **With σ_p0 pinned physical (0.10 mm), the fit demands kY ≈ 2.1–2.2** —
   the legacy R&D value (2.36), *not* the beam's tilt-contaminated 1.12.
   The beam's Y-view c1 and delay are trustworthy; its kY was flagged
   "indicative only" and that flag was correct. det4's Y plane really does
   share ~2× more than X, and the bench data recovers that on its own once
   σ_p0 stops absorbing it.
4. **σ_θ 2.63/2.44° at implied-v spread 4.1/4.5 is where the delay-kernel
   model tops out on det4.** The residual spread (det3 sits at 2.3) and the
   34 → 30 µm/ns implied-v falloff with angle are the remaining kernel-shape
   stress — the share_lp port (§3) is the identified next gain.

## 3. The delay-kernel *shape* is now measurably wrong (both bundles)

Forward-modelling the beam estimator from either bundle (normal-incidence
synthetic event, offsets marginalised over the pitch) over-predicts the ±1
peak (0.55–0.58 vs 0.37 measured) and places it too late (+120–130 ns vs
+54–60 measured), while matching the area budget. The plain
"c1 at τ_s + Gaussian σ_s" copy cannot simultaneously carry ~29 % of the
charge, peak only +60 ns late, and keep a peak ratio of 0.37 — an
RC-dispersed copy can, and that is precisely the `share_lp` structure the
beam measured (`M70V_FLAT_ANALYSIS.md` §3: "the plain-delay parameterisation
and the share_lp one are fitting the same physics; share_lp is the better
description"). Both cosmic fits compensate with σ_p0 ≈ 0.42–0.47 mm — a
half-millimetre initial cloud, unphysical against det3's 0.098 mm.

**Consequence:** porting `share_lp` into `wft/model.py` and fitting it
against the run_71 response library (RAW_RUN71_REANALYSIS §5.2) is now not a
nice-to-have; it is the identified next structural improvement of the fleet
reconstruction. The library (clean median W_d(t), three drift fields) is on
disk and the fitting recipe transfers to every chamber.

## 4. det3: the same estimator on the reference chamber

det3's fitted kernel (R&D: c1 = 0.306, kY = 1.375, τ_s = 47 ns) had never
been checked against a direct, track-fit-free measurement. On the Saturday
long run (reprocessed hits, rebuilt alignment: σ 0.59/0.64 mm, 6,763 matched
within 10 mm), |tan| < 0.06, q0 400–3000 ADC:

| det3 observable | X view | Y view |
|---|---:|---:|
| ±1 peak shift (parabolic median) | **+37 ns** | **+87 ns** |
| ±1 / central, matched-window area (ev) | 0.38 / 0.53 | **0.66 / 0.70** |
| ±1 / central, aligned-stack peak | 0.24 / 0.43 | 0.40 / 0.48 |
| ±2 / central, area (ev) | 0.04 / 0.11 | 0.36 / 0.41 |

- The X-view delay (+37 ns) sits at the fitted τ_s = 47 ns; the **Y view is
  both stronger-shared and ~2× later (+87 ns)**. A single-τ delay kernel with
  an amplitude-only kY cannot represent both planes — kY scales the copy but
  not its arrival. In the RC-sheet picture this is natural: the deeper/more-
  coupled plane sees a more dispersed (hence later-peaking) copy. One more
  argument that `share_lp` (with per-plane dispersion) is the right
  container.
- det3's Y sharing is *larger* than det4's (area 0.66–0.70 vs 0.63–0.66),
  while its X sharing is smaller. Kernels are per-detector — confirmed at
  the observable level, chamber against chamber.
- **The X-view ±1 asymmetry (−1 side stronger) appears here too, same sign
  as det4-bench and det4-H4.** Three datasets, three different mounts, one
  sign. A mechanical tilt story requires all three mounts to lean the same
  way; a common X-view systematic (the X strips' position under the
  resistive layer / readout stack) does not. The beam's
  `tan θ_X = −0.015 ± 0.002` for det4 — measured via the drift-invariant
  centroid walk, which is genuinely geometric — should still not be read as
  purely a mount property until the X-view sharing asymmetry is understood;
  and det3's cathode-tilt story (27.9 mm column) deserves the same second
  look.

## 5. What transfers to the fleet

1. **The kernel-measurement recipe.** Near-normal cosmics + the beam
   estimator measure each chamber's ±1 delay and charge budget directly,
   with no forward model and no v. Run it on det2/6/7 before their next
   recalibration; pin the kernel; pin σ_p0 physical; pin v from the
   implied-v plateau (or the drift scan); fit only (kY, σ_s, Dp). On det4
   this recipe *beat* the lost legacy calibration. This is what breaks the
   det7-class degeneracy *without* a beam — and det2/6/7's bundles are lost
   with the campaign machine anyway, so they need exactly this rebuild.
2. **Kernel invariance** (beam: gain 590→625 V, drift 92–233 V/cm; this
   work: across gas and delivery) — a bench kernel measured at one operating
   point is safe across that detector's conditions. The premise of the wft
   calibration bundles is now supported by three independent levers.
3. **The share_lp port** (§3), fitted to the run_71 library for det4 and to
   the bench estimator stacks for the others.
4. **v_drift stays per-run.** The bench det4 fit lands at v ≈ 39.6–40.1
   µm/ns at 208 V/cm (Ar/iso 95/5, near the velocity peak — plausible for
   dried gas, but not independently pinned). Nothing in the beam transfers
   here: the beam gas was 4–5× slowed by percent-level water. v must come
   from each run's own data (drift scan, end-lobe, or the fit) — unchanged
   conclusion, sharpened by the beam's water story.

## Reproduce

```bash
# data: EOS june_tests/mx17_det4_day_6-24-26/long_run (reprocessed hits + decoded)
mx_june_cosmic_qa/03_alignment_and_tpc.py g_det4 --refit --veto=50
# the winning calibration (now the canonical calib_bundle):
python -m wft.calibrate g_det4 --jobs 13 \
    --fix-hyper "c1=0.25,c2=0.10,tau_s=60,sigma_p0=0.10" --fix-v 34.0
python -m wft.cli reco g_det4 --matched-only
mx_june_wft/01_alignment.py g_det4 && mx_june_wft/02_efficiency.py g_det4 \
    --max-dropped -1 && mx_june_wft/03_angles.py g_det4 && mx_june_wft/digest.py g_det4
mx_june_wft/05_beam_kernel_xcheck.py g_det4 --tan-max 0.06     # the cross-check
mx_june_wft/05_beam_kernel_xcheck.py sat_det3 --tan-max 0.06   # det3's own kernel
# ablation arms kept beside it: calib_bundle_free_degenerate,
# calib_bundle_beampinned (kY=1.12), calib_bundle_beampinned2 (sigma_p0, v free)
```
