# Head-on two-gas A/B on det4 — what the beam data can still say, 2026-08-11

**Question asked** (Dylan, 2026-08-11): before shelving the SPS data for the
T14 campaign, do a close analysis of the head-on (flat-mount) data in det4's
working area, across the two gases we ran, and see what can be extracted.

**Answer in one line: the detector-response observables are gas-invariant at
the few-percent level across every condition the campaign varied — the
event-wise ±1 peak delay is +60 ns in all six arms — so det4's beam data
certifies that kernel-type observables are resistive-layer properties, and
the bench's sim targets cannot be excused by gas uncertainty.**

Companion products: `twogas_headon_lxplus.json` (+ the generator
`twogas_headon.py` here). Ran as lxplus condor cluster 11988756 on the staged
npz extractions; nothing on the laptop. Pre-registration written before the
job returned (`scratchpad twogas_prereg.md`, reproduced in §3).

---

## 1. The six arms

Same detector, same live-band fiducial (X 145–220 mm, uRWELL prediction where
finite, else X-view leading strip), same electronics (DREAM 180 ns, campaign
constant), Y view only (X view carries the standing tilt), central-strip peak
400–3000 ADC:

| arm | gas | resist | drift field | ZS | n clean |
|---|---|---|---|---|---|
| co2_590V | Ar/CO₂/iso 95/3/2 | 590 V | 243 V/cm | 5σ | 2 523 |
| co2_625V | Ar/CO₂/iso 95/3/2 | 625 V | 243 V/cm | 5σ | 3 709 |
| cf4_zs_770V | Ar/CF₄/iso 88/10/2 | 770 V | 243 V/cm | 4σ | 5 315 |
| cf4_raw_d700 | Ar/CF₄/iso 88/10/2 | 770 V | 243 V/cm | **RAW** | 3 992 |
| cf4_raw_d450 | Ar/CF₄/iso 88/10/2 | 770 V | 156 V/cm | **RAW** | 3 172 |
| cf4_raw_d275 | Ar/CF₄/iso 88/10/2 | 770 V | 95 V/cm | **RAW** | 2 380 |

Estimators are **uniform across all arms** — this is the point. The campaign's
own CO₂ ±1 numbers (+29/+36 ns) were mean-trace estimates while CF₄ RAW's
+60 ns was event-wise; they were never comparable. Everything below is
event-wise or per-sample-trim20, identically per arm, with a ZS-aware cleaner
(ZS mode: decoder-subtracted baseline, pile-up gate = any shipped pre-window
sample; RAW mode: the robust_waveforms recipe verbatim).

## 2. Results

| arm | rise 10-90 (stack) | on50→pk (stack) | ±1 shift (event-wise) | df ±1 | pk ratio ±1 | area ratio ±1 (±6-sample window) |
|---|---|---|---|---|---|---|
| co2_590V | 176.0 ns | 115.6 | **+60 / +60** | 0.88 | 0.298/0.300 | 0.354/0.358 |
| co2_625V | 179.1 | 115.7 | **+60 / +60** | 0.95 | 0.296/0.303 | 0.354/0.364 |
| cf4_zs_770V | 170.0 | 119.5 | **+60 / +60** | 0.98 | 0.312/0.304 | 0.356/0.348 |
| cf4_raw_d700 | 241.4 | 134.7 | **+60 / +60** | 0.99 | 0.329/0.323 | 0.422/0.412 |
| cf4_raw_d450 | 192.2 | 127.3 | **+60 / +60** | 1.00 | 0.316/0.326 | 0.410/0.430 |
| cf4_raw_d275 | 189.7 | 129.9 | **+60 / +60** | 0.99 | 0.313/0.316 | 0.403/0.409 |

RAW absolute-time (drift-ladder view), median normalized trace:

| arm | on50 | peak | off50 | rise 10-90 (abs) | late level |
|---|---|---|---|---|---|
| d700 | 830.9 | 1020 | 1480 | 254.8 | **−0.30** (end-of-drift lobe) |
| d450 | 839.2 | 1080 | 1457 | 274.9 | −0.10 |
| d275 | 835.2 | 1080 | 1450 | 310.4 | −0.20 |

d700 reproduces `RAW_RUN71_REANALYSIS_2026-08-04.md` (on50 803→831 under this
amplitude band, peak 1020 exactly, end-lobe −0.30 exactly) — the estimator
chain is certified against the committed record.

### The three findings

**F1 — the ±1 peak delay is a layer constant: +60 ns in all six arms.** Two
gases, three resist voltages (a ×~2 gain range), drift fields 95→243 V/cm,
two ZS thresholds and RAW, both signs of d, symmetric to ±3 %. Combined with
run_56's internal gain scan (c1 moved 1.1 % over 590→625 V) and run_63's
cross-gas c1 (+0.8 % X), the premise "sharing kernel = property of the
resistive layer + readout, not of the gas or operating point" now rests on
every lever the campaign has.

**F2 — the shaped core rise is gas-invariant at matched censoring.** CO₂
176–179 ns vs CF₄ 170 ns at the same drift field and comparable ZS — a 5 %
difference, sign consistent with the drift-velocity ratio (CO₂ drifts ~14 %
slower, feeding the shaper a slightly longer ladder). The detector response,
not the gas, owns the shaped core.

**F3 — ZS censoring is the dominant systematic in every slow observable, now
sized.** Same gas, same HV, same field: RAW rise 241 ns vs ZS 170 ns (the
rising edge's sub-threshold samples are simply absent in ZS), RAW ±1 area
ratio 0.41–0.42 vs ZS 0.35 (late shared charge censored), and the ZS stacks
end at a **+7–8 % pseudo-tail** where RAW shows the true **−4 to −9 %**
undershoot. This retro-explains the τ_s/c2 "non-convergence" across runs
(298→215→850 ns) as acceptance, as suspected — cross-ZS comparisons of any
tail quantity are invalid, full stop.

Bonus, from the RAW drift lever: peak-aligned rise falls (241→190 ns) while
absolute rise grows (255→310 ns) as the field drops — per-event pulses
approach the bare shaper response when cluster arrivals spread out, while the
event-to-event arrival jitter widens. Both directions are what a
drift-ladder ⊗ shaper picture predicts.

## 3. Pre-registration scorecard (written before the job returned)

| prediction | outcome |
|---|---|
| ±1 shift ≈ +60 ns in all arms if RC property; ZS arms may bias low | **+60 everywhere; no ZS bias** |
| core rise gas-invariant within ~15 % | **confirmed at 5 %** |
| df ±2: run56 0.3–0.5, run63 ~0.75, RAW ~1.0 | **WRONG for the ZS arms** — 0.83–0.86 (5σ) / 0.72–0.74 (4σ). The prereg quoted whole-sample acceptances; the q0 400–3000 band selects loud centrals where ±2 is nearly uncensored (the campaign's own 99.5 % at 900–3000 already said this) |
| ZS undershoot/off50 invalid by construction | confirmed (+0.07–0.08 pseudo-tails, discarded) |
| raw700 reproduces documented on50→peak and +60 ns | confirmed |

## 4. What this buys the T14 campaign

1. **The gas-uncertainty excuse is dead for kernel-type observables.** det3's
   bench gas (humidity, iso fraction) is unmeasured — but this A/B shows the
   sharing delay, sharing ratios, and shaped core are invariant across a far
   larger gas change (CO₂-mix → CF₄-mix, dry-equivalent v ratio 1.9) than any
   plausible bench contamination. Sim/data mismatches in these observables are
   detector-model physics, not gas.
2. **The transferable det4 target set** for any future sim-vs-SPS check:
   ±1 delay +60 ns (six conditions), matched-window pk/area ratios ~0.31/0.41
   (RAW), c1 ≈ 0.23–0.28 (campaign), end-lobe −0.30 at 233 V/cm, undershoot
   −4 % (RAW d700). Everything else on the SPS side is censored or
   gas-ambiguous.
3. **Which numbers to stop quoting:** cross-run τ_s and c2 (censoring), any
   ZS amplitude as a gain proxy (was already the rule), any ZS tail.

## 5. The wet-CO₂ bracket — LANDED, and the single-water-fraction test PASSES

Condor 11988752 (5 mixtures, Ar/CO₂/iso 95/3/2 + 0/0.5/1/1.5/2 % H₂O, water
displacing argon, CERN 720.8 Torr, ncoll 5, generator
`garfield_sim/wetco2_one_mixture.py`, tables in
`garfield_sim/results/drift_wetco2_*.json`).

v(243 V/cm): dry 40.10 → 28.66 (0.5 %) → 18.41 (1 %) → **12.73 (1.5 %)** →
9.71 (2 %) µm/ns. Against the measured CO₂-epoch 12.33 (run_57, ladder
method), the **implied water fraction is 1.57 %** — inside the CF₄ epoch's
independent 1.3–1.7 % bracket.

Three independent consistency hits at one water fraction (~1.5 %):

| observable | measured | predicted at 1.5 % H₂O | dry prediction |
|---|---|---|---|
| CO₂-epoch v(243) | 12.33 | 12.73 | 40.1 |
| CF₄-epoch v(233) | 14.0 | 14.4–15.2 (interp 1–2 %) | 74.7 |
| ratio CF₄/CO₂ | 1.14–1.17 | 1.13–1.20 | 1.89 |

The apparent per-epoch contamination discrepancy (measured/dry 0.31 vs 0.19)
was pure water-sensitivity difference between the mixtures — **one ~1.5 %
water floor explains both gases' transport simultaneously**, which is exactly
what the flow-equilibrium (R_outgas/Q) picture demands and a transient would
not give. The H4 gas-system water story is now closed at the level this data
can test. Caveats: η = 0 across the whole grid (the standing Magboltz
ncoll = 5 attachment limitation — this test says nothing about attachment);
the CF₄-at-1.5 % value is interpolated between the 1 % and 2 % tables (v is
strongly nonlinear in fraction, hence the geo/lin spread); do NOT transfer
1.5 % to the det3 cosmic bench, which is a different gas system.

## 6. Open

- A det4-configured run of the (post-transparency-fix) sim chain against the
  RAW arm is now *possible* — the wet-CF₄ drift+diffusion tables and the
  amp-range `.gas` exist — but det4's amplification gap is unpinned (its own
  assessment: continuously varying) so absolute amplitude would be
  uninterpretable; shape-only, if ever.
- No amp-range Magboltz table exists for the CO₂ 95/3/2 mix (drift-range
  only); only needed if a CO₂-epoch gain question ever becomes live.

## Reproduce

```bash
# products live on the staging disk next to the inputs:
#   ~/x17/sps_run53_det4_check/staging/twogas_headon_lxplus.json
# generator (this dir): twogas_headon.py  — ran on lxplus as:
#   python3 twogas_headon.py <dir-with-npz> twogas_headon_lxplus.json
# inputs: wf_m70V.npz, wf_run63_flat.npz, wf_run71_raw_det4only.npz
# condor: cluster 11988756 (espresso), LCG_105 view
```
