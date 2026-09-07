# Handoff — how much charge is the flash actually delivering?

**Written 2026-08-09.** For the next model or operator. Everything claimed here
is reproducible from `analyze.py` plus a mirror of the HV-monitor CSVs; nothing
needs the bulk data.

---

## 0. Why this exists

We have a measured, monotonic, two-decade map of **post-flash dead time vs
resistive HV** (`nTof_x17_DAQ/docs/flash_recovery_run57_HV_map_2026-07-20.md`)
and a well-argued mechanism for it (`daq:analysis/dream_saturation_7-12-26/
DREAM_GAMMA_FLASH_SATURATION.md`): the DREAM CSA is pinned against its rail for
as long as the input current exceeds its ~9–90 nA feedback limit.

What we did **not** have is the abscissa that turns that from an operational
map into physics: **how much charge**. Everything that could tell us goes
through the saturated front end, so the DREAM data cannot answer it.

The resistive-layer HV supply can. It carries the avalanche ion current, so its
average current is the charge delivered to the amplification stage, integrated
over everything one beam pulse does. It is completely outside the readout chain,
it was logged for every sub-run of the whole campaign, and nobody has to take
new data to use it.

**Status: DONE and validated three ways (§2, §3). The scan reduction and the
join to the dead-time map are done, and the join gave a result (§5): dead time
follows charge, t ∝ Q^1.2, with three chambers on one curve. The systematic that
bounded the absolute scale is now CLOSED by direct measurement of the monitor's
impulse response (§4, §8, 2026-08-10): the readback averages over ≥ 0.47 s, so
`mean − median` IS the time-average current. The charges are measurements, not
lower bounds, and there is no correction factor.**

---

## 1. The method, in one line

    Q_per_pulse  =  ( mean(imon) - median(imon) ) / f_pulse

Per sub-run, per detector, on the resist channel (card 5, ch 1–4 = A–D; the
mapping is stable across every run checked, but `run_config.json` carries it and
the code should keep reading it).

Two design choices carry the whole thing:

**The baseline is inside the data.** The CAEN readback samples at ~1 Hz; beam
pulses arrive every ~3.3 s. Most samples therefore sit at the standing leakage
current, so the sub-run **median is the leakage at that exact HV** and
`mean − median` is the beam-induced part. That is what makes an HV scan usable
without a beam-off run at every point — leakage is strongly HV-dependent and a
single subtracted constant would not do.

**The pulse rate is per sub-run**, counted from
`slow_control/beam_intensity/beam_intensity_<date>.csv` over that sub-run's own
time window (pulses = samples above 100e10 protons). Beam availability at n_TOF
varies hour to hour and a run-average rate would smear the scan.

Code: `charge_lib.py` (reduction), `analyze.py` (driver + summary),
`results/flash_charge_subruns.csv` (1 464 rows, committed — this is the product).

---

## 2. It passes three tests

### 2.1 The hard null — no beam, no charge

`run_159` (2026-08-09, beam-off cosmic reference at the production setpoint,
**0.000 Hz** of beam pulses in the log):

| det | leakage [µA] | dI [µA] |
|---|---:|---:|
| A | 0.016 | +0.0002 |
| B | 2.927 | +0.0005 |
| C | 0.008 | −0.0002 |
| D | 0.714 | +0.0015 |

Against a beam-on dI of +0.042 µA on the same channel A. The estimator returns
zero on zero beam, including on a channel carrying **2.9 µA of leakage** —
which is the important part, because it shows a big standing current does not
leak into the answer.

### 2.2 The rate scaling — it tracks the beam, not the detector

`run_157` and `run_158` sit at the identical HV setpoint hours apart on the same
day. `run_157` was taken as a "beam-off" cosmic run, but the intensity log shows
it caught residual beam at **0.031 Hz**, ~10× below production. Charge per pulse
must be unchanged; current must not be:

| run | det | rate [Hz] | dI [µA] | **Q/pulse [nC]** |
|---|---|---:|---:|---:|
| run_157 | A | 0.031 | 0.0052 | **165** |
| run_158 | A | 0.303 | 0.0419 | **142** |
| run_157 | C | 0.031 | 0.0040 | **128** |
| run_158 | C | 0.303 | 0.0293 | **99** |

A 10× change in current, the same charge per pulse to ~25 %. This is the single
strongest argument that the number means what we say it means, and it came free.
(The residual 15–25 % excess at low rate is worth a look — see §6.3.)

### 2.3 The HV dependence is the gain curve

run_57 scanned resist 580→520 V in 2 V steps. Charge per pulse over that range:

| det | 520 V | 580 V | ratio |
|---|---:|---:|---:|
| A | 35 nC | 761 nC | **×21.5** |
| B | 27 nC | 534 nC | **×19.9** |
| C | 74 nC | 821 nC | **×11.1** |
| D | — | — | **×0.3 — broken, do not use** |

Smooth and monotonic on A/B/C over 31 points. That factor is the gas gain,
measured through the charge it delivers rather than the signal it makes — which
is exactly the point, since the signal is what saturates.

**Det D is not usable in run_57** and the reason is not understood. It sits on
its own −10 V voltage grid (510–570 V), carries ~2 µA of leakage, and its curve
*falls* with HV. D is the standing "bad detector" caveat across every
flash-recovery analysis; treat this as one more instance and do not chase it
before §5 is done.

---

## 3. The numbers, as they stand

**Production operating point** (`run_158`, drift 700 V, resist A540/B540/C525/D520):

| det | Q/pulse [nC] | per channel [pC] | × CSA full scale (600 fC) | (50 fC) |
|---|---:|---:|---:|---:|
| **A** | **142 ± 2** | 139 | **231** | 2 774 |
| **C** | **99 ± 1** | 97 | **162** | 1 939 |
| B | 36 ± 2 | 36 | 59 | 711 | leaky, 2.1 µA |
| D | 305 ± 4 | 298 | 496 | 5 957 | leaky, 0.77 µA |

Per-channel divides by 1 024 = the two FEUs' 512 channels each, so it is an
*average over the whole chamber* — the beam spot is worse. DREAM CSA input
range is 50/100/200/600 fC (manual Table 1); which setting we run is a loose end
worth nailing down (§6.1).

**Headline for a talk: ~100–150 nC per beam pulse per chamber, i.e. of order
10²–10³ times the front end's full-scale input charge on an average channel.**
Quote A and C. Areal, over the 398.6 × 362 mm active area: ~0.1 nC/cm²/pulse.

Primary ionisation, if you want it: at gain G the primary charge is Q/G, so at
G ~ 2×10³ that is ~70 pC ≈ 4×10⁸ electrons ≈ 10⁵ MIP-crossings per pulse per
chamber. **The gain at this working point is not measured** — this line is an
illustration, not a result, and should be labelled as one until §6.2 is done.

---

## 4. The one systematic that bounded everything — ✅ CLOSED 2026-08-10

> **CLOSED by measurement, not by spec-reading.** The monitor's impulse response
> was measured directly by phase-folding imon against the individual beam pulses
> (§8). It is a **~1 s averager**: the response to one pulse rises from 0.3 s,
> peaks at 88 nA about 1.1 s later, and is back at zero by 2.3 s, with an area
> equal to the charge. The averaging window is **≥ 0.47 s** from a
> timestamp-free bound, versus the ~10 ms burst. **`mean − median` is the
> time-average current, the charges below are measurements, and there is no
> correction factor.** Route 2 (pulser injection) is no longer needed; route 1
> (the board model) is still a nice-to-have but nothing depends on it.
>
> The paragraphs below are kept as the record of what the argument was before it
> was measured — including the "28 % of samples" inference, which the direct
> measurement confirms (26.6 % on run_79's det C at the same threshold).

Everything above assumes **the CAEN imon readback preserves the time-average of
a current burst much shorter than the sample spacing.**

The evidence for it is indirect but real: 27.8 % of run_158's samples sit above
baseline on **both** clean detectors, with a 1 Hz sample rate and a 3.3 s pulse
period. If the monitor reported a genuinely instantaneous current, a ~10 ms
burst would be caught by ~1 % of samples, not 28 %. 28 % is what you get if each
burst is smoothed over ~0.9 s — and a smoothing filter conserves the integral,
so the sample mean recovers the time-average current.

But that is inference from one number. **If the monitor instead applies a short
boxcar and reports it only occasionally, every Q here is a lower bound**, by a
factor that could be large.

Three ways to settle it, cheapest first:

1. **Read the spec.** Identify the exact CAEN board (card 5 — the model is in
   the HV crate config / `hv_control.py`) and find its imon integration time and
   ripple-rejection filter. If it is a ~1 s average, the method is exact and this
   section closes. *This is a documentation lookup, not an experiment.*
2. **Inject a known charge.** The DAQ has a pulser path
   (`docs/REPORT_2026-07-28_pulser_daq_characterization.md`,
   `PLAN_2026-07-28_pulser_ipd_ladder.md`). Drive a known charge at a known rate
   into a chamber and check that the imon-derived Q reproduces it. This
   calibrates the monitor end to end and is the definitive answer.
3. **Vary the rate on purpose.** §2.2 got a 10× lever by accident. Doing it
   deliberately over 100× would expose any rate-dependent bias directly: a
   monitor that misses bursts gives a Q that *falls* as the rate falls.

~~Until one of these is done, state the number as **"≥ 100 nC per pulse"** or
attach the assumption explicitly.~~ **Superseded — say "100 nC", no inequality.
See §8.**

---

## 5. The join — DONE, and it collapsed

`results/flash_charge_subruns.csv` (charge) and
`daq:analysis/flash_recovery/run57/metrics_run_57_perdet.csv` (recovery) cover
the **same 31 sub-runs** and join cleanly on the sub-run name, 31/31.

Plotted against each other, **A, B and C fall on one curve, t ∝ Q^1.2**, over a
decade in charge and 1.5 decades in dead time. That is the statement worth
having: dead time is set by the **charge delivered**, and HV is only the knob
that sets it. It is a genuinely new result rather than a restatement of the HV
map, and it is the kind of thing a future front end can be specified against.

Figure: `mpgd26/make_status_plots.py --only deadtime_vs_charge` →
`mpgd26/slides/assets/img/status_deadtime_vs_charge.png`. Caveats on it: the
recovery axis is quantised to log-time bins (that is the vertical stepping), and
the power-law index is an unweighted fit drawn to show the trend, not a
measurement of an exponent.

Other scans already reduced and sitting in the CSV, **still unexploited**:

| run | axis | what it would add |
|---|---|---|
| run_58 | drift × resist, singles | does **drift** field change delivered charge at fixed gain? The recovery map says drift barely affects dead time — if charge is also flat in drift, those two facts agree and the story tightens |
| run_61/64 | drift × resist | repeats, for reproducibility across days/gas |
| run_67 | drift × resist × plastic threshold | the run the operating point was chosen from; lets you put "charge delivered" on the same axes as "tracks banked" |
| run_19/42 | resist, **Ar/iso 95/5 vs 90/10** | charge vs gas at matched gain — run_19 is 95/5, run_42/57 are 90/10 |

---

## 6. Loose ends, roughly in priority order

1. **Which CSA input range are we running?** The multiples in §3 span 50–600 fC,
   a factor of 12. It is in the DREAM `state1` register decode
   (`Feu * Dream * 1 = 0xD023891F` — registers 6/7, 2 bits per channel, per the
   saturation note §1). Read it back from a live FEU rather than trusting the
   inferred bit map; the same note flags that the decode is unverified.
2. **Gain at the operating point.** Needed to turn charge into primary
   ionisation, and therefore into "how many MIP-equivalents is the flash". The
   June bench work has gain curves; whether they transfer to 90/10 at n_TOF is
   the question.
3. **The 15–25 % low-rate excess in §2.2.** run_157 gives a slightly *larger*
   Q/pulse than run_158 on both clean detectors. Candidates: the residual pulses
   in a mostly-off period are systematically more intense; the intensity-log
   threshold miscounts at low rate; or a real rate-dependent effect (charging-up
   of the resistive layer at high rate would reduce charge per pulse as rate
   rises — which is a real detector physics effect and would be worth knowing).
4. **Flash vs tail.** The supply current integrates the whole pulse. It cannot
   say what fraction is the prompt γ flash and what is the neutron-induced tail —
   and the CSA-pinning mechanism is specifically about *sustained* current, so
   the split matters for the mechanism argument.

   > **Update, same day: the waveform-level handle now exists.** A parallel
   > session analysed the n_TOF-DAQ micromegas channels (MMA/MMB, 1 GS/s, no
   > charge-sensitive preamp) — `ntof_processing/mm_flash/` and
   > <https://dylan-neff.web.cern.ch/notes/ntof-micromegas-gamma-flash.html>.
   > Headline: **the chamber returns below threshold 0.87 µs after the flash
   > peak** and delivers hits at 30.3 µs (the earliest the DAQ allows), so the
   > millisecond dead time is entirely a DREAM-chain property. Median flash
   > charge **930 pC into 50 Ω** (224302, MMB) and 278 pC (224327, MMA), and the
   > response **compresses**: the dedicated/parasitic charge ratio falls
   > 2.35 → 0.95 across the gain range against a fixed 2.05 intensity ratio.
   >
   > **Do not naively compare 930 pC to the 142 nC here.** They are different
   > quantities: one electrode's prompt flash pulse over ~1 µs, versus a whole
   > chamber's avalanche charge integrated over the entire 80 ms cycle. Making
   > them comparable is the obvious next step and needs the electrode identity.
   > Also note that study's own caveats — the chamber identity is undecidable
   > from the data (A/B/C/D stepped in lockstep), the gas was Ar/CF4/Iso
   > 88/10/2 in 224302, and the 50 Ω figure assumes direct termination.
   >
   > One cross-check worth doing: it reports gain e-folding every ~10.5 V, while
   > the run_57 charge curve here rises ×20 over 60 V, i.e. an e-fold every
   > ~20 V. Those are different measurements of related things and the factor 2
   > is unexplained — space-charge compression is a candidate, since that study
   > sees the charge turn over above 537 V.

   - **Geant4** remains the independent third handle: energy deposition in the
     DriftGas volumes vs time since flash, × W = 26 eV × gain, from the existing
     `full_sim` campaign. Predicts both the charge and its time profile with no
     electronics in the loop.
5. **Det D.** Broken in run_57 (§2.3) but the *largest* charge at the production
   point (§3), on 0.77 µA of leakage. Both cannot be trusted; find out which.
6. **Leakage epochs.** In run_57 (July 19) det A carried 2.5 µA of leakage; in
   run_158 (Aug 9) it carries 0.016 µA and det B carries 2.1 µA. Something
   changed — chambers were worked on, or the leakage healed. Irrelevant to the
   estimator (it subtracts), but it is a detector-history fact worth recording
   somewhere it will be found.

---

## 7. How to re-run

```bash
# 1. mirror the inputs (July runs are on EOS, August runs on the DAQ)
ssh lxplus "cd /eos/experiment/ntof/data/x17/july_beam/runs && \
  tar czf /tmp/hv.tgz \$(for r in run_19 run_42 run_57 run_58 run_61 run_64 run_67 run_79; do \
    ls \$r/run_config.json \$r/*/hv_monitor.csv 2>/dev/null; done)"
scp lxplus:/tmp/hv.tgz <mirror>/ && tar xzf <mirror>/hv.tgz -C <mirror>

for r in run_157 run_158 run_159; do
  rsync -a --include='*/' --include='hv_monitor.csv' --include='run_config.json' \
    --exclude='*' daq:/mnt/data/x17/beam_july/runs/$r/ <mirror>/$r/
done
rsync -a daq:/mnt/data/x17/beam_july/slow_control/beam_intensity <mirror>/

# 2. reduce
.venv/bin/python ntof_july_analysis/flash_charge/analyze.py --src <mirror>
```

Then the readable version:

```bash
# 3. report (regenerates its figures too, via mpgd26/make_status_plots.render)
.venv/bin/python ntof_july_analysis/flash_charge/make_report.py

# ...or publish it where the DAQ web page's Analysis tab will list it
.venv/bin/python ntof_july_analysis/flash_charge/make_report.py \
    --out /mnt/data/x17/beam_july/analysis/flash_charge
```

`results/flash_charge_subruns.csv`, `results/flash_charge_summary.md` and
`results/report.html` are committed. **`results/figures/` is not** — the repo
blanket-ignores `*.png` (250 MB was purged from history on 2026-08-04, policy is
"regenerate from the analysis scripts"), so a fresh clone has the report without
its images until you re-run `make_report.py`, which takes seconds.

The figures are rendered by `mpgd26/make_status_plots.py` — the same code that
builds the MPGD2026 talk — so the report and the talk cannot drift apart.

**Do not** reduce the bulk waveform data for this — the whole point of the
method is that it needs 8 MB of CSV and no reprocessing.

---

# §8. The monitor's impulse response, measured — 2026-08-10

**Verdict up front: the CAEN imon readback is a ~1 s averager, not a snapshot.
`mean − median` is the time-average current. Every charge in this document is a
MEASUREMENT, not a lower bound; there is no correction factor. The residual
uncertainty on the absolute scale is ±3 % from estimator spread on the one clean
chamber, not the factor-of-many that §4 feared.**

Code: `imon_response.py` (analysis), `make_imon_figure.py` (the deck figure).
Products: `results/imon_response_run_79.json`,
`results/imon_fold_run_79_<det>_{isolated,isolated_labels,recon,labels}.csv`,
`results/imon_kernel_run_79_<det>.csv`. Slide markup:
`mpgd26/slides/HANDOFF_imon.md`.

```bash
.venv/bin/python ntof_july_analysis/flash_charge/imon_response.py \
    --src /media/dylan/data/x17/beam_july --run run_79
.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py
```

## 8.1 Which run, and why not run_158

`run_157/158/159` are **not in the local July mirror** and the August tree could
not be fetched (the DAQ at 128.141.177.17 was unreachable, lxplus needed a
Kerberos ticket). The measurement was therefore done on **run_79 sub-runs
`stat090_0000/0001`** (2026-07-26), which is the *same production setpoint* —
resist A540/B540/C525/D520, drift 700 V — and is the only production-point
`hv_monitor.csv` on disk. 7 113 imon samples over 2.01 h, 2 085 logged beam
pulses at 0.288 Hz.

It is the same measurement: **run_79 det C gives 97 nC/pulse, run_158 det C gives
97–101 nC/pulse.** Note that in July det **A** carried ~2 µA of leakage and only
became clean in August (§6.6), so the clean chamber here is **C**; A and D are
cross-checks.

**If you get the August tree, re-run this on run_158 and run_157.** Nothing about
the conclusion is expected to move, and the run_157 comparison would settle §6.3
outright (see §8.7).

## 8.2 The trap was real, and it is the interesting part

`hv_monitor.csv` timestamps are **whole seconds** —
`time.strftime('%Y-%m-%d %H:%M:%S')`, taken in `hv_control.py:monitor_hvs()`
*before* the CAEN read cycle (all 20 channels are read within ~16 ms, so the
channel-order offset is negligible). `monitor_interval` is 1 and the loop is
`reads; write; time.sleep(1)`, so the **real period is 1.0162 s** and the logger
drops a whole labelled second every ~81 samples (115 times in 7 113).

That means the sub-second phase of the true read inside its labelled second
**drifts uniformly over [0, 1)**. Folding on the raw labels therefore convolves
the true response with a 1 s box — σ = 289 ms — which on its own would smear a
delta into a ~1 s feature and hand back exactly the reassuring answer. The
label-based fold also puts apparent response *before* the pulse, which is the
tell.

Two independent defences were used.

**(a) Timestamp-free tests.** These use no timestamp at all, only counting:

| observable | measured (det C) | what an instantaneous read of a 10 ms burst gives |
|---|---:|---:|
| fraction of samples above baseline (+20 nA) | **26.6 %** | 0.29 % |
| largest single-sample excess | **0.216 µA** | 10.1 µA |

Both discriminate by ~2 orders of magnitude, in the same direction. The second
one is a *hard bound*: a sample whose averaging window fully contains one burst
reads `Q/w`, so `w ≥ Q/ΔI_max` = **0.47 s**. (The burst is ms and the window is
≥ 0.47 s, so "fully contains" holds for all but ~1 % of elevated samples.) Run
lengths of consecutive elevated samples — 925 singles, 508 pairs, 112 triples —
say the same thing. Det A and det D give w ≥ 0.49 s and ≥ 0.61 s independently.

**(b) The time base was reconstructed.** The drift is also the cure. For a run of
consecutively-logged samples, `label[k] ≤ t0 + k·P < label[k]+1` is 2N linear
inequalities in two unknowns, so the *pattern of dropped seconds* pins (t0, P).
Solved as an exact convex feasibility problem (ternary search on
`F(P) = a_lo(P) − a_hi(P)`, which is convex — a grid search gives false negatives
because the feasible period interval is ~10 µs wide for a 3 500-sample segment),
on greedy-maximal segments inside each sub-run. Result: **94.9 % of samples
recovered, median 2 ms, p95 16 ms.**

Validations of the reconstruction, all passed: `floor(t̂) == label` for
6 728/6 747 samples (19 boundary-rounding cases); the recovered sub-second phase
histogram is **flat** (699/686/660/678/657/666/681/647/690/683 per 0.1 s bin),
exactly as a drifting `sleep(1)` loop predicts; and the fold sharpens — the label
fold's rms width is 0.49 s against the reconstructed 0.458 s, and the label fold
shows response *before* the pulse, which is impossible.

**Do not turn that rms pair into a quadrature check of the 289 ms box.** I tried;
it does not work, and the handoff should say so rather than quote a number that
looks like a closure. Naively you would expect the label fold at
`sqrt(0.458² + 0.289² + 0.115²) = 0.554 s` (the last term is its own 0.4 s
binning), and 0.49 s is *less* than that — because both rms values are computed
over the finite [−0.8, 2.4] s fold window, which clips the smeared tails and
pulls the label fold's rms down. The qualitative signature (broader, and
acausal) is the check; the quantitative one would need a wider window.

**The clock offset.** `beam_intensity_*.csv`'s `unix_ts` comes from
**NXCALS/pytimber** (`beam_monitor/beam_intensity_controller.py`), i.e. the CERN
timing system, while `hv_monitor.csv` carries the DAQ host clock — so a genuine
host-to-host offset is possible and had to be bounded, not assumed. A ±3 600 s
lag scan of the excess series against the 1 s-binned pulse train gives a global
maximum at **+1 s** (r = 0.302); the nearest competitor is at **+217 s**
(r = 0.277), which is 6 × the 36 s PS supercycle, i.e. a structural alias, not a
rival hypothesis. **An offset shifts the response curve; it cannot widen it**, so
the verdict does not depend on it — and the offset is bounded well below the
response width regardless. The two halves of the run give fold centroids of
1.197 s and 1.222 s, so there is no clock drift within the run either (25 ms).

## 8.3 The measured response

Model-free, on **692 isolated pulses** (no other pulse within 3 s before or 2.4 s
after — both gaps are required; see §8.6), det C:

| quantity | value |
|---|---:|
| rises from | ~0.3 s |
| peak | **88 nA at 1.1 s** |
| FWHM | **1.0 s** |
| rms width | 0.46 s |
| back to zero by | 2.3 s |
| response before the pulse | 0 ± 2 nA |
| area | **98 nC per pulse** |

The same shape appears on det A and det D (each divided by its own area, they
overlay). The **drift-cathode channel `9:2 imon` is exactly constant at
0.1800 µA** across all 7 113 samples — same crate, same host, same 1 Hz logger,
no avalanche current — so this is not crate-wide pickup or a logging artefact.
Randomising the pulse times flattens the fold: χ²/ndf against a flat line is
**1.03 ± 0.49 for the randomised control against 63 for the real fold**. (Use
χ², not `max(fold)`: the maximum of ~30 noisy bin means is biased upward, and a
pure *time shift* is not a valid null here because the 36 s supercycle aliases
onto itself.)

**Whether the ~1 s smoothing is digital (the board's imon integration/ripple
filter) or analog (the RC of the HV output filter feeding the resistive layer)
does not matter, and we cannot tell them apart from this data.** Both are linear,
both conserve charge, and the peak excursion is 0.216 µA on a channel that
demonstrably resolves 1 nA and reads det B's 5.6 µA standing leakage linearly —
26× headroom — so nothing clips. The only scenario that would have biased the
answer is *burst shorter than the sample spacing AND an instantaneous reader*,
and that is what §8.2(a) excludes.

## 8.4 Four estimators, one number

| estimator | det C [nC/pulse] |
|---|---:|
| `mean − median` (the published one) | 97.1 |
| `mean − rolling 20th percentile` (leakage-detrended) | 100.5 |
| **isolated-pulse fold, area** | **98.4** |
| least-squares deconvolution over all 2 085 pulses | 101.7 |

Spread **±2.5 %**, and det C's answer moves only 98.3 → 100.8 nC as the
detrending percentile is swept 30 → 5. Take **±3 %** as the systematic on the
absolute scale at the production point. Any residual bias is in the direction of
`mean − median` being slightly *low* (the median sits marginally inside the
elevated population when 27 % of samples are elevated), which is the safe side.

## 8.5 Two by-products worth having

**Charge is proportional to protons — the readback is linear.** The isolated-pulse
fold split by n_TOF intensity band gives, per 10¹⁰ protons:

| band | pulses | mean intensity | det C | det A | det D |
|---|---:|---:|---:|---:|---:|
| parasitic | 279 | 414e10 | 150.6 pC | 268.7 pC | 816.3 pC |
| dedicated | 413 | 853e10 | 144.4 pC | 228.3 pC | 517.5 pC |

Det C agrees to **4 %** across a factor 2.06 in instantaneous current — so there
is no amplitude-dependent loss, and every logged pulse really does deliver charge
in proportion to its protons (i.e. the intensity log is not over-counting
extractions that never reached the target, which would have inflated `f_pulse`
and deflated Q). The A/D disagreement (18 %, 58 %) tracks their leakage, and D's
58 % is one more entry on the "do not trust det D" list (§6.5).

**Det A is recoverable in July.** Its `mean − median` in run_79 is garbage
(39 nC, and 7–90 nC sub-run to sub-run) purely because its 2 µA leakage drifts
2.11 → 1.29 µA across the run. Detrended, it gives **143–164 nC/pulse**, which
agrees with run_158's *clean* det A at 142 nC. That is a nice consistency check
on both the estimator and the run_79 ↔ run_158 equivalence. **Det D remains
untrustworthy** either way: 277 nC by `mean − median`, 378–414 nC detrended, and
373–497 nC as the detrending percentile is swept — its leakage swamps it.

## 8.6 One methodological trap, for whoever extends this

n_TOF's PS supercycle is **strictly periodic: 36 s, 11 pulses (6 dedicated at
853e10 + 5 parasitic at 414e10), every spacing a multiple of 1.2 s.** Two
consequences:

1. **Do not split the least-squares deconvolution by intensity band.** The fixed
   pattern makes the two bands' kernels near-degenerate, and the fit will happily
   report a 3.3× difference in charge per proton that is entirely an artefact of
   ill-conditioning. (It did, on the first pass.) Split the *isolated-pulse fold*
   instead — that is model-free.
2. **Cut on both gaps, not just the preceding one.** Cutting only on the
   predecessor and then truncating each sample at its successor makes the
   long-τ bins a *different, longer-gap* subset of pulses; because the two
   intensity bands sit at fixed places in the supercycle, that subset has a
   different mean intensity and it shows up as a **spurious dip in the middle of
   the response** — a double-humped "response" that is pure selection.

Also: `charge_lib._parse_ts` interprets the DAQ's local-time strings in **the
local timezone of whatever machine runs the analysis**. It is right on this
laptop (Europe/Paris, same offset as Geneva) and would be silently wrong by
hours from a US machine.

## 8.7 What this says about §6.3 (the 15–25 % low-rate excess)

§8.5's linearity result explains it without new physics. run_79's pulse mix
averages **665e10 protons**; an all-dedicated mix would be **×1.28** in charge
per *counted* pulse. run_157 caught only **14 residual pulses** in a mostly-off
period — a set that is both ±27 % Poisson and quite plausibly dedicated-heavy.
Either effect alone covers the observed 15–25 %.

**The check, when the August mirror is available:** compute the mean logged
intensity per run and compare `Q/pulse ÷ mean intensity` between run_157 and
run_158. If those agree, §6.3 closes as a beam-mix bookkeeping effect. If they do
not, the charging-up hypothesis (charge per pulse falling as rate rises) becomes
interesting and should be chased with a deliberate rate scan.

## 8.8 What this does NOT rule out

- **The absolute DC calibration of the CAEN ammeter.** This measurement shows the
  readback conserves the integral of a short burst; it says nothing about whether
  its nA scale is accurate. That is a board-spec question, and it is the one place
  where knowing the exact card model would still help. *(One-line question for
  Dylan: which CAEN board is card 5 of the crate at 128.141.177.244? The model is
  recorded nowhere in either repo, and the live crate was deliberately not
  probed — four chambers are taking production data and its CFE server has
  crashed mid-scan before, `nTof_x17_DAQ/docs/incident_2026-07-05_hv_cfe_crash.md`.)*
- **Anything about the flash/tail split (§6.4).** The supply current integrates the
  whole 80 ms cycle and the monitor then smears it over a second, so this method
  is structurally blind to time structure inside a pulse. The MM waveform work is
  the handle for that.
- **Det D, and det B.** Their leakage dominates; nothing here rescues them.
- **The run_57 HV-scan numbers specifically.** They were not re-measured this way.
  There is no reason to expect the monitor to behave differently at 580 V than at
  525 V, but the impulse response was only measured at the production setpoint.

---

# §9. The strip-vs-chamber comparison — CLOSED 2026-08-11

§6.4's "do not naively compare 930 pC to the 142 nC" got its proper treatment
in `ntof_processing/mm_flash/` (run 224709: same detector A, same day, same 25
HV plateaus for both instruments). Verdict, in one line: **the two absolute
scales are consistent once the board is accounted for and the flash is ~4×
denser at the measured strip than the chamber average — which the strip's own
intensity compression independently confirms at ~3× through the board's sheet
capacitance, with zero free parameters.** The board accounting (checkerboard
pad combs, exact ½ X/Y split, 0.85 image capture, 17 ms ESL drain to the
y-end buses) is `board_accounting.py`; the write-up is that package's
`report.html` (also on the CERN notes site) and
`HANDOFF_CHARGE_COMPARISON_2026-08-11.md` §8. The July slope-factor-2 puzzle
in §6.4 was already resolved in that package (different gas, unidentifiable
chamber; on det A in 90/10 the slopes agree to 7 %). Still open there and
worth doing from this package's side: nothing — the remaining tests (DREAM
run_160/161 recovery-time profile, pulser through the n_TOF patch, CAEN DC
scale) all live on the mm_flash side.
