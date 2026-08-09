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
follows charge, t ∝ Q^1.2, with three chambers on one curve. The one systematic
that bounds the absolute scale is still open (§4) — until it is closed, every
charge here is potentially a lower bound.**

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

## 4. The one systematic that bounds everything — DO THIS FIRST

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

Until one of these is done, state the number as **"≥ 100 nC per pulse"** or
attach the assumption explicitly.

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
