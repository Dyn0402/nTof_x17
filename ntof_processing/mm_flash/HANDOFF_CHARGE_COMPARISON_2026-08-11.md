# Handoff — why do the two flash-charge measurements differ by 3.5×?

**Written 2026-08-11, for whoever takes the deep dive.** Everything here is
reproducible from `ntof_processing/mm_flash/` plus the products on EOS; nothing
needs the 1.5 TB of raw.

> **STATUS — the deep dive was done the same day (§8, `board_accounting.py`).**
> The board (gerbers + solved electrostatics in `~/CLionProjects/MX17_Geant`)
> replaces the strip accounting of §2: the readout is a checkerboard pad grid,
> the X/Y split is *exactly* ½ by symmetry (the "465, all on Y" reading in §2
> is not physically available), 85 % of the ESL charge images to the pad plane,
> and the uniform-share expectation is **131 pC** → the residual sharpens to
> **4.2 ± 0.3 across all 25 plateaus**. The factor is carried by **local flash
> density**: the strip's own intensity compression, read through the board's
> sheet capacitance c′ with zero free parameters, independently measures the
> local density at **2.9× the chamber average** (lobe says 4.1×) and tracks the
> measured deficit across the whole scan. §4.1 (termination) is direction-dead:
> every passive patch error makes the true strip charge *larger*; §4.2 closed
> (drift imon = pure divider steps, no beam response); §4.3 closed (exact ½ +
> conservation); §3's "periphery ⇒ below average" assumed the flash follows the
> neutron-beam profile, which is the assumption that died. Remaining tests, in
> value order: DREAM run_160/161 per-channel recovery-time map (= a flash-charge
> profile via t ∝ Q^1.2), a pulser through the same patch, a second strip near
> centre. The note (`report.html` / the CERN site) is updated and is the
> canonical write-up.
>
> **Second pass, same day (review feedback):** the residual is properly stated
> as *delivered charge density* = illumination x local gain — one channel can't
> split the product, and the strip's 5–7 mm proximity to the passivation edge
> makes a local gain enhancement as serious a candidate as the flash profile
> (June bench: this chamber's sparks were edge-dominated). Also measured: the
> resist-leakage baseline **steps 0.15–0.57 µA at every drift-HV move** and
> relaxes over tens of minutes (cathode-motion charging current, direction
> excludes a divider path; per-plateau median subtracts it, worst within-plateau
> movement 24 nA). And the imon **cannot** split dedicated/parasitic on 224709 —
> with gaps wide enough to isolate the 1 s response, survivors are 91 dedicated
> vs 1 parasitic (supercycle entangles isolation with class); run_79's
> ms-reconstructed fold remains the imon's class answer. New summary figure
> `figures/compare_final.png` (absolute per-strip curves + ladder, with the
> 600 fC line and the 540–560 V operating band).

---

## 0. The question

Two independent measurements of the gamma-flash charge on MX17 **detector A**, on
the **same day, same gas, same 25 HV working points**:

| | what it integrates | at 700 / 540 V |
|---|---|---|
| **waveform** | one strip, prompt flash, 11–20 µs | 538 pC (pulse mix) |
| **HV supply current** | whole chamber, whole 80 ms cycle | 143 nC |

Divide them and you get the number of strips that would have to share the chamber
charge to explain one strip: **266 ± 15, constant across all 25 points** (5.7 %
spread over a factor 21 in charge and three drift settings).

The geometric expectation is **930** (see §2). So one strip carries **3.5× more
charge than the whole chamber divided by the strips that should be sharing it.**

That factor is the open question. It is *constant*, so it is one multiplicative
constant, not a slope error, a drift, or a calibration that wanders.

**Why it matters.** The absolute charge is what sets "how much can the front end
swallow" (currently quoted as ~1000× the DREAM CSA full scale) and what feeds the
Q/I_feedback dead-time argument. A factor 3.5 does not change the conclusion —
1000× or 3500×, the answer is the same — but it is the difference between a
cross-validated number and two numbers that happen to be within an order of
magnitude.

---

## 1. The two measurements, precisely

### 1.1 Waveform (this package)

`MMA` = strip 32 of detector A's Y plane, cable Y8, digitised by the n_TOF DAQ at
1 GS/s over the full 20 ms cycle, **with no charge-sensitive preamplifier**.

    dV_i = (b - c_i) * V_FS / 2^16          V_FS = 5043.79 mV, 16 bits
    Q    = (dt / R) * sum_i dV_i            dt = 1 ns, R = 50 ohm assumed
         = 1.5392 fC per count*ns

Integrated 11–20 µs (the whole positive lobe; the 20–30 µs window carries −8 to
−12 % of it, the AC-coupling return). Baseline `b` is the per-bunch median of the
first 2 µs. Cable correction to refer it back to the strip is ×1.009 and is *not*
a candidate for the discrepancy (Appendix B of the note shows the charge is the
f → 0 component and the skin-effect term vanishes there; de-attenuating the
measured pulse moves its area by 4×10⁻⁵).

### 1.2 HV supply current (`ntof_july_analysis/flash_charge/`, this package's
`imon_scan.py` for the August application)

    Q_pulse = ( mean(imon) - median(imon) ) / f_pulse

on the resist channel (card 5 ch 1 = det A), per plateau. The median is the
leakage at that exact voltage; `f_pulse` is counted here from the n_TOF side
(every bunch of 224709 carries its own wall clock) rather than from the beam log.

The monitor's impulse response was measured directly by the parallel session: it
is a **~1 s averager** (peak 88 nA at 1.1 s, back to zero by 2.3 s, area = the
charge), with a timestamp-free bound of ≥ 0.47 s against a burst of milliseconds.
So `mean − median` is the time-average current and there is no correction factor.
Four estimators agree to ±2.5 %.

---

## 2. The strip accounting

Done carefully, because it is the denominator of the whole comparison.

- **512 strips per plane**, pitch 398.58/512 = **0.7785 mm**, two planes (X and Y)
  → **1024 readout strips**. The supply current is the whole chamber, so both
  orientations count.
- The strip plane is **passivated over the Y edges**. Measured on this very
  chamber (det A = `mx17_3`, the reference measurement in
  `common/mx17_active_area.py`): **18.0 mm** low, **18.7 mm** high, leaving a live
  band of **361.9 mm**. So **465 of 512 Y strips are live**; 47 are buried.
- X strips all survive but are live over only **90.8 %** of their length.

Multiplier for scaling one live Y strip to the whole chamber:

| assumption | multiplier |
|---|---|
| the two planes share the induced charge equally | **2 × 465 = 930** |
| all the induced charge appears on the Y plane | 465 |
| naive, ignoring passivation | 1024 |

Measured: **266**. Against 930 that is ×3.5; against 465 (the most generous
assumption physically available) it is still ×1.75.

---

## 3. What is already ruled out

**The beam profile.** This was the obvious explanation — strip 32 sits in the beam
spot and sees more than average — and it is wrong. Looking the channel up in
`mx17_m1_map.csv`:

| reading of "strip 32 of cable Y8" | y position | distance to passivation edge |
|---|---|---|
| connector Y8, channel 32 | 374.40 mm | 5.5 mm |
| global y-strip 32 | 24.96 mm | 7.0 mm |

**Both readings put it at the chamber periphery**, where illumination should be at
or below average, not 3.5× above it. Whichever is right, the beam-profile
explanation points the wrong way.

**Lateral charge spreading.** Tempting, but it cannot do this on its own: for a
*uniformly* illuminated chamber, charge conservation makes the average induced
charge per strip independent of the spreading kernel. Spreading redistributes; it
does not create. It can only matter in combination with a non-uniform illumination
— and see above.

**A shape or slope error.** The ratio is constant to 5.7 % over a factor 21 in
charge, across three drift settings. Whatever it is, it is one number.

**Day-to-day drift in the chamber.** The imon value at 700/540 on the evening scan
(143.4 nC) reproduces the independent morning measurement of the same setpoint
(run_158, six sub-runs, 142 ± 3 nC) to better than 1 %.

**The cable.** ×1.009 (§1.1).

**A pulse-counting error.** `f_pulse` = 88 pulses / 300 s = 0.293 Hz at the
reference plateau, consistent with the known ~0.3 Hz. And an error here would have
to *increase* the imon charge to close the gap, i.e. we would have to be
over-counting pulses by 3.5×, which the bunch record excludes.

---

## 4. Candidate explanations, and how to test each

Ranked by my estimate of how likely they are to carry the factor.

### 4.1 The 50 Ω assumption — the single biggest lever

`Q = ∫V dt / R` scales inversely with R. If the digitiser input is not a bare
50 Ω, or if something sits in the patch — a splitter feeding both the DREAM front
end and the n_TOF DAQ, an attenuator, a series resistor — the absolute scale moves
by exactly one constant, which is the shape of the discrepancy.

**Tests, cheapest first.**
1. Look at the physical patch. Is strip 32 split between DREAM and the n_TOF DAQ,
   or moved over entirely? A resistive splitter would be an obvious factor.
2. The S014/ADQ14 input impedance and any input divider, from the card spec.
3. Inject a known charge (the DAQ has a pulser path,
   `docs/REPORT_2026-07-28_pulser_daq_characterization.md`) into the same patch
   and read it out through the same chain. This calibrates the whole path end to
   end and is the definitive answer.

Note the pulse *shape* argues the termination is roughly 50 Ω — a high-impedance
input would integrate on the cable capacitance and decay slowly, whereas the
observed pulse returns to baseline in ~1 µs with a small undershoot. That
constrains the order of magnitude, not the factor of 3.5.

### 4.2 What fraction of the avalanche charge the resist supply actually carries

The comparison assumes imon on the resist channel = the full avalanche charge.
The mesh on these chambers is **grounded** (established on det3,
`mx17-hv-slope-test`), so the resistive layer is the anode and should collect the
avalanche electrons — but "should" is doing work there.

**Tests.** Is any part of the avalanche current returning through the drift supply
(9:0) instead? That channel is logged in the same file
(`imon_224709.csv` carries `A_drift_imon`) and was **exactly constant at
0.1800 µA** through run_79, which argues against it — but it has not been checked
on the August scan, and it is two lines of code.

### 4.3 The induced-charge fraction over a finite window

The waveform integrates 11–20 µs. A resistive-strip detector's readout sees a
capacitively coupled image whose full-time integral is zero (AC coupled); we take
the positive lobe. If the prompt image over that window is a larger fraction of
the local charge than the naive 1/N_strips, that is a real effect — but see the
charge-conservation argument in §3, which limits how far it can go for uniform
illumination.

**Test.** The repo already has measured charge-spreading kernels
(`wft/` share kernels, the RC-ladder adopted for det3; `det4-flat-charge-spreading`
for the response shape). Predict the single-strip prompt image fraction from those
and compare with 1/266 against 1/930.

### 4.4 Non-uniform illumination in the *other* coordinate

Ruled out along Y (§3), but a Y-plane strip integrates over all X. If the flash is
concentrated in X, that does not change a Y strip's charge relative to other Y
strips — it changes nothing here. Included for completeness; I do not think this
can contribute.

---

## 5. Soft spots in my own analysis — check these first

Where I would look for my own mistake, in order:

1. **The pulse-mix weighting.** The waveform number at a plateau is the
   n-weighted mean of the dedicated and parasitic medians (52 and 36 bunches at
   700/540). The imon average weights by whatever actually arrived in that window.
   If the mixes differ, the comparison is biased — but only at the ~20 % level,
   nowhere near 3.5×. Worth doing properly.
2. **Median-as-baseline on the flash block.** `b` is the median of the first
   2 µs. A few-count bias over the 9000-sample window is ~14 pC per count, i.e.
   ~2.6 % per count of bias. It would take a 40-count baseline error to matter,
   which the traces exclude — but I have not checked for a *slope* in the
   pre-flash baseline, which would integrate.
3. **The 11–20 µs window.** Justified by the running integral flattening
   (figure in the note). If there is a slow positive component beyond 30 µs it is
   invisible to the flash block entirely, and it would make the waveform number
   *larger*, widening the gap.
4. **`f_pulse` from n_TOF bunches vs the beam log.** I used the former. The
   parallel session used the latter and got the same charge at the shared point,
   which is reassuring but not a proof they agree in general.

---

## 6. The data and how to re-run

| what | where |
|---|---|
| MMA waveforms, full resolution | `/eos/experiment/ntof/data/x17/mm_raw_2026-08/mm_224709.npz` (154 MB) |
| the same, local | `/media/dylan/data/x17/ntof_mm_flash/mm_224709.npz` |
| imon, 1 Hz, whole scan | `/media/dylan/data/x17/ntof_mm_flash/imon_224709.csv` (8 233 rows) |
| HV plateaus | `ntof_processing/mm_flash/hv_plateaus_224709.csv` (25 rows) |
| July HV-current reduction | `ntof_july_analysis/flash_charge/results/flash_charge_subruns.csv` |

```bash
V=.venv/bin/python
$V ntof_processing/mm_flash/analyse_709.py        # waveform scan -> results_709.json
$V ntof_processing/mm_flash/charge_chain.py       # conversion, cable, RMS
$V ntof_processing/mm_flash/imon_scan.py          # imon on the same plateaus
$V ntof_processing/mm_flash/compare_hv_current.py # the three comparison axes
$V ntof_processing/mm_flash/make_report.py        # the note
```

`imon_scan.py` prints the per-plateau table including the implied strip count; that
table is the whole result and is the thing to attack.

To re-derive the raw products (only if the extraction itself is in question):
`extract_mm_full.py` over the 344 raw files, then `merge_709.py`. Budget ~70 min;
EOS read bandwidth is the bottleneck at ~220 MB/s and more parallel readers on one
node do not help.

---

## 7. What I did not do

- **Nothing on the DREAM side.** DREAM run_160/161 are simultaneous with 224709,
  so the occupancy profile across the Y plane is available on the same bunches and
  would measure the illumination directly — which would turn §3's argument from
  "the strip is at the periphery" into an actual profile.
- **No second strip.** One strip cannot separate a coupling constant from a
  profile. If another channel is ever patched in, put it near the chamber centre.
- **No pulser calibration** of the n_TOF patch (§4.1 test 3), which is the test
  that would most directly settle it.
- **The CSA range is still not confirmed for the beam runs.** A parallel audit of
  44 saved *bench* configs reads `Dream 6/7 = 0xAAAA` → 200 fC at 10 mV/fC. The
  note quotes the conservative 600 fC. If the beam ran at 200 fC, every "× full
  scale" figure is three times worse.

---

## 8. Resolution — the board accounting (2026-08-11, `board_accounting.py`)

Constants and their sources are embedded in `results_board.json`; figures
`figures/board_{stack,ledger,compression}.png`; the note carries the prose.
The three load-bearing board facts:

1. **Checkerboard combs, not strips.** 512×512 pads (680 µm on 780 µm pitch);
   a "Y strip" = 256 pads on 1.56 mm pitch along a row, X combs own the
   intervening pads, both views in the same copper plane
   (`response/common/channel_map.py`, read from the L5/L6 gerber stubs). Uniform
   flash ⇒ exact 50/50 X/Y.
2. **Capture 0.85** of ESL charge images to the pad plane (W2 boundary,
   `wpot_w2.py` / `V6_PAD_GAPS_2026-08-08.md`); mesh takes ~15 %.
3. **ESL = 550/250 µm resistive strips along y, bused at the two y-ends only**;
   τ_drain ≈ 17 ms at the frozen 2 MΩ/sq ⇒ the sheet is charge-conserving on
   the 9 µs window (strip lobe = local image) and the supply integrates
   everything (imon = whole charge). c′ = 0.50 µF/m² converts density → gain
   sag; with the measured ~22 V e-fold this predicts the dedicated/parasitic
   per-proton deficit with no fitted parameters, and the prediction tracks the
   measurement from 8 % (700/540) to ~29 % (700/570). Had the strip carried
   only the chamber-average density the predicted deficit would be 3 % — so
   the compression *discriminates*, and it sides with the lobe.

New measurements folded in: strip's own >30 µs tail = 0 above threshold at the
working point (1–2 % at 570 V); `A_drift_imon` = quantised divider current
only, mean−median negative ⇒ no avalanche return path outside the monitored
channel.

What is still open, and would each close it further:
- **DREAM run_160/161 recovery-time profile** — the flash-density map across
  the plane, from data that already exist (t ∝ Q^1.2 turns per-channel dead
  time into per-channel charge).
- **Pulser through the n_TOF patch** — removes the last instrumental
  alternative (a >50 Ω effective input, the only error that could *shrink*
  the strip charge).
- **CAEN absolute DC scale** — moves chamber charge and residual together;
  board-model-independent.
- The 40 % gap between the two local densitometers (4.1 vs 2.9) is booked to
  the sheet-charging model's roughness (uniform-in-time delivery, e-fold band
  20–24 V, spreading during the ~2 µs sweep) and not chased.
