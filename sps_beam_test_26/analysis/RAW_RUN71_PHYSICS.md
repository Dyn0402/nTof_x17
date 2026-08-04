# run_71 RAW — what it settled, and the new wall it hit

Analysis of the RAW run after the decoder fix. 2026-08-03.

```bash
../../.venv/bin/python decode_dataset.py    run71_raw --feus 03,01
../../.venv/bin/python extract_det4_only.py run71_raw --max-events 24000
../../.venv/bin/python tilt_m70V.py       --wf .../wf_run71_raw_det4only.npz --plateau raw450
../../.venv/bin/python kernel_fit_m70V.py --wf .../wf_run71_raw_det4only.npz --plateau raw450 --raw --q0 400,3000
```

## 1. RAW did exactly what it was taken to do

The ZS censoring is gone, and this is not an inference — it is directly
measured. Samples kept per strip, against that strip's own peak amplitude:

| peak ADC | n strips | RAW: samples kept | ZS (run_63) |
|---:|---:|---:|---:|
| 20–40 | 68,733 | **49** | 5 |
| 40–60 | 22,231 | **49** | 5 |
| 60–100 | 18,771 | **49** | 6 |
| 100–200 | 21,540 | **49** | 17 |
| 200–400 | 12,748 | **49** | 24 |
| 400–800 | 6,046 | **49** | 25 |
| 800–1600 | 2,940 | **49** | 25 |

Flat at 49 of 64 everywhere (the 24 % FEU packet loss), against ZS's 5 → 25.
The amplitude-dependent truncation that was sculpting the measured tail in
run_56 and run_63 is **eliminated**. That was the whole point of the run and it
worked.

Supporting checks: pedestal means are the true raw baselines (median 619 ADC,
range 344–2947), post-CNS noise 10.4 ADC against 297 raw, and the corrected
waveform amplitude distribution sits at 0 with symmetric noise (p1 −36,
p50 +4) and a positive signal tail — i.e. the pedestal + CNS chain reproduced
in `extract_det4_only.py` is behaving.

## 2. The tilt, third independent measurement

| | run_56 (CO₂) | run_63 (CF₄, ZS) | **run_71 (CF₄, RAW)** |
|---|---:|---:|---:|
| X view | 0.40° | 0.39° | **0.22–0.23°** |
| Y view | 0.04° | 0.06° | **0.03–0.06°** |

Same picture across three runs, two gases and a remount: a small standing tilt
in the striped coordinate, nothing in Y. run_71 reads lower than the other two
(0.22° vs ~0.40°); with the drift field different and v_drift unmeasured, I
would not read that difference as a real mount change.

## 3. THE BLOCKER: the tail runs off the end of the DAQ window

Central-strip charge arrival, Y view, within the 3840 ns window:

| plateau | t10 | t50 | t90 | amplitude still present at the LAST sample |
|---|---:|---:|---:|---:|
| drift 450 V | 660 ns | 1740 ns | 3360 ns | **44 % of peak** |
| drift 275 V | 600 ns | 1800 ns | 3360 ns | **52 % of peak** |

**Nearly half the signal is still there when the window ends.** Removing the
zero suppression exposed a second, independent truncation that ZS had been
hiding: 64 × 60 ns is not long enough to contain this detector's response.

That is itself a physics result — the dispersed tail is **longer than 3.8 µs**,
which rules out any short-τ description — but it means `tau_s` and `c2` from
run_71 are an *extrapolation past the window edge*, not a measurement:

| Y view | run_56 (ZS) | run_63 (ZS) | run_71 450 V (RAW) | run_71 275 V (RAW) |
|---|---:|---:|---:|---:|
| `c1` | 0.281 | 0.233 | 0.354 | 0.316 |
| `c2` | 0.111 | 0.084 | 0.257 | 0.222 |
| `tau_s` | 298 ns | 215 ns | 565 ns | 475 ns |

The RAW values are systematically larger, in the direction expected from
recovering previously-cut tail. But they are not converged, because the
recovery is itself incomplete.

**This also retracts a conclusion.** The earlier reports quoted `c1` as stable
at 0.23–0.28 across gas, voltage and threshold, and read that as evidence the
±1 amplitude is a property of the resistive layer. With the tail included
`c1` moves to 0.32–0.35. The apparent stability was partly an artefact of
*consistently* truncated tails, not a physical invariance. `c1` is not yet
established as transferable.

**And the drift-invariance test does not survive.** The truncation is
drift-dependent (44 % vs 52 % of peak at the window edge), so `beta` moving
11–14 % between 450 V and 275 V cannot be separated from the two points being
cut differently. The test needs a window that contains the signal.

## 3b. uRWELL cross-check: flatness confirmed independently

FEU1 (which carries **both** uRWELL planes) had to be re-decoded too — banco's
`combined_hits` for run_71 were made with the pre-fix decoder and the ZS
analyzer flags, and were unusable: 2.5 M hits per group against 38 k from a
correct decode, with the same event-merging signature. FEU1 shows the same
packet loss as FEU3 (23.5 %, 214,426 events, 1 missing), as expected — it is a
link-rate limit, not a detector effect.

With our own FEU1 decode the alignment is clean and independently confirms the
mount:

| | run_56 (flat) | run_63 (flat) | **run_71 (flat, RAW)** |
|---|---:|---:|---:|
| det(A) | 1.0090 | 1.0100 | **0.9942** |
| row scales | 1.0055 / 1.0035 | 1.0067 / 1.0033 | **1.0003 / 0.9939** |
| median residual | 0.51 mm | 0.59 mm | **0.70 mm** |
| fitted z | — | 1100 mm | 1140 mm |

Efficiency (uRWELL-referenced, in live bands): **54.0 % at drift 450 V, 42.3 %
at 275 V** — the expected fall as drift collection degrades.

The waveform-level comparison between the det4-only and uRWELL-referenced
selections is **not done**: `flat_align_eff.py` only knows the flat-256 ZS
baseline, and on RAW that leaves the pedestal (median 619 ADC) and the common
mode in the waveforms. It now refuses to write them rather than emitting
something that looks fine and is not. The kernel numbers above therefore rest
on the det4-only path, which does the pedestal + CNS correction. Porting that
correction into `flat_align_eff.py` is the remaining piece of the cross-check.

## 4. What IS robust

The **±1 peak-time shift**, which is a position on the waveform rather than an
integral, so truncation barely touches it:

| | X view | Y view |
|---|---:|---:|
| run_56, 625 V, CO₂ | +29.0 ns | +35.8 ns |
| run_63, 700 V, CF₄ | +31.3 ns | +34.3 ns |
| run_71, 450 V, CF₄ RAW | +31.6 ns | +33.4 ns |
| run_71, 275 V, CF₄ RAW | +33.6 ns | +34.6 ns |

**+29 to +36 ns across two gases, four drift fields, three resist voltages and
both zero-suppressed and RAW readout.** This is the one number the campaign has
actually pinned, and it sits right on the bench's independently-inferred
τ ≈ 47 ns delay.

Also robust: the Y view is symmetric everywhere (α and β agree between +1/−1 to
~3 %, +2/−2 to ~4 %), while X stays asymmetric and gets *worse* in RAW
(β(+2) 0.010 vs β(−2) 0.089 at 450 V) — consistent with the tilt living in the
tail, which RAW now exposes. **Quote Y, never X.**

## 5. What would close it

The tail needs to fit in the window. Options, in order of preference:

1. **More samples.** 128 × 60 ns = 7.7 µs would contain a tail that is
   currently ≥3.8 µs. Costs a factor 2 in data rate, which is what caused the
   packet loss — so it needs the trigger prescaled too.
2. **A coarser sample period.** 128 samples at 120 ns covers 15 µs at the same
   data rate, at the cost of time resolution on the rising edge. For a tail
   measurement that is a good trade.
3. Failing new data: fit the tail with a model constrained to be
   window-integrable and quote `tau_s` as a bound, not a value.

There is no beam for three years, so 1 and 2 are not available for det4. The
honest position is that **`tau_s` and `c2` are bounded below, not measured**,
and the `share_lp` calibration should carry that as a known limitation rather
than a number.
