# PSA optimization for the X17 EAR2 detectors: what the raw waveforms say

**2026-07-28, afternoon session.** Continues `HANDOFF_2026-07-28_ntof_processing.md`.
Companion document: `FLASH_TIME_BASE.md` (gamma-flash timing and the SiPM
divert, kept separate because it is a different kind of problem).

All numbers below are **[measured]** from raw stream1 waveforms of run 224572
unless marked otherwise. Sample: 7 head chunks spanning the whole run
(bunches 161-163, 398-400, 798-800, 1198-1200, 1598-1600, 1998-2000,
2398-2400), 851 channel-bunch flash blocks, ~30 k isolated late-time pulses,
~350 k zero-suppressed blocks. Reproduce with:

```bash
python ntof_processing/raw_flash_extract.py  <raw_head.bin> <out.npz>
python ntof_processing/flash_finder_emulator.py <raw_head.bin> ...
python ntof_processing/make_pulse_shapes.py <outdir> <raw_head.bin> ...
```

Raw chunks come off EOS with
`xrdfs root://eospublic.cern.ch cat /eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/224572/stream1/run224572_<n>_s1.raw.finished | head -c 450000000 > head_<n>.bin`
(~3 bunches per 450 MB, ~25 MB/s).

---

## 0. Ground truth we did not have before

**Signal polarity.** Walls **positive**, plastics and liquids **negative**,
PKUP positive. Measured on isolated late-time pulses (walls: median +438..+812
against -76..-92 of noise; plastics/liquids the mirror image) and confirmed
independently by the sign of Riccardo's shipped pulse-shape templates
(`X17_WAL*` peak +655..+2165, `X17_LIQ*` trough -747..-2498). Every threshold in
the UserInput is applied as `polarity x pulse`, so this determines which
waveform features the flash finder can even see -- and it is what explains the
wall bistability described in `FLASH_TIME_BASE.md`.

**Pulse shapes.** Median normalised pulse over thousands of isolated hits:

| | rise (10-90 %) | FWHM | tail at 200 ns | tail at 500 ns | undershoot |
|---|---|---|---|---|---|
| WAL | ~20 ns | 74 ns | 4.0-5.8 % | 0.4-0.5 % | none |
| PSS | <10 ns | 13 ns | 0.1-0.3 % | ~0 | none |
| LIQ | <10 ns | 6 ns | 0.1-0.4 % | ~0 | none |

The walls have a genuinely long tail; the plastics and liquids are fast and
clean. **No detector shows a rebound or undershoot at ~350 ns.** That matters
because the 2026-07-27 session attributed a population of small plastic hits to
"rebound fragments trailing the real pulse by ~350 ns". The raw median pulse
does not support that: 350 ns is exactly the divert-gate lead time, so those
in-band hits were an artefact of comparing plastics against a wall time base
that was 350 ns early. **Re-measure the fragment population after
reprocessing; do not tune `EXPAND PULSES` for it now.**

---

## 1. Plastics (PSS)

### 1a. The flash (the headline bug) -- fixed in `v1_flash`

`G-FLASH THRESHOLD = 50.` with no lower time limit. The plastic flash saturates
the ADC (it drives the signal from a baseline of ~30800 down past 0 and wraps),
so 50 channels is ~0.2 % of the feature it is meant to find, and any pre-flash
noise excursion wins. Emulated over 851 channel-bunches: the current setting
returns a median "flash time" of 122-152 ns with a p2-p98 spread of 565-629 ns
-- i.e. it is fitting noise -- against a true flash at 11.63 us.

`2000/1e4` gives median 11627-11633 ns, spread 10-19 ns, 0 % not-found, on
dedicated and parasitic bunches alike. The plateau is wide: **500, 2000 and
5000 give bit-identical results**; only at 15000 does the constant-fraction
point move (+12 ns). 2000 is the midpoint of the plateau.

`G-FLASH MIN WIDTH` is deliberately left at 0. It would be a reasonable second
guard, but the saturating flash *wraps around the ADC range*, which breaks the
"contiguous chunk above baseline" that MIN WIDTH measures. The threshold and
the time limit are sufficient and cannot be confused by the wrap.

### 1b. A quarter of the plastic hits are being eliminated -- `v2_elim`

`AREA/AMP. HIGH THR. = 20` sits inside the bulk of the real distribution:

```
PSS isolated pulses (n = 15408):  area/amp  p1 1.3  p5 7.8  median 17.6  p95 24.2  p99 29.3
                                  current cut 2 .. 20   ->  ~25 % of real pulses eliminated
```

For comparison the walls' cut (10..200) sits a factor 2 above their maximum
observed ratio (110) -- that is what a safe elimination window looks like.
The PSA guide is explicit: *"none of these thresholds should be so severe that
true pulses are being discarded"* and *"if some of the false pulses do pass
into the final root-files, they can and should be eliminated during the later
data analysis"*. Widened to `1 .. 60`.

Caveat on the measurement: this is *our* area (a fixed -50..+400 ns window
around the peak with our own baseline), not the PSA's (derivative-bounded pulse,
adaptive baseline), so the absolute ratio will differ. The conclusion is
robust anyway because it is a comparison of the *same* estimator against the
three detectors' three different cuts, and only the walls' comes out safe.

`AMPLITUDE THRESHOLD` 100 -> 50 to match the walls (plastic amplitude p5 = 234,
so this is not where the spectrum is truncated, but there is no reason to be
2x stricter than the walls). `G-FLASH WINDOW` 0 -> 1000 to match the liquids.

### 1c. Not changed

`EXPAND PULSES = -1`, `STEP SIZE = 3/4`, `SIGNAL WIDTH 10..3000`, and
`AMPLITUDE OPTION = 1` (parabolic top). Plastic pulses are 13 ns FWHM with no
undershoot, so bipolar handling is already appropriate and shape fitting is not
obviously needed. Switching the plastics to shape fitting (`AMPLITUDE
OPTION = 2` with the templates now in `userinputs/v3_shapes/`) is the natural
**v4** experiment -- it would help the 5-6 ns pair resolution the plastic dead
time already achieves -- but it is slow and can misbehave, so it should be its
own variant.

---

## 2. Liquids (LIQ)

The flash finding is fine (0 % failures, and it stays fine with the time-limit
guard added in v1). Riccardo's "the liquids are still kinda bad" is about pulse
recognition and pileup, and there are two concrete defects.

### 2a. `AREA/AMP. HIGH THR. = 10` eliminates ~19 % of real pulses

```
LIQ isolated pulses (n = 2578):  area/amp  p1 1.7  p5 3.3  median 8.2  p95 12.0  p99 13.7
                                 current cut 2 .. 10   ->  ~19 % eliminated
```

Same story as the plastics, and the liquids are the detector where losing
pileup-broadened pulses hurts most. Widened to `1 .. 60` in `v2_elim`.

### 2b. The pulse-shape templates are far too short -- `v3_shapes`

`AMPLITUDE OPTION = 2` means every liquid amplitude and area comes from a fit
to a template, and the pileup deconvolution is only as good as that template.
The shipped ones are **single raw pulses** of

- `X17_LIQA_Signal_7.txt` -- 59 samples
- `X17_LIQB_Signal_0.txt` -- **24 samples**

A 24 ns template for a detector whose pulses have to be disentangled from
pileup is not a template, it is a spike. Replaced by median averages of
191-921 clean isolated pulses per tree per amplitude bin, 551 ns long
(`make_pulse_shapes.py`). All four liquids now get their own shapes instead of
sharing LIQA's and LIQB's; LIQC has too few low-amplitude pulses for a third
bin and gets two.

### 2c. Worth trying next (not in v1-v3)

`EXPAND PULSES = 0` on the liquids while the walls use 1 and the plastics -1.
With `SIGNAL WIDTH LOW THR. = 1` ns the liquids also accept extremely narrow
candidates. Both are plausible follow-ups once v3 is graded; the guide's
advice is to fix width limits with expansion **off** first, which is the
current state, so this is the right order.

---

## 3. SiPM walls (WAL): isolated late-time pulses

### 3a. What the divert costs, quantitatively

After the gate reopens at ~12.3 us the wall baseline is +21500 channels at
13 us and decays to +3800 (15 us), +1300 (20), +830 (25), +400 (30), +100
(36). Against an amplitude threshold of 50 channels that means wall hits are
**unusable from ~11.6 to ~13-15 us and degraded to ~40 us** -- E_n above
roughly 1 keV at 19.5 m. Irrelevant for X17's ms-scale physics; mandatory to
state in any flux or E_n work.

It also means `TIME LIMIT = 40000` in the wall row (the boundary between the
adaptive and the constant baseline) is **well matched to the actual recovery**
and should not be touched. This was worth checking: had the recovery run past
40 us, every hit just after the limit would have been sitting on a biased
baseline.

### 3b. Elimination windows are already safe

```
WAL isolated pulses (n = 11123):  area/amp  p1 41.7  median 87.6  p99 110.2   cut 10..200  -> 0 % lost
                                  full width p1 199  median 746  p99 1435 ns   cut 5..4000  -> 0 % lost
```

No change needed. This is the control that validates the method used on the
plastics and liquids in sections 1b and 2a.

### 3c. Where the improvement actually is: the templates

The wall pulse is still at **4.0-5.8 % of peak 200 ns after the peak** and
0.4-0.5 % at 500 ns. The shipped templates are 314 ns long, i.e. they end at
~250 ns after the peak, inside a tail that still carries several percent.
Every fitted amplitude and area therefore mis-accounts the tail, and the
pileup deconvolution -- the thing that decides whether two late hits 200 ns
apart are resolved into two isolated pulses or merged into one -- is fitting
with a truncated kernel.

Replaced with 720-861 ns median templates per tree per amplitude bin
(387-1327 pulses each). The tail **is** mildly amplitude-dependent (5.8 % at
200 ns in the lowest bin vs 4.1 % in the highest), so the existing three-shape
machinery is kept and now carries three genuinely distinct shapes instead of
three arbitrary single pulses shared across all four walls.

### 3d. Candidate follow-up: `STEP SIZE`

`8/7` (8 ns derivative window, 7x RMS). The guide's first piece of practical
advice is *"reducing the STEP SIZE -- even at the price of worsening the
signal-to-noise ratio in the derivative -- can often help in resolving
pileups"*, and the walls are the detector with a long tail and therefore the
most self-pileup. A `5/6` variant is the obvious **v5**. Not included in
v1-v3 because it interacts with the templates and should be judged after them.

---

## 4. Method note: why there is a local emulator

`/eos/experiment/ntof/bin/SignalAnalyzer` (rebuilt 2026-07-13) is the PSA, but
it is a **ROOT GUI application** -- it needs `LD_LIBRARY_PATH=/eos/experiment/ntof/bin`
and then opens a window and asks for a run file. There is no batch mode
reachable from the command line, and the sources under
`/eos/experiment/ntof/repositories/RawDataSignalAnalyzer/` are 2015-era and
partly unreadable. `RunProcessing.sh` is a 2.8 MB compiled blob with no read
permission. **So there is no cheap headless PSA loop; condor is the only way
to produce a real reprocessed file.**

`ntof_processing/flash_finder_emulator.py` fills the gap for the one decision
that matters most and is cheapest to get wrong: *which waveform feature does
the flash finder latch onto*. It implements `G-FLASH OPTION=0` as documented
(polarity, own baseline, first crossing above threshold after the time limit,
optional min width, constant-fraction timing) and it **reproduces the official
file's failure pattern** -- PSS median 122-152 ns, WALB bimodal against
WALA/C/D -- before reproducing the fix. That is enough to buy confidence in a
parameter set before spending a condor round-trip; it is **not** a substitute
for the real PSA, whose derivative-based recognition and adaptive baseline will
shift the absolute numbers.

---

## 4b. RESULTS of the first reprocessing (2026-07-28, run 224572)

Graded with `ntof_processing/grade_candidate.py` on pooled `completed/` partials
(**421 bunches, 398-3018**) -- partials land within ~25 min of submission, long
before the 26 GB merge, and they are enough for checks 1 and 2. Each partial is
~1.7-2 GB (job 0016 is a small tail chunk). **[measured]**

### The flash fix works, and it works better than the acceptance bar

```
                        official (before)        v1_flash / v2 / v3 (after)
flash-id bad bunches    PSS 37-85 %              0.0 % on 12 of 13 trees
                                                 (PSSD 0.2 %)
consistency vs wall     PSSA -362  PSSB +20      PSSA -3.5  PSSB +0.6
                        PSSC -333  PSSD -336     PSSC -3.0  PSSD -5.9
                        LIQA -373  LIQB  +10     LIQA -6.0  LIQB -2.0
                        LIQC -350  LIQD -348     LIQC -17.8 LIQD -14.6
per-tree modal tflash   WALA/C/D 11245-11275     all four walls 11605
                        WALB/PSS/LIQ 11615-11645 PSS 11635-11645, LIQ 11615-11635
```

Target was <2 % and |offset| <25 ns; everything passes. The plastics come in at
0.6-6 ns. All four walls now time the same waveform feature as the plastics and
the liquids, which is the whole point: **the 350 ns problem is gone at the
source**, and the laptop-side `tflash_repair` becomes a no-op.

The residual to watch is **LIQC/LIQD at -15 to -18 ns** -- inside the bar, but
an order of magnitude larger than the plastics' and the only per-arm structure
left. These are the two liquids with the fewest hits, so it may just be the
coincidence-peak fit; re-check on the merged file before folding these into
`flash_calibration.json`.

### v2_elim recovers exactly the hits it predicted

Hits per bunch, v2 against v1, over 421 bunches:

```
WALA-D  +0.0 %  (control: the wall elimination was not touched)
PSSA +30.7 %   PSSB +13.8 %   PSSC +20.9 %   PSSD +26.8 %
LIQA +21.8 %   LIQB +16.3 %   LIQC +26.6 %   LIQD +19.3 %
```

(The 21-bunch and 421-bunch samples agree to 0.2 %, so the recovery is uniform
across the run rather than concentrated in some bunches.)

Predicted from the raw pulses in sections 1b/2a: ~25 % for the plastics and
~19 % for the liquids. The walls being *exactly* 0.0 % is the control that says
the change did what it was meant to and nothing else.

### v3_shapes: right for the walls, WRONG for the liquids

Fit chi2 is the honest discriminator here, not the hit count:

```
        chi2 p50   chi2 p90   amp p50   hits     verdict
        old  new   old   new  old  new
WALA   0.90 0.85   4.09  4.05  660  674  -2.1 %  BETTER
WALB   1.23 1.00   6.70  5.34  894  923  -2.7 %  BETTER
WALC   1.13 1.00   5.62  5.03  846  879  -2.8 %  BETTER
WALD   1.18 1.06   5.71  5.43  856  885  -2.8 %  BETTER
LIQA   2.40 5.49  26.8  31.4   218  155 -15.7 %  WORSE
LIQB   4.27 7.50  35.2  35.2   311  239  -8.5 %  WORSE
LIQC   1.80 5.35  16.0  29.3   137  100 -33.5 %  WORSE
LIQD   1.77 4.04  18.2  20.9   152  114 -23.7 %  WORSE
```

**The walls behave exactly as argued in 3c**: chi2 falls, amplitudes rise
slightly (the truncated tail is now accounted for), and 2-3 % fewer hits are
spurious splits disappearing.

**The liquids do the opposite, and the reasoning in section 2b was wrong.**
A 551 ns template for a pulse with 6 ns FWHM and a 0.1-0.4 % tail at 200 ns
means the fit is dominated by ~500 ns of baseline noise: chi2 p50 more than
doubles, amplitudes are pulled down ~30 %, the pileup flag rate rises and a
quarter of the hits vanish. Riccardo's 24-59 ns templates were not too short --
they matched the detector. What they lacked was averaging, not length.

Corrected in `make_pulse_shapes.py`: `SPAN` is now per family and matched to
the measured tail (WAL 60/800, PSS 20/80, LIQ 20/60), with the reasoning in
the code so nobody repeats the mistake.

Follow-up variants:
- **`v4_walshapes`** = v2_elim + the wall templates only (the proven win)
- **`v5_liqshort`** = v4 + short *averaged* liquid templates (the retry: same
  length as the shipped ones, but built from hundreds of pulses instead of one)

## 5. What is NOT addressed here

- The absolute wall time offset (the divert-path delay) -- needs the dedicated
  pre-calibration, see `FLASH_TIME_BASE.md` section 4.
- Whether the ~40 ns channel-to-channel spread in the wall gate transient is
  also present in the physics path (per-channel vs per-tree offsets).
- PKUP and SILI: untouched. PKUP has 0 % flash failures and is the time anchor.
- The wall "+330 ns satellite band" from the DREAM matching: with the raw data
  showing no rebound at that delay, this is almost certainly the same
  gate-lead-time artefact and should simply disappear after reprocessing.
  Re-run `match_window.py` and check the band ratios before doing anything.
