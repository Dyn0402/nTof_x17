# The post-flash ringing in the DREAM waveforms (study, 2026-11-22)

Dylan asked whether the ringing after the n_TOF flash can be told apart from ionisation at the waveform level. This
study answers that question on chambers A and C: the waveforms of run_145 (sub-runs 0000-0002), run_104 stat090_0003
and run_139 stat090_0003. Chamber D was added on 2026-11-23, with the other open items (section "The open items,
closed").

**Short answer.** Yes, the ringing can be told apart, and it can be removed from the waveforms before the fit.
- A waveform test ("does the track keep at least one channel above 5 sigma once the ringing is removed?") reaches the
  purity of the 600 ADC charge cut from 8 ms on. It keeps every good late track at every angle, where the charge cut
  loses 0.2 % (A) and 1.1 % (C) of them, and 2.6 % of C's near-normal tracks.
- Together with the charge cut, it raises the purity before 8 ms by 0.04-0.08 (A 3-8 ms: 0.27-0.30 -> 0.35-0.37).
- It does not change the purity from 12 ms on. The non-pointing early tracks that survive the charge cut keep their
  charge once the ringing is removed, so they are real ionisation (partly mis-paired x/y views), not ringing. No
  ringing treatment can remove them.
- (2026-11-23) The plain correction (`deflect`) stays the best one: every variant that also removes the slow
  excursions of the first 1.25 ms takes real track charge. The fit's own window gives the same result as the
  symmetric one. Chamber D's problem is about 50 permanently oscillating channels per view, mostly unmasked in y; the
  waveform test adds nothing to D's `d_faint` cut.

## What the ringing is

We measure it on the pedestal-subtracted waveforms, before and after the plain 64-channel block median that pass 1
subtracts (`wft.io.FeuReader`, CM_MASK = 0). The scripts are `characterise/c1.py`-`c14.py` (study scripts with fixed paths); the
numbers below come from 4,000 consecutive events of tag 000 of run_145 stat090_0000.

- **It is an oscillation of the common mode.** Each 64-channel block carries one oscillation. Its dominant frequency
  is 2.4 MHz on A (period ~7 samples of 60 ns) and 1.0 MHz on C. A real ionisation pulse is a single positive pulse.
- **It is coherent across the chamber.** The block medians of two blocks of the same FEU correlate at 0.92. The x and
  y FEUs of the same chamber correlate at 0.79. Chambers A and C do not correlate (-0.01).
- **Its phase is random from event to event.** Two consecutive events at similar ToF correlate at 0.1 (median). So
  there is no fixed template in time after the flash that could be subtracted.
- **Each channel picks it up with its own factor.** Within one event and one block, the residual after the block
  median is 87-97 % rank one: the oscillation times a per-channel factor (beta - 1). The plain median removes
  beta = 1 everywhere. A channel with beta = 0 is left with minus the oscillation, and a channel with beta = 2 with
  plus it. This is the "both signs, all at the same sample" pattern of the audit (reco review, finding 1).
- **The coupling pattern is stable.** The per-channel beta of one event correlates with that of the next at 0.91-0.98
  (early) and 0.97-0.99 (late). Late events have the same couplings, but there the common mode carries no ringing.
- **It decays with ToF.** The median number of channels below -5 sigma per plane (A x) is 20 at 1-2 ms, 42 at 3-8 ms,
  17 at 8-12 ms, 7 at 12-20 ms and 0 from 20 ms on. A plane has at least 3 such channels in 76 % of the events at
  1-2 ms, 95 % at 3-8 ms, 66 % at 12-20 ms, 20 % at 20-30 ms and 0.8 % after 30 ms. The time since the previous
  trigger does not matter.
- **Before ~2 ms there is a second, local component.** Groups of 10-30 neighbouring channels carry a slow baseline
  excursion of up to -300 ADC that the block does not share (figure `c9_residual_C_y.png`). It looks like a recovery
  after the flash. The correction below does not remove it, and a running median along the channels removes it only
  at the cost of 2-16 % of a real track's charge.

## The correction (`deflect.py`)

Per event and per block, we fit each channel's waveform to the block median, x_ch(t) = beta_ch c(t), and subtract it.
A channel that still has a positive peak above 5 sigma may carry ionisation, which would pull its beta. Such a channel
(and its two neighbours) takes the median beta of the clean channels within 4 channels, since beta is smooth along the
channels. In "gated" mode the correction is applied only to a plane with at least 3 channels below -5 sigma, so every
other plane stays bit-identical to pass 1 (checked: 292 of 292 ungated events identical).

On the 4,000 events (A x, FEU 3):

| ToF | channels below -5 sigma, median (90th percentile) | channels above +5 sigma, median |
|---|---|---|
| 1-3 ms | 16 (127) -> 3 (34) | 11 -> 1 |
| 3-8 ms | 42 (136) -> 2 (51) | 37 -> 4 |
| 8-12 ms | 17 (65) -> 0 (12) | 12 -> 0 |
| 12-20 ms | 7 (22) -> 0 (0) | 4 -> 0 |

On C y (FEU 8) the 3-8 ms median goes from 35 to 7. The signal is preserved: on late events the summed positive
charge of a signal channel keeps 1.000 of its value (median; 10th percentile 0.98) on all four FEUs. A fit with an
intercept, or a fit that also uses the time derivative of c(t), is not better; the intercept version halves real
pulses.

## Does it separate fake tracks from real ones? (`window_charge.py`)

For every gated A/C track we take the channels around the fitted track (|position - p0| <= |tan| x 28 mm + 3
strips) and count the channels whose peak stays above 5 sigma after the correction. A ringing fake loses them; a real
track keeps them. The sample is five sub-runs: run_145 stat090_0000-0002, run_104 stat090_0003 and run_139
stat090_0003 (93,332 gated A/C tracks, 69,872 of them with a time since the flash). We evaluate the cuts exactly as in the charge-cut study. Purity is the
pointing excess of the kept tracks over the late excess. Efficiency is the share of real tracks kept, from the excess.
Each bin holds 2,000-6,000 tracks, so a purity is uncertain by about +-0.03.

**The waveform test identifies the ringing fakes.** On late tracks it keeps 100 % of the reference tracks (>= 30 ms,
wall-confirmed, pointing) on A and C, with no dependence on the angle. The fit-charge cut q >= 600 ADC keeps 99.8 % on
A and 98.9 % on C, and on C it loses 2.6 % of the near-normal tracks (0.974 at |tan x| < 0.05 against 0.997 at
0.4-0.5).

Purity per ToF bin, all five sub-runs (A / C):

| rule | 3-5 ms | 5-8 ms | 8-12 ms | 12-16 ms | 16-20 ms | reference kept |
|---|---|---|---|---|---|---|
| pass-1 gated | 0.11 / 0.07 | 0.09 / 0.15 | 0.35 / 0.34 | 0.69 / 0.72 | 0.93 / 0.88 | 1 / 1 |
| q >= 600 ADC | 0.30 / 0.30 | 0.27 / 0.30 | 0.55 / 0.49 | 0.77 / 0.76 | 0.91 / 0.85 | 0.998 / 0.989 |
| ringing-free (>= 1 channel above 5 sigma after the correction, in both views) | 0.19 / 0.11 | 0.25 / 0.31 | 0.60 / 0.51 | 0.77 / 0.76 | 0.91 / 0.83 | 1.000 / 1.000 |
| both | 0.37 / 0.31 | 0.35 / 0.34 | 0.60 / 0.53 | 0.77 / 0.75 | 0.92 / 0.87 | 0.998 / 0.989 |

The kept fractions of the early tracks are, for the same rules at 5-8 ms: 0.33 / 0.41 (q >= 600), 0.46 / 0.44
(ringing-free) and 0.25 / 0.37 (both). The real-track efficiency of every rule is 0.94-1.03 from 8 ms on, which is 1 within its
statistical error.

What this shows:

- **From 8 ms on, the ringing-free test does what the charge cut does, without its cost.** It reaches the same purity
  (A 0.60 against 0.55 at 8-12 ms, 0.77 against 0.77 at 12-16 ms) and keeps every reference track, at every angle.
- **Before 8 ms the charge cut removes more.** A fake can carry a little real-looking charge after the correction,
  while its fit charge stays low. The two tests together are the best rule: 0.35-0.37 purity at 3-8 ms on A against
  0.27-0.30 with the charge cut alone.
- **After either cut, the remaining non-pointing early tracks are not ringing.** On run_145, in the 3-8 ms bin on A, the tracks
  that pass both tests point at 0.36 of the late rate. The ones the waveform test calls ringing but the charge cut
  keeps point at 0.02 (530 tracks), and the ones the charge cut removes but the waveform test keeps point at 0.07.
  The 8-12 ms numbers are 0.64, 0.19 and 0.00. The non-pointing tracks that pass both tests keep their charge after the
  correction (median 1,610 ADC), and the wall fires for 46-69 % of them on A (the pointing ones: 75-90 %). Their x/y
  charge balance is broader: 24 % have |ln(q_x/q_y)| > 0.5 at 3-8 ms, against 6 % of the pointing ones. So they are
  real ionisation, partly mis-paired x/y views, not ringing.
- The three runs agree within the statistics (run_104, the earliest, has the same pattern).

## Re-running the reconstruction on corrected waveforms (`reco_test.py`, `evaluate.py`)

We ran the pass-1 reconstruction (wft_beam, the pass-1 bundles A_E2_beam and C_E2_beam, min strips 3, hits seeding) on
run_145 stat090_0000 tags 000 and 001, with the correction installed in `wft.io.FeuReader` ("gated": only planes with
at least 3 channels below -5 sigma are corrected). In a corrected plane, an analyzer hit whose corrected waveform stays
below 5 sigma no longer seeds a window. We compare with the canonical pass-1 stage 2 of the same triggers, through
build_tracks on the same footing (frozen v, gated and |tan| < 0.5).

- **The driver reproduces pass 1.** Without the correction (mode `pass1`, tag 000, A) every candidate is found again;
  p0, w and q_sum agree for 99.8-99.9 % of them and the gate for 100 %. The tangents differ on 6 % of the candidates,
  which is the bow-partner fix now in `wft/reco.py` (audit finding 21), not this study.
- **The corrected planes are most of the early ones.** In tag 000, 81 % (A) and 80 % (C) of the planes read were
  corrected, and the hits filter dropped 72,587 (A) and 112,053 (C) analyzer hits. The number of seeded events in the
  trigger list fell from 3,136 to 2,046 on A.
- **Late tracks keep their tangents.** Of the late (>= 30 ms) tracks, 99.3 % (A, 137 tracks) and 100 % (C, 107) keep
  their tangents within 0.01; 93 % and 91 % are bit-identical (the rest is mostly the bow-partner fix). Every late
  wall-confirmed track is found again, in every |tan x| bin.
- **Early tracks.** The kept fraction of the pass-1 gated tracks (A, 2 tags) is 0.48 at 3-5 ms, 0.57 at 5-8 ms, 0.64
  at 8-12 ms and 0.93 at 12-16 ms. The charge cut keeps 0.22, 0.33, 0.57 and 0.94. The purity of the re-reconstructed
  tracks is within the errors of the charge cut's, and with the charge cut on top it is 0.38 / 0.52 at 5-8 / 8-12 ms
  (A; C 0.39 / 0.68) against 0.34 / 0.42 (C 0.26 / 0.54) for the charge cut alone.
- **These purities have large errors.** Each bin holds 100-300 tracks, and the late excess they are divided by comes
  from 107-137 tracks, so a purity is uncertain by about +-0.1. The re-reconstruction agrees in direction with the
  high-statistics window test above, but it does not measure the gain on its own.

## Cost of a campaign pass

- **The waveform flag at table level** (the window test, no re-fit): one pass over the decoded waveforms of the
  production sub-runs. The 293 production sub-runs hold 2,835 GB of decoded data (all eight FEUs); A and C are about
  1.4 TB, and A, C, D about 2.1 TB. On the desktop one sub-run takes ~1.3 min on 4 cores for A and C, dominated by
  reading. On condor that is ~300 short jobs (xrdcp of 5-7 GB each, then a few CPU minutes), writing a per-track
  table that joins on (tag, event_id, arm, track_id).
- **A pass 2 with corrected waveforms**: the fits cost about 2-3 CPU-s per seeded event here (measured under load:
  170 CPU-min for 3,136 events). Restricted to 3-20 ms (50 % of the triggers) and with the hits filter (35 % fewer
  seeded events), it is roughly 3,233 tags x 3 arms x ~1,500 events x 2.5 s, about 10,000 CPU-h. All events would be
  about twice that.

## The open items, closed (2026-11-23)

The first version of this study left four questions open. We answer each of them here, on the same existing data:
run_145 stat090_0000-0002, run_104 stat090_0003 and run_139 stat090_0003. Chamber D's waveforms (FEUs 1 and 2) were
fetched from EOS for the four sub-runs that did not have them locally. Every number below can be regenerated with the
scripts named in each part.

### Summary

- **Chamber D has a different problem.** D's ringing after the flash is small. Instead, about 50 channels per view
  oscillate all the time, also at 30-80 ms. The waveform test removes no D fakes beyond the cut already in `gated`
  (`d_faint`), so D should not use it.
- **The slow local excursions cannot be removed without taking charge from real tracks.** They are confined to the
  first 1.25 ms after the flash. Every correction we tried that removes them also takes 10-40 % of the charge of a
  real track in some events. The plain correction (`deflect`, v1) stays the best correction we have.
- **The fit's own window gives the same answer as the symmetric window.** Purity and efficiency agree within their
  statistical errors in every ToF bin and at every angle.
- **The other questions have simple answers.** The ringing frequency is a fixed property of each chamber. The noise
  after the correction is at or below the pedestal noise, so the 5-sigma thresholds need no new estimate. The gated
  correction is the right choice for A and C.

### 1. Chamber D (`char_all.py`, `d_osc.py`)

We measured D's waveforms in the same way as A's and C's (`char_all.py`: 4,000 events per FEU from tag 000 of run_145
stat090_0000, all six FEUs side by side; out `char_run_145_stat090_0000_000.csv`, `char_cross_*.csv`).

**D rings after the flash, but less than A and C.** The block common mode of D oscillates at 2.5 MHz (x) and 2.75 MHz
(y), close to A's 2.42 MHz. At 3-8 ms its rms is 10 ADC (D x) and 34 ADC (D y), against 26-55 ADC on A and C. The
per-channel coupling is as stable as on A and C (event-to-event correlation 0.87-0.91).

**D's main problem is a set of permanently oscillating channels.** About 50 channels per view carry a 2.5-2.8 MHz
oscillation at every ToF, also at 30-80 ms, where A and C are quiet. Their rms is 75-150 ADC, which is 5-13 times the
noise that the pedestal run measured for them. On D x the oscillation is the same waveform on all of them (|corr|
0.99 within an event). The set is stable: 49 (x) and 46 (y) channels are common to run_104, run_139 and run_145. The
D bundle (`D_E2_beam_L1`) masks 30 of the x channels but only 4 of the y channels. So 19 x channels and 42 y channels
oscillate unmasked: x 1, 27, 50, 54, 128-137, 420, 426, 432, 438, 444; y 128-137, 146, 152 and 30 channels between 191
and 319 at a spacing of 3 or 6 (`d_osc_channels.json`). Channels 128-137 are the first ten channels of the third
64-channel block on both FEUs. The stage-1 hit-occupancy mask (`hot_mask_*.csv`) contains none of them, so they do not
fire the hit analyser more than other channels. They do sit in the fit windows, where the fit weights them with a
noise that is 5-13 times too small.

**This is why D has median 37 channels below -5 sigma at 30-80 ms**, against 0 on A and C. The gate of `deflect`
("at least 3 channels below -5 sigma") therefore fires on every D y plane at every ToF, and on 76-100 % of the D x
planes from 5 ms on. One detail we cannot explain: D x's oscillating channels are quiet from 1 to 5 ms after the flash
(median 0-1 channels below -5 sigma) and come back from 5 ms on. D y's are not quiet.

**The correction does less for D.** `deflect` halves D's negative channels at every ToF (D x 30-80 ms: 34 -> 17). It
does not remove the oscillating channels, because the block median does not carry their waveform. On injected real
track charge (see part 2) it keeps 0.93-1.04 of the charge summed over all events. However, in 10 % of the events a
D track loses 3-22 % of its charge (A and C: 2-10 % from 2 ms on).

**The waveform test does not help D** (`window_own.py`, all five sub-runs, 56,300 gated D tracks). D's cut in
`gated`, `d_faint` (amp_norm < 0.25 or the x window >= 80 % censored), already removes the faint wide fakes. Purity of
the late D tracks relative to the >= 30 ms excess, at 20-30 ms:

| rule | purity | late reference kept | near-normal (\|tan x\| < 0.05) reference kept |
|---|---|---|---|
| gated (view) | 0.47 | 1 | 1 |
| d_faint removed (the rule now in `gated`) | 0.73 | 1 | 0.87 |
| d_faint + >= 1 channel above 5 sigma after `deflect`, own window | 0.72 | 0.99 | 0.77 |
| d_faint + q >= 600 | 0.76 | 0.98 | 0.87 |

The waveform test gives the same purity as `d_faint` alone and loses near-normal tracks. Before 20 ms, D's purity
stays at 0.1-0.2 with every rule. So D's early tracks that remain are not ringing fakes either.

**Do the oscillating channels spoil D's tracks?** 56-83 % of D's gated tracks have at least one unmasked oscillating
channel in their own fit window (`d_osc.py tracks`, `d_osc_tracks.csv`). At 30-80 ms these tracks point as well as the
others (pointing excess 0.36 against 0.34) and the wall confirms them as often (0.97 against 0.94). At 20-30 ms they
point worse (0.17 against 0.23) and the wall confirms fewer (0.58 against 0.75). However, the tracks with such a
channel also have wider windows, so this comparison does not separate cause and effect. Masking the channels changes
the fit itself, so its effect can only be measured on a re-reconstruction of a few sub-runs.

### 2. The local baseline excursions (`excursions.py`, `bench_corr.py`)

**What they are.** After `deflect`, some groups of neighbouring channels sit below the baseline for the whole 1.2 µs
window. We call a channel low when the median of its corrected waveform is below -3 sigma. An excursion is a run of at
least 5 neighbouring low channels (`excursions.py`, 16,000 events per FEU from tags 000-003 of run_145 stat090_0000;
out `exc_summary.csv`, `exc_repeat.csv`, `exc_flash.csv`, `exc_shapes.png`, `exc_channels.png`).

- **They belong to the first 1.25 ms.** On A and C, 33-48 % of the events at 0.5-1 ms carry one and 11-30 % at
  1.0-1.25 ms. From 1.25 ms on, at most 2 % of the events carry one, and almost none do from 1.5 ms on. Only D y keeps
  17-20 % up to 1.5 ms and 4-7 % up to 2 ms.
- **They are short and flat.** The median excursion is 5-6 channels wide (99th percentile 10-14) and -75 to -100 ADC
  deep (6-7 sigma). The groups we see are smaller than the 10-30 channel, -300 ADC example of the first study. That
  example was at < 1 ms. The level does not recover inside the window (the median slope is a few ADC): it is an offset,
  not a decay we could fit.
- **They are not the same channels from event to event.** The overlap of the channel sets of two consecutive early
  events is 0 (median), the same as for two random events. No channel is in an excursion in more than a third of the
  events, and they are spread over all eight blocks.
- **They do not follow the flash size.** We take the rms of the flash trigger of the same proton bunch as the flash
  size. The rank correlation of the flash size with the excursion depth of that bunch's 1-2 ms events is between -0.11
  and +0.06 on all six FEUs. The fraction of events with an excursion is the same in each third of the flash sizes.
  They do follow the size of the event's own common mode (rank correlation 0.1-0.8 with the number of low channels,
  above 0.4 on four of the six FEUs).
- On A x and D y they lie next to a large positive pulse more often than a random place does (within 15 channels: A x
  0.67 against 0.50, D y 0.68 against 0.25). On C there is no difference. So part of them may be the undershoot of a
  large deposit, but this is not established.

**How many analysis tracks are concerned.** The flash veto is at 1 ms, so 1-2 ms is inside the analysis range. Of the
gated tracks with a time in the whole analysis view (`stage3_vt`), 4.5 % (A), 2.5 % (C) and 1.4 % (D) are at
1.0-1.25 ms, and 7.4 % (A), 6.6 % (C) and 2.4 % (D) at 1-2 ms. Their pointing purity is 0.01-0.04 of the late value.
So almost all of them are fakes already, and the analyses that start at 12 or 25 ms never see them.

**Corrections we tried.** The bench (`bench_corr.py`, out `bench_run_145_stat090_0000_<tag>.csv`) runs every variant
on the same events. It counts the channels below and above 5 sigma, and it measures what a real track keeps: a real
late track's pulse (a cluster of >= 3 neighbouring channels above 8 sigma and 100 ADC, with its neighbours) is added
to an event of each ToF bin, and we compare the corrected waveform with and without it. The variants, all in
`deflect.py`:

- `deflect` (v1): the block common mode with a per-channel coupling, the correction of the first study.
- `deflect2`: the same with each pulse protected by leaving its samples out of the fit (> 3 sigma, +-2 samples), with
  optionally a second component per block (the leading time component of the residual of the pulse-free channels,
  "rank 2") and an offset per channel.
- `deflect3`: v1, then the second component only in a block that really carries one (its rms > 1.5-3 times the noise).
- `deflect4`: v1 and the second component, each fit repeated three times with the positive residual samples left out.
- `group_dc`: v1, then each channel of a detected excursion loses its own offset.

The result, on run_145 (A, C and D, 80-150 events per ToF bin):

| variant | A x channels below -5 sigma, 0-1 / 3-5 / 5-8 ms | track charge kept, summed over the events (A, C, from 1 ms) | worst 10 % of the events, 1-2 ms / from 2 ms |
|---|---|---|---|
| plain (pass 1) | 122 / 48 / 47 | 0.98-0.99 | 0.96-0.98 / 0.96-0.99 |
| v1 `deflect` | 60 / 2 / 1 | 0.94-1.02 | 0.81-0.92 / 0.90-0.98 |
| v2, pulse samples left out, rank 1 | 30 / 27 / 21 | 0.94-1.00 | 0.81-0.95 / 0.91-0.97 |
| v2 rank 2 | 6 / 27 / 21 | 0.95-1.01 | 0.83-0.93 / 0.89-0.94 |
| v3 (v1 + second component, 2 sigma) | 30 / 2 / 1 | 0.70-0.97 | 0.47-0.65 / 0.47-0.97 |
| v4 (iterated masks) | 35 / 2 / 0 | 0.87-0.99 | 0.79-0.89 / 0.87-0.97 |
| v1 + group_dc | as v1 | as v1 | as v1 |

The "worst 10 %" column is the 10th percentile of the charge a track keeps, event by event. The v2 rows come from a
second run of the bench on the same events with finer ToF bins (1-1.25, 1.25-1.5, 1.5-2 ms), whose v1 numbers agree
with the first run.

What this shows:

- **v1 is the best correction.** It removes the block ringing from 2 ms on (A x 3-5 ms: 48 -> 2 channels) and keeps
  0.94-1.02 of a real track's charge, summed over the events, at every ToF from 1 ms on, on A, C and D. Its cost is in
  single events: in 10 % of the events at 1-2 ms a track loses 8-19 % of its charge, and from 2 ms on 2-10 %.
- **A second component removes the excursions, but it either takes real charge or misses the ringing.** Fitted on
  every channel together with the block common mode (v2 rank 2), it lowers the channels below -5 sigma at 0-1 ms from
  60 to 6 on A x, and it keeps a track's charge about as well as v1 at 1-2 ms. However, it leaves the 3-8 ms ringing
  in place (27 channels against 2) and costs late tracks more (worst 10 %: 0.89-0.94 against 0.98). Added after v1
  (v3), it also takes the shared charge on a track's neighbouring strips: v3 keeps 0.70-0.97 of a track's charge on
  average, and 0.47-0.65 in the worst 10 % of the events at 1-2 ms. A v1 + rank-2 hybrid applied only at 1-1.5 ms
  would be possible, but it would act on tracks whose pointing purity there is 0.01-0.04.
- **Leaving the pulse samples out of the fit does not protect the pulse.** It hurts the block ringing fit (v2: 27
  channels left at 3-5 ms against 2 for v1), because the ringing's own positive lobes are then left out. Repeating
  the fit with the residual masks (v4) does not do better than v1.
- **The offset correction of the excursions changes nothing measurable** (`group_dc`): the excursions are rare after
  1 ms and seldom under a track.

So the excursions stay. They cost the few real tracks at 1.0-1.25 ms part of their charge and do not make fakes by
themselves, since they are negative.

### 3. The fit's own window (`window_own.py`)

`window_charge.py` used a window that is symmetric about p0 (|position - p0| <= |tan| x 28 mm + 3 strips). The fit's
own window is the seed cluster plus 3 live strips on each side. We rebuild it from the reco's own strip count
(`n_strips`) as that many strips in position order, centred on the column centre p0 + tan x 28 mm / 2. The view
carries the reco sign of tan, which we checked against stage 2 (equal for 100 % of A and C tracks). We measured all
gated A, C and D tracks of the five sub-runs (50,500 A, 42,800 C and 56,300 D tracks; out `wo_<run>_<sub>.parquet`,
`wo_eval.csv`, `wo_angle.csv`).

The two windows give the same result. Purity of the test ">= 1 channel above 5 sigma after `deflect`" (A / C):

| window | 3-5 ms | 5-8 ms | 8-12 ms | 12-16 ms | 16-20 ms | reference kept | near-normal kept |
|---|---|---|---|---|---|---|---|
| symmetric | 0.16 / 0.11 | 0.21 / 0.29 | 0.59 / 0.50 | 0.78 / 0.75 | 0.93 / 0.90 | 1 / 1 | 1 / 1 |
| own | 0.15 / 0.10 | 0.22 / 0.27 | 0.56 / 0.50 | 0.77 / 0.76 | 0.90 / 0.85 | 1 / 1 | 1 / 1 |
| own, with q >= 600 | 0.39 / 0.31 | 0.35 / 0.37 | 0.56 / 0.52 | 0.75 / 0.76 | 0.90 / 0.87 | 0.998 / 0.989 | 0.997 / 0.983 |
| q >= 600 alone | 0.31 / 0.24 | 0.27 / 0.30 | 0.59 / 0.51 | 0.76 / 0.76 | 0.92 / 0.87 | 0.998 / 0.989 | 0.997 / 0.983 |

Each purity is uncertain by about +-0.03, so every difference between the two windows is within the error. The own
window keeps a few more early tracks (A 1-3 ms: 0.22 against 0.15), because it is wider on the side the track does not
reach. The test on the stronger correction (v3) gives the same purity as v1 within the errors (A 3-5 ms 0.22, C 0.12).
Requiring 2 channels instead of 1 adds nothing. The earlier conclusion therefore stands: from 8 ms on the waveform test
matches the charge cut and keeps every reference track at every angle; before 8 ms the two together are best.

### 4. The remaining questions

**Why are 80 % of the early planes corrected?** Because almost every early plane rings. The gate ("at least 3
channels below -5 sigma after the plain median") fires on 51-100 % of the A and C planes in each ToF bin of 1-8 ms
(above 90 % in most bins), on 50-84 % at 8-12 ms, on 10-66 % at 12-20 ms (C y least, A x most) and on 1-24 % at
20-30 ms. From 30 ms on it fires on 0.6-3.6 %. Half of the triggers are at 1-20 ms, which gives the 80 %. On D it fires
on nearly every plane, because of the oscillating channels.

**Should the correction run on every plane, not only the gated ones?** For A and C there is nothing to gain. On late
planes the correction keeps a real track's charge as the plain median does (0.99 summed), and it lowers the residual
noise a little (A x: 0.95 -> 0.83 of the pedestal noise). However, the gated mode keeps every late plane bit-identical
to pass 1, which makes a pass 2 comparable with pass 1 track by track. So we keep the gate. For D, gated and all are
the same thing.

**Should the 5-sigma noise be re-estimated after the correction?** No. We measure the per-event noise as the robust
spread (MAD) of the corrected samples over the well-behaved channels, divided by the pedestal noise (`freq.py`,
`noise_after.csv`). Before the correction it is 1.0-3.0 at 0-1 ms and 0.8-1.5 at 1-3 ms, so in the first millisecond
the 5-sigma threshold sat at only 1.7-5 real sigma. After the correction it is 0.2-0.9 at every ToF and on every FEU
(90th percentile <= 1.1).
So after the correction the threshold is at least as strict as it is on late events.

**deflect_wave against deflect_hits.** Only `deflect_hits` was ever run (`reco/deflect_hits*`). It keeps the pass-1
seeding (the analyser's hits) and drops a hit whose corrected waveform stays below 5 sigma. The
`deflect_hits_all` run is the same reconstruction on more events, not a different mode: its fits agree with
`deflect_hits` on every shared candidate. Seeding from the corrected waveforms (`deflect_wave`) is not tested. Since
`deflect_hits` changes the least, it is the mode to use.

**Why does C ring at 1 MHz and A at 2.4 MHz?** We cannot say what sets the frequency, but it is a fixed property of
each chamber and view (`freq.py`, `freq.csv`). It is the same in run_104, run_139 and run_145 (late July to August)
and at every ToF from 1 to 20 ms: A 2.42 MHz in both views, C 1.00 (x) and 0.96 (y) MHz, D 2.46-2.54 (x) and
2.71-2.79 (y) MHz. A frequency that does not move with time or beam conditions is a resonance of the chamber's own
readout or high-voltage circuit, excited by the flash. The x and y common modes of one chamber are coherent, but with
different signs: A x with A y +0.88, C x with C y -0.84, D x with D y +0.40 (the first study quoted 0.79 for the
magnitude). Different chambers do not correlate (|corr| < 0.17), except D x with A x (+0.40), which ring at nearly the
same frequency.

### Recommendation for a pass 2

- **A and C:** correct the waveforms with `deflect` (v1), gated, in mode `deflect_hits`. It keeps 0.98-1.00 of a real
  track's charge and leaves late planes identical to pass 1. Add the per-track waveform flag (own window, >= 1 channel
  above 5 sigma after the correction, both views) as a column. Without a pass 2, the flag can still be made at table
  level by one pass over the waveforms (~300 condor jobs, see "Cost").
- **D:** do not use the waveform test (no gain over `d_faint`, and it loses near-normal tracks). Do not apply the
  correction to D either: it does not reach D's oscillating channels and costs D tracks 3-22 % of their charge in 10 %
  of the events. The change worth testing for D is to add the 19 x and 42 y oscillating channels to D's hot list, so
  that the fit censors them. Its effect is not measured; a re-reconstruction of a few D sub-runs would measure it.
- **Nothing for the excursions before 1.25 ms.** No correction removes them without taking real charge.

## What is still open

- The effect of masking D's oscillating channels on D's tracks (needs a re-reconstruction of a few sub-runs).
- What drives the local excursions in the first 1.25 ms. They are not the flash size and not fixed channels. Part of
  them sit next to large pulses.
- The re-reconstruction test (`reco_test.py`) still has too few tracks to measure the gain on its own; the window test
  does.
- Out of this study's scope, noted while reading the reco metadata: the reconstruction's `quality_ok` rule takes the
  median chi2/dof of each table as its reference (`quality_rule` in the meta), which the ringing sets early (roadmap
  item 4.6).

## Files

- `deflect.py`: the correction, and `install()` which makes `wft.io.FeuReader` serve corrected waveforms.
- `window_charge.py`: the per-track test on the pass-1 tracks (`measure`, `eval`).
- `reco_test.py`: the pass-1 reconstruction of run_145 with the correction (modes `pass1`, `deflect_hits`,
  `deflect_wave`); `evaluate.py`: the comparison with pass 1 through build_tracks.
- `char_all.py`: the ringing measured the same way on all six FEUs (A, C, D), with a waveform cache
  (`extract`, `table`).
- `bench_corr.py`: every correction variant on the same events: channels below and above 5 sigma, and the charge of
  an injected real track.
- `excursions.py`: the local baseline excursions (where, when, how deep, repeat, flash size).
- `window_own.py`: the per-track test on the fit's own window and the symmetric one, for A, C and D, with `d_faint`.
- `d_osc.py`: D's permanently oscillating channels (`find`) and the D tracks whose window holds one (`tracks`).
- `freq.py`: the ringing frequency per chamber, run and ToF, and the noise after the correction.
- Outputs: `$WORK_DATA/ntof_pass1/ana/ringing/` (`wq_*.parquet`, `wq_eval.csv`, `wq_rules.csv`, `wq_angle.csv`,
  `reco/<mode>/`, `eval_*.csv`, the figures; from 2026-11-23 also `char_*.csv`, `bench_*.csv`, `exc_*`, `wo_*`,
  `d_osc_*`, `freq.csv`, `noise_after.csv`, and the waveform cache `cache/`).
