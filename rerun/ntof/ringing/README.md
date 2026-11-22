# The post-flash ringing in the DREAM waveforms (study, 2026-11-22)

Dylan asked whether the ringing after the n_TOF flash can be told apart from ionisation at the waveform level. This
study answers that question on chambers A and C: the waveforms of run_145 (sub-runs 0000-0002), run_104 stat090_0003
and run_139 stat090_0003. Chamber D was not studied.

**Short answer.** Yes, the ringing can be told apart, and it can be removed from the waveforms before the fit.
- A waveform test ("does the track keep at least one channel above 5 sigma once the ringing is removed?") reaches the
  purity of the 600 ADC charge cut from 8 ms on. It keeps every good late track at every angle, where the charge cut
  loses 0.2 % (A) and 1.1 % (C) of them, and 2.6 % of C's near-normal tracks.
- Together with the charge cut, it raises the purity before 8 ms by 0.04-0.08 (A 3-8 ms: 0.27-0.30 -> 0.35-0.37).
- It does not change the purity from 12 ms on. The non-pointing early tracks that survive the charge cut keep their
  charge once the ringing is removed, so they are real ionisation (partly mis-paired x/y views), not ringing. No
  ringing treatment can remove them.

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

## What is still open

- Chamber D was not studied. Its early fakes are wide and faint (audit finding 15), which the waveform test should
  catch, but this is not measured.
- The local baseline excursions before ~2 ms are not removed by the correction.
- The re-reconstruction test has too few tracks to measure its gain; only the window test does.
- The window test uses a symmetric window around the fitted p0 (+-|tan| x 28 mm + 3 strips), not the fit's own window.

## Files

- `deflect.py`: the correction, and `install()` which makes `wft.io.FeuReader` serve corrected waveforms.
- `window_charge.py`: the per-track test on the pass-1 tracks (`measure`, `eval`).
- `reco_test.py`: the pass-1 reconstruction of run_145 with the correction (modes `pass1`, `deflect_hits`,
  `deflect_wave`); `evaluate.py`: the comparison with pass 1 through build_tracks.
- Outputs: `$WORK_DATA/ntof_pass1/ana/ringing/` (`wq_*.parquet`, `wq_eval.csv`, `wq_rules.csv`, `wq_angle.csv`,
  `reco/<mode>/`, `eval_*.csv`, the figures).
