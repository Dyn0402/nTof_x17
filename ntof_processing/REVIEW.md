# For a reviewer: how to check this work

Written for someone -- human or model -- coming to audit the n_TOF reprocessing
without having done it. `STATUS.md` says *what the state is*; this says *how to
verify it, and where to look hardest*.

Everything below is on run 224572 unless stated. Repo commits `40fbf64`
(first PSA work) through `fc6ead4`.

---

## 1. Reading order

1. `FLASH_TIME_BASE.md` -- the SiPM divert and why the flash timing was wrong.
   Read this first; the rest assumes it.
2. `FINDINGS_2026-07-28_psa_optimization.md` -- every measurement behind the
   UserInput changes, in the order they were made, including the ones that
   failed.
3. `liq_study/FINDINGS_liquids.md` -- the liquids, which are their own story.
4. `userinputs/README.md` -- the variant table and how to run one.
5. `report/comparison.tex` -- the outward-facing version.
6. `HANDOFF_2026-07-28_ntof_processing.md` -- the pre-existing handoff this
   started from. Some of its assumptions were later disproved; see §5.

Adjacent and *not* mine: `flash_timing/` is the absolute flash calibration
(`t_flash = tof_PKUP + C`), done separately. It is the authority on timing;
this work is the authority on hit content.

## 2. Every headline claim, and what produced it

| claim | tool | sample |
|---|---|---|
| PSS flash mis-tagged in 37-85 % of bunches | `grade_candidate.py` | 421 bunches |
| walls time the divert gate, not the flash | `raw_flash_extract.py` + `flash_finder_emulator.py` | 851 channel-bunches, 21 bunches across the run |
| proposed G-FLASH params select the right feature | `flash_finder_emulator.py` | same 851 |
| flash-id 0.0 % after reprocessing | `grade_candidate.py` | 421 bunches |
| coincidence offsets -362 -> -3.5 ns | `grade_candidate.py` | 421 bunches |
| AREA/AMP cut removed ~25 % PSS, ~19 % LIQ | raw waveforms, `threshold_headroom.py` | ~15 k / 2.6 k isolated pulses |
| wall templates improve chi2 | `compare_fits.py` | 1 partial (~200 k hits/tree) |
| matcher 93.7 % -> 96.4 % | `dream_regression.py` | 100 bunches, 10 452 events |
| plastic leg is the limit (98.9 % wall-only) | `dream_regression.py` check 3 | same |
| timing/amplitude quality unchanged | `quality_metrics.py` | 1.9 M top-bottom pairs |
| liquids: photon-statistics floor | `is_it_photon_statistics.py` | 929 / 1164 isolated pulses |
| liquids: PSA beats deconvolution 1.5x | `deconv_vs_psa.py` | 3 bunches, 23 081 PSA hits |
| liquid hit spacing 24-30 ns median | inline, from the PSA output | 40 bunches/tree |

`report/results.json` carries the numbers with the tool that produced each.

## 3. Reproducing it

**What is committed**: all tools, all UserInput variants and their templates,
the figures, `report/results.json`, and `liq_study/liq_pulses.npz` (the
extracted isolated liquid pulses -- input to four of the liquid tools).

**What is NOT, and how to get it back**:

- *Processed output* -- on the ntof disk,
  `/eos/experiment/ntof/data/x17/reproc/<variant>/completed/224572/`. Nothing
  needs reprocessing to re-check any claim in §2.
- *Raw stream1 chunks* (~3 GB, 7 chunks) -- were in `/tmp` and are gone on
  reboot. Regenerate:
  ```bash
  xrdfs root://eospublic.cern.ch cat \
    /eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/224572/stream1/run224572_<N>_s1.raw.finished \
    | head -c 450000000 > head_<N>.bin      # N = 8,20,40,60,80,100,120
  ```
  ~3 bunches per chunk. **Raw ages off EOS eventually** --
  `eos_stream1_inventory_2026-07-28.txt` records what was present on 07-28
  (156 of 329 runs, scattered, not a clean age cutoff).
- *Downloaded partials* (~25 GB) -- re-fetch from the ntof disk as needed.

Most claims can be re-checked from the committed npz plus the ntof-disk output,
without re-downloading anything raw.

## 4. Where I would look hardest

Ranked by how much rests on them versus how well they are established.

1. **The matcher improvement (96.4 %) rests on 100 bunches of one DREAM
   sub-run.** It is the headline number and the thinnest sample in the set.
   `dream_regression.py <partials-dir>` on more bunches, or on
   stat090_0001, would settle it. The bunch range must be one the candidate
   partials cover -- see the trap in §6.
2. **Fit-chi2 comparisons use a single partial** (0016, the short tail chunk,
   ~20 bunches). The chi2 differences were large (2x) so the ordering is
   probably safe, but the magnitudes are not well determined.
3. **`quality_metrics.py` accidental subtraction.** Everything there depends on
   an off-time sideband being a fair model of the accidental background. It
   changes T2 from 38.8 ns to 6.4 ns, so it is doing a lot of work. If the
   sideband is contaminated the widths are wrong.
4. **The liquid photon-statistics conclusion** rests on residual scaling as
   sqrt(A) over a factor 25. The scaling is clean (0.61-0.67 on LIQD) but it is
   one detector family, one run, isolated pulses only.
5. **`deconv_vs_psa.py` is a *simple greedy* deconvolution.** "PSA beats it
   1.5x" is evidence against a large missed pool, not proof that no algorithm
   could do better. A joint/regularised fit was not tried.

## 5. Things I got wrong and corrected -- check these harder

A reviewer should know where the error modes were, because that is where
uncorrected ones are most likely to remain.

- **Deconvolution alignment.** `np.correlate(..., mode='same')` centres the
  template on the output index; I offset by the template's *peak* index,
  misplacing every fit by ~89 samples and getting a yield ratio of 0.01.
  Fixed to `mode='valid'`. **This was the second alignment bug of the same
  kind** -- the first was the strip-map/M3 rotation in the June work -- so
  treat any correlate/interp alignment in this repo as suspect until checked.
- **Width estimator quantisation.** `quality_metrics.py` first reported T1
  sigma as *exactly* 3.18 ns for every variant, and "+33 % worse" for v7. Both
  were FWHM/2.355 with 2.5 ns bins against 1 ns-quantised `tof`. Replaced with
  a background-subtracted second moment; v7's real change is -0.4 %.
- **EOS retention.** I reported "only 40 runs remain, ~3 day retention". That
  was my own `tail -40` inside the inventory command, then counting the
  truncated file. Real answer: 156 of 329 runs, 07-02 to 07-28, scattered.
- **Liquid framing.** I described the liquids as severely pileup-limited using
  an "8-24 % isolated" figure. That figure uses a 200 ns isolation window, so
  it measures *tail* overlap. The fast components are mostly resolvable
  (24-30 ns median gap vs 6 ns FWHM). The original framing overstated it.
- **Liquid two-class hypothesis.** I expected n-like and gamma-like pulse
  classes and found "two shapes" by splitting a *unimodal* PSD distribution at
  its percentiles. Excluded: tail/total is one band at 0.21 above 3000 ADC.
- **Liquid template length.** Argued the shipped 24-59 ns templates were too
  short, shipped 551 ns, it was worse; then 81 ns, also worse. Length was never
  the variable.

## 6. Traps a reviewer will hit

- **`ssh -K` is mandatory** on lxplus. Without delegated credentials there is
  no AFS token and no condor auth, and `/eos/user/d/dneff` appears not to
  exist. Every "permission denied" in this workflow traced back to this.
- **Never merge a run.** The official merge node cannot (1024 MB condor
  transfer cap) and hadd over EOS dies leaving a truncated file that still
  opens. `ntof_io.ntof_paths()` chains the partials instead.
- **Caches are keyed by run number only.** A reprocessed run224572 read through
  the normal paths silently reuses the official file's bunch index. Use the
  sandboxing in `validate_reprocessing.py` / `dream_regression.py`.
- **Build the DREAM<->bunch join BEFORE pointing the reader at a candidate.**
  The join runs off PKUP/index for the whole run; a candidate may be a few
  partials. Getting this backwards silently reports "covers none of the
  bunches".
- **`match_window`'s efficiency is not evidence at early times** -- its own
  false-match probability is ~100 % at 1-3 ms. Quote the singles matcher.
- **Heredocs with f-strings**: several `ssh ... <<PY` attempts failed on
  backslash-in-f-string. Write the script to a file and rsync it.

## 7. What is deliberately not established

- Whether the residual 2.5 % plastic-leg inefficiency is threshold modelling,
  channel selection, dead time or real detector inefficiency. Only that it is
  **not** pulse recognition (three very different reconstructions give the same
  96.4 %).
- Whether a joint multi-pulse fit could recover liquid slow components. Argued
  to be ill-conditioned at 24 ns spacing with 150 ns tails; not attempted.
- Whether the +9.1 % variation of `sqrt(amp_top*amp_bot)` across a wall bar is
  light-collection asymmetry or a reconstruction bias. It is a stable number
  variants can be compared on, nothing more.
- Any absolute energy scale. That is `flash_timing/` plus the Y-88/MIP work.
