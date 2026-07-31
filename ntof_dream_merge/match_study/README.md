# DREAM ↔ n_TOF matching study

How well every DREAM trigger can be tied to an n_TOF sector SINGLES (SiPM wall
top+bottom sum **AND** plastic), how well the n_TOF detectors are aligned in
time, and how tight the accept window can be made.

Run on the **complete reference pair**: run_79 `stat090_0000` + `stat090_0001`
(2061 bunches, 213 420 non-flash DREAM triggers) against n_TOF run 224572
processed with **`v12_liqpileup`**, read on the file's own stored `tflash`
(the laptop-side repair OFF — see below).

Report: `latex/dream_ntof_matching_slides.pdf`.
The calibration, the constants and the per-run recipe:
`../DREAM_NTOF_CALIBRATION.md` -- **the authoritative document**.

## Headline

| at the accept window \|r\| < 25 ns | wall AND plastic | wall only |
|---|---|---|
| efficiency | **95.84 %** | 98.30 % |
| accidental match rate (measured) | **0.049 %** | 1.04 % |
| purity of the matched sample | 99.998 % | 99.982 % |
| >= 2 candidates in the window | 2.46 % | 5.36 % |
| >= 2 arms in the window | **0.15 %** | 3.04 % |

Match resolution 6 ns (68 % half-width), flat from 1 ms to 80 ms, cross-validated.

The window is that tight because the residual band was never a resolution: it
was the DREAM timestamp clock drifting ~1 ppm from bunch to bunch, which smears
the residual in proportion to the time since the flash. Corrected per bunch, the
band is flat.

## Pipeline

Run from this directory's `scripts/` (they import `study_common` by name, so
`cd` there first).

```
python validate_fast.py 40            # fast_singles == dream_trigger, exactly
python build_candidates.py stat090_0000 --chunk 250     # ~7 min per sub-run
python build_candidates.py stat090_0001 --chunk 250
python fit_timebase.py                # K, T0 and the per-arm offsets
python fit_perbunch.py                # the per-bunch clock, cross-validated
python window_scan.py --timebase fitarm     # the global map, for the figures
python window_scan.py --timebase perbunch   # the calibration
python recommend_window.py            # confirms the ±25 ns knee
python align_survey.py --nb 250       # the four levels of internal alignment
python bias_check.py                  # five tests that the per-bunch fit is honest
python tb_offset_compare.py           # official vs v12 wall top/bottom offsets
python make_figures.py
cd ../latex && make
```

`build_candidates.py` is the only expensive step. Everything after it works on
the cached candidate arrays in `data/` and takes seconds.

## What each script is for

| script | what it answers |
|---|---|
| `validate_fast.py` | proves `ntof_dream_merge/fast_singles.py` reproduces `dream_trigger.singles_candidates` bit for bit (it is a speed rewrite: the original is O(N_hits × N_bunches) and cannot run on 2061 bunches) |
| `build_candidates.py` | rebuilds the N1081B sector SINGLES for a whole sub-run, both legs (`wp` = wall AND plastic, `w` = wall only), and caches them |
| `fit_timebase.py` | re-fits `t_nTOF = t_DREAM(1+K) + T0` on the candidate processing, globally, per sub-run and per arm |
| `fit_perbunch.py` | fits the residual clock error per bunch and reports what it buys, **cross-validated** (odd triggers fit, even triggers evaluate) |
| `window_scan.py` | efficiency / accidental rate / purity / ambiguity for any accept window, globally, per time-since-flash bin, per arm, both legs |
| `recommend_window.py` | turns the scans into one number (tightest window within 0.5 % of the efficiency plateau) |
| `align_survey.py` | the four alignment levels: flash-vs-pickup, wall-vs-plastic (per arm and per channel), top-vs-bottom, liquid-vs-wall, plus the stored-vs-repaired `tflash` systematic |
| `tb_offset_compare.py` | the wall top/bottom "cable offsets" on the official file vs v12 |
| `bias_check.py` | five tests that the per-bunch clock fit is not manufacturing its own matches: statistics per bunch, split-half reproducibility, in-sample vs cross-validated width, wide-window invariance, and a wrong-bunch parameter swap |
| `make_figures.py` | every figure in the slides |

## Traps this study had to honour

- **`ntof_io`'s caches are keyed by run number only.** The official and the
  reprocessed run 224572 must never share a cache directory. `study_common.
  use_variant()` points the reader at the candidate partials and gives them
  `ntof_io.variant_cache()`, keyed on the file set (REVIEW.md §5).
- **The tflash repair is for the OFFICIAL file.** On v12 it is not a no-op: it
  would shift LIQC/D by 15 ns and add 25 ns RMS on PSSC, and the alignment
  survey shows the stored time base already has the liquids within 1 ns of the
  walls. `fast_singles.REPAIR_TFLASH = False`.
- **`K` and `T0` do not transfer between processings.** The constants the merge
  was built with were fitted on the official file; on v12 they leave a −45 ns
  offset and a 1.35 % rate error. Re-fit per processing.
- **Neither do the wall top/bottom offsets.** They are ±32…39 ns on the official
  file and within ±5.5 ns on v12 — a reconstruction artifact, not cabling.
- **The per-bunch fit must be cross-validated.** It is fitted on matched
  triggers and then used to match; the in-sample width is optimistic. Every
  number quoted here comes from the half-split.
- **The accidental rate is measured, not modelled** — the identical match with
  the DREAM time shifted ±100 µs. The two shifts agree (0.046 % / 0.062 % at
  ±25 ns).

## Layout

```
scripts/   the pipeline above; study_common.py holds the paths and constants
data/      cached candidates, scans, fitted parameters (npz + json)
figures/   vector PDFs used by the slides
latex/     the slides, the TikZ diagrams (diagrams.tex), and their Makefile
```
