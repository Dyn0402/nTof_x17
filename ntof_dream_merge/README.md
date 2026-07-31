# ntof_dream_merge

Joining the **n_TOF facility DAQ** (SiPM wall / plastic / liquid scintillator hits) to the
**DREAM Micromegas** stream, so that one merged per-event record carries both.

> ## → Start at [`DREAM_NTOF_CALIBRATION.md`](DREAM_NTOF_CALIBRATION.md)
>
> It is the **authoritative** document for the match: the calibration chain, the
> constants, the per-run recipe, what transfers between runs and what does not,
> and the evidence that the per-bunch clock fit does not manufacture its own
> matches. Every other description of the matching in this repository is retired
> and points there.
>
> Slides: `match_study/latex/dream_ntof_matching_slides.pdf`.
> Tooling: [`match_study/`](match_study/).
> Machine-readable constants: `../../nTof_x17_DAQ/calibrations/dream_ntof/`.

## The match, in one table

Run_79 (`stat090_0000` + `stat090_0001`, 2061 bunches, 213 420 non-flash
triggers) ↔ n_TOF 224572 processed with `v12_liqpileup`:

| at the accept window \|r\| < 25 ns | wall AND plastic | wall only |
|---|---|---|
| efficiency | **95.84 %** | 98.30 % |
| accidental match rate (measured) | **0.049 %** | 1.04 % |
| ≥ 2 arms in the window | **0.15 %** | 3.04 % |

Match resolution 6 ns, flat over the whole 80 ms. Coverage, accidental
subtracted: 98.59 % wall leg, **96.00 %** wall AND plastic — the plastic leg
costs 2.58 %, the wall leg 1.41 %.

## Which n_TOF file

**Ours.** We reprocess n_TOF ourselves and the campaign runs on
`v12_liqpileup`; the local copy is `/media/dylan/data/x17/ntof_reproc/`. Read it
on its **own stored `tflash`** — the laptop-side `tflash_repair` is for the
broken *official* flash finding and must be off. See
[`../ntof_processing/STATUS.md`](../ntof_processing/STATUS.md) for the
processing state.

Note: **run 224572 alone covers both real sub-runs of run_79** (`stat090_0000`
bunches 146–1157, `stat090_0001` bunches 1165–2213). The other fourteen
`stat090_*` directories are empty stubs.

## Quick start

```bash
./stage_reference_pair.sh check      # what is staged for the reference pair
./stage_reference_pair.sh ntof       # xrdcp the nTOF run from EOS (resumable)
./stage_reference_pair.sh manifest   # laptop bundle list + rsync command
```

## Layout

| | |
|---|---|
| `DREAM_NTOF_CALIBRATION.md` | **the calibration — read this first** |
| `calibration.py` | the one place code should get the constants from (`load()`) |
| `match_study/` | the pipeline that derives it, its figures and slides |
| `fast_singles.py` | vectorised N1081B sector-SINGLES rebuild (validated bit-identical against `dream_trigger`; the original is O(N_hits × N_bunches) and cannot run a whole run) |
| `dream_trigger.py` | the reference trigger emulation, and the chain it models |
| `bunch_join.py` | DREAM event → n_TOF bunch (100 %) |
| `ntof_io.py` | partial-chaining reader, bunch index, caches |
| `tflash_repair.py` | official-file flash repair — **not for reprocessed files** |
| `match_window.py`, `time_align.py` | earlier window/alignment tools; their docstrings are superseded, see the banners |
| `liq_coincidence.py`, `mm_activity_crosscheck.py` | first physics through the merge |
| `archive/` | retired documentation — do not build on it |

Analysis outputs are **not** in the repo; they live under
`/mnt/data/x17/beam_july/analysis/ntof_dream_merge/`.
