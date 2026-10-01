# Handoff: regenerate the right-hand figure on slide 30

Run this on the Linux machine, where the raw data lives. The Windows machine
has neither the n_TOF waveform archive nor the DREAM cache, so it cannot
render this figure.

## What changed and why

Slide 30 of `ntof_athens_26/slides/ntof_athens_talk.pptx` has two figures.

- **Left, `status_two_readouts_op`: done and already in the deck.** Redrawn
  with two wide bars: n_TOF digitiser (green) alive from 2 µs, DREAM (blue)
  alive from 5 ms, and a ×2,435 arrow. The colours now match the right plot.
- **Right, `status_flash_two_chains`: code edited, PNG not yet regenerated.**
  The text overlapped the plot items. In `fig_two_chains()` in
  `mpgd26/make_flash_slides.py`:
  - The blue "this baseline carries no noise — the channel is still dead, for
    another 5 ms" call-out moved to `xytext=(2.0, 46.0)`, in the empty upper
    band. It no longer crosses the +rail line or its label.
  - The green "back under threshold 2.0 µs after its own peak" call-out moved
    to `xytext=(1.35, 19.0)`, so the two arrows no longer cross.
  - The "4 mV threshold" label moved to the right edge
    (`x=5.95, ha='right'`), off the traces.

  I only tested this against a made-up stand-in for the traces, never the
  real data. Check the result by eye. If anything still collides, nudge those
  three `xytext` / `ax.text` positions.

## Inputs needed

| Input | Path |
|---|---|
| n_TOF digitiser archive | `/media/dylan/data/x17/ntof_mm_flash/mm_224709.npz` |
| DREAM waveforms | `~/.cache/mpgd26_status/wf/*flashOff*A500*.root` (run_32) |
| Recovery numbers | `ntof_processing/mm_flash/results_709.json` (in the repo) |

## Steps

1. Get the repo to the Linux machine with the edit to
   `mpgd26/make_flash_slides.py`. It is **uncommitted** on the Windows
   machine, so commit and push it, or copy the file across.
2. From `mpgd26/`:
   ```
   ../.venv/bin/python make_flash_slides.py --only two_chains
   ```
3. The output is `mpgd26/slides/assets/img/status_flash_two_chains.png`,
   with a copy in `mpgd26/figures/`. Open it and check:
   - Neither blue nor green text crosses the +rail line, a trace, or the
     other arrow.
   - Nothing runs off the right edge of the axes.
4. Copy the PNG to the Windows machine and tell Claude. It will swap it into
   slide 30 as picture "Picture 9", keeping the same position and width and
   fitting the height to the new aspect ratio. Aspect ratio of the render is
   8.1 × 5.60 in, about 1.446.

## Things to know

- Do not run the script without `--only`. It rebuilds every deck figure,
  including ones that open the 154 MB waveform archive.
- The left figure's DREAM "5 ms" was derived as 2,435 × the digitiser's
  2.049 µs, because the run_57 metrics cache was missing on the Windows
  machine. On the Linux machine it will read the cache directly. The right
  figure does not use it, but if you re-run the left one there
  (`--only two_readouts_op`), expect the same 5 ms.
- Nothing in this work has been committed, including the pptx edit.
