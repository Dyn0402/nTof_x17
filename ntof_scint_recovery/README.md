# ntof_scint_recovery — how fast the scintillators come back after the γ-flash

A first, order-of-magnitude look (2026-10-20), asked for by a collaborator. One run, 224572
(DREAM run_79), from data already on the desktop. No tape recall.

- Report: `/media/dylan/data/x17/scint_stack/recovery/report.html` (`paths.spell('scint', 'recovery')`)
- Per-channel table: `recovery_by_channel.csv` in the same directory

```bash
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.raw_tail --jobs 7   # ~1 min, raw baseline after the flash
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.psa_tail --jobs 4   # ~12 min, PSA hits vs time since flash
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.make_report
```

## Inputs

| | what | where |
|---|---|---|
| raw | seven 450 MB stream1 head chunks of 224572, ~2 complete bunches each (15 total), every sample incl. the unsuppressed flash head | `/media/dylan/data/x17/ntof_raw_224572/` |
| PSA | our v12 reprocessing of 224572, 3018 bunches, every found pulse over 80 ms | `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/` |
| time base | `t_flash = tof_PKUP + C` | `ntof_processing/flash_timing/` |

## Three definitions of "recovered"

They give different answers, and the report keeps them apart:

1. **Pulse shape**: the median PSA FWHM is back on its 1–10 ms plateau.
2. **Baseline**: the raw baseline is within ±3 / ±1 mV of its pre-flash level. Inside the head block the
   per-bin median of samples is used, which on the plastics includes pile-up.
3. **Occupancy**: hit rate × the 160 ns production window. This is beam-induced background, not
   electronics, and it scales with intensity.

## Gotchas

- **Gain recovery is not measured.** There is no standard candle in the tail. The PSA amplitude
  median moves with the particle population.
- **The walls undershoot** by 10–17 mV from ~30 µs to ~1 ms after the flash, after the diverted flash
  decays through the baseline. This was verified on individual traces. It is on the digitized signal;
  whether the trigger sum sees it is not known.
- The plastics' head block runs 140–210 µs, because the pulse rate keeps them outside the ZS band.
  That is rate, not a slow baseline.
- `/media/dylan/data` was 100 % full (9.6 GB free) when this was written, so only the small npz
  summaries are written there.

## Data situation (checked 2026-10-20)

- **PSA hits, every run**: on EOS, `/eos/experiment/ntof/processing/official/done/run<N>.root`. The
  shape and occupancy parts can be extended to the whole campaign today.
- **Raw stream1 still on EOS disk for 32 of our runs** (~5.7 TB), under
  `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/<run>/stream1/`. Readable, but they can expire
  at any time. One file is ~20 bunches; the first 2 ms of a bunch is ~48 MB (all scintillator channels).
- **Tape only**: 224572 beyond the local chunks, and the divert-off runs 224356–360 and 224464/466.
  Recall per file with `ntof_processing/waveform_pull/lxplus/recall.sh` and read it the same day.
- `wf_pull` (287 GB on EOS) is ±5 µs around DREAM triggers (≥ ~0.5 ms), so it has no flash tail.
