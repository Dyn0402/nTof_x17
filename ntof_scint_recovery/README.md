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
| raw | seven 450 MB stream1 head chunks of 224572, ~2 complete bunches each (14 complete, 11 dedicated), every sample incl. the unsuppressed flash head | `/media/dylan/data/x17/ntof_raw_224572/` |
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
- **The walls undershoot** by 10–19 mV from ~30 µs to ~1 ms after the flash, after the diverted flash
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

## The flash_tail analysis (2026-10-20)

The first look above used one run. The real study runs on the raw-waveform dataset
`/eos/experiment/ntof/data/x17/flash_tail` (`ntof_processing/flash_pull`), with every result split by beam
intensity. Plan, state and findings: **[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)**.
Slide note: <https://dylan-neff.web.cern.ch/notes/scint-flash-recovery.html>.

```bash
# on lxplus, ~/x17rec (code rsynced from here): per-run reductions, written to EOS
condor_submit summary.sub   -a 'runs=runs.txt'    # flash_summary.py -> flash_tail_summary/
condor_submit pulsehist.sub -a 'runs=runs.txt'    # pulses.py --hist -> flash_tail_pulsehist/
# desktop
rsync -a lxplus:/eos/experiment/ntof/data/x17/flash_tail_summary/ /media/dylan/data/x17/scint_stack/recovery/summaries/
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.quick
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.examples <a local flash_*.root>
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.make_recovery_deck
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    /media/dylan/data/x17/scint_stack/recovery/deck/scint-flash-recovery.html --slug scint-flash-recovery --force --deploy
```

## The full analysis: reconstruction, bias, resolution (2026-10-21) — the headline

From when after the flash a pulse is reconstructed accurately, and its bias and resolution in amplitude,
energy and time. Record: `ANALYSIS_PLAN.md` "Round 5". Slide note:
<https://dylan-neff.web.cern.ch/notes/scint-flash-reconstruction.html>.

```bash
# lxplus ~/x17rec (code rsynced from here)
condor_submit inject.sub                       # inject.py per production run -> flash_tail_inject/
condor_submit g4mix.sub                        # g4_mix.py over full_sim/neutrons_fullrange -> flash_tail_g4mix/
python3 -m ntof_scint_recovery.inject --merge /eos/.../flash_tail_inject/inject_*.npz --out <merged>
python3 -m ntof_scint_recovery.g4_mix --merge /eos/.../flash_tail_g4mix/g4mix_*.npz --out <merged>
# desktop: merged files to out_dir('inject')/inject.npz and out_dir('g4mix')/g4mix_all.npz, pulsehist_v3/ synced
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.inject_analysis
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.g4_mix_analysis
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.gain_ref
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.headline        # headline.csv, performance.csv, report.html
PYTHONPATH=. .venv/bin/python -m ntof_scint_recovery.make_reco_deck
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    /media/dylan/data/x17/scint_stack/recovery/deck/scint-flash-reconstruction.html --slug scint-flash-reconstruction --force --deploy
```

`templates.npz` (the family templates) is force-added to git despite `*.npz` in `.gitignore`: the injection and
the fit both need it.
