# n_TOF micromegas signals — where they are, and what survives

**2026-08-09.** Which n_TOF runs have a micromegas channel in the n_TOF DAQ
(not the DREAM readout), under what name, and whether the waveforms still
exist. Tools and data behind this note: `mm_signals/`.

## Answer

Two distinct detector names, two distinct epochs:

| Name | Runs | Dates | Where | Processed? | Raw waveforms today |
|---|---|---|---|---|---|
| **MMA** | 224708–224710 (3) | 2026-08-09 | EAR2 `X17_measurement` | not yet | all 3, on the DAQ disk |
| **MMA / MMB** | 224297–224339 (43) | 2026-07-05 → 07-09 | EAR2 `X17_measurement` | configured, **never processed** | 3 of 43 runs |
| **MGAS** | 222989–223065 (76) | 2026-02-24 → 03-02 | EAR2 `BeamCommissioning` | configured, **never processed** | 2 of 76 runs |

There is **no MGAS run after 2026-03-02**. The channel came back on 2026-08-09
at 16:56 (see below); between 224340 and 224707 there is no MM channel at all.

## MMA is Det A strip 32, cable Y8 (from 2026-08-09)

Stated by Dylan on 2026-08-09 for the August runs: the `MMA` input carries
**strip 32 of MX17 detector A, cable Y8** — a single strip, not the mesh. The
August configuration also runs a **10× coarser input range** than July:
`fullScalemV` 5043.79 against 504.15, i.e. 76.96 µV per count, with the
baseline parked near zero (measured −186 counts) instead of +200 mV.

Present in **224708** (16:56–17:05), **224709** (17:05–19:38, 1.5 TB, 344
files) and **224710** (19:51–20:01); gone again by 224711. 224709 contains a
**detector-A-only drift × amplification scan**, 17:10–19:31, 25 plateaus of
~5.7 min (drift 700 V × 14 amplification points 565→500 V; drift 600 and 500 V
× 5 points each 570→530 V). Because only A moves, this scan — unlike the July
ones — attributes the response to a named chamber.

## MMA / MMB (July 2026)

- `MMA` and `MMB`, one channel each (`detectorNumber` 1), S014 digitiser,
  1 GS/s, ~504 mV full scale, crate 4 slot 3 (MMA) and slot 5 (MMB).
- **224297–224329** (33 runs, Jul 5–9): MMA **and** MMB, alongside
  `WALL`, `RAMP`, `SILI`, `PKUP`.
- **224330–224339** (10 runs, Jul 9): MMB only, alongside `WALA–D`, `SILI`,
  `PKUP`. These are 3–13 bunch runs — configuration pokes, not data.
- Zero-suppression as stored in `DAQsettings`: run 224297 sits at −374 mV
  (effectively off); from 224298 onward MMB is at **−4.0 mV** (−2.0 mV in
  224300) and MMA at **0.0 mV**, unchanged for the rest of the campaign.

### Which channel is live flips between runs — check, do not assume

The stored thresholds do **not** predict which channel carries signal. Measured
from the raw, three chunks per run (first / middle / late), 20 bunches each,
counting zero-suppressed blocks beyond the mandatory flash block:

| run | MMA | MMB | live channel |
|---|---|---|---|
| 224302 | 2, 3, 2 | **400, 332, 348** | **MMB** |
| 224325 | **918, 910, 218** | 0, 0, 3 | **MMA** |
| 224327 | **1496, 1909, 1116** | 0, 0, 0 | **MMA** |

Both are negative-going on a ~25 700 ADC baseline. The live channel swings
thousands of counts below it (224302 MMB down to 9089; 224325 MMA reaches
−17949, i.e. past the signed-int16 range — expect saturation there), the
silent one never leaves baseline. Since `DAQsettings` is identical across all
three runs, something changed in the *cabling or the detector connection*
between Jul 6 and Jul 8, or the stored threshold is not what was applied.
Whichever it is, **the channel name is not a stable label for the detector** —
identify the live channel per run from the raw before using it.
- The longest runs are 224325 (5277 bunches), 224324 (4131), 224328 (3611),
  224318 (3566), 224320 (3544).

**They were digitised but not processed.** No `MMA`/`MMB` tree exists in any
file under `/eos/experiment/ntof/processing/official/done/`, and the
`UserInput` stored in the `history` object of run 224297 contains no `MM`
entry at all — the PSA was never told those channels existed. So there are no
hits, and there never will be without a reprocess.

That is not a loss as long as the raw survives, because the processed files
strip waveforms anyway — the `waveform` branch in an official file is empty.
**Raw stream1 is the only waveform source for these channels, in every epoch.**
`NTOF_REPROCESSING_REQUEST_2026-08-08.md` covers how a reprocess is requested
and what it costs.

## Do the waveforms still exist?

**Secured 2026-08-09** — the three surviving runs are copied, verified and no
longer at the mercy of the DAQ staging policy:

    /eos/experiment/ntof/data/x17/mm_raw_2026-07/<run>/stream1/

591 files, 24 G, every one checked against the source by length and EOS adler32
(`591 ok, 0 bad`), with a `README.md` alongside. The copies decode: `MMA`,
`MMB`, `WALL`, `RAMP`, `SILI`, `PKUP` all read back through `ntof_raw.py`.

Original state, checked 2026-08-09 against
`/eos/experiment/ntof/DAQ/2026/EAR2/<measurement>/<run>/stream1/`:

| Run | Date | Bunches | stream1 files on disk |
|---|---|---|---|
| **224302** | 2026-07-06 | 3067 | **154** |
| **224325** | 2026-07-08 | 5277 | **264** |
| **224327** | 2026-07-09 | 3444 | **173** |
| all other 40 MM runs | Jul 5–9 | — | 0 |
| **223009** | 2026-02-25 | — | **132** (MGAS) |
| **223011** | 2026-02-26 | — | **142** (MGAS) |
| all other 74 MGAS runs | Feb–Mar | — | 0 |

So the useful physics runs are covered — 224302, 224325 and 224327 are three of
the longest MMA+MMB runs, ~11 800 bunches between them, and each carries a live
micromegas channel — but **the other 40 need a recall before anything can be
done with them**.

Two cautions on the recall:

- The empty directories are empty in the EOS *namespace*, not just off disk:
  `ls` of `224297/stream1` returns nothing, and the directory mtime says the
  files went on 2026-07-19. `eos ls -y` on a surviving run reports `d2::t0`
  — two disk replicas, **no tape replica visible from this instance**. We
  cannot confirm the archive copy ourselves; n_TOF hold it. Ask them to
  confirm the runs are on tape *before* planning around a recall.
- Disk staging here is transient and selective. The three surviving MM runs
  have been staged since at least 2026-07-28 (they match
  `eos_stream1_inventory_2026-07-28.txt` file-for-file), but nothing guarantees
  they stay. If they matter, copy them out.

## What was searched

- **910 runs** started since 2026-06-28 (911 directory entries — 923153 is
  filed under two LAB directory names), from the listing of
  `/eos/experiment/ntof/DAQ/2026/{EAR1,EAR2,EAR3,LAB}/<measurement>/<run>/`:
  X17_measurement 432, MArEX 300, B10_BRAINS 78, 235U_RePPAC 41,
  27STED_25Mg_natCu 23, STAR_Commissioning 20, BRAINS_Aug2026 8, DAQTEST 7,
  Enhanced LaBr Test 1.
- **749 of them** have an official processed file; their `DAQsettings`
  detector lists are in `mm_signals/detector_census_since_2026-06-28.tsv`.
- The remaining **3997 processed runs in the archive** were scanned the same
  way to find MGAS anywhere — that is where the February epoch came from.
- The **71 unprocessed X17 runs** were checked from the raw instead
  (`mm_signals/detectors_from_raw.py`). The 45 with staged raw — 224573
  through today's 224705 — carry only `WAL*`, `PSS*`, `LIQ*`, `RMP*`, `SILI`,
  `PKUP`. No MM.

Detector inventory of the window, for orientation:

| Measurement | Runs | Detectors |
|---|---|---|
| EAR2 X17_measurement | 359 | PKUP, SILI, PSSA–D, WALA–D, WALL, LIQA–D/S/T/1/2, RAMP, RMPA/C, **MMA, MMB** |
| EAR1 MArEX | 294 | DIC6, FC-B, FC-U, PKUP, PPAC, PPAN, STAR, TOF2 |
| EAR1 B10_BRAINS | 55 | DIC6, FC-U, FC-B, PKUP, LABR, PPAC |
| EAR1 235U_RePPAC | 16 | DIC6, PKUP, PPAC, PPAN, TOF2, STAR, FC-B/U |
| EAR2 27STED_25Mg_natCu | 23 | STED, PKUP, SILI |

## What has been done with it

- **`mm_flash/`** — the analysis and the published note
  (<https://dylan-neff.web.cern.ch/notes/ntof-micromegas-gamma-flash.html>).
  Headline: the flash puts **668 pC on one strip** per dedicated pulse at the
  production point, ~1 000× the DREAM CSA full scale; gain cannot fix it; the
  Q/I_feedback drain time (7–74 ms) is the millisecond DREAM dead time. The July
  runs separately show the **chamber** recovers in microseconds, so the dead time
  is a front-end property.
- **The cross-check against the HV supply current**
  (`ntof_july_analysis/flash_charge/`) agrees on *shape* to 5.7 % across all 25
  plateaus but leaves a **constant factor 3.5** on the absolute scale. That is
  the open question, briefed in
  [`mm_flash/HANDOFF_CHARGE_COMPARISON_2026-08-11.md`](mm_flash/HANDOFF_CHARGE_COMPARISON_2026-08-11.md).
- **Strip position**: `mx17_m1_map.csv` puts "strip 32 of cable Y8" at
  y = 374.4 mm (connector Y8 ch 32) or y = 25.0 mm (global strip 32) — **both
  5–7 mm inside a Y passivation edge**, i.e. at the chamber periphery. Which one
  is right should come from the cabling record; they are 350 mm apart.

## Two traps when scanning for the channel

Both of these produced a confident, wrong "there is no MM channel" on
2026-08-09. Neither is visible from the output — the channel just silently
isn't in the list.

1. **`ntof_raw.parse_modh` drops padded names.** It skips any record whose
   4-byte name fails `isalnum()`, and a three-character name is padded, so
   `MMA\x00` is discarded while `WALA` survives. MODH said 51 channels for
   224709; the real number is 52. Re-implement the walk with
   `.strip(' \t\r\n\x00')` *before* the test — `mm_signals/modh_channels.py`
   does. The same padding is why `--blocks MMB` matched nothing in
   `detectors_from_raw.py` until it was fixed.
2. **A head chunk may not hold one whole event.** The July runs were ~14 MB per
   bunch, so a 120 MB head covered several. The August 52-channel runs are
   ~500 MB per bunch, so a 120 MB head covers *part of one* — and the channel
   you want may simply not have appeared yet. `detectors_from_raw.py` prints
   `nev=`; if it is 1, the list is a lower bound, not an inventory. Prefer MODH
   (with the fix) for the channel list, and use ACQC only to confirm.

## How to redo this

On lxplus (`ssh -K lxplus` — without the AFS token every EOS call fails):

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

# 1. runs started in a date window, with EAR + measurement
D=/eos/experiment/ntof/DAQ/2026
for e in EAR1 EAR2 EAR3 LAB; do for m in "$D/$e"/*; do
  ls -l --time-style=+%Y-%m-%d "$m" | awk -v e=$e -v m="$(basename "$m")" \
    '$6>="2026-06-28" {print $6, e, m, $7}'
done; done

# 2. detector names of the processed ones (~1 s/run, 8 threads)
python3 scan_processed_detectors.py runs.txt > census.tsv

# 3. detector names of the unprocessed ones, from the raw
head -c 120000000 $D/EAR2/X17_measurement/224302/stream1/run224302_0_s1.raw.finished > /tmp/h.bin
python3 detectors_from_raw.py /tmp/h.bin --blocks MMB
```

`detectors_from_raw.py` needs `ntof_raw.py` from
`nTof_x17_DAQ/stream1_monitor` (it adds that path itself).

## What this does not rule out

- **A micromegas under a name I did not recognise.** I matched on the literal
  strings `MGAS` and `MM*`. The full census is in the TSV — if n_TOF called a
  chamber something else (a bare `M2` appears in two February runs, 8 channels,
  5 V full scale, unidentified), it would not have flagged.
- **The 26 unprocessed X17 runs from 224451–224565**, whose raw is already
  gone and which have no processed file. Nothing can be said about their
  channel list from here. They sit in the middle of the July campaign, whose
  processed neighbours on both sides are MM-free, so they are very unlikely to
  be MM runs — but it is an inference, not a check.
- **90 non-X17 unprocessed runs** (MArEX, B10_BRAINS, 235U_RePPAC,
  STAR_Commissioning, BRAINS, DAQTEST). Same argument: every processed
  neighbour in those measurements is MM-free.
- **Two unreadable files**, 224405 and 224667 (zero length in `done/`).
- **Whether MMA/MMB were pointed at anything useful.** This note establishes
  that the channels exist, are configured sanely and contain real
  zero-suppressed waveforms. It says nothing about what was connected to them,
  what the gas or HV was, or whether the signal is physics. Nothing here has
  been correlated with the DREAM data.
