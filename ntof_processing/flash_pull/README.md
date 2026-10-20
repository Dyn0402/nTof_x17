# flash_pull — raw waveforms of the γ-flash and the recovery, a sample of every beam run

**What it is.** For a few raw stream1 files of every beam run of the campaign, every
zero-suppressed block of every scintillator channel (12 detectors, 44 channels) and PKUP
that overlaps **tof 0–10 ms** or the reference window **30–32 ms**. The always-kept flash
head is included. This is the raw restricted in time, with no PSA, slim or calibration in the
cut. It exists for the scintillator recovery study (`ntof_scint_recovery/`), which needs
the flash tail pulse by pulse. The PSA output cannot provide that, and the waveform pull
(`../waveform_pull/`) starts at the first DREAM trigger, ~0.5 ms after the flash.

- Output: `/eos/experiment/ntof/data/x17/flash_tail/<run>/flash_<run>_<idx>.root`
  (+ `_provenance.json`, written last; + `_verify.json` where closure was run)
- Plan and inventory: `inputs/run_inventory_2026-10-20.csv`, and `plan.csv` on lxplus (`~/x17flash`)

## The plan (2026-10-20)

| | |
|---|---|
| runs | **299**: every X17_measurement run with ≥ 100 bunches above 10¹² ppp in the official processing (268), plus the 31 runs n_TOF never processed whose raw is > 50 GB |
| files per run | **2**, at evenly spaced positions inside the run (never the first or last file); **6** for the divert-off runs 224356–360, 224464, 224466 (the only undiverted-wall runs) and for 224572 (the first-look run) |
| bunches | ~20 per raw file → ~40 per run, ~12 k in all, parasitic and dedicated as they come |
| window | tof [0, 10 ms) and [30, 32 ms), whole ZS blocks |
| size | 75 (parasitic) – 120 (dedicated) MB of samples per bunch, ZLIB(1) ≈ 2.1× → **~0.6 TB** on EOS |
| raw read | 630 files ≈ 2.2 TB; **42 on EOS disk** (read first, they can expire), **588 recalled from tape** |

Why a sample rather than every bunch: the archive is ~700 k beam bunches and 100 TB. One
bunch has ~10⁴–10⁵ pulses in the first 10 ms, so ~40 bunches per run is ample for
amplitude against time since the flash, per channel and per condition. The cost is the
recall, not the output. If a run turns out to matter, pull more of its files with
the same tool.

Why 0–10 ms: the walls' baseline undershoot is gone by ~1 ms and the plastics' baseline by
~1.6 ms, but the plastics stay occupancy-limited to ~10 ms (first look,
`ntof_scint_recovery/`). The 30–32 ms window is a same-bunch reference far from the flash,
for "is the pulse at full amplitude again".

## Format

`blocks`: `bunch`, `det` (0–11 in the slim's order WALA..LIQD, **12 = PKUP**), `detn` (raw
ACQC channel), `tof0`, `n`, `samples` (**signed** int16). Sample *j* sits at
`tof = tof0 + j`, the PSA's own time base. `tof0 = start − 259` for a ZS block and `0` for
the flash head. Same conventions as `../waveform_pull/`. The ZS fill code −32768 equals the
negative rail.

`bunches`: one row per bunch in the file. Kept blocks/samples, `n_channels`, and, from the
official processed file, `PulseIntensity`, `psTime`, `date`, `time`, `tof_pkup_official`
(`has_official = 0`, NaN/−1 for the unprocessed runs).

`channels`: one row per (bunch, det, detn) seen. `flash_block_n` = 0 means no flash head,
so a missing channel is not mistaken for a quiet one.

Time since the flash: `t = tof − (tof_PKUP + C[det])` with C from `../flash_timing/`. The
PKUP flash is also in the product itself, for the unprocessed runs.

Always TTree: uproot ≥ 5.7 writes RNTuple from `f[name] = dict`, so `pull._put` spells
`mktree`. Products made locally and on the batch workers are identical in format.

## Run

```bash
# local, one raw file
python -m ntof_processing.flash_pull.pull 224572 --raw <raw> --idx 20 --out /tmp/flash
python -m ntof_processing.flash_pull.verify /tmp/flash/flash_224572_0020.root

# campaign (lxplus)
ntof_processing/flash_pull/lxplus/stage.sh             # code + inventory -> lxplus:~/x17flash
ssh lxplus; cd x17flash
./flash_campaign.sh plan                               # plan.csv, runs_disk.txt, tape batches
./flash_campaign.sh disk                               # submit the runs whose files are all on disk
nohup ./flash_campaign.sh tape > tape.log 2>&1 &       # recall -> submit, batch by batch
./flash_campaign.sh status
```

## Checks

- **Bit-identical** to an independent decode (`ntof_raw.parse_acqc`) on 224572 head_20:
  1181/1181 blocks.
- **Closure** (`verify.py`): every official PSA hit in the windows lies in a kept block of
  the same (bunch, det, detn). On 224572 head_20: 434,028 hits, 0 missing, 136 within 3 ns of
  a block end (the PRE_SAMPLES convention). A bunch cut short by a truncated file is
  flagged `bunches_incomplete` by the pull and skipped by the check.

## Gotchas

- `run_inventory` decides "beam" from the official files. Run 224667 could not be read
  that way, but it is > 50 GB so it is in. 224405 is in for the same reason.
- 57 early beam runs have no walls in their configuration. Their products hold PSS/LIQ/PKUP only.
- Tape recalls lapse in under a day, so `flash_campaign.sh tape` requests one batch, waits
  until every file is online, submits, and requests the next. A job whose file went offline
  exits 5. Re-run `tape` (resumable: `*.requested` / `*.submitted` files under `campaign/`).
