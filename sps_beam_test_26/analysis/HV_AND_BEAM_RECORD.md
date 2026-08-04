# HV and SPS beam records — what survives, what does not

We ran parasitically: our HV was the only thing we controlled, and the SPS beam
and H4 line state were the only things that explain rate. Both records have
holes. This file says exactly where, so that no analysis quietly assumes it has
a trace it does not have.

Archived copies (pulled 2026-08-02, before further pruning):

```
/media/dylan/data/x17/sps_run53_det4_check/records/
├── scan_logs/    our det4 scan + hold logs, and run_61's dream_daq.log
├── hv_monitor/   every hv_monitor.csv that still exists anywhere
└── beam/         SPS spill + intensity CSVs and 5 ms profiles
    ├── backfill_nxcals/   ← THE AUTHORITY. Full period, recovered from NXCALS
    │                        2026-08-02 via mx17-daq. Includes h4_tax_*.csv.
    └── from_mx17_daq/     daq's own watcher output, kept as an independent
                           cross-check of the spill columns it shares.
```

> **Read `backfill_nxcals/`, not the files above it or banco's live logs.** It is
> a strict superset of both: same schema as banco (17 spill columns), plus the
> 24 h that banco lost, plus the TAX log neither of them ever had. The loose
> files at the top of `beam/` are the pre-backfill pull, kept only so the
> before/after is auditable. `analysis/beam_record_coverage.py` prints coverage
> for any of these directories, so they can be compared directly.

---

## 1. ⚠ The SPS beam record stopped on 2026-08-01 at 18:49:25

**Nothing has been recorded since.** No `sps_profile_2026-08-02_*` files exist
at all, `sps_spill_2026-08-01.csv`'s last row is `18:49:25`, and no monitor
process is running on banco.

That means **no beam-condition record for**:

- run_58 from 18:49 (the tail of the 25° drift ladder)
- **run_59** — the 41 GB long run
- **run_60** — the entire 12-hour overnight
- **run_61** — everything today: both resist scans, the drift scan, both mount angles
- **and tonight's planned overnight high-statistics run, unless it is restarted**

Likely cause: Kerberos. `beam_intensity_controller.py` authenticates with the
user's ticket, renews with `kinit -R` every 4 h, and its own docstring says that
past the ticket's *renewable* life a manual `kinit dneff@CERN.CH` reseed is
needed. 18:49 Saturday is consistent with the renewable life running out.

**To restart** (before tonight's run — this is the single most time-critical
item in this directory):

```bash
ssh banco_cern
kinit dneff@CERN.CH                  # reseed, not kinit -R
klist                                # confirm renewable life
# then relaunch the watcher/controller under DAQ_Control_Dream_Beam/beam_monitor/
```

### State as of 2026-08-02 ~19:40 — running again, gap NOT filled

- The **lxplus-side watcher is alive again** from `18:50:12` and EOS is growing.
- **`beam_bridge.py`** on banco (tmux `beam_watcher`, restarted 18:48:36) is a
  pure EOS→banco *mirror*, not an NXCALS client — its log reads
  `11 historical file(s) to catch up … catch-up complete`. That is why 08-02
  files appeared. It is forward-only and it auto-pulls anything new on EOS,
  **including a backfill**, so nothing has to be synced by hand afterwards.
- **The gap is identical on banco and on EOS**: `sps_spill_2026-08-01.csv` ends
  `18:49:25` in both, `sps_spill_2026-08-02.csv` starts `18:50:12` in both, and
  EOS's newest profile is still `sps_profile_2026-08-01_18.jsonl.gz`. The lxplus
  watcher died at the same instant as banco's — one Kerberos ticket, both ends.

**Missing window: 2026-08-01 18:49:25 → 2026-08-02 18:50:12** (~24 h). Covers
run_58, run_59 (`detE_long`), all 24 sub-runs of run_60, and every data-taking
sub-run of run_61. Only `meshscan_m100V`'s last ~6 min are inside the restored
window, which is exactly when the linac was already down.

### ⛔ Recovering the gap — the lxplus recipe below does NOT work; use mx17-daq

**Superseded 2026-08-02.** Two things are wrong with it, and both were only
found by trying:

1. **You cannot run it.** `/eos/user/a/akallits/beam_monitor` and
   `/eos/user/a/akallits/nxcals_venv` are both mode `700`, owned by `akallits`.
   From lxplus as `dneff` they are `Operation not permitted` — not readable, not
   writable. The recipe is written for its author's account, not ours.
2. **You do not need lxplus.** `mx17-daq` is TN-trusted and reaches NXCALS
   directly (`cs-ccr-nxcals5-8`), has a working `~/venvs/nxcals`, and — this is
   the part nobody noticed — **its own beam watcher never went down**. banco's
   feed and daq's feed are two independent NXCALS clients; only the lxplus one
   died. daq has continuous 500-cycle/hour coverage straight through the gap.

What was actually done is in §5. The original text is kept below because the
`--what scalars` / `--include-today` semantics still apply verbatim.

### Original (unrunnable as written) — must run on lxplus

NXCALS keeps the history, and `sps_monitor/backfill_nxcals.py` exists for this.
**It cannot be run from here or from banco**: banco has no NXCALS path
(`nxcals-api.cern.ch` does not even resolve there; its venv `~/venvs/nxcals` is
for the separate n_TOF-intensity watcher), and this laptop's key is refused by
lxplus. It needs an lxplus session as the ticket owner.

Split it in two, because only the second half races the live watcher:

```bash
# --- 1. the 08-01 tail. SAFE NOW: the script refuses to touch today's files,
#        and 08-01 is not today, so the watcher can keep running.
SPS_BEAM_LOG_DIR=/eos/user/a/akallits/beam_monitor SPS_SPILL_LOG_DIR=/eos/user/a/akallits/beam_monitor /eos/user/a/akallits/nxcals_venv/bin/python sps_monitor/backfill_nxcals.py     --start 2026-08-01 --end 2026-08-01 --what scalars

# --- 2. today. Stop the lxplus watcher first — --include-today would otherwise
#        read-modify-write the file the watcher is appending to.
#        (banco's beam_bridge can keep running; it only ever reads EOS.)
SPS_BEAM_LOG_DIR=/eos/user/a/akallits/beam_monitor SPS_SPILL_LOG_DIR=/eos/user/a/akallits/beam_monitor /eos/user/a/akallits/nxcals_venv/bin/python sps_monitor/backfill_nxcals.py     --start 2026-08-02 --end 2026-08-02 --include-today --what scalars
# then restart the watcher

# --- 3. optional, slow (~1-2 h): the 5 ms intra-cycle profiles
#        --what profiles ; chunked by hour and resumable, skips hours already there
```

`--what scalars` is the high-value part — per-spill intensity, spill structure
and the H4 counters — and takes minutes. Re-running is safe: every file is
rewritten from the union of what was there and what NXCALS returned,
deduplicated on cycle timestamp.

Once it lands on EOS, banco's `beam_bridge` pulls it down on its own; then
re-run the archive step to bring it here:

```bash
ssh banco_cern 'cd .../beam_monitor/logs && tar cf - sps_spill_2026-08-0*.csv \
  beam_intensity_2026-08-0*.csv sps_profile_2026-08-0*.jsonl.gz' \
  | tar xf - -C /media/dylan/data/x17/sps_run53_det4_check/records/beam/
```

### What the record contains when it is running

`sps_spill_<date>.csv`, one row per SPS cycle:

| column | meaning |
|---|---|
| `destination` | **the beam-stopper / barrier proxy** — `FTARGET` = extraction to the North Area, `SPS_DUMP` = nothing coming, `HIRADMAT` = elsewhere |
| `extracted_e10` | protons extracted, ×10¹⁰ |
| `spill_len_ms`, `duty_factor`, `extraction_time_ms`, `beam_out_time_ms`, `cycle_len_ms` | spill structure |
| `h4_bend_027_a`, `h4_bend_309_a`, `h4_bend_706_a` | **H4 line bending-magnet currents** |
| `h4_gif_001` … `h4_gif_004`, `h4_hna162_005` | **H4 GIF / attenuator states** |

The `h4_*` columns are populated on `FTARGET` cycles only (4650 of 4637
FTARGET rows on 08-01) — which is the useful case anyway.

> **What this record can and cannot tell you.** It is an *upstream* record: SPS
> extraction and the H4 line's own magnet settings. Checked on 08-01, it shows
> `FTARGET` continuously from 00:00 to 18:49 at a steady ~1380×10¹⁰ with
> `h4_bend_027_a`/`_309_a`/`_706_a` pinned at 280.0/478.0/216.4 A and **no
> off-window anywhere in the day** — even though we took two accesses. A zone
> access is made with the H4 beam stopper, which is **not** among the logged
> variables, so the record cannot date accesses and cannot by itself tell you
> whether beam was reaching the detectors. Use it for spill structure, duty
> factor and delivered intensity; use the DAQ gaps and pedestal runs for
> access times.
>
> **⛔ SUPERSEDED 2026-08-02 — the accesses *are* dated now.** The paragraph
> above is right about *banco's* record and wrong about the underlying data. The
> beam stopper is `XTAX_022_023:POSITION_MEAS` in NXCALS; banco's fork of the
> monitor simply never polled it, which is why its H4 story rests on the BEND
> currents — and those stay energised straight through an access, which is
> exactly why no off-window appears. mx17-daq's fork *has* always logged it. It
> is now backfilled for the whole period into `backfill_nxcals/h4_tax_*.csv`
> (465,967 samples, 20 days). Use `analysis/tax_windows.py` for the windows and
> stop inferring access times from DAQ gaps. See §5.

`sps_profile_<date>_<hh>.jsonl.gz` holds the 5 ms-sampled intensity profile of
every cycle. ~3 MB/hour; only the hours around det4's data were pulled locally.
`beam_intensity_<date>.csv` is the per-pulse summary.

### Coverage of what we do have

| file | first row | last row | cycles |
|---|---|---|---|
| `sps_spill_2026-07-30.csv` | 00:00:06 | 23:58:10 | 9849 |
| `sps_spill_2026-07-31.csv` | 00:00:06 | 23:58:54 | 11991 |
| `sps_spill_2026-08-01.csv` | 00:00:06 | **18:49:25** | 9301 |

So Friday's install night and Saturday up to 18:49 are covered; Saturday's flat
and 25° scans are inside that window and **can** be beam-corrected. Nothing
after is.

---

## 2. HV monitor CSVs — recovered, with two windows genuinely gone

`hv_monitor.csv` is written per sub-run into the run directory. Three
independent mechanisms have damaged it; only the second destroys data.

**Current state: 88 files, every sub-run of runs 53–61, archived flat as
`records/hv_monitor/run_NN__<subrun>.csv`.**

### 2a. Runs 53–57 were pruned from banco — but **nothing is actually lost**

Their run directories were deleted from banco's local disk to make room. They
are **complete on EOS**, verified 2026-08-02:

```
root://eospublic.cern.ch//eos/experiment/ntof/data/x17/p2_sps_july/runs/
```

Every run 50–61 is there, and every sub-run carries `hv_monitor.csv`,
`raw_daq_data/`, `decoded_root/`, `hits_root/`, `combined_hits_root/`,
`.subrun_complete` **and `run_config.json`** — so the per-run DAQ configs that
were pruned locally are recoverable too. `backup_watcher.py` syncs to EOS
*before* pruning and only deletes what it has verified landed, which is why.

Recovered locally into `records/hv_monitor/` (see the coverage table in §2d,
which is written against the post-recovery state). To redo it:

```bash
ssh banco_cern
export PATH=/local/home/banco/bin:$PATH        # xrdcp/xrdfs live in ~/bin
klist                                          # needs a live ticket
xrdfs root://eospublic.cern.ch ls /eos/experiment/ntof/data/x17/p2_sps_july/runs
xrdcp root://eospublic.cern.ch//eos/.../runs/run_57/meshscan_m90V/hv_monitor.csv .
```

**Also archived independently:** `detE_resist_scan.log` and `detE_scan.log`
record every det4 HV point with its commanded value, its verified readback and
the P2 sub-run it landed in. Those are the authority for *our* channels even
where the monitor trace exists, because they carry the sub-run join.

> ⚠ **EOS's `run_58` directory is polluted.** It contains its own
> `operating_00…02`, **plus run_59's `detE_long_00/01` and all 24 of run_60's
> `overnight_*`.** The files are byte-identical duplicates of the ones in
> `run_59/` and `run_60/` (checked on the HV CSVs), so nothing is corrupt — but
> anything that pulls "run_58" from EOS gets three runs mixed together under
> one name. Pull run_59 and run_60 from *their own* directories.

The second and third backup destinations in `config/backup_config.json` —
`/eos/project/s/salsachip/Data/T2_tests/P2_SPS_Dream_Data/` (banco's) and
`/eos/user/a/akallits/P2_SPS_backup_temp/` — were not checked.

### 2b. Re-running a sub-run under the same name overwrites its HV log

This is the "HV csvs being overwritten" problem, and the mechanism is specific:
`hv_monitor.csv` is keyed on `<run>/<sub-run>/`, so when run_61 was restarted
and re-entered a sub-run **name it had already used**, the file was rewritten
from scratch. The `.fdf` data survives — those carry a `datrun_<date>_<HH>H<MM>`
stamp in the filename, so both passes coexist — but the HV trace does not.

Two sub-runs hit, both on 2026-08-02:

| sub-run | first pass | surviving hv_monitor.csv | lost |
|---|---|---|---|
| `run_61/meshscan_m00V` | 12:14–12:45 | **15:40:13 → 15:44:14** (240 rows, 92 kB) | the 12:14–12:45 pass |
| `run_61/meshscan_m30V` | 13:46–14:00 | **16:06:56 → 16:29:32** (1348 rows) | the 13:46–14:00 pass |

The EOS copy of `m00V` is the same 240 rows, so EOS does not help here — the
overwrite happened before the backup.

Consequence: the drift scan's last three points (280/210/140 V, 13:48–13:58)
and the `m00V` resist creep have no monitored trace. Both are still recoverable
from `det4_drift_scan.log`, which logs vmon *and* imon per point.

### 2c. Pedestal runs share one HV trace between them

A separate collision, in the pedestal directories. Five pedestal runs carry an
`hv_monitor.csv` that is a byte-identical copy of an earlier run's — distinct
inodes, identical size, identical mtime to the nanosecond:

| pedestal dir | its hv_monitor.csv actually belongs to |
|---|---|
| `pedestals_08-01-26_14-00-23` | `…_12-16-39` |
| `pedestals_08-01-26_17-52-11` | `…_16-20-37` |
| `pedestals_08-01-26_18-12-15` | `…_16-20-37` |
| `pedestals_08-01-26_21-18-04` | `…_21-11-02` |
| `pedestals_08-02-26_15-19-20` | `…_15-02-43` |

**Never read HV out of a pedestal directory** — the timestamp on the directory
does not match the timestamps inside the file.

### 2d. Actual HV monitor coverage, det4 era

After the EOS recovery. Contiguous blocks; a gap of more than 3 minutes starts
a new row. The gaps between blocks are the real ones — accesses, time between
runs, and the two overwritten windows of §2b.

| from | to | rows | runs |
|---|---|---:|---|
| 2026-08-01 12:57:51 | 2026-08-01 14:00:31 | 3684 | run_53, run_54 (2 sub-runs) |
| 2026-08-01 14:13:43 | 2026-08-01 16:00:25 | 6225 | run_55, run_56 (10 sub-runs) |
| 2026-08-01 16:35:35 | 2026-08-01 19:57:12 | 11806 | run_57, run_58 (12 sub-runs) |
| 2026-08-01 20:00:56 | 2026-08-01 20:54:30 | 3182 | run_59 (2 sub-runs) |
| 2026-08-01 21:19:14 | 2026-08-02 09:26:40 | 43127 | run_60 (24 sub-runs) |
| 2026-08-02 12:06:02 | 2026-08-02 12:07:42 | 100 | run_59 (1 sub-runs) |
| 2026-08-02 12:45:12 | 2026-08-02 13:45:58 | 3612 | run_61 (2 sub-runs) |
| 2026-08-02 15:40:13 | 2026-08-02 15:44:14 | 240 | run_61 (1 sub-runs) |
| 2026-08-02 16:06:56 | 2026-08-02 18:56:56 | 10063 | run_61 (8 sub-runs) |

Gaps worth naming:

- **08-02 12:14–12:45** and **13:46–14:00** are the two overwritten windows.
  det4's own voltages for both survive in `det4_drift_scan.log`, which logs
  vmon and imon per point.
- **08-02 09:26–12:06** is the 11:00 access; det4's HV through it is in
  `detE_hold.log` to 11:26.
- **08-01 20:54–21:19** and the shorter inter-run gaps are simply no run.

Channels: **drift = card 8 ch 8, resist = card 12 ch 2.** Confirmed 2026-08-01;
an earlier guess of 12:2 / 12:3 was wrong on both.

---

## 3. End-of-run power-offs — three known, all ours

`hv_control.power_off_hvs()` sweeps every channel on every card. It ignores
`DET_HV`, `included_detectors` and our monitor-only `None` setpoints, so any
run that ends with `power_off_hv_at_end=True` kills det4 too.
`power_off_hv_at_end` now defaults to `False`, but banco's
`quick_scripts/make_*.py` set it `True` explicitly, so generated configs still
do it.

| when | what it hit | how it showed up |
|---|---|---|
| 08-01 14:57:41 | pre-rotation drift scan, at its 5th point | `attempt 1: resist reads 1.25 (want 550.0) — refusing`, then abort |
| 08-02 11:23:48 | the fixed-point hold | `*** HV OFF (re-arm #1)`; auto-recovered by 11:26 |
| 08-02 14:00:46 | drift scan, at its last point (70 V) | logged as `drift channel 8:8 powered OFF (trip?)` — **not a trip** |

The 14:00:46 one is worth being explicit about, because it is written up as a
trip in `rot15_ArCF4iso_88-10-2__run61_1214-1400/README.md` and it was not one: `dream_daq.log`
records `Run finished normally` at **14:00:42**, four seconds before, and the
scan log shows **both** channels collapsing together — drift 23.5 V *and*
resist 642.5 V, down from a held 750 V. A trip on 8:8 cannot pull 12:2 down.

Durable fix, still not applied: patch `power_off_hvs()` to skip 8:8 and 12:2.
It needs an `hv_control` server restart, so it has to happen between runs.

---

## 4. What still needs collecting

- [x] ~~Restart the SPS beam monitor~~ — back up 2026-08-02 18:50; forward
      recording is healthy, so tonight's run *will* have a beam record.
- [x] ~~**Backfill the 08-01 18:49 → 08-02 18:50 gap from NXCALS**~~ — done
      2026-08-02 from **mx17-daq**, not lxplus (§5). 11,005 cycles recovered,
      4,297 of them `FTARGET`. Both days now run 00:00→23:59 with no gap over
      two minutes.
- [x] ~~Pull runs 53–57 `hv_monitor.csv` from EOS~~ — done 2026-08-02, all
      62 sub-runs of runs 53–61 archived under `records/hv_monitor/`.
- [ ] Decode **FEU3** for run_58, run_59 (`detE_long`, 41 GB) and run_60
      (overnight) — the data exists and has never been touched. Pair against
      **FEU1**. 309 GB of det4-era raw is still on banco; the rest is on EOS.
- [ ] Recover the per-run `run_config.json` for runs 53–58 from EOS — they were
      pruned locally and are the only record of each run's sample count, ZS
      settings and FEU list.
- [ ] Keep pulling run_61's later sub-runs as banco prunes.
- [ ] After tonight: archive the flat-mount overnight run's HV and beam records
      immediately rather than after the fact.
- [ ] Re-run `analysis/push_beam_record_to_banco.sh` **after midnight** to land
      the four files the bridge was still overwriting as "today" (§5).
- [ ] Teach banco's own spill monitor to poll `XTAX_022_023:POSITION_MEAS`, so
      the barrier is recorded live instead of only in backfill (§5).

---

## 5. The 2026-08-02 NXCALS backfill — what was recovered and from where

Done on **mx17-daq**, because the §1 lxplus recipe cannot be run by us and is
not needed. Three facts made this easy, none of which were known when §1 was
written:

1. banco's beam feed is **not** its own NXCALS client. It is `beam_bridge.py`,
   a pure `xrdcp` mirror of `/eos/user/a/akallits/beam_monitor`, which an
   lxplus watcher publishes. When that watcher's Kerberos ticket expired, banco
   had no independent way to notice or recover.
2. `mx17-daq` runs a **second, independent** NXCALS client (`beam_watcher.py`,
   up since 07-30, Spark session to `cs-ccr-nxcals5-8`). **It never went down.**
   Its logs cover the whole gap at the usual ~500 cycles/hour.
3. daq is TN-trusted and has a working `~/venvs/nxcals`, so banco's own
   backfill scripts run there unmodified against a staged copy of banco's
   `sps_monitor/` + `beam_monitor/`, writing to a staging dir that no live
   watcher owns. That is what produced everything below.

### What came back

| dataset | before | after | recovered |
|---|---|---|---|
| `sps_spill_*` (per SPS cycle, 17 cols) | 215,529 cycles | 227,853 | **+12,324**, of which 11,005 in the 24 h gap (4,297 `FTARGET`) |
| `beam_intensity_*` (`SPSQC:MEAN_SPILL_INTENSITY`) | 99,413 pts | 105,632 | **+6,219** |
| `h4_tax_*` (H4 beam stopper) | **did not exist** | 465,967 samples | **all 20 days** |
| `sps_profile_*` (5 ms intra-cycle) | 247 hour-files | 445 | **+198 hours**, 1.2 GB |

Plus **539 spill rows blank-filled** — see below, that is a distinct repair.

**The profile set is now complete in the only sense that matters.** 230 hours
were absent; 198 came back and the other 32 returned "no extracting cycles".
That is not a failure — `PROFILE_ARCHIVE_SCOPE` is `extracted`, so an hour with
no `FTARGET` cycle has nothing to archive. Cross-checked against the spill
record: **every one of the 35 hours still without a profile file has zero
`FTARGET` cycles in `sps_spill_*`.** Every hour that had beam has a profile.
Backfilled files are structurally identical to the watcher's own — same seven
keys, `sample_ms = 5.0`, 1815 samples spanning 0 → 9070 ms.

### Two things that look like holes and are not

- **Short days.** 07-16 (9,795 cycles), 07-23 (9,899), 07-30 (9,864) are ~2,000
  cycles below a full day. A whole-day NXCALS query returns *the same count*, so
  this is real SPS supercycle variation, not a logging hole. Only 07-30 gained
  anything (+73).
- **Empty `h4_*` cells on `FTARGET` rows** (753 of them). Every one is an
  `extracted_e10 = 0.0`, `extraction_time_ms = -1`, 7,200 ms-cycle `FTARGET`
  carrying no spill. `EFF_SPILL_LENGHT`, `SPILL_DUTY_FACTOR` and the H4 counters
  are published **only for extracting cycles**, and `_nearest` uses a
  deliberate 1.0 s tolerance so that it will *not* borrow the neighbouring
  cycle's value. Those cells are correctly empty and no query will fill them.

### The repair that *was* real: 539 blank-filled rows

banco's `backfill_nxcals.py` keyed purely on "is this cycle timestamp already on
disk", so a row that existed but was half-empty was skipped forever. The live
watcher creates exactly such rows: it matches the companion scalars onto the
`DESTINATION` spine inside a bounded lookback, and each SPSQC variable lands in
NXCALS with its own slightly different timestamp, so cycles near the trailing
edge of that window get written with those columns empty and are never revisited.
mx17-daq's newer sibling script had grown a repair pass for this; it was ported
in for this run. It touched **07-27 (48), 07-28 (97), 07-29 (114), 07-30 (71),
07-31 (108), 08-01 (94), 08-02 (7)** — i.e. only the days the H4 columns have
existed. Repair-only: it writes into empty cells and never over a value.

### Reproducing it

Staging tree, patched script and logs are on daq at `~/dylan/sps_backfill/`:

```bash
ssh daq
cd ~/dylan/sps_backfill
export SPS_BEAM_LOG_DIR=$PWD/beam_monitor/logs SPS_SPILL_LOG_DIR=$PWD/beam_monitor/logs
~/venvs/nxcals/bin/python sps_monitor/backfill_nxcals.py \
    --start 2026-07-14 --end 2026-08-02 --include-today --force-beam-today --what scalars
~/venvs/nxcals/bin/python sps_monitor/backfill_tax_nxcals.py \
    --start 2026-07-14 --end 2026-08-02 --include-today --driver-port 5041
./run_profiles.sh          # profiles, scoped to the missing hours only
```

Safe to re-run: every day is rewritten from the union of disk and NXCALS,
deduplicated on cycle timestamp. Give each concurrent job its own
`spark.driver.port` — 5011 is daq's live watcher, 5031/5041/5051 are these.

### Pushing it back to banco — the one trap

`beam_bridge.py` re-`xrdcp`s **four names on every 20 s poll**, unconditionally,
overwriting whatever banco has:

```
beam_intensity_<today>.csv   sps_spill_<today>.csv
sps_profile_<today>_<this hour>.jsonl.gz   sps_profile_<today>_<prev hour>.jsonl.gz
```

Everything else is safe, because the only other writer is the start-up catch-up
plan and that skips any file where banco's copy is **>= the EOS copy by size** —
and a backfilled file is always larger. `h4_tax_*` is safe unconditionally: the
bridge has no TAX code and does not know the prefix.

`analysis/push_beam_record_to_banco.sh` encodes exactly this skip list. Run it
again after midnight to land the four that were "today".
