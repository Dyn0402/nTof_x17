# det4 at the SPS H4 — run and data timeline

det4 (`mx17_E` in the DAQ, det4 in the June cosmic bench, "detector E" in the
scan scripts, and det E at n_TOF — the same chamber, four names) went to the
North Area H4 line and ran parasitically inside banco's P2 uRWELL test beam
`TB_July2026_H4`. We controlled **only our own HV**. Everything else — when
runs started, how long sub-runs were, the trigger, the DAQ config — was
banco's, and we moved our voltages underneath their schedule.

Narrative recorded from Dylan 2026-08-02; every claim below is then marked with
what it rests on. **✓ = confirmed against the machine record.**
**✗ = the account and the record disagree, see the note.**
**? = no independent evidence either way.**

Regenerate the supporting tables:

```bash
cd sps_beam_test_26/analysis
../../.venv/bin/python harvest_inventory.py --table   # run/sub-run inventory
../../.venv/bin/python build_run_map.py --table       # det4 HV point -> P2 sub-run
```

---

## Rule of authority for this campaign

1. **`run_config.json` is not authoritative.** run_61's still says `Ar/Iso 95/5`
   and the 25.64° mount from run_57. Its `start_time` reads
   `2026-08-02 16:06:54` — the fourth DAQ restart, not when the run began
   (12:13:12).
2. **`hv_monitor.csv` is** — an independent monitor trace, not a setpoint echo.
   Runs 53–61 are complete again (recovered from EOS, archived under
   `records/hv_monitor/`), with two sub-run windows genuinely overwritten —
   see `HV_AND_BEAM_RECORD.md`.
3. **Our own scan logs are authoritative for det4's HV**, and are not reflected
   in the DAQ's `hvs` dict at all. They are also the only record that carries
   the P2 sub-run each point landed in.
4. **`dream_daq.log`** is authoritative for run/sub-run boundaries where it
   exists (run_61 only) — it is the one place the restarts are visible.

---

## 1. Narrative

### Friday 2026-07-31 — arrival and install

| | |
|---|---|
| **~21:00** | Installed at the 1-hour Friday-night access. **Perpendicular to the beam** (flat, 0°), **upstream of the back uRWELL and downstream of the last P2 detector**, in the ~25 cm of free space between them. |
| **overnight** | Flushed with **Ar/CO₂/iso 95/3/2** — the mixture P2 was already running, but **on our own gas line**. No data taken. |

**?** Nothing in the machine record dates the install: banco's runs ran
straight through Friday night into Saturday morning, and det4 was not in the
readout until run_53, so its arrival leaves no trace. What the other FEUs were
carrying before then is P2's business and out of scope here.

**✓ Gas — Ar/CO₂/iso 95/3/2** (confirmed 2026-08-02). `run_config.json` says
`Ar/Iso 95/5` for every run of the campaign, including run_61 today; **the
config is stale and is to be ignored**, exactly as it is for the mount angle.
This is the gas for every flat-mount and 25° measurement, and it is what any
Magboltz or gain comparison has to use — not the config string, and not the
June bench mixture.

### Saturday 2026-08-01 — first data, first scans, rotation to 25°

| when | what | evidence |
|---|---|---|
| 10:29 | pedestal run `pedestals_08-01-26_10-29-54` | ✓ |
| 10:44–11:55 | runs 50/51/52 — **FEU 01 and 05 only, no FEU 3** | ✓ det4 not yet in the readout |
| ~12:00–12:59 | FEU3 brought into the readout | ✓ inferred from the gap |
| 12:16, (14:00) | pedestal runs | ✓ — and the 14:00 one is the first HV-log collision, §`HV_AND_BEAM_RECORD.md` |
| **12:59:31** | **run_53 — det4's first data**, FEU [01,03,05] | ✓ first FEU3 file; banco's auto-QA emits `mx17_E` from here |
| 13:07–13:57 | **first resist scan**, flat mount: 535→485 V in 5 V steps, then 540/545/550 V, 180 s/point, drift held 700 V | ✓ `detE_resist_scan.log` |
| 14:00:32 | run_54 ends | ✓ |
| 14:25–14:56 | **drift scan, pre-rotation**: 700, 600, 500, 400 V at resist 550, 600 s/point, in **run_55** | ✓ `detE_scan.log` |
| **14:57:41** | drift scan **aborted** — `resist reads 1.25 (want 550.0) — refusing`. An **end-of-run power-off** killed our channels, not a detector fault | ✓ |
| 16:01–16:50 | **ACCESS: rotated to 25.64°** | ✓ resolved 2026-08-02 — see below |
| 16:20:37 | post-access pedestal run — and it **silently reverted det E's ZS from 2σ to 5σ** | ✓ `ZS_TIMELINE.md` |
| 16:55–18:03 | **resist ladder at 25°**: 670→625 V, then 620→400 V, then back up 670→685 V, during run_57 `meshscan_m90V` … `driftscan_gap300V` | ✓ `detE_scan.log` |
| 18:14–19:45 | **drift ladder at 25°**: 700→10 V, 300 s/point, spanning run_57 `driftscan_gap350V` → run_58 `operating_02` | ✓ |
| 18:04 / 18:12 | det E ZS dropped to **2σ**, then raised to **3σ** | ✓ `ZS_TIMELINE.md` |
| 20:00:54 | **run_59 `detE_long`** started — 64 samples, 4×30 min planned, 2 written (41 GB + 0.4 GB) | ✓ `start_detE_long.log` |
| **~21:00** | **ACCESS: gas changed to Ar/CF₄/iso 88/10/2.** Left to flush overnight | ✓ consistent with run_60 starting 21:20 |
| 21:20 → 09:26+1 | **run_60 `overnight_00…23`**, 24×30 min — **taken while the gas was still changing** | ✓ |

**✓ The 25° rotation is the 16:01–16:50 access** (resolved 2026-08-02; the
account's "~14:00" was a misremembering, confirmed against the record):

- `pedestals_08-01-26_16-20-37` is called *the post-access pedestal run* in
  `ZS_TIMELINE.md`, written on the day.
- The same note records det E's noise σ = 8.20 ADC as **unchanged across the
  access**, i.e. the rotation is the thing that access did.
- `POST_ACCESS_RUNBOOK.md` §4 says run_55's drift points are "**pre-rotation**
  … plan to retake them" — run_55 is 14:30–14:55, so the rotation is after it.
- It is the longest gap in P2 data-taking all afternoon (36.8 min between
  sub-runs, 49 min between runs). The only other candidate, 14:00:32→14:17:26,
  is **17 minutes** — too short for a mount change, and no pedestal run
  followed it.
- The recovered HV monitor confirms it independently: monitoring runs
  continuously to **16:00:25** and does not resume until **16:35:35**, a 35-min
  hole with nothing else in the day like it.
- Our own directory naming already agrees: `flat_ArCO2iso_95-3-2__run53-56` /
  `rot25_ArCO2iso_95-3-2__run57`.

> **The H4 beam record cannot see accesses, so it could not be used to decide
> this.** Checked explicitly: on 08-01 the SPS extracted to `FTARGET`
> continuously from 00:00 to 18:49 at a steady ~1380×10¹⁰, and
> `h4_bend_027_a` / `_309_a` / `_706_a` held 280.0 / 478.0 / 216.4 A all day
> with **no off-window anywhere**. An H4 zone access is made with the line's
> beam stopper, which is not one of the logged variables. Access times have to
> come from the DAQ gaps and the pedestal runs, as above.

The 14:00 event the account may be blending in is real but different: run_54
ended at 14:00:32, a pedestal run fired at 14:00:23, and at 14:57 an
end-of-run power-off killed our HV and aborted the drift scan.

### Sunday 2026-08-02 — 15°, then back to 25°

| when | what | evidence |
|---|---|---|
| 09:26:41 | run_60 ends | ✓ |
| 09:26–11:26 | `detE_hold.py` holding drift 700 / resist 650 V, no scan | ✓ `detE_hold.log` |
| 11:23:48 | `*** HV OFF (re-arm #1) … almost certainly an end-of-run power-off` — auto-recovered by 11:26 | ✓ |
| **~11:00** | **ACCESS: rotated to ~15° (15.465°)** | ✓ consistent — hold log stops 11:26, run_61 starts 12:13 |
| 12:13:12 | **run_61** starts; `meshscan_m00V` 12:14 | ✓ `dream_daq.log` |
| 12:45–13:13 | resist creep 724.8→789.9 V — **not a scripted ladder**, read off the HV monitor trace afterwards | ✓ `run61_conditions.py` |
| 13:13–14:00 | **drift scan at 15°**: resist fixed 750 V, drift 700→70 V, 10 points × 5 min | ✓ `det4_drift_scan.log` |
| **14:00:46** | drift scan **aborted**. Logged as `drift channel 8:8 powered OFF (trip?)` — **it was not a trip** | ✗ see below |
| **~14:00** | **ACCESS: rotated back to 25°** | ✓ **independently confirmed by the alignment** — see below |
| 14:00–16:06 | DAQ cycling through restarts (15:24, 15:40, 16:06); `m00V` restarted twice, `m30V` once | ✓ `dream_daq.log` |
| 15:02 / 15:19 | pedestal runs — new pedestal set for everything after | ✓ |
| 16:42–17:14 | **scripted resist scan**: 720→580 V, 5 V steps, 60 s, drift held 700 V, 29 points | ✓ `det4_resist_scan_720_580.log` |
| 17:34→ | run_61 `meshscan_m70V`/`m80V`/`m90V`/`m100V` continuing | ✓ |
| 18:53:21 | **SPS linac down, T2 off** — beam to 0 mid `meshscan_m100V` | ✓ `M100V_PARTIAL.md` on EOS |
| 18:56:32 | run_61 stopped rather than record beamless data; `m100V` left as a 70 %-statistics point, deliberately not retaken | ✓ |
| 17:56:29 | `det4_post_run_sequence.py` armed: resist 770 V, drift 700 V, then drift 700→50 V in −50 V steps, 8 min/point, hard stop 21:00 | ✓ `det4_post_run_sequence.log` |
| **~21:00** | **PLANNED: rotate back to flat**, overnight high statistics at operating voltage, for charge spreading / sharing response | account |

**✗ The 14:00:46 abort was not a trip.** `dream_daq.log` records
`Run finished normally` at **14:00:42**, four seconds earlier, and the scan log
shows **both** channels collapsing at once — drift 23.5 V *and* resist 642.5 V,
down from a held 750 V. A trip on drift 8:8 cannot pull resist 12:2 down. This
is the known `hv_control.power_off_hvs()` end-of-run sweep (POST_ACCESS_RUNBOOK
§5), the third time it hit us — 14:57 and 11:23 being the others.
`rot15_ArCF4iso_88-10-2__run61_1214-1400/README.md` still says "trip"; that should be corrected.

**✓ The rotation back to 25° is confirmed by the alignment fit.** The det4↔uRWELL
transform jumps across the 14:00–16:06 gap: `det(A)` **1.036 → 1.120**, +8.1 %,
with ~18 mm of Y. A rotation about the vertical axis scales the projected
footprint by 1/cos θ, and

| | |
|---|---|
| observed det(A) ratio | 1.081 |
| cos(15.465°)/cos(25.64°) | **1.069** |
| implied θ₂ from the fit | **26.9°** |

So the geometry change across that gap **is** the rotation, at the right size.
It was never a mystery — it was the access.

### Consequence, now fixed: the run_61 scans were one curve across two angles

`resist_scan_gas2.py` / `gain_scan_gas2.py` labelled the whole combined scan
`15.465 deg`. Session 1 (12:45–13:13, 724.8–789.9 V) is at 15°; **session 2
(16:29–17:14, 580–719.8 V) is at ~25°**. The combined efficiency and gain
curves therefore had a mount-angle change buried at their ~720 V seam, on top
of a pedestal-set change at the same place.

**Resolved 2026-08-02.** The scans are now split by condition — see
`/media/dylan/data/x17/sps_run53_det4_check/CONDITIONS.md`. Conditions,
sub-run start times and HV plateau windows are defined once in
`mapping_check/run61_conditions.py`; `resist_scan_run61.py` and
`gain_scan_run61.py` loop over them and write into per-condition directories.
There is no combined curve any more.

The drift scan (`driftscan_run61.py`, 13:13–14:00) is entirely pre-rotation and
is cleanly **15°**.

---

## 2. Physical configuration epochs

Mount angle is a rotation of the detector **down about the vertical axis,
right-hand rule** (`DAQ_DETE_ROT_Y` / `DETE_ROT_Y_DEG`). Gas tracked P2's own
line throughout — same mixture, our own line.

Directory naming on the data disk follows this table; the key is
`/media/dylan/data/x17/sps_run53_det4_check/CONDITIONS.md`.

| # | from | to | mount | gas | notes |
|---|---|---|---|---|---|
| 0 | Fri 07-31 ~21:00 | Sat 08-01 12:59 | flat (0°) | Ar/CO₂/iso 95/3/2 | install + flush, no data |
| 1 | Sat 12:59 | Sat 16:01 | **flat** | Ar/CO₂/iso 95/3/2 | runs 53–56 |
| 2 | Sat 16:50 | Sat ~21:00 | **25.64°** | Ar/CO₂/iso 95/3/2 | runs 57, 58, 59 |
| 3 | Sat ~21:00 | Sun 08-02 ~11:00 | 25.64° | **Ar/CF₄/iso 88/10/2, flushing** | run 60 — gas in transition, do not treat as either mixture |
| 4 | Sun ~11:00 | Sun ~14:00 | **15.465°** | Ar/CF₄/iso 88/10/2 | run 61 `m00V`–`m30V` (to 14:00) |
| 5 | Sun ~14:00 | **Mon 08-03 00:40** | **25.64°** | Ar/CF₄/iso 88/10/2 | run 61 `m30V` (from 16:08) onward, run 62, run 63 `operating_00/_01` — see §3b, the ~21:00 rotation did NOT happen |
| 6 | **Mon 08-03 01:00** | — | **flat** | Ar/CF₄/iso 88/10/2 | run 63 `operating_02` tail + `operating_03`, then run 71 (RAW). Access timed by the H4 TAX stopper: blocked 00:40:11–00:57:55, beam back 01:00:50 |

Epoch 3 is the one to be careful with: run_60 is 12 hours of data taken while
the gas was being exchanged. Whether it is usable depends on the flush time
constant, which nobody has measured for this chamber on this line.

---

## 3. DAQ configuration — what we changed in banco's readout

| | banco's original | ours |
|---|---|---|
| zero suppression | on, TPC mode (`ZsTyp=1`) | same |
| uRWELL thresholds | **5σ** | untouched |
| det E (FEU3) thresholds | 5σ by default | 2σ → reverted to 5σ at 16:20 → 2σ at 18:04 → **3σ** from 18:12 |
| samples per waveform | **16** | **32** normally, **64** for drift scans |

**✓** run_59 ran at 64 samples (`start_detE_long.log`); run_61 runs at 32
(`run_config.json`). Both match the account.

**The rate cost was measured, and it was not only ours.** From `ZS_TIMELINE.md`,
run_57 per-second data rates:

| sub-run | det E ZS | FEU 1 (uRWELL) | FEU 3 (det E) |
|---|---|---|---|
| `driftscan_gap300V` | 5σ | 1.54 MB/s | 2.08 MB/s |
| `driftscan_gap350V` | **2σ** | **0.92 MB/s** | **23.76 MB/s** |

FEU 3 up 11.6×, and **FEU 1 down 40 % although its thresholds were never
touched** — our FEU floods, asserts BUSY, the shared TCM withholds triggers,
and banco loses rate too. At 64 samples in TPC mode, 2σ keeps a channel with
probability 1−(1−p)ⁿ = **77 %**, not the 2.3 % the per-sample figure suggests.
That is the mechanism behind "extending the window really killed the event
rate": it is the sample count and the threshold *together*.

---

## 3b. Corrections established 2026-08-03/04

From decoding the beam data itself. Each supersedes a claim above; the original
text is left in place so the correction is visible rather than silent.

**run_56 ran at 64 samples, not 32.** §3 says 32 normally and 64 only for drift
scans. `run_56/run_config.json` says `n_samples_per_waveform = 64`; the flat
resist ladder was also at 64.

**det E was at 5 sigma during run_56, not 2 sigma.** §3's threshold history is
wrong before 16:20. The `..._03_thr.prg` copied into the sub-run directory
carries its own header — *"Threshold value: 5.000000 sigmas"* — and that is the
set the FEU actually ran.

**`dream_daq.log` exists on EOS for runs 54, 55 and 56**, not run_61 only (§1
rule 4). Real sub-run boundaries differ from §5's QA-PNG-mtime table by 10-15
min; e.g. run_55 `meshscan_m00V` is 14:15:23-14:27:28, not "14:30".

**Zone accesses CAN be dated to the second**, contrary to the box in §1. The H4
TAX beam stopper (`XTAX_022_023:POSITION_MEAS`) is logged by the mx17-daq
NXCALS client — see `tax_windows.py` and the note in `HV_AND_BEAM_RECORD.md`.
banco's mirror does not carry it, which is where the original claim came from.

**The ~21:00 08-02 rotation back to flat did not happen then.** §1 records it as
`account`, never confirmed, and it is wrong. det4 was still at 25.64 deg through
run_62 and the first half of run_63: the alignment gives det(A) = 1.1132
(1/cos -> 26.1 deg) against 1.009 for a genuinely flat mount. **The rotation to
flat happened at the 08-03 00:40:11-00:57:55 access**, timed by the TAX
stopper, with beam back at 01:00:50.

**run_63 therefore straddles an access and is TWO conditions.** Split in
`datasets.py` as `run63_rot25` (operating_00/_01, 25.64 deg, the drift ladder)
and `run63_flat` (operating_02 tail + operating_03, flat, 53.4 min at the
operating point). What first looked like a beam dip inside operating_01 is the
stopper closing, not a machine fault.

**Every RAW run decoded before 2026-08-03 has merged events.** The decoder
closed an event only on the FEU end-of-event marker, and under RAW bandwidth
the FEU drops ~24 % of sample-group packets including some carrying that
marker. Fixed by delimiting on `eventID`
(`mm_strip_reconstruction`, branch `raw-mode-event-splitting-and-loss-reporting`).
This affects banco's auto-produced `combined_hits` for run_71 as well as our
own decodes, so the uRWELL side had to be re-decoded too.

## 4. Open questions

**Answered by this pass** — recorded so they are not re-asked:

- **Saturday's rotation to 25° is the 16:01–16:50 access**, not ~14:00.
  Confirmed 2026-08-02. The H4 beam record cannot resolve accesses at all —
  see the box in §1.
- **The first-night gas is Ar/CO₂/iso 95/3/2**; `Ar/Iso 95/5` in the config is
  stale for every run. Confirmed 2026-08-02.
- **Sunday's ~14:00 rotation back to 25°** is confirmed, and it is what the
  14:00–16:06 alignment jump was.
- run_61's 14:00 abort: an end-of-run power-off, not a trip.
- run_60 thinning out after `overnight_14`: **answered 2026-08-05** from the
  backfilled spill record (`GAS_FLUSH_TIMELINE.md` §1): SPS FTARGET
  extractions stop at ~04:50 and return only ~08:30 (at near-zero intensity
  until ~09:30). `overnight_15`–`overnight_23` are beamless. The live record
  had stopped before it; the NXCALS backfill covers it.

- **The gas tracked P2's own line throughout** — the same mixture, run on our
  own line. Ar/CO₂/iso **95/3/2** from the Friday install to the Saturday
  ~21:00 access, Ar/CF₄/iso **88/10/2** after it. Confirmed 2026-08-02.
- **Rotation sign convention**: the detector rotates **down about the vertical
  axis, signed by the right-hand rule** — the angle that belongs in
  `DAQ_DETE_ROT_Y` / `DETE_ROT_Y_DEG`. Confirmed 2026-08-02, closing the item
  `POST_ACCESS_RUNBOOK.md` §1 raised as unrecoverable from the number alone.
- **What FEU3 and FEU4 carried before run_53** is out of scope. The campaign
  starts at the Friday-night install; the other FEUs are P2's.

**Still open:**

1. **Where do 15.465° and 25.64° come from** — measured how, against what
   reference? The precision implies a measurement, not a setting, and the
   cos θ cross-check on the alignment only constrains the *difference*.
2. ~~**run_59 `detE_long`** was configured for 4×30 min sub-runs; only
   `detE_long_00` (41 GB) and `_01` (0.4 GB) exist, and `_01` is 100× smaller.
   Beam loss, or was it stopped?~~ **Answered 2026-08-05** (TAX record,
   `GAS_FLUSH_TIMELINE.md` §1): the gas-change access blocked H4 at
   **20:24:07**, 22 min into `detE_long_00`. `_01` started into the access
   with no beam. Not a DAQ fault.
3. **The 764.8 V discharge spike** (2.8 %, 47k tracks, in the 15° condition) —
   anything seen at the time?
4. **run_61 `meshscan_m00V` at 12:14** — was that sub-run's data usable at all?
   Its HV log is the one that got overwritten, and the sub-run was restarted
   twice.
5. **The 15° vs 25° efficiency gap.** At comparable resist voltage the 15°
   condition reads 36.7 % (724.8 V) and the 25° condition 32.7 % (719.8 V).
   Angle, pedestal set and time all differ, so nothing is attributable yet.

---

## 5. Machine-harvested sub-run table

See `run_inventory.json` (regenerate with `harvest_inventory.py`) and
`run_map.csv` (`build_run_map.py`), which lists every det4 HV point with the P2
run/sub-run that was live at the time. `join=logged` means the scan driver
stamped the sub-run itself (Saturday); `join=by-time` means it was matched
against sub-run boundaries afterwards (Sunday, whose driver dropped the
window-gating).

`raw` timings come from the `.fdf` files (DAQ wall clock; trailing `_NN` is the
FEU: **01** uRWELL front, **03** det4, **04** a P2 FEU present only to run_49,
**05** uRWELL back). Runs 55–58 were pruned from banco's disk, so only banco's
auto-QA PNG mtimes survive for them — those are *processing* times, minutes
after the sub-run, and their detector list is the auto-pipeline's, which stops
emitting `mx17_E` after run_56/`meshscan_m40V` even though FEU3 keeps writing
all the way through run_61.

### Saturday 2026-08-01

| start | end | run | sub-run | src | FEUs | fdf | GB | det4 doing |
|---|---|---|---|---|---|---:|---:|---|
| 10:44 | 11:07 | run_50 | cfg_gain3.0_peaktime200_opt | raw | 01,05 | 5 | 2.5 | not in readout |
| 11:08 | 11:31 | run_51 | cfg_gain3.0_peaktime200_deflt | raw | 01,05 | 5 | 2.5 | not in readout |
| 11:32 | 11:55 | run_52 | cfg_gain4.5_peaktime200_opt | raw | 01,05 | 5 | 2.5 | not in readout |
| 12:59 | 13:29 | run_53 | cfg_gain4.5_peaktime200_deflt | raw | 01,03,05 | 11 | 6.1 | **first data**; resist 535→505 V |
| 13:30 | 14:00 | run_54 | cfg_gain4.5_peaktime50 | raw | 01,03,05 | 11 | 6.1 | resist 500→485, 540→550 V |
| 14:30 | 14:30 | run_55 | meshscan_m00V | qa | +mx17_E | — | — | drift 700, 600 V |
| 14:42 | 14:42 | run_55 | meshscan_m10V | qa | +mx17_E | — | — | drift 500 V |
| 14:54 | 14:55 | run_55 | meshscan_m20V | qa | +mx17_E | — | — | drift 400 V; **aborted 14:57 by power-off** |
| 15:11 | 15:12 | run_56 | meshscan_m30V | qa | +mx17_E | — | — | idle |
| 15:24 | 15:24 | run_56 | meshscan_m40V | qa | +mx17_E | — | — | idle |
| 15:36 | 16:01 | run_56 | meshscan_m50/60/70V | qa | uRWELL only | — | — | idle |
| — | — | — | **ACCESS — rotate to 25.64°** | — | — | — | — | ped run 16:20, ZS reverted to 5σ |
| 16:50 | 16:51 | run_57 | meshscan_m80V | qa | uRWELL only | — | — | idle |
| 17:03 | 17:03 | run_57 | meshscan_m90V | qa | uRWELL only | — | — | resist 670→655 V |
| 17:15 | 17:16 | run_57 | meshscan_m100V | qa | uRWELL only | — | — | resist 650→625 V |
| 17:28 | 17:28 | run_57 | driftscan_gap150V | qa | uRWELL only | — | — | resist 620→595 V |
| 17:41 | 17:41 | run_57 | driftscan_gap200V | qa | uRWELL only | — | — | resist 590→565 V |
| 17:53 | 17:53 | run_57 | driftscan_gap250V | qa | uRWELL only | — | — | resist 560→480 V |
| 18:05 | 18:06 | run_57 | driftscan_gap300V | qa | uRWELL only | — | — | resist 460→400, then 670→685 V |
| 18:19 | 18:19 | run_57 | driftscan_gap350V | qa | uRWELL only | — | — | **drift** 700, 650 V; ZS → 2σ |
| 18:31 | 18:31 | run_57 | driftscan_gap400V | qa | uRWELL only | — | — | drift 600, 550 V; ZS → 3σ |
| 19:04 | 19:05 | run_58 | operating_00 | qa | uRWELL only | — | — | drift 500→300 V |
| 19:33 | 19:33 | run_58 | operating_01 | qa | uRWELL only | — | — | drift 250→70 V |
| 20:01 | 20:01 | run_58 | operating_02 | qa | uRWELL only | — | — | drift 40→10 V |
| 20:02 | 20:35 | run_59 | detE_long_00 | raw | 01,03,05 | 82 | 41.0 | **long run, 64 samples** |
| 20:34 | 20:54 | run_59 | detE_long_01 | raw | 01,03,05 | 10 | 0.4 | long run (thin — why?) |
| — | — | — | **ACCESS — gas → Ar/CF₄/iso 88/10/2** | — | — | — | — | |
| 21:20 | 09:26+1 | run_60 | overnight_00…23 | raw | 01,03,05 | 5–15 ea | 0.2–8.5 ea | **overnight, gas in transition** |

run_60's sub-runs are on a clean 30-minute cadence. Sizes hold near 8 GB
through `overnight_13`, then collapse to 0.2 GB from `overnight_15` (04:54)
onward.

### Sunday 2026-08-02

| start | end | run | sub-run | src | FEUs | fdf | GB | det4 doing |
|---|---|---|---|---|---|---:|---:|---|
| 12:07 | 12:07 | run_59 | meshscan_m00V | raw | (empty) | 0 | 0.0 | stray mis-targeted start |
| 12:14 | 12:45 | run_61 | meshscan_m00V | raw | 01,03,05 | 44 | 21.0 | 15°; restarted 15:25 and 15:41 |
| 12:45 | 13:15 | run_61 | meshscan_m10V | raw | 01,03,05 | 42 | 22.7 | 15°; resist creep 724.8→789.9 V |
| 13:15 | 13:45 | run_61 | meshscan_m20V | raw | 01,03,05 | 34 | 18.7 | 15°; **drift scan** 700→350 V |
| 13:46 | 14:00 | run_61 | meshscan_m30V | raw | 01,03,05 | 46 | 23.7 | 15°; drift 280→140 V, then **power-off 14:00:46** |
| — | — | — | **ACCESS — rotate back to 25.64°** | — | — | — | — | DAQ restarts 15:24 / 15:40 / 16:06; peds 15:02, 15:19 |
| 16:08 | 16:32 | run_61 | meshscan_m30V (2nd) | raw | 01,03,05 | — | — | **25°** |
| 16:29 | 16:53 | run_61 | meshscan_m40V | raw | 01,03,05 | 22 | 10.7 | **25°**; resist 719.8→690 V |
| 16:51 | 17:14 | run_61 | meshscan_m50V | raw | 01,03,05 | 18 | 10.4 | **25°**; resist 684.8→590 V |
| 17:12 | 17:36 | run_61 | meshscan_m60V | raw | 01,03,05 | 22 | 10.9 | **25°**; resist 584.8→580 V |
| 17:34 | 17:57 | run_61 | meshscan_m70V | raw | 01,03,05 | 22 | 11.1 | **25°**; not yet analysed |
| 17:55 | 18:19 | run_61 | meshscan_m80V | raw | 01,03,05 | 22 | 10.9 | **25°**; not yet analysed |
| 18:17 | 18:33 | run_61 | meshscan_m90V | raw | 01,03,05 | 16 | 7.5 | **25°**; not yet analysed |
| 18:38→ | | run_61 | meshscan_m100V | raw | 01,03,05 | — | — | live at time of writing |

---

## 6. What is analysed vs. what is not

| data | analysed? | where |
|---|---|---|
| run_53 / run_56 flat-mount efficiency maps | yes | `flat_ArCO2iso_95-3-2__run53-56/`, `DET4_EFFICIENCY_H4_2026-08-01.md` |
| run_57 resist ladder at 25° | yes | `rot25_ArCO2iso_95-3-2__run57/` |
| run_61 drift scan (15°, 700→70 V) | yes | `rot15_ArCF4iso_88-10-2__run61_1214-1400/`, `driftscan_run61.py` |
| run_61 resist + gain, 15° half | yes | `rot15_ArCF4iso_88-10-2__run61_1214-1400/resist_scan.*`, `gain_scan.*` |
| run_61 resist + gain, 25° half | yes | `rot25_ArCF4iso_88-10-2__run61_1606on/` |
| run_61 `m00V` single-point eff map | yes | `rot15_ArCF4iso_88-10-2__run61_1214-1400/effmap_m00v/` |
| run_61 `m20V`, `m30V` (both halves) | staged 2026-08-05, not analysed | the 15° drift-scan sub-runs; m30V holds TWO same-named passes (13H46 = 15°, 16H08 = 25°) — split by datrun stamp |
| run_61 `m70V`–`m100V` | **yes, 2026-08-05** | NOT scan points — hv_monitor shows drift 700.2/resist 769.8 held ~80 min = the 25.64° **operating block** (`run61_op25` in `datasets.py`): 454k clean tracks, 62.2 % in-band, and the 4th plateau of the wft ladder fit |
| run_59 det4 side | **yes, 2026-08-05** | `run59_co2`: the last CO₂ dataset (beam dies 20:24:07 — the gas access); resist was 669.8 V; the CO₂ span anchor of the flush fit |
| run_60 det4 side | **yes, 2026-08-05** | `run60_flush`: THE gas-flush transient — lag 1.72 h, τ 3.49 h, see `GAS_FLUSH_TIMELINE.md` §4a |
| run_58 det4 side | **no** | not staged (its own sub-runs are the Saturday ladder tail) |
| Saturday's 25° drift ladder (18:14–19:45, run_57/58) | **no** | 17 points, 700→10 V, ~100 GB — needs a fresh Kerberos ticket; the CF₄ ladder (`run63_rot25` + `ladder_span.py`/`wft_beam_fit.py`) covers the same physics in the other mixture |

Data root on the analysis laptop:
`/media/dylan/data/x17/sps_run53_det4_check` (mirror `~/x17/…`), with the
recovered banco records now under `records/`.

---

## 7. Data locations

| what | where |
|---|---|
| raw `.fdf`, live | `banco_cern:/local/home/banco/P2_data/TB_July2026_H4/runs/run_{59,61}/` |
| raw `.fdf`, older | `.../dream_run/run_{22..54,59,60,61}/` |
| runs 55–58 raw | pruned from banco, **complete on EOS** at `root://eospublic.cern.ch//eos/experiment/ntof/data/x17/p2_sps_july/runs/` |
| our scan scripts + logs | `banco_cern:/local/home/banco/dylan/` |
| det4 scan logs (archived) | `…/sps_run53_det4_check/records/scan_logs/` |
| HV monitor CSVs (archived) | `…/records/hv_monitor/` — 88 files, runs 53–61 complete, recovered from EOS |
| SPS beam + H4 line records | `…/records/beam/` — **stops 08-01 18:49, see `HV_AND_BEAM_RECORD.md`** |
| EOS backups | `/eos/experiment/ntof/data/x17/p2_sps_july/` and `/eos/project/s/salsachip/Data/T2_tests/P2_SPS_Dream_Data/` |
| working ssh alias | `banco_cern` only |
