# HANDOFF — run 224489 (2026-07-17 plastic HV scan #2, first run WITH liquid scintillators)

**Audience: a fresh Claude session on Dylan's desktop, picking this up cold.**
Read `README.md` in this directory first — it is the handover doc for the July
beam QA analysis (runs 224404/224460/224466) whose machinery you will reuse.
This file only covers what is NEW for run 224489 and the full task list.

---

## 0. What run 224489 is

- Second plastic-scintillator HV scan, taken 2026-07-17, EAR2, campaign
  `X17_measurement`. 1750 beam bunches; PKUP `psTime` ends 17:17:04 UTC
  (= 19:17 local CERN). Beware: some PKUP entries have `psTime == 0` — filter
  them before computing time ranges.
- **First run with the liquid scintillators (LIQ) in the readout.** Trees
  `LIQA-D`, ONE channel each (`detn == 1`), 5–22 M hits per arm
  (C lowest: 5.3 M, B highest: 21.6 M).
- Processed on lxplus by Claude on 2026-07-17 with Riccardo Mucciola's new
  PSA `UserInput_2026_EAR2_X17.h` (adds LIQ + pulse-shape fitting for WALL
  and LIQ; WALL/plastic params retuned for the new signal path — see §2).
  DAG completed OK, all detector trees verified present.

### Get the file (16 GB)

```
# from lxplus (ssh lxplus works from Dylan's machines):
scp lxplus:/eos/user/d/dneff/ntof_x17_processing/run224489/done/run224489.root \
    ~/x17/beam_july/data/
# or, usually faster:
xrdcp root://eoshome-d.cern.ch//eos/user/d/dneff/ntof_x17_processing/run224489/done/run224489.root \
    ~/x17/beam_july/data/
```

Local destination follows the existing convention: `~/x17/beam_july/data/`
(next to run224404/224460/224464/224466.root). Verify size ≈ 15,976,600,691 B.

Note this file is in Dylan's user EOS, NOT the official processing area — the
official area was not used because this run needed Riccardo's brand-new
UserInput. Partial files live in `.../run224489/completed/` (deletable).

## 1. Code you need (git)

Everything lives in `github.com:Dyn0402/nTof_x17.git`, directory
`mx_july_beam_qa/`. The laptop was 4 commits ahead of origin including all of
the July QA work — **make sure `git pull` on the desktop actually brings in
commit `e866dc9` ("July beam QA: plastic HV scan, trigger thresholds,
imaging/vertical calib, fast read pass") and this HANDOFF file.** If they are
missing, the push from the laptop didn't happen — stop and ask Dylan.

Build the fast reader once: `cd mx_july_beam_qa/fastread && make`
(needs `root-config`; ROOT 6.36 works). Python: the repo venv (`.venv`),
numpy 1.x + uproot.

## 2. Hardware/DAQ changes between run 224466 (last scan) and 224489

These change the interpretation of EVERYTHING downstream — read carefully.

1. **Plastic signal path: linear fan-in/fan-out (FIFO) instead of BNC-T split**
   to DAQ + trigger logic. Expected consequence: **plastic amplitudes ~2×**
   relative to 224466 at the same HV (the T split halved the signal). Treat 2×
   as a hypothesis to MEASURE, not assume (the coincident-median power-law fits
   per PMT give this directly by comparing runs at equal HV).
2. **≥16 ns extra delay on the plastics** from the new path. So the
   run-224466 `calib/time_offsets_*.json` are INVALID for wall–plastic dt.
   Redo the offset scan from scratch (03), same sliding-window method; expect
   the dt peaks shifted by roughly −16 ns (sign check: dt = t_wall − t_pss,
   so plastics later ⇒ dt more negative... verify, don't trust this sentence).
3. **Liquids never timed in at all.** First task with LIQ is a wide coincidence
   window scan (02-style) WAL×LIQ and PSS×LIQ to even find the dt peak —
   scan generously (±200 ns or more) before narrowing.
4. **Cross-wiring fix**: after run 224404/224460, a crossed/duplicated channel
   was identified (the "duplication" between same-side channels one group over,
   arms A/D — see `17_duplication_veto.py` docstring and README flag). The
   cabling was supposedly fixed before this run. **Confirm it** (task T2).
5. **SiPM gating: ON for the whole run** (per Dylan). Unlike 224466 (which
   flipped mid-run), no gating boundary to handle — but flash-window wall
   occupancy is not comparable to 224466 pass-1 steps.

## 3. The HV scan log — pull it yourself, no Dylan needed

The desktop has passwordless ssh to BOTH machines: `ssh lxplus` (data) and
`ssh daq_lxplus` (DAQ machine, host mx17-daq, user mx17). Verified working
2026-07-17. Pull the scan log with:

```
mkdir -p ~/beam_july/scint_hv_scan
scp -r daq_lxplus:beam_july/scint_hv_scan/2026-07-17_17-41-11_plastic_scan_2 \
    ~/beam_july/scint_hv_scan/
```

Contents (already read on 2026-07-17): `hv_beam_monitor.csv` (947 rows, 5 s
cadence: `timestamp, step_label, target_v`, per-PMT vmon/imon, beam state)
and `RUN_NOTES.md`. Key facts from the notes so you can plan without it:

- **Same 9-step ladder as the 07-16 scan**: pass 1 = 1600, 1500, 1400, 1300,
  1200 V (steps 1–5); pass 2 = 1550, 1450, 1350, 1250 V (steps 6–9); merged
  = uniform 50 V grid 1200–1600 V. ~100k e10 protons per step.
- Step start times (LOCAL = UTC+2): 1(1600) 17:41 · 2(1500) 17:50 ·
  3(1400) 18:00 · 4(1300) 18:09 · 5(1200) 18:19 · 6(1550) 18:28 ·
  7(1450) 18:38 · 8(1350) 18:47 · 9(1250) 18:56. Scan done 19:05:29.
- **SiPM gating ON the entire run** — single condition, no boundary
  (matches Dylan; RUN_NOTES confirms).
- No HV trips; imon steady 160–175 µA; END_ACTION='nominal' ramped back to
  per-channel 1275–1325 V at 19:05.
- **Post-scan reference window**: 19:05–19:17 local (run's last PKUP pulse
  19:17:04 local) at per-channel nominal HV — ~12 min, use like 224466's
  post-scan window. A PRE-scan window is short or absent (first raw file
  closed 17:40, scan started 17:41) — check in the data, don't assume one.
- **Plastics only were stepped — liquid HVs fixed throughout** (Dylan), so
  the LIQ tasks (T5/T6) can integrate over the whole run (subject to
  plastic-HV step splitting for anything involving plastic amplitudes).

Bunch→step assignment: match PKUP `psTime` (ns, UTC) against the CSV
timestamps (LOCAL, UTC+2) — machinery in `12_plastic_hv_scan.py`.

## 4. Task list (ordered; each builds on the previous)

### T0 — Infrastructure: extend the hit cache to LIQ
`fastread/extract_hits.cpp` line ~44: `HIT_TREES` only lists WALA-D, PSSA-D.
Add `LIQA","LIQB","LIQC","LIQD` (and SILI if useful), rebuild, re-extract.
Check `hitcache.py` for any hard-coded tree lists too. Then
`./run_readpass.sh ~/x17/beam_july/data/run224489.root` (~10 min).
LIQ hits have a single detn=1 — the pairing helpers should cope, but watch
for any per-channel loops assuming ≥2 channels.

### T1 — Standard read pass + QA (scripts 01–09 conventions)
Run 01 (signal QA) and 02 (coincidence scan) first. Because of §2 items 2–3,
run 02 with a WIDE dt range for all pairings: WAL×PSS, WAL×LIQ, PSS×LIQ
(same arm). Then 03 for fresh per-channel offsets →
`calib/time_offsets_run224489.json`. Sanity checks along the way:
- wall outage check (224404 had 46% of bunches with all-WAL-dead; check
  bunch-vs-hits map before trusting anything);
- gamma-flash position/`tflash` sane; ADC→mV via `adc_mv.py`
  (`fullScalemV/65536`, regenerate `calib/adc_to_mv_run224489.json`).

### T2 — Confirm the cross-wiring fix
Re-run `17_duplication_veto.py` on 224489. Success criteria:
- the equal-amplitude ±4 ns same-side neighbor duplication band on arms A/D
  is GONE (flagged fraction ≈ B-arm control level);
- A/D wall–plastic coincident spectra now show MIP bumps WITHOUT needing the
  veto (in 224404, A/D had 6–8× fewer true coincidences and no MIP bump —
  README "OPEN FLAGS"). If A/D now look like B/C, both the wiring fix is
  confirmed AND the old A/D deficit is explained. Report this loudly either
  way — it closes (or reshapes) a standing collaboration question.

### T3 — SiPM wall MIP peaks per HV step (repeat of 224466 analysis)
`12_plastic_hv_scan.py` + `12b` machinery: per-step sideband-subtracted
wall MIP peaks. Expected: wall MIP position again immune to plastic HV.
This revalidates the B/C calibration transfer — and now hopefully A/D too.

### T4 — Plastic HV scan → gain power laws + NEW equalization table
Same as 224466 (12/12b/12c): per-PMT coincident (wall-tagged,
sideband-subtracted) median vs V, fit ~V^n. Deliverables:
- per-PMT n and G(V) (224466 gave n = 4.8–6.9);
- **FIFO vs BNC-T gain ratio**: compare same-HV points between runs —
  expect ≈2×, measure it per PMT;
- updated HV equalization table (224466 table: AL −22, AR −33, BL +51,
  BR −21, CL −120, CR +7, DL +3, DR +117 V — superseded by this run).

### T5 — Triple coincidences WAL×PSS×LIQ → plastic MIP attempt  ★ new physics
The whole point of adding the liquids. A wall–plastic tag selects wall MIPs
only (plastic is at the back of that pair); adding a LIQ hit BEHIND the
plastic selects through-going particles ⇒ the plastic spectrum in triples
is a genuine plastic MIP candidate spectrum.
- Build triples: for each same-arm wall–plastic pair (|dt_cal| ≤ 8 ns after
  new offsets), search LIQ hits near the pair time (window from T1's WAL×LIQ
  dt peak; sideband-subtract in the LIQ dt too — 2D sidebands or sequential).
- Stack order (as-built, measured 2026-07-15, see MX17_Full_Geant commit
  47cb1d2): MM → SiPM wall → plastics → single LS layer (2 cm LAB in CFRP
  box). So LIQ is the outermost — correct geometry for this tag.
- **Statistics will be limited** (LIQC only has 5.3 M inclusive hits and the
  scan splits the run into steps). First try inclusive-over-all-steps at
  fixed nominal-HV windows (pre/post-scan reference windows, if this scan
  had them like 224466 did). Only split by step if the peak is fat.
- If a plastic MIP peak appears: fit position per PMT per step where
  possible → this is the direct plastic gain measurement that replaces the
  coincident-median proxy.

### T6 — Liquid gain-vs-position map  ★ new physics
Motivation: suspected large LIQ gain variation with distance from its PMT
(see updated detailed geometry in `~/CLionProjects/MX17_Full_Geant` — check
where the PMT couples to the LAB volume before interpreting).
Position estimators, all from EXISTING machinery:
- **vertical**: wall top/bottom SiPM asymmetry ln(A_top/A_bot) and timing
  dt_tb (16_wall_vertical.py — both estimators validated on 224460-era data);
- **horizontal**: which wall group (4 bins across the wall) + which plastic
  bar (2 bins).
For triple-tagged hits, accumulate LIQ amplitude (median or MIP-ish
landmark) in position bins → 2D map per arm. Even a coarse 4×few map
confirms/kills the gain-gradient hypothesis. Compare gradient direction
with the actual PMT location from the Geant geometry.

### T7 — Timing program
- New wall–plastic offsets (T1) — compare to 224466 offsets: is the shift a
  clean common ~16 ns on all plastic channels, or per-channel?
- First-ever LIQ offsets: WAL×LIQ per (wall ch × arm) → extend the offsets
  JSON schema (new keys, don't break old consumers).
- Check the −60 ns satellite bump (README open flag) in the new signal path —
  did it move/disappear with the FIFO? That discriminates reflection vs
  afterpulse.

### T8 — Absolute energy calibration (only if T5 finds a plastic MIP)
- Compute expected MIP dE in the plastic from geometry (2.5 cm thick bars
  now — thickness was updated in the Geant geometry; ~2 MeV/cm ⇒ ~5 MeV, but
  use the sim/geometry numbers, and mind path-length spread from incidence
  angles — the position estimators from T6 can give a crude angle cut).
- MIP peak (mV) + dE (MeV) ⇒ mV/MeV per PMT per HV step ⇒ absolute gain
  G(V) curves, not just relative. Cross-check against T4's power laws.

## 5. Deliverables Dylan expects

1. Updated figures/caches in the standard layout (`figures/`, `cache/`,
   `calib/*_run224489.json`).
2. A written summary (extend README.md "Key results" with a run-224489
   section, or a new report in `report/`) covering, in order of priority:
   cross-wiring confirmation (T2), plastic MIP found/not-found (T5), FIFO
   gain ratio + new HV equalization (T4), LIQ gain map (T6), timing (T7),
   absolute calibration if reached (T8).
3. Keep the old runs' caches intact — everything is keyed by run stem.

## 6. Known gotchas (learned the hard way, don't rediscover)

- `tof` is time since acquisition start, not gamma flash; use `tof − tflash`
  and the standard late cut (>0.1 ms) for spectra.
- Inclusive-spectrum medians are NOT a gain proxy (threshold bias) — always
  use coincident sideband-subtracted spectra for gain statements.
- Beam-on/wall-active bunch selection first (see 02's `select_bunches`).
- TTree::Draw estimate caps at 16,777,216 — use uproot/hitcache for counts.
- PKUP has psTime==0 entries in this run — filter.
- The official-processing `waveform` branch is empty — hit-level only.
- lxplus condor fallback for the read pass exists (`lxplus/README.md`) if
  the desktop is busy: ~89 min/run, ~2.4 MB of caches come back.

## 7. Already answered — nothing blocks on Dylan

1. Scan log: pull it yourself from the DAQ machine (§3, exact scp command).
2. Scan scope: **plastics only**; liquid HVs fixed the whole run.
3. SiPM gating: **ON the whole run** (Dylan + RUN_NOTES agree).
4. Step grid, timings, reference windows: all in §3.
5. Nominal (non-scan) plastic HVs: per-channel 1275–1325 V (restored at
   19:05; exact per-channel values in the CSV `end_state` row and in
   README.md §"Detector naming": AL 1325, AR 1275, BL 1325, rest 1300).

You have ssh to `lxplus` (EOS data, condor) and `daq_lxplus` (scan logs,
DAQ config) — if something is missing, go look for it there before asking.
