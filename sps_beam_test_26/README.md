# sps_beam_test_26 — det4 in the H4 test beam, SPS, 2026-08

det4 (`mx17_E`) was taken to the SPS North Area H4 line and run parasitically
inside the banco/P2 uRWELL test beam (`TB_July2026_H4`), which supplies the
trigger and the reference tracker. This directory holds everything about that.

## Layout

- **`det4_sps_assessment/`** — the *pre-trip* feasibility study: is det4 worth
  putting in a beam, where would it sit, what does the mount and the beam
  geometry look like. Written 07-31/08-01 from June cosmic-bench data plus the
  P2 beam-profile CSVs. Headline: 62 % of det4's area does not amplify, in
  fixed ~35 mm stripes, and no HV setting fills them in — but the live 38 % is
  a normal detector. Reports: `DET4_SPS_ASSESSMENT.md`,
  `SPS_BEAM_GEOMETRY_2026-07-31.md`, `SPS_MOUNT_2026-07-31.md`. Its
  `mapping_check/` subdirectory has since grown into the working scripts for
  the beam data itself (`effmap.py`, `driftscan_run61.py`,
  `resist_scan_run61.py`, `gain_scan_run61.py`, all keyed off
  `run61_conditions.py`) and the two in-beam reports
  `DET4_EFFICIENCY_H4_2026-08-01.md` and `DET4_URW_MAPPING_2026-08-01.md`.
  Moved here from `mx_june_cosmic_qa/` on 08-02; path references were updated
  with it.
- **`analysis/`** — analysis of the beam data. **`analysis/README.md` is the
  entry point**: it indexes the reports, the script chain and the conditions
  registry, and lists the traps this directory has already fallen into.
  - `RUN_TIMELINE.md` — the narrative of the beam time, each claim marked with
    whether the machine record confirms it, plus the configuration epochs every
    analysis has to key off. §3b carries the corrections established from the
    beam data itself.
  - `HV_AND_BEAM_RECORD.md` — what survives of the HV and SPS beam records and
    what does not. Read before assuming you have a trace.
  - `datasets.py` — every flat-mount dataset as a (mount, gas, drift, resist)
    condition, with HV plateau windows and DAQ settings. The single source of
    truth; analysis scripts must not restate any of it.
  - `M70V_FLAT_ANALYSIS.md`, `FLAT_CF4_RUN63.md`, `RAW_RUN71_STATUS.md`,
    `RAW_RUN71_PHYSICS.md` — the charge-spreading measurement, run by run.
  - `harvest_inventory.py` → `run_inventory.json` — objective run/sub-run
    inventory harvested from banco.
  - `build_run_map.py` → `run_map.csv` — every det4 HV point with the P2
    run/sub-run that was live at the time.
  - `beam_record_coverage.py` — per-day coverage of the beam record (spill
    cycles, intensity points, TAX samples) for any directory of day-files, so
    the archive and banco's live logs can be compared with one command.
    `--gaps` lists intra-day holes.
  - `tax_windows.py` — H4 beam-stopper open/blocked windows. **This is how you
    date a zone access**; do not infer them from DAQ gaps any more.
  - `push_beam_record_to_banco.sh` — copy the archive back up to banco,
    skipping exactly the four files banco's `beam_bridge.py` overwrites every
    20 s. Read the header before changing it.

## Where the data is

| | |
|---|---|
| on banco | `banco_cern:/local/home/banco/P2_data/TB_July2026_H4/` |
| EOS backup (complete, incl. pruned runs) | `root://eospublic.cern.ch//eos/experiment/ntof/data/x17/p2_sps_july/runs/` |
| local outputs + paired caches | `/media/dylan/data/x17/sps_run53_det4_check/` (mirror `~/x17/...`) |
| **SPS beam record (the authority)** | `…/sps_run53_det4_check/records/beam/backfill_nxcals/` — full period 07-14→08-02, NXCALS-backfilled 08-02 |

The beam record deserves one warning of its own. banco's copy is a mirror of
EOS, published by a watcher on lxplus, and that watcher died for 24 h over the
det4 data-taking. **`mx17-daq` runs a second, independent NXCALS client that
never went down** — that is what the gap was recovered from, and it is where to
go if it ever happens again. `analysis/HV_AND_BEAM_RECORD.md` §5 has the whole
recipe; the staging tree is on daq at `~/dylan/sps_backfill/`.

Local outputs are **one directory per measurement condition** —
`<mount>_<gas>__<runs>`, e.g. `rot25_ArCF4iso_88-10-2__run61_1606on/`. A
condition is a (mount angle, gas) pair, because those are the two things that
were changed during the beam time and either one invalidates a comparison. The
key is `CONDITIONS.md` at the root of that directory; the run_61 conditions are
defined in code, once, in `det4_sps_assessment/mapping_check/run61_conditions.py`.

det4 reads out on **FEU3**; the uRWELL reference is FEU1 (front) and FEU5
(back). banco's auto-pipeline does not decode FEU3 past run_56 — det4 always
has to be decoded by hand and paired against the uRWELL tracks.

## Three things to know before writing anything here

0. **We controlled only our own HV.** banco started every run and sub-run on
   their own schedule; we moved our voltages underneath it. Any det4 quantity
   has to be joined to a P2 sub-run by wall clock — `run_map.csv` is that join.


1. **The DAQ config lies on this campaign.** `run_config.json` carries stale
   gas and mount angle, and a `start_time` that is the last restart. Take HV
   from `hv_monitor.csv` and from the per-detector scan logs, and say which.
   See the authority rules at the top of `analysis/RUN_TIMELINE.md`.
2. **The repo reconstruction rule still applies** (`../CLAUDE.md`,
   `../RECONSTRUCTION_BASIS.md`): no position, angle or drift depth out of
   `combined_hits` times. Everything here so far is efficiency, amplitude and
   discharge rate — QA-side quantities, which hits are the right input for —
   plus uRWELL-track-referenced pointing, where the *reference* geometry comes
   from the uRWELL, not from det4's hit times.
