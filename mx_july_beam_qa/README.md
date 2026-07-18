# mx_july_beam_qa — July 2026 beam: SiPM wall + plastic scintillator analysis

**Status (2026-07-16): first-run analysis complete; MIP calibration achieved on arms B/C;
report + slides ready for the 7/17 collaboration presentation. This README is the handover
document.**

## Context

First beam data of the n_TOF EAR2 X17 campaign. Goal: cross-calibrate the SiPM trigger
walls (WAL) and back plastic scintillators (PSS) to MIPs using coincidences.

- Data: official n_TOF processed files, `lxplus:/eos/experiment/ntof/processing/official/done/runXXXX.root`
  (hit-level PSA results; `waveform` branch exists but is EMPTY in official files).
  Local copy: `~/x17/beam_july/data/run224404.root` (13 GB, verified vs EOS).
  Run numbers from the DAQ website; July beam runs are ~224xxx.
- File layout: trees `WALA-D` (SiPM walls, detn 1-8), `PSSA-D` (plastics, detn 1-2),
  `SILI` (Si monitor), `PKUP`, `index` (per-bunch), `DAQsettings` (channel map, full
  scale). One entry = one hit; grouped by `BunchNumber`; acquisition = 20 ms/bunch at
  1 GS/s; `tof` is time since acquisition start, NOT since gamma flash; gamma flash at
  ~10.8-11.9 us, per-hit fitted arrival in `tflash`.
- ADC -> mV: per channel `fullScalemV/65536` ~ 0.0305 mV/count (see `adc_mv.py`,
  `calib/adc_to_mv_run224404.json`).

## Detector naming / geometry (partially confirmed in data)

- Tree letter = arm (4 arms around beam; per arm: Micromegas -> SiPM wall -> back
  plastic -> LS-1 -> LS-2 in the back-first stack assumed for the diagrams).
- WAL: 20 vertical bars, 16 read out, summed in 4 groups of 4 bars; each group read by
  top+bottom SiPMs -> detn pairs (1,2)(3,4)(5,6)(7,8) = groups 1-4 ordered left->right
  seen from the back (CONFIRMED in data: block-diagonal coincidence matrices, identical
  bar-fractions within pairs).
- PSS: detn 1 = left bar, 2 = right bar (seen from back; per Dylan).
- Cross-arm corner hotspots => A-D and B-C are ADJACENT arms; ring order consistent with
  A->D->C->B. (Earlier "A-D opposite" reading was wrong.)
- Current plastic PMT HV: AL 1325, AR 1275, BL 1325, BR/CL/CR/DL/DR 1300 V.

## Key results (run 224404)

1. **Coincidence method works**: every WALx-PSSx pair shows a sharp dt peak (-8..-18 ns,
   rms 1.5-3.5 ns) on a flat combinatorial pedestal; sidebands (|dt|>100 ns) give the
   background; excess plateaus by W ~ +-10-15 ns. Purity ~90% for tof>1 ms.
2. **Per-channel time offsets** measured and stored: `calib/time_offsets_run224404.json`
   (convention: dt_cal = (t_wall - t_pss) - offset). Spread up to ~9 ns within an arm.
3. **SiPM-wall MIP calibration, arms B/C**: clean Landau-like peaks 29-39 mV
   (~960-1270 ADC) in all 16 channels (see 07 output table / `07_mip_run224404.npz`).
4. **Arms A/D**: 6-8x fewer true coincidences, NO MIP bump — coincident spectrum traces
   the inclusive shape. Cause OPEN (see flags). Wall-only top-bottom coincidences show
   MIP-scale bumps on ALL arms => SiPM response/gain healthy everywhere; but top+bottom
   view the same bars, so gammas also fire both ends — NOT proof of through-going flux.
5. **Plastic physics note** (Dylan): plastic is at the BACK => wall-plastic coincidence
   is a MIP selection for the WALL only; plastic spectra are hit spectra. True plastic
   MIP calibration needs the LIQ readout (wall x plastic x LS-1).
6. **Plastic PMT relative gains** within +-30% (AL lowest 0.74, DL highest 1.26); HV
   equalization table in 10 output (G ~ V^7 assumed — needs a 2-point HV scan).

## Key results (plastic HV scan, run224466, 2026-07-16)

Scan log + handoff doc: `~/beam_july/scint_hv_scan/2026-07-16_13-32-34_plastic_scan_1/`
(pulled from the DAQ machine). 9 steps x ~10 min, all 8 plastic PMTs at a common HV
(pass 1: 1600->1200 V, SiPM flash-gating OFF; pass 2: 1550->1250 V, gating ON), plus
two free reference windows in the same run: pre-scan (all at 1500 V) and post-scan
(per-channel nominal). Bunches assigned to steps by PKUP `psTime` (ns UTC; local =
UTC+2). Analysis: `12_plastic_hv_scan.py` (+`12b` figures in `figures/12_hv_scan/`).

1. **Wall MIP peaks are immune to the plastic HV** (the question that motivated the
   scan): ch-summed sideband-subtracted wall MIP peak stays within +-1 amplitude bin
   (+-2.6%) on all 4 arms across 1200-1600 V — a 3x change in tag rate — and across
   the gating boundary. The B/C MIP calibration transfers unchanged.
2. **Per-PMT gain power laws measured**: coincident (wall-tagged, sideband-subtracted)
   plastic median ~ V^n with n = 4.8-6.9 per PMT, passes consistent, pre-scan 1500 V
   point reproduces step 2 exactly. (Inclusive-spectrum medians are NOT a gain proxy —
   threshold bias flattens them; use the coincident spectra.)
3. **HV equalization table** (12b output, targets fleet geo-mean coincident median):
   AL -22, AR -33, BL +51, BR -21, CL -120, CR +7, DL +3, DR +117 V. Spread at current
   nominals is 2.8x (CL highest, DR lowest) — supersedes the G~V^7 guess in 10.
4. **No rate plateau up to 1600 V** (threshold sits inside the spectrum) and the
   late-tof (>0.1 ms) rates are unaffected by the flash gating; flash-window wall
   occupancy drops only ~4-6% with gating ON (weaker effect than expected — check
   what M6.C actually blanks).

## Key results (run 224489, 2026-07-17: HV scan 2, FIFO path, first LIQ)

Second plastic HV scan (same 9-step ladder, SiPM gating ON throughout, scan log
`~/beam_july/scint_hv_scan/2026-07-17_17-41-11_plastic_scan_2/`), first run with
the liquid scintillators (LIQA-D, 1 ch each), plastics on the new linear
fan-in/fan-out (FIFO) instead of the BNC-T split, +~16 ns cable delay, cabling
fix after the A/D cross-wiring. Full task list + context: `HANDOFF_RUN224489.md`.
New scripts: `19_liq_triples.py` (+`19b`); 01/02/adc_mv extended to LIQ; 12
step tables now per-run (`SCANS`). No wall outage this run (0/1750 bunches).

1. **Cross-wiring fix CONFIRMED, A/D deficit explained** (17 on 224489): the
   equal-amplitude ±4 ns same-side duplication band on A/D is gone (flagged
   fractions ≤4%, B-control level, vs 23–31% on the crossed channels in
   224460), and all four arms now show equal wall-plastic excess (~1.7–1.8 M)
   and clean A/D wall MIP bumps with NO veto. The standing A/D question is
   closed: it was the cabling.
2. **Plastic MIP found — first plastic calibration** (19: WAL×PSS×LIQ triples,
   double sideband subtraction in wall-plastic dt AND LIQ dt): clear MIP humps
   that march with HV; modes at V≥1400 V transported to nominal V with the
   coincident-median n give MIP = AL 11.5, AR 8.9, BL 4.5, BR 8.4, CL 14.3
   (±78%, one outlier step), CR 8.0, DL 5.2, DR 3.7 mV (±10–25%).
   **DR/BL MIPs sit at/below the 4.9 mV trigger threshold** — equalize first.
   Absolute: 2.5 cm PVT ⇒ 5.05 MeV ⇒ 0.73–2.83 mV/MeV per PMT
   (`calib/pss_mip_calib_run224489.json`; normal-incidence, no path-length corr).
3. **FIFO gain ratio is NOT 2×**: same-HV coincident medians give fleet
   geo-mean ×1.40, strongly per-PMT (DL 1.13 … AL 1.65), ~flat vs HV.
4. **New HV equalization** (12b, target 64.2 mV fleet geo-mean): AL −54,
   AR −64, BL +84, BR −38, CL −142, CR −7, DL +68, DR +133 V (supersedes the
   224466 table; exported to the DAQ repo via 12c). Wall MIP again immune to
   plastic HV (±1 bin, all arms; arm A now measurable too). Coincident-median
   power laws n = 3.8–7.1 (PSSB1/PSSD2 inclusive-median anomalies are pure
   threshold bias — coincident fits are clean).
5. **LIQ timed in** (02 wide scan + 19 per-channel): LIQ leads the plastic by
   ~30 ns, |t_wall − t_liq| ≤ 7 ns everywhere; per-channel offsets (rms
   3.2–4.5 ns) in `calib/liq_offsets_run224489.json` (new file, wall-plastic
   offsets JSON untouched).
6. **LIQ gain-vs-position gradient CONFIRMED** (19 position map, triple-tagged):
   ~1.5–2× median-amplitude variation along the vessel axis, higher near the
   PMT, and the gradient axis flips with the surveyed orientation — horizontal
   (toward wall group 4 = +u) on A/D, vertical (toward top) on B/C. Matches the
   Geant as-built geometry (vessels vertical PMT-up on B/C, horizontal PMT+u
   on A/D).
7. **Wall-plastic offsets shifted by a single common −22.27 ns (rms 0.22 ns,
   64/64 channels)** — the new plastic path is a clean common delay; new
   offsets in `calib/time_offsets_run224489.json`.
8. **−60 ns satellite: NOT a plastic-cable reflection.** Its absolute dt
   position stayed at −60…−68 ns while the main peak moved −22 ns, so it is
   injected downstream of the FIFO (digitizer side); it GREW to 19–34% of the
   main peak (was 6–9%) — worth a scope look. It stays outside signal and
   sideband windows.

## OPEN FLAGS / questions for the collaboration

- **[ ] SiPM wall outage, run 224404**: all 32 WAL channels dead for bunches ~643-2212
  (46% of beam-on bunches; WALD ch5 only partial). Present in raw data (EOS reprocess
  checked identical). Ask DAQ/shift crew; check other runs. Analyses mask to bunch>2212.
- **[ ] Top/bottom SiPM assignment ASSUMED**: odd detn (first DAQ card slot) = top.
  Confirm from cabling sheet.
- **[ ] Which 4 of 20 wall bars are unread** (assumed 2 per edge) + physical positions
  of arms A-D + per-arm stack order (standard vs back-first) — needed to settle the A/D
  question.
- **[x] A/D coincidence deficit — RESOLVED (run 224489)**: it was the crossed cabling;
  after the fix all four arms show equal excess and clean MIP bumps (see 224489 §1).
- **[ ] dt satellite bump at ~-60 ns** (WALB-PSSB, WALC-PSSC, late tof): NOT a
  plastic-cable reflection (didn't move with the −22 ns path change in 224489, and grew
  to 19–34% of the main peak) — likely injected at/after the FIFO; scope check wanted.
  Excluded from signal window and sidebands.

## Fast read pass (2026-07-16): C++ hit cache + vectorized pairing

The read pass used to take ~89 min/run (six scripts each re-reading the 10-18 GB
root file with uproot, then per-bunch python pairing loops). It now runs in ~10 min
total on the laptop:

1. **Extract once** (C++, ~2-4 min/run): `fastread/extract_hits` reads the 9 needed
   branches of every hit tree in parallel threads, sorts by (bunch, tof), and dumps
   flat `.npy` arrays to `<data_dir>/hitcache/<run_stem>/` (~8 GB/run, memmapped by
   the scripts). Build with `make` in `fastread/` (needs `root-config` in PATH;
   local ROOT 6.36.06 works). `satuflag`/`pileup1` are stored as uint8 (!=0).
2. **Analyze** (~seconds-minutes/script): all six read-pass scripts load via
   `hitcache.py`, which memmaps the cache when present and falls back to the old
   chunked-uproot read when it isn't (so the lxplus/condor path still works
   unchanged, minus the speedup). The per-bunch pairing loops are replaced by one
   global sorted-key search (`hitcache.iter_pairs`; key = bunch*1e9 + tof ns),
   yielding the exact same pair windows as the old code.

Validated on run224404 against the pre-existing caches: 01/03/06/07/09 and the
offsets JSON are bit-identical. 02 is bit-identical after summing over tof-regions;
the region split itself moves ~0.002% of hits between adjacent decade regions from
one benign convention change — the per-bunch `tflash` is now the earliest-tof hit's
tflash instead of the first hit in file order (tflash is per-hit and varies ~µs
within a bunch, so both picks are arbitrary; the new one is deterministic).

Timing on the laptop (8 cores, run224404 = 299M hits): extract 225 s, then
01: 146 s, 02: 146 s, 03: 29 s, 06: 54 s, 07: 36 s, 09: 46 s (~11 min total
including extraction, vs ~89 min for the old read pass on lxplus).

## Pipeline (scripts, in order; all cache to `cache/*.npz`, figures to `figures/`)

| # | script | what it does |
|---|--------|--------------|
| 01 | `01_signal_qa.py` (+`01b`) | chunked QA pass: per-channel amp/area/fwhm/tof hists, rates, flash zoom, stability |
| 02 | `02_coincidence_scan.py` (+`02b`) | WALxPSS dt hists per tree pair x tof-region x pulse type; window scan |
| 03 | `03_time_offsets.py` | per-channel (wall ch x pss bar) dt offsets -> `calib/time_offsets_*.json` |
| 04 | `04_channel_mapping.py` | cache-only: wall-ch<->bar excess matrices, cross-arm coupling -> naming/geometry |
| 05 | `05_sideband_diagram.py` | explanatory figure of the sideband method on real data |
| 06 | `06_wall_geometry_test.py` | within-wall 8x8 + cross-arm (A-D, B-C) channel coincidence matrices |
| 07 | `07_mip_amplitude.py` (+`07b`) | MIP spectra: wall x bar pairs, offsets applied, +-8 ns window, sideband-subtracted; also renders 06 figures |
| 08 | `08_inclusive_vs_coinc.py` | shape comparison inclusive vs coincidence (uses 09 late cache when present) |
| 09 | `09_late_inclusive.py` | late-tof inclusive spectra + wall-only top-bottom pair spectra |
| 10 | `10_pmt_gain.py` | plastic PMT raw-vs-coinc overlays, relative gains, HV suggestions |
| 11 | `11_concept_diagrams.py` | top-down concept sketches (current selection; with LIQ readout) |
| 12 | `12_plastic_hv_scan.py` (+`12b`) | plastic HV scan (run224466 x CAEN log): per-step spectra/rates, wall-MIP stability, gain power laws, HV equalization |

The whole read pass now also runs locally in ~10 min via `./run_readpass.sh <run.root>` (C++ hit-cache extraction + vectorized pairing, see "Fast read pass" above). It can still run on lxplus/HTCondor next to the
EOS data instead of pulling the 13-18 GB file local — see `lxplus/README.md` (benchmarked
on run224461: ~89 min/run, only ~2.4 MB of caches come back, plotting stays local).

Shared helpers: `adc_mv.py` (ADC->mV factors), `hitcache.py` (binary hit cache /
uproot fallback + global pair finder; see "Fast read pass" above). Scripts
03/06/07/09 import `02_coincidence_scan.py` via importlib for `select_bunches`
and friends.
Selection everywhere: beam-on & wall-active bunches (>2212) and tof-tflash > 0.1 ms.
Python: repo venv `.venv` (numpy 1.x: use `np.trapz`; uproot; python-pptx; PIL).

## Deliverables

- `report/mip_report.pdf` — 5-page LaTeX writeup (rebuild: `pdflatex mip_report.tex` in `report/`).
- `report/hv_scan_report.pdf` — 8-page LaTeX writeup of the run-224466 plastic HV
  scan (scan timeline, gain power laws with linear+log spectra in mV, HV
  equalization table + construction diagram, target-choice rationale, wall-MIP
  stability, gating check, first trigger-threshold recommendation 4-5 mV).
- `slides/mip_slides.pdf` — 16-frame beamer deck for 7/17 (rebuild in `slides/`).
- `slides/mip_slides.pptx` — same content in Dylan's usual Google-Slides-derived
  template (regenerate: `python slides/build_pptx.py`; template read from
  `~/Downloads/X17 Overview 7_13.pptx`). Frozen at 14 slides — the two concept-diagram
  frames and the toned-down A/D wording are in the BEAMER deck only (per Dylan:
  "stop editing the pptx, work with the latex slides").

## Suggested next steps

1. Quantify pre-flash cosmic wall-plastic coincidence rates per arm (extends 02
   machinery; discriminates plastic response vs flux composition for A/D).
2. When LIQ readout arrives: wall x plastic x LS-1 triple coincidence -> plastic MIP
   calibration; also settles the A/D absorption hypothesis.
3. Plastic HV scan -> calibrate G(V), apply equalization (10 output).
4. Apply the B/C MIP constants + top-bottom group landmarks as a wall energy scale;
   propagate to trigger-threshold settings.
5. Re-run pipeline on further runs (all scripts take the run file/stem as argv[1]).
