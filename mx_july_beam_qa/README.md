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

## OPEN FLAGS / questions for the collaboration

- **[ ] SiPM wall outage, run 224404**: all 32 WAL channels dead for bunches ~643-2212
  (46% of beam-on bunches; WALD ch5 only partial). Present in raw data (EOS reprocess
  checked identical). Ask DAQ/shift crew; check other runs. Analyses mask to bunch>2212.
- **[ ] Top/bottom SiPM assignment ASSUMED**: odd detn (first DAQ card slot) = top.
  Confirm from cabling sheet.
- **[ ] Which 4 of 20 wall bars are unread** (assumed 2 per edge) + physical positions
  of arms A-D + per-arm stack order (standard vs back-first) — needed to settle the A/D
  question.
- **[ ] A/D coincidence deficit unattributed**: plastic response vs absorption between
  wall and plastic vs flux composition. Discriminators: pre-flash cosmic wall-plastic
  coincidences on A/D (gray curves in `dt_by_region.png` show they exist — quantify!),
  LIQ readout, plastic HV scan.
- **[ ] dt satellite bump at ~-60 ns** (WALB-PSSB, WALC-PSSC, late tof): afterpulse or
  reflection? Excluded from signal window and sidebands.

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

Shared helpers: `adc_mv.py` (ADC->mV factors). Scripts 03/06/07/09 import
`02_coincidence_scan.py` via importlib for `select_bunches`/`pair_dts`.
Selection everywhere: beam-on & wall-active bunches (>2212) and tof-tflash > 0.1 ms.
Python: repo venv `.venv` (numpy 1.x: use `np.trapz`; uproot; python-pptx; PIL).

## Deliverables

- `report/mip_report.pdf` — 5-page LaTeX writeup (rebuild: `pdflatex mip_report.tex` in `report/`).
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
