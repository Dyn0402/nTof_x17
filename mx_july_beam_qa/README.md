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
   double sideband subtraction in wall-plastic dt AND LIQ dt; per-step linear
   spectra + error bars in 19d, `figures/19_triples/steps_linear/`): a single
   significant peak tracks the gain power law step after step while the fixed
   acquisition threshold stays put. Calibration = smoothed peak of the
   gain-aligned summed linear spectrum (V≥1400 V steps), Poisson-bootstrap
   stat errors, per-step transport scatter as the fuller uncertainty:
   MPV = AL 10.3±1.5, AR 8.1±1.2 (scatter 42% — flagged), BL 3.7±0.5,
   BR 5.9±0.5, CL 6.5±1.6 (38%), CR 6.3±0.3, DL 5.0±0.6, DR 3.3±0.3 mV.
   Validated against the no-MIP-selection pair-coincident median: same power
   law, factor ~10 below (`figures/19_triples/mip_vs_v_median_compare.png`).
   NOTE: log-binned modes are 15–35% above the linear MPV (mode of dN/dlogA ≠
   MPV of dN/dA); superseded values kept as `mip_mv_logmode` in the JSON.
   **DR/BL MPVs sit below the 4.9 mV trigger threshold** — equalize first.
   Absolute: 2.5 cm PVT ⇒ 5.05 MeV ⇒ 0.65–2.05 mV/MeV per PMT
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

## Key results (Y-88 source scan, runs 224476-79, 2026-07-17 AM)

Dedicated ~13-min source runs, one ⁸⁸Y source between the two plastic bars of
one arm per run (224476=A, 224477=B, 224478=C, 224479=D). Beam-off; the source
is bright, so the illuminated-arm plastics have >3 M hits — statistics are not
the limit the handoff feared. The four runs were **reprocessed** with the
official `RunProcessing.sh` + Mucciola's LIQ PSA UserInput (see
`../ntof_daq_processing/PROCESSING.md`), which adds the LIQ trees AND cleans the
wall reconstruction (WALL noise hits drop ~10×). Scripts `21_y88_spectra.py`
(spectra + cache), `22_y88_edges.py` (edge fits + `calib/y88_edges_<run>.json`),
`23_y88_energy_calib.py` (`calib/y88_energy_calib.json`); figures in
`figures/21_y88/`. All numbers below are from the reprocessed (new-PSA) files.

1. **Both Y-88 Compton edges seen on every illuminated plastic** (699 & 1612
   keVee), fit as TWO INDEPENDENT smeared steps (no ratio imposed). The measured
   1612/699 ratio (median 2.4, per channel 1.8-2.6) reproduces the expected
   2.31 — confirms the energy assignment from the data. 699 edge 20-27 mV is the
   robust primary landmark; response linear to ~10-20%.
2. **Plastic HV was RAISED for these runs** ("PLASTIC CALIBRATION" titles):
   24-39 mV/MeVee, ~20-25× the nominal-HV MIP scale (0.65-2.05 mV/MeV, 224489).
   Exact run HV is NOT in DAQsettings — **needed from Dylan** to transport to
   nominal and close the plastic absolute calibration.
3. **First LIQ absolute energy scale**: the 699 keVee edge is a clean bump at
   22-26 mV on all four liquids (32-37 mV/MeVee), consistent to ~10% arm-to-arm
   despite modest stats (6-110 k hits). New result, enabled by the reprocessing.
4. **Wall 699 keVee edge = clean bump at 20-30 mV** on source-facing SiPM
   channels (walls now noise-suppressed by the new PSA). Across all three
   detector types the 699 edge lands within 21-27 mV per arm (same source).
   The 224404 wall-MIP cross-check is retired (224404 = old PSA, mismatched);
   a same-PSA wall MIP from 224489 would restore it.
5. Reprocessed files in `~/x17/beam_july/data/` (LIQ trees present);
   `report/y88_report.pdf` written.

**Still needs Dylan:** exact plastic+wall HV for 224476-79 (not in the data) to
transport plastics to nominal; optionally a same-PSA (224489) wall MIP for the
wall cross-check.

## Key results (run_55 DREAM resist-HV scan + ZS optimization, 2026-07-18/19)

DREAM-side (not official-nTOF-files) analysis of the cyclical resist scan
r560→r520, scint-doubles trigger, 30 ms gate; DAQ machine died mid-run, data
recovered from EOS (24 subruns). Full stories: `HV_SCAN_RUN55_ANALYSIS.md`,
`ZS_OPTIMIZATION_RUN55.md`; working context for a follow-up model:
**`HANDOFF_RUN55_HV_ZS.md`**; slides `slides/hv_scan_run55_slides.pdf`;
scripts 25/25b (HV scan), 26/26b (ZS sim). Parallel tracking analysis:
`TRACKING_RUN55_ANALYSIS.md` (27*).

1. **DAQ FIFO comb**: no-ZS events → gate sampled only at 0–0.5 / 8–12 /
   16–28 ms; 2–6 ms (incl. the 3 ms target) unsampled. MM dead at flash at
   every HV; recovered by 8 ms at ≤550 V; C/D still sagged at 10 ms at
   555–560 V (saturation time grows with HV).
2. **³He capture flood** at thermal times floods det D (≳90 % of windows);
   at low HV capture blobs shrink into MIP-sized clusters (fake efficiency).
3. **Operating points for ≥3 ms**: A ≥560 V (no plateau; drift capped
   600–700, sparks at 800), B 550–555 (ringing unresolved), C 545–550,
   D 540–550.
4. **ZS: common-mode correction is mandatory** — in-beam wander 10–20×
   beam-off σ, coherent per Dream chip; `Feu_RunCtrl_CM=1` + **3.5σ**
   recommended → ~9 % volume, ~0.2 ms/event (10× readout, closes the comb),
   strip survival 92–99.7 % (losses = CM signal bias, threshold-independent).

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
| 21 | `21_y88_spectra.py` | Y-88 source scan (224476-79): per-channel linear+log mV spectra, source-arm overview figure, `cache/21_y88_<run>.npz` |
| 22 | `22_y88_edges.py` | Compton-edge fits: PSS double-erfc-step (699+1612 keVee, ratio 2.307), WAL Gaussian bump; bootstrap errors -> `calib/y88_edges_<run>.json` + diagnostic grid |
| 23 | `23_y88_energy_calib.py` | mV/keVee per channel, plastic linearity, wall edge-vs-224404-MIP cross-check -> `calib/y88_energy_calib.json` |

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
