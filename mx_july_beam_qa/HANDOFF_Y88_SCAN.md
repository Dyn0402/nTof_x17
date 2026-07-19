# HANDOFF — Y-88 source scan: amplitude spectrum analysis per detector

> **STATUS 2026-07-18 (updated):** T0-T4 essentially DONE, incl. LIQ. The four
> runs were REPROCESSED with the LIQ PSA UserInput (official `RunProcessing.sh`;
> recipe now in `../ntof_daq_processing/PROCESSING.md`) — LIQ trees present, and
> the new PSA also cleaned the walls (~10× less noise). Analysis re-run on the
> reprocessed files. Scripts `21/22/23`; `calib/y88_edges_<run>.json`,
> `calib/y88_energy_calib.json`; figures `figures/21_y88/`; `report/y88_report.pdf`.
> Findings: both plastic edges seen (699 & 1612 keVee, measured ratio ~2.4,
> linear ~10-20%); **plastic HV was RAISED** (24-39 mV/MeVee); **first LIQ scale**
> (699 bump 22-26 mV, 32-37 mV/MeVee, consistent all arms); clean wall 699 bumps
> 20-30 mV. Only OPEN item for Dylan: exact 224476-79 HV values (not in the data)
> to transport plastics to nominal; optionally a same-PSA (224489) wall MIP for
> the wall cross-check.

**Audience: a fresh Claude session on Dylan's desktop, picking this up cold.**
Read `README.md` in this directory first (machinery + conventions), then this
file. The infrastructure from the run-224489 analysis (hit cache, adc_mv,
per-channel spectra) is all reusable; the physics and the event selection are
different — this is a SOURCE run analysis, not a beam analysis.

## 0. What this is

An explicit Y-88 source scan of each detector arm: dedicated ~13-min runs
with ONE ⁸⁸Y source placed **between the two plastic scintillator bars of the
arm, near the middle** (so both bars are illuminated ~symmetrically, and the
same arm's wall (in front) and LIQ vessel (behind) also see the source).
Run list (from Dylan, DAQ page; campaign `X17_measurement`):

| run | arm | title | start–end (local, 2026-07-17) | triggers |
|--------|---|-------------------------|---------------------|------|
| 224476 | A | PLASTIC CALIBRATION A | 10:25:57–10:38:55 | 1507 |
| 224477 | B | PLASTIC CALIBRATION B | 10:46:45–10:59:41 | 1502 |
| 224478 | C | PLASTIC CALIBRATION C | 11:12:43–11:26:38 | 1501 |
| 224479 | D | PLASTIC CALIBRATION D | 11:29:19–11:42:16 | 1503 |

Facts and immediate implications:

- **Beam-off** (protons column 0; ~1500 triggers in ~13 min ≈ 1.9 Hz timer
  trigger, 20 ms windows ⇒ ~30 s of live time per run). The §2/§3 source-run
  gotchas (empty index/PKUP, no bunch selection, no tflash) fully apply.
- Taken the **morning of 2026-07-17** — after HV scan 1 (07-16) and ~6 h
  BEFORE run 224489 (17:39). Plastic HVs presumably the per-channel nominals
  (AL 1325, AR 1275, BL 1325, rest 1300 V) — **confirm with Dylan** (§6).
- **Unknown hardware state — ask (§6)**: were the FIFO signal path and the
  A/D cabling fix already in place at 10:30–11:45 that morning, or did that
  work happen between these runs and 224489? This changes the expected
  amplitude scale by the per-PMT FIFO ratio (×1.13–1.65) and whether the A/D
  channel mapping matches 224489. If unclear from Dylan, the data itself can
  tell: the A/D duplication signature (`17_duplication_veto.py` logic) is a
  cabling fingerprint.
- Check whether official processed files exist for these runs and **whether
  they contain the LIQ trees** — the official PSA of that morning may
  predate R. Mucciola's LIQ UserInput (224489 had to be processed privately
  in Dylan's user EOS for that reason). If LIQ trees are absent and LIQ
  edges are wanted, the runs need a reprocessing pass like 224489's.
- Source between the bars near the middle ⇒ ONE LIQ position only — the
  edge-vs-position gradient measurement (§5 T3.3) is not available from
  these runs; instead compare the single-position LIQ edge against the
  triples position map (19 cache `liq_pos`) at the source's location
  (mid-vessel, i.e. mid-gradient).
- ~30 s effective live time per run at source rates: statistics will be
  modest — check raw hit counts per tree right after extraction before
  planning fits.

⁸⁸Y emits two gammas: **898.04 keV** and **1836.06 keV**. In organic
scintillators (PVT plastic, LAB liquid, the SiPM-wall bars) there is no
photopeak — the observable landmarks are the **Compton edges**:

- 898 keV  → **699 keV** edge
- 1836 keV → **1612 keV** edge  (weaker rate, cleaner separation)

(Edge energy = 2E²/(m_e c² + 2E), keV electron-equivalent = keVee.)

## 1. Why we care / what it plugs into (run224489 context)

The triples analysis (`19_liq_triples.py`, `19d_mip_step_panels.py`,
`calib/pss_mip_calib_run224489.json`) gave the first absolute plastic scale:
MIP MPV 3.3–10.3 mV ⇒ 0.65–2.05 mV/MeV per PMT at nominal HV, with 15–40%
uncertainty on the weaker PMTs. A Y-88 edge at a known keVee is an
**independent absolute calibration** of every channel it reaches:

- **SiPM walls**: MIP (through-going, ~0.3 cm PVT ⇒ ~0.6 MeV) sits at
  29–39 mV on B/C — i.e. roughly 50–65 mV/MeV, so expect the 699 keV edge
  around ~35–45 mV and the 1612 keV edge around ~80–105 mV. Both comfortably
  above threshold. Direct cross-check of the wall MIP scale.
- **Plastics (PSS)**: at 0.65–2.05 mV/MeV nominal-HV gains, the 699 keV edge
  lands at **0.45–1.4 mV — below or at the ~1.5 mV acquisition threshold**
  (~50 ADC). The 1612 keV edge (1.0–3.3 mV) is marginal on the low-gain PMTs
  (DR, BL). **If the scan was taken at nominal HV, expect to see little or
  nothing on the plastics; if it includes raised-HV steps, analyze those.**
  Use the measured power laws (n per PMT in `pss_mip_calib` /
  `12_hvscan` cache) to transport any raised-HV edge back to nominal.
- **Liquids (LIQ)**: no absolute scale exists yet at all — a Y-88 edge would
  be the first. ALSO: the triples found a 1.5–2× gain-vs-position gradient
  toward the PMT (README §"Key results (run 224489)" pt. 6). If the source
  was placed at several positions along a vessel, the edge-vs-position curve
  measures that gradient with a monoenergetic landmark — note positions
  carefully in the run table.

## 2. Getting and extracting the data

- Runs live on EOS like the beam runs. Check BOTH
  `lxplus:/eos/experiment/ntof/processing/official/done/runXXXXXX.root` and
  Dylan's user area `lxplus:/eos/user/d/dneff/ntof_x17_processing/` (224489
  needed the new PSA and lived in the user area — ask which applies).
  `ssh lxplus` works passwordless; scp to `~/x17/beam_july/data/` (203 GB
  free on that disk; the root filesystem is nearly full — never put data
  in `~`).
- Extract with `fastread/extract_hits <run.root>` (already built for this
  desktop's ROOT 6.30; extracts WAL/PSS/LIQ trees). Source runs are short —
  extraction is fast.
- **Gotcha — extractor may FATAL on source runs**: `write_index()` exits if
  both `index` and `PKUP` are empty/absent, which is plausible with no beam.
  If it dies there, patch it to write empty index arrays instead (2-line
  change) — nothing downstream needs intensity for source runs.

## 3. Event selection — do NOT reuse the beam selection

The beam-analysis selection chain **does not apply**:

- `select_bunches` (02) needs beam-on/wall-active logic and the 01 cache —
  meaningless without beam. Use ALL bunches/triggers.
- `tof − tflash > 0.1 ms` late cut — meaningless; `tflash` is a gamma-flash
  fit and will be garbage. Ignore tof entirely for spectra (or use it only
  to check rate uniformity across the acquisition window).
- PKUP `psTime`: if present, still useful to time-order the run (e.g. HV
  changes mid-run); beware denormal ~0 entries (filter `t > 1e18`).

Write a new script (suggest `21_y88_spectra.py`, argv = run file) rather
than bending 01: load per-channel `amp` via `hitcache.load`, histogram in
BOTH log bins (overview) and **linear bins (edge extraction — this matters,
see below)**, convert to mV via `adc_mv.mv_factors(run_file)` (regenerates
the per-run JSON automatically; LIQ included).

## 4. Edge extraction — lessons already paid for (don't rediscover)

1. **Work in linear amplitude.** The run224489 MIP analysis found log-binned
   spectrum modes sit 15–35% above the true linear-space landmark (dN/dlogA
   = A·dN/dA). Same trap applies to Compton edges. Extract on linear-binned
   dN/dA.
2. **Edge estimator**: the resolution-smeared Compton edge has no unique
   "position" — pick ONE convention and state it. Recommended, in order of
   robustness at modest statistics:
   - fit the shoulder with a Gaussian-smeared step (erfc) + linear
     background; quote the step center;
   - or the half-height point of the shoulder (between local plateau and
     valley);
   - or the extremum of the smoothed derivative −dN/dA.
   Cross-check at least two on a few channels; they differ by O(σ/2) — if a
   simulation-anchored convention is needed later, that offset is the
   systematic.
3. **Error bars + bootstrap**: propagate Poisson errors per bin; bootstrap
   the edge position (fixed RNG seed, ~200 resamples) exactly as
   `19d_mip_step_panels.py` does for the MIP peak — argmax/half-height
   estimators are bin-quantized, floor the error at ~half a bin.
4. **Background**: source runs still see cosmics + room background. If
   feasible take/ask for a background (no-source) run and subtract
   rate-normalized spectra; otherwise fit the local background under the
   shoulder. Wall channels: the top/bottom coincidence (detn pairs
   (1,2)(3,4)… view the same bars, `16_wall_vertical.py` machinery) is a
   powerful cosmics/noise discriminator if needed — but note gammas also
   fire both ends.
5. **Two edges = linearity check**: where both 699 and 1612 keV edges are
   visible in one channel, their mV ratio tests linearity (expect 1612/699 =
   2.31 × any nonlinearity).

## 5. Task list

- **T0** — Locate runs 224476–79 on EOS (official first, then user area);
  pull to `~/x17/beam_july/data/`; extract hit caches (patch `write_index`
  if needed, §2); report per-tree hit counts + whether LIQ trees exist.
- **T1** — `21_y88_spectra.py`: per-channel linear+log amplitude spectra per
  run, mV-calibrated; overview figure per run (all channels of the source's
  detector + a far detector as background reference).
- **T2** — Edge extraction (§4) per channel: 699 keV edge (and 1612 keV
  where visible) in mV, with bootstrap errors; store to
  `calib/y88_edges_<run>.json` (per channel: edge_mv, err, convention, HV).
- **T3** — Energy calibration: mV per keVee per channel. Cross-checks, in
  priority order:
  1. walls: compare with the MIP scale (MIP ≈ 0.6 MeV in 0.3 cm) — one
     number per channel, should agree at the tens-of-% level;
  2. plastics (raised-HV runs only): transport to nominal with the per-PMT
     n and compare with `pss_mip_calib_run224489.json` (0.65–2.05 mV/MeV);
  3. LIQ: first absolute scale; if multiple source positions exist, edge
     vs position vs the triples gain-gradient map (19 cache `liq_pos`).
- **T4** — Write-up: extend README "Key results" with a Y88 section; figures
  under `figures/21_y88/`; per-detector calibration table. If it merits it,
  a report/slides pass like `report/run224489_report.tex` (repo now has
  full texlive; `pdflatex` works).

## 6. Open questions for Dylan (ask before deep work)

1. Plastic (and LIQ) HVs during runs 224476–79 — the nominal set, or raised
   for the calibration?
2. Hardware state that morning: FIFO installed? A/D cabling already fixed?
   (See §0 — the duplication fingerprint can also answer this from data.)
3. Official or user-EOS processing? Do the official files include LIQ trees
   (pre- or post-Mucciola-UserInput)?
4. Was a background (no-source) run taken nearby in time?
5. If the plastic edges turn out to be threshold-buried at nominal HV (§1
   expectation: 699 keV edge at 0.45–1.4 mV vs ~1.5 mV threshold), flag to
   Dylan that a raised-HV Y88 repeat would complete the plastic absolute
   calibration — the per-PMT power laws can transport it back to nominal.

## 7. Environment quick facts (desktop)

- Repo venv: `venv/` (`.venv` is a symlink to it); numpy 2.2.6, uproot,
  scipy, matplotlib all present.
- `fastread/extract_hits` built (ROOT 6.30, no-IMT include fix applied).
- Data disk: `~/x17/` (NTFS, 203 GB free). Beam data + hitcaches:
  `~/x17/beam_july/data/`. Analysis outputs for humans:
  `~/x17/beam_july/<name>_qa/` (see `run224489_qa/` for the pattern).
- `ssh lxplus` (EOS) and `ssh daq_lxplus` (DAQ machine, scan logs) both
  passwordless.
- Full run224489 context: `HANDOFF_RUN224489.md`, README §"Key results
  (run 224489)", `report/run224489_report.pdf`.
