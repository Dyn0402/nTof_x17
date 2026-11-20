# The SPS CO₂-epoch gas is the ATLAS NSW mixture: Ar/CO₂/iC₄H₁₀ **93/5/2**

Settled 2026-11-20. **Use 93/5/2, not 95/3/2.** The rest of this note is the evidence, so the question does not have to
be reopened.

## What chamber E breathed at the SPS H4 test (1–3 August 2026)

| epoch | runs | gas (% by volume) | source |
|---|---|---|---|
| CO₂ (first day, incl. run 59's beam data before the 20:24 access) | 53–58, 59 (beam part) | **Ar/CO₂/iC₄H₁₀ 93/5/2**, the ATLAS NSW gas | Dylan; `run_config.json` on EOS |
| flush | 60 | mixed, changing | `GAS_FLUSH_TIMELINE.md` |
| CF₄ (next two days) | 61–72 | Ar/CF₄/iC₄H₁₀ 88/10/2 | `run_config.json` on EOS (confirmed by A. Kallitsopoulou) |

The DAQ's own `run_config.json` said `Ar/Iso 95/5` for every run; that string is stale and is never to be used.
Per-run authority in code: `sps_beam_test_26/analysis/datasets.py`; narrative: `sps_beam_test_26/analysis/RUN_TIMELINE.md`.

## Evidence that the NSW gas is 93/5/2

1. **Our own record.** The corrected `run_config.json` of runs 53, 55, 56, 57 and 58 on EOS
   (`/eos/experiment/ntof/data/x17/p2_sps_july/runs/run_NN/run_config.json`, rewritten 2026-08-16) reads
   `"Ar/CO2/Iso 93/5/2"`, with the note "the bottle flowing for this run was Ar/CO2/Iso 93/5/2". Re-read over xrootd on
   2026-11-20. Dylan confirmed the same mixture on 2026-11-20.
2. **ATLAS's own description.** The ATLAS Run 3 detector paper lists the NSW Micromegas gases as baseline
   **Ar:CO₂:iC₄H₁₀ 93:5:2** and backup Ar:CO₂ 93:7 (ATLAS Collaboration, JINST 19 (2024) P05063,
   [arXiv:2305.16623](https://arxiv.org/abs/2305.16623)).
3. **How the mixture came about, which fixes the numbers.** The NSW Micromegas were built for Ar:CO₂ **93:7**. For HV
   stability, 2 % of the CO₂ was replaced by isobutane, so argon stays at 93 and CO₂ drops from 7 to 5: 93/5/2. The
   operating point fell from ~570 V (93:7) to ~500–520 V on the resistive strips. Every ATLAS-side source found says
   93:5:2:
   - GIF++ long-term irradiation of NSW modules, [PoS 398 (2022) 757](https://pos.sissa.it/398/757/pdf)
   - ATLAS Micromegas performance with Run 3 data, [arXiv:2503.16735](https://arxiv.org/abs/2503.16735)
   - ATLAS MM summary at RD51, Dec 2023 (Iengo), [slides](https://indico.cern.ch/event/1327482/contributions/5701265/attachments/2765617/4817258/20231205-AtlasMMsummaryRD51-Iengo.pdf)
   - MPGD22 and PIC2023 talks (Mancini), [MPGD22](https://indico.cern.ch/event/1219224/contributions/5130184/attachments/2564953/4421732/MPGD22_Mancini.pdf), [PIC2023](https://indico.cern.ch/event/1190468/contributions/5599028/attachments/2732911/4751476/PIC2023_Mancini.pdf)
   - F. Vogel, PhD thesis (LMU), [pdf](https://www.etp.physik.uni-muenchen.de/publications/theses/download/phd_fvogel.pdf)

## Where "95/3/2" comes from, so it is not mistaken for a second source

- **Our own earlier notes.** From 2026-08-02 to 2026-11-20 the SPS notes, tables and the Magboltz inputs said
  Ar/CO₂/iso 95/3/2 ("confirmed 2026-08-02"). That was wrong; it was relabelled in commit `007fee02`.
- **The T2K gas.** Ar/**CF₄**/iC₄H₁₀ 95/3/2 is the well-known T2K TPC mixture. It has CF₄, not CO₂.
- **One ePIC talk.** "CyMBaL – Gas requirements" (M. Vandenbroucke, F. Bossù, MPGD-DSC Workfest, 2024-07-26,
  [pdf](https://indico.bnl.gov/event/20727/contributions/94339/attachments/56161/96121/ePIC_Collab_meeting_CyMBaL_Gas.pdf))
  has a slide line "Ar:iC₄H₁₀:CO₂ 95:3:2 (NSW gas)". Read literally that is 3 % isobutane and 2 % CO₂, which is neither
  93/5/2 nor Ar/CO₂/iso 95/3/2. It is a loose label in a talk about a different detector, not a source on the NSW gas.
  It is not ours.

## The refit on 93/5/2 (GAS9352, done 2026-11-20)

- **Magboltz:** `BASES['co2']` in `sps_beam_test_26/analysis/cloud_basics/magboltz_drift.py` is 93/5/2. All 432 `magboltz_co2_*`
  points of `results/air_hs` and `results/air_merged` were regenerated on it (condor 4694770, 3e8 collisions, the same
  water × air × field nodes; each JSON carries its `comp`). The tag stays `co2`: no 95/3/2 file is left where `gasmodel.py`
  looks. The 47 older 95/3/2 files of `air`, `air_fine`, `air_mid` were moved to `legacy_co2_95-3-2/` there, out of the
  loader's glob. At 1 % water and 225 V/cm the drift is 21 % slower in 93/5/2 (13.0 against 16.4 µm/ns).
- **Gas fit** (`rerun/sps_gasfit/`, baseline + 10 arms + verdict + report): CF₄ epoch unchanged to the last digit; CO₂
  epoch water 1.94–2.12 → **1.56–1.75 %**, O₂ 322–443 → **282–403 ppm**; χ² about the same, no grid edge, the O₂ ordering
  (run55 > run59) kept in all arms. The CO₂ water now matches the CF₄ epoch's (1.7–1.85 %), as one gas line at one flow
  should.
- **Figures:** `rerun/paper1_spsfigs/figures/attachment` and `attachment_charge` rebuilt (CO₂ column: 403 ppm, χ² 271 against
  9063 without O₂; was 443, 258 against 10 600). Paper I text and figure copies wait for Dylan (`\analysis{GAS9352}`).

## What still carries 95/3/2

- `garfield_sim/wetco2_one_mixture.py` and `garfield_sim/results/drift_velocity_beamtest_co2_CERN.json`, `drift_wetco2_*.json`
  (Garfield runs for the August notes; nothing current reads them).
- The CO₂ midpoint interpolation check (`cloud_basics/results/mb_mid_check.json`), not redone (same grid spacing).
- `papers/paper2/hits_daq/sps_dp_scan.py`'s refit at `Dp` 0.006 (`SPS_EXCESS_2026-11-18.md`): the 93/5/2 grid gives 0.0050–0.0055.

**Directory names.** The data directories `sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/` and
`rot25_ArCO2iso_95-3-2__run57/` keep their names because scripts and notes point at them. The `95-3-2` in those names is
the old wrong label; the gas in them is 93/5/2.
