# sps_beam_test_26 / analysis — start here

Analysis of det4 (`mx17_E`) in the H4 test beam, 2026-07-31 → 08-03.

Two strands, kept separate because they answer different questions:

1. **The record** — what was actually taken, when, at what HV, at what mount
   angle. Every physics number depends on getting this right, and on this
   campaign the DAQ config lies about gas, mount and start time.
2. **The charge-spreading measurement** — det4 flat (normal to the beam) is a
   rare geometry in which the resistive sharing kernel can be measured
   *directly* rather than inferred from a track fit. That is what most of the
   beam time was spent on.

---

## Read in this order

| | |
|---|---|
| `RUN_TIMELINE.md` | the narrative and the configuration epochs. **Read first.** Each claim is marked with whether the machine record confirms it. |
| `HV_AND_BEAM_RECORD.md` | what survives of the HV and SPS beam records, and what does not. |
| `M70V_FLAT_ANALYSIS.md` | run_56 m70V — flat, Ar/CO₂/iso, the first kernel measurement. |
| `FLAT_CF4_RUN63.md` | run_63 — flat at the operating point in the new CF₄ gas. |
| `RAW_RUN71_STATUS.md` | run_71 RAW — the FEU packet loss, its diagnosis, and the decoder fix. |
| `RAW_RUN71_PHYSICS.md` | run_71 RAW — what it settled, what it retracted, and the wall it hit. **Partially superseded.** |
| `RAW_RUN71_REANALYSIS_2026-08-04.md` | ground-up rework: the ">3.8 µs tail" was pile-up + two oscillating channels + no baselines; drift is slow (v ≈ 13–15 µm/ns at 233 V/cm, 4× below dry Magboltz — open item) but **fine at the operating point**; kernel drift-invariance passes; charge budget measured. **Current authority for the flat-data conclusions.** |
| `RERUN_2026-08-04_NEW_MACHINE.md` | the campaign machine is gone: decoder patch recovered & pushed, chain rebuilt from EOS, clean scripts recreated **in the repo** (`robust_waveforms.py`, `kernel_refit_clean.py`, `tilt_clean.py`). Charge budget + invariance reproduce (now across all 3 fields); cascade parameters shift with the lost recipe details and were already ruled non-physical. **Tilt corrected: θ_X ≈ 0.9° — the old 0.2–0.4° was the dry-gas v_drift; the invariant is tan θ_X = −0.015 ± 0.002.** `mapping_urwell.csv` still to reconstruct (blocks `flat_align_eff.py` only). |

> **2026-08-04: the two sections below are superseded by
> `RAW_RUN71_REANALYSIS_2026-08-04.md`.** The window wall was three
> measurement artefacts; the clean central response is contained in the
> window; the kernel's drift-invariance test passes; what genuinely runs off
> the window at 450/275 V is the *drift ladder* (slow gas at deliberately low
> fields), and the ±2/±3 few-percent surface tails. The paragraphs are left
> as written so the correction is visible.

## The conclusion, in one paragraph

det4 at normal incidence lets the ±1 sharing be split into a **prompt**
component (transverse diffusion, arriving with the central strip) and a
**dispersed** component (through the resistive layer, arriving late), because
at w = 0 the two separate in time. The sharing is RC-*dispersed*, not a delayed
copy — the `share_lp` branch of `wft/model.py`, not the plain-delay one. The
one number the campaign genuinely pinned is the **±1 peak-time shift, +29 to
+36 ns**, stable across two gases, four drift fields, three resist voltages and
both zero-suppressed and RAW readout, and sitting on the bench's
independently-inferred τ ≈ 47 ns. `c1`, `c2` and `tau_s` are **not** pinned:
all three are integrals over a tail that no readout configuration in this
campaign contained (§*The window wall* below).

## The window wall — why the kernel is not finished

Three truncations, discovered in this order, each hidden by the one before it:

1. **ZS closes the neighbour's window** near threshold (5 samples kept at
   threshold vs 25 at high amplitude) — amplitude-dependent, so it sculpts the
   measured shape. Fixed by taking run_71 in RAW.
2. **ZS closes the central strip's own window** ~400 ns after its peak, so the
   basis waveform of the fit was itself truncated. Also fixed by RAW.
3. **The DAQ window is too short.** With RAW, **44–52 % of the central strip's
   amplitude is still present at the last of the 64 samples.** The dispersed
   tail is longer than 3.8 µs.

(3) is not fixable in analysis and there is no beam for three years. So
`tau_s` and `c2` are **bounds, not measurements**, and the drift-invariance
test fails because the truncation is itself drift-dependent. An earlier
conclusion that `c1` = 0.23–0.28 was invariant across gas and voltage is
**retracted**: with more tail included it moves to 0.32–0.35, so the apparent
stability was an artefact of *consistently* truncated tails.

## Two standing gotchas

- **det4 carries a tilt in X** (the striped coordinate), measured from the
  charge centroid walking as the column drifts in; the uRWELL track slopes
  cannot see it. It contaminates the X view of the kernel and gets *worse* in
  RAW because the tilt lives in the tail. **Quote the Y view, never X.**
  *2026-08-04 correction:* the historical "~0.2–0.4°" divided the walk by the
  DRY-gas v_drift; with the measured wet-gas v the same walk is **θ_X ≈ 0.9°**
  (and θ_Y ≈ +0.2–0.3° is not zero). The v-independent invariant is
  **tan θ_X = −0.015 ± 0.002**, drift-field-invariant across 92–233 V/cm —
  see `RERUN_2026-08-04_NEW_MACHINE.md` §4.
- **`run_config.json` is wrong** about gas, mount angle and `start_time` for
  this entire campaign. Authorities are `hv_monitor.csv` for HV, `dream_daq.log`
  for sub-run boundaries, `*_thr.prg` headers for the ZS threshold, the
  H4 TAX stopper log for accesses, and our own scan logs for det4's HV.

---

## Scripts

### Conditions — the single source of truth

`datasets.py` defines every flat-mount dataset as a (mount, gas, drift, resist)
condition with its HV plateau windows, sub-run boundaries, ZS threshold and
pedestal set. **Nothing may restate these in an analysis script.** This file
exists because the run_61 scans were once labelled with one mount angle in two
copy-pasted scripts that then drifted, and half of each combined curve turned
out to be at the other angle.

```
run56_m70V      flat, Ar/CO2/iso 95/3/2, ZS 5 sigma, resist 590 then 625 V
run63_rot25     25.64 deg (PRE-access), Ar/CF4/iso, drift ladder at fixed resist
run63_flat      flat (POST-access), Ar/CF4/iso, 53 min at the operating point
run71_raw       flat, Ar/CF4/iso, RAW (no ZS), drift 700 / 450 / 275 V
```

### The chain

```bash
python decode_dataset.py    <dataset> [--feus 03,01]   # fdf -> waveforms + hits
python pair_dataset.py      <dataset>                  # det4 <-> uRWELL by eventId
python flat_align_eff.py    <dataset>                  # fit z, align, efficiency
python extract_det4_only.py <dataset> [--cm masked]    # waveforms, det4-only selection
python kernel_fit_m70V.py --wf <wf.npz> --plateau <p> [--raw]
python tilt_m70V.py       --wf <wf.npz> --plateau <p>
# the clean (artefact-free) RAW chain, recreated in-repo 2026-08-04:
python robust_waveforms.py   <dataset> --wf <wf.npz>   # clean median/trim library
python kernel_refit_clean.py <dataset> --lib <lib.npz> # invariance test + budget
python tilt_clean.py         <dataset> --wf <wf.npz>   # tilt on the clean selection
```

`decode_dataset.py` derives every analyzer flag from the dataset record rather
than accepting them: matched-filter width from the Dream shaping time, and —
critically — `--cns 1 --zs-baseline 0` for RAW against `--cns 0 --zs-baseline 1`
for ZS. Those are opposite on both counts and getting them wrong is silent.

### Diagnostics

| | |
|---|---|
| `fdf_scan.py` | walks the raw 16-bit words (**big-endian** — `read16` does `ntohs`) and reports FEU frame structure, eventID continuity and sample completeness. Use when a decode looks wrong. |
| `decode_loss_report.py` | reads the `decode_stats` tree and `sample_acceptance` histogram the decoder now writes into every output, and returns the acceptance array a mean waveform must be divided by. Returns `None` for files decoded before that existed, so old files cannot be silently assumed clean. |
| `tax_windows.py` | H4 TAX beam-stopper open/blocked windows — dates every zone access to the second. |
| `beam_record_coverage.py` | per-day coverage of the SPS beam record. |

### The older per-run scripts

`pair_m70V.py`, `align_eff_m70V.py`, `extract_waveforms_m70V.py` and
`charge_spreading_m70V.py` are the run_56-specific originals that
`pair_dataset.py` / `flat_align_eff.py` / `extract_det4_only.py` generalise.
They are kept because `M70V_FLAT_ANALYSIS.md` quotes their output directly, and
because `charge_spreading_m70V.per_strip` is imported by the fit and the tilt
scripts.

---

## Traps this directory has already fallen into

All four produced plausible-looking output rather than an error, which is why
they are listed:

1. **Silent numpy string truncation.** `"flat700"` in a `<U6` array became
   `"flat70"` and every plateau match failed; `"cfg_gain4.5_peaktime50"` in a
   `<U16` array broke the sub-run join. Size string dtypes generously.
2. **The flat-256 baseline on RAW data.** Correct for ZS, badly wrong for RAW
   (which sits on raw per-channel pedestals, median 619 ADC, plus common mode).
   `flat_align_eff.py` now *refuses* to write RAW waveforms rather than emit
   something that looks fine; use `extract_det4_only.py`, which does the
   pedestal + CNS correction.
3. **A patch that silently no-op'd** because its anchor text did not match.
   Assert the anchor count before writing.
4. **Normalising against the wrong denominator.** run_71's loss was first
   assessed against the decoder's entry count rather than the true event count,
   and came out as "no loss" when a quarter of the data was missing.

## Data

Staged under `/media/dylan/data/x17/sps_run53_det4_check/staging/<run>/`.
Raw on EOS at
`root://eospublic.cern.ch//eos/experiment/ntof/data/x17/p2_sps_july/runs/`
(complete, including runs pruned from banco, and it now carries run_71).
banco itself is `banco_cern:/local/home/banco/P2_data/TB_July2026_H4/`.
