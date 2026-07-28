# The SiPM-wall divert was OFF in runs 224356–224360 and 224464, 224466

**Written 2026-07-28. Answers: "find a run where the γ-flash divert (SiPM
blanking) was disabled, so the walls record the true flash."**

> **UPDATE, same day.** A scan of *every* processed run in the campaign (306
> runs) found **two more** divert-off runs beyond the five below: **224464 and
> 224466**, 2026-07-16 — a second, independent epoch. The complete list is
> therefore **224356, 224357, 224358, 224359, 224360, 224464, 224466**, and
> nothing else. Scan output: `flash_timing/data/divert_state_by_run.csv`.
>
> **The timing calibration built on these runs now lives in
> [`flash_timing/`](flash_timing/README.md)** — that directory supersedes §4 of
> this file, which was a first look.

**Answer: n_TOF runs 224356, 224357, 224358, 224359, 224360** — 2026-07-10
~21:44 → 2026-07-11 ~10:09 — **plus 224464 and 224466** on 2026-07-16. All 32
wall channels see the undiverted γ-flash. **Use 224357** (cleanest; see §3).

---

## 1. How they were found

Two independent lines, one config-side and one data-side.

### 1a. Config side — N1081B per-sub-run snapshots  [checked, negative]

The SiPM blank/enable line is **M6 (.245) SEC_C outputs 0 and 1** —
`n1081b/n1081b_module_map.py:366` (`dst="SiPM enable / blank"`, mono 1000 ns,
inverted TTL, "2 used"). `SUBRUN_CONFIG_SNAPSHOT.md` writes a full board dump
to `<subrun>/n1081b_config.json` on every DREAM sub-run.

All **389** snapshots on the DAQ machine (`/mnt/data/x17/beam_july/{runs,pedestals}`,
2026-07-13 → 07-28) plus the 27 manual dumps in `n1081b/snapshots/`
(07-09 → 07-18) were scanned:

- **SEC_C outputs are `1111` in every single one.** The blanking line was never
  switched off inside the snapshot era.
- The only SEC_B anomaly is `0001` (outs 1–3 off) in three run_68 sub-runs,
  **2026-07-23 15:38–16:08** — but n_TOF was not taking data then (run 224544
  ended 09:54:02, run 224545 started 17:31:20), so it yields nothing. Those were
  `cos_*` cosmic sub-runs, i.e. beam-off — no flash to record anyway.
- SEC_B `1111`↔`1100` toggling everywhere else is just the mesh-injection scan
  (`acmeshOn_*`/`acmeshOff_*` sub-run names), not the SiPM line.

The snapshot era starts 07-13, so it **cannot** cover the window that actually
matters. What does cover it is a note in
`n1081b/HANDOFF_2026-07-11_trigger_timing.md`:

> **M6 (.245) offline**: pulser (→ M4.C input), mesh charge injection (SecB),
> **SiPM enable (SecC) all dead in the water**; also still on old fw
> 2022.3.0.0 → upgrade via web GUI when back and safe.

That is the 07-11 switch outage; `.245` was moved onto the DAQ net and upgraded
during the recovery (`n1081b/README.md`). **With M6 down, nothing drove the
blanking gate.**

### 1b. Data side — the walls' own tflash  [decisive]

Per-run, from the official processed files
(`/eos/experiment/ntof/processing/official/done/run<N>.root`):

| | blanking ON (every other run) | blanking OFF (224356–360) |
|---|---|---|
| WALA/C/D tflash | 0.7–1.2 µs **before** WALB | equal to WALB within **±6 ns** |
| flash-peak amplitude | ~850 ADC (clamped transient, area negative) | **~28 500 ADC** |
| all-4-walls-agree | 0.8 % of bunches | **90–99.8 %** of bunches |

Scan of runs 224345–224364 (`WALB` as reference, flash hit = `|tof−tflash|<60`):

```
224355  A:-735  B:  0  C:-695  D:  -5   flashamp ~800    blanking ON
224356  A:  +5  B:  0  C:  +5  D:  +5   flashamp 12890*  BLANKING OFF
224357  A: +10  B:  0  C:  +5  D:  +5   flashamp 13128*  BLANKING OFF
224358  A:  +5  B:  0  C:  +5  D:  +5   flashamp 13084*  BLANKING OFF
224359  A:  +5  B:  0  C:  +5  D:  +5   flashamp 12942*  BLANKING OFF
224360  A:  +5  B:  0  C:  +5  D:+10    flashamp 12963*  BLANKING OFF
224361  A: -30  B:  0  C:-1025 D: -25   flashamp ~840    blanking ON
```
`*` median of the PSA-tagged hit; the true per-bunch **peak** in the flash
window is ~28 500 (the finder splits the giant pulse into fragments).

Both edges are inside a run boundary, so the five runs are clean end to end:
224355 is blanked to its last bunch, 224361 is blanked from its first.

## 2. Per-run detail

| run | ends (local) | bunches | intensity | all-4-agree |
|---|---|---|---|---|
| 224356 | 07-11 01:00:47 | 3412 | dedicated 8.3e12 | 90.1 % |
| **224357** | **07-11 03:51:06** | **3286** | **mixed: 47 % dedicated 8.5e12 / 53 % parasitic 4.1e12** | **96.7 %** (full run) |
| 224358 | 07-11 06:40:56 | 3269 | mixed, 48 % dedicated | 93.2 % |
| 224359 | 07-11 09:40:03 | 3426 | dedicated 8.4e12 | 90.0 % |
| 224360 | 07-11 10:08:51 | 549 | dedicated | 90.4 % |

(224356 starts ≈ 07-10 21:44, right after 224355 ends. The non-agreeing few %
are the known per-bunch PSA flash mis-tags — [[ntof-pss-tflash-bug]] — not
blanking coming back.)

All five have `WALA-D`, `PSSA-D`, `SILI`, `PKUP`. **No `LIQ` trees** — the
liquid scintillators were installed 07-17.

## 3. Recommendation

**224357** — mixed dedicated/parasitic intensity in one run (lets you test
intensity dependence without a cross-run comparison), 3286 bunches, and the
highest per-bunch consistency (96.7 % of bunches have all four walls tagging
the same feature; per-bunch spread wall-to-wall σ ≈ 5 ns).

Use **224359** instead if you want maximum stats at nominal dedicated
intensity, and 224358 as the independent repeat.

## 4. What these runs give you — and what they do not

### They give the absolute wall flash time  [superseded by `flash_timing/`]

> The numbers below are the first-look, stored-`tflash` version. The proper
> measurement — per channel, per epoch, with the timing resolution decomposed
> and the transport across the campaign verified against the liquid
> scintillators — is in `flash_timing/README.md`. Its headline is
> `t_flash = tof_PKUP + C`, `C ≈ −1719 ns` per wall channel.

Referenced to `PKUP` tflash (bunch-matched, median over the run):

| run | state | WALA | WALB | WALC | WALD |
|---|---|---|---|---|---|
| 224356 | OFF | −1712.6 | −1716.0 | −1710.8 | −1711.5 |
| 224357 | OFF | −1710.7 | −1715.8 | −1709.8 | −1711.7 |
| 224358 | OFF | −1712.5 | −1715.9 | −1710.9 | −1711.3 |
| 224359 | OFF | −1711.4 | −1716.2 | −1711.2 | −1712.5 |
| 224360 | OFF | −1714.9 | −1717.3 | −1711.7 | −1713.3 |
| 224361 | ON | −2906.3 | −1890.7 | −2887.4 | −2891.9 |
| 224362 | ON | −2861.3 | −1879.8 | −2877.1 | −2881.1 |
| 224363 | ON | −2855.3 | −1874.5 | −2871.8 | −2876.7 |
| 224364 | ON | −2856.2 | −1879.0 | −2877.7 | −2881.2 |
| 224572 | ON | −2076.7 | −1709.3 | −2050.4 | −2056.0 |

Reading:

- **True γ-flash in the walls = PKUP tflash − 1713 ns**, reproducible to ±2 ns
  run-to-run and ±6 ns wall-to-wall. This is the absolute wall term that
  `FLASH_TIME_BASE.md` §4 said needed a dedicated pulser measurement — it can
  be read straight off these runs instead.
- Gate-on WALA/C/D tag the gate transient **1165 ns early** (07-11 era) or
  **340 ns early** (224572, after the SEC_C in0 delay went 200 → 1000 ns on
  2026-07-22 10:13 — visible in the snapshots).
- **WALB is the bridge**: it tags the flash leak, and in 224572 it sits at
  −1709.3 vs −1715.8 measured with the gate off — **agreement to 6.5 ns across
  15 days**. So WALB's stored tflash already *is* the true flash time, and PKUP
  is stable enough to transport the calibration across the campaign.
- Corrections to add to stored tflash to get the true flash, run 224572:
  **WALA +364, WALC +337, WALD +343, WALB −4 ns** (recompute per run; the
  gate-transient offset changed on 07-22).

### They do not give an amplitude/energy calibration

The undiverted flash **saturates the wall**. Doubling the proton intensity does
not change the response:

```
run224357, flash amp, parasitic 4.1e12 → dedicated 8.5e12 (median, per channel)
  WALA  1:13743→12780 r=0.93   ...  8:13498→12699 r=0.94
  WALB  1:13146→12644 r=0.96   ...  8:13536→13256 r=0.98
  WALC  1:13412→13077 r=0.97   ...  8:12836→12416 r=0.97
  WALD  1:13263→13191 r=0.99   ...  8:14186→12644 r=0.89
```
r ≈ 1 for all 32 channels: the amplitude is flat in beam intensity. It is not
ADC clipping (`satuflag` = 0 on every hit; peak ~28 500 of a 65 536 range with
baseline offset −949 mV) — it is the SiPM/front-end itself. Which is, of
course, why the divert exists.

So: **timing calibration yes, linear amplitude calibration no.** For absolute
light-yield scale keep using the Y-88 source runs (224476–224479).

## 5. Practical notes

- **Raw data is gone from EOS disk** for these runs (only ~07-14 onward
  survives). The official processed ROOT files exist for all five. If you need
  waveforms — and for a saturation/shape study you do — stage from CTA tape
  with `StageRuns.sh` (`ntof_daq_processing/PROCESSING.md`, "Data location").
- The PSS trees in these runs carry the **unfixed** flash bug (5–9 % of bunches
  in the core when referenced to PKUP), so do not use plastic tflash here
  without `tflash_repair` / a reprocessing.
- These runs predate: the FIFO plastic path (07-17), the LIQ vessels (07-17),
  the plastic HV equalization (07-16), and the SEC_C delay change (07-22).
  Wall HV/thresholds of 07-10/11 are the pre-recalibration set.

Scan scripts used (kept in the session scratchpad, trivial to regenerate):
per-run `tflash`/`amp` probes against the official files on lxplus with uproot
from `LCG_105`, and a walker over `n1081b_config.json` on the DAQ machine.
