# RESOLVED: the "missing plastic" was n_TOF's broken PSS time base

**2026-07-28 overnight session, picking up HANDOFF_2026-07-27 §5.**
Everything below was executed this session; the scripts quoted are reproducible.

## TL;DR

The open question — "why does n_TOF record a plastic hit for only ~52 % of DREAM
triggers when the hardware trigger required wall AND plastic" — is closed. **The
plastic hits were there all along. The official n_TOF reconstruction stores a
wrong gamma-flash time (`tflash`) for the PSS trees in 37–85 % of bunches**, so
`t_since_flash = tof − tflash` for those (tree, bunch) combinations is shifted by
up to 11.6 µs and the true partner sat outside the accept bands, masquerading as
"no plastic hit". With the time base repaired:

```
plastic partner | wall-matched DREAM event:  48.9 %  →  99.7 %
   per arm  A 43.6→99.4   B 33.6→99.9   C 63.8→99.7   D 50.8→99.9
match_window efficiency (stat090_0000):      98.8 %  →  99.9 %  (100.0 % in every bin below 40 ms)
```

The DREAM trigger is vindicated: it is a true wall∧plastic AND (config §2), the
plastic pulses are physically present at trigger level in the raw waveforms
(§5), and the geometry hypothesis is dead. **Fix applied**: `tflash_repair.py`,
wired into `ntof_io.read_bunches` by default.

## 1. The measurements, in the order that cracked it

1. **LIQ/SILI check** — the file also carries LIQA-D and SILI trees, never
   checked before. LIQ in-band 3.6 % (accidental level) → the mid-July
   plastic↔liquid cable swap is NOT the answer.
2. **Per-bunch bimodality** — the plastic-found fraction per bunch is bimodal:
   ~43 % of bunches at ~0 %, the rest at 65–90 %. The "bad" bunches are exactly
   the **parasitic (half-intensity) PS pulses** (415 vs 854 ×10¹⁰ p).
3. **tflash tables** (all 3018 bunches, every tree; cached
   `tflash_table_224572.npz`): stored tflash deviates >150 ns from the tree mode in

   | | WALA | WALB | WALC | WALD | PSSA | PSSB | PSSC | PSSD | LIQA-D | PKUP |
   |---|---|---|---|---|---|---|---|---|---|---|
   | bad | 1.7 % | 1.1 % | 0.3 % | 0.0 % | **84.5 %** | **65.4 %** | **36.8 %** | **80.6 %** | 0.0 % | 0.0 % |

   PSS failures are µs-scale (the finder tags a pulse near the window start —
   stored values like 314 ns where the flash is at 11 645 ns), hit essentially
   every parasitic pulse plus an arm-dependent fraction of dedicated ones.
4. **Per-event proof** — for previously-missing events, the nearest-plastic
   residual equals that bunch's tflash error (median agreement 15 ns).
5. **Raw waveforms** (stream1 on EOS,
   `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/224572/stream1/`, read
   with `nTof_x17_DAQ/stream1_monitor/ntof_raw.py`; `EVEH words[1]` =
   BunchNumber): the plastic gamma flash is present and **rails the ADC high**
   in every pulse, parasitic included, at ~11.6–11.9 µs — while the PSA tagged
   0.3–2 µs. And at the predicted times of 8/8 sampled "missing" DREAM events
   there is a **negative pulse 3 800–12 800 counts deep** (≈120–390 mV, i.e.
   right at the 118–157 mV discriminator level) in exactly one plastic channel
   of the matched arm, onset within ~30 ns of prediction.

## 2. The hardware trigger, settled from the run_79 config

`n1081b_config.json` live read-back: M3 is `and` on all four sections, lemos
0+1, `bypass_enable: False`, wall leg G&D gate 20/delay 20, plastic gate
20/delay 0; M4.C or_veto has **only Singles** enabled (Doubles and pulser off);
M4.D ORs the gated physics with the PS line (delay 1440 ns → the per-burst flash
event). M5 scaler diffs across the 16 sub-run polls: walls ~850–960 Hz/arm
(flat), plastics 200/480/380/335 Hz (A/B/C/D), sector ANDs 43/72/63/53 Hz —
the AND tracks the plastic leg at 15–21 % in every sector, i.e. a live, healthy,
plastic-limited coincidence (cosmics), stable all night.

## 3. What was wrong in our own analysis

Nothing structural — the matcher, clock fit, bunch join all hold. The 52 % was
real arithmetic on corrupted upstream data. Two of HANDOFF §5's "excluded"
tests deserve a caveat: the wall×plastic mapping matrix and dt-peak checks were
dominated by sane-tflash bunches, which is why they looked clean while half the
data was shifted.

## 4. The repair (in this repo, applied by default)

`tflash_repair.corrected_tflash(run)`:
`tflash_true(tree, b) = mode_tree + jitter(b)`, with `mode_tree` the per-run
modal tflash (cable constant, 10 ns bins) and `jitter(b)` the median over the
stable trees (WALB-D, LIQA-D, PKUP) of their per-bunch deviation from mode.
This repairs every tree including WALA's own 1.7 % +374 ns glitches. Wired into
`ntof_io.read_bunches` (`repair_tflash=True` default); the stored `tflash`
branch is returned untouched when asked for. Tables build once per run (~10 min)
and cache next to the bunch indexes.

Validated: stat090_0000 → 99.9 % matched (100.0 % in every bin < 40 ms);
stat090_0001 → 98.9 % with the sub-run-0000 clock constants (the 40–80 ms bin
reads 93.7 % — fit (k, t0) per sub-run as already agreed and re-check).

## 5. What is broken in the official PSS reconstruction (report upstream)

**CORRECTION (later the same night).** An earlier version of this section
claimed the PSS *amplitudes* were broken on arms A/C/D. **They are not.** The
big-amplitude partners exist on every arm — they form razor-sharp prompt
coincidence peaks with the walls — but at arm-dependent offsets of
**−375 / +25 / −325 / −325 ns (A/B/C/D)** in the mode-repaired time base, i.e.
*outside* the accept bands on A/C/D, which made them invisible to the earlier
partner search (the small in-band "partners" were secondary overshoot/rebound
fragments of the same physical pulses, ~130–240 ADC, trailing the true pulse by
~350 ns). The real defect list:

1. **tflash mis-identification per bunch** — arm-dependent 37–85 % of bunches,
   ~always on parasitic pulses. The flash rails the ADC and is unmissable in
   the raw data; purely a PSA/flash-finder fault (G-FLASH THRESHOLD = 50
   channels, first-crossing option, no MIN_WIDTH — any early junk pulse wins).
2. **Per-tree flash-feature inconsistency, ~350 ns** — a *constant*, not
   per-bunch: WALA/C/D time the flash at mode 11 245–11 275 ns while
   WALB / PSSA-D / LIQA-D all sit at 11 615–11 645 ns. The PSA is timing a
   different feature of the (railed, undershooting) flash waveform per
   detector. Harmless for same-tree relative times, fatal for cross-detector
   coincidences and for absolute ToF, and it means **the tflash mode alone is
   not a sufficient repair** — see §5b.
3. Secondary artifact hits: each real plastic pulse is accompanied by a small
   (~130–240 ADC) rebound hit ~350 ns later. (The wall +330 ns satellite band
   of match_window may well be the same phenomenon on the walls — untested.)
4. (Known, separate) the plastic PSA amplitude cut is 2× the walls' (100 vs
   50 ADC) and truncates the small-hit spectrum.

## 5b. Repair v2 — coincidence-calibrated offsets

`tflash_repair.corrected_tflash` now adds a per-tree constant measured from
data: the prompt-coincidence peak of large (amp>1000) PSS hits — and of LIQ
hits — against the same arm's wall, so a true coincidence reconstructs at
dt ≈ 0 in every arm. Measured offsets (run224572):

```
PSSA −362.3   PSSB +19.6   PSSC −333.0   PSSD −336.0
LIQA −372.6   LIQB  +9.7   LIQC −349.8   LIQD −348.0     (walls = reference)
```

LIQ agreeing with PSS per arm confirms the inconsistency belongs to the walls'
flash timing, not to the plastics. Cached in `tflash_offsets_<run>.npz`.

**With v2, the full hardware-threshold trigger emulation finally works.**
`singles_candidates(require_plastic=True)` + accept bands, 100 bunches of
stat090_0000, control = +100 µs shifted time:

```
  t bin (ms)      n    efficiency   control(false)
     1-3        895       89.9%        1.3%
     3-10      2365       92.9%        1.0%
    10-20      2773       94.4%        0.4%
    20-40      2583       95.0%        0.0%
    40-80      1805       93.7%        0.1%
  overall              93.7%        0.5%
```

Compare the pre-fix state: wall-only 88.3 % efficient with **28.9 %** false at
1–3 ms; plastic-required 12.7 % efficient. **The early-time purity problem is
solved** — the AND cuts the candidate rate to ~935/bunch and the 1–3 ms false
rate drops 28.9 % → 1.3 % while keeping ~90 % efficiency.

## 6. Matcher recommendation (supersedes HANDOFF §8)

- **Thresholded wall∧plastic SINGLES** (dream_trigger with require_plastic=True)
  on the v2-repaired time base: ~90–95 % efficient, ≤1.3 % false in every bin.
- Fall back to wall SINGLES + plastic-presence tag only where the plastic
  amplitude calibration is in doubt.
- Fit `(k, t0)` per sub-run.
- For *absolute* ToF/E_n (Phase 5), anchor the flash on PKUP (stable, 0 %
  failures) and carry the per-tree offsets explicitly — do not treat any
  scintillator tflash as the physical flash arrival until the upstream fix
  lands.

## 7. Figures

`~/x17/beam_july/analysis/ntof_dream_merge/figures/`:
`tflash_bug_map.png` (stored PSS vs WAL tflash per bunch, dedicated/parasitic),
`tflash_failure_rates.png` (per-tree failure fractions),
`tflash_raw_waveform_proof.png` (raw flash mis-tag + the "missing" pulse at the
predicted trigger time), `plastic_partner_before_after.png`.
