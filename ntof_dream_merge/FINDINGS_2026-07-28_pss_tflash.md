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

## 5. What is STILL broken in the official PSS reconstruction (report upstream)

1. **tflash mis-identification** (the bug above) — arm-dependent 37–85 % of
   bunches, ~always on parasitic pulses. The flash rails the ADC and is
   unmissable in the raw data; this is purely a PSA/flash-finder fault.
2. **Amplitude reconstruction broken on PSSA/C/D**: raw pulses 3.8–12.8k counts
   deep are stored as `amp` ≈ 130–240 ADC (~4–7 mV), piled just above the
   100 ADC PSA cut, and `area` is proportionally wrong. **Arm B alone is
   correct** (trigger partners: sharp +15 ns dt peak, median 4 445 ADC ≈
   137 mV). Practical consequence: no amplitude discrimination is available
   from the official file on A/C/D — discriminator emulation
   (`singles_candidates(require_plastic=True)`) cannot work there.
3. **Time resolution degraded on A/C/D**: the wall→plastic dt distribution has
   no ns-scale peak — partner times scatter over ~±250 ns (still inside the
   accept bands, which is why presence-matching works at 99.7 %).
4. (Known, separate) the plastic PSA amplitude cut is 2× the walls' (100 vs
   50 ADC) and truncates the spectrum in its bulk.

## 6. Matcher recommendation (supersedes HANDOFF §8)

- **Wall SINGLES** for timing (unchanged), with the repaired time base.
- **Plastic PRESENCE in band** (no amplitude cut) as the AND tag — now carried
  by 99.7 % of true triggers, so it can be *required* without the 47-point
  efficiency loss it used to cost. Amplitude thresholds only on arm B.
- Fit `(k, t0)` per sub-run.
- Early-time purity is still limited by the plastic accidental rate at 1–10 ms;
  the tag helps mostly >10 ms until the upstream amplitude bug is fixed.

## 7. Figures

`~/x17/beam_july/analysis/ntof_dream_merge/figures/`:
`tflash_bug_map.png` (stored PSS vs WAL tflash per bunch, dedicated/parasitic),
`tflash_failure_rates.png` (per-tree failure fractions),
`tflash_raw_waveform_proof.png` (raw flash mis-tag + the "missing" pulse at the
predicted trigger time), `plastic_partner_before_after.png`.
