> # ⛔ RETIRED — do not build on this
>
> **Superseded by `DREAM_NTOF_CALIBRATION.md`.** Archived 2026-07-30.
>
> The findings note that first re-derived the window on v12, written earlier the same day. Its content has been folded into the calibration document, which supersedes it in three ways: the coverage figures are now accidental-subtracted (98.59 / 96.00 / 2.58 / 1.41 %, against 98.7 / 96.0 / 2.7 / 1.3 % here), the operating point is quoted at exactly ±25 ns rather than at the 23.6 ns knee, and the per-bunch fit now carries a five-test bias study it did not have.
>
> **Read `../DREAM_NTOF_CALIBRATION.md`.**

---

# The DREAM ↔ n_TOF accept window, re-derived on the reprocessed file

**2026-07-30.** Everything below is measured on the **complete reference pair**
— run_79 `stat090_0000` + `stat090_0001`, 2061 bunches, **213 420** non-flash
DREAM triggers — against n_TOF run 224572 processed with `v12_liqpileup`, read
on the file's own stored `tflash`. Tooling, figures and slides:
`match_study/` (`latex/dream_ntof_matching_slides.pdf`).

This supersedes the window calibration in `match_window.py`'s docstring, which
was measured on the **official** processing.

## 1. Every DREAM trigger, and its wall+plastic partner

The n_TOF sector SINGLES is rebuilt from the hit trees exactly as the N1081B
formed it (`dream_trigger.py` for the chain; `fast_singles.py` is a vectorised
rewrite, validated bit-identical): the 428F analogue **sum** of the two bar ends
over the wall threshold, ORed over the four bar segments, ANDed with a plastic
bar over its threshold inside the 20 ns logic pulse. Thresholds come from each
sub-run's `n1081b_config.json` (wall 25/35/34/36 mV, plastic 118/139/157/134 mV).

| | fraction of the 213 420 triggers |
|---|---|
| matched to a wall coincidence (wall leg alone) | **98.7 %** |
| matched to a wall **AND** plastic SINGLES | **96.0 %** |
| plastic partner present, given a wall match | 97.3 % |
| plastic leg costs | 2.7 % |
| wall leg costs | 1.3 % |

So the answer to "can we match a wall top+bottom sum to a plastic pulse for each
DREAM trigger" is **yes for 96 % of them**, and the 4 % that fail is two thirds
plastic leg. That 2.7 % is the same number the v12 acceptance test reports and
is the one thing the reprocessing has not closed (the official file with the
laptop tflash repair costs 5.9 %).

## 2. The n_TOF detectors are aligned — to a few ns, measured on v12

`match_study/scripts/align_survey.py`, 250 bunches spread over the whole run,
stored `tflash` (no repair).

**[1] Absolute, against the beam pickup.** `tflash(tree) − tflash(PKUP)` per
bunch is the only estimator that compares detectors with no common particle, so
it is what puts arm A and arm D on one time base.

- walls −1719.3 / −1719.6 / −1721.3 / −1723.3 ns (A/B/C/D) — a **4.0 ns**
  spread, per-bunch σ 5–10 ns;
- **the liquids reproduce the divert-off calibration of `ntof_processing/
  flash_timing/` to 0.1–0.5 ns** (LIQA −1708.1 vs −1708.2, LIQB −1710.5 vs
  −1710.3, LIQC −1695.7 vs −1695.6, LIQD −1701.0 vs −1701.6). Two independent
  measurements, one on seven divert-off runs and one on this run's own data,
  agreeing at half a nanosecond;
- the plastics sit 31–50 ns from the same calibration, which is exactly what
  `flash_timing/README.md` says will happen (*"do NOT transport the PSS
  constants"*). Take PSS per run.

**[2] Wall vs plastic**, prompt coincidences of late hits (t > 100 µs):
peak −3.8 to −8.8 ns per arm, median −1.7 to −5.1 ns, σ ≈ 11 ns. Per wall
channel the medians have RMS **2.3 ns** (range −7.7 … +1.9 over 32 channels);
per plastic bar RMS 1.8 ns. The stale −25…−40 ns offsets in
`mx_july_beam_qa/calib/time_offsets_run*.json` remain inapplicable here.

**[3] Top vs bottom of a bar**: within **±6 ns** on v12 — see §4.

**[4] Liquid vs wall**: −0.8 … +0.2 ns. **Aligned to under a nanosecond**, so
the liquid leg needs no offset at all when it enters the merge.

## 3. On v12, do not run the tflash repair

`tflash_repair` was built for the *broken* official flash finding. Stored minus
repaired, per bunch on v12:

| tree | median (ns) | RMS (ns) | > 25 ns |
|---|---|---|---|
| WALA–D | 0.5 … 4.2 | 4.3 … 9.1 | 0.4 – 2.2 % |
| PSSA–D | 1.3 … 8.1 | 4.4 … 25.5 | < 0.1 % |
| LIQA, LIQB | 2.0, 3.8 | 18 | 2.8 – 3.0 % |
| LIQC, LIQD | 15.1, 14.7 | 3.9, 4.2 | 0.1 – 0.2 % |

It would inject a 15 ns shift on LIQC/D — which §2[4] shows are already within
1 ns of the walls — and 25 ns of RMS on PSSC. `fast_singles.REPAIR_TFLASH` is
`False`, and any analysis of a reprocessed file should do the same.

## 4. The wall "cable offsets" were a reconstruction artifact

`dream_trigger.py` carries a measured table of per-segment `t_top − t_bottom` of
either ~0 or ~±32–40 ns and reads it as a cabling difference. Measured with the
**same estimator on the same bunches (1007–1156)**:

| | seg 0 | seg 1 | seg 2 | seg 3 |
|---|---|---|---|---|
| WALA official | +38.5 | −31.5 | +0.5 | +35.5 |
| WALB official | −0.5 | +38.5 | −28.5 | +1.5 |
| WALC official | +34.5 | −32.5 | +0.5 | +39.5 |
| WALD official | +31.5 | −0.5 | **−77.5** | +32.5 |
| WALA v12 | +0.5 | +5.5 | +2.5 | +2.5 |
| WALB v12 | −0.5 | +0.5 | +3.5 | +4.5 |
| WALC v12 | +0.5 | +0.5 | +0.5 | −0.5 |
| WALD v12 | +0.5 | −2.5 | −5.5 | −2.5 |

So the structure was the old flash-finder / leading-edge timing, removed by the
wall shape fitting introduced in `v4_walshapes`. **The stored table must not be
reused on a reprocessed file** — pairing the bar ends around a 38 ns offset that
is no longer there would lose most genuine pairs. Measure it on the file being
analysed; `fast_singles.measure_tb_offsets` does it in seconds.
(`match_study/scripts/tb_offset_compare.py`.)

## 5. The residual band was clock drift, not resolution

With the constants the merge was built with (K = 1.089e−4, T0 = −197.5 ns,
fitted on the *official* file) the match residual on v12 is offset by −45 ns and
**fans out with time since flash**: 68 % half-width 9.1 ns at 1–3 ms, 36.6 ns at
40–80 ms. A width proportional to elapsed time is a rate error.

Three corrections, each measured on this file:

1. **re-fit K, T0**: K = **1.10372e−4** (+1.35 %), T0 = **−253.64 ns**. Fitted
   robustly (per-time-bin median, then a line); the per-bin scatter about the
   line is 0.4 ns, and the two sub-runs agree to ±1 ns.
2. **per-arm offsets**: A −16.8, B +7.6, C +1.6, D −0.8 ns, reproducible between
   the two independent hours to ≤2.6 ns. This is a **trigger-path** delay, not a
   detector misalignment: §2[1] sees the four wall flash times within 4 ns.
3. **per-bunch clock fit** — `t_nTOF = t_DREAM(1+K+δk_b) + T0 + δa_b`, from that
   bunch's own ~100 matched triggers. δk has RMS **0.92–0.96 ppm** and is
   *structured* in bunch number (neighbouring bursts drift together, so it is a
   real oscillator drift, not fit noise); δa has RMS 6.5–6.8 ns. All 2061
   bunches fit.

**Cross-validated** (parameters from the odd-numbered triggers of a bunch,
applied to all the even-numbered ones), the residual 68 % half-width becomes:

| t since flash | before | after |
|---|---|---|
| 1–3 ms | 9.1 ns | 6.8 ns |
| 3–10 ms | 10.7 | 6.6 |
| 10–20 ms | 14.4 | 6.2 |
| 20–40 ms | 21.1 | 5.8 |
| 40–80 ms | 36.6 | 6.0 |

**Flat at ≈6 ns over the whole 80 ms.** The "37 ns DREAM trigger jitter" of
`time_align.py`'s budget was this drift envelope, not jitter — the real
DREAM↔n_TOF match resolution is ~6 ns.

## 6. The window: ±25 ns, single band

Criterion: the tightest window still within 0.5 % (relative) of the efficiency
plateau. It lands at 23.6 ns on both legs and in every time bin → quote
**±25 ns**. Accidental rates are measured with the DREAM time shifted +100 µs
(the −100 µs control agrees: 0.062 % against 0.046 %).

Wall AND plastic, corrected time base:

| t since flash | efficiency | accidental |
|---|---|---|
| 1–3 ms | 94.3 % | 0.141 % |
| 3–10 ms | 94.6 % | 0.117 % |
| 10–20 ms | 95.7 % | 0.022 % |
| 20–40 ms | 96.6 % | 0.002 % |
| 40–80 ms | 96.8 % | 0.005 % |
| **all** | **95.8 %** | **0.046 %** |

Against the as-built ±150 ns + [250,450] ns: **same efficiency** (+0.05 points),
**7.1×** less accidental background, and the two-arm ambiguity of the matched
sample drops 0.50 % → 0.15 %. Wall-only leg: 98.2 % at 0.98 %, a 5.8×
suppression.

**Drop the satellite band.** On v12 `[250,450] ns` carries no signal at all:
±150 ns alone gives 95.71 % at 0.33 %, adding the satellite gives 95.71 % at
0.54 %. Zero efficiency, +0.21 points of background. The delayed wall lobe was a
feature of the old pulse reconstruction; the plastics never had it.

## 7. What to change in the merge

1. `K = 1.10372e-4`, `T0 = -253.64 ns` — and **re-fit them per reprocessing**.
2. per-arm offsets A −16.8, B +7.6, C +1.6, D −0.8 ns.
3. per-bunch (δa_b, δk_b) from the bunch's own matched triggers; two-pass,
   ≥20 triggers per bunch (2061/2061 qualify on this pair).
4. accept window **±25 ns**, one band.
5. measure the wall top/bottom offsets on the file being analysed.
6. tflash repair **off** on v12.

For the Micromegas work now running in parallel: the arm assignment is what the
MM cross-check keys on, and at ±25 ns the two-arm ambiguity is 0.15 % against
0.50 %, with 7× less accidental contamination. A 6 ns match residual also means
the n_TOF wall time is now usable as an absolute reference for the MM drift
window, which a ±150 ns window could not support.

## Reproduce

`match_study/README.md` has the pipeline. Nothing here reads the official file
except `tb_offset_compare.py`, which reads both on purpose.
