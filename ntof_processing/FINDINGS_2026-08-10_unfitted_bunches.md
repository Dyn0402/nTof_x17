# The bunches with no per-bunch clock correction are empty PS pulses

**2026-08-10.** 1,658 of the 1,660 campaign bunches that never get their own
`(da_b, dk_b)` delivered **no protons**: PKUP intensity < 10e10 and, from a
completely separate code path, `tflash = 0` — the n_TOF flash finder found no
gamma flash in them either. They are not parasitic pulses, they are not
recoverable, and there is nothing in them to recover.

Report (figures, tables): [`slim_study/unfitted/report.html`](slim_study/unfitted/report.html).
Scripts: [`slim_study/unfitted_bunches.py`](slim_study/unfitted_bunches.py) (two
stages) and `slim_study/make_unfitted_report.py`.

---

## 1. The three questions, answered

| asked | answer |
|---|---|
| how does a bunch lose its correction? | it has < 20 DREAM triggers, always because there was no beam |
| are those bunches good / recoverable? | no — no protons, no flash, no n_TOF coincidences. 0.05 % of campaign triggers, all background, all correctly unmatched |
| do they skew parasitic? | **no.** The parasitic family (45,225 bunches at ~413e10) is fitted 100.0000 % |
| is the gamma flash absent? | **yes, and that is the whole story.** `tflash = 0` on 100 % of them |

## 2. How it was measured

`clockfit.fit_perbunch` needs `PB_MIN_EVENTS = 20` DREAM triggers already inside
±200 ns of the global map before it will fit a bunch. Two stages, both in
`slim_study/unfitted_bunches.py`:

* **A** — one row per (segment, bunch) from the written slims alone: DREAM
  triggers, flash-tagged triggers, matched triggers, residual summary, first and
  last trigger time. 96,356 rows over the 119-segment campaign.
* **B** — join the n_TOF beam record (`ntof_io.pkup_bunches`): pulse intensity,
  `tflash`, `psTime`, for all 44 n_TOF runs.

## 3. What it says

**The threshold is never marginal.** Unfitted bunches hold 0–19 physics
triggers (median 1); fitted ones hold 46–139 (median 92). The two populations do
not touch, so no bunch is lost to match quality and lowering the threshold
recovers nothing.

**The trigger deficit is a beam deficit.** Fitted-bunch intensity is bimodal —
parasitic ~413e10 (47.8 %) and dedicated ~851e10 (52.2 %) — and the unfitted
population sits at zero, 1,658 of 1,660. The DREAM rate is ~1.2 kHz in a beam
burst and ~22 Hz in an empty one, which is the detector background: DREAM's gate
opens on the PS timing whether or not protons arrive, so an empty pulse still
produces a "burst" of a couple of dark-count coincidences, and `bunch_join`
labels the first of them `is_flash` — which is why their time base is
meaningless and **not one of the 2,764 triggers matches**.

**Nothing is there to find.** On the worst segment (`run_116/stat090_0014`,
22.3 % empty), empty-pulse triggers carry 2.82 slim hits each against 16.8 for
beam triggers, and the composition gives it away: WAL 2.80, PSS 0.017, LIQ
0.000 per trigger. SiPM dark counts in the ±1 µs window; the plastics and
liquids see nothing.

**Two exceptions**, both real beam: `run_77/stat090_0003` bunch 44 and
`run_118/stat090_0019` bunch 1758, with 19 and 14 triggers. Both are the *first*
bunch of their sub-run — a burst DREAM joined part-way through. 33 events.

## 4. Consequences for the QA and the format

1. **`clock_qa`'s 'bunches fitted' check is measuring beam availability, not the
   clock fit.** In 114 of 116 segments the unfitted count *equals* the empty-pulse
   count exactly, and every PKUP bunch in a segment's range reaches the slim
   (verified on 224636 and 224603: zero missing), so the fitted fraction *is* the
   delivered fraction. Its four WARNs are hours when the PS delivered 86–92 % of
   the pulses n_TOF was scheduled — 224636 was 11.8 % empty over the whole run,
   224603 10.2 %. Nothing is wrong with those segments' clocks.
2. **The slim carries 4,424 no-beam triggers** (2,764 physics + 1,660 flash-tagged,
   0.05 % of the file), all with `matched = 0`, all with a `t_since_flash`
   referenced to a background trigger rather than a flash. Any analysis that
   counts unmatched triggers, or that trusts `t_dream_ns`, should exclude them.
3. **Applied on 2026-08-10, in the pipeline:**
   * `config.EMPTY_PULSE_E10 = 10` (and `PARASITIC_E10 = 600`, reporting only).
   * `slim.bunch_table` drops empty pulses at the **join**, before the candidate
     pass reads anything, and the `bunches` tree gains `has_beam` and
     `intensity_e10` — spanning every bunch the sub-run touched, so the table is
     both the beam record and the record of what was dropped. A bunch whose
     intensity is NaN (a burst the join could not place) counts as beam and is
     kept: an unknown is not a reason to throw data away.
   * `qa.json` gains `beam_availability`, `n_bunches_empty`, `n_triggers_empty`,
     `parasitic_fraction`, `intensity_median_e10`.
   * `clock_qa` asks 'bunches fitted' **of the bunches that had beam**, adds
     'no-beam pulses filtered out' (FAIL if any survived, NA on older files),
     and reports availability and beam mix without ever judging them — the PS
     is not something the QA can act on. Legacy files fall back to the old
     scope and say so in the check detail.
   * A guard on the filter itself: an empty pulse holds 1–2 triggers against
     ~92 in a beam bunch, so a *dropped* pulse holding a full burst is not a
     beam statement — it means bursts are landing on the wrong bunches.
     `clock_qa` checks the ratio ('dropped pulses look like no beam', warn 0.25,
     fail 0.50) and `slim` logs it during the run. This was written because the
     smoke test found one: `run_116/stat090_0013 × 224636`, a 13 %-overlap
     proposal whose join fitted a **−1,324 s** burst-to-pulse offset and paired
     unrelated bursts to unrelated pulses, showing 22 "empty" bunches with
     66–108 triggers each. That segment is not in the campaign's 116 — it fails
     its clock fit — but the ratio names the cause minutes earlier, and it is
     the only check that can see a mis-assigned join from the written file.
   * `clock_qa`'s CLI no longer aborts the whole sweep when one directory is
     unreadable; it reports the failures and exits non-zero.
   * `tests/test_clock_qa.py` gains five cases (empty pulses with a healthy
     beam must stay PASS; a real deficit on top must still FAIL; leaked no-beam
     triggers must FAIL; a legacy file must be NA), and now reseeds per case so
     one case's draws cannot move another across a threshold.
   * The dashboard grows a "The beam" section: availability and parasitic
     fraction, stated as beam facts rather than quality metrics.

   **This takes effect on a re-slim only.** The 119 campaign segments on lxplus
   predate it, and their `frac_fitted` is beam availability, not fit quality.

4. **Smoke-tested on n_TOF 224636** (the worst run for availability: 493 of
   4,179 pulses empty), condor 11983888, 4 segments, 85 min.

   | | pre-filter | filtered |
   |---|---|---|
   | triggers written | 81,088 | 80,405 (−683, none of which matched) |
   | hits | 1,931,825 | 1,930,505 (−1,320, all wall dark counts) |
   | bunches fitted | 925 of 1,190 → **WARN** | 925 of 925 with beam → **PASS** |
   | efficiency | 94.58 % | 94.77 % |
   | per-arm residual centring | ≤ 0.2 ns | ≤ 0.5 ns |

   The drop is exactly the predicted set. Two honest caveats:

   * On a segment that *has* empty pulses the calibration moves a little, so a
     re-slim is not a pure subtraction there. The cause is not the filter as
     such: `measure_tb_offsets` samples the first 150 bunches, that sample is
     now 150 *beam* bunches, and the modal top/bottom offsets are quantised at
     1 ns and moved by one bin on several bar groups. It propagates to a
     coherent +1.1…+1.9 ns in the four arm offsets and flips 0.24 % of the
     triggers — 168 gained, 20 lost, and the flipped ones move from |residual|
     13.6 ns to 5.5 ns, i.e. toward the centre. Both builds pass 'per-arm
     residuals centred'. If the 1 ns sampling noise is ever worth removing, the
     lever is `OFFSET_BUNCHES`, not this filter.
   * `run_116/stat090_0017` joined **zero** events, and the first version of
     this change reported it as "every joined bunch was an empty pulse" — a lie
     about the accelerator. Fixed: the two are now distinguished and tested.

5. **Unrelated bug the smoke test surfaced**: `calibration.json` recorded
   `tb_offsets_ns` as `[0, 1, 2, 3]` for every arm in **every slim ever
   written**, because `measure_tb_offsets` returns `{group: offset}` and the
   writer iterated it, yielding the group indices. The data was never affected
   (the real dict goes to `fast_singles.all_arms`), but a provenance sidecar
   that misreports the calibration is the one thing it exists not to do. Fixed;
   files written before 2026-08-10 carry the wrong record, and the true values
   are in the job logs.

## 5. By-product: the fleet efficiency spread is the beam mix

Per-segment match efficiency against that segment's parasitic fraction:
**r = −0.82** over 116 segments. Measured per family, 97.7 % on dedicated pulses
against 91.2 % on parasitic; the per-segment line extrapolates to 97.8 % / 92.1 %.
The parasitic fraction runs 9 %–61 % across the campaign, which accounts for most
of the 93.58–97.31 % efficiency range that
[`SLIM_CAMPAIGN_2026-08-09.md`](SLIM_CAMPAIGN_2026-08-09.md) reported as
unexplained scatter.

**Why parasitic pulses match ~6 points worse is not established.** Half the
protons means fewer n_TOF candidates per trigger, so a DREAM trigger is more
often left without a partner above the singles threshold — plausible, unmeasured.

## 6. What this does not touch

The 54 sub-runs that could not be slimmed at all (the ~0.982 ms, ~6 µs-wide
association) are a separate and still-open question. Empty pulses are not
involved: those segments failed at the coarse search, before any per-bunch fit.
