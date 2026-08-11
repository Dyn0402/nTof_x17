# The runs we processed ourselves are equivalent to n_TOF's own

**2026-08-11.** The 17 runs the self-processing campaign has moved to
`/eos/experiment/ntof/data/x17/reproc/prod_v12/` were checked against the runs
n_TOF processed themselves in `/eos/experiment/ntof/processing/official/completed/`.
**They pass.** The configuration is byte-identical to n_TOF's, every partial reads,
and on the 13 runs that have beam every measurable quantity overlaps the official
range.

Report (tables, figures): [`campaign_qa/results/report.html`](campaign_qa/results/report.html).
Scripts and the gotchas: [`campaign_qa/README.md`](campaign_qa/README.md).

---

## 1. The awkward part

> **Superseded in part, the same day.** Later on 08-11 n_TOF merged 27
> previously-unmerged runs, two of which (224573, 224577) we had processed
> ourselves — so runs *do* now exist in both sets, and the direct comparison
> passes bit for bit. See
> [`FINDINGS_2026-08-11_official_ledger.md`](FINDINGS_2026-08-11_official_ledger.md).
> What follows still holds for the 224688-224718 block, which n_TOF has not
> touched.

**No run of the block we are processing exists in both sets.** 224688-224718 is
exactly the block n_TOF never processed — that is why we are processing it. So this
cannot be a file-against-file diff. It is an equivalence argument in two parts:

* **configuration**, compared exactly, because every product records the UserInput
  it was made with in its own `history` object;
* **behaviour**, compared statistically against the nearest official runs in time
  (224660-224676), with everything normalised to delivered protons — a raw hit count
  is a beam measurement, not a processing measurement.

## 2. Configuration: identical, not merely similar

n_TOF's production `UserInput_2026_EAR2_X17_v4.h` **is** our `v12_liqpileup`. They
adopted it after the July handoff, so both processings run the same recipe — the
file in `official/completed/224687/history_224687.root` still carries our
`# X17 EAR2 2026 -- variant v12_liqpileup` comment block verbatim, and so does
`done/run224297.root` from July.

| compared | result |
|---|---|
| parameter columns, all 65 lines | **identical** |
| the 26 referenced pulse-shape templates, by md5 | **26 of 26 byte-identical** |
| what differs | line 0 (`UserInput.h` vs `UserInput_2026_EAR2_X17_v4.h`) and the template *directory* — our AFS staging vs their EOS `shapes_X17_v4` |

A raw md5 of `history` therefore never matches and that fact carries no physics;
`campaign_qa/history_diff.py` drops path prefixes and diffs what is left.

**This is what the check is and is not.** It establishes equivalence *to the
official product*. A defect in the shared recipe affects ours and n_TOF's
identically and is invisible here.

Housekeeping: three files in our staging directory
(`X17_WALA_Signal_3.txt`, `X17_WALB_Signal_0.txt`, `X17_WALC_Signal_0.txt`) are
unreferenced leftovers — the UserInput points only at the `_avg0/1/2` set. They
were not used; clear them so a future reader does not think otherwise.

## 3. Structure: every partial, not a sample

`campaign_qa/verify_transferred.py` opens **every** partial, requires all 16
top-level objects, and issues a real array read on all 14 hit trees so a truncated
basket is hit rather than only the header.

| | |
|---|---|
| runs on the ntof disk | **17** (224688-224700, 224706, 224716-224718) |
| partials / volume | **494 / 384 GB** |
| unreadable files | **0** |
| runs not contiguous, or off `ceil(raw/4)` | **0** |
| bunch gaps within a run | **0** |
| runs with a different `history` md5 | **0** |

## 4. Physics, on the 13 runs with beam

`campaign_qa/compare_campaign.py`, one partial per run, gated on `PKUP amp > 0`,
normalised to the protons those bunches carried.

| axis | ours (13 runs) | official (6 runs) | verdict |
|---|---|---|---|
| hits per 1e12 p, WALA | 1483-1556 | 1470-1558 | overlap |
| hits per 1e12 p, PSSC | 6813-7083 | 6899-7073 | overlap |
| hits per 1e12 p, LIQA | 2647-2706 | 2637-2659 | overlap |
| **all 12 trees** | | | **all overlap** |
| median hit amplitude, **all 12 trees** | | | **all overlap** |
| modal `tflash`, per tree | | | **same 10 ns bin as official on every tree** |
| bunches > 150 ns off flash | **0.00 %** | 0.00 % | target < 2 %; broken July processing was 37-85 % on PSS |

The slow rise on LIQA/LIQD across run number is a **time trend that runs through
both sets** — official 224660 sits at 2637 and our 224700 at 2688 — not a step at
the boundary between the two processings.

**Arm offsets.** Prompt-coincidence peak of large plastic hits and of liquid hits
against the same arm's wall, after removing each tree's modal `tflash`: |peak| ≤ 1 ns
almost everywhere, except **PSSC at 33-35 ns** and PSSD at 27 ns in some runs. **The
official runs show the same thing in the same trees** (PSSC 33-35 ns on
224660/224667/224674/224676), so it is a property of those channels under this
recipe, not something our processing introduced. Worth chasing separately; not a
campaign defect.

**Hit quality, not just hit count.** A processing can buy hits by measuring them
worse, which no rate comparison would show. `quality_metrics.py`, two partials each
of our 224691 against official 224672, everything accidental-subtracted:

| metric | ours | official | |
|---|---|---|---|
| T1 wall top↔bottom σ | 6.66 ns | 6.69 ns | +0.5 % |
| T2 wall↔plastic σ | 6.12 ns | 5.86 ns | −4.2 % |
| T3 timing walk over amplitude deciles | 1.43 ns | 1.37 ns | −4.3 % |
| A1 MIP peak | 1057 ADC | 1047 ADC | −0.9 % |
| A1 MIP relative width | 1.22 | 1.22 | equal |
| A2 log-ratio residual | 0.37 | 0.38 | +0.5 % |

Different runs, so the few-percent T2/T3 spread is within run-to-run variation.

## 5. Two traps that make a healthy run look broken

**The official runs next door have no beam.** 224678-224687 all sit at zero
`PulseIntensity` and zero PKUP amplitude. Picking 224687 — the run immediately
before our block — as the control makes our output look **400× too busy** and its
partials 170× too large. It is not; that run had no protons, and its 4.4 MB partials
are what a quiet run costs.

**Empty PS pulses inside our own runs.** Those bunches have no gamma flash, so
`tflash` is 0 or garbage and every flash check flags them — the mechanism already
established in [`FINDINGS_2026-08-10_unfitted_bunches.md`](FINDINGS_2026-08-10_unfitted_bunches.md).
The **first partial of 224692 is 75 % empty**, which alone reads as 25 % off-flash on
the walls and 69-75 % on the plastics. Whole-run, 224692 is **98.0 % beam** and its
intensity-normalised rates match the other twelve runs to ~1 %; of its beam bunches,
**0.0 %** have `tflash = 0`.

Both traps are now handled by the tooling: `campaign_qa/beam_state.py` reads the
whole-run beam state from the `index` tree (replicated in full in every partial, so
one open per run), and every number in `compare_campaign.py` is gated on protons.

## 6. Four of the transferred runs have no beam at all

| run | bunches | with protons |
|---|---|---|
| 224706 | 2 615 | **0** |
| 224716 | 4 | **0** |
| 224717 | 8 | **0** |
| 224718 | 16 | **0** |

They processed correctly and are simply quiet — a few MB per partial instead of
~800 MB. Expect more of these through 224701-224718, and note that **the acceptance
test for them can only be structural**: with no protons there is no flash and no
rate to compare.

## 7. What this does not rule out

* **Shared systematics.** See § 2 — equivalence, not absolute correctness.
* **The physics comparison samples one partial per run** (two for the quality
  metrics), 63-80 bunches out of thousands. The structural pass is the one that
  covers every file; a defect confined to late partials would survive this.
* **The 14 runs still in flight** (submitted 13:24 on 08-11) are not covered.
* **No downstream check.** Nothing here runs the DREAM slim over the new block, so
  its association efficiency and clock QA are still open — and those are what would
  catch a timing problem that survives everything above.
