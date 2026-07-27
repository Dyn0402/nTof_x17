# Handoff: DREAM ↔ n_TOF event matching

> **⚠ 2026-07-28: §5 (the missing plastic) is RESOLVED — see
> `FINDINGS_2026-07-28_pss_tflash.md`.** The official n_TOF PSS `tflash` is
> wrong in 37–85 % of bunches; with the time base repaired
> (`tflash_repair.py`, on by default in `ntof_io`) the plastic partner is
> present for 99.7 % of wall-matched DREAM events and match_window reads
> 99.9 %. The geometry hypothesis is dead; the trigger AND is real. The PSS
> *amplitude* remains broken on arms A/C/D in the official file.

**Written 2026-07-27 evening. Audience: someone picking this up cold, overnight.**
**You do not need to have run anything before.**

Everything marked **[verified]** was executed and observed in this session and the
numbers are reproducible from the commands quoted. Everything marked
**[inferred]** is a reading of the code or the geometry that has *not* been
confirmed by running it — several of the interesting claims are inferred, please
keep the distinction.

Commits: `43e82de` … `6d540e3` on `main` (10 commits), all pushed to nothing yet —
**`main` is 30 commits ahead of `origin/main`, nothing has been pushed.**

---

## 1. The one-paragraph state

`ntof_dream_merge` joins the n_TOF facility DAQ (scintillator hits, official
hit-level ROOT on EOS) to the DREAM Micromegas stream, so one record carries both
for the same beam pulse and the same event. **PLAN.md Phase 3 (which bunch) is
closed at 100 %. Phase 4 (which pulse inside the bunch) is closed to 37 ns.** The
matcher works: 88.3 % of DREAM events get a wall-SINGLES partner, with a
false-match probability of 0.2–4 % past 10 ms and 29 % at 1–3 ms. What is *not*
settled is why n_TOF records a plastic hit for only ~52 % of DREAM triggers when
the hardware trigger nominally required wall **AND** plastic. Five explanations
have been excluded with data (§5); one — geometric acceptance — survives and is
the cheapest thing left to test. **That is the job.**

---

## 2. Where everything is

| | |
|---|---|
| Code | `ntof_dream_merge/` in this repo |
| DREAM data | `~/x17/beam_july/runs/run_79/stat090_000{0,1}/combined_hits_root/` |
| n_TOF data | `~/x17/beam_july/ntof_data/run224572.root` (26.1 GB, byte-exact vs EOS) |
| Caches / figures | `~/x17/beam_july/analysis/ntof_dream_merge/` |
| Trigger config | `~/x17/beam_july/runs/run_79/stat090_*/n1081b_config.json` |
| venv | `.venv` in the repo root |

Reference pair: DREAM `run_79` sub-runs `stat090_0000` + `stat090_0001` ↔ n_TOF
run **224572**.

Paths resolve through `common/beam_july_paths.py`: `$X17_BEAM_JULY`, else
`/mnt/data/x17/beam_july` (DAQ machine), else `~/x17/beam_july` (laptop). Before
this session `pulse_match.py`, `ntof_tracking/reco/io.py` and
`track_rate_hv_time_intensity/build_cache.py` hardcoded the DAQ path and could not
run on the laptop at all.

### Modules, in dependency order

| module | what it does |
|---|---|
| `ntof_io.py` | per-bunch access into the 616 M-hit trees; PKUP beam record |
| `bunch_join.py` | Phase 3 — DREAM event → `BunchNumber` |
| `intra_burst_align.py` | Phase 4 — the clock fit `(k, t0)` and peak resolution |
| `match_window.py` | the accept window, calibrated against rate vs time |
| `dream_trigger.py` | rebuilds the N1081B SINGLES chain from n_TOF hits |
| `time_align.py` | verifies n_TOF internal timing before matching |
| `fake_trigger_study.py` | random-time control: are unmatched events spurious? |
| `mapping_and_deadtime.py` | wall×plastic mapping matrix; PSA gap distribution |
| `plot_alignment.py`, `plot_plastic_amplitude.py` | the two QA figures |

Every module runs standalone: `python <module>.py run_79 stat090_0000 224572 <n_bunches>`.
Start with `python ntof_dream_merge/bunch_join.py` (~2 min) to check the stack works.

---

## 3. The clock chain — do not re-derive these  [verified]

### 3.1 Phase 3: burst → BunchNumber, 100 %

```
median(burst_epoch − psTime) = 0.8290 s
```

reproducing the NXCALS publication latency PLAN.md measured independently from the
CSV/PKUP pair — and reproducing it **separately on each sub-run** despite different
`pulse_match` offsets (+27.917 s and +51.918 s). Residual MAD 5.3 ms, max 11.5 ms.

* stat090_0000: **1012/1012** bursts, bunches 146–1157
* stat090_0001: **1049/1049** bursts, bunches 1165–2213
* both perfectly **contiguous**, zero duplicates, nearest bunch ≥23× closer than
  the runner-up
* cross-checked: `PKUP.PulseIntensity` vs the `pulse_match` CSV intensity — two
  files sharing no code path — agree to ≤0.0005e10 on every burst

**Two PLAN.md corrections.** (a) The "126 bursts missing at exactly −1.2 s = ~12 %
n_TOF acceptance to track as an efficiency" is **not real**; matched through
`pulse_match`'s fitted offset it is 100 % and contiguous. (b) The `psTime` gotcha
is **two** 20-bunch blocks (2038–2057, 2638–2657), not one block 2038–2077, and
*interpolating is the wrong repair* — spacing is irregular (1.2–12 s) so a linear
fill lands seconds off and the match fails outright, which was costing
stat090_0001 exactly 20 bursts. Repaired instead from beam structure: PS pulses sit
on an **exact 1.2 s grid** (all 3017 spacings integer multiples, residual mod
1.2 s = 0.0000 s), and `index.Date/Time` is filled for every bunch and good to
±0.5 s — inside the 0.6 s half-period — so each bad bunch snaps to a unique grid
point exactly.

### 3.2 Phase 4: the two clocks run at different rates

The smear is not jitter. Both sides time from the same gamma flash, so a
fractional rate error grows linearly across the burst:

```
dt = t0 + k · t_since_flash        k = 108.9 ppm,  t0 = −198 ns
```

109 ppm is an ordinary free-running-crystal difference; over a 73 ms burst it
integrates to ~8 µs, exactly the observed smear. Removing it collapses the excess
to **σ = 37 ns**. Stable across independent bunch sets and both sub-runs
(k 108.6–110.7 ppm, t0 −197 to −203 ns).

**Fit `(k, t0)` per sub-run** (agreed with Dylan), comparing against neighbours if
a fit fails. `intra_burst_align.fit_clock()` does the fit; it is a 1-D scan over k
with t0 read off as the mode, so there is no seed to get wrong.

### 3.3 The accept window: two discrete bands

Measured at t > 40 ms where the accidental floor is 2 %:

```
   0–150 ns   main band       \  nothing between 150 and 250 ns,
 250–450 ns   second band     /  and ZERO counts from 500 ns out to 20 µs
```

The **second band is wall-only**: satellite/main is 0.00–0.01 in the four plastics
and 0.68–1.97 in the four walls. So the SiPM walls deliver a hit ~330 ns after the
trigger that the plastics never do — delayed bar/WLS light, SiPM afterpulsing, or
the PSA timing the later lobe of a double-peaked wall pulse. **[inferred]** which.
It is *not* wall-vs-plastic misalignment (§4.1). 31 % of events have only the
delayed wall hit, so the band must be accepted, not vetoed.

Efficiency was once reported as 66 % — that was an artifact of a ±100 ns window
discarding the whole second band. Ignore any "66 %" you find in older text.

---

## 4. n_TOF internal timing — measure it, never carry it  [verified]

### 4.1 Wall vs plastic
`mx_july_beam_qa/calib/time_offsets_run*.json` records a real −25 to −40 ns
wall-vs-plastic delay for runs 224404–224489. **It is gone by run224572**:
measured in situ, station A −0.5, B +0.5, C −0.5, D +0.5 ns, σ ≈ 13.5 ns, with a
per-wall-channel spread of RMS 1.2 ns (range −3.0…+4.3). Applying the stored
run224489 offsets would inject a −32 ns shift that is not in the data. The
channel-to-channel *shape* of the old files is stable to ~1 ns across
224404–224489, but the common level moved +33 ns by 224572 — **do not carry the
calibration across the recalibration boundary.** `time_align.py` measures it and
warns above 5 ns.

Consequence: the 37 ns match width is not detector misalignment. A wall-plastic
pair resolves to σ 13.5 ns → one n_TOF detector ≈ 9.5 ns → ~36 ns is on the DREAM
side (10 ns timestamp granularity plus trigger latency jitter).

### 4.2 Top vs bottom of a wall bar — this one bites
The two ends of a bar are **not simultaneous**. Per (arm, segment), from late hits:

```
  A  +38.5  −31.5   +0.5  +34.5        C  +34.5  −31.5   +0.5  +39.5
  B   −0.5  +38.5  −28.5   +1.5        D  +32.5   −1.5   +0.5  +32.5
```

σ ≈ 4 ns once removed. `30_trigger_emulation.py`'s bare ±15 ns pairing window
keeps only **27.6 %** of genuine pairs here, so the analog sum is never formed and
the wall trigger silently disappears — it took wall-only match efficiency from
88.8 % down to 29.7 %. `dream_trigger.measure_tb_offsets()` measures it; pair
around the measured offset with ±25 ns.

### 4.3 Wall channel layout — verified, don't re-litigate
Each wall has 4 top and 4 bottom channels. An 8×8 channel coincidence matrix gives
strongest partners **1↔2, 3↔4, 5↔6, 7↔8** in every wall (adjacent-segment terms an
order of magnitude smaller), so the channels are **interleaved** in `detn`
(1,3,5,7 top; 2,4,6,8 bottom) and the `(2g+1, 2g+2)` segment pairing is right.

---

## 5. The open question: the missing plastic

**The fact.** For clean late events, 96.5 % of DREAM events have a wall segment sum
over threshold, but only ~52 % have *any* plastic hit in the accept bands. The
hardware trigger nominally required wall **AND** plastic (N1081B M3 "Sector
coincidence (AND)"), so every DREAM event should have a plastic pulse by
construction. Requiring M2 gives 12.7 % efficiency against the wall's 88.3 %.

### Excluded, with data  [verified]

| hypothesis | test | result |
|---|---|---|
| **Tree mismapping** | 4×4 wall×plastic coincidence matrix | strongly diagonal: on-diagonal 324–742 vs off-diagonal 10–44 |
| **PSA dead time / double-pulse merging** | gap between consecutive hits in a channel | no truncation — plastic PSA resolves pulses **5–6 ns** apart, distribution rises monotonically into the smallest bin |
| **Too-high PSA amplitude cut** | low edge of the amplitude spectrum | real and worth fixing (**plastic 100 ADC vs walls 50 ADC**) but sits at ~3.1 mV, **40× below** the 118–157 mV discriminator |
| **Spurious DREAM triggers** | random-time control at t+100 µs | bounded at **~2 %** of events; the no-wall class matches plastic at 33.1 % against a 1.4 % control, so those are real triggers |
| **PSSD1 excluded by our own code** | `lemo_enables` + data | was a genuine bug of ours, cost 11 points (§6) |

The fake-trigger study is worth reading in full — it also shows the matched
plastic amplitudes are trigger-level (median 7059 ADC, 86.2 % above the ~4463 ADC
discriminator) while the control's are background (median 374 ADC, 10.5 % above).
Real triggers pick up the pulse that fired them.

### What survives  [inferred]

**Geometric acceptance.** `MX17_Full_Geant/src/DetectorConstruction.cc:698-702`
puts two wrapped bars behind each wall (`BackTapeL`/`BackTapeR` at ∓uOff), so
`PSS<arm>` detn 1/2 are the Left/Right **bar**. Two 20×30 cm bars cover 40 cm in u
but only **30 cm of the wall's 50 cm in v**, so a particle crossing the wall has
roughly a 60 % chance of also crossing a plastic — close to the measured 52 %.

**The tension you have to resolve:** if geometry is the answer, the DREAM trigger
cannot have been a strict wall∧plastic AND, which contradicts the module map. So
either the trigger is more wall-driven than `n1081b_module_map.py` implies, or
n_TOF's plastic efficiency genuinely is ~60 % of its wall efficiency for a reason
not yet found.

### Suggested attack, in order

1. **Compute the plastic-given-wall geometric acceptance in Geant4.** The model
   already has the real bar dimensions and the measured per-arm depths
   (`gap_sipm_to_plastic_cm`). If it returns ~60 %, geometry explains everything
   and the trigger is not a strict AND. Cheapest decisive test.
2. **Trace the actual trigger path.** `n1081b_module_map.py` M4/`SEC_C` is
   `OR(Singles lemo0)`; confirm from the run_79 `n1081b_config.json` output
   configuration which board/section actually drove the DREAM trigger input, and
   whether the AND was bypassed (`bypass_enable` exists in the section config).
   This settles it from the hardware side rather than statistically.
3. **Check the wall/plastic coincidence rate in n_TOF alone** against the DREAM
   trigger rate. n_TOF wall∧plastic SINGLES ≈ 423/bunch vs DREAM 106 events/burst.
   Wall-only is ~11 600/bunch, 100× the DREAM rate. If the trigger were wall-only
   the DREAM rate should track the wall rate modulo busy — it does not, which
   *supports* the AND and is in tension with (1). **Reconciling these two is
   probably where the answer is.**
4. If the plastic can be recovered, the AND cuts the candidate rate ~30× and would
   fix the early-time purity outright.

---

## 6. Bugs found — some fixed here, some still live elsewhere

**Fixed in this session:**
- `intra_burst_align` reported excess *pairs* over event count and called it the
  fraction of events matched (32.6 % printed where the distinct-event figure was
  18.9 %). Now counts unique `eventId`.
- `D_PMTS = {'D': (2,)}` was carried over from the mid-July D-L fault. **PSSD1 was
  repaired before run_79** — `SEC_D` reads back lemo 0 *and* 1 enabled, and in the
  data PSSD1 is the *stronger* partner of WALD (615 vs 133). Wrong for the trigger
  (both bars live) and wrong for hit selection regardless (the digitiser records
  both bars whatever the N1081B is doing). Now read from `lemo_enables`. Worth
  11 points of plastic match rate.
- `stage_reference_pair.sh ntof` cannot cold-start: `xrdcp --continue` errors with
  "no such file or directory (destination)" unless the target exists. Worked
  around with `touch`; **not yet fixed in the script.**

**Still live, NOT fixed — will bite anyone running them:**
- `mx_july_beam_qa/30_trigger_emulation.py` carries **three** stale constants:
  `PLA_THR` at the mid-July 0.5-MIP values (65/78/86/83 mV) where run_79 ran at
  0.90 MIP (118/139/157/134 mV); `D_PMTS = {'D': [2]}`; and `TB_MAX = 15.0` ns,
  which per §4.2 guts the wall trigger on run224572. Any run of that script after
  mid-July inherits all three.
- The plastic PSA amplitude cut is 2× the wall's (100 vs 50 ADC) and truncates the
  plastic spectrum in its bulk. That is an official-processing issue, worth
  raising with the n_TOF processing people regardless of the matching question.

---

## 7. Gotchas

- `tof` and `tflash` are **ns**, not µs. n_TOF acquisition is **80 ms/bunch** now
  (was 20 ms mid-July) — check per era.
- Each tree carries its **own** `tflash` (~11.2 µs on WALA vs ~13.3 µs on PKUP)
  because each detector has its own cable delay. Subtract each tree's own tflash;
  never take PKUP's for a scintillator hit.
- The dt search window must be centred on the **predicted** position
  `t + k·t + t0`, not the raw event time — at 80 ms the drift term is 8.7 µs and a
  window centred on the raw time misses the match entirely. This cost an hour.
- `BunchNumber` is monotonically non-decreasing in entry order in every tree,
  which is what makes `ntof_io.bunch_edges` work. It is asserted, not assumed.
- `ftst` (the DREAM 3-bit fine timestamp, 10 ns/unit) is **not** in
  `trigger_timestamp_ns` — `WaveformAnalyzer.cpp:499-505` folds it into the *sample*
  positions instead, and `trigger_timestamp_ns` is exactly 10 ns granular (mod 10
  ≡ 0 for every hit). It is not a branch in `combined_hits_root`; it lives in
  `decoded_root`, which is deliberately **not** on the laptop. Recovering it from
  the hit `time` branch does not work cleanly because the per-event MM drift spread
  (~570 ns) swamps the ≤70 ns ftst shift. It is a sub-leading correction on bands
  300 ns apart — worth doing eventually, not the blocker.
- Analysis outputs go to `~/x17/beam_july/analysis/ntof_dream_merge/`, **not** the
  repo.

---

## 8. What Phase 5 needs when the above is settled

Per DREAM event with a reconstructed MM track: the matched arm's wall / plastic /
LIQ amplitudes in mV and MeVee (`mx_july_beam_qa/calib/` has ADC→mV, per-channel
time offsets and the Y-88 absolute scale), pulse intensity, `t_since_flash`, and
E_n from ToF over the 19.5 m EAR2 path. Validation figure: track rate and
scint-tagged track rate vs `t_since_flash` with the E_n bands overlaid — the
mid-window turn-off and the ³He capture flood must land where the July QA says.

Current recommendation for the matcher: **wall SINGLES**, 88.3 % efficient,
0.2–4 % false past 10 ms, 29 % false at 1–3 ms. Carry a match-confidence flag so
downstream can cut on the early-time region rather than trusting it blindly.
