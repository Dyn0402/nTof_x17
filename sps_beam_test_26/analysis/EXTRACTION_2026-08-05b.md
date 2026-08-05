# The follow-up pass — 2026-08-05 (late), with lxplus access

`EXTRACTION_2026-08-05.md` closed with six named follow-ups. This is what
happened to five of them once `ssh lxplus` was available again. Headlines:

- the **Saturday CO₂ 25° ladder is measured** — its top point gives
  v(CF₄)/v(CO₂) = **1.14** against the flush transient's independent **1.17**;
- the **shared-σ_p0 joint ladder fit runs, and returns a negative result**:
  σ_p0 is not identifiable even with the ladder tied together, and the reason
  is the χ² floor, not the parameterisation;
- the **referenced RAW cross-check passes** — the det4-only leading-strip
  selection is not biasing the kernel (τ within 6 %, c1 within 4 %, ±1 peak
  shift +60 ns at all three fields), after fixing a selection bug that had
  made it look unusable;
- run_55's plateau windows were **wrong by 20–45 s each** and are now measured;
- **run_61's 15° ladder cannot give v(E) at all** — it was recorded at 32
  samples and the drift ladder does not fit in the 1.92 µs window. Closed, not
  deferred;
- **det3's lost calibration is rebuilt** under `share_lp` and reproduces the
  dead bundle on every metric (σ_θ 1.176/1.143°, within-5 mm 93.41 %).

All six follow-ups are closed. Then, asked whether the SPS test was *finished*,
the EOS directory was audited run-by-run against the analysis record rather
than against the documents' own summaries — and it was not:

- **eight runs carrying det4 data appear in no analysis table** (§7). The whole
  Monday-morning flat block, runs 64–70, is missing from every document
  because RUN_TIMELINE's narrative stops on Sunday. Their `cfg_gain*_peaktime*`
  sub-run names are **P2's VMM settings, not ours** — verified against the
  Dream configs, which are byte-identical across the supposed scan (§7);
- **run_62 is a second, independent 25.64° CF₄ drift ladder** under conditions
  identical to run_63's. Pulled, decoded and spanned here: the two ladders'
  v(E) curves agree to **0.6 % RMS** (§8). The campaign's most load-bearing gas
  number is now reproduced rather than resting on one dataset;
- two documented claims are wrong: there *are* flat CF₄ drift scans after the
  access (runs 68 and 70), and there are *two* CF₄ ladders, not one.

So the answer to "is it closed?" is **no** — see §11 for what is left and what
it is worth.

## 1. The Saturday CO₂ 25° drift ladder — pulled and measured

`datasets.py: run57_rot25_co2` (new). Pulled from EOS with
`staging/pull_wave3.sh` (new): **the high-field half only**. run_58's three
`operating_*` sub-runs are 86 GB of FEU03 against 37 GB of free disk, and
their points (500 V and below) are where the ladder starts running off the
3.84 µs window anyway. What was taken — run_57 `driftscan_gap350V` +
`gap400V`, 29 GB — carries drift 700/650/600/550 V = **243/226/208/191 V/cm**,
deliberately the same range as the CF₄ ladder's top four
(235/217/200/182 V/cm), so the two mixtures compare point for point.

Everything about the ladder is now dated to the second from the run_57+run_58
`hv_monitor` traces (58,115 rows, no gap 16:35 → 09:26+1) — all 17 points, not
just the staged four; see the table in `datasets.py`. Two things that were not
in the record before:

- **resist was 669.75 V** during the ladder, not run_60's 649.75;
- run_58's **first point (500 V) is not gain-comparable**: its resist reads a
  mean of 623 V, still recovering from the 18:30:47–18:32:53 end-of-run
  power-off between run_57 and run_58.

### 1a. The ZS trap, and why the 700 V window is short

Both sub-run directories carry the 16H21 pedestal run's `_thr.prg`, header
*"Threshold value: 5.000000 sigmas"*. It is **stale**: our own script dropped
det E to 2σ at 18:04 and raised it to 3σ at 18:12 without a new pedestal run.
`RUN_TIMELINE.md` §3's rate table is the evidence — FEU3 goes 2.08 → 23.76
MB/s across exactly that boundary. So the 700 V dwell straddles a threshold
change, and `d700`'s window starts at 18:12:30 to keep all four plateaus at
3σ. This is the second time on this campaign that the `_thr.prg` header has
been the wrong authority (the first, run_56, was the opposite direction —
§3b of RUN_TIMELINE).

### 1b. The result, and the mixture comparison

`ladder_span.py run57_rot25_co2 --c0 30`:

| plateau | V/cm | hits | t10 | t90 | span | v [µm/ns] |
|---|---|---|---|---|---|---|
| d700 | 243 | 504k | 812 | 3178 | 2366 | **12.33** |
| d650 | 226 | 1,360k | 810 | 3279 | 2469 | 11.81 |
| d600 | 208 | 1,535k | 819 | 3302 | 2482 | 11.74 |
| d550 | 191 | 948k | 817 | 3300 | 2482 | 11.74 |

The bottom two points are equal to the nanosecond, which is the window
truncation signature again (t90 3300 against the 3.84 µs edge) — lower bounds,
same as the CF₄ ladder's 165/148/113 V/cm points. **The top two are the
measurement.**

Against the CF₄ ladder at the same mount angle, same estimator, same shaping
constant:

| | CO₂ (run_57) | CF₄ (run_63) |
|---|---|---|
| 243 V/cm | **12.33** | — |
| 235 V/cm | — | **14.00** |
| 226 V/cm | 11.81 | — |
| 217 V/cm | — | 13.22 |
| ratio at ~240 V/cm | **1.14** | |

and the gas-flush transient (`GAS_FLUSH_TIMELINE.md` §4a, run_60, a completely
different dataset and method — the span stepping *within* one run as the
mixture exchanged) measured **1.17** at 243 V/cm. Two independent routes to
the same mixture ratio, agreeing to 3 %. The flush measurement's absolute CO₂
span (2340 ns) also lands on this ladder's 2366 ns at the same field, at a
different resist and on a different day-half.

### 1c. A bug this exposed in `ladder_span.py`

The v column above needs `--c0 30` because **the anchor does not transfer
between mixtures**. `span = gap/v + c0`: the shaping constant `c0` is an
electronics property (same shaping, same 64×60 ns window) and *does* transfer,
but the 14 µm/ns anchor is a wet-CF₄ measurement. The old auto-anchor took any
plateau within 10 V/cm of 233 V/cm, and the CO₂ ladder's 243 V/cm point sat
exactly inside that — so the CO₂ curve silently rescaled itself to make its own
top point read 14.0 instead of 12.3 µm/ns, erasing precisely the mixture
difference the ladder exists to measure. `ladder_span.py` now refuses to
auto-anchor on a gas other than the anchor's (`--anchor-gas`, default CF4),
tightens the tolerance to ±5 V/cm, and takes `--c0` directly.

### 1d. run_55 (flat CO₂) re-read against it

The flat CO₂ ladder's plateau windows were the last approximate thing in
`datasets.py`. Measured from the four staged `meshscan_*/hv_monitor.csv`
(2,434 rows at 1 Hz, 14:13:43–14:55:05, no gap):

| | measured dwell | old window | error |
|---|---|---|---|
| 700.3 V | 14:13:43–14:25:45 | 14:15:30–14:26:30 | ran **45 s past** the step |
| 600.3 V | 14:25:48–14:35:51 | 14:28:00–14:36:30 | 39 s past |
| 500.3 V | 14:35:54–14:46:08 | 14:37:00–14:46:30 | 22 s past |
| 400.1 V | 14:46:11–14:55:05 | 14:47:00–14:55:06 | ok |

Each old window leaked into the *next, slower* point — a drift-time
contamination of exactly the kind the span estimator is sensitive to. With the
measured windows (and `--c0 30`):

| plateau | V/cm | span, old windows | span, measured | v [µm/ns] |
|---|---|---|---|---|
| d700 | 243 | 2188 | 2184 | 13.37 |
| d600 | 208 | 2564 | 2508 | 11.62 |
| d500 | 174 | 2338 | 2399 | 12.15 |
| d400 | 139 | 2363 | 2362 | 12.35 |

**The non-monotonicity survives the fix** — so it is not a windowing artefact,
and it is not the amplitude cut either (the span ordering is stable across
A>100/150/250/400/600, checked). What it most likely is: run_55 ran at **5σ**
ZS where run_57 ran at 3σ, and at low drift field the deeper, more diffused
charge falls under threshold first, shortening the *apparent* ladder and making
low-field v read high. Supporting that reading, the two CO₂ datasets agree
where the effect should be smallest and disagree where it should be largest:
208 V/cm gives 11.62 (run_55) vs 11.74 (run_57), while 243 V/cm gives 13.37 vs
12.33. **Quote run_57 for CO₂ v(E), not run_55.** run_55 remains the flat-CO₂
dataset for anything that is not a drift time.

## 2. The shared-σ_p0 joint ladder fit — run, and negative

`wft_ladder_joint.py` (new). §4 of the previous pass identified this as the
next step and called it cheap; it is now done, and the answer is that it does
not work — for a reason worth recording.

The lever is that the two degenerate parameters depend on the drift field
*differently*: σ_p0 (initial cloud at the mesh) is field-independent, Dp
(diffusion) is not. So σ_p0 is fitted once for the whole ladder while (v, Dp)
stay per plateau. The script profiles rather than minimising in 15 dimensions —
for each σ_p0 on a grid, each plateau's (v, Dp) is minimised in its own
process — which gives the profile curve, i.e. an actual uncertainty, instead of
a point.

**Run 1 (sign auto-detected per plateau)** produced two symptoms that matter
more than its numbers:

- the joint χ²(σ_p0) profile is **not parabolic and has no clear minimum**:
  1.709, 1.717, 1.688, 1.703, **1.662**, 1.683, 1.706, 1.667 ×10⁷ across
  σ_p0 = 0.03…0.80 mm. The two lowest points are 0.35 and 0.80 mm — as far
  apart as the grid allows.
- the auto-detected **rotation sign disagreed between plateaus** (−1 on five,
  +1 on two). The mount does not turn round between drift points. A
  per-plateau sign is the fit finding different local minima, not different
  geometry.

**Run 2** therefore forces one sign for the whole ladder (`--sign -1`, the
consensus: the five −1 votes had margins up to 20 %, both +1 votes under 15 %).
Removing that spurious freedom makes the result *worse-behaved*, not better:

| σ_p0 [mm] | 0.03 | 0.08 | 0.15 | 0.25 | 0.35 | 0.45 | 0.60 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| joint χ² ×10⁷ | 1.740 | 1.741 | 1.721 | 1.731 | 1.710 | 1.742 | 1.752 | **1.708** |
| χ²/dof | 172.8 | 173.0 | 171.4 | 171.4 | 169.4 | 172.5 | 172.9 | 168.8 |

The minimum is **at the upper guard**, the curve is not monotonic on either
side of it, and the whole 0.03 → 0.80 mm range spans 2.6 % in χ². The ladder
at that "minimum":

| plateau | 235 | 217 | 200 | 182 | 165 | 148 | 113 V/cm |
|---|---|---|---|---|---|---|---|
| v [µm/ns] | 16.46 | 16.13 | 13.30 | 15.40 | 12.79 | 14.93 | 14.92 |
| Dp | 0.022 | 0.0015 | 0.092 | 0.133 | 0.108 | 0.0014 | 0.080 |
| χ²/dof | 188 | 215 | 158 | 131 | 131 | 218 | 146 |

Dp scatters over two orders of magnitude with no field dependence, and v is
non-monotonic and sits ~15 % above the span estimator at every field.

**The conclusion is that the joint fit is not the missing lever.** σ_p0 is
unconstrained across the entire physical range at the 3 % χ² level; sharing it
across seven plateaus does not change that, because what limits the fit is not
the number of free parameters but the **χ²/dof ≈ 170 systematic floor** — the
noise model and the thin rot25 alignment, exactly as the previous pass
diagnosed. Note also that the auto-sign run reached a *lower* total χ²
(1.662 vs 1.708 ×10⁷): a fit that gains 2.8 % from letting the mount rotate the
other way between drift points is not measuring geometry.

So the next step on σ_p0 is not another fitter. It is (a) a real per-sample
noise model instead of the flat 10 ADC, and (b) a rot25 alignment that is not
thin — and until one of those exists, **σ_p0 = 0.404 mm from the bench arm C
remains the only supported value**, and the ladder's contribution stays what
the span estimator already gives: v(E).

## 3. The referenced RAW path — the cross-check now passes

§4b left this as "runs end to end, but the quantitative comparison still needs
the robust-library aggregation". Doing that turned up a selection bug first.

**The bug.** `robust_waveforms.py` picks each event's *leading* trace as the
central strip. `extract_det4_only.py` stores ±4 strips about the cluster, so
"leading trace of the event" there already means "leading trace near the
track". `flat_align_eff.py` writes **all 512 channels** — and over 512 channels
the leading trace is whichever channel swings hardest, which on run_71 is
ch 510 or 372, the two known oscillators. They won **1,859 of 1,943** Y events.
The Y view of the referenced selection was not noisy, it was 96 % oscillator.
`--win-strips 4` restricts the leading-strip search to the reference window and
the two selections mean the same thing again.

**The comparison**, same script, same q0 gate (200–3000 ADC), same fitter,
Y view:

| plateau | | τ_s [ns] | c1 | c2 | α(±1) | ±1 area | ±1 peak | n |
|---|---|---|---|---|---|---|---|---|
| raw275 | det4-only | 873 | 0.542 | 0.449 | 0.199 | 0.738 | 0.303 | 5,941 |
| | **referenced** | 854 | 0.563 | 0.440 | 0.203 | 0.757 | 0.304 | 127 |
| raw450 | det4-only | 922 | 0.554 | 0.468 | 0.209 | 0.728 | 0.314 | 7,543 |
| | **referenced** | 977 | 0.582 | 0.474 | 0.204 | 0.728 | 0.307 | 175 |
| raw700 | det4-only | 941 | 0.549 | 0.474 | 0.207 | 0.736 | 0.321 | 9,444 |
| | **referenced** | 913 | 0.539 | 0.459 | 0.207 | 0.723 | 0.318 | 293 |

τ within 6 %, c1 within 4 %, c2 within 2 %, α within 2 %, ±1 area within 2.5 %,
±1 peak within 1 % — on samples that differ by a factor 30–50 in size and,
more to the point, are chosen by two different things: det4's own charge in one
case, an external tracker in the other. **The det4-only selection is not
selecting on the thing it measures.** The event-wise ±1 peak shift comes out
**+60 ns at all three drift fields** on the referenced sample too, matching the
det4-only value and the bench.

Drift invariance on the referenced sample: τ ±6.7 %, c1 ±3.9 %, c2 ±3.7 %
(det4-only: ±3.7 / ±1.1 / ±2.8 %) — consistent, with the extra spread the
30× smaller sample buys. **§4b is closed.**

## 4. run_61's 15° drift ladder — closed, because it cannot be done

`datasets.py: run61_rot15_ladder` (new): `meshscan_m20V` + the **13H46** pass
of `meshscan_m30V` (the 16H08 pass of the same name is at 25.64° with the
resist scan creeping underneath it — the trap the previous pass flagged;
the entry lists the stem explicitly so nothing can glob it wrong). All ten
drift points come from `det4_drift_scan.log`, which is a scripted, fully
time-stamped scan (resist held 750.0 V, 700→70 V, 5 min each), so the windows
come from the driver rather than the monitor.

`ladder_span.py` on it:

| plateau | 243 | 219 | 195 | 170 | 146 | 122 | 97 | 73 | 49 | 24 V/cm |
|---|---|---|---|---|---|---|---|---|---|---|
| t90 [ns] | 1694 | 1693 | 1692 | 1691 | 1689 | 1688 | 1685 | 1683 | 1680 | 1677 |

t90 is **pinned at the window edge at every field**. Confirmed from the decoded
data, not the config: run_61's decoded `sample` index maxes at **31**
(1920 ns window) against run_57's **63** (3840 ns). The CF₄ drift ladder is
2000–2500 ns long. **It does not fit.** There is no v(E) in this dataset and no
analysis can put one there — which is the answer to why it was never analysed,
and it should stop appearing on follow-up lists.

What the dataset *does* carry is the charge lever, and that is clean:

| V/cm | 243 | 219 | 195 | 170 | 146 | 122 | 97 | 73 | 49 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| hits/event | 8.17 | 7.74 | 7.27 | 6.78 | 6.35 | 5.90 | 5.45 | 5.06 | 4.77 | 4.54 |
| median amp | 209 | 210 | 202 | 195 | 187 | 178 | 169 | 160 | 155 | 153 |
| p90 amp | 1100 | 1129 | 1120 | 1110 | 1109 | 1122 | 1104 | 1072 | 1126 | 1227 |

Strip multiplicity falls monotonically by a factor 1.8 and the *median* hit
amplitude by 27 % as the drift field drops 10×, while the **p90 amplitude does
not move at all**. That is the mesh-transparency / attachment curve at
15.465°: fewer primaries arrive, so small hits fall under threshold and
multiplicity drops, but wherever a large cluster lands the signal is unchanged.

## 5. Files

New: `wft_ladder_joint.py`, `staging/pull_wave3.sh`, `datasets.py` entries
`run57_rot25_co2` and `run61_rot15_ladder`.
Changed: `ladder_span.py` (`--c0`, `--anchor-gas`, `--anchor-tol`),
`robust_waveforms.py` (`--referenced`, `--win-strips`, `--tag`),
`datasets.py` (`run55_flatdrift` plateaus measured).

```bash
cd sps_beam_test_26/analysis
../../.venv/bin/python decode_dataset.py run57_rot25_co2 --jobs 4 --feus 03
../../.venv/bin/python ladder_span.py run63_rot25                 # c0 = 30 ns
../../.venv/bin/python ladder_span.py run57_rot25_co2 --c0 30
../../.venv/bin/python ladder_span.py run55_flatdrift  --c0 30
../../.venv/bin/python decode_dataset.py run61_rot15_ladder --jobs 3 --feus 03
../../.venv/bin/python ladder_span.py run61_rot15_ladder          # window-railed
../../.venv/bin/python robust_waveforms.py run71_raw --referenced --win-strips 4 \
    --q0 200,3000 --tag _referenced_q200 --wf .../wf_run71_raw.npz
../../.venv/bin/python robust_waveforms.py run71_raw --q0 200,3000 --tag _det4only_q200
../../.venv/bin/python kernel_refit_clean.py run71_raw --view y --lib <either>
../../.venv/bin/python wft_ladder_joint.py run63_rot25 --view x --sign -1 \
    --events-per-plateau 150 --jobs 7
```

det3's rebuild, from the repo root (`W` = det3's `mx17_3/wft` directory):

```bash
.venv/bin/python -m wft.calibrate sat_det3 --jobs 12 --share-mode lp \
    --fix-hyper "c1=0.306,c2=0.12" --fix-v 36.6 --out $W/calib_bundle_lp_sp0free
.venv/bin/python -m wft.cli reco sat_det3 --matched-only --jobs 12 \
    --bundle $W/calib_bundle_lp_sp0free --out $W/events_lp.parquet
.venv/bin/python mx_june_wft/01_alignment.py sat_det3 --table $W/events_lp.parquet \
    --out $W/alignment_lp
.venv/bin/python mx_june_wft/03_angles.py sat_det3 --table $W/events_lp.parquet \
    --alignment $W/alignment_lp/alignment.json --out $W/angles_lp
.venv/bin/python mx_june_wft/02_efficiency.py sat_det3 --table $W/events_lp.parquet \
    --alignment $W/alignment_lp/alignment.json --max-dropped -1   # no --out flag
```

## 5b. The report

`make_figures.py` + `make_report.py` (new) build
`~/x17/sps_beam_test_26/extraction_2026-08-05/report.html` from the JSON
products, so re-running the analysis updates figures, tables and verdict
together. The DAQ page's Analysis tab lists and opens it; figures are
referenced with relative links so the file also works from disk.

## 6. det3 rebuilt under `share_lp` — the lost bundle is recovered

This item was larger than the follow-up line implied: det3 had **no
`calib_bundle` on this machine at all**. Its wft directory held only
`calib_work/calib_cache.pkl` and `beam_xcheck` — the bundle behind the quoted
σ_θ 1.20/1.14° went with the campaign desktop. So this was not "refit det3
under lp", it was "rebuild det3's calibration from the cache, under lp".

Recipe, the det4 arm-C analogue: `share_mode lp`, kernel pinned to det3's own
R&D values (c1 = 0.306, c2 = 0.12), v pinned to 36.6 µm/ns (the drift-scan
value its free fit already agreed with), **σ_p0 free**, fitting (τ_RC, kY,
σ_s, Dp):

    kY = 1.222, tau_RC = 127 ns, sigma_s = 96 ns, sigma_p0 = 0.242 mm,
    Dp = 0.0125                              [calib_bundle_lp_sp0free]

Full chain against the numbers the lost bundle produced:

| metric | lost delay-mode bundle | **rebuilt, lp** |
|---|---:|---:|
| σ_θ X / Y | 1.20 / 1.15° | **1.176 / 1.143°** |
| s68 X / Y | 1.25 / 1.18° | 1.246 / 1.183° |
| bias X / Y | −0.04 / −0.29° | −0.06 / −0.31° |
| implied-v spread X / Y | 2.3 / 2.4 | 2.13 / 2.61 µm/ns |
| within 5 mm | 93.47 % | **93.41 %** |
| core σ \|r\| | 0.470 mm | **0.461 mm** |
| median \|r\| | 0.763 mm | **0.737 mm** |
| reco-at-all | 97.29 % | 97.29 % |
| spark_frac | 8.22 % | 8.22 % |

**Reproduced, and marginally ahead on every metric** — which is the bar det4's
arm C had to clear before promotion. det3's angular resolution is a live,
reproducible number again rather than a citation to a dead bundle.

Two things worth carrying forward:

- **lp buys det3 almost nothing** (1.20 → 1.18°), against det4's
  2.63 → 2.53°. That is consistent, not contradictory: det4's gain came from
  fixing a kernel *shape* that its fit had been compensating for with a
  half-millimetre σ_p0. det3 was never in that regime, so there was nothing
  for the better kernel to recover. The lp port is a rescue for degenerate
  chambers, not a general upgrade.
- **σ_p0 = 0.242 mm** under lp against det3's delay-mode R&D 0.098 mm. Same
  caveat as det4's 0.404 mm: σ_p0 means a different thing under the two
  kernels and the two numbers are not comparable.

## 7. The audit — the campaign was NOT fully extracted

Prompted by "is the SPS test closed?", the EOS run directory was walked
run-by-run against `datasets.py` and RUN_TIMELINE §6's "what is analysed"
table, instead of trusting either. **Eight runs carrying det4 data were in
neither.**

> ### ⚠ The sub-run names are P2's, not ours — verified 2026-08-05 (late)
>
> Runs 64–70's sub-runs are called `cfg_gain3.0_peaktime200_opt`,
> `cfg_gain4.5_peaktime100`, `cfg_gain3.0_peaktime50` and so on. **Those gains
> and peaking times are P2's VMM settings.** banco named the runs after what
> *they* were scanning; det4 hangs off the **Dream** readout and was untouched
> by any of it. Checked three ways, not inferred:
>
> 1. **The Dream config is byte-identical across the "scan".**
>    `md5sum` of `P2B_Beam.cfg`: run_64 (`peaktime200_opt`), run_66
>    (`gain4.5_peaktime200_opt`) and run_70 (`peaktime50`) all hash to
>    `81c16e94…`. `diff run_64 run_70` returns **nothing**.
> 2. **Every Dream register is a single constant campaign-wide.** Over all 29
>    `P2B_Beam.cfg` copies staged from runs 55/57/61/62/63/64/66/68/70, each
>    `Feu * Dream * <n>` line has exactly one value — e.g. register 1 is
>    `0x081f 0xd023` everywhere. The Dream gain and peaking time never moved.
> 3. **Only three lines ever differ between any two Dream configs**, and they
>    are exactly what we controlled: `Sys NbOfSamples` (32 ↔ 64),
>    `Sys PedRun Threshold` (5.00 ↔ 4.00 σ), and `Sys DaqRun Time` (banco's
>    sub-run length, 720/1260/1800 s). Nothing else.
>
> `run_config.json` corroborates it: `triggered_by = vmm_daq@…` on every one of
> these runs, and the `sub_runs` entries carry only a name plus P2's HV
> channels — 8:8 and 12:2 (ours) are `null`, because our voltages were driven
> outside banco's DAQ entirely.
>
> **So what we ever varied on det4 across this whole campaign was: HV (drift
> and resist), sample count, and the ZS threshold / RAW-vs-ZS.** That is the
> complete list. An earlier draft of this section read the sub-run names as a
> det4 shaping scan and called it "the campaign's one lever on its own biggest
> caveat" — that was wrong, and the tail-vs-shaping confound stays exactly as
> unresolved as `RAW_RUN71_PHYSICS.md` leaves it.

| run | when | mount | det4 (FEU03) | what it is **for us** |
|---|---|---|---|---|
| **run_62** | Sun 22:00–23:11 | 25.64° | 4.2 GB, **64 smp** | a SECOND CF₄ drift ladder — see §8 |
| run_64 | Mon 01:58–02:28 | flat | 3.4 GB, 32 smp | flat, operating point, no det4 scan running |
| run_65 | Mon 02:29–02:59 | flat | 4.0 GB, 32 smp | ditto |
| run_66 | Mon 03:00–03:30 | flat | 3.9 GB, 32 smp | **our flat resist scan 780→400 V** starts 03:00:34 |
| run_67 | Mon 03:31–04:01 | flat | 3.2 GB, 32 smp | …the same resist scan continues, to 03:56:35 |
| run_68 | Mon 04:02–04:18 | flat | 0.02 GB, **64 smp** | our flat drift scan 700→100 V — **but NO BEAM**, see §11 |
| run_69 | Mon 04:19–04:50 | flat | 0.02 GB, 32 smp | …same scan, also beamless |
| run_70 | Mon 04:50–05:21 | flat | 2.7 GB, 32 smp | **our flat drift scan 600→100 V**, 05:02:47–05:19:17 |

Note the scans cross run boundaries: the resist scan spans runs 66+67 and the
drift scan spans 68+69, because banco restarted runs on their own schedule
underneath our scan scripts — the same pattern as Saturday's ladder spanning
run_57→run_58.

Why they were missed: RUN_TIMELINE's narrative **stops on Sunday**. Runs 64–70
are Monday 01:56–05:21, between the 00:40 rotation-to-flat access and run_71's
05:22 RAW run — a 3.5-hour block of det4 beam data with no section in any
document. run_62 does appear in the epoch table but never in the analysis
table, and it is named exactly once in the whole corpus
(`GAS_FLUSH_TIMELINE.md` §5, only to say the gas was exchanged by then).

Two documented claims are **wrong** as a result:

- *"No drift lever here — every drift scan ran before the access"*
  (`datasets.py: run63_flat`) and `FLAT_CF4_RUN63.md`'s "no drift lever in the
  flat data". **run_68 and run_70 are flat CF₄ drift scans taken after the
  access**, run_68's at 64 samples. The 2026-08-05 pass narrowed this claim to
  "true only of the CF₄ era" on the strength of run_55 (CO₂); it is not true of
  the CF₄ era either.
- The 2026-08-05 pass called run63_rot25 the CF₄ drift ladder. There are
  **two** (§8).

The det4 scan logs for all of these are now archived under
`records/scan_logs_late/run_{62,64..70}/` — five scripted, timestamped CSVs
(`det4_drift_scan_A_700_100`, `_B_650_50`, `det4_resist_scan_780_400`,
`det4_drift_scan_700_100_64smp`, `det4_drift_scan_600_100`) that no analysis
had ever read.

## 8. run_62 — the v(E) curve is now independently reproduced

`datasets.py: run62_rot25_ladder` (new), pulled FEU03-only
(`staging/pull_wave4.sh`), decoded, spanned. Conditions are **identical** to
run63_rot25 — 25.64°, Ar/CF₄/iso, 64 samples, ZS 4σ, resist 769.75 V, same
15H04 pedestal set — but the drift points are different (700/600/500/400 V
against 675/625/575/525/475/425/325 V), and it was taken 90 minutes earlier.
That makes it a genuine reproducibility check rather than a repeat.

| plateau | V/cm | hits | span [ns] | v [µm/ns] |
|---|---|---|---|---|
| d700 | 243 | 1,985k | 2033 | **14.38** |
| d600 | 209 | 1,911k | 2293 | **12.73** |
| d500 | 174 | 1,586k | 2515 | 11.59 |
| d400 | 139 | 1,125k | 2521 | 11.56 |

Against run_63's curve interpolated to run_62's fields, same estimator, same
c0 = 30 ns:

| V/cm | run_62 | run_63 (interp) | difference |
|---|---|---|---|
| 139 | 11.56 | 11.53 | **+0.2 %** |
| 174 | 11.59 | 11.67 | **−0.7 %** |
| 209 | 12.73 | 12.83 | **−0.8 %** |
| 243 | 14.38 | 14.39 | **−0.1 %** (extrapolated) |

**RMS 0.6 %, max 0.8 %.** Two ladders, two nights' different drift points, one
curve. The wet-CF₄ v(E) result — the campaign's most load-bearing gas number,
and the one that reset the tilt from 0.2–0.4° to 0.9° — is now reproduced on
independent data instead of resting on a single ladder. The bottom two points
(174, 139 V/cm) sit on the same ~2520 ns truncation floor the run_63 ladder
hits, confirming that floor is a window property and not a fit artefact.

`d300`/`d200` are defined in the dataset but return no hits: det4's data in
`driftscan_gap200V` **stops at 22:32:14**, 13 minutes before the sub-run's
logged end (22:45:23), and EOS holds only two datrun files for it. No physics
is lost — both points are at 104 and 69 V/cm, far below the truncation floor —
but the early stop is unexplained and is the one loose end this dataset adds.

## 9. The flat resist scan — kernel gain-invariance, widened and controlled

`datasets.py: run66_flat_resist` (new), `gain_scan_flat.py` (new). Our own
scripted scan (`det4_resist_scan_780_400.csv`), 780 → 405 V in 25 V steps,
205 s each, drift held 700.5 V, **flat** mount, fully-exchanged CF₄. Nine
plateaus (780 → 580 V) fall inside run_66's sub-run; the rest continue into
run_67, which is not staged because 555–405 V is far below det4's 769.8 V
operating point. Beam covers every plateau (FTARGET 02:13:14–04:00:45).

Before this, kernel gain-invariance rested on run_56's flat **590 → 625 V** —
a 6 % swing. This is a **1.34× span in resist voltage**, same geometry, one
sub-run, one pedestal set, one ZS threshold (so plateaus are directly
comparable, unlike the cross-run gain comparison `GAS_FLUSH_TIMELINE.md`
warns about).

### 11a. The raw numbers, and why they mislead

| resist | Y strips/ev | Y q_lead | Y share (raw) | X strips/ev | X q_lead | X share (raw) |
|---|---|---|---|---|---|---|
| 779.8 | 5.65 | 237.7 | 0.410 | 4.40 | 242.7 | 0.522 |
| 705.0 | 5.00 | 231.2 | 0.397 | 4.07 | 232.8 | 0.518 |
| 629.8 | 4.37 | 211.3 | 0.350 | 3.71 | 209.2 | 0.506 |
| 580.0 | 3.92 | 198.6 | **0.301** | 3.48 | 192.4 | **0.494** |
| **span** | **×1.44** | ×1.20 | **×1.36** | ×1.26 | ×1.26 | ×1.06 |

Read naively this says the Y-view sharing *falls 36 %* with gain — i.e. the
kernel is not a layer property at all. It does not say that, and the giveaway
is `q_lead`: a 200 V resist swing moving the mean amplitude by only 20 % is not
credible for a resistive amplification stage. **The amplitude estimator is
ZS-censored**, and the censoring is gain-dependent: a ±1 neighbour carries
roughly half the leading strip, so as gain drops the *neighbours* cross the 4σ
threshold before the leading strip does, and the ratio falls for a reason that
has nothing to do with the kernel. The real gain lever here is **strips/event
(×1.44) and the event yield (235k → 109k, ×2.15)**, not the amplitude.

### 11b. The control, and the result

Restricting to a fixed leading-strip amplitude window (q_lead 400–3000 ADC,
the kernel work's own q0 window) makes the plateaus compare like with like: at
the same q_lead the neighbour sits at the same absolute ADC and is censored
identically at every gain.

| resist | q_lead (matched) | **Y share (matched)** | q_lead (matched) | **X share (matched)** |
|---|---|---|---|---|
| 779.8 | 801.9 | **0.613** | 790.4 | **0.598** |
| 705.0 | 784.5 | **0.613** | 781.3 | **0.602** |
| 629.8 | 773.0 | **0.613** | 773.7 | **0.613** |
| 580.0 | 768.5 | **0.618** | 768.7 | **0.617** |
| **span over the whole scan** | | **×0.99** | | **×0.97** |

**The amplitude-matched sharing ratio is flat to 1 % (Y) and 3 % (X) across a
1.34× resist swing** — while the uncontrolled version moved 36 %. That is the
invariance premise measured directly, at normal incidence, over the widest
gain lever the campaign has, and it also *explains* the raw variation instead
of merely tolerating it.

Two caveats, both important:

- **The absolute ratio is not `c1`.** It is a hits-level (matched-filter peak,
  ZS-truncated, ±180 ns time-matched) proxy summed over both ±1 neighbours.
  Its value (~0.60) is not comparable with the waveform library's per-side
  `c1` (Y 0.51–0.56, X 0.240–0.249), and the matched selection equalises the
  two views' censoring, which is why X and Y read alike here while the model's
  `kY ≈ 2.1` says they are not. **Only the constancy is being claimed.**
- The scan is in resist voltage, i.e. gain. It says nothing about drift field,
  which `kernel_refit_clean.py` covers separately (invariant over 92–233 V/cm).

## 10. run_70 — the flat drift lever, and what it can and cannot say

`datasets.py: run70_flat_drift` (new). Our scripted flat CF₄ drift scan,
600 → 100 V at the operating resist (769.75 V), plus a 700 V dwell at each end.
Windows measured from the sub-run's own `hv_monitor.csv` (1,816 rows at 1 Hz),
which is cleaner than the scan CSV. Beam covers the whole scan (FTARGET
resumes 04:59:19); the *first* 700 V dwell starts 8.6 min before beam returns,
so `d700`'s window is trimmed to 04:59:35–05:01:00.

**No v(E) is obtainable** — 32 samples = 1.92 µs against a 2.0–2.5 µs ladder,
the same wall run_61 hits. What it gives is the charge / mesh-transparency
curve at *normal* incidence:

| drift [V] | 700b | 700 | 600 | 500 | 400 | 300 | 200 | 100 |
|---|---|---|---|---|---|---|---|---|
| V/cm | 243 | 243 | 208 | 174 | 139 | 104 | 69 | 35 |
| Y q_lead | 243.5 | 266.1 | 229.7 | 204.5 | 183.4 | 163.4 | 146.9 | 125.5 |
| Y strips/ev | 5.48 | 5.84 | 5.32 | 5.06 | 4.88 | 4.71 | 4.70 | 4.89 |
| X q_lead | 249.5 | 276.8 | 233.5 | 201.7 | 175.0 | 149.6 | 127.5 | 107.1 |
| X strips/ev | 4.39 | 4.63 | 4.26 | 4.01 | 3.80 | 3.59 | 3.31 | 2.97 |

Leading-strip amplitude falls **×2.1 (Y) / ×2.6 (X)** over a 7× drift-field
range, with strips/event falling ×1.2 / ×1.6 — the transparency/attachment
curve, and the flat counterpart of run_61's 15° version (which fell ×1.4 in
multiplicity over the same field range).

**The d700/d700b bracket is the internal control and it passes.** Two slices
of the same condition, taken 18 minutes apart at opposite ends of the scan:
matched sharing 0.621 vs 0.605 (Y, 2.6 %) and 0.606 vs 0.598 (X, 1.3 %). The
plateau windows and the estimator are stable within the run.

**On sharing, this dataset is weak evidence and should be quoted as such.** The
raw ratio censors completely (median → 0 below ~140 V/cm, because at low field
both neighbours drop under the 4σ threshold); the amplitude-matched ratio moves
0.621 → 0.526 (Y) and 0.606 → 0.537 (X), ~11–15 %. That residual is very
likely *still* censoring — the matched window fixes the *leading* strip's
amplitude, not the neighbour's, and `q(match)` itself drifts upward (799 → 855
ADC) as the low-field plateaus keep only their brightest events. Two reasons
not to read it as physics:

- the sign is **wrong for diffusion**. Lower drift field means more transverse
  diffusion, hence *more* sharing; the measured trend goes the other way.
- the waveform-level measurement on the same detector already settles this
  properly and disagrees: `kernel_refit_clean.py` on run_71 RAW gives
  τ ±3.7 %, c1 ±1.1 %, c2 ±2.8 % over 92–233 V/cm.

So: use run_70 for the transparency curve, and use run_71 for drift-invariance
of the kernel. The hits-level sharing proxy is only trustworthy where the
*gain* is scanned and the drift (hence the absolute signal) is held — which is
exactly run_66, §9.

## 11. What is still open

- **det2 / det6 / det7 `share_lp`.** det3 is done (§7); the other three
  chambers' bundles are lost with the campaign machine in the same way, and
  det6/det7 additionally carry the unresolved thin-template problem
  (`FLEET_2026-07-29.md`), so they need the template refit *before* a kernel
  is pinned, not after.
- **run_58's half of the CO₂ ladder** (500 → 10 V, 86 GB): still not pulled,
  and now with a reason beyond disk — its first point is at a recovering
  resist, and everything below ~190 V/cm is window-truncated in the same way
  the staged points already show.
- **σ_p0.** See §2. The next lever is not another fit: it is the noise model
  and the rot25 alignment.

### From the audit (§7) — genuinely unexploited data

Ranked by what they would settle, not by size. With the peaking-time
misreading removed (§7), the list is shorter and more modest than it first
looked — **none of these would change a published number.**

1. ~~**The flat resist scan, runs 66+67**~~ — **DONE, §9.** run_66's nine
   plateaus (780→580 V) were pulled, decoded and analysed. The amplitude-matched
   sharing ratio is flat to **1 % (Y) / 3 % (X)** across the scan. run_67
   (555→405 V) is still not staged and is not worth it: run_66's own bottom
   plateau already shows the yield halving and the raw ratio censoring away.
2. ~~**run_68+69's flat 64-sample drift scan (700→100 V)** — the only flat CF₄
   ladder that is not window-railed, in the geometry where σ_p0/Dp separate
   best.~~ **DEAD — there was no beam.** The scan ran 04:02:50–04:31:15 and
   SPS FTARGET extraction stopped at **04:00:45** and did not resume until
   **04:59:19**; every spill in that hour went to `SPS_DUMP`. That is why det4
   wrote only ~20 MB in each of runs 68 and 69. This looked like the most
   interesting item on the list and it is worth nothing. Do not pull it.
3. ~~**run_70's flat drift scan**~~ — **DONE, §10.** The transparency curve at
   normal incidence (q_lead ×2.1–2.6 over 243→35 V/cm); no v(E), as predicted.
4. **runs 64/65 (7.4 GB)** — flat, operating point, no scan running: pure extra
   statistics for the flat kernel measurement, which `FLAT_CF4_RUN63.md`
   already calls statistics-stable at 42.6k events. Lowest value here, and the
   only remaining unpulled det4 data on the whole campaign apart from run_67
   and run_58's ladder tail.
5. ~~**run_63 `beam_commissioning_00`**~~ — **CLOSED.** It is a **37-second
   aborted start**: `dream_daq.log` has it beginning 23:50:48 with a 10-minute
   budget and finishing 23:51:25, the run ending at 23:51:35 and being
   restarted at 23:52:57 as `operating_00`. 100 MB of FEU03. Correctly omitted
   from `datasets.py`; nothing to analyse.
6. ~~run_62's unexplained early stop~~ — **CLOSED: it was beam.** det4's data
   in `driftscan_gap200V` ends 22:32:14; SPS **FTARGET extraction stopped at
   22:33:09** and did not resume until 22:49:00. The DAQ kept the sub-run open
   to 22:45:23 and simply had no triggers to write — the same signature as
   run_59's thin `detE_long_01` and run_60's `overnight_15` collapse. Not a
   FEU or decoder fault: FEU01 stopped at the same point (both FEUs have 2
   datrun files for that sub-run). The H4 TAX stopper was *open* throughout
   (21:13:35→21:52:34 and 22:05:56→end of day), so this is the machine, not
   the zone. run_62's scan B (650→50 V, 22:50–23:36) ran entirely after the
   last sub-run with FEU03 in it and has no det4 data at all.

**The campaign's binding systematic is unchanged and unfixable in analysis.**
`c1`, `c2` and `tau_s` remain integrals over a tail longer than the 3.84 µs
window (`RAW_RUN71_PHYSICS.md`, "the window wall"). Nothing in runs 62–70
addresses it, because addressing it needed a longer window at the time and
there is no beam for three years.
