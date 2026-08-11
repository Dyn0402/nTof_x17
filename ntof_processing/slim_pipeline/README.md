# slim_pipeline — n_TOF hits, keyed to DREAM event IDs

Turns one **(DREAM sub-run × n_TOF run) segment** into an `ntof_hits/` directory
beside the DREAM sub-run on EOS. **First full campaign: 2026-08-09, 119 of 202
segments over 59 n_TOF runs — see
[`../SLIM_CAMPAIGN_2026-08-09.md`](../SLIM_CAMPAIGN_2026-08-09.md) for what ran,
what it found and what is still open.** Feasibility, window choice and sizes:
[`../SLIM_FEASIBILITY_2026-08-08.md`](../SLIM_FEASIBILITY_2026-08-08.md).
The time calibration authority is
[`../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`](../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md).

```bash
# reference pair, local v12 copy, scratch output
python run_segment.py run_79 stat090_0000 224572 \
    --ntof-source /media/dylan/data/x17/ntof_reproc/v12_liqpileup \
    --out /tmp/slim_test [--nb 60]

# production: reprocessed n_TOF on EOS, output beside the DREAM sub-run
python run_segment.py run_79 stat090_0000 224572

# does it reproduce what was measured on the full source?
python validate.py /tmp/slim_test/runs/run_79/stat090_0000/ntof_hits/ntof_hits_*.root
```

## What it does

```
[0] join    bunch_join: DREAM eventId -> BunchNumber, t_since_flash, is_flash
[1] pass 1  wall top/bottom offsets, then the N1081B SINGLES emulation
            -> candidate list                          reads WAL + PSS
[2] fit     COARSE SEARCH for T0, then K, T0, per-arm offsets, then
            per-bunch (da_b, dk_b)                               seconds
    (empty PS pulses are dropped at [0], before any of this)
[3] pass 2  every scintillator hit within +-1 us of the FULLY CORRECTED
            prediction, plus the same width at +100 us  reads all 12 trees
[4] write   one ROOT file + three JSON sidecars
```

**The clock is fully fitted before anything is cut.** No hit is ever discarded
on a provisional calibration. Nothing reads a stored constant: `K`, `T0` and the
per-arm offsets are per (DREAM run, n_TOF processing) pair and do not transfer.

**Step [2] begins with a coarse search, and that is not optional.** The iterated
fit only sees candidates within +-250 ns of where it is currently looking, so it
cannot walk to a peak further away than that. Until 2026-08-09 it started from a
hard-coded seed instead, which happened to be right for run_79 (T0 = -253 ns) and
wrong for run_77 (T0 = +110 ns): 7 of 9 run_77 segments died, and the 2 that
lived started from 312 candidates against a floor of 200 -- luck, not physics.
`clockfit.bootstrap` now histograms every candidate within +-50 us and takes the
peak, requiring S/N >= 6 over the accidental floor beside it. On the reference
pair it lands +-10 ns from the old seed at S/N ~1850, and every published
constant reproduces to 4 decimals.

Flash triggers are tagged in `events` and get no n_TOF hits.

## Output

`<eos>/july_beam/runs/<run>/<subrun>/ntof_hits/`

| file | |
|---|---|
| `ntof_hits_<run>_<subrun>_<ntofrun>.root` | trees `hits`, `events`, `bunches` |
| `calibration.json` | K, T0, arm offsets, tb offsets, thresholds, fit trace |
| `qa.json` | efficiency, accidental, purity, counts, runtime |
| `provenance.json` | which n_TOF files, which processing, det codes, branches |
| `clock_qa.json` | written by `clock_qa.py`: every check, the verdict, histograms |

`hits` — `eventId`, `det` (0-11, see `provenance.det_code`), `detn`, `tof`,
`dt_ns` (to the corrected prediction), `amp`, `amp_0`, `area_0`, `fwhm`,
`risetime`, `chi2`, `satuflag`, `pileup1`, `pulseshape`, `is_control`, and
(since 2026-08-09) `shadow_amp`/`shadow_dt` — the largest `amp_0` on the same
(bunch, channel) in the previous 1 µs and the ns since it, computed on the
**full** per-bunch stream before the window cut. The adopted plastic ringing
cut is `amp_0 < 0.05 × shadow_amp` (`../pss_ringing/`); storing the floats
instead of the boolean keeps the thresholds re-tunable without re-slimming.

`events` — one row per DREAM trigger of a bunch **that had beam**, flash and
unmatched ones included, so "no n_TOF partner" is distinguishable from "not
written": `eventId`, `bunch`, `t_dream_ns`, `is_flash`, `t_pred_ns`, `matched`,
`residual_ns`, `arm`, `da_ns`, `dk`, `corr_ns`, `corr_cv_ns`.

`bunches` — `bunch`, `n_triggers`, `fitted`, `da_ns`, `dk`, `n_core`, and
(since 2026-08-10) `has_beam`/`intensity_e10`. **This table spans every bunch
the sub-run touched, including the ones whose triggers were filtered out**, so
it is both the beam record and the record of what the slim dropped.

### Empty pulses are filtered, and the file says so

A PS pulse that delivered **no protons** (`intensity_e10 < 10`, and `tflash = 0`
independently) is dropped at the join, before anything is read or fitted. Its
bunch stays in `bunches` with `has_beam = 0` and its trigger count intact.

Measured over the whole first campaign
(`../FINDINGS_2026-08-10_unfitted_bunches.md`): 1,658 of 96,206 bunches, holding
0–19 DREAM triggers each against 46–139 in a beam bunch. DREAM's gate opens on
the PS timing whether or not protons arrive, so an empty pulse still produces a
burst of detector background — SiPM dark counts, WAL 2.80 hits per trigger
against PSS 0.017 and LIQ 0.000 — and `bunch_join` labels the first of those
background triggers `is_flash`, which leaves the whole burst referenced to
nothing. None of the campaign's 2,764 such triggers matched an n_TOF candidate.

The filter guards itself: a dropped pulse that holds a *full* DREAM burst is not
a beam statement but a mis-assigned join, and `clock_qa` fails the segment on
the ratio ('dropped pulses look like no beam'). Found in the wild on
`run_116/stat090_0013 × 224636`, a 13 %-overlap proposal whose join fitted a
−1,324 s burst-to-pulse offset.

**Two things follow for an analysis.** Use `intensity_e10` for the
dedicated/parasitic split — the two families sit at ~851e10 and ~413e10, and
they do not match equally well (97.7 % against 91.2 %, which is most of the
fleet's 93.6–97.3 % efficiency spread). And do not read `frac_fitted` as a
quality number on a file written before 2026-08-10: there it silently includes
empty pulses, i.e. it is beam availability.

### `is_control` is not optional

The measured 0.049 % accidental rate is the identical match with the DREAM time
shifted **+100 µs** — not a local sideband, because the singles rate varies far
too much across the 80 ms. Hits in that shifted window are written with
`is_control = 1`. Without them the file cannot measure its own background.
They cost ~12 % of the file.

## Running it on condor

`lxplus/` holds the batch machinery. One job = **one n_TOF run** = every DREAM
sub-run that overlaps it, so the 30 GB source is copied and indexed once.

```bash
./lxplus/stage.sh                    # push source to ~/x17slim (no data)
ssh -K lxplus                        # -K is mandatory: no token, no EOS
cd x17slim && myschedd bump
condor_submit slim.sub run=224572
./publish_to_eos.sh out              # from lxplus, after it finishes
```

Measured on the first real job (224572, 3 segments): **xrdcp 30 GB in 133 s,
28.0 min wall, peak RSS 3.16 GB**, one core. Budget ~10 min per DREAM sub-run
plus the copy.

Two things that cost a submit each, recorded so they do not again:

- **Condor does not accept a trailing `# comment` on a value line** — it parses
  it into the expression. `request_disk = 45 GB   # note` is a syntax error.
- **`transfer_input_files = pkg/` with a trailing slash** transfers the
  *contents* into the scratch root and flattens the package layout. Use `pkg`.
  The wrapper now preflights every input path and imports the package *before*
  the 30 GB copy, so this costs 5 s instead of two wasted xrdcps.

## Validated

`validate.py` on run_79/stat090_0000 x 224572, against numbers measured without
this pipeline. Every check passes, and it passes identically whether the segment
ran locally or on a condor worker (efficiency 95.8864 % both ways, to four
decimals):

| check | slim | published |
|---|---|---|
| K | 1.106350e-4 | 1.103724e-4 (both sub-runs) |
| T0 | -252.60 ns | -253.64 ns |
| arm offsets A/B/C/D | -17.06/+7.79/+1.86/-1.01 | -16.81/+7.55/+1.62/-0.83 |
| efficiency @ +-25 ns | 95.8864 % (cv 95.8493 %) | 95.84 % |
| accidental | 0.0457 % | 0.049 % |
| per-bunch da / dk RMS | 6.55 ns / 0.92 ppm | 6.5-6.8 ns / 0.92-0.96 ppm |
| **liquid same-arm diagonal** | **0.163/0.150/0.018/0.093** | **0.165/0.151/0.018/0.094** |

The liquid row is the one that matters: it is a physics result recomputed **from
the slim alone** and it exercises the window, the accidental control, the arm
assignment and the liquid leg at once.

### Segment independence

The condor job also produced `stat090_0001` and `stat090_0002`, neither of which
had ever been run. `stat090_0001` is a disjoint hour with its **own** published
diagonal, and it reproduces that too:

| | slim | published |
|---|---|---|
| efficiency | 95.8848 % | — |
| diagonal | 0.162 / 0.145 / 0.016 / 0.090 | 0.164 / 0.146 / 0.016 / 0.092 |
| background-subtracted | 0.135 / 0.117 / 0.011 / 0.070 | (0000 gives 0.136 / 0.119 / 0.013 / 0.075) |

and the clock is genuinely refitted per segment rather than reused —
`K` = 1.106350e−4 on 0000 against 1.101174e−4 on 0001, 0.47 % apart, each within
0.24 % of the published two-sub-run fit.

**Why the window is 150 ns** — and a correction. An earlier version of this
file said ±100 ns clipped the liquid diagonal and ±250 was needed. That was
wrong. The raw `sig` column does move (0.158 → 0.163 on arm A) but so does the
control (0.023 → 0.027), and the **background-subtracted signal is identical**:
0.135/0.119/0.012/0.075 at ±100 against 0.136/0.119/0.013/0.075 at ±250. What
±250 recovered was accidental floor inside `liq_coincidence`'s deliberately wide
±100 ns integration window, and that cancels in the subtraction.

Measured containment of the background-subtracted excess (sig − ctl), on the
±10 µs probe (`slim_study/pss_tail_probe.py`, 2026-08-09):

| PSS cumulative capture | ±25 | ±150 | ±250 | ±500 | ±1000 | ±2000 | ±5000 ns |
|---|---|---|---|---|---|---|---|
| | 13 % | 46 % | 57 % | 71 % | **80 %** | 86 % | 93 % |

**The window is ±1 µs, and the reason is the plastics.** Walls are the trigger
and are contained at ±25 ns; liquids are contained at ±150. The plastics are
not, and the tail is large:

| family | early (< −150 ns) | late (> +150 ns) | late/early | core (\|dt\| < 25) |
|---|---|---|---|---|
| WAL | −572 | 3,136 | — | 32,026 |
| PSS | 3,147 | **69,199** | **22×** | 17,490 |
| LIQ | 3,701 | 2,224 | **0.6×** | 879 |

±1000 captures 93 % of the PSS excess lying within ±2 µs at 2.24× the hits
(~72 MB/segment, ~14 GB for the campaign). Past ~2 µs each extra microsecond
adds 1–2.4 k counts against an early-side noise level of ~1 k. The choice is one
constant, `config.SLIM_NS`; re-slimming a sub-run costs ~10 minutes.

**The plastic late tail is EXPLAINED: the plastics ring** (`../pss_ringing/`,
2026-08-09). Every large plastic pulse is followed by a train of real
secondary pulses out to ~1 µs (~4.4 extra PSA hits per large pulse, against
0.007 on the walls), plus a 2 ns-wide cable echo at 81–82 ns. Established four
ways — event-mixed control, time reversal, the walls as a null, and the raw
traces. The tail does not affect the trigger match (candidates are wall-side,
and the walls do not ring) or the liquid physics. **A plastic hit yield is
quotable against the shadow cut** (`amp_0 < 0.05 × shadow_amp`): it removes
99.5 % of the 150–1000 ns excess for 10.4 % of the core, all small-amplitude.
Per trigger, the question "is the main plastic peak where it should be" must
use the **largest-amplitude** hit on the trigger's arm (92 % within ±25 ns),
never the earliest (31 % — it picks unrelated singles at 720 kHz).

Two earlier readings of this were wrong and are corrected above: that ±100 ns
clipped the liquid coincidence (it clipped the accidental floor inside
`liq_coincidence`'s ±100 ns integration window, which cancels in the
subtraction), and that the liquids gained from a wider window (their apparent
tail is symmetric, i.e. subtraction noise).

## Judging a segment: `clock_qa.py` and the dashboard

The clock fit is the load-bearing step: get it wrong and every hit in the file
is attached to the wrong trigger while the file looks perfectly healthy. Two
independent layers exist because they catch different things.

```bash
python clock_qa.py <ntof_hits_dir> [--json]       # 17 absolute checks, verdict
python dashboard/make_clock_dashboard.py <root>   # the fleet, as one HTML page
python tests/test_clock_qa.py                     # 24 injected defects, ~10 s
python segment_diagnose.py <run> <subrun> <ntof>  # why did THIS one find nothing?
python lxplus/campaign_status.py                  # coverage vs the proposal
```

Three checks added and one fixed on 2026-08-09, after the ringing measurement
and the first campaign's QA sweep:

* **per-arm residuals centred** — a wrong per-arm offset hides from every
  global check (the arms average to zero, the efficiency barely moves inside
  ±25 ns). What cannot hide is that arm's matched residuals centring off zero.
  This is the check that catches `run_78/stat090_lat051_c0_0005`, whose stored
  arm C offset was 12 ns off (see below).
* **plastic primary within accept** — per matched trigger, the largest plastic
  pulse on the trigger's own arm lands within ±25 ns (92 % reference).
* **PSS late tail is ringing** — the shadow flag explains ≥ 90 % of the
  150–1000 ns excess (100.6 % reference; > 100 % means explained to within
  subtraction noise). If it ever stops doing so, the late tail is no longer
  ringing and someone should look.
* **containment is now SIDED, against an empirical null** — the old |dt| edge
  test lit up 45 of 116 campaign segments, all on LIQ, but the fleet-total
  edge excess was early +14,491 against late +15,623: *symmetric*, which no
  truncated coincidence is. It was the +100 µs control mis-stating the local
  floor by a little — a pedestal, common mode between the two edges. The
  check now flags only the edge **asymmetry** (|late − early| against core
  density), and judges it against the **mid-band (0.3–0.7 W) per-decile
  asymmetry RMS** rather than pure Poisson: measured fleet-wide, the pedestal
  wobbles 2–3× wider than Poisson everywhere in the window, and against the
  Poisson null 18 segments still lit up with *random sign*. Against the
  mid-band null the fleet |z| tops out at 3.2, sign-random; the gate is
  z ≥ 4, far below what real truncation gives (the ±150 ns mistake was a
  coherent 22× one-sided tail).

**The per-arm offset estimator was also fixed** (`clockfit.fit_global`). It
used to keep the intercept of a free per-arm line fit — an extrapolation to
t = 0 from data starting at 0.1 ms, with a slope the model says should not
exist per arm. On 3-minute segments the slope noise displaced the stored
offsets by up to 12 ns (lat051, arm C at −10.3 ns against a fleet median of
+1.5, matched residuals unimodal at +12 ns to prove it). Sliced-reference
measurement: old estimator 2–5 ns RMS with 9–18 ns worst cases on
lat051-sized slices; refined median 1.3–1.8 ns RMS. Full-segment values move
≤ 1.1 ns, inside the validate tolerance.

The published dashboard for the July campaign is at
<https://dylan-neff.web.cern.ch/notes/ntof-dream-clock-qa.html>.

### When a segment finds no coincidence

`segment_diagnose.py` separates four causes the error message cannot, building
the candidate list once and reusing it:

| it finds | means |
|---|---|
| too few events | skip it; the wall-clock proposal did not pan out |
| a peak at a non-zero **bunch shift** | bunch assignment is off; recoverable |
| a peak at a **large lag** in the full-burst FFT scan | the fine ±50 µs search was too narrow |
| a lag that will not **sharpen** under refinement | a broad association, NOT a coincidence |

The last is what the July campaign's 54 uncovered sub-runs turned out to be:
34.7 σ at 2 µs bins, 5.0 σ at 500 ns, i.e. ~6 µs wide against ~6 ns for a real
coincidence. Knowing it is broad rather than absent is the difference between
"these hours have no data" and "these hours were triggered on something else".

* **absolute** (`clock_qa.TH`) — thresholds from measurement, each with its
  reason in the source. PASS / WARN / FAIL, and **NA** when a file predates the
  field rather than silently passing.
* **population** (`make_clock_dashboard.population`) — robust z against the
  fleet median. This is the layer that matters: in the synthetic fleet test a
  segment whose T0 sits 380 ns from its neighbours passes *every* absolute check
  and is caught only here. That is exactly the 2026-08-09 bug class. T0 is
  compared only within one DREAM run, since it is not comparable across.

`tests/test_clock_qa.py` injects one defect per case and asserts the right check
fires at the right level. It has already paid for itself twice: it caught
`uproot.arrays()` returning a dict for some files and a structured array for
others, and it exposed that the inherited window check was near-useless — it
only detected a distribution still *rising* at the edge, and returned "flat,
window is wide enough" on the reference while 23 % of the plastic was being cut.

## Notes

- `area` is deliberately absent: with `AMPLITUDE OPTION=2` it is
  `amp × integral(shape)` by construction. `amp_0`/`area_0` are the measured pair.
- Saturation is **not** cut at slim time — the flags are kept, cut downstream
  with `ntof_io.saturated()` semantics.
- The wall top/bottom offsets are re-measured per segment. They are per
  *processing*, not per cabling (±32–39 ns on the official file, within ±5.5 ns
  on v12); a stored table pairs the bar ends around an offset that is no longer
  there and loses most genuine pairs.
- `ntof_io`'s caches are keyed by run number only, so `_bind_ntof` gives each
  source its own `variant_cache` fingerprinted on the file set.
