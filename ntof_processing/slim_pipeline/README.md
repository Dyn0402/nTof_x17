# slim_pipeline — n_TOF hits, keyed to DREAM event IDs

Turns one **(DREAM sub-run × n_TOF run) segment** into an `ntof_hits/` directory
beside the DREAM sub-run on EOS. Feasibility, window choice and sizes:
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
`risetime`, `chi2`, `satuflag`, `pileup1`, `pulseshape`, `is_control`.

`events` — one row per DREAM trigger including flash and unmatched ones, so
"no n_TOF partner" is distinguishable from "not written": `eventId`, `bunch`,
`t_dream_ns`, `is_flash`, `t_pred_ns`, `matched`, `residual_ns`, `arm`,
`da_ns`, `dk`, `corr_ns`, `corr_cv_ns`.

`bunches` — `bunch`, `n_triggers`, `fitted`.

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

**Known, unexplained: the plastic late tail.** One-sided (22× late over early),
smooth and monotonic with **no discrete echoes**, so it is not ringing at a
fixed period — more likely afterpulsing, late light, or the 101 ns PSS template
fitter splitting a long pulse. It is contained by the ±1 µs window, so it can be
diagnosed from the slims without re-reading the 21 TB source. **Do not quote a
plastic hit yield until it is understood.** It does not affect the trigger match
or the liquid physics.

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
python clock_qa.py <ntof_hits_dir> [--json]       # 13 absolute checks, verdict
python dashboard/make_clock_dashboard.py <root>   # the fleet, as one HTML page
python tests/test_clock_qa.py                     # 19 injected defects, ~5 s
```

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
