> # ⛔ RETIRED — do not build on this
>
> **Superseded by `../STATUS.md`.** Archived 2026-07-30.
>
> The work order that started the UserInput iteration. It was executed: the sweep ran, `v12_liqpileup` was chosen and shipped. The proposed fixes marked [inferred] here were decided by the iteration and several were rejected.
>
> **Read `../STATUS.md` for the state, `../userinputs/README.md` for the variant table, and `../FINDINGS_2026-07-28_psa_optimization.md` for what was measured.**

---

# Handoff: fix the n_TOF official processing ourselves (UserInput iteration)

**Written 2026-07-28 morning, at the end of the overnight session that solved
the "missing plastic" mystery. Audience: someone picking this up cold for a
full day's work. We are now the n_TOF processing people.**

Everything marked **[verified]** was executed and observed in the overnight
session (2026-07-27→28) and is reproducible from the quoted scripts. Everything
marked **[inferred]** is a reading of the PSA guide or the data that has not
been confirmed by running the processing — several of the proposed UserInput
fixes are inferred; keep the distinction and let the iteration loop decide.

---

## 0. The one-paragraph mission

The official n_TOF hit files for the X17 EAR2 campaign are produced by a Pulse
Shape Analysis (PSA) whose per-detector configuration lives in a single
`UserInput.h` text file. The overnight session proved the current
configuration mis-identifies the γ-flash on the plastic (PSS) trees in
**37–85 % of bunches** and times the flash on **inconsistent waveform features
across detectors (~350 ns spread)** — see §2. We have laptop-side repairs
(`ntof_dream_merge/tflash_repair.py`) that make the *existing* files usable,
but the right fix is upstream: **write corrected UserInput file(s), reprocess a
reference run on lxplus/condor with `RunProcessing.sh`, grade the output with
`ntof_processing/validate_reprocessing.py`, and iterate until it passes — for
all detectors (SiPM walls WAL, plastics PSS, liquids LIQ; PKUP is already
clean).** Then reprocess the runs the DREAM campaign needs. Raw data ages off
EOS after ~2 weeks (run 224572 was taken 2026-07-26), so start now.

---

## 1. Where everything is

| | |
|---|---|
| Processing docs pulled by Dylan | `/media/dylan/data/x17/ntof_processing/` |
| — current official UserInput | `UserInput_2026_EAR2_X17.h` (from Riccardo, 2026-07-17; see `riccardo_email.txt`) |
| — pulse-shape files it references | `X17_WAL{A,B,C}_Signal_*.txt`, `X17_LIQ{A,B}_Signal_*.txt` (must sit next to the UserInput, **full paths** inside it when processing) |
| — PSA manual | `PSA_Guide_20240704.pdf` (Žugec; READ IT — parameter semantics below cite it) |
| — lxplus/condor how-to | `Lxplus _ NTOF _ TWiki.html` (§3 is the processing section, §3.1 staging) |
| Overnight findings | `ntof_dream_merge/FINDINGS_2026-07-28_pss_tflash.md` (+ the 27th's `HANDOFF_2026-07-27_dream_ntof_matching.md` for the matching context) |
| Validation tool (acceptance test) | `ntof_processing/validate_reprocessing.py` |
| Laptop-side repair being replaced | `ntof_dream_merge/tflash_repair.py` |
| Raw waveform reader | `~/PycharmProjects/nTof_x17_DAQ/stream1_monitor/ntof_raw.py` (format doc: `docs/NTOF_RAW_FORMAT.md` in that repo) |
| Raw-waveform evidence extracts | `~/x17/beam_july/analysis/ntof_dream_merge/raw_evidence/` (+ figures in `../figures/`) |
| Reference official file | `~/x17/beam_july/ntof_data/run224572.root` (26 GB, byte-exact vs EOS) |
| Raw data on EOS | `root://eospublic.cern.ch//eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/<run>/stream1/run<run>_<seq>_s1.raw.finished` (~2.8 GB / ~70 s chunk, 152 files for 224572; `stream0` is tiny/aux) |
| Kerberos | `dneff@CERN.CH`; a ticket was valid through 2026-07-28 ~10:34 — `kinit dneff@CERN.CH` to renew. lxplus login `dneff`, computer group `za` (required, TWiki §1) |

Campaign identifiers for the processing scripts: **year 2026, area EAR2,
campaign `X17_measurement`** (this is the DAQ "Exp. / Archive Folder" — it is
the directory name on EOS), reference run **224572**.

---

## 2. What is broken, precisely  [verified]

Full evidence chain in `FINDINGS_2026-07-28_pss_tflash.md`. Summary with the
numbers you will re-measure after each iteration:

### 2a. PSS γ-flash mis-identification (per bunch)

Stored `tflash` deviates >150 ns from the tree's per-run mode in:

| WALA | WALB | WALC | WALD | PSSA | PSSB | PSSC | PSSD | LIQA-D | PKUP |
|---|---|---|---|---|---|---|---|---|---|
| 1.7 % | 1.1 % | 0.3 % | 0.0 % | **84.5 %** | **65.4 %** | **36.8 %** | **80.6 %** | 0.0 % | 0.0 % |

Failures are µs-scale (stored values like 314 ns against a true flash at
~11 645 ns — the finder tags junk near the window start) and hit essentially
**every parasitic (half-intensity) pulse** plus an arm-dependent fraction of
dedicated ones. The raw waveforms show the plastic flash **rails the ADC high
in every pulse, parasitic included** — nothing subtle, purely a config problem.
Because every hit's physics time is `t_since_flash = tof − tflash`, each failed
(tree, bunch) has ALL its hits time-shifted by the tflash error. This is what
masqueraded as "n_TOF only records a plastic for 52 % of DREAM triggers".

### 2b. Cross-detector flash-feature inconsistency (~350 ns, constant)

Per-run modal tflash: WALA/C/D = 11 245–11 275 ns, but **WALB / PSSA-D /
LIQA-D = 11 615–11 645 ns** (PKUP 13 335, its own cable). The coincidence peak
of large (amp>1000) plastic hits against same-arm wall hits sits at
**−375 / +25 / −325 / −325 ns (A/B/C/D)** after mode removal; LIQ shows the
same per-arm offsets (−373/+10/−350/−348), proving the inconsistency belongs to
the **walls'** flash timing (WALA/C/D time a different feature of the railed,
undershooting flash than everything else). All four walls share one UserInput
row, so this is a waveform-shape interaction, not a config difference between
arms.  **[verified]**  Why WALB differs from WALA/C/D in hardware is unknown —
compare raw wall flash waveforms per arm if you want the mechanism (§6, q2).

### 2c. Rebound/fragment artifact hits

Each real plastic pulse is accompanied by a small (~130–240 ADC, i.e. just
above the 100 ADC amplitude threshold) secondary hit ~350 ns later — the
pulse's rebound/overshoot re-triggering the finder. The wall "+330 ns satellite
band" seen in the DREAM matching (31 % of events had ONLY the delayed wall hit)
is plausibly the same phenomenon on the walls **[inferred]** — if a UserInput
change kills it, the matcher's accept window collapses to a single clean band.

### 2d. What is NOT broken (don't "fix" it)

- PSS/WAL **amplitudes and areas** are fine (an interim conclusion that they
  were broken was wrong — the true partners were simply outside the search
  window). Raw pulses are fast: 4–7 ns rise, 11–16 ns FWHM, up to ~13k counts.
- Plastic PSA dead time (resolves 5–6 ns pairs), wall↔plastic tree mapping,
  channel interleaving, `BunchNumber` entry-order sortedness (asserted by our
  reader; note TWiki mentions `sort_runs` for multi-rack detectors if a future
  file ever fails that assert).
- PKUP: 0 % flash failures — it is the natural absolute-time anchor.

### 2e. Also worth improving while you're in there

- Plastic amplitude threshold is 100 ADC vs the walls' 50 — truncates the
  small-hit spectrum in its bulk (fragments pile right above it). After the
  flash fix, consider whether it should change at all — the PSA guide
  recommends loose elimination ("eliminate in later data analysis").
- The LIQ row was tuned quickly ("liquids are still kinda bad" — Riccardo).
  LIQ flash-finding is *fine*; it's pulse recognition/pileup that was hard.

---

## 3. Reading the current UserInput (the knobs that matter)

`UserInput_2026_EAR2_X17.h`, columns per the PSA guide (parameter names in
guide's "UserInput file" slide; semantics in the pages named below). Current
rows, abridged to the flash- and recognition-relevant fields:

```
tree  STEP   TIME    GFLASH  GFLASH  GFLASH     GFLASH  BASELINE  AMP     AMP    SIGWIDTH  #SHAPES
      SIZE   LIMIT   OPTION  THRESH  MIN_WIDTH  WINDOW  OPT/FILT  OPT    THRESH  LO/HI
WAL*  8/7    40000   0       500.    0.         0       4/150,800  2      50     5/100,4000  3 (WALA_3, WALC_0, WALB_0)
PSS*  3/4    25000   0       50.     0.         0       1,200      1     100     10,3000     0
LIQ*  2/4    25000   0       500.    100.       1000    1,100      2      50     1,5000      2 (LIQA_7, LIQB_0)
PKUP  300/6  100000  0       100     1          0       -1,300     0     300     1,4000      0
```

Semantics you need (guide pages):

- **G-FLASH OPTION** (Locating γ-flash): 0 = first pulse crossing G-FLASH
  THRESHOLD (in channels, relative to baseline); 1 = first pulse going into
  saturation (optional `MIN_WIDTH = min_width/min_saturation`); 2 = oscillatory
  treatment. Optional CF fraction: `flash_option/constant_fraction` (default
  0.3). **Latest updates** adds a *lower time limit* for the flash search:
  `G-FLASH THRESHOLD = flash_threshold/time_limit` (e.g. `1e3/1e4`) — made
  precisely for "competing pulses before the γ-flash", which is our exact
  failure mode.
- **G-FLASH MIN_WIDTH**: minimal expected width of the flash pulse, to reject
  false flashes. **G-FLASH WINDOW**: protection window after the flash where
  pulses are never eliminated.
- **Polarity** is inferred from the ZS-threshold sign (all our trees read back
  `polarity=1` = regular); pulses are analyzed as negative. The PSS flash
  *rails high* (positive) then undershoots — so what the PSS flash-finder can
  see as a "pulse" is the **undershoot** (~23k deep, ~1–2 µs wide, at
  ~11.9–14 µs) **[inferred]** — consistent with the sane-bunch PSS mode of
  11 645 ns.
- **STEP SIZE** `ns/rms_multiplier`: derivative window for pulse recognition.
- **BASELINE OPTION/FILTER, TIME LIMIT**: baseline machinery (guide, several
  pages). WAL uses 4/150 (moving-maximum), PSS 1 (weighted moving average).
- **AMPLITUDE OPTION**: 0 max-point, 1 parabolic top fit, 2 pulse-shape fit
  (needs shapes; walls & LIQ use 2). γ-flash shape subtraction is possible via
  `NUMBER OF PULSE SHAPES = regular/flash` with a special flash-shape file
  (guide: "Pulse Shape fitting (γ-flash)") — the heavy-duty tool if simple
  options can't fix the wall flash timing.

### First-iteration proposal  [inferred — the loop decides]

1. **PSS flash fix** (the big one). Change `G-FLASH THRESHOLD` from `50.` to
   `2000/1e4` (threshold 2000 channels, don't look before 10 µs — the flash
   sits at 11.6–11.9 µs by hardware, well after 10 µs in every bunch) and set
   `G-FLASH MIN_WIDTH` ~ `300` (the undershoot is ~µs wide; real pulses are
   ~10–100 ns above baseline). Rationale: with threshold 50 ch (~1.5 mV) and no
   width/time guard, the first dark/junk pulse in 0–11.6 µs wins. LIQ proves
   the machinery works when configured (500 + MIN_WIDTH 100 + WINDOW 1000 →
   0 % failures); PSS at `50./0/0` is simply unprotected.
2. **Wall flash-feature consistency.** First try the same lower-time-limit
   guard on WAL (fixes WALA's 1.7 % of +374 ns outliers). For the ~350 ns
   WALA/C/D-vs-WALB feature question, experiment with the flash CF fraction
   (`0/0.5`…) and MIN_WIDTH; if nothing makes the four walls and the plastics
   time-consistent, the fallback is γ-flash **pulse-shape subtraction**
   (`#SHAPES = 3/1` + a flash-shape file per the guide) — or accept the
   constants and keep the offline offset calibration, which
   `validate_reprocessing.py` measures anyway. Consistency to <25 ns is the
   acceptance bar; *how* is free.
3. **PSS rebound fragments**: try `EXPAND PULSES` −1/−2 (bipolar-undershoot
   handling, guide "Pulse expansion (special case)") or a min `SIGNAL WIDTH
   LOW THR.` so the ~350 ns-late rebound is absorbed into the parent pulse
   rather than reported. Check the wall satellite band afterwards (§6 q3).
4. **Leave alone**: everything in §2d. Make ONE change-set at a time; the
   guide's own optimization chapter ("Optimizing the UserInput parameters",
   "Little practical advices", "Think creatively!") is short and worth reading
   before iterating.

Batch idea (user's suggestion, endorsed): since condor runs are slow-ish and
independent, prepare a small **batch of UserInput variants** (e.g. PSS
threshold {500, 2000, 8000} × time-limit {none, 1e4}) and submit them as
parallel processings of the same run into separate output dirs, then grade all
with the validator and keep the winner. Each variant = one `RunProcessing.sh`
invocation with its own `-p` and `-o`.

---

## 4. How to run the processing (TWiki §3, condensed)

```bash
ssh dneff@lxplus.cern.ch          # needs group 'za'
# optional env: source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_105 x86_64-el9-gcc13-opt
cd /afs/cern.ch/work/d/dneff/...  # MUST run from an /afs path (aux files per run appear in cwd)

/eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh \
    -y 2026 -a EAR2 -c X17_measurement \
    -r 224572 \
    -p /afs/.../userinputs/v1_pssflash/UserInput.h \
    -o /eos/user/d/dneff/x17/reproc/v1_pssflash
```

- `-l` for a run range; omitted = just `-r`. `-s 1` skips already-done runs.
- **Pulse-shape files must be referenced by FULL PATH inside the UserInput**
  when processing (Riccardo's email + TWiki both stress this). Copy the
  `X17_*_Signal_*.txt` files up with the UserInput and rewrite the last column.
- The script finds data on EOS or stages from CTA (tape) automatically and
  submits condor jobs; output lands in `<out>/completed/` (partials) and
  `<out>/done/run224572.root` (final). Aux folders appear in your cwd — clean
  them yourself.
- Raw data: EOS copy lives ~2 weeks post-acquisition
  (`/eos/experiment/ntof/DAQ`), afterwards CTA
  (`xrdfs root://eosctapublicdisk.cern.ch/ ls /eos/ctapublicdisk/archive/ntof/`,
  `query prepare` / `prepare -s -f` to stage; or `StageRuns.sh` with the same
  -y/-a/-c/-r flags). **224572 was taken 2026-07-26 → on EOS until ~08-09.**
- Scale sanity: 224572 is 3 018 bunches / 152 raw files / 26 GB output. For
  iteration speed, ask whether `RunProcessing.sh` can process a subset (it has
  no obvious flag for it) — if not, iterate on the full run (the condor fan-out
  is per-file, wall-clock should be tolerable) or look in
  `/eos/experiment/ntof/bin` + `/eos/experiment/ntof/repositories` for the
  standalone/GUI PSA the guide implies exists, which would let you iterate on
  a single downloaded raw chunk locally — **that is the fastest possible loop
  if it exists; spend 20 minutes looking for it before the first condor
  round-trip.** Riccardo (author of the current UserInput) is the person to
  ask.

Local raw chunks for offline eyeballing (no lxplus needed):

```bash
xrdfs root://eospublic.cern.ch cat \
  /eos/.../224572/stream1/run224572_8_s1.raw.finished | head -c 900000000 > head.bin
python nTof_x17_DAQ/stream1_monitor/ntof_raw.py head.bin   # summary
# EVEH words[1] == BunchNumber [verified]; PSS flash rails HIGH, walls LOW
```

---

## 5. The validation loop (our unique asset — use it every iteration)

```
new UserInput → RunProcessing.sh → done/run224572.root
      → xrdcp to laptop (or run the validator on lxplus with the repo)
      → .venv/bin/python ntof_processing/validate_reprocessing.py 224572 /path/to/candidate.root
      → PASS?  no → adjust knobs, repeat
               yes → full DREAM-side regression (below), then reprocess the campaign
```

`validate_reprocessing.py <run> <path>` sandboxes its caches (it monkey-patches
the reader at the top — the official file's caches are untouched) and grades:

1. **Flash identification**: per-tree bad-bunch fraction. Target <2 % (LIQ and
   PKUP prove ~0 % is achievable).
2. **Flash consistency**: PSS/LIQ-vs-wall coincidence-peak offsets. Target
   |peak| <25 ns per arm.
3. **Prompt-coincidence capture + per-tree hit counts** (no hard target —
   compare iterations; the hardware sector/plastic scaler ratio is 15–21 %, and
   watch the hit counts so a "fix" doesn't silently eat real hits).

Final regression once the validator passes, using the DREAM reference pair
(run_79 stat090_0000 ↔ 224572):

```bash
# point the analysis at the candidate file in a SEPARATE tree, e.g.:
mkdir -p /tmp/reproc_tree/ntof_data && cp candidate.root /tmp/reproc_tree/ntof_data/run224572.root
X17_BEAM_JULY=/tmp/reproc_tree ...   # plus copy/symlink runs/ and analysis/ from ~/x17/beam_july
.venv/bin/python ntof_dream_merge/match_window.py run_79 stat090_0000 224572 100
.venv/bin/python ntof_dream_merge/eval_singles_matcher.py 100
```

CRITICAL CACHE GOTCHA: `bunch_edges`, `tflash_table_*`, `tflash_offsets_*` are
cached under `~/x17/beam_july/analysis/ntof_dream_merge/cache/` **keyed by run
number only**. A reprocessed run224572.root read through the normal paths will
happily reuse stale caches from the official file. Either use the validator's
sandbox, or a separate `$X17_BEAM_JULY` tree, or delete
`cache/{bunchidx,tflash_table,tflash_offsets}_224572*` first.

Acceptance numbers to beat (current state, official file + laptop repair
**[verified]**): match_window 99.9 % (stat090_0000); thresholded wall∧plastic
SINGLES matcher 93.7 % efficient overall / 0.5 % false (89.9 % / 1.3 % at
1–3 ms). A *correctly processed* file should reproduce these **with the repair
disabled** (`repair_tflash=False` in `ntof_io.read_bunches`, offsets ~0) — that
is the definition of done. Then the laptop repair becomes a no-op (modes
consistent, offsets <25 ns, jitter tiny) and can stay enabled harmlessly.

---

## 6. Open questions worth an hour each (not blockers)

1. **Why parasitic pulses fail ~100 %** on PSS flash-finding while dedicated
   fail 35–85 %: unknown. With the time-limit guard it stops mattering; but if
   the fix underperforms, look at what sits in 0–11.6 µs of PSS block0 on
   parasitic vs dedicated pulses (raw extracts + `raw_extract.py` pattern in
   `~/x17/.../raw_evidence/`, script preserved at
   `.../analysis/ntof_dream_merge/raw_evidence/` — re-stream a head chunk as in §4).
2. **Why WALB times the flash ~350 ns later than WALA/C/D** with identical
   config: compare the four walls' raw flash waveforms (one head chunk gives
   all channels). Whatever the shape difference is, it may also explain WALB's
   1.1 % of −300 ns outlier bunches.
3. **Is the wall +330 ns satellite band a rebound artifact?** After any wall
   UserInput change, re-run `match_window.py` and look at the band ratios; if
   the satellite dies, simplify `BANDS` to the single ±150 ns core and re-run
   `eval_singles_matcher` (expect purity to improve further).
4. **Absolute ToF anchor for Phase 5**: PKUP is clean; decide and document
   whether E_n uses PKUP tflash + per-tree offsets or the fixed scintillator
   tflash. (19.5 m EAR2 path; the July QA E_n bands are the cross-check.)
5. **Which runs to reprocess** once the UserInput passes: at minimum every
   n_TOF run paired with a DREAM run (224572 is the reference; the DREAM runs
   and their pairs are in `ntof_july_analysis/pulse_match.py` machinery /
   `~/x17/beam_july/runs/*/`). Budget EOS-vs-CTA staging time for the older
   mid-July runs (224404–224524 era — those are past the 2-week EOS window).

## 7. State of the repo (as of this handoff)

- `203051e` — tflash repair v1 + FINDINGS doc + figures (the overnight solve).
- Uncommitted at handoff-writing time (commit lands right after this file):
  repair **v2** (coincidence-calibrated per-tree offsets in `tflash_repair.py`,
  cached `tflash_offsets_<run>.npz`), corrected FINDINGS §5/§5b (the amplitude
  claim was wrong — timing, not amplitude), `eval_singles_matcher.py` promoted
  into `ntof_dream_merge/`, this directory.
- Nothing pushed to origin (repo is ~30+ commits ahead; that predates tonight).
- Figures: `~/x17/beam_july/analysis/ntof_dream_merge/figures/tflash_*.png`,
  `plastic_partner_before_after.png` — ready to paste into any report to the
  n_TOF collaboration.
- Memory files: `ntof-pss-tflash-bug.md` (needs its amp-claim correction if you
  touch it — the FINDINGS doc is the source of truth).
