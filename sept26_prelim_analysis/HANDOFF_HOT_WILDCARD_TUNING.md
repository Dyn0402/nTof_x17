# HANDOFF — the hot-channel wildcard: tuned locally, and the answer is that
# the wildcard belongs downstream, not in the seeder

**Written 2026-09-08 after the condor regression; resolved the same day by the
local, event-by-event pass this file originally asked for. Local work on
run_145 only — no condor, no EOS, nothing here needs either.**

Companions: [`HANDOFF_D_NOISY_CHANNELS.md`](HANDOFF_D_NOISY_CHANNELS.md) (the
original spec), [`STATUS.md`](STATUS.md), `hot_seed_strata.py`,
`compare_hotmasked_rerun.py`, `k_robustness.py`.

---

## 0 · Resolved — read this first, then §3–4 for the evidence

**The re-run was not a regression, and neither of the two constants this file
blamed was the problem.** Both §4.1 and §4.2 below were written as guesses and
both have now been measured against real windows; the guesses were wrong in
opposite directions, and the reason the condor numbers looked catastrophic is
something neither anticipated.

**1 · The frozen D sample is two populations, and the mask separates them.**
29 % of D's triggers have a largest x cluster made **entirely** of flagged
channels — channels running ~35× their neighbours' occupancy, on connector
boundaries. Those noise columns **fit better than real tracks do**:

| stratum of the baseline x cluster | n | median χ²/dof | quality_ok | fitted p0 |
|---|---:|---:|---:|---|
| **all-hot** (zero clean strips) | 1 948 | **1.27** | **99.6 %** | 65 mm, IQR 142 |
| mostly-hot (<50 % clean) | 1 853 | 20.2 | 93.4 % | 137 mm, IQR 308 |
| clean (≥50 % clean) | 2 339 | 22.5 | 94.6 % | 257 mm, IQR 228 |

A smooth, wide, dilute coherent-noise deposit is *easy* for the forward model;
a real track has to be fitted against real diffusion and sharing. And the
all-hot fits pile up at p0 ≈ 65 mm — the u ∈ [50, 70) mm hot band of
`HANDOFF_D_NOISY_CHANNELS.md` §2.2 — while real tracks spread across the plane.

So the frozen table's 72.6 % convergence and 10.7 median χ²/dof were being
**held up by the noise**. Removing it must make both look worse. §3's "0 gained
a fit, 16 152 lost one" is that and nothing more.

> ⚠ **`HANDOFF_D_NOISY_CHANNELS.md` §3.3's criterion "the number of gated
> tracks in D goes UP" cannot be met by any correct mask, and should be struck.**
> It was written on the assumption that the flagged channels cost D tracks. They
> do not — they *manufacture* triggers, 29 % of them. The other four criteria in
> that section stand, and are what is used below.

**2 · `HOT_NOISE_INFLATION` does not matter at all.** §4.1's scan, run over one
fixed set of 3 600 real triggers: 10, 30 and 100 agree **to three decimals on
every metric in every stratum**. Paired per event, a window whose *seed* the
mask leaves alone fits identically with the mask on (χ²/dof 39.45 → 38.54,
median |Δp0| **0.00 mm**). Everything the wildcard does, it does through
seeding. The constant stays at 10; `WFT_HOT_NOISE_INFLATION` re-runs the scan.

**3 · §4.2 correctly identified seeding as the driver, and its fix does not
help either.** The admission rule is what drops the events (48 % of D's
x-planes lose every candidate), so §4.2 was right about the mechanism. But
neither its one-line fix nor the better rule now in `wft/seed.py` improves the
fits that survive. Paired per event, **when the mask changes which cluster is
seeded, the fit gets worse** — clean stratum χ²/dof 13.1 → 20.6, p0 moving a
median 4.8 mm. Seeding on the mask buys nothing measurable today.

**4 · What does work: use the classification downstream, on the frozen
products, with no re-reconstruction.** `hot_seed_strata.py` labels every
trigger by the hot content of its raw cluster in 4 minutes over all four arms,
and `k_robustness`'s new `no_hotstrip` variant cuts on it:

| arm | variant | removes | k shift | estimator spread | reproducibility |
|---|---|---:|---:|---|---|
| **D** | `no_hot` (existing, 2D post-fit) | 22.4 % | 3.76 % | 0.103 → 0.056 | 0.059 → 0.029 |
| **D** | **`no_hotstrip` (new)** | **3.6 %** | **0.55 %** | 0.103 → **0.087** | 0.059 → **0.029** |
| A | `no_hot` | 1.2 % | 0.02 % | 0.089 → **0.103** | 0.014 → **0.043** |
| A | `no_hotstrip` | **0.0 %** | 0.00 % | unchanged | unchanged |
| C | `no_hot` | 4.1 % | 0.94 % | 0.128 → **0.136** | 0.048 → 0.039 |
| C | `no_hotstrip` | **0.0 %** | 0.00 % | unchanged | unchanged |

It takes **all** of `no_hot`'s reproducibility gain and most of its spread gain
**for a sixth of the sample**, and it shifts k by 0.55 % instead of 3.76 % — so
it is not buying its improvement by moving the calibration. And it is the only
one of the two that meets §3.3's "A and C do not move": B and C have literally
zero all-hot triggers in either plane, so the cut cannot touch them, whereas
`no_hot` moves C by ~1 % and makes A's spread and reproducibility *worse*.

**That is the "something that works", and it is deliberately the simple
version.** It needs no condor cycle, leaves every frozen product valid, and is
one join on `(subrun, event_id)`. Seeding on the wildcard is the October
question — §6.

---

## 1 · What is built (unchanged from the original handoff, plus this session)

| piece | file | what |
|---|---|---|
| per-strip dead/hot/noisy classifier | `noisy_channels.py` | unchanged; **validated this session** — D's 42 hot x channels carry 55.6 % of the plane's hits at ~35× the good-channel median occupancy, in connector-aligned runs. Not the beam. |
| **per-trigger hot strata** | **`hot_seed_strata.py`** | **new.** all-hot / mostly-hot / clean per trigger per plane, from raw hits. 4 min for all four arms, all sub-runs. Keyed `(subrun, event_id)`. |
| **the downstream cut** | **`k_robustness.py` `no_hotstrip`** | **new.** the table in §0.4. Not folded into `clean`, so the published variants stay comparable. |
| **stratified comparison** | **`compare_hotmasked_rerun.py --strata`** | **new.** the aggregate comparison is a mixture on D and will mislead; this compares like with like. |
| bundle field / fit down-weighting / provenance | `wft/calib.py`, `wft/model.py`, `wft/reco.py` | unchanged, and measured to be a no-op (§0.2) |
| seeding admission | `wft/seed.py` `seed_candidates` | **rewritten**: clusters are now FORMED from the clean strips instead of formed from all strips and then judged. Better defined, measured a wash — see the docstring's own warning. |
| tests | `wft/tests/test_hot_mask.py` | 4 → 7 checks incl. the weld/split cases. Suite green (`test_model_regression` needs a bench cache not staged here; pre-existing). |

---

## 2 · The three tools, and what each is for

```bash
# 1. label every trigger by the hot content of its raw cluster (~4 min, all arms)
python -m sept26_prelim_analysis.hot_seed_strata --arm all --run run_145

# 2. the cut, under the falsification framework that was set before the numbers
python -m sept26_prelim_analysis.k_robustness --run run_145

# 3. only if you re-reconstruct: compare like with like, never in aggregate
python -m sept26_prelim_analysis.compare_hotmasked_rerun \
    --new .../events_prelim.parquet \
    --strata /media/dylan/data/x17/sept26_prelim/hot_strata/hot_strata_run_145_D.parquet
```

---

## 3 · What the condor run actually measured (the original §3, reinterpreted)

Cluster 4141149, D/run_145/stat090_0000, 7 tags, `--hot` carrying 42 x / 57 y
channels. Merged into `/home/dylan/x17/wft_beam145_hotmasked/analysis/`.

| | frozen | hot-masked | reading |
|---|---:|---:|---|
| events attempted | 46 218 | 33 393 | the 29 % all-hot population, mostly |
| both-plane fit converges | 72.6 % | 33.4 % | **a population shift, not a failure** |
| median x_chi2/dof | 10.7 | 41.9 | the 1.27 stratum is gone; 22–25 is what real D tracks cost |
| median x_n_strips | 42 | 32 | toward A's 25 — §3.3 criterion, met |

**The local harness is exact, which is why none of this needed condor.** Run
with the unmasked bundle over 3 600 triggers it reproduces the frozen products
*bit for bit*: `x_ok` agrees on 100.0 %, and `x_p0`, `x_tan_theta` and `x_chi2`
are identical to the last digit on 99.8 % of fitted events (the remainder is
NNLS tie-breaking, |Δ| < 4e-10 on χ² and 0 on the geometry). A configuration
takes ~5 minutes on 8 cores against ~15 minutes of condor round-trip, and it
can be diffed per event, which the condor run could not.

---

## 4 · The two candidate causes, both now answered

### 4.1 `HOT_NOISE_INFLATION = 10.0` — **NO. It is irrelevant.**

Scanned at 10 / 30 / 100 over one fixed set of 3 600 real triggers
(`WFT_HOT_NOISE_INFLATION`). Identical to three decimals everywhere. The
original argument — "windows touching a flagged strip show χ²/dof 45.6 against
6.6" — was reading the stratum effect, not a weighting effect: those windows
have a worse χ²/dof because of *what is in them*, and inflating the noise on a
handful of rows in a 30-strip window cannot move a median.

### 4.2 The seeding admission rule — **right mechanism, no gain.**

It is the driver: 48 % of D's x-planes lose every candidate, and 31 % of them
because the baseline cluster is entirely hot (correctly rejected). The rule is
now the better-defined one — cluster the CLEAN strips, so a hot band can never
weld a noise column onto a real track, and a band narrower than the 12 mm gap
threshold is crossed for free (D's x 448–461 is 14 strips; y 44–63 is 20 and
does split). Tested both ways in `test_hot_mask.py`.

But measured against the old rule on the same 3 600 triggers it is a wash, and
against no mask at all it is worse wherever it changes the seed. **Do not
re-run condor to install it.**

---

## 5 · What this does NOT settle

- **Whether the all-hot triggers contain any real tracks.** They are 29 % of
  D's sample and every piece of evidence says the *cluster* is noise, but a
  real particle could have crossed the chamber in the same window and been
  lost in it. Nothing here tests that; the scintillator tag would.
- **A-y.** It has its own hot connector-8 run (448–460, 21.7 % of the plane's
  hits) that this session found but did not chase. `no_hotstrip` on x alone is
  a no-op there by construction, so A is unaffected either way — but A-y is
  not clean, and anything that uses A's y angles should know that.
- **The other runs.** Everything above is run_145. The classification is per
  run condition (CLAUDE.md) and has to be rebuilt per run.

---

## 6 · Done, and what is left

**DONE — `no_hotstrip` is the production default (2026-09-08).**
`k_arm.coincident_tracks(..., drop_hotstrip=True)` applies it, before the
charge window (a noise column carries charge like a track — median 0.94× — so
leaving them in would set the percentiles the sample is then cut on). It
propagates to everything that reads that sample: the angle scale, and through
it `source_imaging` and the S2/S4 chain. D re-certifies PROVISIONAL at
**k = 1.767**, spread 9.0 %, reproducibility **2.9 %** (was 1.757 / 10.3 % /
5.9 %). A, B and C are bit-identical — the cut removes literally nothing there.

`rerun_chain.sh` runs `noisy_channels` and `hot_seed_strata` **first** now,
ahead of `k_arm`. They used to run last; leaving them there would have
calibrated on the previous run's strata.

**(a) Rebuild the strata for the other runs** once their full passes exist.
`hot_seed_strata.py --arm all --run <run>`, ~4 minutes each. The
classification is per run condition (CLAUDE.md) — a run_145 strata table used
on another run is a silent error, and `dropped_events` returns `{}` rather
than guessing if the table is absent.

---

### The October list

**(b) Candidate ranking, which is the real reason seeding does not help.**
Today `seed_candidates` ranks clusters by strip count, and the old rule and the
new one only change *what gets counted* — neither changes the ranking's
premise. So when the mask makes a real track's cluster smaller than a
competing one, the seed moves, and §0.3 measures what that costs
(χ²/dof 13.1 → 20.6). The fit already computes `_candidate_score` (a
plausibility flag plus Δχ² against "no signal") and is already offered 5
candidates per plane — ranking on that instead would sidestep the strip-count
premise entirely and would be testable in the same 5-minute local loop this
session used. **This is a study, not a constant**, which is why it is not a
one-line change.

**(c) Whether `mostly-hot` should be cut too.** Another 22 % of D. They still
contain real tracks — fitted p0 spread across the whole plane, IQR 308 mm,
against 142 mm for the all-hot pile-up — and cutting them moves the angle
scale by a rounding amount. Not worth the sample for the preliminary; worth
revisiting if D's angle *resolution* ever matters.

**(d) Whether the all-hot triggers hide real tracks.** Every piece of evidence
says the *cluster* is noise, so cutting it is right for the angle scale. But a
real particle could have crossed D in the same window and been lost inside
one — which would make this a 29 % efficiency hole in D rather than a clean
cut. The scintillator tag decides it: do all-hot triggers carry a wall+plastic
coincidence at the rate real tracks do? Nothing here tests that.

**(e) A-y.** Its own hot connector-8 run (448–460, 21.7 % of the plane's hits),
found this session and not chased. `DROP_PLANES = ('x',)` means A is untouched
either way, but A-y is not clean and anything using A's y angles should know.

**(f) The physical fix.** Hot and dead are the same connector fault in the same
places, on D and now A-y. Nothing in software recovers 55.6 % of a plane's hits
being noise.

---

## 7 · Where everything is

| | |
|---|---|
| git commit with the wildcard mechanism | `a77c262` |
| the frozen baseline (untouched) | `/media/dylan/data/x17/sept26_prelim/fullpass/run_145/stat090_0000/mx17_D/events_prelim.parquet` |
| hot-channel classification (source of truth) | `<out>/noisy_channels/noisy_channels_run_145.csv` |
| **per-trigger strata, all four arms** | `<out>/hot_strata/hot_strata_run_145_<arm>.parquet` |
| **k under every variant incl. `no_hotstrip`** | `<out>/kcal/k_robustness_run_145.csv` |
| the condor hot-masked table (kept, for reference) | `/home/dylan/x17/wft_beam145_hotmasked/analysis/run_145/stat090_0000/mx17_D/events_prelim.parquet` |
| condor package (local + lxplus) | `/home/dylan/x17/wft_beam145_hotmasked/`, `lxplus:~/wft_beam145_hotmasked/` |
| local raw data (combined_hits + decoded_root) | `/media/dylan/data/x17/beam_july/runs/run_145/stat090_0000/` |
