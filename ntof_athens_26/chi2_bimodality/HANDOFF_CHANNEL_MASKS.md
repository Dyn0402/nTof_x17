# HANDOFF — no reconstruction in this campaign masks a single channel

**Written 2026-09-12, out of `chi2_bimodality/`.** Local work only; nothing
re-reconstructed.

Companions: [`HANDOFF_D_NOISY_CHANNELS.md`](../../sept26_prelim_analysis/HANDOFF_D_NOISY_CHANNELS.md)
(what the channels *are*), [`HANDOFF_HOT_WILDCARD_TUNING.md`](../../sept26_prelim_analysis/HANDOFF_HOT_WILDCARD_TUNING.md)
(why the first mask attempt made D worse), [`README.md`](README.md).

---

> ## §5 steps 1–2 attempted 2026-09-12 — [`channel_masks/`](../channel_masks/README.md)
>
> **Step 2 (stability) is answered: the classification is hardware, not
> occupancy.** Run over the 5 campaign runs whose `combined_hits` is staged
> locally (run_79, 86, 116, 145, 162 — six weeks, both access conditions):
>
> - The classifier **blindly recovers the A-x connector-8 fault** that
>   `CLAUDE.md` records from the DAQ side — 43 of channels 448–511 dead in
>   run_79, 0 of 64 in every post-access run. Nothing in it is told that
>   connector exists.
> - The hot **set** is stable: Jaccard against run_145 is 0.52–0.81 (D·x),
>   0.70–0.89 (D·y), 0.57–0.86 (A·x). Same physical channels.
> - **D's headline generalises** — `hits_in_hot` is 45–56 % on D·x and 33–53 %
>   on D·y in *every* run, not just run_145.
>
> So step 2's worry is retired and step 3 is not blocked on it.
>
> ### ⚠ One correction: "Chamber C is clean" is false for run_79
>
> §2 and §4 both lean on C having **zero** hot channels. That is true from
> run_86 on. In **run_79 — the first long production run** — C·x carries
> **15.8 %** of its hits in hot channels and C·y **15.0 %**, and B·x carries
> **32.7 %**; the 27 July access removed all three. Measured in each of run_79's
> 11 sub-runs separately, so it is not a pooling artefact.
>
> §4's *argument* survives (the A-vs-C χ² comparison is campaign-wide, where C is
> overwhelmingly post-access), but the sentence must not be carried to run_79 —
> which already carries the dead A-x connector and now has a second, unrelated
> condition on two other chambers.
>
> ### Step 1 is blocked on staging, not on analysis
>
> Of the **289** campaign sub-runs with a `combined_hits_root`, this machine has
> **21, in 5 runs**; the rest of the directories exist and are *empty*, which is
> why a campaign pass returns quietly instead of failing.
> `<out>/channel_masks/missing_runs.csv` names every one. Note also that the
> 23 July `RdClk_Div` split does **not** cut this sample — every campaign run is
> run_79 or later, so all of them are on the noisy side.

---

## 1 · The gap, in one sentence

Every piece of machinery exists — a per-channel classifier, a bundle patcher,
and `wft/model.py` support for both wildcards — and **none of it is in the
bundles the campaign actually ran.**

| | state |
|---|---|
| `noisy_channels.py` | works, classifies good/hot/dead/noisy per channel | **run_145 only** |
| `apply_hot_wildcards.py` | works, writes `calib_bundle_hotmasked` | **run_145 / arm D only** |
| `wft.model.prep_plane` | reads `dead` (censored) and `hot` (noise inflated) | ready |
| bench bundles `mx17_A..D` | `dead = {}` on A, B, D; absent on C; **no `hot` key on any** | empty |
| `calib_bundle_prelim` (what the campaign ran) | derived from the above by `wft_beam.make_bundle`, which adds no masks | **empty** |

So the whole 36-run campaign reconstruction ran with every dead and every hot
channel in the fit at full weight.

## 2 · The scale of it, measured

`noisy_channels_summary_run_145.csv`, per plane, fraction of the 512 channels
and the fraction of all hits those channels carry:

| arm·plane | dead | hot | hot bands | **hits in hot** |
|---|---:|---:|---:|---:|
| A·x | 0 % | 1.4 % | 7 | 6.0 % |
| A·y | 0 % | 3.1 % | 13 | **21.7 %** |
| B·x | 1.2 % | 0 % | 0 | 0 % |
| B·y | 0 % | 0.2 % | 1 | 0.6 % |
| **C·x** | 0.4 % | **0 %** | 0 | **0 %** |
| **C·y** | 0 % | **0 %** | 0 | **0 %** |
| D·x | 2.1 % | 8.2 % | 25 | **55.6 %** |
| D·y | 4.3 % | 11.1 % | 32 | **50.1 %** |

**Chamber D is half noise by hit count on both planes. Chamber C is clean.**

## 3 · Why this is not just "D is noisy" — it corrupts χ² in the wrong direction

From `compare_hotmasked_rerun.py`'s docstring, measured on the 2026-09-08 D
re-run:

> D's 42 hot x channels carry 55.6 % of the plane's hits, and a third of the
> frozen events' largest x cluster is made ENTIRELY of them. Those noise
> columns fit **better** than real tracks — median χ²/dof **1.27** against 25
> for an uncontaminated window, 99.6 % `quality_ok` — because a smooth
> coherent-noise deposit is easy for the forward model to explain.

So unmasked hot channels do not inflate χ². **They manufacture low-χ² "tracks".**
Any χ²-based quality argument on an unmasked chamber is flattered by them, and
this is exactly why the first mask attempt read as a catastrophic regression
(→ `HANDOFF_HOT_WILDCARD_TUNING.md`) and why `--strata` exists.

### ⚠ What this does to the χ² bimodality result — bounded, not dismissed

The obvious worry is that chamber A's low-χ² mode is partly noise columns. It is
mostly not, and here is the bound. Mapping each selected track's `x_p0` onto the
run_145 channel classification:

| arm | selected tracks (run_145) | centred on a hot/noisy channel | window touches one |
|---|---:|---:|---:|
| A | 423 | **1.2 %** | 13.7 % |
| C | 350 | 0 % | 0 % |
| D | 435 | 0.2 % | 17.7 % |

The 5 A tracks centred on a bad channel do behave as the quote predicts — median
χ²/dof 1.15, 80 % at the floor — but at 1.2 % of the sample they cannot make a
mode holding 37 %. The published selection (gated, angle-calibrated, DCA < 30 mm,
paired) evidently removes most noise columns, which is expected: a noise column
points nowhere near the beam axis, so the DCA cut kills it.

**This is run_145 only, n = 5 in the critical cell, and x-plane only.** It bounds
the contamination; it does not characterise it. Redo it per run and on y (where
A carries 21.7 % of hits in hot channels) before quoting the bound anywhere else.

## 4 · What masking will and will not fix

**Will:** chamber D. Half its hits are in channels that should be down-weighted,
and `HANDOFF_D_NOISY_CHANNELS.md` already shows masking moves D toward a better
calibration on every metric (estimator spread 0.103 → 0.056, sub-run
reproducibility 0.059 → 0.029, both then the best of any chamber).

**Will not:** the χ² ordering that `chi2_bimodality` found. C has **zero** hot
channels and still sits a factor 1.9 above A on short tracks. So masking is not
the explanation for A-vs-C, and anyone hoping this handoff closes that question
should read `README.md` §5 instead — the live suspect there is C's
`sigma_p0`/`Dp`, an order of magnitude off the other three and carried into the
beam bundle verbatim.

## 5 · What to do

1. **Run `noisy_channels.py` over the whole campaign**, per run and per run
   condition, not just run_145. The 23 July `RdClk_Div` change doubled the noise
   floor campaign-wide (CLAUDE.md), so a classification from one side of it does
   not describe the other, and the thresholds in the meta
   (`hot_factor = 5.0`, `dead_thresh = 0.2`) were tuned on run_145 alone.
2. **Check stability first.** Before masking anything, confirm a channel
   classified hot in run_145 is hot in run_100 and run_162. If the classification
   moves run to run it is measuring occupancy, not hardware, and the connector
   argument in `HANDOFF_D_NOISY_CHANNELS.md` §2.3 says it should be hardware.
3. **Patch the bench bundles, not the derived ones.** `make_bundle` is a pure
   function; adding masks upstream means every future beam bundle inherits them
   and the provenance stays honest. Per CLAUDE.md, write a new bundle name —
   never mutate one a frozen product points at.
4. **Re-reconstruct stratified, and judge it stratified.** Use
   `hot_seed_strata.py` + `compare_hotmasked_rerun.py --strata`. An aggregate
   before/after on D will say the mask made things worse; that is the known trap,
   not a result.
5. **Then re-run `chi2_bimodality`** and see whether D's low-χ² population
   shrinks. If the "noise columns fit better" mechanism is right, masking should
   *remove* low-χ² tracks from D and A·y — a prediction worth declaring now, so
   it counts when it is checked.

## 6 · What would make this wrong

- If the classification is not stable run to run, everything above is measuring
  a run's occupancy rather than a broken channel, and step 1 produces a mask that
  varies for no physical reason.
- If masking D removes so much of its sample that nothing is left to measure —
  D's unbiased hit map already fails at the loosest mask scanned, with 45 % of the
  sample gone at the tightest (`ntof_athens_26/README.md`). Masking may simply
  reveal that D cannot carry this analysis, which is a legitimate outcome and
  should be reported as one rather than tuned around.
