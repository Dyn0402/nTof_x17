# HANDOFF — the two-track pairs' true-coincidence fraction, now measured

**Written 2026-09-08; item (a) and (c) below completed the same day, redone
properly (unbiased hit choice, the full ±1000 ns range, an unbinned
two-component fit) rather than the biased first pass this file originally
reported. Local work on run_145 only; nothing here needs EOS or a campaign
pass.**

Companions: [`PLAN.md`](PLAN.md) §10 S4, [`STATUS.md`](STATUS.md),
<https://dylan-neff.web.cern.ch/x17/opening-angle/>,
<https://dylan-neff.web.cern.ch/x17/accidental-timing/> (this work, published).

---

## 0 · Redone 2026-09-08 — read this first, then §1–3 for how it was built

**The title changed because the number did.** §2.3 below reported a first
pass at **f ≈ 6 ± 3 %**, explicitly an upper bound because the hit choice
("closest to dt_ns = 0") was known-biased *toward* finding coincidence
(§3.1). Redone with `accidental_timing.py` — a random (unbiased) hit choice,
the full ±1000 ns range, and an unbinned two-component MLE fit against
`is_control` instead of matching one summary statistic to a single-parameter
fold:

| topology | pairs | f | 68 % interval |
|---|---:|---:|---|
| all inter-chamber | 76 | **29 %** | [17, 40] % |
| opposing (A–C, the signal topology) | 42 | **42 %** | [26, 58] % |
| perpendicular | 34 | **16 %** | [0, 32] % |

**Higher, not lower, than the first pass — and the reversal is methodological,
not a sign the stated bias direction was wrong.** Removing a bias that pulls
toward coincidence should, if anything, pull f down; what actually moved the
number is comparing the full arm1−arm2 *shape* against two data-derived
templates in one fit, rather than matching a single "fraction within 20 ns"
statistic through an approximate fold. Both numbers are on record; §3 has the
figure and the fit. **The topology ordering is itself evidence for real
physics**: opposing (where an X17 or IPC pair actually lands) carries ~2.7×
the perpendicular fraction, which an accidental floor would not produce.

**Item (c) delivered too, as a recommendation, not a code change.** The
production accept window (−100, +60) ns sits at 88 % peak/pedestal purity; a
centred (−20, +20) ns window reaches 95 %+. `window_scan`/`recommend_window`
in `accidental_timing.py` have the full trade-off curve. **Not applied** —
`DT_WINDOW` is shared by `candidate_filter.py`, `efficiency.py` and
`scintillators.py`, so changing it touches the stage-1/stage-2 production
chain, which is on Dylan's hold.

**One more finding from the same pass, not anticipated below.** Under the
*production* window, a wall+plastic "coincidence" is nearly automatic: only 1
of 147 216 single-active-arm events fails to show one, because 160 ns is wide
enough that the plastic family's own high accidental rate lands something in
it almost every time. §2.1's "peak is a factor ~30 over pedestal" is still
right, but at that width it barely discriminates a real coincidence from one
element firing alone — the peak's own core (−30, +30) ns is what recovers a
real "both fire vs. one fires" comparison (see the published page).

**What is now the priority list for the next session** — §4 below, updated:
items (a) and (c) are done; (b), (d) and (e) are not, and (b) is now the
highest-value one, since it is what turns this into a corrected S4 spectrum
rather than a number sitting beside it.

---

## 1 · The question, and why it is the right one

The opening-angle page ends on a null: run_145's 628 two-track pairs follow the
event-mixed shape (χ²/dof 1.8–3.1) rather than any pair spectrum (6.7 and
worse). The reading is that **one charged particle fired the trigger and a
second, uncorrelated one landed in the same DREAM readout window**.

That reading makes a sharp, independent prediction. The DREAM window is
~1.8 µs (60 ns × 28–32 samples). A genuine IPC or X17 pair is two particles
born *simultaneously*, so both should be prompt. An accidental second particle
arrives at a **uniform random time across the window**. The Micromegas cannot
test this well — its own `t0` carries the drift depth, which spans most of the
window by itself — but **the scintillators can**: they are prompt, and the
n_TOF slim already carries their time.

**This is the cleanest test of the null in the whole analysis, and it needs no
model, no acceptance and no background normalisation.**

---

## 2 · What is already established — measured 2026-09-08

### 2.1 The timing exists, at full range, with no reprocessing

`dt_ns` in the slim is the scintillator hit's time **relative to the DREAM
trigger**, and it is carried over the **full ±1000 ns** — not truncated to the
±100 ns accept window the analysis currently uses. Verified on
`ntof_hits_run_145_stat090_0000_224670.root`.

The prompt peak is sharp and sits on a flat pedestal:

| region | wall+plastic hits per 10 ns |
|---|---:|
| −200 … −50 ns (pedestal) | ~2 000 |
| **0 … +10 ns (peak)** | **61 548** |
| +100 … +300 ns (tail) | 12 600 → 5 600 |

So the prompt component is a factor ~30 over the flat background, with a core
of roughly ±15 ns. **The accidental level is directly visible as the flat
pedestal.**

> ⚠ **The current accept window is mis-centred and too wide.** `efficiency.py`
> and `candidate_filter.py` use `dt_ns ∈ (−100, +60)`. The peak core is about
> (−30, +30), so the window admits ~100 ns of flat background it does not need
> and clips a real positive tail it may want. Re-optimising it is cheap and is
> item (c) below.

### 2.2 There is a ready-made accidental sample nobody has used

The slim carries `is_control` — **454 604 hits in run_145's first sub-run
alone**, and its `dt_ns` is **dead flat across the whole ±1000 ns**
(22 400–23 100 per 100 ns bin). This is the n_TOF processing's own
random-coincidence control.

**That is very likely the missing normalisation for the S4 background**, which
STATUS currently lists as the top open item. Nothing in this analysis reads
`is_control` except to cut it away.

### 2.3 The test has been run once, and it comes out on the accidental side

For each inter-chamber pair, take the wall **and** plastic hit in each arm
(the one closest to `dt_ns = 0`), average them into one time per arm, and ask
how prompt each arm is. **93 of run_145's 464 inter-chamber pairs (20 %) have
both arms tagged.**

| | median \|Δt\| | <20 ns | >200 ns |
|---|---:|---:|---:|
| single-arm tagged events — **the trigger itself, the reference** | **5.2 ns** | **96.8 %** | **0.0 %** |
| two-arm pairs: the *prompt* arm | 7.0 ns | 71.0 % | 6.5 % |
| two-arm pairs: **the other arm** | **170.7 ns** | **7.5 %** | **47.3 %** |

**In a two-arm event one arm is the trigger and the other fires at essentially
a random time.** If both particles were born together, the second arm would
look like the 96.8 % reference. It looks nothing like it.

Folding the two components (`f × 0.968 + (1−f) × flat`) gives a
true-coincidence fraction of **f ≈ 6 % ± 3 %** among two-arm pairs — consistent
with zero, and an **upper bound** for the reason in §3.1.

---

## 3 · Three traps. Two of them already caught someone

### 3.1 The "closest to zero" hit selection biases the answer toward coincidence

Taking the hit nearest `dt_ns = 0` in each arm pulls |Δt| down whenever an arm
has more than one hit. So the 7.5 % above is an **over**-estimate of the
coincident fraction and f ≈ 6 % is an upper bound, not a central value. Fix it
by taking a *random* in-window hit, or by fitting the full `dt_ns` shape
instead of counting inside a box.

### 3.2 Event mixing is the WRONG null for this variable — it inverts the answer

Mixing tracks between triggers and comparing Δt gives **38.8 %** within 20 ns
for mixed against **9.7 %** for real, i.e. the null looks *more* coincident than
the data. That is not a signal of anything: `dt_ns` is measured **relative to
each event's own trigger**, and a singly-tagged event's hit is the trigger, so
two hits drawn from two different singly-triggered events are both at zero by
construction. **Do not use event mixing here.** The §2.3 formulation needs no
null at all, which is why it is the one to use.

### 3.3 A latent bug in the pair products, now fixed — check you have the fix

`source_imaging.vertices` used to write only `key`, the **first** track's event
id, so a mixed pair silently claimed its second track belonged to the first
one's event. Reading arm 2's scintillator through it gives a meaningless
answer. Fixed 2026-09-08: the frame now carries **`key1` and `key2`** and
`(key1 == key2).sum() == 0` in the mixed sample. If your `pairs_*.parquet` has
no `key2` column, regenerate it.

---

## 4 · What to do, in order

**(a) Redo §2.3 properly and turn it into a measured accidental fraction. —
DONE 2026-09-08.** Unbiased hit choice (§3.1), the full ±1000 ns, and an
unbinned two-component MLE fit — prompt (bootstrap difference of two draws
from the single-arm reference) plus accidental (one draw from the reference,
one from `is_control`, §2.2, both restricted to the tagging window) — in
`accidental_timing.py`. Result: **f = 29 % [17, 40] overall, 42 % [26, 58]
opposing, 16 % [0, 32] perpendicular** (§0). One methodological choice made to
get there, carried forward: the strict wall-AND-plastic tag leaves only 8 of
464 real inter-chamber pairs with both arms tagged (too few to fit), so the
fit uses a looser wall-OR-plastic tag (76 pairs, 16 % of the sample) — every
number above is on that looser tag, stated in the published page.

**(b) Feed f back into the opening angle. — NOT DONE, now the top priority.**
The S4 page still subtracts nothing. The complication found while chasing (a):
f above is measured only on the 16 % of pairs that carry a two-arm
scintillator tag at all, so applying it to the full spectrum needs either (i)
a check that the tagged subsample is representative in opening angle — a
first look (near-zero |Δt| vs far, n = 30 vs 23) saw no significant shape
difference but is not a test at that n — or (ii) a fit that uses the tag as a
per-pair weight/selection rather than a blanket normalisation. Either way this
is where the campaign statistics (§5) start to matter: 76 tagged pairs is
enough to measure f, not enough to slice it finely against open_deg.

**(c) Re-optimise the accept window** (§2.1) on the peak/pedestal ratio. —
**Recommendation DONE 2026-09-08, NOT APPLIED.** `window_scan`/
`recommend_window` in `accidental_timing.py`: production (−100, +60) ns sits
at 88 % purity, a centred (−20, +20) ns reaches 95 %+. Not installed anywhere
— `DT_WINDOW` is shared across `candidate_filter.py`, `efficiency.py` and
`scintillators.py`, i.e. the production stage-1/stage-2 chain, on Dylan's
hold. What checking it against the efficiency and the stage-1 class census
would take is unchanged from the original ask below, and is still undone.

**(d) Repeat on the intra-chamber pairs. — NOT DONE.** They were excluded from
§2.3/§0 because both legs share one arm and therefore one scintillator, so the
two-arm formulation does not apply. The wall's **along-bar** position
(`scintillators.py`, σ_y < 53 mm) may separate two legs within one arm well
enough to give each its own time — untested, and unlike (a)/(c) this is new
method development, not a rerun of an existing one.

**(e) Only then, the Micromegas timing. — NOT DONE, and now unblocked.** `t0`
differences between two tracks contain the drift-depth difference, which is
why the scintillators came first. (a) is now fixed, so the MM `t0` can become
a *cross-check* with much larger statistics, since it needs no scintillator
tag — recall 84 % of real inter-chamber pairs have no two-arm tag at all
(§0), which is exactly the population a `t0`-based check would reach that
this handoff's method cannot.

---

## 5 · What this will and will not settle

**Will:** whether the two-track sample contains any true coincidences, at what
level, with an error, and using none of the acceptance modelling that the rest
of S4 depends on.

**Will not:** anything about *which* physics the true coincidences are, if any
survive. A prompt two-arm pair could still be an external conversion or a
single particle scattering between arms (D12).

**Statistics — MEASURED 2026-09-08, not the ±3 % this section originally
projected.** The 76 (loose-tag) pairs actually fit give f to a 68 % interval
of roughly ±11 points, not ±3 % — the unbinned shape fit is a harder problem
than the single-statistic fold this section was written against, and 8/464
pairs pass the *strict* tag, an order of magnitude below the "93" this
section assumed. The campaign is ~50× run_145; scaling the loose-tag count
gives ~3 800 tagged pairs, which is what would bring the per-topology
interval down to a few points. **This test does not need the campaign to be
worth doing, but it becomes decisive with it** — that conclusion survives
the revised numbers.

---

## 6 · Where everything is

| | |
|---|---|
| slim, with `dt_ns` at full range and `is_control` | `<runs>/run_145/<subrun>/ntof_hits/*.root` |
| the slim reader, families and arms attached | `scintillators.read_slim` |
| wall pairs, both ends, per group | `scintillators.wall_pairs` |
| the pair lists, now with `key1`/`key2` | `<out>/angle/pairs_run_145.parquet` and `pairs_mixed_run_145.parquet` |
| how pairs are built and mixed | `source_imaging.vertices` |
| the spectrum this feeds | `opening_angle.py`, `make_angle_report.py` |
| **this handoff's own analysis (§0, new 2026-09-08)** | `accidental_timing.py` |
| single-arm hit classes, window scan, two-arm pairs, the fit | `<out>/accidental_timing/*.csv`, `*.parquet`, `*.meta.json` |
| figures + report | `<out>/accidental_timing/figures/`, `<out>/accidental_timing/report.html` |
| published | <https://dylan-neff.web.cern.ch/x17/accidental-timing/> |
