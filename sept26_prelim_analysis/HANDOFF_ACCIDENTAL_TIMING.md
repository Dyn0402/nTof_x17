# HANDOFF — the two-track pairs are accidentals, and the scintillator timing says so directly

**Written 2026-09-08. Local work on run_145 only; nothing here needs EOS or a
campaign pass, and the measurement below is already half done.**

Companions: [`PLAN.md`](PLAN.md) §10 S4, [`STATUS.md`](STATUS.md),
<https://dylan-neff.web.cern.ch/x17/opening-angle/>.

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

**(a) Redo §2.3 properly and turn it into a measured accidental fraction.**
Unbiased hit choice (§3.1), the full ±1000 ns, and a two-component fit — prompt
(shape taken from the single-arm tagged events, which is the trigger's own
resolution function) plus accidental (shape taken from `is_control`, §2.2) —
with the prompt fraction floating. That yields **f, the true-pair fraction, with
an error**, per topology and per arm pair. *This is the number S4 is missing.*

**(b) Feed f back into the opening angle.** The S4 page currently subtracts
nothing and says so. With f measured, the accidental component can be
subtracted with the mixed sample supplying the shape and f the normalisation,
and the spectrum becomes a measurement rather than a shape check.

**(c) Re-optimise the accept window** (§2.1) on the peak/pedestal ratio, and
check what it does to the efficiency and to the stage-1 class census. Expect
a purity gain; verify it is not an efficiency loss.

**(d) Repeat on the intra-chamber pairs.** They were excluded from §2.3 because
both legs share one arm and therefore one scintillator, so the two-arm
formulation does not apply. The wall's **along-bar** position
(`scintillators.py`, σ_y < 53 mm) may separate two legs within one arm well
enough to give each its own time — untested.

**(e) Only then, the Micromegas timing.** `t0` differences between two tracks
contain the drift-depth difference, which is why the scintillators come first.
Once (a) fixes the scale, the MM `t0` becomes a *cross-check* with much larger
statistics, since it needs no scintillator tag — and 80 % of pairs have no
second-arm tag at all.

---

## 5 · What this will and will not settle

**Will:** whether the two-track sample contains any true coincidences, at what
level, with an error, and using none of the acceptance modelling that the rest
of S4 depends on.

**Will not:** anything about *which* physics the true coincidences are, if any
survive. A prompt two-arm pair could still be an external conversion or a
single particle scattering between arms (D12).

**Statistics.** 93 pairs with both arms tagged in run_145 gives f to ±3 %. The
campaign is ~50× — about 4 600 tagged pairs, so f to a few tenths of a per
cent. **This test does not need the campaign to be worth doing, but it becomes
decisive with it.**

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
