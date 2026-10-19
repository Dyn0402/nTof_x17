# HANDOFF — chamber D's noisy channels: identify them, and make them wildcards

**Written 2026-09-08. Local work on run_145 only.**

Companions: [`STATUS.md`](STATUS.md), `k_robustness.py`, `source_imaging.py`,
<https://dylan-neff.web.cern.ch/x17/detector-response/>.

---

## 1 · Why this matters more than it looks

Chamber D's noisy cells are, on the measurements so far, **the single thing
standing between D and a clean calibration**:

- they take **22.4 %** of D's pointing-coincident sample, against 1–5 % for the
  other chambers;
- they are the **only** cut of four that moves D's angle scale (`k_robustness.py`:
  outer ring +0.93 %, dead channels 0.00 %, **hot cells +3.76 %**);
- removing them moves D toward a *better* calibration on every metric —
  estimator spread 0.103 → 0.056 and sub-run reproducibility 0.059 → 0.029, both
  then the best of any chamber;
- they also show up independently in the head-on study, where D has **1.33×**
  more unreliable-slope tracks than geometry allows even after masking them.

D is one of only three chambers carrying angles, and it is half of the B–D
opposing pair the X17 topology needs. **Fixing D properly is worth more than
any other single detector task.**

---

## 2 · What they are — measured, and it is not what was assumed

### 2.1 Not discharges, and not simply small signals

Dylan's two hypotheses were discharges (large signals) or noise (small
signals). Neither is right on its own:

| | median q | q p10 | median n_strips |
|---|---:|---:|---:|
| D, hot cells | 1 851 | 245 | **42** |
| D, normal cells | 1 971 | 548 | **25** |
| **ratio** | **0.94** | 0.45 | **1.7** |

**The median charge is the same** (0.94, and arm A gives the identical 0.94),
so these are not discharges. What differs is the *shape*: an excess of
low-charge clusters (p10 down 2.2×) spread over **1.7× more strips**. That is
the signature of **wide, dilute, low-density deposits** — coherent or
correlated noise being fitted as a cluster — rather than either of the two
hypotheses.

> ⚠ `x_q_sum` has a broken upper tail (p90 reaches 10¹³). This is the same
> `q_uend` railing that forced `q_per_len` to be withdrawn (STATUS). **Use
> medians and low percentiles; never a mean, and never p90.**

### 2.2 They are whole x COLUMNS, not regions — so the algorithm is per-channel

Mapped on a 10 mm grid, D's hot cells are not blobs. They are **vertical bands
in u that span the entire plane height**:

```
  u [ 50, 60) mm : 10 of 40 v-cells hot, v range [0,400)   <- full height
  u [ 60, 70) mm : 29 of 40 v-cells hot, v range [0,400)   <- full height
  u [200,210) mm :  8 of 40                v range [0,400)
  u [300,310) mm : 13 of 40                v range [50,390)
  u [390,400) mm : 15 of 40                v range [0,400)
```

A noisy *region* would be localised in both coordinates. A noisy *channel* is
hot at every v. **These are bad x strips.**

### 2.3 They sit on connector boundaries — the same fault as the dead channels

512 channels over 398.58 mm is 64 channels per connector = **49.8 mm**, so
connector boundaries fall at u ≈ 50, 100, 150, 200, 250, 300, 350, 400 mm. The
hot bands begin at **50, 200, 250, 300 and 380–400 mm**.

D's *dead* runs were already found on connector boundaries
(STATUS, 2026-09-08). **Hot and dead are the same class of fault in the same
places** — a connector, grounding or cabling problem, not a detector-surface
problem. That is a strong hint for the physical fix and it means the ID
algorithm should be **connector-aware**.

### 2.4 The scale of it

**80.6 % of all D's fitted clusters sit in cells covering 10.8 % of the plane.**
D's occupancy is dominated by a small, structured minority of its channels.
(Arm A for contrast: 12.1 % of clusters in 2.1 % of the plane.)

---

## 3 · What to build

### 3.1 A per-channel classifier, not a per-cell mask

The current `k_robustness.hot_cells` is a blunt 2D instrument built to *test*
whether hot cells move k. It is not the right thing to put in the
reconstruction. Replace it with a **per-strip** classifier on the raw
`combined_hits` occupancy, per run condition, producing for each channel one of:

| class | meaning | how to find it |
|---|---|---|
| `dead` | no hits | occupancy < 20 % of the plane median (already in `source_imaging.dead_ranges`) |
| `hot` | fires far above its neighbours | occupancy > N× the **local** median, N tuned on A/C which are clean |
| `noisy` | ordinary rate, wrong *shape* | normal occupancy but a low-charge / wide-cluster excess (§2.1) |
| `good` | — | everything else |

Two design points learned the hard way:

- **Use a local median, not the plane median.** The plane median is dragged by
  the trigger's own two-lobe illumination structure (`plastic_acceptance.py`),
  which is real physics and must not be flagged as hot.
- **Take the median over OCCUPIED channels only.** Letting dead channels into
  the median drags the threshold down and calls ordinary channels hot — this is
  already handled in `hot_cells` and the same trap applies per-strip.

### 3.2 Wildcards in the reconstruction — Dylan's specification

> *"don't seed on them or take them too seriously but don't outright exclude
> them and kill tracks either"*

Concretely, in `wft/reco.py`:

1. **Never seed on a flagged channel.** Seeding is where a noise cluster
   becomes a track, and it is the cheapest place to stop it. This alone should
   recover most of what masking recovers.
2. **Keep flagged channels in the fit, down-weighted.** Inflate their
   uncertainty rather than dropping the sample, so a real track crossing a bad
   strip keeps its continuity and its χ² is not distorted by a hole. A track
   should never be lost *because* it crossed a bad channel.
3. **Cap their influence.** A flagged channel must not be able to pull the fit
   on its own — the natural implementation is a per-channel weight in the NNLS
   design matrix, which is where the model already lives.
4. **Record it.** A `n_flagged_strips` column per plane in the track row, so any
   downstream cut can be re-derived without re-running the fit — the same
   contract the rest of the schema holds to.

### 3.3 How to know it worked

Success criteria, stated now so they cannot be chosen afterwards:

- D's angle-scale estimator spread falls toward the ~0.056 the hot-cell mask
  already reaches, **without** discarding 22 % of the sample;
- D's head-on excess (1.33× geometry, §1) closes;
- D's cluster width falls from 42 strips toward A's 25;
- **A and C do not move.** They are nearly clean, so the classifier must be a
  no-op on them. If it moves them, the threshold is wrong.
- the number of gated tracks in D goes **up**, not down — the point is to stop
  trusting bad channels, not to throw away the tracks that cross them.

---

## 4 · Order of work

1. Build the per-strip classifier standalone and **look at the map** for all
   four chambers before touching the reconstruction. Confirm §2.2 and §2.3 at
   channel granularity, and confirm A and C come out nearly empty.
2. Check it against the *hits* level, not the fitted clusters — the current
   evidence is all from `events_prelim`, which is post-fit and therefore
   partly circular. `combined_hits` is the honest input and is already local.
3. Persist it as a per-(run condition, chamber) product, since anything
   calibrated is per detector **and** per run condition (CLAUDE.md).
4. Only then wire it into `wft/reco.py`, and re-run run_145 to compare against
   the frozen products, which are all on disk.

---

## 5 · Where everything is

| | |
|---|---|
| the 2D hot-cell finder used for the tests | `k_robustness.hot_cells`, `k_robustness.Masks` |
| the dead-channel finder (per-run, from occupancy) | `source_imaging.dead_ranges` |
| what the cuts do to k | `<out>/kcal/k_robustness_run_145.csv` |
| D's surface summary | `<out>/kcal/k_robustness_surface_run_145.csv` |
| cluster shape per chamber | `<out>/chamber_b/chamber_b_shape_run_145.csv` |
| the head-on cross-check | `<out>/angle/slope_reliable_crosscheck_run_145.csv` |
| raw hits (the honest input for §4.2) | `<runs>/run_145/<subrun>/combined_hits_root/` |

**Nothing here needs EOS or a campaign pass.** run_145 is complete on disk and
every product above rebuilds with `bash sept26_prelim_analysis/rerun_chain.sh`.
