# ntof_athens_26 — the Athens talk

The n_TOF conversion of the MPGD2026 deck. `slides/ntof_athens_talk.pptx` is the
deck; the scripts in this directory build figures for it, and the
subdirectories are studies the figures raised.

## What is here

| where | what | entry point |
|---|---|---|
| this directory | the deck's figure builders: the overhead fans and the side view, opening-angle topology, pair quality, in-situ performance (sections below) | this README |
| `chi2_bimodality/` | why chamber A's worst-view χ²/dof is double-humped; holds `HANDOFF_T0_PRIOR.md` and `HANDOFF_CHANNEL_MASKS.md` | `chi2_bimodality/README.md` |
| `xy_t0/` | what the two planes agree on, and what `dt_xy` is | `xy_t0/README.md` |
| `channel_masks/` | whether the hot/dead channel classification is hardware or occupancy | `channel_masks/README.md` |
| `pair_vertex_imaging/` | the two-track vertex series, 2026-09-12 → 13: why the pair vertex is not a 10 mm image, the blurred image it is in x, where z went (chamber D), a balanced 3D density, the vertex along the beam, and why same-chamber pairs cannot be tested yet — four published notes | `pair_vertex_imaging/README.md` |

Two handoffs came out of this package and live with the other campaign
handoffs: `sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md` (the
reconstruction work `pair_vertex_imaging` raised) and
`sept26_prelim_analysis/HANDOFF_CAPSULE_Y.md` (the ray-tracing test that turns
the side view's +30 mm into a fitted capsule position).

## The opening-angle topology figures

```bash
../.venv/bin/python make_topology_figures.py            # all of them
../.venv/bin/python make_topology_figures.py --only pairings
../.venv/bin/python make_topology_figures.py --norm counts --bin 15
../.venv/bin/python make_topology_figures.py --b2b drop  # cut back-to-back
```

On the Windows box the data disk is a drive letter, so point the tree at it
first: `X17_ROOT=D:/x17`. Everything else resolves through
`sept26_prelim_analysis/paths.py`.

Each figure is written three ways into `figures/`: `.png` for a quick paste,
`.pdf` because the type stays live and scales to any slide, and `.csv` so the
numbers can be checked without re-running anything. `figures/index.html` is a
contact sheet of the set.

| figure | what it is |
|---|---|
| `topology_split` | **the explainer.** Three maps of the four chambers in transverse section, one representative pair each, and under them the angular reach each class actually has in the data. |
| `topology_map` | the same three maps without the reach strip, for a slide that builds the strip in afterwards. |
| `pairings` | **the result.** The measured distributions, data only, one panel per class with the arm pairs overlaid — three intra (A–A, C–C, D–D), two perpendicular (A–D, C–D), one opposing (A–C). |
| `pairings_intra`, `_perpendicular`, `_opposing` | one panel each, for a build. |
| `pairings_counts` | the same three panels as raw counts rather than unit-area shapes, for when the yield is the point. |

### Where the numbers come from

One file: `<out>/angle_campaign/pairs.parquet`, which `campaign_angle.py`
writes. 33 runs of the condor full pass, **each run on its own angle scale**
(not the borrowed run_145 `k`), every unordered pair of selected tracks inside
one trigger. 59 588 pairs after the two cuts below.

### Three things the figures decide, and why

**Chamber B is in no pairing.** It has no field-shaping rings, so its drift
field is not uniform, there is no clean time↔depth ladder and it measures no
angle (`sept26_prelim_analysis/STATUS.md`, 2026-09-08). It is still *drawn* on
the maps, hatched, because its absence is what costs the B–D opposing channel —
half the signal topology. That is why there is one opposing pair and not two.

**Back-to-back pairs are kept.** One charged particle crossing the target and
punching through both opposing chambers would read as a pair at ~180°, and
would be perfectly time-coincident because it is one particle. That is 1 540 of
the 13 082 A–C pairs — but it is a hypothesis about what they are, not a
demonstration, and they sit *inside* the X17 region. So they are in the
spectrum, the ≥ 170° band is shaded copper, and the cut version is drawn beside
them as a dashed ghost. `--b2b drop` swaps which of the two is solid.

**Nothing is subtracted and no model is drawn.** The event-mixed sample in the
same file is a shape and carries no normalisation; the Poisson accidental rate
over-predicts the observed pair count by 2–4× because the trigger correlates the
arms. Both are argued out in `opening_angle.py`'s docstring. Data only is the
honest version of this plot today.

### The caveat to say out loud

**The angle scale is preliminary.** `k` moves run to run by up to 25 % over one
contiguous 48-hour block (runs 128–147), and the arm-A scintillator wall puts
arm A's own scale 33 % out. A real per-detector recalibration is deferred to
October. So the *ordering* of the three classes, and the agreement of the arm
pairs within a class, are robust; an absolute angle is not. Both figures carry
the `PRELIMINARY` badge for exactly this.

## The overhead figure — the fans and the capsule they cross at

```bash
X17_ROOT=D:/x17 ../.venv/Scripts/python.exe make_overhead_figure.py
../.venv/bin/python make_overhead_figure.py --no-clean --no-charge-window
```

`figures/overhead_run145.{png,pdf,csv}` + `.json`. The Athens rebuild of the
MPGD2026 closing slide (`mpgd26/make_run145_pointing.py`,
`run145_overhead_AC`), which is left alone as the record of that talk. What is
different, and why, is in this script's docstring; in short it is the **full
pass, all three sub-runs**, the confirmed sample **`k_arm.coincident_tracks`**
(so the picture and the published crossings are the same tracks), the
**hot-strip** and **charge-window** cuts that sample carries, the
**pair-vertex cleaning** (slope measured, no noisy column) and the **active-area
v** cut, run_145's **own `k`**, and **chamber D** as a third fan.

| arm | drawn | crossing on exactly these tracks | 33-run campaign |
|---|---:|---:|---:|
| A | 4 344 | X = −7.7 ± 0.3 mm | X = −9.3 ± 0.3 (A and C) |
| C | 3 175 | X = −8.9 ± 0.4 mm | |
| D | 2 342 | Z = −3.8 ± 0.4 mm | Z = −3.5 ± 0.8 |

**D is worth drawing and B is not, for one reason.** The band crossing is
scale-free, so it survives the open `k` question — but *drawing a fan* needs an
angle per track, and B has no certified `k` (no field-shaping rings, no
time-to-depth ladder), so its stage-3 directions are null. B is drawn hatched
with no fan, and is off the right-hand panels entirely — its one available
number is not worth the space. It is still measured and lands in the JSON: on
this cleaned selection B gives Z = +2.9 ± 3.0 mm (415 tracks) against D's
−3.8 ± 0.4, uncleaned −3.8 ± 2.0, and campaign-wide its run-to-run spread is
1.63 mm against D's 0.83 and A's 0.30. That instability, on the one observable
B has, is the statement.

The campaign numbers set **where the bore is shaded** and are no longer printed
in the panels; each panel quotes only what this run's own tracks say.

**Every piece of hardware in both views is drawn to scale** (2026-09-14), from
one helper (`_local_poly`) that uses the same transform as the tracks. A chamber
is its 30.1 mm drift gap, on the target side of the strip plane, across the full
398.6 mm of strips. In the side view the measured passivated ends
(`common.mx17_active_area.TRUE_ACTIVE_BY_DET`, ~18–20 mm each, confirmed on
run_79 beam data) are shaded. The SiPM wall is its 3 mm of active scintillator
at 25 mm pitch. The plastics are 200 × 300 × 20 mm — the sim's 2026-07-20 "20
mm, not 25" correction, centred on the surveyed position; the ±2.5 mm this
leaves open is invisible at this scale. The side view is tall enough for the
full 500 mm wall. Only B's scintillators are left out, on purpose.

Two things to know when reading the pictures. **At true thickness the wall's
four read-out groups hardly show:** each group's outline is as thick as the
3 mm bar. **In the side view, some tracks run past the plastic vertically.**
That is real, not a drawing error: the coincidence only ever tested u (§6.5 of
`HANDOFF_CAPSULE_Y.md`).

**Both scintillator layers are drawn from the DAQ survey, and the tracks are
extrapolated out through them.** The SiPM trigger wall sits 97.4 mm past the
strips — 16 instrumented bars of 25 × 500 mm, **centred on the structure**, so
it carries the pinwheel term the plastics do not — and the plastic bars 189–193
mm past them. Positions come from run_145's own `run_config.json`, which
places every scintillator in the **global** frame —
not from `geometry.py`'s constants, whose active-PVT mid-plane sits 2.5 mm
short of the surveyed bar centre (the sim carries a 20 mm bar, the config says
25 mm). `plastics()` projects each surveyed centre onto the arm's own axes and
**asserts the two conventions** that would otherwise rot silently: the two bars
are centred on the MM (midpoint 0 in the plane's coordinate, with the
beam-axis foot a pinwheel away), and they sit ±101.72 mm from it. Surveyed
depths past the strips: **A 190.6, C 188.6, D 192.6 mm**.

Nothing is fitted to the bars — every track here was *selected* by pointing at
the wall segment and the bar that fired, so the outward half of each line is
the same measured line continued. **Why some lines still overshoot a bar edge:
the coincidence tested the raw tan, and the figure draws `k · tan`.** At the
raw scale 99.8–99.96 % of the drawn tracks land on a bar (which also confirms
this module reproduces the selection's own geometry); at the applied `k` that
falls to **A 96.3 %, C 94.1 %, D 94.9 %**. The 4–6 % that spill past an edge
are the angle scale at a 190 mm lever, not a geometry error. The gap between
each arm's two bars is the ~7 mm shadow the in-situ maps measure.

**Two things in the picture that are not the source.** The dark red marks on a
plane are dead readout, found from the occupancy, not transcribed. The wedge
down the middle of each fan is the `|tan| < 0.08` tracks, where `wft` says the
drift timing carries no slope information — they are recorded and not drawn,
because drawing one is drawing an angle nobody measured.

**The caveat to say out loud:** the fans' *focus* is drawn with run_145's own
`k`, which is provisional, sits at the peak of the runs 128–147 excursion, and
which arm A's scintillator wall says is 33 % out. The crossings quoted on the
figure are scale-free and do not depend on any of that.

## The side view — where the source sits along the beam

```bash
X17_ROOT=D:/x17 ../.venv/Scripts/python.exe make_overhead_figure.py --projection y
#                                                                  --projection both
```

`figures/sideview_run145.{png,pdf,csv}` + `.json`, from the same builder and
the same confirmed sample as the overhead figure, so the two cannot describe
different populations. The right-hand panel carries **all three chambers on one
axis**, unit-normalised — the curves land on top of each other, and that
agreement is the result. Two things are added on top of the overhead cleaning,
both `y_image`'s "y-clean" tier: the **y** slope must be measured
(|tan y| ≥ 0.08) and the track must not sit in a noisy **y** column. That costs
A 14 %, C 22 %, D 32 % of the drawn sample.

| arm | drawn | y band crossing (scale-free) | `y_image`, campaign |
|---|---:|---:|---:|
| A | 3 726 | **+30.7 ± 0.8 mm** | +30.4 |
| C | 2 481 | **+32.6 ± 2.1 mm** | +31.9 |
| D | 1 593 | **+34.1 ± 2.2 mm** | +31.9 |

**The result is the offset.** The nominal gas centroid is **+0.8 mm** and all
three chambers independently put the source **~30 mm above it**, agreeing with
each other to a few mm and with yesterday's campaign-wide `y_image` numbers.
Like the x crossing, this is the zero of the band and is **scale-free**, which
matters more in y than in x because the y angle scale is not calibrated at all.

**Why there is no sharp focus to look at.** The ³He gas is 80 mm long along the
beam (polycone −29.5 → +50.7 mm), so there is no point source in y; the fans
narrow from 340 mm of plane to that column and no further. The per-track y blur
is ~27 mm, comparable to the source itself.

**The capsule is drawn at its measured height** (2026-09-14, after
`HANDOFF_CAPSULE_Y.md` was run). The CAD polycone is shifted along the beam so its
gas centroid sits at the inverse-variance mean of the three chambers' band
crossings on this run: **+31.3 ± 0.7 mm**, statistical only, i.e. a shift of
+30.5 mm on the CAD's +0.8. The same shift is applied to the capsule outline,
the horizontal gas band and the shaded column on the right. It is recorded as
`capsule_y_calibration` in `sideview_run145.json`. The handoff's campaign value,
**+33 ± 3 mm** (the ±3 is A–C and an eff(v) tilt), agrees with it. That is the
number to use anywhere a systematic matters.

### The capsule sits ~30 mm up the beam, and the number is not final — 2026-09-14

**Where the current y comes from.** Nowhere measured. `run_config.json` has no
target entry at all — only `target_type: 3He` — so the capsule's height in our
frame is `geometry.HE3_GAS_Y/R`, the STEP-derived polycone ported from the
Geant4 stack (`MX17_Full_Geant/scripts/plot_geometry.py`, 2026-07-15), placed
with its axis on +Y. The chambers define the frame and are aligned to the
millimetre; the capsule's placement in it is a centimetre-scale guess nobody
surveyed. This figure measures the thing that was never measured — it is not
contradicting something that was.

**Why the crossing is not yet the number to write down.** It is an ensemble
estimator with the **v acceptance left out**: the chambers are 340 mm tall, the
plastics 300 and the wall 500, the trigger demanded both, and a robust line
through an asymmetrically illuminated band pulls toward the populated side —
which is exactly what `source_imaging.symmetric_crossing` exists to undo in u,
with no counterpart in v. The source shape is a second systematic: the pairs
are expected from the capsule's **aluminium**, and at y ≈ +30 the polycone has
tapered to R ≈ 4.8 mm, the neck, where the aluminium is thickest.

Folding the acceptance in is a forward fit, and it is now scoped:

> **`sept26_prelim_analysis/HANDOFF_CAPSULE_Y.md`** — the ray-tracing test that
> settles it. `plastic_acceptance.py` already ray-traces the capsule through
> the bars **and already computes the v intersection**; what it needs is the
> polycone instead of its y-centred stadium, a free source offset, the SiPM
> wall imposed alongside the plastics, and a v projection to fit against.

Worth noting what *has* improved: on uncleaned `target_y_mm` PLAN.md recorded
A +16.2, C +14.3, **D −33.5**; with the y-clean + confirmed selection the three
agree at +30.7 / +32.6 / +34.1.

**Three caveats for this figure specifically.**

- **The y angle scale is bracketed, not measured.** The band slope on this
  sample comes out 0.72 (A), 0.39 (C), 0.43 (D) where a point source demands
  1.0, and `y_image`'s focus scan comes out *above* 1 — the two disagree in the
  directions their biases predict, and neither is adopted. Everything drawn is
  at the scale as reconstructed.
- **The plastic bars were never tested in v.** `pointing_coincidence` checks
  the u coordinate only, so tracks are free to leave the bar vertically in this
  view, and some do. `det_a_scint` is the module that does check v.
- **C's band is the most diluted** (scale 0.39, crossing error ±2.1 mm), so the
  A–C agreement here is weaker than the same comparison in x.

## Slide 40 — what the thermal measurement can see

```bash
../.venv/bin/python make_thermal_sim_figures.py            # both, + thermal_sim.json
../.venv/bin/python make_thermal_sim_figures.py --only branching
```

Needs no data disk: ENDF, the staged line lists, and numbers quoted in the
script with their sources.

| figure | what it is |
|---|---|
| `thermal_branching` | **why the gas self-shields.** (a) ³He cross sections, which all rise as 1/v. (b) σ(n,γ)/σ(n,p), **1.0×10⁻⁸ at 25 meV against 8.8×10⁻⁵ at 1 MeV**. (c) ³He(n,γ) per energy decade over **30 days at slide 39's flux** (1.93×10⁴ pulses/day), thin-target against self-shielded. For 0.01–1 eV: **3.5×10⁶ → 2.9×10⁴ ⁴He\***, ≈ 130 IPC pairs made. That is ×120 analytic; the Geant4 thermal note gives ×50–100. The 30-day bins are also in `thermal_branching_30d.csv`. No footnote, on purpose — this is the audience-facing version. |
| `thermal_pair_sources` | **interim, to be replaced** by the five-step funnel in [`HANDOFF_THERMAL_ACCOUNTING.md`](HANDOFF_THERMAL_ACCOUNTING.md), built from Geant4 truth on the Linux box. (a) expected per neutron entering the capsule: wall pairs outnumber ³He pairs **6×10⁴–8×10⁵ : 1**; (b) Geant4 trigger provenance, **96 % of trigger legs aluminium**; (c) measured true-coincidence fraction of our two-arm pairs, 29 % [17, 40] overall. |

**The funnel that replaces `thermal_pair_sources`**
(`HANDOFF_THERMAL_ACCOUNTING.md`, for the Linux box). One figure and one
diagram per step:

- **F1** ³He(n,p), which we cannot see, against (n,γ).
- **F2** capture γ that touch no detector against those that make a charged particle.
- **F3** pair production in or near the capsule against charged particles made elsewhere.
- **F4** the capsule pairs by source reaction, with internal pair creation added analytically because Geant4 does not simulate it.
- **F5** separately, what fires a trigger leg.

The contract is `data/thermal_accounting/accounting.json`.

**Why the ratio and not the cross section.** Once the optical depth is ~150
the cell absorbs every neutron whatever σ is, so a radiative capture happens
with probability σ(n,γ)/σ(abs) per neutron — 1.0×10⁻⁸ — and adding gas buys
nothing. The thin-target N·σ(n,γ) of slide 39's table overstates that by the
optical depth; at MeV the cell is thin and the two agree.

**What is quoted rather than computed.** The Geant4 numbers (trigger
provenance, per-day yields, the ×50–100) come from `ntof_run_report` §6, whose
sources (`MX17_Full_Geant/…/trigger_provenance`, `al_pair_background/VERDICT.md`,
`docs/report/thermal_note.pdf`) are on the Linux box and were not re-read. The
timing fractions are `sept26_prelim_analysis/HANDOFF_ACCIDENTAL_TIMING.md` §0,
run_145 only, 76 pairs.

**The caveats to say out loud.** Panel (a) counts internal conversion only;
external conversion of the 7.7 MeV γ in the wall adds to the wall. The ³He pair
yield is `ipc_born` M1 + E0 (4.7×10⁻³ per radiative capture), not 2.1×10⁻³;
Geant4's 0.44 /day IPC is the generator's own number. And a prompt pair in (c)
is **not identified** as aluminium — the capsule shape fits better than the gas,
but nothing in this setup measures total pair energy.

## The pair-quality figures

```bash
# once: build the table (a few minutes, 8 workers)
../.venv/bin/python -m sept26_prelim_analysis.pair_qa --jobs 8 --verify
# then: the figures
../.venv/bin/python make_pair_qa_figures.py
../.venv/bin/python make_pair_qa_figures.py --only chi2dof_worst
```

Nine figures, `qa_<metric>.png`, contact sheet `figures/qa_index.html`. Each is
three panels on the same split as `pairings` — three intra, two perpendicular,
one opposing — so a feature in a spectrum can be traced to the sample that made
it. **Every panel carries each arm pair's own event-mixed null as a dashed
curve in the same colour.** That is the point of the set: where the null lies
under the data, a cut on that variable removes signal and background alike and
buys nothing; where they separate, there is something to cut on.

| metric | what it is |
|---|---|
| `chi2dof_worst` | worst χ²/dof of the four track fits in the pair |
| `n_strips_min` | fewest strips on any view of either leg |
| `q_total_min` | total charge of the quieter leg |
| `dca_worst` | worse leg's closest approach to the beam axis |
| `sep_mm` | how close the two legs come to *each other* |
| `v_r` | reconstructed vertex radius from the beam axis |
| `dt_track_ns` | difference of the two fitted track times |
| `delta_t` | scintillator Δt between the two arms |
| `t_flash_ms` | neutron arrival time since the flash |

`pair_qa.py` writes the table to `<out>/pair_qa/`: one row per pair with both
legs' quality columns, plus `pair_qa_summary.csv`, which is the percentile
table a cut would be chosen from.

### The table is the same sample as the angle figures, and that is checked

`--verify` compares the per (arm1, arm2, mixed) counts against
`angle_campaign/pairs.parquet` and prints the difference. It is currently zero
on all twenty rows. A QA table that is only *nearly* the published sample would
be worse than none, because every distribution drawn from it would describe a
population nobody else has.

### Three things to know before reading them

**Only `delta_t` is a coincidence.** `dt_track_ns` is the difference of the two
legs' fitted track times, and both legs of a real pair share one trigger and
one sampling window while a leg's `t0` moves with its depth in the 30 mm gap —
so it is dominated by drift depth, not by simultaneity. `delta_t` is the
scintillator measurement and it exists only for the ~5 300 pairs where both
arms are tagged.

**`delta_t` does not exist for intra pairs.** One chamber is one arm is one
time. `tight_coincidence.py` writes no intra rows at all, and the intra panel
of that figure says so rather than showing an empty axis.

**There is no event-mixed null for `delta_t`** — `tight_coincidence.py` writes
real pairs only — so that figure's key does not claim one.

## The in-situ performance figures

```bash
../.venv/bin/python insitu_maps.py --jobs 6        # the analysis, ~20 min
../.venv/bin/python insitu_maps.py --derive-only   # re-pool, seconds
../.venv/bin/python make_insitu_figures.py         # the five figures
../.venv/bin/python make_insitu_report.py          # figures/report.html
```

`insitu_maps.py` reads the stage-3 track tables and the n_TOF slim and writes
`<out>/athens_insitu/`; the two `make_*` scripts only draw and only report.
`--derive-only` rebuilds every pooled table from the per-run products already on
disk, so changing a mask, a window or a summary costs seconds rather than
another pass over the campaign.

| figure | what it is |
|---|---|
| `hitmap_triggered` | **the trigger-biased maps.** A, C and D on scintillator-matched tracks, with the wall groups and the plastic bars projected onto the chamber through the lever arm. |
| `hitmap_unbiased` | **the result.** All four chambers read out on another arm's trigger. B is on this figure and on no other. |
| `hitmap_profiles` | both samples along u and v, normalised to their own means — the proof that the trigger footprint is gone. |
| `shadow_test` | the declared test, answered. |
| `detector_b` | B's map, its cluster shape against the other three, and the HV monitor that identified the missing ring chain. |

### The idea, in one paragraph

The production trigger is a wall **and** plastic coincidence in **one** arm,
OR'd over the four, and **97 % of the read-outs this analysis can tag have
exactly one arm lit**. So each chamber spends most of its life being read out
on somebody else's trigger — a **bystander** — and in those events its
occupancy carries no trace of its own scintillator acceptance. That is the
unbiased map. It needs no track removal, because the bystander sample never
contained the triggering particle, and it needs **no angle**, which is why it
is the one map chamber B can make.

### The test that decides it, declared before it was run

The two plastic bars leave a gap that shadows u ≈ +7 mm on every chamber. If
that dip is the trigger's it must vanish when another arm triggers — *except*
where the chamber is blind underneath it. It does:

| chamber | matched | bystander | dead channels there |
|---|---:|---:|---|
| A | 0.526 | **−0.193** | none |
| B | — | −0.212 | none |
| C | 0.476 | 0.504 | ~10 |
| D | 0.862 | 0.822 | ~130 |

A's dip inverts because the beam illuminates the centre more, which is what the
trigger had been carving a hole in.

The removal method the bystander construction replaces is built anyway, as
`leftover`, and it agrees **where the dip is real** — C 0.517 against 0.504
(0.8σ), D 0.845 against 0.822 (1.3σ). On A, where the shadow is genuinely
absent, it does not: −0.042 against −0.193, 4.9σ, with the removal method left
closer to a shadow. That is the expected direction and it argues *for* the
bystander construction: a leftover track shares its event with the particle
that fired the trigger, so it is still weighted by the trigger's acceptance
through whatever correlated the two. A bystander has no such partner. It is
also 20× smaller, so it is the cross-check and not the result.

### Three things to say out loud

**These are occupancy maps, not efficiency maps.** No independent denominator,
so a cold cell is "less illuminated or less efficient" and the analysis does not
say which. What the construction removes is the chamber's own scintillator
acceptance; that is all it claims to remove.

**The bystander sample is not purity-selected.** Nothing confirms a track in it
— that is the price of not using the chamber's own scintillators — so junk
accumulates there *by construction*: a cluster made by a noisy strip points at
no scintillator, so it can never be matched. Hence the hot-cell mask, and hence
the cluster-shape table being computed on a different sample.

**Chamber D's unbiased map does not work**, and no mask setting rescues it — its
99th-percentile-to-median cell ratio is 207 at the loosest setting scanned and
68 at the tightest, where the mask has already taken 45 % of its sample. A
quarter of D's x plane is dead. That is a statement about D, not about the
method.

### Style

`mpgd26/plotstyle.py`, imported rather than copied — the Athens deck inherits
the MPGD2026 palette so a chamber does not change colour between the two talks.
Chamber hues are the Okabe-Ito subset (A `#0072B2`, B `#D55E00`, C `#009E73`,
D `#CC79A7`); within a class a curve is identified by the arm that is *not*
shared, so a hue always means a chamber and never a class.
