# ntof_athens_26 — the Athens talk

The n_TOF conversion of the MPGD2026 deck. `slides/ntof_athens_talk.pptx` is the
deck; the scripts in this directory build figures for it, and the
subdirectories are studies the figures raised.

## What is here

| where | what | entry point |
|---|---|---|
| this directory | the deck's figure builders: opening-angle topology, pair quality, in-situ performance (sections below) | this README |
| `chi2_bimodality/` | why chamber A's worst-view χ²/dof is double-humped; holds `HANDOFF_T0_PRIOR.md` and `HANDOFF_CHANNEL_MASKS.md` | `chi2_bimodality/README.md` |
| `xy_t0/` | what the two planes agree on, and what `dt_xy` is | `xy_t0/README.md` |
| `channel_masks/` | whether the hot/dead channel classification is hardware or occupancy | `channel_masks/README.md` |
| `pair_vertex_imaging/` | the two-track vertex series, 2026-09-12 → 13: why the pair vertex is not a 10 mm image, the blurred image it is in x, where z went (chamber D), a balanced 3D density, the vertex along the beam, and why same-chamber pairs cannot be tested yet — four published notes | `pair_vertex_imaging/README.md` |

The reconstruction handoff that came out of `pair_vertex_imaging` lives with the
other campaign handoffs: `sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md`.

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
