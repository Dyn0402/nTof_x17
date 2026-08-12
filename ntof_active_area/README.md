# `ntof_active_area/` — detector active areas from n_TOF beam data

**Answer: the MX17 chambers are 39.9 × 36.0 cm of active area, not the
38 × 34 cm the Geant simulations assume — 11 % more area. The scintillator
sizes should be left alone; chamber pointing is a factor ~30 too blurry to
check them.**

Three write-ups, same numbers, different readers:

| | for | |
|---|---|---|
| [`report.html`](report.html) | the DAQ Analysis tab | verdict, tables, figures, caveats |
| [`ACTIVE_AREA_2026-08-11.md`](ACTIVE_AREA_2026-08-11.md) | whoever picks this up next | method, the three estimator guards, exactly what was changed in the Geant repos, open items |
| <https://dylan-neff.web.cern.ch/notes/mx17-active-area.html> | anyone, on a phone, offline | the reasoning spelled out, figures embedded |

## Why this was worth doing

`SimConfig.hh` in both `~/CLionProjects/MX17_Geant` and `MX17_Full_Geant`
carries

```c++
double mm_size_u_cm    = 38.0;   // MM active area: u [cm]
double mm_size_v_cm    = 34.0;   // MM active area: v (along beam) [cm]
```

with no provenance comment, unlike every other dimension in that file. They were
an estimate. Every other detector's size there *does* carry a survey date
(SiPM wall 2026-07-15/17, plastics 2026-07-17/20, LS from the STEP file), and
those are millimetre-accurate measurements that this analysis cannot improve on.

## What is measured

A **paired track** — exactly one particle-like strip cluster on each plane of the
same chamber in the same event, with the two planes' charges balanced, since an
MX17 avalanche splits ~50/50 between them. That balance requirement is the whole
trick: it removes the uncorrelated per-plane noise that dominates a raw
occupancy precisely at the board edges. On chambers B, C and D the raw `y`-plane
occupancy beyond 380 mm — outside the chamber — is **2–15× higher** than in the
chamber's interior (B 3.6×, C 2.1×, D 15×). Paired tracks go to zero at 379 on
all four.

The beam illuminates each chamber smoothly and well past its edges, so a step in
the profile is geometry and a gradient is illumination. Edges are found by
walking **outward from the interior**, comparing each strip to the strips just
inside it.

No hit times are used, so this stays within `RECONSTRUCTION_BASIS.md`: strip
identity and charge are detection/QA quantities, and an occupancy edge is
exactly that.

Coordinates: `x` plane = **u** (tangential), `y` plane = **v** (along beam),
as in `ntof_tracking/run79_merge_prelim.track_frame`.

## Results

| | measured | was |
|---|---|---|
| u (tangential) | **39.9 cm** — full metallised strip region, no passivation | 38.0 |
| v (along beam) | **36.0 cm** — 359.9 ± 1.8 mm, a ~19 mm dead band at each end | 34.0 |
| centring | midpoint 199.1 mm vs strip-plane centre 199.3 mm — unchanged | — |

Chambers A, B and C each give a determined v edge at both ends; all agree with
the June cosmic-bench telescope measurement in `common/mx17_active_area.py`
(taken with an external reference, so the better of the two) to 1–2 mm. This is
the independent confirmation *in the n_TOF configuration*.

Two run-79 readout defects are separated out rather than folded in:

* **chamber A X-plane connector 8 is dead** (strips 448–511, u = 349–399 mm).
  It was alive on 18 July in run_55, so this is a campaign fault, not the
  chamber.
* **chamber D's u plane is mostly dark** in this run; its edges are reported as
  undetermined rather than fitted. Chamber C has a real interior dead stripe
  near u = 190 mm.

### Scintillators

The merged n_TOF ↔ DREAM arm-A sample confirms *placement* but cannot measure
*size*: the plastic L/R boundary lands at −6.8 ± 5.3 mm where the geometry puts
it at 0, and the four wall segments map onto the chamber in the right order
(r = +0.97). But the pointing blur is σ ≈ 47 mm at the plastic plane and the
tagged plateau stands only ~1.5–2.5× above the accidental-tag pedestal, so every
outer-dimension fit is unconstrained and lands on both sides of the survey.
Keep the surveyed numbers.

## Running it

```bash
.venv/bin/python -m ntof_active_area.run_all      # everything, ~2 min
```

or piecewise:

```bash
.venv/bin/python -m ntof_active_area.mm_edges           # -> results_mm.json, profiles.npz
.venv/bin/python -m ntof_active_area.scint_acceptance   # -> results_scint.json
.venv/bin/python -m ntof_active_area.figures_mm
.venv/bin/python -m ntof_active_area.figures_scint
.venv/bin/python -m ntof_active_area.figures_note
.venv/bin/python -m ntof_active_area.make_report        # -> report.html
.venv/bin/python -m ntof_active_area.make_note          # -> note_active_area.html
```

To republish the site note after re-running:

```bash
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    ntof_active_area/note_active_area.html --slug mx17-active-area --force \
    --tags "X17, nTOF, micromegas, simulation, Geant4" --deploy
```

## Files

| | |
|---|---|
| `clusters.py` | vectorised strip clustering on `combined_hits` (validated against a loop reference, identical to float rounding) |
| `mm_edges.py` | the chamber measurement: paired tracks, span profiles, live edges, dead bands, connector health |
| `scint_acceptance.py` | arm-A scintillator acceptance from the merge, with the blur and the accidental pedestal floated |
| `figures_mm.py`, `figures_scint.py` | analysis figures |
| `figures_note.py` | the three explanatory figures for the published note |
| `make_report.py`, `make_note.py` | build the two HTML write-ups from the JSONs — rerun after remeasuring |
| `results_mm.json`, `results_scint.json` | the committed results — every edge, every fit, and the per-strip span profiles |
| `profiles.npz` | **not committed** (`.gitignore`): a ~2.4 MB cache of the profiles and the paired-track table, rebuilt in ~80 s by `mm_edges`. The figure scripts need it; the results do not. |
| `figures/*.png` | **not committed** — the repo ignores `*.png` by policy (they were purged from history on 2026-08-04). Regenerate with `figures_mm` / `figures_scint` / `figures_note`. |
| `note_active_area.html` | committed *because* of that: with the PNGs untracked and `profiles.npz` needing the raw data to rebuild, this 1.2 MB self-contained page is the only copy of the figures that survives in git. Same precedent as `ntof_processing/mm_flash/report_standalone.html`. |

## Inputs

* `run_79/stat090_0000` and `stat090_0001`, 27 `combined_hits` files, 215 481
  DREAM events — the only two sub-runs mirrored locally.
* `merged_prelim.parquet` from `RUN79_PRELIM_2026-07-30` (arm A, n_TOF 224572,
  tags 000–002) for the scintillator section.
