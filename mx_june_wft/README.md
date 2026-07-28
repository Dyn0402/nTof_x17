# mx_june_wft — the June cosmic analysis on the waveform-first basis

The parallel chain that replaces the hits-based one, detector by detector. Why
it exists: `../RECONSTRUCTION_BASIS.md`. The reconstruction library it runs on:
`../wft/`.

The old chain (`../mx_june_cosmic_qa/`) still runs and is the comparison
reference. Nothing here overwrites it — all output goes to `.../<det>/wft/`.

## Chain

```bash
# one detector, end to end (bundle -> reco -> alignment -> efficiency -> angles -> maps -> digest)
mx_june_wft/run_chain.sh sat_det3 --jobs 12

# a detector with no calibration yet (det6/det7): fit one first
mx_june_wft/run_chain.sh g_det6_long --calibrate --seed-bundle <det3 bundle>

# the whole fleet, sequential
mx_june_wft/run_fleet.sh
```

| stage | script | output |
|---|---|---|
| calibration bundle | `python -m wft.cli bundle <key>` / `python -m wft.calibrate <key>` | `<det>/wft/calib_bundle/` |
| reconstruction | `python -m wft.cli reco <key> --matched-only` | `<det>/wft/events.parquet` (+ `.meta.json`) |
| alignment | `01_alignment.py` | `<det>/wft/alignment/alignment.json` |
| efficiency | `02_efficiency.py` | `<det>/wft/efficiency/efficiency_breakdown.{json,txt,png}` |
| angles | `03_angles.py` | `<det>/wft/angles/angular_resolution.json` |
| maps | `04_maps.py` | `<det>/wft/maps/` |
| comparison | `digest.py <keys...>` | markdown table vs `rerun_baseline.json` + gate verdict |

## What is different from the hits chain, and what deliberately is not

**Different — everything geometric.** Position at the mesh, angle and drift
depth come from fitting the (strip x sample) waveform picture, so the 20-30 %
ladder compression is gone. Expect `sigma_theta` to improve a lot and `v_drift`
to move (34.3 -> 36.6 on det3) — the velocity change *is* the correction.

**Deliberately the same — detection.** Whether the detector saw the muon is a
property of the analyzer's trigger, not of the fit, so seeding and the
efficiency numerator/denominator still come from the hits tree with the same
significance floor, gap threshold and spark veto. `has_any` and `spark_frac`
should therefore not move at all; if they do, something is wrong with the
seeding, not with the physics.

**Deliberately reference-free.** The M3 track is used to *choose which events*
to reconstruct and to score the result — never to seed a fit. The R&D study
seeded fits at the reference; production cannot, because alignment and
efficiency would then be circular. This costs a little mesh resolution (4 % on
X, 17 % on Y in a controlled test) and nothing in angle.

## Reading the numbers

- **Angle resolution below ~1 deg per event is not a better reconstruction** —
  it is the measured diffusion/charge-granularity floor
  (`WAVEFORM_FIRST_THREADING.md` §12). Treat a sub-floor number as a bug.
- **Planes with |tan| < 0.08 carry no slope information** (`slope_reliable`).
  Any angle average that includes them re-introduces a bias.
- **chi2/dof is large** (median ~110 on X, ~180 on Y for det3) because every
  sample counts against a model that is imperfect at the percent level. It is
  not a goodness-of-fit p-value; `quality_ok` (chi2/dof < 300) is a flag for
  showers and multi-track events, not a filter, and the chain does not apply it.
- **`implied-v spread`** in `03_angles.py` is the honesty check: the median
  `w / tan_ref` must be the same in every angle bin. The hits ladder falls
  56 -> 39 um/ns across the angle range; that spread is the compression
  signature, and a small number here is the evidence it is gone.

## Not ported yet

Time resolution (PLAN_42), sparks, charge balance, fringe field, the drift-HV
v(E) scan and the U50 gap maps still run in `../mx_june_cosmic_qa/`. The v(E)
scan and gap maps already have waveform-first results from the R&D study
(`waveform_first_threading/` scripts 19-21 and 29-35) — they need packaging,
not redoing. Status table: `../RECONSTRUCTION_BASIS.md`.
