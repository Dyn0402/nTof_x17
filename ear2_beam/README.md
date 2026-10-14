# ear2_beam — the n_TOF EAR2 neutron beam at our station

Started 2026-10-14. Report: `/media/dylan/data/x17/ear2_beam/report.html` (`make_report.py`).

## Why this exists: `Lambda2D` is not a beam profile

Until 2026-10-14 every beam footprint in our code took the beam's radius from `Lambda2D` in
`MX17_Full_Geant/data/lamda2DvsEn_EAR2.root`. That covers the Geant4 neutron gun, and therefore
every `full_sim/neutrons*` campaign, plus `sept26_prelim_analysis/capsule_pose.py`. **`Lambda2D`
is the n_TOF resolution function.** Its y axis is titled "lambda (cm)". It is the optical transport code's
`HIST_LAMBDA` output, `lambda = (t·v − L)·100` (`/eos/experiment/ntof/simul/transport/transport.cc`).
It sits next to its own SAMMY user-defined-resolution export (`…root_sammyudr.txt`) in
`/afs/cern.ch/work/m/msabateg/public/n_TOF-files/`. Read as a radius, it put half the beam
inside 2.5 mm.

## The beam, from n_TOF's own inputs

`optical_transport.py` reimplements the transport code's method in numpy:
- **Source:** each neutron of n_TOF's phase-4 FLUKA dump toward EAR2 is a point source.
  The dump is `/eos/experiment/ntof/simul/phase4/data_fluka_4.5.1`, October 2025. 400 of its
  3003 runs give 5.0 M neutrons.
- **Rays:** rays to a grid at height L are kept if they pass every aperture of n_TOF's
  collimation file (`simul/transport/input/*.inp`).
- **Variants:**
  - `july2021`: the post-upgrade line, small second collimator (21.8 mm exit).
  - `survey2017`: the first collimator where the 2017 survey found it, (−1.4, −1.1) cm.
  - `miss2mm`: the second-collimator exit 2 mm off.
  - `fission`: the big second collimator (60 mm exit).

**Validation.** At 19.95 m, thermal, small collimator, centred disks of 1.5 / 2 / 3 / 4 cm:

| | 1.5 cm | 2 cm | 3 cm | 4 cm |
|---|---|---|---|---|
| this transport | 0.212 | 0.351 | 0.645 | 0.876 |
| PPACMon (measured) | 0.215 | 0.355 | 0.655 | 0.875 |

The PPACMon values are read off Fig. 14 of Pavón-Rodríguez et al., EPJ A 61 (2025),
arXiv:2505.00042.

**Results (small collimator, thermal).**
- **No energy dependence below 1 keV.** The profile is set by the collimator geometry.
- **No long tail.** The beam ends sharply at the collimator's geometric edge.
- **Width with height:**

  | flight path | r50 (half the beam inside) | FWHM | on a 20 mm bore | r99 |
  |---|---|---|---|---|
  | 19.0 m | 8.8 mm | 19 mm | 0.61 | 17 mm |
  | 19.5 m | 10.5 mm | 22 mm | 0.46 | 22 mm |
  | 19.95 m | 12.5 mm | 27 mm | 0.35 | 25 mm |
  | 20.5 m | 15.2 mm | 34 mm | 0.24 | 30 mm |

  For comparison, the old model gave r50 = 2.5 mm and 0.95 on a 20 mm bore.
- **Divergence:** the beam is a cone. A ray's angle is ~3.5–4 mrad per cm of distance from
  the axis, and the cone's apex is ~2.5 m below the station. On top of that there is a local
  spread of 2–3 mrad.
- **Centring:**
  - The FLUKA source alone puts the centroid ~1 mm off.
  - The known misalignments move it 1–3 mm.
  - PPACMon puts the centre 1.3 mm (x) and < 0.1 mm (y) from the pipe axis.
  - A 10 mm beam offset would need the defining aperture 1 cm off.
- **The big collimator** gives ~54 mm FWHM, and a 20 mm bore then sees ~0.09 of the beam.

**The flux.** Our code used the Phase-3 (pre-2021 target) evaluation. The Phase-4 one
(`data/Final_evaluated_flux_Phase4_EAR2.root`, from `2021_Commissioning/flux/EAR2`) is higher:
- ×1.19 at 14–124 meV;
- ×1.32 for everything below 1 keV;
- ×1.41 at 1–10 eV.

## Not known

- **Which second collimator we had.** The small one is assumed: it is n_TOF's capture
  standard and the one the PPACMon profile was measured with. Our raw headers are gone from EOS.
  Ask the n_TOF logbook or the coordinators.
- **The capsule's flight path.** Our code uses 19.5 m and n_TOF's capture reference is 19.95 m.
  Between the two, the bore fraction goes from 0.46 to 0.35.
- **How n_TOF's (x, z) map onto our chamber frame.**

## Consumers of the old beam (audit 2026-10-14)

| what | uses | status |
|---|---|---|
| Geant4 gun (`X17PrimaryGenerator::GenerateNeutron`) | Lambda2D as radius, Ph3 flux | fixed on branch `ear2-beam` (2D table + cone, Ph4 flux, refuses Lambda2D) |
| all `full_sim/neutrons*` campaigns, incl. `neutrons_thermal_trig_2cm_nose` (10⁹) | old gun | to rerun |
| `sept26_prelim_analysis/capsule_pose.py` | Lambda2D as radius | switched to `ear2_beam.beam` + Ph4 flux; old outputs in `capsule_pose/`, new in `capsule_pose_ear2/` |
| `same_chamber_g4.py`, `same_chamber_compare.py`, `same_chamber_scatter.py` (the 1 : 8) | `neutrons_thermal_trig_2cm_nose` | needs the rerun |
| `leg_penetration.py`, `make_status_oct15_deck.py` | same Geant4 sample | numbers are narrow-beam |
| `ntof_cosmics/g4_angle`, `g4_digi` (capsule-focus angle scale, capsule estimators) | same Geant4 sample (track library) | source distribution changes; check |
| `ntof_athens_26/make_thermal_funnel_figures.py`, Geant4 `thermal_accounting.py`, `docs/al_gamma_yield_check` | same | per-neutron budgets change |
| `mpgd26/make_x17_rate.py`, `data/x17_rate_3He.txt` (Dec 2025 rate table) | the whole Phase-3 beam through a 4 cm cell (BIF 1) | as built (20 mm bore) the ³He sees 0.46 (19.5 m) / 0.35 (19.95 m) of the beam; with the Phase-4 flux (×1.19 thermal) the IPC/X17 rate is ×0.55 / ×0.42 of the table |
