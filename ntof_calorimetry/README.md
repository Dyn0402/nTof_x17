# ntof_calorimetry — scintillator energy scale, liquid salvage, MM dE/dx

The plan, and the results so far, are in `PLAN.md`. Outputs go to
`/media/dylan/data/x17/calorimetry/` (`paths.spell('calo')`, override with
`X17_CALO_OUT`); the report is `report.html` there.

```bash
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.mip_sample beam     # ~1 min
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.mip_sample cosmic   # ~40 min, reads n_TOF trees
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.scint_ecal          # C1, ~5 min
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.liquid_salvage      # C4
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_report
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.wall_plastic        # wall + plastic vs beam e- and Geant4 (needs stack_edep.py output)
PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_wp_deck        # notes/scint-calorimetry
```

**Gotchas**
- The plastic keVee scale of `srccal_energy_calib.json` under-reads by 3-24 % at
  the MIP. For energies above ~2 MeV use `c1/calib_plastic_e.json` (MIP-anchored).
- The liquid keVee scale of the same file is wrong by ×5-7: it was never a
  Compton-edge calibration. Use the MIP scale in `c4/mip_scale.csv`.
- In-beam through-goers are not a MIP sample: most do not penetrate the stack.

**MM energy deposit per track (2026-10-14):** `heed_edep` → `mm_edep_sim` / `mm_edep_data` → `mm_edep_ana` → `mm_edep_combo`. Results and status: `MM_EDEP.md`.

**Track passport (2026-10-15):** after `mm_edep_ana`, run `python -m ntof_calorimetry.track_passport` (~10 min) and
then `python -m ntof_calorimetry.make_passport_report` and `python -m ntof_calorimetry.make_passport_deck` (slide note `track-passport`; delete `passport/deck_cache_beamcal.pkl` after an analysis rerun). This combines every layer per track, and also holds the
beam-sim MM calibration. Output: `<calo>/passport/report.html`. Summary in `MM_EDEP.md`.
The pickles in `<calo>` were written with numpy 2. A numpy-1.26 venv cannot read them; re-save them through
parquet in a numpy-2 env.
