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
```

**Gotchas**
- The plastic keVee scale of `srccal_energy_calib.json` under-reads by 3-24 % at
  the MIP. For energies above ~2 MeV use `c1/calib_plastic_e.json` (MIP-anchored).
- The liquid keVee scale of the same file is wrong by ×5-7: it was never a
  Compton-edge calibration. Use the MIP scale in `c4/mip_scale.csv`.
- In-beam through-goers are not a MIP sample: most do not penetrate the stack.
