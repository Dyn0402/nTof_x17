# ntof_scint_stack — the scintillator stack, mapped by the MM tracks

Every gated MM track of the n_TOF campaign full pass, walked outward through the
SiPM wall, the plastic and the liquid behind its chamber: efficiency and response
maps of all twelve counters, the wall's position along its bars, and whether to
demand both wall ends. Moved here from `sept26_prelim_analysis/scint_stack*.py`
on 2026-10-06.

- Slide note: <https://dylan-neff.web.cern.ch/notes/scint-stack.html>
- Long report: `/media/dylan/data/x17/scint_stack/report.html`
- Outputs: `/media/dylan/data/x17/scint_stack/` (`paths.spell('scint')`, override
  with `X17_SCINT_OUT`). `ana_v1_freescale/`, `figures_v1_freescale/` and
  `report_v1_freescale.html` there are the 2 October pass, kept for comparison.

```bash
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.extract --jobs 8   # ~3 min, per-track tables
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.ana --jobs 4       # ~9 min
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.make_figures
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.make_report
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.make_deck          # the slide note
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    /media/dylan/data/x17/scint_stack/deck/scint-stack.html --slug scint-stack --force --deploy
```

## The track calibration (2026-10-06)

Slopes are the **single-track imaging calibration**: `m = k · tan_raw`, each
run's own `k` from `<out>/kcal` (the stage-3 `tanx` *is* `k · tan_raw`; runs
with no `k` for an arm take its campaign median, flagged `k_fill`). The crossing
at a layer a lever `L` past the strip plane is

    u_layer = u + L (α a + λ m) − δ,     a = (u − u_capsule) / w_strip

with the capsule at the imaged position (`ana.CAPSULE_XYZ`). α, λ and the wall
alignment δ are fitted on which wall group fires, at the three internal group
boundaries held at the survey (`ana.fit_pointing`). Result: α ≈ 0, λ ≈ 0.59–0.76.
The wall then sits within 1–6 mm of the survey, and the edges are sharper
than with the bare `k` or with `k` shrunk toward the capsule.

**Gotchas**
- `tanx = k · tan_raw`, not `tan_raw / k`. The 2 October pass compared its
  fitted scale with 1/k; that comparison (and "arm A is k × 1.10") was wrong.
- λ < 1 says the scintillators prefer a *shallower* slope than `k`. That is a
  predictor calibration, not an angle-scale measurement. A boundary fit
  regresses on a noisy slope. The cosmic review of 2026-10-06 points the same
  way, with a smaller effect.
- The unbiased tag sample (another arm triggered, > 10 ms) is only a few
  thousand tracks per arm. The deck rebins its maps to 100 mm, and the grids are
  aligned to the channel edges (`ana.efficiencies`, `ana.gains`) so that this works.
- `tan_err` in the track tables is a constant placeholder. Don't split on it.
