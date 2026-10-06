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
PYTHONPATH=. .venv/bin/python -m ntof_scint_stack.checks              # trigger edges, time-cut scan
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
- **Coverage is the whole full pass** (34 runs, all but the pre-access run_79/81).
  The thin part is the *unbiased* tag sample (another arm triggered, > 10 ms):
  a few thousand tracks per arm on A–C, because 97.7 % of triggers fire exactly
  one arm (`checks.trigger`). So the **main maps are the `*_full` layers on
  `all_late`** — every late trigger, whole face, boundary-tolerant probes
  (`predict`'s `_tol` columns), 25 mm cells, ~30× the statistics, level biased
  by the arm's own trigger. The unbiased maps are a separate slide, rebinned
  to 100 mm; the grids are aligned to the channel edges so that this works.
- **The emulated trigger thresholds are measured** (`extract.WALL_THR/PLAS_THR`,
  2026-10-06): the 0.5 % low edge of each arm's own triggers, flat to ~1 mV
  over the campaign. The run_79 read-back had the plastic 3–6 mV high.
- **`LATE_MS` stays at 10 ms** (`checks.late_scan`): earlier bins hold < 15 %
  of the late statistics and their accidental-corrected efficiency has not
  converged. Loosening it buys nothing.
- Arm D's unbiased sample is ~15× the others' (a third of its late tracks sit
  on events another arm triggered). Not yet understood.
- `tan_err` in the track tables is a constant placeholder. Don't split on it.
