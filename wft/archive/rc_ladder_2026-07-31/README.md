# Archived `wft/` core — the RC-ladder R&D branch, 2026-07-31

**This code is not on the import path and is not what `wft/` runs.** It is kept
as a reference, to be reworked into the shipped model later.

These four files are the `ntof-signed-decoding` branch's version of `wft/`,
as they stood when the branch was merged into `main` on 2026-08-06. They are
the *original R&D* implementation of the RC-ladder sharing kernel, developed
on the bench in late July.

## Why there are two implementations

The same feature was built twice, independently:

- **This copy** — the July bench R&D, measured directly on near-vertical det3
  tracks by `mx_june_wft/bench/rc_line_step3.py`.
- **The shipped `wft/`** — commit `8e52e69` (2026-08-05), *"share_lp into the
  fleet model"*. Its test file describes itself as a **port**, written after
  "the R&D artefacts that were lost with the campaign machine". It is a
  clean-room reimplementation of the same physics.

At merge time the shipped version was kept, because it is the newer line of
work and the one the SPS campaign ran on. Neither is a superset of the other.

## What this copy has that the shipped model does not

| feature | what it does |
|---|---|
| `kTauY` | per-plane lateral RC constant. Measured **410 ns (Y) / 230 ns (X) on det3**, ratio 1.78 — the shipped model uses one `tau_s` for both planes |
| `tau2_fac_y` | the ±2-strip copy arrives at `fac·tau`, default 2.0 (linear). Set 4.0 on Y to model RC *diffusion*, where delay grows quadratically with distance |
| `sigma_sY` | per-plane template smearing |
| `MODEL_FRAC` | fractional model-error term in the chi2 weights (production used 0.03). Settable via `WFT_MODEL_FRAC` so calibration workers and the reco driver agree |
| `sample_weights()` | exposes the per-sample 1/sigma weights, used by the residual audit |
| `_tau_eff` / `_tau2_delay` | the per-plane delay helpers the above are built on |

Kernel shape differs too: this copy convolves the template with a normalised
exponential truncated at 6·tau, then Gaussian-smooths. The shipped model uses a
discrete one-pole IIR over a grid zero-padded to +6 us — which is the more
careful treatment of the RC tail beyond the template grid, and it ships with a
runnable test (`wft/tests/test_share_modes.py`, needs nothing on disk).

The shipped model also has a `SHARE_MODE = 'delay' | 'lp'` switch that keeps the
legacy kernel reachable. This copy has no switch; `share_lp` is a hyper key.

## Scripts that were written against this copy

These merged into `main` alongside it and still reference its symbols. Against
the shipped `wft/` they do **not** all fail loudly:

- `mx_june_wft/bench/residual_audit.py` — **hard failure**: calls
  `wm.sample_weights(...)`, which the shipped model does not define.
- `mx_june_wft/bench/run_bench.py` — **silently ineffective**: its variant
  table sets `MODEL_FRAC` and `tau2_fac_y`, which the shipped model ignores.
  Variants `mf3`/`mf5`/`mf10`/`prod`/`rc4`/`rc4_prod` therefore all collapse to
  the same configuration.
- `mx_june_wft/bench/gap_study.py`, `ntof_tracking/run79_event_display.py` —
  set `WFT_MODEL_FRAC` in the environment; ignored by the shipped model.
- `ntof_tracking/wft_beam.py` — passes `model_frac` in `reco_config`.

The silent cases matter more than the loud one: numbers produced by re-running
those scripts against the shipped model are **not** comparable to the numbers in
`mx_june_wft/GAP_STUDY_2026-07-30.md`, `GAP_CONSISTENCY_2026-07-30.md`,
`WINDOW_ABLATION_2026-07-30.md` or `ANALYSIS_STATE_2026-07-31.md`, all of which
were produced with this kernel at `MODEL_FRAC = 0.03`.

The gap column in `RECONSTRUCTION_BASIS.md` (det2 30.6, det3 27.9, det6 27.9,
det7 27.5, det4 25.6 mm) also came from this kernel.

## Recovering it

```bash
git show ntof-signed-decoding:wft/model.py      # or calibrate.py, reco.py, calib.py
```

Branch tip at archive time: `9dd7d6e`.
