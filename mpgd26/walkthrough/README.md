# The forward fit, one muon at a time

A step-by-step walkthrough of `wft`'s forward model on **one real det3 cosmic**
— event 1663 of the ref-pinned calibration cache, the same event the MPGD26
deck's "One muon through the forward fit" slide uses. Written 2026-08-18 to
answer a direct question: *is the forward fit what I think it is?*

    ../../.venv/bin/python make_figures.py [--only f1_raw,f9_scan] [--fast]
    ../../.venv/bin/python make_note.py

`make_figures.py` writes `figures/*.png` **and** `steps.json`; `make_note.py`
builds the standalone note `forward_fit_det3.html` from both, so the prose and
the pictures cannot drift apart. `--fast` skips the 220-event ensemble (step 11),
which is the only slow stage.

Published: <https://dylan-neff.web.cern.ch/notes/forward-fit-det3.html>

## What it establishes

The description under test was: *"we guess a charge distribution over each
strip, produce the 0, ±1, ±2 waveforms, then sum over all strips and iterate
track angle."* Right in outline; three corrections, all drawn:

1. **The free charge is per 60 ns arrival slice, not per strip** — 18 of them, a
   charge-versus-depth profile. Where a slice lands transversely is *fixed* by
   (p0, w) through the strip integral.
2. **The charges are solved, not searched.** NNLS, exactly, at every trial
   geometry. Three numbers are searched: p0, w, t0.
3. **The ±1/±2 copies live inside the design matrix**, so a strip and the
   neighbour that donated to it are fitted with one consistent charge set.

## Which calibration it runs on — read this first

**`calib_bundle_r06`**, i.e. the CORRECTED kernel (c2 = 0.6 c1) — now the only
kind that will load. The bundle the frozen MPGD26 reco used carried
c2/c1 = 1.14, a ±2 copy larger than the ±1 copy, which cannot happen on a
resistive film; it was retired on 2026-08-21 together with every product built
on it, and `wft.calib.check_kernel_ordering` refuses to load one.

The old side-by-side (section 10) is gone with it. What it measured, before it
was deleted: the kernel change moves σθ by under 0.6σ on this event set. The
population-level cost is in `mx_june_wft/R06_GATE_2026-08-19.md`, which is the
record to read.

## Numbers worth having (event 1663, Y, `calib_bundle_r06`)

| | |
|---|---|
| window | 19 strips × 32 samples = 608 measurements, 18 charges + 3 geometry |
| geometry | p0 231.478 mm, w −0.008549 mm/ns, t0 230.0 ns |
| kernel (Y) | c1 0.145, c2 0.087 (pinned ratio 0.6), τ_s 166 ns, `share_mode = 'delay'` |
| neighbours' share of the predicted peak | 19 % on the core, up to 64 % two strips out |
| fit quality | χ²/dof 18.8; residual rms 32 ADC = 3.1 % of peak against 0.7 % noise |
| held-out resolution | σ68 1.03° (X) / 1.07° (Y) on 220 events (superseded kernel: 0.99 / 1.03) |
| cross-check | the same code on the frozen bundle reproduces the production `events.parquet` row exactly (230.9404 / −0.008545 / 290.0) |

## Traps found while writing it

- **χ²/dof is 21, not 1.** The residuals are model error, not noise, so the
  curvature errors are optimistic. Quote resolution against the reference.
- **The bench position residual (≈ 650 µm) is not a detector resolution** — it
  is M3 pointing plus scattering. Position comes from the SPS beam.
- **The deck attributes σ68 to "the full 7,093-event run"**; the resolution is
  measured on the 6,852/6,850 events that also have an M3 reference.
- **The σ = 5 ns t0 prior does not arbitrate between the 60 ns-spaced minima.**
  With χ²/dof ≈ 20 it contributes ~50 units against a ~400-unit χ² difference
  between adjacent depth-bin minima; it works by *seeding* the search at the
  prediction, not by out-weighing χ². On event 1663 the frozen fit sat in the
  *higher*-χ² of the two minima and the corrected one did not. Not visible in
  the ensemble numbers — but worth a sweep during the re-freeze.
- **Pick well-centred slices for the "one slice → five waveforms" figure.** A
  slice straddling a strip boundary splits geometrically and buries the kernel
  copies. `f5_column` now selects the best-centred early and late slice.
- **σ_p0 ≈ 0.42 mm on a 0.78 mm pitch means the ±1 strip gets real charge from
  geometry alone** (~84 % of its single-slice signal is its own). The kernel
  copies only dominate from ±2 out. Do not claim otherwise.
- `share_lp` in the hyper dict is inert — the form is read off the bundle's
  `share_mode`, which on the frozen bundle is `delay`.

## Relationship to the c2 > c1 work

Section 10 is that defect on one event, and this package is now the clearest
statement of it: the frozen bundle draws the ±2 copy *taller* than the ±1.
Refitting with `c2_over_c1 = 0.6` moves this event's angle by 0.006°
(χ²/dof 21.1 → 18.8) and is free in resolution fleet-wide. See
`sps_beam_test_26/analysis/sharing_kernel/` and the "sharing kernel, measured"
note. **Nothing is re-frozen** — and the deck's own kernel figure
(`mpgd26/make_share.py`) still reads the frozen bundle, so it still draws
c2 > c1. That is queued in `slides/SLIDE_EDITS_TODO.md`.
