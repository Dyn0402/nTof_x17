# chi2_bimodality — the double bump in χ²/dof, and why only chamber A has it

Asked of `qa_chi2dof_worst` in the pair-quality set: chamber A's χ²/dof is
double-humped — a sharp peak near 2 and a broad one near 20 — and every arm pair
containing A carries the low bump while no pair without A does. What is it, and
why not on the others.

```bash
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe chi2_shape.py        # tables, ~1 min
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe make_chi2_figures.py # five figures
X17_ROOT=D:/x17 ../../.venv/Scripts/python.exe make_chi2_report.py  # figures/report.html
```

`chi2_shape.py` writes to `<out>/chi2_bimodality/`; the two `make_*` scripts only
draw and only report. `--all-tracks` runs on the whole 2.11 M-track stage-3
population instead of the published pair selection.

---

## 1 · The answer, short

**Most of the bump was a plotting bug — now fixed — and the part underneath it
is real.**

`make_pair_qa_figures._hist` returned `n / tot / np.diff(edges)`. The edges are
`np.geomspace`, so `np.diff(edges)` is a *linear* width that grows in proportion
to χ²/dof — but the axis is log. The plotted height was therefore the per-bin
fraction **divided by χ²/dof**, which lifts the left of the axis by about 5×
relative to the peak and draws a small shoulder as a mode of comparable height.
Redrawn as a per-bin fraction the A–A curve is single-peaked near 25 with a
shoulder below 5 holding **5.3 %** of pairs.

**This affected five of the nine panels in that set** — every log-scaled one:
`chi2dof_worst`, `q_total_min`, `sep_mm`, `v_r`, `t_flash_ms`.

> **Fixed 2026-09-12.** `_hist` now returns the per-bin fraction, which is what
> its y-axis label always claimed, and the nine figures are regenerated. Bins
> are uniform in the plotted coordinate on both scales (`linspace` for linear,
> `geomspace` = uniform in log), so a plain fraction is correct on either axis.
> The correction also **revealed** a feature the lift had flattened: D–D's χ²
> peaks near 250, an order of magnitude above every other arm pair.

The shoulder is nonetheless A's own: 5.3 % on A–A against **1.5 % on C–C and
0.8 % on D–D**. And at the single-track level, where the pair variable's
worst-of-four stops hiding it, the bimodality is strong and real.

### On a linear axis, at 0.1 resolution

A linear axis binned at a width of 1 put 36 % of chamber A in a single bin —
that bin *was* the peak. At 0.1 the shape is there, and it says something the
coarse view could not: **A's peak sits at χ²/dof = 1.05**, on the noise floor
rather than near it. `figures/axis_and_binning.png`.

| arm | peak at | FWHM | peak density | frac < 1 | **frac 1–2** | frac < 10 |
|---|---:|---|---:|---:|---:|---:|
| A | **1.05** | 0.7–1.9 | 0.475 | 14.3 % | **36.3 %** | 77.3 % |
| C | 1.65 | 0.8–3.9 | 0.133 | 2.6 % | 11.9 % | 53.1 % |
| D | 1.35 | 0.7–2.8 | 0.109 | 5.4 % | 9.0 % | 37.8 % |

**This forces a refinement on the headline.** All three chambers *do* have a
peak at the floor. The difference is how much of the sample is in it and how
tight it is — A's is 3.6× taller than C's and about half the width. "C and D
never reach the floor" was too strong; **"C and D put a fifth as many tracks
there"** is what the data says. The short-track *medians* (A 1.5, C 2.8, D 3.1)
are unchanged and still correct — they are a different measurement.

On the linear axis the high mode never appears, because it is spread over
decades. That is not a disagreement with the log view, it is the two axes
answering different questions, and it is why the figure carries both. In log
space with the correct per-bin normalisation A has a dip at 11.0 that is **5.4×
below its low mode**; dip depth against the *shallower* of the two modes, where
1.0 would mean no dip: **A 0.63, C 0.89, D 0.76**.

> A density per unit χ² — dividing by the bin width — is the **correct**
> normalisation on a linear axis, and it is what lets the two linear panels use
> finer bins at low χ² and still be comparable. It is the *same operation* that
> is wrong on the log axis. The mistake was never the division; it was dividing
> by a linear width while binning geometrically.

## 2 · What χ²/dof is here, because everything turns on it

Not a track-residual χ². `wft.model.chi2_plane` fits the forward model to the raw
waveform window — every strip, every sample — and `dof = (~saturated).sum()`,
measured here as **exactly 20 × n_strips on 99.2 % of tracks**. So χ²/dof is the
mean squared residual *per sample*, in units of that strip's own noise, which
`wft.model.prep_plane` takes from the event itself.

**1.0 is the noise floor and it means the same thing on all four chambers.** A
chamber whose best tracks sit at 3 has a model that does not describe its data;
it is not a units problem. That is what makes the comparison below a statement
about the chambers rather than about normalisation.

## 3 · What the bimodality correlates with

**Track length, which is the mixture.** χ²/dof climbs monotonically with
`x_n_strips` on every chamber. At fixed length every chamber is single-peaked —
there is no second mechanism, just two populations.

| n_strips | A | C | D |
|---|---:|---:|---:|
| 11–13 | 1.5 | 2.8 | 3.1 |
| 14–17 | 1.4 | 3.5 | 4.2 |
| 18–23 | 1.7 | 5.2 | 25.8 |
| 24–33 | 8.9 | 12.6 | 64.8 |
| 34–59 | 16.7 | 21.3 | 27.9 |
| 60+ | 20.8 | 25.8 | 128.8 |

**Pulse amplitude, which is the mechanism.** At fixed length χ²/dof rises with
charge per strip, because the model's residual is a fixed *fraction* of the
pulse: on a quiet track it is buried in the noise, on a loud one it is not. A's
low-χ² tracks carry ~2.5× less charge than its high-χ² ones at identical
`n_strips`. (In the *unselected* population there is also an upturn at the very
quiet end — fits that explained nothing — but the selection removes most of it.)

**Not the length spectrum.** Fraction at the floor (χ²/dof < 1.5), each chamber
re-averaged over another's `n_strips` distribution. If the spectrum were the
explanation, a reweighted row would land on the column chamber's own value. It
does not — A given C's lengths is still 0.296 against C's own 0.084.

| fraction at floor | own | on A's lengths | on C's | on D's |
|---|---:|---:|---:|---:|
| A | 0.369 | — | 0.296 | 0.255 |
| C | 0.084 | 0.110 | — | 0.068 |
| D | 0.101 | 0.113 | 0.095 | — |

**Not the run, the period or the 27 July access.** A holds its factor 2.3 under
C and D in every one of the 26 runs with enough short tracks to measure, and in
both access conditions. (C and D interleave with each other, so this separates A
from the pair and does not rank C against D.)

## 4 · So why only A

Both modes exist on all three chambers. **A is the only one whose short-track
mode reaches the noise floor** (median 1.5), which puts it 13× below its
long-track mode at 19 and leaves clear air between them. C's short mode is at 2.8
and D's at 3.1 — not resolved from the long-track continuum, so the same two
populations read as one broad blob.

**The low bump is not missing on C and D. It is not separated.**

Chamber B is in none of this: it carries no angle calibration, so it is in no
pairing. On the full stage-3 population it is the worst of the four, which is
what `RECONSTRUCTION_BASIS.md` and the in-situ work already say — no field-shaping
rings, so no clean drift ladder for the model to fit.

## 5 · Where a chamber constant could live — and what is NOT established

The offset is fixed per chamber, so it points at the per-detector calibration
bundle. The four differ structurally. **This is a matter of record, not a result:
nothing here ranks these differences or shows that any one of them causes the
offset.**

> ⚠ **Read `bundle_diff`'s docstring before using that table.** It reads the
> *bench* bundles. The campaign ran `calib_bundle_prelim`, which
> `ntof_tracking.wft_beam.make_bundle` derives from them: it carries the impulse
> template and the kernel hypers through verbatim — `sigma_p0` and `Dp`
> included — replaces `v_drift` with one shared Magboltz prior, and **drops
> `t0_abs` and `t0_prior_sigma` for every arm alike**.

- **No chamber masks a single channel, at any stage.** `dead` is empty on A, B
  and D and absent on C; no bundle has a `hot` map at all. The classifier and the
  bundle patcher both exist and were only ever run on run_145. →
  [`HANDOFF_CHANNEL_MASKS.md`](HANDOFF_CHANNEL_MASKS.md). **This does not explain
  the χ² ordering**: chamber C has *zero* hot channels and still sits a factor
  1.9 above A.
- **C is a bundle generation behind** — `calib_bundle_lp`: no `share_mode`, no
  `dead` key, a bare `c2` rather than `c2_over_c1`, and — the part that survives
  into the beam bundle, and which `make_bundle` itself calls "the largest
  un-validated assumption in this chain" — `sigma_p0` and `Dp` an order of
  magnitude below A's, B's and D's. **This is the live suspect.**
- **A is the only chamber calibrated on a dedicated scan.** Its `run_key` names
  the resistive and drift voltages; D's `conditions` are empty strings.
- **The t0 prior is off on all four chambers**, deliberately, so it explains no
  chamber difference — but nothing replaced it, and the two planes of one chamber
  now disagree on t0 by more than a full 60 ns depth bin on ~43 % of tracks. →
  [`HANDOFF_T0_PRIOR.md`](HANDOFF_T0_PRIOR.md).

**The decisive test is a refit with one knob moved at a time** — D with
`t0_prior_sigma = 5`, C on an `r06`-generation bundle — which needs the waveform
path (`decoded_root` is staged locally) rather than these tables. Not run here.

Also untouched: `q_total` and `x_q_sum` run to 10³⁰ on a tail of tracks, the NNLS
charge profile escaping into a near-null direction. Excluded from the amplitude
tables here; it is its own bug.

## 6 · Figures

| figure | what it is |
|---|---|
| `pair_link` | **the artefact.** The intra pairs as the QA set draws them, beside the same counts per bin. |
| `axis_and_binning` | the low peak at 0.1 linear resolution, its full-range context in the same units, and the correctly normalised log view beside them. |
| `chi2_decomposition` | **the answer.** Per chamber, the distribution with its two length populations in place, and below it each normalised to itself. |
| `chi2_vs_length` | the universal rise and the per-chamber offset, with the noise floor drawn. |
| `chi2_vs_charge` | at fixed length, the amplitude dependence — the mechanism. |
| `chi2_by_run` | the offset is a chamber constant: 26 runs, both conditions. |

`figures/report.html` is the write-up; the DAQ Analysis tab serves it.

## 7 · Handoffs this produced

| | |
|---|---|
| [`HANDOFF_CHANNEL_MASKS.md`](HANDOFF_CHANNEL_MASKS.md) | No reconstruction in this campaign masks a dead or hot channel, though every piece of machinery to do it exists. D is half noise by hit count; C is clean. Includes the measured bound on how much hot-channel contamination could be inside A's low-χ² mode (~1 %). |
| [`HANDOFF_T0_PRIOR.md`](HANDOFF_T0_PRIOR.md) | The t0 prior is off on all four chambers by design, and nothing replaced it. The two planes of one chamber disagree on t0 by more than a 60 ns depth bin on ~43 % of tracks, and the `dt_xy` offset meant to police that **never applies on A or C**, by parity of its `ftst_diff` keys. |
