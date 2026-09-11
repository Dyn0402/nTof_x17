# Preliminary analysis — status

**Keep this file current. It is the resume point if a session drops.**
Plan of record: [`PLAN.md`](PLAN.md). Board:
<https://dylan-neff.web.cern.ch/x17/analysis.html>.

**One line:** **every arm-A track now points back at the scintillators behind
it, and the wall says the arm-A angle scale is 33 % out.** 2 498 384 gated
tracks over 31 runs, extrapolated to the SiPM wall and the plastic bars: 39 %
of the confirmable ones point at a channel that fired against a 0.7 % accidental
floor, 65 % on the beam-pointing tracks, and 68 % of those are confirmed by both
layers at once. Two things fell out of it that nothing else in the chain could
see. **The wall's surveyed group boundaries appear to move with the track's own
slope** — 32.5 mm per unit tan, which only an angle-scale error produces — and
the plastic, at 1.96x the lever arm, measures 1.99x the shift and the same
eps = +33 %. **And a fifth of the arm-A track table lands outside the chamber in
v**, 14 % of all tracks piled in one 20 mm window where the fitted y rails; the
scintillators confirm those at 7 % against 46 % inside. Report:
`<out>/det_a_scint/report.html`.

Before that: **the detector-A intra-chamber control is consistent with
accidentals.** 296 218 intra-A pairs whose opening-angle distribution matches
two tracks that never shared a trigger (31.1 deg against 32.4; 49.2 against 47.4
where the angle is measured), with one positive surviving: on the
slope-selected pairs the two legs converge on the beam axis at 2.06 +- 0.29
times the mixed rate, 4.4 sigma. Report: `<out>/det_a_intra/report.html`.

And before that: **the acceptance is now per run, and it was not the fault.**
The scintillator-tagged efficiency is measured in all 36 runs and moves by
6-10 % p10-p90; run_145's borrowed acceptance was within 1.5 % of the campaign
median. What *is* wrong with it is smaller and more specific: the efficiency map
stops at |u| = 160 mm and the code turns that edge into a zero, cutting
7-11 % of accepted legs. **Folding the aluminium capsule continuum in beats the
helium gas in every topology** (coincident perpendicular chi2/dof 7.4 vs 11.0)
and, on the acceptance-corrected perpendicular sample, describes everything
below 109 deg at chi2/dof 4.5 -- leaving a **2.2x, 5 sigma excess above it**
that the accidental template covers. Report:
`<out>/fold_campaign/report.html`. The angle scale remains the open blocker.

Previously: **the whole campaign is now reconstructed blind** — 12 929 condor
jobs over 36 runs / 293 sub-runs / 25.27 M triggers, **47.8 M events, 9.5x the
allowlist pass**, in `<out>/reco_fullpass`. The allowlist was retired because
it kept only **12.8 %** of the triggers that reconstruct into a two-track
event. **The blocker is no longer statistics, it is the angle scale.** With
all 36 runs calibrated alike on the full pass, `k` varies run to run --
but **not randomly**. The spread is one contiguous 48-hour block, **runs
128-147 (3-5 Aug)**, where all three arms rise together (A +2.3 %, C +7.8 %,
D +7.3 %) and **run_145 -- whose `k` every calibrated angle in the current
track table borrows -- sits at its peak**. **It is not a drift-velocity
change:** the drift HV never moved and the k-independent drift end time rises
~1 % against the ~7 % a real velocity drop demands. Decision 2026-09-10: **push the full pass through with per-run `k`,
and defer a real per-detector recalibration to October** — meanwhile open up
the per-track tracking distributions run by run and tag by tag to find what
drifts. **Do not build an opening-angle spectrum on a borrowed `k` for C or D.**

Last updated **2026-09-10** (arm-A tracks are now confirmed positionally against the scintillators, and the wall measures the angle scale; the FULL pass is complete; per-run capsule imaging says the geometry is sound and the angle scale is the fault; the campaign opening-angle spectra exist).

> **START HERE:** [`HANDOFF_FULLPASS_2026-09-10.md`](HANDOFF_FULLPASS_2026-09-10.md)
> — why the full pass happened, what it produced, and the run_86 test that now
> blocks the opening angle. The entries below are the working record behind it.

> ## DETECTOR A -> SCINTILLATORS: A PER-TRACK POSITIONAL CONFIRMATION, AND THE WALL MEASURES THE ANGLE SCALE -- 2026-09-10
>
> New: `det_a_scint.py`, `make_det_a_scint_figures.py`,
> `make_det_a_scint_report.py`, `det_a_scint_chain_2026-09-10.sh`. Report at
> `<out>/det_a_scint/report.html`; per-track tables under
> `<out>/det_a_scint/tracks/`, one parquet per run.
>
> **What the analysis had before this, and why none of it was enough.**
> `wall_A`/`plastic_A` in the track table are PER-TRIGGER stage-1 booleans, so a
> track on the far side of the chamber from the fired bar carries the same flag
> as one pointing at it -- `det_a_intra`'s `tag_A` is this.
> `build_tracks.predictions` has been writing `pred_sipm_bar`/`pred_plastic`
> into all 586 track files and **nothing ever compared them with the slim**.
> `run145_target_imaging.pointing_coincidence` does compare them, but in the x
> plane only: no v check, no residual, and used as a purity cut.
>
> **THE CONFIRMATION.** 2 498 384 gated arm-A tracks, 31 runs, extrapolated in
> 3D to each layer at positions read from the DAQ's own `run_config.json`.
>
> | selection | wall | floor | plastic | floor | both layers |
> |---|---:|---:|---:|---:|---:|
> | all | 39.0 % | 0.70 % | 53.3 % | 3.7 % | 43.8 % |
> | fiducial | 46.3 % | | 54.4 % | | 44.8 % |
> | beam-pointing | 64.9 % | 0.31 % | 75.8 % | 1.2 % | **68.2 %** |
>
> **Every control window is the SAME WIDTH as the signal window it controls**,
> and the first version of this module got that wrong: a 700 ns control against
> a 160 ns signal window made the plastic's confirmation rate (59 %) equal its
> apparent floor (58 %). Two independent floors are quoted and they agree --
> the slim's own `is_control` sample and a pre-trigger window. **Pre-trigger,
> not post:** the wall's `dt_ns` is flat outside one 100 ns peak but the
> **plastic decays for the best part of a microsecond after the trigger** (27 154
> hits in (0,100) falling through 6 223 at +400, still ~15 % above pedestal at
> +900), so a +400 ns control sits in a real tail and reports 21 % where the
> truth is 5 %. In the control windows the "match" lands on the predicted one of
> four wall groups 28 % of the time and one of two plastic bars 53 % -- chance,
> exactly, which is what validates the floor.
>
> ## THE WALL MEASURES THE ANGLE SCALE, INDEPENDENTLY OF THE IMAGING
>
> The wall's group boundaries are surveyed and fixed. Fit where each boundary
> APPEARS to sit, separately in bins of the track's own in-plane slope:
> **the boundary moves at 32.5 +- 5.4 mm per unit tan**, on a rigid offset of
> 1.6 +- 1.5 mm. Only an angle scale does that -- a survey error, a swapped
> read-out order or a plane-fit bias shifts every bin alike and lands in the
> intercept. Dividing by the 97.4 mm lever arm:
>
> **eps = +33.4 +- 5.5 %, i.e. the wall prefers k(arm A) x 1.33.**
>
> **The decisive check is the second lever arm.** The plastic sits 190.6 mm
> past the strip plane against the wall's 97.4, a ratio of **1.96**; it measures
> **64.6 mm per unit tan, a ratio of 1.99**, and returns eps = +33.9 +- 8.2 %.
> Two layers, two levers, one number. It holds on every selection that does not
> cut on the direction: 33.4 (all), 31.3 (fiducial), 31.4 (slope), 30.2
> (fiducial+slope). **The `pointing` cut must NOT be used for it** -- that cut is
> computed FROM the reconstructed direction, so it selects on tan correlated
> with position, and it duly returns +17.7 +- 11.3 at chi2/dof 83.
>
> **What it is not.** One arm, one axis (the wall's u), one estimator, and the
> wall fit has chi2/dof 17.8 so the shift is not perfectly linear in tan (the
> quoted error is inflated to cover it). **It is not a replacement calibration
> and must not be applied as one.** It is an independent handle on the standing
> angle-scale blocker, from a direction the target imaging cannot see, and it
> says arm A's tangents are too large -- the same sign the imaging's k > 1
> already implies, but a much larger size.
>
> ## A FIFTH OF THE ARM-A TRACK TABLE LANDS OUTSIDE THE CHAMBER
>
> Found by drawing the wall projection, which carries a hard horizontal stripe
> at v ~ -195 mm that no part of the apparatus sits at. **The fitted y position
> rails just outside the active area.** Arm A is 340 mm tall, so |v| <= 170 is
> all of it:
>
> | where the track lands | tracks | of all | wall confirms | plastic |
> |---|---:|---:|---:|---:|
> | inside the active area | 2 015 612 | 80.7 % | 46.0 % | 54.0 % |
> | outside it in v | 482 772 | **19.3 %** | 7.7 % | 21.7 % |
> | in the v rail alone | 348 234 | **13.9 %** | **6.6 %** | 15.3 % |
>
> and the railed tracks' accidental floor goes the other way (1.06 % against
> 0.63 %), which is what a population of junk in busier triggers looks like.
> **Reported, not cut** -- whether a railed y should fail the 3D gate is a
> decision for `wft`, not for a confirmation study. But the "all" row of every
> table in this analysis is diluted by it, which is why a `fiducial` selection
> is now carried beside it.
>
> ## Two more things worth keeping
>
> **The position tolerance is 15.5 mm, not the 1.8 mm the fit errors claim.**
> The formal plane-fit error extrapolated to the wall is a floor: it carries the
> plane fit alone, not the angle scale, not scattering, not the survey. The
> width that matters is how sharply the fired group switches across a boundary,
> and that is measurable on single-track single-group events. The plateaux sit
> at 0.80-0.91, not 1.0 -- the wall's own single-group inefficiency and
> cross-talk, which no amount of pointing removes.
>
> **run_126, run_154 and run_156 are excluded and their tracks are not lost.**
> Their stage-3 tables were built before their `k_arm_<run>.json` existed, so
> `angle_calibrated` is false and every direction is null. The module **refuses
> them by name**: projecting a null direction lands on no channel and arrives at
> the far end as a confirmation rate of exactly 0.0 %, which is the one wrong
> answer that looks like a measurement. **Rebuilding stage 3 for those three
> returns ~370 000 tracks, 13 % more than this page has.**
>
> Re-running: `det_a_scint --from-tracks` rebuilds every summary from the stored
> per-track tables without re-reading the 10 GB slim.

> ## DETECTOR A, INTRA-CHAMBER: THE INTRA CONTROL IS CONSISTENT WITH ACCIDENTALS -- 2026-09-10
>
> New: `det_a_intra.py`, `make_det_a_figures.py`, `make_det_a_report.py`,
> `det_a_chain_2026-09-10.sh`. Report at `<out>/det_a_intra/report.html`.
> A deliberately narrow pass: one chamber, one topology, every step checked
> against data before the next was built.
>
> **296 218 intra-A pairs over 31 runs**, against 1 773 525 event-mixed. The
> existing campaign product holds **7 915** intra-A pairs -- and they are
> exactly this module's `pointing` selection. **The campaign's intra-chamber
> control IS the 30 mm-pointing subset, 2.7 % of the intra-A pairs that
> exist**, and the cut is applied silently inside
> `source_imaging._track_table`.
>
> **THE HEADLINE IS A NULL.** The measured opening-angle distribution is the
> same as the distribution of two arm-A tracks that never shared a trigger:
>
> | selection | n | median | event-mixed median |
> |---|---:|---:|---:|
> | all pairs | 296 218 | 31.1 deg | 32.4 deg |
> | both legs slope-reliable | 55 399 | 49.2 deg | 47.4 deg |
>
> and the event-mixed shape beats **every** folded continuum by 10-50x in
> chi2/dof (all pairs: 911 mixed against 9 776 Al capsule, 10 137 3He E0;
> slope: 202 against 10 793 and 7 933).
>
> **THE ONE POSITIVE RESULT.** On the slope-selected pairs the two legs
> converge on the beam axis (lines within 30 mm of each other, within 20 mm of
> the axis) at **2.06 +- 0.29 times the event-mixed rate -- 74 pairs against
> ~36 expected, 4.4 sigma**. On the full sample there is nothing (0.92 +-
> 0.07). It appears only where the direction is actually measured, which is
> where it should. The mixed sample is not separation-matched, so 2.06 is a
> **lower** bound.
>
> ## Three things that had to be established first, and all three are new
>
> **1. The two-track resolution, measured in situ, and it is missing from
> `acceptance.py` entirely.** Real over event-mixed: **0 pairs below 20 mm,
> 56 of 296 218 below 40 mm** of radial separation where ~12 400 are expected.
> `acceptance.py` treats the two legs as independently reconstructed, so it
> accepts a pair 10 mm apart at full efficiency. Folding the measured loss in
> moves the capsule's folded median from **19.0 to 24.7 deg** against 31.1
> observed; it does not change any conclusion.
>
> **And the loss is a CROSS, not a disc** -- found only after the figures were
> made legible. On the map of real over mixed against (|du|, |dv|) the whole
> first row and the whole first column sit at **0.00-0.03** of the plateau
> whatever the other coordinate does: a pair 12 mm apart in u and 300 mm apart
> in v is lost as completely as one 12 mm apart in both. Two independent strip
> planes, and the fit needs both. **A radial efficiency passes those pairs**,
> so the correction is measured once per view (the other held above 100 mm) and
> applied as the product eff(|du|) x eff(|dv|). Per view: 0.005 at 0-20 mm,
> 0.38 (u) / 0.51 (v) at 20-40, complete from 40. The separable product is
> tested against the 2D map, rms residual 0.19 over 169 cells -- it reproduces
> the cross and leaves the leg-to-leg correlation behind, which is not an
> efficiency.
> This is `PLAN.md`'s own D1/D2 and it is now measured. It applies to the
> intra topology only -- two legs in different chambers are reconstructed
> independently -- which is exactly the sample PLAN sec S4 uses as the
> background normalisation.
>
> **2. The in-chamber `dt0` peak is not usable as a coincidence.** `t0` is the
> charge arrival at the mesh, so two legs born together share it; the
> difference shows a clear peak on a broad pedestal. Three tests:
>
> - **the prompt fraction is not identifiable** -- five two-component models
>   with near-identical likelihood return 0.23 to 0.67;
> - **the scintillators do not see it** -- arm-A tagged pairs have the SMALLER
>   prompt-to-off-time ratio, 1.41 against 1.69;
> - **it lives entirely below `wft.reco.TAN_MIN_SLOPE`** -- ratio 1.84 in the
>   lowest |tan| bin, **0.97-1.15 above the threshold, consistent with no peak
>   at all**. (`x_slope_reliable & y_slope_reliable` is exactly
>   `min|tan| >= 0.08`; verified.)
>
> **The obvious explanation is falsified.** A slope-less fit whose `t0`
> collapses onto its prior would do this -- but single-track `t0` is *widest*
> for the low-slope tracks (sd 293 ns against 105 ns). Origin left open.
> **Operationally it does not matter:** on the pairs whose angle is a
> measurement, there is no prompt excess to select on.
>
> **3. A two-dimensional efficiency map for A, in the toy's own frame.**
> Measured per run off the exported slim, stable campaign-wide (p10-p90 6.2 %).
> **And a frame bug: `efficiency.py` bins in `local_x - PINWHEEL`, the lever
> from the beam-axis foot, while `acceptance.Chambers.cross` returns the offset
> from the PLANE CENTRE.** For arm A those differ by 16.35 mm, so the published
> 1-D map is indexed 0.4 bins away from the toy that consumes it. Not fixed in
> `acceptance.py`; `det_a_intra` builds its map in the plane-centre frame.
>
> ## Two bugs worth remembering
>
> **The figures in this entry and the one below were unreadable, and the cause
> is a trap in `figstyle`.** `figstyle.title` sets the headline on an AXES with
> `loc='left'`. On a single full-width axes that is fine. On a two- or
> three-panel figure the headline is several times wider than the panel it is
> anchored to, so it runs off the canvas -- and `savefig(bbox_inches='tight')`,
> which the house rcParams turn on, then EXPANDS the saved image sideways to
> contain it. Measured: a two-panel `WIDE` figure that should be 2133 x 800 px
> came out **3262 x 680**, aspect 2.7:1 -> 4.8:1, and a report that scales the
> image to its column width was left with panels a third of their intended
> height. New `figstyle.fig_title(fig, ...)` anchors the headline to the
> FIGURE, wraps it to the canvas width, and lays the panels out itself; call it
> last and do not call `tight_layout` at all. **Use it for every multi-panel
> figure.** The earlier reports (`angle_campaign`, `imaging_campaign`,
> `tracking_qa_fullpass`) were checked and are unaffected -- aspects 1.05-2.11.
>
> `pandas.Series.to_numpy()` may return a VIEW of the frame's buffer. An
> in-place `&=` on it silently overwrote the `mixed` column mid-run, so the
> census and the saved parquet disagreed about which pairs were real. Every
> mask in `det_a_intra.select` is now `copy=True`. Audited: no other module in
> this package does in-place boolean ops on a `to_numpy()` result.
>
> **Re-run with** `bash sept26_prelim_analysis/det_a_chain_2026-09-10.sh`
> (~15 min; the second pass throws N pairs per run once the two-track
> resolution has been measured on the pooled sample).

> ## THE ACCEPTANCE IS PER-RUN NOW, AND THE BORROWING WAS NOT THE FAULT -- 2026-09-10
>
> New: `campaign_efficiency.py`, `campaign_acceptance.py`, `campaign_fold.py`,
> `make_fold_figures.py`, `make_fold_report.py`,
> `acceptance_fold_chain_2026-09-10.sh`. Report at
> `<out>/fold_campaign/report.html`. Nothing was written into
> `<out>/efficiency` or `<out>/angle` -- the published single-run products stay
> put for the comparison.
>
> **The scintillator-tagged efficiency is now measured in all 36 runs** (off
> the exported n_TOF slim, validated to give the identical tag set to the ROOT
> slim on run_145 arm A) and it barely moves:
>
> | arm | runs | median | p10-p90 | run_145 vs median |
> |---|---:|---:|---:|---:|
> | A | 34 | 23.9 % | **6.2 %** | **+1.3 %** |
> | B (hits) | 34 | 16.5 % | 60.7 % | +6.2 % |
> | C | 34 | 16.9 % | **9.9 %** | **+1.5 %** |
> | D | 34 | 13.1 % | **10.5 %** | **+1.3 %** |
>
> The acceptance curve's own run-to-run shape scatter has a median cv of
> **0.029**, and run_145's curve differs from the pair-weighted campaign one by
> 5 % in integral and 2-5 % rms in shape. **`campaign_angle`'s
> `acceptance_source = run_145 (BORROWED)` was worth ~1.5 % on the efficiency
> scale.** It is not why nothing fits.
>
> **The head-on dip is real, campaign-wide, and smaller in effect than it
> looks.** The head-on wall group's tracking rate over its two positional
> neighbours: A 0.833, B 0.861, C 0.675, D 0.726, **below both neighbours in
> 100 % of run-arm pairs on A, C and D**. But entering it into the toy as a
> factor on the leg's own incidence lands almost exactly on a flat efficiency
> -- the band is narrow in |tan| and a pair from a 20 mm source averages over
> it. Three variants (`flat`, `u_map`, `incidence`), and **`u_map` -- what every
> published number so far used -- is the outlier**: perpendicular fraction above
> 109 deg 0.1999 against 0.2305 (flat) and 0.2290 (incidence).
>
> **Why: the map has an edge and the code turns it into a cut.** The measured
> map spans |u| <= 160 mm (the scintillators stop covering beyond ~150) while
> the active area runs to 190. `acceptance.Chambers.efficiency` interpolates
> with `left=nan, right=nan` and the NaN becomes a zero, so **7.4-10.8 % of
> accepted legs are given zero efficiency** in `u_map`. Reproduced rather than
> corrected, because it is what the published numbers did; measured by
> `campaign_acceptance.edge_cost`.
>
> ## THE ALUMINIUM CAPSULE FITS BETTER THAN THE GAS, AND STILL LEAVES A TAIL
>
> `ipc_aluminium.capsule_spectrum` and `shape_comparison` folded through the
> per-run acceptance, capsule components on a **wall** vertex distribution (the
> skin of the He-3 polycone, where an Al pair is actually born; 2 % in integral,
> nothing in shape) and gas components on the gas volume.
>
> | | median at birth | frac > 109 deg |
> |---|---:|---:|
> | Al capsule, after the wall | **50.6 deg** | **0.125** |
> | 3He gas M1+E0, after the wall | 36.9 deg | 0.065 |
>
> Single-shape chi2/dof on the coincident sample, capsule against gas:
> **opposing 27.5 vs 41.6, perpendicular 7.4 vs 11.0** -- the capsule wins in
> every topology and every selection. The event-mixed accidental template still
> wins overall (12.9 opposing, 2.1 perpendicular).
>
> **The sharpest result is the corrected spectrum, and it needs no model.**
> Dividing the data by its own acceptance, the three topologies imply
> **completely different birth spectra** -- medians 37.6, 81.7 and 112.5 deg --
> when one source and one physics feed all three and a correct acceptance would
> collapse them onto one curve.
>
> **Where there IS agreement.** Perpendicular is the only topology whose
> acceptance does not decide the answer. Normalising below 109 deg only, on the
> coincident sample:
>
> | birth model | chi2/dof below 109 | data/model above 109 |
> |---|---:|---:|
> | **Al capsule (after wall)** | **4.5** | **2.16x (5.0 sigma)** |
> | 3He gas E0 only | 4.8 | 2.49x |
> | 3He gas M1+E0 | 6.2 | 3.14x |
>
> So **the aluminium accounts for the bulk of the continuum and not for the
> wide-angle tail**, and the tail is where the accidentals and the residual
> single-particle background live. The two-component fit (capsule + event-mixed)
> reaches chi2/dof **1.40** on the coincident perpendicular sample -- but at an
> accidental share of **0.70 [0.61, 0.80]** against the timing measurement's
> **0.16 [0.00, 0.32]**, and on opposing it runs to the f = 1 boundary. Two
> independent handles on the same quantity disagree.
>
> **What this does not do:** it does not identify the pairs as aluminium. Only
> the total pair energy separates a 2-4 MeV capsule pair from a 20.6 MeV gas
> pair and this setup does not measure it. The after-wall curves also carry a
> Highland Gaussian that is untrustworthy for 18 % of the capsule weight.
>
> **Re-run with** `bash sept26_prelim_analysis/acceptance_fold_chain_2026-09-10.sh`
> (~25 min, mostly the 8 M-pair-per-run throws). `campaign_acceptance
> --summarise-only` re-derives the summary tables without re-throwing.

> ## PER-RUN CAPSULE IMAGING: the geometry is SOUND -- 2026-09-10
>
> New: `campaign_imaging.py`, `make_campaign_imaging_figures.py`,
> `make_campaign_imaging_report.py`. Report at
> `<out>/imaging_campaign/report.html`.
>
> **The pointing crossing is scale-free** -- multiplying every angle by `k`
> scales the band's slope and intercept together and leaves
> `-intercept/slope` untouched -- so it is the one geometric observable the
> angle-scale problem cannot touch. Run once per run on the condor full pass,
> **33 of 36 runs post-access**:
>
> | | value | spread over 33 runs |
> |---|---:|---:|
> | capsule X (mean of A and C) | **-9.32 mm** | **+-0.30 mm** |
> | A-C alignment (half their difference) | +1.04 mm | +-0.27 mm |
> | capsule Z (D alone) | -3.45 mm | +-0.83 mm |
>
> Per arm the run-to-run sd is A 0.30, C 0.49, D 0.83, B 1.63 mm. **Divide by
> each arm's own sub-run spread and every ratio is BELOW 1** (A 0.59, B 0.66,
> C 0.71, D 0.99): the runs do not differ from each other by more than one
> run's sub-runs differ among themselves. **On this observable there is no
> run-to-run effect at all**, against an angle scale whose p10-p90 is
> A 3.4 %, C 10.7 %, D 9.6 % over the same runs.
>
> **What that settles.** The alignment, the frame, the in-plane signs, the
> pointing sample and the track finding are reproducible campaign-wide. The
> fault is in the depth-to-length conversion alone. It does NOT make `k`
> right -- `gap_check` still fails on A and C.
>
> **A scale-free observable DOES move inside the 128-147 block, slightly.**
> C's crossing +0.50 mm (1.26 sd, p = 0.011), D's -0.77 mm (p = 0.055), and the
> A-C alignment tightens by 0.35 mm (p = 0.021). Sub-millimetre against an 8 %
> shift in `k`, so the excursion is not mostly geometric -- but it is not
> purely an estimator artefact either. **Crossing against `k` is correlated on
> C (rho +0.48, p = 0.012) and D (rho -0.37, p = 0.043)** and `k` cancels
> algebraically, so a third thing moved and changed both. Same lead as the
> `x_local` step; still a lead, not a mechanism.
>
> **The y half is badly wrong and it is NOT new.** `target_y_mm` carries `k`,
> and against the polycone forward model every chamber sits 15-27 mm off in
> median with an IQR 2.3-5.6x the model's. Run-to-run sd under 1 mm on A and C,
> so the failure is campaign-wide and stable -- either the polycone acceptance
> model or the y reconstruction, and this does not separate them. run_145's
> published `y_compare` had exactly these numbers; nobody had checked it was
> universal.
>
> ## THE CAMPAIGN OPENING-ANGLE SPECTRA EXIST -- 2026-09-10
>
> New: `campaign_angle.py`, `make_campaign_angle_figures.py`,
> `make_campaign_angle_report.py`. Report at
> `<out>/angle_campaign/report.html`. Pairs also censused directly off the
> track table: **68 542 real pairs**, A-C **14 594** (PLAN sec S4 projected
> ~11 450 from run_145's 229 x 50).
>
> | topology | all pairs | tagged | tight | tight_pair | median | >109 deg |
> |---|---:|---:|---:|---:|---:|---:|
> | intra | 19 707 | -- | -- | -- | 38.7 | 0.000 |
> | perpendicular | 29 617 | 2 290 | 503 | **503** | 87.7 | 0.252 |
> | opposing | 13 405 | 3 027 | 949 | **571** | 144.1 | 0.955 |
>
> (Pre-access run_79/81 excluded; 62 729 pairs enter, 1 074 tight_pair.)
>
> **Four things this makes plain, none of them a signal.**
>
> **1. The topology decides the angle before the physics does.** Opposing
> produces nothing below ~90 deg and intra nothing above ~110. "Fraction above
> 109 deg" is a statement about the chambers first: every folded model predicts
> 1.000 for opposing and 0.000 for intra.
>
> **2. The back-to-back single particle is 12 % of opposing pairs and the
> TIGHT CUT ENRICHES IT TO 40 %.** One particle through both chambers is
> perfectly time-coincident because it is one particle, so a cut built to
> remove accidentals raises its share 3.4x -- and it lands inside the X17
> region. `tight_coincidence` already flags it; `opening_angle.py` does not
> know about it at all.
>
> **3. Nothing fits.** On the tight opposing sample chi2/dof is 29 (M1),
> 38 (thermal M1+E0), 85 (E0), 29 (X17) and **13 for the event-mixed
> accidental template** -- the accidental shape is still the best description,
> as the S4 null said on 1/10 the sample. With one free normalisation, chi2/dof
> of this size is what a WRONG ACCEPTANCE looks like, and the acceptance here
> is run_145's, borrowed, applying efficiency independently of incidence when
> the measured head-on tracking ratio is 0.80.
>
> **4. The intra control has no timing cut and cannot have one.** Both legs in
> one chamber means one arm and no `t1 - t2`, so the mutual half of the tight
> cut does not exist for it. Comparing intra (uncut) against opposing (cut) is
> not like for like -- and PLAN sec S4 makes intra the background
> normalisation.
>
> **Two bugs fixed on the way.** `source_imaging._pairs_real` returned an
> object-dtype empty frame and `_vertex_frame` a column-less one, so any run
> with no calibrated pairs (run_126, run_143, run_156) crashed every campaign
> consumer instead of contributing nothing. `campaign_imaging` now also drops a
> sub-run missing any arm's merged table rather than failing the run --
> `run_104/stat090_0016`, the corrupt tag, is the only one.
>
> **Re-run both with** `bash sept26_prelim_analysis/campaign_qa_chain_2026-09-10.sh`
> (after `fullpass_chain_2026-09-10.sh`). It archives the previous
> tight-coincidence campaign products before overwriting them, and it does
> **not** pass `--campaign` to `tight_coincidence` -- that flag points at
> `<out>/stage3_campaign`, the allowlist pass.

> ## FULL PASS LAUNCHED -- Dylan's call, 2026-09-09 21:10 CEST
>
> The timing check below argued the full pass was roughly break-even on shape
> significance. **Dylan overrode it: "not worth missing anything."** Recorded
> so the reasoning on both sides survives -- the check stands, the decision is
> to reconstruct everything anyway.
>
> | | |
> |---|---|
> | jobs | **12 932** = 3 233 (run, sub-run, tag) x 4 arms |
> | cost | ~1.2 core-h/job, **~16 000 core-hours** (~20x the allowlist pass's 760) |
> | EOS out | `/eos/user/d/dneff/x17/sept26_fullpass` (NOT the allowlist pass's dir) |
> | local out | `<out>/reco_fullpass` -- **`<out>/fullpass` already holds the ALLOWLIST pass** despite its name; unpacking on top would replace a known sample with a superset and lose the comparison |
> | driver | `overnight_fullpass_2026-09-09.sh`, resumable (skips what is on EOS) |
>
> ## The run_86 test came back, and it FALSIFIES the borrowed k -- 2026-09-10
>
> The 14-hour run_86 local full pass finished at 07:26 and **its `k_arm` step
> never ran**: `unattended_2026-09-09.sh` called `k_arm.py --out <dir>`, and
> that flag does not exist. Worse, `k_arm` has NO `--out` -- it always writes
> `paths.out('kcal')/k_arm_<run>.json` -- so the "fix" of dropping the flag
> would have **overwritten run_86's calibration-pass file**, the exact trap
> that deleted run_145's published calibration on 2026-09-09. Re-run by hand
> with the calibration-pass result copied aside first; both now coexist as
> `k_arm_run_86.{fullpass,calibpass}.json` and `k_arm_run_86.json` is restored
> to the calibration-pass content it had, so nothing downstream moved.
>
> | arm | r86 FULL | r86 calib | full/calib | r145 FULL | **r86/r145** |
> |---|---:|---:|---:|---:|---:|
> | A | 1.2294 | 1.1840 | +3.8 % | 1.2662 | **-2.9 %** |
> | B | 1.5664 | -- | -- | 2.1421 | -26.9 % (never certifies) |
> | C | 1.1560 | 1.3500 | **-14.4 %** | 1.6163 | **-28.5 %** |
> | D | 1.3413 | -- | -- | 1.7667 | **-24.1 %** |
>
> **The method hypothesis fails on its own terms.** It predicted the full pass
> would sit ~6 % ABOVE the calibration pass on both arms. A moves +3.8 %,
> **C moves -14.4 %** -- opposite direction, larger size. There is no single
> method offset.
>
> **And with the method now MATCHED -- both are full passes -- k still differs
> between the two runs by 24-29 % on C and D**, while A agrees to 2.9 %. run_86
> and run_145 are on the same side of the 27 July access. This is `PLAN.md`'s
> own stated falsifier arriving: *if `k` varies strongly run to run, one bundle
> is wrong.* **Applying run_145's k campaign-wide is not supported for C or D.**
> C is half of the A-C opposing pair, which is the signal topology.
>
> **Three caveats, none of which rescue the borrowed k.** Every run_86 verdict
> is PROVISIONAL with the focus scan flat over 30-37 %, so the estimator is
> weak; the within-run estimator spread is 8-15 %, below the 24-29 % gap but
> not negligible; and **run_86 has no hot-strata table, so its hot-channel cut
> was NOT applied** -- on run_145 that cut moved D's k by 3.65 % and removed
> 22.4 % of D's coincident sample, so D's comparison is partly confounded and
> C's is not.
>
> **What this changes.** The full pass just made per-run `k` affordable for all
> 36 runs, and this says it is also necessary. `campaign_tracks --k-from
> run_145` stamps `k_source`, so nothing already built is silently wrong -- but
> 83 % of the current track table's calibrated angles rest on a scale this test
> does not support for two of the three usable arms.
>
> Also corrected: the "run_86 calibration-pass k (A 1.184, C 1.350)" quoted
> below was **never certified** -- that file's `apply` is `{}` and both arms
> read NOT CALIBRATED. They are raw fit values, not measurements.

> ## The track database is REBUILT on the full pass, with per-run k -- 2026-09-10
>
> `fullpass_chain_2026-09-10.sh` finished 10:14. **`<out>/stage3_fullpass`,
> 11.4 GB, 292 sub-runs** (the corrupt tag's sub-run skipped, as designed).
>
> | | full pass | allowlist pass | ratio |
> |---|---:|---:|---:|
> | track segments | **29 159 045** | 2 111 162 | 13.8x |
> | gated | **14 263 041** | 1 137 029 | 12.5x |
> | calibrated | 22 758 421 | 1 750 293 | -- |
> | file tags | 3 232 | 3 150 | run_145 now tagged properly |
>
> **Every run carries its OWN k** (`k_source` = `self`), not run_145's. 35 of
> 36 runs certified at least one arm; run_126 certified none.
>
> **The full pass is a CLEANER sample, and that retires a number quoted all
> week.** The allowlist selected busy multi-arm events, which are the hard ones
> to fit. Median chi2/dof, allowlist -> full pass:
>
> | arm | chi2/dof x | chi2/dof y | p25 x | frac chi2 > 100 |
> |---|---|---|---|---|
> | A | 6.0 -> **6.3** | 5.9 -> 5.0 | 1.65 -> 1.55 | 0.018 -> 0.007 |
> | B | 19.1 -> **14.9** | 16.3 -> 14.1 | 9.70 -> 7.12 | 0.050 -> 0.031 |
> | C | 16.8 -> **12.2** | 16.5 -> 11.9 | 6.95 -> 4.82 | 0.052 -> 0.032 |
> | D | 38.6 -> **16.9** | 55.0 -> 21.6 | 7.67 -> 2.64 | 0.260 -> 0.145 |
>
> **"Arm D fits at chi2/dof 39" was largely an allowlist selection effect.**
> D is still the worst arm, at 16.9, not 38.6. A is unchanged, which is what
> you would expect of the arm whose model already described its data.
>
> **The charge blow-up got WORSE, not better**: `frac_q_gt_1e6` 0.282 -> 0.295
> on A, 0.265 -> 0.314 on B, **0.281 -> 0.362 on D**. Campaign-wide **29.3 %**.
> It is not a selection artefact and it is not going away.
>
> ## The geometric bound now fails on C as well as A -- 2026-09-10
>
> Re-run on the full pass with per-run `k`, `gap_check` flips C from pass to
> fail, because C's per-run `k` (1.436) is well below the borrowed run_145
> value (1.616) that had been holding its span inside the gap:
>
> | arm | gap | k applied | deepest possible | median unrailed span | **past the gap** | k the gap demands |
> |---|---:|---:|---:|---:|---:|---:|
> | A | 27.9 | 1.229 | 37.4 | 29.3 | **0.678** | **1.649** |
> | B | 30.5 | 1.959 | 23.5 | 19.4 | 0.000 | 1.508 |
> | C | 30.0 | 1.436 | 32.0 | 27.2 | **0.400** | **1.534** |
> | D | 30.0 | 1.613 | 28.5 | 24.6 | 0.000 | 1.534 |
>
> **Note the DIRECTION, which is the whole point.** The pointing estimators
> want a SMALLER `k` on A and C; this bound wants a LARGER one. They disagree
> in a fixed direction on the two arms that fail, and agree on the two that
> pass. **That disagreement, not the run-to-run scatter, is the sharpest single
> statement available about the calibration -- and it argues against treating
> any one arm as the reference in October.**
>
> ## What the bigger sample says about drift -- 2026-09-10
>
> Post-access Spearman on ~2 980 tags per arm, all far past any plausible
> multiple-comparison threshold:
>
> | arm | variable | rho after the access | pre -> last tenth |
> |---|---|---:|---|
> | D | chi2dof_y | **+0.57** | 20.1 -> 29.5 |
> | B | t0_y | +0.57 | 96.7 -> 96.0 (dips to 89 mid-campaign) |
> | B | t0_x | +0.55 | 83.0 -> 86.0 |
> | C | frac_gated | **-0.53** | 0.461 -> 0.473 |
> | A | chi2dof_x | **+0.48** | 5.12 -> 6.67 |
>
> **D's y-view degradation is confirmed and stronger on the bigger sample**,
> and **A now shows the same thing** (chi2/dof x climbing 5.1 -> 6.7 through
> the campaign) where the small sample showed nothing. Both fits get worse with
> time; neither the HV nor the DAQ configuration changed.
>
> Report: `<out>/tracking_qa_fullpass/report.html`. The allowlist-pass QA is
> kept beside it at `<out>/tracking_qa/` for the comparison above.

> ## CORRECTION: all 36 runs calibrated alike, and the scatter is ONE 48-HOUR BLOCK -- 2026-09-10
>
> **This supersedes the "13-17 %" in the section below.** That figure compared
> run_145's FULL pass against four runs measured on the prescaled CALIBRATION
> pass. `fullpass_chain_2026-09-10.sh` has now run `k_arm` on the condor full
> pass for every run -- **35 of 36 certified at least one arm** -- so the
> comparison is finally like for like.
>
> | arm | n runs | min | median | max | p10-p90 spread | run_145 vs the rest |
> |---|---:|---:|---:|---:|---:|---:|
> | A | 32 | 1.2055 | 1.2260 | 1.2848 | **3.4 %** | +3.3 % |
> | B | 6 | 1.8405 | 2.0098 | 2.4000 | 27.8 % | -- |
> | C | 30 | 1.3866 | 1.4497 | 1.6165 | **13.2 %** | +11.5 % |
> | D | 34 | 1.5650 | 1.6104 | 1.8000 | **9.8 %** | +9.9 % |
>
> **And the scatter is not scatter.** Nearly all of C's and D's spread is one
> CONTIGUOUS BLOCK -- **runs 128 through 147, 3 August 17:26 to 5 August 17:34**
> -- in which all three arms rise together:
>
> | arm | outside the block | inside | shift |
> |---|---|---|---:|
> | A | 1.2221 (n=23, 1.2055-1.2400) | 1.2499 (n=9, 1.2141-1.2848) | **+2.3 %** |
> | C | 1.4349 (n=22, 1.3866-1.4819) | 1.5472 (n=8, 1.5103-1.6165) | **+7.8 %** |
> | D | 1.6023 (n=26, 1.5650-1.6637) | 1.7194 (n=8, 1.6500-1.8000) | **+7.3 %** |
>
> **C's inside and outside ranges do not overlap at all.** Both edges of the
> block fall in beam-off gaps (run_124 -> run_126 is ~18 h; run_147 -> run_150
> is ~43 h). **run_145 sits at the peak of it** -- which is why borrowing its
> `k` looked so wrong, and why run_86 (27 July, outside) looked so different.
>
> ## ...and the excursion is NOT a drift-velocity change -- 2026-09-10
>
> The obvious reading is gas: water slows the drift, `k = v_assumed/v_true`
> rises, and A moves least because A was the dry line (`V_DRIFT_PRIOR`). Three
> checks, and it does not hold.
>
> **1. The high voltage never moved.** Pulled `hv_monitor.csv` per sub-run from
> `/eos/experiment/ntof/data/x17/july_beam/runs` for 14 runs spanning the
> block: drift **700 V on all four chambers** and mesh **540/540/524/520**,
> bit-identical from run_104 to run_162. The field is not the cause.
>
> **2. The drift END TIME barely moves, and it is the one observable that
> cannot lie about this.** `drift_t_end_ns` is read straight off the waveform
> fit and `k` never touches it. A genuine 7-8 % fall in `v_true` must raise it
> by 7-8 %. Measured on unrailed gated tracks:
>
> | arm | outside | inside | shift |
> |---|---:|---:|---:|
> | A | 830.6 ns | 835.6 ns | **+0.6 %** |
> | C | 884.4 ns | 891.8 ns | **+0.8 %** |
> | D | 915.0 ns | 924.0 ns | **+1.0 %** |
>
> One percent, against the seven the `k` shift demands. **The gas hypothesis
> is falsified.**
>
> **3. The scale-free source position moves less than a millimetre, and
> incoherently** -- A +0.19 mm, C +0.54 mm, D -0.83 mm, C and D in opposite
> directions while their `k` moved the same way. Not that either.
>
> **What DID move: the illumination.** Chamber A's median `x_local` steps from
> **+3.4 mm to +7.6 mm** inside exactly that window (C +1.1 mm, D -0.9 mm),
> while the raw angular distribution is flat to 0.3-2.5 %. So the excursion
> lives in the pointing geometry the estimator reads, not in the chamber.
> **Correlation, not mechanism -- this is a lead for October, not a result.**
>
> **The consequence, and it is not small.** If the excursion is an estimator
> artefact rather than a velocity change, then **applying per-run `k` inside
> runs 128-147 injects an ~8 % angle error rather than removing one.** The
> build now running stamps `k_source` on every row, so this is recoverable
> either way -- but the block is the first thing to settle before any
> opening-angle spectrum is drawn from those nine runs.

> ## Per-track tracking QA -- three things the medians were hiding -- 2026-09-10
>
> New: `tracking_qa.py`, `make_tracking_qa_figures.py`,
> `make_tracking_qa_report.py`. Every tracking number quoted before this was a
> campaign median. These profile the DISTRIBUTIONS at three levels -- per arm,
> per (run, arm) and per (tag, arm), ~3 150 tags, each a few minutes of beam --
> in quantiles, never means, because `q_total` reaches 1e34 in this table.
> Report at `<out>/tracking_qa/report.html`.
>
> **1. One gated track in four carries a charge that cannot be real.**
> **25.6 %** of tracks have `q_total` > 1e6 ADC and the p95 reaches
> **1e14-1e17** on a 12-bit ADC over a ~330-count pedestal. `q_total` is
> `x_q_sum + y_q_sum` and both plane sums diverge together, so it is the fit's
> amplitude solution running away on some depth bins, not a units error. Their
> chi2 is unremarkable, which is why nothing caught it -- the waveform still
> fits. **Every charge-based statement in this analysis runs on a column with a
> 25 % tail of garbage**: gain comparisons, `q_per_len`, and the charge
> percentile window `k_arm` itself cuts on.
>
> | | A | B | C | D |
> |---|---:|---:|---:|---:|
> | charge > 1e6 ADC | 0.282 | 0.265 | 0.197 | 0.281 |
> | chi2/dof > 100 | 0.019 | 0.050 | 0.052 | **0.260** |
> | >= 200 strips in one track | 0.035 | **0.113** | 0.038 | 0.000 |
>
> **2. Chamber A has a good population the others do not have at all.** A's
> chi2/dof is BIMODAL -- a clean peak at ~1.3 and a lower quartile of **1.65**
> -- against B 9.7, C 7.0, D 7.7. Same split as the angle scale: A is the only
> arm whose forward model describes its data and the only one whose `k` is
> stable. One fact, not two.
>
> **3. The 27 July access is a STEP, not a drift -- except on D.** A Spearman
> across the whole campaign scores that step as a strong trend; split at the
> access and almost everything flattens. `drift()` now reports both rhos and
> the difference between them is the result. The survivor is **chamber D's
> y-view chi2/dof, rho = +0.33 after the access over 2 796 tags, p = 7e-74**,
> climbing 33 (pre-access) -> 70 (post) -> **81** (last tenth). **D's fits get
> worse as the campaign runs.** Nothing else exceeds rho_post 0.20.
>
> The step itself reproduces the known hardware: A's dropped strips 41.5 -> 5
> and its chi2/dof 8.5 -> 5.3 across the access, which is the dead x-view
> connector being repaired. run_79 and run_81 dominate the outlier list for
> exactly that reason -- a check that the method finds what it should.
>
> ## A geometric bound on k, and arm A FAILS IT -- 2026-09-10
>
> `tracking_qa.gap_check`. This owes nothing to the pointing estimators and so
> checks them. `drift_len_mm = t_end * v` with `v = 42.6/k`, and the depth grid
> stops at 18 x 60 = **1080 ns**, so the deepest span the reconstruction can
> produce is fixed once `k` is -- and it has to fit in the drift gap:
>
>     k  >=  1080 * 0.0426 / gap_mm
>
> | arm | gap | k applied | v | deepest possible | span p50 | p50 unrailed | **past the gap** | k needed |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|
> | A | 27.9 | 1.266 | 33.6 | 36.3 | 34.3 | 28.3 | **0.741** | **1.649** |
> | B | 30.5 | -- | -- | -- | -- | -- | -- | 1.508 |
> | C | 30.0 | 1.616 | 26.4 | 28.5 | 26.9 | 23.7 | 0.000 | 1.534 |
> | D | 30.0 | 1.767 | 24.1 | 26.0 | 26.0 | 23.1 | 0.000 | 1.534 |
>
> **C and D pass. A does not: 74 % of its gated tracks reconstruct deeper than
> its own gap, and its median UNRAILED span is still past it.** A's `k` would
> have to be >= 1.649 (>= 1.534 on the 30 mm the run config records instead of
> the 27.9 mm `BEAM_DETS` carries) for its deepest track to fit in the gas. The
> pointing estimators say 1.14-1.27 -- **low by 25-30 %**.
>
> **Two readings, both calibration faults, and this test cannot choose.**
> Either A's angle scale is ~30 % too small, or A's depth-grid origin sits
> outside the gas and the span is inflated with `k` innocent -- and that is the
> same fitted `t0` whose median moves ~58 ns (one full sample period) between
> runs on D. What it does settle: **arm A, the arm this analysis has been
> treating as its reference, fails an independent geometric check that C and D
> pass.** October should start on A, not assume A.
>
> **What this does NOT do: it does not explain the run-to-run `k` scatter.**
> Nothing in the per-run distributions moves in a way that tracks it. The
> charge blow-up and D's degradation are real and are, on this evidence, not
> the cause.

> ## The angle scale varies RUN TO RUN, and it is not the sample -- 2026-09-10
>
> The run_86-vs-run_145 test above framed this as two runs disagreeing. With
> the four other calibrated runs read off `<out>/kcal`, the picture is sharper
> and worse: **run_145 is the outlier, and the other runs agree with each
> other.**
>
> | arm | run_104 | run_124 | run_156 | run_162 | run_86 full | **run_145** |
> |---|---:|---:|---:|---:|---:|---:|
> | A | 1.175 | 1.154 | 1.167 | 1.141 | 1.229 | **1.266** |
> | C | 1.425 | 1.375 | 1.375 | 1.383 | 1.156 | **1.616** |
> | D | -- | 1.550 | 1.650 | 1.553 | 1.341 | **1.767** |
>
> **The sampling explanation was tested and does not hold.** run_145's `k` came
> from its blind FULL pass; the other four came from the prescaled CALIBRATION
> pass (`<out>/fullpass_calib`, SINGLE at 0.25), so the samples were not
> comparable and that was the obvious suspect. Both trees exist for run_145, so
> the test is direct: take run_145's full-pass pointing-coincident sample,
> restrict it to the event ids the calibration pass actually kept, re-run the
> same estimators.
>
> | arm | full pass, band | restricted to the calib selection | shift |
> |---|---:|---:|---:|
> | A | 1.306 | 1.298 | -0.6 % |
> | C | 1.786 | 1.755 | -1.7 % |
> | D | 1.914 | 1.841 | -3.8 % |
>
> The selection buys 2-4 %. The gap is 13-17 %. **The scatter is real.**
>
> **Nothing else in the chain is per-run, by construction.** Verified rather
> than assumed:
>
> | input | scope | varies run to run? |
> |---|---|---|
> | wft bundle (`c1`, `c2/c1`, `kY`, `tau_s`, `sigma_s`, `sigma_p0`, `Dp`, `w0`, `kw`, `dt_xy`, `t0_abs`, `dead`) | per ARM, one bundle for all 36 runs | no -- byte-identical |
> | `v_drift` | pinned to 42.6 um/ns on `run_beam_job.py`'s command line | no -- the bundle's own `v` is discarded |
> | gas, target, detector centres and orientations, DREAM `sample_period`/`n_samples`/`latency` | `run_config.json` | **no** -- identical run_79 -> run_162 |
> | `k` | per RUN, per arm, measured in situ | **YES -- this is the whole problem** |
> | hot-strip strata, imaging summary | per run | exist for **run_145 only** |
>
> So there is no known physical reason for `k` to move 15 %, and the one
> physical variable not yet checked is the **drift HV**: `hv_monitor.csv` is
> written per sub-run but only run_145's is mirrored locally. Pull the rest
> from EOS -- they are small, and it is the cheapest remaining discriminator.
>
> **Where the October effort should start: arm A works and the others do not.**
> Median reduced chi-squared on gated campaign tracks --
>
> | arm | chi2/dof x | chi2/dof y | strips in fit |
> |---|---:|---:|---:|
> | A | 6.0 | 5.9 | 22 |
> | B | 19.1 | 16.3 | 46 |
> | C | 16.8 | 16.5 | 37 |
> | D | 38.6 | 55.0 | 31 |
>
> **A is the only arm whose forward model describes its data, and the only arm
> whose `k` is stable.** That is one fact, not two.
>
> **Arm C is also on a different bundle generation from every other arm** --
> `calib_bundle_lp`, not `calib_bundle_r06` -- and its diffusion parameters are
> an order of magnitude off the fleet:
>
> | | A | B | **C** | D |
> |---|---:|---:|---:|---:|
> | `sigma_p0` | 0.425 | 0.393 | **0.039** | 0.434 |
> | `Dp` | 0.0144 | 0.0151 | **0.0016** | 0.0153 |
> | `c2/c1` | 0.60 | 0.60 | **0.82** | 0.60 |
>
> Those two set how much cluster widening the fit attributes to diffusion; if
> they are too small the model must explain the width with the transverse speed
> `w`, which is exactly what becomes the angle. C's clusters span 37 strips
> while its bundle carries almost no diffusion. **Candidate mechanism,
> UNTESTED** -- it is a hypothesis for October, not a result.
>
> **The decision, 2026-09-10 (Dylan).** Do not stop for this. Push the full
> pass through with per-run `k` where it certifies, build per-track tracking
> QA distributions run by run and tag by tag to look for outliers and drift,
> and take the real per-detector calibration up in October.

> **THE FULL PASS IS DONE -- 2026-09-10 07:19.** 22:27 to 06:31 on condor, then
> a 48-minute pull and unpack. **12 929 of 12 932 outputs, 0 problems on
> unpack, all 293 sub-runs present** in `<out>/reco_fullpass` (28 GB).
>
> | arm | full pass | allowlist pass | ratio |
> |---|---:|---:|---:|
> | A | 9 458 115 | 1 096 449 | 8.6x |
> | B | 9 377 791 | 1 175 912 | 8.0x |
> | C | 9 573 426 | 1 362 793 | 7.0x |
> | D | 19 356 728 | 1 406 033 | 13.8x |
> | **all** | **47 766 060** | **5 041 187** | **9.5x** |
>
> The only 3 missing jobs are arms A/C/D of the corrupt tag (arm B of it ran).
> They were released once, failed identically, and were retired as
> deterministic -- so campaign-wide this pass retired 3 jobs and lost nothing
> else.
>
> **Next, and NOT done because it is a calibration decision:** `k_arm` per run
> on the new reco, then `campaign_tracks --fullpass <out>/reco_fullpass`. The
> sample is now large enough to measure the angle scale per run instead of
> borrowing run_145's, which is what the 4-7 % systematic rests on.
>
> **Smoke gate PASSED, 22:27 CEST, and it was exact.** Four jobs (one per arm,
> run_145/stat090_0000 tag 000) reproduced the August blind pass **event for
> event** -- A 3233, B 3098, C 3256, D 6690 -- so the jobs are fitting the
> whole tag and no allowlist survives anywhere in the path. Wall clock 33-41
> min for A/B/C and 76 min for D (it seeds ~2x as often); memory 733-977 MB
> against the 2 GB request. **12 928 jobs then went in as cluster 4172031.**
> Tarballs are ~1.5-1.8 MB, so ~22 GB on EOS and ~40 GB unpacked locally
> (164 GB free).
>
> **The corrupt file is three files, not one -- correcting the entry below.**
> The campaign-pass note says the only data loss found campaign-wide was one
> `decoded_root` file, `run_104/stat090_0016` tag `260730_11H29_000` FEU 03.
> The full pass reads every arm of every tag and found **FEUs 02, 03 and 07 of
> that same tag all empty** -- full size on EOS (75-85 MB), zero ROOT keys, the
> identical signature. Arms A, C and D held on it; arm B ran fine. The
> allowlist pass saw only FEU 03 because it never asked the other two arms for
> that tag. One tag of one sub-run is still the whole of the campaign's data
> loss; it is three quarters of that tag rather than one quarter.
>
> **And a bug in the driver's own watch loop, found the same way.** Held jobs
> count as queued, so `wait until the queue is empty` never exits once a
> deterministic failure holds -- the loop would have spun until morning with
> the pass finished. Fixed to wait on running-plus-idle only, with a release
> pass and then an explicit record-and-clear of whatever stays held. The submit
> step also now refuses to run while anything is in the queue, because a job
> that is RUNNING has no EOS tarball yet and the done-list alone would submit a
> second copy of everything in flight.
>
> New: `--full-pass` in `condor/make_stage2_campaign.py`, plus
> `stage2_fullpass.sub` and `run_stage2_fullpass_wrapper.sh`. The allowlist
> path is untouched, so the existing pass stays reproducible.
>
> **Three things done differently from the allowlist pass, each for a measured
> reason.** Tags come from the **stage-1 candidate tables** (one row per
> trigger, so their distinct tags are the sub-run's tags) rather than from the
> allowlist, which only lists tags the filter selected. `JobFlavour` is
> **workday (8 h)**, not longlunch: these jobs are ~20x longer, a median near
> 40 min with a tail that would run past 2 h, and a job killed on the flavour
> limit is held and retried into the same limit. stderr is **sharded by run**
> into ~36 directories, because AFS caps entries per directory and a flat
> 22k-file log dir already degraded the shared schedd once.
>
> **Unchanged, and still the open question:** every arm is still seeded from
> its run_145 bundle with `v_drift` pinned at 42.6 um/ns, and the angle scale
> is still run_145's borrowed `k` with its 4-7 % systematic. The full pass
> multiplies statistics; it does not touch either.

> ## The stage-1 filter keeps an eighth of the real pairs -- and a full pass
> would not obviously help -- 2026-09-09
>
> **Asked because the campaign gave 8.2x run_145's A-C pairs, not the 50x the
> plan projected.** The cause is not stage 2, which behaved identically
> everywhere: every campaign run converts its `INTER` triggers into two-track
> events at 2.1-3.9 %, and run_145's own `INTER` triggers convert at 1.8 %. The
> difference is entirely **which events stage 1 offered it.**
>
> **Measured against run_145's blind full pass, the one run that has one.** Of
> the 595 triggers the full pass turns into two gated, angle-calibrated tracks
> within 30 mm of the axis, stage 1 put only **76 (12.8 %)** into the classes
> `allowlist.py` reconstructs in full. 355 sat in `NONE` and 161 in `SINGLE`,
> drawn at 1 % and 5 %.
>
> | stage-1 rule | triggers selected | two-track events captured |
> |---|---:|---:|
> | current (`INTER`/`INTRA`/`IMPLIED`) | 3.80 % | **12.8 %** |
> | `n_arms_loose >= 2` | 13.77 % | 47.7 % |
> | `n_arms_loose >= 1` | 57.77 % | 89.2 % |
>
> **No cheap retune closes it** -- 89 % capture costs 58 % of all triggers,
> which is a full pass in all but name. The earlier 98.3 % seeding number is
> not in tension: it measures whether stage 2 can see what stage 1 *chose*, and
> says nothing about what stage 1 discarded.
>
> ### The scintillator timing check -- and it argues AGAINST the full pass
>
> `accidental_timing.fit_by_stage1_class` (new), the same unbinned two-component
> MLE as the published `f`, split by whether stage 1 would have kept the event.
> The two-arm-tagged sample is small (76 pairs, 65 of them missed), so the
> **non-parametric cross-check carries the result** -- no KDE, no template fit:
>
> | population | n | median \|dt\| | within 30 ns |
> |---|---:|---:|---:|
> | prompt template (both arms born together) | -- | 15.2 ns | 0.82 |
> | **stage-1 selected** | 11 | 18.8 ns | **0.82** |
> | **stage-1 missed** | 65 | 33.0 ns | **0.46** |
> | accidental template | -- | -- | 0.37 |
>
> The pairs stage 1 keeps are **prompt**. The ones it misses are **not**:
> 46 % within 30 ns is inconsistent with pure prompt at p = 8e-11 and only
> marginally above pure accidental (p = 0.09). The two differ at p = 0.02
> (Mann-Whitney) / p = 0.03 (Fisher). Implied true-coincidence fraction of the
> missed population **0.20**, independently reproduced by the MLE at
> **f = 0.19 [0.07, 0.31]** (1.6 sigma from zero) against the pooled
> f = 0.29 [0.17, 0.40].
>
> **So a blind full pass buys ~6.6x more inter-chamber pairs of which ~80 % are
> not coincidences.** Folding both fractions through run_145's 61 selected and
> 405 missed inter pairs: true pairs rise 2.3-3.1x, purity falls 0.8 -> 0.27,
> and **S/sqrt(S+B) moves by 0.93x** -- i.e. roughly break-even, spanning
> 0.57-1.29x across the missed-side interval. **~18 000 core-hours (760 bought
> 4.1 % of the arm fits) for no clear gain in shape significance.**
>
> **The weak link is n = 11 on the selected side, and it is fixable for free.**
> The campaign already reconstructed `NONE` and `SINGLE` at 1 % / 5 % in all
> four arms and carries 151x run_145's `INTER`, so repeating this split
> campaign-wide needs no new reconstruction -- only pointing the slim read at
> the exported parquet instead of `read_slim`'s ROOT. **Do that before deciding
> on a full pass.**
>
> Not yet done: the accidental-timing page still shows the pooled fit only; the
> new `fit_by_stage1_class_run_145.csv` is written but not published.

> ## THE CAMPAIGN PASS IS DONE -- 2026-09-09
>
> | pass | result |
> |---|---|
> | stage-1 census | **293/293 sub-runs** |
> | stage-2 reco | **12,705** condor outputs |
> | calibration pass | **1,040** outputs (SINGLE prescale 0.25 vs the main 0.05) |
> | n_TOF slim | **293/293**, 9.6 GB local parquet |
> | **tracks** | **2,111,162 segments, 1,137,029 gated, 843 MB**, 287 sub-runs |
>
> 324 transient condor holds released, **0 retired as deterministic**, so the
> only data loss found campaign-wide is one corrupt `decoded_root` file
> (`run_104/stat090_0016` tag `260730_11H29_000` FEU 03 -- full size on EOS,
> zero ROOT keys).
>
> ### The angle scale: run_145's k, campaign-wide, with a stated systematic
>
> `campaign_tracks --k-from run_145` applies A=1.2662, C=1.6163, D=1.7667 to
> every run and stamps **`k_source`** on every row, so a borrowed scale can
> never be read back as a per-run measurement. 1,750,293 of 2,111,162 segments
> (83 %) carry a calibrated angle; B never certifies anywhere.
>
> **Why one k rather than per-run:** the calibration pass measured k on six
> runs spanning 27 Jul - 10 Aug. Arm A's spread is **3.7 %**, but the SAME run
> (run_145) measured from the calibration pass instead of its full pass differs
> by **-6.5 %**. The method-to-method offset is larger than the run-to-run
> drift, and every run's A sits below the full-pass value rather than
> scattering about it -- the signature of a systematic, not of drift. So k does
> not meaningfully drift, the estimator is sample-dependent, and the price of
> one k campaign-wide is a **~4-7 % systematic on the angle scale**.
>
> **This is being tested, not assumed.** run_86 (8.9 days before run_145, same
> side of the 27 Jul access) has been downloaded in full (37 GB, 4 sub-runs)
> and `unattended_2026-09-09.sh` is running a LOCAL FULL PASS on it to produce
> its full-pass k. If that sits ~6 % above run_86's calibration-pass k
> (A 1.184, C 1.350) the offset is method; if it lands near it, run_145 is the
> outlier and using its k everywhere is wrong.
>
> ### Two errors made and fixed the same day, recorded because both were silent
>
> 1. **The published run_145 calibration was deleted** -- the calibration loop
>    ran `rm -f k_arm_<run>.json` per run and included run_145, overwriting
>    certified values with an uncertified calibration-pass result. The first
>    `--k-from` rebuild consequently produced **0 calibrated angles**. Restored
>    from the full pass, reproducing A=1.2662, C=1.6163, D=1.7667 exactly.
> 2. **run_145 was double-counted.** It carried BOTH its original full-pass
>    `events_prelim.candidates.parquet` and the campaign per-tag files, and
>    `build_tracks.load_reco` globs `events_*.candidates.parquet`, matching
>    both. 192 campaign per-tag files removed, keeping the full pass (a strict
>    superset). Its segment count rising 74,187 -> 78,403 was this bug, NOT the
>    campaign adding tags as first reported.

> ## The campaign pass -- IN FLIGHT on condor, 2026-09-09
>
> **The overnight run of 2026-09-08/09 did not happen.** The `/loop` that was
> supposed to carry it re-arms only if `ScheduleWakeup` is the last action of
> each turn, and on the first tick it was not called, so the loop died after
> one iteration. Stage 1 was still 11/293 and no condor job had ever been
> submitted. Recorded because the failure is invisible from the products.
>
> **Stage 1 moved from the desktop to condor.** `campaign_census.sh` streams
> each sub-run from EOS and runs locally -- its own header puts that at 17-25 h,
> "a multi-night job". The data is already at CERN and `paths.py` resolves every
> root through an environment variable, so the *unmodified* CLI runs on a worker
> at 29.3 ev/s (~1.2 h per sub-run), all sub-runs in parallel.
>
> | pass | cluster | jobs | state |
> |---|---|---|---|
> | stage 1 census | 4141478, 4141479 | 282 | running |
> | n_TOF slim -> parquet | 4141481 | 293 | queued |
> | stage 2 reco | not yet submitted | ~7 800 | package built + smoke-tested |
>
> New, in `sept26_prelim_analysis/condor/`: `make_stage1_package.py`,
> `make_stage2_campaign.py`, the two wrappers and submit files, and
> `fetch_stage1.sh` / `fetch_stage2.sh`. Plus `slim_export.py`.
>
> **Four bugs the smoke tests caught before they reached the fleet:** xrootd
> needs `root://host//eos` (a single slash is read as a relative path and
> refused); `ntof_tracking.reco.io` resolves through
> `common/beam_july_paths.py::X17_BEAM_JULY`, which wants the PARENT of `runs/`,
> not `runs/` itself; rsync of `lxplus:~/dir` expands `~` LOCALLY; and the
> LCG_105 pyarrow is built without the zstd codec, so the slim export writes
> snappy.
>
> **The one calibration assumption, stated because it is not free.** Stage 2
> seeds every arm from its run_145 bundle with `v_drift` PINNED at 42.6 um/ns
> (Magboltz), exactly as the published run_145 bundles were -- so no run_145 gas
> fit travels into another run, and the whole in-sample set is one noise
> configuration and one gas mixture (Ar/Iso 90/10, 36 runs, checked in
> `sample.csv`). Per-run gas variation is absorbed downstream by `k_arm`.
> **The falsifier: if `k` varies strongly run to run, one bundle is wrong and
> stage 2 must be re-cut by condition.** Nothing in this pass tests that; it has
> to come from the campaign `k_arm` output.
>
> ### Tight scintillator coincidence -- built, and it found something
>
> `tight_coincidence.py` + `make_tight_figures.py`, downstream only: the
> production `DT_WINDOW` is UNCHANGED, so the window stays re-tunable offline
> without another campaign pass. A pair is coincident when each arm is within
> +-30 ns of its own trigger AND the two arms are within 20 ns of each other.
>
> **The cut ENRICHES a single-particle background.** On run_145, 5 of the 20
> survivors are opposing-chamber (A-C) pairs above 170 deg -- one particle
> crossing the target and punching through both opposite chambers, which is
> perfectly time-coincident *because it is one particle*. 8 of the 11 tight
> opposing pairs are above 150 deg, against 25 % of the loose sample. This is
> `HANDOFF_ACCIDENTAL_TIMING.md` sec 5's D12 caveat arriving in the data.
> Flagged as `back_to_back` / `tight_pair`, never silently dropped -- it is also
> the cleanest back-to-back calibration line available.
> **Any X17 statement from a timing-coincident sample needs this veto first.**
>
> Two things run_145 cannot settle, both waiting on campaign statistics: at
> n = 20 the shape is indistinguishable from both nulls (chi2/dof 1.32 against
> the loose sample, 1.23 against event-mixed); and the topology asymmetry the
> cut predicts -- it should bite harder on perpendicular, where the accidental
> fraction is higher -- is NOT visible (0.262 against 0.265).
>
> One known non-independence, recorded not fixed: `source_imaging.vertices`
> gives a multi-track event one row per track combination, so those rows share
> a single scintillator tag. Two such events in run_145's 76; campaign-wide it
> has to be handled before the tagged pairs are treated as independent.

> ## S1-S4 are DONE and published -- 2026-09-08
>
> All four workstreams of PLAN.md §10 ran end to end on run_145 and shipped a
> generated page. `rerun_chain.sh` now rebuilds and republishes all four in
> dependency order.
>
> | | result | page |
> |---|---|---|
> | **S1** | the scintillators are a **filter** (three boolean roles, three branches) and nothing else. The wall is read at **both ends**, and that gives position along its 500 mm bars to **σ_y < 53 mm** -- 9× better than which group fired. Both estimators agree with each other at r = −0.69 to −0.83 in all four chambers, **B included**, with no Micromegas in the argument. | [`x17/scintillators/`](https://dylan-neff.web.cern.ch/x17/scintillators/) |
> | **S2** | the capsule is at **X = −8.4, Z = −4.1 mm**, 9.3 mm off the beam axis and **inside its own r = 10 mm bore**, chamber-to-chamber spread ±0.6 mm. y is +22 mm and much weaker (σ_y 44-109 mm). | [`x17/source-imaging/`](https://dylan-neff.web.cern.ch/x17/source-imaging/) |
> | **S3** | v falls monotonically **down the gas line** at the same 700 V in every chamber: 33.6 → 26.4 → 24.2 µm/ns = **0.37 / 0.67 / 0.77 % H₂O**. O₂, air and N₂ are all excluded. | folded into [`x17/reco-funnel/`](https://dylan-neff.web.cern.ch/x17/reco-funnel/) |
> | **S4** | the pair generator reproduces the full Geant4 sim to **KS = 0.003**; the acceptance toy collapses 6× across the middle; and **the measured pairs follow the event-mixed shape** (χ²/dof 1.8-3.1) rather than any pair spectrum (6.7 and worse). | [`x17/opening-angle/`](https://dylan-neff.web.cern.ch/x17/opening-angle/) |
>
> ### Three corrections this phase forced
>
> 1. **Chamber D was never 40 mm off axis.** The −48/−36/−36 mm came from a
>    cached `imaging_summary.json` that went stale under the y sign fix. The
>    identical estimator on the current reconstruction gives −10.0/−9.6/−6.1,
>    and −4.3 ± 0.5 with the charge window. `k_arm` now **computes** the
>    crossing instead of reading it; `make_funnel_report` reads that.
> 2. **Chamber B does contribute an alignment point.** The crossing needs no
>    angle scale, only where the reconstructed angle is zero, so B gives Z at
>    ±2 mm. Z is cross-checked, not single-sided.
> 3. **N₂ cannot explain any chamber's drift velocity.** 5 % N₂ reaches only
>    35.2 µm/ns, above all three.
>
> ### The autonomous shift of 2026-09-08 — threads (a), (b), (c)
>
> | | result |
> |---|---|
> | **(a) D's angle scale** | **survives.** `k_robustness.py` re-measures k under four sample cuts against a threshold declared first (the baseline estimator half-spread). D moves 3.65 % against a 5.1 % tolerance. **The hot cells are the whole story** — they remove 22.4 % of D's coincident sample and are the only cut that moves k, and they move it toward a *better* calibration (spread 0.103 → 0.056). The outer ring is already gone: 42 % of D's fitted clusters sit there but only 6.7 % of its coincident ones. Dead channels remove 0.2 %, because a dead channel produces no track. |
> | **(b) the normal-incidence hole** | **does not exist.** `TAN_MIN_SLOPE` sets a flag and gates nothing in this chain — verified in `wft/reco.py`/`wft/compat.py` (no caller passes `require_slope=True`, and every caller is a June bench package) and in the data (23-30 % of gated tracks carry the flag False and are all still there). The real cost is **resolution**: if a reliable slope were ever required on both planes of both legs it would take **~50 % of pairs**, near-flat across topologies. `normal_incidence.py`. |
> | **(c) chamber B** | characterised in **both** in-plane coordinates with no angle anywhere. Position information real (lift 1.45×) but **not better than D** — that claim is withdrawn above. What separates B is **charge density: 0.45× A's**, the widest and most dilute clusters of the four, which is what a fringing field predicts. `chamber_b.py`. |
>
> **One new systematic on S4, from (b) — and it is now measured, not inferred.**
> The acceptance toy predicts the per-leg head-on fraction from geometry alone
> and lands within ~25 % for A and C, but the data has *fewer* head-on tracks
> than geometry allows. That pointed at an incidence-dependent reconstruction
> efficiency, and `normal_incidence.efficiency_vs_incidence` measures it
> **without using the Micromegas at all**: the fired wall group supplies the
> abscissa, since the four groups sample tan = −0.53, −0.23, **+0.08**, +0.38
> and the third is head-on. The head-on group is *interior*, with a neighbour
> either side, so a surface effect would interpolate between them:
>
> | arm | seeding vs neighbours | tracking vs neighbours | below both? |
> |---|---:|---:|:--:|
> | A | 1.01 | 0.84 | yes |
> | B | 0.93 | 0.86 | yes |
> | C | 0.96 | 0.67 | yes |
> | D | 0.98 | 0.76 | yes |
>
> **Below both neighbours in 4 of 4 chambers, median tracking ratio 0.80 — and
> seeding is flat at 0.97.** The chamber sees the head-on particle; the fit is
> what fails to return a gated track, which is the mechanism expected when a
> charge column arrives at every strip at once.
>
> **`acceptance.py` applies efficiency independent of incidence, so it
> over-counts head-on legs by roughly the reciprocal of those ratios** — a real,
> θ-dependent bias on the opening-angle acceptance. Not corrected: correcting it
> needs the efficiency map to become 2D in (position, incidence), which is more
> than run_145 supports. D's own head-on surplus (1.33× after hot cells) is
> separate and stays unexplained.

> ### Two handoff documents, written 2026-09-08
>
> | | |
> |---|---|
> | [`HANDOFF_ACCIDENTAL_TIMING.md`](HANDOFF_ACCIDENTAL_TIMING.md) | the scintillator timing test of the S4 null. **Half done already**: in two-arm events one arm is the trigger (median \|Δt\| 7 ns, against a 5.2 ns single-arm reference) and **the other fires at a random time — median 171 ns, 47 % beyond 200 ns**. True-coincidence fraction **≤ 6 ± 3 %**. Also finds the accept window is mis-centred, and that `is_control` is an unused flat accidental sample. |
> | [`HANDOFF_D_NOISY_CHANNELS.md`](HANDOFF_D_NOISY_CHANNELS.md) | identify D's noisy channels and make them reconstruction wildcards. They are **not discharges** (median charge ratio 0.94) but wide, dilute, low-density clusters; they are **whole x columns spanning the full plane height**, i.e. bad channels not bad regions; and they sit on the **49.8 mm connector boundaries** — the same fault class as D's dead runs. |
>
> **`source_imaging.vertices` had a latent bug, now fixed**: the pair frames
> carried only `key`, the *first* track's event id, so a mixed pair silently
> claimed its second track belonged to the first one's event. They now carry
> `key1` and `key2`. No published number changed — nothing downstream read
> `key` — but the timing study reads it, and it burned an hour.

> ### HANDOFF_ACCIDENTAL_TIMING.md, item (a) done — the true-coincidence
> fraction is measured — 2026-09-08
>
> Redone per the handoff's own §4(a): unbiased hit choice instead of
> "closest to dt_ns = 0" (sec 3.1's own bias, which the first pass carried),
> the full ±1000 ns range, and an unbinned two-component MLE fit against
> `is_control` (sec 2.2's unused accidental control) instead of matching one
> summary statistic to a single-parameter fold. New module
> `accidental_timing.py`.
>
> **f = 29 % [17, 40] overall, 42 % [26, 58] opposing (A–C, the signal
> topology), 16 % [0, 32] perpendicular** — well above the first pass's
> 6 ± 3 % upper bound, and topology-ordered the way a real source predicts
> (opposing carries ~2.7× the perpendicular fraction). The direction reversal
> is methodological, not a sign the stated bias was wrong about its own
> direction: both numbers are on record and the difference is explained on
> the page rather than quietly superseded.
>
> Also delivered: item (c), the accept window's purity/width trade-off — the
> production window (−100, +60) ns sits at 88 % peak/pedestal purity where a
> centred (−20, +20) reaches 95 % — **not applied**, since `DT_WINDOW` is
> shared by `candidate_filter.py`/`efficiency.py`/`scintillators.py` and
> changing it touches the stage-1/stage-2 production chain, on Dylan's hold.
> One more finding from the same pass: under the production window, a
> wall+plastic "coincidence" is nearly automatic — only 1 of 147 216
> single-active-arm events fails to show one, because the window is wide
> enough that the plastic family's own accidental rate lands something in it
> almost every time. The peak's own core (−30, +30) ns is what recovers a
> real "coincidence vs. one element" comparison.
>
> Still open, in the handoff's own order: item (b), feeding f back into the
> opening-angle spectrum — the measured f covers only the 16 % of real
> inter-chamber pairs that carry a two-arm scintillator tag at all, so
> applying it to the rest needs an unchecked representativeness assumption,
> not done here; item (d), intra-chamber pairs (needs the wall's along-bar
> position to separate two legs in one arm, untested); item (e), the
> Micromegas `t0` cross-check (needs (a) fixed first, since `t0` carries the
> drift depth). `HANDOFF_ACCIDENTAL_TIMING.md` updated with the full result
> and the next-session priority list. Published:
> <https://dylan-neff.web.cern.ch/x17/accidental-timing/>

> ### HANDOFF_D_NOISY_CHANNELS.md, steps 1-3 done — 2026-09-08
>
> `noisy_channels.py` is the standalone per-strip classifier the handoff's
> §4 asked for, built and run on run_145 **before touching the
> reconstruction**. Per (arm, plane, channel) from raw `combined_hits` —
> honest, pre-fit occupancy, not `events_prelim.x_p0/y_p0` like
> `k_robustness.hot_cells`/`source_imaging.dead_ranges` — with a local
> (64-channel = one connector) median so the plane's own illumination lobes
> are not flagged. Three classes: `dead`/`hot` from occupancy alone,
> `noisy` from a hits-level cluster-shape check (`wft.seed`'s production
> significance-floor + gap clustering — no waveform fit).
>
> **Confirms the handoff's finding independently, at channel granularity**:
> hot-channel fraction lands within a point of `k_robustness`'s 2D-cell
> numbers on every chamber (A 1.4/3.1 % x/y, B 0/0.2 %, C 0/0 %, D 8.2/11.1 %
> — vs. k_robustness's A 2.1, B 1.5, C 1.9, D 10.75 %), **with HOT_FACTOR = 5
> reused unchanged from k_robustness** — nothing was fit to make the numbers
> agree. D's flagged channels carry 56-70 % of its own hits despite being
> 8-11 % of the plane (echoes the handoff's 80.6 %-of-clusters-in-10.8 %-of-
> plane), and the widest single band is 14 channels sitting exactly on the
> 448-462 connector boundary. The `noisy` shape class is real but small
> (0-2 % of channels) and not yet validated the way `hot` is.
>
> **A second tuning pass found and fixed two real bugs in the shape (`noisy`)
> pass, both upstream of this module.** (1) The DAQ's `eventId` resets every
> subrun, so concatenating subruns and grouping on `eventId` alone silently
> merges unrelated events — `noisy_channels.py` now carries a `subrun` column
> and groups on `(subrun, eventId)` everywhere. (2) Raw `combined_hits` is
> dominated by a documented ~1.2-1.3 MHz coherent-band noise residual
> (`ntof_tracking.reco.noise.py`) that `wft.seed`'s own significance floor
> does not remove — it is relative to the event's own max, and a coherent
> band shares one amplitude scale across hundreds of strips at once. One
> inspected A-x "track-like" event had hits over channels 57-509 at near-
> uniform amplitude, and the 12 mm gap threshold merged all of it into one
> supercluster. Before both fixes the median hits-level cluster width on
> clean A-x was 45-110 strips (nearly a quarter of the plane); after running
> hits through `noise.flag_noise` (the existing production filter) and
> restricting to track-like events (`{arm}_trk_x/y >= 1` from
> `candidate_filter`'s own per-plane classification), it is 18 — physically
> sane, and the flagged `noisy` sets got smaller and tighter (A-x: 9 scattered
> channels → 2; D-y: 7 bands → 3). `dead`/`hot` were unaffected — they only
> ever used the full, unselected occupancy. **Even after the fix, `noisy` stays
> the weaker class**: its flagged count is sensitive to its own threshold
> (27 → 2 → 0 channels on A-x across a modest (width, density) factor sweep),
> and on D-x — the plane the handoff's own shape evidence is actually about —
> it flags nothing at any threshold tried, because `hot` already accounts for
> the shape-anomalous channels there. Read `noisy` as "worth a second look",
> not validated the way `hot` is (which matches `k_robustness`'s independent
> measurement to ~1 point on every chamber, unchanged by this pass).
> `load_hits` now costs ~15 min for the default 3 subruns (`noise.flag_noise`
> is an unvectorized per-event pass) — was ~1 min before the fix.
>
> Output: `<out>/noisy_channels/noisy_channels_run_145.csv` (per-channel) and
> `noisy_channels_summary_run_145.csv` (per arm/plane), wired into
> `rerun_chain.sh`.

> ### §3.2/§4 step 4 done — the `hot` wildcard is wired into `wft/`, not yet re-run — 2026-09-08
>
> `dead` already had exactly this plumbing (T1.3): `CalibrationBundle.dead` ->
> `wft.model.DEAD` -> `prep_plane` censors those rows entirely. Dylan's spec
> for `hot` is different on purpose — a hot channel DOES carry signal, so it
> must stay in the fit, just capped — which meant a second, parallel
> mechanism rather than reusing `dead`'s:
>
> | spec item | where | mechanism |
> |---|---|---|
> | never seed on a flagged channel | `wft/seed.py` `seed_candidates` | ranking + the `min_strips` admission test now use the CLEAN (non-hot) strip count, not the raw one — a cluster made entirely of hot strips can't outrank or become a seed. Membership is untouched: a hot strip inside a real cluster stays in `Seed.channels`, so gap-clustering still bridges across it and a track is never split or lost for crossing one. Threaded through `wft.reco.reconstruct_run` and `ntof_tracking.wft_beam.seeds_from_hits_beam`/`reconstruct_subrun` (`cal.hot`). |
> | keep it in the fit, down-weighted | `wft/model.py` `prep_plane` | `HOT` (mirrors `DEAD`, from `cal.hot`) inflates that row's noise by `HOT_NOISE_INFLATION` (10x, **not tuned yet** — first cut) instead of `DEAD`'s 1e9; `sat` is left alone, so the row keeps its dof, unlike `dead`. |
> | cap its influence | same `HOT_NOISE_INFLATION` | bounded, not infinite, by construction — the row survives, its pull on the NNLS profile is 1/100 of an unflagged row's. |
> | record it | `wft/reco.py` `PlaneFit.n_flagged_strips` | dead+hot channels in the fit window, per plane, in every track row (including the null-fit row, so the column is never missing). |
>
> `CalibrationBundle` gained a `hot` field (mirrors `dead`, same save/load
> round-trip). All locally verified, no condor needed: `wft/tests/
> test_hot_mask.py` (new, 4 checks — down-weighted-not-censored, bundle
> round-trip, an all-hot cluster is rejected as a seed, a track crossing one
> hot strip stays one cluster with all its channels) plus the full existing
> suite, **23/23 passing** (`.venv/bin/python -m pytest wft/tests/`).
>
> `sept26_prelim_analysis/apply_hot_wildcards.py` builds the concrete D/
> run_145 case: it re-derives `calib_bundle_prelim` (the exact bundle the
> frozen tracks were built from — that bundle itself lived on a condor worker
> and is gone, but `wft_beam.make_bundle` is a pure function of the bench
> source bundle + `run_config.json`, and the script verifies the re-derived
> hyper/v_drift/sat_adc against the frozen run's own `events_prelim.meta.json`
> before trusting it — **caught a real staleness bug in the process**:
> `wft_beam.V_DRIFT_PRIOR['D']` is now 36.0 µm/ns, but the frozen run_145/D
> reco actually used 42.6 — the table moved after that reco ran, and
> re-deriving without checking would have silently shipped a bundle that
> differs from baseline in v_drift AND hot channels at once, confounding any
> comparison), attaches the 42 x / 57 y hot channels from
> `noisy_channels.py`, and writes `calib_bundle_hotmasked` alongside the
> untouched baseline — never overwriting it, since a bundle is per detector
> **and** per run condition.
>
> **Not done, deliberately**: actually launching the re-reconstruction. That
> needs `ntof_tracking.wft_beam reco` on lxplus/condor — remote compute with
> real wall-clock cost — so it was left as an explicit next step rather than
> triggered automatically. Once run, compare against the frozen `run_145/D`
> track table on the success criteria HANDOFF_D_NOISY_CHANNELS.md §3.3 already
> declared (angle-scale spread toward ~0.056 without discarding 22 % of the
> sample; the head-on excess closes; cluster width falls toward A's 25; A/C
> must not move; gated-track COUNT goes up, not down).

> ### The re-run happened — and the wildcard, as first tuned, makes D WORSE — 2026-09-08
>
> `calib_bundle_hotmasked` ran on lxplus/condor (cluster 4141149, 7 jobs, all
> D/run_145/stat090_0000, no allowlist — same full reconstruction as the
> frozen baseline, +hot wildcards). All 7 completed clean, no holds. Merged
> with `merge_beam_tags.py` into a dedicated directory, never touching the
> frozen products. Direct comparison against the frozen table, same
> event_ids:
>
> | | frozen (no wildcards) | hot-masked |
> |---|---:|---:|
> | rows (events attempted) | 46 218 | 33 393 |
> | both-plane fit converges | 72.6 % | 33.4 % |
> | quality_ok (both planes) | 67.4 % | 29.4 % |
> | median x_chi2/dof | 10.7 | 41.9 |
> | median x_n_strips | 42 | 32 |
>
> Of the 33 393 events common to both tables: **0 gained a good fit, 16 152
> lost one that the frozen run had.** This fails §3.3's own criterion
> outright (gated tracks must go UP, not down) — the wildcard as built makes
> the reconstruction substantially worse, not better.
>
> **Two candidate causes, not yet disentangled — both are tuning choices
> flagged as unvalidated when they were written, now confirmed to matter**:
>
> 1. **HOT_NOISE_INFLATION = 10.0 is too weak.** Splitting the hot-masked
>    table by whether a window touches a flagged strip: windows that do
>    (84.6 % of attempted events — far more than the 8-12 % raw flagged-
>    strip rate per window, because D's hot channels are common enough that
>    most windows touch at least one) show median chi2/dof 45.6; windows
>    with none show 6.6, close to baseline. A hot channel's actual amplitude
>    excursion looks large enough that 10x noise inflation still lets it pull
>    real chi2 weight — it needs to be inflated much harder, or the
>    down-weighting approach needs rethinking.
> 2. **The seeding admission rule may be too strict.** `seed_candidates`
>    rejects a cluster when its CLEAN strip count is below `min_strips`
>    (5 for beam) — but "never seed on a flagged channel" (HANDOFF_D_NOISY_
>    CHANNELS.md item 1) most plausibly means a cluster made ENTIRELY of
>    flagged strips, not one that merely falls a strip or two short of
>    `min_strips` once its flagged strips are discounted. In a chamber with
>    D's hot-channel density, a real 5-6 strip cluster grazing 1-2 hot
>    strips now loses its seed ENTIRELY (46 218 -> 33 393 attempted events,
>    -28 %) rather than being kept and down-weighted — arguably the opposite
>    of "a track should never be lost because it crossed a bad channel."
>
> Neither of these is a code bug — both are exactly the "first cut,
> not tuned" choices flagged when they were written (`wft/model.py`
> `HOT_NOISE_INFLATION`, `wft/seed.py`'s clean-count admission threshold).
> They need retuning (and probably re-deriving from data rather than
> guessed constants) before this wildcard is a net improvement. **Not done
> without checking in first**: iterating the inflation factor and/or the
> seeding threshold burns another condor cycle (~15 min) each time, and the
> right fix changes what the mechanism actually does, which is worth a
> second opinion before spending more of it blindly.
>
> **Handed off**: [`HANDOFF_HOT_WILDCARD_TUNING.md`](HANDOFF_HOT_WILDCARD_TUNING.md).
> Dylan's instruction for the next pass: tune locally, event by event --
> `combined_hits_root` AND `decoded_root` for run_145/D are both already
> staged locally, so building a handful of real fit windows and scanning
> `HOT_NOISE_INFLATION` / the seeding admission rule against them needs no
> condor at all. Only re-run the condor cluster once a local check looks
> right. `compare_hotmasked_rerun.py` (new) is the frozen-vs-rerun comparison
> tool, reusable on any `events_prelim.parquet`.

> ### RESOLVED: it was never a regression, and the cut belongs downstream — 2026-09-08
>
> Tuned locally, event by event, as instructed — and **no condor at all**. The
> local harness reproduces the frozen products *bit for bit* (`x_ok` agrees on
> 100.0 %; `x_p0`, `x_tan_theta` and `x_chi2` identical to the last digit on
> 99.8 % of fitted events), so a configuration costs ~5 min on 8 cores against
> a ~15 min condor round-trip, and it can be diffed per event.
>
> **The frozen D sample is two populations and the mask separates them.**
> 29 % of D's triggers have a largest x cluster made ENTIRELY of flagged
> channels, and those noise columns **fit better than real tracks do**:
>
> | stratum of the baseline x cluster | n | median χ²/dof | quality_ok | fitted p0 |
> |---|---:|---:|---:|---|
> | **all-hot** (zero clean strips) | 1 948 | **1.27** | **99.6 %** | 65 mm, IQR 142 |
> | mostly-hot (<50 % clean) | 1 853 | 20.2 | 93.4 % | 137 mm, IQR 308 |
> | clean (≥50 % clean) | 2 339 | 22.5 | 94.6 % | 257 mm, IQR 228 |
>
> A smooth, wide, dilute coherent-noise deposit is *easy* for the forward
> model; a real track is not. The all-hot fits pile up at p0 ≈ 65 mm — the
> hot band itself — while real tracks spread across the plane. So the frozen
> table's 72.6 % convergence and 10.7 median χ²/dof were being held **up** by
> the noise, and the table above is the whole of "0 gained, 16 152 lost".
>
> **§3.3's criterion "gated tracks in D go UP" is struck.** The flagged
> channels do not cost D tracks; they manufacture 29 % of them. The other four
> criteria stand and are what was used.
>
> **Both suspected causes answered, both negative:**
>
> 1. **`HOT_NOISE_INFLATION` is irrelevant.** 10 / 30 / 100 over one fixed set
>    of 3 600 real triggers agree to three decimals on every metric in every
>    stratum. Paired per event, a window whose *seed* is unchanged fits
>    identically with the mask on (χ²/dof 39.45 → 38.54, median |Δp0| **0.00
>    mm**). Everything the wildcard does, it does through seeding. The
>    "84.6 % of windows touch a flagged strip → χ²/dof 45.6" reading above was
>    the stratum effect, not a weighting effect.
> 2. **Seeding is the driver, and no seeding rule helps.** It is what drops
>    the events (48 % of D's x planes lose every candidate; 31 % because the
>    cluster is entirely hot, correctly). `seed_candidates` is rewritten to
>    *form* clusters from the clean strips — so a hot band can never weld a
>    noise column onto a real track, and a band narrower than the 12 mm gap is
>    crossed for free — which is better defined and newly tested. But against
>    the old rule it is a wash, and **wherever the mask relocates a seed the
>    fit gets worse** (clean stratum χ²/dof 13.1 → 20.6, p0 moving 4.8 mm).
>    Do not spend a condor pass installing it.
>
> **What works, and is now the default.** Use the same classification as a
> **downstream cut on the frozen products** — no re-reconstruction, every
> frozen product still valid. `hot_seed_strata.py` labels every trigger by the
> hot content of its raw cluster (4 min, all four arms, all sub-runs, keyed
> `(subrun, event_id)` because ids restart); `dropped_events` is the single
> definition; `k_arm.coincident_tracks` applies it by default (before the
> charge window, since a noise column carries charge like a track);
> `k_robustness` asks for the *uncut* sample so its `baseline` stays a
> baseline and its new `no_hotstrip` row stays a measurement.
>
> | arm | variant | removes | k shift | spread | reproducibility |
> |---|---|---:|---:|---|---|
> | **D** | `no_hot` (old, 2D post-fit) | 22.4 % | 3.76 % | 0.103 → 0.056 | 0.059 → 0.029 |
> | **D** | **`no_hotstrip` (default)** | **3.6 %** | **0.55 %** | 0.103 → 0.087 | 0.059 → **0.029** |
> | A | `no_hot` | 1.2 % | 0.02 % | 0.089 → **0.103** | 0.014 → **0.043** |
> | A | `no_hotstrip` | **0.0 %** | 0.00 % | unchanged | unchanged |
> | C | `no_hot` | 4.1 % | 0.94 % | 0.128 → **0.136** | 0.048 → 0.039 |
> | C | `no_hotstrip` | **0.0 %** | 0.00 % | unchanged | unchanged |
>
> All of the old mask's reproducibility gain for a sixth of the sample,
> without its 3.76 % k shift — and the only one of the two that meets "A and C
> do not move": B and C have literally **zero** all-hot triggers in either
> plane, so it cannot touch them, whereas `no_hot` moves C by ~1 % and makes
> A *worse*. D re-certifies PROVISIONAL at **k = 1.767**, spread 9.0 %,
> reproducibility 2.9 % (was 1.757 / 10.3 % / 5.9 %).
>
> `rerun_chain.sh` now runs `noisy_channels` and `hot_seed_strata` **first**,
> ahead of `k_arm`, since the cut feeds the angle scale and everything below
> it. Running them last, as before, would calibrate on the previous run's
> strata.
>
> **Found and not chased**: A's *y* view has its own hot connector-8 run
> (448–460, 21.7 % of the plane's hits), the same fault class as D's. The cut
> is on x only, so A is untouched either way — but A-y is not clean, and
> anything using A's y angles should know it.
>
> Written up in [`HANDOFF_HOT_WILDCARD_TUNING.md`](HANDOFF_HOT_WILDCARD_TUNING.md),
> which now carries the October angles: candidate ranking by the fit's own
> `_candidate_score` rather than by strip count (the real reason seeding does
> not help), whether `mostly-hot` should be cut too, whether the all-hot
> triggers hide real tracks (the scintillator tag would say), and the physical
> connector fix.

> ### The three things S1-S4 leave open, in priority order
>
> 1. ~~The accidental normalisation for pairs.~~ **MEASURED 2026-09-08** — not
>    via a control chamber in the end, but directly from scintillator timing:
>    f = 29 % [17, 40] true-coincidence overall, 42 % [26, 58] opposing,
>    16 % [0, 32] perpendicular (see `HANDOFF_ACCIDENTAL_TIMING.md` above).
>    **Not yet folded into the S4 spectrum itself** — the tagged subsample is
>    only 16 % of real inter-chamber pairs, so that is a new, smaller open
>    item rather than this one closed outright.
> 2. **Chamber D's sign flip against the wall**, degenerate between D's two
>    wall ends swapped and D's MM y plane mirrored. Needs an external fact --
>    the cabling map or the y strip mapping order. **A question for Dylan.**
> 3. **The common y offset (+22 mm) and the common X offset (~9 mm)**, both
>    degenerate between a real target position and a shared convention. D8.

> ## The next phase is planned: PLAN.md §10, workstreams S1-S4
>
> Set 2026-09-08 by Dylan, after the run_145 chain closed. The local run is
> complete; what is left before a campaign pass is worth launching is
> understanding its **geometry**, because every number in the spectrum is
> divided by an acceptance and the acceptance is geometry.
>
> | | | ships |
> |---|---|---|
> | **S1** | the scintillators: integrated as a *filter*, not as a measurement | `x17/scintillators/` |
> | **S2** | imaging the He-3 capsule; chamber-to-chamber spread = alignment | `x17/source-imaging/` |
> | **S3** | drift velocity along the gas chain, and the H2O it implies | into `x17/reco-funnel/` |
> | **S4** | the opening angle against an acceptance-folded expectation | `x17/opening-angle/` |
>
> Order is dependency, not size: **S4 needs S2**, and S2's y handle needs one
> measurement out of S1. Full scope, and the caveats each carries, in
> [`PLAN.md`](PLAN.md) §10.
>
> **lxplus is reachable again** -- single probe 2026-09-08, `SSH_OK` on
> lxplus942, ticket renewable to 13 Sep. The connection-storm warning below
> stands as a lesson; the block it describes is lifted.

> ## A second beam: NFS/GANIL at 1-40 MeV, and there is a quiet window -- 2026-09-09
>
> Asked as a separate question: what do the same two backgrounds look like at a
> MeV neutron beam? New: `ganil_background.py`, `endf.py` (a 40-line MF=3
> reader), `make_ganil_figures.py`, `make_ganil_report.py`, `GANIL_NOTES.md`,
> and the ENDF/B-VIII.0 MF=3 records for Al-27, C-12 and He-3 in
> `data/nuclear/`. Publishes `/x17/ganil-background/`.
>
> **The excitation stops being a constant.** `E_x = 20.578 + 0.749 E_n`, so it
> runs 21.3-50.6 MeV and the X17 minimum opening angle slides **104 deg -> 39
> deg**. 109 deg is a property of 20.58 MeV, not of the boson.
>
> **Which is a handle, not only a loss.** E_n is measured per event, so the
> signal becomes a *correlation*: theta_peak tracking a known function of a
> measured quantity. The gas continuum follows it (same excitation); the
> capsule background does not follow it at all. n_TOF cannot have this.
>
> **Three numbers that all favour MeV.**
>
> | | thermal | 2 MeV |
> |---|---|---|
> | radiative captures per neutron entering the cell | 1.0e-8 | 2.9e-6 (**280x**) |
> | (n,p) two-prongs per radiative capture | 9.7e7 | 8.9e3 (**10^4** better) |
> | capsule wide-angle pairs per gas pair | 10^4-10^6 | **3** |
>
> The first is not because (n,gamma) rises -- it barely moves -- but because
> the 5333 b (n,p) that consumes every thermal neutron and makes nothing
> collapses to under a barn.
>
> **And there is a quiet window below 2.29 MeV.** The capsule's two strongest
> inelastic lines (843.8 and 1014.5 keV in 27Al) are BELOW the 1.022 MeV pair
> threshold, and the first level that is not -- 2.211 MeV -- needs a 2.29 MeV
> neutron to open. `quiet_band()` computes that edge from the evaluation rather
> than taking it as given. Above 4.8 MeV the 12C 4.44 MeV level (460 mb, E2, in
> the fibre which outweighs the aluminium) ends it.
>
> **Recommendation: run below 2.29 MeV.** Four to five decades of capsule
> background removed by choosing the beam energy, no change to the apparatus,
> and the X17 angle stays at 98-104 deg so the same acceptance applies.
>
> **A hard stop at 20 MeV, from both directions.** ENDF/B-VIII.0 and TENDL-2021
> both end 3He(n,gamma) there, AND the Al/C discrete inelastic levels are zeroed
> above it in favour of the MT=91 continuum. The study runs 1-20 MeV and says
> so. Running above that needs a cross section that does not exist yet.
>
> **What the page deliberately does not do is quote a signal rate.** At
> E_x = 21-51 MeV the compound is above the 20.21 and 21.01 MeV states the
> anomaly is reported for, so the X17 branching there is a model statement. The
> background is calculable; the signal is not.

> ## The IPC expectation is now a spectrum, and the aluminium is costed -- 2026-09-09
>
> Two things asked for and both delivered; a third fell out and is the largest
> open number on the page.
>
> **The deliverable is the curve, not a threshold.** `ipc_born.grid_spectrum()`
> returns dN/dtheta by quadrature rather than by sampling -- reproducible to the
> last digit, validated against the sampled version in total variation over the
> whole 0-180 deg range (a new row in `validate()`, 0.3 % at 1 deg bins). Every
> table on the page is now a spectrum with a running "beyond this bin" column,
> so the reader picks the threshold. The old "fraction beyond 90/109/130 deg"
> tables are gone.
>
> **It does not depend on neutron energy inside the window.** Four things could
> make it, and two of them cancel *exactly*: the E0:M1 mix and the Al-to-3He
> capture ratio are both ratios of 1/v s-wave channels. The transition energy
> gains (3/4)E_n, which at the top of the window is 1.5 eV on 20.58 MeV, and
> folding that through moves the spectrum by **2e-9** in total variation between
> 1 ms and 1 s. **One template covers the whole window**, which frees
> time-of-flight binning to be spent on backgrounds that do vary.
>
> **Aluminium, from the actual capture scheme.** 215 prompt lines with absolute
> partial cross sections (IAEA PGAA) on the EGAF level scheme, both staged in
> `data/nuclear/`. Three results, and the first two say the old estimate was
> looking at the wrong lines:
>
> 1. The two hard primaries are **M1, not E1** -- 7724.0 keV feeds the 3+ ground
>    state of 28Al and 7693.4 keV the 2+ at 30.6 keV, both positive parity
>    against a 2+/3+ capture state. Assuming E1 overstated them by 3.0x.
> 2. **72 % of the wide-angle yield comes from the 2-5 MeV primaries**, which
>    *are* E1 (they feed the negative-parity levels). The 7724 keV line is the
>    tallest single contributor at 13 %, and the group beats it three to one.
>    So the Al pair background is a 2-4 MeV background, trivially separable by
>    energy and barely separable by angle.
> 3. Per capture the two sources are the same problem -- 2.2e-4 wide-angle pairs
>    per capsule capture against 2.8e-4 per 3He radiative capture. The entire
>    difficulty is that there are 10^4-10^6 times more capsule captures.
>
> **And the wall is not only aluminium.** The same machinery over 12C(n,g)
> puts carbon at 11 % of the wall's captures and **14 % of its wide-angle
> pairs** -- per capture it is worse than aluminium, because both its strong
> primaries (4945 keV to the 1/2- ground state, 1262 keV to the 3/2- at 3685)
> are E1 and soft. Everything downstream now runs on the capsule, both species.
>
> **The largest open number is not nuclear.** The 500 atm 3He cell has an
> optical depth of ~150 to (n,p) at thermal, so it absorbs essentially every
> neutron entering it -- but `calculation_tables/results_3He` computes its
> radiative captures with what looks like a thin-target formula. If that reading
> is right, **every expected IPC and X17 yield in that table, and in the INTC
> proposal quoting it, is high by ~2 orders of magnitude.** The report gives the
> Al:3He comparison three ways rather than picking one. **A question for whoever
> produced the table.**
>
> New: `ipc_aluminium.py`, `data/nuclear/` (+README), `IPC_MISSING.md` -- the
> standing list of the eight gaps, ordered by how much each could move the
> answer, with `ipc_aluminium.missing()` as its machine-readable twin so the
> report renders it and it cannot drift. Three new figures. `ipc_channels.py`
> lost its aluminium section to the new module.

---

## Resume here

Written on Windows, executed on the **Ubuntu laptop** on 2026-09-07.
**N0–N5 are done** and kept below for the record, each marked with what it
found. Stages 0, 1 and 2 all run.

### The night of 2026-09-07 — what changed

Six things, in rough order of how much they matter.

1. **A geometry bug: the Y plane never got the in-plane sign flip X got.** The
   2026-08-20 sign measurement was made on the target image, which lives in the
   XZ projection and is blind to the y sign, so y silently kept the raw strip
   direction — **every 3D track direction has carried a mirrored vertical
   component since**. Fixed. It moves the median target height from below the
   He-3 capsule to near its centre and roughly doubles the fraction pointing
   into its y span. `dca` in XZ barely notices, which is why it survived; the
   opening angle between two chambers very much does.
2. **Chamber D is calibrated.** k = 1.751, v = 24.3 µm/ns. Three of four
   chambers now carry angles.
3. **The two-chamber rate is null, with a limit and a systematic.** Combined
   A–C excess 1.5 ± 11.2 events; second-track rate < 0.07–0.12 % of triggers at
   95 % CL. The dominant systematic is the *choice of control chamber*, which
   alone moves the significance by 3.4 σ.
4. **The geometry is validated end to end** by the opening angles: opposing
   chambers 144°, perpendicular ones 83–97°.
5. **Chamber B is closed out** — not statistics, not the scan range, not the
   charge window. Most likely the bench-transferred sharing kernel does not
   describe B in the beam.
6. **Two methodological faults found and fixed in my own tooling**: the focus
   estimator was reading a value that double-counted the per-track one, and my
   own scan grid was narrow enough that a railed optimum could have passed for
   a measurement. `focus_scan` now refuses one.

Everything is published: <https://dylan-neff.web.cern.ch/x17/reco-funnel/>,
linked from the X17 hub, with the board's log carrying each result.

> ### ⚠ CERN access is down, and I caused a connection storm against it
>
> **2026-09-08 03:20.** `ssh lxplus` began refusing with `Permission denied
> (publickey,keyboard-interactive)` despite a valid, forwardable Kerberos
> ticket (`Flags: FPRA`, renewable to 09-12). No `ControlPersist` master was
> alive, so §N1's stale-master recipe does not apply.
>
> **The campaign census had no back-off, and turned that into 1 318 failed
> connection attempts in about a minute** before I stopped it by hand. On a
> staging failure the worker released its claim and immediately took the next
> sub-run; a systemic failure fails every sub-run, so eight workers walked the
> worklist as fast as they could. That is abusive to shared CERN
> infrastructure, and it was my bug. `campaign_census.sh` now backs off and
> aborts after five consecutive staging failures.
>
> It is **not established** whether the auth failure caused the storm or the
> storm tripped a rate limit — the timing does not separate them. Assume the
> latter is possible and stay off lxplus until a single probe succeeds:
>
> ```
> timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=30 lxplus 'echo SSH_OK'
> ```
>
> **Do not retry in a loop.** One probe per attempt. If it still fails, a fresh
> `kinit dneff@CERN.CH` is the first thing to try, and it may simply need time.
> Nothing local is blocked: run_145 is complete and every product is on disk.
> The overnight cron now probes once and does local-only work if it fails.
>
> Census progress when stopped: **6 of 293 sub-runs done** (run_79), all
> correct — finished sub-runs are skipped on restart, so nothing is lost.

> ### Chamber B: no field-shaping rings. It is hardware, and B moves to hits.
>
> **CORRECTED 2026-09-08 by Dylan.** My first reading of the HV monitor was
> wrong and is kept here because the wrong version is the tempting one.
>
> | arm | drift v0 | drift vmon | drift imon | resist imon |
> |---|---:|---:|---:|---:|
> | A | 700.0 | 699.8 | 0.180 µA | 0.088 µA |
> | **B** | 700.0 | 699.8 | **0.000 µA** | **2.136 µA** |
> | C | 700.0 | 699.8 | 0.180 µA | 0.013 µA |
> | D | 700.0 | 700.0 | 0.180 µA | 0.800 µA |
>
> I read B's zero drift current as an open circuit — the cathode unpowered.
> **It is not.** A, C and D ground their degrader rings through three ~1 GΩ
> resistors in series, and *that chain is what draws the current*:
> 700 V / 0.180 µA = **3.89 GΩ = 3 × 1.30 GΩ**, which is the divider and
> nothing else. **B simply has no ring chain**, so zero current is exactly what
> it should read, and the monitor says nothing about whether B's cathode is at
> voltage. It should be.
>
> The physics that follows is different and better. The degrader rings are what
> make the drift field *uniform*; without them the field between cathode and
> mesh fringes badly, so drift lines are distorted and there is **no clean
> time ↔ depth ladder** even with the cathode powered. That explains everything
> B does — the angle-scale estimators that never agree, the focus objective
> that never turns over, the plateau spanning the whole scan grid — without
> requiring an unpowered cathode. B's resistive channel drawing 2.136 µA
> against A's 0.088 is a separate anomaly and still unexplained.
>
> **It still retires the sharing-kernel explanation.** Everything chased for B
> overnight was aimed at the reconstruction when the answer was in the
> detector's field cage.
>
> **B is a good hit detector, and that is where it goes** — on its own
> processing chain, with its own metrics, before recombining with the others.
> Position needs only the amplification stage. Asking whether B's cluster
> position predicts which wall segment fired, using no angle at all:
>
> | arm | segment match | shuffled | lift |
> |---|---:|---:|---:|
> | A | 78.0 % | 30.2 % | 2.58 |
> | C | 62.1 % | 29.2 % | 2.13 |
> | **B** | **49.7 %** | 27.4 % | **1.82** |
> | D | 48.6 % | 29.3 % | 1.66 |
>
> B's position information is real. B's efficiency is measured on **hits, not
> tracks**.
>
> **CORRECTED 2026-09-08 (later), by `chamber_b.py`.** The table above was an
> ad-hoc comparison and its *ordering* does not survive a proper one. Using the
> geometric lever prediction (a particle at u_c reaches the wall at
> `foot + 1.409·(u_c − foot)`) on single-group events, the lifts are
> **A 2.70, C 2.25, D 1.71, B 1.45** — so **B is real but NOT better than D**.
> The ordering is robust: dropping the single-group requirement gives
> 1.42 / 1.71, and scanning the lever from 1.0 to 1.6 leaves B at 1.41–1.45 and
> D at 1.57–1.71 throughout. The first half of the claim stands; the second is
> withdrawn.
>
> **What does separate B is its cluster shape**, and it is what the missing ring
> chain predicts — a fringing field spreads the same charge over more strips, so
> the signature is a *wide, dilute* cluster and not a weak one:
>
> | arm | width_x [strips] | vs A | charge/strip | vs A |
> |---|---:|---:|---:|---:|
> | A | 25 | 1.00 | 94.7 | 1.00 |
> | C | 34 | 1.36 | 62.1 | 0.66 |
> | D | 42 | 1.68 | 74.0 | 0.78 |
> | **B** | **43** | **1.72** | **42.3** | **0.45** |
>
> D is nearly as wide, so width alone does not isolate the fault; the charge
> **density** does, and B sits well clear of the other three.

> ### Chamber D has ~130 dead channels; A is clean; the "acceptance hole" was mine
>
> **2026-09-08, and this corrects an explanation I committed confidently and
> wrongly.** I read the central gap in the filtered hit maps as the
> reconstruction blanking normal-incidence tracks via
> `wft.reco.TAN_MIN_SLOPE = 0.08`. It is not. The gate is *"time-coincident AND
> both members plausible"* — `slope_reliable` is recorded and never gates
> anything — and the gaps are present before any selection is applied. The
> occupancy projected on x, no cuts, normalised to its own median:
>
> | | x = −8 | +8 | +22 | +38 | |
> |---|---:|---:|---:|---:|---|
> | A | 0.93 | 1.03 | 1.12 | 0.95 | flat |
> | C | 0.94 | 0.34 | 0.15 | 0.43 | already there |
> | D | 2.17 | 0.01 | 0.07 | 0.03 | already there |
>
> **It is dead readout channels, on connector boundaries:**
>
> | arm | dead runs (≥8 ch, <20 % of median) | x_local | connector |
> |---|---|---|---|
> | A | **none** | | |
> | C | ch 227–236 (10) | +15…+22 mm | 3 |
> | D | ch 183–212 (30) | +34…+57 mm | 2–3 |
> | D | ch 219–227 (9) | +22…+29 mm | 3 |
> | D | ch 234–255 (22) | +0…+17 mm | 3 |
> | D | ch 27–63 (37) | +150…+178 mm | 0 |
>
> **D has ~130 dead channels of 512 — a quarter of its x plane** — the same
> class of fault as chamber A's connector-8 outage in run_79 (CLAUDE.md). That
> is the 13.9 % dead area the 2D map showed, diluted there because D's y plane
> is healthy.
>
> A's dip is real, appears only once the scintillator cuts are applied, and is
> **unexplained**. Stated as unexplained rather than given the nearest
> plausible story, which is what went wrong the first time.
>
> D's angle scale is nonetheless sound: re-measured with the outer ring
> excluded it moves 2 % and the estimator spread improves, because the pointing
> coincidence already strips the junk (D goes from 44.1 % of fitted clusters in
> the ring to 6.8 % of the k sample).

> ### The two-lobe structure is the TRIGGER, not the detectors
>
> **2026-09-08, Dylan's explanation, confirmed from the Geant geometry.** The
> production trigger needs a coincidence with the two plastic bars behind each
> wall. There is a gap between them, so a charged particle threading it makes no
> plastic signal and no trigger — the chamber is not inefficient there, the
> trigger is blind.
>
> Nothing was tuned to make this work. The Geant config
> (`~/CLionProjects/MX17_Full_Geant/include/SimConfig.hh`) and the reconstruction
> constants are independent sources and they agree where they overlap:
>
> * `bscTape_hu + bsc_gap/2 = 100.22 + 1.5 = 101.72 mm` = `PLASTIC_U_OFFSET`
> * `mm_pinwheel_shift_cm = {1.55, 1.575, 1.635, 1.73}` = `PINWHEEL` for D, B, A, C
>
> **The predicted shadow centre** is `x = foot_x · L/(234.6 + L)`: **+7.3, +7.0,
> +7.7, +6.9 mm** for A, B, C, D — essentially identical across arms, as the
> symmetry demands. Measured minimum in arm A's purest tier: **+5 mm**, one bin
> away.
>
> `plastic_acceptance.py` ray-traces the He-3 capsule (r = 10 mm, 60 mm long
> with hemispherical caps) through the bars and produces the acceptance per
> chamber surface. The measured occupancy sits **inside the predicted contours**
> for all four chambers — both lobes, the gap between them, and the ±85 mm
> extent in v that the 300 mm bar imposes at the 1.8× lever.
>
> **Depth needs ~5 mm of inactive scintillator at each bar edge**: the bare
> 3.4 mm geometric gap gives only a 3 % dip, against 86 % measured (A's target
> tier falls to 0.14 of its median). At a 5 mm dead edge the model gives 0.08.
> The measured dip is *wider* than the model's — ~30 mm against 5-15 mm — which
> is expected and not a failure of the model: the `dca < 30 mm` selection alone
> smears the plastic crossing by 30 × L/D ≈ 24 mm.
>
> **This does not retract the dead channels**, it sits on top of them. Chamber A
> has no dead runs, so its dip is purely the trigger. C and D have dead runs at
> x ≈ +15…+57 mm, which is *the same place*, so their dip is both effects at
> once — and that is why theirs shows in the raw occupancy while A's needs the
> scintillator cut to appear.

## HANDOFF — the state on 2026-09-08

**Published, both linked from <https://dylan-neff.web.cern.ch/x17/>:**

| page | what it is |
|---|---|
| [`x17/reco-funnel/`](https://dylan-neff.web.cern.ch/x17/reco-funnel/) | the reconstruction chain end to end, the angle scale, the two-chamber rate |
| [`x17/detector-response/`](https://dylan-neff.web.cern.ch/x17/detector-response/) | efficiency, hit maps, dead channels, the plastic trigger acceptance |
| [`x17/analysis.html`](https://dylan-neff.web.cern.ch/x17/analysis.html) | the board: pipeline, open questions, dated log |

Both reports are **generated** (`make_funnel_report.py`,
`make_response_report.py`) — re-run the analysis and then the maker, never edit
the HTML.

**Run 145 is complete**: three sub-runs, 189 724 triggers, full waveform pass
with no prescale, everything downstream rebuilt on it.

| | A | B | C | D |
|---|---|---|---|---|
| angle scale k | 1.266 | **none** | 1.616 | 1.757 |
| efficiency | 24.2 % (tracks) | 17.5 % (**hits**) | 17.1 % (tracks) | 13.3 % (tracks) |
| dead channels | 0 | — | 10 | **~130 of 512** |
| state | healthy | field cage broken | healthy | badly compromised surface, sound angle scale |

**The four things a new session must not re-derive:**

1. **Chamber B has no field-shaping ring chain**, so no uniform drift field and
   no time↔depth ladder. It is a **hit detector**: position and timing only,
   angle columns null, efficiency quoted on hits. Its zero drift current is
   *expected* (that current is the degrader divider, 3 × 1.30 GΩ) and is not
   evidence of anything.
2. **The two-lobe structure in every hit map is the trigger**, not the
   detectors — the gap between the two plastic bars, predicted from the Geant
   geometry with nothing tuned. Dead channels sit on top of it in C and D.
3. **`k` is applied, never assumed.** An arm without a certified scale gets NaN
   angles, not k = 1. Positions never depend on v.
4. **The Y plane carries the same in-plane sign flip as X** — this was wrong
   until 2026-09-08 and mirrored every vertical component.

**Blocked on Dylan's hold** ("hold off on any large processing over all runs
until I understand the results of the local tests"): the campaign stage-1
census (11 of 293 sub-runs done, resumable via `campaign_census.sh`), and the
`t_since_flash_ns` slim patch, which needs the slims regenerated.

**Deferred to October** — see the list below, principally whether the
scintillator tag really means a charged particle crossed the MM.

**Package map** (`sept26_prelim_analysis/`):

```
freeze_sample     stage 0: the frozen run/sub-run sample
candidate_filter  stage 1: one class per trigger from combined_hits + slim
allowlist         stage 2: which (event, arm) pairs get reconstructed
merge_fullpass    puts the flat CERN pass into the per-sub-run layout
build_tracks      stage 3: one row per 3D segment; applies k, nulls what it cannot
k_arm             the in-situ angle scale, three estimators + verdict
funnel            trigger -> track -> n_TOF confirmation, per chamber
pairs             the controlled two-chamber rate, with its systematic
efficiency        scintillator-tagged, accidental-corrected
hit_maps          occupancy, relative, by angle, and the purity ladder
plastic_acceptance the trigger's own acceptance from the Geant geometry
neutron_energy    time since flash -> E_n, and the resolution it delivers
trigger_time      the per-trigger flash time out of the slim, into stage 3
beam_cache        ref-free training cache (built, measured, does NOT work)
refit_B_beam      the beam-calibration harness (kept as the record of why not)
make_*_report     the two published pages
rerun_chain.sh    everything downstream of the pass, in dependency order
campaign_census.sh streamed, resumable stage-1 over the campaign — ON HOLD
```

**Next, in order:**

1. **Chamber B.** The only chamber still without angles. Four suspects have now
   been eliminated, so what is left is the detector or its kernel:

   | suspect | test | verdict |
   |---|---|---|
   | too few tracks | subsample A and D to B's 412 | **no** — they recover k to ±0.03–0.06 |
   | scan grid too narrow | open it to k = 6 | **no** — B peaks at 2.10 and 4.45 in its two sub-runs |
   | the charge window | vary it 25–75 → 0–100 | **no** — A, C, D move less than a plateau width; only B jumps |
   | wall/plastic readout order reversed for the Z-view pair | try all four orderings per chamber | **no** — ascending/ascending wins in all four (A 44.8 %, B 20.5 %, C 35.6 %, D 15.0 %; every reversal collapses to 2–6 %) |

   The conditional funnel says B's loss is diffuse, not one stage — and the one
   place the chambers split into pairs is pointing confirmation per gated
   track: **A 35.9 % and C 28.9 % against B 10.1 % and D 11.6 %**. That split is
   {X-view} vs {Z-view}, which is what motivated the readout-order test above;
   it survives the test, so it is a real detector difference. B's remaining
   distinction from D is simply yield — 824 confirmed tracks against D's 1 865,
   412 against 933 after the charge window — and since 412 is demonstrably
   enough for A and D, B's individual tracks must carry less angle information.
   That points at the bench-transferred sharing kernel (B: kY 5.40,
   sigma_s 172 ns) not describing B in the beam.

   **A beam-side refit is not a re-run — it is development.** `wft.calibrate`
   cannot fit on beam data at all: `build_cache` is **ref-pinned**, selecting
   its training events along the M3 reference corridor and fitting the model
   against per-event reference track parameters. The beam has no reference
   telescope. This also settles what the bundles are: **all four are bench
   transfers, and none has ever been fitted on beam data** — arm C's
   `"fitted": "wft.calibrate"` describes its parent *bench* fit, alongside
   `"transferred": "template + sharing kernel + w0/kw (bench)"`.

   What would make it possible is a **ref-free training selector**. The beam
   has a constraint the bench does not: the source is a point 234.6 mm away, so
   position and angle are not independent — the very relation `k_arm.py`
   already exploits. A calibration could pin on the target the same way,
   fitting the kernel against `tan = (u − foot)/d_perp` instead of a reference
   ray.

   **That selector now exists** — `beam_cache.py`, 2026-09-08. It produces a
   cache in exactly `build_cache`'s format from the target constraint, and
   `refit_B_beam.py --use-beam-cache` drops it in where `calibrate()` looks, so
   the fit is runnable. Validated on run_145: it builds for all four arms, and
   its truth independently reproduces `k_arm`'s per-track estimator, because it
   is the same relation — median `tan_reco/tan_target` is 1/k to a few percent
   (A 0.804 → k 1.24, C 0.600 → 1.67, D 0.544 → 1.84, B 0.403 → 2.48).

   **And it does not work — measured, 2026-09-08.** The fit runs end to end and
   has no power:

   | | χ² improvement |
   |---|---:|
   | beam, target-pinned, 60 training events | 0.028 % |
   | beam, target-pinned, 180 training events | 0.080 % |
   | bench, ref-pinned (all four bundles) | **23–27 %** |

   Three orders of magnitude less traction, and the training-set size is not
   the explanation. At 180 events the optimiser wandered to c1 = 0.743 (seed
   0.0513), tau_s = 1.9 ns (134), sigma_s = 5 ns (172) for that 0.08 % — a long
   walk for nothing, which is what a flat landscape looks like. That c1 puts
   **238 % of the charge on the neighbours** and still passes the `c2 < c1`
   gate, which only catches inversion; `evaluate()` now checks
   2(c1+c2) < 1 as well, and reports χ² traction against the bench's 23–27 %.

   **Why**, and it is the circularity surfacing as powerlessness rather than as
   a wrong answer: the bench truth is a reference ray measured *outside* the
   waveform, so the model must reproduce a given track with the right kernel.
   The target truth is `tan = (u − foot)/d_perp` with `u` from the fit's own
   `p0`, so the χ² can be satisfied by moving the track instead of by getting
   the kernel right. **The target constrains one number — the relation between
   position and angle, which is exactly what `k_arm` measures — not a
   seven-parameter kernel.** A ref-free calibration needs truth that is
   independent of the waveform fit, and the target is not.

   **And no external reference in the beam is good enough either** — measured,
   not asserted. A calibration needs truth independent of the waveform, so the
   candidates are the scintillators, and their granularity settles it (uniform
   segments, σ = half-width/√3):

   | external pair | lever | d(tan) |
   |---|---:|---:|
   | target + wall group | 331 mm | **0.089** |
   | target + plastic bar | 421 mm | 0.138 |
   | wall + plastic | 90 mm | 0.720 |

   The best is 0.089 — **twice as coarse as the circular target truth that
   already failed**, and 70 % of a typical |tan| in chamber B. The wall's u
   granularity is 100 mm because `detn` resolves 4 groups of 4 bars; its 8
   values are those groups × top/bottom, and the parity is a *y* distinction,
   not a finer u one.

   **So chamber B stays tagging-only for the preliminary.** Its kernel cannot
   be calibrated from beam data with any reference that exists. Two avenues
   remain, both new method development rather than re-runs:

   * **An ensemble calibration** instead of per-event truth. The target already
     pins one number from a distribution; a kernel might be pinned the same way
     by matching distributions the kernel controls — cluster width, χ²/dof, the
     residual structure across strips — rather than event by event.
   * **The wall's top/bottom amplitude ratio** should give position along the
     bar, which is the y handle the capsule's 80 mm length denies. That would
     bear on kY only, not the shared kernel.

   The two limitations below stand regardless:

   * **kY cannot be fitted.** The capsule is a 10 mm point in XZ but 80.2 mm
     long along y, so d(tan_x) = 0.043 (21–34 % per track, 1.6–2.5 % on a
     180-event mean) while d(tan_y) = 0.171 — 85 % per track, comparable to the
     angle itself. `hypers_to_fit()` returns the shared kernel only. For
     chamber B, whose kY of 5.40 is itself one of the suspects, that is a real
     limitation.
   * **The acceptance test is not written.** The truth derives from `p0`, from
     a fit made with the bundle being replaced, so a result must be shown to be
     a **fixed point**: fit → re-reconstruct → re-fit, and require the hypers to
     stop moving. `k_arm` did exactly this for the angle scale (8–21 % of rows
     moved, k changed by less than a grid step). Until that runs, any bundle
     from this cache is a candidate and must not be installed. Its focus objective
   never turns over across the whole scan grid (k = 0.60–2.55) and its k jumps
   1.35 → 2.30 depending on the charge window, so this is a detector question,
   not a fitting one. B also has the fewest tracks by far (1 258
   pointing-confirmed against A's 5 909) and sub-run 0001 leaves under 200 in
   the charge window. Start from the funnel: B is the chamber with the lowest
   lift (1.15×) and the weakest pointing confirmation (16.1 %).
2. ~~Sub-run 0002~~ — **done 2026-09-08 02:15**, all four arms rc=0, and the
   whole chain re-run over all three sub-runs (189 724 triggers). `run_145` is
   complete.
3. **The B–D pair direction.** With D certified, D-track/B-trigger is
   measurable; its mirror needs B. Until then the X17 topology has one
   measurable channel and a half.
4. **Campaign stage-1 census — RUNNING** since 2026-09-08 02:33
   (`campaign_census.sh`, 8 workers, streamed and resumable). 293 sub-runs,
   25.6 M triggers. Honest sizing: `candidate_filter` is single-process at
   20–30 events/s, so this is **17–25 h wall**, a multi-night job rather than
   an overnight one. It skips any sub-run whose census exists, so stopping and
   restarting is free — `bash campaign_census.sh --status` for progress, and
   just re-run it to continue. Then stage 2 (~760 core-hours).
5. **The time base — DONE 2026-09-11. There was no gap.** Every one of the
   29.16 M tracks now carries `t_since_flash_ns` and `e_neutron_keV`.

   The slim's `events` tree already held the trigger's time since the gamma
   flash, under two names nothing downstream read: **`t_dream_ns`** (the DREAM
   clock, `trigger_timestamp_ns` minus the burst's first trigger, and the first
   trigger of a burst *is* the flash trigger) and **`t_pred_ns`** (the same
   instant on the n_TOF time base through the segment's own fitted clock,
   per-bunch correction included). `slim.py` writes `ev['t_since_flash_ns']`
   out as `t_dream_ns` and that rename is the whole of why this sat open for a
   month. Stage 3 is filled from `t_pred_ns`.

   **No slim was regenerated and no reprocessing was asked for.** The pull was
   one 90 s pass over the 292 slims on EOS for their `events` trees (380 MB),
   then a local backfill. `sept26_prelim_analysis/trigger_time.py` has the
   evidence, the extractor and the backfill; `build_tracks` now fills the two
   columns natively, so `NOT_POPULATED` is empty.

   **And it corrects a number that was on the board.** `neutron_energy.py`
   inferred the window as "2.4 MeV down to 0.36 meV" by taking the hits' `tof`
   floor of 1.00408e6 ns to be the flash. The real offset is `tflash`, a
   per-tree cable delay of **~11.6 µs** — recovered directly as
   `tof − (t_pred_ns + dt_ns)`, 11.60–11.65 µs per detector tree with a 5–14 ns
   spread, which also closes the loop that the hits and the triggers share one
   flash reference. So the window is **2.0 eV down to 0.44 meV**, median 31 meV,
   the thermal peak. That is what `../CLAUDE.md` says independently and what
   the E0/M1 argument in `ipc_born.py` rests on. **There is no resonance
   region in this dataset and no cut can make one.**

   Resolution over this window is irrelevant to any cut: dE/E is 1e-5 at 1 eV.
   Where it would have mattered — 0.84 % at 1 MeV — there is no data.

   ~~The 2026-09-08 investigation, kept because its three upstream facts are
   right and only its conclusion was wrong:~~

   * `ntof_processing/flash_timing` measures `t_flash(bunch) = tof_PKUP + C`,
     C ≈ −1708 ns per channel, good to **0.5 ns** run-to-run within an epoch
     and **3.2 ns** per bunch.
   * `ntof_dream_merge.ntof_io.read_bunches` already returns
     `t_since_flash_ns = tof − tflash` against a **repaired** tflash.
   * `slim.py` **computes** it — it matches DREAM to n_TOF on exactly that
     quantity — and then does not write it. The slim carries `tof` and the
     match residual `dt_ns` but neither `tflash` nor `BunchNumber`, so
     t_since_flash cannot be reconstructed from a slim file.

   `sept26_prelim_analysis/neutron_energy.py` has the relativistic conversion
   (19.5 m EAR2 flight path, cross-checked to 1e-14) and `slim_patch()` states
   the three-line fix. ~~**Applying it needs the slims regenerated, so it needs
   EOS.**~~ **It did not — see above; `slim_patch()` is marked superseded and
   must not be applied.** Resolution once filled: dE/E = 0.001 % at 1 eV,
   0.084 % at 10 keV, 0.84 % at 1 MeV, 2.7 % at 10 MeV.

   ~~The slim window is *not* the limitation: all three detector families share
   a tof floor of 1.00408e6 ns and ceiling of 7.505e7 ns, and taking the floor
   as the flash gives 2.4 MeV down to 0.36 meV~~ — **wrong, the offset is
   `tflash` ≈ 11.6 µs and the top of the window is 2.0 eV** — with the busiest
   bins at 40–125 meV, the thermal peak, where run_55 independently found the
   ³He(n,p) capture flood. ~~That last is an inference until one bunch's
   `tflash` is read from the raw files.~~ `tflash` is now measured.


**Deferred to October** (added 2026-09-08):

* **Is the scintillator tag really "a charged particle crossed the MM"?** The
  efficiency below uses wall AND plastic as truth, and chamber A comes out at
  63 % where 80–90 % is expected. A neutron or gamma passing through the MM and
  converting in the PCB would fire the scintillators without a charged particle
  having crossed the gas — inflating the denominator and pushing the measured
  efficiency down, which is the direction of the discrepancy. Needs a study of
  the conversion probability and, if it matters, a tag that is not
  scintillator-only.
* **Chamber B's resistive channel draws 2.136 µA against A's 0.088** — 24×,
  unexplained, and separate from the missing ring chain.

### N0 · Land on the machine — ✅ done

Kerberos ticket live (`dneff@CERN.CH`, renewable to 12 Sep), `ssh lxplus`
works, both repos clean and current.

```bash
cd ~/PycharmProjects/nTof_x17      && git status && git pull
cd ~/PycharmProjects/dylan-cern-site && git status && git pull
```

Check `git status` **before** pulling on each: the Ubuntu clones may carry
their own unpushed commits (this has bitten before — `../CLAUDE.md` warns about
it for the DAQ clone). Then:

```bash
kinit dneff@CERN.CH
ssh lxplus true && echo "lxplus OK"
```

The two Claude skills (`publish-note`, `x17-board`) live in `~/.claude/skills/`,
which is machine-local and does **not** travel with the repositories. Canonical
copies are committed in the site repo, so install them once on the new machine:

```bash
cp -r ~/PycharmProjects/dylan-cern-site/skills/publish-note \
      ~/PycharmProjects/dylan-cern-site/skills/x17-board ~/.claude/skills/
```

### N1 · Publish what is already written — ✅ done

Both pages return `200`: the board and the plan note are live.

```bash
cd ~/PycharmProjects/dylan-cern-site && ./scripts/deploy-eos.sh
```

That puts the restructured board and the plan note live. Confirm:

```bash
curl -sI https://dylan-neff.web.cern.ch/x17/analysis.html | head -1
curl -sI https://dylan-neff.web.cern.ch/notes/x17-prelim-plan.html | head -1
```

If `/eos` comes back "Permission denied", the forwarded ticket is stale behind a
live `ControlPersist` master — `ssh -O exit lxplus`, then retry.

### N2 · Verify the five CERN assumptions — ✅ done

**All five answered; the answers are in
[Verified at CERN](#verified-at-cern).** Two of them change the plan and are
written up under [What the verification changed](#what-the-verification-changed).
The original five questions follow, for the record.

1. **Is the re-slim complete?** `../ntof_processing/SLIM_CAMPAIGN_2026-08-12.md`
   reports 170 fitted segments, 107 failures, and a recovery campaign started
   and stopped at 20 of 83 jobs. The site's pulse ledger (frozen 2026-08-16,
   from `slim_recovery_2026-08-13`) says 99.45 % of beam pulses are matched,
   which implies the recovery finished. **Confirm from the files on EOS, not
   from either document**, and produce the list of sub-runs that actually have a
   slim product. This is the single most important check: stage 1 reads the
   slim files, and a sub-run without one cannot enter the sample.
2. **Which runs already have a wft reconstruction**, with which bundle and at
   which code commit? run_79 and run_145 are known; anything else is not.
   Check `CODE_COMMIT.txt` beside each product — a mismatch there has bitten
   this analysis before (`../ntof_tracking/RUN145_R06_2026-08-19.md` §4, where
   45 jobs ran on a stale `code.tar.gz`).
3. **run_79's products still carry the two geometry defects** — the mirrored
   in-plane sign and the wrong pointing lever
   (`../ntof_tracking/RUN145_ALIGNMENT_2026-08-20.md` §6). They were fixed in
   the *analysis*, not in the parquet. Anything reading run_79's imaging output
   must apply the fix or be rebuilt. Decide which, and note it.
4. **Measure the link speed to CERN from the Ubuntu laptop.** 310 kB/s was
   measured in August; if it still holds, the run_145 pull below is an overnight
   job and the download order in `PLAN.md` §4 matters.
5. **Condor throughput and quota** — what the `workday` flavour delivers now,
   and whether `/afs/cern.ch/work/d/dneff` has room for the campaign outputs.

### N3 · Stage the run_145 development bundle — ✅ partly done

Staged under `/media/dylan/data/x17/beam_july/analysis/`:

| what | where | size |
|---|---|---|
| the 60 reco tarballs + `CODE_COMMIT.txt` + `jobs.txt` | `wft_beam145/results/` | 129 MB |
| the same, unpacked to `out/mx17_{A,B,C,D}/` | `wft_beam145/extracted/` | 143 MB |
| run_145 `stat090_0000` slim + all sidecars | `slim_dev/run_145_stat090_0000/` | 52 MB |

**Not pulled, and now a disk question rather than a link question:** the
`combined_hits_root` (515 MB per sub-run — the stage-1 input) and any
`decoded_root` (6.0–7.0 GB per sub-run). `/media/dylan/data` has **6.8 GB free
of 477 GB**. At 36 MB/s one sub-run's waveforms is ~3 minutes to transfer and
does not fit. **Free disk before pulling waveforms** — that is the real N3
blocker, and it did not exist when the plan was written.

### N4 · Scaffold the package — ✅ done

`figstyle.py` and `paths.py`, both with a runnable `__main__`.

- **`paths.py`** — every root resolves through it (`X17_ROOT` and four
  per-tree overrides), raises with the variable name that would fix it rather
  than returning a path that does not exist, and carries the CERN-side paths as
  strings so scripts spell EOS the same way. `python paths.py` prints what
  resolves and what exists.
- **`figstyle.py`** — ordinary document figure sizes (~3:2 single panel,
  10.5 pt type, light ink), the Okabe-Ito four-chamber
  palette **re-validated for this package** (ALL CHECKS PASS; the two warnings
  are discharged by `det_style` always returning a marker with its colour, and
  by direct labelling). `save()` **refuses to write a PNG without its CSV** —
  pass `data=NO_DATA` for a schematic and expect to justify it. `python
  figstyle.py` renders the smoke test.
  - `end_labels()` de-collides direct labels with leader lines. It exists
    because the first smoke test collided chambers B and C at the right-hand
    edge, which is exactly what converging efficiency curves will do.

### N5 · Start the chain — **stage 0 done, stage 1 next**

Two small modules, before any analysis code, because everything else imports
them:

- `figstyle.py` — the shared matplotlib style. Normal document figure sizes
  and 10.5 pt type (shape follows the data, nothing is pinned to a slide
  frame), a `preliminary(ax)` badge
  helper, and a `save(fig, path)` that writes the PNG **and** the numbers as
  CSV beside it. Every figure in the deck comes through this.
- `paths.py` — machine-aware data roots, so no script hard-codes
  `/media/dylan/data/x17/`. Resolve from an environment variable with the
  laptop path as the default, and fail loudly rather than silently returning a
  path that does not exist.

### N5 · Then start the chain

**Stage 0 is written and run** — `freeze_sample.py`. See
[Stage 0, frozen](#stage-0-frozen) below for the numbers it produced.

**Stage 1 is the next code to write.** Its input is staged:
`analysis/run145_hits/stat090_0000/` — 7 file tags, 515 MB, a flat hit table
(`eventId, feu, channel, amplitude, time, time_of_max, integral, significance,
saturated, …`, ~1.9 M hits per tag). `PLAN.md` §3 stage 1 has the
specification; `../ntof_tracking/reco/noise.py` and `reco/segments.py` already
do the de-noising and the cluster taxonomy.

Two things to settle first, both raised by today's staging:

1. **Check the cluster taxonomy per arm before believing any census** — see
   [C3](#c3--arm-d-seeds-twice-as-often-as-a-b-and-c).
2. The n_TOF arm flags come from the slim file, staged for `stat090_0000` at
   `analysis/slim_dev/run_145_stat090_0000/`.

---

## Stage board

| stage | state | what exists | next |
|---|---|---|---|
| 0 · sample | **DONE** | the cuts have been *applied* to the frozen registry (2026-09-07): `mode=beam & phys & run≥79` gives 40 runs / 329 sub-runs / **25.87 M triggers** / 8.41 TB, and every one of those 40 is already ³He + Ar/Iso 90/10 + `st=complete` + 8 FEUs, so those cuts cost nothing. Dropping run_82 (watermark × IPD scan) and run_161 (detector-A resist × drift scan) leaves the core sample: **38 runs / 296 sub-runs / 25.62 M triggers / 8.32 TB** | — (see [Stage 0, frozen](#stage-0-frozen)) |
| 1 · time base | **todo** | the flash calibration and clock QA are done and published | pick the veto window; write `t_since_flash` and E_n per trigger |
| 2 · candidate filter | **RUNS** | `../ntof_tracking/reco/noise.py` and `reco/segments.py` already do the clustering and the taxonomy; the slim files carry the n_TOF arm flags | census on a full sub-run; then the stage-2 event-id allowlist |
| 3 · reco | **RAN ON CONDOR** | 28 jobs (4 arms x 7 tags) on cluster 4139919, all succeeded, allowlist honoured. A **full** pass (no allowlist) already exists locally for sub-runs 0000 and 0001 — `analysis/wft_beam145/extracted/out`, 15 tags x 4 arms — and is what the funnel and `k_arm` are built on | sub-run 0002 (the only one with no full pass), then the campaign |
| 4 · database | **RUNS** | `build_tracks.py` — 4 216 segments / 2 263 gated for `stat090_0000`. `k_arm` is **measured and applied** (A 1.25, C 1.58; B and D uncertified, angles null). `t_since_flash` and `e_neutron` remain declared nulls | re-certify D, diagnose B; then the time base |
| 5 · scint positions | **todo** | `../ntof_processing/quality_metrics.py` A1/A2 has both estimators and their caveats | recalibrate λ and the Δt scale against MM tracks |
| 6 · pairs & spectrum | **todo** | nothing | after 4 |
| — · figures | **scaffolded** | `figstyle.py` (validated palette, document sizing, PNG+CSV enforced) and `paths.py` | build them as each stage lands |

The board carries the same eleven stages with their full descriptions; this
table is the short form. Keep the two in step by moving the board stage
(`x17_board.py stage <slug> --status …`) rather than editing its numbers.

---

## Blockers

**B1 · lxplus access — resolved by switching machines, 2026-09-07.**
On the Windows box `ssh lxplus` fails with
`Permission denied (publickey,gssapi-with-mic,keyboard-interactive)`: no
Kerberos ticket (`klist` → 0 cached), no `kinit` installed, and `~/.ssh/id_rsa`
(2019) is not accepted. The `Host lxplus` block in `~/.ssh/config` is correct
and requests GSSAPI delegation — there was simply no ticket to delegate. The
same gap blocks `deploy-eos.sh`, which is why the board and the note were built
but not live.

**The fix is not to repair Windows: the analysis moves to the Ubuntu laptop**,
which has a working `kinit`, the data under `/media/dylan/data/x17/`, and every
path the repo's documentation already assumes. Nothing else was blocked by B1,
and nothing needs redoing after the move — everything written today is in git.

**Confirmed closed on Ubuntu the same evening:** ticket live, `ssh lxplus` OK,
`deploy-eos.sh` run, and both pages return `200`.

**B2 · Local disk is full — open.** `/media/dylan/data` has 6.8 GB free of
477 GB and `/home` 9.9 GB of 115 GB. The parquet and slim products fit; a
sub-run of `decoded_root` (6–7 GB) does not. This blocks nothing until N3's
waveform step, and it is the only thing in this file that has to be fixed on
the laptop rather than at CERN. See [C1](#c1--the-link-is-no-longer-the-bottleneck-so-the-disk-is).

**B3 · The acceptance is run_145-only, and it now blocks the spectrum — open,
2026-09-10.** The campaign opening-angle spectra exist and no folded model
describes them (chi2/dof 13–85 with one free normalisation). The leading
suspect is not the physics but `acceptance.py`, which runs only on run_145
because it needs an efficiency map, and `efficiency.py` has been run only
there. It also applies efficiency independently of incidence when the measured
head-on tracking ratio is 0.80 — a real theta-dependent bias in exactly the
variable being fitted. **A campaign acceptance needs a per-run efficiency
measurement, and that is the next piece of real work on S4.**

---

## Deferred, on purpose

**D-QA1 · A detailed run-by-run / tag-by-tag QA page — deferred 2026-09-10,
Dylan's request.** `<out>/tracking_qa_fullpass/report.html` and the two new
campaign pages (`imaging_campaign`, `angle_campaign`) each answer one question
across runs. What does not exist is the browsable page: pick a run, or a tag,
and see *its* distributions — chi2/dof, strips, charge, t0, drift span, the
crossing, the pair yield — against the campaign band, with the outliers
clickable. `tracking_qa.py` already writes `per_run.csv` and `per_tag.csv`
(~3 150 tags), so the data is there and this is a presentation job, not an
analysis one. **Wanted for the control-room browser, after the acceptance work,
not before it.**

---

## Verified at CERN

Filled in by N2 on 2026-09-07 — **all five checked against the files on EOS and
lxplus, not against the documents.**

| # | question | answer | when |
|---|---|---|---|
| 1 | is the re-slim complete, and which sub-runs have a slim product? | **Yes — 326 of 329 beam+phys sub-runs in run_79–162, 99.1 %.** 450 slim segments on EOS over 49 runs, 26.0 GB (24.3 GB from run_79 on). Three gaps, all trivial: run_120/`stat090_0000` (22 476 ev), run_137/`stat090_0000` (44 ev), run_145/`stat090_0003` (a 3.3 MB stub sub-run). The 46 runs in 79–162 with no slim at all are the **beam-off cosmic and pulser runs** — no n_TOF join by design, not a gap. | 2026-09-07 |
| 2 | which runs have a wft reco, with which bundle and code commit? | **Only run_145, and only 2 of its 4 sub-runs.** `lxplus:~/wft_beam145/` holds 60 tarballs = 4 arms × 15 file tags, 129 MB, `CODE_COMMIT` `5f1ee4a9`. Tags cover `stat090_0000` (7) and `stat090_0001` (8); `stat090_0002` (9 tags, 7.0 GB) is **not** reconstructed. Bundles verified kernel-ordered: A/B/D `calib_bundle_r06` with `c2_over_c1 = 0.6`, C `calib_bundle_lp` with a stored c2 = 0.0528 < c1 = 0.0642 (0.82), physical as intended. **No other beam reco exists at CERN.** | 2026-09-07 |
| 3 | do run_79's products get rebuilt, or patched at read time? | **Moot — there is nothing to patch.** No run_79 wft product exists on the laptop or on lxplus. The 2026-07-30 prelim was arm A only, tags 000–002 of 13, on `calib_bundle_prelim` (c2/c1 = 1.14 — the inverted kernel retired 2026-08-21), so it would have to be re-run regardless. **Rebuild.** | 2026-09-07 |
| 4 | link speed to CERN from the laptop | **17.5 MB/s from AFS, 36 MB/s from EOS** — 56–115× the August figure of 310 kB/s. Measured by `rsync` over ssh: 129 MB of tarballs in 7.7 s; a 54 MB slim file in 1.5 s. **The link is no longer the bottleneck; the local disk is** (`/media/dylan/data` 6.8 GB free of 477 GB, `/home` 9.9 GB of 115 GB). One measurement, one time of day — re-measure before committing to a multi-hundred-GB pull. | 2026-09-07 |
| 5 | condor throughput and AFS/EOS quota | **Queue is clear and fast** — an 8-job × 8-core `workday` probe was fully started in 78 s and done in 131 s. Cluster-wide 285 running / 263 idle, 0 for dneff. **Storage: use EOS, not AFS.** `/afs/.../work/d/dneff` is 74 % of 100 GB → ~26 GB free; AFS `~` is 62 % of 10 GB. `/eos/user/d/dneff` is 41 % filled of a **2 TB** quota — ample for the campaign outputs. | 2026-09-07 |

---

## Stage 1 across the campaign — homogeneous from run_116 on

Four single-tag censuses, ~8,200 triggers each, spanning six weeks
(`candidate_filter.py`, one sub-run of each run staged by
`stage_validation_runs.sh`):

| run | date | INTER | INTRA | BUSY | NONE |
|---|---|---:|---:|---:|---:|
| run_79 | 26 Jul | **1.86 %** | **5.73 %** | **2.64 %** | 72.95 % |
| run_116 | 31 Jul | 0.96 % | 2.65 % | 1.20 % | 77.97 % |
| run_145 | 5 Aug | 1.01 % | 2.26 % | 1.27 % | 78.74 % |
| run_162 | 9 Aug | 0.98 % | 2.34 % | 1.15 % | 78.77 % |

**116, 145 and 162 agree to a few percent** — inside the ~10 % single-tag
sampling error, which is measured rather than assumed: run_145's one-tag census
gives INTER 1.006 % against 1.114 % for all seven tags.

**The campaign census's first full sub-run confirms both the method and the
anomaly.** run_79/`stat090_0006`, 14 tags and 109 272 triggers — 13× the spot
check — gives INTER 2.00 %, INTRA 5.40 %, BUSY 2.56 %, NONE 73.82 %, against
the single-tag 1.86 / 5.73 / 2.64 / 72.95 %. Everything agrees to ~7 %, so the
single-tag comparison above is sound, and run_79's excess is real at full
statistics rather than a small-sample effect. Its INTER events also carry a
median 409 clean strips against run_145's 162, which is the same noisier state
seen in the per-arm rates. So **run_145 is
representative of the bulk**, and the class partition is stable across the
production period.

**run_79 is genuinely different**, and the per-arm rates locate it: chamber A is
busy in **9.2 %** of triggers against 1.4–1.6 % in the later runs, a factor of
six. INTER, INTRA and BUSY all inflate together, which is what extra spurious
activity in one arm does to this partition.

**It is not the dead connector**, which was the obvious suspect — run_79 is the
only run in the sample carrying it. Channels 448–511 of FEU 3 fire at **0.3×**
the per-channel rate of the rest of that FEU, with 23 of 64 alive: exactly what
a partly disconnected connector should look like, and consistent with the
41-silent-channel record. What is elevated is the **rest of the FEU** — 34.98
hits/event against 13.53 (run_116) and 15.19 (run_162), 2.5× per channel.
Chamber A's x plane was broadly noisier in run_79, not locally broken, and
whatever changed had changed by 31 July; the 27 July access sits in between.

Stage 1 has no run_79-specific mask — its hot-channel cut is generic occupancy
(53 channels masked in run_79, 75 in run_116, 106 in run_162). The frozen sample
already carries `flag_a_x_mask` for run_79, so the campaign pass should either
mask it or quote run_79 separately.

**Cost, measured rather than extrapolated:** 8.3–14.0 core-hours per 10⁶
triggers, and every one of those numbers was taken while 14 reconstruction jobs
were saturating the machine, so they are upper bounds. At 25.6 M triggers the
campaign census is ~210–360 core-hours loaded, and the earlier ~135 estimate
looks right for an idle machine.

## Stage 3 — the track database, rebuilt on the full pass

`stage3_fullpass/tracks_run_145_stat090_{0000,0001,0002}.parquet`, built
2026-09-08. **240 065 track segments, 116 839 gated**, across all three
sub-runs — against 4 216 / 2 263 for the old allowlist-based table, a factor of
52.

It was rebuilt because the old one described the wrong sample. Stage 2 keeps
`INTER`/`INTRA`/`IMPLIED` whole and prescales `SINGLE`/`NONE` to 5 %/1 %, so a
track table built on it has the *prescale's* composition, not the campaign's:

| class | full pass | old (allowlist) | stage-1 census |
|---|---:|---:|---:|
| NONE | 70.1 % | 12.2 % | 78.2 % |
| SINGLE | 21.7 % | 18.1 % | 16.7 % |
| INTRA | 5.2 % | **34.1 %** | 2.5 % |
| INTER | 2.2 % | **26.6 %** | 1.1 % |

The full pass tracks the census, with the residual excess in `INTER`/`INTRA`
being the real fact that those events are likelier to yield a gated track —
which is what they are selected for. The old table is kept: it is the right
sample for studying the stage-2 selection itself, and nothing else.

Consistency checks, all exact: the gated count matches the funnel's 116 839
and every per-arm count matches too (A 25 550, B 15 208, C 23 340, D 52 741).
The null discipline holds — A, C and D carry finite angles on 100 % of gated
rows and **B on 0 %**, because B has no certified angle scale.

## The two-chamber rate, measured against a control — and it is null

`pairs.py`. An X17 at 16.8 MeV has a minimum opening angle of 109 deg, so its
pair lands in two chambers, and the chambers are opposed in pairs: **A +94.0
against C −85.8**, **D +3.8 against B −176.2** (measured from `run_config`).

**Counting two-chamber events directly does not work, and this retires the
earlier "two-chamber excess".** The trigger is a wall AND plastic coincidence
in ONE arm, which partitions events by arm: an event with an A track is
overwhelmingly an A-triggered event and is therefore *less* likely to carry a C
track. Measured: the A-pointing and C-pointing event sets overlap **3.5× less**
than independent expectation. An excess quoted against a product-of-marginals
null measures the trigger, not the physics — which is exactly what the stage-2
number was doing.

The controlled measurement fixes the track chamber and varies the **trigger**
chamber. B and D can serve as controls even without usable angles, because the
trigger arm comes from the n_TOF slim, not the reconstruction. Over 118 190
single-arm triggers:

| target cut | A track / C trig | C track / A trig | D track / B trig |
|---|---:|---:|---:|
| 20 mm | +1.78 σ | −1.04 σ | +0.67 σ |
| 30 mm | +1.95 σ | −0.24 σ | +1.36 σ |
| 50 mm | +2.83 σ | −1.79 σ | +2.25 σ |

**Combining the two directions** — the same physics measured twice, and a real
signal cannot average away while an acceptance asymmetry can. Over all three
sub-runs, **183 361 single-arm triggers**:

| target cut | A–C combined excess | significance | 95 % CL limit |
|---|---:|---:|---|
| 20 mm | −17.9 ± 14.2 | −1.26 σ | < 0.05 % of triggers |
| 30 mm | −11.8 ± 17.4 | −0.68 σ | < 0.06 % |
| 50 mm | −43.5 ± 23.0 | −1.89 σ | < 0.08 % |

**The apparent excess did not scale with statistics** — the cleanest argument
that it was never signal. A-track/C-trigger at dca < 50 mm reads 2.83 σ on two
sub-runs and **0.88 σ on three**; a real signal grows as √N. The control-choice
systematic grew with the sample instead, to **4.3 σ**.

**Null.** Three arguments, not one:

1. A back-to-back signal must be **symmetric**, and it is not — A-in-C-triggered
   is positive at every cut while its mirror C-in-A-triggered is negative at
   every cut.
2. The apparent excess **grows as the target cut is loosened**, which is
   backwards for something that points at the target.
3. The asymmetry has an ordinary explanation: A's ambient rate in non-A
   triggers is ~1.2 % against C's ~0.78 %, the same A–C quality gap the funnel,
   the lift and the pointing-confirmation rate all show.

B–D is the other X17 channel and only half of it is measurable until B has
angles.

### The geometry, validated end to end — for free

The same two-chamber events that give the null give the best validation in the
package, because it costs nothing extra: two tracks from the target into
**opposing** chambers must open wide, into **perpendicular** ones ~90°.
Measured at dca < 50 mm:

| pair | | n | median opening angle | above 109° |
|---|---|---:|---:|---:|
| A–C | opposing | 153 | **143.7°** | 93.5 % |
| A–D | perpendicular | 285 | 82.6° | 22.1 % |
| C–D | perpendicular | 194 | 96.9° | 30.4 % |

A–C has essentially no density below 95°. That one separation validates the
strip maps, **both** in-plane signs, the pinwheel, the chamber transforms and
the angle scale together — if any of them were wrong the distributions would
not separate. It is not evidence of a pair; the rate above is null.

## The funnel — published, and what it measures

`funnel.py` + `make_funnel_report.py`. Built on the **full** August waveform
pass (every trigger, every tag of sub-runs 0000 and 0001, no allowlist, no
prescale), so no number in it inherits a hits-based selection. `combined_hits`
enters at exactly one stage, the seeder, and only as a set of channels; the
seeder's acceptance is therefore a row in the funnel rather than an assumption.

122 280 triggers -> 75 473 gated 3D track segments -> 15 398 pointing-confirmed.

| | A | B | C | D |
|---|---:|---:|---:|---:|
| seeded / triggers | 38.9 % | 36.7 % | 39.6 % | **79.8 %** |
| gated tracks | 16 391 | 10 368 | 15 034 | 33 680 |
| wall+plastic \| tracked | 56.0 % | 39.9 % | 54.6 % | 32.0 % |
| same, seeded but NO track (control) | 37.7 % | 34.6 % | 45.6 % | 27.3 % |
| **lift** | **1.49x** | 1.15x | 1.20x | 1.17x |
| pointing-confirmed \| predictable | 43.9 % | 16.1 % | 35.9 % | 15.2 % |

Every chamber sits above its own no-track control, so the tracking selects real
particles rather than following the trigger. Two things to carry forward:

- **D seeds twice as often as anyone else and confirms worst.** Same threshold,
  same seeder — D passes clusters n_TOF does not back. Its track counts are an
  upper bound until that is understood.
- **LIQ C is effectively dead in run_145**: 891 in-time hits against 7 227 in A.
  LIQ is excluded from the partition for that reason.

**Published** to `/eos/user/d/dneff/www/x17/reco-funnel.html` ->
<https://dylan-neff.web.cern.ch/x17/reco-funnel.html> (HTTP 200 verified
2026-09-07). The DAQ machine (`daq_lxplus`, 128.141.177.17 and .103) is
**unreachable from lxplus** — by IP and by name, port 22 closed — so the DAQ
page's Analysis tab was not an option; that is expected with our run ended
2026-08-10, but it means the DAQ route needs re-testing before it is relied on.
`report.html` is a complete document (doctype, head, body) and `body.html` the
fragment form for the artifact publisher.

## Stage 3 — the track database, and what it says

`build_tracks.py`, 2026-09-07. One row per **3D track segment**: a paired
(x, y) candidate in one chamber of one trigger, keyed on
`(run, subrun, tag, event_id, arm, track_id)`.

### The condor pass

28 jobs (4 arms x 7 tags), cluster 4139919, **all succeeded, no held jobs, no
`FATAL`**. The workers honoured the allowlist — arm A tag 000 reports
`319 -> 174 seeded`, identical to the laptop run. First product:

| | |
|---|---|
| segments | **4 216** (A 921, B 860, C 1 272, D 1 163) |
| gated | **2 263** |
| events with >= 1 segment | 2 765 |
| by stage-1 class | INTRA 1 296, INTER 1 172, SINGLE 771, NONE 563, IMPLIED 271, BUSY 143 |

### The reco is deterministic per platform, not across platforms

Comparing the condor tables against the laptop's on the same tag, same
allowlist, same bundle: **12 of 770 events (1.6 %) differ**, and where they do
the difference is large (relative 0.4–1.0) — a *different candidate cluster*
winning a near-tie, not floating-point drift. `x_ok`, `y_ok` and `n_tracks` are
identical everywhere, so the gate decisions are stable; only the choice among
near-tied candidates moves. Per arm: A 1.1 %, B 1.0 %, C 1.5 %, D 2.7 %.

**This corrects an earlier claim in this file.** The 1-of-145 disagreement
between the filtered run and the August full pass was attributed to `wft/`
moving between the two. A ~1.6 % platform-dependent rate explains it on its
own, and the code-movement explanation is not needed and was not established.
A re-run *on the same machine* still reproduces bit-for-bit.

### Two rules the table enforces

**Every X/Y pairing is a row, gated or not.** `wft` pairs candidates and then
gates on `quality_ok & plausible` in both planes; `n_tracks` counts only
survivors — 69 of 128 pairings on tag 000 / arm A. Writing only the survivors
would make the gate's own efficiency unmeasurable from the product, and the 33
that are `quality_ok` but not `plausible` are exactly the marginal population a
later cut has to argue about. **The gate is a column, never a filter.**

**Nothing is invented.** `t_since_flash_ns` and `e_neutron_keV` are null (the
stage-1 time base is not written); `k_arm` is null (no in-situ angle scale is
published). Each carries its reason in the sidecar.

### The angle scale, measured

*Superseded the 2026-09-07 morning scan in this section, which minimised a
median over 23–60 tracks from one tag. Measured properly the same day on the
August full pass, both sub-runs, `sept26_prelim_analysis/k_arm.py`.*

Every bundle pins `v_drift = 42.6 um/ns` for all four arms — a Magboltz prior
for Ar/iso 90/10, never measured in these chambers with this gas. The fit
measures a transverse **speed** `w`; only `tan = w/v` needs the velocity, so
**positions are measured and angles are measured × an assumed constant**.
`k = v_prior/v_true` is the correction.

Three estimators, failure modes deliberately non-overlapping, all on the
pointing-coincident sample (wall segment **and** plastic bar confirmed):

| arm | k | v in situ | band | track | focus | focus plateau | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| A | 1.25 | 34.1 | 1.29 | 1.25 | 1.15 | 1.00–1.35 | PROVISIONAL |
| C | 1.58 | 27.0 | 1.79 | 1.58 | 1.55 | 1.30–1.85 | PROVISIONAL |
| D | 1.73 | 24.7 | 3.29 | 1.71 | 1.73 | 1.45–1.95 | NOT CALIBRATED |
| B | 1.62 | 26.3 | 8.48 | 1.62 | 1.33 | **0.60–2.55** | NOT CALIBRATED |

A and C reproduce between sub-runs to 2 %; their point estimates agree to
12–16 %. They are **provisional, not certified**, because the focus objective
is flat across ~35 % — that is the honest uncertainty on k, and it is large.
B's plateau spans the entire scan grid: B carries no angle information.

**A third sub-run confirms it.** Sub-run 0002, reconstructed on 2026-09-07 and
never used to fit anything, reproduces A and C exactly — A gives band 1.30,
track 1.27, focus 1.20 against 1.30/1.26/1.20 and 1.32/1.27/1.20 on the two
sub-runs the calibration was made on; C gives 1.78/1.61/1.60 against
1.77/1.58/1.55 and 1.81/1.66/1.60. And it fails B for a third time, on yield
alone: the charge-window coincident sample is **1 635 tracks in A, 1 203 in C
and 143 in B** — B is down by an order of magnitude on the very sample the
calibration needs.

**Two methodological traps, both live in `run145_target_imaging.py`:**

- `k_phys` is set to `k_track_coincident` *verbatim*. Reading it as a third
  opinion counts the per-track estimator twice and makes any arm look
  self-consistent. It is not read.
- `k_opt` minimises `r_core`, the median of the sub-30 mm population — a median
  conditioned on a cut that k itself moves, so it can be "improved" by
  shrinking the core rather than focusing it. (The imaging source already says
  its scan rails; this is why.) `k_arm.py` re-derives the focus estimator as a
  **count inside a fixed radius**, where the selection cannot move with the
  parameter, and reports the plateau.

**The stage-2 sample is background-dominated, and k exposed it.** Applying the
measured k makes the median axis-miss *worse* on the stage-2 allowlist reco
(A 54.6 → 56.7 mm, C 79.4 → 99.8 mm). On that sample the k that maximises
target pointing is ≈1.1 for *every* arm, regardless of the chamber — i.e. the
sample carries almost no target-track information. On the pointing-coincident
sample the optima are sharp and chamber-specific (A 1.20, C 1.55, D 1.70).
Anything measured on the allowlist sample alone inherits this.

`build_tracks.py` now **applies** k rather than recording it: once, in
`local_and_global`, so the direction, the pointing, the scintillator
predictions and the path length cannot land on different calibrations, with
`v_insitu = v_prior/k` for the depth. **An arm with no certified k gets NaN
angles, never a silent k = 1.** Positions are untouched.

The **source position** (zero crossing of the pointing band, −intercept/slope)
is scale-free — scaling every angle by k scales intercept and slope together —
so it is a geometry check, not evidence about v. A and C measure the same
global axis: −7.7 ± 0.4 mm and −11.5 ± 0.7 mm against a surveyed 0, stable to
0.01 mm and 0.9 mm between sub-runs.

### `q_uend` rails, so `q_per_len` had to be withdrawn

`drift_len_mm` first came out at a median 43–46 mm against a **30 mm** gap. Not
the drift velocity: `q_uend` is the last depth bin above 5 % of the profile
peak, `n_depth_bins = 18` at 60 ns, and **50.4 % of gated tracks sit exactly on
that 1080 ns edge**. For them q_uend is a censoring bound. So `drift_railed` is
a column, the raw time is kept, and `q_per_len` is **null where it rails** — a
censored denominator makes it a wrong number, not an uncertain one. Per arm:
A 33 %, C 45 %, D 47 %, B 61 %.

Consequence worth knowing: `wft`'s plausibility window is `250 <= q_uend <=
1100`, and the grid cannot produce more than 1080. **The upper bound is
unreachable**; only the shallow cut ever bites.

### The two-chamber excess — real, and not yet interpretable

Counting events with a gated track in >= 2 distinct chambers:

| stage-1 class | 1 arm | 2 arms | 3 | 4 |
|---|---:|---:|---:|---:|
| `INTER` | 310 | **111** | 0 | 0 |
| `INTRA` | 632 | 0 | 0 | 0 |
| `SINGLE` | 242 | 55 | 9 | 0 |
| `NONE` | 158 | 26 | 4 | 2 |
| `IMPLIED` | 73 | 25 | 0 | 0 |
| `BUSY` | 17 | 18 | 3 | 0 |

Two things fall out, and only the first is safe.

**Stage 1's `INTER` is not a clean selector of the reconstructed topology.**
Of 421 `INTER` events with any gated track, only **111 (26 %)** reconstruct in
two chambers; the rest lose an arm at the fit. And 142 two-chamber events come
from classes stage 1 did *not* call `INTER`.

**The extrapolation is where it stops being safe.** `SINGLE` is sampled at 5 %
and `NONE` at 1 %, so 64 and 32 found there scale to ~1 280 and ~3 200 across
the sub-run — which would make `INTER` a few per cent of the two-chamber
population. **Do not quote that number.** Three reasons: the gate is loose
(`quality_ok & plausible`, and half of `plausible` is unreachable, above); the
angles are uncalibrated, so the pointing that would separate a real pair from
two unrelated clusters does not yet discriminate (median axis-miss is 75–98 mm
in *every* class, flat); and a `NONE` event with two gated tracks is exactly
what an over-permissive gate produces. The measurement is real; the
interpretation waits on `k_arm`.

---

## Stage 2 — the event-id allowlist

Built 2026-09-07. `allowlist.py` turns the stage-1 class table into the list of
triggers stage 2 fits; `wft_beam.py --allow` consumes it; the condor packager
ships it and builds the job list from it.

### The unit of selection is the (event, arm) pair, not the event

This is where most of the saving comes from and it is worth stating plainly: an
`INTER` event needs its **two lit arms** fitted. The other two chambers have
nothing in them, and fitting them buys nothing but CPU.

| class | arms reconstructed | why |
|---|---|---|
| `INTER` | the 2 lit arms | the pair |
| `INTRA` | the 1 lit arm | both tracks are in it |
| `IMPLIED` | the lit arm **plus** every arm with an n_TOF coincidence and no track | forced — "was there a hint the hit-level finder missed?" is the only question `IMPLIED` exists to ask |
| `SINGLE` / `BUSY` / `NONE` | **all four**, prescaled | a control that only fitted the arms stage 1 already liked could not measure what stage 1 missed |

Control prescales are **per class**, not one global rate: `SINGLE` 5 %,
`BUSY` 10 %, `NONE` 1 %. Those classes differ by two orders of magnitude in
population, so a single rate would either bankrupt us on `NONE` or leave `BUSY`
in single digits. The draw is `blake2b(salt|run|subrun|event) < rate` — no RNG
state, reproducible on any machine, and adding a run does not repartition the
runs already drawn. **Changing `PRESCALE_SALT` redraws the whole control**, so
a redrawn control's efficiency is not comparable to the one before it; say so
here if it is ever bumped.

### The cost, measured

run_145/`stat090_0000`, and the three censused sub-runs agree to 2 %:

| class | triggers | selected | (arm, event) fits | of a full reco |
|---|---:|---:|---:|---:|
| `INTER` | 643 | 643 (100 %) | 1 286 | 0.56 % |
| `INTRA` | 1 430 | 1 430 (100 %) | 1 430 | 0.62 % |
| `IMPLIED` | 181 | 181 (100 %) | 398 | 0.17 % |
| `SINGLE` | 9 629 | 475 (4.9 %) | 1 900 | 0.82 % |
| `BUSY` | 687 | 687 (100 %) | 2 748 | 1.19 % |
| `NONE` | 45 172 | 439 (1.0 %) | 1 756 | 0.76 % |
| **total** | **57 742** | **3 855 (6.7 %)** | **9 518** | **4.12 %** |

Signal is 3 114 of those fits, the control 6 404 — **the control is two thirds
of the budget.** That is the price of having an acceptance at all; it is the
first number to cut if the budget moves, and cutting it means saying what the
spectrum's acceptance is then based on.

Timing, tag `260805_14H06_000`, all four arms, 8 workers on the laptop
(`stage2/bench/bench_run145_*.csv`):

| arm | allowed | seeded/fitted | wall | core-s | core-s/fit |
|---|---:|---:|---:|---:|---:|
| A | 319 | 174 | 27.3 s | 171 | 0.98 |
| B | 352 | 209 | 48.7 s | 306 | 1.47 |
| C | 372 | 201 | 38.2 s | 254 | 1.26 |
| D | 302 | 186 | 28.5 s | 162 | 0.87 |

893 core-s for 770 fits over 8 353 triggers = **0.107 core-s per trigger across
all four arms**, so:

| | triggers | core-hours |
|---|---:|---:|
| one sub-run | 57 742 | **1.7** |
| run_145 (24 tags) | 189 724 | **5.6** |
| **the campaign** | 25 598 064 | **≈ 760** |

**PLAN.md §5 budgeted ~1 900 core-hours** from the 3.80 % event-level selection.
The measured number is 40 % of that, for two reasons the plan did not model:
arm scoping (2.47 arms per selected event, not 4) and the seeder firing on only
56 % of allowlisted arm-events. Includes interpreter start-up and I/O in every
figure, so it is an upper bound.

Taking `BUSY` whole rather than at 10 % cost **+24 %** (612 → 760 core-hours) —
more than its share of fits, because the `BUSY` events that *do* seed are the
dear ones: the 93 extra fits per tag cost 1.87 core-s each against a 1.16
average.

**CPU is not the constraint** — 8 200 jobs each pulling ~290 MB from EOS is, and
the current one-job-per-(arm, tag) design fetches the same `combined_hits` file
four times. Left alone for now; noted as the thing to fix if the condor pass is
I/O-bound.

### It reproduces the August full pass, row for row

run_145 `stat090_0000`/tag 000/arm A already has a **full, unfiltered** table
from the 2026-08-19 condor pass. The filtered run against it:

- all 145 event ids are a **subset** of the full table's 3 233 — the allowlist
  removes events, it does not invent them;
- **144 of 145 agree on all 52 columns exactly**;
- the one difference (event 1048) is a *different candidate cluster* chosen —
  31 vs 34 strips, χ² 17 386 vs 18 800, both far past `quality_ok`. Re-running
  the filtered job reproduces itself bit-for-bit, so this is not
  non-determinism: `wft/` moved between the August pass and HEAD (`d044073`,
  `b9c7856`, `6247750` — the kernel-inversion work). **Not an allowlist effect.**

### What the allowlist costs at the seeder — the number that must not be lost

An allowlisted event the beam seeder finds no cluster for produces **no row**,
and a missing row is indistinguishable downstream from an event that was never
selected. Unmeasured, that seeding loss silently becomes a reconstruction
inefficiency attributed to physics. So it is measured, per tag and per arm, and
written into every `.meta.json` as `allowlist.n_allowed/n_seeded/n_missing`.
Against the full August pass (`allowlist.py --seed-eff`):

| class | reason | n | seeded | frac |
|---|---|---:|---:|---:|
| `INTER` | lit | 1 286 | 1 267 | **0.985** |
| `INTRA` | lit | 1 430 | 1 404 | **0.982** |
| `IMPLIED` | lit | 181 | 176 | 0.972 |
| `IMPLIED` | forced_silent | 217 | 173 | **0.797** |
| `SINGLE` | control | 1 900 | 994 | 0.523 |
| `NONE` | control | 1 756 | 843 | 0.480 |
| `BUSY` | control | 2 748 | 478 | **0.174** |
| | **all** | 9 518 | 5 335 | 0.561 |

Three things to read off it:

1. **The signal path is safe.** 98.3 % of `lit` (arm, event)s seed — 97.3–99.8 %
   across arms. Stage 1 → stage 2 loses 1.7 % of what stage 1 identified.
2. **`forced_silent` seeds at 80 %.** In four cases out of five where n_TOF says
   a particle crossed a chamber and stage 1 found no track, the *waveform*
   seeder does find a clusterable deposit. `IMPLIED` is not an empty class, and
   the forced fit has something to work on.
3. **`BUSY` seeds at 17 %** — and the reason is not a threshold mismatch to be
   tuned away. See below. **Resolved by taking the class whole.**

Per-arm seeding runs A 0.64, B 0.64, C 0.69, **D 0.84** — D seeds most, as its
higher raw track rate in stage 1 already said.

### `BUSY` is flooded, not crowded — so it is taken whole

The first reading of that 17 % was that stage 1's "busy" (> 120 strips in an
arm) and the seeder's (> 150 hits in a plane) are different tests and the seeder
was throwing away crowded multi-track events. **That is wrong, and the data
says so plainly.** Splitting the 256 `BUSY` arm-events of `stat090_0000` by
clean strips in the arm:

| clean strips in the arm | n | seeded | frac |
|---|---:|---:|---:|
| ≤ 200 | 16 | 16 | **1.000** |
| 200–400 | 57 | 12 | 0.211 |
| > 400 | 183 | 16 | 0.087 |

97 % of the arms in a `BUSY` event carry more than 120 clean strips, and the
median arm that fails to seed carries **493** — roughly half the chamber's
channels lit. Those are discharges and flashes, not tracks, and
`BUSY_PLANE_HITS = 150` is rejecting them correctly. Everything a chamber can
still be read out for seeds at **100 %**.

So there is nothing to fix in the seeder, and no prescale recovers the flooded
part — that information is not in the data. What the 10 % prescale *was* doing
was leaving 44 arm-events to characterise the whole class. **`BUSY` prescale is
now 1.00.** It costs little (the class is 1.2 % of triggers and 83 % of it
never reaches the fit) and it buys a real number on the reconstructable part
plus a measured count of the flooded part, which is itself a detector-QA
quantity worth having per run.

With `BUSY` whole, `stat090_0000` selects **9 518** (arm, event) fits — 4.12 %
of a full reco, 5 335 of them seeded.

### Files

```
sept26_prelim_analysis/allowlist.py          build, cost, seed efficiency
ntof_tracking/wft_beam.py                    --allow, load_allowlist(), sidecar
ntof_tracking/condor/run_beam_job.py         --allow, fails loud if missing
ntof_tracking/condor/make_beam_package.py    --allow: ships it, builds jobs from it
ntof_tracking/condor/beam_reco.sub           $(allowfile) in transfer_input_files
<out>/stage2/allowlist_<run>_<subrun>.json     what the worker reads
<out>/stage2/allowlist_<run>_<subrun>.parquet  the join key stage 3 needs
<out>/stage2/allowlist_cost_*.csv, seed_efficiency_*.csv, bench/
```

A job that runs **without** its allowlist reconstructs the whole tag and looks
perfectly fine doing it, so both the packager and the worker fail loudly rather
than fall back: `run_beam_job.py` exits if `--allow` names a file condor did not
transfer, and `load_allowlist` raises on an arm the document does not mention.

---

## Stage 1 — the full run_145 census, and what it says about `INTER`

`candidate_filter.py` over **all three sub-runs of run_145: 24 file tags,
189 724 triggers**, ~59 min wall at 52–53 triggers/s.

### The census is stable; the selection fraction is 3.80 %

| class | n | fraction | `0000` | `0001` | `0002` |
|---|---|---|---|---|---|
| `INTER` | 1 935 | **1.020 %** | 1.114 % | 1.077 % | 0.885 % |
| `INTRA` | 4 713 | **2.484 %** | 2.477 % | 2.583 % | 2.396 % |
| `IMPLIED` | 566 | **0.298 %** | 0.314 % | 0.285 % | 0.298 % |
| `SINGLE` | 30 703 | 16.183 % | 16.68 % | 16.10 % | 15.84 % |
| `BUSY` | 2 222 | 1.171 % | 1.19 % | 1.17 % | 1.16 % |
| `NONE` | 149 585 | 78.843 % | 78.23 % | 78.79 % | 79.42 % |

`INTER + INTRA + IMPLIED` = **3.80 %** → stage 2 ≈ **1 900 core-hours** at ~2 arms
per selected trigger. Stage 1 itself: **5.2–5.3 core-hours per 10⁶**, so
~135 core-hours for the campaign.

**The hot mask is stable across sub-runs** — 102, 103, 105 channels, each
measured independently from a different file tag. That is the evidence for
treating it as a per-run-condition calibration rather than something that
drifts file to file.

### `INTER` is the one class that is not Poisson — and it is not understood

χ² against a constant rate over the 23 full tags:

| class | mean | min | max | χ²/dof |
|---|---|---|---|---|
| **`INTER`** | 1.022 % | 0.685 % | 1.351 % | **2.85 — over-dispersed** |
| `INTRA` | 2.479 % | 2.083 % | 2.975 % | 1.53 |
| `IMPLIED` | 0.297 % | 0.192 % | 0.396 % | 0.67 |
| `SINGLE` | 16.18 % | 15.32 % | 17.55 % | 1.67 |
| `BUSY` | 1.172 % | 1.095 % | 1.298 % | 0.17 |
| `NONE` | 78.85 % | 76.86 % | 80.08 % | 0.74 |

**`INTER` should be Poisson.** The trigger is a wall+plastic coincidence in any
one arm, so every event contains one guaranteed particle; the second track is
the pair partner from the same interaction, i.e. pair-creation physics, whose
probability per triggered event is a constant. It scales with neither the
per-arm rate squared nor the beam intensity. So over-dispersion is an artefact
to be found, not a feature to be explained.

And it is not scatter. **Five *consecutive* tags spanning the sub-run boundary**
(`14H06_004,005,006` → `15H07_000,001`, roughly 14:40–15:25 on 5 August) sit at
**1.333 %** against **0.935 %** for the other eighteen — a 43 % elevation
lasting ~40 minutes.

**What has been excluded, each by measurement:**

| candidate cause | verdict |
|---|---|
| High voltage | **no** — mesh stable to 0.003–0.09 V, drift to 0.16 V over the whole run |
| Mesh current (≈ rate) | **no** — currents step up 13–35 % at 14:30 and *stay* up to 17:00, while `INTER` rises at 14:40 and falls back at 15:24 |
| Hot-mask scoping | **no** — see below |
| Beam intensity | ruled out for the pair component on physics grounds. *Not* excluded for an accidental second-track component, which would scale with rate |

**The mask does drift, and it does not matter.** Measuring the mask
independently on each of the 7 tags of `stat090_0000`, its membership moves
steadily away from the tag-000 mask that `run_tags` applies to all of them —
Jaccard 1.000 → 0.931 → 0.885 → 0.905 → 0.824 → 0.814 → 0.843, with arm A's
masked count growing 18 → 26 and arm C acquiring channels it did not have
early on. So the detector state genuinely evolves within an hour.

But re-running the first elevated tag (`14H06_004`) with **its own** mask
(98 channels) instead of the shared one (102) gives **`INTER` = 111 events
either way — identical to the event**. The classification is insensitive to the
~16 % mask churn, so the elevation is real data, not a processing artefact, and
the "measure once per sub-run" scoping stands.

**Status: deferred to October, by Dylan, 2026-09-07.** A ~40 % time-dependent
excursion on `INTER`, cause unknown, in the one class the analysis is built
around. Dylan's read: most likely one of the marginal chambers going noisy for
a while, or any of several similar things, and **it is not diagnosable at
stage 1** — telling a noisy chamber from a real pair excess needs the
reconstructed tracks and their match to the n_TOF scintillators, which is
stage 2 and stage 4. Do not spend more stage-1 effort on it.

It does not move the 3.80 % selection fraction much and it does not change what
stage 2 does. **Re-open it once the track database exists**: the same 24-tag
split, but cutting on `quality_ok`, on the two arms' `t0` agreement, and on
whether both tracks point at the target. If it is noise, the excess events fail
those cuts; if it survives them, it is something else. Whoever picks this up
wants the tag table in `stage1/census_per_tag_run_145_*.csv` and the window
14:40–15:25 on 5 August 2026.

### What stage 1 cannot do, and two things I got wrong on the way

**Stage 1 has no inter-arm timing.** A real pair is simultaneous; two unrelated
tracks are spread over the ~1.2 µs drift window. Nothing in this stage
distinguishes them — the per-arm waveform `t0` from stage 2 is that
discriminant. **No statement about the opening angle, or about how much of
`INTER` is signal, can be made from the stage-1 ledger alone.**

Two errors made and corrected while getting here, recorded because both are
easy to repeat:

1. **Treating the four arms as independent.** An "accidental" model of the form
   P(2 arms) = Σᵢⱼ pᵢpⱼ, and the symmetric event mixing built on it, are both
   **wrong for a triggered sample**. The trigger guarantees one particle, so the
   arms are anti-correlated by construction and the second track is not p². The
   29 % "deficit against accidentals" this produced was the trigger constraint,
   not a physics result. `event_mixing_background()` now raises rather than run.

   **Fixing it did not rescue the method.** The corrected estimator
   (`accidental_second_track()`) identifies the trigger arm from its
   wall+plastic coincidence, holds it fixed and mixes only the others — and
   returns observed 6.2335 % against "accidental" 6.3154 ± 0.0082 %, a ratio
   of 0.99. That is an identity, not a measurement: **permuting an arm's flag
   among the events where it is not the trigger arm conserves that arm's total
   count exactly**, so mixing only reassigns which event each second track
   lands in. Measured directly — 11 459 non-trigger lit arms over 11 077
   events, and the whole 1.3 % gap is clumping (372 events with ≥ 2), not
   content.

   **The limit is structural.** The stage-1 ledger holds only counts, and no
   permutation of counts can separate a pair partner from an unrelated track:
   what distinguishes them is that the pair is simultaneous and points at the
   target, i.e. timing and geometry. `PLAN.md` §5 puts event mixing at stage 5
   for exactly this reason; at stage 1 it is vacuous. The function is kept,
   renamed to the quantity it does measure (second-arm clumping), and says so
   in its own docstring.
2. **Concluding `IMPLIED` was broken.** From the same p² error: "if the
   coincidence flags were independent P(≥2 arms) would be 24.9 %, and it is
   1.65 %". The trigger requires *one* arm's coincidence and does not suppress a
   second, so the 1.65 % is the genuine second-particle population plus
   accidentals — exactly the parent `IMPLIED` draws from. The class is sound.

What the n_TOF data does show, and this stands: per arm,
P(coincidence | arm lit) against P(coincidence | arm not lit) is 68 %/16 % on A,
73 %/23 % on C, 63 %/23 % on D and 43 %/27 % on B — lifts of 4.3, 3.2, 2.8 and
1.6. The track finding is real, and B is the weakest, consistent with
everything else known about B.

---

## Stage 1, earlier — and the arm-D anomaly is solved

`candidate_filter.py` → `/media/dylan/data/x17/sept26_prelim/stage1/`
(`candidates_*.parquet`, `census_*.csv`, `arm_rates_*.csv`, `hot_mask_*.csv`,
`bench_*.json`). Numbers below are the **full file tag**,
run_145/stat090_0000/260805_14H06_000, 8 353 triggers.

### The class census

| class | n | fraction | before the `INTRA` fix |
|---|---|---|---|
| `INTER` | 84 | **1.01 %** | 1.01 % |
| `INTRA` | 189 | **2.26 %** | 2.66 % |
| `IMPLIED` | 26 | **0.31 %** | 0.31 % |
| `SINGLE` | 1 371 | 16.4 % | 16.0 % |
| `BUSY` | 106 | 1.27 % | 1.27 % |
| `NONE` | 6 577 | 78.7 % | 78.7 % |

**`INTER + INTRA + IMPLIED` is 3.6 % of triggers**, not the ~1 % `PLAN.md` §5
hoped for. At ~2 arms reconstructed per selected trigger that is
**≈ 1 800 core-hours** of stage 2 across the campaign — a day or two on condor
at this morning's throughput, so it is affordable, but it is *the* number that
decides whether the filter tightens, and one file tag of one sub-run of one run
is not yet the campaign. A sweep over all three sub-runs is running.

### Two bugs found by reading the code, not by watching it run

Both were caught before they reached a result that mattered; the second one
moved a headline number.

**The tag path loaded the whole sub-run first.** Asking for one file tag read
all ~13 M hits of the sub-run, discarded them, then read the one file — about
**8x the necessary I/O** across a `run_tags` sweep, with the 54 MB slim
re-parsed once per tag on top. Fixed, and `arm_flags` is now memoised per
sub-run: **load per tag 18.3 s → 1.5 s**. Some of the slowness earlier
attributed to rsync contention was actually this.

**`INTRA` was counting other chambers' doubles.** `n_sep_best` took the maximum
separated-cluster count over **all four arms**, but it is only consulted where
exactly one arm is lit — so two resolvable clusters in a *different, unlit*
chamber classified the event `INTRA`, booking one chamber's pair against
another. It now counts inside the lit arm only, with the all-arms value kept as
`n_sep_any_arm` so the size of what is excluded stays measurable.

It cost **15 % of `INTRA`** (222 → 189 on the reference tag; the 33 events moved
to `SINGLE`) and nothing else — `INTER`, `IMPLIED`, `BUSY` and `NONE` are
unchanged to the event, which is exactly what the fix predicts, since the bug
lived only in the one-lit-arm branch. The selection fraction went 4.0 % → 3.6 %
and the stage-2 estimate down by ~10 %.

### C3 resolved: arm D was half hot channels

`PLAN.md` §3 stage 1 step 1 calls for a dead/hot channel mask; none existed
(D10 is deferred). Measured per channel on run_145: **every arm's per-plane
median occupancy is 1.1–2.5 %**, so the arms are alike in the bulk. The tail is
not — **arm D has 26 channels above 20 % occupancy on each plane against 0–4
for A, B and C**, clustered at connector boundaries (x 92–127, x 448–450). An
excess concentrated on individual channels while the plane median is
unremarkable is instrumental; physics spreads across a plane.

`candidate_filter.hot_channel_mask` masks a channel firing in **> 8 % of
triggers *and* > 8× its own plane's median** — both, not either. What it does:

Both sides measured on the full 8 353-trigger tag:

| | A | B | C | D |
|---|---|---|---|---|
| channels masked (of 1 024) | 19 | 1 | **0** | **82** |
| strips/event removed | 2.7 | 0.2 | **0.0** | **23.2** |
| track rate (both planes), **unmasked** | 8.00 % | 3.69 % | 8.84 % | **6.17 %** |
| track rate, **masked** | 7.99 % | 3.71 % | 8.84 % | **4.91 %** |
| median clean strips, unmasked | 3 | 3 | 0 | **39** |
| median clean strips, masked | 0 | 3 | 0 | **11** |

Figure: `stage1/figures/arm_rates_mask_stat090_0000_tag000.png`.

Its effect on the census is to remove candidates, not create them —
`INTER` 1.22 % → 1.01 %, `INTRA` 2.93 % → 2.66 %, `NONE` 77.7 % → 78.7 %. About
1 % of triggers were being promoted out of `NONE` by D's hot channels alone.

**B and C are unchanged to the digit and only D moves** — the signature of a
targeted instrumental fix rather than a cut that flatters the data. After it
the arms order the way the calibration says they should: A and C, the good
pair, ahead of B and D. About 1.3 % of triggers were being promoted out of
`NONE` by D's hot channels alone.

Two honest caveats, both in the docstring: the mask is **measured on the same
data it is applied to** (a dedicated pedestal-run map is D10), and it is **per
run condition** — it must be re-measured, never carried across the 23-July
boundary. It is written out beside the candidates so a later D10 map can be
diffed against it.

### Benchmark

| what | value |
|---|---|
| throughput | **51 triggers/s** single core (noise + classify), 8 353 triggers |
| campaign cost, stage 1 | **5.4 core-hours per 10⁶ triggers → ≈ 140 core-hours for 25.6 M** |
| hot channels masked | 102 of 4 096 (run_145, both planes, four arms) |

`PLAN.md` §5 budgeted ~10³ core-hours for stage 1; it is ~7× cheaper than that,
so stage 1 over the whole campaign is an overnight condor job and not a
constraint on the week.

⚠ **Do not benchmark on this laptop while rsync is running.** The analysis and
both transfers share one `fuseblk` mount; with two rsyncs active the classifier
ran at ~23 % duty cycle (99 % CPU but 2 min of CPU per 8 min of wall). The
51 ev/s above is from a run that overlapped the transfers and is therefore a floor, not a ceiling.

---

## Stage 0, frozen

`freeze_sample.py` → `/media/dylan/data/x17/sept26_prelim/stage0/`
(`sample.csv` 2 270 rows, `cut_ledger.csv`, `sample_summary.json`,
`figures/sample_timeline.png` + `.csv`). Re-runnable; nothing here is typed in
by hand.

| | |
|---|---|
| **runs** | 36 |
| **sub-runs** | 293 |
| **triggers** | 25 598 064 |
| **beam hours** | 238.2 of 498 in the campaign |
| **matched pulses** | 239 675 of 239 679 delivered |
| **slim volume** | 20.2 GB |
| **EOS footprint** | 8.31 TB |
| **carries the arm-A x mask** | run_79 only (§2.2) |

### The cut ledger

Each sub-run is charged to the **first** cut it failed, so the column sums to
the campaign and nothing is double-counted.

| cut | sub-runs | runs | beam h |
|---|---|---|---|
| before the production trigger (run_79) | 1 822 | 71 | 239.9 |
| **kept** | **293** | **36** | **238.2** |
| not beam physics (cosmics) | 111 | 47 | 16.7 |
| detector-A resist × drift HV scan on beam (run_161) | 25 | 1 | 2.4 |
| not beam physics (pulser) | 9 | 3 | 0.4 |
| watermark × inter-packet-delay DAQ scan (run_82) | 8 | 1 | 0.6 |
| not joined to n_TOF | 2 | 2 | 0.1 |

**The production-trigger cut is the whole story**: it costs 240 of the 498 beam
hours, and after it almost nothing else is removed. Every one of the 36
surviving runs was already ³He + Ar/Iso 90/10 + `st=complete` + 8 FEUs, so
those four cuts cost *nothing* — they are worth stating precisely because they
turned out to be free.

### The honest denominator

The sample is 293 sub-runs; **296 exist**. `freeze_sample.unenumerated()`
reports the difference rather than letting it vanish:

- **run_120** and **run_137** (44 events) are in the ledger but never joined to
  n_TOF — they appear as cut rows.
- **run_145 `stat090_0003`** has no pulse-ledger record *and* no slim product,
  so neither input names it and it would otherwise be invisible. It is a 3.3 MB
  stub sub-run of the development run.

Sub-run names differ across all three inputs (`stat090_0000` in the slim
inventory, `0000` in the pulse ledger, unnamed in the run survey).
`freeze_sample._short()` is the single place that reconciles them and
`_check_join()` **raises** if the reconciliation ever stops covering the
sample — a silent join failure would look exactly like a missing slim product
and would quietly shrink every downstream denominator.

---

## What the verification changed

Three things N2 found that the plan did not assume. Each is a *change to the
plan*, not a detail — recorded here so §5 of `PLAN.md` is read against them.

### C1 · The link is no longer the bottleneck, so the disk is

`PLAN.md` §4 and the run_145 handoff both build on **310 kB/s**, measured in
August, and that number is what makes "run the reco at CERN and bring back the
parquet" the only route. Measured today from this laptop: **17.5 MB/s from AFS
and 36 MB/s from EOS** — 56–115× faster. run_145's entire `decoded_root`
(19.7 GB over its three real sub-runs) is **~10 minutes**, not 60 hours.

This does *not* overturn the condor design — 8.3 TB of campaign waveforms is
still work that belongs where the data is, and one measurement at one time of
day is not a guarantee. What it overturns is the **development** loop: the
waveform path can now be debugged locally on real data, which `PLAN.md` §4
explicitly gave up on.

**The new constraint is local disk.** `/media/dylan/data` is 99 % full
(6.8 GB free of 477 GB) and `/home` has 9.9 GB. One sub-run of waveforms does
not fit. Freeing disk is now a prerequisite for N3's step 4, and it is the one
thing here that cannot be done from the CERN side.

### C2 · Campaign outputs go to EOS, not AFS

`/afs/cern.ch/work/d/dneff` is at 74 % of its 100 GB — **~26 GB free**, and AFS
`~` has ~4 GB. The candidate ledger alone is "a few GB" by `PLAN.md` §3, before
any reco output. `/eos/user/d/dneff` is at 41 % of a **2 TB** quota. Every
campaign product from stage 1 on should be written there; the condor package
directory can stay on AFS because it is small.

### C3 · Arm D seeds twice as often as A, B and C

Across all 15 reconstructed run_145 tags: **A 47 546, B 44 896, C 48 470,
D 97 537** events. Per tag that is ~80 % of triggers seeded on D against
37–39 % on the other three, and it is not duplication — every `event_id` is
unique and all four arms span the same [1, ~8 350] range per tag.

This matters directly for stage 1. `INTER` is defined as "track-like clusters
in exactly 2 arms", so an arm that fires on twice as many triggers will
dominate the class census and push real one-arm events into `INTER`. **The
cluster taxonomy has to be checked per arm before the census is believed**, and
D's seeding threshold is the first thing to look at. Whether this is occupancy,
noise or a threshold difference is not yet known — it is a new open question,
not a known one.

---

## Decisions taken

| date | decision | why |
|---|---|---|
| 2026-09-07 | **run_145 is the local development run** | post-23-July and post-run_83 so neither campaign-wide condition bites; small (62 GB, 4 sub-runs); already reconstructed on all four arms with `calib_bundle_r06`; its slim join exists; its pointing result is a ready-made sanity check |
| 2026-09-07 | **intra-chamber = both tracks in one chamber; inter-chamber = one track in each of two** | fixes the naming used in the original brief. The X17 opening angle is ≥ 109°, which cannot fit in one chamber's ±40°, so **the signal is the inter-chamber topology** and the intra-chamber one is the low-angle control |
| 2026-09-07 | **filter on hits before reconstructing** | reconstructing the campaign blind is ~10⁵ core-hours. The filter is what makes the week possible, and its own efficiency is the price (D13) |
| 2026-09-07 | **B and D are tagging chambers this week** | their in-situ k (1.99, 1.70) is unphysical and B truncates half its columns. They say *whether* there was a track; their angles carry a flag and are never quoted alone |
| 2026-09-07 | **run_67 and run_68 are out of the sample** | run_67 is the quiet noise configuration and run_68 is inside the pedestal bracket and unplaced. Neither can be pooled with the production period (D14) |
| 2026-09-07 | **no invariant mass** | needs scintillator calorimetry, which needs a calibration we do not have (D6). The opening angle is the deliverable |
| 2026-09-07 | **the analysis runs on the Ubuntu laptop** | it is the only machine here with lxplus, and every path in the repo's documentation already assumes it |

---

## Benchmarks

Filled in as they are measured. Empty is honest; a guess is not.

| what | value | measured on |
|---|---|---|
| wft reco throughput | — | |
| stage-2 filter throughput | — | |
| class census (fraction per class) | — | |
| campaign reco cost estimate | — | |
| link speed to CERN | **17.5 MB/s** (AFS), **36 MB/s** (EOS), rsync over ssh | 2026-09-07, Ubuntu laptop, 14:10 CEST |
| per-arm seeding rate, run_145 | A 39 %, B 37 %, C 39 %, **D 80 %** of triggers | 2026-09-07, 15 tags, `wft_beam145` |
| condor queue latency, `workday` 8-core | **78 s** to first start, 101 s for the last of 8; every job got its 8 CPUs | 2026-09-07, 8-job probe, cluster 4139571 |
| condor worker spread | ±15 % on a fixed sha256 loop across 6 distinct workers | 2026-09-07, same probe |

The only number in hand is the historical one: run_79, 12 534 events on
6 cores in ~1 h 50 m ≈ **1 100 events per core-hour per chamber**, on
20-sample windows with `WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1`. Treat it as an
order of magnitude until re-measured.

---

## Log

**2026-09-09 (GANIL)** — a second, separate analysis: the same 3He and capsule
pair backgrounds at a 1-40 MeV neutron beam (NFS at GANIL). Section above for
the detail. The three headlines: the 4He excitation becomes a variable so the
X17 opening angle slides 104->39 deg, which is a discriminant rather than a loss
because E_n is measured per event; the gas converts ~280x more of its neutrons
into radiative capture than at thermal, and the (n,p) two-prong load falls 10^4;
and below 2.29 MeV the capsule cannot make a pair at all, because its two
strongest inelastic lines are under the pair threshold. Recommendation: run
below 2.29 MeV. Both nuclear-data libraries stop at 20 MeV, which caps the study
rather than the facility.

**2026-09-09 (IPC)** — the expected internal-pair spectrum, done as a spectrum,
plus the aluminium capsule from its real capture scheme. Section above for the
detail. The three headlines: the prediction is one curve for the whole >1 ms
window (2e-9 total variation between 1 ms and 1 s, and two of the four possible
energy dependences cancel exactly because they are ratios of 1/v channels); the
capsule's wide-angle background is made by the 2-5 MeV E1 primaries and not by
the 7.7 MeV ones, which are M1 (and 14 % of it is carbon fibre, not aluminium);
and per capture the capsule and the gas are equally
dangerous, so the whole problem is that there are 10^4-10^6 times more capsule
captures. Along the way: `results_3He` appears to compute 3He radiative captures
without self-shielding in a cell that is optically thick to thermal neutrons,
which would be worth ~10^2 on every expected yield the experiment quotes — not
confirmed, and it is a question for the table's author, not a finding.

**2026-09-07 (latest+1, Ubuntu)** — stage 2 ran at CERN and stage 3 exists.
Two corrections and one thing that has to be fixed before any physics.

*The condor pass.* 28 jobs, all succeeded, allowlist honoured — arm A tag 000
gave `319 -> 174 seeded`, identical to the laptop. 4 216 track segments,
2 263 gated.

*Correction: the reco is deterministic per platform, not across platforms.*
12 of 770 events (1.6 %) differ between lxplus and the laptop on identical
inputs, always as a different candidate winning a near-tie; gate decisions are
identical everywhere. That rate explains the earlier 1-of-145 disagreement with
the August pass on its own, so my attributing it to `wft/` moving between the
two was not established and is withdrawn.

*Correction: `q_per_len` was wrong and is now null where it rails.*
`drift_len_mm` read 43-46 mm against a 30 mm gap. `q_uend` is quantised to the
model's 18-bin depth grid and half the gated tracks sit exactly on its 1080 ns
edge, so it is a censoring bound, not a depth. The derived rate is withdrawn
rather than reported; `q_total` stands. Same inspection found that `wft`'s
plausibility window has an unreachable upper bound.

*The thing to fix.* Every bundle pins v = 42.6 um/ns for all four arms and the
pointing says that is wrong by 1.1x (A) to 1.8x (D) — exactly the wet/dry split
`V_DRIFT_PRIOR` predicted. Angle scale goes as 1/v, so **no angle in the track
table means anything yet**, and that is why `k_arm` is a declared null instead
of a silent 1.0. It also means the two-chamber excess found in `SINGLE` and
`NONE` — which prescale-corrects to something far larger than `INTER` — is a
real measurement with no safe interpretation: the pointing that would separate
a pair from two unrelated clusters is flat across every class at 75-98 mm.
Measure `k_arm` first.

**2026-09-07 (latest, Ubuntu)** — Stage 2's allowlist: built, benchmarked,
validated against the August full pass, and a third the budgeted cost.

*The `INTER` excursion is parked, on Dylan's call.* It is not diagnosable at
stage 1 — separating a chamber going noisy from a real pair excess needs
reconstructed tracks matched to the n_TOF scintillators. October, with the
track database in hand; the re-open recipe is written into the section above.

*The allowlist.* `allowlist.py` selects **(event, arm) pairs**, not events, and
that scoping is most of the saving: 2.17 arms per selected event rather than 4.
`INTER`/`INTRA`/`IMPLIED` in full, plus `IMPLIED`'s n_TOF-coincident silent arms
forced, plus a per-class-prescaled control over all four arms. The draw is a
blake2b hash, not an RNG, so it is reproducible with no stored state and a rate
change moves only the events near the old threshold.

*It reproduces the August pass.* On tag 000/arm A, 144 of 145 events agree with
the unfiltered table on all 52 columns; the one difference is a different
candidate cluster in a χ² ≈ 18 000 event, and re-running reproduces itself
bit-for-bit — `wft/` moved between 2026-08-19 and HEAD, the allowlist did not
cause it.

*The cost is a third of the plan's.* 0.086 core-s per trigger across four arms
→ **≈ 610 core-hours campaign-wide** against PLAN.md §5's ~1 900. Arm scoping
and a 70 % seeder hit rate account for the difference. The constraint is not
CPU: it is 8 200 condor jobs pulling ~290 MB each from EOS, with `combined_hits`
fetched four times over. Left as-is, flagged.

*One number worth having found.* Of the events the allowlist selects, the beam
seeder produces a cluster for 98.3 % of the `lit` ones — the signal path is
safe — but only **17 % of `BUSY`**, because stage 1's "busy" (> 120 strips in an
arm) and the seeder's (> 150 hits in a plane) are different tests and the
seeder throws the planes away. At a 10 % prescale that is 44 seeded arm-events:
not a measurement. Either the `BUSY` prescale goes up or `BUSY` is stated to
have no acceptance. Open, and cheap to fix before the campaign pass.

**2026-09-07 (late, Ubuntu)** — Stage 1 written and running; the arm-D anomaly
turned out to be instrumental.

*The upload was never slow.* The "5 MB/s" in the previous entry was a
measurement error of mine: every window it was averaged over also contained a
download, a 3-route upload benchmark, or an EOS tree walk. Two clean windows
with nothing else running give **23 and 25 MB/s**, and a single 512 MB file
reaches EOS at 13 MB/s *while contending*. Nothing needed fixing.

*There is a real contention effect, on disk not network.* The analysis and both
rsyncs share one `fuseblk` mount; with two transfers active the classifier ran
at ~23 % duty cycle (99 % CPU, ~2 min of CPU per 8 min of wall). Benchmark on a
quiet disk or treat the number as a floor.

*Stage 1.* `candidate_filter.py` classifies every trigger into the six-class
partition from `combined_hits` + the slim, no fitting and no geometry.
`reco.search.sift_events` was deliberately **not** reused — it ranks rather than
partitions, and it does 3D pairing per event, which is both the wrong basis
(`../RECONSTRUCTION_BASIS.md`) and far too slow at campaign scale. `io`, `noise`
and `segments` are reused as-is, plus one backward-compatible `measure=False`
on `find_segments` to skip the anchored fit that candidate-finding never uses.

*The finding.* C3 is resolved and it was not physics. Per-channel occupancy
shows all four arms have the same **1.1–2.5 % per-plane median**; what differs
is the tail, and arm D's tail is 26 channels above 20 % occupancy *per plane*
against 0–4 elsewhere, sitting at connector boundaries. The mask built from
that (`> 8 %` **and** `> 8x` the plane median) touches **82 channels on D, 19 on
A, 1 on B and none at all on C** — B and C's track rates come back identical to
the digit while D's falls by a third. A cut that only moves the arm that was
anomalous, and leaves the two good chambers untouched, is the shape an
instrumental fix should have.

*What it costs.* 51 triggers/s, **5.4 core-hours per 10⁶** → ~140 core-hours for
the campaign, ~7x cheaper than `PLAN.md` §5 budgeted. But
`INTER+INTRA+IMPLIED` is **4.0 %**, not the ~1 % the plan hoped, which puts
stage 2 at ≈ 2 000 core-hours. Affordable, and still one file tag of one
sub-run — it needs re-measuring across sub-runs before anything is launched.

**2026-09-07 (night, Ubuntu)** — Disk reclaimed, package scaffolded, stage 0 done.

*Disk.* `sps_run53_det4_check` was 220 GB of the 477 GB volume. Checked it
against EOS file by file: **all 294 `.fdf` and all 175 `combined_hits` ROOT
files matched an EOS file of identical size**, so those 140 GB went
immediately — the volume is at 73 % instead of 99 %. The other 92 GB is
`dec_*`/`hits_*` ROOT that was decoded *locally* and is on EOS nowhere; it is
being uploaded to `/eos/user/d/dneff/x17/sps_run53_det4_check/` before being
deleted here. The 3.5 GB of `.npz`/`.json`/`.csv`/figures — the actual SPS
analysis products — were mirrored to EOS **and kept on disk**.
`sps_beam_test_26/RESTORE_LOCAL_STAGING.md` is the record;
`DELETED_MANIFEST_2026-09-07.json` beside the data lists all 1 824 files.

⚠ **EOS's own `decoded_root` is not a substitute for ours** — it holds FEU `_01`
(the P2 detector) only, decoded with different settings. Our det4 side is FEU
`_03`, which EOS never decoded.

*What was already local.* `beam_july` holds 111 GB of n_TOF processed ROOT for
the **July commissioning** runs (224466, 224476–79, 224489, 224503) plus a 50 GB
`hitcache`, and 15 GB of MM `run_55`. **None of it is in the September sample**,
which starts at n_TOF run 224572. Two things worth knowing about it: the local
`run224*.root` are *a different processing vintage* from EOS — consistently
81–85 % of the EOS size, and both open cleanly, so this is not truncation — and
nothing else from the production period was ever staged here.

*Scaffolding and stage 0.* `paths.py` and `figstyle.py` (N4), then
`freeze_sample.py`, which produced the frozen sample above. The sample is
**36 runs / 293 sub-runs / 238 beam hours**, and the cut ledger says the whole
cost is the production-trigger cut — the four condition cuts (target, gas,
status, FEUs) turned out to remove nothing at all.

*Measured on the way.* Link holds ~16 MB/s **in both directions at once** (a
515 MB `combined_hits` pull ran at 15.8 MB/s while the 92 GB upload was
saturating the other direction). Upload to EOS is slower per byte than
download, ~5 MB/s across many mid-size files.

**2026-09-07 (evening, Ubuntu)** — Landed on the laptop and ran N0–N3. Kerberos,
lxplus and the deploy all work; the board and the plan note are live. All five
CERN assumptions checked against the files rather than the documents:

- **The re-slim is complete** — 326 of 329 beam+phys sub-runs from run_79 on,
  99.1 %. The 46 runs in that range with no slim at all turned out to be the
  beam-off cosmic and pulser runs, which have no n_TOF join by design; the
  alternating gap in the run numbers is that alternation, not a failure. Three
  real gaps remain and all are negligible (run_120, run_137's 44 events, and
  run_145's stub 4th sub-run).
- **run_145's reconstruction covers 2 of its 4 sub-runs**, not the whole run —
  `stat090_0002` (9 tags, 7.0 GB) was never reconstructed. All four arms'
  bundles verified kernel-ordered on the way in.
- **run_79 has no surviving reconstruction anywhere**, so the "rebuild or
  patch?" question answers itself: there is nothing to patch, and the 7-30
  product was arm A only on the retired inverted kernel regardless.
- **The link is 56–115× faster than the August measurement** and the local disk
  is now the constraint instead (C1).
- **AFS work is nearly full and EOS user has 2 TB** (C2).

One genuinely new open question came out of staging the products: **arm D seeds
twice as many triggers as A, B and C** (C3), which stage 1's class census
cannot be believed without explaining.

Nothing was run beyond verification and staging — the package scaffolding (N4)
is still the next code to write.

**2026-09-07 (later)** — Moving to the Ubuntu laptop; lxplus is unreachable from
the Windows box and repairing it there is not worth the time when the machine
that already works is a reboot away. Everything written today is committed and
pushed: the plan, this status file, the package README, the note generator, and
on the site side the restructured board plus the plan note, both **built but not
deployed** — `./scripts/deploy-eos.sh` on Ubuntu is the one command that has
been waiting. Next steps written up as N0–N5 above.

**2026-09-07** — Plan written. Took stock of the git history and the inherited
state: the DREAM ↔ n_TOF match is effectively done (99.45 % of beam pulses
since run_79), n_TOF reprocessing is complete (445/445 runs), and the
waveform-first reconstruction is proven on run_79 and run_145 with the
corrected sharing kernel and the corrected geometry. Established that the X17
opening angle of ≥ 109° cannot fit inside one chamber's ±40° acceptance, which
makes the two-chamber topology the signal region and puts chambers B and D —
the two that are not calibrated — on the critical path. Chose run_145 as the
local development run.
