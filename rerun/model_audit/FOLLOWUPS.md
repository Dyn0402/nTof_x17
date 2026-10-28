# Model audit — Dylan's follow-up questions (2026-10-27 evening)

Context: the eight workstreams on `FINDINGS.md` are finished (outcomes table at the top of `FINDINGS.md`, reports in
`ws_*/REPORT.md`, slide note https://dylan-neff.web.cern.ch/notes/mx17-model-audit.html, generator
`make_audit_deck.py`). Dylan read the note and sent the 13 points below. For each one: his words (condensed), the first
answer from what the workstreams already measured, and the follow-up task. The follow-up workstreams are listed at the
end (FA–FE) and write to `followups/<id>/`.

Standing rules for every follow-up: no new data taking (CLAUDE.md); geometry from waveforms via `wft/`, never from
combined_hits times; anything calibrated is per detector **and** per run condition; one reconstruction, not options
(switches only as `wft/diag.py` study overrides); CERN write-ups are figure-first slide notes (`slide-note` skill), not
prose; lxplus only via `ssh lxplus`, condor outputs to EOS.

## Outcomes of FA–FE (2026-10-27/28) — reports in `followups/*/REPORT.md`
| item | outcome |
|---|---|
| 1 | v(t) from the model-free column duration on every trigger (no M3; works at n_TOF); block size from the Allan minimum (8.5–49 min). v(x,y) separated from the bow via dln v/dln T (bow alone −0.46): det2/3/7 bow-only, det4/det6 gas still flushing (v rms 6–9 %). Gate: det6 cell spread 8.5→2.6 %, s68 −0.6°; v(t) slightly worsens X on stable det2/det3. In the reco: v(x,y) map slot (replaces the bow scale term); v(t) slot NOT applied — event record lacks the DREAM timestamp. (FA) |
| 2 | det3 X/Y +1.7 % is a per-view angle scale of the fit, not v (per-view column duration differs −0.6 %); one σ0 halves it (+0.8 %); follows the chamber across mountings; det4's −3.5 % has a different cause. (FA) |
| 3 | rigid tilt constant within a mounting, changes on remount; bench slot sets Y tilt (bottom −0.5…−0.8°, top −0.2…−0.36°). (FA) |
| 4 | **bow real**: mesh centre 1.5–2.3 mm lower than the rim (further from the cathode) on all five chambers, three independent routes, sign from external facts, static per chamber (15/19 runs within 2σ; shape transfers across runs with 94 % of own-map gain). Absolute depth 0.5–1.3× (M3 planes could bow too). Correction = `bow` bundle block: angle s68 −0.07…−0.56°. Notes mx17-chamber-bow; paper `papers/paper1/sections/08b_bow.tex`. Never to n_TOF. (FB, FE) |
| 5 | **one σ0 = 0.30 ± 0.02 mm for both views**: the X/Y difference was σ0_X trading with Dp_X; physics: both views read the same pads, the 0.3 mm is mostly resistive-sheet spread inside the first 60 ns sample; the only view-dependent part is bounded and would make X wider. (FC) |
| 6 | trade survives at −1.2 ± 0.5 % common compression (det4 Y, det7 Y ~−4 %); bright-track effect mostly removed by dropping the per-event σ0 (−4 % left on det6/det7, the saturating chambers). (FC) |
| 7 | t0 pinned at the latency in bench bundles; x–y gate on column time; χ²/dof +3…+37 % under the pin (model error); never at n_TOF (enforced). (FE) |
| 8 | map gallery note mx17-chamber-maps (USED / PARTLY / NOT USED badges). (FB) |
| 10 | [10-28: gain-map rank withdrawn (GA); M3 term 0.05° (GB)] 60 ns jump scales with the single-electron footprint (onset L = 1.55 σ_eff), clustering doubles it; 20 ns flat to 45°. Realism ladder reaches 0.41–0.76° vs measured 1.06–1.76°; + 2D tilt 0.43–1.09°; 0.72–1.38° unexplained and χ²-blind (points to fine-scale geometry, not the waveform model). Note mx17-angle-floor. (FD) |
| 11 | pairing per class 84–99 %; on det6/det7 the limit is saturation, not tolerance. (FE) |
| 12 | `fe` arm recommended as the single reconstruction (s68 better on every view, detection +0.15…+3.3 pt except det4 −0.12); 20 ns not adopted on the bench; promotion prepared (`promote_fe.sh`), **not applied**. (FE) |
| 13 | band strips censored + re-seed: det3 band sample 145 → 221/252 found, controls 400/400. Hotspot census deferred. (FE) |

Pending: `fefa` det4/det6/det7 + `fe_nobow` det7 gate rows (lxplus auth expired 10-27 night); v(t) needs the DREAM
timestamp in the event record; measured strip gain map (slot, all 1.0); promotion decision (Dylan); per-chamber
`CHI2DOF_BAD`; item 9 (gas fit) and the hotspot census after reprocessing.

---

## 1. v changes over the run (gas flush) and over the surface → make it standard procedure
**Dylan:** expected, we start taking data after a short flush; measure and correct the gas change over time as standard;
over the surface too (hopefully smaller).
**Known (ws_v_w0):** time: det4 −9…−10 % over its day run, det6 +12 %, det7 +4 % over the shared night; det2/3 ~2 %.
Two independent routes (ladder angle route and column duration 1/T) move together → real gas. Surface: quadrant v
differs (det3 X 35.6 → 37.5 µm/ns, i.e. ±3 %; up to ±8 % on det6). Today the bundle carries one v per view per run.
**Task (FA):** a standard v(t) and v(x, y) calibration in the bundle: v(t) from the column duration (model-free, per time
block, needs no M3, so it also works at n_TOF) cross-checked by the ladder; v(x, y) from the ladder in cells. Decide
the time-block size from the data (statistics vs drift rate). Separate v(x, y) from the bow (item 4): the bow changes
the local gap and the local tilt, which also changes the column duration and the ladder; the two must be fitted
together or one of them pinned. Wire both into the bundle and the angle conversion; gate.

## 2. det3 X–Y v difference (+2.1 ± 0.4 %)
**Known:** same events, stable in time, varies by quadrant (−0.2 … +3.7 %), sign differs between chambers (det2 0.0,
det4 −3.0, det6 +1.9, det7 +0.6 %) → not strip-layer geometry. Candidate: per-view angle-scale error of the size of the
width–tilt trade (Y carries more modelled width: RC spreading is on Y only).
**Task (FA):** chase it. Does it survive the position map (bow), a common σ0 (item 5) and the pinned t0? Is it a v
difference at all, or a per-view tan scale (compare the column duration per view, which has no angle in it)? Per
quadrant, per run, per HV row of det3.

## 3. Tilt w0 relative to M3, per run
**Dylan:** certainly exists, slightly different per data set → characterise per run.
**Known:** w0 = chamber tilt against M3 (r = 0.81 against the position-route tilt). det3 in the bottom slot (6-25) has
a different tilt from the top slot. `set_w0kw.py` already measures w0 per finalize row; it is per run in effect but was
never compared across runs of the same chamber.
**Task (FA, with FB):** a table of tilt (angle route and position route) for every bench run of every chamber; check it
is constant within a mounting and changes when the chamber was remounted (dates from the run registry). Split the
tilt (rigid, per mounting) from the bow (item 4, per chamber).

## 4. The chamber bow — major claim, needs cross-checks, slides, consistency, correction, paper section
**Dylan:** plausible (thin boards under mesh tension) but a major claim: cross-check thoroughly; slides demonstrating the
effect and the measurements; it should be static per detector — is it consistent between runs?; correct for it in
reconstruction (started); a section in the detector paper.
**Known (ws_widthtilt):** w0 ramps ≈ −0.6 → +0.6 µm/ns across every chamber and both views; the mesh-height curvature
from M3 position residuals vs tan has the same sign on 10/10 views (120–250 /km, 1–3 mm dish), predicted 40–200 /km from
the w0 ramp. Not resolved: which way the centre sits (frame z sign), rim bins flip sign on det3 X / det2, an M3
plane-scale artefact excluded only by plausibility (needs a 5–10 % scale error). The fix (`angle_map`) is in `wft/` and
cross-validated (s68 −1 … −27 %) but in no production bundle. The SPS det4 result (ws_spsv) also allows a local gap of
~26.5 mm at the beam spot.
**Task (FB):** (a) cross-checks independent of the angle fit: column duration T(x, y) (a dish shortens the gap at the
centre or at the edges → T map at fixed v; must be separated from v(x, y), see item 1), head-on cluster width vs position,
position residual vs tan per cell; M3 alignment / plane-scale systematics; the drift-gap study of 7-30 (det3 "dished
~2 mm short", det2 flat) — compare. (b) Consistency: every run of each chamber, both slots, different HV rows — same
shape and size? (c) Sign of the dish from an external fact (M3 z up, gravity, mounting), never self-consistency. (d) A
physical model (edge-supported plate under mesh tension / gravity) only as a plausibility check. (e) Slide note with
the evidence. (f) Draft paper section (`papers/`). (g) Correction in reconstruction: per-chamber static bow map + per-run
rigid tilt in the bundle; gate.

## 5. σ0: does it make physical sense for X and Y to differ?
**Dylan:** glad SPS gives a consistent value; does X ≠ Y make physical sense? Convince me. (Gas not transferring is fine.)
**First answer (not yet convincing, by design):** the avalanche and the gas diffusion are isotropic, so any X ≠ Y in the
prompt footprint must come from the readout, and there are candidates in both directions:
- the two readout strip layers sit at different depths below the resistive layer; the induced footprint widens with
  distance, so the deeper layer should see a wider σ0;
- the resistive strips run along y, so Y (the view along them) has RC spreading. Any part of that spreading faster than
  the first 60 ns sample is indistinguishable from a wider σ0 in Y — the ZS fits could not separate σ0_Y from D_rc,Y;
- X is quantised by the resistive-strip pitch (charge landing on a resistive strip is shared over its width).
The measurement itself is weaker than the numbers suggest: σ0_X 0.29 is an upper bound (X width grows with drift time at
2× Magboltz, an unmodelled X effect absorbs part of it), so "Y 0.32, X ≤ 0.29" is not yet evidence of a real
difference. **Task (FC):** test one common σ0 for both views with the Y RC fast component free (and the X anomaly
modelled or bounded): if the common value fits as well, use one σ0; if not, identify which mechanism (layer depth from
the board stack-up — check the drawings / Geant4 geometry `MX17_Geant`; RC fast component; pitch quantisation) and
show its size is right. Report as a slide.

## 6. Is the width–tilt trade still there after everything else?
**Known:** on det4/det6 σ0 0.24 → 0.54 mm lowers the slope 1–3 % on steep tracks; no (σ0, Dp) makes the M3 angle the χ²
minimum; after the bow map a symmetric ~2 % compression (det4 Y 7 %) remains, and bright tracks read 5–10 % shallow on
det6/det7. Never re-tested with all corrections at once.
**Task (FC):** re-run the trade with the pinned t0, the position map, the fixed fleet σ0 (no per-event choice) and the
20 ns grid together, on all five chambers (the earlier data trade rests on det4/det6 only). What is left of the
compression and of the brightness dependence?

## 7. Is the t0 treatment good now?
**Answer: the understanding is complete; the implementation is half done.** t0 is one bench latency (C ≈ 300–340 ns,
t0 = C − 10 ns·ftst), pinning improves position 4–31 % and leaves angles unchanged, continuous t0 is on. Still to do:
pin t0 in the bench bundles (`set_w0kw.py --latency-cache --pin-t0`; no bundle uses it yet), replace the x–y t0 gate by
the column-time comparison (t0 + u50; not implemented), and decide on χ²/dof rising 4–20 % (model error the free t0 used
to absorb). Never pin at n_TOF (trigger ≠ particle). **Task (FE).**

## 8. Show the position/tilt maps, gain maps, everything; what is actually used in reconstruction?
**Answer — what the reconstruction uses today** (the calibration bundle, `wft/calib.py`):
| bundle content | used in reco | per |
|---|---|---|
| v (ladder) | yes, one value per view | chamber × run condition |
| electronics templates, RC kernel (D_rc,Y; c1, c2/c1 legacy) | yes | chamber (template per plane) |
| σ0, Dp (physical RC model) | yes; σ0 still has the per-event wide/narrow choice | chamber |
| w0 / kw (one tilt and one angle scale per view) | yes | chamber × run |
| dead list (censored in the fit, bridged by the seeder) | yes | chamber (bench), none at n_TOF |
| x–y pairing window, head-on charge tolerance, tan_floor | yes | chamber × run |
| t0 information (t0_abs, latency) | free t0 in practice; pin not in any bundle | — |
| per-channel gain map (`gainmap.npz`) | loaded and applied, but all 1.0 (unmeasured) in all five promoted bench bundles — checked 10-27 | — |
| angle map (bow, position-dependent w0 and scale) | code exists, **in no production bundle** | — |
| v(x, y), v(t) | **not used** | — |
**Task (FB):** a slide note of every map per chamber: tilt/bow, angle scale, v(x, y) and v(t), gain (from charge, the
F15 connector gains and the efficiency maps), dead strips, noise bands — with a line saying which ones the
reconstruction uses.

## 9. SPS gas composition fit (H₂O + O₂) with Magboltz — for the paper, after reprocessing
**Known:** v(243 V/cm) = 14.1 +0.6 −1.1 µm/ns implies ~1.4–1.6 % water if O₂-free; the column loses ~50 % of its charge
from cathode to mesh (attachment); `beam-gas-attachment` (10-10) settled beam 1.5 % H₂O + ~160 ppm O₂ for n_TOF.
No bearing on the reconstruction algorithm (gas-dependent quantities are measured per run, not transferred). **Deferred
until after reprocessing; not spawned now.** When done: joint fit of v(E) and the attachment profile (column charge
gradient) vs Magboltz over (H₂O, O₂).

## 10. Physics floor: why the 60 ns curve jumps at high angle, and why we don't reach the floor
**First answer to the jump:** a 60 ns depth bin on det3 is v·Δt ≈ 36 µm/ns × 60 ns ≈ 2.2 mm of drift; across the strips
it moves 2.2 mm × tan θ: 0.4 mm at 10°, 0.8 mm (one strip pitch) at ~20°, 1.0 mm at 25°, 1.8 mm at 40°. The model
spreads each bin's charge uniformly along that segment, but real ionisation is clustered (a few clusters per bin), so
once the segment is longer than a strip the per-bin centroid is wrong by up to half a segment and the strips see
lumps the model cannot represent. That is why the curve is flat to 20° and rises steeply after ~25° (onset at one
pitch), why it vanishes without cluster-size fluctuations, and why it is weaker on slow det6. **To confirm (FD)** with
the toy: onset angle vs v·Δt/pitch across chambers and bin widths.
**Why we don't reach the floor:** the toy fits with its own exact mean model; the data fit has model error (toy χ²/dof
4–40 with production weighting against 60–700 on data). Candidates the toy idealises: the RC/sharing kernel shape, the
one-sided X copy, saturation, common-mode residuals and noise bands, gain non-uniformity across a cluster, dead strips,
δ-ray tails, the free t0, the per-event σ0 choice, the bow/tilt (now separable), the M3 reference (≤ 0.15°).
**Task (FD):** a toy ladder that adds one realism at a time (generate with the realistic effect, fit with the production
model) until it reproduces the measured resolution and χ²/dof per chamber; rank the terms. Slide note.

## 11. Charge balance — what does the result mean?
**Answer:** X and Y strips read the same avalanche, so for one track qx/qy should be the same number every time; the
width of ln(qx/qy) is how precisely the two views agree. That width sets how well we can tell which X cluster belongs
to which Y cluster when two particles cross at once. Clean, unsaturated tracks: 5 % (det3) … 7–10 % (det4/det6/det7)
→ two tracks whose charges differ by more than ~15–25 % can be paired by charge; with Landau-distributed charges that
is most pairs (pairing by charge 96–100 % in the det3 study). The wide numbers in the first table (det7 0.32) were not
the detector: they came from wrong x–y pairs in the sample, saturated tracks (clipped charge in one view) and windows
containing dead strips. It is **good news for pairing**, with two conditions the reconstruction must honour: a pairing
tolerance per event class (saturated / censored / clean) instead of one per chamber, and handling saturation (69 % of
det6/det7 events saturate). **Task (FE):** implement per-class pairing tolerances; measure pairing efficiency per class.

## 12. Keep pushing the single unified reconstruction
**Task (FE):** the decided items, as bundle content and code constants only, gated together: fleet σ0 and drop the
per-event choice (with FC's verdict on X vs Y), t0 pin in bench bundles + column-time x–y gate, per-class pairing
tolerances, angle map (from FB) and v(t)/v(x, y) (from FA) as bundle slots, 20 ns grid (gate it), dead lists for n_TOF
and the corrected golden masks (ws_dead_charge). Then re-finalise and promote every bundle, re-measure n_TOF pair
efficiency. FE owns `wft/` core edits; FA–FD deliver bundle content or study overrides through `wft/diag.py`.

## 13. Noise bands as discharges at fixed places → hotspot maps (after reprocessing); algorithm awareness now
**Dylan:** characterise for all detectors to pick out hotspots — can wait until after reprocessing. For now the
algorithms must be aware of them and reconstruct with them ignored.
**Known:** the band veto (`DROP_FLAT_HITS`) is now always on in the single reconstruction, but it acts only in the
bench driver; det3 bands are whole-window baseline offsets on a contiguous patch (`rerun/satstudy/`, `rerun/bands/`).
**Task (FE, now):** make band awareness part of the one reconstruction for every dataset (bench, SPS, n_TOF): detect the
band strips per event, exclude them from the fit (censor, like dead strips) rather than dropping the event; check on the
det3 band sample that tracks under a band are recovered. **Hotspot census: deferred until after reprocessing.**

---

## Follow-up workstreams (spawned 2026-10-27 evening)
| id | items | owns | output |
|---|---|---|---|
| FA | 1, 2, 3 | v(t), v(x, y), X–Y v, per-run tilt table; bundle slots proposed to FE | `followups/fa_gas_v_tilt/` |
| FB | 4, 8 | bow cross-checks, consistency, sign, correction map, slides, paper-section draft; map gallery | `followups/fb_bow_maps/` |
| FC | 5, 6 | σ0 X vs Y physics; width–tilt with all corrections | `followups/fc_sigma0_trade/` |
| FD | 10 | 60 ns jump confirmed; realistic toy ladder to the measured resolution | `followups/fd_floor_realism/` |
| FE | 7, 11, 12, 13 | single reconstruction: implement + gate decided items, band awareness | `followups/fe_unify2/`, `wft/` |
Deferred (after reprocessing): 9 (gas fit), 13 hotspot census.

---

## Round 3 — Dylan's questions (2026-10-28)
1. Gain map: is it unimportant? it has the largest variation → **GA** (`followups/ga_gainmap/`)
2. Is M3 (+ multiple scattering) limiting the angle resolution? what does SPS show? → **GB** (`followups/gb_m3_angle/`)
3. Gas change vs time for every dataset; consequence for reco; can it wait until after reprocessing? → **GC** (`followups/gc_gas_time/`)
4. Bow: Dylan has another session cross-check it; results look reasonable.
5. **Decided:** one σ0 for X and Y (FC) is the physical choice.
6. lxplus back → pending `fefa` / `fe_nobow` gate rows → **GF** (fe_unify2)
7. Spark detection re-checked after the FE changes; make it optimal → **GD** (`followups/gd_sparks/`)
8. Has the fixed χ²/dof cut been useful? hard to trust on this model; keep only if useful → **GE** (`followups/ge_chi2cut/`)
9. **Decided:** before the full reprocessing, every calibration is redone from scratch — no shortcuts, no reuse of
   intermediate products from this audit.

### Round 3 outcomes (2026-10-28) — reports in `followups/g*/REPORT.md`, `fe_unify2/REPORT.md`
| item | outcome |
|---|---|
| 1 | **GA: gain map not needed** (any chamber). Large-scale variation (±20–40 %, det4 X ×34) only rescales the free charge profile; strip-to-strip gain 0.02–0.04 (noise-limited); cross-validated maps change s68 by 0 ± 0.03°. FD's "gain 0.55° det4" withdrawn. Gain slot stays 1.0. By-product: 14 reproducible weak/hot channels → masks (below). |
| 2 | **GB: M3 + MS = 0.05° (≤ 0.11–0.14° measured)** — not limiting; the det6/det7 shared 0.5–0.9° term is the bow. SPS det4 head-on (beam reference 0.03°) agrees with the bench. FD budget M3 row → 0.05°. |
| 3 | **GC: post-correctable** (v enters only seeding, u_window and tan). Before the pass: DREAM timestamp + acquisition tag in the event record (**done**: `timestamp`, `acq_tag` columns); own v for every row with ≥ 500 clean triggers (**done**: `make_registry.py` rule; 117 own-v rows, 37 inherit the nearest-in-time calibrated row of their group). After: per-time-block kw from the M3 ladder (bench), column end (n_TOF A/C); n_TOF B/D within-run v not measurable (stated). |
| 6 | **GF: all rows in; promote `fe`, not `fefa`** (v(x,y) map gains ≤ 0.05° on det6, in-sample, loses on det7 X; revisit with v(t)). |
| 7 | GD running. |
| 8 | **GE: fixed χ²/dof 300 replaced by 5 × the table median per view** (**done**: `wft.reco.CHI2DOF_REL`, `quality_flags`, sidecar `quality_rule`; old tables re-flagged on load; n_TOF stage 3 per sub-run × arm). Validated on the fe gate tables: det2/det3 remove 9–16 % with 81–91 % of bad views; det6 gives back 60–70 pt of efficiency; blind on det4 / det7 X (missing-charge faults, hardware). |

**To double-check before the reprocessing (Dylan 10-28): the GA bad channels** now in
`followups/fe_unify2/masks/bench_det{3,4}.json` (`ga_added`): det4 X weak 94, 125, 196, 376, 400, 435, 439, 455, 472, 498
→ dead; det4 X 274 hot; det3 X 62 hot; det3 Y 41 weak, 510 dead-like → dead. They come from the in-fit gain map
(|ln g| > 0.5 in both event halves), not from the census/flank test the other masks passed. Check each with the flank test
on the hits and on a waveform display; det4 X 274 sits beside the chamber-side dead run 271–273 (a short partner?).
`set_w0kw.py --dead-json` now also installs the mask's `hot` list.
