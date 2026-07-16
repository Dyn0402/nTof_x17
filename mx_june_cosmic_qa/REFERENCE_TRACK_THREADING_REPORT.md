# Reference-track threading in the MX17 micro-TPC 3-D displays — a systematics report

**Detector A (mx17_3), Saturday long run `sat_det3` (`long_run_resist_490V_drift_1000V`, 490 V amplification / 1000 V drift), M3 recipe χ²<1 & NClus≥4, spark veto 50.**
Author: analysis session 2026-07-15. Intended for independent review.

---

## 0. TL;DR and the honest headline

The question that triggered this report: *in the 3-D event displays, the M3 reference track agrees with the charge cloud at the mesh but appears to diverge as the track ascends into the drift gap — is that a misalignment, and is it consistent with our claimed ~0.5 mm resolution?*

Findings, in one paragraph:

1. **It is not a misalignment.** The chamber-to-reference match at the mesh is centred on zero to better than 10 µm in both axes, with a per-axis core σ of 0.67/0.76 mm — consistent with the quoted resolution.
2. **The divergence with depth is real, systematic, and per-event.** The raw time–position ladder is **~4° too steep**; the reference threads the cloud at the mesh and fans away from it going up. Root cause: resistive-strip **charge sharing** biases the per-strip drift-time ladder.
3. **Therefore the reviewer's/user's skepticism is correct:** on an **event-by-event** basis we do **not** cleanly thread the reference track through the *full-depth* charge cloud with the production (raw) reconstruction. We thread at the mesh; we diverge with depth. The pretty "tight" display events were selected on the *mesh* residual, which does **not** guarantee full-depth threading.
4. **The hit-level "unsharing" correction fixes this — but statistically, not per-event.** Over a population it removes ~80–90% of the depth-dependent bias slope. On a single event the deconvolution *amplifies* per-strip noise, and the fitted angle barely moves. So there is currently **no per-event reconstruction that threads a single muon cleanly through the whole 30 mm gap**; the win from unsharing is in ensemble angular bias/resolution.

This report documents the metrics, the numbers, the figures, the reconstruction chain, the open questions, and how to reproduce everything.

---

## 1. Geometry and how the reference line is drawn

**Micro-TPC coordinates (detector-local, "raw strip frame"):**
- `x` = strip position from the X strips [mm] (FEU 7)
- `y` = strip position from the Y strips [mm] (FEU 8)
- `z` = drift **depth** from the mesh [mm] = `(t − t0) · v_drift / 1000`, `v_drift = 34 µm/ns`, `t0` = earliest hit time in the event.

A single muon makes one cluster on X and one on Y. Each strip contributes a point `(position, drift-time→depth)`. In the 3-D display the complementary coordinate (e.g. `y` for an X-strip hit) is predicted from a straight-line fit of the other plane.

**The reference line (M3 telescope):**
```
x_ref(z) = ref_mesh_x_mm + z · tan_x_raw
y_ref(z) = ref_mesh_y_mm + z · tan_y_raw
```
- `ref_mesh_x/y_mm` = the M3 track position at the mesh (z=0), obtained by inverse-applying the alignment transform to the reference position at the DUT plane (`cosmic_micro_tpc_analysis.py:1364-1377`).
- `tan_x_raw, tan_y_raw` = the M3 angle tangents **rotated into the raw strip frame** by the alignment rotation θ≈89.45°:
  `tan_x_raw = cosθ·tan_x + sinθ·tan_y`, `tan_y_raw = −sinθ·tan_x + cosθ·tan_y`
  (`cm._rotate_ref_tangents`, added this session). **NB:** the original `plot_event_display_3d` drew the tangents un-rotated — a genuine bug that made the line fan away for any large-θ alignment (det3, θ≈89.45°). Fixed; verified the drawn slope now matches the measured ladder direction.

**Anchor convention matters.** The line is pinned to the M3 position at the mesh (z=0), and the detector position used for the alignment is the earliest-hit strip. So by construction the comparison is tightest at z=0 and any slope disagreement grows with z.

---

## 2. Is it a misalignment? — No.

Population: all events with both planes, matched to a single clean M3 ray, finite radial residual (no upper radial cut here). N = 6,978. (§3's depth table adds a radial < 10 mm cut → 6,692 events / 106,013 hits.)

The residual is `ref − det` where `det` = **earliest-hit strip position** (`mesh_position_mm`; `StripFitResult.mesh_position_mm`, `cosmic_micro_tpc_analysis.py:326, 660-693`), and `ref` = M3 at the DUT plane.

| quantity | X | Y |
|---|---|---|
| core mean (systematic offset) | **−0.005 mm** | **−0.006 mm** |
| median | +0.004 mm | +0.001 mm |
| core σ (resolution) | **0.67 mm** | **0.76 mm** |

- Both axes centred on **zero to < 10 µm** → no global misalignment.
- Core σ 0.67/0.76 mm matches the report's quoted 0.61/0.73 mm (and *includes* the M3 pointing error, so the chamber alone is tighter).
- **Radial (2-D) residual** is a positive Rayleigh quantity: for σ≈0.72 mm/axis with perfect alignment the *median* radial distance is 0.85 mm — observed median **0.81 mm** (p10/50/90 = 0.29/0.81/2.11 mm). So a "typical" event sits ~0.8 mm off in 2-D distance *even with perfect alignment and the claimed resolution*. This is pure geometry (you cannot get a visually-zero 2-D offset from a real 0.7 mm/axis detector).

Resolution degrades mildly with inclination: core σ_X = 0.64 mm (0–5°) → 0.94 mm (15–25°).

**Figure:** `engineer_package/event_displays_3d/_depth_residual_diagnostic.png` panel-free version; residual histograms are in the session's scratch `resid_deepdive.py`.
**Conclusion:** alignment is sound; the mesh-anchored position metric is unbiased and near-optimal (see §5).

---

## 3. The depth-resolved metric (the key diagnostic)

For every strip hit in the matched population, compute the signed distance from the reference **line** at that hit's depth, in the measured axis:
- X-strip hit: `dx(z) = x_meas − (ref_mesh_x + z·tan_x_raw)`
- Y-strip hit: `dy(z) = y_meas − (ref_mesh_y + z·tan_y_raw)`

Sign convention: multiply by `sign(tan)` so a **steeper** raw ladder reads positive. Bin by depth `z`.

Full population (all strips, N=106,013 hits):

| depth z [mm] | N | signed bias [mm] | core σ [mm] |
|---|---|---|---|
| 0.0–2.5 | 24226 | −0.155 | 0.932 |
| 2.5–5.0 | 12342 | −0.152 | 1.240 |
| 5.0–7.5 | 8836 | +0.129 | 1.323 |
| 7.5–10.0 | 7709 | +0.388 | 1.328 |
| 10.0–12.5 | 7539 | +0.536 | 1.419 |
| 12.5–15.0 | 7610 | +0.711 | 1.522 |
| 15.0–17.5 | 8096 | +0.881 | 1.585 |
| 17.5–20.0 | 8852 | +1.062 | 1.669 |
| 20.0–22.5 | 9432 | +1.232 | 1.788 |
| 22.5–25.0 | 6700 | +1.275 | 1.904 |
| 25.0–27.5 | 2520 | +1.189 | 2.155 |
| 27.5–30.0 | 684 | +0.620 | 2.976 |

Two distinct effects fall straight out:

**(A) A depth-dependent BIAS** (column 3). Near-zero at the mesh, growing to ~+1.3 mm by mid/upper drift, then turning over near the cathode. Linear fit +48 µm/mm; a quadratic captures the concavity (−0.0031·z²). This *is* the "divergence with depth" the displays show. Per event, the raw ladder over-steepening is `|tan_raw| − |tan_ref|` = **+0.080 (median)** → ≈ **+4.4°** at the 9.2° median reference angle (≈ 50% too steep in slope, equivalently drift-velocity read ~30% low).

**(B) Resolution vs depth** (column 4). Core σ grows **0.93 → 1.9 → 3.0 mm** from mesh to cathode. Driven by diffusion + attachment: binning by amplitude, low-amplitude (deep, attenuated) hits have σ = 2.17 mm vs 1.07 mm for the brightest quartile.

**Figure:** `engineer_package/event_displays_3d/_depth_residual_diagnostic.png`.
**Reproduce:** `../.venv/bin/python engineer_package/make_event_displays_3d.py --diagnostic`.

### 3.1 Directly addressing the event-by-event skepticism

The bias in (A) is *systematic and per-event* (it sign-follows each track; it is not random scatter that averages away). So on any single inclined muon, the reference line and the raw cloud share the mesh point and then separate — by roughly `(tan_raw − tan_ref)·z`, i.e. ~1–2 mm at the top of the gap for a 15–20° track. A representative decomposition of four real events (offset at the mesh vs. the extra angle-driven divergence at the top of the ~26 mm cloud):

| event | ref θ | raw ladder θ | offset @ mesh | extra divergence @ top |
|---|---|---|---|---|
| 9125 | 13.5° | 16.3° | 1.6 mm | 1.4 mm over 26 mm |
| 14783 | 14.8° | 17.2° | 1.3 mm | 1.6 mm over 26 mm |
| 7451 | 19.8° | 21.5° | 1.4 mm | 1.1 mm over 25 mm |
| 7534 | 21.3° | 24.3° | 1.2 mm | 1.8 mm over 27 mm |

**This confirms the skepticism:** we do not thread the full-depth cloud event-by-event. The "tight" display picks (radial match 0.1–0.3 mm) were selected on the **mesh** residual (`pick_by_residual` in `make_event_displays_3d.py`), which optimises z≈0 agreement and says nothing about the slope; those events still fan at the top by ~1 mm.

---

## 4. Root cause — resistive-strip charge sharing

The resistive strips deliberately share avalanche charge with neighbours (~45–52% first-neighbour, a few % second-neighbour, with a ~69 ns delay). The shared charge arrives on a neighbour with the *neighbour's* timing, which distorts the per-strip drift-time ladder and makes the fitted slope too steep. This is a measured **design property** of the detector, not a defect (see `18-det3A-charge-sharing-measured.png`, `MICROTPC_RUNBOOK.md`).

Measured det3 kernel (from vertical tracks, script 26; runbook): `c1/c2 = 0.449/0.052` (X, FEU 7), `0.516/0.151` (Y, FEU 8), neighbour delay +69 ns.

The **unsharing** correction is a hit-level waveform deconvolution: over each contiguous strip block it solves the mixed prompt/delayed banded system `(I + c1·E±1 + c2·E±2) x = w` per time sample, then re-extracts per-strip CFD time + amplitude and re-fits. Prototyped/validated in `26/27/28_*.py`; **also run in production-adjacent form by `31_microtpc_metrics.py`** (caches `microtpc_segments.csv`, the unshared segment slope). Lifted this session into a reusable module `engineer_package/sota_reco.py` (`unshare()`, `sota_hits()`, `sota_track_points()`).

**Granularity (verified from code):** the unsharing is **hit-by-hit** (per strip, per time sample) followed by a re-fit — it is *not* a post-fit angle scaling. On top of it, `28_angle_calibration.py` adds one **global additive per-plane** offset (`b_x≈+0.032, b_y≈+0.028`) to remove a residual ~2° diffusion floor. So: hit-level deconvolve → re-fit slope → single scalar per-plane calibration.

---

## 5. Does unsharing thread the track? Population: yes. Per-event: no.

### 5.1 Population proof

Depth-resolved signed bias, **core strips**, RAW vs UNSHARED, ~500 muons (`sota_reco` deconvolution):

| depth z [mm] | RAW bias [mm] | UNSHARED bias [mm] |
|---|---|---|
| 1.2 | +0.324 | +0.348 |
| 3.8 | +0.447 | +0.535 |
| 6.2 | +0.617 | +0.439 |
| 8.8 | +0.741 | +0.479 |
| 11.2 | +0.708 | +0.270 |
| 13.8 | +0.775 | +0.568 |
| 16.2 | +0.856 | +0.522 |
| 18.8 | +0.860 | +0.538 |
| 21.2 | +0.932 | +0.723 |
| 23.8 | +1.054 | +0.386 |
| 26.2 | +0.643 | +0.337 |

(Exact output of the figures/09 generator, `render_unsharing_depth_proof`, bins 0–27.5 mm, core strips, 500 events.)

Linear bias slope: **RAW +19 µm/mm → UNSHARED +2 µm/mm (89% reduction)** for this figure. The *raw* slope estimate is subset/binning-sensitive because the curve is concave (an independent scratch run over bins 0–30 mm gave +27 → +5 µm/mm, 79%; a 300-event run gave +32 → +10 µm/mm, 69%). Across all runs: **raw +19…32, unshared +2…10 µm/mm, i.e. the depth-dependent bias is reduced ~70–90% and the unshared trend is consistently near-flat.**

Two residual features the reviewer should note:
- The unshared curve keeps a **constant ~+0.4 mm offset** (all points > 0). This is almost certainly a **re-alignment artefact**: `ref_mesh` and the alignment were fit on the *raw* hits, so the unshared reconstruction inherits a small constant registration offset. Re-deriving the alignment on unshared hits should remove it — **not yet done** (open item).
- The residual *slope* (~+2 µm/mm) is the diffusion floor removed downstream by the per-plane additive calibration (§4).

**Figure:** `engineer_package/figures/09-det3A-unsharing-depth-bias.png` (+ .pdf).
**Reproduce:** `../.venv/bin/python engineer_package/make_event_displays_3d.py --compare --compare-n 500`.

### 5.2 Per-event reality (the crux)

On a single event the deconvolution **amplifies per-strip noise** — it is an ill-conditioned inverse. Concretely, event 25066:
- fitted ladder angle: **raw 21.6° → unshared 21.5°** (reference 20.3°). The angle barely moves.
- per-hit scatter *increases* (visible in `event_displays_3d/_unsharing_event_compare_25066.png`, right panel).

So the naive expectation — "unshared hits collapse onto the reference line" — is **false at the single-event level**. Unsharing shifts the *average* slope across many events (fixing the ensemble bias) while adding noise to each event. There is currently **no reconstruction that threads a single muon cleanly through the full depth**: raw has the cleanest per-hit cloud but the wrong slope; unshared has the right average slope but noisier hits.

This is the precise, honest answer to the skepticism. It is a genuine limitation, not a display artefact.

---

## 6. Why the mesh (earliest-strip) anchor is still the right position metric

The depth-bias curve crosses ≈zero near the mesh, so the earliest-hit anchor sits at the **low-bias pivot** and is largely immune to the charge-sharing slope bias. Quantitatively (raw-frame per-axis residual, matched population):

| position estimator | mean | σ |
|---|---|---|
| earliest-hit (mesh, production) | +0.002 mm | **0.773 mm** |
| line-fit intercept @ z=0 | −0.004 mm | 0.937 mm |

The earliest hit **beats** extrapolating the (biased) ladder to z=0 — extrapolating the biased slope injects noise. A cluster centroid taken at mid-depth would instead inherit ~+0.7 mm of the slope bias. So the production position choice is sound. (The best *position* estimator, script 36's early-charge centroid, reaches 0.47/0.54 mm and is a separate axis of improvement from unsharing — see §7.)

Subtlety for the reviewer: the near-mesh sign of the bias depends on strip selection — the all-strips full-population metric is slightly **negative** at z<5 (pivot at z≈5 mm, not 0), while the core-strip subset is slightly positive there. This ~0.15 mm near-mesh structure and the ~5 mm pivot hint at a small **t0 / z-registration** offset (the earliest *detected* hit is a few mm of drift above the true mesh crossing). It is well within the resolution but is a real, unexplained second-order feature.

---

## 7. Audit — which shipped analyses/figures use state-of-the-art reconstruction?

Reconstruction paths: **RAW** = production combined-hits (earliest-strip position, raw per-strip times, no unsharing). **SOTA** = unshared + calibrated (via `31_microtpc_metrics.py` → `34_hybrid_tracking.py` → `make_hybrid_figures.py`), and/or the early-charge-centroid position (script 36 / `sota_reco`).

| figure / number | reconstruction | would change if unshared? | evidence |
|---|---|---|---|
| Angular resolution 1.66/1.84/1.75°, bias ~0.1° (report, slides, figs 15/16) | **SOTA** (unshared+calibrated hybrid) | already corrected | `make_hybrid_figures.py:159,174-183,225-227,280,329-332`; seg CSV `31:192,213-218` |
| Fig 17 tracking dashboard (back-up, unreferenced) | shows raw `tan_prod` for contrast | — | `34_hybrid_tracking.py:321-324,457` |
| Position residual figs 10–14 (0.61/0.73 mm) | **RAW** earliest-strip | yes, improves | `03_alignment_and_tpc.py:191-192`; `cosmic_...py:405-410,660-663` |
| "Best estimator 0.47/0.54 mm" (report/slides text only) | SOTA combo (script 36) — **study-only, not in any figure** | corrected | `36_position_estimators.py:82-94`; `main.tex:363-366` |
| 3-D displays 07/08/hero/gallery | **RAW** combined-hits markers | markers raw (why they fan) | `make_event_displays_3d.py` |
| Fig 09 unsharing depth-bias | RAW vs SOTA (demonstration) | — | this session |
| Efficiency 20–26, HV-scan 30–33, sparks 40–44, time-res 50–52, charge-balance 60–61 | RAW hit-presence/position | **no** (no ladder-angle dependence) | scripts 08/09/12/10/39/40/42/38 |
| Drift-velocity 70–72 | RAW ladder slope | yes (already flagged pre-recipe) | scripts 21/23 |

**Bottom line of the audit:** the shipped **angle** numbers are already SOTA (unsharing routed through script 31). **No raw biased angle is presented as a headline.** The genuine gaps: (i) position-residual *figures* 10–14 are raw earliest-strip while the *text* quotes the study-only 0.47/0.54 mm — a figure/text inconsistency; (ii) the 3-D display markers are raw; (iii) a handful of velocity/geometry scripts read the raw ladder. Efficiency/spark/time-res/charge-balance are unaffected.

**Architectural risk:** unsharing is a **parallel side-channel** (`31` re-streams waveforms → CSV), *not* in the production combined-hits converter. Any new analysis that naively reads combined-hits gets the raw biased angle. "Baking it in as the default" ultimately means pushing the deconvolution into the hit converter (upstream, partly out-of-repo DAQ), or standardising all consumers on `sota_reco`/script-31 output.

---

## 8. Open questions / for the independent reviewer to challenge

1. **Is the depth divergence *fully* charge sharing, or is some of it a velocity / z-registration error?** The near-mesh negative dip and the pivot at z≈5 mm (§6) suggest a small t0/mesh registration offset that is not accounted for. Worth: fit the pivot depth per angle and see if it is constant (registration) or angle-dependent (something else).
2. **Re-alignment on unshared hits.** The residual constant offset in §5.1 is attributed to alignment being fit on raw hits. This should be tested by re-running the alignment on unshared hits and checking the offset vanishes. *Not done.*
3. **Is the deconvolution kernel/α optimal, and can per-event threading be recovered?** Unsharing adds per-hit noise. Is there a regularised deconvolution, a template/cluster fit, or a combined position+angle per-event estimator that threads a single muon cleanly? This is the real open problem — *we currently have no per-event reconstruction that threads the full depth.*
4. **Velocity used for depth (34 µm/ns).** The display and metric use v_geom = 34. The raw ladder implies v ~30% lower; the unshared ladder converges to ~34 (`45_slope_reference_vdrift_scan.py`). If the *display* used the raw-implied velocity, the depth axis would rescale and change the apparent divergence. Confirm the metric is insensitive to this choice (it should be, since both cloud and line use the same v, but state it explicitly).
5. **Selection bias in the display events.** The "tight" picks optimise the mesh residual. A fairer full-depth-threading selection would minimise the *perpendicular* distance to the reference line over all depths. Consider re-selecting on that and reporting how many events (if any) thread to < resolution at all depths.
6. **Radial-residual tail.** Radial mean (4.44 mm) ≫ median (0.81 mm): a heavy tail from mis-association / multi-cluster events survives the < 10 mm cut. Characterise the tail (is it real physics or reconstruction failures?).

---

## 9. Reproduction

All from `mx_june_cosmic_qa/` with `../.venv/bin/python`.

```bash
# 3-D displays (raw hits) + selection
python engineer_package/make_event_displays_3d.py --figures            # hero 07 + gallery 08
# depth-resolved residual diagnostic (population)
python engineer_package/make_event_displays_3d.py --diagnostic         # _depth_residual_diagnostic.png
# unsharing: population proof (fig 09) + single-event compare
python engineer_package/make_event_displays_3d.py --compare --compare-n 500
```

Key code:
- `engineer_package/sota_reco.py` — reusable unsharing (kernel, `unshare`, `sota_hits`, `sota_track_points`).
- `engineer_package/make_event_displays_3d.py` — displays, `_rotate_ref_tangents` usage, `render_depth_diagnostic`, `render_unsharing_depth_proof`, `render_unsharing_comparison`, `pick_by_residual`.
- `cosmic_bench_analysis/cosmic_micro_tpc_analysis.py` — `EventResult`, `attach_reference_positions` (line 1301), `_rotate_ref_tangents`, `plot_event_display_3d` (tangent-rotation fix).
- `26/27/28/31/36_*.py` — the unsharing/calibration/position-estimator pipeline.

Data: det3 Saturday run under `~/x17/cosmic_bench/`, decoded_root waveforms local for this run. Event cache: `event_results_veto50.pkl`; alignment: `alignment_tpc_veto50/alignment.json` (z=714 mm, θ=89.45°).

Kernel (det3): `c1/c2 = 0.449/0.052` (X, FEU 7), `0.516/0.151` (Y, FEU 8), delay +69 ns, α=0.5.

---

## 10. Figures referenced

| file | what |
|---|---|
| `engineer_package/figures/07-det3A-event-3d-display.png` | hero 3-D display (raw hits, tight-match event) |
| `engineer_package/figures/08-det3A-event-3d-gallery.png` | 2×2 gallery, ~6–20°, raw hits |
| `engineer_package/figures/09-det3A-unsharing-depth-bias.png` | **population proof: raw vs unshared depth bias** |
| `engineer_package/event_displays_3d/_depth_residual_diagnostic.png` | depth-resolved bias + resolution + per-event steepening |
| `engineer_package/event_displays_3d/_unsharing_event_compare_25066.png` | single-event raw vs unshared (per-hit noise, angle unchanged) |

---

## 11. Independent audit (2026-07-15, second session) — findings confirmed, one blind spot closed, pointing band added

An independent re-derivation from the cached EventResults (`event_results_veto50.pkl`, same alignment) reproduced the report's claims:

1. **Population and mesh anchor reproduce.** Matched population = 6,978 events exactly. Mesh residual (raw strip frame): core mean +0.001/−0.002 mm, core σ 0.83/0.75 mm — the raw-frame axes are swapped relative to §2's aligned-frame 0.67/0.76 by the θ≈89.45° rotation; same conclusion (no misalignment).
2. **Blind spot closed — the steepening is not a sign/handedness artefact.** The report's §3 metrics compare *absolute* tangents (`|tan_raw| − |tan_ref|`), which would silently mask a wrong-signed tangent rotation (the exact class of bug fixed this session). A **signed** per-event comparison of the production ladder slope (`slope_mm_per_ns·1000/v`) against the rotated reference tangent gives sign agreement **97.6 % (X) / 98.1 % (Y)** for |tan_ref| > 0.05, with median signed steepening **+0.069/+0.073** along the reference direction — matching §3's +0.080. The rotation/handedness is correct; the over-steepening is real.
3. **Transforms verified.** Forward-transforming `ref_mesh_x/y_mm` reproduces `ref_x/y_mm` to machine precision (< 1e-13 mm); `_rotate_ref_tangents` is the matching inverse rotation.
4. **NEW — the reference line now carries its own measured uncertainty.** All 3-D displays (canonical `plot_event_display_3d`/`_rotating`, package hero/gallery/compare) draw the M3 track as a line inside translucent **±1σ/±2σ elliptical pointing tubes**, using the M3 self-resolution study (`m3_self_resolution/results.json`, pointing at the DUT plane z=702: σ_X = 206 µm, σ_Y = 242 µm aligned frame; covariance rotated into the raw strip frame via `cm.ref_sigma_raw_frame`, effectively an axis swap for det3). The M3 *angular* error contributes < 15 µm over the 30 mm gap (4-plane fit, σ_slope ≈ 0.35 mrad), so the constant-width tube is accurate. **Verdict: the reference's own resolution (~0.2 mm) is 3–9× smaller than the observed 0.7–1.3 mm full-depth divergence and cannot be the explanation** — the divergence is the chamber-side charge-sharing slope bias, as §4 concluded.

5. **Threading census (answers open question 5) — "no per-event threading" needs sharpening.** Unsharing is genuinely per-event (hit-level waveform deconvolution + per-event refit); the accurate statement is that no reconstruction makes the *typical* event thread, not that threaded events don't exist. Defining "threads at T" as *every* core-strip hit on both planes within T of the reference line over the full depth (events with z_max > 15 mm, N = 5,081, raw production hits): **0.4% thread at 0.5 mm, 17.7% at 1.0 mm, 53.6% at 1.5 mm; median worst-hit 1.44 mm** — nearly independent of angle (15–25°: 16.4% at 1.0 mm), because the worst-hit metric is dominated by deep-hit diffusion scatter rather than the slope bias alone. Genuine full-depth threaders exist even at large angle: eid 32235 (13.2°, worst hit 0.35 mm over 21 mm), eid 11718 (17.6°, 0.38 mm over 23 mm). So honest "threading" display events CAN be picked — they are the ~1-in-6 tail, and should be captioned as such. A full-depth (worst-hit) pick criterion would replace `pick_by_residual`'s mesh-only criterion for that purpose.

Figures 07/08 and the per-band statics/GIFs were regenerated with the band; the legend quotes the σ values. Audit scripts: session scratch `audit_threading.py` (signed-slope check, anchor round-trip), `threading_census.py` (full-depth threading fractions).

*Prepared for independent review. The central, deliberately un-sugar-coated claim: we thread the reference at the mesh (position resolution ~0.5–0.8 mm, no misalignment), but we do NOT thread the full-depth charge cloud event-by-event with any current reconstruction — the raw ladder is ~4° too steep (charge sharing), and the hit-level unsharing that fixes this is a statistical/ensemble correction that adds per-event noise. Whether a per-event reconstruction can be built that threads a single muon through the whole gap is the open problem.*
