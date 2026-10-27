# ws_widthtilt — F4 / F18 / F21: why steep tracks read shallow, and the width–tilt trade, 2026-10-27

(Written by the coordinating session from the workstream's final message. Commits cad1cce, 774daf1.)

## Answer
Mostly NOT model inadequacy and NOT the width law. Two causes:
1. **Selection artefact.** `03_angles.py` drops fits with |tan_fit| < 0.08; in the shallowest reference bin that removes
   the fits that came out too shallow → the shallow bin is inflated 3–13 %. Without the cut (w0 removed, ± averaged) the
   steepest bin (|tan| ≈ 0.36) is within 0–6 % of the shallow ones: det7 0, det3 ~1.5, det4 3–4, det2 Y 4.5, det6 X 6 %.
2. **Chamber bow.** w0 is not constant: it ramps across every chamber and both views (≈ −0.6 → +0.6 µm/ns over
   300 mm, zero near centre). Through the telescope acceptance (position ↔ angle correlated, strongly on det6) this
   looks like an angle nonlinearity. M3 positions alone confirm it: the mesh-height curvature fitted from position
   residuals vs tan has the same sign on all 10 chamber-views, 120–250 /km (40–200 /km predicted from the w0 ramp) →
   a 1–3 mm dish. Gravity sag of an edge-supported board is the candidate; which way the centre sits is not pinned
   (frame z-sign not resolved here).

## Evidence
- Toy closure (all five chambers, real windows/noise/gains): true slope to ≤ 0.3 % at every angle.
- Injected physics (Landau, one-sided X copy ±0.13, wider true footprint): no bias.
- Model wider than truth (footprint 0.8×, diffusion 0.5×): ~2 % compression at |tan| 0.1, ~1 % at 0.36 — the
  opposite of F4's angle trend.
- Width–tilt trade on data (det4, det6): σ0 0.24 → 0.54 mm lowers the slope 3–10 %, most at shallow angles; no
  (σ0, Dp) on the grid makes the M3 angle the χ² minimum → real but percent-level, not the steep compression.
- Residual after the position correction: symmetric compression |tan| 0.15 → 0.36 of ~2 ± 0.5 % (det3, det4 X, det6 X),
  4 % det2 Y, 7 % det4 Y, none on det7. **Unexplained.**

## F18 / σ0
Per-event wide/narrow choice is harmful (agrees with ws_sps): in toys a truly wider footprint picked this way is
unbiased; in data the wide-pick fits read 5–15 % shallower even in the chamber centre (not the bow) and are 1.3–5×
brighter. With the choice off, bright tracks still read 5–10 % shallower on det6/det7 (det7 Y 0.95) → a
brightness-dependent mismatch the model doesn't describe.
Recommendation to H: fixed fleet σ0 from SPS head-on (Y 0.32–0.343, X 0.24–0.29), drop the per-event choice for the
angle, keep the bow correction separate (geometry).

## Fix (opt-in): per-chamber position map of w0 and angle scale
Measured on the bench from M3, applied in the angle conversion. Cross-validated (measure on one half, apply to the
other): s68 X / Y det2 −12/−26 %, det3 −12/−16, det4 −3/−25, det6 −26/−27, det7 −1/−18 %; far fractions unchanged.
Code: `wft/calib.py` optional `angle_map` (survives save/load); `wft/reco.py` `tan_of()` in fit_plane/_child_fit,
table meta records planes with a map; `wft/tests/test_angle_map.py`; this dir: set_angle_map.py, toy_closure.py,
width_probe.py, gate_artefact.py, data_split.py, column_vs_angle.py, position_scale.py, curvature_fit.py,
w0_profile.py, w0map_xval.py, bow_test.py, width_summary.py, make_ws_dag.py. Maps: `angle_maps.json`,
`bundles_amap/<det>/` under `~/x17/cosmic_bench/model_audit/ws_widthtilt/`.

## Risks / open
- Bow is mounting / board-sag geometry → never transfer the bench map to n_TOF without an in-situ check (like w0).
- Rim bins (< 30 mm from edge) flip sign on det3 X and det2 — kept, unchecked.
- M3 plane-scale artefact excluded only by plausibility (would need a 5–10 % scale error).
- Bow biases position by tan × height (~±0.45 mm at tan 0.3) — likely part of F14's 0.5–0.76 mm.
- Width probes on det2/det3/det7 and toy-mode probes still on condor (ws1b, ws2; EOS
  `/eos/user/d/dneff/x17rerun/model_audit/ws_widthtilt/`) — the data trade rests on det4/det6 only.

Figures (`figures/`, copied from `~/x17/cosmic_bench/model_audit/ws_widthtilt/`): fig_f4_gate, fig_position_map,
fig_w0_profile, fig_bow_test, fig_w0map_xval, fig_resolution_map, fig_toys, fig_width_trade.
