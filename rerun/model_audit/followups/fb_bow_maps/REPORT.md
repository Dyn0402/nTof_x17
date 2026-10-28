# FB — FOLLOWUPS items 4 (chamber bow) and 8 (map gallery), 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commit dc22ae3; site master 2533ce7.)

Notes: https://dylan-neff.web.cern.ch/notes/mx17-chamber-bow.html · https://dylan-neff.web.cern.ch/notes/mx17-chamber-maps.html
Paper: `papers/paper1/sections/08b_bow.tex` (after 08_tracking; figures bow_profiles, bow_closure in figures.toml).

> **Reviewed 2026-10-28** (`SKEPTIC_REVIEW_2026-10-28.md`): the drift tilt stands; the position-route depth is
> 1.2–1.9 mm ± 0.4 systematic (not ± 0.2), det6's map is mostly a slope-scale reconstruction bias, det7 X is unmasked;
> the M3 common dish is bounded at 0.0–0.6 mm; the column duration puts the sag on the mesh (det2/det7), cathode
> < 0.5 mm; the det3 x–y difference is not established and the Y-view w0 gradient excess is a fit effect. Numbers
> below are the original ones.

## Verdict: the bow is real
On all five chambers the mesh centre sits 1.5–2.3 mm lower than its rim (further from the cathode); the drift direction
tilts with it. Static per chamber to ~20 %. Three routes with no shared systematics see it; the fourth (column duration)
is inconclusive.

**For**
- Positions: position-residual slope vs M3 slope → mesh height per cell. X and Y strips each show a dish in both
  directions through independent M3 projections; all 20 curvature terms positive. Depth at 150 mm: det2 1.77 ± 0.20,
  det3 1.48 ± 0.24, det4 1.85 ± 0.18, det6 1.79 ± 0.19, det7 2.31 ± 0.37 mm (corners ~2×).
- Drift tilt w0 ramps ±0.6 µm/ns, sign as the geometry predicts on 10/10 views; measured/predicted gradient 0.4–0.9 on
  8/10 views (det6 1.1, 1.9). Physical range ½ (flat cathode) – 1 (cathode bows with board) → cathode mostly flat.
- Narrowest cluster (amplitudes only, no times, no fitted angle): its slope drifts across the chamber with the predicted
  sign on 9/10 views.
- Shape from positions alone + 2 numbers fitted to angles recovers 15 % angle resolution on held-out tracks (median) vs
  16 % for a free 2-D map.
- Excluded: M3 plane-scale (needs 6 % at the chamber; measured ≤ 0.15 %); sliding t0 (pinning per trigger class keeps
  79–97 % of the curvature); fringe fields (die within ~10 mm of the frame; ramp linear over 300 mm).
- The 7-30 gap study agrees (det2 rim droop, det3 deepest at centre → longer column in the middle).

**Against / limits**
- A common bow of the M3 planes cannot move the tilt or width routes but could change the position-route depth → the
  chamber dish is 0.5–1.3× the quoted numbers.
- Column route: gap = T × local ladder slope (slope = v, independent of gap) — but X and Y, timing the same column,
  disagree by 1–4 mm/(100 mm)², more than the bow's 1–2 → inconclusive.
- Own-axis curvature reads 20–40 % high (edge truncation); the other-axis values (1.0–2.2) are the conservative ones.

## Consistency (19 runs, own alignment + M3 rays, golden bundle per chamber)
Other subruns, nights (det2 after remount), both slots, other drift/mesh HV: 15/19 within 2σ of the chamber mean. Outlier
det3 6-25 (0.93 mm, 3.2σ low); det3 same bottom slot 6-23 reads 1.70 → the slot doesn't change the dish. det3 6-23
(ladder 4.4 µm/ns) and det2 6-17 (half-dead X) have broken angles → positions only. Transfer test: golden tilt shape on
other runs, only rigid tilt + kw re-measured → median 94 % of each run's own-map gain, across slots too (det3 6-25 X
1.16 → 0.99°, own map 0.99°).

## Sign (external facts only)
M3 z up (ray files Z_Up = 1302, Z_Down = 24; slot P2 702 mm above P1 232 mm); chambers cathode-up (rail board under the
readout; kw > 0 on every chamber) → "h lowest at centre" = mesh centre lower. Gravity, gas overpressure and mesh tension
all push that way. Plate model (1.7 mm PCB, 0.402 m aperture): own weight 0.10–0.24 mm, 0.3–0.7 mm per mbar
overpressure, ~0.8 mm from mesh laminated at 10 N/cm → measured depth needs a few mbar or combined loads (plausibility).

## Gate
Map acts after the fit (`tan_of`) → exact at table level. Median s68 gain: 2-D bow 16 %, shape-from-positions 15 %, old
1-D 18 %; position resolution 4–15 % better subtracting tan·h. With FC's settings (σ0 0.30, no per-event choice) the map
still gains 13–25 % on Y views and removes det6 X's −8 % compression; det4 X, det7 X gain nothing.

## Spec for FE (no wft changes made)
1. Bundle block `bow` per chamber (static, golden row): h surface (gu, gw, kuu, kww, kuw; mm); drift-tilt surfaces A_x,
   A_y (cu, cw, auu, aww, auw; µm/ns, constant removed); scale gradients s_x, s_y (su, sw); U = (u−200)/100,
   W = (w−200)/100, u, w = fitted p0 + 15 mm·tan of each view. Prototype `results/bow_block_<det>.json` (`set_bow.py`).
   Survives save/load like angle_map; dropped on any transfer to n_TOF.
2. Per run only w0, kw (`set_w0kw.py` after subtracting the bow shape).
3. `tan_of`: tan = (w − w0 − A(U,W)) / (kw·v·(1 + su·U + sw·W)), both views' coordinates, one iteration; single-view
   tracks take the other coordinate 0; `bow` precedes `angle_map`.
4. Track point: p0 at the mesh with z offset h(U,W); at the nominal plane position = p0 − tan·h.
5. Gate: re-finalise the five golden rows with the block; rerun held-out comparison + transfer rows.
6. Unit test: synthetic dish (kuu = kww = 2) reproduces input tan; block round-trips save/load.

## Map gallery (item 8)
11 slides (bow, w0, ladder/angle scale, v(t), column duration, gain from charge, connector gains, efficiency, dead
strips, noise bands) badged USED (dead list) / PARTLY (one constant per view of w0, kw, v; per-event band veto) /
NOT USED (everything else). FA's v(x,y)/v(t) not out yet → that slide shows existing time-quintile v, marked pending.

## Risks
M3-bow ambiguity bounds absolute depth only to 0.5–1.3×; bench map must never reach n_TOF (chambers vertical there,
different gas flow); HV rows small (380–715 muons); gallery efficiency maps pre-rerun (Aug 13–19, labelled); consistency
tables used promoted bundles (old σ0 handling); add-note also changed hub/index.html and sw.js (committed with the notes).

Files: scripts in this dir (xcheck.py, profiles.py, gap_route.py, physmodel.py, gate_bow.py, set_bow.py, make_bow_deck.py,
make_maps_deck.py, paper_figs.py, …), results/, deck/, figures/. EOS `/eos/user/d/dneff/x17rerun/followups/fb/`.
