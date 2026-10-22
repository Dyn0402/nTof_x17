# X neighbour asymmetry: contingency plan (2026-10-22)

**Decision gate: 2026-10-23.** One more day on the mechanism. If it is still unexplained at the end of that day,
this plan replaces the mechanism work. Nothing in it needs new data.

Slide note: <https://dylan-neff.web.cern.ch/notes/mx17-xasym-plan.html> (`make_xasym_plan_deck.py`).
Evidence: FINDINGS §38, §42 here; the route ledger in MX17_Geant `scratchpad/2026-10-19/asym_deck/FINDINGS.md`
(40 routes, eighth round = the reconstruction side); `wft/xasym.py`.

## How big it is

| quantity | size | source |
|---|---|---|
| raw ±1 asymmetry | aQ up to ±0.2; X only, smooth, two-signed, fixed per board; every June build; det3 June-15 (= chamber A) strongest | ledger routes 22–24 |
| production X position bias | K = −0.77 ± 0.13 × raw centroid map; board mean removed, max cell 44 µm (det3), 116 µm (det2) | route 40, `wft/xasym_maps.json` |
| X resolution | unchanged ≤ 5 µm after correction + re-alignment (det3 0.457 → 0.453 mm) | eighth round §6 |
| angle resolution | not measured separately; the quoted 1.15–2.51° already contains it; slopes do not follow it (det3 0.963, the best) | §38 |
| fit quality | the X χ² structure a symmetric footprint cannot follow; the σ0_x tension (per event 0.20–0.25 vs stacks 0.35–0.43) | §38, §42 |

At n_TOF, ≤ 0.1 mm is small against track pointing errors of millimetres. Same-chamber pairs see a common bias:
they are mm apart inside 50 mm cells, so it cancels in the separation.

## Decision: do not absorb it

- **Not a kernel term.** No mechanism makes the size: the best Y-compatible sim (700 µm strips + funnel +
  across-strip anisotropy, 5 MΩ/sq) gives ×1.8, the data need ×2.5–5. A fitted term would be a free parameter per
  board and per run condition.
- **Not the empirical correction as nominal.** Validated only for det2 and det3 June-15 (det4 worse, det6/det7 no
  gain), so chamber C has none. It needs a per-run amplitude scale (det3 0.44–1.57) and a re-alignment, and buys no
  resolution.
- **Instead:** freeze the symmetric model and carry `wft.xasym` as a systematic variation. Write the asymmetry up as
  a measured detector property.

## Plan by workstream

**Detector characterization**
1. Lift the kernel-freeze hold: freeze r06 (paper 1 question C2).
2. Per-event σ0_x 0.25 (§38: never costs overall resolution, slopes toward 1).
3. `wft.xasym` is a ± variation for anything that compares positions across a board or between chambers.
4. det3's D_T excess and the asymmetry become one det3 limitation (a plausible link, untested); gas transport is
   validated on det2/det4.

**n_TOF**
1. Nominal reconstruction unchanged.
2. Chamber A (det3 June-15): session scale from the n_TOF raw ADC (the raw-ADC asymmetry estimator of the ledger),
   `xasym.correct_x` as a ± variation in capsule pointing, alignment and vertexing. Re-align on corrected positions.
3. Chamber C (det6): flat ±0.1 mm X-position systematic.

**Papers**
- Paper 1: a subsection: every June resistive print shows an X-only, pad-pitch-locked, board-fixed field; what is
  excluded (electronics, connectors, copper/bus, track angle, the 800 µm beat); the leading hypothesis
  (resistive-strip registration, unproven); its position effect; the strip mask was never obtained.
- Paper 2: the forward model is symmetric in X; this is its main known mismatch (χ², σ0_x); quote the angle cost
  from the check below.

## The one check (simulation, ~1 day)

Inject the measured one-sided ±1 term (aQ ≈ 0.165, ~110 ns late) into the paper-2 physics-floor toy, reconstruct
with the symmetric model, and read off the angle bias and resolution cost. Expected a few hundredths of a degree
(§38 head-on +0.04°); unmeasured until it runs.

## Stop

The mechanism simulations (strip width, funnelling, ρ_s, anisotropy) and the strip-mask question.

## What this does not rule out

- A larger angle cost than §38 suggests, until the toy injection runs.
- An unmapped position bias on chamber C, carried as a flat ±0.1 mm.
- That det3's D_T excess is the same effect.
- A mechanism that also changes the Y view or the gap; nothing seen there so far.
