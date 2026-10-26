# x↔y pairing of tracks in one chamber: simulation study and its validation on cosmics

Dylan 2026-10-26: "The x-y pairing looks very good, something I was worried we'd have degeneracy problems with. This
should be a full section of the reconstruction paper and should have a dedicated simulation study to go with it,
please get that prepared and then also prepare to validate the sim study with cosmic data."

Status (2026-10-26 night): **data gates done on all five chambers; S1 run and FAILS the σ(lq) and u50/u90 gates;
S2 run as a pessimistic bound; S3–S5 not run (gun ready).** Report: `make_report.py` →
`~/x17/cosmic_bench/satstudy/pairing_sim/report/report.html` (notes/mx17-xy-pairing-sim). See §7.

## 1 · The question

Each MX17 view measures one projection. With N tracks in a chamber there are N! ways to combine N x clusters with N y
clusters; for N = 2 the "ghost" assignment is as consistent geometrically as the true one. Time does not break it for
prompt tracks (both reach the mesh together; t0 alone pairs 65 % on real events). What does: the two views read the
**same avalanche**, so one particle leaves the same charge (and the same depth profile) in x and y, while two
particles almost never leave the same charge (Landau).

What the paper has to show: how often pairing by charge is right, as a function of everything that can degrade it
(the two tracks' charge ratio, separation, opening angle, gain, noise, saturation, attachment, N), with the
mechanism understood, a simulation that reproduces the measured ingredients, and the simulation then used where the
data cannot go (X17 pair kinematics, n_TOF conditions, N = 3).

## 2 · What the data already established (det3 golden row, sat_det3, 490 V)

| measurement | value | script / output |
|---|---|---|
| x/y charge ratio of one track, robust σ of ln(q_x/q_y) | 0.054 (median −0.059) | pairing.py / pairing_toy.json `cal` |
| x/y spread of the charge-arrival quantiles u50, u90 | robust σ 93 ns each | same |
| toy (5 882 real singles, random pairs): right by charge / + profile | 95.8 / 97.6 % | pairing_toy.json |
| toy, right vs \|ln Q1/Q2\| | 46 % < 0.05, 91 % 0.05–0.1, ≥ 99 % above | pairing_toy.json `by_ratio` |
| real two-track events (110, M3 truth on one track) | 96.4 / 98.2 % (+ t0) | multitrack.json `pairing` |
| bench overlay, truth on both, ≥ 24 mm | 96.7 / 100 %; production's pairing 50 % | overlay.json |
| both tracks found vs separation | 0 % < 12 mm, 22 % 12–18, 98 % 18–24, 100 % above | overlay.json |
| earlier: n_TOF run_145 overlay (beam seeder) | swaps 24 → 14 % (A), 23 → 7 % (C) | wft/MULTITRACK_2026-09-14.md |

The mechanism in one line: pairing fails only when |ln Q1/Q2| ≲ 2σ(lq) ≈ 0.1, which Landau makes rare (12 % of
pairs); the depth profile is an independent handle that halves those failures.

## 3 · Simulation

Base: the response simulation `~/CLionProjects/MX17_Geant` (Stage A Geant4 → Stage B digitiser → Stage C DREAM →
`sim_decoded.root`, read by wft unmodified; `design/RESPONSE_SIM_PLAN.md`). The production wft reconstruction is run
on the simulated files exactly as on data (phys bundle of the golden row; the planned unified reconstruction once it
exists).

### 3.1 What the simulation must get right — and what is known to be off

Pairing depends on (a) the x/y charge split of one avalanche and its event-to-event spread, (b) the Landau spread of
the total charge, (c) the depth profile (diffusion, attachment, δ-rays), (d) noise and thresholds, (e) the sharing
kernel (how close two tracks can be). Known response-sim discrepancies to carry as systematics, not to fix here:
amplitude deficit ×1.46 at fixed gain (T14), HV-slope 1.52× the sim's, the det3 X asymmetry (closed, cause not
identified). (a) is the critical one: the measured 5.4 % spread must come out of the sim, not be put in.

### 3.2 Event samples

| sample | how | purpose |
|---|---|---|
| S1 single cosmic muons | existing cosmic gun, bench geometry, det3 conditions (490 V, bench gas) | validation set (§4) |
| S2 sim overlay | two S1 events summed in the waveforms, as `rerun/satstudy/overlay.py` (same code, sim inputs) | apples-to-apples with the data overlay |
| S3 prompt pairs | two-primary gun: e+e− from a vertex 0–30 cm upstream, opening angle 5–180°, energy split from ipc_born.py (X17 at 17.6 MeV and the IPC continuum) | the physics case: X17 / IPC pair kinematics |
| S4 n_TOF conditions | S3 with beam gas (1.5 % H2O, ~160 ppm O2 attachment), noise ×2 (post-23-July), n_TOF HV | does pairing survive the real run? |
| S5 three tracks | three-primary events, N! = 6 assignments | beyond pairs (showers, pile-up) |

Stage A needs one change: the gun fires k primaries per event with a common vertex and time (`--n-primaries`, opening
angle / energy split from a table). Everything downstream is per-hit and already handles any number of tracks.

### 3.3 Grid and outputs

Per sample, the pairing right-fraction (charge; charge + profile; + t0) against: |ln Q1/Q2|, separation in each view,
opening angle, gain (scan), noise ×1/×2, saturated strips present or not. Also: both-found fraction vs separation; the
two-track fit (WFT_TWO_TRACK_FIT) below 12 mm. Truth is exact in the sim (each cluster's parent track id), so swaps
and ghosts are counted directly.

## 4 · Validation on cosmics (the gate before any sim number goes in the paper)

Same reconstruction, same scripts, sim S1/S2 against data. The sim is accepted for pairing if, within statistical +
stated systematic bands:

| quantity | data (det3 golden row) | test |
|---|---|---|
| σ(ln q_x/q_y) of single tracks | 0.054 | ratio sim/data within ±20 % |
| median ln q_x/q_y | −0.059 | stated (sets the cost's centre, not its power) |
| total-charge distribution (Landau width) | from the final_phys q_sum | KS / quantile match after the T14 gain factor |
| σ of u50, u90 differences | 93 ns | within ±20 % |
| pairing right vs \|ln Q1/Q2\| (toy built from S1 exactly as pairing.py toy) | pairing_toy.json `by_ratio` | per bin within errors |
| S2 overlay: both-found vs separation, right ≥ 24 mm | overlay.json | per bin within errors |
| real-event check (sim has no M3; skip) | — | — |

If σ(lq) disagrees, that is itself a detector-model finding (the x/y split is the readout's anode geometry) and
goes back to MX17_Geant before S3–S5 are used. The data numbers above are from det3 only; repeat the data side on the
other bench chambers (det2/4/6/7: their own golden rows, `multitrack.py feat/count`, `pairing.py toy`) so the
validation is not one chamber's.

## 5 · Paper section (reconstruction paper)

1. The ambiguity: N! assignments, why time does not resolve prompt tracks.
2. The handle: one avalanche read by two views; the x/y charge identity (σ = 5 %) and the depth profile.
3. The rule (cost = Σ min(z², 25) over lq, u50, u90 [, t0]), its calibration on single tracks.
4. Data: toy from real singles, the overlay bench (truth on both), the real two-track events — three independent
   numbers that agree.
5. Simulation: validated on cosmics (§4), then X17/IPC pair kinematics and n_TOF conditions, N = 3.
6. Limits: equal-charge pairs, < 12–18 mm separation (merging, two-track fit), saturation.

## 6 · Order of work

1. Data side on all bench chambers (cheap; existing scripts).
2. Sim S1 at det3 conditions → §4 single-track checks (σ(lq), Landau, u50/u90).
3. S2 overlay through overlay.py → §4 overlay checks.
4. Gun change (k primaries) → S3, S4, S5.
5. Paper figures via papers/<paper>/figures.toml.

## 7 · Results so far (2026-10-26, dylan-Yoga)

Scripts here: `gates_data.py` (data gates, five chambers), `mfratio.py` (model-free x/y ratio, data and sim),
`sim_gates.py` (S1 vs gates; truth split; the xysplit diagnostic), `sim_overlay.py` (S2), `make_report.py`
(+ `report_template.html`). Outputs: `~/x17/cosmic_bench/satstudy/pairing_sim/`. Sim reco on condor:
`/eos/user/d/dneff/pairing_sim/` (code tarball, bundles, `s1/` tables; job script and .sub in MX17_Geant
`scratchpad/2026-10-26/pairing_sim/`).

**Data gates** (clean singles = detected interior M3 rays, one candidate per view within 5 mm; det3 = the satstudy
class-S set exactly): σ ln(qx/qy) det2 0.068, det3 0.055, det4 0.099, det6 0.179, det7 0.135 (flat in angle; a 50 mm
position map is 5–10 % of it); σ u50 91/93/90/166/157 ns; toy right (q / q+u) det2 95.7/97.5, det3 95.8/97.6,
det4 96.8/98.0, det6 87.1/90.3, det7 90.4/93.0 %. det4's broad Landau (σ ln Q 1.5, stripes) helps it.

**S1** (v4c, 12 gun angles × 3 000, production wft with det3's phys bundle, t0 prior dropped, COL_GUARD/PIVOT on):
σ(lq) 0.086–0.101 vs data 0.047–0.056 at the same inclination (×1.8); σ u50 86–186 vs 69–105 ns; Landau σ ln Q 0.35
vs data 0.38 after a 25 mm gain map (passes). Toy from sim singles: 91.2 % (q), the profile adds nothing (u50 too wide).
The sim-calibrated bundle gives the same σ(lq) (0.100): not the bundle.
Decomposition (th00 / data < 5°): truth induced split 0.059 = 0.85/√N (per-avalanche: Stage B lands each avalanche
and its feedback secondaries at one point) → model-free waveform ratio 0.071–0.087 (data 0.031–0.039) → wft fit 0.101
(data 0.054). Diagnostic Stage B with the feedback secondaries displaced (`--feedback-spread-um` 150 / 400; prediction
registered in `stageBC_xysplit.sub`): truth 0.035 / 0.034 as predicted, model-free 0.051–0.061, but the fit stays at
0.108 — the sim's waveforms are fitted worse than data by the det3 bundle (cf. RECO_TRUTH's unexplained wft
nonlinearity on sim). Two sim defects, not one.

**Remaining:** fix the sim's x/y split (MX17_Geant: displaced feedback or an avalanche cloud, then re-check the v4c
pre-registered observables) and the waveform-level excess; then rerun S1/S2; then S3 (pair table from ipc_born.py +
n_TOF chamber geometry → `mm_sim --primaries-table`), S4 (beam gas needs an attachment table), S5.
