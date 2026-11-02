# Paper I story spine (2026-11-01)

One line of argument per section, in Paper I order: the claim, the headline numbers (source), the figures, the caveat,
the open decisions. Numbers come from `rerun/PAPER1_NUMBERS_2026-10-31.md` (the sheet), checked for cross-section consistency in
`rerun/consistency/README.md`. Provisional choices where Dylan hasn't decided: run 63, interior ≥ 30 mm, no-clip s68,
pass arm (the v(t) arm in brackets).

**The paper in one sentence:** a resistive micro-TPC Micromegas whose geometry is reconstructed from waveforms, not hit
times, reaches
- ~1° angles on cosmics;
- ~190–210 µm per view at normal incidence in a beam;
- ~40 ns event time;
- 90–93 % efficiency on the good chambers.

Every loss and every residual term is accounted for, except a ~0.6° angle remainder that is measured and bounded, but not
identified.

| § | claim | headline numbers (source) | figures | caveat | decisions |
|---|---|---|---|---|---|
| 1 Introduction | X17 at n_TOF needs pair angles; a micro-TPC MM gives them from one plane | — | (physics case, MPGD) | open section | — |
| 2 Design | 400 × 360 mm resistive-strip MM, 0.78 mm pitch, 30 mm drift | efficient area 399.4 × 359.9 mm² (not a pass product) | chamber_stack, readout_layout (placeholders) | spec page: 9 unknowns, 31 assumed | — |
| 3 Setups | cosmic bench with the M3 reference (z up, slots 702/232 mm, scintillator trigger); SPS H4 with a uRWELL telescope | M3 pointing 0.21–0.28 mm per axis (sheet MPGD row) | bench_setup (current), sps_setup (placeholder) | — | — |
| 4 Operation | gain, discharges and dead time set the operating point | gain ×2 every 16.6–19.1 V; discharges ×10 every 21–42 V (∝ gain^2–2.7); dead time per discharge ≤ 1–3 ms (det4 ≤ 15); optima 480 / 470 / 480 / 430 V (det2/3/6/7), with the golden rows 10–65 V above; det4 at SPS ×1.65 over 580–780 V (OPS, CHG) | hv_scan, spark_deadtime (rebuilt) | 6-23 night excluded (chambers not yet at gain; CONS §1); "muon-induced 4–6×", "edges" not re-derived; det4's SPS HV path unknown (Stephan) | G, H |
| 5 Charge balance, efficiency | X and Y share the charge evenly; efficiency is set by the chamber, and every loss is classified | f = 0.485–0.513, s68 0.014–0.032 (det7 0.116) (CHG); efficiency within 5 mm, interior: det2 90.3, det3 92.6, det4 40.9 (71.3 in live bands), det6 74.4, det7 68.0 % (EFF); a drift-field fringe 20–30 mm in from the unpassivated x sides sets the 30 mm fiducial (FIGS) | charge_balance, efficiency_map, edge_turnon (rebuilt) | det7 X on ch 256–511 only; det2's edge drift time open (2.5σ, CONS) | E, I, Q |
| 6 Gas | the bench gas was Ar/iso + 0.4–1.0 % water and dried over hours; v(E) follows Magboltz at one water | det3 0.97 % [0.95, 0.99] at all 7 fields, χ² 2.5/3; golden rows det2 0.69, det4 0.42, det6 1.00, det7 0.48 %; 6-26 det6 1.47 → 0.76 %, det7 0.68 → 0.35 % (one time shape); det6 early-night attachment ~600 → 370 ppm O₂-eq (×2) (GAS) | gas_vE_timing (recommended), gas_vE, attachment (pending: SPS gas fit) | bench vs SPS comparison waits for the SPS gas fit; det4 6-23 water needs an attachment caveat (CONS); no gas log | K, L |
| 7 Timing | the chamber times a cosmic to ~40 ns on its own | event time vs the trigger 40–46 ns (1.5–1.7 mm); per view 22–28 ns; common to X/Y 35–40 ns; det3 drift field 114 → 38.5 ns (104 → 382 V/cm); SPS not measurable (ZS + low signal: 140–218 ns upper bound) (`rerun/paper1_timing`) | timing (rebuilt) | det4 bench timing carries the 6-23/6-24 trigger term; the fit's t0 is pinned, so the time comes from the column start | M, N |
| 8 Tracking | waveform reconstruction gives ~1° angles and a chamber position term of ~120–180 µm near normal (bench), 176–209 µm in the beam | s68 det2 1.045/1.125, det3 0.991/0.901, det4 2.412/2.076, det6 1.256/1.506, det7 1.232/1.461° (RES; v(t) arm ≤ 0.063° lower); SPS run63 X 209 ± 11 (p_mid), Y 190 ± 10 (p0), run53 Y 176 ± 10 (hits) (SPS, sps_junk); bench chamber near normal det2 163/176, det3 126/116 µm (FLOOR); xasym systematic ≤ 5 µm; vs the hits chain 2–3× better in angle (fleet) | tracking_perf (rebuilt), sps_resolution (now buildable: the SPS final pass is done), fleet (Paper II) | unexplained remainder 0.56–0.65° on clean events (det4 1.42°): not M3, not a common cause, not K1 (FLOOR, K1); residual bootstrap pending. det4/6/7 in-sample. "176 µm = 0.30 × pitch" is wrong (0.23 × pitch) | A, B, C, D, O, P |
| 8b Bow | the readout boards are bowed, and one block per chamber transfers across days | dish per view 0.90–2.45 mm; angle gain median 15.7 % (3.6–28.9); position 1.7–10.6 %; the gate improves every row (BOW, RES) | bow_profiles, bow_closure (rebuilt, v(t) arm) | w0/kw stay per row | F |
| 9 det4 stripes | det4's non-amplifying stripes are a gain continuum, the same on the bench and at SPS | 52 % of the x range below half the plateau; 41 % two-view detection; bench vs SPS agree on 82–85 % of 1 mm bins (FIGS) | det4_stripes (rebuilt) | the old "62 % never amplifies" is replaced | — |
| 10 Conclusions | — | — | — | open | — |
| A1 Loss categories | every lost track has a class | det3: 92.64 % detected; noise band 0.22, second particle 1.02 + 1.43, discharge 0.50 (0.37 + 0.12 burst) …; det6/det7 discharges 16.7 / 24.2 % (real, muon often visible) (EFF, FIGS) | 14 loss_* displays (rebuilt) | class rules tuned on det3, audited by eye on all chambers | E, H, J |

## Decisions (one list; letters A–J from the sheet, K–Q new from the section READMEs)

- **A.** SPS run: run63 (recommended) or run53.
- **B.** SPS position estimator, per view.
- **C.** v(t) arm or pass arm for the bench angles.
- **D.** s68: no clip, or the ±5° clip.
- **E.** A1: one column per chamber, or det3 only.
- **F.** Bow numbers: per view, or a per-chamber mean.
- **G.** §4: quote det6/det7 at the golden HV and/or at their scan optima; which chambers "480/480/440 V" names.
- **H.** Spark-veto bursts: a separate class, or one spark class with a stated note.
- **I.** Fiducial: 30 mm on every side, or 10 mm on y for efficiency-only numbers.
- **J.** Recovering det6/det7 discharge events (a reconstruction change): do it, or state it.
- **K.** §6: `gas_vE_timing` (recommended) or `gas_vE`.
- **L.** Quote det6's early-night attachment (×2 model factor)?
- **M.** §7: quote the event time 40–46 ns, with the per-view/common split (recommended)?
- **N.** MPGD proceedings "33 ns / 1.1 mm": replace it with the waveform number, or keep it labelled as hits?
- **O.** §8 wording on the angle remainder. The 10-27 "not a waveform misfit" is reversed on pass 1, and K1 is ruled out.
- **P.** §8 bench position: the chamber term (~120–180 µm near normal), or the residual with the M3-scattering and pointing
  terms named.
- **Q.** §5 edge figure: det3 (the paper's chamber) or det2 (10× the triggers).

### Dylan, 2026-11-02 (answers to the laptop outline review, session 40c9307a)

- **Focus on det3.** The fleet efficiency maps appear early and show that det4/det6/det7 have production problems;
  no detailed efficiency study of the other chambers. (Settles E: A1 is det3 only.)
- **Efficiency definition (det3):** the default number gives credit when one of several reconstructed tracks is the M3
  track (95.1 %, not 92.6 %). δ-electrons (DELTA + FIT_SAT) are **losses**: a true inefficiency, named as a category.
  Break det3 down into single-track and multi-track events separately. Open: which further classes leave the
  denominator (proposal in the 11-02 reply: bent-M3 position tail and the other-coordinate edge = reference/geometry).
- **Discharges:** a well-built chamber reaches full efficiency before discharges limit it; show efficiency vs HV with
  the discharge loss overlaid.
- **det4 at SPS:** state that det4 went to SPS because the four good chambers (det3/det2/det6/det7 = A/B/C/D) were
  installed at n_TOF over the same weeks (SPS runs 30 Jul–2 Aug 2026). Dylan to confirm wording.
- **Body, briefly:** X/Y charge sharing and the time resolution (M: yes, in the body). Edge study: expand, possibly
  grouped with the bow as "geometrical / mechanical deviations" (they are separate mechanisms: the bow predicts the
  edge column shorter, the fringe makes it longer; both are field/geometry non-uniformities).
- **Resolution:** quote the intrinsic resolution only with the reference subtraction stated and realistic
  uncertainties, on the bench (det3 near normal 126/116 ± 35 stat, ±~50 µm from ±10 % on the M3 pointing) and at SPS
  (region R 138/120 µm, telescope-f range 112–153 / 89–136). (P: the chamber term, with the caveat.)
- **n_TOF:** nothing in Paper I beyond motivation and requirements.
- **Design section:** starter notes for Stephan in `papers/paper1/sections/02_design.tex` §Design choices (pads for
  X/Y balance, resistive TPC trade-off, strips vs layer and their orientation, passivation).
- **AI disclosure:** Claude Code's role is stated explicitly (draft in `main.tex` Acknowledgements; methods paragraph
  in §3 still to write).

(The HANDOFF's G/H, the duplicate reproc files and the bench gas fit, are not paper decisions. The gas fit is done, in §6.)

## Facts only Dylan or Stephan can supply (the data can't)

- Had det6 just been opened or refilled before 6-26?
- What changed in the trigger or the paddles on 6-23 and 6-24? The data show a 40–60 ns timing term, and the only config
  difference is the pre-6-26 DAQ config with no fresh pedestal run.
- Was det3 or det4 opened on 6-22/6-23, and what was the gas line order? That would identify (a) in CONS §1.
- For Stephan: is det4's SPS resist HV fed through a filter or a series resistor (imon ~0.7 µA at 770 V)?
