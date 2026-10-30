# Parameters of the reconstruction and the bench/SPS analysis

What every model and analysis parameter means, where it is set, and where it is used. These are the values of the from-scratch
pass (`Analysis_pass1`, code `6a0bb1f`; the 01–04 stages `b5a29af`; SPS run71 `a72b889`). When a value changes, change this file in the same commit.

**Where parameters live**, from the most physical to the most procedural:

| layer | file | per |
|---|---|---|
| physics inputs, measured outside the reconstruction | `rerun/calib/physics_inputs.json` | chamber |
| calibration bundle (`bundle.json` + `arrays.npz`) | `<row>/wft/calib_bundle/`; class `wft/calib.py:CalibrationBundle` | chamber × run condition (golden), row (vrefit) |
| pass registry: which row inherits what | `rerun/pass_registry.csv` (`rerun/make_registry.py`), `rerun/sps_registry.csv` | row |
| algorithm constants (no per-row values) | top of `wft/reco.py`, `wft/model.py`, `wft/seed.py`, `wft/compat.py` | global |
| stage arguments (alignment, efficiency, angles) | `mx_june_wft/01–04_*.py`, `cosmic_bench_analysis/cosmic_micro_tpc_analysis.py` | global |

Frame: detector-local x/y from the strip maps (pitch 0.78 mm, plate 0–400 mm), z = drift depth from the mesh. +y is up
(`COORDINATES.md`).

---

## 1 · What the fit measures per event and view

Each view (x, y) of an event is fitted independently. The model is a **straight line of charge**. Charge arriving at time t0 + u comes from transverse position

  p(u) = p0 + w · u

for drift time u after t0. The free parameters are p0, w and t0, and the charge in each of K depth bins of 60 ns (a linear sub-problem, solved exactly for each (p0, w, t0)).

| column | meaning | unit |
|---|---|---|
| `p0` | track position **at the mesh** (the first-arriving end of the column), not the charge centroid | mm |
| `w` | transverse speed of the arriving charge along the strips | mm/ns |
| `t0` | arrival time of the charge from the mesh | ns |
| `tan_theta` | track slope, from w through the angle mapping (§3), with the bow applied | – |
| `q_sum` | total fitted charge | ADC units |
| `q_u50`, `q_u90` | drift time after t0 by which 50 / 90 % of the charge has arrived | ns |
| `q_uend` | last depth bin above 5 % of the profile peak: the **column duration** | ns |
| `chi2`, `dof` | over every sample of every strip in the window (so χ²/dof is large and grows with signal) | – |
| `p0_err`, `w_err`, `t0_err`, `tan_err` | 1-D curvature errors; `tan_err` includes the per-chamber `tan_floor` | |
| `n_strips` / `n_seed` / `n_dropped` | strips in the fit window / in the seed cluster / in competing clusters | |
| `n_flagged_strips`, `n_band_strips`, `n_sat_strips` | of `n_strips`: dead/hot wildcards, noise-band strips (censored), strips reaching the ADC rail | |
| `ok` | a fit was made | bool |
| `quality_ok` | χ²/dof ≤ `CHI2DOF_REL` (5) × the median of this view over this table (§5) | bool |
| `slope_reliable` | \|tan\| ≥ `TAN_MIN_SLOPE` (0.08); below that the timing carries no slope information | bool |
| `isochronous` | the column arrives all at once (head-on): angle poorly constrained | bool |
| `x_bow_h`, `y_bow_h` | the chamber bow's mesh height at this track (§3.3); the position at the nominal plane is p0 − tan · h | mm |
| `n_tracks`, `n_splits`, `n_candidates` | multi-track bookkeeping (`wft/MULTITRACK_*.md`, `wft/TWO_TRACK_FIT_2026-09-16.md`) | |
| `timestamp`, `acq_tag` | DREAM timestamp and acquisition, for v(t) | |
| SPS only: `ref_x`, `ref_y` | det4's **own** charge centroid over its seed window, **not** an external reference (§8) | mm |

---

## 2 · The forward model (footprint and electronics): `hyper` and friends

Charge drifting for a time u lands with a Gaussian footprint, then spreads along the resistive strips:

  σ²(u) = σ0² + Dp² · u + (w · 60 ns)²/12  then, on Y only, σ² += 2 · rc_D_y · s   (s = time since landing; the middle
  term is the track's own extent within one 60 ns depth bin)

and every increment of charge on a strip is seen through the electronics template (`wft/model.py:build_matrix_rc`).

| parameter | where | value (pass 1) | meaning | source |
|---|---|---|---|---|
| `sigma_p0` (σ0) | `hyper` | 0.30 mm, both views, all chambers | prompt footprint: avalanche + induction, a readout constant | FC (`fc_sigma0_trade`), SPS-measured; free bench σ0 is not a measurement |
| `Dp` | `hyper` | det2 0.025, det3 0.023, det4 0.020, det6 0.018, det7 0.0225 mm/√ns | transverse drift diffusion (Dp² = D_T² v) | unsaturated head-on stack; agrees with Magboltz on det3/det4 |
| `rc_D_y` | `hyper` | det2 5.1e-4, det3 3.8e-4, det6 4.6e-4, det7 6.8e-4 mm²/ns; det4 2.2e-4 (stated, H4) | RC spreading along the resistive strips (Y view) | fitted in the calibration (DRC node, pinned t0, unsaturated planes) |
| `rc_D_x` | `hyper` | 0 | the same across the strips: none | physics |
| `rc_tmpl` | `hyper` | 1 (= 'x') | the X-derived electronics template drives both views (Y's own carries RC undershoot) | TEMPLATE_STUDY T14/T15 |
| template (`grid`, `tmpl`) | `arrays.npz` | measured per chamber | electronics impulse response | `measure_templates` on the calibration cache |
| `gain` | `arrays.npz` | 1.0 everywhere | per-channel gain | no gain map (GA: not needed) |
| `t0_abs`, `t0_prior_sigma` | bundle | t0 = C − 10 ns · ftst; σ 0.05 ns | **pinned** trigger latency: t0 is fixed by the scintillator trigger up to the 10 ns ftst clock phase | `measure_latency` (det3 C = 318.6 X / 326.5 Y ns). **Bench only**: dropped on transfer to SPS / n_TOF |
| `sat_adc` | bundle | 3550 ADC | the ADC rail. Saturation uses the locked simple model (K1–K9) | `rerun/calib/saturation/DECISION_2026-10-29.md` |
| `n_depth_bins` (K) | bundle / registry | bench 24 (30 at drift ≤ 500 V); SPS 50–57 | number of 60 ns depth bins: the longest column the fit can represent | registry rule; SPS K from v_span + 15 bins |
| `sample_ns`, `depth_subdiv` | bundle | 60 ns, 1 | DREAM sampling period; depth bins per sample (3 = the 20 ns grid, not adopted) | |
| `pitch_mm` | bundle | 0.78 | strip pitch | |
| `share_mode` | bundle | 'delay' | legacy sharing-kernel switch; inactive when `rc_D_y` is set (the physical RC kernel replaces c1/c2/kY/τ_s/σ_s) | |
| `dead` | bundle | e.g. det3 x [307, 398, 400], y [510] | no signal rate: **censored** (unknown charge, not zero), and the seeder bridges them | `rerun/channel_layer/` census + shorts |
| `hot` | bundle | mostly empty | signal-carrying noisy channels: kept in the fit with noise ×10 (`HOT_NOISE_INFLATION`), never seeded on | |

The old copy-amplitude parameters (`c1`, `c2`, `kY`, `tau_s`, `sigma_s`) are retired. A bundle that still has them must have c2 < c1
(`wft.calib.check_kernel_ordering`), and on an r06 bundle the ratio is in `c2_over_c1` (read it with `wft.calib.effective_c2`).

---

## 3 · From w to an angle: `v_drift`, `w0`, `kw`, the bow

### 3.1 The constant mapping

  tan θ = (w · 1000 − w0) / (kw · v)        (w in mm/ns, w0 in µm/ns, v in µm/ns)

| parameter | per | meaning |
|---|---|---|
| `v_drift` (v) | golden row: the chamber's input; own-v rows: their own | drift velocity. Golden rows carry the **input**, the timing v = gap / column duration (`physics_inputs.json`: det2 39.3, det3 35.8, det4 33.0, det6 24.17, det7 36.8 µm/ns). Own-v rows (registry `refit = v`) carry v_golden × the mean of their two measured kw (`set_w0kw --fold-kw`, e.g. det6 6-26 longer_run 24.17 × 0.824 = 19.9). Rows with `refit = none` take their `calib_key`'s |
| `w0` | row × view | **tilt**: the transverse speed of a perpendicular track (drift field or mount not exactly along z). A pure offset in angle |
| `kw` | row × view | **scale**: multiplies v, measured on each row against the M3 reference angles. After the fold, an own-v row's kw is the X/Y split around 1 (0.96–1.04); a golden row's kw is unfolded (1.00–1.08) |

**Why both v and kw.** For the angle they are degenerate: only kw · v enters. They are kept apart because they do different jobs:
- **v sets the reconstruction's windows.** The fit itself works in time (w in mm/ns, depth bins in ns), so v only sizes things inside the reconstruction:
  - the plausible column length 30 mm / v (`wft.reco.u_window`, 0.30–1.60 × that);
  - the position lever arm 15 mm / v;
  - the start guess for w;
  - the registry's K.
- **kw is the angle scale measured against M3.** It absorbs two things:
  - the row's real drift velocity relative to the bundle's v: drift field, or gas exchange in a night;
  - the method's own angle scale. On golden rows, where v was measured, kw is still 1.00–1.08, so the timing v and the angle-implied v disagree by a few %.

**How the pass uses them (vrefit, `make_pass_dag.py vrefit --keep-v`).**
1. Reconstruct 8000 events with the golden bundle.
2. Measure w0 and kw on them against M3.
3. **Fold** the mean kw into v (v_row = v_golden × mean kw).
4. Reconstruct the row with that bundle.

So every own-v row is reconstructed with its own effective drift velocity, and its windows match its column. Before the fold, the measured
kw is the ratio to the golden v (det3 drift scan 0.06–1.09 against field; det6 6-26 night 1.08 → 0.73 as the gas changed). The
provenance keeps both (`kw_folded_into_v`: v_was, kw_mean), and `w0kw.json` lists the unfolded `kw` next to `kw_folded`.

**What remains:**
- v drifting *within* a row: one v per row in the tables. On the 6-26 night v **rises** with clock time in both det6 (17.3 → 31.5 µm/ns, +20 % inside the
  golden long_run) and det7 (+24 %, +4 % inside long_run), the same time-shape (the gas). Post-correction `r(t) = v_rel` per event, applied as v → v·r(t)
  in the angle mapping: `rerun/vt_within_rows/` (det6/det7 6-26 only; corrected tables in `/eos/user/d/dneff/x17rerun/vt_correction/`).
- The low-field rows, whose column overruns the 32-sample window whatever v is (100 V, stated limitation).
- `refit = none` rows, which inherit their `calib_key`'s folded v.

| also | meaning |
|---|---|
| `tan_floor` | per-view model floor of the per-event angle error (det3 0.0185 X / 0.0163 Y in tan, ≈ 1°); enters `tan_err` |
| `angle_map`, `v_map` | position- and time-dependent mappings: present in the bundle class, **empty in pass 1** (FA's v(x,y) not adopted). v(t) is applied to the tables afterwards, not in the bundle: per row `vt_map.json` = the `v_map['t']` block (`t_h` = hours since 2026-06-26 00:00 from acq_tag stem + DREAM timestamp × 10 ns, `v_rel`), `rerun/vt_within_rows/`. Every row: `rerun/vt_all/` (`VT_PLAN_2026-10-30.md`), clock hours since **2026-06-01** 00:00 (stated in each file) |
| VT gate (`rerun/vt_all/vt_bench.py`) | `N_PER` 300 kw-window tracks per block and view, `N_MIN` 150 (a row with < 300 gets no time measurement); a row is `correct` when the joint line moves \|Δv\| ≥ `GATE_DV` 2 % over the row at ≥ `GATE_Z` 3σ, on ≥ 3 blocks, with x and y agreeing; `unstable` when the line's χ²/dof > `GATE_CHI2` 3 (dof ≥ 5). SPS: both views ≥ max(2 %, 3√2 σ_win), σ_win = 1.2 % per 1000-event window (run63 flat700 control). n_TOF: k per sub-run where one k per run fails (χ² p < 10⁻³ or 2 % at 3σ) |

### 3.2 Where w0/kw come from
`rerun/w12/set_w0kw.py` fits w = w0 + kw · v · tan_ref over the row's M3-matched events (03_angles `per_event.npz`). With a bow block it fits
**after** subtracting the bow shape. In `w0kw.json`, `kw` is the measured scale relative to the input v, and `kw_folded` is the same divided by
the mean of the two views (the fold, §3.1), which is what the bundle carries (det3 480 V: 1.0312 → 1.0151 X, 1.0005 → 0.9849 Y).

### 3.3 The chamber bow (`bundle['bow']`)
This is a static block per chamber, measured on its golden row and inherited by every row of the chamber. It is bench only and never sent to SPS or n_TOF.
Coordinates: U = (u − 200 mm)/100 mm and W = (w − 200 mm)/100 mm, where u = p0_x + 15 mm · tan_x and w = p0_y + 15 mm · tan_y, i.e. the track at mid-gap
(`BOW_MID_MM` = 15). A view whose point is off the plate (|U| or |W| > `BOW_UW_MAX` = 2.2) counts as missing (U or W = 0).

| term | meaning | unit |
|---|---|---|
| `a_x`, `a_y` (cu, cw, auu, aww, auw) | **drift-tilt surface in tan form**: tan = (w − w0)/(kw · v · S) − a(U, W). This is the form that transfers across drift fields | tan |
| `s_x`, `s_y` (su, sw) | scale gradient: S = 1 + su · U + sw · W | – |
| `h_x`, `h_y` (gu, gw, kuu, kww, kuw) | **mesh height** surface (the mesh is not flat): position at the nominal plane = p0 − tan · h | mm |
| `A_x`, `A_y`, `B_*` | the older µm/ns form; `a_*` takes precedence | |

Evidence that one block per chamber is enough: `rerun/calib/bow/results_pass1/GATE_RESULT_2026-10-30.md`. With per-row w0/kw, every row improves by
0.06–0.53° in angle and 0.01–0.05 mm in position. With the golden w0/kw as well, rows of another day get worse, which is why w0/kw stay per row.

---

## 4 · Pairing the two views and several tracks

| parameter | where | meaning |
|---|---|---|
| x–y gate | `wft.reco` (`xy_gate`, `tc_tol_ns` in the events meta) | an x and a y fit belong together if their column times (t0 + u50) agree within the tolerance (p98 from data, ~160–750 ns per chamber) |
| `xy_pairing` | bundle | median and robust width of x − y features (`lq` = ln qx/qy, `u50`, `u90`, `tc`) on clean singles, for pairing several tracks |
| `dt_xy` | bundle | legacy t0x − t0y per ftst difference; superseded by the column-time gate |
| `TWO_TRACK`, `TWO_TRACK_F` (300), `MAX_TRACKS` (3), `two_track_min_sep_mm` | `wft.reco` | joint two-track fit and its trigger. See `wft/TWO_TRACK_FIT_2026-09-16.md` |
| `HEADON_RESCUE` | `wft.reco` | when one view has a track and the other no seed, look for a drift column at that t0 in the other view's full waveform; keep it only if ln(qx/qy) matches within 3 × the bundle's `xy_pairing` width (part of the algorithm since 2026-10-27) |

---

## 5 · Algorithm constants (the same everywhere; changing one means a new code hash)

| constant | file | value | meaning |
|---|---|---|---|
| `PIVOT_START` | `wft/reco.py` | True | start the fit from lines forced through the charge centroid (x_c, t_c), scanning w ± 0.032 mm/ns |
| `CHI2DOF_REL` | `wft/reco.py` | 5 | `quality_ok` if χ²/dof ≤ 5 × the **table's** median (relative: χ²/dof runs 4–740 by chamber). A flag for second deposits and showers, not a reconstruction filter |
| `TAN_MIN_SLOPE` | `wft/reco.py` | 0.08 | below it, `slope_reliable` = False |
| `U_MIN_FRAC`, `U_MAX_FRAC` | `wft/reco.py` | 0.30, 1.60 × (30 mm / v) | plausible column-end window |
| `SPARK_VETO_HITS` | `wft/seed.py` | 50 | more hits than this in the detector = discharge: the event is vetoed |
| `SIG_REL_FLOOR` | `wft/seed.py` | 0.10 | per plane, strips below 10 % of the strongest are not counted toward multiplicity |
| band censoring (`band_nsig` 5) | `wft/model.py` | on bench and SPS windows ≥ 1800 ns | strips in a noise band in this event are censored and the event re-seeded without them |
| `MAX_DROPPED` | `wft/compat.py` | 2 | on load, a view with > 2 strips in competing clusters → not ok (efficiency: "hit, no reco") |
| `TAN_SANE`, `P0_SANE_MM` | `wft/compat.py` | 3, [−20, 420] mm | on load, a fit with \|tan\| > 3 or p0 off the plate is not a track → not ok (pass-1 review P1) |
| `BOW_UW_MAX` | `wft/reco.py` | 2.2 | an off-plate view does not set the other view's bow (applies to reconstructions after 2026-10-30) |
| `SAT_SOLVE` | `wft/model.py` | True | the locked simple saturation treatment; `SAT_RAIL` off (K4) |

---

## 6 · The pass registry (`rerun/pass_registry.csv`)

| column | meaning |
|---|---|
| `cls` | `golden_calib` (the chamber's calibration row), `golden_quoted` (largest clean run, candidate for quoting), `consistency`, `skeptical` (angle scale not calibrated or chamber suspect: positions/efficiency only), `excluded` |
| `group` | (det, run, drift V): rows sharing gas, night and drift field |
| `calib_key` | the row whose bundle this row uses |
| `refit` | `golden` = full calibration; `v` = own w0/kw against M3 (≥ `N_OWN_V` = 500 clean triggers; ~0.5 % on kw); `none` = inherits `calib_key`'s bundle |
| `align_key` | the row whose waveform alignment this row uses (§7) |
| `K` | depth bins (§2) |

Golden calibration rows: det2 6-22 longer_run (`o22_long_det2`), det3 6-27 long 490/1000 V (`sat_det3`), det4 6-24 long (`g_det4`),
det6/det7 6-26 long (`g_det6_long`, `g_det7_long`).

---

## 7 · Alignment (bench: detector strip frame ↔ M3 telescope)

`alignment.json` per row (01_alignment). The M3 ray is evaluated at z and compared with the detector position at the nominal plane (p0 − tan · h).

| parameter | meaning |
|---|---|
| `z_x`, `z_y` | height of the detector plane in the M3 frame, per view, from a scan of the residual width over ±60 mm (`Z_HALF_WINDOW`) around the nominal. ~713 mm on the top slot, ~242 mm on the bottom. The width is a fitted Gaussian core, falling back to 1.4826 × MAD (never the RMS) |
| `theta_deg` | rotation between strip frame and M3 (≈ 90°: the strip maps and M3 are swapped) |
| `x_offset`, `y_offset`, `centre_x/y` | translation, about the plate centre (200, 200) mm |
| `ref_x_sign` | +1: handedness of M3 x relative to the strip frame |

Checks (`rerun/pass1_ledger.py`): z within 5 mm of its night's median, z_x ≈ z_y, and reco_far ≤ 15 %.
These caught four rows whose scan had railed to the window edge.

---

## 8 · Analysis outputs (efficiency, resolution)

| quantity | file | definition |
|---|---|---|
| M3 reference track | `qa_config.py` | χ² < 1.0 and NClus = 4 (`M3_CHI2_CUT`, `M3_MIN_NCLUS`; pass explicitly) |
| `within_R` | `efficiency/efficiency_breakdown_cut.json` | % of M3 rays in the active box with an x+y waveform point within R = 5 mm. Denominator `n_rays` |
| `reco_far` / `hit_no_reco` / `no_hit` / `spark_cat` | same | reconstructed but > R away / fired but no accepted fit / nothing fired / discharge (> 50 hits) |
| active box | 02_efficiency | 0.5–99.5 percentiles of the row's own reconstructed positions (so absurd positions widen it, one reason for the sanity cut) |
| `core_sigma_mm` | same | robust width of the radial distance r between waveform point and M3 ray, over r < 15 mm |
| `s68_deg` | `angles/angular_resolution.json` | 68th percentile of \|Δθ − median Δθ\| (Δθ = detector − M3), per view. bow_finalize uses ½(p84 − p16) instead: close, not identical |
| `implied_v` | same | v that would make the angle scale exact, per \|tan_ref\| bin: a check on kw · v |
| SPS `ref_x/ref_y` | events table | det4's own window centroid. On RAW rows the window is mostly noise strips, so it is **not** a position reference. Use the uRWELL telescope or the other view |

---

## 9 · Calibration and pass gates

| gate | criterion |
|---|---|
| convergence (`rerun/calib/convergence.py`) | template L2 < 1 % of peak; latency ±2 ns; D_rc,y ±5 %; w0 ±0.02 µm/ns; kw ±0.5 %; bow h < 0.4 mm, a < 0.002; alignment 0.1 mm / 0.05° (z: one 1 mm step). Failing → one more CAL4 iteration |
| G1 (`rerun/calib/g1_gate.py`) | pass − audit: detection ≥ −0.5 pt, far ≤ +0.3 pt, s68 ≤ +0.03° per view, and converged |
| bow transfer (`rerun/calib/bow/bow_finalize.py --manifest`) | no row ≥ 2000 events worse in angle or position s68 beyond its error |
| ledger (`rerun/pass1_ledger.py`) | no row missing, no overrides, code = archive, alignment within its night, far ≤ 15 % |

---

## 10 · SPS (det4 at H4) specifics

| parameter | where | value | meaning |
|---|---|---|---|
| bundle | `make_pass_dag.py sps` | the pass-1 det4 golden bundle with bench bow, v map, w0 and t0_abs **dropped** | bench geometry and trigger do not transfer |
| v | `sps_registry.csv` `v_span_um_ns` | 11.5–14.0 µm/ns where measured; **34 rows have none** (angle scale uncalibrated: positions only) | |
| K | registry | v_span-based + 15 bins (50–57) | |
| seed window | `sps_build_cache.py` | brightest strip ± 16 strips per view, peak ≥ 300 ADC, ≥ 4 strips with samples | self-seeded, no uRWELL; on low-gain ZS rows (run53-like) the ≥ 4-strip cut keeps only the high-charge tail (`rerun/sps_audit`) |
| seed exclusion | `sps_build_cache.SEED_BAD`, `hot_channels` | ch 372, 510 always; per plateau, a channel seeding > 3 % (`HOT_FRAC`) of the first 2000 (`HOT_SCAN`) events and > 10× (`HOT_RATIO`) the median of its ±4 neighbours | oscillating Y strips (and ch 373, some X) won the seed: 41–44 % of run71 Y fits on ch 372 before 2026-10-30; `sps_reco` censors the hot ones in the fit (`hot_seed_channels` in the meta) |
| ZS rows: zeros | `sps_reco.py` | a suppressed sample = a measured zero (below threshold) | closest to the RAW answer (ws_sps) |
| RAW rows (run71): zeros | `sps_reco.py` | an absent sample = the FEU dropped it (~23 %): censored | |
| RAW baseline | `sps_build_cache._baseline` | per-channel pre-sample median, then the per-sample median of each 64-channel block excluding ±10 strips around each view's lead and channels 510/372 (`RAW_CM_MASK`, `RAW_CM_BAD`), then the pre-sample median again | without the common-mode step every strip carries ~260 ADC (pass-1 review P2) |
| ZS baseline | same | subtract 256 (the FEU's flat ZS offset) | |
