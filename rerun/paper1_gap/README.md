# The drift gap of chamber A across its surface — pass 1 (Paper I §5.1), 2026-11-23

Report: [`report.html`](report.html) (built by `make_report.py`; PNGs are git-ignored, a copy with figures is on EOS
`/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_gap/report/`). Paper candidate figure: `figures/gap_profiles.pdf`.
Numbers: `results/compare.json`.

## Verdict

- **The gap of chamber A is not uniform.** Three independent measurements show the same pattern, in both golden runs:
  - geometric with the X strips (the S4 estimator, ±tan halves averaged per cell);
  - geometric with the Y strips (peak sample, relative only);
  - angle-scale velocity × column duration, measured per cell (no β, no gap input).
- **Pattern:**
  - Along x the gap rises by 0.5–0.8 mm per 100 mm (X geometric 0.79 ± 0.23, Y geometric 0.49 ± 0.20,
    v×T 0.51 ± 0.05). That is a cathode tilt of about 0.3–0.5° relative to the mesh.
  - Along y the gap is largest near the centre and about 0.5 mm shorter at ±100 mm. That is the shape of a flat cathode
    above the measured dish of the board (fitted scale 0.5–0.8 of the dish depth).
  - Over the interior the gap runs from about 27 to 29 mm. The cell scatter is 0.5–0.7 mm around 28.2 mm.
- **The runs reproduce each other.** Run 1 − Run 2 gives χ² ≈ number of cells for every method.
- **It is the gap, not the gas.** d ln v / d ln T over the cells is −0.50 ± 0.02. A gap change at fixed voltage gives
  −0.41 (β = 0.68), and a gas change gives −1.
- **The chamber value is unchanged.** Run 1 28.17 ± 0.17, Run 2 28.20 ± 0.13 mm (S4: 28.19 ± 0.19).

## The two surfaces and tracking (2026-11-23, Paper I appendix `app:gapmap`)

- **One model for the bow and the gap** (`cellobs.py` → `joint.py`, `results/joint.json`).
  - It uses a quadratic mesh h and a quadratic cathode c, with the gap g = c − h.
  - Fitted per 4×4 cell over both runs:
    - mesh height from the positions (the bow route);
    - the drift tilt w0 = v (∂h + ∂c)/2;
    - the same tilt, timing-free, from the ±tan asymmetry of the geometric gap;
    - G and GY;
    - T ∝ g^(1+β) and v ∝ g^(−β).
  - Each class gets an error floor so that its reduced χ² is 1. The floors are what a quadratic does not describe:
    hp 0.18 mm, w0 0.18 µm/ns, G 0.1 mm, T 1.9 %, v 1.4 %.
- **The board dish is measured twice.**
  - Without positions, the drift tilt (h + c) and the gap (c − h) give a mesh 1.01 / 1.55 mm deep at 150 mm (x / y).
  - The positions give 0.77–0.88 / 1.38–1.47 mm.
  - A free reference-telescope dish fits to −0.25 ± 0.16 / −0.05 ± 0.20 mm, consistent with zero.
- **Cathode along y:** nearly flat in every subset (−0.2 to 0.4 mm at 150 mm), so the gap domes by about 1.2 mm.
- **Cathode along x:** it follows the board, and the gap rises about 0.4 mm per 100 mm.
  - Only the no-timing subset prefers a flat cathode along x (2.5σ). The X geometric gap covers x 100–320 mm only.
  - The w0 ramp along x steepens toward the edges (0.39 against 0.31 µm/ns per 100 mm), so r_f along x reads 1.2. The board
    is not a pure paraboloid.
- **β:** 1.0 ± 0.2 (Magboltz 0.68: Δχ² +5; drift scan 0.87: +0.7). The surfaces do not depend on it.
- **Tracking** (`track_test.py`, `track_grid.py`, cross-validated: the map comes from the other run, only two constants are
  re-measured).
  - A 42 mm-cell map of w0 and v improves the angle s68 by 0.15–0.17° (16–21 %) on X and Y in both runs:
    X 0.905 / 0.929 → 0.736 / 0.776°, Y 0.796 / 0.809 → 0.627 / 0.651°.
  - The smooth two-surface model gains only 0–0.04°. The velocity map alone gains nothing.
  - The structure is 2-D and smooth, with no steps at the 64-strip chip boundaries, so it is the drift field, not chip timing.
  - Not shipped: a production change (a bundle `v_map` of w0, v per chamber and mounting) is Dylan's call. Not tested at n_TOF.

## Limits
- v×T shows the pattern with 0.7 ± 0.13 of the geometric amplitude. The v-only map varies more than a pure gap change
  predicts. So the amplitude is known to about 30 %.
- Structure beyond tilt + dish remains in the timing maps (χ² 64/12 for v×T). Its origin is not identified.
- The geometric methods see only the centre: X strips x ≈ 100–320 mm, Y strips y ≈ 60–290 mm. This is the M3
  acceptance for inclined tracks, not a cut. Outside it, the map rests on timing.
- One usable mounting only. The mean shortfall (1.8 mm below the 30 mm design) is not explained.

## How to run
1. `~/.venvs/xrd/bin/python mirror.py $WORK_DATA/claude-jobs/<job>/mirror` copies the inputs from EOS over xrootd
   (8.2 GB, adler32-checked). It takes the rr1 FEU 7/8 waveforms, the M3 v2 rays, the decode audit, the pass-1
   alignment and bundle, and S4's cache.
2. `build_incl.py <key> <mirror> <work>/build_<key>` for `sat_det3` and `g_det3_wknd` builds the inclined and head-on
   caches through `rerun/inputs.py` + `s4_stacks.build`. The rebuilt sat_det3 cache is identical to S4's. Make empty
   `combined_hits_root_rr1_cm0/` dirs in the mirror: the stager lists them.
3. Copy the pass-1 tables (`bow_gate_table.parquet`, per row) to `<work>/pass1/<run>/<subrun>/`.
   Copy colsum.npz (`x17rerun/s4_inputs/.../fa/colsum.npz` and `x17rerun/followups/gc/.../gc/colsum.npz`) to
   `<work>/colsum/{sat,p2}_colsum.npz`.
4. `gapmap.py <work>` (about 5 min, 14 processes).
5. `gapcompare.py <work>`, then `cellobs.py <work>`, `joint.py <work>`, `track_test.py <work>` and `track_grid.py <work>`.
6. `gapfig.py <work>`, then `make_report.py`.

Run 2's caches and gapmap.json are on EOS (`x17rerun/paper1_gap/caches/g_det3_wknd/`), so steps 1–2 are needed only to
rebuild them.

Supersedes: the July map (`mx_june_wft/GAP_STUDY_2026-07-30.md`, 25.7–29.2 mm, old kernel) and the FA follow-up maps
(`rerun/model_audit/followups/fa_gas_v_tilt`, pre-pass bundles, Run 1 only).
