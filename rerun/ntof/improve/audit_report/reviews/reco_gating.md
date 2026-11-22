<!-- Verbatim report of the 'Review beam reco and gating' code review, part of the 2026-11-21 n_TOF tracking audit (session be7f0e66).
     Scratch scripts it cites under /home/dylan/.claude/jobs/be7f0e66/tmp/ were copied to
     $WORK_DATA/ntof_pass1/ana/tracking_audit/review_scripts/. -->

# Review of the n_TOF pass-1 reconstruction and its `gated` selection (read-only)

**Bottom line.** Before about 12 ms, most gated tracks are reconstruction fakes: post-flash baseline ringing that the gate lets through. After 20 ms the reconstruction explains little of the non-pointing fraction. The largest late item is in stage 3: `dca_axis_mm` is measured about the wrong axis.

All checks use run_145 (three sub-runs), arms A and C. Scratch scripts are in `/home/dylan/.claude/jobs/be7f0e66/tmp/review_reco/` (`c1.py` to `c14.py`). The repo is untouched.

## What makes a track `gated`

- **Reco gate:** both views `plausible`, and the x–y column times agree within `tc_tol` (`_gate`, `wft/reco.py:1716`).
- **`plausible`:** column end `q_uend` inside `u_window`, and raw |tan| < 0.6 (`wft/reco.py:636-644`).
- **Stage 3 adds:** v(t) |tan| < 0.5 in both views, and not `censored_window`.
- **Not in the gate:** χ², Δχ², charge, x–y charge balance, strip count, `quality_ok`, `slope_reliable`, t0.

## Findings, ranked

**1. The gate accepts post-flash baseline ringing. CONFIRMED.** (`wft/reco.py:636-644`, `1716`, `1824-1864`; `ntof_tracking/wft_beam.py:316-326`, `465`, `474`)
- In `cost` mode the pairing cost only chooses partners. No cost, Δχ² or charge-balance threshold rejects a pair.
- I applied an illustrative cut (not tuned): |ln(qx/qy) − median| < 4σ of the late width, and Δχ²/(χ²/dof) > 1000 in both views.

| Window | A kept | A pointing, all → kept | C kept | C pointing, all → kept |
|---|---|---|---|---|
| 1–5 ms | 191 of 3,823 | 0.11 → 0.36 | 287 of 3,947 | 0.09 → 0.27 |
| 5–10 ms | 868 of 4,993 | 0.16 → 0.37 | 1,097 of 3,700 | 0.17 → 0.30 |
| 12–20 ms | 3,350 of 3,768 | 0.50 → 0.54 | 2,819 of 3,090 | 0.48 → 0.50 |
| > 20 ms | 12,575 of 12,881 | 0.60 → 0.61 | 9,963 of 10,351 | 0.57 → 0.58 |

- Pointing here is |miss| < 15 mm about the fitted source. The removed early tracks in A point at 0.09–0.11, which is chance (mixing null 0.08–0.09).
- Early gated pairs in A have corr(ln qx, ln qy) = 0.09 and a ln(qx/qy) width of 0.81. Late pairs have 0.95 and 0.075.
- Raw waveforms, A x, 40 early low-significance gated tracks against 40 late tracks:
  - Early: a median of 86 strips above +5σ and 80 strips below −5σ across the plane, and all strips peak at the same sample.
  - Late: 6.5 strips above +5σ, none below −5σ, and peak times spread over 9 samples.
- Why the seeder passes it: the isochronous veto needs ≥ 10 strips, the busy rule needs > 150 hits, and min strips is 3.

**2. `dca_axis_mm` is measured about global (0, 0), but the source sits at x ≈ −10 mm. CONFIRMED.** (`sept26_prelim_analysis/build_tracks.py:330-354`)
- Late fraction within 15 mm: A 0.521 about (0, 0) against 0.614 about (−10, −4); C 0.511 against 0.586.
- In addition, 6–7 % of late gated tracks sit where pointing would need |tan| > 0.5. They cannot point by construction, and 0.11 of them do.

**3. `quality_ok` is referenced to a median that the ringing sets. CONFIRMED.** (`wft/reco.py:178-208`, `build_tracks.py:412`)
- Sub-run median χ²/dof of rank-0 candidates: A x 7.4, C x 11.5. Late real tracks have 1.42 and 2.09.
- The 5× threshold is therefore 37 (A x) and 58 (C x).
- Early gated tracks pass `quality_ok` in both views at 0.61 (A) and 0.88 (C). Referenced to the late median they would pass at 0.21 and 0.34.

**4. The coincidence and `plausible` tests reject little. CONFIRMED.** (`wft/reco.py:608-617`, `1571`; `rerun/ntof/ntof_bundle.py:108-117`)
- `tc_tol` is the bench value: 197 ns (A) and 456 ns (C).
- Unrelated, plausible x and y candidates from different late events pass the coincidence at 70 % (A) and 89 % (C). Two in-time particles in one chamber are therefore not separated by it.
- The upper column bound is 1.6 T = 1243 ns (A) and 1708 ns (C). The profile grid ends at 1200 ns, so this bound never rejects.
- The lower bound rejects 0.3–5.5 % of candidates. In practice `plausible` is the |tan| < 0.6 cut.
- The pairing cost has three correlated timing terms (`u50`, `u90`, `tc`) and one charge term.

**5. `apply_bow` overwrites a paired track's tangent with the wrong partner's coordinate. CONFIRMED bug.** (`wft/reco.py:2154-2162`)
- After the pairs are mapped, the plane winners are mapped again together. When rank-0 x and rank-0 y belong to different pairs, each gets the other's coordinate.
- Recomputing from `w` and `p0` shows every pair with exactly one rank-0 member has the wrong x or y tangent. Pairs with both or neither are exact.
- That is 19 % (A) and 26 % (C) of all pairings in stat090_0000, 11–13 % of gated tracks over the run, and about 1.4 % of late ones.
- Size: p90 |Δtan x| = 0.011 (A) and 0.063 (C). This matters most for same-chamber pairs.

**6. An out-of-time track is biased, not rejected. CONFIRMED on synthetics with the A bundle; small effect on late data.** (`wft/reco.py:128`, `382`; `wft/model.py:280`, `1435`)
- The prior is weak: σ = 15 ns is 16 χ² units per 60 ns bin, against a data Δχ² of about 10⁴. The fitted t0 − pin has a median of −26 ns and a 16–84 % range of −57 to +5 ns (A x).
- The real constraint is the hard box of pin ± 180 ns.
- Late arrival (+120 to +600 ns): the profile absorbs the delay, the slope is within 5 %, and p0 moves by −w·Δt (up to 9.5 mm at tan 0.45).
- Early arrival (−360 to −480 ns): t0 rails at −180 ns and the slope reads 14–33 % too steep. `plausible` stays true.
- Late data: 94 % (A) and 99 % (C) of late gated tracks have their own wall in the trigger. The miss does not follow tan·v·(u50 shift) in A, and only partly in C (regression slope 0.09 and 0.35). Tracks at the t0 box edge are about 1 %.

**7. Stage 3 never applies the bow mesh height. CONFIRMED by reading.** (`build_tracks.py:203-231`, `256-264`)
- The tangent uses the bow block, but the position stays the raw `p0`. The candidates table has no `bow_h`, so `p0 − tan·h` cannot be formed.
- h is 1.0 mm at the median and 4–7 mm at the 99th percentile. |tan·h| reaches 2.6 mm (A) and 5.1 mm (C) at p95 over all pairs.

**8. Two-track children ignore the t0 pin. CONFIRMED.** (`wft/reco.py:1089-1186`)
- 31 % of gated children have |t0 − pin| > 180 ns. Children are about 1 % of late gated tracks and 4–10 % of early ones.
- `split_child` is not carried into stage 3.

**9. C x windows with many censored strips. CONFIRMED, small.** (`rerun/ntof/stage3/censor_join.py:40-45`)
- Late C tracks with 30–99 % of the x window censored: 43 tracks, none point.
- With 10–30 % censored: 917 tracks at 0.50, against 0.58 with none. The rule only removes x windows that are fully censored.

**10. Seeding keeps the five widest clusters, not the most significant. PLAUSIBLE.** (`wft/seed.py:351`, `369`)
- 12–15 % of early planes reach the cap of five. A narrow real track can lose its place to wider ringing clusters.

**11. The head-on rescue paths are not wired into the beam driver. CONFIRMED by reading.** (`wft/reco.py:2096-2107`, `wft/seed.py:496-499`)
- `wft_beam` never supplies `_rescue_` windows and never calls `headon_seeds`.
- C has 957 late tracks at |tan x| < 0.08 against A's 2,178. The pointing width there is 22–32 mm in C against 17 mm elsewhere.

**12. Smaller points.**
- The events table pairs the two plane winners. They are not track 0 in 24 % (A) and 31 % (C) of paired events (stat090_0000).
- `tan_err` sits at the floor (median equals `tan_floor`), so it carries no per-track information.
- `cal.dt_xy.get(..., -18.8)` reads bench keys (`wft/reco.py:1670`). It is inert today because no active feature uses it.

## Checked and found correct

- **p0 reference and signs:** pointing tracks follow x_local = x0 + 234.2·tan x (A) and 233.1·tan x (C), correlation 0.99. That is the strip-plane distance (234.6 mm), not mid-gap (219.6 mm). `local_and_global` treats x and y consistently.
- **Mirror-slope fits:** no peak, below 1 % (0.6–0.9 % against neighbouring bins).
- **Window clipping:** synthetic windows covering only the central 30–60 % of a column give tan within 0.004. On data the window is narrower than the column in 1.1–1.7 % of late tracks.
- **Plane edge:** 0.3–0.4 % of late tracks lie within 10 mm of it.
- **Near-normal incidence in A:** no loss of pointing (width 17.7 mm).
- **Saturation:** 0.3–0.8 % of late gated pairs have a saturated strip.
- **Censoring in the χ²:** dead strips leave the fit and the dof with noise 10⁹; saturated samples are one-sided lower bounds.
- **t0 prior lookup:** active for ftst 0–5, which is all the data has.
- **Candidate ranking:** stable, and `best` equals `ranked[0]`.
- **`track_id` and `rank` indexing after splits:** consistent between `select_tracks` and `candidate_rows`.
- **Bench pairing constants at n_TOF:** late ln(qx/qy) width 0.073 (A) and 0.126 (C), close to the bench values.

## Not explained

The late miss has a 68 % width of 17 mm, flat in angle and in every fit variable. Cosmic angular resolution plus the capsule radius would give about 9 mm. Nothing I tested in the reconstruction produces the difference, and I did not test multiple scattering.
