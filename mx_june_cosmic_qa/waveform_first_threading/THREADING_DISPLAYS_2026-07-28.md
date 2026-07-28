# Do M3 tracks thread the reconstructed cluster better? — det3 event displays

**2026-07-28.** Answers the display question left open by
`REFERENCE_TRACK_THREADING_REPORT.md` §0 ("the reference agrees at the mesh but
fans away with depth") on the waveform-first reconstruction of
`WAVEFORM_FIRST_THREADING.md`.

- Script: `37_threading_displays.py`
- Data: `sat_det3` (`mx17_det3_saturday_scan_6-27-26 /
  long_run_resist_490V_drift_1000V`), the same `wfcache.pkl` as the rest of this
  directory (6,317 matched muons, both planes, M3 radial residual < 10 mm)
- Outputs: `<Analysis>/mx17_3/waveform_first/threading_displays/`
  — `event_<eid>_planes.png` (12 events, X and Y), `event_<eid>_3d.png`,
  `threading_census.png/.json/.csv` (600 events)

---

## 1. The circularity problem, and how the displays avoid it

The forward fit returns a *line*. Drawing its own charge profile along that line
and observing that the M3 track lies on top of it would prove nothing. So the
cluster in these displays is never taken from the fit. Two line-free estimators
are used instead, both built only from the calibrated impulse template and the
resistive-sharing kernel (scripts 03/11/13 products):

1. **2-D deconvolution** (`deconv2d`) — charge `Q[strip j, depth bin k] >= 0`
   with

   ```
   data(i,t) = sum_{j,k} Q[j,k] . [ h0(t-t0-u_k)          i = j
                                  + c1 h1(t-t0-u_k-tau)   |i-j| = 1
                                  + c2 h2(t-t0-u_k-2tau)  |i-j| = 2 ]
   ```

   solved by NNLS with a second-difference Tikhonov penalty along depth. It is
   drawn as the blue density. No track, no slope, no position model enters.

2. **Free ladder** (`free_ladder`) — one *continuous* transverse position
   `mu_k` and one charge `q_k` per 60 ns depth bin, again with no relation
   imposed between bins. This exists because the deconvolution's strip-wise
   centroid quantises to the 0.78 mm pitch and collapses onto one strip when the
   cloud is narrower than a pitch (visible in event 11211 Y). Solved by exact
   coordinate descent: for a trial `mu_k` the optimal `q_k` is a one-line
   non-negative least-squares update, so a 40 µm position grid is cheap. This is
   the estimator behind the quoted numbers — it is the natural generalisation of
   the production ladder (one point per *strip*, sharing left in) to one point
   per *depth bin* with the sharing removed.

The only quantity taken from the forward fit is `t0`, the arrival time of charge
from the mesh, which fixes the z origin. It is a common offset — it moves both
clusters together, not one relative to the other.

**The production cluster** on the same axes is what the analysis actually fits:
hits past the per-plane relative significance floor (0.10, the 7-25 fix),
spatially clustered with the production gap threshold, largest cluster kept
(`cosmic_micro_tpc_analysis._fit_single_axis`), each strip placed at the depth
implied by its aggregate hit time. Both clusters use the same z origin and the
same drift velocity, so the panels differ *only* in how the charge is timed.

**Metric:** charge-weighted median |cluster − M3 line| measured horizontally,
over the 0–29 mm gap and in depth slices. It contains the M3 pointing error
(σ ≈ 0.4 mm at the DUT, per `m3_self_resolution`) as an irreducible floor —
which is why per-event values below ~0.3 mm are not meaningful and events with a
large M3 residual (e.g. 40073, 2.0–2.3 mm in both methods) fail for both.

## 2. Result: yes, and specifically at depth

600 random events, |tan θ| in 0.08–0.45:

| charge-weighted median &#124;dev&#124; to the M3 line | X | Y |
|---|---|---|
| production cluster | 0.49 mm | 0.55 mm |
| production, own t0 and v (what the existing 3-D displays draw) | 0.60 mm | 0.63 mm |
| **waveform-first cluster** | **0.44 mm** | **0.39 mm** |
| production, above 15 mm depth | 0.64 mm | 0.70 mm |
| production own frame, above 15 mm depth | 0.98 mm | 0.91 mm |
| **waveform-first, above 15 mm depth** | **0.52 mm** | **0.51 mm** |

Deviation vs depth — the actual claim under test:

| depth [mm] | 3 | 9 | 15 | 21 | 26.5 |
|---|---|---|---|---|---|
| X production | 0.45 | 0.50 | 0.61 | 0.67 | 0.63 |
| X production own frame | 0.43 | 0.63 | 0.88 | 1.07 | 1.17 |
| X waveform-first | 0.39 | 0.44 | 0.49 | 0.50 | 0.55 |
| Y production | 0.44 | 0.61 | 0.65 | 0.70 | 0.66 |
| Y production own frame | 0.47 | 0.67 | 0.90 | 0.94 | 0.92 |
| Y waveform-first | 0.37 | 0.40 | 0.41 | 0.49 | 0.57 |

Both methods agree at the mesh (0.4–0.47 mm ≈ the M3 pointing floor). The
production cluster then walks away with depth — by 1.5× in the common frame and
by **2.5×** in its own frame — while the waveform-first cluster stays within
0.4 → 0.55 mm across the whole gap. That is the "fans away as it ascends"
effect, measured, and it is largely removed.

The residual rise of the waveform-first cluster in the last slice (24–29 mm) is
expected: per §21 of the main report the local charge column ends between 26.5
and 29.5 mm depending on position, so part of that slice is past the cathode for
many tracks.

**Robustness.** Production is not handicapped by the common frame: in its own
convention (earliest hit as z = 0, its own calibrated v = 31.5/33.5 µm/ns) it is
*worse*, not better, because the low calibrated v compresses the cloud against a
reference line that is anchored at the mesh. Both variants are plotted.

## 3. What the individual events show

Twelve events spanning |tan θ| 0.13–0.33 (35th–96th percentile). The clearest
single illustration is **16786 X**: the production ladder reads tan +0.121
against the reference's +0.072, its four hit points fan to the right of the M3
line (0.59 mm), while the deconvolved cluster tracks the line to 0.16 mm and the
forward fit returns +0.062. Event **4421** shows the same in Y (production
−0.317 vs reference −0.222).

Honest counter-examples are in the set and worth keeping:

- **11211 Y** — the charge sits on ~2 strips, the deconvolution centroid
  collapses to one strip, and even the free ladder reads 1.2 mm off while the
  production per-strip points follow the reference better (0.80 mm). Most of
  that offset is a whole-cluster shift, i.e. the M3 pointing error for this
  event (radial residual 0.99 mm), not a fan.
- **40073 Y** — 2.0/2.3 mm for both methods: a bad reference, not a bad
  reconstruction.
- Per-event, the waveform-first cluster is *not* uniformly better (see the
  printed table in the script log): 13 of the 24 plane-fits improve; averaging
  the two planes, 8 of the 12 events improve, one (26977) is clearly worse and
  the rest are a wash. The population statistics are where the claim lives.

## 4. Reproduction

```bash
cd mx_june_cosmic_qa/waveform_first_threading
../../.venv/bin/python 37_threading_displays.py                 # 6 displays + 300-event census
../../.venv/bin/python 37_threading_displays.py --n 12 --census 600 --jobs 12
../../.venv/bin/python 37_threading_displays.py --eids 16786,4421 --census 0
```

~4 s per event (both planes: forward fit + deconvolution + free ladder); the
census parallelises over `--jobs`.

## 5. Open items

1. The free ladder is a *display/diagnostic* estimator, not a production one: it
   has no continuity prior, so single bins can wander (visible in the deepest
   bins of some events). A track-free but smoothness-regularised version would
   be the right thing if this is ever used for physics rather than for showing.
2. Only det3 (`sat_det3`) is done here. det2 and det4 have calibration bundles
   from scripts 26/26b and would run with a `--calib` pointer; the script
   currently hardwires `forward_model2.BASE`.
3. `V_PROD = {'x': 31.5, 'y': 33.5}` for the own-frame check is read off
   `alignment_tpc_veto50/angular_resolution.json` by hand; it should be loaded.
