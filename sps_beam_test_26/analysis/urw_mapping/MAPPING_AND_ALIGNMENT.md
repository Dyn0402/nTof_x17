# uRWELL mapping and alignment — the record

Written 2026-07-26T16:00:41 from `highstat_eff_1` (6 sub-runs).

Everything the uRWELL-referenced P2 measurement rests on that is not in the
raw data. The authoritative source is the code — `urw_lib.VIEW_MODE_DEFAULT`,
`urw_lib.AXIS_FLIP_DEFAULT`, and the fits in `urw_p2_efficiency.py` — and this
file is generated from it by `record_mapping_alignment.py`. Regenerate rather
than edit.

| file | what it is |
|---|---|
| `mapping_urwell.csv` | one row per FEU channel: view, local position, pitch, zone. Join on (detector, channel); no code needed. |
| `mapping_alignment.json` | the same numbers, machine readable. |
| this file | the same numbers, with the evidence. |

## 1. Channel → strip wiring

A uRWELL view is read out on **two** 64-channel Dream connectors, so the
wiring has two binary degrees of freedom and four possible answers:

| mode | meaning |
|---|---|
| `AB` | map connector 0 -> lower FEU connector, channel order as-is |
| `BA` | the two connectors of the view interchanged |
| `AB_rev` | as AB, channel order reversed inside each connector |
| `BA_rev` | as BA, channel order reversed inside each connector |

Measured:

| detector | x view | y view | axis flip |
|---|---|---|---|
| `EIC_uRWELL_front` | `BA` | `AB` | none |
| `EIC_uRWELL_back` | `AB_rev` | `AB_rev` | x, y |

**How.** width of (back - front) per candidate wiring, with the front as reference; explore4_back_map.py on highstat_eff_1/beam_commissioning_00. Width of `back − front` per candidate:

| view | `AB` | `BA` | `AB_rev` | `BA_rev` |
|---|---|---|---|---|
| back x | 45.60 mm | 5.42 mm | **0.80 mm** | 47.00 mm |
| back y | 35.20 mm | 6.40 mm | **0.88 mm** | 38.80 mm |

**Mirror ambiguity.** the front's own data cannot separate BA/AB from the mirror partner AB_rev/BA_rev; the choice recorded here is the one that maps the front into the P2 pad frame by a PROPER rotation (det > 0), which a reflection could not do.

**Supersedes.** CONNECTOR_SWAP_DEFAULT (removed 2026-07-26), which had back x and back y as 'BA' and left the back pointing at 4.4 mm.

The axis flip is a pure labelling choice: with the correct wiring the back
reads anti-parallel to the front on both views, so its two axes are mirrored
in software to keep the front→back slope at +1. No strip → position
assignment is touched.

## 2. Front → back alignment

`back = slope * front + offset, per axis, in local mm`, lever arm dz = 1370 mm.

| axis | slope | offset [mm] | core sigma [mm] |
|---|---|---|---|
| x | +0.99960 ± 0.00003 | -0.958 | 0.77 |
| y | +1.04177 ± 0.00003 | -4.107 | 0.72 |

The y slope is **not** a scale error — see §4. The core sigma is the two
planes' resolutions in quadrature plus the beam's angular spread, so each
plane is better than ~0.6 mm.

## 3. uRWELL → P2 pad frame

`(X, Y)_P2 = A @ (x, y)_uRWELL_track_at_z + t, both in mm`

A is a free 2x2 fitted per station per sub_run; the values here are the mean and spread over the sub_runs of the source run. The rigid fit (dx, dy, theta) is reported alongside as the check that A stayed orthogonal.

| station | z [mm] | rotation [deg] | det(A) | stretch xx / yy | shear | rigid rmse [mm] | residual rms x / y [mm] |
|---|---|---|---|---|---|---|---|
| P2_IN | 320 | -59.663 ± 0.006 | +1.0046 | 0.9974 / 1.0072 | -0.0013 | 4.85 | 3.39 / 3.46 |
| P2_MID | 630 | -59.732 ± 0.006 | +1.0123 | 0.9990 / 1.0134 | -0.0012 | 4.83 | 3.37 / 3.44 |
| P2_OUT | 940 | -59.999 ± 0.008 | +1.0225 | 0.9993 / 1.0232 | -0.0008 | 4.87 | 3.39 / 3.45 |

A proper rotation of about −60° with no reflection and no shear at every
station. Because a reflection is geometrically impossible here — the two
detectors are seen from the same side along the same beam — getting an
orthogonal `A` with det = +1 is a real test of both strip maps and the pad
map, not a fit artefact.

The full per-sub-run matrices are in the `frame` block of each entry of the
run's `urw_p2_efficiency_<run>.json`.

## 4. Beam optics

`stretch_ii(z) = 1 + z / L_i, fitted across the three P2 z` — the departure of A from orthogonality; a divergent beam, not a detector distortion - see explore6_divergence.py.

| axis | d(scale)/dz [1/mm] | virtual source [m upstream] |
|---|---|---|
| x | +3.121e-06 | +320.4 |
| y | +2.577e-05 | +38.8 |

The beam diverges in y and is essentially parallel in x. Extrapolating the
y term to dz = 1370 mm reproduces the front→back slope of §2, which is why
that slope is optics rather than a bad pitch. This is also why the applied
uRWELL → P2 transform is the affine and not the rigid one: otherwise a known
±1 mm optical term leaks into the residuals.

## 5. The mapping table

`mapping_urwell.csv` has 512 rows. Columns: `detector`,
`det_type`, `feu`, `channel` (0–511 global), `feu_connector`,
`connector_channel`, `view` (the coordinate the strip MEASURES), `position_mm`
(local, 0 at the low edge), `pitch_mm`, `interpitch_mm`, `zone`, `view_mode`,
`axis_flipped`, `z_mm`.

Zone summary:

| detector | view | zone | pitch [mm] | strips | position range [mm] |
|---|---|---|---|---|---|
| EIC_uRWELL_back | x | 3 | 1.0 | 64 | 64.44 – 127.44 |
| EIC_uRWELL_back | x | 4 | 0.5 | 32 | 0.00 – 15.50 |
| EIC_uRWELL_back | x | 5 | 1.5 | 32 | 16.88 – 63.38 |
| EIC_uRWELL_back | y | 0 | 1.0 | 64 | 64.25 – 127.25 |
| EIC_uRWELL_back | y | 1 | 0.5 | 32 | 0.00 – 15.50 |
| EIC_uRWELL_back | y | 2 | 1.5 | 32 | 16.50 – 63.00 |
| EIC_uRWELL_front | x | 2 | 1.0 | 64 | 0.00 – 63.00 |
| EIC_uRWELL_front | x | 3 | 1.0 | 64 | 64.12 – 127.12 |
| EIC_uRWELL_front | y | 0 | 1.0 | 64 | 0.00 – 63.00 |
| EIC_uRWELL_front | y | 1 | 1.0 | 64 | 64.00 – 127.00 |

> Reminder: in the raw map files `axis` is the direction the strip **runs**,
> so `axis=y` measures x. The `view` column here is already the measured
> coordinate, so no further inversion is needed.

