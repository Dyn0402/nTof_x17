# Audit F: statistical audit of the joint mesh + cathode fit (2026-11-24)

Scripts (all new, in `rerun/paper1_gap/audit/`): `f_lib.py` (vectorised re-implementation of `joint.Model`, with options for extra terms, mixing weight a, constraints, cell weights, 2x2 covariance), `f_01_repro.py`, `f_02_runs_resample.py`, `f_02b_timing.py`, `f_03_report_resample.py`, `f_04_order.py`, `f_04b_depth.py`, `f_05_mix.py`, `f_06_hyp.py`, `f_07_pins.py`, `f_08_decomp.py`, `f_09_cov.py`, `f_10_final.py`, `f_11_nohp.py`.
Figures (this directory): `jackknife_influence.png`, `chi2_vs_a.png`. Raw pickles: `resample.pkl`, `order.pkl`, `mix.pkl`, ...

Notation: h_kuu etc. are the quadratic coefficients (mm per (100 mm)^2 /… as in `joint.py`); "depth at 150 mm" = 1.125 k (even part, quartic terms included where fitted); gap tilt in mm per 100 mm. "Fit error" is the least-squares error with the committed floors; "resampling error" is the jackknife / bootstrap over the 16 cells (a cell is dropped in both runs and all classes).

## 0. Reproduction (done first)
All seven arms (all, tilt, gap, geo, timing, geo_gap, no_hp) are reproduced from `$WORK/cellobs.json` with the committed floors: chi2 identical to 0.01, parameters to < 1e-3 (max difference 1.4e-4 in `timing`, 3e-7 elsewhere). Re-running the floor iteration of `joint.main` from zero returns the committed floors to 4 decimals. All numbers below use my vectorised copy, which gives the same fits.

Facts about the inputs that matter later (from `cellobs.json`):
- The X geometric gap G exists in only **8 of 16 cells per run, and only in the two middle columns (U = -0.43 and +0.42, i.e. x = 157 and 242 mm)**. The timing-free tilt dX has the same coverage. G therefore cannot constrain a curvature along U at all, only a slope.
- GY (and dY) exist in 3 cells of run 1 (one W row) and 12 cells of run 2 (all four U values).
- The per-run constants Wx, Wy absorb the mean w0, so **w0 constrains the U-dependence of the tilt, not its mean**. The absolute gap tilt comes only from dX/dY (timing-free) and from T and v. In my `hp + w0` fit the quoted gap-tilt error of 0.04 is a degenerate direction (pinv gives zero error); it is not a measurement.
- In the `timing` arm (hp, w0, T, v) g0 is degenerate with the per-run offsets of ln T and ln v. Under resampling g0 wanders and the gap tilt blows up (jackknife error 72). I therefore re-ran that arm with g0 fixed at 28.2 (section 2b).

## 1. Per-run fits


**all**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 119/114 | +0.80+-0.11 | +1.41+-0.14 | +1.12+-0.15 | +0.32+-0.21 | +0.31+-0.15 | -1.09+-0.26 | +0.37+-0.10 | -0.32+-0.09 | +1.06+-0.30 |
| run2 | 162/132 | +0.76+-0.09 | +1.24+-0.11 | +1.00+-0.13 | +0.15+-0.18 | +0.25+-0.13 | -1.09+-0.20 | +0.39+-0.08 | -0.31+-0.07 | +0.99+-0.23 |
| run1+run2 | 285/259 | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | -1.09+-0.16 | +0.39+-0.06 | -0.32+-0.05 | +1.02+-0.18 |
| (run1-run2)/err | | +0.3 | +1.0 | +0.6 | +0.6 | +0.3 | +0.0 | -0.2 | -0.1 | +0.2 |

**tilt**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 71/61 | +0.71+-0.15 | +1.48+-0.19 | +1.34+-0.30 | +0.21+-0.35 | +0.62+-0.40 | -1.27+-0.48 | +0.54+-0.53 | +0.99+-1.11 | +0.68+-0.00 |
| run2 | 84/70 | +0.66+-0.11 | +1.18+-0.14 | +1.34+-0.28 | +0.41+-0.30 | +0.68+-0.34 | -0.76+-0.39 | +0.36+-0.44 | +0.40+-0.45 | +0.68+-0.00 |
| run1+run2 | 158/141 | +0.68+-0.09 | +1.28+-0.11 | +1.35+-0.20 | +0.35+-0.23 | +0.67+-0.26 | -0.94+-0.30 | +0.44+-0.34 | +0.48+-0.42 | +0.68+-0.00 |
| (run1-run2)/err | | +0.3 | +1.3 | -0.0 | -0.4 | -0.1 | -0.8 | +0.3 | +0.5 | +nan |

**gap**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 76/73 | +0.71+-0.15 | +1.48+-0.19 | +0.96+-0.22 | +0.51+-0.35 | +0.25+-0.15 | -0.96+-0.30 | +0.33+-0.11 | -0.30+-0.10 | +1.21+-0.43 |
| run2 | 105/82 | +0.66+-0.11 | +1.18+-0.14 | +0.84+-0.19 | -0.02+-0.28 | +0.18+-0.15 | -1.19+-0.24 | +0.42+-0.08 | -0.34+-0.08 | +0.91+-0.22 |
| run1+run2 | 184/168 | +0.68+-0.09 | +1.28+-0.11 | +0.90+-0.14 | +0.16+-0.22 | +0.22+-0.11 | -1.12+-0.19 | +0.39+-0.07 | -0.33+-0.06 | +1.00+-0.20 |
| (run1-run2)/err | | +0.3 | +1.3 | +0.4 | +1.2 | +0.3 | +0.6 | -0.6 | +0.3 | +0.6 |

**geo**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 43/40 | +0.72+-0.15 | +1.45+-0.19 | +0.61+-0.99 | +0.25+-0.68 | -0.11+-1.01 | -1.21+-0.66 | +0.64+-0.33 | -0.27+-0.22 | +0.68+-0.00 |
| run2 | 68/58 | +0.66+-0.11 | +1.12+-0.13 | -0.17+-0.45 | -0.31+-0.32 | -0.83+-0.45 | -1.43+-0.32 | +0.54+-0.17 | -0.41+-0.13 | +0.68+-0.00 |
| run1+run2 | 115/110 | +0.69+-0.09 | +1.23+-0.11 | +0.01+-0.41 | -0.19+-0.29 | -0.68+-0.41 | -1.42+-0.28 | +0.58+-0.15 | -0.37+-0.11 | +0.68+-0.00 |
| (run1-run2)/err | | +0.3 | +1.4 | +0.7 | +0.7 | +0.7 | +0.3 | +0.3 | +0.6 | +nan |

**timing**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 205/93 | +0.71+-0.15 | +1.48+-0.19 | +1.35+-0.31 | +0.21+-0.35 | +0.64+-0.40 | -1.27+-0.48 | -0.32+-0.07 | -33.51+-0.07 | +0.68+-0.00 |
| run2 | 114/93 | +0.77+-0.09 | +1.23+-0.14 | +1.04+-0.15 | +0.40+-0.31 | +0.27+-0.16 | -0.83+-0.38 | +0.30+-0.15 | -0.25+-0.13 | +1.44+-0.66 |
| run1+run2 | 221/198 | +0.77+-0.07 | +1.33+-0.11 | +1.11+-0.12 | +0.30+-0.22 | +0.33+-0.13 | -1.03+-0.29 | +0.37+-0.12 | -0.32+-0.11 | +1.42+-0.47 |
| (run1-run2)/err | | -0.3 | +1.1 | +0.9 | -0.4 | +0.9 | -0.7 | -3.8 | -223.8 | -1.1 |

**no_hp**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 87/84 | +0.90+-0.15 | +1.33+-0.21 | +1.15+-0.15 | +0.36+-0.21 | +0.25+-0.15 | -0.96+-0.30 | +0.33+-0.11 | -0.30+-0.10 | +1.21+-0.43 |
| run2 | 126/102 | +0.91+-0.15 | +1.39+-0.18 | +1.09+-0.15 | +0.20+-0.18 | +0.18+-0.15 | -1.19+-0.24 | +0.42+-0.08 | -0.34+-0.08 | +0.91+-0.22 |
| run1+run2 | 215/199 | +0.90+-0.11 | +1.38+-0.14 | +1.12+-0.11 | +0.25+-0.14 | +0.22+-0.11 | -1.12+-0.19 | +0.39+-0.07 | -0.33+-0.06 | +1.00+-0.20 |
| (run1-run2)/err | | -0.1 | -0.2 | +0.3 | +0.6 | +0.3 | +0.6 | -0.6 | +0.3 | +0.6 |

**geo_gap**

| fit | chi2/ndf | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | gap_kww | gtilt_U | gtilt_W | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 33/29 | +0.71+-0.15 | +1.48+-0.19 | -0.36+-1.74 | +0.44+-0.70 | -1.07+-1.73 | -1.04+-0.67 | +0.70+-0.45 | -0.28+-0.23 | +0.68+-0.00 |
| run2 | 48/38 | +0.66+-0.11 | +1.18+-0.14 | -0.21+-0.52 | -0.11+-0.39 | -0.87+-0.50 | -1.29+-0.36 | +0.61+-0.19 | -0.44+-0.13 | +0.68+-0.00 |
| run1+run2 | 84/79 | +0.68+-0.09 | +1.28+-0.11 | -0.16+-0.48 | +0.05+-0.33 | -0.83+-0.47 | -1.23+-0.31 | +0.61+-0.17 | -0.40+-0.11 | +0.68+-0.00 |
| (run1-run2)/err | | +0.3 | +1.3 | -0.1 | +0.7 | -0.1 | +0.3 | +0.2 | +0.6 | +nan |


Reading: in every arm the two runs agree within 1.4 sigma for every curvature, the gap tilt and beta (largest: h_kww, +1.0 to +1.4 sigma, always run 1 higher, and the `timing` arm in run 1 alone, which is degenerate, see section 0). The two runs share chamber and M3, so this is a statistical check only and it passes. Run 1 alone has 3 GY cells, so it constrains c_kww and the Y tilt poorly (c_kww 0.32 +- 0.21 against 0.15 +- 0.18).

## 2. Resampling errors (jackknife over 16 cells, bootstrap 400 x over cells, leave-one-row/column)
### 2. 

**all** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.78 | 0.07 | 0.16 | 0.16 | 0.15 | +0.74..+0.81 | +0.66..+0.92 |
| h_kww | +1.31 | 0.09 | 0.25 | 0.22 | 0.22 | +1.10..+1.51 | +1.30..+1.33 |
| c_kuu | +1.06 | 0.10 | 0.23 | 0.22 | 0.21 | +0.78..+1.20 | +0.84..+1.23 |
| c_kww | +0.22 | 0.13 | 0.34 | 0.32 | 0.30 | +0.05..+0.53 | +0.02..+0.35 |
| gap_kuu | +0.28 | 0.10 | 0.27 | 0.26 | 0.25 | +0.03..+0.46 | +0.17..+0.38 |
| gap_kww | -1.09 | 0.16 | 0.32 | 0.31 | 0.30 | -1.15..-0.92 | -1.31..-0.95 |
| gtilt_U | +0.39 | 0.06 | 0.16 | 0.14 | 0.14 | +0.33..+0.50 | +0.28..+0.58 |
| gtilt_W | -0.32 | 0.05 | 0.13 | 0.15 | 0.10 | -0.45..-0.18 | -0.38..-0.28 |
| beta | +1.02 | 0.18 | 0.26 | 0.25 | 0.26 | +0.94..+1.21 | +0.81..+1.20 |

**tilt** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.68 | 0.09 | 0.19 | 0.20 | 0.17 | +0.52..+0.84 | +0.51..+0.84 |
| h_kww | +1.28 | 0.11 | 0.23 | 0.25 | 0.23 | +1.08..+1.46 | +1.17..+1.40 |
| c_kuu | +1.35 | 0.20 | 0.19 | 0.24 | 0.20 | +1.06..+1.55 | +1.28..+1.48 |
| c_kww | +0.35 | 0.23 | 0.35 | 0.36 | 0.34 | +0.14..+0.65 | +0.19..+0.48 |
| gap_kuu | +0.67 | 0.26 | 0.29 | 0.38 | 0.30 | +0.22..+0.93 | +0.47..+0.78 |
| gap_kww | -0.94 | 0.30 | 0.33 | 0.43 | 0.37 | -0.99..-0.78 | -1.22..-0.76 |
| gtilt_U | +0.44 | 0.34 | 0.52 | 0.46 | 0.44 | +0.01..+0.97 | +0.28..+0.60 |
| gtilt_W | +0.48 | 0.42 | 0.41 | 0.40 | 0.36 | +0.23..+0.89 | +0.29..+0.81 |
| beta | +0.68 | 0.00 | 0.00 | 0.00 | 0.00 | +0.68..+0.68 | +0.68..+0.68 |

**gap** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.68 | 0.09 | 0.19 | 0.19 | 0.18 | +0.52..+0.84 | +0.51..+0.84 |
| h_kww | +1.28 | 0.11 | 0.23 | 0.23 | 0.22 | +1.08..+1.46 | +1.17..+1.40 |
| c_kuu | +0.90 | 0.14 | 0.35 | 0.40 | 0.35 | +0.44..+1.37 | +0.60..+1.08 |
| c_kww | +0.16 | 0.22 | 0.54 | 0.48 | 0.47 | +0.02..+0.44 | -0.27..+0.46 |
| gap_kuu | +0.22 | 0.11 | 0.30 | 0.33 | 0.28 | -0.08..+0.53 | +0.09..+0.31 |
| gap_kww | -1.12 | 0.19 | 0.46 | 0.41 | 0.40 | -1.22..-0.98 | -1.44..-0.87 |
| gtilt_U | +0.39 | 0.07 | 0.18 | 0.18 | 0.17 | +0.33..+0.55 | +0.26..+0.59 |
| gtilt_W | -0.33 | 0.06 | 0.14 | 0.17 | 0.13 | -0.47..-0.19 | -0.45..-0.27 |
| beta | +1.00 | 0.20 | 0.33 | 0.47 | 0.23 | +0.93..+1.15 | +0.79..+1.33 |

**geo** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.69 | 0.09 | 0.19 | 0.19 | 0.17 | +0.63..+0.76 | +0.50..+0.86 |
| h_kww | +1.23 | 0.11 | 0.23 | 0.24 | 0.22 | +0.99..+1.42 | +1.19..+1.25 |
| c_kuu | +0.01 | 0.41 | 0.50 | 0.43 | 0.40 | -0.19..+0.32 | -0.29..+0.28 |
| c_kww | -0.19 | 0.29 | 0.53 | 0.51 | 0.45 | -0.48..+0.14 | -0.61..+0.06 |
| gap_kuu | -0.68 | 0.41 | 0.49 | 0.43 | 0.41 | -0.81..-0.42 | -0.79..-0.58 |
| gap_kww | -1.42 | 0.28 | 0.46 | 0.49 | 0.39 | -1.58..-1.25 | -1.80..-1.18 |
| gtilt_U | +0.58 | 0.15 | 0.20 | 0.20 | 0.19 | +0.59..+0.67 | +0.39..+0.74 |
| gtilt_W | -0.37 | 0.11 | 0.18 | 0.24 | 0.18 | -0.53..-0.02 | -0.57..-0.34 |
| beta | +0.68 | 0.00 | 0.00 | 0.00 | 0.00 | +0.68..+0.68 | +0.68..+0.68 |

**timing** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.77 | 0.07 | 0.25 | 0.19 | 0.18 | +0.62..+0.84 | +0.51..+0.92 |
| h_kww | +1.33 | 0.11 | 0.23 | 0.24 | 0.22 | +1.08..+1.48 | +1.24..+1.45 |
| c_kuu | +1.11 | 0.12 | 0.54 | 0.25 | 0.24 | +0.88..+1.55 | +1.10..+1.40 |
| c_kww | +0.30 | 0.22 | 0.40 | 0.34 | 0.32 | +0.06..+0.73 | +0.16..+0.51 |
| gap_kuu | +0.33 | 0.13 | 0.73 | 0.37 | 0.36 | +0.15..+0.93 | +0.19..+0.87 |
| gap_kww | -1.03 | 0.29 | 0.39 | 0.40 | 0.35 | -1.10..-0.69 | -1.29..-0.73 |
| gtilt_U | +0.37 | 0.12 | 72.34 | 16.45 | 13.91 | -2.48..+17.60 | -9.24..+3.43 |
| gtilt_W | -0.32 | 0.11 | 89.03 | 20.75 | 21.05 | -0.27..+47.91 | -34.93..+31.08 |
| beta | +1.42 | 0.47 | 1.33 | 0.42 | 0.37 | +0.66..+1.58 | +0.67..+1.33 |

**no_hp** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.90 | 0.11 | 0.19 | 0.17 | 0.16 | +0.68..+1.01 | +0.79..+1.02 |
| h_kww | +1.38 | 0.14 | 0.36 | 0.30 | 0.28 | +1.14..+1.60 | +1.27..+1.49 |
| c_kuu | +1.12 | 0.11 | 0.20 | 0.19 | 0.17 | +0.82..+1.22 | +0.94..+1.28 |
| c_kww | +0.25 | 0.14 | 0.32 | 0.29 | 0.27 | +0.08..+0.54 | +0.00..+0.39 |
| gap_kuu | +0.22 | 0.11 | 0.30 | 0.29 | 0.26 | -0.08..+0.53 | +0.09..+0.31 |
| gap_kww | -1.12 | 0.19 | 0.46 | 0.41 | 0.40 | -1.22..-0.98 | -1.44..-0.87 |
| gtilt_U | +0.39 | 0.07 | 0.18 | 0.16 | 0.16 | +0.33..+0.55 | +0.26..+0.59 |
| gtilt_W | -0.33 | 0.06 | 0.14 | 0.15 | 0.13 | -0.47..-0.19 | -0.45..-0.27 |
| beta | +1.00 | 0.20 | 0.33 | 0.48 | 0.24 | +0.93..+1.15 | +0.79..+1.33 |

**geo_gap** (bootstrap n=400)

| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |
|---|---|---|---|---|---|---|---|
| h_kuu | +0.68 | 0.09 | 0.19 | 0.20 | 0.19 | +0.52..+0.84 | +0.51..+0.84 |
| h_kww | +1.28 | 0.11 | 0.23 | 0.27 | 0.25 | +1.08..+1.46 | +1.17..+1.40 |
| c_kuu | -0.16 | 0.48 | 0.65 | 5.35 | 0.56 | -0.86..+0.58 | -0.29..-0.01 |
| c_kww | +0.05 | 0.33 | 0.75 | 7.31 | 0.70 | -0.30..+0.29 | -0.68..+0.94 |
| gap_kuu | -0.83 | 0.47 | 0.65 | 5.38 | 0.54 | -1.38..-0.26 | -1.03..-0.74 |
| gap_kww | -1.23 | 0.31 | 0.66 | 7.31 | 0.65 | -1.42..-1.03 | -1.85..-0.46 |
| gtilt_U | +0.61 | 0.17 | 0.28 | 2.31 | 0.30 | +0.70..+0.82 | +0.43..+0.90 |
| gtilt_W | -0.40 | 0.11 | 0.21 | 4.53 | 0.25 | -0.71..-0.00 | -0.69..-0.33 |
| beta | +0.68 | 0.00 | 0.00 | 0.00 | 0.00 | +0.68..+0.68 | +0.68..+0.68 |

**Leave-one-cell-out influence on c_kuu (value when the cell is dropped; full fit in header)**

all: full +1.06

| i\j | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|
| i=0 | +1.02 (-0.04) | +1.04 (-0.02) | +1.03 (-0.03) | +1.03 (-0.02) |
| i=1 | +1.08 (+0.02) | +1.01 (-0.05) | +1.06 (+0.00) | +1.08 (+0.03) |
| i=2 | +1.12 (+0.06) | +1.11 (+0.06) | +0.99 (-0.07) | +1.07 (+0.02) |
| i=3 | +1.22 (+0.17) | +1.06 (+0.00) | +0.95 (-0.10) | +1.06 (+0.00) |

tilt: full +1.35

| i\j | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|
| i=0 | +1.30 (-0.04) | +1.31 (-0.03) | +1.31 (-0.03) | +1.32 (-0.02) |
| i=1 | +1.39 (+0.04) | +1.40 (+0.06) | +1.36 (+0.02) | +1.36 (+0.01) |
| i=2 | +1.36 (+0.02) | +1.32 (-0.02) | +1.34 (-0.01) | +1.39 (+0.05) |
| i=3 | +1.41 (+0.07) | +1.35 (+0.00) | +1.34 (-0.00) | +1.20 (-0.14) |

geo: full +0.01

| i\j | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|
| i=0 | -0.26 (-0.27) | +0.18 (+0.17) | +0.01 (-0.00) | +0.06 (+0.05) |
| i=1 | -0.02 (-0.04) | -0.11 (-0.12) | -0.03 (-0.04) | +0.08 (+0.07) |
| i=2 | +0.20 (+0.19) | -0.02 (-0.03) | -0.14 (-0.15) | +0.16 (+0.15) |
| i=3 | +0.24 (+0.23) | -0.00 (-0.01) | -0.06 (-0.07) | +0.04 (+0.02) |

gap: full +0.90

| i\j | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|
| i=0 | +0.86 (-0.03) | +0.90 (-0.00) | +0.88 (-0.02) | +0.87 (-0.03) |
| i=1 | +0.90 (-0.00) | +0.79 (-0.11) | +0.88 (-0.02) | +0.95 (+0.05) |
| i=2 | +0.99 (+0.09) | +0.99 (+0.09) | +0.79 (-0.11) | +0.88 (-0.02) |
| i=3 | +1.13 (+0.23) | +0.91 (+0.01) | +0.73 (-0.17) | +0.98 (+0.08) |


### 2b. `timing` arm with g0 fixed
**timing, g0 fixed at 28.2 (T and v alone cannot fix g0: it is degenerate with the per-run offsets)**

| quantity | full | fit err | jackknife | bootstrap std | boot 16-84 hw |
|---|---|---|---|---|---|
| h_kuu | +0.79 | 0.07 | 0.15 | 0.15 | 0.15 |
| h_kww | +1.30 | 0.09 | 0.23 | 0.21 | 0.21 |
| c_kuu | +1.07 | 0.10 | 0.25 | 0.23 | 0.21 |
| c_kww | +0.40 | 0.16 | 0.31 | 0.28 | 0.28 |
| gap_kuu | +0.28 | 0.10 | 0.26 | 0.25 | 0.24 |
| gap_kww | -0.90 | 0.19 | 0.20 | 0.24 | 0.23 |
| gtilt_U | +0.31 | 0.07 | 0.14 | 0.13 | 0.12 |
| gtilt_W | -0.27 | 0.07 | 0.13 | 0.12 | 0.10 |
| beta | +1.26 | 0.32 | 0.31 | 235.09 | 0.29 |

Notes:
- The `geo_gap` bootstrap std (5.4, 7.3) is a few degenerate resamples; its 16-84 half-width (0.56, 0.70) and jackknife (0.65, 0.75) are the usable numbers. The bootstrap std of beta in `timing` is the same artefact (235), the half-width is 0.29.
- **Resampling errors are 2 to 3 times the fit errors for the surface curvatures in `all`**: h_kuu 0.16 vs 0.07, h_kww 0.25 vs 0.09, c_kuu 0.23 vs 0.10, c_kww 0.34 vs 0.13, gap_kuu 0.27 vs 0.10, gap_kww 0.32 vs 0.16, beta 0.26 vs 0.18. The floors, tuned to chi2/n = 1 per class, treat the cell scatter as independent; the real scatter is shared between classes (the same cell is high in hp, w0 and v), so the fit error is too small by this factor. For `tilt` the two agree (c_kuu 0.20 fit, 0.19-0.24 resampled) because that arm has almost no redundancy to be fooled by.
- **Which cells drive c_kuu**: none. In `all` the largest single-cell change is cell (i=3, j=0): c_kuu is 1.22 when it is removed (+0.17); removing (3,2) gives 0.95 (-0.10); the spread over 16 cells is +- 0.23. Leave-one-column-out gives 0.84-1.23, leave-one-row-out 0.78-1.20. In `tilt` the edge cell (3,3) matters most (-0.14). In `geo` (c_kuu = 0.01) the shifts are +- 0.27 with no dominant cell (largest -0.27 for (0,0), +0.23 for (3,0), +0.19 for (2,0)): the geo value is noisy, not driven by one cell. See `jackknife_influence.png`.
- With resampling errors, `geo` c_kuu (0.01 +- 0.43..0.50) differs from `all` (1.06 +- 0.23) by 1.9 sigma, not the 2.5 sigma of the fit errors. The tilt-vs-geo_gap gap_kuu tension (+0.67 vs -0.83) is 1.5 / sqrt(0.29^2 + 0.65^2) = 2.1 sigma with jackknife errors.

## 3. Model order (committed floors, and floors re-tuned)
| model | npar | chi2 (committed floors) | dchi2 vs base | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta |
|---|---|---|---|---|---|---|---|---|---|
| base | 27 | 284.8 | +0.0 (0 par) | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | +1.02+-0.18 |
| cubic h | 31 | 274.2 | -10.6 (4 par) | +0.78+-0.07 | +1.29+-0.09 | +1.03+-0.09 | +0.30+-0.13 | +0.26+-0.09 | +1.21+-0.22 |
| cubic c | 31 | 258.1 | -26.7 (4 par) | +0.78+-0.07 | +1.29+-0.09 | +1.05+-0.09 | +0.28+-0.13 | +0.26+-0.09 | +1.15+-0.19 |
| cubic h+c | 35 | 255.2 | -29.5 (8 par) | +0.78+-0.07 | +1.30+-0.09 | +1.05+-0.10 | +0.26+-0.13 | +0.27+-0.10 | +1.10+-0.18 |
| quartic u4,w4 h | 29 | 276.6 | -8.2 (2 par) | -0.03+-0.31 | +0.97+-0.37 | +0.92+-0.11 | +0.14+-0.17 | +0.94+-0.26 | +1.00+-0.18 |
| quartic u4,w4 c | 29 | 276.3 | -8.5 (2 par) | +0.69+-0.08 | +1.27+-0.10 | +0.18+-0.33 | -0.15+-0.43 | -0.51+-0.30 | +1.00+-0.18 |
| quartic u4,w4 h+c | 31 | 275.4 | -9.4 (4 par) | +4.76+-4.43 | +0.40+-3.45 | -3.89+-4.44 | +0.72+-3.46 | -8.64+-8.86 | +1.02+-0.19 |
| radial r4 h | 28 | 261.5 | -23.2 (1 par) | +1.53+-0.17 | +2.05+-0.18 | +1.14+-0.10 | +0.30+-0.13 | -0.39+-0.17 | +0.97+-0.17 |
| radial r4 c | 28 | 280.1 | -4.6 (1 par) | +0.80+-0.07 | +1.36+-0.09 | +1.51+-0.23 | +0.62+-0.23 | +0.71+-0.22 | +0.96+-0.17 |
| radial r4 h+c | 29 | 258.4 | -26.3 (2 par) | +1.53+-0.17 | +2.03+-0.17 | +1.49+-0.22 | +0.68+-0.24 | -0.04+-0.26 | +1.06+-0.19 |
| u4 only h | 28 | 277.5 | -7.3 (1 par) | -0.03+-0.31 | +1.31+-0.09 | +0.91+-0.11 | +0.23+-0.13 | +0.94+-0.26 | +1.04+-0.19 |
| u4 only c | 28 | 277.1 | -7.6 (1 par) | +0.69+-0.08 | +1.31+-0.09 | +0.17+-0.33 | +0.23+-0.13 | -0.51+-0.30 | +1.05+-0.19 |
| w4 only h | 28 | 283.8 | -0.9 (1 par) | +0.78+-0.07 | +0.97+-0.37 | +1.06+-0.10 | +0.13+-0.16 | +0.29+-0.10 | +0.98+-0.18 |
| w4 only c | 28 | 283.9 | -0.9 (1 par) | +0.78+-0.07 | +1.27+-0.10 | +1.06+-0.10 | -0.17+-0.43 | +0.29+-0.10 | +0.98+-0.18 |

Re-tuned floors (chi2/n = 1 per class re-imposed with the extra terms):

| model | floor hp | w0 | d | G | GY | T | v | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.1772 | 0.1807 | 0.0053 | 0.0975 | 0.0000 | 0.0187 | 0.0143 | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | +1.02+-0.18 |
| cubic h | 0.1829 | 0.1864 | 0.0057 | 0.1423 | 0.0000 | 0.0150 | 0.0140 | +0.78+-0.07 | +1.27+-0.09 | +1.01+-0.09 | +0.33+-0.13 | +0.23+-0.08 | +1.28+-0.23 |
| cubic c | 0.1777 | 0.1850 | 0.0050 | 0.0728 | 0.0000 | 0.0128 | 0.0137 | +0.79+-0.07 | +1.29+-0.08 | +1.02+-0.09 | +0.28+-0.12 | +0.23+-0.08 | +1.15+-0.19 |
| cubic h+c | 0.1797 | 0.1822 | 0.0051 | 0.0000 | 0.0000 | 0.0126 | 0.0131 | +0.78+-0.07 | +1.30+-0.08 | +1.03+-0.09 | +0.26+-0.12 | +0.24+-0.08 | +1.10+-0.17 |
| quartic u4,w4 h | 0.1722 | 0.1728 | 0.0054 | 0.0857 | 0.0000 | 0.0186 | 0.0144 | -0.02+-0.30 | +0.96+-0.36 | +0.92+-0.11 | +0.14+-0.16 | +0.94+-0.26 | +1.00+-0.18 |
| quartic u4,w4 c | 0.1724 | 0.1729 | 0.0054 | 0.0883 | 0.0000 | 0.0186 | 0.0143 | +0.69+-0.07 | +1.27+-0.09 | +0.18+-0.33 | -0.16+-0.42 | -0.51+-0.30 | +1.00+-0.18 |
| quartic u4,w4 h+c | 0.1764 | 0.1728 | 0.0054 | 0.0000 | 0.0000 | 0.0183 | 0.0140 | +4.93+-4.36 | +0.20+-3.40 | -4.05+-4.37 | +0.92+-3.41 | -8.98+-8.72 | +1.01+-0.18 |
| radial r4 h | 0.1117 | 0.1665 | 0.0050 | 0.0513 | 0.0000 | 0.0206 | 0.0153 | +1.58+-0.15 | +2.11+-0.15 | +1.17+-0.10 | +0.32+-0.13 | -0.42+-0.16 | +0.95+-0.16 |
| radial r4 c | 0.1886 | 0.1679 | 0.0051 | 0.0684 | 0.0717 | 0.0187 | 0.0151 | +0.82+-0.07 | +1.38+-0.09 | +1.58+-0.23 | +0.68+-0.23 | +0.76+-0.22 | +0.94+-0.17 |
| radial r4 h+c | 0.1198 | 0.1661 | 0.0050 | 0.1083 | 0.0000 | 0.0195 | 0.0143 | +1.54+-0.15 | +2.06+-0.15 | +1.46+-0.21 | +0.64+-0.23 | -0.08+-0.25 | +1.05+-0.19 |
| u4 only h | 0.1725 | 0.1748 | 0.0052 | 0.1014 | 0.0000 | 0.0186 | 0.0144 | -0.03+-0.30 | +1.31+-0.09 | +0.91+-0.11 | +0.23+-0.13 | +0.94+-0.26 | +1.04+-0.19 |
| u4 only c | 0.1728 | 0.1748 | 0.0052 | 0.1029 | 0.0000 | 0.0185 | 0.0143 | +0.69+-0.07 | +1.31+-0.09 | +0.18+-0.33 | +0.23+-0.13 | -0.51+-0.30 | +1.05+-0.19 |
| w4 only h | 0.1767 | 0.1786 | 0.0056 | 0.0814 | 0.0000 | 0.0188 | 0.0143 | +0.78+-0.07 | +0.96+-0.37 | +1.06+-0.10 | +0.13+-0.16 | +0.29+-0.10 | +0.97+-0.17 |
| w4 only c | 0.1765 | 0.1786 | 0.0056 | 0.0834 | 0.0000 | 0.0188 | 0.0143 | +0.78+-0.07 | +1.28+-0.10 | +1.06+-0.10 | -0.18+-0.43 | +0.29+-0.10 | +0.97+-0.18 |

Fitted higher-order coefficients (committed floors; units mm per (100 mm)^n):

| model | terms |
|---|---|
| cubic h | h_u3 +0.007+-0.043 (0.2 sigma), h_u2w -0.071+-0.036 (2.0 sigma), h_uw2 +0.107+-0.041 (2.6 sigma), h_w3 +0.020+-0.059 (0.3 sigma) |
| cubic c | c_u3 -0.080+-0.054 (1.5 sigma), c_u2w +0.150+-0.048 (3.1 sigma), c_uw2 -0.193+-0.055 (3.5 sigma), c_w3 +0.051+-0.071 (0.7 sigma) |
| cubic h+c | h_u3 +0.002+-0.044 (0.0 sigma), h_u2w -0.029+-0.040 (0.7 sigma), h_uw2 +0.065+-0.044 (1.5 sigma), h_w3 +0.016+-0.060 (0.3 sigma), c_u3 -0.080+-0.055 (1.5 sigma), c_u2w +0.137+-0.053 (2.6 sigma), c_uw2 -0.171+-0.059 (2.9 sigma), c_w3 +0.049+-0.072 (0.7 sigma) |
| quartic u4,w4 h | h_u4 +0.200+-0.074 (2.7 sigma), h_w4 +0.105+-0.111 (0.9 sigma) |
| quartic u4,w4 c | c_u4 +0.204+-0.074 (2.8 sigma), c_w4 +0.102+-0.111 (0.9 sigma) |
| quartic u4,w4 h+c | h_u4 -1.136+-1.237 (0.9 sigma), h_w4 +0.303+-1.210 (0.3 sigma), c_u4 +1.339+-1.237 (1.1 sigma), c_w4 -0.202+-1.211 (0.2 sigma) |
| radial r4 h | h_r4 -0.113+-0.023 (4.8 sigma) |
| radial r4 c | c_r4 -0.063+-0.029 (2.2 sigma) |
| radial r4 h+c | h_r4 -0.108+-0.023 (4.7 sigma), c_r4 -0.052+-0.028 (1.8 sigma) |
| u4 only h | h_u4 +0.200+-0.074 (2.7 sigma) |
| u4 only c | c_u4 +0.205+-0.074 (2.8 sigma) |
| w4 only h | h_w4 +0.108+-0.111 (1.0 sigma) |
| w4 only c | c_w4 +0.106+-0.111 (1.0 sigma) |

Per-class chi2 (committed floors), base vs best:


Depth at 150 mm (even part along each axis, all fitted terms included) and the gap tilt, per variant:

| model | dchi2 | mesh x | mesh y | cathode x | cathode y | gap tilt x |
|---|---|---|---|---|---|---|
| base | +0.0 | +0.87+-0.08 | +1.48+-0.10 | +1.19+-0.11 | +0.25+-0.15 | +0.39+-0.06 |
| cubic h+c | -29.5 | +0.87+-0.08 | +1.47+-0.10 | +1.18+-0.11 | +0.29+-0.15 | +0.65+-0.13 |
| cubic c | -26.7 | +0.88+-0.08 | +1.46+-0.10 | +1.18+-0.11 | +0.31+-0.15 | +0.60+-0.11 |
| cubic h | -10.6 | +0.87+-0.08 | +1.45+-0.10 | +1.16+-0.10 | +0.33+-0.15 | +0.42+-0.09 |
| u4,w4 on h | -8.2 | +0.98+-0.09 | +1.62+-0.19 | +1.03+-0.13 | +0.15+-0.19 | +0.39+-0.06 |
| u4,w4 on c | -8.5 | +0.77+-0.09 | +1.43+-0.11 | +1.24+-0.11 | +0.35+-0.18 | +0.39+-0.06 |
| u4 on h | -7.3 | +0.98+-0.09 | +1.47+-0.10 | +1.03+-0.12 | +0.26+-0.15 | +0.38+-0.06 |
| u4 on c | -7.6 | +0.77+-0.09 | +1.47+-0.10 | +1.23+-0.11 | +0.26+-0.15 | +0.38+-0.06 |
| r4 on h | -23.2 | +1.15+-0.10 | +1.74+-0.11 | +1.29+-0.11 | +0.34+-0.15 | +0.40+-0.06 |
| r4 on c | -4.6 | +0.90+-0.08 | +1.53+-0.10 | +1.38+-0.14 | +0.38+-0.16 | +0.40+-0.06 |
| r4 on h+c | -26.3 | +1.17+-0.10 | +1.74+-0.11 | +1.42+-0.14 | +0.50+-0.17 | +0.38+-0.06 |

Reading:
- The statement "the board is not a pure paraboloid" **holds, but the data do not say which surface carries the extra term.** A separable quartic along U improves chi2 by 7.3 (1 parameter, 2.7 sigma) on the mesh and by 7.6 on the cathode, equally (the fit sees only h + c in w0 and the difference weakly). Along W it is not needed (-0.9). A radial r^4 term is much better on the mesh (-23.2, 4.8 sigma, coefficient -0.113) than on the cathode (-4.6); a cubic set on the cathode (-26.7, 4 parameters, c_u2w 3.1 sigma, c_uw2 3.5 sigma) beats the cubic set on the mesh (-10.6). So the mesh takes the radial term and the cathode takes the odd terms. Both together are degenerate (u4,w4 on h+c: coefficients +-1.2).
- **The quadratic coefficients are not stable against these terms; the depths at 150 mm are.** With u4 on the mesh h_kuu falls from 0.78 to -0.03 +- 0.31; with u4 on the cathode c_kuu falls from 1.06 to 0.18 +- 0.33; with r4 on the mesh h_kuu and h_kww jump to 1.5 and 2.1. Yet the depths at 150 mm stay in 0.77-1.17 (mesh x), 1.43-1.74 (mesh y), 1.03-1.42 (cathode x), 0.15-0.50 (cathode y). So "c_kuu = 1.06" should not be quoted as a curvature of a paraboloid; "the cathode is about 1.2 mm lower at x = +-150 mm than at the centre" is the stable statement.
- Re-tuning the floors moves them by < 4 % for the quartic variants (hp 0.17-0.19, w0 0.17-0.19) but the G floor swings from 0.098 to 0.00-0.14 and T and v by up to 30 % for the cubic variants: the floors are not a property of the data, they depend on the model.
- beta moves 0.96-1.28 (error 0.17-0.23): unchanged within errors.

## 4. Mixing weight a (w0 = v[(1-a) grad h + a grad c]); d kept at 1/2
**all**

| a | chi2 | dchi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 284.4 | -0.3 | +0.80+-0.07 | +1.22+-0.08 | +1.08+-0.11 | +0.01+-0.16 | +0.27+-0.11 | +0.93+-0.17 | +0.91+-0.07 | +1.21+-0.13 | +0.01+-0.18 |
| 0.4 | 284.1 | -0.7 | +0.79+-0.07 | +1.27+-0.08 | +1.07+-0.11 | +0.11+-0.15 | +0.28+-0.11 | +0.97+-0.17 | +0.89+-0.08 | +1.20+-0.12 | +0.12+-0.17 |
| 0.5 | 284.8 | +0.0 | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | +1.02+-0.18 | +0.87+-0.08 | +1.19+-0.11 | +0.25+-0.15 |
| 0.6 | 286.2 | +1.4 | +0.77+-0.07 | +1.34+-0.09 | +1.04+-0.09 | +0.33+-0.12 | +0.27+-0.09 | +1.09+-0.20 | +0.86+-0.08 | +1.17+-0.10 | +0.37+-0.14 |
| 0.7 | 288.1 | +3.3 | +0.76+-0.07 | +1.37+-0.09 | +1.02+-0.09 | +0.43+-0.11 | +0.26+-0.09 | +1.16+-0.21 | +0.85+-0.08 | +1.15+-0.10 | +0.48+-0.13 |
| free a (one): a_x=0.38+-0.13, a_y=0.38+-0.13 | 284.1 | -0.7 | +0.79+-0.07 | +1.26+-0.11 | +1.07+-0.11 | +0.09+-0.20 | +0.28+-0.11 | +0.96+-0.18 | +0.89+-0.08 | +1.21+-0.13 | +0.10+-0.22 |
| free a_x,a_y: a_x=1.73+-1.04, a_y=0.30+-0.13 | 278.9 | -5.8 | +0.66+-0.09 | +1.22+-0.11 | +0.88+-0.14 | +0.02+-0.20 | +0.22+-0.11 | +0.95+-0.18 | +0.75+-0.10 | +0.99+-0.16 | +0.02+-0.23 |

**no_hp**

| a | chi2 | dchi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 213.8 | -1.1 | +0.95+-0.10 | +1.18+-0.12 | +1.16+-0.12 | +0.04+-0.16 | +0.21+-0.11 | +0.99+-0.19 | +1.06+-0.11 | +1.30+-0.13 | +0.04+-0.18 |
| 0.4 | 214.2 | -0.6 | +0.92+-0.10 | +1.28+-0.13 | +1.14+-0.11 | +0.14+-0.15 | +0.22+-0.11 | +0.99+-0.19 | +1.04+-0.11 | +1.28+-0.12 | +0.16+-0.17 |
| 0.5 | 214.8 | +0.0 | +0.90+-0.11 | +1.38+-0.14 | +1.12+-0.11 | +0.25+-0.14 | +0.22+-0.11 | +1.00+-0.20 | +1.02+-0.12 | +1.26+-0.12 | +0.29+-0.15 |
| 0.6 | 215.7 | +0.8 | +0.88+-0.11 | +1.47+-0.15 | +1.10+-0.10 | +0.36+-0.12 | +0.22+-0.11 | +1.01+-0.20 | +0.99+-0.12 | +1.24+-0.11 | +0.41+-0.14 |
| 0.7 | 216.6 | +1.8 | +0.86+-0.12 | +1.56+-0.16 | +1.08+-0.10 | +0.47+-0.12 | +0.22+-0.11 | +1.03+-0.20 | +0.97+-0.13 | +1.22+-0.11 | +0.53+-0.13 |
| free a (one): a_x=0.11+-0.33, a_y=0.11+-0.33 | 213.4 | -1.4 | +0.99+-0.11 | +0.98+-0.36 | +1.19+-0.14 | -0.15+-0.36 | +0.20+-0.11 | +0.99+-0.20 | +1.11+-0.13 | +1.34+-0.16 | -0.17+-0.41 |
| free a_x,a_y: a_x=2.97+-2.18, a_y=0.02+-0.34 | 211.1 | -3.7 | +0.39+-0.39 | +0.88+-0.37 | +0.61+-0.39 | -0.24+-0.37 | +0.22+-0.11 | +1.00+-0.20 | +0.44+-0.44 | +0.68+-0.44 | -0.27+-0.41 |

**tilt**

| a | chi2 | dchi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 157.4 | -0.1 | +0.70+-0.09 | +1.30+-0.11 | +1.62+-0.35 | -0.26+-0.39 | +0.92+-0.40 | +0.68+-0.00 | +0.79+-0.10 | +1.83+-0.39 | -0.29+-0.44 |
| 0.4 | 157.3 | -0.2 | +0.69+-0.09 | +1.29+-0.11 | +1.47+-0.26 | +0.11+-0.29 | +0.79+-0.32 | +0.68+-0.00 | +0.77+-0.10 | +1.66+-0.29 | +0.12+-0.33 |
| 0.5 | 157.5 | +0.0 | +0.68+-0.09 | +1.28+-0.11 | +1.35+-0.20 | +0.35+-0.23 | +0.67+-0.26 | +0.68+-0.00 | +0.76+-0.10 | +1.52+-0.23 | +0.39+-0.26 |
| 0.6 | 157.8 | +0.3 | +0.67+-0.09 | +1.27+-0.11 | +1.25+-0.16 | +0.51+-0.18 | +0.58+-0.22 | +0.68+-0.00 | +0.76+-0.10 | +1.41+-0.18 | +0.58+-0.21 |
| 0.7 | 158.1 | +0.5 | +0.67+-0.09 | +1.27+-0.11 | +1.18+-0.14 | +0.63+-0.15 | +0.50+-0.19 | +0.68+-0.00 | +0.76+-0.10 | +1.32+-0.15 | +0.71+-0.17 |
| free a (one): a_x=0.37+-0.19, a_y=0.37+-0.19 | 157.3 | -0.3 | +0.69+-0.09 | +1.30+-0.11 | +1.52+-0.47 | -0.00+-0.64 | +0.83+-0.51 | +0.68+-0.00 | +0.78+-0.10 | +1.71+-0.53 | -0.00+-0.72 |
| free a_x,a_y: a_x=1283.63+-50.77, a_y=0.22+-0.10 | 153.6 | -3.9 | +0.67+-0.09 | +1.28+-0.11 | +0.67+-0.09 | -0.64+-0.72 | +0.00+-0.00 | +0.68+-0.00 | +0.75+-0.10 | +0.75+-0.10 | -0.72+-0.81 |

**timing**

| a | chi2 | dchi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 223.7 | +2.4 | +0.81+-0.07 | +1.22+-0.08 | +1.11+-0.11 | +0.21+-0.20 | +0.29+-0.11 | +1.14+-0.30 | +0.92+-0.08 | +1.24+-0.13 | +0.24+-0.23 |
| 0.4 | 222.1 | +0.9 | +0.80+-0.07 | +1.26+-0.09 | +1.09+-0.11 | +0.30+-0.18 | +0.29+-0.10 | +1.18+-0.30 | +0.90+-0.08 | +1.23+-0.12 | +0.34+-0.20 |
| 0.5 | 221.2 | +0.0 | +0.79+-0.07 | +1.30+-0.09 | +1.07+-0.10 | +0.40+-0.16 | +0.28+-0.10 | +1.26+-0.32 | +0.89+-0.08 | +1.20+-0.11 | +0.45+-0.18 |
| 0.6 | 220.9 | -0.3 | +0.78+-0.07 | +1.32+-0.10 | +1.04+-0.09 | +0.49+-0.14 | +0.27+-0.09 | +1.37+-0.34 | +0.88+-0.08 | +1.18+-0.10 | +0.55+-0.16 |
| 0.7 | 220.9 | -0.3 | +0.77+-0.07 | +1.34+-0.10 | +1.02+-0.09 | +0.58+-0.12 | +0.25+-0.08 | +1.49+-0.37 | +0.87+-0.08 | +1.15+-0.10 | +0.65+-0.14 |
| free a (one): a_x=0.64+-0.26, a_y=0.64+-0.26 | 220.9 | -0.4 | +0.77+-0.07 | +1.33+-0.11 | +1.03+-0.11 | +0.53+-0.26 | +0.26+-0.10 | +1.42+-0.47 | +0.87+-0.08 | +1.16+-0.13 | +0.60+-0.29 |
| free a_x,a_y: a_x=1.69+-1.00, a_y=0.51+-0.24 | 217.9 | -3.3 | +0.68+-0.09 | +1.28+-0.11 | +0.89+-0.14 | +0.45+-0.27 | +0.21+-0.10 | +1.40+-0.47 | +0.76+-0.10 | +1.00+-0.15 | +0.51+-0.30 |

profile chi2 all: min at a=0.40 (chi2 284.1); delta chi2<1 for a in [0.25,0.50]; chi2(0.5)-min=0.7
profile chi2 no_hp: min at a=0.10 (chi2 213.4); delta chi2<1 for a in [0.00,0.40]; chi2(0.5)-min=1.4
profile chi2 tilt: min at a=0.35 (chi2 157.3); delta chi2<1 for a in [0.25,0.75]; chi2(0.5)-min=0.2

![chi2 vs a](chi2_vs_a.png)

Reading:
- The data do not determine a. One common a: 0.38 +- 0.13 in `all` (profile chi2 within 1 of the minimum for a in 0.25-0.50), 0.11 +- 0.33 in `no_hp`, 0.37 +- 0.19 in `tilt`, 0.64 +- 0.26 in `timing`. a = 1/2 is within 1 sigma everywhere (chi2 +0.7 at most). Separate a_x, a_y: a_y = 0.30 +- 0.13 but a_x is not constrained (1.7 +- 1.0).
- **Cathode curvature along x is insensitive to a**: c_kuu 1.08 -> 1.02 and depth 1.21 -> 1.15 mm for a = 0.3 -> 0.7. The mesh depth along x is also stable (0.91 -> 0.85).
- **Cathode curvature along y depends strongly on a**: c_kww 0.01 +- 0.16 (a = 0.3), 0.22 +- 0.13 (0.5), 0.43 +- 0.11 (0.7); depth at 150 mm 0.01 -> 0.25 -> 0.48 mm. "Nearly flat along y" is therefore a statement about a: it is exactly flat for a = 0.3 and 0.5 +- 0.13 mm deep for a = 0.7. In `no_hp`, h_kww goes 1.18 -> 1.38 -> 1.56 over the same range (the 'mesh without positions' along y depends on a by +-0.19, comparable to the position-based value 1.28).
- The no-position mesh (0.90, 1.38 at a = 0.5) moves to (0.95, 1.18) at a = 0.3 and (0.86, 1.56) at a = 0.7: the agreement with positions in x holds for all a; in y the difference from the position value (1.28) is -0.10 at a = 0.3 and +0.28 at 0.7, both inside the 0.3 error of the comparison.

## 5. Constrained hypotheses (all classes, committed floors, and T,v floors x0.5 and x2)
**committed floors** (free fit chi2 284.8; per class hp 65/64, w0 66/64, d 32/31, G 16/16, GY 14/15, T 32/32, v 60/64)

| hypothesis | n constraints | dchi2 | p (chi2) | sigma-equiv | per-class dchi2 (hp, w0, d, G, GY, T, v) |
|---|---|---|---|---|---|
| H1 flat cathode | 3 | +106.9 | 5e-23 | 9.9 | -0.2 +58.0 -0.9 -2.1 -5.6 +20.6 +37.2 |
| H2 cathode parallel to mesh | 3 | +106.3 | 6.9e-23 | 9.8 | +8.6 +3.6 +3.3 +12.7 +2.2 +34.6 +41.3 |
| H3 isotropic h and c | 2 | +92.4 | 8.6e-21 | 9.4 | +17.4 +2.1 +0.9 +6.3 -1.7 +26.4 +41.0 |
| H4 isotropic cathode | 1 | +25.4 | 4.7e-07 | 5.0 | -0.9 +11.8 +3.3 +8.8 +0.5 +0.9 +0.9 |
| H5 taut foil on board-following frame | 2 | +145.5 | 2.6e-32 | 11.8 | +11.0 +46.4 -2.2 -2.7 -4.8 +39.3 +58.4 |
| H6 cathode x follows board, y flat (c_kuu=h_kuu, c_kww=0) | 2 | +11.2 | 0.0037 | 2.9 | +2.3 +5.6 -1.4 -1.0 -2.8 +0.8 +7.8 |
| H7 c_kuu=h_kuu only | 1 | +8.8 | 0.003 | 3.0 | +2.3 +2.6 -0.3 -0.1 -2.4 -0.1 +6.9 |
| H8 c_kww=0 only | 1 | +2.5 | 0.11 | 1.6 | -0.2 +2.9 -1.1 -0.9 -0.2 +0.9 +1.1 |

**T,v floors x0.5** (free fit chi2 449.8; per class hp 65/64, w0 66/64, d 33/31, G 17/16, GY 14/15, T 91/32, v 165/64)

| hypothesis | n constraints | dchi2 | p (chi2) | sigma-equiv | per-class dchi2 (hp, w0, d, G, GY, T, v) |
|---|---|---|---|---|---|
| H1 flat cathode | 3 | +181.9 | 3.4e-39 | 13.1 | +19.3 +96.3 -0.9 -2.4 -4.5 +16.8 +57.5 |
| H2 cathode parallel to mesh | 3 | +253.2 | 1.4e-54 | 15.6 | +8.5 +3.8 +3.0 +12.3 +2.0 +90.5 +133.0 |
| H3 isotropic h and c | 2 | +224.2 | 2.1e-49 | 14.8 | +17.0 +1.7 +0.8 +6.8 -1.3 +69.1 +130.2 |
| H4 isotropic cathode | 1 | +27.5 | 1.6e-07 | 5.2 | -0.6 +13.7 +3.6 +7.7 +0.7 +2.4 +0.1 |
| H5 taut foil on board-following frame | 2 | +321.0 | 2e-70 | 17.7 | +12.0 +61.5 -3.0 -2.9 -4.3 +91.2 +166.6 |
| H6 cathode x follows board, y flat (c_kuu=h_kuu, c_kww=0) | 2 | +27.2 | 1.3e-06 | 4.8 | +2.2 +7.0 -1.9 -1.7 -2.7 +1.2 +23.1 |
| H7 c_kuu=h_kuu only | 1 | +21.9 | 2.8e-06 | 4.7 | +2.1 +2.5 -0.3 -0.2 -2.2 -0.9 +21.0 |
| H8 c_kww=0 only | 1 | +5.9 | 0.015 | 2.4 | -0.1 +4.6 -1.6 -1.7 -0.2 +2.3 +2.7 |

**T,v floors x2** (free fit chi2 218.7; per class hp 65/64, w0 67/64, d 32/31, G 15/16, GY 14/15, T 9/32, v 18/64)

| hypothesis | n constraints | dchi2 | p (chi2) | sigma-equiv | per-class dchi2 (hp, w0, d, G, GY, T, v) |
|---|---|---|---|---|---|
| H1 flat cathode | 3 | +59.3 | 8.2e-13 | 7.2 | +3.6 +43.1 -1.3 -1.3 -5.7 +8.3 +12.6 |
| H2 cathode parallel to mesh | 3 | +51.6 | 3.7e-11 | 6.6 | +8.7 +3.5 +3.3 +12.5 +1.9 +10.0 +11.6 |
| H3 isotropic h and c | 2 | +43.4 | 3.7e-10 | 6.3 | +18.2 +3.0 +0.6 +5.3 -2.7 +7.6 +11.5 |
| H4 isotropic cathode | 1 | +21.1 | 4.4e-06 | 4.6 | -0.1 +10.8 +2.7 +4.3 -1.8 +1.6 +3.6 |
| H5 taut foil on board-following frame | 2 | +73.7 | 9.7e-17 | 8.3 | +14.0 +40.0 -1.9 -1.7 -5.1 +12.0 +16.5 |
| H6 cathode x follows board, y flat (c_kuu=h_kuu, c_kww=0) | 2 | +5.2 | 0.075 | 1.8 | +2.5 +4.9 -1.2 -0.4 -2.9 +0.2 +2.0 |
| H7 c_kuu=h_kuu only | 1 | +3.9 | 0.048 | 2.0 | +2.4 +2.7 -0.3 -0.1 -2.7 +0.0 +1.8 |
| H8 c_kww=0 only | 1 | +1.2 | 0.27 | 1.1 | +0.0 +2.1 -0.8 -0.3 -0.2 +0.2 +0.2 |


Reading:
- Chi2 differences are not trustworthy as quoted. Halving the T and v floors makes their chi2 91/32 and 165/64 and every Delta chi2 grows by a factor 1.5-2.5; doubling shrinks them by about a factor 2. The `p` and sigma values above inherit that. Resampling inflates the fit errors by 2-3 on the curvatures, so divide every Delta chi2 by about 5 (2.3^2) for a realistic comparison.
- H1 (flat cathode): rejected at all floors (Delta chi2 107 / 59 / 182), carried by w0 (+58) and v (+37), T (+21): the cathode is not flat, and this does not need the geometric classes (G, GY, d actually prefer a flat cathode, -2, -6, -1). This is the strongest statement in the paper.
- H2 (cathode parallel to the mesh) and H3 (isotropic): rejected (Delta chi2 106 and 92), carried by T and v (35 + 41, 26 + 41) and, for H2, G (+13): the gap does vary across the chamber.
- H4 (isotropic cathode, c_kuu = c_kww): rejected, Delta chi2 25 (5.0 sigma), carried by w0 (+12), G (+9): the cathode curvature is anisotropic, with c_kuu > c_kww.
- H5 (taut foil on the board-following frame): rejected, Delta chi2 146, carried by w0, T, v.
- H7 (c_kuu = h_kuu: cathode parallel to the mesh along x only): Delta chi2 8.8 (3.0 sigma; 2.0 sigma with T,v floors x2, 4.7 with x0.5), carried by v (+6.9) and w0 (+2.6). After the resampling correction this is below 1.5 sigma: "the cathode follows the board along x" is not contradicted.
- H8 (c_kww = 0): Delta chi2 2.5 (1.6 sigma). A flat cathode along y is allowed. H6 (c_kuu = h_kuu and c_kww = 0): Delta chi2 11.2 (2.9 sigma; 1.8 with x2) with 2 parameters: acceptable after resampling.

## 6. Which data say "the cathode follows the board along x"
Gap curvature along U (c_kuu - h_kuu) and along W, and gap tilt along U, from each class separately. G alone has U = -0.43 and +0.42 only, so its curvature along U is NOT MEASURABLE (the 'G alone' row below shows +224 +- 0.01: that is the minimum-norm solution of a rank-deficient fit; ignore it, and its tilt 2.28 with it). For the same reason the first table compares tilt along U on the columns where G exists in the second table (curvature along U switched off, all classes restricted to the same cells).
**All cells (as in the joint fit)**

| method | gap_kuu (c-h) | gap_kww | gap tilt U | chi2/ndf |
|---|---|---|---|---|
| G alone | +224.07 +- 0.01 | -1.21 +- 0.34 | +2.28 +- 0.24 | 11/10 |
| GY alone | -0.70 +- 0.49 | -0.68 +- 1.13 | +0.75 +- 0.26 | 5/8 |
| T alone (beta=1.0) [g0 fixed 28.2] | +0.13 +- 0.14 | -1.08 +- 0.18 | +0.39 +- 0.06 | 29/25 |
| v_x alone (beta=1.0) | +0.42 +- 0.15 | -1.10 +- 0.19 | +0.36 +- 0.06 | 56/55 |
| T alone (beta=0.68) [g0 fixed 28.2] | +0.16 +- 0.17 | -1.28 +- 0.21 | +0.46 +- 0.07 | 29/25 |
| v_x alone (beta=0.68) | +0.62 +- 0.22 | -1.60 +- 0.27 | +0.52 +- 0.09 | 56/55 |
| v_x only (beta=1.0) | +0.15 +- 0.21 | -1.05 +- 0.26 | +0.63 +- 0.08 | 17/23 |
| v_y only (beta=1.0) | +0.65 +- 0.22 | -1.12 +- 0.27 | +0.09 +- 0.08 | 14/23 |
| v_x*T product (beta-free) | +0.12 +- 0.30 | -1.12 +- 0.37 | +0.15 +- 0.12 | 9/25 |
| v_y*T product (beta-free) | -0.37 +- 0.30 | -1.01 +- 0.37 | +0.68 +- 0.12 | 12/25 |
| tilt route: hp + w0 (a=1/2) | +0.73 +- 0.26 | -0.85 +- 0.31 | -1.06 +- 0.04 | 127/110 |
| hp + d (timing-free tilt) | -0.34 +- 0.80 | -1.92 +- 0.75 | +0.46 +- 0.34 | 90/81 |
| hp + w0 + d | +0.67 +- 0.26 | -0.94 +- 0.30 | +0.44 +- 0.34 | 158/141 |
| G+GY+hp (geo_gap) | -0.83 +- 0.47 | -1.23 +- 0.31 | +0.61 +- 0.17 | 84/79 |
| T+v (beta=1) alone joint, g0 fixed | +0.27 +- 0.11 | -1.09 +- 0.13 | +0.37 +- 0.04 | 92/85 |

**Same cells as G, curvature along U switched off (c_kuu = c_kuw = h_kuu = h_kuw = 0; G has only two columns, U = -0.43 and +0.42): gap TILT along U is the quantity**

| method | gap_kuu (c-h) | gap_kww | gap tilt U | chi2/ndf |
|---|---|---|---|---|
| G alone | not constrained | -1.19 +- 0.34 | +0.76 +- 0.23 | 12/12 |
| GY alone | not constrained | -1.30 +- 1.45 | +0.62 +- 0.47 | 2/3 |
| T alone (beta=1.0) [g0 fixed 28.2] | not constrained | -1.04 +- 0.25 | +0.54 +- 0.17 | 13/11 |
| v_x alone (beta=1.0) | not constrained | -1.16 +- 0.25 | +0.38 +- 0.17 | 18/25 |
| T alone (beta=0.68) [g0 fixed 28.2] | not constrained | -1.23 +- 0.29 | +0.65 +- 0.20 | 13/11 |
| v_x alone (beta=0.68) | not constrained | -1.69 +- 0.37 | +0.54 +- 0.25 | 18/25 |
| v_x only (beta=1.0) | not constrained | -1.04 +- 0.36 | +0.61 +- 0.24 | 8/9 |
| v_y only (beta=1.0) | not constrained | -1.29 +- 0.36 | +0.13 +- 0.25 | 8/9 |
| v_x*T product (beta-free) | not constrained | -1.06 +- 0.50 | +0.46 +- 0.34 | 3/11 |
| v_y*T product (beta-free) | not constrained | -0.79 +- 0.51 | +0.97 +- 0.35 | 4/11 |
| tilt route: hp + w0 (a=1/2) | not constrained | -0.54 +- 0.42 | +1.52 +- 0.11 | 39/50 |
| hp + d (timing-free tilt) | not constrained | -1.51 +- 1.01 | +0.35 +- 0.38 | 37/46 |
| hp + w0 + d | not constrained | -0.62 +- 0.41 | +0.35 +- 0.38 | 59/74 |
| G+GY+hp (geo_gap) | not constrained | -1.25 +- 0.33 | +0.74 +- 0.21 | 32/44 |
| T+v (beta=1) alone joint, g0 fixed | not constrained | -1.10 +- 0.18 | +0.46 +- 0.12 | 33/39 |

Reading:
- **Gap curvature along U from methods that cover all four columns**: T alone +0.13 +- 0.14 (beta = 1) or +0.16 +- 0.17 (0.68); v_x +0.42 +- 0.15; v_x*T (beta-free) +0.12 +- 0.30, v_y*T -0.37 +- 0.30; GY alone -0.70 +- 0.49; tilt route (hp + w0) +0.73 +- 0.26; joint T + v +0.27 +- 0.11. The spread is -0.7 to +0.7; the combination in `all` is +0.28. **No single method gives a gap curvature along U that is significantly non-zero except the tilt route (2.8 sigma) and v_x (2.8 sigma, but v_y gives +0.65 and the product v_y*T -0.37).** The positive value in `all` is carried by w0 and v together, which depends on a = 1/2 (section 4) and on the single per-view velocity.
- The only method that is negative is the one that uses GY along U (`geo_gap`, `GY alone`). It rests on Y-strip peak samples, which `gapmap.py` itself calls 'relative only' (the peak version reads 0.6-2.3 mm high with blur, and the Y blur depends on the RC distance, i.e. on position along the strip). I did not test that this blur varies with U; it is the obvious candidate for the geo_gap sign. This is not demonstrated.
- **Coverage, not a direct conflict, explains most of the geo tension**: dX and G sit in two columns 0.85 apart, so the curvature along U in `geo` rests on dX(U = +-0.43) (a difference of (h_kuu + c_kuu) * 0.85 / 200 = 0.0077 against floors 0.0053), and on dY/GY. The tilt-based curvature uses w0 over the whole U range +-1.3. With the same cells and curvature switched off (third table) every method gives a positive gap tilt along U of 0.35 to 0.76 (G 0.76 +- 0.23, T 0.54 +- 0.17, v_x 0.38 +- 0.17, hp + d 0.35 +- 0.38, T + v 0.46 +- 0.12, geo_gap 0.74 +- 0.21; v_y alone is lower, 0.13 +- 0.25). The only outlier is the hp + w0 row (1.52 +- 0.11), whose tilt is degenerate with the constants (section 0). The tilt along U is therefore well established; its curvature is not.
- v_x and v_y give different gap curvatures along U from the same quadratic model (+0.15 +- 0.21 and +0.65 +- 0.22 with the same cells), a 1.6 sigma difference that should not exist if both measure the gap alone: part of the 'gap' signal in v is not gap.

## 7. Sign and consistency pins (no fit)
(a) Per cell and run, regression of the timing-free tilt dX on w0_x / v (expected +1; cells with dX: 8, 8; dY: 3, 12). Slopes with the cross-cell regression error (bootstrap error in brackets; attenuation by noise in w0/v is 1-2 %):
- run 1 x: +1.57 +- 0.37 (0.42), r = +0.87, n = 8; run 1 y: +0.79 +- 0.06 (0.26), r = +1.00, n = 3 (trivial).
- run 2 x: +0.91 +- 0.24 (0.25), r = +0.84, n = 8; run 2 y: +0.97 +- 0.24 (0.21), r = +0.79, n = 12.
So w0 / v and the timing-free tilt agree in sign and size (run 1 x is 1.4 sigma above 1). This is an independent pin of "w0 is a drift tilt" and of the factor v/1 in w0 = v tau, at the 25 % level.
(b) Correlation of w0 across cells with the own-axis gradient of hp (central difference along the row/column), slope in um/ns per (mm per 100 mm): run 1 x +0.45 +- 0.19 (r +0.54), run 1 y +0.24 +- 0.04 (r +0.83), run 2 x +0.48 +- 0.15 (r +0.66), run 2 y +0.39 +- 0.05 (r +0.92). All four positive. The expected value is (v / 200) (1 + dc/dh) = 0.185 (1 + dc/dh), so dc/dh = 0.3-1.6: the sign is that of a dished mesh pushing the drift lines outward, and the size is consistent with the cathode following the mesh (1.0) within the noise.
(c) The ladder slope v (w*1e3 against tan_ref) is positive in every cell: run 1 x 35.0-39.1, y 35.4-37.9; run 2 x 35.8-39.5, y 36.0-39.4 um/ns. No sign is applied to the tables in `GM.load_table` (grep: no flip, rotate or sign in `gapmap.py`, `cellobs.py`, `track_test.py`, `track_grid.py`). The 90 deg mounting is handled upstream: `alignment.json` has `theta_deg` 89.45 and `ref_x_sign` +1.0 for both runs, the bundle has no sign/flip key, and `tan_ref_x/y`, `ref_mesh_x/y` are read as given. I did not re-derive the M3-to-chamber rotation; a positive v only shows that x_w and tan_ref share a sign convention. The sign of 'cathode above the mesh' is pinned instead by (b): w0 rises with the mesh gradient, and by the fact that the gap g = c - h comes out at +28 mm (G, T and v all see a positive gap; the mesh is the sign-fixed hp from M3 z up).

## 8. Correlation of w0 and v
From the line-fit covariance per cell and view (64 cell-views): median |corr(w0, v)| = 0.10, max 0.22 (the mean tan_ref of ladder tracks is 0.013 median, 0.024 max, so the intercept and slope are nearly orthogonal). Refit with the full 2x2 block per cell and view (floors added on the diagonal): chi2 284.78 against 284.76, c_kuu 1.057 against 1.056, beta 1.018 against 1.018; the same at the 0.01 level in `no_hp` and `timing`. **It does not matter.** (The 2x2 table is in `sec8.md`.)
Per-cell corr(w0, v) from the line-fit covariance (64 cell-views): median |rho| 0.10, max |rho| 0.22; mean tan_ref of the ladder tracks: median |<tan>| 0.013, max 0.024

| arm | fit | chi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | diagonal (committed) | 284.8 | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | +1.02+-0.18 | +0.87+-0.08 | +1.19+-0.11 | +0.25+-0.15 |
| all | 2x2 w0-v covariance | 284.8 | +0.78+-0.07 | +1.31+-0.09 | +1.06+-0.10 | +0.22+-0.13 | +0.28+-0.10 | +1.02+-0.18 | +0.87+-0.08 | +1.19+-0.11 | +0.25+-0.15 |
| no_hp | diagonal (committed) | 214.8 | +0.90+-0.11 | +1.38+-0.14 | +1.12+-0.11 | +0.25+-0.14 | +0.22+-0.11 | +1.00+-0.20 | +1.02+-0.12 | +1.26+-0.12 | +0.29+-0.15 |
| no_hp | 2x2 w0-v covariance | 214.9 | +0.90+-0.10 | +1.38+-0.14 | +1.12+-0.11 | +0.26+-0.14 | +0.22+-0.11 | +1.00+-0.20 | +1.01+-0.12 | +1.26+-0.12 | +0.29+-0.15 |
| timing | diagonal (committed) | 221.2 | +0.79+-0.07 | +1.30+-0.09 | +1.07+-0.10 | +0.40+-0.16 | +0.28+-0.10 | +1.26+-0.32 | +0.89+-0.08 | +1.20+-0.11 | +0.45+-0.18 |
| timing | 2x2 w0-v covariance | 221.2 | +0.79+-0.07 | +1.30+-0.09 | +1.07+-0.10 | +0.40+-0.16 | +0.28+-0.10 | +1.26+-0.32 | +0.89+-0.08 | +1.20+-0.11 | +0.45+-0.18 |
| tilt | diagonal (committed) | 157.5 | +0.68+-0.09 | +1.28+-0.11 | +1.35+-0.20 | +0.35+-0.23 | +0.67+-0.26 | +0.68+-0.00 | +0.76+-0.10 | +1.52+-0.23 | +0.39+-0.26 |

## Extra check: is the mesh 'without positions' independent of the mesh from positions?
`f_11_nohp.py`. The positions alone (class hp only, h 5 parameters) give h_kuu 0.68 +- 0.09 and h_kww 1.28 +- 0.11 (this is the h of the `tilt` and `gap` arms). `no_hp` gives 0.90 +- 0.11 and 1.38 +- 0.14. Re-tuning the floors without hp barely moves them (w0 0.176, G 0.086, T 0.0187, v 0.0143) and gives the same mesh (0.90, 1.38), so the floors do not manufacture the agreement. Difference no_hp - hp: +0.22 +- 0.22 (bootstrap 16-84 half-width, correlation of the two estimators -0.3) in kuu and +0.09 +- 0.30 in kww. They agree. The test has limited power: the no-position mesh is only known to +-0.2 / +-0.3 (a quarter of its value) and depends on a (section 4).

## 9. My table (depth at 150 mm in mm, positive = centre lower; gap tilt in mm per 100 mm)
Errors: the 'resampling' column is 1.125 x max(jackknife, bootstrap) of the corresponding coefficient in `all`; the other three are half the range over the variants of sections 3, 4 and the floor scalings of this table. Total = quadrature of these four.

| quantity | committed fit | fit error | resampling error (max of jackknife, bootstrap over cells) | model-order range | a = 0.3..0.7 range | floor x0.5..x2 range | total (quadrature of fit-err-free parts: resampling + half-ranges) |
|---|---|---|---|---|---|---|---|
| mesh_x | +0.87 | 0.08 | 0.18 | +0.77..+1.17 | +0.85..+0.91 | +0.82..+0.93 | 0.28 |
| mesh_y | +1.48 | 0.10 | 0.28 | +1.43..+1.74 | +1.37..+1.54 | +1.39..+1.54 | 0.34 |
| cath_x | +1.19 | 0.11 | 0.26 | +1.03..+1.42 | +1.15..+1.21 | +1.07..+1.29 | 0.34 |
| cath_y | +0.25 | 0.15 | 0.38 | +0.15..+0.50 | +0.01..+0.48 | +0.11..+0.37 | 0.50 |
| gtilt_U | +0.39 | 0.06 | 0.16 | +0.38..+0.65 | +0.35..+0.41 | +0.34..+0.45 | 0.21 |

Floor-sensitivity detail:

| floors | mesh x | mesh y | cathode x | cathode y | gap tilt x |
|---|---|---|---|---|---|
| all floors x0.5 | +0.93+-0.05 | +1.50+-0.07 | +1.24+-0.06 | +0.37+-0.09 | +0.35+-0.04 |
| all floors x2 | +0.85+-0.13 | +1.47+-0.16 | +1.07+-0.21 | +0.11+-0.25 | +0.45+-0.10 |
| T,v floors x0.5 | +0.88+-0.07 | +1.45+-0.09 | +1.17+-0.09 | +0.32+-0.13 | +0.34+-0.04 |
| T,v floors x2 | +0.87+-0.08 | +1.50+-0.10 | +1.21+-0.15 | +0.19+-0.17 | +0.45+-0.09 |
| hp,w0 floors x0.5 | +0.91+-0.06 | +1.54+-0.07 | +1.29+-0.08 | +0.35+-0.11 | +0.38+-0.06 |
| hp,w0 floors x2 | +0.82+-0.12 | +1.39+-0.15 | +1.09+-0.15 | +0.11+-0.20 | +0.39+-0.06 |

Quoted version (committed value, total error):
- mesh x 0.87 +- 0.28 (position route alone 0.77 +- 0.10 fit; model-order range 0.77-1.17)
- mesh y 1.48 +- 0.34
- cathode x 1.19 +- 0.34
- cathode y 0.25 +- 0.50
- gap tilt x 0.39 +- 0.21 (cubic terms raise it to 0.65; it is the only quantity measured by every method, 0.35-0.76)

## Verdict on the three statements
(i) **Mesh dish from positions agrees with the mesh without positions: survives, weakly.** h_kuu 0.68 vs 0.90 and h_kww 1.28 vs 1.38 agree (difference +0.22 +- 0.22 and +0.09 +- 0.30), the agreement is not built in by the floors. But the no-position mesh has errors 2-3 times the quoted ones, it depends on a (h_kww 1.18-1.56 for a = 0.3-0.7), and the quadratic coefficients move with the quartic terms (the depth at 150 mm does not).
(ii) **Cathode nearly flat along y (c_kww 0.22 +- 0.13) and follows the board along x (c_kuu 1.06 +- 0.10): weakened.** With resampling errors c_kuu = 1.06 +- 0.23 and c_kww = 0.22 +- 0.34; the y-curvature is zero at a = 0.3 and 0.5 mm deep at a = 0.7, so 'flat along y' is allowed (H8: 1.6 sigma) but not demonstrated, and 'follows the board along x' rests on w0 and v (not on the geometric gap, which covers two columns); `geo` gives 0.01 +- 0.45 (1.9 sigma lower), and one quartic term on the cathode brings c_kuu to 0.18 +- 0.33. What is stable is the depth: the cathode is 1.2 +- 0.3 mm lower at +-150 mm along x than at the centre (with a gap curvature +0.28 +- 0.27 consistent with zero), and not flat (H1 rejected, 4 sigma even after the resampling correction).
(iii) **beta = 1.0 +- 0.2: weakened.** Fit 1.02 +- 0.18, jackknife 0.26, bootstrap 0.25; across model order 0.96-1.28; `timing` 1.26-1.42 (+-0.3-0.47) and `gap` 1.00 +- 0.33; Magboltz 0.68 is 1.3 sigma away with the resampling error. It is compatible with 0.68 and with 1.3.

## Caveats and what I could not do
- Cell scatter in the classes is spatially structured and shared between classes; the jackknife over 16 cells captures it only with 16 'independent' units (cells of one chamber are not independent; neighbouring cells share the board dish). The true errors may be larger than my resampling errors.
- The floors were not re-derived per class beyond what `joint.main` does; chi2 differences between nested models depend on them (section 5).
- GY and dY exist in only 3 cells for run 1; any claim resting on `geo` for the Y view in run 1 is a single row.
- I did not test the Y-strip blur as the origin of the geo_gap sign.
- A real straight-line fit with early-weighted mixing was not done (a is fitted, not derived).
