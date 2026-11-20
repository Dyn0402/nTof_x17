# The SPS known-angle rows against the bench law — 2026-11-18

Question (IIb §5 `\tbw`, HANDOFF 11-13): chamber E at 25.4° in Ar/CO2/iso reads a hits-ladder / waveform-fit slope
ratio of 1.07–1.13 on the inclined (Y) view, where the bench law 1 + C/N² gives 1.03–1.05. Does the beam need its own
arm (gas, field), or is it something else?

**Answer: the ladder follows the bench law. The excess is hits that are not on the ladder.** Existing data only
(the 11-09 window pickles, EOS `hits_daq/windows/sps_*.pkl`; desktop copy `$WORK_DATA/paper2_hits/sps/`).

> **Correction 2026-11-20 (GAS9352): the gas Dp was 0.006 from a Magboltz grid for Ar/CO2/iso 95/3/2, a wrong label; the
> gas was the ATLAS NSW mixture 93/5/2 (`docs/GAS_MIXTURES.md`).** On 93/5/2, at each row's composition from the SPS gas
> fit (run57 1.69 % water / 322 ppm O2, run59 1.75 % / 282 ppm) and the row's own v, `sps_resim.gas_dp` gives
> D_T = 179 / 175 / 172 / 170 / 169 µm/√cm and **Dp = 0.0049 / 0.0051 / 0.0052 / 0.0054 / 0.0053** (d550 … d700, run59).
> Everything that used the gas Dp was rerun at those values; every other arm reproduced to the last digit.
> - "model, Magboltz Dp" column: 1.040 / 1.032 / 1.027 / 1.024 / 1.023 → **1.038 / 1.030 / 1.025 / 1.023 / 1.022**;
>   "gas Dp, RC off": 1.026 / 1.021 / 1.016 / 1.015 / 1.015 → **1.023 / 1.019 / 1.015 / 1.014 / 1.014**.
>   Data without far hits minus model: +0.008…+0.011 → **+0.009…+0.013**. Model far hits per plane 0.23 / 0.30 → 0.24 / 0.30
>   (d550 / d700). `results/sps_resim.json` (now written by `sps_resim.py --summary`) and `results/sps_outliers.json`.
> - **Dp refit** (`sps_dp_scan.py --dp 0.003,gas,0.006,0.009,0.014,0.020`, 200 planes per view, the 0.006 column
>   reproduces the table below exactly): at d550's gas value 0.0049, **74 % (X) / 81 % (Y)** of planes have χ² below their
>   χ²(0.020) (75 / 81 % at 0.006), median Δχ²/χ² **−13 % / −18 %** (−13 / −18), 69 % / 80 % have their best χ² at ≤ 0.009;
>   median |w| 0.192 / 3.599 µm/ns against 0.229 / 3.543 at 0.020, so **the Y slope moves by 1.6 %** (1.4 % at 0.006).
>   At d700's 0.0054: 80 % / 83 %, −13 % / −12 %, 94 % / 92 % of planes best below 0.020, Y slope **+0.65 %** (4.859 against
>   4.828; 0.6 % at 0.006). The summed χ² still favours 0.020. Conclusions unchanged.
> - The old products are kept beside the new ones: `$WORK_DATA/paper2_hits/sps/resim_sps_dp0.006_95-3-2.parquet`,
>   `dpscan_d{550,700}_95-3-2grid.parquet`.
> The numbers below are as first written.

## Numbers (inclined view, 3000 planes per row)

| row | field [V/cm] | v [µm/ns] | data, all hits | data, robust line | data, hits within 4 pitches of the track | model, Magboltz Dp 0.006 | model, pass Dp 0.020 | model, gas Dp, RC off | geometry only |
|---|---|---|---|---|---|---|---|---|---|
| run57 d550 | 191 | 7.5 | 1.134 | 1.048 | 1.050 | 1.040 | 1.098 | 1.026 | 1.009 |
| run57 d600 | 208 | 8.4 | 1.103 | 1.041 | 1.042 | 1.032 | 1.083 | 1.021 | 1.008 |
| run57 d650 | 226 | 9.2 | 1.092 | 1.037 | 1.038 | 1.027 | 1.071 | 1.016 | 1.006 |
| run57 d700 | 243 | 10.1 | 1.087 | 1.033 | 1.035 | 1.024 | 1.062 | 1.015 | 1.004 |
| run59 | 243 | 10.0 | 1.076 | 1.031 | 1.031 | 1.023 | 1.062 | 1.015 | 1.004 |

"data, all hits" is the `res`/`base` arm of `sps_resim.py` (the plane's own residual added back to its own model, i.e.
the data with 4 padded strips a side); it reproduces the tiers tables' V0 (1.129 / 1.099 / 1.091 / 1.082 / 1.074) to
0.005. The model columns are the `noise` mode: each plane's fitted (p0, w, t0, q) through `model_waveforms` with white
noise, no residual. Statistical error of a median ratio: 0.003–0.004.

1. **Far hits.** 60–65 % of the planes have leading-edge hits more than 4 pitches from the fitted track in the
   (strip, time) plane: 1.8–2.2 per plane, small (the hits more than 180 ns off the robust line have a median amplitude of 0.12 of
   the plane's largest; 0.26–0.29 for the kept ones; 62–65 % of the rejected hits are on the mesh side of the track). The one-track model with noise and the
   data's zero suppression makes 0.2–0.3 per plane. Any hit uncorrelated with the ladder lowers the regression slope
   dt/dx, so the line reads steeper whichever side the hit is on (`sps_outliers.py`, `results/sps_outliers.json`).
2. **Without them the plain amplitude-weighted line equals the robust line**, 1.031–1.050, which is the bench law for
   N = 16–17 (1.03–1.05).
3. **The pure model with the gas's diffusion reads 1.023–1.040**, falling with field as the data do (the faster column
   puts more strips in the window: 13.2 → 15.6 hits). Data without far hits minus model: +0.008…+0.011 on every row
   (one unit of C is 0.005 at this N; the bench toy is also about one unit below the data).
4. **RC is a small part here:** model with the gas Dp 1.040 → 1.026 with RC off (d550). A leading-edge time is taken
   early, and the RC variance grows with the time since landing, so the long drift of the slow gas does not multiply it.
5. **What the far hits are is not established** (second beam particles in the 3.84 µs window, δ electrons). They are
   also where the waveform fit's tail is: W within 5° of the stage angle is 92–93 % on planes with no far hit, 86–90 %
   with one, 51–56 % with two or more (a third of the planes; their χ²/dof is twice the others'). That answers the
   IIa §6 question about W's 77–79 % in5 at SPS against 96–97 % on the bench.

## A model finding on the way: the SPS rows carry the bench Dp

The SPS pass rows are fitted with chamber E's bench bundle, `Dp = 0.020 mm/√ns` (window meta `hyper`; registry: "bench
det4 calib_bundle … v = v_span"). Dp is a gas constant and does not transfer (model_audit F20). For the beam's CO2
mixture with 1.6–1.7 % water (cloud_basics FINDINGS §18), Magboltz (`cloud_basics/results/air/magboltz_co2_w1p7_a0p1.json`)
gives v = 8.8 / 11.3 µm/ns and D_T = 184 / 179 µm/√cm at 200 / 243 V/cm, i.e. Dp = D_T √v = 0.0055–0.0060: the pass
value is 3.5× too large (12× in variance). With the pass value the pure model would read 1.06–1.10 and look like an
explanation of the plain line's 1.07–1.13; that agreement is accidental.

Refit at fixed Dp (`sps_dp_scan.py`, pass fitter, everything else as the pass): see the section below.

Result on run57 d550 (191 V/cm), 200 planes per view, Dp = 0.003 / 0.006 / 0.009 / 0.014 / 0.020:

| view | planes with χ²(0.006) < χ²(0.020) | median Δχ²/χ² at 0.006 | planes with their best χ² at ≤ 0.009 | median \|w\| at 0.006 / 0.020 [µm/ns] |
|---|---|---|---|---|
| X (normal incidence, no RC) | 75 % | −13 % | 69 % | 0.20 / 0.23 |
| Y (inclined) | 81 % | −18 % | 80 % | 3.593 / 3.543 |

- **The typical plane prefers the gas value** over the bench value, in both views.
- **The summed χ² goes the other way** (lowest at 0.020): it is carried by a minority of planes with large χ². On Y the
  planes that prefer 0.020 have a median χ² of 21 000 against 8 500 for the others; a wide diffusion absorbs charge that
  is not on the track (the far-hit planes above). A summed-χ² hyperfit on beam rows would therefore pick too large a Dp.
- **The fitted slope moves by 1.4 %** (Y) between 0.020 and 0.006. On these rows v is set from W at the stage angle, so
  the angle is unaffected by construction; where v is independent (the CF4 known-angle rows) a 1–2 % scale shift is
  inside the quoted 0.98 ± 0.03, but it has not been measured there.
- **run57 d700 (243 V/cm) agrees:** the median plane's χ² is lower at every Dp below 0.020 in both views, 91–93 % of planes
  have their best χ² below 0.020, and the Y slope moves by 0.6 % (4.855 against 4.828 µm/ns).

## Files

- `sps_resim.py` → `$WORK_DATA/paper2_hits/sps/resim_sps.parquet`, summary `results/sps_resim.json`
- `sps_outliers.py` → `results/sps_outliers.json` (also holds the plain line without far hits, per row)
- `sps_dp_scan.py` → `$WORK_DATA/paper2_hits/sps/dpscan_d{550,700}.parquet`
- Figure: IIb `fig:knownangle` (`papers/tpc_mixing_review/scripts/make_figures.py:known_angle`)
- Text: IIb §5 "A known angle in the beam" and conclusions; IIa §6 (W's tail) and §4.3

## Not done / limits

- The far hits are not identified. The X view (normal incidence) of the same events could say whether a second track is
  there; not looked at.
- The model columns use each plane's q fitted under the pass Dp; a plane refitted at the gas Dp would have a slightly
  different q. The ratio is insensitive to q's detail (it is a median of slopes), not tested.
- CF4 rows (the 13 known-angle settings of Paper I) also carry the bench Dp; not looked at here.
