# leadshield_compare — did the 2026-08-04 lead removal lengthen DAQ saturation?

On the morning of 2026-08-04, during an access, a large amount of shielding
lead was removed from the setup (run_132 was operator-killed at 08:44 for it).
Concern: more gamma-flash radiation reaching the detectors/electronics ->
longer post-flash saturation -> less track efficiency at early
time-since-flash (~1-5 ms).

## Runs (identical config: stat090 PRODUCTION at the run_67 optimum)

| run | period | when | sub-runs |
|---|---|---|---|
| run_130 | before | Aug 3 18:08-22:08 | 5 (night-to-night control) |
| run_132 | before | Aug 3 22:33 - Aug 4 08:45 | 11 |
| run_139 | after  | Aug 4 22:01 - Aug 5 08:58 | 11 |

All: PS+SINGLES, drift 700 V, resist A540/B540/C525/D520, 0.90 MIP, RAW
20 smp x 60 ns, latency 27, Hwm 1/Lwm 0, Ar/Iso 90/10, 3He, no beam filter.

## Pipeline

```
process.py       # reco cache per sub-run (ntof_tracking.reco chain, verbatim
                 #   run67_scan conventions; --jobs 2 on this 15 GB box)
feu_presence.py  # per-event readout_*/live_* flags  (RERUN WITH --force AFTER
                 #   PROCESSING MORE SUB-RUNS — stale table silently drops
                 #   events from every denominator)
compare.py       # everything: acceptance vs dt, blindness vs dt, boxcar
                 #   efficiency curves, fixed-window z-tests, intensity match,
                 #   flash-leader size, first-accept quantiles, SUMMARY.md
```

`lib.py` is run67_scan/scan_lib.py adapted to stat090 sub-run names (no HV
axes; drift fixed 700 V; meta = run/subrun/seq/period). Efficiency metric and
conventions are run67_scan's exactly: P(3D x/y pair) per recorded trigger,
denominator = readout_*, blind_frac an observable never a cut, Det A (clean
M1) the reference, B/C/D single-plane yields noise-inflated (bad M1).

Outputs -> `<ANALYSIS_DIR>/lead_shielding_compare/` (= ~/beam_july/analysis/…).

## Result (2026-08-05)

**No lengthening.** Det A t50 (half of the 40-76 ms plateau) 5.44 +/- 0.13 ms
before -> 5.43 +/- 0.12 ms after; the before-vs-before control moved 20x more.
95% upper bound on any increase +0.32 ms (<6%). Nothing in 1-5 ms reaches
1.5 sigma. Acceptance, first-accept time, blindness and flash-leader size all
agree. Written up in `<OUT_BASE>/RESULT.md`, headline plot
`figures/VERDICT_detA.png`. Caveat: the accept gate opens at 1.00 ms, so this
bounds the 1-76 ms recovery only — a sub-ms change is invisible here and would
need a flash-trigger run (run_18/45 style).

## Reading the result

The saturation-time signature is EARLY-dt-only: acceptance and/or efficiency
down at 1-5 ms in run_139 with the 40-76 ms window unchanged. A uniform shift
at all dt is a different story (gas, HV, beam mix — check run_130 vs run_132
for the night-to-night scatter and the intensity-matched curves before
believing either). If the flash-leader size distributions differ, the flash
itself changed at the detector — report that alongside.
