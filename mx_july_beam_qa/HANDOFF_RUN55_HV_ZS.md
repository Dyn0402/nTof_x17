# HANDOFF — run_55 resist-HV scan + ZS optimization (2026-07-19)

**For the next Claude model. Two complete analyses live here: (A) the resist-HV
scan of the DREAM Micromegas vs time-since-flash (operating-point search), and
(B) the zero-suppression threshold optimization simulated on the same raw
data. Read `HV_SCAN_RUN55_ANALYSIS.md` and `ZS_OPTIMIZATION_RUN55.md` for the
full stories; this file is the working context you need to continue.**

A third run_55 analysis (micro-TPC 3D tracking, scripts 27*) was done in a
parallel session — see `TRACKING_RUN55_ANALYSIS.md`.

## The run and the data

- run_55, 2026-07-18 19:11 → 23:26 CERN: cyclical resist scan r560→r520 V
  (−5 V, 9 pts), 10 min/subrun, all four dets stepped together. Drift fixed
  800 V B/C/D, **600 V A** (A sparks on drift at 800; Dylan: stable at 600,
  maybe 700 max — that's the ceiling). Ar/iso 90/10, ³He target, no filter.
- Trigger: scint DOUBLES (≥2-of-4 wall×plastic) in a 30 ms beam gate
  (N93B), MM-independent. DREAM: 32 smp × 60 ns, latency 35, **no ZS**.
- The DAQ machine DIED mid-run (subrun `c02_024` in flight). Data had
  auto-synced to EOS: `lxplus:/eos/experiment/ntof/data/x17/july_beam/runs/
  run_55/` — 24 complete subruns survived (3 cycles r535–560, 2 for
  r520–530), 76 214 triggers. Nothing else lost.
- Local: `~/x17/beam_july/runs/run_55/` has `combined_hits_root` +
  `decoded_root` (15 GB). `raw_daq_data` (~31 GB) and `hits_root` are EOS-only
  — pull with the rsync in the repo-root pattern (see git log or
  `download_new_runs.py`; ssh alias `lxplus`, kinit needed).
- Pedestal/threshold run used by analysis (B):
  `~/x17/beam_july/pedestals/pedestals_07-18-26_14-06-43/` (from EOS
  `july_beam/pedestals/`, taken 5 h before run_55).

## Pipeline (all in mx_july_beam_qa/)

| script | in → out |
|---|---|
| `25_hv_scan_extract.py` | decoded `_01` (trigger list) + combined_hits → per-event top-5 clusters/plane/det, burst t₀, t_since_flash → `cache/25_run55/*.npz` |
| `25b_hv_scan_analysis.py` | 25 cache → `figures/25_hv_scan/*` (11 figs), `calib/25_hv_scan_summary.json` |
| `26_zs_sim_extract.py` | decoded waveforms (all FEUs) + pedestal aux + 25 cache → ZS simulation `cache/26_run55/*.npz` (shardable: `<shard> <nshards>` args) |
| `26b_zs_analysis.py` | 26 cache → `figures/26_zs/*` (7 figs), `calib/26_zs_summary.json` |
| slides | `slides/hv_scan_run55_slides.tex` (beamer, compiled PDF committed) |

Key definitions: MIP-like cluster = 3–20 strips, ≤25 mm extent; MIP track =
x+y MIP clusters with sample-range overlap ≥ −2. Burst t₀ = first trigger of
each burst (= gate open = flash; valid for ALL bursts — the batch comb is
identical whether the MM railed empty or recorded flash garbage).

## Established facts a follow-up must not re-derive

1. **DAQ FIFO comb**: no-ZS events (~1.6 MB) → gate sampled only at
   0–0.5 ms, ~8–12 ms (b1), ~16–28 ms (b2). **2–6 ms (incl. the 3 ms physics
   target) unsampled.** ~2 ms/event readout, ~4–5 event FIFO.
2. **³He capture flood**: b1 sits ON the thermal peak (19 m / 2200 m/s =
   8.6 ms). n+³He→p+t (764 keV) blobs light det D in ≳90 % of late windows;
   at LOW HV the blobs shrink into MIP-sized clusters (D's "MIP" rate at
   520 V is captures — do not read D's low-HV points as efficiency). Arms
   are equidistant (pinwheel, geometry in `~/CLionProjects/MX17_Full_Geant/`);
   D's visibility = its ~2× gain.
3. **Saturation time grows with HV**: all dets dead 0–0.5 ms at every HV;
   recovered by ~8 ms at ≤550 V; at 555–560 V, C (strongly) and D (mildly)
   are still gain-sagged at 8–12 ms, recovered by ~20 ms. Early amplitude
   suppression shows as rate loss + survivor-biased-high medians.
4. **Det A declines through the gate** (opposite of C/D) — open; suspect
   space charge in the 600 V drift. Det B has wide (~200 mm) low-amp
   ringing clusters at all HVs — blocks its efficiency reading; waveform
   diagnosis needs `raw_daq_data`/`hits_root` (EOS).
5. **Operating points (for ≥3 ms goal)**: A ≥560 (not plateaued; also try
   drift 700), B 550–555 (tentative), C 545–550, D 540–550.
6. **ZS mechanism**: thr_ch = ped + N·σ_ch from beam-off pedestal run
   (`Sys PedRun Threshold`, deployed 5.0); ZsTyp=1 tpc, ZsChkSmp=4; runs
   with Pd subtraction ON (CmOffset 256). Parse `_thr.aux` with
   `C\s*\d+` channel field and keep the LAST of 2 blocks per file.
7. **CM correction is mandatory for ZS in beam**: in-beam wander =
   10–20× beam-off σ, >99 % coherent per Dream chip (64 ch). Without
   `Feu_RunCtrl_CM=1`, even 5σ keeps 60–95 % of channels. With per-Dream CM,
   residual = 4.1 ADC ≈ beam-off σ. Per-FEU CM is insufficient (36 ADC).
   CM also absorbs the +38 ADC post-flash baseline shift.
8. **ZS recommendation**: CM=1 + **3.5σ** → ~9.4 % sample volume,
   ~0.2 ms/event (10× speedup — closes the 2–6 ms comb gap). Strip survival
   A 99.7 / B 92.4 / C 93.6 / D 96.8 %; B/C/D losses are
   threshold-INdependent CM signal bias in busy windows (firmware will do
   the same) — going below 3σ buys nothing.
9. `common/Mx17StripMap.py` now skips detectors without `dream_feus`
   (July configs list plastics/SiPMs) — needed to load run_55's config.
10. Decoded `timestamp` = 10 ns ticks (×10 = `trigger_timestamp_ns`).
    Decoded files `_0N.root` = FEU N, tree `nt`, all triggers (incl. the
    27 % with zero stored hits — keep them in efficiency denominators).

## Open threads (concrete next actions)

1. **Next scan with ZS** (waiting on DAQ machine repair): CM=1, 3.5σ,
   short no-ZS reference subrun for a data-driven survival check. This
   samples 1–8 ms and answers "alive at 3 ms" directly.
2. **Det A drift test at 700 V** (its resist curve hasn't plateaued and its
   in-gate decline may be drift-field starvation).
3. **Higher resist point for A** (565–575) once drift question settled.
4. **Det B ringing autopsy** — needs waveform-level data from EOS.
5. **Check FEU CM algorithm** for outlier/hit rejection (would recover the
   3–7 % busy-window strip loss). CentOS FEU docs / Irakli.
6. If absolute per-arm efficiency wanted: tag-and-probe across arms
   (doubles → correlated pair; 16-pattern likelihood; lib in
   `ntof_tracking/`).

## Memory pointers

Auto-memory: `july-run55-hv-scan.md`, `july-zs-optimization.md` (+
`july-run55-microtpc-tracking.md` from the parallel tracking session).
