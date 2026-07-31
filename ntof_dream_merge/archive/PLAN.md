> # ⛔ RETIRED — do not build on this
>
> **Superseded by `DREAM_NTOF_CALIBRATION.md`.** Archived 2026-07-30.
>
> The original plan for the merge, written 2026-07-27. Phases 3 and 4 are closed and their numbers here (88 % bunch join, 37 ns match) are wrong: the join is 100 % and the match resolves to 6 ns. The data-location tables are still broadly right but are duplicated, current, in `../README.md`.
>
> **Read `../DREAM_NTOF_CALIBRATION.md` for the matching, and `../README.md` for where things live.**

---

# ntof_dream_merge — joining the n_TOF facility DAQ to the DREAM Micromegas stream

**Goal: one merged per-event record that carries the DREAM Micromegas tracks AND the
n_TOF-DAQ scintillator hits (SiPM wall / plastic / liquid) of the same beam pulse, at the
same time reference. Then run it at scale on condor, on a rolling basis as data arrives.**

Written 2026-07-27. Everything in "Verified facts" below was measured during planning,
not assumed — the numbers are reproducible from the commands quoted.

---

## 1. Where the two streams live

| | DREAM (Micromegas) | n_TOF facility DAQ |
|---|---|---|
| Raw | `/mnt/data/x17/beam_july/runs/run_NN/<subrun>/raw_daq_data` | `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/<run>/stream1/` (~2 wk retention, then CTA tape) |
| Processed | `<subrun>/combined_hits_root/*_feu-combined_hits.root`, tree `hits` | `/eos/experiment/ntof/processing/official/done/run<N>.root` |
| Size | ~1.2 GB combined hits per 60-min subrun (18 GB raw) | ~26 GB per ~2.8 h run |
| Unit | one **event** = one trigger accept | one **hit** = one PSA-fitted pulse, grouped by `BunchNumber` |
| Processing | DAQ `processor_watcher`, already done on disk | official `RunProcessing.sh` (auto, see `../ntof_daq_processing/PROCESSING.md`) |

Prior work read for this plan:
- `mx_july_beam_qa/` — the whole n_TOF-side analysis (README is the handover). Wall×plastic×LIQ
  coincidence machinery (02/03/07/19), `hitcache.py` fast reader, `calib/*.json` (ADC→mV,
  per-channel time offsets, Y-88 absolute energy scale).
- `ntof_july_analysis/pulse_match.py` — DREAM burst → PS pulse matching (this is the clock bridge).
- `ntof_tracking/TRACK_PLAN_07_timing_tof.md` — the intended time chain; this plan closes its
  risk item (see §3).
- `~/beam_july/analysis/tt_dream_match/FINDINGS_2026-07-18.md` — the earlier attempt at
  per-event cross-DAQ matching, via the N1081B TT stream. Read this before Phase 4.
- `~/beam_july/analysis/flash_timing_threshold/flash_timing_lib.py` — burst/flash anchoring.

---

## 2. Recommended first pair: DREAM `run_79` (subruns 0000+0001) ↔ n_TOF `224572`

**Why this one:**

1. **n_TOF 224572 is already officially processed** — `done/run224572.root`, 26 GB, written
   2026-07-27 10:07. Trees `PKUP index DAQsettings SILI WALA-D PSSA-D LIQA-D` — the **LIQ trees
   are present**, i.e. the official 2026 PSA is now the LIQ-enabled Mucciola UserInput. No condor
   run needed to start; we get to work on the *merge* rather than on processing.
2. **Full time containment.** 224572 ran 18:04:24 → 20:53:35 local (2026-07-26). DREAM `run_79`
   `stat090_0000` = 18:07:19–19:07:29 and `stat090_0001` = 19:07:43–20:07:53 are both entirely
   inside it. Two clean hours.
3. **run_79 is the production operating point** — the run_67 optimum, no scan axis, every subrun
   identical (drift 700 V, resist A540/B540/C525/D520, latency 27, n_samples 20, RAW readout,
   Hwm 2/Lwm 1, PS+SINGLES trigger). Whatever we build here is what the rolling pipeline runs.
4. **Small DREAM side**: 1.2 GB combined hits per subrun → trivially laptop-portable.
5. **The acquisition windows now overlap completely** — see §3.

**Why not run_79 in full:** its 16 subruns span 18:07 on 7/26 → 10:00 on 7/27 and therefore
straddle **eight** n_TOF runs (224572–224579), of which 224573 and 224576 are not yet in `done/`.
Restricting to subruns 0000+0001 keeps it to a single n_TOF run.

**Why not run_81 first:** it is actually the *cleanest* containment (`stat090_0000`
11:28:45–12:28:56 sits wholly inside 224580, 11:29:03–12:43:43, one subrun ↔ one n_TOF run) but
**224580 is not processed yet** (62 raw files on EOS, run ended 12:43 today). Keep run_81 as
**run #2** — it is the natural "does it generalise" test and the first real exercise of the
condor path.

Run→n_TOF mapping is machine-readable from
`~/beam_july/slow_control/stream1_filesize/stream1_waveform_<date>.csv`, which the DAQ's own
monitor writes: columns `timestamp,run,seq,event,det,...` where `run` is the **n_TOF run
number**. Group by `run` for its time span. This is the lookup the rolling pipeline should use.

---

## 3. The time chain — verified

```
DREAM event  --(a)-->  burst (beam pulse)  --(b)-->  PS pulse  --(c)-->  n_TOF BunchNumber
     |                      |                                                  |
     +--(d) t_since_flash --+                                                  |
                                            n_TOF hit: t_since_flash = tof - tflash
```

**(a) Burst clustering — solid.** `trigger_timestamp_ns` (10 ns granularity), 0.5 s gap split.
`run_79/stat090_0000`: 1012 bursts, 106 127 events, median 113 events/burst. The first event of
each burst is the PS/flash trigger (t=0); the N93B gate then admits singles from ~1 ms.
Measured flash→first-single = **1.0060 ms** median (p5–p95 0.9947–1.0376) — the 1 ms gate edge,
a hard landmark.

**(b) Burst → PS pulse — solid, already implemented.** `pulse_match.match_subrun('run_79',
'stat090_0000')` → offset +27.917 s, **1012/1012 bursts matched (100 %)**, residual RMS 6 ms,
median intensity 850e10.

**(c) PS pulse → n_TOF bunch — verified this session.** The DAQ's beam-intensity CSV and n_TOF
`PKUP.psTime` are the **same pulse stream with a rigid offset**:

```
psTime/1e9  =  beam_intensity CSV unix_ts  -  0.829 s     (MAD = 0.0000 s over 3018 bunches)
```

and the intensities agree to 0.1e10 pulse by pulse (854.8 / 843.9 / 412.4 / 849.2 …). The
0.829 s is NXCALS publication latency in the CSV, not a clock error.

Joining DREAM bursts directly to `psTime`: **886/1012 bursts land on a unique bunch, zero
duplicates**, residual MAD 6 ms after the constant offset. The 126 non-matches sit at exactly
−1.2 s (one PS basic period) = **beam pulses DREAM took that n_TOF did not record** (n_TOF
recorded 3018 bunches at 2.4 s median spacing while pulses arrive at 1.2 s multiples). That is a
real ~12 % n_TOF-side acceptance, to be *tracked as an efficiency*, not debugged.

**(d) Common t=0 = the gamma flash — the acquisition windows now match.** n_TOF `tof`/`tflash`
are in **ns** (`tflash` ≈ 13 330 ns, spread 13 322–13 333 over the run). Max `tof` = 79 999 663 ns:
the n_TOF acquisition is **80 ms/bunch** (it was 20 ms back in mid-July — it has been extended).
The DREAM N93B gate is 1–81 ms, bursts span 73 ms median. **The two windows now cover each other
completely** — 100 % of DREAM events are inside n_TOF's acquisition, not the 58 % a 20 ms window
would have given.

**This closes the TRACK_PLAN_07 risk item** ("is `trigger_timestamp_ns` good at ns level?") from
the *coarse* side: bunch identity is unambiguous. The remaining question is per-event alignment
inside the burst, which is Phase 4.

---

## 4. Phases

### Phase 1 — DREAM event table (local, ~1 h)
Per event, from `combined_hits_root` + `beam_track_finding.py`:
`run, subrun, eventId, burst_id, is_flash, t_since_flash_ns, n_hits, n_sat, bunch_number,
pulse_e10` plus the reconstructed tracks (per-detector clusters → X/Y pairs → 3D segments, the
existing `ntof_tracking/reco` + `run67_scan` path). Write parquet under
`~/beam_july/analysis/ntof_dream_merge/`.
Reuse: `pulse_match.py`, `flash_timing_lib.py`, `track_rate_hv_time_intensity/build_cache.py`.

### Phase 2 — n_TOF slim (lxplus, then pull down)
1. Full `PKUP` + `index` dump (3018 rows — already done for 224572, cached at
   `~/beam_july/analysis/ntof_dream_merge/cache/pkup_224572.csv`).
2. Slim the 12 scint trees to `(BunchNumber, tree, detn, t_since_flash_ns = int32(tof−tflash),
   amp, area, satuflag, pileup1)`. Full run ≈ 610 M hits ≈ 8.5 GB packed — so for local
   development slim **only ~200 bunches** (≈ 300 MB, enough for alignment + the first
   coincidence peak), and defer the full run to Phase 6 where the DREAM event list filters it to
   ~1 % of the volume.
Run it as the established lxplus/HTCondor job (`mx_july_beam_qa/lxplus/`); needs `kinit` on
lxplus (AFS home is currently unreadable without Kerberos — ssh key alone is not enough).

### Phase 3 — Bunch join + QA (local)
Implement the §3 chain as one function `dream_event_to_bunch(run, subrun) -> table`. QA figures:
burst-residual histogram (expect a spike at 0 plus a −1.2 s satellite), match fraction per
subrun, and **`PKUP.PulseIntensity` vs `pulse_match` intensity per event** — a free
per-pulse verification that the join is right, since the two come from independent files.

### Phase 4 — Intra-burst alignment (the real work)
The DREAM trigger is **PS + SINGLES**: every DREAM event was triggered by a scintillator single
whose PMT the n_TOF DAQ *also digitised*. So for each DREAM event there must be an n_TOF
plastic/wall hit above the trigger threshold at a fixed offset. Build the Δt histogram
(DREAM `t_since_flash` − nearest n_TOF scint `t_since_flash`) per arm, fit the peak, measure the
accidental pedestal **as a function of t_since_flash** (hit density is high early on: PSSA alone
has ~32 k hits/bunch). This is exactly `mx_july_beam_qa/02_coincidence_scan.py` machinery applied
across DAQs — sideband subtraction included.

Two known hazards, both already documented:
- **Accept-time artifacts.** `tt_dream_match` found that at run_53 settings intra-burst accepts
  were quantised at ~4 µs (pipeline drain) and only burst *leaders* were physical. **Re-checked
  at run_79 settings: the pathology is largely gone** — intra-burst Δt is smooth (median 322 µs,
  p5 51 µs, min 3 µs) with only 0.07 % of Δt in the 3.8–4.4 µs band. Re-verify per run before
  trusting per-event times.
- **Fallback if the Δt peak will not resolve**: the order-based assignment from
  `tt_dream_match` (k-th DREAM accept ↔ k-th surviving n_TOF trigger candidate in the window,
  plus a DREAM busy model). Decide from the measured peak/pedestal ratio, not a priori.

### Phase 5 — Merged record
For each DREAM event (especially those with a reconstructed MM track): the matched arm's wall /
plastic / LIQ amplitudes in **mV and MeVee** (`mx_july_beam_qa/calib/` already has ADC→mV,
per-channel time offsets, and the Y-88 absolute scale), pulse intensity, `t_since_flash`, and
**E_n from ToF over the 19.5 m EAR2 path** (`neutron_energy_vs_flight_time.py`). This is the
deliverable TRACK_PLAN_07 asked for. Validation figure: track rate and scint-tagged track rate
vs `t_since_flash`, with the E_n bands overlaid — the mid-window turn-off and the ³He capture
flood must show up where the July QA says they do.

### Phase 6 — Scale (condor)
Split by where the data is: **DREAM tracking local / on the DAQ machine** (data is on
`/mnt/data`), **n_TOF slimming on lxplus HTCondor** (data is on EOS). Push the small DREAM event
table *up* (≈1 MB per subrun), do the bunch join and the window filter on lxplus, pull down only
the matched hit table. That reduces 26 GB → tens of MB per run.

### Phase 7 — Rolling
Driver keyed on `stream1_filesize/stream1_waveform_<date>.csv` (n_TOF run number + span) and
`done/`: for each new DREAM subrun, find its n_TOF run, wait for `done/run<N>.root`, submit the
slim+join job, land the merged table. Reuse the DAQ repo's `processor_watcher` pattern.

### Phase 8 — Laptop
Portable bundle = DREAM combined hits for the chosen subruns (2.4 GB) + slimmed n_TOF tables
(hundreds of MB) + `calib/`. No 26 GB file ever needs to reach the laptop. See §5.

---

## 5. Staging (`stage_reference_pair.sh`)

Everything below runs on the DAQ machine (`mx17`); data lands under
`/mnt/data/x17/beam_july/`. **No Kerberos needed** — the n_TOF official processing area is
world-readable over xrootd, and `xrdcp`/`xrdfs` are already on the box. (`kinit` is only needed
for the Phase 2/6 *condor* jobs on lxplus.)

```bash
cd ntof_dream_merge
./stage_reference_pair.sh ntof      # xrdcp done/run224572.root -> ntof_data/ (26.1 GB, ~50 min @ 9 MB/s)
./stage_reference_pair.sh pkup      # per-bunch index straight off EOS (3018 rows, seconds)
./stage_reference_pair.sh denom     # portable decoded_root denominators (~0.5 MB/subrun)
./stage_reference_pair.sh check     # what is staged, with a byte-exact size check vs EOS
./stage_reference_pair.sh manifest  # laptop bundle file list + the rsync command
```

`ntof` uses `xrdcp --continue`, so it is safe to re-run after an interruption; `check` compares
against `xrdfs stat` so "COMPLETE" means byte-exact, not merely present. Override the pair with
`NTOF_RUN=` / `DREAM_RUN=` / `DREAM_SUBRUNS=` for run #2 (run_81 ↔ 224580).

Staged so far:

| | |
|---|---|
| `ntof_data/run224572.root` | 26 108 826 793 B |
| `runs/run_79/stat090_0000/combined_hits_root/` | 13 files, 1.2 GB — 1012 flashes, 106 127 events |
| `runs/run_79/stat090_0001/combined_hits_root/` | 14 files, 1.2 GB — 1049 flashes, 109 354 events |
| `analysis/ntof_dream_merge/cache/pkup_224572.csv` | 3018 bunches |
| `analysis/ntof_dream_merge/cache/denom_run_79_stat090_000{0,1}.npz` | portable denominators |

**The laptop bundle is ~28.5 GB** and deliberately excludes `raw_daq_data` (18 GB/subrun) and
`decoded_root` (10 GB/subrun). `decoded_root` is only needed for the trigger denominator, which
is why `denom` pre-extracts it: note that `flash_timing_lib.load_subrun` keys its own cache on an
**md5 of the decoded-file list**, which a machine without `decoded_root` can never reproduce — so
the portable copies are keyed on run/subrun only, and Phase 1 should read those.

## 6. Gotchas to carry in

- `tof` and `tflash` are **ns**, not µs (the mx_july README's "flash at 10.8–11.9 µs" is the same
  number in different units). n_TOF acquisition is **80 ms** now, was 20 ms in mid-July — check
  per era.
- **1.3 % of bunches have `psTime <= 0`** (40/3018 in 224572, one contiguous block 2038–2077 =
  one segment). `psTime` *is* monotonic in `BunchNumber` for the good ones, so interpolate.
- **~12 % of DREAM bursts have no n_TOF bunch** (n_TOF didn't record that pulse). Expected.
- DREAM hit `time` has diverged-pulse-fit outliers (seen here: −500 ms … +143 ms in a 1200 ns
  window) — affects **all** reprocessed runs; `beam_track_finding.py` must keep filtering them.
- Official n_TOF files have an **empty `waveform` branch** — hit-level only. Fine for this;
  a waveform need means private reprocessing.
- lxplus needs `kinit` for AFS/EOS; ssh key alone gives a shell but `/afs/.../.bashrc` is
  unreadable and EOS writes will fail.
- Local `psTime`→UTC: `psTime/1e9` is UTC; the DAQ logs and scan windows are **local = UTC+2**.

## 7. Open items to confirm
- Which PSA UserInput the *official* 2026 processing uses — the LIQ trees are there, so it is
  Mucciola's or a descendant, but confirm it matches `../ntof_daq_processing/psa_userinput/`
  before mixing calibrations across eras.
- Whether the DREAM SINGLES trigger leg is the plastic OR, the wall OR, or both (sets which
  n_TOF tree is the primary matching partner in Phase 4). `M4.C = OR(Singles lemo0)` per run_80's
  config — trace the lemo.
