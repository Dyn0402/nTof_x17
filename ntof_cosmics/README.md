# ntof_cosmics — the beam-off cosmic runs, and the through-going-particle background

**Started 2026-10-02.** Prompted by the ILL feasibility study
(`~/PycharmProjects/x17_facility_search/ill/`), where cosmics came out as a
substantial background with two handles (arm-to-arm time of flight, and
collinearity). The question was whether we have done the same on the n_TOF
data. The answer: partly. `sept26_prelim_analysis/tight_coincidence.py` flags
opposing pairs above 170° (`BACK_TO_BACK_DEG`) as `back_to_back`. Nothing uses
the timing direction, and the beam-off runs have never been read.

| file | what |
|---|---|
| `inventory.py` | one row per cosmic sub-run: era, HV, noise condition, A-x dead connector, n_TOF beam pulses (edge vs interior), n_TOF runs recording alongside → `results/cosmic_subruns.csv`, `results/cosmic_runs.csv` |
| `make_report.py` | `results/report.html` from those CSVs |
| `clock_match.py` | beam-off DREAM triggers onto the n_TOF clock, staged (coarse → drift → κ → per-bunch δ_b, leave-one-out) → `results/clock_match/` |
| `make_clock_report.py` | `results/clock_match/report.html` + figures |
| `cosmic_tracks.py` | per-sub-run fetch / build (borrowed k) / analyse of the full-pass reco → `results/tracking/k_<run>/` |
| `pool_tracking.py` | pools the per-sub-run pairs of a run (bootstrap over sub-runs) → `results/tracking/pooled/` |
| `angle_response.py` | the A/C angle response against the joined A–C line, resolution vs angle, `slope_reliable` test, beam test → `results/tracking/pooled/` |
| `make_pooled_report.py` | `results/tracking/pooled/report.html` + figures; current state in `HANDOFF_TRACKING_2026-10-06.md` §7 |
| `make_deck.py` | the slide note, live at <https://dylan-neff.web.cern.ch/notes/beam-off-cosmics.html> → `results/deck/beam-off-cosmics.html` |

Inputs: `dylan-cern-site/data/x17-runs.json` and `x17-match.json`; the local
`run_config.json` mirror; the `beam_class_*.csv` slow-control logs, copied to
`/media/dylan/data/x17/beam_july/slow_control/beam_intensity/` from
`/eos/experiment/ntof/data/x17/july_beam/slow_control/beam_intensity`; and the
n_TOF completed ledger in `ntof_processing/campaign_qa/results/`.

## What the inventory found (2026-10-02)

- **47 runs tagged cosmics.** Of these, 41 are at the production operating point
  (run_80–159), holding **57.6 h and 5.18 M triggers**. All are decoded on EOS
  (~1.9 TB in total) and all use the run_79 HV and readout. The trigger is
  scintillator singles with the veto open at about 25 Hz. One arm is enough to
  trigger.
- **The beam sits at run edges, not inside the running.** 35.2 h never saw an
  n_TOF pulse. In the other runs the pulses fall at the boundaries, from beam
  loss and beam return around the auto-substituted run. The only exception is
  run_73, a pre-production run. Veto by pulse time, not by the minute-level flag.
- **The best background sample:** run_149 (21.8 h), run_133, run_89, run_103,
  run_134, which together hold 41 of the 57.6 h. PLAN §S4 had named run_83 and
  run_146 (0.2 h and 0.35 h), which are a poor choice.
- **n_TOF was recording with no protons for 15.1 h**, almost all of it in run_149
  (8.6 h, n_TOF 224678–687) and run_103 (4.3 h, 224608–613). That is the only
  place a cosmic scintillator time, and so a time of flight, could come from.
  The clock match below shows it works there.
- **Pre-production runs** (54–74, 5 h) are at other HV and timing and are mostly
  resist ladders. **run_80** has chamber A x connector 8 dead. **run_148** is
  empty.

## How to think about the background — notes for the beam-on analysis

These are Dylan's framing (2026-10-02), plus the handles discussed alongside it.

1. **What makes it background is the topology, not where the particle came
   from.** A single particle that crosses the set-up in a straight line from one
   side to the other is one background category, whether it is a cosmic or
   something beam-related. Do not spend effort separating "cosmic" from "beam
   punch-through" for the X17 veto. The beam-off runs are useful as a clean,
   high-statistics sample of that topology: they show its signature and let us
   measure a veto's efficiency. They are not a separate background to subtract.
2. **The capsule vertex is a handle.** A through-going straight track has no
   reason to pass through the ³He capsule beyond what the acceptance geometry
   gives it. A real pair comes from the capsule. So the distance of closest
   approach of the combined line to the capsule, or the two legs' common vertex,
   should separate them: through-goers should fill it roughly flat, at whatever
   the geometric acceptance allows, while signal peaks on the capsule. The
   cosmic runs give that through-goer distribution directly, with no beam on
   top, to compare with the beam-on `back_to_back` sample. Use pointing
   estimators, not shape fits (CLAUDE.md, "Take positions from pointing").
3. **Collinearity.** This is already in use: `back_to_back` means an opening
   angle above 170°. The cosmic runs give the true opening-angle distribution of
   one particle through two arms, including scattering and resolution, so the
   170° threshold can be set from data instead of chosen.
4. **Direction-of-flight timing.** One particle through arms A and C arrives
   ~2–3 ns later at the second arm, always in the same order for cosmics
   (downward). A real pair arrives with Δt ≈ 0. The per-arm trigger resolution
   (~5 ns, so ~7 ns on the difference) probably rules out an event-by-event cut
   at n_TOF, but a shift of ⟨t1−t2⟩ in the >170° sample would show up
   statistically. The ILL design leans on this handle. At n_TOF it needs the
   scintillator times, which only exist in the n_TOF-quiet windows above.
5. **Smaller cross-checks.** In beam-on data, through-goers that are cosmics
   should be flat in time since the flash and mostly vertical. Beam-related ones
   should not. This is not needed for the veto (see 1), but it says what the
   >170° population is made of.

## Clock match without beam (2026-10-02) — works

run_149/cos_0000 ↔ n_TOF 224678: **2467 triggers matched within ±50 ns, 92.2 %
of the in-window triggers (beam-on 96 %), 10.6 ns core MAD, accidentals
unmeasurably small.** With no beam n_TOF free-runs (0.5009 s, 80 ms window) and
PKUP still stamps every bunch with `psTime`. The map is the beam-on one with
the flash replaced by psTime:

    t_DREAM_on_nTOF = t_log + S + td·(1+k)       S = +6.450 s, k = −3.8 ppm
    off = t_DREAM_on_nTOF − psTime_b = δ_b + tof·(1−κ)      κ = 116.6 ppm

δ_b, each window's offset from its psTime, is new: a slow wander (~280 µs
over the sub-run, curvature the straight drift line misses) plus ~29 µs MAD
of bunch-to-bunch jitter. It is fitted from the bunch's own triggers, and
validated leave-one-out. Gotchas found on the way:
- `combined_hits` holds only events with MM activity; use `decoded_root` FEU 01
  `timestamp` (10 ns ticks) for every trigger.
- Raw `tof` has no common zero across trees once `tflash` is meaningless: the
  plastics sit ~35–42 ns late, outside the 20 ns AND. Measured in situ and
  shifted back (`measure_pss_shift`).
- 7.5 % of quiet bunches have psTime = 0 (dropped for now).
- The DAQ log's "Subrun started" line (ms) is the anchor. On beam sub-runs,
  file-name minute + `pulse_match` offset − log line − 0.829 s (NXCALS lag) puts
  DREAM go 5.5–13.4 s after it on the psTime clock (14 of 21 sub-runs). The
  other 7 sit at −100 s or +21–24 s: supercycle mislocks in the local
  `ntof_july_analysis/cache_pulse_match/`, which predates the 2026-08-12 lock
  fix. Fine as a prior, never as a constant; S per sub-run is a ±60 s search.
  Log lines and file names: `results/clock_match/beam_on_log_starts.txt`.

## Reproduce

Stage the inputs (from a machine with a CERN Kerberos ticket; lxplus has no
uproot, so copy rather than read in place):

```bash
B=/media/dylan/data/x17/beam_july
# slow-control beam logs (inventory)
scp -r lxplus:/eos/experiment/ntof/data/x17/july_beam/slow_control/beam_intensity $B/slow_control/
# one DREAM sub-run: FEU 01 decoded trees (~330 MB), its thresholds and the DAQ log
mkdir -p $B/runs/run_149/cosbounce_cos_0000/decoded_root
scp 'lxplus:/eos/experiment/ntof/data/x17/july_beam/runs/run_149/cosbounce_cos_0000/decoded_root/*_01.root' \
    $B/runs/run_149/cosbounce_cos_0000/decoded_root/
ssh lxplus 'cd /eos/experiment/ntof/data/x17/july_beam/runs/run_149 && tar cf - */n1081b_config.json dream_daq.log' \
    | tar xf - -C $B/runs/run_149
# the n_TOF run recording alongside it (official partials, ~255 MB)
mkdir -p $B/ntof_data/run224678.parts
scp 'lxplus:/eos/experiment/ntof/processing/official/completed/224678/run224678_*.root' $B/ntof_data/run224678.parts/
```

Then, from the repo root:

```bash
.venv/bin/python ntof_cosmics/inventory.py          # results/cosmic_{runs,subruns}.csv
.venv/bin/python ntof_cosmics/make_report.py        # results/report.html
.venv/bin/python ntof_cosmics/clock_match.py \
    --run run_149 --subrun cosbounce_cos_0000 --ntof 224678   # results/clock_match/*, ~15 s
.venv/bin/python ntof_cosmics/make_clock_report.py  # results/clock_match/report.html + figures/
.venv/bin/python ntof_cosmics/make_deck.py          # results/deck/beam-off-cosmics.html
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py \
    ntof_cosmics/results/deck/beam-off-cosmics.html --slug beam-off-cosmics --force --deploy
```

`clock_match.py` outputs, per (DREAM sub-run, n_TOF run): `summary_*.json`
(every stage's numbers), `pairs_*.csv` (every DREAM trigger within ±300 µs of a
candidate: arm, tof, offset, δ_b, leave-one-out residual), `delta_b_*.csv`,
`drift_pairs_*.csv`, `control_*.csv`.

The deck reads only these outputs (plus the pulse-match cache for the latency
prior), so the note is extended by adding slide functions to `make_deck.py`
and rerunning the last two commands.

## Next

**Picked up in [`HANDOFF_TRACKING_2026-10-02.md`](HANDOFF_TRACKING_2026-10-02.md):**
tracking first (run_149/cos_0000), the clock match scaled up alongside it.
**First tracks (2026-10-02 evening):** `cosmic_tracks.py`, report
`results/tracking/report.html`. Only 74–77 % of clean A–C through-goers pass
the 170° cut, and a capsule-free slope check says the borrowed k reads A ~8 %
shallow and C ~12 % steep (~40 events). The rest of run_149 is on condor;
progress and next steps in the handoff §5.

- **Clock match, the rest of it:** a smooth drift model (absorbs the slow
  δ_b); several n_TOF runs per sub-run (cos_0001 straddles 224678/224679, so key
  by (run, bunch)); recover the psTime = 0 bunches from the 0.5009 s grid; show
  k and κ transfer between sub-runs; then all of run_149 and run_103. The
  output is, per DREAM trigger, the n_TOF arm and time to ~10 ns.
- **Understand 92 % against the beam-on 96 %:** the unfitted per-bunch rate and
  the 418 single-trigger bunches are the candidates.
- **Tracking:** production tracking over run_149 first; count triggers with a
  track in two arms; their opening-angle and capsule-DCA distributions.
- **The three handles:** calibrate the 170° cut on data; the capsule-DCA shape
  of through-goers; the arm-to-arm Δt sign on two-arm cosmics.
- Read `hv_monitor.csv` for trips before trusting a sub-run.
