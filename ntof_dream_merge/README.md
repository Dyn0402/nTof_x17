# ntof_dream_merge

Joining the **n_TOF facility DAQ** (SiPM wall / plastic / liquid scintillator hits, official
hit-level ROOT on EOS) to the **DREAM Micromegas** stream (tracks, on `/mnt/data`), so that one
merged per-event record carries both — then running it on condor, rolling, as data arrives.

**Start with [`HANDOFF_2026-07-27_dream_ntof_matching.md`](HANDOFF_2026-07-27_dream_ntof_matching.md)**
— current state, what is closed, what is open, and the bugs to know about.
[`PLAN.md`](PLAN.md) is the original plan; parts of its §3 and §6 are superseded
by the handoff (the bunch join is 100 %, not 88 %, and the `psTime` repair is
different).

## Quick start

```bash
./stage_reference_pair.sh check      # what is staged for the reference pair
./stage_reference_pair.sh ntof       # xrdcp the nTOF run from EOS (resumable)
./stage_reference_pair.sh manifest   # laptop bundle list + rsync command
```

Reference pair: DREAM `run_79` subruns `stat090_0000`+`stat090_0001` ↔ n_TOF run **224572**
(already officially processed, LIQ trees present, both DREAM subruns fully time-contained).

## Layout

- `PLAN.md` — the plan and all verified facts.
- `stage_reference_pair.sh` — stage/verify the data (`ntof`, `pkup`, `denom`, `check`, `manifest`).
- Analysis outputs are **not** in the repo; they live under
  `/mnt/data/x17/beam_july/analysis/ntof_dream_merge/` (same convention as
  `ntof_july_analysis/track_rate_hv_time_intensity`).

## Comparing against our OWN n_TOF processing

Everything in this package defaults to the **official** n_TOF file. We now
reprocess n_TOF ourselves and the reprocessed data is better -- the matcher goes
from 95.3 % to 96.3 % efficient at the same 0.5 % false rate.

To run against ours, start at
[`../ntof_processing/HANDOFF_2026-07-29_dream_vs_reprocessed.md`](../ntof_processing/HANDOFF_2026-07-29_dream_vs_reprocessed.md).
It has the data inventory, the three-step recipe (build the join first, point the
reader at the candidate, sandbox the caches, turn `repair_tflash` off), and the
traps. `../ntof_processing/dream_regression.py` is the working example.

Note: **run 224572 alone covers both real sub-runs of run_79** (`stat090_0000`
bunches 146-1157, `stat090_0001` bunches 1165-2213). The other fourteen
`stat090_*` directories are empty stubs.
