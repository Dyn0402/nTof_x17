# nTOF DAQ processing — raw → hit-level ROOT (official pipeline)

How to turn raw n_TOF DAQ data into the hit-level PSA ROOT files the analysis
(`mx_july_beam_qa/`) consumes. This is the **official** n_TOF processing path
(`RunProcessing.sh`), not the old feb_beam `scripts/run_processor.py` route.

First done for **run 224489** (2026-07-17, EAR2 X17_measurement, the 2nd plastic
HV scan + first run with liquid scintillators). That run is the worked example
throughout.

Reference: n_TOF TWiki "Lxplus" page, §3 "Large scale data processing"
(https://twiki.cern.ch/twiki/bin/view/NTOF/Lxplus).

---

## TL;DR (the command that produced run224489.root)

Run from a directory **inside `/afs/`** (aux DAG files land in the cwd):

```bash
ssh lxplus
cd /afs/cern.ch/work/d/dneff/ntof_processing/run224489_launch   # any afs dir

/eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh \
    -y 2026 \
    -a EAR2 \
    -c X17_measurement \
    -r 224489 \
    -p /afs/cern.ch/work/d/dneff/ntof_processing/UserInput_2026_EAR2_X17.h \
    -o /eos/user/d/dneff/ntof_x17_processing/run224489
```

`RunProcessing.sh` itself submits an HTCondor DAG (N per-file processing jobs +
1 merge, 3 retries each) — nothing heavy runs on the login node. Final output:
`<-o>/done/run224489.root`. Launch it detached (`nohup … &`) if your connection
is flaky; the DAG survives your logout.

## Options (`RunProcessing.sh -help`)

| flag | meaning |
|------|---------|
| `-y` | year of the campaign (e.g. 2026) |
| `-a` | area: EAR1 / EAR2 / EAR3 / LAB |
| `-c` | campaign name = the DAQ "Exp. / Archive Folder" field (e.g. `X17_measurement`) |
| `-r` | first run number (required) |
| `-l` | last run number (optional; default = first run only) |
| `-p` | PSA UserInput file path (required) — relative or full |
| `-o` | output folder (optional; default = cwd). EOS or AFS, but **not** `/afs/cern.ch/user` (too small) |
| `-s` | `1` = skip runs already processed (looks in `completed/`+`done/`); default `0` |

Companion: `.../processingscripts/StageRuns.sh` (same options) stages a run's
files from CTA tape — see "Data location" below.

## Data location & staging

Raw DAQ data lives in two places at once:

- **EOS** `/eos/experiment/ntof/DAQ/<year>/<area>/<campaign>/<run>/` — normal
  `ls`, but only kept ~2 weeks after acquisition. (Run 224489:
  `/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement/224489/`, streams
  `stream0/` = `.idx`, `stream1/` = 88 × ~3 GB `.raw` ≈ 250 GB.)
- **CTA tape** `/eos/ctapublicdisk/archive/ntof/` after that — browse with
  `xrdfs root://eosctapublicdisk.cern.ch/ ls …`. `RunProcessing.sh` auto-stages
  from tape and waits for files to come online, so you usually don't need to
  stage manually; if you want to pre-stage, use `StageRuns.sh` (submits condor
  jobs, batch name `stage.dag`).

## PSA UserInput — the one real gotcha

The UserInput (`psa_userinput/UserInput_2026_EAR2_X17.h` here — Riccardo
Mucciola's 2026-07-17 version) defines per-detector PSA parameters. For X17 it
enables **pulse-shape fitting** for the WALL (WALA-D) and LIQ (LIQA-D) detectors,
which reference external pulse-shape `.txt` files (also in `psa_userinput/`).

**Pulse-shape files must be referenced by FULL path inside the UserInput.** As
Riccardo ships it (and as committed here) the last column has bare filenames:

```
WALA … 3   X17_WALA_Signal_3.txt X17_WALC_Signal_0.txt X17_WALB_Signal_0.txt
LIQA … 2   X17_LIQA_Signal_7.txt X17_LIQB_Signal_0.txt
```

The condor workers do **not** run from the UserInput's directory, so bare names
fail silently-ish (fit falls back / errors) → wrong reconstruction. Before
launching you must rewrite them to absolute paths pointing at wherever you put
the shape files. The TWiki says the same: "if Pulse Shape Fitting is activated,
the files containing the pulse shapes must be referenced by their full path."

`prepare_userinput.sh` in this directory does the rewrite for you (copies the
UserInput + shapes to a target afs dir and patches the paths). Or by hand:

```bash
DIR=/afs/cern.ch/work/d/dneff/ntof_processing
for f in X17_WALA_Signal_3 X17_WALB_Signal_0 X17_WALC_Signal_0 \
         X17_LIQA_Signal_7 X17_LIQB_Signal_0; do
  sed -i "s|\b$f.txt|$DIR/$f.txt|g" "$DIR/UserInput_2026_EAR2_X17.h"
done
```

(The committed copy keeps bare names on purpose, so it's portable — patch a
working copy, don't commit the patched one.)

## Output layout

Under `-o`:
- `completed/<run>/` — per-job partial ROOT files (`run<N>_NNNN.root`) +
  `history_<N>.root`. Deletable once `done/` is written; cleanup is left to you.
- `done/run<N>.root` — the final merged hit-level file. **Run 224489: 16 GB**,
  trees `PKUP SILI WALA-D PSSA-D LIQA-D index DAQsettings` (+ `history`).

## Verify a run finished cleanly

1. DAG: `<launch cwd>/<run>/run<N>.dag.dagman.out` ends with `DAG_STATUS_OK`
   and **no** `*.rescue*` file exists in that dir.
2. File: `done/run<N>.root` present, ~expected size, and opens with all detector
   trees (quick ROOT `f->ls()` / `GetEntries()`).
3. Benign noise you can ignore in the per-job `.err`: `history_*.root already
   exists`, g-flash warnings, one `ntoflib Exception: processing index file`
   per job. The DAG counts these jobs successful (no retry) — trust the DAG.

Timing: run 224489 (88 files, ~250 GB) took ~1.7 h wall on 9 condor jobs
(4 CPU, `longlunch`), ~8 min per raw file, once the jobs started running.

## Where run 224489's artifacts are (for reference)

- Launch dir (afs): `/afs/cern.ch/work/d/dneff/ntof_processing/run224489_launch/`
- Patched UserInput + shapes (afs): `/afs/cern.ch/work/d/dneff/ntof_processing/`
- Output (EOS): `/eos/user/d/dneff/ntof_x17_processing/run224489/`
  (note: user EOS, not the official processing area — this run needed Riccardo's
  brand-new UserInput, so it was processed privately.)

## Operational notes

- Must launch from inside `/afs/` (aux DAG files write to cwd; they are **not**
  auto-deleted — clean up yourself).
- Home afs (`/afs/cern.ch/user/...`) is too small for output; use work-afs or EOS.
- A single `condor_q` read can transiently return empty — don't read one empty
  result as "the DAG finished." Confirm with the dagman.out / `done/` file.
