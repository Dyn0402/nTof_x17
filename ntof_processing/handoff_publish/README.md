# Handing the X17 tail block to n_TOF

Two scripts for the n_TOF processing account. They copy the runs the X17 group
processed itself into `official/completed/`, so the block ends up where everyone
looks for it.

| | |
|---|---|
| `publish_x17_block.sh` | the 30 finished runs of 224688-224718 |
| `publish_224709.sh` | 224709 alone, once it finishes on our side |

```bash
# on lxplus, as the processing account
./publish_x17_block.sh --dry-run     # see exactly what it would do
./publish_x17_block.sh --go          # copy
./publish_x17_block.sh --verify      # re-check afterwards

./publish_224709.sh --wait           # blocks until 224709 is ready, then copies
```

Only two EOS paths are needed and no AFS:

| | |
|---|---|
| from | `/eos/experiment/ntof/data/x17/reproc/prod_v12/<run>/completed/<run>/` |
| to | `/eos/experiment/ntof/processing/official/completed/<run>/` |

## Why there is anything to hand over

The 2026 X17 EAR2 pass stopped after run 224687 (last output 08-07 19:56), so
**224688-224718 was never processed** — 31 runs, 12.76 TB of raw, the final three
days of the campaign. We processed them ourselves with **the same configuration**:
`UserInput_2026_EAR2_X17_v4.h`, every parameter line and all 26 pulse-shape
templates byte-identical.

That is not just a claim about the config. On the runs that now exist in **both**
processings (224572-224579, after your 08-10/08-11 merges), our output and yours
agree **hit for hit** — same hit count and all 22 per-hit columns exact on
WAL A-D, PSS A-D, SILI and PKUP.

The products are the per-job partials, `run<run>_NNNN.root`, contiguous from
`_0001`, exactly as `RunProcessing.sh` leaves them. **They are not merged** — our
merge node dies on condor's 1024 MB file-transfer cap. Merge them if you want to;
nothing downstream needs it.

## What the scripts will not do

* **Never overwrite.** A destination that already holds partials is left
  completely alone. If you have since processed a run yourselves, yours wins and
  the script says so.
* **Never delete** anything, anywhere.
* **Never publish an incomplete run.** The source must have its `history` file and
  partials contiguous `1..N` with `N = ceil(raw stream1 files / 4)`.
* Each file is size-verified after copying, and a file already at the destination
  with the right size is skipped — so re-running after an interruption is cheap
  and safe.

## Two things worth knowing

**The `history` object records our staging path.** Each product stores the
UserInput it was made with, and ours was read from
`/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/...` rather than your EOS
`shapes_X17_v4` directory. The parameters are identical; only the directory
string differs. We left it as it is, because it is honest provenance — these
files were produced outside your pass.

**224576 is deliberately not in the list.** We hold it only in an older variant
(`v11_pssfit_width`, which differs from v4 on the four LIQ rows), and you are
reprocessing it yourselves. Publishing ours would put a different recipe into the
official area under the same name.

## If something looks wrong

`./publish_x17_block.sh --verify` re-checks every file against our copy and
names anything missing or the wrong size. Questions to Dylan Neff,
dneff@cern.ch.
