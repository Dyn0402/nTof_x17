# UserInput variants for the X17 EAR2 2026 reprocessing

Each directory is one change-set against Riccardo's
`UserInput_2026_EAR2_X17.h` (2026-07-17), cumulative in the order below, so a
regression can be attributed to a single group of parameters.

| variant | what changes | risk | expected effect |
|---|---|---|---|
| `v1_flash` | `G-FLASH THRESHOLD` only, on WAL / PSS / LIQ | low | PSS flash-id failures 37-85 % -> ~0 %; walls stop timing the divert gate, so the arm-dependent ~350 ns offsets collapse |
| `v2_elim` | v1 + `AREA/AMP` windows on PSS and LIQ, PSS `AMPLITUDE THRESHOLD` 100->50, PSS `G-FLASH WINDOW` 0->1000 | medium | recovers ~25 % of plastic and ~19 % of liquid hits that the current windows eliminate; hit counts go UP, so watch for junk |
| `v3_shapes` | v2 + measured pulse-shape templates for WAL and LIQ | medium | better fitted amplitude/area and pileup deconvolution, mainly for the liquids |

Full reasoning and the raw-waveform evidence:
`../FLASH_TIME_BASE.md` (flash timing) and
`../FINDINGS_2026-07-28_psa_optimization.md` (everything else).

## Parameter diff

```
             STEP  TIME    G-FLASH                       BASE  AMP  AREA/AMP  SIGWIDTH  SHAPES
             SIZE  LIMIT   OPT  THRESH      MINW  WINDOW OPT   THR  LO   HI   LO    HI
WAL  base    8/7   40000    0   500.        0.    0      4/150  50  10   200  5/100 4000  3 shipped
     v1..v3  8/7   40000    0   250/11400   0.    0      4/150  50  10   200  5/100 4000  3 measured (v3)

PSS  base    3/4   25000    0   50.         0.    0      1      100  2   20   10    3000  0
     v1      3/4   25000    0   2000/1e4    0.    0      1      100  2   20   10    3000  0
     v2,v3   3/4   25000    0   2000/1e4    0.    1000   1       50  1   60   10    3000  0

LIQ  base    2/4   25000    0   500.        100.  1000   1       50  2   10   1     5000  2 shipped
     v1      2/4   25000    0   500/1e4     100.  1000   1       50  2   10   1     5000  2 shipped
     v2      2/4   25000    0   500/1e4     100.  1000   1       50  1   60   1     5000  2 shipped
     v3      2/4   25000    0   500/1e4     100.  1000   1       50  1   60   1     5000  3 measured

PKUP, SILI   unchanged in every variant (PKUP has 0 % flash failures and is the time anchor)
```

## Running a variant

```bash
# 1. stage it (rewrites the pulse-shape addresses to full paths and checks them)
./ntof_processing/deploy_userinput.sh v1_flash /afs/cern.ch/work/d/dneff/x17_reproc/userinputs
rsync -a /afs/.../userinputs/ dneff@lxplus.cern.ch:/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/

# 2. process.  ssh -K is REQUIRED: without delegated credentials you get no AFS
#    token (writes fail) and no condor auth.
ssh -K dneff@lxplus.cern.ch
cd /afs/cern.ch/work/d/dneff/x17_reproc          # must be an /afs path: aux dirs land in cwd
/eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh \
    -y 2026 -a EAR2 -c X17_measurement -r 224572 \
    -p /afs/cern.ch/work/d/dneff/x17_reproc/userinputs/v1_flash/UserInput.h \
    -o /afs/cern.ch/work/d/dneff/x17_reproc/out/v1_flash

# 3. grade it
.venv/bin/python ntof_processing/validate_reprocessing.py 224572 <candidate>.root
```

### Two logistics facts, learned the hard way

- **`/eos/user/d/dneff` is not accessible** ("permission denied" on both `ls`
  and `eos quota`) -- CERNBox has probably never been activated for this
  account. Until it is, output has to go to `/afs/cern.ch/work/d/dneff`, which
  has ~44 GB free -- **one 26 GB run at a time**, so the variants run in
  sequence, not in parallel. Activating CERNBox at `cernbox.cern.ch` is the
  one thing that unblocks a parallel batch, and only you can do it.
- **You do not need the merged 26 GB file to grade a variant.**
  `RunProcessing.sh` writes per-raw-file partials into `<out>/completed/`
  before merging into `<out>/done/`. A handful of those carries the same
  per-bunch flash information at ~1/50 the transfer cost. Use them for the
  iteration loop and only pull the merged file for the final DREAM regression.

### Acceptance (from the handoff, unchanged)

1. flash-id bad-bunch fraction < 2 % per tree
2. per-tree coincidence offsets < 25 ns
3. per-tree hit counts must not fall (a "fix" that eats real hits is not a fix)
4. then the DREAM regression: `match_window.py` >= 99.9 %,
   `eval_singles_matcher.py` >= 93.7 % / <= 0.5 % false --- and it must reach
   those with the laptop-side repair DISABLED.
