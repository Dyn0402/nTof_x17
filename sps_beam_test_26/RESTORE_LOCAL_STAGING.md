# Restoring the local `sps_run53_det4_check` staging tree

**232 GB of `/media/dylan/data/x17/sps_run53_det4_check` was deleted on
2026-09-07** to make room for the September X17 preliminary analysis, which had
6.5 GB of disk to work with. Nothing unique to that machine was lost; this is
how to get any of it back.

The full file-by-file record is
`/media/dylan/data/x17/sps_run53_det4_check/DELETED_MANIFEST_2026-09-07.json` —
every one of the 1 824 files that was in the tree, with its size, in three
classes.

| class | files | size | where it is now |
|---|---|---|---|
| `on_eos` — `.fdf` and `combined_hits_root/*.root` | 469 | 139.9 GB | `/eos/experiment/ntof/data/x17/p2_sps_july`. **Verified before deleting**: all 469 matched an EOS file of identical size |
| `regenerable` — `dec_*.root`, `hits_*.root`, `ped_*.root` | 569 | 91.8 GB | nowhere. Local decode output; reproduce from the `.fdf` |
| `kept` — `.npz`, `.json`, `.csv`, `.png`, `.log`, `.cfg`, `.prg` | 786 | 3.5 GB | **still on disk**, and mirrored to `/eos/user/d/dneff/x17/sps_run53_det4_check/` |

## Re-pulling the raw

`staging/pull_wave1.sh` … `pull_wave6.sh` (and `resume_all.sh`) are the original
rsync recipes and still work — the EOS paths have not moved. They pull
selectively (FEU3 `.fdf` plus FEU1 `combined_hits`), which is why 140 GB covers
a 2.6 TB source tree.

## Re-decoding — EOS's `decoded_root` is NOT a substitute

EOS carries a `decoded_root` for these runs, and it is the wrong one twice over:

- it holds FEU `_01` only — the P2 detector. **Our det4 side is FEU `_03`, which
  EOS never decoded at all.**
- even for `_01` it was decoded with different settings; the file sizes differ
  from ours by ~10 kB each.

The exact commands are logged one per file in the `decode*.log` beside each run.
From `run_63/decode_rot25.log`:

```
decode EicP2Bt_operating_00_datrun_260802_23H53_001_01.fdf dec_operating_00_001_01.root
analyze_waveforms dec_operating_00_001_01.root hits_operating_00_001_01.root ped_01.root \
    --tps 60 --thr 4.0 --mf 5 --cns 0 --zs-baseline 1
```

Each log's **first line** records the configuration it ran under — angle, gas,
ZS threshold, sampling window, matched-filter length. `run_63` has two,
`flat` and `rot25`, and they are not interchangeable. Those logs were kept.

## What survived, and why that is the part that mattered

The 3.4 GB of `.npz` are the fit and pairing products — `pair_run63_rot25.npz`,
`kernel_fit_flat700.npz` and the rest — together with every `.json` result,
`.csv` scan table, `.log` and figure. Those are what the SPS conclusions in
`sps_beam_test_26/` actually rest on, they are not reproducible without
re-running the whole analysis, and they are small, so they stayed on disk and
were also copied to EOS user space.
