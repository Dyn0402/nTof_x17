# dream_return_cea — proof-of-life pedestals, DAQ back at CEA

One pedestal run per FEU, taken one FEU at a time on 5 October 2026 after the
MX17 DREAM DAQ came back from n_TOF to CEA. Output: [`report.html`](report.html).

Raw files: `~/Desktop/ped_validation_5-10-26` (FDFs, `ped_val.cfg`, RunCtrl logs).

## FEU identity

File `_feuN_` = cfg slot `N` (only that `Sys Topo Feu N` line was active for the
run) → `Feu N Feu_RunCtrl_Id`. This is cross-checked against the FEU ID the decoder
reads from the data headers (`reading FEU …`); all nine agree.

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| FEU ID | 32 | 71 | 98 | 99 | 106 | 31 | 69 | 70 | 121 |

The first feu9 attempt (13:36) failed with `FeuCtrl_Open failed`; the 13:40 retry worked.

## Rebuild

```bash
../.venv/bin/python -m dream_return_cea.pedestals    # decode FDF -> data/*.root, stats -> data/ped_stats.npz
../.venv/bin/python -m dream_return_cea.figures      # figures/*.png
../.venv/bin/python -m dream_return_cea.make_report  # report.html  (--embed PATH for a self-contained copy)
```

Decoder: `~/CLionProjects/mm_strip_reconstruction/cmake-build-release/decoder/decode`.
The noise decomposition (raw σ / common mode / σ after subtraction) is imported
from `ntof_pedestal_qa/lxplus/extract_pedestals.py`, so the numbers mean the same
thing as in the n_TOF pedestal history.
