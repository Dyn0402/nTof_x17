# data/ — inventories pulled from CERN, kept so stage 0 is reproducible offline

| file | what | how it was made |
|---|---|---|
| `slim_inventory_2026-09-07.tsv` | every n_TOF slim product on EOS: `<filename>\t<bytes>`, 450 rows | `find /eos/experiment/ntof/data/x17/july_beam/runs -mindepth 4 -maxdepth 4 -name 'ntof_hits_*.root' -printf '%f\t%s\n'` on lxplus, ~2 min |

Filenames parse as `ntof_hits_run_<run>_<subrun>_<ntof run>.root`. Joined
against `dylan-cern-site/data/x17-runs.json` this is the slim half of stage 0's
sample; see `../STATUS.md` "Verified at CERN" row 1 for what it showed.

Re-generate rather than trust this file if more than a few weeks have passed —
n_TOF keeps running other experiments and the tree is not frozen.
