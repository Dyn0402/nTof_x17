# Pass 1 completion ledger

Bench: 156 non-excluded rows, 156 complete; SPS: 71 rows, 71 complete.
Expected code: bench ['6a0bb1f'], SPS ['3ac0e79'] (seed fix + head-on estimators: meta ['hot_seed_channels'], columns ['x_p_mid', 'y_p_mid', 'x_c_pos', 'y_c_pos', 'x_c_n_strips', 'y_c_n_strips', 'ftst']); the rows of sps_junk/results/promoted_rows.txt ['48244778', 'c36322ec'] (+ coupled channels: meta ['hot_seed_channels', 'coupled_channels', 'n_events_coupled_censored'], columns x/y_n_coupled); 01-04: ['b5a29af']. Alignment within 5 mm of its night, reco_far <= 15 %.

**No row missing, no overrides, code = the archive, every alignment within its night, no far collapse.**

Bench events reconstructed: 215,797; SPS events: 212,599

## Addendum 2026-11-02 (hand-added, not produced by `pass1_ledger.py`): pairing refresh

- **Location:** every bench row has a sibling directory `Analysis_pass1/<rel>/wft/pairing_cost/`, holding:
  - `events.candidates.parquet` and `events.parquet`, with x–y tracks re-selected in `select_tracks` mode `cost`;
  - `PAIRING_COST.json`, recording the tool, commit `1d537b04`, the bundle, and sha1 of the source tables.
- **Coverage:** 157 tables (156 registry rows and det3_quick), 215 831 events.
- **The originals above are unchanged.** This ledger's checks still refer to them.
- **References:** `rerun/effbudget/single/PAIRING_FIX_2026-11-02.md` §5 and `PASS1_STATUS_2026-10-30.md`.
