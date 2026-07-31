# Archive — retired n_TOF processing documentation

**Nothing in this directory is current. Do not build on it, do not quote it, do
not let a search result from here become an input to an analysis.**

Every file carries a retirement banner naming its successor. It is kept only so
that the history of a number is traceable.

The current documentation is:

| | |
|---|---|
| state of the reprocessing, the resume point | [`../STATUS.md`](../STATUS.md) |
| how to audit any of it | [`../REVIEW.md`](../REVIEW.md) |
| what was measured to build v12 | [`../FINDINGS_2026-07-28_psa_optimization.md`](../FINDINGS_2026-07-28_psa_optimization.md) |
| the γ-flash time base | [`../FLASH_TIME_BASE.md`](../FLASH_TIME_BASE.md), [`../flash_timing/README.md`](../flash_timing/README.md) |
| the variant table and how to run one | [`../userinputs/README.md`](../userinputs/README.md) |
| **DREAM ↔ n_TOF matching** | [`../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`](../../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md) |

## What is in here and why it went

| file | retired because |
|---|---|
| `HANDOFF_2026-07-28_ntof_processing.md` | the work order that started the UserInput iteration; executed, v12 shipped |
| `HANDOFF_2026-07-29_dream_vs_reprocessed.md` | its recipe is now `ntof_dream_merge/match_study/`, its numbers predate the re-derived time map |
| `PRE_SHIP_TESTS.md` | the work order; run 07-29, results kept in `../FINDINGS_2026-07-29_pre_ship_tests.md` |
| `FINDINGS_2026-07-29_dream_crosscheck.md` | all three sections superseded; the MM cross-check in its §2 needs re-running at ±25 ns |
