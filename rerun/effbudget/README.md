# Efficiency loss budget (2026-10-26)

Where every M3 track that crosses an MX17 bench chamber ends up (Dylan 10-25: "if 95 % efficient, exactly where did the
other 5 % go?"). Report: `~/x17/cosmic_bench/effbudget/report.html`, published as notes/mx17-efficiency-budget.

| step | script | output |
|---|---|---|
| stage the pass inputs of a golden row (rr1_cm0 hits, M3 v2, incomplete events) | `rerun/inputs.py stage <key> <root> --parts run_config,m3,hits` (lxplus) | staged bench tree |
| budget per chamber | `X17_BENCH_ROOT=<root> budget.py <key> <det> --table <final_phys>/events.parquet --alignment <...>/alignment.json --out ~/x17/cosmic_bench/effbudget` | `rays_<det>.parquet`, `budget_<det>.json` |
| report | `make_report.py [--inline]` | `report.html`, `figures/` |

Needs the C22 outputs (`~/x17/cosmic_bench/outliers/classified_<det>.parquet`, `an/live_det4.parquet`) for the far
classes and det4's band map. Reproduces `mx_june_wft/02_efficiency.py` (det3 far 4.45 % in both). Downstream of the
reconstruction: re-run on the pass tables.
