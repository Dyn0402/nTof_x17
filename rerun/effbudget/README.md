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

## Pass 1 (2026-10-31): `pass1/`

The budget on every pass-1 bench row, the interior_good fiducial (`common/mx17_fiducial.py`, det7 X ch 256–511), the
trigger-track rule / multi-track credit / position-only arms, A1 on the golden rows, the noise-band census and the
fine (8 mm) maps. Result: `PASS1_RESULT_2026-10-31.md`; report `pass1/report/report.html` (with figures on EOS
`/eos/user/d/dneff/x17rerun/paper1_eff/report/`). Run order: `pass1/make_dag.py <code>` on lxplus (condor), then
`collect.py`, `a1_recat.py`, `fine_maps.py` with the lock venv on lxplus, then `make_report.py`.
