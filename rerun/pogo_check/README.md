# pogo_check: M1 pogo-field pad shorts, for any data set

Start with `report.html` (verdict, figures, per-card table). `METHOD.md` covers why each test is there and what it
cannot see.

| file | what |
|---|---|
| `layout.py` + `m1_pogo_to_mec8.json` | DREAM channel → pogo pad (card, col, row); pogo-neighbour matrix |
| `accumulate.py` | per-pair event statistics from hits (QA only): firing, jumps across quiet strips, log-amplitude regression, census correlation |
| `run_bench.py` | bench reader (hits_root) → `out/<run>_<plane>.npz` |
| `pogo_check.py` | links, nets, cross-card null, layout test, verdict → `results/<run>_<plane>.json` |
| `make_report.py` | `figures/` + `report.html` |

Bench fleet, 2026-10-25: only det7 X card 1 (g_det7_long) is POGO. det7 Y and det6 Y carry a strong non-pogo
correlation between their two cards. The census "dense block" on det7 X does not respond to tracks.
