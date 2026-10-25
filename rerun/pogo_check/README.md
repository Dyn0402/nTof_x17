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
| `run_mountings.py` | every bench mounting (run × chamber in `pass_registry.csv`, all sub-runs pooled; FEUs from the campaign manifest) → `out/m_<run>__<det>_<plane>.npz` |
| `mountings_table.py` | `results/mountings_summary.{json,md}`: one line per mounting, plane and card |

Bench fleet, 2026-10-25: only det7 X card 1 (g_det7_long) is POGO. det7 Y and det6 Y carry a strong non-pogo
correlation between their two cards. The census "dense block" on det7 X does not respond to tracks.

## Every mounting (2026-10-26, `run_mountings.py`, `mountings_table.py`)

15 mountings (14 nights × chamber, 6-17 to 6-27; pooled sub-runs, local hits_root, QA only). Table:
`results/mountings_summary.md`. Read against the nets, not the label alone:

- **det7 X card 1 is a pogo fault on every 6-26 mounting** (HV scan: POGO, 42 pogo-net channels in 7 nets; overnight
  pooled: same vertical chains 88→123 and 230→250 step 5, label drops to POGO? only because pooling dilutes the
  enrichment to 8.8 < ENRICH_MIN). Unchanged conclusion: quote det7 X from ch 256–511.
- **det2 on 6-17, X card 1: POGO by the rule, weak in substance** — one 3-channel pogo net (56, 57, 62) and a pair,
  inside a 77-channel noisy OTHER net (33 noisy channels) on the card. This is the night's known half-dead /
  noisy X; the night is already `skeptical`. det2 is clean of pogo nets on 6-22 (remounted).
- **det3 on 6-22 (overnight), Y card 2: not pogo** — an 11-channel run of consecutive channels 450–459, 462 linked only
  by the census correlation (C ≈ 0.9, zero jumps): a correlated noisy block (cf. det7 X's dense block), which a
  contiguous run satisfies trivially as "pogo-shaped". Gone on 6-23, 6-25, 6-27.
- det3 6-23 Y card 2: mirror pairs about ch 394 (385↔403 … 391↔397), a connector/cable pattern, OTHER (not a pad).
- Every other "POGO?" is a single pogo-adjacent pair, the same as on the golden rows already judged clean.
- No new pogo fault on det3, det4 or det6 on any night.
- **SPS det4**: no pogo net on run_56 (Ar/CO₂/iso, flat, first epoch), run_63 or run_71 (`run_sps.py --tag`): CLEAN or
  OTHER on both planes and both cards (beam-lit channels only). The H4 mount changes were rotations of the stand, not
  reseatings, so these cover the campaign.

**Verdict for the pass:** the only pogo fault in the June bench + SPS data is det7 X card 1 on 6-26 (both
mountings); det2 6-17 X card 1 carries weak pogo evidence on an already-skeptical night. No masks change.
