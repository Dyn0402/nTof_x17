# Per-view c2/c1 — the model-free evidence (2026-10-09)

Step 0 of the per-plane ratio work (PAPER_PLAN C2). Neighbour areas relative to
the centre strip, from 20 %-trimmed peak-aligned stacks with absent strips as zero.
These are head-on samples only, and no fit is involved.

| sample | view | +1 | −1 | (±2)/(±1) |
|---|---|---|---|---|
| det4 SPS run_71 RAW, 3 fields | Y | 0.49–0.57 | same | **0.39–0.45** |
| det4 SPS run_71 RAW, 3 fields | X | 0.26–0.28 (prompt) | 0.40–0.43 (delayed) | **0.10–0.12** |
| det3 bench, \|tan\| < 0.025–0.1 | Y | 0.61–0.65 | 0.64–0.65 | 0.42–0.43 ± 0.02 |
| det3 bench | X | 0.29–0.43 | 0.57–0.60 | 0.16–0.19 ± 0.03 |
| det4 bench | Y | 0.67–0.72 | 0.62–0.69 | 0.38–0.39 ± 0.02 |
| det4 bench | X | 0.41–0.46 | 0.31–0.44 | 0.20–0.23 ± 0.03 |

- **Y: c2/c1 ≈ 0.40 everywhere.** That covers two boards, beam and bench, and CF₄ and Ar/iso.
  It transfers.
- **X: the ±2 reach is less than half of Y's.** It is 0.10–0.12 in the beam and 0.16–0.23
  on the bench. r06's single global 0.6 is wrong on both views, and worst on X.
- **X ±1 is one-sided on det4 in SPS.** The asymmetry is the same in the flat and rotated mounts
  and at 81–233 V/cm (`angled_kernel/results.json`), so it is not a mount tilt. det3 on the bench
  shows the same sign. det4 on the bench is near-symmetric, but on only 30–120 events.
  **Open:** this needs the full bench waveform sample, not the calib cache.
- The comment at `wft/model.py:288` says "X cannot have resistive sharing". The data show
  ±1 sharing of 0.27–0.43 on X. What X lacks is the ±2 term.

SPS numbers come from `sharing_kernel/stacks.py` (`stacks_run71_raw.npz`), and the
mount comparison from `angled_kernel/results.json`. For the bench:

    ../../../.venv/bin/python bench_sides.py 0.05     # |tan| cut
