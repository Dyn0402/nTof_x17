# Noise bands (satstudy follow-up 1, 2026-10-26)

What the "noise bands" are: whole-window baseline offsets on a contiguous patch of strips (rerun/satstudy/bands.py).
Report: `report.html` (published at https://dylan-neff.web.cern.ch/notes/mx17-noise-bands.html). Outputs (npz/json) in
`$BANDS_OUT`, default `~/x17/cosmic_bench/satstudy/bands/`. Inputs: the per-trigger baseline scans of
`../satstudy/bands.py scan` (`~/x17/cosmic_bench/satstudy/bands_scan*.npz`).

| step | script | output |
|---|---|---|
| (a) raw .fdf vs decoded_root, word by word, band + quiet events; header/trailer words by chip | `fdfcheck.py [N]` | `fdfcheck.json` |
| (b, c, d) where (per strip), patch edges vs connectors, x/y sign, persistence into the next trigger, vs discharges | `patches.py [KEY ...]` | `patches.json`, `patches_<key>.npz` |
| (f) band rate vs resist / drift HV: det3 6-27 scans, det6/det7 6-26 scan | `hvscan.py scan`, `hvscan.py summary` | `hvscan_<scan>.npz`, `hvscan.json` |
| (e) n_TOF: run_149 + run_89 (beam-off cosmics), run_145 (beam); on lxplus with LCG_105 | `ntof_scan.py <run_dir> <out.npz> [nsub nfile nmax]` | `/eos/user/d/dneff/bands/ntof_run*.npz` (copy to `$BANDS_OUT`) |
| figures, summary.json (incl. cross-chamber coincidence, bench at the n_TOF threshold), report.html; verdict text in `verdict.html` | `make_report.py` | `figures/`, `report.html`, `summary.json` |

Run locally with the main checkout's `.venv/bin/python`. The scans are I/O-bound and use about 1 GB of RAM.

## Result

There are two families of bands.

**Fixed-patch bands (det3, det7)**
- They are a chamber effect in the amplification stage: untriggered disturbances that last about 1 ms.
- Their rate rises ×1.8–1.9 per 10 V of resist HV (gain^1.4–1.6) and is flat in the drift field.
- They are not:
  - corruption: 0 of 3.9 M words differ;
  - pickup: two chambers read in the same triggers band independently;
  - tails of triggered discharges.
- They are absent at n_TOF, as the gain-equivalent HV predicts.

**Whole-plane bands (det2, det4, det6)**
- Their rate is flat in HV and they ride on spark triggers.
- They are not characterised here.

**Not identified (stated limitation):** the mechanism at the weak spot.
