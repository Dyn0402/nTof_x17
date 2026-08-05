# The waveform-first reconstruction — reference document

A complete, figure-driven account of the `wft/` reconstruction, from raw FEU
ADC counts to physics outputs. Nine parts, 42 figures, all generated from the
live `sat_det3` products beside the data — nothing here is a redrawn old plot.

    figsrc/wftdoc.py     shared setup: paths, palette, display-event ranking
    figsrc/f_raw.py      Part I    raw ADC, pedestal, CNS, noise, the window
    figsrc/f_hits.py     Part II   compression, estimator independence, implied v
    figsrc/f_model.py    Part III  template, kernels, F_ik, design matrix
    figsrc/f_fit.py      Part IV   chi2 landscape, global scan, NNLS, errors
    figsrc/f_seed.py     Part V    seeding, window pad, candidates, dt_xy
    figsrc/f_calib.py    Part VI   corridor, template build, the degeneracy map
    figsrc/f_valid.py    Part VII  angles, position, pulls, quality, column
    figsrc/f_gallery.py  Part VII  outcome-ordered event gallery
    sections/*.html      the document, in order
    build.py             inlines the figures, assembles page.html

## Regenerate

```bash
cd docs/wft_reference/figsrc
for m in f_raw f_hits f_model f_fit f_seed f_calib f_valid f_gallery; do
    ../../../.venv/bin/python $m.py
done
cd .. && ../../.venv/bin/python build.py
```

Figures land in the scratchpad directory by default; override with
`WFT_DOC_FIGDIR` or `build.py --figs DIR`.

## Inputs

All under `<Analysis>/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/`:

- `wft/calib_bundle_lp_sp0free/` — the promoted lp calibration bundle
- `wft/calib_work/calib_cache.pkl` — 400 ref-pinned calibration events
- `wft/events_lp.parquet` — the 7,093-event reconstruction
- `wft/alignment_lp/alignment.json`, `wft/efficiency/`
- and the decoded ROOT files under the run's `decoded_root/`

det4 (`g_det4`) is read only for the noise comparison in Figure 5.

## Findings this document produced

Two things fell out of re-measuring rather than re-quoting:

1. **The `p0` global scan does not bracket the answer for ~21 % of planes**
   (§21.1). It is centred on the window's charge centroid, but `p0` is the
   position at the *mesh*, and those differ by half the transverse span.
   Nelder-Mead walks out and usually recovers, but those planes carry 5× the
   rate of >2 mm failures. Widening or re-centring the scan looks like cheap
   headroom.
2. **The candidate score is not scale-free** (§27). `Δχ²` is computed inside
   each candidate's own window, so a bright compact cluster outscores a dimmer
   extended one regardless of which looks like a track — visible in Figure 29.
