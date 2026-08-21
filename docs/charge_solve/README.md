# The charge solve — a deep dive on A and NNLS

A standalone note answering one question in detail: *what is the design matrix
`A` in `wft/model.py::build_matrix`, and what does the NNLS step do with it?*
Sixteen figures, all generated from live `sat_det3` products, plus a 12 × 2
worked example small enough to check by hand.

```bash
../../.venv/bin/python figs.py        # figures + numbers.json  (~15 s)
../../.venv/bin/python make_note.py   # charge_solve.html, figures inlined
python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py charge_solve.html \
    --slug wft-charge-solve --tags "X17, cosmic bench, micromegas, reconstruction, waveforms" --force
```

Inputs (overridable): `WFT_DOC_BUNDLE` picks the calibration bundle — the
default is `calib_bundle_r06`, the corrected sharing kernel, which is what the
reconstruction on disk was produced with since 2026-08-21. An inverted bundle
(c2 > c1) will not load at all: `wft.calib.check_kernel_ordering`.
`CS_FIGDIR` moves the PNGs.

`figs.py` carries an instrumented Lawson–Hanson NNLS so the solver's steps can
be plotted; it is checked against `scipy.optimize.nnls` on every run (agreement
2e-12 on the display event) and is *not* used by the reconstruction.

Every number quoted in the prose is read from `numbers.json`, which `figs.py`
writes — the text cannot drift away from the plots. `charge_solve.html` is
generated and gitignored.

Companion: `docs/wft_reference/` is the full nine-part account of the chain;
this note goes deeper on Parts III–IV only.
