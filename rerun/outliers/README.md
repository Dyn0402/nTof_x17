# Outliers vs M3 (CHECKS C22)

Every golden-row event that disagrees with the M3 reference, looked at through its waveforms, the M3 cluster trees and
a two-chamber cross-check; the det4/det7 head-on far tail; the edge zone (PAPER_PLAN D4); det4 at H4.
Report: `~/x17/cosmic_bench/outliers/report/report.html` (published notes/mx17-outliers-vs-m3.html).

Run order (laptop, `../../.venv/bin/python`, data in `~/x17/cosmic_bench/outliers/`):

| step | script | output |
|---|---|---|
| pull finalize_phys tables of the golden rows from EOS (`Analysis_rr1/<run>/<sub>/<det>/wft/final_phys`) | — | `final/` |
| per-event join (wft + M3 as 03_angles + every ray + hits QA) | `join.py <key> --table --alignment --out` | `joined_<det>.parquet` |
| M3 cluster trees from raw (lxplus condor) | `m3clus/job.sh`, `m3clus.sub`, `jobs.txt` | `m3clus/*_{analyse,rays}.root` |
| M3 features: clusters per plane, doublet kink, alternative combinations | `m3feat.py <det>` | `m3feat_<det>.parquet` |
| waveforms of outliers + 300 controls | `display.extract` (see `classify.outlier_ids`) | `wf_<det>.pkl` |
| waveform features | `wffeat.py <det>` | `wffeat_<det>.parquet` |
| classes | `classify.py <det>` | `classified_<det>.parquet` |
| bent tracks: kink / doublet vertex, mixture fit | `kinkvtx.py`, `BENT_ONLY=1 kinkvtx.py` | `kinkvtx_<det>.parquet`, `an/kinkvtx.json`, `an/bent_fraction.json` |
| two chambers at once (6-26) | `sixpt.py` | `sixpt.parquet` |
| edge zone | `edge_response.py` | `an/edge_response.{png,json}` |
| det4 at H4 vs bench, same fitter | `../../sps_beam_test_26/analysis/plane_ratio/sps_cache.py`, `sps_headon.py` | `sps/*.json` |
| edges by distance band, every category | `fiducial.py` | `report/fiducial.{csv,json,txt}` |
| competing-cluster events: nearest candidate to the reference | `nearest.py` | `report/nearest.json` |
| two-track refit of the outliers (arms off / on / onF30; env as finalize + `WFT_TWO_TRACK_*`) | `reco_subset.py`, `twotrack.py det3 --arms off on onF30` | `reco2/`, `report/twotrack_det3.json` |
| numbers, figures, report | `summary.py`, `make_figures.py --examples`, `make_report.py` | `report/` |

Displays: `display.py <det> <eid>...`, `montage.py <det> <class> --n 4` (`OL_ZOOM=10` to zoom).
`olib.py` holds the loader and the ladder-scaled residual (named so it does not shadow the repo's `common` package).
