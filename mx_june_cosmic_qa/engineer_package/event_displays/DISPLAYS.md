# Detector 3 micro-TPC event displays (engineer slide package)

Run: `mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V` (resist 490 V, drift 1000 V, 27 June 2026). Drift velocity 34 um/ns; strip pitch 0.78 mm; 30 mm drift gap; 32 samples x 60 ns readout.

| file | event | angle (X plane) | caption |
|---|---|---|---|
| `event_display_track_1.png/.pdf` | 14109 | 7 deg | Cosmic muon crossing the 30 mm drift gap; each point is one strip pulse, the line is the fitted track segment. |
| `event_display_track_2.png/.pdf` | 18400 | 18 deg | Cosmic muon crossing the 30 mm drift gap; each point is one strip pulse, the line is the fitted track segment. |
| `event_display_track_3.png/.pdf` | 9727 | 26 deg | Cosmic muon crossing the 30 mm drift gap; each point is one strip pulse, the line is the fitted track segment. |
| `event_display_track_4.png/.pdf` | 16659 | 34 deg | Cosmic muon crossing the 30 mm drift gap; each point is one strip pulse, the line is the fitted track segment. |
| `event_display_waveforms.png/.pdf` | 9727 | — | Raw waveforms of one inclined muon: the diagonal charge stripe shows deeper ionization arriving later; insets show single-strip pulses. |
| `event_display_spark.png/.pdf` | 11309 | — | A quenched discharge for contrast: hundreds of strips fire simultaneously, trivially separated from muon tracks. |
| `event_display_gallery.png/.pdf` | 6 events | 7–34 deg | Gallery of micro-TPC muon tracks at increasing angle (X plane). |

Selection: clean single cluster in both planes; events below ~23 deg are matched to a single reference-telescope ray with fitted angle consistent with the telescope; steeper events (tracks 3-4) lie outside the telescope acceptance and are selected on internal micro-TPC consistency alone (full-gap drift-time span and clean linear ladders in BOTH planes) — their quoted angles come from the micro-TPC fit only. Absolute drift time contains an arbitrary trigger offset; the secondary axis converts relative time to drift distance (34 um/ns).
