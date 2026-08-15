# ntof_pedestal_qa — DREAM pedestal stability over the n_TOF campaign

What each of the 4 096 channels does when nothing is happening, measured before
every run, for the whole 1 July – 10 August 2026 campaign. 56 pedestal
acquisitions, 8 FEUs, per channel and per DREAM chip.

**Output**: [`report.html`](report.html) here, and the interactive page on the
CERN site at
<https://dylan-neff.web.cern.ch/x17/qa-pedestals.html> (linked from the X17
dashboard's QA row). A self-contained copy with the figures embedded is
published as a note at
<https://dylan-neff.web.cern.ch/notes/ntof-pedestal-stability.html>.

## What it found

| | |
|---|---|
| **23 July ~13:00** | `RdClk_Div` 6→4 (readout clock 1.5× faster, sampling clock unchanged). Residual noise **×2.0 on all eight FEUs at once**, common mode unchanged, and it stayed that way for the last 18 days — the whole production period. |
| **21–27 July** | Chamber **A**'s common mode ×3.3 on all sixteen chips of both views, no other chamber. Fixed at an access on 27 July between 11:23 and 14:11. |
| **22–27 July** | Chamber **A** x-view **connector 8 (channels 448–511) electrically disconnected**. Covers **every sub-run of run_79**; confirmed in the hits — 41 of those 64 channels recorded nothing at all. |
| three more | Connector-8 dropouts on A-y and B-y (early July) and A-y again (9 August), all recovered. |
| stable | Baselines flat across six weeks; no chamber drifted. |

## Vocabulary

Fixed in `pedestals.py` and used everywhere, and the same decomposition
`ntof_run_report` uses, so the numbers on both pages are the same numbers:

- **raw σ** — per-channel std of the raw ADC about its own baseline
- **common mode** — per chip, per time sample, the median over its 64 channels
  of (amplitude − channel mean); `cm_rms` is its std
- **residual σ** — what is left of a channel once that common mode is
  subtracted; this is what sets the 5 σ zero-suppression threshold

Raw σ is dominated by the coherent part — about 8× the residual at the end of
the run — so "the noise" is meaningless without saying which of the three.

## Rebuilding

`data/ped_stats.npz` and `figures/*.png` are **not in git** — the repository
ignores `*.npz` and `*.png` — so a fresh clone has `report.html` with no
pictures until the steps below are run. `data/ped_usage.csv` and
`data/ped_context.json` are small and are committed.

Three read-only passes on lxplus (EOS is POSIX there; nothing is mounted
locally), then everything else here:

```bash
scp lxplus/*.py lxplus:x17_pedqa/
ssh lxplus 'cd ~/x17_pedqa && source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
            python3 extract_pedestals.py --out ped_stats.npz     # ~5 min
            python3 extract_usage.py     --out ped_usage.csv
            python3 extract_context.py   --out ped_context.json'
scp lxplus:x17_pedqa/{ped_stats.npz,ped_usage.csv,ped_context.json} data/

../.venv/bin/python -m ntof_pedestal_qa.figures        # figures/*.png
../.venv/bin/python -m ntof_pedestal_qa.make_report    # report.html
../.venv/bin/python -m ntof_pedestal_qa.export_site    # the site's JSON
```

| | |
|---|---|
| `lxplus/extract_pedestals.py` | decodes every pedestal ROOT under `/eos/experiment/ntof/data/x17/july_beam/pedestals/` into per-channel mean / raw σ / residual σ and per-chip common mode, and parses the firmware's own `_ped.aux` / `_thr.aux` alongside |
| `lxplus/extract_usage.py` | which pedestal each of the 2 695 sub-runs actually loaded, from its `pedestal_run.txt` |
| `lxplus/extract_context.py` | the configuration and HV each pedestal was taken under — what makes a clock change distinguishable from a cable change |
| `pedestals.py` | loading, the dead/loud cuts, silent-connector episodes |
| `figures.py` · `make_report.py` · `export_site.py` | the three outputs |

`make_report.py --inline --out X.html` embeds the figures for publishing as a
note; the in-repo `report.html` links `figures/` relatively instead, because the
DAQ's Analysis tab serves them by path.

## Two things to be careful with

**Cut thresholds are relative to the same acquisition, deliberately.** The
absolute ADC scale doubled on 23 July; a fixed "dead below N ADC" cut would have
read that as four thousand new dead channels overnight. A channel counts as
disconnected when its raw σ is below 12 % of its FEU's median *or* below 8 ADC
outright, and loud when its residual is above 3× the FEU median.

**Disconnection is diagnosed on the raw σ, not the residual.** Losing the strip
loses the common-mode pickup, so the raw σ collapses while the residual barely
moves — the residual is what is left after the common mode is removed either
way.

## What it does not establish

A pedestal measures noise, never signal: the 23 July step says the *noise floor*
doubled, not that the physics got worse, and settling that needs hit amplitudes
across the boundary. A pedestal measures a FEU and its cable, not a chamber. And
these are 56 snapshots over six weeks — anything that started and finished
between two of them leaves no trace here. Fuller list at the end of the report.

Chamber↔FEU map from `ntof_active_area/clusters.py`: A = FEU 3/4, B = 5/6,
C = 7/8, D = 1/2.
