# ntof_italy_26 — X17 talk at the n_TOF Italy meeting (15 Oct 2026)

The Athens talk (`ntof_athens_26/slides/neff_x17_ntof_athens_26.pptx`) with two
new slides near the end, and the meeting name, date and page numbers updated
(25 → 27 main slides).

| slide | what | source |
|---|---|---|
| 43 (24 / 27) | **The Al pairs are soft: few trigger, poor pointing** — Al legs ~2 MeV vs X17 4–16 MeV; 11 % vs 82 % reach the trigger level; ~9° scattering → 33 mm vertex miss; but ~3,700 correlated same-chamber double tracks in the data (rate ~⅛ of Geant4, open) | status deck for 15 Oct (`sept26_prelim_analysis/make_status_oct15_deck.py`, notes/x17-status-2026-10-15) |
| 45 (26 / 27) | **Next: repeat the ATOMKI measurement at LNL** — n_TOF vs LNL table, 10-day Geant4 spectra (S/B ≈ 8 %), X17/day and time-to-3σ, the big plastics (×7 trigger acceptance, ×7.8 X17/day, ~1 d vs ~17 d, ≈ €93k), other facilities in one line | `x17_facility_search/lnl/` (`out/talk/`, `out/syst/`), notes/lnl-x17-feasibility |

Speaker notes on both new slides carry the numbers and their provenance.

## Rebuild

```bash
bash ntof_italy_26/tools/make_figures.sh          # figures from dylan-cern-site notes + facility search
.venv/Scripts/python.exe ntof_italy_26/make_italy_deck.py --src ntof_athens_26/slides/neff_x17_ntof_athens_26.pptx
```

`tools/render_note.py` renders one slide (or one SVG of it) of a published
slidedoc note to PNG with headless Chrome. The Athens deck is read, never
written. The meeting name and date are `MEETING` / `DATE` at the top of
`make_italy_deck.py`.
