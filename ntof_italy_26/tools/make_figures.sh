#!/usr/bin/env bash
# Pull the figures of the two new Italy-talk slides from the published slide
# notes (dylan-cern-site) and the facility-search outputs into ../figures/.
#   bash ntof_italy_26/tools/make_figures.sh        (from the repo root)
set -e
PY=${PY:-/c/Users/Dylan/PycharmProjects/nTof_x17/.venv/Scripts/python.exe}
SITE=${SITE:-$HOME/PycharmProjects/dylan-cern-site}
FAC=${FAC:-$HOME/PycharmProjects/x17_facility_search}
F=ntof_italy_26/figures
R=ntof_italy_26/tools/render_note.py
ST=$SITE/notes/x17-status-2026-10-15.html
LN=$SITE/notes/lnl-x17-feasibility.html
mkdir -p $F
# Al pairs slide (status deck for 15 Oct)
"$PY" $R $ST penetration $F/al_penetration.png --bg ffffff --crop 120,330,1080,620 --scale 2.5
"$PY" $R $ST scatter $F/al_scatter_schematic.png --svg "schematic of a scattered conversion pair" --bg ffffff --scale 2.5
"$PY" $R $ST scatter $F/al_miss_vs_energy.png --svg "miss vs energy" --bg ffffff --scale 2.5
"$PY" $R $ST correlated $F/al_correlated_fraction.png --svg "pair fraction vs time" --bg ffffff --scale 2.5
# LNL slide (lnl-x17-feasibility pitch block)
"$PY" $R $LN pitch-setup $F/lnl_arm_ntof.png --svg "one arm face-on" --nth 0 --bg ffffff --scale 3
"$PY" $R $LN pitch-setup $F/lnl_arm_big.png --svg "one arm face-on" --nth 1 --bg ffffff --scale 3
"$PY" $R $LN pitch-setup $F/lnl_four_arms_big.png --svg "four arms, view along the beam" --nth 1 --bg ffffff --scale 3
"$PY" $R $LN pitch-setup $F/lnl_four_arms_ntof.png --svg "four arms, view along the beam" --nth 0 --bg ffffff --scale 3
cp $FAC/lnl/out/talk/lnl_summary.png $FAC/lnl/out/talk/significance_days.png $FAC/lnl/out/talk/x17_per_day.png $F/
cp $FAC/lnl/out/syst/figures/syst_spectra.png $FAC/lnl/out/syst/figures/atomki_repro.png $F/
cp $FAC/lnl/out/talk/numbers.json $F/lnl_numbers.json
