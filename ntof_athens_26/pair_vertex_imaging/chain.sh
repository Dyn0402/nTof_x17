#!/usr/bin/env bash
# The whole pair-vertex diagnosis, end to end.  Run from ntof_athens_26/.
#   bash pair_vertex_imaging/chain.sh
# Every path resolves through sept26_prelim_analysis/paths.py, so moving the
# tree is one environment variable and not an edit here.
set -euo pipefail
PY="${PY:-../.venv/Scripts/python.exe}"
J="${JOBS:-8}"
$PY -m pair_vertex_imaging.vertex_lab  --jobs "$J"
$PY -m pair_vertex_imaging.diagnostics --jobs "$J"
$PY -m pair_vertex_imaging.make_figures
$PY -m pair_vertex_imaging.make_report
# the image follow-up: the 3D density against a no-source null, and its note
$PY -m pair_vertex_imaging.vertex_image --jobs "$J"
$PY -m pair_vertex_imaging.make_image_figures
$PY -m pair_vertex_imaging.make_image_note
# where z went: D's track quality, D-D and A-C pairs, and the A/C alignment test
$PY -m pair_vertex_imaging.z_image --jobs "$J"
$PY -m pair_vertex_imaging.z_image --jobs "$J" --align-ac
$PY -m pair_vertex_imaging.make_z_figures
$PY -m pair_vertex_imaging.make_z_note
# every clean pair in one 3D density, then y
$PY -m pair_vertex_imaging.vertex3d
$PY -m pair_vertex_imaging.y_image --jobs "$J"
$PY -m pair_vertex_imaging.make_y_figures
$PY -m pair_vertex_imaging.make_y_note
# same-chamber pairs, and the track-quality-against-multiplicity study
$PY -m pair_vertex_imaging.intra_vertex --jobs "$J"
$PY -m pair_vertex_imaging.intra_vertex --jobs "$J" --multiplicity
$PY -m pair_vertex_imaging.make_v3d_intra_note
echo
echo "3D + intra note -> pair_vertex_imaging/figures/v3d_intra_note.html"
echo "z note -> pair_vertex_imaging/figures/z_image_note.html"
echo "y note -> pair_vertex_imaging/figures/y_image_note.html"
echo "report -> pair_vertex_imaging/figures/report.html"
echo "note   -> pair_vertex_imaging/figures/vertex_image_note.html"
