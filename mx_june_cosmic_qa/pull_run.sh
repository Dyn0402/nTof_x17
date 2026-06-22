#!/usr/bin/env bash
# Pull the analysis inputs for one cosmic-bench run/subrun from the DAQ PC into
# ~/x17/cosmic_bench (next to where the QA writes its output). Only the two
# directories the QA needs (combined_hits_root, m3_tracking_root) plus
# run_config.json are pulled -- NOT the bulky raw .fdf / decoded data.
#
# Usage:
#   ./pull_run.sh <run_name> <sub_run> [area] [host]
# e.g.
#   ./pull_run.sh mx17_det1_det2_short_6-18-26 short_run det1_det2
#
# Defaults: area=det1_det2  host=rays_daplxa
# Remote layout assumed: <host>:/mnt/cosmic_data/MX17/Run/<run_name>/
# Local  layout written:  ~/x17/cosmic_bench/<area>/<run_name>/
#
# ALT SOURCE: the June runs are also archived on lxplus (Kerberos: kinit first) at
#   lxplus:/afs/cern.ch/user/d/dneff/x17/cosmic_bench/june_tests/<run_name>/<sub_run>/
# (same combined_hits_root + m3_tracking_root layout). rays_daplxa may be wiped between
# beam periods, so if a run is gone there, pull from lxplus with a direct rsync, e.g.:
#   rsync -a --update -e "ssh -o BatchMode=yes" \
#     lxplus:.../june_tests/<run>/<sub_run>/{combined_hits_root,m3_tracking_root} \
#     ~/x17/cosmic_bench/<area>/<run>/<sub_run>/
# This script's REMOTE_BASE uses the rays /mnt path, NOT the AFS path.
set -euo pipefail

RUN="${1:?usage: pull_run.sh <run_name> <sub_run> [area] [host]}"
SUBRUN="${2:?usage: pull_run.sh <run_name> <sub_run> [area] [host]}"
AREA="${3:-det1_det2}"
HOST="${4:-rays_daplxa}"

REMOTE_BASE="/mnt/cosmic_data/MX17/Run/${RUN}"
LOCAL_BASE="${HOME}/x17/cosmic_bench/${AREA}/${RUN}"

echo "Pull  ${HOST}:${REMOTE_BASE}/${SUBRUN}"
echo "  ->  ${LOCAL_BASE}/${SUBRUN}   (combined_hits_root + m3_tracking_root + run_config.json)"

# Size check on the remote before transferring.
echo "Remote sizes:"
ssh "${HOST}" "du -sh '${REMOTE_BASE}/${SUBRUN}/combined_hits_root' '${REMOTE_BASE}/${SUBRUN}/m3_tracking_root' 2>/dev/null" || {
    echo "ERROR: remote subrun dirs not found (combined_hits_root / m3_tracking_root)."; exit 1; }
echo "Local free space:"; df -h "${HOME}/x17" | tail -1

mkdir -p "${LOCAL_BASE}/${SUBRUN}"
# rsync 3.0.9 on the DAQ PC: no --info; -a preserves structure. Permission-preserve
# warnings on the mounted data disk are harmless (data still copies).
rsync -a "${HOST}:${REMOTE_BASE}/run_config.json"               "${LOCAL_BASE}/"                 || true
rsync -a "${HOST}:${REMOTE_BASE}/${SUBRUN}/combined_hits_root"  "${LOCAL_BASE}/${SUBRUN}/"
rsync -a "${HOST}:${REMOTE_BASE}/${SUBRUN}/m3_tracking_root"    "${LOCAL_BASE}/${SUBRUN}/"

echo "Pulled:"
du -sh "${LOCAL_BASE}/${SUBRUN}/combined_hits_root" "${LOCAL_BASE}/${SUBRUN}/m3_tracking_root"
echo "combined_hits files: $(ls "${LOCAL_BASE}/${SUBRUN}/combined_hits_root"/*.root 2>/dev/null | wc -l)"
echo "Done. Register in qa_config.py with base_path='${HOME}/x17/cosmic_bench/${AREA}/'."
