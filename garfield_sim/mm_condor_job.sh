#!/bin/bash
# mm_condor_job.sh — HTCondor job wrapper
# =========================================
# Sources the LCG environment then calls mm_condor_worker.py.
# All arguments are passed through from the JDL's Arguments line.
#
# This script is transferred to the worker node by HTCondor along with
# mm_condor_worker.py.

set -e

echo "[job.sh] Starting on $(hostname) at $(date)"
echo "[job.sh] Arguments: $@"

# ── LCG environment ────────────────────────────────────────────────────────────
LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/setup.sh

if [ ! -f "$LCG_VIEW" ]; then
    echo "[job.sh] ERROR: LCG view not found: $LCG_VIEW" >&2
    exit 1
fi

source "$LCG_VIEW"
echo "[job.sh] LCG_108 sourced — ROOT $(root-config --version 2>/dev/null || echo 'unknown')"

# ── Garfield setup ─────────────────────────────────────────────────────────────
# The LCG view includes Garfield++. Source its setup script to set
# GARFIELD_INSTALL and HEED_DATABASE.
GARFIELD_SETUP=$(find /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt \
    -name "setupGarfield.sh" 2>/dev/null | head -1)

if [ -n "$GARFIELD_SETUP" ]; then
    source "$GARFIELD_SETUP"
    echo "[job.sh] Garfield++ setup sourced from $GARFIELD_SETUP"
else
    echo "[job.sh] WARNING: setupGarfield.sh not found in LCG view — HEED_DATABASE may be unset"
fi

# ── Run worker ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[job.sh] Script dir: $SCRIPT_DIR"
echo "[job.sh] Python: $(which python3) ($(python3 --version))"

python3 "${SCRIPT_DIR}/mm_condor_worker.py" "$@"
EXIT_CODE=$?

echo "[job.sh] Worker exited with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
