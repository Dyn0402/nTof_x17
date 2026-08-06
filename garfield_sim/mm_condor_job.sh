#!/bin/bash
# mm_condor_job.sh — HTCondor job wrapper
# =========================================
# Unpacks the shipped Garfield++ tarball, sets the environment via
# setup_garfield.sh, then calls mm_condor_worker.py. All arguments are passed
# through from the JDL's Arguments line.
#
# This script is transferred to the worker node by HTCondor along with
# mm_condor_worker.py, setup_garfield.sh and garfield-<pin>.tar.gz.

set -e

echo "[job.sh] Starting on $(hostname) at $(date)"
echo "[job.sh] Arguments: $@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Environment ───────────────────────────────────────────────────────────────
# setup_garfield.sh unpacks the shipped garfield-<pin>.tar.gz and exports
# everything; it is the only place that names a Garfield or LCG path.
source "${SCRIPT_DIR}/setup_garfield.sh"

# ── Run worker ─────────────────────────────────────────────────────────────────
echo "[job.sh] Script dir: $SCRIPT_DIR"
echo "[job.sh] Python: $(which python3) ($(python3 --version))"

# `set -e` is active, so a bare call followed by `EXIT_CODE=$?` would abort the
# script before the exit code was ever captured or logged — a failing worker
# would look like a silent death in the condor log. Running the worker as an
# `if` condition suspends `set -e` for that command so we can report properly.
if python3 "${SCRIPT_DIR}/mm_condor_worker.py" "$@"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

echo "[job.sh] Worker exited with code $EXIT_CODE at $(date)"
exit $EXIT_CODE
