#!/usr/bin/env bash
# Waveform-first chain for one run key:
#   bundle (import or calibrate) -> reco -> alignment -> efficiency -> angles -> maps
#
#   mx_june_wft/run_chain.sh <run_key> [--legacy DIR] [--hyper-file F] [--jobs N]
#                                      [--calibrate] [--seed-bundle DIR] [--skip-reco]
#
# Every stage writes under <Analysis>/<run>/<subrun>/<det>/wft/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
KEY=${1:?run key}; shift || true

JOBS=12; LEGACY=""; HYPER="hyper_v2.json"; CALIB=0; SEED=""; SKIP_RECO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS=$2; shift 2;;
    --legacy) LEGACY=$2; shift 2;;
    --hyper-file) HYPER=$2; shift 2;;
    --calibrate) CALIB=1; shift;;
    --seed-bundle) SEED=$2; shift 2;;
    --skip-reco) SKIP_RECO=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

LOG_DIR=/tmp/wft_logs; mkdir -p "$LOG_DIR"
log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "$*"; "$@" || { log "FAILED: $*"; exit 1; }; }

if [ "$CALIB" = 1 ]; then
  ARGS=(--jobs "$JOBS")
  [ -n "$SEED" ] && ARGS+=(--seed-bundle "$SEED")
  run $PY -m wft.calibrate "$KEY" "${ARGS[@]}"
else
  ARGS=(--hyper-file "$HYPER")
  [ -n "$LEGACY" ] && ARGS+=(--from-legacy "$LEGACY")
  run $PY -m wft.cli bundle "$KEY" "${ARGS[@]}"
fi

if [ "$SKIP_RECO" = 0 ]; then
  run $PY -m wft.cli reco "$KEY" --jobs "$JOBS" --matched-only
fi

run $PY mx_june_wft/01_alignment.py "$KEY"
run $PY mx_june_wft/02_efficiency.py "$KEY" --max-dropped -1     # headline: no cluster cut
run $PY mx_june_wft/02_efficiency.py "$KEY"                       # variant: n_dropped<=2
$PY mx_june_wft/02_efficiency.py "$KEY" --source hits || log "hits-source comparison unavailable"
run $PY mx_june_wft/03_angles.py "$KEY"
run $PY mx_june_wft/04_maps.py "$KEY" || log "maps stage failed (non-fatal)"
run $PY mx_june_wft/digest.py "$KEY"
log "chain complete for $KEY"
