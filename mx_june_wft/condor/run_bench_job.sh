#!/usr/bin/env bash
# run_bench_job.sh — condor worker for one reconstruction-benchmark scan point.
#
#   run_bench_job.sh <cache.pkl> <bundle.tgz> <scan_points.json> <index> <tag>
#
# Reads scan_points.json[index] for the hyper patch / window crop / event split
# and runs bench/run_bench.py, which scores the fit against the cached M3 truth.
# Writes result_bench_<tag>.tar.gz for condor to bring back.
set -eu

CACHE=$1; BUNDLE_TGZ=$2; SCAN=$3; IDX=$4; TAG=$5

echo "[$(date -u +%H:%M:%S)] $(hostname) scan point $IDX ($TAG)"

LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
if [ -f "$LCG" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$LCG"
  set -u
else
  echo "WARNING: $LCG not found, falling back to the system python" >&2
fi

tar xzf payload.tar.gz
tar xzf "$BUNDLE_TGZ"                 # -> bundle/
BUNDLE=$(dirname "$(find . -name bundle.json | head -1)")

# the production reconstruction configuration (HANDOFF_2026-07-30)
export WFT_MODEL_FRAC=0.03 WFT_PRESCAN=1 WFT_CHI2DOF_BAD=250

read -r RUN_KEY PATCH CROP EVENTS SPLIT <<EOF
$(python - "$SCAN" "$IDX" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))[int(sys.argv[2])]
print(p['run_key'],
      json.dumps(p['patch'], separators=(',', ':')),
      p['crop'] or 'none', p['events'], p['split'])
PY
)
EOF

ARGS=(--cache "$CACHE" --bundle "$BUNDLE" --variant prod --tag "$TAG"
      --subset "$EVENTS" --subset-mod "$SPLIT" --out-dir out
      --jobs "${OMP_NUM_THREADS:-2}")
[ "$PATCH" != '{}' ] && ARGS+=(--patch "$PATCH")
[ "$CROP" != 'none' ] && ARGS+=(--crop "$CROP")

mkdir -p out
time python payload/mx_june_wft/bench/run_bench.py "$RUN_KEY" "${ARGS[@]}"

tar czf "result_bench_${TAG}.tar.gz" out
echo "[$(date -u +%H:%M:%S)] done"
