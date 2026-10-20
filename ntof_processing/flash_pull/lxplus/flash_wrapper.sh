#!/bin/bash
# Condor payload: the flash pull for one n_TOF run, published to EOS from the worker.
#
#   ./flash_wrapper.sh <ntof_run>          (plan.csv must be in the sandbox)
#
# Each planned file is read from its EOS disk copy if it is still there, else
# from tape, which must already be online (flash_campaign.sh recalls a batch and
# submits it in the same pass: recalled files go offline again in under a day).
# Products are published, then DELETED from the scratch dir, so nothing large
# can come back to AFS.
# No `set -u`: LCG's setup.sh references an unbound COMPILER and dies under it.
set -eo pipefail

RUN=${1:?usage: flash_wrapper.sh <ntof_run>}
XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
DEST=${X17_FLASH_DEST:-root://eosuser.cern.ch//eos/experiment/ntof/data/x17/flash_tail}

echo "=== flash pull, run $RUN, $(date -u), $(hostname)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

WORK=$PWD
OUT=$WORK/out
STAGE=$WORK/stage
mkdir -p "$OUT" "$STAGE"
cleanup() { rm -rf "$OUT" "$STAGE"; }
trap cleanup EXIT

[ -f plan.csv ] && [ -f ntof_processing/flash_pull/pull.py ] \
    || { echo "MISSING INPUT"; ls -la; exit 3; }
python -c "import sys; sys.path.insert(0,'.'); import ntof_processing.flash_pull.pull" \
    || { echo "package imports are broken in the sandbox"; exit 3; }
if ! klist -s 2>/dev/null; then
    echo "FAIL: no valid Kerberos credential on the worker; the EOS publish would fail."
    exit 3
fi
echo "preflight OK"

# ---- this run's files: disk if still there, else tape if online -------------
RAWS=() IDXS=()
while IFS=, read -r run idx source path _rest; do
    [ "$run" = "$RUN" ] || continue
    if [ "$source" = disk ] && [ -f "$path" ]; then
        RAWS+=("$path")
    else
        url=$(python -c "import sys; sys.path.insert(0,'.')
from ntof_processing.waveform_pull import config as C; print(C.cta_url($RUN, $idx))")
        p=${url#"$XRD"}
        if ! xrdfs "$XRD" query prepare 0 "$p" 2>/dev/null | tr -d ' \n' | grep -q '"online":true'; then
            echo "FAIL: file $idx of $RUN is not on disk and not online on tape -- recall it first"
            exit 5
        fi
        RAWS+=("$url")
    fi
    IDXS+=("$idx")
done < plan.csv
[ "${#RAWS[@]}" -gt 0 ] || { echo "FAIL: $RUN has no rows in plan.csv"; exit 3; }
echo "${#RAWS[@]} raw files: ${IDXS[*]}"

set +e
python -m ntof_processing.flash_pull.pull "$RUN" --raw "${RAWS[@]}" --idx "${IDXS[@]}" \
    --out "$OUT" --stage-dir "$STAGE"
rc=$?
set -e
echo "pull exit $rc"

# ---- publish finished products only (provenance is written last) -----------
published=0
for prov in "$OUT"/flash_*_provenance.json; do
    [ -f "$prov" ] || continue
    stem=${prov%_provenance.json}
    [ -f "$stem.root" ] || { echo "  ORPHAN $prov"; continue; }
    for f in "$stem.root" "$prov"; do
        xrdcp --force --silent "$f" "$DEST/$RUN/$(basename "$f")" \
            || { echo "  XRDCP FAILED: $f"; exit 4; }
        published=$((published + 1))
    done
done
echo "published $published files to $DEST/$RUN"
echo "=== done, exit $rc, $(date -u)"
exit $rc
