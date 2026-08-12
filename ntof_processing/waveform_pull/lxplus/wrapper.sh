#!/bin/bash
# Condor payload: pull one n_TOF run's waveforms and publish them to EOS.
#
#   ./wrapper.sh <ntof_run>
#
# Raw source is chosen per run: the EOS disk copy if it is COMPLETE, otherwise
# the CTA copy (which must already have been recalled).  A partial disk copy is
# never used -- it is indistinguishable downstream from a quiet detector, and
# `pull_run.py` would (correctly) fail the run at the end after doing all the
# work.  Better to notice here.
# `set -u` is deliberately NOT set: LCG's own setup.sh references an unbound
# COMPILER and dies instantly under it, taking the job with it.
set -eo pipefail

RUN=${1:?usage: wrapper.sh <ntof_run>}
XRD=${X17_CTA_XRD:-root://eosctapublicdisk.cern.ch/}
CTA=${X17_CTA_BASE:-/eos/ctapublicdisk/archive/ntof/2026/EAR2/X17_measurement}
DISK=${X17_NTOF_RAW:-/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement}
SLIM=${X17_SLIM_BASE:-/eos/experiment/ntof/data/x17/july_beam}
DEST=${X17_WF_DEST:-root://eosuser.cern.ch//eos/experiment/ntof/data/x17/july_beam}
WINDOW=${X17_WF_WINDOW_NS:-5000}
EXTRA=${X17_WF_EXTRA:-}

echo "=== waveform pull, run $RUN, window +-$WINDOW ns, $(date -u)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

WORK=$PWD
OUT=$WORK/out
STAGE=$WORK/stage
mkdir -p "$OUT" "$STAGE"

# `common.beam_july_paths` resolves the DREAM tree AT IMPORT and raises if it
# finds none, and `ntof_io` imports it at module level -- so without this the
# job dies on the first tflash read, i.e. after the raw list is already built.
export X17_BEAM_JULY=${X17_BEAM_JULY:-/eos/experiment/ntof/data/x17/july_beam}
export X17_SLIM_CACHE=${_CONDOR_SCRATCH_DIR:-$WORK}/ntof_cache
mkdir -p "$X17_SLIM_CACHE"

# Preflight BEFORE any raw is read. `transfer_input_files = pkg/` with a
# trailing slash flattens the package into the scratch root, and discovering
# that after a 0.5 TB read is an expensive way to learn it.
for f in ntof_processing/waveform_pull/pull_run.py \
         ntof_dream_merge/ntof_io.py common/beam_july_paths.py; do
    [ -f "$f" ] || { echo "MISSING INPUT: $f"; ls -la; exit 3; }
done
python -c "import sys; sys.path.insert(0,'.')
import ntof_processing.waveform_pull.pull_run
from ntof_dream_merge import ntof_io" \
    || { echo "package imports are broken in the sandbox"; exit 3; }
# The EOS write depends on a Kerberos credential forwarded by the schedd
# (MY.SendCredential). It is valid ~19 h from SUBMIT, not from start, so a job
# that queued a long time can wake with a dead ticket -- and would then do the
# entire 0.5 TB read before failing to publish a byte of it. Check first.
if ! klist -s 2>/dev/null; then
    echo "FAIL: no valid Kerberos credential on the worker; the EOS publish"
    echo "      would fail after the whole raw read. Resubmit with a fresh"
    echo "      ticket (kinit on the submit node, then condor_submit)."
    klist 2>&1 | head -5
    exit 3
fi
echo "credential OK: $(klist 2>/dev/null | awk '/krbtgt/ {print $1, $2, "->", $3, $4}' | head -1)"
echo "preflight OK"

# ---- decide the raw source -------------------------------------------------
n_cta=$(xrdfs "$XRD" ls "$CTA/$RUN/stream1" 2>/dev/null | grep -c '_s1\.raw\.finished$' || true)
n_disk=$(ls "$DISK/$RUN/stream1" 2>/dev/null | grep -c '_s1\.raw' || true)
echo "raw: $n_disk on disk, $n_cta on tape"

RAWLIST=$WORK/raw_$RUN.list
if [ "$n_disk" -gt 0 ] && [ "$n_disk" -ge "$n_cta" ]; then
    echo "using the EOS disk copy (complete)"
    # The disk copies carry the SAME `.finished` suffix as the tape ones, so
    # the glob must allow it -- `*_s1.raw` alone matches nothing at all.
    ls "$DISK/$RUN/stream1"/run${RUN}_*_s1.raw* |
        sed 's|.*/run[0-9]*_\([0-9]*\)_s1\.raw.*|\1 &|' | sort -n | cut -d' ' -f2- \
        > "$RAWLIST"
    [ -s "$RAWLIST" ] || { echo "FAIL: disk listing produced no files"; exit 3; }
    STAGE_ARG=""
else
    if [ "$n_cta" -eq 0 ]; then
        echo "FAIL: $RUN has no complete raw anywhere (disk $n_disk, tape $n_cta)"
        exit 3
    fi
    echo "disk copy is absent or partial ($n_disk of $n_cta) -- using tape"
    xrdfs "$XRD" ls "$CTA/$RUN/stream1" | grep '_s1\.raw\.finished$' |
        sed "s|.*/run${RUN}_\([0-9]*\)_s1.*|\1 $XRD&|" | sort -n | cut -d' ' -f2- \
        > "$RAWLIST"
    STAGE_ARG="--stage-dir $STAGE"
fi
echo "$(wc -l < "$RAWLIST") raw files to read"

# ---- the pull --------------------------------------------------------------
set +e
python -m ntof_processing.waveform_pull.pull_run "$RUN" \
    --slim-base "$SLIM" --raw-list "$RAWLIST" --out "$OUT" \
    --window-ns "$WINDOW" $STAGE_ARG $EXTRA
rc=$?
set -e
echo "pull_run exit $rc"

# ---- publish, whatever the exit code ---------------------------------------
# A run that failed its completeness check still produced real data for the
# bunches it did see, and the provenance JSON records exactly which. Publish it
# with the verdict attached rather than throwing the raw read away.
published=0
while IFS= read -r f; do
    rel=${f#"$OUT"/}
    echo "  publish $rel"
    xrdcp --force --silent "$f" "$DEST/$rel" || { echo "  XRDCP FAILED: $rel"; exit 4; }
    published=$((published + 1))
done < <(find "$OUT" -type f \( -name '*.root' -o -name '*.json' \))
echo "published $published files to $DEST"

# ---- verify what landed ----------------------------------------------------
find "$OUT" -name 'ntof_wf_*.root' -print0 |
    xargs -0 -r python -m ntof_processing.waveform_pull.verify || true

echo "=== done, exit $rc, $(date -u)"
exit $rc
