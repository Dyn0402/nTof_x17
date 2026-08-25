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
SLIM=${X17_SLIM_BASE:-/eos/experiment/ntof/data/x17/wf_pull/slim_input}
DEST=${X17_WF_DEST:-root://eosuser.cern.ch//eos/experiment/ntof/data/x17/wf_pull/out}
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

# ---- is there anything to build? -------------------------------------------
# BEFORE the raw listing, which takes minutes to hours when the whole fleet
# hits EOS and CTA at once. A run with no slim under $SLIM can produce nothing,
# and discovering that after the listing is how 56 jobs burned ~3 h each.
n_slim=$(find "$SLIM" -name "ntof_hits_*_${RUN}.root" 2>/dev/null | wc -l)
if [ "$n_slim" -eq 0 ]; then
    echo "FAIL: no slim products for $RUN under $SLIM"
    echo "      (X17_SLIM_BASE=${X17_SLIM_BASE:-<unset, using default>})"
    exit 3
fi
echo "$n_slim slim segment(s) for $RUN"

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

    # A CTA recall does NOT stay staged. Measured 2026-08-13: a recall that read
    # 3187/3187 files online was back to online:false on 26 of 27 runs inside a
    # day, with no pin held (`requested:false`), and 224614/224616 died on
    # `[3005] no disk replica exists` at their FIRST xrdcp. So the recall must be
    # consumed, not banked -- see campaign.sh, which couples recall to submit.
    #
    # Check before reading rather than after: a job that starts on an evicted
    # run wastes a slot and, worse, could xrdcp its way to a PARTIAL read that
    # looks like a quiet detector downstream.
    sed "s|^$XRD||" "$RAWLIST" > "$RAWLIST.paths"
    n_online=$(xargs -a "$RAWLIST.paths" -n 50 xrdfs "$XRD" query prepare 0 2>/dev/null |
               tr -d ' \n' | grep -o '"online":true' | wc -l)
    n_want=$(wc -l < "$RAWLIST")
    echo "tape staging: $n_online of $n_want files online"
    if [ "$n_online" -lt "$n_want" ]; then
        echo "FAIL: $RUN is not fully staged ($n_online/$n_want online)."
        echo "      The recall has expired or never completed. Re-request it and"
        echo "      submit this run WITHIN THE SAME PASS -- recalled files fall"
        echo "      back to tape-only in well under a day."
        exit 3
    fi
fi
echo "$(wc -l < "$RAWLIST") raw files to read"

# ---- the pull --------------------------------------------------------------
set +e
python -m ntof_processing.waveform_pull.pull_run "$RUN" \
    --slim-base "$SLIM" --raw-list "$RAWLIST" --out "$OUT" \
    --window-ns "$WINDOW" --tflash-fallback-on-incomplete $STAGE_ARG $EXTRA
rc=$?
set -e
echo "pull_run exit $rc"

# ---- verify BEFORE publishing ----------------------------------------------
# `--write-json` drops a <product>_verify.json beside each product so the fleet
# verdict is read from summaries rather than grepped out of prose in job logs
# (that habit produced four confident wrong numbers in one night on this
# project). The verdict does not gate the publish -- a product that fails
# closure is still evidence, and the raw may be gone by the time anyone looks --
# but it travels WITH the product instead of dying in a log.
find "$OUT" -name 'ntof_wf_*.root' -print0 |
    xargs -0 -r python -m ntof_processing.waveform_pull.verify --write-json || true

# ---- publish COMPLETE products only ----------------------------------------
# The provenance JSON is written last, by `SegmentWriter.close`, so its presence
# is exactly the statement "this product is finished". Publishing on the .root
# alone put three files with no `blocks` tree at all onto EOS on 2026-08-13,
# from a job that died mid-scan -- indistinguishable from a real product until
# something tries to open one.
published=0 skipped=0
while IFS= read -r prov; do
    stem=${prov%_provenance.json}
    if [ ! -f "$stem.root" ]; then
        echo "  ORPHAN provenance with no product: $prov"; continue
    fi
    for f in "$stem.root" "$prov" "${stem}_verify.json"; do
        [ -f "$f" ] || continue
        rel=${f#"$OUT"/}
        echo "  publish $rel"
        xrdcp --force --silent "$f" "$DEST/$rel" \
            || { echo "  XRDCP FAILED: $rel"; exit 4; }
        published=$((published + 1))
    done
done < <(find "$OUT" -type f -name 'ntof_wf_*_provenance.json')

while IFS= read -r f; do
    [ -f "${f%.root}_provenance.json" ] && continue
    echo "  NOT PUBLISHED (half-written, no provenance): ${f#"$OUT"/}"
    skipped=$((skipped + 1))
done < <(find "$OUT" -type f -name 'ntof_wf_*.root')

# The run-level summary is not a product and carries no provenance of its own.
if [ -f "$OUT/pull_$RUN.json" ]; then
    xrdcp --force --silent "$OUT/pull_$RUN.json" "$DEST/pull_$RUN.json" \
        && published=$((published + 1))
fi
echo "published $published files to $DEST ($skipped incomplete products withheld)"

echo "=== done, exit $rc, $(date -u)"
exit $rc
