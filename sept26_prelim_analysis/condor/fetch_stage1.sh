#!/usr/bin/env bash
# Pull finished condor outputs home and unpack them into the local product
# tree, so campaign products sit exactly where a desktop run would have left
# them. Parameterised over the three campaign passes -- override PREFIX,
# REMOTE, LOCALPKG and OUT together:
#
#   stage 1  PREFIX=stage1 REMOTE=lxplus:sept26_stage1 OUT=<out>/stage1   (default)
#   slim     PREFIX=slim   REMOTE=lxplus:sept26_slim   OUT=<out>/slim
#   stage 2  PREFIX=run_   REMOTE=lxplus:sept26_stage2 OUT=<out>/fullpass
#
# Every tarball is <something>/out/<stage>/..., so --strip-components=2 lands
# its contents directly in OUT in all three cases.
#
# Idempotent and safe to run while jobs are still finishing: it only ever adds
# files, and a sub-run already unpacked is simply overwritten with an identical
# copy. Run it repeatedly as the queue drains.
#
#   bash fetch_stage1.sh            # pull + unpack whatever is ready
#   bash fetch_stage1.sh --status   # how many are home, how many still out
set -u
# No "~": the shell expands it LOCALLY (to /home/dylan) before rsync sees it.
# A bare relative path is already relative to the remote home.
PREFIX=${PREFIX:-stage1}
REMOTE=${REMOTE:-lxplus:sept26_stage1}
# The tree comes from paths.py, so this and the chains cannot disagree about
# where it is; $X17_ROOT / $X17_SEPT26_OUT move both. A subshell: a fetch script
# has no business changing the caller's working directory.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
x17_path () { (cd "$REPO" && .venv/bin/python -m sept26_prelim_analysis.paths --path "$1"); }
LOCALPKG=${LOCALPKG:-$(x17_path x17)/sept26_stage1}
OUT=${OUT:-$(x17_path out)/stage1}
[ -n "$LOCALPKG" ] && [ -n "$OUT" ] || exit 1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
mkdir -p "$LOCALPKG/tarballs" "$OUT"

if [ "${1:-}" = "--status" ]; then
  n_remote=$($SSH lxplus "ls ${REMOTE#*:}/*.tar.gz 2>/dev/null | wc -l")
  n_local=$(ls "$LOCALPKG/tarballs"/*.tar.gz 2>/dev/null | wc -l)
  n_census=$(ls "$OUT"/census_run_*_stat090_*.csv 2>/dev/null \
             | grep -cE 'census_run_[0-9]+_stat090_[0-9]{4}\.csv$')
  echo "tarballs at CERN : $n_remote"
  echo "tarballs pulled  : $n_local"
  echo "sub-runs unpacked: $n_census / 293   (11 predate this pass)"
  exit 0
fi

echo "=== pulling $(date -Is)"
rsync -a --info=stats1 -e "$SSH" \
      --include="${PREFIX}*.tar.gz" --exclude='*' \
      "$REMOTE/" "$LOCALPKG/tarballs/"
# 24 is "some files vanished before they could be transferred". That is NORMAL
# here: jobs finish and the drain deletes while rsync is walking the directory.
# Treating it as fatal once cost a 3.7 GB pull that was then never unpacked.
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 24 ]; then echo "!! rsync failed (rc=$rc)"; exit 1; fi

echo "=== unpacking"
n=0
for t in "$LOCALPKG/tarballs"/${PREFIX}*.tar.gz; do
  [ -e "$t" ] || continue
  # Each tarball is out/stage1/<the per-sub-run product files>. Strip the two
  # leading components so they land directly in $OUT, matching the layout a
  # local campaign_census.sh run produces.
  tar xzf "$t" -C "$OUT" --strip-components=2 || { echo "!! bad tarball: $t"; continue; }
  n=$((n+1))
done
echo "unpacked $n tarball(s) -> $OUT"

# DRAIN. AFS home is a 10 GB volume shared with everything else at CERN, and
# the slim pass alone produces ~8.8 GB of tarballs. Left to accumulate they
# fill the quota and condor starts failing the output transfer -- which looks
# like a mysterious stall, not a disk error. So once a tarball is pulled AND
# unpacked locally, delete the CERN copy: the local product is the artefact,
# and any tarball lost this way is one condor job to regenerate.
if [ "${DRAIN:-0}" = "1" ] && [ "$n" -gt 0 ]; then
  ls "$LOCALPKG/tarballs"/${PREFIX}*.tar.gz 2>/dev/null \
    | xargs -n1 basename \
    | $SSH lxplus "cd ${REMOTE#*:} && xargs -r rm -f && echo drained" \
    || echo "!! drain failed -- tarballs still at CERN, quota still at risk"
fi
if [ "$PREFIX" = stage1 ]; then
  c=$(ls "$OUT"/census_run_*_stat090_*.csv 2>/dev/null \
      | grep -cE 'census_run_[0-9]+_stat090_[0-9]{4}\.csv$')
  echo "sub-runs with a census now: $c / 293"
else
  echo "product files in $OUT now: $(ls "$OUT" 2>/dev/null | wc -l)"
fi
