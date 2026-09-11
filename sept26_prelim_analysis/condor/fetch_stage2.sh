#!/usr/bin/env bash
# Pull finished campaign stage-2 reco tarballs home and unpack each into the
# NESTED per-sub-run layout every downstream consumer wants:
#
#   <out>/fullpass/<run>/<subrun>/mx17_<arm>/events_<tag>[.candidates].parquet
#
# WHY NOT merge_fullpass.py. That module solves the same problem for the August
# CERN pass, but it maps tag -> sub-run through a hardcoded three-entry table
# for run_145 and cannot generalise to 293 sub-runs. Here the tarball NAME
# already carries run and sub-run (make_stage2_campaign.py builds it that way
# precisely so this step needs no map), so the nesting is read off the filename
# and no lookup table can go stale.
#
# WHY THE NESTING MATTERS. Unpacked flat, two sub-runs' arms would land in one
# mx17_<arm>/ directory and load_reco would silently concatenate them into a
# single "sub-run". Tags happen to be globally unique today (they encode a
# timestamp), so the collision would not overwrite -- it would just quietly
# merge sub-runs, which is worse.
#
# Idempotent: re-unpacking a tarball overwrites with identical content.
set -u
# EOS, not AFS: the stage-2 jobs xrdcp their own tarballs there, because the
# access point cannot write to EOS and AFS home is only 10 GB.
# See EOS_WRITE_TEST.md.
REMOTE=${REMOTE:-lxplus:/eos/user/d/dneff/x17/sept26_stage2}
LOCALPKG=${LOCALPKG:-/home/dylan/x17/sept26_stage2}
OUT=${OUT:-/media/dylan/data/x17/sept26_prelim/fullpass}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
mkdir -p "$LOCALPKG/tarballs" "$OUT"

if [ "${1:-}" = "--status" ]; then
  echo "tarballs at CERN : $($SSH lxplus "ls ${REMOTE#*:}/run_*.tar.gz 2>/dev/null | wc -l")"
  echo "tarballs pulled  : $(ls "$LOCALPKG/tarballs"/run_*.tar.gz 2>/dev/null | wc -l)"
  echo "sub-runs unpacked: $(find "$OUT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
  exit 0
fi

echo "=== pulling $(date -Is)"
rsync -a --info=stats1 -e "$SSH" \
      --include='run_*.tar.gz' --exclude='*' \
      "$REMOTE/" "$LOCALPKG/tarballs/" || { echo "!! rsync failed"; exit 1; }

echo "=== unpacking"
n=0; bad=0
for t in "$LOCALPKG/tarballs"/run_*.tar.gz; do
  [ -e "$t" ] || continue
  b=$(basename "$t" .tar.gz)              # run_<N>_<subrun>_beam_<arm>_<tag>
  run=${b%%_stat090_*}                    # run_<N>
  rest=${b#*_stat090_}                    # <NNNN>_beam_<arm>_<tag>
  sub=stat090_${rest%%_beam_*}
  case "$run/$sub" in
    run_*/stat090_[0-9][0-9][0-9][0-9]) ;;
    *) echo "!! cannot parse run/sub-run from $b -- skipped"; bad=$((bad+1)); continue;;
  esac
  d="$OUT/$run/$sub"
  mkdir -p "$d"
  # strip the leading out/ so mx17_<arm>/ lands directly under the sub-run
  tar xzf "$t" -C "$d" --strip-components=1 || { echo "!! bad tarball: $t"; bad=$((bad+1)); continue; }
  n=$((n+1))
done
echo "unpacked $n tarball(s), $bad problem(s) -> $OUT"
echo "sub-runs present: $(find "$OUT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
