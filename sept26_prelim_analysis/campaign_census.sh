#!/usr/bin/env bash
# Stage-1 census over the frozen campaign sample, streamed and resumable.
#
# WHY STREAMED. The sample is 293 sub-runs and ~150 GB of combined_hits, more
# than is comfortable to hold alongside everything else on a 477 GB disk. So
# each sub-run is staged from EOS, classified, and its staged hits deleted
# before the next -- peak local cost is one sub-run per worker, a few GB in
# total, instead of 150 GB.
#
# WHY RESUMABLE. candidate_filter runs at 20-30 events/s in one process, so the
# 25.6 M-trigger sample is 17-25 h wall even across 8 workers. That is a
# multi-night job, not an overnight one. A sub-run whose census CSV already
# exists is skipped, so stopping and restarting costs nothing, and the claim is
# an atomic mkdir so two workers never take the same sub-run.
#
# WHY 8 WORKERS AND NOT 16. Half the work is an rsync from EOS, and EOS is a
# shared facility system. 8 concurrent streams is a reasonable neighbour; the
# CPU is not the binding constraint anyway.
#
#   bash campaign_census.sh            # run until the sample is done
#   WORKERS=4 bash campaign_census.sh  # gentler
#   bash campaign_census.sh --status   # what is done, what is left
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNS=/media/dylan/data/x17/beam_july/runs
EOS=/eos/experiment/ntof/data/x17/july_beam/runs
OUT=/media/dylan/data/x17/sept26_prelim/stage1
WORK=/media/dylan/data/x17/sept26_prelim/stage1/.claims
LOG=/media/dylan/data/x17/sept26_prelim/stage1/campaign.log
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
WORKERS=${WORKERS:-8}
#: consecutive staging failures a worker tolerates before it gives up.
#: Small on purpose -- see the back-off comment in worker().
MAX_FAILS=${MAX_FAILS:-5}
LIST=/media/dylan/data/x17/sept26_prelim/stage1/campaign_worklist.txt

# ---------------------------------------------------------------- the worklist
if [ ! -s "$LIST" ]; then
  $PY -W ignore -c "
import pandas as pd
s = pd.read_csv('/media/dylan/data/x17/sept26_prelim/stage0/sample.csv')
k = s[s.in_sample]
for _, r in k.sort_values(['run', 'subrun']).iterrows():
    print(f'run_{r[\"run\"]} {r[\"subrun\"]}')
" > "$LIST"
fi
TOTAL=$(wc -l < "$LIST")

# Exactly census_<run>_<subrun>.csv -- the per-TAG files written by a --tag
# run (census_run_79_stat090_0000_260726_18H07_000.csv) carry a trailing tag
# and must not be counted as a finished sub-run.
done_count () { ls "$OUT"/census_run_*_stat090_*.csv 2>/dev/null \
                 | grep -cE 'census_run_[0-9]+_stat090_[0-9]{4}\.csv$'; }

if [ "${1:-}" = "--status" ]; then
  d=$(done_count)
  echo "sample:    $TOTAL sub-runs"
  echo "done:      $d  ($(awk -v a=$d -v b=$TOTAL 'BEGIN{printf "%.1f", 100*a/b}') %)"
  # A claim is never removed on success, so counting claims counts finished
  # sub-runs too. In flight = claimed AND no census written.
  inf=0
  for c in "$WORK"/*; do
    [ -d "$c" ] || continue
    b=$(basename "$c"); r=${b%_stat090_*}; sub=stat090_${b##*_stat090_}
    [ -s "$OUT/census_${r}_${sub}.csv" ] || inf=$((inf+1))
  done
  echo "in flight: $inf"
  echo
  # candidate_filter's numpy warnings go to the same log, so filter for the
  # progress lines rather than tailing raw -- this is the line Dylan reads.
  grep -E '^\[.*\] (w[0-9]+ .*(start|done|FAILED)|campaign census)' "$LOG" \
    2>/dev/null | tail -8
  exit 0
fi

mkdir -p "$WORK" "$OUT"

# A claim is a directory held for the life of a worker, so a killed run leaves
# claims behind and those sub-runs would be skipped forever on restart -- the
# claim blocks re-claiming and the census it should have written never appears.
# Clear any claim with no finished census and no live process: nothing else can
# be holding one, because this script is the only thing that makes them and it
# is not running yet.
if ! pgrep -f 'campaign_census\.sh' | grep -qv "^$$\$"; then
  n=0
  for c in "$WORK"/*; do
    [ -d "$c" ] || continue
    b=$(basename "$c")
    r=${b%_stat090_*}; sub=stat090_${b##*_stat090_}
    [ -s "$OUT/census_${r}_${sub}.csv" ] || { rmdir "$c" 2>/dev/null && n=$((n+1)); }
  done
  [ "$n" -gt 0 ] && echo "cleared $n stale claim(s) from a previous run" | tee -a "$LOG"
fi

worker () {
  local id=$1
  local fails=0
  while read -r RUN SUB; do
    [ -s "$OUT/census_${RUN}_${SUB}.csv" ] && continue          # already done
    mkdir "$WORK/${RUN}_${SUB}" 2>/dev/null || continue         # atomic claim
    local D="$RUNS/$RUN/$SUB/combined_hits_root"
    # ALWAYS rsync, never "the directory exists so it must be complete".
    # A partially staged sub-run -- one tag pulled by hand for a spot check,
    # say -- would otherwise be classified as if it were the whole thing and
    # written out under the full sub-run's name. rsync is incremental, so this
    # costs nothing when the directory really is complete.
    mkdir -p "$D"
    if ! rsync -a -e "$SSH" "lxplus:$EOS/$RUN/$SUB/combined_hits_root/" "$D/" \
         >>"$LOG" 2>&1; then
      echo "[$(date -Is)] w$id $RUN/$SUB STAGE FAILED" >>"$LOG"
      rmdir "$WORK/${RUN}_${SUB}" 2>/dev/null
      # BACK OFF, DO NOT STORM. A systemic failure -- ssh refused, EOS down,
      # ticket expired -- fails EVERY sub-run, and without this the worker
      # simply walks the worklist retrying, which on 2026-09-08 turned one
      # refused connection into 1,318 attempts against lxplus in about a
      # minute. That is abusive to shared infrastructure regardless of intent.
      fails=$((fails + 1))
      if [ "$fails" -ge "$MAX_FAILS" ]; then
        echo "[$(date -Is)] w$id ABORTING: $fails consecutive staging failures." \
             "This is a systemic problem (check: ssh lxplus true), not this" \
             "sub-run. Fix it, then re-run -- finished sub-runs are skipped." \
             | tee -a "$LOG"
        return 1
      fi
      sleep $((fails * 10))
      continue
    fi
    fails=0
    # the slim too, for the IMPLIED class -- small, and incremental as well
    rsync -a -e "$SSH" "lxplus:$EOS/$RUN/$SUB/ntof_hits/" \
          "$RUNS/$RUN/$SUB/ntof_hits/" >>"$LOG" 2>&1 || true
    rsync -a -e "$SSH" "lxplus:$EOS/$RUN/run_config.json" \
          "$RUNS/$RUN/" >>"$LOG" 2>&1 || true
    echo "[$(date -Is)] w$id $RUN/$SUB start" >>"$LOG"
    if nice -n 10 $PY -W ignore -m sept26_prelim_analysis.candidate_filter \
         --run "$RUN" --subrun "$SUB" --all-tags >>"$LOG" 2>&1; then
      echo "[$(date -Is)] w$id $RUN/$SUB done  ($(done_count)/$TOTAL)" >>"$LOG"
    else
      echo "[$(date -Is)] w$id $RUN/$SUB CLASSIFY FAILED" >>"$LOG"
    fi
    # Give the disk back. run_145 is Dylan's own local copy and predates this
    # script, so it is never deleted; everything else was pulled here and can
    # be pulled again.
    [ "$RUN" != "run_145" ] && rm -rf "$D"
  done < "$LIST"
}

echo "[$(date -Is)] campaign census: $TOTAL sub-runs, $(done_count) already done, $WORKERS workers" | tee -a "$LOG"
for i in $(seq 1 "$WORKERS"); do worker "$i" & done
wait
echo "[$(date -Is)] campaign census finished: $(done_count)/$TOTAL" | tee -a "$LOG"
