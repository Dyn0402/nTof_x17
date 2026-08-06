#!/usr/bin/env bash
# gap_campaign.sh — every remaining June dataset through the gap pipeline,
# sequentially (8 cores, one dataset at a time). See gap_consistency.sh.
#
# det3 : sat_det3 (done) + the 6-27 P2 overnight (same slot, next day)
#        + the 6-22 long_run (different day AND slot: bottom, FEU 3/4)
# det2 : o22_long_det2 (done) + the 6-22 long_run subrun (8x stats)
# det4/6/7 : first maps (old-kernel generation backed up, full chain re-run)
set -u
cd "$(dirname "$0")/../.."
G=mx_june_wft/bench/gap_consistency.sh
LOG=/tmp/wft_logs; mkdir -p "$LOG"

# wait for a dataset already in flight
while pgrep -f "bench/gap_consistency.sh g_det3_wknd" > /dev/null; do sleep 30; done

run () {  # run <key> <v> [extra args...]
  local key=$1; shift
  local v=$1; shift
  echo "[$(date +%H:%M:%S)] === starting $key"
  $G "$key" "$v" "$@" > "$LOG/GAPRUN_$key.log" 2>&1
  echo "[$(date +%H:%M:%S)] === $key exit $? ; $(tail -1 "$LOG/GAPRUN_$key.log")"
}

run g_det3      36.6  --limit 12000 --jobs 8
run g_det2      39.94 --limit 12000 --jobs 8
run g_det4      34.16 --jobs 8 --lowgain --backup --chain
run g_det6_long 26.7  --jobs 8 --lowgain --backup --chain
run g_det7_long 36.6  --jobs 8 --lowgain --backup --chain
echo "[$(date +%H:%M:%S)] === campaign complete"
