#!/usr/bin/env bash
# gap_campaign2.sh — remaining fleet datasets, 4 jobs each so the machine stays
# usable for other work. Waits for any dataset already in flight.
set -u
cd "$(dirname "$0")/../.."
G=mx_june_wft/bench/gap_consistency.sh
LOG=/tmp/wft_logs; mkdir -p "$LOG"

while pgrep -f "bench/gap_consistency.sh" > /dev/null; do sleep 30; done

run () {
  local key=$1; shift
  local v=$1; shift
  echo "[$(date +%H:%M:%S)] === starting $key"
  nice -n 5 $G "$key" "$v" "$@" > "$LOG/GAPRUN_$key.log" 2>&1
  echo "[$(date +%H:%M:%S)] === $key exit $? ; $(tail -1 "$LOG/GAPRUN_$key.log")"
}

run g_det4      34.16 --jobs 4 --lowgain --backup --chain
run g_det6_long 26.7  --jobs 4 --lowgain --backup --chain
run g_det7_long 36.6  --jobs 4 --lowgain --backup --chain
echo "[$(date +%H:%M:%S)] === campaign complete"
