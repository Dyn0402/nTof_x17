#!/bin/bash
# Push the backfilled beam record from the local archive up to banco's live log
# directory, skipping exactly the files banco's own bridge would overwrite.
#
# WHY THE SKIP LIST. beam_bridge.py re-xrdcp's four names from EOS on EVERY 20 s
# poll, unconditionally, overwriting whatever is on banco:
#
#     beam_intensity_<today>.csv
#     sps_spill_<today>.csv
#     sps_profile_<today>_<this hour>.jsonl.gz
#     sps_profile_<today>_<previous hour>.jsonl.gz
#
# Pushing those is pointless while the bridge runs — they are clobbered within
# 20 s. Everything else is safe, because the only other path that writes is the
# start-up catch-up plan, and that skips any file where the local copy is >= the
# remote one by size (_catch_up_plan in beam_bridge.py). Our merged files are
# strictly larger than what is on EOS, so they are never re-pulled.
#
# h4_tax_*.csv is safe unconditionally: the bridge has no TAX code at all and
# does not know the prefix.
#
# So: run this now for everything historical, and re-run it after midnight to
# land the four files that were today's. Re-running is harmless — it is a plain
# copy of a file the bridge has been told to leave alone.

set -euo pipefail

ARCHIVE=${ARCHIVE:-/media/dylan/data/x17/sps_run53_det4_check/records/beam/backfill_nxcals}
REMOTE=${REMOTE:-banco_cern}
DEST=${DEST:-DAQ_Control_Dream_Beam/beam_monitor/logs}
TODAY=$(date +%Y-%m-%d)
HOUR=$(date +%H)
PREV=$(date -d '1 hour ago' +%Y-%m-%d_%H)

cd "$ARCHIVE"

skip=(
  "beam_intensity_${TODAY}.csv"
  "sps_spill_${TODAY}.csv"
  "sps_profile_${TODAY}_${HOUR}.jsonl.gz"
  "sps_profile_${PREV}.jsonl.gz"
)

send=()
while IFS= read -r f; do
  for s in "${skip[@]}"; do [[ "$f" == "$s" ]] && continue 2; done
  send+=("$f")
done < <(ls sps_spill_2026-*.csv beam_intensity_2026-*.csv h4_tax_2026-*.csv \
             sps_profile_2026-*.jsonl.gz 2>/dev/null)

echo "archive : $ARCHIVE"
echo "dest    : $REMOTE:$DEST"
echo "sending : ${#send[@]} file(s)"
echo "skipping: ${skip[*]}"
echo

if [[ ${#send[@]} -eq 0 ]]; then echo "nothing to send"; exit 0; fi

# One tar stream: 200+ small-ish files over one connection, and banco's disk is
# the constraint, not ours.
tar cf - "${send[@]}" | ssh "$REMOTE" "mkdir -p '$DEST' && tar xf - -C '$DEST'"
echo "done"
