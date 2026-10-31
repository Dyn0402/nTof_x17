#!/bin/bash
# wait_streams.sh -- block until no run_streams.sh process is left on the shared lxplus node (local helper)
while true; do
  n=$(timeout 120 ssh lxplus 'ps -u dneff -o args | grep -c "[r]un_streams.sh"' 2>/dev/null)
  [ "$n" = "0" ] && break
  sleep 600
done
echo "streams done"
