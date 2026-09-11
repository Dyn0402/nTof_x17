#!/usr/bin/env bash
# The campaign stage-2 FULL PASS -- gate on a smoke test, submit, watch, pull.
#
# Dylan's call, 2026-09-09: the stage-1 filter keeps only 12.8 % of the
# triggers that reconstruct into a two-track event, so reconstruct everything
# rather than risk missing pairs. ~12 932 condor jobs, ~16 000 core-hours.
# STATUS.md carries the measurement and the timing caveat that argued against.
#
# Products land in a NEW tree. `<out>/fullpass` already holds the ALLOWLIST
# pass's reco despite its name; unpacking on top of it would silently replace
# a known sample with a superset and destroy the comparison between them.
#
# Resumable at every step: the submit skips anything already on EOS, so
# re-running after a partial night picks up only what is missing.
set -u
cd "$(dirname "$0")/.."
# The tree comes from paths.py, so a chain and the Python it calls cannot
# disagree about where it is; $X17_ROOT / $X17_SEPT26_OUT still move both.
OUT=$(.venv/bin/python -m sept26_prelim_analysis.paths --path out) || exit 1
PKG=$(.venv/bin/python -m sept26_prelim_analysis.paths --path x17)/sept26_fullpass
EOSDIR=/eos/user/d/dneff/x17/sept26_fullpass
SMOKE=4156277
PY=.venv/bin/python
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=60"
log () { echo "[$(date -Is)] $*"; }

# --------------------------------------------------------------------------
log "=== STEP 0: wait for the smoke cluster $SMOKE (4 jobs, run_145/stat090_0000 tag 000)"
$SSH lxplus "until [ -z \"\$(condor_q $SMOKE -af ClusterId 2>/dev/null)\" ]; do sleep 60; done
  echo '  drained'; condor_history $SMOKE -af ProcId ExitCode RemoteWallClockTime MemoryUsage 2>/dev/null"

log "=== STEP 0b: VALIDATE the smoke output against the August full pass"
# The August blind pass of this exact tag gives A 3233, B 3098, C 3256, D 6690
# events. A full pass that reproduces those counts is fitting everything the
# seeder finds; one that comes back near the allowlist's ~4 % is not.
mkdir -p "$PKG/smoke"
rsync -a -e "$SSH" \
  --include='run_145_stat090_0000_beam_*_260805_14H06_000.tar.gz' --exclude='*' \
  "lxplus:$EOSDIR/" "$PKG/smoke/" || { log "!! smoke rsync failed"; exit 1; }
n=$(ls "$PKG/smoke"/*.tar.gz 2>/dev/null | wc -l)
log "  pulled $n/4 smoke tarballs"
[ "$n" -eq 4 ] || { log "!! FATAL: expected 4 smoke tarballs, got $n -- NOT submitting"; exit 1; }

rm -rf "$PKG/smoke/x"; mkdir -p "$PKG/smoke/x"
for t in "$PKG/smoke"/*.tar.gz; do tar xzf "$t" -C "$PKG/smoke/x" --strip-components=1; done
$PY - "$PKG/smoke/x" <<'PYEOF'
import sys, os, pandas as pd
base = sys.argv[1]
want = dict(A=3233, B=3098, C=3256, D=6690)     # August blind pass, same tag
bad = []
for arm, exp in want.items():
    p = os.path.join(base, f'mx17_{arm}', 'events_260805_14H06_000.parquet')
    if not os.path.exists(p):
        bad.append(f'{arm}: MISSING {p}'); continue
    n = pd.read_parquet(p, columns=['event_id']).event_id.nunique()
    ok = abs(n - exp) <= 0.02 * exp
    print(f'  {arm}: {n:6d} events, August full pass {exp:6d}  '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        bad.append(f'{arm}: {n} vs {exp}')
if bad:
    print('SMOKE FAILED: ' + '; '.join(bad)); sys.exit(1)
print('SMOKE PASSED -- the jobs are fitting the whole tag')
PYEOF
[ $? -eq 0 ] || { log "!! smoke validation FAILED -- NOT submitting the campaign"; exit 1; }

# --------------------------------------------------------------------------
log "=== STEP 1: submit everything not already on EOS"
# GUARD: submit only when the queue is EMPTY. This script has to be safe to
# re-run -- the watch loop below is where a night goes wrong -- and a job that
# is RUNNING has produced no EOS tarball yet, so the done-list alone would
# happily submit a second copy of everything in flight.
if [ -n "$($SSH lxplus 'condor_q -af ClusterId 2>/dev/null' | head -1)" ]; then
  log "  jobs already in the queue -- skipping submission, going straight to the watch"
else
$SSH lxplus "cd ~/sept26_fullpass
  eos root://eosuser.cern.ch ls $EOSDIR 2>/dev/null | sed 's/\.tar\.gz\$//' > done.txt
  awk -F, 'NR==FNR{d[\$1];next} !(\$3 in d)' done.txt jobs.txt > jobs_todo.txt
  echo \"  already done \$(wc -l < done.txt), to submit \$(wc -l < jobs_todo.txt)\"
  if [ -s jobs_todo.txt ]; then
    condor_submit stage2_fullpass.sub -append 'jobfile = jobs_todo.txt' | tail -2
  else
    echo '  nothing left to submit'
  fi"
fi

log "=== STEP 2: watch until nothing is left RUNNING OR IDLE"
# Held jobs are deliberately NOT counted here. A deterministically-failing job
# holds, gets released, fails again and holds again forever, so a loop that
# waits for the queue to be empty never exits -- which is exactly what happened
# on the first attempt at 00:27 with run_104/stat090_0016.
$SSH lxplus '
  while true; do
    s=$(condor_q -af JobStatus 2>/dev/null)
    a=$(echo "$s" | grep -c "^[12]" || true)
    h=$(echo "$s" | grep -c "^5" || true)
    e=$(eos root://eosuser.cern.ch ls '"$EOSDIR"' 2>/dev/null | wc -l)
    echo "  $(date -Is) active=$a held=$h on_eos=$e"
    [ "$a" -eq 0 ] && break
    sleep 600
  done'

log "=== STEP 3: release the held once -- transient or deterministic?"
# The allowlist pass released 324 holds and retired 0 as deterministic, so a
# blanket release is the right first move. What survives it is real.
$SSH lxplus 'condor_release dneff 2>&1 | tail -1
  sleep 120
  while [ "$(condor_q -af JobStatus 2>/dev/null | grep -c "^[12]" || true)" -gt 0 ]; do sleep 600; done'

log "=== STEP 3b: record what stayed held, then clear it so the queue drains"
$SSH lxplus 'condor_q -hold -af:h ClusterId ProcId Arguments HoldReason 2>/dev/null | head -40
  n=$(condor_q -af JobStatus 2>/dev/null | grep -c "^5" || true)
  echo "  $n job(s) held after a release -- treating as deterministic"
  [ "$n" -gt 0 ] && condor_rm -constraint "JobStatus == 5" 2>&1 | tail -1
  echo "  final on_eos=$(eos root://eosuser.cern.ch ls '"$EOSDIR"' 2>/dev/null | wc -l) of 12932"
  true'

log "=== STEP 4: pull and unpack into a NEW tree"
REMOTE=lxplus:$EOSDIR LOCALPKG=$PKG OUT=$OUT/reco_fullpass \
  bash sept26_prelim_analysis/condor/fetch_stage2.sh

log "=== DONE.  Left for a decision, NOT run here:"
log "  k_arm per run on the new reco, then campaign_tracks --fullpass $OUT/reco_fullpass"
log "  The borrowed run_145 k and its 4-7 % systematic are unchanged by this pass."
