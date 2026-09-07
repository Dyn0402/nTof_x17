#!/usr/bin/env bash
# Re-run everything downstream of the waveform pass, in dependency order, and
# publish. Idempotent: safe to run twice, and safe to run from cron.
#
#   merge_fullpass  the flat CERN pass -> the nested per-sub-run layout
#   k_arm           the in-situ angle scale, per chamber (needs the pass)
#   funnel          trigger -> track -> n_TOF confirmation (needs stage-1 census)
#   pairs           the controlled two-chamber rate (needs k_arm)
#   make_figures    the five figures (needs funnel + k_arm)
#   make_report     report.html + body.html (needs all of the above)
#   rsync           -> lxplus:/eos/user/d/dneff/www/x17/reco-funnel/
#
# Each step must succeed before the next runs: a report built on a half-updated
# k_arm would be worse than no report.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN=${RUN:-run_145}
SUBRUNS=${SUBRUNS:-stat090_0000,stat090_0001,stat090_0002}
OUT=/media/dylan/data/x17/sept26_prelim/funnel

step () {  # step <name> <command...>
  echo; echo "=== $1  $(date -Is)"
  shift
  if ! "$@"; then echo "!! FAILED: $*"; exit 1; fi
}

step "merge_fullpass" $PY -W ignore -m sept26_prelim_analysis.merge_fullpass --run "$RUN"
step "k_arm"        $PY -W ignore -m sept26_prelim_analysis.k_arm --run "$RUN" --subruns "$SUBRUNS"
step "funnel"       $PY -W ignore -m sept26_prelim_analysis.funnel --run "$RUN" --subruns "$SUBRUNS"
step "pairs"        $PY -W ignore -m sept26_prelim_analysis.pairs --run "$RUN" --subruns "$SUBRUNS"
step "figures"      $PY -W ignore -m sept26_prelim_analysis.make_figures --run "$RUN"
step "report"       $PY -W ignore -m sept26_prelim_analysis.make_funnel_report --run "$RUN"

echo; echo "=== publish  $(date -Is)"
cp "$OUT/report.html" "$OUT/index.html"
rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=25" \
      "$OUT/index.html" "$OUT/figures" \
      lxplus:/eos/user/d/dneff/www/x17/reco-funnel/ || {
  echo "!! rsync failed -- the local products are still good"; exit 1; }
code=$(curl -sS -o /dev/null -w '%{http_code}' \
       https://dylan-neff.web.cern.ch/x17/reco-funnel/ || echo 000)
echo "live: HTTP $code"
echo "=== done $(date -Is)"
