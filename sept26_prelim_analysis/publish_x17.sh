#!/usr/bin/env bash
# publish_x17.sh -- push report directories to https://dylan-neff.web.cern.ch/x17/
#
# One place that knows which output directory is served at which URL. The
# publish loop used to live inline at the bottom of rerun_chain.sh, which meant
# a report built by any other chain (the campaign QA chain, the det-A chain,
# the acceptance/fold chain) had no way to go live without editing that file.
#
# Each report ships as a DIRECTORY, not a single file: index.html beside its
# figures/, so the "numbers" link under every figure still resolves to the CSV
# the plot was drawn from. That is the whole reason these are not notes -- a
# note is one self-contained document with the PNGs inlined, and inlining
# throws the CSVs away.
#
#   ./publish_x17.sh                 # everything in the registry that exists
#   ./publish_x17.sh det-a-pairs …   # only the slugs named
#   DRY=1 ./publish_x17.sh           # say what would be sent, send nothing
#
# Needs a Kerberos ticket (`kinit dneff@CERN.CH`). rsync runs with --delete
# INSIDE each slug directory only, so a figure dropped from a report stops
# being served; nothing outside x17/<slug>/ is ever touched.
set -u

BASE=${BASE:-/media/dylan/data/x17/sept26_prelim}
DEST=${DEST:-lxplus:/eos/user/d/dneff/www/x17}
DRY=${DRY:-}

# output directory under $BASE  :  the slug it is served at under /x17/
#
# The first six are the run_145 pass (rerun_chain.sh builds them); the rest are
# the campaign-wide reports of 2026-09-10. Keep the comment on any slug whose
# name does not say which sample it is over -- opening-angle is run_145 and
# opening-angle-campaign is all 36 runs, and that is not obvious from the URL.
REGISTRY=(
  "funnel:reco-funnel"                     # run_145 reconstruction funnel
  "scint:scintillators"                    # run_145 scintillator study
  "imaging:source-imaging"                 # run_145 capsule pointing
  "angle:opening-angle"                    # run_145 opening angle
  "ipc:ipc-continuum"                      # data-free: the IPC prediction
  "ganil:ganil-background"                 # data-free: the same at GANIL/NFS
  "det_a_intra:det-a-pairs"                # campaign: intra-A pairs
  "det_a_scint:det-a-scintillators"        # campaign: A vs the scintillators
  "fold_campaign:acceptance-fold"          # campaign: per-run acceptance + fold
  "angle_campaign:opening-angle-campaign"  # campaign: opening angle, 36 runs
  "imaging_campaign:capsule-imaging"       # campaign: capsule, once per run
)

want=("$@")
sent=0; skipped=0; failed=0

for pair in "${REGISTRY[@]}"; do
  dir=${pair%%:*}; slug=${pair##*:}

  if (( ${#want[@]} )); then
    hit=
    for w in "${want[@]}"; do [ "$w" = "$slug" ] && hit=1; done
    [ -n "$hit" ] || continue
  fi

  src="$BASE/$dir"
  if [ ! -f "$src/report.html" ]; then
    echo "-- $slug: no $src/report.html, skipping"
    skipped=$((skipped + 1)); continue
  fi

  # The web server wants index.html; the DAQ Analysis tab lists report.html.
  # Both are the same bytes, and the copy is here rather than in the report
  # generators so that a report is not born knowing where it will be served.
  cp "$src/report.html" "$src/index.html"

  payload=("$src/index.html")
  [ -d "$src/figures" ] && payload+=("$src/figures")

  echo "== $slug  <- $dir  ($(du -sh "$src/figures" 2>/dev/null | cut -f1 || echo 0) of figures)"
  if [ -n "$DRY" ]; then
    echo "   dry run: rsync ${payload[*]} -> $DEST/$slug/"
    continue
  fi

  if ! rsync -a --delete -e "ssh -o BatchMode=yes -o ConnectTimeout=25" \
       "${payload[@]}" "$DEST/$slug/"; then
    echo "!! rsync failed for $slug -- the local products are still good"
    failed=$((failed + 1)); continue
  fi

  code=$(curl -sS -o /dev/null -w '%{http_code}' \
         "https://dylan-neff.web.cern.ch/x17/$slug/" || echo 000)
  echo "   live: https://dylan-neff.web.cern.ch/x17/$slug/  HTTP $code"
  [ "$code" = 200 ] || failed=$((failed + 1))
  sent=$((sent + 1))
done

echo
echo "published $sent, skipped $skipped, failed $failed"
exit $(( failed > 0 ))
