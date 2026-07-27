#!/usr/bin/env bash
# Stage the ntof_dream_merge reference pair: DREAM run_79 stat090_0000+0001 <-> nTOF 224572.
#
#   ./stage_reference_pair.sh ntof     # xrdcp the official nTOF file from EOS (26 GB)
#   ./stage_reference_pair.sh denom    # precompute the decoded_root denominator caches (~1 MB each)
#   ./stage_reference_pair.sh check    # report what is staged and verify sizes
#   ./stage_reference_pair.sh manifest # write the laptop-bundle file list + rsync command
#
# No Kerberos needed: the nTOF official processing area is world-readable over xrootd.
# Override the pair with NTOF_RUN / DREAM_RUN / DREAM_SUBRUNS.
set -euo pipefail

NTOF_RUN=${NTOF_RUN:-224572}
DREAM_RUN=${DREAM_RUN:-run_79}
DREAM_SUBRUNS=${DREAM_SUBRUNS:-"stat090_0000 stat090_0001"}

BASE=/mnt/data/x17/beam_july
RUNS=$BASE/runs
NTOF_DIR=$BASE/ntof_data
ANA=$BASE/analysis/ntof_dream_merge
FT_CACHE=$BASE/analysis/flash_timing_threshold/cache
EOS_DONE=/eos/experiment/ntof/processing/official/done
XRD=root://eosexperiment.cern.ch
VENV=${VENV:-$HOME/PycharmProjects/nTof_x17/.venv/bin/python}

mkdir -p "$NTOF_DIR" "$ANA/cache"

case "${1:-check}" in

ntof)
    dst=$NTOF_DIR/run${NTOF_RUN}.root
    # --continue resumes a partial transfer, so this is safe to re-run.
    xrdcp --continue --nopbar --streams 15 --retry 3 \
        "$XRD/$EOS_DONE/run${NTOF_RUN}.root" "$dst"
    echo "staged $dst ($(stat -c%s "$dst") bytes)"
    ;;

pkup)
    # Small per-bunch index (BunchNumber, psTime, PulseIntensity, tflash) straight off EOS.
    # Useful on its own: the bunch join in Phase 3 needs nothing else from the nTOF side.
    "$VENV" - "$NTOF_RUN" "$ANA/cache" <<'PY'
import sys, uproot, numpy as np
from pathlib import Path
run, out = sys.argv[1], Path(sys.argv[2])
url = f'root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/run{run}.root'
a = uproot.open(url)['PKUP'].arrays(
    ['BunchNumber', 'psTime', 'PulseIntensity', 'tflash'], library='np')
o = np.argsort(a['BunchNumber'])
p = out / f'pkup_{run}.csv'
with p.open('w') as fh:
    fh.write('BunchNumber,psTime_ns,PulseIntensity,tflash_ns\n')
    for i in o:
        fh.write('%d,%.1f,%.6g,%.4f\n' % (a['BunchNumber'][i], a['psTime'][i],
                                          a['PulseIntensity'][i], a['tflash'][i]))
print(f'wrote {p} ({len(o)} bunches)')
PY
    ;;

denom)
    # flash_timing_lib caches dt_ms + n_flash per subrun from decoded_root, but keys the
    # cache on an md5 of the decoded-file LIST -- which a laptop without decoded_root can
    # never reproduce. So re-export a portable copy keyed only on run/subrun. Shipping
    # those means the laptop never needs decoded_root (~10 GB/subrun).
    "$VENV" - "$DREAM_RUN" "$ANA/cache" $DREAM_SUBRUNS <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/mnt/data/x17/beam_july/analysis/flash_timing_threshold')
import flash_timing_lib as FT
run, out, subs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
for s in subs:
    d = FT.load_subrun(run, s)
    p = out / f'denom_{run}_{s}.npz'
    np.savez_compressed(p, dt_ms=d['dt_ms'], n_flash=d['n_flash'], n_events=d['n_events'])
    print(f"{run}/{s}: n_flash={d['n_flash']} n_events={d['n_events']} "
          f"n_single={d['n_single']} -> {p.name}")
PY
    ;;

check)
    dst=$NTOF_DIR/run${NTOF_RUN}.root
    want=$(xrdfs "$XRD" stat "$EOS_DONE/run${NTOF_RUN}.root" 2>/dev/null \
           | awk '/^Size:/{print $2}')
    have=$(stat -c%s "$dst" 2>/dev/null || echo 0)
    printf 'nTOF run%s : %s / %s bytes' "$NTOF_RUN" "$have" "${want:-?}"
    [[ -n "${want:-}" && "$have" == "$want" ]] && echo '  COMPLETE' || echo '  INCOMPLETE'
    for s in $DREAM_SUBRUNS; do
        d=$RUNS/$DREAM_RUN/$s/combined_hits_root
        printf 'DREAM %s/%-14s : %s files, %s\n' "$DREAM_RUN" "$s" \
            "$(ls "$d" 2>/dev/null | wc -l)" "$(du -sh "$d" 2>/dev/null | cut -f1)"
        ls "$FT_CACHE"/${DREAM_RUN}_${s}_*.npz 2>/dev/null | sed 's/^/    denom: /'
    done
    ls "$ANA/cache"/pkup_*.csv 2>/dev/null | sed 's/^/pkup: /'
    ;;

manifest)
    m=$ANA/laptop_bundle.txt
    : > "$m"
    for s in $DREAM_SUBRUNS; do
        echo "runs/$DREAM_RUN/run_config.json" >> "$m"
        echo "runs/$DREAM_RUN/dream_daq.log" >> "$m"
        echo "runs/$DREAM_RUN/$s/combined_hits_root/" >> "$m"
        echo "runs/$DREAM_RUN/$s/hv_monitor.csv" >> "$m"
    done
    echo "ntof_data/run${NTOF_RUN}.root" >> "$m"
    echo "analysis/ntof_dream_merge/cache/" >> "$m"
    echo "slow_control/beam_intensity/" >> "$m"
    echo "slow_control/stream1_filesize/" >> "$m"
    sort -u -o "$m" "$m"
    echo "wrote $m:"; cat "$m"
    cat <<EOF

Pull to the laptop with (run FROM the laptop):
  rsync -av --info=progress2 --files-from=<(ssh mx17 cat $m) \\
        mx17:$BASE/  ~/x17/beam_july/
EOF
    ;;

*) echo "unknown mode: $1" >&2; exit 2 ;;
esac
