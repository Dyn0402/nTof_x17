#!/usr/bin/env bash
# Stage a UserInput variant (and its pulse-shape files) somewhere RunProcessing.sh
# can use it, rewriting the PULSE SHAPE ADDRESS column to FULL PATHS -- which
# both the TWiki and Riccardo's mail insist on.
#
#   ./deploy_userinput.sh v1_flash /afs/cern.ch/work/d/dneff/x17_reproc/userinputs
#
# Then, on lxplus (ssh -K, so you actually get AFS tokens and condor auth):
#
#   /eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh \
#       -y 2026 -a EAR2 -c X17_measurement -r 224572 \
#       -p <dest>/v1_flash/UserInput.h -o <outdir>
#
set -euo pipefail

VARIANT=${1:?usage: deploy_userinput.sh <variant> <dest-dir> [path-as-seen-by-the-processing]}
DEST=${2:?usage: deploy_userinput.sh <variant> <dest-dir> [path-as-seen-by-the-processing]}
# Third argument: the directory the PROCESSING will see, when it differs from
# where we are writing (staging locally for an lxplus path, say).  The template
# existence check is then skipped, since those paths are not local.
REMOTE_BASE=${3:-}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/userinputs/$VARIANT"
SHIPPED=/media/dylan/data/x17/ntof_processing     # Riccardo's original shapes

[ -d "$SRC" ] || { echo "no such variant: $SRC" >&2; exit 1; }

OUT="$DEST/$VARIANT"
mkdir -p "$OUT"
cp "$SRC"/*.txt "$OUT"/ 2>/dev/null || true
# variants that reuse the shipped templates need them alongside
for f in "$SHIPPED"/X17_*_Signal_*.txt; do
    [ -e "$f" ] && [ ! -e "$OUT/$(basename "$f")" ] && cp "$f" "$OUT"/ || true
done

# rewrite bare template filenames to absolute paths, and check every reference
# resolves and that each row's address count matches NUMBER OF PULSE SHAPES
python3 - "$SRC/UserInput.h" "$OUT/UserInput.h" "${REMOTE_BASE:+$REMOTE_BASE/$VARIANT}" "$OUT" <<'PY'
import os, re, sys
src, dst, remote, local = sys.argv[1:5]
base = (remote or local).rstrip('/')
check = not remote
txt = open(src).read()
txt = re.sub(r'(?<![\w/.])(X17_\w+_Signal_\w+\.txt)', base + r'/\1', txt)
open(dst, 'w').write(txt)

bad = 0
for line in txt.splitlines():
    if not line or line[0] in '#.=~ ' or line.startswith('DETECTOR') or \
            line.startswith('  NAME'):
        continue
    tok = line.split()
    if len(tok) < 22:
        continue
    n_declared = int(float(tok[21]))
    addrs = [t for t in tok[22:] if t.endswith('.txt')]
    if len(addrs) != n_declared:
        print(f'  ERROR {tok[0]}: declares {n_declared} shapes, '
              f'{len(addrs)} addresses'); bad += 1
    for a in addrs:
        probe = a if check else os.path.join(local, os.path.basename(a))
        if not os.path.isfile(probe):
            print(f'  MISSING TEMPLATE for {tok[0]}: {a}'); bad += 1
print(f'  {"OK" if not bad else str(bad) + " PROBLEM(S)"}: '
      f'{len(set(re.findall(r"X17_\w+_Signal_\w+.txt", txt)))} distinct templates')
sys.exit(1 if bad else 0)
PY

echo "staged $VARIANT -> $OUT"
