#!/usr/bin/env bash
# Stage a PSA UserInput + its pulse-shape files into a target AFS directory and
# rewrite the pulse-shape references from bare filenames to FULL paths, as the
# official n_TOF processing (condor workers) requires. See PROCESSING.md.
#
# Usage:
#   ./prepare_userinput.sh <target_afs_dir> [userinput.h]
#
# Example (reproduces the run 224489 setup):
#   ./prepare_userinput.sh /afs/cern.ch/work/d/dneff/ntof_processing
#
# Idempotent: rewriting an already-absolute path is a no-op. Run it on lxplus
# (or anywhere the target dir is reachable); commit the SOURCE copy with bare
# names, never the patched one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE/psa_userinput"

TARGET="${1:?usage: prepare_userinput.sh <target_afs_dir> [userinput.h]}"
UI_NAME="${2:-UserInput_2026_EAR2_X17.h}"

mkdir -p "$TARGET"
cp "$SRC_DIR/$UI_NAME" "$TARGET/"
# copy every pulse-shape file that ships alongside the UserInput
shopt -s nullglob
for f in "$SRC_DIR"/X17_*_Signal_*.txt; do cp "$f" "$TARGET/"; done
shopt -u nullglob

UI="$TARGET/$UI_NAME"

# Rewrite each shape filename referenced in the UserInput to its full path.
# Match a bare "NAME.txt" not already preceded by "/" (so re-running is safe).
for f in "$TARGET"/X17_*_Signal_*.txt; do
    base="$(basename "$f")"
    sed -i -E "s#([^/[:alnum:]_])${base}#\1${TARGET%/}/${base}#g; s#^${base}#${TARGET%/}/${base}#g" "$UI"
done

echo "Prepared UserInput at: $UI"
echo "Pulse-shape references now:"
grep -oE "/[^[:space:]]*X17_[A-Z]+_Signal_[0-9]+\.txt" "$UI" | sort -u
echo
echo "Next: launch RunProcessing.sh with  -p $UI   (see PROCESSING.md)"
