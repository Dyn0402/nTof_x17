#!/bin/bash
# Push the flash-pull code (no data) to lxplus:~/x17flash.
#
#   ./stage.sh            then on lxplus: cd x17flash && ./flash_campaign.sh plan
set -eu
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
DIR=x17flash
ssh lxplus "mkdir -p $DIR/logs $DIR/inputs $DIR/ntof_processing"
rsync -a --delete --include='*/' --include='*.py' --exclude='*' \
    "$REPO/ntof_processing/" "lxplus:$DIR/ntof_processing/"
rsync -a "$REPO/ntof_processing/flash_pull/lxplus/"{flash_wrapper.sh,flash.sub,flash_campaign.sh} "lxplus:$DIR/"
rsync -a "$REPO/ntof_processing/flash_pull/inputs/run_inventory_2026-10-20.csv" "lxplus:$DIR/inputs/run_inventory.csv"
ssh lxplus "chmod +x $DIR/*.sh"
echo "staged to lxplus:$DIR"
