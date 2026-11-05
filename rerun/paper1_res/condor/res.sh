#!/bin/bash
# res.sh <code_dir> <script> <args...> -- one table-level paper1_res job (res_rows.py / vt_window.py / dp_arms.py) on
# the lock venv, reading the frozen code archive.  Writes only to EOS (its --out); leaves nothing in the scratch dir.
CODE=$1; shift; S=$1; shift
export RES_CODE=$CODE MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
/afs/cern.ch/user/d/dneff/bin/lockpy $CODE/rerun/paper1_res/$S "$@"
