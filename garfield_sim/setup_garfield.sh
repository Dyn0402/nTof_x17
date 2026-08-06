#!/bin/bash
# setup_garfield.sh — the single place that says which Garfield++ we run.
# =============================================================================
# Source it (do not execute it):
#
#     source setup_garfield.sh
#
# It exports ROOT + Garfield++ for whichever host it finds itself on, then
# leaves GARFIELD_INSTALL, LD_LIBRARY_PATH, PYTHONPATH and HEED_DATABASE set so
# that `import ROOT; import Garfield` works from plain python3.
#
# PINNED VERSION: garfieldpp master 927e5c21 (2026-08-06).
# ---------------------------------------------------------------------------
# Why a private build instead of the CVMFS Garfield: the LCG views are far
# behind, and the MX17 response chain (design/RESPONSE_SIM_PLAN.md in
# MX17_Geant) needs things that only exist on master —
#   * Examples/ResistiveMicromegas: woven-mesh + dynamic weighting-potential
#     COMSOL maps, the reference implementation for S2 and the Stage B slow path
#   * AvalancheMicroscopic::GetIons(), for the ion component of the induced signal
#   * the neBEM OpenMP race fix in the SVD inversion (S2 solves)
#   * interface-crossing checks (electrons no longer tunnel through mesh wires,
#     which is exactly the transparency observable)
#   * the FFT convolution fix and the arbitrary-PSD noise generators (Stage C)
# For the record: LCG_108 ships Garfield 6fb94b35 (2025-07-07, 664 commits
# behind the pin) and LCG_109 ships 78fe1bd3 (2026-02-02, 281 behind).
#
# To move the pin: rebuild on each host, then edit MX17_GARFIELD_PIN below and
# the install paths if they change. Nothing else in this directory names a
# Garfield or LCG path.
# =============================================================================

export MX17_GARFIELD_PIN=927e5c21

# LCG view used on lxplus and on condor workers (ROOT 6.38, gcc 14.3).
# Only its ROOT/python/compiler runtime is used — never its Garfield.
MX17_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc14-opt

# ── Condor: unpack the shipped build ─────────────────────────────────────────
# We ship our own pinned build rather than using the CVMFS Garfield. The
# tarball is ~7 MB, so transferring it per job costs nothing and — unlike
# reading an AFS install — does not depend on the worker holding a token.
if [ ! -d "$PWD/garfield/include/Garfield" ]; then
    for _tgz in "$PWD"/garfield-*.tar.gz; do
        [ -f "$_tgz" ] || continue
        echo "[setup_garfield] unpacking $(basename "$_tgz")"
        tar xzf "$_tgz"
        for _d in "$PWD"/*/; do
            if [ -d "$_d/include/Garfield" ]; then
                ln -sfn "${_d%/}" "$PWD/garfield"
                break
            fi
        done
        break
    done
fi

# ── Locate ROOT + the Garfield install ───────────────────────────────────────
# Precedence:
#   1. MX17_GARFIELD_INSTALL set by the caller  (explicit override)
#   2. ./garfield  in the working directory     (condor: shipped tarball)
#   3. CVMFS present                            (lxplus: AFS install)
#   4. a known workstation path                 (desktop / laptop)

if [ -n "$MX17_GARFIELD_INSTALL" ]; then
    _gf_install="$MX17_GARFIELD_INSTALL"
    [ -f "$MX17_LCG_VIEW/setup.sh" ] && source "$MX17_LCG_VIEW/setup.sh"

elif [ -d "$PWD/garfield/include/Garfield" ]; then
    # HTCondor worker: the tarball unpacked above (or a previous job step) left
    # the pinned build in the scratch directory as ./garfield.
    _gf_install="$PWD/garfield"
    source "$MX17_LCG_VIEW/setup.sh"

elif [ -d /cvmfs/sft.cern.ch/lcg/views ]; then
    # lxplus interactive.
    _gf_install=/afs/cern.ch/user/d/dneff/work/garfield_install/lcg109-${MX17_GARFIELD_PIN}
    source "$MX17_LCG_VIEW/setup.sh"

elif [ -d "$HOME/Software/garfield/install" ]; then
    # desktop (dylan-MS-7C84): local ROOT 6.30, gcc 11.4.
    source "$HOME/Software/root_6_30/bin/thisroot.sh"
    _gf_install="$HOME/Software/garfield/install"

elif [ -d "$HOME/garfield/install" ]; then
    # laptop: local ROOT 6.36, gcc 13.3.
    source "$HOME/Software/root_6_36_06/bin/thisroot.sh"
    _gf_install="$HOME/garfield/install"

else
    echo "[setup_garfield] ERROR: no Garfield++ install found on $(hostname)." >&2
    echo "[setup_garfield] Set MX17_GARFIELD_INSTALL to point at one." >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$_gf_install/include/Garfield" ]; then
    echo "[setup_garfield] ERROR: $_gf_install is not a Garfield++ install." >&2
    return 1 2>/dev/null || exit 1
fi

export GARFIELD_INSTALL="$_gf_install"

# lib vs lib64, and the python version in the site-packages path, both vary by
# host — glob rather than hard-code.
for _d in "$GARFIELD_INSTALL"/lib64 "$GARFIELD_INSTALL"/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:$LD_LIBRARY_PATH"
done
for _d in "$GARFIELD_INSTALL"/lib64/python*/site-packages \
          "$GARFIELD_INSTALL"/lib/python*/site-packages; do
    [ -d "$_d" ] && export PYTHONPATH="$_d:$PYTHONPATH"
done
export HEED_DATABASE="$GARFIELD_INSTALL/share/Heed/database"

unset _gf_install _d

echo "[setup_garfield] host=$(hostname -s)  ROOT $(root-config --version)  python $(python3 --version 2>&1 | awk '{print $2}')"
echo "[setup_garfield] Garfield++ $MX17_GARFIELD_PIN at $GARFIELD_INSTALL"
