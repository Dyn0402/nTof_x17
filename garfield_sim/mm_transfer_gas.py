#!/usr/bin/env python3
"""
mm_transfer_gas.py — Copy gas tables to a remote machine
=========================================================
Transfers .gas files from the local gas_tables/ directory to a remote host
(lxplus, another lab machine, etc.) using rsync over SSH.

Gas files only need to be generated once (Magboltz is slow).  Run this
after mm_generate_gas.py to make the tables available on the remote before
submitting HTCondor jobs.

Usage:
    # Default: to lxplus EOS (uses stored defaults below)
    python3 mm_transfer_gas.py

    # Explicit host / user / path
    python3 mm_transfer_gas.py --host lxplus.cern.ch --user dneff \\
        --target /eos/user/d/dneff/garfield_sim/gas_tables

    # Custom SSH key
    python3 mm_transfer_gas.py --ssh-key ~/.ssh/id_rsa_lxplus

    # Copy to a different machine (e.g. another lab server)
    python3 mm_transfer_gas.py --host myserver.in2p3.fr --user jdoe \\
        --target /data/garfield/gas_tables

    # Preview without transferring
    python3 mm_transfer_gas.py --dry-run

After the transfer, update the gas_tables path used by mm_condor_submit.py
if you placed the files somewhere other than the default EOS path.
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
import mm_config as cfg

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_HOST       = "lxplus.cern.ch"
DEFAULT_USER       = "dneff"
DEFAULT_REMOTE_DIR = "/eos/user/d/dneff/garfield_sim/gas_tables"


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Magboltz gas tables to a remote machine via rsync"
    )
    parser.add_argument("--host",    default=DEFAULT_HOST,
                        help=f"Remote hostname (default: {DEFAULT_HOST})")
    parser.add_argument("--user",    default=DEFAULT_USER,
                        help=f"Remote username (default: {DEFAULT_USER})")
    parser.add_argument("--target",  default=DEFAULT_REMOTE_DIR,
                        help=f"Remote target directory "
                             f"(default: {DEFAULT_REMOTE_DIR})")
    parser.add_argument("--ssh-key", default=None,
                        metavar="PATH",
                        help="Path to SSH private key (optional)")
    parser.add_argument("--port",    default=None, type=int,
                        help="SSH port if non-standard")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show rsync command without executing")
    args = parser.parse_args()

    # ── Discover local gas files ───────────────────────────────────────────────
    if not os.path.isdir(cfg.GAS_DIR):
        print(f"ERROR: gas_tables directory not found: {cfg.GAS_DIR}")
        sys.exit(1)

    gas_files = sorted(f for f in os.listdir(cfg.GAS_DIR) if f.endswith(".gas"))

    if not gas_files:
        print(f"No .gas files found in {cfg.GAS_DIR}")
        print("Run mm_generate_gas.py first.")
        sys.exit(1)

    # ── Print plan ─────────────────────────────────────────────────────────────
    total_bytes = sum(
        os.path.getsize(os.path.join(cfg.GAS_DIR, f)) for f in gas_files
    )

    print("Gas Table Transfer")
    print("=" * 55)
    print(f"Source : {cfg.GAS_DIR}/")
    print(f"Target : {args.user}@{args.host}:{args.target}/")
    print(f"Files  :")
    for f in gas_files:
        size_kb = os.path.getsize(os.path.join(cfg.GAS_DIR, f)) / 1024
        print(f"  {f}  ({size_kb:.0f} KB)")
    print(f"Total  : {total_bytes/1024/1024:.1f} MB")
    print()

    # ── Build rsync command ────────────────────────────────────────────────────
    ssh_cmd = "ssh -o StrictHostKeyChecking=accept-new"
    if args.ssh_key:
        ssh_cmd += f" -i {args.ssh_key}"
    if args.port:
        ssh_cmd += f" -p {args.port}"

    cmd = [
        "rsync",
        "--archive",          # preserves permissions, timestamps, etc.
        "--verbose",
        "--compress",         # compress during transfer (gas files compress well)
        "--progress",
        "--human-readable",
        "-e", ssh_cmd,
        cfg.GAS_DIR + "/",   # trailing slash = copy contents, not the dir itself
        f"{args.user}@{args.host}:{args.target}/",
    ]

    print("rsync command:")
    print("  " + " ".join(cmd))
    print()

    if args.dry_run:
        print("(dry-run — not executing)")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("Transfer complete.")
        print(f"Gas files are now at {args.user}@{args.host}:{args.target}/")
        print()
        print("Next steps:")
        print("  On lxplus: python3 mm_condor_submit.py --dry-run")
    else:
        print(f"\nTransfer failed (rsync exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
