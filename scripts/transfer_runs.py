#!/usr/bin/env python3
"""
transfer_runs.py — Interactive run selector and rsync transfer tool
for n_TOF X17 February beam data.

Directory structure:
  runs/
    run_88/
      resist_525V_drift_500V/
        raw_daq_data/
        hits_root/
        decoded_root/
        combined_hits_root/
        hv_monitor.csv
      resist_530V_drift_1000V/
        ...
      run_config.json

Filters:
  • Runs below MIN_RUN always excluded
  • EXCLUDE_RUNS: manually excluded run numbers
  • hits_root/ and combined_hits_root/ skipped globally (all runs)
  • decoded_root/ skipped only for runs in SKIP_DECODED_RUNS
  • .fdf files skipped globally if SKIP_FDF = True
"""

import glob
import os
import re
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR  = "/media/dylan/data/x17/feb_beam/runs"
DEST_HOST   = "192.168.0.89"
DEST_DIR    = "/media/dylan/data/x17/feb_beam/runs"
DEST_BUDGET = 470 * 1024 ** 3   # 470 GiB in bytes
MIN_RUN     = 17                 # skip run_N where N < MIN_RUN

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_run_number(name: str) -> int | None:
    m = re.fullmatch(r"run_(\d+\w*)", name)
    return m.group(1) if m else None


def du_bytes(paths: list[str]) -> dict[str, int]:
    """Run du -sb on a list of paths, return {path: bytes}."""
    if not paths:
        return {}
    r = subprocess.run(["du", "-sb"] + paths, capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            out[parts[1].strip()] = int(parts[0])
    return out


def subconfig_dirs(run_path: str) -> list[str]:
    """Return all immediate subdirectories inside a run dir, regardless of naming."""
    try:
        return sorted(e.path for e in os.scandir(run_path) if e.is_dir())
    except OSError:
        return []


def scan_run(run_path: str) -> dict:
    """
    Return size breakdown for one run:
      total            — full du of run dir
      hits             — sum of all */hits_root/ sizes
      combined_hits    — sum of all */combined_hits_root/ sizes
      decoded          — sum of all */decoded_root/ sizes
      fdf              — sum of all *.fdf files (recursive)
    """
    total = du_bytes([run_path]).get(run_path, 0)

    hits_dirs    = []
    combined_dirs = []
    decoded_dirs = []

    for hv_dir in subconfig_dirs(run_path):
        for subdir, bucket in [
            ("hits_root",          hits_dirs),
            ("combined_hits_root", combined_dirs),
            ("decoded_root",       decoded_dirs),
        ]:
            p = os.path.join(hv_dir, subdir)
            if os.path.isdir(p):
                bucket.append(p)

    hits_total     = sum(du_bytes(hits_dirs).values())
    combined_total = sum(du_bytes(combined_dirs).values())
    decoded_total  = sum(du_bytes(decoded_dirs).values())

    fdf_total = 0
    for f in glob.glob(os.path.join(run_path, "**", "*.fdf"), recursive=True):
        try:
            fdf_total += os.path.getsize(f)
        except OSError:
            pass

    return {
        "total":         total,
        "hits":          hits_total,
        "combined_hits": combined_total,
        "decoded":       decoded_total,
        "fdf":           fdf_total,
    }


def get_run_info(source: str) -> dict[str, dict]:
    run_dirs = sorted(glob.glob(os.path.join(source, "run_*")))
    if not run_dirs:
        return {}
    info = {}
    for run_path in run_dirs:
        name = os.path.basename(run_path)
        if re.fullmatch(r"run_\w+", name):
            print(f"    scanning {name}...{' '*20}", end="\r", flush=True)
            info[name] = scan_run(run_path)
    print(" " * 50, end="\r")
    return info


def effective_size(info: dict, skip_hits: bool, skip_decoded: bool,
                   skip_fdf: bool) -> int:
    size = info["total"]
    if skip_hits:
        size -= info["hits"]
        size -= info["combined_hits"]
    if skip_decoded:
        size -= info["decoded"]
    if skip_fdf:
        size -= info["fdf"]
    return max(size, 0)


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def bar(used: int, total: int, width: int = 30) -> str:
    frac = min(used / total, 1.0)
    filled = int(frac * width)
    color = "\033[92m" if frac < 0.85 else "\033[93m" if frac < 1.0 else "\033[91m"
    return f"{color}[{'#' * filled}{'.' * (width - filled)}]\033[0m {frac*100:.1f}%"


def print_summary(sizes: dict[str, int], budget: int, skip_fdf: bool,
                  skip_decoded_runs: set[str]) -> None:
    total = sum(sizes.values())
    print(f"\n  Runs selected        : {len(sizes)}")
    print(f"  Total xfer size      : {human(total)}")
    print(f"  Budget               : {human(budget)}")
    print(f"  Usage                : {bar(total, budget)}")
    print(f"  Skip hits_root       : always (all runs)")
    print(f"  Skip comb. hits_root : always (all runs)")
    print(f"  Skip .fdf            : {'yes' if skip_fdf else 'no'}")
    if skip_decoded_runs:
        print(f"  Skip decoded_root    : {', '.join(sorted(skip_decoded_runs))}")
    else:
        print(f"  Skip decoded_root    : none (copied for all runs)")
    if total > budget:
        print(f"\033[91m  WARNING: OVER BUDGET by {human(total - budget)}\033[0m")
    else:
        print(f"\033[92m  OK: {human(budget - total)} remaining\033[0m")


def print_table(sizes: dict[str, int], all_info: dict, skip_fdf: bool,
                skip_decoded_runs: set[str]) -> None:
    sorted_runs = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    total = sum(sizes.values())

    print(f"\n  {'RUN':<16} {'TOTAL':>9}  {'--HITS':>8}  {'--CHITS':>8}  {'--DEC':>8}  {'--FDF':>7}  {'XFER':>9}  {'CUMUL':>9}")
    print("  " + "-" * 90)
    running = 0
    for name, xfer in sorted_runs:
        running += xfer
        info = all_info[name]
        pct      = xfer / total * 100 if total else 0
        mini_bar = "#" * max(1, int(pct / 3))
        dec_col  = human(info["decoded"]) if name in skip_decoded_runs else "  -"
        fdf_col  = human(info["fdf"])     if skip_fdf                  else "  -"
        print(f"  {name:<16} {human(info['total']):>9}  {human(info['hits']):>8}  "
              f"{human(info['combined_hits']):>8}  {dec_col:>8}  {fdf_col:>7}  "
              f"{human(xfer):>9}  {human(running):>9}   {mini_bar} {pct:.1f}%")
    print()


def normalise_run(tok: str) -> str:
    tok = tok.strip()
    if tok.isdigit():
        return f"run_{tok}"
    if not tok.startswith("run_"):
        return f"run_{re.sub(r'^run', '', tok)}"
    return tok


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Scripting config — edit these ─────────────────────────────────────────
    DRY_RUN  = False   # set False to actually transfer
    SKIP_FDF = True   # skip *.fdf files globally

    # Run numbers to exclude entirely
    EXCLUDE_RUNS = []

    # hits_root/ and combined_hits_root/ are ALWAYS skipped for all runs.
    # decoded_root/ is skipped ONLY for runs listed here:
    SKIP_DECODED_RUNS: list[int] = [83, 99, 100, 101, 102, 103, 104, 105, 106]
    # ──────────────────────────────────────────────────────────────────────────

    exclude_set       = {f"run_{n}" for n in EXCLUDE_RUNS}
    skip_decoded_runs = {f"run_{n}" for n in SKIP_DECODED_RUNS}

    print("\n-- n_TOF Run Transfer Tool --")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Dest   : {DEST_HOST}:{DEST_DIR}")
    print(f"  Budget : {human(DEST_BUDGET)}\n")

    print("  Scanning run sizes...")
    try:
        all_info = get_run_info(SOURCE_DIR)
    except Exception as e:
        print(f"  Error scanning source: {e}")
        sys.exit(1)

    if not all_info:
        print("  No run_* directories found. Check SOURCE_DIR.")
        sys.exit(1)

    # Auto-exclude runs below MIN_RUN
    auto_excluded = {
        name for name in all_info
        if (n := parse_run_number(name)) is not None and int(re.search(r'\d+', n).group()) < MIN_RUN
    }
    excluded = auto_excluded | exclude_set

    print(f"  Found {len(all_info)} runs total.")
    print(f"  Auto-excluded {len(auto_excluded)} runs (run_N < {MIN_RUN}).")
    if exclude_set:
        print(f"  Script-excluded : {', '.join(sorted(exclude_set))}")

    def compute_sizes() -> dict[str, int]:
        return {
            name: effective_size(
                all_info[name],
                skip_hits=True,
                skip_decoded=(name in skip_decoded_runs),
                skip_fdf=SKIP_FDF,
            )
            for name in all_info if name not in excluded
        }

    # Interactive loop
    while True:
        sizes = compute_sizes()
        total = sum(sizes.values())

        print_summary(sizes, DEST_BUDGET, SKIP_FDF, skip_decoded_runs)

        if total <= DEST_BUDGET:
            print("\n  Within budget. Run list (largest transfer first):\n")
        else:
            print(f"\n  Over budget. Exclude runs to fit under {human(DEST_BUDGET)}:\n")

        print_table(sizes, all_info, SKIP_FDF, skip_decoded_runs)

        prompt = (
            "  Commands:\n"
            "    x  <runs>  -- exclude runs entirely       (e.g. x 107 114)\n"
            "    sd <runs>  -- skip decoded_root for runs  (e.g. sd 107 118)\n"
            "    Enter      -- proceed to transfer\n"
            "  > "
        )
        answer = input(prompt).strip()
        if not answer:
            break

        tokens = re.split(r"\s+", answer, maxsplit=1)
        cmd  = tokens[0].lower()
        rest = tokens[1] if len(tokens) > 1 else ""
        names = [normalise_run(t) for t in re.split(r"[\s,]+", rest) if t.strip()]

        if cmd == "x":
            added = set()
            for name in names:
                if name in all_info and name not in excluded:
                    excluded.add(name)
                    added.add(name)
                else:
                    print(f"  '{name}' not found or already excluded.")
            if added:
                print(f"  Excluded: {', '.join(sorted(added))}")

        elif cmd == "sd":
            added = set()
            for name in names:
                if name in all_info and name not in excluded:
                    skip_decoded_runs.add(name)
                    added.add(name)
                else:
                    print(f"  '{name}' not found or already excluded.")
            if added:
                print(f"  Will skip decoded_root for: {', '.join(sorted(added))}")

        else:
            print("  Unknown command -- use 'x' to exclude or 'sd' to skip decoded_root.")

    # Build rsync filter flags
    sizes = compute_sizes()
    filter_flags: list[str] = []

    # Exclude entire runs
    for run in sorted(excluded):
        filter_flags += ["--exclude", f"{run}/"]

    # hits_root and combined_hits_root — excluded at any depth
    filter_flags += ["--exclude", "hits_root/"]
    filter_flags += ["--exclude", "combined_hits_root/"]

    # decoded_root — excluded only for specific runs: run_N/*/decoded_root/
    for run in sorted(skip_decoded_runs):
        if run not in excluded:
            filter_flags += ["--exclude", f"{run}/*/decoded_root/"]

    # .fdf files
    if SKIP_FDF:
        filter_flags += ["--exclude", "*.fdf"]

    rsync_cmd = [
        "rsync", "-avh", "--progress", "--stats",
        *filter_flags,
        f"{SOURCE_DIR}/",
        f"dylan@{DEST_HOST}:{DEST_DIR}/"
    ]

    print("\n-- Transfer Plan --")
    print(f"  Runs to transfer : {len(sizes)}")
    print(f"  Estimated size   : {human(sum(sizes.values()))}")
    print(f"\n  rsync command:\n    " + " \\\n      ".join(rsync_cmd) + "\n")

    if DRY_RUN:
        print("  DRY_RUN = True -- running rsync --dry-run:\n")
        subprocess.run(rsync_cmd + ["--dry-run"])
        return

    confirm = input("  Proceed with transfer? [y/N] ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        sys.exit(0)

    print("\n-- Transferring... --\n")
    result = subprocess.run(rsync_cmd)
    if result.returncode == 0:
        print("\nTransfer complete.")
    else:
        print(f"\nrsync exited with code {result.returncode}.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
