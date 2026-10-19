#!/usr/bin/env python3
"""make_figures.py -- fill a paper's figures/ directory from its figures.toml.

    python papers/make_figures.py paper1            # all figures of Paper I
    python papers/make_figures.py paper2 --only kernel_forms
    python papers/make_figures.py paper1 --run      # run each producer first
    python papers/make_figures.py all --list        # status table, nothing written

Each [fig.<id>] entry in <paper>/figures.toml becomes figures/<id>.pdf or
figures/<id>.png, so the .tex includes it as \\includegraphics{figures/<id>}
(no extension) and never changes when a figure is swapped.

    state = "current"     copy `source` (repo-relative) as it is
    state = "superseded"  render `source` faded under a red "do not quote" stamp
    state = "placeholder" draw a grey box with the title, note and producer

A current/superseded entry whose source is missing falls back to a placeholder,
so the paper always compiles. To swap in a new figure: point `source` at it,
set state = "current", re-run, commit, then `papers/overleaf.sh sync <paper>`.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAPERS = ["paper1", "paper2"]
EXTS = (".pdf", ".png", ".jpg")

STATE_COLOUR = {"ready": "#2e7d32", "partial": "#ef6c00", "open": "#c62828",
                "requote": "#6a1b9a", "deferred": "#616161"}


def load(paper):
    with open(HERE / paper / "figures.toml", "rb") as f:
        return tomllib.load(f)["fig"]


def clear(figdir, fid):
    for ext in EXTS:
        (figdir / f"{fid}{ext}").unlink(missing_ok=True)


def wrap(s, n):
    return "\n".join(textwrap.wrap(" ".join(s.split()), n))


def placeholder(out, fid, e, reason=""):
    aspect = e.get("aspect", 0.6)
    fig = plt.figure(figsize=(6.4, 6.4 * aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=True, fc="#f3f3f3",
                               ec="#9e9e9e", lw=1.5, ls="--", transform=ax.transAxes))
    sec = e.get("section_state", "open")
    ax.text(0.04, 0.93, f"PLACEHOLDER  ·  {fid}", fontsize=9, color="#757575",
            family="monospace", va="top", transform=ax.transAxes)
    ax.text(0.96, 0.93, sec, fontsize=9, color="white", va="top", ha="right",
            bbox=dict(fc=STATE_COLOUR.get(sec, "#616161"), ec="none", pad=3),
            transform=ax.transAxes)
    ax.text(0.5, 0.66, wrap(e["title"], 48), fontsize=14, weight="bold",
            ha="center", va="center", transform=ax.transAxes)
    body = wrap(e.get("note", ""), 78)
    if reason:
        body = wrap(reason, 78) + ("\n\n" + body if body else "")
    ax.text(0.5, 0.40, body, fontsize=8, ha="center", va="center",
            color="#424242", transform=ax.transAxes)
    if e.get("producer"):
        ax.text(0.04, 0.05, "producer: " + wrap(e["producer"], 90), fontsize=6.5,
                family="monospace", color="#616161", va="bottom", transform=ax.transAxes)
    fig.savefig(out)
    plt.close(fig)


def raster(src):
    """An image array of `src` (png/jpg directly, pdf via pdftoppm page 1)."""
    if src.suffix.lower() != ".pdf":
        return plt.imread(src)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1",
                        "-singlefile", str(src), f"{tmp}/p"], check=True)
        return plt.imread(f"{tmp}/p.png")


def superseded(out, fid, e, src):
    img = raster(src)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(6.4, 6.4 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(img, alpha=0.35)
    ax.text(0.5, 0.5, "SUPERSEDED — DO NOT QUOTE", fontsize=20, weight="bold",
            color="#c62828", alpha=0.85, rotation=20, ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.08, wrap(e.get("note", ""), 90), fontsize=7, color="#b71c1c",
            ha="center", va="bottom", transform=ax.transAxes,
            bbox=dict(fc="white", ec="#c62828", alpha=0.9))
    fig.savefig(out)
    plt.close(fig)


def build(paper, only=None, run=False, listing=False):
    figdir = HERE / paper / "figures"
    figdir.mkdir(exist_ok=True)
    rows = []
    for fid, e in load(paper).items():
        if only and fid not in only:
            continue
        state = e.get("state", "placeholder")
        src = REPO / e["source"] if e.get("source") else None
        if run and e.get("producer") and not listing:
            print(f"[{fid}] running: {e['producer']}")
            r = subprocess.run(e["producer"], shell=True, cwd=REPO)
            if r.returncode:
                print(f"[{fid}] producer failed (rc={r.returncode}); keeping fallback")
        have = src is not None and src.exists()
        if state in ("current", "superseded") and not have:
            made = "placeholder (source missing)"
        else:
            made = state
        rows.append((fid, made, e.get("source", "")))
        if listing:
            continue
        clear(figdir, fid)
        if made == "current":
            shutil.copy2(src, figdir / f"{fid}{src.suffix.lower()}")
        elif made == "superseded":
            superseded(figdir / f"{fid}.pdf", fid, e, src)
        else:
            reason = f"source not on this machine: {e['source']}" if made.endswith("missing)") else ""
            placeholder(figdir / f"{fid}.pdf", fid, e, reason)
    w = max(len(r[0]) for r in rows) if rows else 0
    print(f"\n{paper}: {len(rows)} figures")
    for fid, made, s in rows:
        print(f"  {fid:<{w}}  {made:<28}  {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paper", choices=PAPERS + ["all"])
    ap.add_argument("--only", nargs="+", help="figure ids")
    ap.add_argument("--run", action="store_true", help="run each entry's producer first")
    ap.add_argument("--list", action="store_true", help="print the status table only")
    a = ap.parse_args()
    for p in PAPERS if a.paper == "all" else [a.paper]:
        build(p, a.only, a.run, a.list)


if __name__ == "__main__":
    sys.exit(main())
