"""Figure inventory for the end-of-run report.

Every figure the report shows is produced somewhere else in the campaign's
analysis packages.  This module is the single place that says *where*, so the
report can be rebuilt on a machine that has the sources, and so a missing
source is a loud error rather than a silently absent image.

Two of the figures are made here (`ntof_run_report/figures_local.py`) because
they did not exist anywhere else: the beam-availability history and the
campaign timeline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path(os.path.expanduser("~"))
DATA = Path("/media/dylan/data/x17")
DECKS = DATA / "Documents/presentations"
MPGD = REPO / "mpgd26/slides/assets/img"
SITE = HOME / "PycharmProjects/dylan-cern-site"

# name in figures/  ->  source path
SOURCES: dict[str, Path] = {
    # --- photographs (the run, as it stood) -------------------------------
    "photo_topdown.jpg": HOME / "Downloads/PXL_20260722_081056955.jpg",
    "photo_side.jpg": HOME / "Downloads/PXL_20260723_080251836.MP.jpg",
    "photo_daq.jpg": HOME / "Downloads/PXL_20260723_080246594.jpg",
    "photo_beamline.jpg": HOME / "Downloads/PXL_20260810_072331838.jpg",
    # --- the setup, as modelled ------------------------------------------
    "setup_3d.png": MPGD / "setup3d_9_full.png",
    # --- the gamma flash and what it does to the readout ------------------
    "flash_waveform.png": MPGD / "status_flash_waveform.png",
    "two_readouts.png": MPGD / "status_two_readouts.png",
    "recovery_vs_hv.png": MPGD / "status_recovery_vs_hv.png",
    "deadtime_vs_charge.png": MPGD / "status_deadtime_vs_charge.png",
    "charge_compare.png": REPO / "ntof_processing/mm_flash/figures/compare_final.png",
    # --- what we recorded --------------------------------------------------
    "track_rate.png": MPGD / "status_track_rate.png",
    # --- tracking ----------------------------------------------------------
    "wall_segment_tour.png": DECKS / "summary_2026-07-31/figures/wall_segment_tour_all.png",
    "event_display.png": DECKS / "summary_2026-07-31/figures/evt24931_3d.png",
    # --- detector performance ---------------------------------------------
    "mm_maps.png": REPO / "ntof_active_area/figures/mm_maps.png",
    # --- simulation and the yield comparison ------------------------------
    "sim_timedist_bysource.png": DECKS / "summary_2026-07-22/figures/geant/timedist_bysource.png",
    "yield_data_vs_sim.png": DECKS
    / "summary_2026-07-22/figures/trigger/singles_vs_time_datasim_run224524.png",
    # --- SPS ---------------------------------------------------------------
    "det4_sps_efficiency.png": DATA
    / "sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/mapping_check/det4_efficiency_summary.png",
}

# Figures embedded as base64 inside a published note rather than living as
# files.  (index into the note's <img> sequence, output name)
FROM_NOTE: dict[str, tuple[Path, int]] = {
    "run145_pointing.png": (SITE / "notes/run145-target-imaging.html", 0),
    "run145_image.png": (SITE / "notes/run145-target-imaging.html", 1),
}


PHOTO_MAX_PX = 1800  # phone photos are 12 Mpx; the page never shows more than this


def _copy_photo(src: Path, dst: Path) -> None:
    """Downscale a phone photo on the way in.

    The originals are 3-5 MB each and the page renders them at ~400 px wide.
    Shipping them whole would make the published note ~15 MB for no visible
    gain.  Falls back to a plain copy if Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        shutil.copyfile(src, dst)
        return
    im = Image.open(src)
    im.thumbnail((PHOTO_MAX_PX, PHOTO_MAX_PX), Image.LANCZOS)
    im.convert("RGB").save(dst, "JPEG", quality=82, optimize=True)


def stage(outdir: Path) -> list[str]:
    """Copy every source figure into ``outdir``.  Returns the names missing."""
    outdir.mkdir(parents=True, exist_ok=True)
    keep = (set(SOURCES) | set(FROM_NOTE)
            | {"beam_availability.png", "events_collected.png",
               "capsule_pressure.png"}                           # figures_local
            | {"hv_current_scan.png"}                            # figures_flash
            | {"setup_topdown.png"}                              # figures_geometry
            | {"comb_evolution.png"})                            # figures_comb
    for stale in outdir.iterdir():  # a figure dropped from the report
        if stale.name not in keep:  # must not linger and get published
            stale.unlink()
    missing = []
    for name, src in SOURCES.items():
        if not src.exists():
            missing.append(f"{name} <- {src}")
        elif name.endswith(".jpg"):
            _copy_photo(src, outdir / name)
        else:
            shutil.copyfile(src, outdir / name)

    import base64
    import re

    for name, (note, idx) in FROM_NOTE.items():
        if not note.exists():
            missing.append(f"{name} <- {note}")
            continue
        imgs = re.findall(r'src="data:image/png;base64,([^"]+)"', note.read_text())
        if len(imgs) <= idx:
            missing.append(f"{name} <- {note} [img {idx} of {len(imgs)}]")
            continue
        (outdir / name).write_bytes(base64.b64decode(imgs[idx]))
    return missing
