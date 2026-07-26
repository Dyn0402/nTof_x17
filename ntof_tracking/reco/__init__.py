#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_tracking.reco — beam-data track reconstruction foundation (July 2026).

Pipeline (per event, per Micromegas plane):
    io.load_subrun_hits  ->  noise.flag_noise  ->  segments.find_segments
    ->  pairing.pair_xy_3d  ->  geometry.segment_to_global  ->  display.*

search.sift_events runs the front half of that chain over every event of a
subrun and emits a ranked track-candidate table — the event filter that
replaces "number of hits" selections.

First target dataset: run_48 scint-doubles triggers (32 samples x 60 ns,
Ar/iso 95/5, drift 800 V). Driver: ntof_july_analysis/run48_tracking.py.
"""
from . import io, noise, segments, pairing, geometry, display, search  # noqa: F401
