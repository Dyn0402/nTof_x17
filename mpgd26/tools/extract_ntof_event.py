#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_ntof_event.py -- pick the n_TOF setup figure's event out of Geant4.

The build-up figure (``make_ntof.py``) shows one neutron arriving in the He-3
capsule and one X17 e+e- pair leaving it.  Geant4 cannot give both in a single
event -- the neutron run transports a real EAR2 neutron and lets the physics
list decide what it does (overwhelmingly 3He(n,p)t), while the X17 pair is a
*generator* mode that throws the decay products from a vertex sampled in the
gas.

**The e+e- pair is the real event**; this script picks it, on how legible the
event is (below).  What the figure then draws for the neutron is a straight
line up the beam axis to that vertex -- not a transported trajectory -- and it
says so in the caption.

A neutron event is still selected and stored, translated so its interaction
point sits on the pair vertex, for anyone who wants the real history; and the
neutron RUN is needed regardless, because the beam envelope in the figure is
measured from its sampled primaries.  What the drawing code uses from it is
therefore ``beam``, not ``neutron``.

  1. rank neutron events by how much of their history is inside the gas, and
     keep the ones that actually interact there;
  2. rank pair events by how much of the setup they light up (arms crossed,
     layers reached, both legs out through the barrel wall);
  3. pick the combination whose SHIFTED neutron primary still starts inside the
     beam profile -- since the whole history is translated, any neutron can be
     moved, so the constraint is where the moved one starts, not whose vertex
     happened to be closest.

Inputs are the per-step CSVs written by ``mx17_full_sim --trajdump``:

    mx17_full_sim -n 400  -t 1 --trajdump 400  --ipc 0 -o pairs
    mx17_full_sim -n 4000 -t 1 --trajdump 4000 --neutron <flux> <profile> \
                  --emin 1e-3 --emax 1000 -o neutrons

Usage:
    ../../.venv/bin/python tools/extract_ntof_event.py \
        --pairs   /path/pairs_traj_t0.csv \
        --neutrons /path/neutrons_traj_t0.csv \
        -o data/ntof_event.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), 'data', 'ntof_event.json')

# --------------------------------------------------------------------------- #
# Volume -> layer.  The names come from DetectorConstruction.cc; the two traps
# worth knowing are that "PlasticScint" is a bar of the *SiPM trigger wall* and
# that the actual plastic scintillators are the "BackScint" volumes.
# --------------------------------------------------------------------------- #
MM_VOLS = {
    'DriftGas', 'AmpGas', 'WindowGapGas', 'Micromesh', 'ResistivePaste',
    'BulkPillar', 'FieldCagePCB', 'GasWindow_Mylar', 'GasWindow_Al',
    'GasFrame_Al', 'WindowFlange_Al', 'DriftCathode_Cu', 'DriftCathode_Kapton',
    'SupportPlate_Al',
}
# The sensitive volume of each layer -- where a deposit is a *measurement*.
SENSITIVE = {'DriftGas': 'mm', 'PlasticScint': 'sipm', 'BackScintL': 'plastic',
             'BackScintR': 'plastic', 'LiqScint_1': 'ls'}
LAYER_ORDER = ['capsule', 'mm', 'sipm', 'plastic', 'ls']
SENSITIVE_LAYERS = frozenset(LAYER_ORDER[1:])

# The generator fires the e- then the e+, so the two legs are always the first
# two tracks of the event; everything else in it is their shower.
LEG_IDS = (1, 2)

# Where a leg is allowed to leave the vessel.  The capsule is a barrel with a
# dome at each end: 20 mm of bore over |y| < 20 mm, closing to a 7 mm neck at
# the top.  A leg that leaves through a dome crosses several times the wall
# thickness at a glancing angle, so it scatters more and it *looks* wrong --
# the picture wants the case the detector was designed around, which is a pair
# leaving sideways through 0.6 mm of Al and 0.9 mm of CFRP.
CAP_VOLS = ('He3Gas', 'He3Cap_Al', 'He3Cap_CFRP')
SIDE_EXIT_R_MM = 10.4       # the barrel wall; anything less is an end
SIDE_EXIT_Y_MM = 22.0       # ... and outside this the profile has started to taper
SIDE_EXIT_DY_MAX = 0.55     # |dy| of the leg where it leaves: within ~33 deg of
                            # transverse, so it crosses the wall once and does
                            # not then run up the barrel alongside it

# Arms the figure draws solid (--prefer-arms); set by main().
PREFER_ARMS = None
# Birth energy above which a secondary is drawn (--secondary-min-MeV); the
# event ranking penalises events with many of them, because every one is
# another line on the picture.
SECONDARY_MIN_MEV = 0.30


def layer_of(vol: str) -> str:
    if vol.startswith('He3'):
        return 'capsule'
    if vol in MM_VOLS or vol.startswith('PCB_'):
        return 'mm'
    if vol.startswith('PlasticScint'):
        return 'sipm'
    if vol.startswith('BackScint') or vol.startswith('BackTape'):
        return 'plastic'
    if vol.startswith('LiqScint') or vol.startswith('LS_'):
        return 'ls'
    return 'world'


def arm_of(x: float, z: float) -> int:
    """Arm index 0..3 = D(+X), B(-X), A(+Z), C(-Z), by which face a point is on.

    The arms sit on the +-X and +-Z faces of a 41 cm box, so the dominant
    transverse coordinate names the arm unambiguously anywhere outside the
    capsule.
    """
    if abs(x) >= abs(z):
        return 0 if x > 0 else 1
    return 2 if z > 0 else 3


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_traj(path, max_events=None):
    """{eventID: {trackID: track}}, a track being the step arrays in order."""
    events = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            eid = int(r['eventID'])
            if max_events is not None and eid >= max_events:
                continue
            tid = int(r['trackID'])
            t = events[eid].get(tid)
            if t is None:
                t = events[eid][tid] = dict(
                    trackID=tid, parentID=int(r['parentID']),
                    particle=r['particle'], p0=[], p1=[], ke=[], ke1=[],
                    vol=[], proc=[], edep=[])
            t['p0'].append((float(r['pre_x']), float(r['pre_y']),
                            float(r['pre_z'])))
            t['p1'].append((float(r['post_x']), float(r['post_y']),
                            float(r['post_z'])))
            t['ke'].append(float(r['ke_pre_MeV']))
            t['ke1'].append(float(r['ke_post_MeV']))
            t['vol'].append(r['volume'])
            t['proc'].append(r['process'])
            t['edep'].append(float(r['edep_MeV']))
    return events


def polyline(track):
    """Step endpoints as one (n+1, 3) polyline, with KE at each vertex."""
    pts = np.array(track['p0'] + [track['p1'][-1]], float)
    ke = np.array(track['ke'] + [track['ke1'][-1]], float)
    return pts, ke


# --------------------------------------------------------------------------- #
# Selection -- pair events
# --------------------------------------------------------------------------- #
def descendants(tracks, roots):
    """Track ids descending from ``roots`` (Geant4 numbers children after
    parents, so one forward pass over sorted ids closes the tree)."""
    keep = set(roots)
    for tid in sorted(tracks):
        if tracks[tid]['parentID'] in keep:
            keep.add(tid)
    return keep


def leaves_sideways(track):
    """Did this leg leave the vessel through the barrel wall, not an end cap?

    Returns the exit point, or None.  The exit is the far end of the last step
    the track takes inside any capsule volume; a leg that leaves through the
    side is at the barrel radius there, one that leaves through a dome is on
    the axis-ward part of the profile or past the end of the straight section.
    """
    idx = [i for i, v in enumerate(track['vol']) if v in CAP_VOLS]
    if not idx:
        return None
    i = idx[-1]
    x, y, z = track['p1'][i]
    if np.hypot(x, z) < SIDE_EXIT_R_MM or abs(y) > SIDE_EXIT_Y_MM:
        return None
    d = np.array(track['p1'][i], float) - np.array(track['p0'][i], float)
    n = np.linalg.norm(d)
    if n > 0 and abs(d[1]) / n > SIDE_EXIT_DY_MAX:
        return None                              # leaving along the beam axis
    return np.array([x, y, z], float)


def kinematics(tracks):
    """(opening angle [deg], the two birth energies [MeV]) of the two legs.

    The angle is taken between the legs' *initial* directions -- what the decay
    produced, before the vessel wall touches them.
    """
    d, E = [], []
    for t in LEG_IDS:
        tr = tracks[t]
        v = np.array(tr['p1'][0], float) - np.array(tr['p0'][0], float)
        d.append(v / np.linalg.norm(v))
        E.append(float(tr['ke'][0]))
    cos = float(np.clip(np.dot(d[0], d[1]), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos))), E


def pair_quality(tracks):
    """Score an X17 event by how much of the setup its two legs light up.

    What makes a good build-up figure is not a typical event but a *legible*
    one: the two legs in different arms (so the pinwheel reads as four
    independent arms), and at least one leg deep enough to put light in the
    trigger wall, the plastic and the liquid, which is the story the last three
    build stages tell.
    """
    legs = [tracks[t] for t in LEG_IDS if t in tracks
            and tracks[t]['parentID'] == 0]
    if len(legs) != 2:
        return None
    exits = [leaves_sideways(leg) for leg in legs]
    if any(e is None for e in exits):
        return None                              # one or both left through a cap
    fam = descendants(tracks, set(LEG_IDS))

    arms, per_leg, depth = set(), [], defaultdict(int)
    per_leg_layers = [set(layer_of(v) for v in leg['vol']) for leg in legs]
    for leg in legs:
        la = set()
        reach = 0
        for (x, _, z), vol in zip(leg['p0'], leg['vol']):
            lay = layer_of(vol)
            if lay in ('mm', 'sipm', 'plastic', 'ls'):
                la.add(arm_of(x, z))
                reach = max(reach, LAYER_ORDER.index(lay))
        arms |= la
        per_leg.append((la, reach))
    # What "lights a layer up" is the energy it collects, not the number of
    # steps Geant4 happened to take there: a single leg rattling around inside
    # one plastic bar racks up hundreds of steps and still only makes one glow.
    for tid in fam:
        t = tracks[tid]
        for i, vol in enumerate(t['vol']):
            lay = SENSITIVE.get(vol)
            if lay is not None:
                depth[lay] += t['edep'][i]

    if not all(la for la, _ in per_leg):        # both legs must reach a chamber
        return None
    split = len(arms) >= 2                       # legs in different arms
    reach = min(r for _, r in per_leg)           # the *weaker* leg's depth
    # A rendering preference, not a physics one: the figure's camera looks
    # through two of the four arms and draws them as outlines, so a leg that
    # ends in one of those leaves its deposits glowing inside a wireframe.
    drawn = 6.0 if PREFER_ARMS is None or arms <= PREFER_ARMS else 0.0
    # The event the figure most wants: BOTH legs crossing all four layers, so
    # each one can be followed chamber -> trigger wall -> plastic -> liquid and
    # the build-up has something to add at every step.  It is rare -- a leg can
    # pass between the two plastic bars, or out of the side of the stack -- so
    # this is a heavy bonus rather than a cut, which keeps the selector working
    # on a run that happens not to contain one.
    full_chain = all(SENSITIVE_LAYERS <= s for s in per_leg_layers)
    chain = 12.0 if full_chain else 0.0
    # Every drawn secondary is another line on the picture, and the ones that
    # wander (a few-MeV electron spiralling out of a scintillator) cost more
    # legibility than they buy.
    nsec = sum(1 for tid in fam if tid not in LEG_IDS
               and tracks[tid]['ke'][0] >= SECONDARY_MIN_MEV)
    score = (10 * split + 3 * reach + 2 * len(arms) + drawn + chain - 0.4 * nsec
             + min(3, depth['ls'] / 2.0) + min(2, depth['plastic'] / 4.0))
    return dict(score=score, arms=sorted(arms), split=split, reach=reach,
                full_chain=full_chain,
                vertex=np.array(legs[0]['p0'][0], float), family=fam,
                exits=[np.round(e, 2).tolist() for e in exits])


# --------------------------------------------------------------------------- #
# Selection -- neutron events
# --------------------------------------------------------------------------- #
def neutron_quality(tracks):
    """Score a neutron event by the length of its history inside the gas."""
    n = tracks.get(1)
    if n is None or n['particle'] != 'neutron' or n['parentID'] != 0:
        return None
    end = None
    for i, proc in enumerate(n['proc']):
        if proc in ('neutronInelastic', 'nCapture') and n['vol'][i] == 'He3Gas':
            end = i
            break
    if end is None:
        return None                              # escaped, or interacted in Al

    p0 = np.array(n['p0'][:end + 1], float)
    p1 = np.array(n['p1'][:end + 1], float)
    in_gas = np.array([v == 'He3Gas' for v in n['vol'][:end + 1]])
    path = float(np.linalg.norm(p1 - p0, axis=1)[in_gas].sum())
    nscat = int(sum(1 for pr in n['proc'][:end] if pr == 'hadElastic'))
    return dict(score=path / 10.0 + 2.0 * min(nscat, 4), n_steps=end + 1,
                path_mm=path, n_scatter=nscat, end=end,
                proc=n['proc'][end], vertex=np.array(n['p1'][end], float),
                start=np.array(n['p0'][0], float),
                E_eV=float(n['ke'][0]) * 1e6)


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def pack_track(track, shift=None, kind='primary'):
    pts, ke = polyline(track)
    if shift is not None:
        pts = pts + shift
    return dict(particle=track['particle'], trackID=track['trackID'],
                parentID=track['parentID'], kind=kind,
                points=np.round(pts, 3).tolist(),
                ke=np.round(ke, 5).tolist(),
                layers=[layer_of(v) for v in track['vol']])


def deposits(tracks, ids, shift=None, min_MeV=2e-3):
    """One glow marker per (track, sensitive volume) visit.

    Consecutive steps of one track in one sensitive volume are a single
    crossing, so they are merged into one marker at their energy-weighted
    centroid; the marker's weight is the energy that crossing actually left
    behind.  That is what each detector layer sees, at the size it sees it.
    """
    out = []
    for tid in sorted(ids):
        t = tracks[tid]
        cur = None
        for i, vol in enumerate(t['vol']):
            lay = SENSITIVE.get(vol)
            if lay is None:
                if cur:
                    out.append(cur)
                    cur = None
                continue
            mid = (np.array(t['p0'][i], float) + np.array(t['p1'][i])) / 2
            e = t['edep'][i]
            if cur is None or cur['layer'] != lay:
                if cur:
                    out.append(cur)
                cur = dict(layer=lay, edep=0.0, wsum=np.zeros(3), w=0.0,
                           particle=t['particle'],
                           a=np.array(t['p0'][i], float))
            cur['edep'] += e
            cur['wsum'] += mid * max(e, 1e-9)
            cur['w'] += max(e, 1e-9)
            cur['b'] = np.array(t['p1'][i], float)
        if cur:
            out.append(cur)

    packed = []
    for d in out:
        if d['edep'] < min_MeV:
            continue
        p = d['wsum'] / d['w']
        if shift is not None:
            p = p + shift
        a, b = d['a'], d.get('b', d['a'])
        if shift is not None:
            a, b = a + shift, b + shift
        packed.append(dict(layer=d['layer'], particle=d['particle'],
                           edep_MeV=round(d['edep'], 5),
                           arm=arm_of(p[0], p[2]),
                           p=np.round(p, 2).tolist(),
                           # the crossing's extent, so a deposit in the 30 mm
                           # drift gap can be drawn as the ionisation trail it
                           # is rather than as a dot at its centroid
                           a=np.round(a, 2).tolist(),
                           b=np.round(b, 2).tolist()))
    packed.sort(key=lambda d: -d['edep_MeV'])
    return packed


def beam_envelope(events):
    """Transverse spread of the *sampled* neutron start points.

    The generator throws primaries from the measured EAR2 profile, so the start
    points of a few thousand of them are that profile -- which is what the beam
    cylinder in the figure is drawn at.  It is the beam, not the collimator.
    """
    r = []
    for tracks in events.values():
        t = tracks.get(1)
        if t is None or t['particle'] != 'neutron':
            continue
        x, _, z = t['p0'][0]
        r.append(np.hypot(x, z))
    r = np.asarray(r)
    return dict(n=int(r.size), rms_mm=float(np.sqrt((r ** 2).mean())),
                r90_mm=float(np.percentile(r, 90)))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', required=True, help='X17 mode _traj CSV')
    ap.add_argument('--neutrons', required=True, help='neutron mode _traj CSV')
    ap.add_argument('-o', '--out', default=DEFAULT_OUT)
    ap.add_argument('--n-best', type=int, default=1,
                    help='how many top-ranked pair events to consider when '
                         'matching vertices.  The default picks the pair on '
                         'legibility alone and then the closest neutron to it; '
                         'raising it lets a slightly worse pair win on a '
                         'smaller vertex residual, which with thousands of '
                         'neutron candidates in a 20 mm bore is never the '
                         'binding constraint')
    ap.add_argument('--secondary-min-MeV', type=float, default=0.30,
                    help='drop showers below this birth energy (they are '
                         'sub-millimetre and only add visual noise)')
    ap.add_argument('--min-gas-path', type=float, default=25.0,
                    help='mm the neutron must travel inside the He-3 before it '
                         'interacts.  A hard cut, not a preference: the splice '
                         'is chosen on where the shifted primary starts, and '
                         'that objective on its own prefers a thermal neutron '
                         'that stopped as soon as it entered')
    ap.add_argument('--prefer-arms', default='0,3',
                    help='arm indices the figure draws SOLID (0=D +X, 1=B -X, '
                         '2=A +Z, 3=C -Z); events that stay inside them score '
                         'higher, so no deposit ends up inside a ghosted arm. '
                         'Must agree with scenes_ntof.NEAR_ARMS.  Empty to '
                         'disable.')
    args = ap.parse_args()

    global PREFER_ARMS, SECONDARY_MIN_MEV
    PREFER_ARMS = (set(int(a) for a in args.prefer_arms.split(','))
                   if args.prefer_arms.strip() else None)
    SECONDARY_MIN_MEV = args.secondary_min_MeV

    print(f'reading {args.pairs}')
    pair_ev = read_traj(args.pairs)
    print(f'  {len(pair_ev)} pair events')
    print(f'reading {args.neutrons}')
    neut_ev = read_traj(args.neutrons)
    print(f'  {len(neut_ev)} neutron events')

    beam = beam_envelope(neut_ev)
    print(f'  beam envelope: rms {beam["rms_mm"]:.1f} mm, '
          f'90% inside {beam["r90_mm"]:.1f} mm ({beam["n"]} primaries)')

    pq = {e: q for e, q in ((e, pair_quality(t)) for e, t in pair_ev.items())
          if q}
    nq = {e: q for e, q in ((e, neutron_quality(t)) for e, t in neut_ev.items())
          if q}
    print(f'  usable: {len(pq)} pair, {len(nq)} neutron')
    if not pq or not nq:
        raise SystemExit('no usable events -- run more, or relax the cuts')

    best_pairs = sorted(pq, key=lambda e: -pq[e]['score'])[:args.n_best]
    # Neutrons are then chosen on the splice (below), so the quality bar has to
    # be a HARD cut rather than a ranking: without it the splice happily picks
    # a thermal neutron that interacted 0.1 mm inside the gas, which lands the
    # start point beautifully and shows nothing.
    best_neuts = [e for e in nq if nq[e]['path_mm'] >= args.min_gas_path]
    print(f'  {len(best_neuts)} neutrons with >= {args.min_gas_path:.0f} mm '
          f'of gas path')
    if not best_neuts:
        raise SystemExit('no neutron travels far enough in the gas')

    # Which neutron to splice onto the pair.  The splice translates the whole
    # neutron history so its interaction point lands on the pair vertex, so the
    # question is not "whose vertex was already closest" -- any neutron can be
    # moved -- but "does the moved primary still start somewhere the beam
    # actually delivers neutrons".  So the combination is chosen on the
    # TRANSVERSE RADIUS OF THE SHIFTED START POINT, which is the thing that has
    # to stay inside the measured EAR2 profile.
    def start_r(p, n):
        s = nq[n]['start'] + (pq[p]['vertex'] - nq[n]['vertex'])
        return float(np.hypot(s[0], s[2]))

    pe, ne, r_start = min(((p, n, start_r(p, n))
                           for p in best_pairs for n in best_neuts),
                          key=lambda c: c[2])
    shift = pq[pe]['vertex'] - nq[ne]['vertex']
    resid = float(np.linalg.norm(shift))
    ang, Es = kinematics(pair_ev[pe])
    print(f'\npair event {pe}: score {pq[pe]["score"]:.1f}, arms '
          f'{pq[pe]["arms"]}, weakest leg reaches '
          f'{LAYER_ORDER[pq[pe]["reach"]]}')
    print(f'  {Es[0]:.2f} + {Es[1]:.2f} MeV at {ang:.1f} deg, both legs out '
          f'through the barrel at {pq[pe]["exits"]}')
    print(f'  both legs cross all four layers: {pq[pe]["full_chain"]}')
    print(f'neutron event {ne}: E = {nq[ne]["E_eV"]:.3g} eV, '
          f'{nq[ne]["path_mm"]:.1f} mm in gas, {nq[ne]["n_scatter"]} elastic '
          f'scatters, ends on {nq[ne]["proc"]}')
    print(f'neutron history translated by {shift.round(2).tolist()} mm '
          f'(|shift| = {resid:.2f}); the shifted primary starts at '
          f'r = {r_start:.2f} mm, vs 90 % of the beam inside '
          f'{beam["r90_mm"]:.1f} mm')

    ptracks = pair_ev[pe]
    fam = pq[pe]['family']
    legs = [pack_track(ptracks[t], kind='primary') for t in LEG_IDS]
    shower = []
    for tid in sorted(fam - set(LEG_IDS)):
        t = ptracks[tid]
        if t['ke'][0] < args.secondary_min_MeV:
            continue
        shower.append(pack_track(t, kind='secondary'))

    ntracks = neut_ev[ne]
    nq_e = nq[ne]
    n = ntracks[1]
    npts = np.array(n['p0'][:nq_e['end'] + 1] + [n['p1'][nq_e['end']]], float)
    nke = np.array(n['ke'][:nq_e['end'] + 1] + [n['ke1'][nq_e['end']]], float)
    products = []
    for tid in sorted(descendants(ntracks, {1}) - {1}):
        t = ntracks[tid]
        if t['particle'] in ('proton', 'triton', 'alpha', 'deuteron'):
            products.append(pack_track(t, shift=shift, kind='product'))

    doc = dict(
        provenance=dict(
            generator='MX17_Full_Geant/build/mx17_full_sim --trajdump',
            pairs_csv=os.path.abspath(args.pairs),
            neutrons_csv=os.path.abspath(args.neutrons),
            pair_event=pe, neutron_event=ne,
            composite=('the X17 pair is one real Geant4 event.  A neutron '
                       'event is selected and stored here too, translated so '
                       'its interaction point coincides with the pair vertex, '
                       'but scenes_ntof does NOT draw that history: it draws '
                       'the arriving neutron as a straight line up the beam '
                       'axis to the vertex.  The transported history belonged '
                       'to a different event, so what it added to the picture '
                       'was its own in-gas scattering -- a fact about that '
                       'neutron, not about this figure.  The neutron run is '
                       'still needed for the beam envelope, which is measured '
                       'from its sampled primaries.'),
            why_composite=('no single simulated event can contain both. The '
                           'radiative branch that forms the 4He* is ~1e-8 of '
                           '3He(n,p)t, so a neutron run never produces one: '
                           'the pair is thrown by the generator from a vertex '
                           'sampled in the gas, which is also why the capture '
                           'products of the transported neutron (the (n,p)t '
                           'proton and triton, kept in "products") belong to '
                           'the competing channel and are not drawn.'),
            vertex_residual_mm=round(resid, 3),
            neutron_shift_mm=np.round(shift, 3).tolist(),
            shifted_start_r_mm=round(r_start, 3),
            units='mm, world frame, beam along +Y, origin at the target centre',
        ),
        beam=beam,
        neutron=dict(
            E_eV=nq_e['E_eV'], process=nq_e['proc'],
            path_in_gas_mm=round(nq_e['path_mm'], 2),
            n_elastic=nq_e['n_scatter'],
            vertex=np.round(nq_e['vertex'] + shift, 3).tolist(),
            points=np.round(npts + shift, 3).tolist(),
            ke=nke.tolist(),
        ),
        products=products,
        pair=dict(
            vertex=np.round(pq[pe]['vertex'], 3).tolist(),
            arms=pq[pe]['arms'],
            exits=pq[pe]['exits'],
            E_MeV=[round(float(ptracks[t]['ke'][0]), 3) for t in LEG_IDS],
            opening_deg=round(kinematics(ptracks)[0], 1),
            legs=legs, shower=shower,
            deposits=deposits(ptracks, fam),
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(doc, f, indent=1)
    size = os.path.getsize(args.out) / 1e6
    nd = len(doc['pair']['deposits'])
    print(f'\nwrote {args.out}  ({size:.1f} MB, {len(legs)} legs, '
          f'{len(shower)} shower tracks, {len(products)} capture products, '
          f'{nd} deposits)')
    for lay in LAYER_ORDER[1:]:
        e = sum(d['edep_MeV'] for d in doc['pair']['deposits']
                if d['layer'] == lay)
        k = sum(1 for d in doc['pair']['deposits'] if d['layer'] == lay)
        print(f'   {lay:8s} {k:3d} crossings, {e:8.3f} MeV deposited')


if __name__ == '__main__':
    main()
