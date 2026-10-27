#!/usr/bin/env python3
"""wft.diag: the reconstruction reads nothing from the environment except through wft.diag, which records every
override; a retired switch set to the algorithm's own value is a no-op; a production stage refuses overrides.

    ../../.venv/bin/python -m pytest -q wft/tests/test_diag.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE = ('import json, wft.model, wft.seed, wft.io, wft.reco, wft.diag as d; '
         'from wft import model as m, seed as s, reco as r; '
         'print(json.dumps(dict(active={k: str(v) for k, v in d.ACTIVE.items()}, sat=m.SAT_SOLVE, guard=m.COL_GUARD, '
         'tt=r.TWO_TRACK, pivot=r.PIVOT_START, headon=s.HEADON_SIG, drop=r.DROP_FLAT_HITS, tan=r.TAN_MAX)))')


def probe(**env):
    e = {k: v for k, v in os.environ.items() if not k.startswith('WFT_')}
    e.update(env)
    out = subprocess.run([sys.executable, '-c', PROBE], cwd=REPO, env=e, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_are_the_algorithm():
    p = probe()
    assert p['active'] == {}
    assert p['sat'] is True and p['guard'] == 0.01 and p['tt'] is True and p['pivot'] is True
    assert p['headon'] == 20.0 and p['drop'] == 'both' and p['tan'] == 0.6


def test_legacy_switch_at_default_is_a_noop():
    p = probe(WFT_COL_GUARD='0.01', WFT_PIVOT_START='1', WFT_DEAD_SEED='1')
    assert p['active'] == {}


def test_legacy_switch_is_a_recorded_override():
    p = probe(WFT_TWO_TRACK_FIT='0', WFT_DROP_FLAT_HITS='0', WFT_TAN_MAX='1.2')
    assert p['tt'] is False and p['drop'] is None and p['tan'] == 1.2
    assert set(p['active']) == {'reco.TWO_TRACK', 'reco.DROP_FLAT_HITS', 'reco.TAN_MAX'}


def test_wft_diag_variable():
    p = probe(WFT_DIAG='model.SAT_SOLVE=0, seed.HEADON_SIG=0')
    assert p['sat'] is False and p['headon'] == 0.0
    assert set(p['active']) == {'model.SAT_SOLVE', 'seed.HEADON_SIG'}


def test_production_refuses_overrides():
    code = ('import wft.reco, wft.diag as d\n'
            'try:\n    d.require_production("x")\nexcept RuntimeError:\n    print("refused")\nelse:\n    print("ran")')
    e = {k: v for k, v in os.environ.items() if not k.startswith('WFT_')}
    for env, want in (({}, 'ran'), ({'WFT_SAT_SOLVE': '0'}, 'refused')):
        out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=dict(e, **env), capture_output=True, text=True,
                             check=True).stdout.strip()
        assert out == want, (env, out)
