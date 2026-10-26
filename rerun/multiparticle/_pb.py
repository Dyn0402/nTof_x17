import json, os
d = json.load(open(os.path.expanduser('~/x17/cosmic_bench/satstudy/multiparticle/bench.json')))
for a, v in d['arms'].items():
    if not v:
        print(a, None); continue
    print(a, 'single ok', v['single_ok'], 'split', v['single_split'])
    for b in v['bins']:
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in b.items()})
    print('ge24', {k: (round(x, 3) if isinstance(x, float) else x) for k, x in (v.get('ge24') or {}).items()})
r = d['real']
for b in r['bins']:
    print(b)
print({k: v for k, v in r.items() if k != 'bins'})
