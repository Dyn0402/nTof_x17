import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path.home()/'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, ZS_FILL_CODE
POS = 32767
tot = {}
for path in sys.argv[1:]:
    for _o, t, _v, p in iter_banks(path):
        if t != 'ACQC':
            continue
        det, ch, blks = parse_acqc(p, with_samples=True)
        det = str(det).strip(' \t\r\n\x00')
        if not det.startswith('MM'):
            continue
        r = tot.setdefault(det, dict(nsamp=0, pos=0, neg=0, jump=0, maxjump=0, blocks=0))
        for st, s in blks:
            s = s.astype(np.int32)
            r['blocks'] += 1
            r['nsamp'] += len(s)
            r['pos'] += int((s >= POS).sum())
            r['neg'] += int((s <= ZS_FILL_CODE).sum())
            if len(s) > 1:
                d = np.abs(np.diff(s))
                r['jump'] += int((d > 20000).sum())
                r['maxjump'] = max(r['maxjump'], int(d.max()))
for det, r in sorted(tot.items()):
    print(f"{det}: {r['blocks']} blocks, {r['nsamp']} samples | at +rail {r['pos']} | "
          f"at -rail/fill {r['neg']} | sample-to-sample jumps >20000: {r['jump']} "
          f"(largest {r['maxjump']})")
