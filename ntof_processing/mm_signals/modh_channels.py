"""MODH channel list, tolerant of NUL/space-padded detector names.

ntof_raw.parse_modh drops any record whose 4-byte name fails isalnum(), which
silently discards 3-character names like 'MMA\x00'.  This re-implements the same
walk without that filter.
"""
import struct, sys, collections, pathlib
sys.path.insert(0, str(pathlib.Path.home()/'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks

def modh(payload):
    out = []
    stride = 88
    n = struct.unpack_from('<I', payload, 0)[0]
    for i in range(n):
        b = 4 + i*stride
        if b + stride > len(payload):
            break
        name = payload[b:b+4].decode('ascii', 'replace').strip(' \t\r\n\x00')
        if not name.isalnum():
            continue
        chan, card = struct.unpack_from('<I4s', payload, b+4)
        out.append((name, chan))
    return out

for p in sys.argv[1:]:
    got = None
    for _o, t, _v, pay in iter_banks(p):
        if t == 'MODH':
            got = modh(pay); break
    if got is None:
        print(f'{p}: no MODH'); continue
    c = collections.Counter(n for n, _ in got)
    mm = sorted({n for n, _ in got if n.startswith('MM')})
    print(f'{len(got)} channels  MM={mm if mm else "none"}  {dict(sorted(c.items()))}')
