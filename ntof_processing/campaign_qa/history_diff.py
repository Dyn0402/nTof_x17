#!/usr/bin/env python3
"""Diff the UserInput actually used by two processings, from the files themselves.

`history_<run>.root` stores the whole UserInput as a ROOT string object, so the
product records its own configuration.  A raw md5 comparison always fails
between our products and n_TOF's for reasons that carry no physics -- line 0 is
the staged file name (`UserInput.h` vs `UserInput_2026_EAR2_X17_v4.h`) and every
pulse-shape template is referenced by an absolute path that starts with our AFS
prefix.  This strips both and diffs what is left, so what prints is only the
parameters.

Usage:  python history_diff.py <ours_history.root> <official_history.root>
"""
import difflib
import re
import sys

import uproot


def history_string(path):
    o = uproot.open(path)['history']
    for attr in ('fString', 'fTitle', 'fName'):
        try:
            v = o.member(attr)
            if isinstance(v, (str, bytes)) and len(v) > 200:
                return v.decode() if isinstance(v, bytes) else v
        except Exception:
            pass
    return str(o)


def normalise(text):
    """Drop path prefixes and the header file name; keep parameters."""
    out = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        # any absolute path -> its basename, so /afs/... and /eos/... compare
        line = re.sub(r'(/[\w.\-]+)+/([\w.\-]+)', r'\2', line)
        out.append(line)
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_raw, b_raw = history_string(a_path), history_string(b_path)
    a, b = normalise(a_raw), normalise(b_raw)
    print(f'A = {a_path}   ({len(a_raw)} chars, {len(a)} non-empty lines)')
    print(f'B = {b_path}   ({len(b_raw)} chars, {len(b)} non-empty lines)')
    print()
    diff = list(difflib.unified_diff(b, a, fromfile='official', tofile='ours',
                                     lineterm='', n=1))
    if not diff:
        print('IDENTICAL after dropping path prefixes -- same parameters.')
        return 0
    print(f'{sum(1 for d in diff if d.startswith(("+", "-")) and not d.startswith(("+++", "---")))} '
          f'differing lines:\n')
    print('\n'.join(diff))
    return 0


if __name__ == '__main__':
    sys.exit(main())
