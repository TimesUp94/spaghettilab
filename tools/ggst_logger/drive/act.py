"""Print each player's action-name changes across the frames of one or more dumps."""
import glob
import os
import sys

import dumps

P1, P2, ACT = 0x274D90, 0x322910, 0x38A8


def name(objs, om, obj, off=ACT):
    for a, b in objs:
        if a - om == obj:
            return b[off:off + 32].split(b"\0")[0].decode(errors="replace")
    return None


for p in sorted(glob.glob(sys.argv[1])):
    fr = dumps.read(p)
    om = fr[0][1][0][0]
    out = []
    for f, objs in fr:
        s = (name(objs, om, P1), name(objs, om, P2))
        if not out or out[-1][1] != s:
            out.append((f, s))
    print(os.path.basename(p), out)
