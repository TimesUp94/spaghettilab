"""Read memory snapshots written by the logger mod (F6/F7) and search them.

A dump is a sequence of frames: int32 game_frame, int32 n_objects, then per
object int64 address, int32 size and the bytes.
"""
import re
import struct
import sys


def read(path):
    frames = []
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data):
        frame, n = struct.unpack_from("<ii", data, i); i += 8
        objs = []
        for _ in range(n):
            addr, size = struct.unpack_from("<qi", data, i); i += 12
            objs.append((addr, data[i:i + size])); i += size
        frames.append((frame, objs))
    return frames


if __name__ == "__main__":
    frames = read(sys.argv[1])
    pat = re.compile(rb"(Cmn|Nml)[A-Za-z0-9_]{3,30}")
    om = frames[0][1][0][0]
    for k, (addr, b) in enumerate(frames[0][1]):
        hits = [(m.start(), m.group().decode()) for m in pat.finditer(b)]
        if hits:
            print(f"obj{k} addr={addr:X} size={len(b):X} (om{addr - om:+X}) hits={len(hits)}", hits[:12])
