"""Read input/output feature names and MultiArray shapes from Core ML protobuf."""

import sys
from pathlib import Path


def varint(buf, pos):
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def fields(buf):
    pos = 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = buf[pos:pos + size]
            pos += size
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"Unsupported wire type {wire}")
        yield number, wire, value


data = Path(sys.argv[1]).read_bytes()
description = next(v for n, w, v in fields(data) if n == 2 and w == 2)
for n, w, feature in fields(description):
    if n not in (1, 10) or w != 2:
        continue
    parts = list(fields(feature))
    name = next(v.decode() for fn, fw, v in parts if fn == 1 and fw == 2)
    kind = next(v for fn, fw, v in parts if fn == 3 and fw == 2)
    array = next(v for fn, fw, v in fields(kind) if fn == 5 and fw == 2)
    shape = []
    dtype = None
    for fn, fw, v in fields(array):
        if fn == 1:
            if fw == 0:
                shape.append(v)
            elif fw == 2:
                pos = 0
                while pos < len(v):
                    value, pos = varint(v, pos)
                    shape.append(value)
        elif fn == 2 and fw == 0:
            dtype = v
    print(f"{'input' if n == 1 else 'output'}\t{name}\tshape={shape}\tCoreML dtype code={dtype}")
