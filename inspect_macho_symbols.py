"""List selected Mach-O symbols and their VM addresses."""

import argparse
import re
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("pattern")
args = p.parse_args()
data = args.binary.read_bytes()
header = struct.unpack_from("<IiiIIIII", data)
if header[0] != 0xFEEDFACF:
    raise ValueError("Expected ARM64 Mach-O")
off = 32
symoff = nsyms = stroff = strsize = None
for _ in range(header[4]):
    cmd, size = struct.unpack_from("<II", data, off)
    if cmd == 2:
        symoff, nsyms, stroff, strsize = struct.unpack_from("<IIII", data, off + 8)
    off += size
if symoff is None:
    raise ValueError("No symbol table")
pattern = re.compile(args.pattern, re.I)
for i in range(nsyms):
    strx, typ, sect, desc, value = struct.unpack_from("<IBBHQ", data, symoff + i * 16)
    end = data.index(b"\0", stroff + strx)
    name = data[stroff + strx:end].decode(errors="replace")
    if pattern.search(name):
        print(f"0x{value:010x}\t{typ:02x}\t{name}")
