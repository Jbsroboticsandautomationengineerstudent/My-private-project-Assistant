"""Print Mach-O dynamic-library load commands (64-bit little-endian)."""

import struct
import sys
from pathlib import Path

KINDS = {0xC: "LOAD_DYLIB", 0xD: "ID_DYLIB", 0x18: "LOAD_WEAK_DYLIB", 0x1F: "REEXPORT_DYLIB", 0x23: "LOAD_UPWARD_DYLIB"}

for name in sys.argv[1:]:
    path = Path(name)
    data = path.read_bytes()
    magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack_from("<IiiIIIII", data)
    if magic != 0xFEEDFACF:
        raise ValueError(f"Unexpected Mach-O magic in {path}")
    print(f"{path}: filetype={filetype}, commands={ncmds}")
    off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        kind = KINDS.get(cmd & 0x7FFFFFFF)
        if kind:
            str_off = struct.unpack_from("<I", data, off + 8)[0]
            raw = data[off + str_off:off + size].split(b"\0", 1)[0]
            print(f"  {kind}: {raw.decode(errors='replace')}")
        off += size
