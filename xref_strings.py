"""Find ARM64 ADRP/ADD references to literal strings in selected functions."""

import argparse
import struct
from pathlib import Path

from dump_objc import MachO

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("ranges", nargs="+", help="start:end VM address in hex")
a = p.parse_args()
m = MachO(a.binary)
for span in a.ranges:
    start, end = [int(x, 16) for x in span.split(":")]
    print(f"\nFunction range 0x{start:x}:0x{end:x}")
    pages = {}
    seen = set()
    for pc in range(start, end, 4):
        word = struct.unpack_from("<I", m.data, m.fileoff(pc))[0]
        if word & 0x9F000000 == 0x90000000:
            imm = ((word >> 5) & 0x7FFFF) << 2 | ((word >> 29) & 3)
            if imm & (1 << 20):
                imm -= 1 << 21
            pages[word & 31] = ((pc & ~0xFFF) + (imm << 12), pc)
        elif word & 0xFF000000 == 0x91000000 or word & 0xFFC00000 == 0xF9400000:
            rd, rn = word & 31, (word >> 5) & 31
            if rn in pages and pc - pages[rn][1] <= 64:
                is_load = word & 0xFFC00000 == 0xF9400000
                shift = (word >> 22) & 3
                if is_load or shift <= 1:
                    target = pages[rn][0] + (((word >> 10) & 0xFFF) << (3 if is_load else 12 if shift else 0))
                    for section in ("__cstring", "__objc_methname", "__objc_classname", "__cfstring", "__objc_selrefs"):
                        addr, size, _ = m.sections[section]
                        if addr <= target < addr + size and target not in seen:
                            seen.add(target)
                            try:
                                if section == "__cfstring":
                                    value = m.cstr(m.u64(target + 16))
                                elif section == "__objc_selrefs":
                                    value = m.cstr(m.u64(target))
                                else:
                                    value = m.cstr(target)
                            except ValueError:
                                continue
                            print(f"0x{pc:x} -> {section} 0x{target:x}: {value}")
                    if not is_load:
                        pages[rd] = (target, pc)
