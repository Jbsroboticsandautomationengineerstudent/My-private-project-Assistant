"""Find direct ARM64 BL call sites for selected Mach-O functions."""

import argparse
import bisect
import struct
from pathlib import Path

from dump_objc import MachO


def uleb(data, off):
    value = shift = 0
    while True:
        byte = data[off]
        off += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, off
        shift += 7


p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("targets", nargs="+")
a = p.parse_args()
m = MachO(a.binary)
data = m.data
h = struct.unpack_from("<IiiIIIII", data)
off = 32
func_off = func_size = symoff = nsyms = stroff = indirect_off = None
stub_info = None
for _ in range(h[4]):
    cmd, size = struct.unpack_from("<II", data, off)
    if cmd == 0x26:
        func_off, func_size = struct.unpack_from("<II", data, off + 8)
    elif cmd == 2:
        symoff, nsyms, stroff, _ = struct.unpack_from("<IIII", data, off + 8)
    elif cmd == 0xB:
        indirect_off = struct.unpack_from("<I", data, off + 56)[0]
    elif cmd == 0x19:
        nsects = struct.unpack_from("<I", data, off + 64)[0]
        for j in range(nsects):
            sec = off + 72 + j * 80
            name = data[sec:sec + 16].split(b"\0")[0]
            if name == b"__stubs":
                stub_addr = struct.unpack_from("<Q", data, sec + 32)[0]
                reserved1, reserved2 = struct.unpack_from("<II", data, sec + 68)
                stub_info = (stub_addr, reserved1, reserved2)
    off += size
symbols = {}
symbol_names = []
for i in range(nsyms):
    strx, typ, sect, desc, value = struct.unpack_from("<IBBHQ", data, symoff + i * 16)
    name = data[stroff + strx:data.index(b"\0", stroff + strx)].decode(errors="replace")
    symbol_names.append(name)
    if value:
        symbols[value] = name
if stub_info and indirect_off is not None:
    stub_addr, reserved1, stub_size = stub_info
    stubs_size = m.sections["__stubs"][1]
    for j in range(stubs_size // stub_size):
        index = struct.unpack_from("<I", data, indirect_off + (reserved1 + j) * 4)[0]
        if index < len(symbol_names):
            symbols[stub_addr + j * stub_size] = symbol_names[index]
starts = []
if func_off is not None:
    pos = 0
    cursor = func_off
    while cursor < func_off + func_size:
        delta, cursor = uleb(data, cursor)
        if delta == 0:
            break
        pos += delta
        starts.append(pos)
labels = {}
addr, size, _ = m.sections["__objc_classlist"]
for item in range(addr, addr + size, 8):
    cls = m.class_info(m.u64(item))
    for sel, typ, imp in m.methods(cls["methods"]):
        labels[imp] = f"-[{cls['name']} {sel}]"
    meta = m.class_info(cls["isa"])
    for sel, typ, imp in m.methods(meta["methods"]):
        labels[imp] = f"+[{cls['name']} {sel}]"
targets = {addr: name for addr, name in symbols.items() if any(t.lower() in name.lower() for t in a.targets)}
print("Targets:", {hex(k): v for k, v in targets.items()})
text_addr, text_size, text_off = m.sections["__text"]
for pc in range(text_addr, text_addr + text_size, 4):
    word = struct.unpack_from("<I", data, text_off + pc - text_addr)[0]
    if word & 0xFC000000 != 0x94000000:
        continue
    imm = word & 0x03FFFFFF
    if imm & 0x02000000:
        imm -= 0x04000000
    dest = pc + imm * 4
    if dest not in targets:
        continue
    idx = bisect.bisect_right(starts, pc) - 1
    parent = starts[idx] if idx >= 0 else 0
    print(f"0x{pc:x} in 0x{parent:x} {labels.get(parent, symbols.get(parent, ''))} -> {targets[dest]} (0x{dest:x})")
