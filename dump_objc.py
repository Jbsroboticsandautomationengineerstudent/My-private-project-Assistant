"""Dump Objective-C classes, methods and type encodings from a 64-bit Mach-O."""

import argparse
import struct
from pathlib import Path


class MachO:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        header = struct.unpack_from("<IiiIIIII", self.data)
        if header[0] != 0xFEEDFACF:
            raise ValueError("Expected little-endian 64-bit Mach-O")
        self.sections = {}
        self.segments = []
        off = 32
        for _ in range(header[4]):
            cmd, size = struct.unpack_from("<II", self.data, off)
            if cmd == 0x19:
                segname, vmaddr, vmsize, fileoff, filesize, _, _, nsects, _ = struct.unpack_from("<16sQQQQiiII", self.data, off + 8)
                self.segments.append((vmaddr, vmsize, fileoff, filesize, segname.split(b"\0")[0].decode()))
                sec_off = off + 72
                for _ in range(nsects):
                    sectname, ssegname, addr, ssize, offset = struct.unpack_from("<16s16sQQI", self.data, sec_off)
                    name = sectname.split(b"\0")[0].decode()
                    self.sections[name] = (addr, ssize, offset)
                    sec_off += 80
            off += size
        self.base = min(vmaddr for vmaddr, vmsize, fileoff, filesize, name in self.segments if filesize and name != "__PAGEZERO")

    def fileoff(self, address):
        # Arm64e dyld chained rebase pointers retain metadata in their high bits.
        address &= 0x0000FFFFFFFFFFFF
        if address < self.base:
            address += self.base
        for vmaddr, vmsize, fileoff, filesize, _ in self.segments:
            if vmaddr <= address < vmaddr + filesize:
                return fileoff + address - vmaddr
        raise ValueError(f"Unmapped address 0x{address:x}")

    def u64(self, address):
        return struct.unpack_from("<Q", self.data, self.fileoff(address))[0]

    def cstr(self, address):
        off = self.fileoff(address)
        return self.data[off:self.data.index(b"\0", off)].decode(errors="replace")

    def methods(self, list_address):
        if not list_address:
            return []
        off = self.fileoff(list_address)
        entsize_flags, count = struct.unpack_from("<II", self.data, off)
        small = bool(entsize_flags & 0x80000000)
        direct_selector = bool(entsize_flags & 0x40000000)
        entsize = entsize_flags & 0xFFFF
        if not 0 < entsize <= 32 or count > 10000:
            raise ValueError(f"Bad method list 0x{list_address:x}: 0x{entsize_flags:x}, {count}")
        out = []
        for i in range(count):
            item = list_address + 8 + i * entsize
            if small:
                name_delta, type_delta, imp_delta = struct.unpack_from("<iii", self.data, self.fileoff(item))
                name_addr = item + name_delta
                name = self.cstr(name_addr if direct_selector else self.u64(name_addr))
                types = self.cstr(item + 4 + type_delta)
                imp = item + 8 + imp_delta
            else:
                name_addr, types_addr, imp = struct.unpack_from("<QQQ", self.data, self.fileoff(item))
                name = self.cstr(name_addr)
                types = self.cstr(types_addr)
            out.append((name, types, imp))
        return out

    def class_info(self, address):
        isa, superclass, cache, vtable, data = struct.unpack_from("<QQQQQ", self.data, self.fileoff(address))
        ro = (data & 0x0000FFFFFFFFFFFF) & ~7
        flags, instance_start, instance_size, reserved = struct.unpack_from("<IIII", self.data, self.fileoff(ro))
        ivar_layout, name, methods, protocols, ivars, weak_layout, properties = struct.unpack_from("<QQQQQQQ", self.data, self.fileoff(ro + 16))
        return {"name": self.cstr(name), "isa": isa & 0x0000FFFFFFFFFFFF, "superclass": superclass, "methods": methods & 0x0000FFFFFFFFFFFF, "ro": ro}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--sections", action="store_true")
    args = parser.parse_args()
    m = MachO(args.binary)
    if args.sections:
        for name, (addr, size, off) in m.sections.items():
            print(f"{name:25s} VM=0x{addr:x} size=0x{size:x} file=0x{off:x}")
    addr, size, _ = m.sections["__objc_classlist"]
    for item in range(addr, addr + size, 8):
        class_addr = m.u64(item) & 0x0000FFFFFFFFFFFF
        info = m.class_info(class_addr)
        print(f"@interface {info['name']}  // class VM 0x{class_addr:x}")
        for selector, types, imp in m.methods(info["methods"]):
            print(f"- {selector}\t{types}\tIMP 0x{imp:x}")
        meta = m.class_info(info["isa"])
        for selector, types, imp in m.methods(meta["methods"]):
            print(f"+ {selector}\t{types}\tIMP 0x{imp:x}")
        print("@end\n")


if __name__ == "__main__":
    main()
