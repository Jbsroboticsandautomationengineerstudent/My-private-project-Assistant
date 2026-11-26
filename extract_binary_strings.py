"""Extract printable strings and relevant hits from a Mach-O binary."""

import argparse
import re
from pathlib import Path


KEYWORDS = re.compile(
    rb"stockfish|maia|onnx|engine|best.?move|evaluat|board|arrow|overlay|uiwindow|https?://|fen|uci|coreml|chessassistant|chess\.dylib",
    re.IGNORECASE,
)
ASCII = re.compile(rb"[\x20-\x7e]{4,}")
UTF16 = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.binary.read_bytes()
    all_lines = []
    hits = []
    for pattern, encoding in ((ASCII, "ascii"), (UTF16, "utf-16le")):
        for match in pattern.finditer(data):
            value = match.group().decode(encoding, errors="replace")
            line = f"0x{match.start():08x}\t{value}"
            all_lines.append(line)
            if KEYWORDS.search(match.group() if encoding == "ascii" else value.encode()):
                hits.append(line)
    args.output.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    args.output.with_name(args.output.stem + "_hits.txt").write_text(
        "\n".join(hits) + "\n", encoding="utf-8"
    )
    print(f"{args.binary}: {len(all_lines)} strings; {len(hits)} keyword hits")


if __name__ == "__main__":
    main()
