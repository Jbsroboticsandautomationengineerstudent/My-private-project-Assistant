#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REVISION=bd309e0673a5a851941f0c19d15b01cbaba4baa5
NET_SHA256=b069fc089421bdf2dfcc5d7ea4de2345d7af4795cd996373c07d5f5cf67e0670
NET_NAME=nn-consilium-plebis.nnue
NET_URL="https://github.com/Tors3/Triumviratus/releases/download/v8.1/$NET_NAME"
SOURCE_DIR=.build-sources/triumviratus
CACHE_DIR=.asset-cache/triumviratus
DEST_DIR=layout/usr/bin
mkdir -p "$CACHE_DIR" "$DEST_DIR" .build-sources

if [[ ! -f "$CACHE_DIR/$NET_NAME" ]] || ! printf '%s  %s\n' "$NET_SHA256" "$CACHE_DIR/$NET_NAME" | shasum -a 256 -c - >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "$CACHE_DIR/$NET_NAME.tmp" "$NET_URL"
  printf '%s  %s\n' "$NET_SHA256" "$CACHE_DIR/$NET_NAME.tmp" | shasum -a 256 -c -
  mv "$CACHE_DIR/$NET_NAME.tmp" "$CACHE_DIR/$NET_NAME"
fi

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/Tors3/Triumviratus.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch --depth 1 origin "$REVISION"
git -C "$SOURCE_DIR" checkout --detach "$REVISION"

# Upstream resolves its network with /proc/self/exe on POSIX. iOS needs dyld.
python3 - "$SOURCE_DIR/source/main.cpp" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
marker = '#include <unistd.h>'
patch = r'''
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <mach-o/dyld.h>
#include <string.h>
static ssize_t cap_executable_path(char *buffer, size_t capacity) {
    uint32_t size = static_cast<uint32_t>(capacity);
    return _NSGetExecutablePath(buffer, &size) == 0 ? static_cast<ssize_t>(strlen(buffer)) : -1;
}
#define readlink(path, buffer, capacity) cap_executable_path(buffer, capacity)
#endif
#endif
'''
if 'cap_executable_path' not in source:
    if marker not in source:
        raise SystemExit('Triumviratus main.cpp changed: executable path patch did not apply')
    path.write_text(source.replace(marker, marker + '\n' + patch, 1))
PY

SDK=$(xcrun --sdk iphoneos --show-sdk-path)
CXX=$(xcrun --sdk iphoneos --find clang++)
CC=$(xcrun --sdk iphoneos --find clang)
TARGET_FLAGS="-target arm64-apple-ios17.0 -isysroot $SDK"
CXXFLAGS="-std=c++17 -O3 -DNDEBUG -DNNUE_EMBEDDING_OFF -DTRIUMV_C2 -DTRIUMV5_THREATS_INCR -DUSE_POPCNT -DUSE_NEON=8 $TARGET_FLAGS -pthread -I. -Ifathom"
CFLAGS="-O3 $TARGET_FLAGS -pthread -I. -Ifathom"
make -C "$SOURCE_DIR/source" clean EXE=triumviratus_ios
make -C "$SOURCE_DIR/source" -j "$(sysctl -n hw.ncpu)" EXE=triumviratus_ios \
  CXX="$CXX" CC="$CC" CXXFLAGS="$CXXFLAGS" CFLAGS="$CFLAGS" LIBS=-lm
cp "$SOURCE_DIR/source/triumviratus_ios" "$DEST_DIR/triumviratus"
cp "$CACHE_DIR/$NET_NAME" "$DEST_DIR/$NET_NAME"
chmod 755 "$DEST_DIR/triumviratus"
codesign --force --sign - "$DEST_DIR/triumviratus"
file "$DEST_DIR/triumviratus"
xcrun lipo -archs "$DEST_DIR/triumviratus" | grep -qw arm64
otool -l "$DEST_DIR/triumviratus" | grep -A4 LC_BUILD_VERSION | grep -q 'platform 2'
