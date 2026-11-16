#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
SOURCE_DIR=.build-sources/stockfish19
CACHE_DIR=.asset-cache/stockfish19
DEST_DIR=layout/usr/bin
NET_NAME=nn-1a298aa575a0.nnue
NET_URL="https://tests.stockfishchess.org/api/nn/$NET_NAME"
mkdir -p "$CACHE_DIR" "$DEST_DIR" .build-sources

verify_net() {
  [[ -f "$1" ]] && [[ "$(shasum -a 256 "$1" | cut -c 1-12)" == 1a298aa575a0 ]]
}

if ! verify_net "$CACHE_DIR/$NET_NAME"; then
  curl --fail --location --retry 3 --max-time 300 --output "$CACHE_DIR/$NET_NAME.tmp" "$NET_URL"
  verify_net "$CACHE_DIR/$NET_NAME.tmp" || { echo "Stockfish 19 NNUE checksum mismatch" >&2; exit 1; }
  mv "$CACHE_DIR/$NET_NAME.tmp" "$CACHE_DIR/$NET_NAME"
fi

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/official-stockfish/Stockfish.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" fetch --depth 1 origin tag sf_19
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
[[ "$(git -C "$SOURCE_DIR" rev-parse --short=7 HEAD)" == edb0d9d ]] || {
  echo "Stockfish 19 tag no longer points to the audited release" >&2
  exit 1
}
grep -q "EvalFileDefaultName \"$NET_NAME\"" "$SOURCE_DIR/src/evaluate.h"
cp "$CACHE_DIR/$NET_NAME" "$SOURCE_DIR/src/$NET_NAME"

SDK=$(xcrun --sdk iphoneos --show-sdk-path)
CXX=$(xcrun --sdk iphoneos --find clang++)
TARGET_FLAGS="-target arm64-apple-ios17.0 -isysroot $SDK"

# Build a single iOS slice. The upstream net is embedded into the executable,
# so there is no runtime EvalFile path or network download on the phone.
make -C "$SOURCE_DIR/src" clean ARCH=armv8 COMP=clang TARGET_KERNEL=iOS
make -C "$SOURCE_DIR/src" -j "$(sysctl -n hw.ncpu)" all \
  ARCH=armv8 COMP=clang TARGET_KERNEL=iOS COMPCXX="$CXX" optimize=no \
  EXTRACXXFLAGS="$TARGET_FLAGS -O3 -DNDEBUG -fPIE" \
  LDFLAGS="$TARGET_FLAGS"

cp "$SOURCE_DIR/src/stockfish" "$DEST_DIR/stockfish19"
chmod 755 "$DEST_DIR/stockfish19"
codesign --force --sign - "$DEST_DIR/stockfish19"
file "$DEST_DIR/stockfish19"
xcrun lipo -archs "$DEST_DIR/stockfish19" | grep -qw arm64
otool -l "$DEST_DIR/stockfish19" | grep -A4 LC_BUILD_VERSION | grep -q 'platform 2'
