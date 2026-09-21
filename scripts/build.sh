#!/usr/bin/env bash
# build.sh - One-command build for all three language components.
#
# Usage:
#   ./scripts/build.sh          # Build everything
#   ./scripts/build.sh --tests  # Build with CUDA kernel tests
#   ./scripts/build.sh --clean  # Clean all build artifacts
#
# Prerequisites:
#   - GHC 9.6+ and Cabal (via ghcup)
#   - Rust/Cargo (for tokenizer-ffi)
#   - CUDA Toolkit 12.x with nvcc (for csrc/)
#   - CMake 3.18+
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TESTS=OFF
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --tests) BUILD_TESTS=ON ;;
    --clean) CLEAN=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if [ "$CLEAN" -eq 1 ]; then
  echo "=== Cleaning ==="
  rm -rf "$ROOT/csrc/build"
  rm -rf "$ROOT/tokenizer-ffi/target"
  rm -rf "$ROOT/dist-newstyle"
  echo "Clean done."
  exit 0
fi

# ---------------------------------------------------------------------------
# 1. Rust tokenizer FFI
# ---------------------------------------------------------------------------
echo "=== Building tokenizer-ffi (Rust) ==="
cd "$ROOT/tokenizer-ffi"
cargo build --release
echo "  -> tokenizer-ffi/target/release/libtokenizer_ffi.so"

# ---------------------------------------------------------------------------
# 2. C/CUDA engine
# ---------------------------------------------------------------------------
echo "=== Building CUDA engine ==="
mkdir -p "$ROOT/csrc/build"
cd "$ROOT/csrc/build"
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS="$BUILD_TESTS"
cmake --build . -j"$(nproc)"
echo "  -> csrc/build/libengine.a"

if [ "$BUILD_TESTS" = "ON" ]; then
  echo "=== Running CUDA kernel tests ==="
  ctest --output-on-failure
fi

# ---------------------------------------------------------------------------
# 3. Haskell
# ---------------------------------------------------------------------------
echo "=== Building Haskell ==="
cd "$ROOT"

# Ensure ghcup env is loaded
if [ -f "$HOME/.ghcup/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.ghcup/env"
fi

cabal build all
echo "  -> dist-newstyle/.../haskell-infer-demo"

echo ""
echo "=== Build complete ==="
echo "Run with:"
echo "  cabal run haskell-infer-demo -- hello-gpu --device 0"
echo "  cabal run haskell-infer-demo -- generate --model-dir /path/to/model --gpus 0,1 -p 'Hello'"
