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
#   - CUDA 12.9+, FlashInfer headers, Triton 3.4.0 and fla-core 0.5.2
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
  rm -rf "$ROOT/csrc/build-libs"
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
mkdir -p "$ROOT/csrc/build-libs"
cd "$ROOT/csrc/build-libs"
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS="$BUILD_TESTS" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-86}" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build . -j"${BUILD_JOBS:-2}"
echo "  -> csrc/build-libs/libengine.so"

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

cabal build exe:haskell-infer-demo --extra-lib-dirs="$CUDA_HOME/lib64" \
  --ghc-options="-optl-Wl,-rpath,$ROOT/csrc/build-libs -optl-Wl,-rpath,$ROOT/tokenizer-ffi/target/release"
echo "  -> dist-newstyle/.../haskell-infer-demo"

echo ""
echo "=== Build complete ==="
echo "Run with:"
echo "  cabal run haskell-infer-demo -- hello-gpu --device 0"
echo "  cabal run haskell-infer-demo -- generate --model-dir /path/to/model --gpus 0,1 -p 'Hello'"
