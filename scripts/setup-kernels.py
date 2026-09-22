#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request
import zipfile


CUDA_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST_SHA = "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630"
WHEELS = {
    "flashinfer_python-0.5.3-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/76/78/6dc7e7da8cb87c9965644ea0d2439457a1bc9256c45ceda0044595be4143/flashinfer_python-0.5.3-py3-none-any.whl",
        "b601293b72f9138bad173edc28df84b9f239a013be974e2e79d4ba98aeb38cf5"),
    "fla_core-0.5.2-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/2d/ed/dfe19c4da779957eb6a42a26812f9b4e2280bf757a17a71933ff59ffcb98/fla_core-0.5.2-py3-none-any.whl",
        "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761"),
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Isolated dependency directory on local SSD")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for name in ("downloads", "cuda", "tmp", "cache", "flashinfer"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(root / "tmp"), XDG_CACHE_HOME=str(root / "cache"),
                      PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", FLA_TILELANG="0")

    def fetch(url, filename, sha):
        target = root / "downloads" / filename
        if target.exists():
            if digest(target) != sha:
                raise RuntimeError(f"Existing download checksum mismatch: {target}")
            return target
        partial = target.with_suffix(target.suffix + ".partial")
        urllib.request.urlretrieve(url, partial)
        if digest(partial) != sha:
            raise RuntimeError(f"Download checksum mismatch: {filename}")
        partial.replace(target)
        return target

    manifest = fetch(CUDA_URL + "redistrib_12.9.1.json", "redistrib_12.9.1.json", MANIFEST_SHA)
    metadata = json.loads(manifest.read_text())
    components = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas", "cuda_nvrtc", "libcurand")

    def component(name):
        info = metadata[name]["linux-x86_64"]
        return fetch(CUDA_URL + info["relative_path"], Path(info["relative_path"]).name, info["sha256"])

    with ThreadPoolExecutor(max_workers=3) as pool:
        archives = list(pool.map(component, components))
    marker = root / "cuda/.redistrib-sha256"
    if not marker.exists() or marker.read_text() != MANIFEST_SHA:
        for archive in archives:
            subprocess.run(["tar", "-xf", str(archive), "--strip-components=1", "--no-same-owner",
                            "-C", str(root / "cuda")], check=True)
        marker.write_text(MANIFEST_SHA)
    if not (root / "cuda/lib64").exists():
        (root / "cuda/lib64").symlink_to("lib", target_is_directory=True)
    for filename, (url, sha) in WHEELS.items():
        fetch(url, filename, sha)
    with zipfile.ZipFile(root / "downloads/flashinfer_python-0.5.3-py3-none-any.whl") as archive:
        for entry in archive.infolist():
            prefix = "flashinfer/data/"
            if not entry.filename.startswith(prefix + "include/") or entry.is_dir():
                continue
            relative = Path(entry.filename[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe dependency archive path")
            output = root / "flashinfer" / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(entry))
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(root / "venv")], check=True)
    python = str(root / "venv/bin/python")
    subprocess.run([python, "-m", "pip", "--isolated", "install", "--no-deps", "--no-index", "--no-cache-dir",
                    "--no-compile", str(root / "downloads/fla_core-0.5.2-py3-none-any.whl")], check=True)
    subprocess.run([python, "-c", "import torch,triton,einops,fla; assert triton.__version__ == '3.4.0'; print('FLA',fla.__version__,'Triton',triton.__version__)"], check=True)
    environment = 'export KERNEL_DEPS=' + shlex.quote(str(root)) + '\n' + '''export CUDA_HOME="$KERNEL_DEPS/cuda"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$KERNEL_DEPS/venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export FLASHINFER_INCLUDE="$KERNEL_DEPS/flashinfer/include"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_CACHE_DIR="$KERNEL_DEPS/cache/triton"
export CUDA_CACHE_PATH="$KERNEL_DEPS/cache/cuda"
export XDG_CACHE_HOME="$KERNEL_DEPS/cache"
export TMPDIR="$KERNEL_DEPS/tmp"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export FLA_TILELANG=0
'''
    (root / "env.sh").write_text(environment)
    subprocess.run([str(root / "cuda/bin/nvcc"), "--version"], check=True)
    print(f"Dependencies ready; source {root / 'env.sh'}")


if __name__ == "__main__":
    main()
