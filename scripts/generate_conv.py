import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    code = (source / "causal_conv1d_update.cu").read_text()
    code = "\n".join(line for line in code.splitlines()
                     if not line.startswith("#include")
                     and not line.startswith("template void causal_conv1d_update_cuda"))
    code = code.replace("C10_CUDA_KERNEL_LAUNCH_CHECK();", "check_conv_launch();")
    output.mkdir(parents=True, exist_ok=True)
    (output / "causal_conv1d.h").write_text((source / "causal_conv1d.h").read_text())
    (output / "causal_conv1d_update.cuh").write_text(code + "\n")


if __name__ == "__main__":
    main()
