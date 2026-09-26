#!/usr/bin/env python3
"""CLI-level checks for the temperature-sampling migration (plan T3/T4).

No GPU, no weights and no model directory: every check here is about the options and the
*early* validation, so the runner points ``--model-dir`` at a path that does not exist and
requires the failure to name the configuration rather than the model. That ordering is the
plan's "invalid configuration must fail before any model allocation", and a non-existent
model directory is what makes the ordering observable.

Usage:
    python3 tests/test_sampling_cli.py --exe "$(cabal list-bin haskell-infer-demo)"

The plan's real-model smoke commands (greedy regression plus a fixed-seed temperature run)
are deliberately *not* here: they need the checkpoint and a GPU, so they live in the
worklog's reproduction section instead of in a gate that would have to skip itself.
"""

import argparse
import subprocess
import sys

FAILURES = []
CHECKS = 0


def check(condition, what, detail=""):
    global CHECKS
    CHECKS += 1
    if condition:
        return True
    FAILURES.append(f"{what}{(': ' + detail) if detail else ''}")
    print(f"FAIL {what}" + (f" — {detail}" if detail else ""))
    return False


def run(exe, args):
    proc = subprocess.run([exe, "generate", *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def expect_early_refusal(exe, args, needle, what):
    """A configuration error must be reported and must not mention the model directory."""
    code, out, err = run(exe, args)
    text = out + err
    check(code != 0, what + ": the run exited 0", text.strip()[:200])
    check(needle in text, what + f": the error does not mention {needle!r}", text.strip()[:200])
    check("tokenizer" not in text.lower() and "safetensors" not in text.lower(),
          what + ": the failure came from model loading, not from the configuration",
          text.strip()[:200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, help="path to the haskell-infer-demo binary")
    args = parser.parse_args()

    # A model directory that cannot exist, so any failure past validation would name it.
    missing = "/nonexistent/sampling-cli-check"

    # 1. The options exist and are documented.
    help_proc = subprocess.run([args.exe, "generate", "--help"], capture_output=True, text=True)
    help_text = help_proc.stdout
    check(help_proc.returncode == 0, "--help exits 0", help_text[:200])
    check("--temperature" in help_text, "help documents --temperature")
    check("--seed" in help_text, "help documents --seed")

    # 2. Every invalid configuration is refused before anything is loaded.
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--temperature", "-1"],
                         "must not be negative", "a negative temperature")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--temperature", "-1e-400"],
                         "must not be negative", "a negative underflowing temperature")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--temperature", "1e-400"],
                         "underflows to zero", "a temperature that underflows to zero")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--temperature", "1e400"],
                         "must be finite", "a temperature that overflows to infinity")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--temperature", "warm"],
                         "expects a decimal number", "a non-numeric temperature")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--seed", "18446744073709551616"],
                         "out of range", "a seed above 2^64-1")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--seed", "-1"],
                         "unsigned decimal", "a negative seed")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--seed", "0x10"],
                         "unsigned decimal", "a hexadecimal seed")
    expect_early_refusal(args.exe, ["--model-dir", missing, "-p", "x", "--seed", ""],
                         "unsigned decimal", "an empty seed")

    # 3. A valid configuration with no model still fails, but at the model — which is the
    #    control for check 2: the validation does not reject *everything*.
    code, out, err = run(args.exe, ["--model-dir", missing, "-p", "x",
                                    "--temperature", "0", "--seed", "42"])
    text = out + err
    check(code != 0, "a valid configuration with no model exits non-zero")
    check("sampling:" in text, "the resolved sampling metadata is reported on stderr", text[:200])

    if FAILURES:
        print(f"\ntest_sampling_cli: {len(FAILURES)} of {CHECKS} check(s) failed", file=sys.stderr)
        return 1
    print(f"test_sampling_cli: the sampling options validate before any model is loaded "
          f"({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
