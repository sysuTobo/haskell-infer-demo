#!/usr/bin/env python3
"""S0's gate: the speculative CLI's early refusals, and (with a model) its output.

The plan's S0 is an explicitly greedy experiment that owns two runtimes, so the combinations it
refuses are part of its contract: a window without a draft, a window above temperature 0, a
window with streaming, and a window outside 1..16. Those are configuration errors, so they must
be reported *before* anything is loaded - which is what the first half checks by pointing
--model-dir at a path that does not exist and requiring the failure to name the configuration
rather than the model.

The second half is the correctness property S0 exists to establish: **the speculative run
produces the target-only output**. It needs a model and a GPU, so it runs only when
--target-dir is given; the CPU half is the part that always runs.

Usage:
    python3 tests/test_speculative.py --exe "$(cabal list-bin haskell-infer-demo)" \\
        [--target-dir DIR [--target-desc D] [--draft-dir DIR] [--speculative-k K] ...]
"""

import argparse
import os
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


def expect_refusal(exe, args, needle, what):
    """A configuration error must be reported and must not come from loading a model."""
    code, out, err = run(exe, args)
    text = out + err
    check(code != 0, what + ": the run exited 0", text.strip()[:200])
    check(needle in text, what + f": the error does not mention {needle!r}", text.strip()[:200])
    check("tokenizer" not in text.lower() and "safetensors" not in text.lower(),
          what + ": the failure came from model loading, not from the configuration",
          text.strip()[:200])


def generated_text(out):
    """The text between the CLI's two '---' markers, which is what was generated."""
    parts = out.split("---", 2)
    if len(parts) < 3:
        return None
    return parts[1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--missing-model", default="/nonexistent/model-dir",
                        help="a path that must not exist, so an early refusal is observable")
    parser.add_argument("--target-dir", default=None,
                        help="when given, run the real comparison (needs a GPU and weights)")
    parser.add_argument("--target-desc", default=None,
                        help="the target's descriptor; the draft derives its own from its "
                             "config.json, because S0 admits a draft with a different "
                             "architecture and reusing the target's would defeat that")
    parser.add_argument("--draft-dir", default=None)
    parser.add_argument("--speculative-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    if not os.path.exists(args.exe):
        print(f"test_speculative: skipped (no executable at {args.exe})")
        return 0

    print("=== the combinations S0 refuses, before anything is loaded ===")
    missing = ["--model-dir", args.missing_model, "-p", "hello"]
    expect_refusal(args.exe, missing + ["--speculative-k", "3"],
                   "needs --draft-model-dir", "a window without a draft")
    expect_refusal(args.exe, missing + ["--draft-model-dir", "/nonexistent/draft",
                                        "--speculative-k", "3", "--temperature", "1.0"],
                   "greedy-only", "a window above temperature 0")
    expect_refusal(args.exe, missing + ["--draft-model-dir", "/nonexistent/draft",
                                        "--speculative-k", "3", "--temperature", "0",
                                        "--stream"],
                   "does not stream", "a window with streaming")
    for bad, needle in (("0", "between 1 and 16"), ("17", "between 1 and 16"),
                        ("two", "must be an integer")):
        expect_refusal(args.exe, missing + ["--draft-model-dir", "/nonexistent/draft",
                                            "--speculative-k", bad],
                       needle, f"a window of {bad}")

    if args.target_dir is None:
        print("test_speculative: the model half needs --target-dir; skipped")
    else:
        if not os.path.isdir(args.target_dir):
            print(f"test_speculative: FAIL: no target checkpoint at {args.target_dir}")
            return 1
        print("=== the speculative output must be the target-only output ===")
        target = ["--model-dir", args.target_dir, "-p", args.prompt,
                  "--max-tokens", str(args.max_tokens), "--temperature", "0"]
        if args.target_desc:
            target += ["--descriptor", args.target_desc]
        code, out, err = run(args.exe, target)
        if not check(code == 0, "the target-only run failed", (out + err).strip()[-300:]):
            return 1
        baseline = generated_text(out)
        print(f"  target-only: {baseline!r}")

        draft_dir = args.draft_dir or args.target_dir
        draft = ["--model-dir", args.target_dir, "-p", args.prompt,
                 "--max-tokens", str(args.max_tokens), "--temperature", "0",
                 "--draft-model-dir", draft_dir,
                 "--speculative-k", str(args.speculative_k)]
        if args.target_desc:
            draft += ["--descriptor", args.target_desc]
        code, out, err = run(args.exe, draft)
        if not check(code == 0, "the speculative run failed", (out + err).strip()[-300:]):
            return 1
        speculative = generated_text(out)
        print(f"  speculative: {speculative!r}")
        check(speculative is not None and speculative == baseline,
              "the speculative run did not reproduce the target-only output",
              f"{speculative!r} vs {baseline!r}")

    if FAILURES:
        print(f"test_speculative: {len(FAILURES)} of {CHECKS} check(s) failed")
        return 1
    print(f"test_speculative: {CHECKS} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
