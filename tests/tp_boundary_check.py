"""Is the tp1-vs-tp2 logit difference at the 128/129 chunk boundary prompt-specific?

Compares prefill logits of the two arms on:
  natural128 - 128 natural tokens (one full chunk, no boundary)
  natural129 - 129 natural tokens (full chunk + 1-token chunk)
  repeat129  - the repeated-token fixture from test_tp.py (degenerate, chunk + 1)

If the natural prompts stay small and only the degenerate one shows O(1) rms, the
boundary difference is the model's own sensitivity amplification of BF16-level
path differences, not a placement bug (the engine documents rms ~1.0 for any two
different prefill chunkings at that boundary).
"""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_bindings import bind, create_engine, load_descriptor, ptr  # noqa: E402
from test_longseq import PROMPT as PASSAGE  # noqa: E402

PROMPT_MAIN = [760, 6511, 314, 9564, 369]


def prefill_logits(lib, model_dir, desc, prompt):
    engine, vocab = create_engine(lib, model_dir, desc, [0, 1])
    try:
        logits = np.empty(vocab, dtype=np.float32)
        ids = np.ascontiguousarray(prompt, dtype=np.int64)
        rc = lib.engine_prefill(engine, ptr(ids), len(ids), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
        return logits.copy()
    finally:
        lib.engine_destroy(engine)


def main():
    lib = bind(ctypes.CDLL(sys.argv[1]))
    model_dir = sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    natural = tok(PASSAGE, return_tensors="np").input_ids[0].astype(np.int64).tolist()
    print(f"natural passage has {len(natural)} tokens", flush=True)

    base = load_descriptor("descriptors/qwen38-27b.json", max_seq_len=256)
    tp2 = dict(base)
    tp2["tp_size"] = 2

    cases = {
        "natural128": natural[:128],
        "natural129": natural[:129],
        "repeat129": np.resize(PROMPT_MAIN, 129).astype(int).tolist(),
    }
    for name, prompt in cases.items():
        a = prefill_logits(lib, model_dir, base, prompt)
        b = prefill_logits(lib, model_dir, tp2, prompt)
        rms = float(np.sqrt(np.mean((a - b) ** 2)))
        print(f"{name}: tokens={len(prompt)} rms={rms:.4g} "
              f"(std={float(a.std()):.3g}, top1 {int(a.argmax())}/{int(b.argmax())})",
              flush=True)


if __name__ == "__main__":
    main()
