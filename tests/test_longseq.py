import argparse
import ctypes
from collections import Counter

import numpy as np


class EngineConfig(ctypes.Structure):
    _fields_ = [("num_layers", ctypes.c_int), ("num_devices", ctypes.c_int),
                ("devices", ctypes.POINTER(ctypes.c_int)),
                ("layer_devices", ctypes.POINTER(ctypes.c_int)),
                ("max_seq_len", ctypes.c_int)]


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def repetition_rate(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0, None
    most = Counter(grams).most_common(1)[0]
    return 1.0 - len(set(grams)) / len(grams), most


# ~300 token expository passage; invites continued coherent prose.
PROMPT = (
    "The history of artificial intelligence as a formal discipline begins in the "
    "mid twentieth century, although its roots stretch back to ancient myths about "
    "artificial beings endowed with intelligence by master craftsmen. The study of "
    "logic and formal reasoning, from Greek philosophers through Leibniz, Boole "
    "and Frege to Turing, laid the conceptual groundwork for machines that could "
    "manipulate symbols according to precise rules. Alan Turing's 1950 paper posed "
    "the question of whether machines could think, and proposed an imitation game "
    "as a practical substitute for the ill-defined notion. The field itself was "
    "named at a summer workshop at Dartmouth College in 1956, organized by John "
    "McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon. The decades "
    "that followed saw waves of optimism and disappointment: early programs solved "
    "algebra word problems and proved theorems, but the combinatorial difficulty of "
    "real-world tasks soon became apparent, leading to the first AI winter in the "
    "1970s. Expert systems revived commercial interest in the 1980s, only to be "
    "followed by a second winter when their brittleness and maintenance costs became "
    "clear. The modern era, beginning in the late 1990s and accelerating after 2012, "
    "has been driven by machine learning, large datasets, and graphics processors, "
    "with deep neural networks achieving superhuman performance on tasks ranging from "
    "image classification to the game of Go. Large language models, trained on vast "
    "corpora of text, now generate fluent prose, summarize documents, translate "
    "languages, and assist with programming. Yet fundamental questions remain open: "
    "whether such systems genuinely understand, how to make them reliable and "
    "aligned with human values, and what the long-term societal consequences of "
    "increasingly capable machines will be. Researchers continue to explore these "
    "frontiers, balancing enthusiasm about the technology's potential against careful "
    "attention to risk. The story of artificial intelligence is therefore not a "
    "simple march of progress but a complex interplay of ideas, funding cycles, "
    "hardware constraints, and shifting expectations about what intelligence itself "
    "means, a story that is still being written today as new breakthroughs and new "
    "concerns emerge in roughly equal measure across laboratories around the world."
)
EOS = (248046, 248044)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--gen-tokens", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    lib = ctypes.CDLL(args.library)
    lib.engine_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(EngineConfig)]
    lib.engine_create.restype = ctypes.c_void_p
    lib.engine_destroy.argtypes = [ctypes.c_void_p]
    lib.engine_reset.argtypes = [ctypes.c_void_p]
    lib.engine_seq_len.argtypes = [ctypes.c_void_p]
    lib.engine_seq_len.restype = ctypes.c_int
    lib.engine_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.engine_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.engine_last_error.restype = ctypes.c_char_p

    devices = [int(x) for x in args.devices.split(",")]
    dev_arr = (ctypes.c_int * len(devices))(*devices)
    assign = (ctypes.c_int * 64)(*(devices[min(i * len(devices) // 64, len(devices) - 1)]
                                   for i in range(64)))
    cfg = EngineConfig(64, len(devices), dev_arr, assign, 1024)

    eng = lib.engine_create(args.model_dir.encode(), ctypes.byref(cfg))
    assert eng, lib.engine_last_error().decode()

    vocab = 248320
    logits = np.empty(vocab, dtype=np.float32)

    def prefill(idarr):
        a = np.ascontiguousarray(idarr, dtype=np.int64)
        rc = lib.engine_prefill(eng, ptr(a), len(a), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
        return logits.copy()

    def decode(token_id):
        rc = lib.engine_decode(eng, int(token_id), ptr(logits))
        assert rc == 0, lib.engine_last_error().decode()
        return logits.copy()

    prompt_ids = tok(PROMPT, return_tensors="np").input_ids[0].astype(np.int64)
    n = len(prompt_ids)
    print(f"Prompt tokens: {n} (max_seq=1024)", flush=True)
    assert n > 128, "prompt must exceed one chunk"

    failures = []
    try:
        # 1) Chunk-split self-consistency: one-shot (internal 128+128+..) vs a
        #    different manual split. Both must agree on the next token.
        lib.engine_reset(eng)
        whole = prefill(prompt_ids)
        seq_whole = lib.engine_seq_len(eng)
        lib.engine_reset(eng)
        prefill(prompt_ids[:150])
        split = prefill(prompt_ids[150:])
        seq_split = lib.engine_seq_len(eng)
        rms = float(np.sqrt(np.mean((whole - split) ** 2)))
        t_w, t_s = int(whole.argmax()), int(split.argmax())
        print(f"[chunk-split] seq_len {seq_whole}/{seq_split}, top1 {t_w}/{t_s}, "
              f"rms={rms:.4g} (logit std={whole.std():.3g})", flush=True)
        if seq_whole != n or seq_split != n:
            failures.append({"seq_len_mismatch": [seq_whole, seq_split, n]})
        if t_w != t_s or not np.isfinite(rms) or rms > 5.0:
            failures.append({"chunk_split": {"top1": [t_w, t_s], "rms": rms}})

        # 2) Long generation from the one-shot prefill state.
        lib.engine_reset(eng)
        cur = prefill(prompt_ids)
        gen = []
        nxt = int(cur.argmax())
        for _ in range(args.gen_tokens):
            gen.append(nxt)
            cur = decode(nxt)
            nxt = int(cur.argmax())
            if nxt in EOS:
                break
        text = tok.decode(gen)
        rep, top_gram = repetition_rate(gen)
        first = repetition_rate(gen[:64])[0]
        last = repetition_rate(gen[-64:])[0] if len(gen) >= 64 else rep
        print(f"[generate] {len(gen)} tokens, stopped_at_eos={gen[-1] in EOS if gen else None}",
              flush=True)
        print(f"[generate] repetition_rate(4-gram)={rep:.3f} "
              f"(first64={first:.3f}, last64={last:.3f})", flush=True)
        if top_gram:
            print(f"[generate] most common 4-gram x{top_gram[1]}: {tok.decode(top_gram[0])!r}",
                  flush=True)
        print("=== GENERATED TEXT ===", flush=True)
        print(text, flush=True)
        print("=== END ===", flush=True)
        # Degradation heuristics: collapsed text repeats heavily; coherent prose
        # keeps the tail no worse than the head by a wide margin.
        if rep > 0.5:
            failures.append({"repetition_rate": rep})
        if len(gen) >= 64 and last > first + 0.35:
            failures.append({"tail_degradation": {"first64": first, "last64": last}})
        if len(gen) < 20 and (not gen or gen[-1] not in EOS):
            failures.append({"too_short_without_eos": len(gen)})
    finally:
        lib.engine_destroy(eng)

    assert not failures, failures
    print("Long-sequence coherence verification passed", flush=True)


if __name__ == "__main__":
    main()
