import argparse
import ctypes
import json
import time

import numpy as np

from engine_bindings import bind, create_engine, describe, load_descriptor, ptr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--desc", default="descriptors/qwen38-27b.json")
    parser.add_argument("--devices", default="0,1")
    args = parser.parse_args()
    lib = bind(ctypes.CDLL(args.library))
    devices = [int(x) for x in args.devices.split(",")]
    descriptor = load_descriptor(args.desc, max_seq_len=256)
    reference = np.load(args.reference)
    start = time.monotonic()
    engine, vocab = create_engine(lib, args.model_dir, descriptor, devices)
    print(f"Loaded in {time.monotonic() - start:.1f}s (vocab {vocab})", flush=True)
    assert describe(lib, engine) == descriptor, "descriptor round-trip mismatch"
    print("Descriptor round-trip: OK", flush=True)
    logits = np.empty(vocab, dtype=np.float32)
    failures = []

    def status(code):
        assert code == 0, lib.engine_last_error().decode()

    try:
        for case in range(2):
            lib.engine_reset(engine)
            assert lib.engine_seq_len(engine) == 0
            prompt = np.ascontiguousarray(reference[f"prompt_{case}"], dtype=np.int64)
            status(lib.engine_prefill(engine, ptr(prompt), len(prompt), ptr(logits)))
            for step, token in enumerate(reference[f"tokens_{case}"]):
                golden = reference[f"logits_{case}_{step}"]
                assert np.isfinite(logits).all()
                error = logits - golden
                actual = int(logits.argmax())
                rms = float(np.sqrt(np.mean(error ** 2)))
                maxima = np.flatnonzero(golden == golden.max()).tolist()
                result = {"case": case, "step": step, "actual": actual,
                          "expected": int(token), "reference_maxima": maxima,
                          "rms": rms, "max_abs": float(np.abs(error).max())}
                print(json.dumps(result), flush=True)
                # BF16 reference logits can tie while the engine returns FP32 logits.
                if actual not in maxima or rms > 0.1:
                    failures.append(result)
                if step + 1 < len(reference[f"tokens_{case}"]):
                    status(lib.engine_decode(engine, int(token), ptr(logits)))
            assert lib.engine_seq_len(engine) == len(prompt) + len(reference[f"tokens_{case}"]) - 1

        previous_length = lib.engine_seq_len(engine)
        invalid = np.array([-1], dtype=np.int64)
        assert lib.engine_prefill(engine, ptr(invalid), 1, ptr(logits)) == -4
        assert lib.engine_seq_len(engine) == previous_length
        overflow = np.full(257, 760, dtype=np.int64)
        assert lib.engine_prefill(engine, ptr(overflow), len(overflow), ptr(logits)) == -6
        assert lib.engine_seq_len(engine) == previous_length
        print("Invalid-input and capacity checks passed", flush=True)

        # 128 is the engine's chunk size; 129 crosses a chunk boundary.
        prompt = np.resize(reference["prompt_0"], 129).astype(np.int64)
        lib.engine_reset(engine)
        status(lib.engine_prefill(engine, ptr(prompt), len(prompt), ptr(logits)))
        whole = logits.copy()
        assert lib.engine_seq_len(engine) == 129
        lib.engine_reset(engine)
        status(lib.engine_prefill(engine, ptr(prompt[:64]), 64, ptr(logits)))
        status(lib.engine_prefill(engine, ptr(prompt[64:]), 65, ptr(logits)))
        assert lib.engine_seq_len(engine) == 129
        # 不同 chunk 切分走不同 BF16 数值路径（FlashInfer 按 qo_len 选 CTA tile、
        # chunk pipeline 内部 rounding 点不同），经 64 层残差放大后 logit RMS 可达
        # ~1，远大于引擎 vs 参考的 0.02（短 prompt 单 chunk）。功能正确性以 top1
        # 一致为准；RMS 上限仅用于捕获 state 丢失/爆炸（真 bug 会到几十或 NaN）。
        rms = float(np.sqrt(np.mean((whole - logits) ** 2)))
        print(f"129-token boundary: rms={rms:.7g} (logit std={whole.std():.3g}), "
              f"top1={whole.argmax()}/{logits.argmax()}", flush=True)
        if not np.isfinite(rms) or rms > 5.0 or whole.argmax() != logits.argmax():
            failures.append({"chunk_boundary_rms": rms})
        lib.engine_reset(engine)
        for start, count in ((0, 64), (64, 64), (128, 1)):
            part = np.ascontiguousarray(prompt[start:start + count])
            status(lib.engine_prefill(engine, ptr(part), count, ptr(logits)))
        aligned_rms = float(np.sqrt(np.mean((whole - logits) ** 2)))
        print(f"64+64+1 boundary: rms={aligned_rms:.7g} (logit std={whole.std():.3g}), "
              f"top1={whole.argmax()}/{logits.argmax()}", flush=True)
        if not np.isfinite(aligned_rms) or aligned_rms > 5.0 or whole.argmax() != logits.argmax():
            failures.append({"aligned_boundary_rms": aligned_rms})
    finally:
        lib.engine_destroy(engine)
    assert not failures, json.dumps(failures, indent=2)
    print("Engine numerical and state tests passed", flush=True)


if __name__ == "__main__":
    main()
