import argparse
import ctypes
import json
import time

import numpy as np


class EngineConfig(ctypes.Structure):
    _fields_ = [("num_layers", ctypes.c_int), ("num_devices", ctypes.c_int),
                ("devices", ctypes.POINTER(ctypes.c_int)),
                ("layer_devices", ctypes.POINTER(ctypes.c_int)),
                ("max_seq_len", ctypes.c_int)]


def pointer(array):
    return ctypes.c_void_p(array.ctypes.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--devices", default="0,1")
    args = parser.parse_args()
    lib = ctypes.CDLL(args.library)
    lib.engine_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(EngineConfig)]
    lib.engine_create.restype = ctypes.c_void_p
    lib.engine_destroy.argtypes = [ctypes.c_void_p]
    lib.engine_reset.argtypes = [ctypes.c_void_p]
    lib.engine_seq_len.argtypes = [ctypes.c_void_p]
    lib.engine_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.engine_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.engine_last_error.restype = ctypes.c_char_p
    devices = [int(x) for x in args.devices.split(",")]
    ids = (ctypes.c_int * len(devices))(*devices)
    assignment = (ctypes.c_int * 64)(*(devices[min(i * len(devices) // 64, len(devices) - 1)] for i in range(64)))
    config = EngineConfig(64, len(devices), ids, assignment, 256)
    reference = np.load(args.reference)
    start = time.monotonic()
    engine = lib.engine_create(args.model_dir.encode(), ctypes.byref(config))
    assert engine, lib.engine_last_error().decode()
    print(f"Loaded in {time.monotonic() - start:.1f}s", flush=True)
    logits = np.empty(248320, dtype=np.float32)
    failures = []

    def status(code):
        assert code == 0, lib.engine_last_error().decode()

    try:
        for case in range(2):
            lib.engine_reset(engine)
            assert lib.engine_seq_len(engine) == 0
            prompt = np.ascontiguousarray(reference[f"prompt_{case}"], dtype=np.int64)
            status(lib.engine_prefill(engine, pointer(prompt), len(prompt), pointer(logits)))
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
                    status(lib.engine_decode(engine, int(token), pointer(logits)))
            assert lib.engine_seq_len(engine) == len(prompt) + len(reference[f"tokens_{case}"]) - 1

        previous_length = lib.engine_seq_len(engine)
        invalid = np.array([-1], dtype=np.int64)
        assert lib.engine_prefill(engine, pointer(invalid), 1, pointer(logits)) == -4
        assert lib.engine_seq_len(engine) == previous_length
        overflow = np.full(257, 760, dtype=np.int64)
        assert lib.engine_prefill(engine, pointer(overflow), len(overflow), pointer(logits)) == -6
        assert lib.engine_seq_len(engine) == previous_length
        print("Invalid-input and capacity checks passed", flush=True)

        prompt = np.resize(reference["prompt_0"], 129).astype(np.int64)
        lib.engine_reset(engine)
        status(lib.engine_prefill(engine, pointer(prompt), len(prompt), pointer(logits)))
        whole = logits.copy()
        assert lib.engine_seq_len(engine) == 129
        lib.engine_reset(engine)
        status(lib.engine_prefill(engine, pointer(prompt[:64]), 64, pointer(logits)))
        status(lib.engine_prefill(engine, pointer(prompt[64:]), 65, pointer(logits)))
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
            status(lib.engine_prefill(engine, pointer(part), count, pointer(logits)))
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
