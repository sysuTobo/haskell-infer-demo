"""Shared ctypes bindings for the engine C ABI (descriptor based).

The engine is created from the model descriptor JSON plus a layer placement, so
the tests no longer mirror a hand-packed struct. Dimension lookups (vocab size)
come from the engine itself.
"""

import ctypes
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import manifest_check  # noqa: E402


def bind(lib):
    lib.engine_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.engine_create.restype = ctypes.c_void_p
    lib.engine_destroy.argtypes = [ctypes.c_void_p]
    lib.engine_reset.argtypes = [ctypes.c_void_p]
    lib.engine_seq_len.argtypes = [ctypes.c_void_p]
    lib.engine_seq_len.restype = ctypes.c_int
    lib.engine_vocab_size.argtypes = [ctypes.c_void_p]
    lib.engine_vocab_size.restype = ctypes.c_int
    lib.engine_prefill.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.engine_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.engine_last_error.restype = ctypes.c_char_p
    if hasattr(lib, "engine_manifest"):
        lib.engine_manifest.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.engine_manifest.restype = ctypes.c_int
    if hasattr(lib, "engine_manifest_version"):
        lib.engine_manifest_version.restype = ctypes.c_int
    if hasattr(lib, "engine_describe"):
        lib.engine_describe.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.engine_describe.restype = ctypes.c_int
    if hasattr(lib, "engine_desc_version"):
        lib.engine_desc_version.restype = ctypes.c_int
    # Stage 3: the training path. Bound only when the library exports it, so the
    # other tests keep working against an older build.
    if hasattr(lib, "engine_train_attach"):
        lib.engine_train_attach.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.engine_train_attach.restype = ctypes.c_void_p
        lib.engine_train_store.argtypes = [ctypes.c_void_p]
        lib.engine_train_store.restype = ctypes.c_void_p
        lib.engine_train_begin_update.argtypes = [ctypes.c_void_p]
        lib.engine_train_write_master.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p]
        lib.engine_train_publish.argtypes = [ctypes.c_void_p]
        lib.engine_train_end_update.argtypes = [ctypes.c_void_p]
        lib.engine_train_step_plan.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_int64)]
        lib.engine_train_step_begin.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_void_p)]
        lib.engine_train_step_end.argtypes = [ctypes.c_void_p]
        lib.engine_train_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_int, ctypes.c_void_p]
        # The store's own accessors, for the tying and lifetime assertions.
        lib.train_store_logical_count.argtypes = [ctypes.c_void_p]
        lib.train_store_logical_count.restype = ctypes.c_int
        lib.train_store_spec_count.argtypes = [ctypes.c_void_p]
        lib.train_store_spec_count.restype = ctypes.c_int
        lib.train_store_logical_of.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.train_store_logical_of.restype = ctypes.c_int
        lib.train_store_alias_count.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.train_store_alias_count.restype = ctypes.c_int
        lib.train_store_is_frozen.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.train_store_is_frozen.restype = ctypes.c_int
        lib.train_store_version.argtypes = [ctypes.c_void_p]
        lib.train_store_version.restype = ctypes.c_int64
        lib.train_store_stale_derived_count.argtypes = [ctypes.c_void_p]
        lib.train_store_stale_derived_count.restype = ctypes.c_int
        lib.train_last_error.restype = ctypes.c_char_p
    # Stage 5: the SFT step. Bound only when the library exports it.
    if hasattr(lib, "engine_train_forward_retain"):
        lib.engine_train_forward_retain.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_int]
        lib.engine_train_loss.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_void_p]
        lib.engine_train_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.engine_train_zero_grads.argtypes = [ctypes.c_void_p]
        lib.engine_train_apply.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.engine_train_export_state.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p]
        lib.engine_train_import_state.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p]
    # Stage 5: the synchronous rollout and the phase machine's bookkeeping. Bound only
    # when the library exports the rollout entry point.
    if hasattr(lib, "engine_rollout_sample"):
        lib.engine_train_version.argtypes = [ctypes.c_void_p]
        lib.engine_train_version.restype = ctypes.c_int64
        lib.engine_rollout_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                                              ctypes.c_void_p, ctypes.c_void_p]
        lib.backward_rng_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.train_loop_create.argtypes = [ctypes.c_void_p]
        lib.train_loop_create.restype = ctypes.c_void_p
        lib.train_loop_destroy.argtypes = [ctypes.c_void_p]
        lib.train_loop_enter.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_int64)]
        lib.train_loop_leave.argtypes = [ctypes.c_void_p]
        lib.train_loop_version.argtypes = [ctypes.c_void_p]
        lib.train_loop_version.restype = ctypes.c_int64
        lib.train_loop_sequence_resets.argtypes = [ctypes.c_void_p]
        lib.train_loop_sequence_resets.restype = ctypes.c_int
        lib.train_loop_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.train_loop_read_reward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_float)]
        lib.train_loop_ratio.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_double)]
        lib.train_loop_last_error.restype = ctypes.c_char_p
    if hasattr(lib, "model_desc_role_from_name"):
        lib.model_desc_role_from_name.argtypes = [ctypes.c_char_p]
        lib.model_desc_role_from_name.restype = ctypes.c_int
    return lib


def load_descriptor(path, max_seq_len=None):
    """Read a descriptor JSON file, optionally overriding the context length."""
    with open(path) as handle:
        desc = json.load(handle)
    if max_seq_len is not None:
        desc["max_seq_len"] = max_seq_len
    return desc


def contiguous_assignment(num_layers, devices):
    """Layer -> device mapping: contiguous blocks, remainder on the first devices."""
    count = len(devices)
    base, extra = divmod(num_layers, count)
    assignment = []
    for index, device in enumerate(devices):
        assignment += [device] * (base + (1 if index < extra else 0))
    return assignment


def ptr(array):
    return ctypes.c_void_p(array.ctypes.data)


def create_engine(lib, model_dir, desc, devices):
    """Create an engine from a descriptor dict. Returns (engine, vocab_size)."""
    layer_devices = contiguous_assignment(desc["num_layers"], devices)
    dev_arr = (ctypes.c_int * len(devices))(*devices)
    layer_arr = (ctypes.c_int * len(layer_devices))(*layer_devices)
    payload = json.dumps(desc).encode()
    engine = lib.engine_create(model_dir.encode(), payload, len(devices), dev_arr, layer_arr)
    assert engine, lib.engine_last_error().decode()
    return engine, lib.engine_vocab_size(engine)


def describe(lib, engine, capacity=64 * 1024):
    """Round-trip the descriptor the engine parsed."""
    buf = ctypes.create_string_buffer(capacity)
    written = lib.engine_describe(engine, buf, capacity)
    assert written >= 0, lib.engine_last_error().decode()
    return json.loads(buf.value.decode())


# ENGINE_MANIFEST_MAX: the engine refuses a manifest that does not fit rather than
# truncating one, so the buffer is sized to the contract, not to a guess.
MANIFEST_CAPACITY = 256 * 1024


def manifest(lib, engine, capacity=MANIFEST_CAPACITY):
    """The engine's canonical execution manifest, verified before it is used.

    A capture must not record a manifest whose own digests do not re-derive in a
    second implementation, so the verification happens here, at the boundary.
    """
    buf = ctypes.create_string_buffer(capacity)
    written = lib.engine_manifest(engine, buf, capacity)
    assert written >= 0, lib.engine_last_error().decode()
    text = buf.raw[:written].decode("utf-8")
    problems = manifest_check.verify(text)
    assert not problems, "the engine's manifest does not verify: " + "; ".join(problems)
    return text
