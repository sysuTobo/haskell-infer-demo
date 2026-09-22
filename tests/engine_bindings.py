"""Shared ctypes bindings for the engine C ABI (descriptor based).

The engine is created from the model descriptor JSON plus a layer placement, so
the tests no longer mirror a hand-packed struct. Dimension lookups (vocab size)
come from the engine itself.
"""

import ctypes
import json

import numpy as np


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
    if hasattr(lib, "engine_describe"):
        lib.engine_describe.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.engine_describe.restype = ctypes.c_int
    if hasattr(lib, "engine_desc_version"):
        lib.engine_desc_version.restype = ctypes.c_int
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
