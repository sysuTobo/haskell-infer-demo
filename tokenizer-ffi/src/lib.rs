//! tokenizer-ffi: C ABI wrapper around HuggingFace `tokenizers`.
//!
//! Compiled as `libtokenizer_ffi.so` (cdylib) and linked from Haskell
//! via `foreign import ccall`.
//!
//! Safety: all `extern "C"` functions are `unsafe` by contract. The caller
//! (Haskell) must ensure pointer validity and lifetime.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;
use tokenizers::Tokenizer;

/// Opaque handle to a loaded tokenizer.
pub struct TokenizerHandle {
    inner: Tokenizer,
}

/// Load a tokenizer from a HuggingFace `tokenizer.json` file.
///
/// # Safety
/// `path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_load(path: *const c_char) -> *mut TokenizerHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let c_str = CStr::from_ptr(path);
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    match Tokenizer::from_file(path_str) {
        Ok(tok) => Box::into_raw(Box::new(TokenizerHandle { inner: tok })),
        Err(_) => ptr::null_mut(),
    }
}

/// Encode text into token IDs.
///
/// Returns the number of tokens written to `out_ids`, or -1 on error.
/// `out_ids` must have capacity for at least `max_len` int64 values.
///
/// # Safety
/// All pointers must be valid. `text` must be null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    out_ids: *mut i64,
    max_len: c_int,
) -> c_int {
    if handle.is_null() || text.is_null() || out_ids.is_null() || max_len <= 0 {
        return -1;
    }

    let h = &*handle;
    let c_str = CStr::from_ptr(text);
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match h.inner.encode(text_str, false) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len as usize);
            for (i, &id) in ids.iter().take(len).enumerate() {
                *out_ids.add(i) = id as i64;
            }
            len as c_int
        }
        Err(_) => -1,
    }
}

/// Decode token IDs back to text.
///
/// Returns the number of bytes written to `out_buf` (excluding null
/// terminator), or -1 on error. The output is null-terminated UTF-8.
///
/// # Safety
/// All pointers must be valid. `out_buf` must have capacity for `buf_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode(
    handle: *mut TokenizerHandle,
    ids: *const i64,
    len: c_int,
    out_buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if handle.is_null() || ids.is_null() || out_buf.is_null() || len <= 0 || buf_size <= 0 {
        return -1;
    }

    let h = &*handle;
    let token_ids: Vec<u32> = (0..len as usize)
        .map(|i| *ids.add(i) as u32)
        .collect();

    match h.inner.decode(&token_ids, true) {
        Ok(text) => {
            let bytes = text.as_bytes();
            let write_len = bytes.len().min((buf_size - 1) as usize);
            ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, write_len);
            *out_buf.add(write_len) = 0; // null terminator
            write_len as c_int
        }
        Err(_) => -1,
    }
}

/// Decode a single token ID to text (for streaming output).
///
/// Returns bytes written, or -1 on error.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode_single(
    handle: *mut TokenizerHandle,
    id: i64,
    out_buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if handle.is_null() || out_buf.is_null() || buf_size <= 0 {
        return -1;
    }

    let h = &*handle;
    match h.inner.decode(&[id as u32], false) {
        Ok(text) => {
            let bytes = text.as_bytes();
            let write_len = bytes.len().min((buf_size - 1) as usize);
            ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, write_len);
            *out_buf.add(write_len) = 0;
            write_len as c_int
        }
        Err(_) => -1,
    }
}

/// Get the vocabulary size.
///
/// # Safety
/// `handle` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_vocab_size(handle: *mut TokenizerHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    h.inner.get_vocab_size(true) as c_int
}

/// Free a tokenizer handle.
///
/// # Safety
/// `handle` must be a valid pointer returned by `tokenizer_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
