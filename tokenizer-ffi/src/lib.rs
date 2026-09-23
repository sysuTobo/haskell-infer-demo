//! tokenizer-ffi: C ABI wrapper around HuggingFace `tokenizers`.
//!
//! Compiled as `libtokenizer_ffi.so` (cdylib) and linked from Haskell
//! via `foreign import ccall`.
//!
//! Capacity protocol: every converting call has a companion length query that
//! reports the buffer size the conversion needs. A conversion whose buffer is
//! too small writes *nothing* and returns [`CAPACITY_INSUFFICIENT`]; an empty
//! result is legal and reports `0` bytes. Text lengths always include the
//! terminating NUL, so `length == 1` means "the empty string".
//!
//! Safety: all `extern "C"` functions are `unsafe` by contract. The caller
//! (Haskell) must ensure pointer validity and lifetime.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use tokenizers::Tokenizer;

/// The caller's buffer is smaller than the reported required size; nothing was
/// written. Every other negative return is a generic failure (-1).
pub const CAPACITY_INSUFFICIENT: c_int = -2;

/// Streaming decode uses the same special-token policy as whole-sequence
/// decode, so the two paths produce identical text for the same ids.
const SKIP_SPECIAL_TOKENS: bool = true;

/// Opaque handle to a loaded tokenizer.
pub struct TokenizerHandle {
    inner: Tokenizer,
}

/// Opaque handle to an incremental decode session.
///
/// Owns a clone of the tokenizer plus the decode state, so the stream does not
/// borrow from [`TokenizerHandle`] and has no lifetime tied to it.
pub struct TokenizerStream {
    handle: Tokenizer,
    skip_special_tokens: bool,
    /// Token ids kept because they may still be part of an incomplete chunk.
    ids: Vec<u32>,
    /// Text of `ids` that was already emitted (trimmed off the next chunk).
    prefix: String,
    /// Index in `ids` where the emitted prefix starts.
    prefix_index: usize,
    /// Second prefix kept only so a leading token can apply its own effect.
    read_index: usize,
    /// Produced-but-not-yet-drained text.
    pending: String,
    /// Set by `finish`: no further feed is meaningful until `reset`.
    finished: bool,
}

impl TokenizerStream {
    fn new(handle: Tokenizer) -> Self {
        Self {
            handle,
            skip_special_tokens: SKIP_SPECIAL_TOKENS,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            read_index: 0,
            pending: String::new(),
            finished: false,
        }
    }

    /// Advance the decode state by one token, appending any newly available
    /// text to `pending`. Returns `false` on a decoding error.
    ///
    /// `step_decode_stream` in tokenizers 0.20.4 drains the retained ids from
    /// `read_index`, which it sets to the *previous* call's `prefix_index`. When
    /// chunks arrive back to back that stale index trims too little, the
    /// retained window drifts and `ids.len() - prefix_index` eventually
    /// underflows; upstream fixed the same defect in 0.22 by draining from the
    /// current `prefix_index`. Seeding `read_index` from `prefix_index` before
    /// each call reproduces the fixed behavior on top of the pinned crate.
    ///
    /// The call is also isolated from unwinding: a panic in a dependency must
    /// never abort the process across the C ABI.
    fn feed(&mut self, id: u32) -> bool {
        self.read_index = self.prefix_index;
        let step = catch_unwind(AssertUnwindSafe(|| {
            tokenizers::step_decode_stream(
                &self.handle,
                id,
                self.skip_special_tokens,
                &mut self.ids,
                &mut self.prefix,
                &mut self.prefix_index,
                &mut self.read_index,
            )
        }));
        match step {
            Ok(Ok(Some(text))) => {
                self.pending.push_str(&text);
                true
            }
            // `None` means the ids so far do not yet form a valid UTF-8 chunk
            // (byte fallback across tokens); nothing is emitted, not an error.
            Ok(Ok(None)) => true,
            Ok(Err(_)) | Err(_) => false,
        }
    }

    /// Move the not-yet-decodable tail into `pending`.
    ///
    /// `step_decode_stream` withholds ids whose bytes are an incomplete UTF-8
    /// sequence. At the end of a generation those bytes would be lost, so the
    /// tail is decoded as-is (yielding U+FFFD for the truncated bytes) exactly
    /// as whole-sequence `decode` would.
    fn flush_tail(&mut self) {
        if self.finished {
            return;
        }
        if let Ok(full) = self.handle.decode(&self.ids, self.skip_special_tokens) {
            let tail = match full.strip_prefix(self.prefix.as_str()) {
                Some(rest) => rest.to_string(),
                None => full,
            };
            self.pending.push_str(&tail);
        }
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        self.read_index = 0;
        self.finished = true;
    }
}

/// Copy `text` into `out` as a NUL-terminated C string.
///
/// Returns the number of bytes written excluding the NUL, or
/// [`CAPACITY_INSUFFICIENT`] (writing nothing) when the buffer is too small.
///
/// # Safety
/// `out` must be valid for `buf_size` bytes, or null when `buf_size` is 0.
unsafe fn copy_out(text: &str, out: *mut c_char, buf_size: c_int) -> c_int {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        // An empty result is legal and needs no room; write the NUL only if
        // there is space, and never fail on a zero-sized buffer.
        if !out.is_null() && buf_size >= 1 {
            *out = 0;
        }
        return 0;
    }
    if out.is_null() || (buf_size as i64) < bytes.len() as i64 + 1 {
        return CAPACITY_INSUFFICIENT;
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
    *out.add(bytes.len()) = 0;
    bytes.len() as c_int
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

/// Free a tokenizer handle.
///
/// # Safety
/// `handle` must be a pointer returned by `tokenizer_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
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

/// Number of token slots `tokenizer_encode` needs for `text`, or -1 on error.
///
/// # Safety
/// `handle` and `text` must be valid pointers.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_encode_len(
    handle: *mut TokenizerHandle,
    text: *const c_char,
) -> c_int {
    if handle.is_null() || text.is_null() {
        return -1;
    }
    let h = &*handle;
    let c_str = CStr::from_ptr(text);
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match h.inner.encode(text_str, false) {
        Ok(encoding) => encoding.get_ids().len() as c_int,
        Err(_) => -1,
    }
}

/// Encode text into token IDs.
///
/// Returns the number of ids written (>= 0), or [`CAPACITY_INSUFFICIENT`] when
/// `max_len` is smaller than the required count (nothing is written), or -1 on
/// error. `out_ids` may be null only when the required count is 0.
///
/// # Safety
/// `handle` and `text` must be valid pointers; `out_ids` must be valid for
/// `max_len` int64 values.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    out_ids: *mut i64,
    max_len: c_int,
) -> c_int {
    if handle.is_null() || text.is_null() {
        return -1;
    }
    let h = &*handle;
    let c_str = CStr::from_ptr(text);
    let text_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let encoding = match h.inner.encode(text_str, false) {
        Ok(encoding) => encoding,
        Err(_) => return -1,
    };
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return 0;
    }
    if out_ids.is_null() || (max_len as i64) < ids.len() as i64 {
        return CAPACITY_INSUFFICIENT;
    }
    for (i, &id) in ids.iter().enumerate() {
        *out_ids.add(i) = id as i64;
    }
    ids.len() as c_int
}

/// Buffer size in bytes (including the NUL) that `tokenizer_decode` needs for
/// `ids`, or -1 on error.
///
/// # Safety
/// `handle` must be valid; `ids` must be valid for `len` values when `len > 0`.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode_len(
    handle: *mut TokenizerHandle,
    ids: *const i64,
    len: c_int,
) -> c_int {
    if handle.is_null() || len < 0 || (len > 0 && ids.is_null()) {
        return -1;
    }
    let h = &*handle;
    let token_ids: Vec<u32> = (0..len as usize).map(|i| *ids.add(i) as u32).collect();
    match h.inner.decode(&token_ids, SKIP_SPECIAL_TOKENS) {
        Ok(text) => text.len() as c_int + 1,
        Err(_) => -1,
    }
}

/// Decode token IDs back to text.
///
/// Returns the number of bytes written excluding the NUL terminator, or
/// [`CAPACITY_INSUFFICIENT`] when `buf_size` is smaller than the required size
/// (nothing is written), or -1 on error.
///
/// # Safety
/// `handle` must be valid; `ids` must be valid for `len` values when `len > 0`;
/// `out_buf` must be valid for `buf_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode(
    handle: *mut TokenizerHandle,
    ids: *const i64,
    len: c_int,
    out_buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if handle.is_null() || len < 0 || (len > 0 && ids.is_null()) || buf_size < 0 {
        return -1;
    }
    let h = &*handle;
    let token_ids: Vec<u32> = (0..len as usize).map(|i| *ids.add(i) as u32).collect();
    match h.inner.decode(&token_ids, SKIP_SPECIAL_TOKENS) {
        Ok(text) => copy_out(&text, out_buf, buf_size),
        Err(_) => -1,
    }
}

/// Create an incremental decode session over a clone of `handle`.
///
/// Returns null when `handle` is null or the tokenizer cannot be cloned.
///
/// # Safety
/// `handle` must be a valid pointer returned by `tokenizer_load`.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_new(handle: *mut TokenizerHandle) -> *mut TokenizerStream {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let h = &*handle;
    Box::into_raw(Box::new(TokenizerStream::new(h.inner.clone())))
}

/// Free a stream handle.
///
/// # Safety
/// `stream` must be a pointer returned by `tokenizer_stream_new`, or null.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_free(stream: *mut TokenizerStream) {
    if !stream.is_null() {
        drop(Box::from_raw(stream));
    }
}

/// Reset the stream to its initial state (all decode state and pending text).
///
/// # Safety
/// `stream` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_reset(stream: *mut TokenizerStream) -> c_int {
    if stream.is_null() {
        return -1;
    }
    let s = &mut *stream;
    s.ids.clear();
    s.prefix.clear();
    s.prefix_index = 0;
    s.read_index = 0;
    s.pending.clear();
    s.finished = false;
    0
}

/// Feed one token id, advancing the decode state exactly once.
///
/// The available text is buffered and read back with `tokenizer_stream_pending`
/// / `tokenizer_stream_drain`; feeding is never implied by reading, so a
/// length query followed by a retry cannot double-advance the state.
/// Returns 0 on success, -1 on error.
///
/// # Safety
/// `stream` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_feed(stream: *mut TokenizerStream, id: i64) -> c_int {
    if stream.is_null() || id < 0 {
        return -1;
    }
    let s = &mut *stream;
    if s.finished {
        return -1;
    }
    if s.feed(id as u32) {
        0
    } else {
        -1
    }
}

/// Bytes (including the NUL) that `tokenizer_stream_drain` will write, or -1.
///
/// # Safety
/// `stream` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_pending(stream: *mut TokenizerStream) -> c_int {
    if stream.is_null() {
        return -1;
    }
    let s = &*stream;
    s.pending.len() as c_int + 1
}

/// Copy the pending text out and clear it.
///
/// Returns the bytes written excluding the NUL, or [`CAPACITY_INSUFFICIENT`]
/// when the buffer is too small (the pending text is kept for a retry), or -1.
///
/// # Safety
/// `buf` must be valid for `buf_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_drain(
    stream: *mut TokenizerStream,
    buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if stream.is_null() || buf_size < 0 {
        return -1;
    }
    let s = &mut *stream;
    if s.pending.is_empty() {
        return copy_out("", buf, buf_size);
    }
    if buf.is_null() || (buf_size as i64) < s.pending.len() as i64 + 1 {
        return CAPACITY_INSUFFICIENT;
    }
    let text = std::mem::take(&mut s.pending);
    copy_out(&text, buf, buf_size)
}

/// Flush the incomplete tail of the stream and copy it out.
///
/// Idempotent: the tail is decoded once, and later calls return an empty
/// result without an error. Same return convention as `tokenizer_stream_drain`.
///
/// # Safety
/// `buf` must be valid for `buf_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn tokenizer_stream_finish(
    stream: *mut TokenizerStream,
    buf: *mut c_char,
    buf_size: c_int,
) -> c_int {
    if stream.is_null() || buf_size < 0 {
        return -1;
    }
    let s = &mut *stream;
    s.flush_tail();
    tokenizer_stream_drain(stream, buf, buf_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::CString;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer};

    /// In-memory byte-level BPE: the vocabulary holds only the 256 `<0xXX>`
    /// byte tokens, so every character falls back to its UTF-8 bytes and decode
    /// reassembles the input exactly. A multi-byte character therefore spans
    /// several tokens, which is what the streaming tests rely on. Nothing is
    /// downloaded or read from disk.
    fn fixture() -> Tokenizer {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for byte in 0..=255u32 {
            vocab.insert(format!("<0x{byte:02X}>"), byte);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .byte_fallback(true)
            .build()
            .expect("fixture BPE builds");
        let mut tok = Tokenizer::new(bpe);
        tok.with_decoder(Some(ByteFallback::new()));
        tok.add_special_tokens(&[AddedToken::from("<eos>", true)]);
        tok
    }

    fn byte_id(tok: &Tokenizer, byte: u8) -> i64 {
        *tok.get_vocab(true)
            .get(&format!("<0x{byte:02X}>"))
            .expect("byte token") as i64
    }

    fn eos_id(tok: &Tokenizer) -> i64 {
        *tok.get_vocab(true).get("<eos>").expect("special token") as i64
    }

    /// Owns a loaded C handle and frees it on drop.
    struct Handle {
        ptr: *mut TokenizerHandle,
    }

    impl Handle {
        fn new(tok: Tokenizer) -> Self {
            Self {
                ptr: Box::into_raw(Box::new(TokenizerHandle { inner: tok })),
            }
        }

        fn ptr(&self) -> *mut TokenizerHandle {
            self.ptr
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { tokenizer_free(self.ptr) };
        }
    }

    fn encode(handle: *mut TokenizerHandle, text: &str, cap: usize) -> Vec<i64> {
        let c_text = CString::new(text).unwrap();
        let mut out = vec![0i64; cap];
        let written =
            unsafe { tokenizer_encode(handle, c_text.as_ptr(), out.as_mut_ptr(), cap as c_int) };
        assert!(written >= 0, "encode failed: {written}");
        out.truncate(written as usize);
        out
    }

    fn decode(handle: *mut TokenizerHandle, ids: &[i64], cap: usize) -> String {
        let mut buf = vec![0i8; cap];
        let written = unsafe {
            tokenizer_decode(
                handle,
                ids.as_ptr(),
                ids.len() as c_int,
                buf.as_mut_ptr(),
                cap as c_int,
            )
        };
        assert!(written >= 0, "decode failed: {written}");
        String::from_utf8(buf[..written as usize].iter().map(|b| *b as u8).collect()).unwrap()
    }

    /// Flush the stream's tail and return everything that is left.
    ///
    /// Follows the documented two-step protocol: a `finish` with no room at all
    /// flushes the tail into the pending buffer and reports the capacity error,
    /// after which `pending` gives the size to allocate.
    fn finish_text(stream: *mut TokenizerStream) -> String {
        let status = unsafe { tokenizer_stream_finish(stream, ptr::null_mut(), 0) };
        assert!(status == 0 || status == CAPACITY_INSUFFICIENT, "finish failed: {status}");
        let pending = unsafe { tokenizer_stream_pending(stream) };
        assert!(pending >= 1);
        let (_, text) = drain(stream, pending as usize);
        text
    }

    fn drain(stream: *mut TokenizerStream, cap: usize) -> (c_int, String) {
        let mut buf = vec![0i8; cap];
        let written =
            unsafe { tokenizer_stream_drain(stream, buf.as_mut_ptr(), cap as c_int) };
        let text = if written > 0 {
            String::from_utf8(buf[..written as usize].iter().map(|b| *b as u8).collect()).unwrap()
        } else {
            String::new()
        };
        (written, text)
    }

    #[test]
    fn encode_reports_the_required_length_and_rejects_short_buffers() {
        let h = Handle::new(fixture());
        let text = CString::new("Hello world").unwrap();
        let required = unsafe { tokenizer_encode_len(h.ptr(), text.as_ptr()) };
        assert!(required > 0);
        assert_eq!(required as usize, encode(h.ptr(), "Hello world", 64).len());

        // A short buffer is reported as a capacity error and left untouched.
        let mut out = vec![7i64; 8];
        let status = unsafe { tokenizer_encode(h.ptr(), text.as_ptr(), out.as_mut_ptr(), 1) };
        assert_eq!(status, CAPACITY_INSUFFICIENT);
        assert!(out.iter().all(|v| *v == 7), "a short encode must not write");
    }

    #[test]
    fn decode_round_trips_ascii_and_multibyte_text() {
        let h = Handle::new(fixture());
        for text in ["Hello world", "你好，世界", "", "Hello 你好 world", "a\tb\n"] {
            let ids = encode(h.ptr(), text, 512);
            let required =
                unsafe { tokenizer_decode_len(h.ptr(), ids.as_ptr(), ids.len() as c_int) };
            assert_eq!(required as usize, text.len() + 1, "the length includes the NUL");
            assert_eq!(decode(h.ptr(), &ids, required as usize), text);
        }
    }

    #[test]
    fn decode_never_partially_writes_a_short_buffer() {
        let h = Handle::new(fixture());
        let ids = encode(h.ptr(), "Hello world", 64);
        let mut buf = vec![b'#' as i8; 64];
        let status = unsafe {
            tokenizer_decode(h.ptr(), ids.as_ptr(), ids.len() as c_int, buf.as_mut_ptr(), 2)
        };
        assert_eq!(status, CAPACITY_INSUFFICIENT);
        assert!(buf.iter().all(|b| *b == b'#' as i8), "a short decode must not write");
    }

    #[test]
    fn decode_handles_text_longer_than_the_old_fixed_buffer() {
        // The previous Haskell wrapper allocated a fixed 4096-byte buffer and
        // silently truncated; the length query must report the full size.
        let h = Handle::new(fixture());
        let text = "Hello world ".repeat(600);
        let ids = encode(h.ptr(), &text, 1 << 16);
        let required =
            unsafe { tokenizer_decode_len(h.ptr(), ids.as_ptr(), ids.len() as c_int) };
        assert!(required as usize > 4096 + 1);
        assert_eq!(decode(h.ptr(), &ids, required as usize), text);
    }

    #[test]
    fn special_tokens_follow_the_same_policy_in_both_paths() {
        let tok = fixture();
        let eos = eos_id(&tok);
        let h = Handle::new(tok);
        // Whole-sequence decode and a stream must agree: the special token is
        // skipped in both.
        assert_eq!(decode(h.ptr(), &[eos], 16), "");
        let stream = unsafe { tokenizer_stream_new(h.ptr()) };
        assert!(!stream.is_null());
        assert_eq!(unsafe { tokenizer_stream_feed(stream, eos) }, 0);
        let (written, text) = drain(stream, 16);
        assert_eq!((written, text.as_str()), (0, ""));
        unsafe { tokenizer_stream_free(stream) };
    }

    #[test]
    fn stream_holds_back_a_multibyte_character_split_across_tokens() {
        let tok = fixture();
        let h = Handle::new(tok);
        let stream = unsafe { tokenizer_stream_new(h.ptr()) };
        assert!(!stream.is_null());

        // The first byte of "你" cannot decode on its own: the stream buffers it
        // instead of emitting a replacement character.
        let ids = encode(h.ptr(), "你", 16);
        assert_eq!(ids.len(), 3, "a byte-level fixture needs three tokens here");
        assert_eq!(unsafe { tokenizer_stream_feed(stream, ids[0]) }, 0);
        let (written, _) = drain(stream, 64);
        assert_eq!(written, 0, "an incomplete sequence must not be emitted");

        for id in &ids[1..] {
            assert_eq!(unsafe { tokenizer_stream_feed(stream, *id) }, 0);
        }
        assert_eq!(drain(stream, 64).1, "你", "surfaces once its bytes are complete");
        unsafe { tokenizer_stream_free(stream) };
    }

    #[test]
    fn stream_drain_is_retry_safe_and_advances_only_once() {
        let tok = fixture();
        let h = Handle::new(tok);
        let stream = unsafe { tokenizer_stream_new(h.ptr()) };
        for id in encode(h.ptr(), "Hello world", 64) {
            assert_eq!(unsafe { tokenizer_stream_feed(stream, id) }, 0);
        }
        let pending = unsafe { tokenizer_stream_pending(stream) };
        assert!(pending > 1);

        // A failed drain keeps the text so the caller can retry with a bigger
        // buffer: neither the query nor the failure advanced the state.
        let mut small = vec![0i8; 2];
        assert_eq!(
            unsafe { tokenizer_stream_drain(stream, small.as_mut_ptr(), 2) },
            CAPACITY_INSUFFICIENT
        );
        assert_eq!(unsafe { tokenizer_stream_pending(stream) }, pending);

        let (written, text) = drain(stream, pending as usize);
        assert_eq!(written, pending - 1);
        assert_eq!(text, "Hello world");
        assert_eq!(unsafe { tokenizer_stream_pending(stream) }, 1, "pending drained");
        unsafe { tokenizer_stream_free(stream) };
    }

    #[test]
    fn streaming_one_token_at_a_time_matches_whole_sequence_decode() {
        // The property the streaming path exists for: feeding the ids one by
        // one and concatenating the drained chunks must equal a single decode.
        // This is what pins the index bookkeeping around `step_decode_stream`.
        let tok = fixture();
        let h = Handle::new(tok);
        for text in [
            "Hello world",
            "你好，世界",
            "Hello 你好 world",
            "a\tb\nc",
            "",
            &"Hello world ".repeat(200),
        ] {
            let ids = encode(h.ptr(), text, 1 << 16);
            let expected = decode(h.ptr(), &ids, text.len() + 1);
            assert_eq!(expected, text, "whole-sequence decode must round-trip");

            let stream = unsafe { tokenizer_stream_new(h.ptr()) };
            let mut got = String::new();
            for id in ids.iter() {
                assert_eq!(unsafe { tokenizer_stream_feed(stream, *id) }, 0, "feed failed");
                let pending = unsafe { tokenizer_stream_pending(stream) };
                assert!(pending >= 1);
                // Drain with an undersized buffer first: a retry must not
                // advance the state or lose text.
                let (_, chunk) = drain(stream, pending as usize);
                got.push_str(&chunk);
            }
            got.push_str(&finish_text(stream));
            assert_eq!(got, expected, "streamed text differs for {text:?}");
            unsafe { tokenizer_stream_free(stream) };
        }
    }

    #[test]
    fn finish_flushes_the_incomplete_tail_once() {
        let tok = fixture();
        let truncated = byte_id(&tok, 0xE4);
        let h = Handle::new(tok);
        let stream = unsafe { tokenizer_stream_new(h.ptr()) };
        assert_eq!(unsafe { tokenizer_stream_feed(stream, truncated) }, 0);

        let mut buf = vec![0i8; 64];
        let written = unsafe { tokenizer_stream_finish(stream, buf.as_mut_ptr(), 64) };
        assert!(written > 0, "the truncated byte must surface at finish");
        let text =
            String::from_utf8(buf[..written as usize].iter().map(|b| *b as u8).collect()).unwrap();
        assert!(text.contains('\u{FFFD}'), "got {text:?}");

        // Idempotent: a second finish emits nothing and does not error.
        let written = unsafe { tokenizer_stream_finish(stream, buf.as_mut_ptr(), 64) };
        assert_eq!(written, 0);
        unsafe { tokenizer_stream_free(stream) };
    }

    #[test]
    fn reset_clears_the_state_and_allows_reuse() {
        let tok = fixture();
        let h = Handle::new(tok);
        let stream = unsafe { tokenizer_stream_new(h.ptr()) };
        for id in encode(h.ptr(), "Hello", 32) {
            unsafe { tokenizer_stream_feed(stream, id) };
        }
        assert_eq!(unsafe { tokenizer_stream_reset(stream) }, 0);
        assert_eq!(unsafe { tokenizer_stream_pending(stream) }, 1);

        let again = encode(h.ptr(), "Hello", 32);
        for id in again.iter() {
            assert_eq!(unsafe { tokenizer_stream_feed(stream, *id) }, 0);
        }
        let pending = unsafe { tokenizer_stream_pending(stream) };
        let (written, text) = drain(stream, pending as usize);
        assert_eq!(written, pending - 1);
        assert_eq!(text, "Hello");
        unsafe { tokenizer_stream_free(stream) };
    }

    #[test]
    fn null_handles_are_rejected_without_touching_memory() {
        unsafe {
            assert!(tokenizer_stream_new(ptr::null_mut()).is_null());
            assert_eq!(tokenizer_stream_feed(ptr::null_mut(), 1), -1);
            assert_eq!(tokenizer_stream_pending(ptr::null_mut()), -1);
            assert_eq!(tokenizer_stream_reset(ptr::null_mut()), -1);
            assert_eq!(tokenizer_vocab_size(ptr::null_mut()), 0);
            tokenizer_free(ptr::null_mut());
            tokenizer_stream_free(ptr::null_mut());
        }
    }
}
