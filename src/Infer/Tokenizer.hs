{-# LANGUAGE ForeignFunctionInterface #-}

-- | Tokenizer FFI bindings to the Rust @tokenizer-ffi@ library.
--
-- Wraps HuggingFace tokenizers via a C ABI. The tokenizer is loaded from
-- a @tokenizer.json@ file in the model directory.
--
-- Every conversion is sized from the library's length query and the buffers are
-- allocated to that size, so a long prompt or a long generation is never
-- silently truncated. Text crosses the boundary as explicit UTF-8: the locale
-- encoding is irrelevant, and a failure is reported instead of being flattened
-- into an empty result.
module Infer.Tokenizer
  ( Tokenizer
  , loadTokenizer
  , freeTokenizer
  , encode
  , decode
  , vocabSize
    -- * Incremental decoding
  , DecodeStream
  , newDecodeStream
  , freeDecodeStream
  , feedToken
  , pendingLength
  , drainStream
  , finishStream
  , resetStream
  ) where

import Control.Exception (throwIO)
import qualified Data.ByteString as BS
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Int (Int64)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Marshal.Array (allocaArray, peekArray, withArray)
import Foreign.Ptr

-- | Opaque tokenizer handle.
type Tokenizer = Ptr ()

-- | Opaque incremental decode handle (see 'newDecodeStream').
type DecodeStream = Ptr ()

foreign import ccall unsafe "tokenizer_load"
  c_tokenizer_load :: CString -> IO Tokenizer

foreign import ccall unsafe "tokenizer_free"
  c_tokenizer_free :: Tokenizer -> IO ()

foreign import ccall unsafe "tokenizer_vocab_size"
  c_tokenizer_vocab_size :: Tokenizer -> IO CInt

foreign import ccall unsafe "tokenizer_encode_len"
  c_tokenizer_encode_len :: Tokenizer -> CString -> IO CInt

foreign import ccall unsafe "tokenizer_encode"
  c_tokenizer_encode :: Tokenizer -> CString -> Ptr Int64 -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_decode_len"
  c_tokenizer_decode_len :: Tokenizer -> Ptr Int64 -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_decode"
  c_tokenizer_decode :: Tokenizer -> Ptr Int64 -> CInt -> CString -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_stream_new"
  c_tokenizer_stream_new :: Tokenizer -> IO DecodeStream

foreign import ccall unsafe "tokenizer_stream_free"
  c_tokenizer_stream_free :: DecodeStream -> IO ()

foreign import ccall unsafe "tokenizer_stream_reset"
  c_tokenizer_stream_reset :: DecodeStream -> IO CInt

foreign import ccall unsafe "tokenizer_stream_feed"
  c_tokenizer_stream_feed :: DecodeStream -> Int64 -> IO CInt

foreign import ccall unsafe "tokenizer_stream_pending"
  c_tokenizer_stream_pending :: DecodeStream -> IO CInt

foreign import ccall unsafe "tokenizer_stream_drain"
  c_tokenizer_stream_drain :: DecodeStream -> CString -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_stream_finish"
  c_tokenizer_stream_finish :: DecodeStream -> CString -> CInt -> IO CInt

-- | The Rust side reports a buffer that is too small as -2; every other
-- negative value is a real failure.
capacityInsufficient :: CInt
capacityInsufficient = -2

-- | Load a tokenizer from a @tokenizer.json@ file path.
loadTokenizer :: FilePath -> IO (Maybe Tokenizer)
loadTokenizer path =
  withCString path $ \cPath -> do
    tok <- c_tokenizer_load cPath
    if tok == nullPtr then return Nothing else return (Just tok)

freeTokenizer :: Tokenizer -> IO ()
freeTokenizer = c_tokenizer_free

-- | Encode text to token IDs.
encode :: Tokenizer -> String -> IO [Int64]
encode tok text =
  withCStringUtf8 text $ \cText -> do
    required <- c_tokenizer_encode_len tok cText
    if required < 0
      then throwIO (userError "tokenizer_encode_len failed (invalid handle or input)")
      else allocaArray (max 1 (fromIntegral required)) $ \pIds -> do
        written <- c_tokenizer_encode tok cText pIds required
        if written < 0
          then throwIO (userError ("tokenizer_encode failed with status " ++ show written))
          else peekArray (fromIntegral written) pIds

-- | Decode token IDs back to text.
decode :: Tokenizer -> [Int64] -> IO String
decode tok ids =
  withArray ids $ \pIds -> do
    let len = fromIntegral (length ids)
    required <- c_tokenizer_decode_len tok pIds len
    if required < 0
      then throwIO (userError "tokenizer_decode_len failed (invalid handle or input)")
      else allocaBytes (fromIntegral required) $ \pBuf -> do
        written <- c_tokenizer_decode tok pIds len pBuf required
        decodeWritten written pBuf

-- | Get the vocabulary size.
vocabSize :: Tokenizer -> IO Int
vocabSize tok = fromIntegral <$> c_tokenizer_vocab_size tok

-- ---------------------------------------------------------------------------
-- Incremental decoding
-- ---------------------------------------------------------------------------

-- | Start an incremental decode session. The session owns its own decode state,
-- so several generations can stream independently and the handle outlives the
-- scope of any single token.
newDecodeStream :: Tokenizer -> IO DecodeStream
newDecodeStream tok = do
  stream <- c_tokenizer_stream_new tok
  if stream == nullPtr
    then throwIO (userError "tokenizer_stream_new failed")
    else return stream

freeDecodeStream :: DecodeStream -> IO ()
freeDecodeStream = c_tokenizer_stream_free

-- | Advance the stream by exactly one token. The text it unlocks is buffered;
-- read it with 'pendingLength' / 'drainStream'. Feeding twice would advance the
-- state twice, so callers query and drain without re-feeding.
feedToken :: DecodeStream -> Int64 -> IO ()
feedToken stream tid = do
  status <- c_tokenizer_stream_feed stream tid
  if status /= 0
    then throwIO (userError ("tokenizer_stream_feed failed with status " ++ show status))
    else return ()

-- | Bytes (including the NUL) the pending output needs. A zero-length result is
-- legal and reports 1.
pendingLength :: DecodeStream -> IO Int
pendingLength stream = do
  pending <- c_tokenizer_stream_pending stream
  if pending < 0
    then throwIO (userError "tokenizer_stream_pending failed")
    else return (fromIntegral pending)

-- | Take the buffered text out of the stream (empty when an incomplete
-- character is still being assembled).
drainStream :: DecodeStream -> IO String
drainStream stream = do
  required <- pendingLength stream
  allocaBytes required $ \pBuf -> do
    written <- c_tokenizer_stream_drain stream pBuf (fromIntegral required)
    decodeWritten written pBuf

-- | Flush the stream's incomplete tail and take it (empty when nothing is left).
finishStream :: DecodeStream -> IO String
finishStream stream = do
  -- First flush with no room: on a capacity error the tail is already buffered,
  -- so the length query below reports the size to retry with.
  status <- c_tokenizer_stream_finish stream nullPtr 0
  if status /= 0 && status /= capacityInsufficient
    then throwIO (userError ("tokenizer_stream_finish failed with status " ++ show status))
    else do
      required <- pendingLength stream
      allocaBytes required $ \pBuf -> do
        written <- c_tokenizer_stream_finish stream pBuf (fromIntegral required)
        decodeWritten written pBuf

-- | Drop all decode state so the stream can be reused for a new sequence.
resetStream :: DecodeStream -> IO ()
resetStream stream = do
  status <- c_tokenizer_stream_reset stream
  if status /= 0
    then throwIO (userError "tokenizer_stream_reset failed")
    else return ()

-- | Turn a length-checked C return value into the text it wrote.
decodeWritten :: CInt -> CString -> IO String
decodeWritten written pBuf
  | written < 0 = throwIO (userError ("tokenizer decode failed with status " ++ show written))
  | otherwise = do
      bytes <- BS.packCStringLen (pBuf, fromIntegral written)
      -- The library only ever emits UTF-8; a decode failure here means the
      -- bytes are not text, which must surface rather than be mangled.
      case TE.decodeUtf8' bytes of
        Left err -> throwIO (userError ("tokenizer produced invalid UTF-8: " ++ show err))
        Right text -> return (T.unpack text)

-- | Pass the UTF-8 encoding of a 'String' as a NUL-terminated C string. The
-- locale encoding is deliberately not used: the engine vocabulary is UTF-8.
withCStringUtf8 :: String -> (CString -> IO a) -> IO a
withCStringUtf8 text = BS.useAsCString (TE.encodeUtf8 (T.pack text))
