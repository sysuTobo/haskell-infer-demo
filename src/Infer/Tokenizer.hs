{-# LANGUAGE ForeignFunctionInterface #-}

-- | Tokenizer FFI bindings to the Rust @tokenizer-ffi@ library.
--
-- Wraps HuggingFace tokenizers via a C ABI. The tokenizer is loaded from
-- a @tokenizer.json@ file in the model directory.
module Infer.Tokenizer
  ( Tokenizer
  , loadTokenizer
  , freeTokenizer
  , encode
  , decode
  , decodeSingle
  , vocabSize
  ) where

import Foreign.Ptr
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Data.Int (Int64)

-- | Opaque tokenizer handle.
type Tokenizer = Ptr ()

foreign import ccall unsafe "tokenizer_load"
  c_tokenizer_load :: CString -> IO Tokenizer

foreign import ccall unsafe "tokenizer_encode"
  c_tokenizer_encode :: Tokenizer -> CString -> Ptr Int64 -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_decode"
  c_tokenizer_decode :: Tokenizer -> Ptr Int64 -> CInt -> CString -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_decode_single"
  c_tokenizer_decode_single :: Tokenizer -> Int64 -> CString -> CInt -> IO CInt

foreign import ccall unsafe "tokenizer_vocab_size"
  c_tokenizer_vocab_size :: Tokenizer -> IO CInt

foreign import ccall unsafe "tokenizer_free"
  c_tokenizer_free :: Tokenizer -> IO ()

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
  withCString text $ \cText ->
    -- Allocate a generous buffer (max 8192 tokens for a single prompt)
    allocaArray 8192 $ \pIds -> do
      n <- c_tokenizer_encode tok cText pIds 8192
      if n < 0
        then return []
        else peekArray (fromIntegral n) pIds

-- | Decode token IDs back to text.
decode :: Tokenizer -> [Int64] -> IO String
decode tok ids =
  withArray ids $ \pIds ->
    allocaArray 4096 $ \pBuf -> do
      n <- c_tokenizer_decode tok pIds (fromIntegral (length ids)) pBuf 4096
      if n < 0
        then return ""
        else peekCString pBuf

-- | Decode a single token (for streaming output).
decodeSingle :: Tokenizer -> Int64 -> IO String
decodeSingle tok tid =
  allocaArray 256 $ \pBuf -> do
    n <- c_tokenizer_decode_single tok tid pBuf 256
    if n < 0
      then return ""
      else peekCString pBuf

-- | Get the vocabulary size.
vocabSize :: Tokenizer -> IO Int
vocabSize tok = fromIntegral <$> c_tokenizer_vocab_size tok
