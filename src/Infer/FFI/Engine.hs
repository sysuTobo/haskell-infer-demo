{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE CApiFFI #-}

-- | FFI bindings to the C engine API (csrc/include/engine.h).
--
-- The engine is created from a *descriptor*: a flat JSON document describing the
-- architecture (see "Infer.Descriptor"). Passing data instead of a hand-packed
-- struct keeps a single authoring point for the model layout and removes the
-- byte-offset marshalling that used to be duplicated across Haskell, C and the
-- Python tests.
--
-- All functions are synchronous and must be called from the same OS thread
-- that created the engine (CUDA context affinity).
module Infer.FFI.Engine
  ( -- * Engine handle
    EngineHandle
    -- * Lifecycle
  , engineCreate
  , engineDestroy
    -- * Inference
  , enginePrefill
  , engineDecode
  , engineReset
    -- * Queries
  , engineVocabSize
  , engineSeqLen
  , engineDescVersion
  , engineDescribe
    -- * Errors
  , engineLastError
    -- * Hello-world (Phase 1)
  , engineHelloGpu
  ) where

import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as BSC
import Data.Int (Int64)
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc (allocaBytes)
import Foreign.Marshal.Array (allocaArray, peekArray, withArray)
import Foreign.Ptr

-- | Opaque engine handle.
type EngineHandle = ()

-- -----------------------------------------------------------------------
-- Raw FFI imports
-- -----------------------------------------------------------------------

foreign import ccall unsafe "engine.h engine_create"
  c_engine_create :: CString -> CString -> CInt -> Ptr CInt -> Ptr CInt -> IO (Ptr EngineHandle)

foreign import ccall unsafe "engine.h engine_destroy"
  c_engine_destroy :: Ptr EngineHandle -> IO ()

foreign import ccall unsafe "engine.h engine_prefill"
  c_engine_prefill :: Ptr EngineHandle -> Ptr Int64 -> CInt -> Ptr CFloat -> IO CInt

foreign import ccall unsafe "engine.h engine_decode"
  c_engine_decode :: Ptr EngineHandle -> Int64 -> Ptr CFloat -> IO CInt

foreign import ccall unsafe "engine.h engine_reset"
  c_engine_reset :: Ptr EngineHandle -> IO ()

foreign import ccall unsafe "engine.h engine_vocab_size"
  c_engine_vocab_size :: Ptr EngineHandle -> IO CInt

foreign import ccall unsafe "engine.h engine_seq_len"
  c_engine_seq_len :: Ptr EngineHandle -> IO CInt

foreign import ccall unsafe "engine.h engine_desc_version"
  c_engine_desc_version :: IO CInt

foreign import ccall unsafe "engine.h engine_describe"
  c_engine_describe :: Ptr EngineHandle -> CString -> CInt -> IO CInt

foreign import ccall unsafe "engine.h engine_last_error"
  c_engine_last_error :: IO CString

foreign import ccall unsafe "engine.h engine_hello_gpu"
  c_engine_hello_gpu :: CInt -> CInt -> IO CInt

-- -----------------------------------------------------------------------
-- Haskell wrappers
-- -----------------------------------------------------------------------

-- | Create an engine from a model directory, a descriptor JSON document and a
-- layer placement (@devices@ plus a device ordinal per layer).
-- Returns 'Nothing' on failure (check 'engineLastError').
engineCreate :: FilePath -> ByteString -> [Int] -> [Int] -> IO (Maybe (Ptr EngineHandle))
engineCreate modelDir descriptorJson devices layerDevices =
  BSC.useAsCString descriptorJson $ \cDesc ->
    withCString modelDir $ \cDir ->
      withArray (map fromIntegral devices) $ \pDevices ->
        withArray (map fromIntegral layerDevices) $ \pLayerDevices -> do
          ptr <- c_engine_create cDir cDesc (fromIntegral (length devices)) pDevices pLayerDevices
          if ptr == nullPtr then return Nothing else return (Just ptr)

engineDestroy :: Ptr EngineHandle -> IO ()
engineDestroy = c_engine_destroy

-- | Prefill: process tokens and get logits for the last position.
-- @vocabSize@ must come from 'engineVocabSize'.
enginePrefill :: Ptr EngineHandle -> [Int64] -> Int -> IO (Either String [Float])
enginePrefill h tokens vocabSize =
  withArray tokens $ \pTokens ->
    allocaArray vocabSize $ \pLogits -> do
      rc <- c_engine_prefill h pTokens (fromIntegral (length tokens)) pLogits
      if rc /= 0
        then do
          err <- engineLastError
          return (Left err)
        else do
          logits <- peekArray vocabSize pLogits
          return (Right (map realToFrac logits))

-- | Decode: process one token and get logits.
engineDecode :: Ptr EngineHandle -> Int64 -> Int -> IO (Either String [Float])
engineDecode h token vocabSize =
  allocaArray vocabSize $ \pLogits -> do
    rc <- c_engine_decode h token pLogits
    if rc /= 0
      then do
        err <- engineLastError
        return (Left err)
      else do
        logits <- peekArray vocabSize pLogits
        return (Right (map realToFrac logits))

engineReset :: Ptr EngineHandle -> IO ()
engineReset = c_engine_reset

engineVocabSize :: Ptr EngineHandle -> IO Int
engineVocabSize h = fromIntegral <$> c_engine_vocab_size h

engineSeqLen :: Ptr EngineHandle -> IO Int
engineSeqLen h = fromIntegral <$> c_engine_seq_len h

-- | Descriptor wire-format version supported by the linked engine.
engineDescVersion :: IO Int
engineDescVersion = fromIntegral <$> c_engine_desc_version

-- | Round-trip the descriptor the engine actually parsed. Used to prove that C
-- and Haskell agree on the model layout.
engineDescribe :: Ptr EngineHandle -> IO (Either String ByteString)
engineDescribe h =
  allocaBytes bufSize $ \buf -> do
    written <- c_engine_describe h buf (fromIntegral bufSize)
    if written < 0
      then do
        err <- engineLastError
        return (Left err)
      else Right <$> BSC.packCStringLen (buf, fromIntegral written)
  where
    bufSize = 64 * 1024

engineLastError :: IO String
engineLastError = c_engine_last_error >>= peekCString

-- | Phase 1 FFI verification: round-trip an integer through the GPU.
engineHelloGpu :: Int -> Int -> IO Int
engineHelloGpu device value =
  fromIntegral <$> c_engine_hello_gpu (fromIntegral device) (fromIntegral value)
