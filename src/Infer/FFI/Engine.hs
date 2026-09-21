{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE CApiFFI #-}

-- | FFI bindings to the C engine API (csrc/include/engine.h).
--
-- All functions are synchronous and must be called from the same OS thread
-- that created the engine (CUDA context affinity). Use
-- @System.Posix.Thread.runInBoundThread@ if calling from Haskell threads.
module Infer.FFI.Engine
  ( -- * Engine handle
    EngineHandle
  , EngineConfig(..)
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
    -- * Errors
  , engineLastError
    -- * Hello-world (Phase 1)
  , engineHelloGpu
  ) where

import Foreign.Ptr
import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Array
import Foreign.Marshal.Alloc
import Foreign.Storable
import Data.Int (Int64)

-- | Opaque engine handle.
type EngineHandle = ()

-- | Engine configuration mirroring the C @EngineConfig@ struct.
data EngineConfig = EngineConfig
  { ecNumLayers      :: !Int
  , ecNumDevices     :: !Int
  , ecDevices        :: [Int]   -- ^ Device ordinals
  , ecLayerDevices   :: [Int]   -- ^ Per-layer device assignment
  , ecMaxSeqLen      :: !Int
  }

-- -----------------------------------------------------------------------
-- Raw FFI imports
-- -----------------------------------------------------------------------

foreign import ccall unsafe "engine.h engine_create"
  c_engine_create :: CString -> Ptr () -> IO (Ptr EngineHandle)

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

foreign import ccall unsafe "engine.h engine_last_error"
  c_engine_last_error :: IO CString

foreign import ccall unsafe "engine.h engine_hello_gpu"
  c_engine_hello_gpu :: CInt -> CInt -> IO CInt

-- -----------------------------------------------------------------------
-- Haskell wrappers
-- -----------------------------------------------------------------------

-- | Create an engine. Returns 'Nothing' on failure (check 'engineLastError').
engineCreate :: FilePath -> EngineConfig -> IO (Maybe (Ptr EngineHandle))
engineCreate modelDir cfg =
  withCString modelDir $ \cDir ->
    -- Allocate the C EngineConfig struct
    -- struct layout: { int num_layers; int num_devices; const int *devices;
    --                  const int *layer_devices; int max_seq_len; }
    allocaBytes (5 * sizeOf (0 :: CInt) + 2 * sizeOf (nullPtr :: Ptr CInt)) $ \pCfg -> do
      -- We need to marshal the config struct manually
      withArray (map fromIntegral (ecDevices cfg) :: [CInt]) $ \pDevices ->
        withArray (map fromIntegral (ecLayerDevices cfg) :: [CInt]) $ \pLayerDev -> do
          pokeByteOff pCfg 0 (fromIntegral (ecNumLayers cfg) :: CInt)
          pokeByteOff pCfg 4 (fromIntegral (ecNumDevices cfg) :: CInt)
          pokeByteOff pCfg 8 pDevices
          pokeByteOff pCfg (8 + sizeOf (nullPtr :: Ptr CInt)) pLayerDev
          pokeByteOff pCfg (8 + 2 * sizeOf (nullPtr :: Ptr CInt))
            (fromIntegral (ecMaxSeqLen cfg) :: CInt)
          ptr <- c_engine_create cDir pCfg
          if ptr == nullPtr then return Nothing else return (Just ptr)

engineDestroy :: Ptr EngineHandle -> IO ()
engineDestroy = c_engine_destroy

-- | Prefill: process tokens and get logits for the last position.
-- The output logits are returned as a list of Floats (length = vocab_size).
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

engineLastError :: IO String
engineLastError = c_engine_last_error >>= peekCString

-- | Phase 1 FFI verification: round-trip an integer through the GPU.
engineHelloGpu :: Int -> Int -> IO Int
engineHelloGpu device value =
  fromIntegral <$> c_engine_hello_gpu (fromIntegral device) (fromIntegral value)
