-- | Runtime orchestration: engine lifecycle, weight loading, multi-GPU setup.
--
-- This module bridges the Haskell model definition with the C engine API.
-- It computes the GPU partition, marshals configuration to C, and manages
-- the engine handle lifecycle.
module Infer.Runtime
  ( Runtime(..)
  , initRuntime
  , shutdownRuntime
  , runtimeModelConfig
  , runtimeEngine
  ) where

import Foreign.Ptr
import System.Exit (exitFailure)

import Infer.Config
import Infer.Model
import Infer.FFI.Engine
import Infer.Tokenizer

-- | Runtime state: holds the engine handle, tokenizer, and model definition.
data Runtime = Runtime
  { rtEngine   :: !(Ptr EngineHandle)
  , rtTokenizer :: !Tokenizer
  , rtModel    :: !ModelDef
  , rtConfig   :: !RuntimeConfig
  }

runtimeModelConfig :: Runtime -> ModelConfig
runtimeModelConfig = mdConfig . rtModel

runtimeEngine :: Runtime -> Ptr EngineHandle
runtimeEngine = rtEngine

-- | Initialize the runtime: load tokenizer, create engine, load weights.
initRuntime :: RuntimeConfig -> IO Runtime
initRuntime cfg = do
  let modelDef = qwen38_27bModel (rcDevices cfg)
      mc = mdConfig modelDef

  -- Load tokenizer
  putStrLn $ "Loading tokenizer from " ++ rcModelDir cfg ++ "/tokenizer.json"
  mTok <- loadTokenizer (rcModelDir cfg ++ "/tokenizer.json")
  tok <- case mTok of
    Nothing -> do
      putStrLn "ERROR: Failed to load tokenizer.json"
      exitFailure
    Just t -> return t

  -- Build engine config
  let partition = mdPartition modelDef
      engineCfg = EngineConfig
        { ecNumLayers    = mcNumLayers mc
        , ecNumDevices   = length (rcDevices cfg)
        , ecDevices      = gpDevices partition
        , ecLayerDevices = gpLayerDevices partition
        , ecMaxSeqLen    = rcMaxSeqLen cfg
        }

  -- Create engine (loads weights, allocates GPU memory)
  putStrLn $ "Initializing engine on " ++ show (length (rcDevices cfg)) ++ " GPU(s)..."
  putStrLn $ "  Layer partition: " ++ show (gpLayersPerDev partition)
  mEngine <- engineCreate (rcModelDir cfg) engineCfg
  engine <- case mEngine of
    Nothing -> do
      err <- engineLastError
      putStrLn $ "ERROR: engine_create failed: " ++ err
      exitFailure
    Just e -> return e

  putStrLn "Engine initialized successfully."
  return Runtime
    { rtEngine    = engine
    , rtTokenizer = tok
    , rtModel     = modelDef
    , rtConfig    = cfg
    }

-- | Shut down the runtime: destroy engine, free tokenizer.
shutdownRuntime :: Runtime -> IO ()
shutdownRuntime rt = do
  engineDestroy (rtEngine rt)
  freeTokenizer (rtTokenizer rt)
