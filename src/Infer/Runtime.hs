-- | Runtime orchestration: descriptor loading, placement, engine lifecycle.
module Infer.Runtime
  ( Runtime(..)
  , initRuntime
  , shutdownRuntime
  , loadDescriptor
  , runtimeDescriptor
  , runtimeEngine
  , runtimeVocabSize
  ) where

import qualified Data.ByteString as BS
import Foreign.Ptr
import System.Exit (exitFailure)

import Infer.Config
import Infer.Descriptor
import Infer.Descriptor.Adapter.Qwen35
import Infer.FFI.Engine
import Infer.Model
import Infer.Placement
import Infer.Tokenizer

-- | Runtime state: engine handle, tokenizer, model definition, vocabulary size.
data Runtime = Runtime
  { rtEngine :: !(Ptr EngineHandle)
  , rtTokenizer :: !Tokenizer
  , rtModel :: !ModelDef
  , rtConfig :: !RuntimeConfig
  , rtVocabSize :: !Int
  }

runtimeDescriptor :: Runtime -> Descriptor
runtimeDescriptor = mdDescriptor . rtModel

runtimeEngine :: Runtime -> Ptr EngineHandle
runtimeEngine = rtEngine

runtimeVocabSize :: Runtime -> Int
runtimeVocabSize = rtVocabSize

-- | Load the descriptor for a model: an explicit descriptor file wins, otherwise
-- the family adapter derives it from the model directory's @config.json@.
loadDescriptor :: RuntimeConfig -> IO Descriptor
loadDescriptor cfg = case rcDescriptor cfg of
  Just path -> do
    result <- decodeDescriptor <$> BS.readFile path
    finish result
  Nothing -> do
    adapted <- qwen35DescriptorFromDir (rcModelDir cfg)
    finish (adapted >>= validated)
  where
    validated desc = validateDescriptor desc >> pure desc
    finish (Left err) = do
      putStrLn $ "ERROR: cannot load model descriptor: " ++ err
      exitFailure
    finish (Right desc) = pure (withMaxSeqLen (rcMaxSeqLen cfg) desc)

-- | Initialize the runtime: load descriptor, tokenizer, engine (weights + state).
initRuntime :: RuntimeConfig -> IO Runtime
initRuntime cfg = do
  desc <- loadDescriptor cfg
  putStrLn $ "Descriptor: " ++ dFamily desc ++ " (" ++ dModelType desc ++ "), "
    ++ show (dNumLayers desc) ++ " layers, hidden " ++ show (dHiddenSize desc)
    ++ ", vocab " ++ show (dVocabSize desc)

  model <- case modelDef desc Pipelined (rcDevices cfg) of
    Left err -> do
      putStrLn $ "ERROR: " ++ err
      exitFailure
    Right md -> return md
  putStrLn $ "  Layers per device: " ++ show (plLayersPerDevice (mdPlacement model))

  putStrLn $ "Loading tokenizer from " ++ rcModelDir cfg ++ "/tokenizer.json"
  mTok <- loadTokenizer (rcModelDir cfg ++ "/tokenizer.json")
  tok <- case mTok of
    Nothing -> do
      putStrLn "ERROR: Failed to load tokenizer.json"
      exitFailure
    Just t -> return t

  putStrLn $ "Initializing engine on " ++ show (length (rcDevices cfg)) ++ " GPU(s)..."
  mEngine <- engineCreate (rcModelDir cfg) (encodeDescriptor desc)
    (plDevices (mdPlacement model)) (plLayerDevices (mdPlacement model))
  engine <- case mEngine of
    Nothing -> do
      err <- engineLastError
      putStrLn $ "ERROR: engine_create failed: " ++ err
      exitFailure
    Just e -> return e
  vocab <- engineVocabSize engine

  putStrLn "Engine initialized successfully."
  return Runtime
    { rtEngine = engine
    , rtTokenizer = tok
    , rtModel = model
    , rtConfig = cfg
    , rtVocabSize = vocab
    }

-- | Shut down the runtime: destroy engine, free tokenizer.
shutdownRuntime :: Runtime -> IO ()
shutdownRuntime rt = do
  engineDestroy (rtEngine rt)
  freeTokenizer (rtTokenizer rt)
