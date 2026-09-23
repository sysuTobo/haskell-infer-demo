-- | Runtime orchestration: descriptor loading, placement, engine lifecycle.
module Infer.Runtime
  ( Runtime(..)
  , initRuntime
  , withRuntimeResources
  , shutdownRuntime
  , loadDescriptor
  , runtimeDescriptor
  , runtimeEngine
  , runtimeVocabSize
  ) where

import qualified Data.ByteString as BS
import Control.Exception (onException)
import Foreign.Ptr
import System.Exit (exitFailure)

import Infer.Config
import Infer.Descriptor
import Infer.Descriptor.Adapter
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
    adapted <- descriptorFromModelDir (rcModelDir cfg)
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
  base <- loadDescriptor cfg
  (policy, desc) <- case resolveTopology (rcTp cfg) (rcEp cfg) base of
    Left err -> do
      putStrLn $ "ERROR: " ++ err
      exitFailure
    Right resolved -> return resolved
  putStrLn $ "Descriptor: " ++ dFamily desc ++ " (" ++ dModelType desc ++ "), "
    ++ show (dNumLayers desc) ++ " layers, hidden " ++ show (dHiddenSize desc)
    ++ ", vocab " ++ show (dVocabSize desc)

  model <- case modelDef desc policy (rcDevices cfg) of
    Left err -> do
      putStrLn $ "ERROR: " ++ err
      exitFailure
    Right md -> return md
  putStrLn $ "  Placement (" ++ show (plPolicy (mdPlacement model)) ++ "): "
    ++ show (plLayersPerDevice (mdPlacement model))

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
      -- The tokenizer is already live; release it before bailing out so a
      -- failed engine brings the whole runtime down without a leak.
      freeTokenizer tok
      exitFailure
    Just e -> return e
  -- Past this point the tokenizer and the engine are both live but the Runtime
  -- does not exist yet: if anything throws before it is returned, the caller's
  -- bracket never sees an acquisition, so release them here rather than leaking.
  withRuntimeResources engine tok $ do
    vocab <- engineVocabSize engine
    putStrLn "Engine initialized successfully."
    return Runtime
      { rtEngine = engine
      , rtTokenizer = tok
      , rtModel = model
      , rtConfig = cfg
      , rtVocabSize = vocab
      }

-- | Run the last step of initialization, releasing the engine and its tokenizer
-- if it throws instead of returning. The caller's 'bracket' only releases after a
-- /successful/ acquisition, so the window between \"engine created\" and
-- \"Runtime returned\" needs its own cleanup. Exported so the CPU suite can pin
-- that boundary with the stub engine.
withRuntimeResources :: Ptr EngineHandle -> Tokenizer -> IO a -> IO a
withRuntimeResources engine tok action =
  action `onException` (engineDestroy engine >> freeTokenizer tok)

-- | Shut down the runtime: destroy engine, free tokenizer.
shutdownRuntime :: Runtime -> IO ()
shutdownRuntime rt = do
  engineDestroy (rtEngine rt)
  freeTokenizer (rtTokenizer rt)
