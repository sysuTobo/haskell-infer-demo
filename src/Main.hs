-- | CLI entry point for the Haskell inference demo.
--
-- Usage:
--   haskell-infer-demo generate --model-dir /path/to/Qwen3.8-27B --gpus 0,1 -p "Hello"
--   haskell-infer-demo show-config --descriptor descriptors/qwen38-27b.json
--   haskell-infer-demo descriptor --model-dir /path/to/model [--write FILE]
--   haskell-infer-demo manifest --model-dir /path/to/model [--write FILE] [--check]
--   haskell-infer-demo manifest-compare OLD.json NEW.json [--mode strict|diagnostic]
--   haskell-infer-demo hello-gpu -d 0 -v 42
module Main (main) where

import Control.Exception (bracket)
import qualified Data.ByteString as BS
import Data.List (intercalate)
import qualified Data.Text as T
import Options.Applicative
import System.Exit (ExitCode(..), exitFailure, exitSuccess, exitWith)
import System.IO.Error (ioeGetErrorString, tryIOError)
import Text.Read (readMaybe)

import Infer.Config
import Infer.Descriptor
import Infer.FFI.Engine
import Infer.Generation
import Infer.Manifest
import Infer.Model
import Infer.Placement
import Infer.Runtime
import Infer.Tokenizer

-- -----------------------------------------------------------------------
-- CLI options
-- -----------------------------------------------------------------------

data Options = Options
  { optCommand :: Command
  }

data Command
  = Generate GenOptions
  | HelloGpu Int Int
  | ShowConfig ConfigSource [Int]
  | DumpDescriptor FilePath (Maybe FilePath)
  | ShowManifest ManifestOptions
  | CompareManifestFiles CompareOptions

-- | Where a descriptor comes from: an explicit JSON file, or a model directory
-- whose config.json is read by the family adapter.
data ConfigSource = FromFile FilePath | FromDir FilePath

data GenOptions = GenOptions
  { genModelDir  :: FilePath
  , genDesc      :: Maybe FilePath
  , genGpus      :: [Int]
  , genTp        :: Int
  , genEp        :: Int
  , genMaxSeqLen :: Int
  , genMaxTokens :: Int
  , genPrompt    :: String
  , genStreaming :: Bool
  , genCheckDesc :: Bool
  }

-- | The manifest has no request-level state, so there is no prompt or token
-- budget: it describes the execution a capture would be taken under.
data ManifestOptions = ManifestOptions
  { manModelDir  :: FilePath
  , manDesc      :: Maybe FilePath
  , manGpus      :: [Int]
  , manTp        :: Int
  , manEp        :: Int
  , manMaxSeqLen :: Int
  , manWrite     :: Maybe FilePath
  , manCheck     :: Bool
  }

data CompareOptions = CompareOptions
  { cmpLeft             :: FilePath
  , cmpRight            :: FilePath
  , cmpMode             :: String
  , cmpDeploymentScoped :: Bool
  }

optionsParser :: Parser Options
optionsParser = Options <$> hsubparser
  ( command "generate" (info generateCmd (progDesc "Generate text from a prompt"))
 <> command "hello-gpu" (info helloGpuCmd (progDesc "Phase 1: FFI verification"))
 <> command "show-config" (info showConfigCmd (progDesc "Print the model descriptor"))
 <> command "descriptor" (info descriptorCmd (progDesc "Dump the canonical descriptor for a model dir"))
 <> command "manifest" (info manifestCmd
      (progDesc "Print the canonical execution manifest (identities + provenance)"))
 <> command "manifest-compare" (info manifestCompareCmd
      (progDesc "Compare two manifest JSON documents: strict admission / diagnostic"))
  )

generateCmd :: Parser Command
generateCmd = Generate <$> (GenOptions
  <$> strOption
      ( long "model-dir"
     <> metavar "DIR"
     <> help "Path to model directory (containing safetensors + tokenizer.json)"
      )
  <*> optional (strOption
      ( long "descriptor"
     <> metavar "FILE"
     <> help "Descriptor JSON overriding the model directory's config.json"
      ))
  <*> option parseGpuList
      ( long "gpus"
     <> metavar "0,1,..."
     <> value [0, 1]
     <> showDefault
     <> help "Comma-separated CUDA device ordinals"
      )
  <*> option auto
      ( long "tp"
     <> metavar "N"
     <> value 1
     <> showDefault
     <> help "Tensor-parallel ranks: N > 1 keeps the whole model on N devices and loads weight shards"
      )
  <*> option auto
      ( long "ep"
     <> metavar "N"
     <> value 1
     <> showDefault
     <> help "Expert-parallel ranks: N > 1 splits the MoE experts across N devices"
      )
  <*> option auto
      ( long "max-seq-len"
     <> metavar "N"
     <> value 4096
     <> showDefault
     <> help "Maximum context length"
      )
  <*> option nonNegativeInt
      ( long "max-tokens"
     <> metavar "N"
     <> value 256
     <> showDefault
     <> help "Maximum new tokens to generate (>= 0; 0 generates nothing)"
      )
  <*> strOption
      ( long "prompt"
     <> short 'p'
     <> metavar "TEXT"
     <> help "Input prompt"
      )
  <*> switch
      ( long "stream"
     <> help "Stream tokens as they are generated"
      )
  <*> switch
      ( long "check-descriptor"
     <> help "Round-trip the descriptor through the engine and compare"
      ))

helloGpuCmd :: Parser Command
helloGpuCmd = HelloGpu
  <$> option auto
      ( long "device"
     <> short 'd'
     <> value 0
     <> showDefault
     <> help "CUDA device ordinal"
      )
  <*> option auto
      ( long "value"
     <> short 'v'
     <> value 42
     <> showDefault
     <> help "Integer to round-trip through GPU"
      )

showConfigCmd :: Parser Command
showConfigCmd = ShowConfig
  <$> configSource
  <*> option parseGpuList
      ( long "gpus"
     <> metavar "0,1,..."
     <> value [0, 1]
     <> showDefault
     <> help "Comma-separated CUDA device ordinals (for placement preview)"
      )

descriptorCmd :: Parser Command
descriptorCmd = DumpDescriptor
  <$> strOption
      ( long "model-dir"
     <> metavar "DIR"
     <> help "Model directory whose config.json is adapted"
      )
  <*> optional (strOption
      ( long "write"
     <> metavar "FILE"
     <> help "Write the canonical descriptor to FILE instead of stdout"
      ))

configSource :: Parser ConfigSource
configSource =
      (FromFile <$> strOption
        ( long "descriptor"
       <> metavar "FILE"
       <> help "Descriptor JSON (no model directory needed)"
        ))
  <|> (FromDir <$> strOption
        ( long "model-dir"
       <> metavar "DIR"
       <> help "Derive the descriptor from DIR/config.json"
        ))

manifestCmd :: Parser Command
manifestCmd = ShowManifest <$> (ManifestOptions
  <$> strOption
      ( long "model-dir"
     <> metavar "DIR"
     <> help "Path to model directory (required: the manifest reports the parameter identity)"
      )
  <*> optional (strOption
      ( long "descriptor"
     <> metavar "FILE"
     <> help "Descriptor JSON overriding the model directory's config.json"
      ))
  <*> option parseGpuList
      ( long "gpus"
     <> metavar "0,1,..."
     <> value [0, 1]
     <> showDefault
     <> help "Comma-separated CUDA device ordinals"
      )
  <*> option auto
      ( long "tp"
     <> metavar "N"
     <> value 1
     <> showDefault
     <> help "Tensor-parallel ranks (changes the numerical policy, not just the placement)"
      )
  <*> option auto
      ( long "ep"
     <> metavar "N"
     <> value 1
     <> showDefault
     <> help "Expert-parallel ranks"
      )
  <*> option auto
      ( long "max-seq-len"
     <> metavar "N"
     <> value 4096
     <> showDefault
     <> help "Maximum context length"
      )
  <*> optional (strOption
      ( long "write"
     <> metavar "FILE"
     <> help "Write the canonical manifest to FILE instead of stdout"
      ))
  <*> switch
      ( long "check"
     <> help "Re-query the manifest and verify it is byte-identical (and report unestablished facts)"
      ))

manifestCompareCmd :: Parser Command
manifestCompareCmd = CompareManifestFiles <$> (CompareOptions
  <$> argument str (metavar "LEFT")
  <*> argument str (metavar "RIGHT")
  <*> strOption
      ( long "mode"
     <> metavar "MODE"
     <> value "strict"
     <> showDefault
     <> help "'strict' requires the identities and provenance to agree; 'diagnostic' reports a deliberate difference without calling it a pass"
      )
  <*> switch
      ( long "deployment-scoped"
     <> help "Declare that a scoped invariance claim covers a placement-only difference"
      ))

-- | A token budget: the CLI rejects a negative count instead of handing it to
-- the generation loop (which treats <= 0 as "generate nothing").
nonNegativeInt :: ReadM Int
nonNegativeInt = eitherReader $ \s -> case readMaybe s :: Maybe Int of
  Nothing -> Left ("Invalid token budget: " ++ s)
  Just n
    | n < 0 -> Left ("max-tokens must be non-negative, got " ++ show n)
    | otherwise -> Right n

parseGpuList :: ReadM [Int]
parseGpuList = eitherReader $ \s ->
  let parts = splitOn ',' s
      parsed = mapM (readMaybe . trim) parts
  in case parsed of
       Just xs -> Right xs
       Nothing -> Left ("Invalid GPU list: " ++ s)

splitOn :: Char -> String -> [String]
splitOn c s = case break (== c) s of
  (before, []) -> [before]
  (before, _:after) -> before : splitOn c after

trim :: String -> String
trim = dropWhile (== ' ') . reverse . dropWhile (== ' ') . reverse

-- -----------------------------------------------------------------------
-- Commands
-- -----------------------------------------------------------------------

runHelloGpu :: Int -> Int -> IO ()
runHelloGpu device value = do
  putStrLn $ "Testing FFI: Haskell -> C -> CUDA (device " ++ show device ++ ")"
  result <- engineHelloGpu device value
  if result == value
    then do
      putStrLn $ "SUCCESS: GPU returned " ++ show result
      exitSuccess
    else do
      putStrLn $ "FAILED: expected " ++ show value ++ ", got " ++ show result
      err <- engineLastError
      putStrLn $ "Error: " ++ err
      exitFailure

-- | Print the descriptor and the placement it implies.
runShowConfig :: ConfigSource -> [Int] -> IO ()
runShowConfig source devices = do
  desc <- loadFor source
  putStrLn $ "Descriptor " ++ show (dVersion desc) ++ ": " ++ dFamily desc
    ++ " (" ++ dModelType desc ++ ")"
  putStrLn $ "  Layers:            " ++ show (dNumLayers desc)
  putStrLn $ "  Hidden size:       " ++ show (dHiddenSize desc)
  putStrLn $ "  Intermediate size: " ++ show (dIntermediateSize desc)
  putStrLn $ "  Vocab size:        " ++ show (dVocabSize desc)
  putStrLn $ "  Num heads:         " ++ show (dNumHeads desc)
  putStrLn $ "  Num KV heads:      " ++ show (dNumKvHeads desc)
  putStrLn $ "  Head dim:          " ++ show (dHeadDim desc)
  putStrLn $ "  Rotary dim:        " ++ show (dRotaryDim desc)
  putStrLn $ "  Output gate:       " ++ show (dAttnOutputGate desc)
  putStrLn $ "  GDN v-heads:       " ++ show (dGdnNumVHeads desc)
  putStrLn $ "  GDN head dim:      " ++ show (dGdnHeadDim desc)
  putStrLn $ "  EOS tokens:        " ++ show (dEosTokens desc)
  putStrLn $ "  Max seq len:       " ++ show (dMaxSeqLen desc)
  putStrLn ""
  putStrLn "Layer layout (A=full_attn, G=gdn, M=mla):"
  putStrLn $ "  " ++ intercalate " " (chunksOf 16 (map mixerLetter (dLayerMixers desc)))
  putStrLn ""
  case modelDef desc Pipelined devices of
    Left err -> putStrLn $ "Placement error: " ++ err
    Right md -> do
      putStrLn $ "Placement (" ++ show (plPolicy (mdPlacement md)) ++ "):"
      putStrLn $ "  Layers per device: " ++ show (plLayersPerDevice (mdPlacement md))
  where
    mixerLetter MFullAttention = 'A'
    mixerLetter MGatedDeltaNet = 'G'
    mixerLetter MMlaAttention = 'M'
    chunksOf _ [] = []
    chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- | Dump the canonical descriptor produced by the family adapter.
runDumpDescriptor :: FilePath -> Maybe FilePath -> IO ()
runDumpDescriptor modelDir writeTo = do
  desc <- loadFor (FromDir modelDir)
  let bytes = encodeDescriptor desc
  case writeTo of
    Nothing -> BS.putStr bytes >> putStrLn ""
    Just path -> do
      BS.writeFile path bytes
      putStrLn $ "Wrote " ++ path ++ " (" ++ show (BS.length bytes) ++ " bytes)"

loadFor :: ConfigSource -> IO Descriptor
loadFor source = loadDescriptor $ case source of
  FromFile path -> defaultRuntimeConfig { rcDescriptor = Just path }
  FromDir dir -> defaultRuntimeConfig { rcModelDir = dir }

-- | Print the execution manifest for this model, device set and placement.
--
-- Unlike the descriptor (which is portable architecture) the manifest describes
-- *this* execution, so it needs a load: the parameter identity and the runtime
-- provenance come from the engine, and the build provenance from the header the
-- build generated.
runShowManifest :: ManifestOptions -> IO ()
runShowManifest opts = do
  let cfg = defaultRuntimeConfig
        { rcModelDir   = manModelDir opts
        , rcDescriptor = manDesc opts
        , rcDevices    = manGpus opts
        , rcTp         = manTp opts
        , rcEp         = manEp opts
        , rcMaxSeqLen  = manMaxSeqLen opts
        }
  bracket (initRuntime cfg) shutdownRuntime $ \rt -> do
    let engine = runtimeEngine rt
    queried <- engineManifest engine
    text <- case queried of
      Left err -> do
        putStrLn $ "ERROR: engine_manifest failed: " ++ err
        exitFailure
      Right bytes -> return bytes
    case parseManifest text of
      Left err -> do
        putStrLn $ "ERROR: the engine's manifest does not parse: " ++ err
        exitFailure
      Right manifest -> do
        mapM_ putStrLn (manifestSummary manifest)
        let unestablished = unestablishedPaths manifest
        if null unestablished
          then putStrLn "provenance: fully established"
          else do
            putStrLn $ "provenance: NOT established for " ++ show (length unestablished)
                     ++ " field(s): " ++ intercalate ", " (map T.unpack unestablished)
            putStrLn "  (strict admission will refuse this manifest until they are resolved)"
        if manCheck opts
          then do
            again <- engineManifest engine
            case again of
              Right second
                | second == text ->
                    putStrLn "Manifest stability: OK (two queries are byte-identical)"
                | otherwise -> do
                    putStrLn "Manifest stability: MISMATCH (the manifest is not deterministic)"
                    exitFailure
              Left err -> do
                putStrLn $ "ERROR: engine_manifest failed on the second query: " ++ err
                exitFailure
          else return ()
    case manWrite opts of
      Nothing -> BS.putStr text >> putStrLn ""
      Just path -> do
        BS.writeFile path text
        putStrLn $ "Wrote " ++ path ++ " (" ++ show (BS.length text) ++ " bytes)"

-- | Compare two manifest documents under one of the contract's modes.
--
-- Exit codes: 0 admitted (strict, or with a declared deployment scope), 1
-- rejected, 2 legacy/unverified, 3 diagnostic-only. A legacy or diagnostic
-- result is deliberately not 0, so neither can be mistaken for a contract pass.
runCompareManifestFiles :: CompareOptions -> IO ()
runCompareManifestFiles opts = do
  mode <- case parseCompareMode (cmpMode opts) of
    Left err -> putStrLn ("ERROR: " ++ err) >> exitFailure
    Right parsed -> return parsed
  left <- readDocument (cmpLeft opts)
  right <- readDocument (cmpRight opts)
  case compareDocuments mode (cmpDeploymentScoped opts) left right of
    Left err -> do
      putStrLn $ "ERROR: " ++ err
      exitFailure
    Right (verdict, _) -> do
      putStrLn $ "mode: " ++ compareModeName mode
              ++ ", deployment scope declared: " ++ show (cmpDeploymentScoped opts)
      putStrLn $ "left:  " ++ cmpLeft opts ++ "  (" ++ documentLabel left ++ ")"
      putStrLn $ "right: " ++ cmpRight opts ++ "  (" ++ documentLabel right ++ ")"
      putStrLn $ "verdict: " ++ verdictKind verdict
      putStrLn (renderVerdict verdict)
      case verdictExitCode verdict of
        0 -> exitSuccess
        code -> exitWith (ExitFailure code)
  where
    readDocument path = do
      outcome <- tryIOError (BS.readFile path)
      case outcome of
        Left err -> do
          putStrLn $ "ERROR: cannot read " ++ path ++ ": " ++ ioeGetErrorString err
          exitFailure
        Right bytes -> return bytes
    documentLabel bytes = case parseManifest bytes of
      Left _ -> "no manifest version"
      Right manifest -> "semantic " ++ T.unpack (ibSemanticId (mIdentities manifest))
                        ++ ", numerical " ++ T.unpack (ibNumericalPolicyId (mIdentities manifest))

runGenerate :: GenOptions -> IO ()
runGenerate opts = do
  let cfg = RuntimeConfig
        { rcModelDir  = genModelDir opts
        , rcDescriptor = genDesc opts
        , rcDevices   = genGpus opts
        , rcTp        = genTp opts
        , rcEp        = genEp opts
        , rcMaxSeqLen = genMaxSeqLen opts
        , rcMaxTokens = genMaxTokens opts
        , rcPrompt    = genPrompt opts
        }

  -- bracket: the tokenizer and the engine are released on every exit path,
  -- including a generation failure or an output error.
  bracket (initRuntime cfg) shutdownRuntime $ \rt -> do
    let engine = runtimeEngine rt
        tok = rtTokenizer rt
        vocab = rtVocabSize rt
        desc = runtimeDescriptor rt

    if genCheckDesc opts
      then do
        echoed <- engineDescribe engine
        case echoed of
          Left err -> do
            putStrLn $ "ERROR: engine_describe failed: " ++ err
            exitFailure
          Right text -> case decodeDescriptor text of
            Left err -> do
              putStrLn $ "ERROR: engine descriptor echo is not decodable: " ++ err
              exitFailure
            Right echoedDesc ->
              if echoedDesc == desc
                then putStrLn "Descriptor round-trip: OK"
                else do
                  putStrLn "Descriptor round-trip: MISMATCH"
                  putStrLn $ "  sent: " ++ show desc
                  putStrLn $ "  echo: " ++ show echoedDesc
                  exitFailure
      else return ()

    putStrLn "Tokenizing prompt..."
    promptTokens <- encode tok (genPrompt opts)
    putStrLn $ "  " ++ show (length promptTokens) ++ " tokens"

    if length promptTokens > genMaxSeqLen opts
      then do
        putStrLn "ERROR: prompt exceeds max-seq-len"
        exitFailure
      else return ()

    putStrLn $ "Generating (max " ++ show (genMaxTokens opts) ++ " tokens)..."
    putStrLn "---"
    -- An engine failure is reported as a failed run, never as a short output
    -- that exits 0.
    outcome <- tryIOError $ if genStreaming opts
      then bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming engine vocab (dEosTokens desc) stream promptTokens (genMaxTokens opts)
      else do
        toks <- generate engine vocab (dEosTokens desc) promptTokens (genMaxTokens opts)
        text <- decode tok toks
        putStrLn text
        return toks

    outputTokens <- case outcome of
      Left err -> do
        putStrLn $ "ERROR: generation failed: " ++ ioeGetErrorString err
        exitFailure
      Right toks -> return toks

    putStrLn "---"
    putStrLn $ "Generated " ++ show (length outputTokens) ++ " tokens"

-- -----------------------------------------------------------------------
-- Main
-- -----------------------------------------------------------------------

main :: IO ()
main = do
  opts <- execParser optsInfo
  case optCommand opts of
    HelloGpu device value -> runHelloGpu device value
    ShowConfig source devices -> runShowConfig source devices
    DumpDescriptor dir writeTo -> runDumpDescriptor dir writeTo
    ShowManifest manifestOpts -> runShowManifest manifestOpts
    CompareManifestFiles compareOpts -> runCompareManifestFiles compareOpts
    Generate genOpts -> runGenerate genOpts
  where
    optsInfo = info (optionsParser <**> helper)
      ( fullDesc
     <> progDesc "Haskell inference framework demo for Qwen3.8-27B"
     <> header "haskell-infer-demo - a Haskell LLM inference engine"
      )
