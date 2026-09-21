-- | CLI entry point for the Haskell inference demo.
--
-- Usage:
--   haskell-infer-demo --model-dir /path/to/Qwen3.8-27B --gpus 0,1 --prompt "Hello"
--   haskell-infer-demo --hello-gpu 0    # Phase 1 FFI verification
module Main (main) where

import Data.Int (Int64)
import Data.List (intercalate)
import Data.Maybe (fromMaybe)
import Options.Applicative
import System.Exit (exitFailure, exitSuccess)
import Text.Read (readMaybe)

import Infer.Config
import Infer.FFI.Engine
import Infer.Generation
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
  | ShowConfig

data GenOptions = GenOptions
  { genModelDir  :: FilePath
  , genGpus      :: [Int]
  , genMaxSeqLen :: Int
  , genMaxTokens :: Int
  , genPrompt    :: String
  , genStreaming :: Bool
  }

optionsParser :: Parser Options
optionsParser = Options <$> hsubparser
  ( command "generate" (info generateCmd (progDesc "Generate text from a prompt"))
 <> command "hello-gpu" (info helloGpuCmd (progDesc "Phase 1: FFI verification"))
 <> command "show-config" (info showConfigCmd (progDesc "Print model configuration"))
  )

generateCmd :: Parser Command
generateCmd = Generate <$> (GenOptions
  <$> strOption
      ( long "model-dir"
     <> metavar "DIR"
     <> help "Path to model directory (containing safetensors + tokenizer.json)"
      )
  <*> option parseGpuList
      ( long "gpus"
     <> metavar "0,1,..."
     <> value [0, 1]
     <> showDefault
     <> help "Comma-separated CUDA device ordinals"
      )
  <*> option auto
      ( long "max-seq-len"
     <> metavar "N"
     <> value 4096
     <> showDefault
     <> help "Maximum context length"
      )
  <*> option auto
      ( long "max-tokens"
     <> metavar "N"
     <> value 256
     <> showDefault
     <> help "Maximum new tokens to generate"
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
showConfigCmd = pure ShowConfig

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

runShowConfig :: IO ()
runShowConfig = do
  let cfg = qwen38_27bConfig
  putStrLn "Qwen3.8-27B Configuration:"
  putStrLn $ "  Layers:            " ++ show (mcNumLayers cfg)
  putStrLn $ "  Hidden size:       " ++ show (mcHiddenSize cfg)
  putStrLn $ "  Intermediate size: " ++ show (mcIntermediateSize cfg)
  putStrLn $ "  Vocab size:        " ++ show (mcVocabSize cfg)
  putStrLn $ "  Num heads:         " ++ show (mcNumHeads cfg)
  putStrLn $ "  Num KV heads:      " ++ show (mcNumKvHeads cfg)
  putStrLn $ "  Head dim:          " ++ show (mcHeadDim cfg)
  putStrLn $ "  Rotary dim:        " ++ show (mcRotaryDim cfg)
  putStrLn $ "  Full attn interval:" ++ show (mcFullAttnInterval cfg)
  putStrLn $ "  GDN v-heads:       " ++ show (mcGdnNumVHeads cfg)
  putStrLn $ "  GDN head dim:      " ++ show (mcGdnHeadDim cfg)
  putStrLn $ "  EOS tokens:        " ++ show (mcEosTokens cfg)
  putStrLn ""
  putStrLn "Layer types (A=attention, G=GDN):"
  let layerStr = [if isAttentionIdx cfg i then 'A' else 'G' | i <- [0..63]]
  putStrLn $ "  " ++ intercalate " " (chunksOf 16 layerStr)
  where
    isAttentionIdx c i = (i + 1) `mod` mcFullAttnInterval c == 0
    chunksOf n [] = []
    chunksOf n xs = take n xs : chunksOf n (drop n xs)

runGenerate :: GenOptions -> IO ()
runGenerate opts = do
  let cfg = RuntimeConfig
        { rcModelDir  = genModelDir opts
        , rcDevices   = genGpus opts
        , rcMaxSeqLen = genMaxSeqLen opts
        , rcMaxTokens = genMaxTokens opts
        , rcPrompt    = genPrompt opts
        }

  -- Initialize runtime
  rt <- initRuntime cfg
  let mc = runtimeModelConfig rt
      engine = runtimeEngine rt
      tok = rtTokenizer rt

  -- Tokenize prompt
  putStrLn $ "Tokenizing prompt..."
  promptTokens <- encode tok (genPrompt opts)
  putStrLn $ "  " ++ show (length promptTokens) ++ " tokens"

  if length promptTokens > genMaxSeqLen opts
    then do
      putStrLn "ERROR: prompt exceeds max-seq-len"
      shutdownRuntime rt
      exitFailure
    else return ()

  -- Generate
  putStrLn $ "Generating (max " ++ show (genMaxTokens opts) ++ " tokens)..."
  putStrLn "---"
  outputTokens <- if genStreaming opts
    then generateStreaming engine mc tok promptTokens (genMaxTokens opts)
    else do
      toks <- generate engine mc tok promptTokens (genMaxTokens opts)
      -- Print all at once
      text <- decode tok toks
      putStrLn text
      return toks

  putStrLn "---"
  putStrLn $ "Generated " ++ show (length outputTokens) ++ " tokens"

  -- Cleanup
  shutdownRuntime rt

-- -----------------------------------------------------------------------
-- Main
-- -----------------------------------------------------------------------

main :: IO ()
main = do
  opts <- execParser optsInfo
  case optCommand opts of
    HelloGpu device value -> runHelloGpu device value
    ShowConfig -> runShowConfig
    Generate genOpts -> runGenerate genOpts
  where
    optsInfo = info (optionsParser <**> helper)
      ( fullDesc
     <> progDesc "Haskell inference framework demo for Qwen3.8-27B"
     <> header "haskell-infer-demo - a Haskell LLM inference engine"
      )
