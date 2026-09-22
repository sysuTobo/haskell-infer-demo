-- | Runtime configuration for the CLI (user-specified, not model knowledge).
--
-- Model architecture lives in "Infer.Descriptor"; this module only carries the
-- choices the user makes on the command line.
module Infer.Config
  ( RuntimeConfig(..)
  , defaultRuntimeConfig
  ) where

-- | Runtime configuration (user-specified).
data RuntimeConfig = RuntimeConfig
  { rcModelDir     :: FilePath        -- ^ Path to model weights directory
  , rcDescriptor   :: Maybe FilePath  -- ^ Optional descriptor JSON override
  , rcDevices      :: [Int]           -- ^ CUDA device ordinals to use
  , rcTp           :: Int             -- ^ Tensor-parallel ranks (1 = pipelined layer split)
  , rcMaxSeqLen    :: Int             -- ^ Maximum context length
  , rcMaxTokens    :: Int             -- ^ Maximum new tokens to generate
  , rcPrompt       :: String          -- ^ Input prompt
  } deriving (Eq, Show)

defaultRuntimeConfig :: RuntimeConfig
defaultRuntimeConfig = RuntimeConfig
  { rcModelDir    = "weights/Qwen3.8-27B"
  , rcDescriptor  = Nothing
  , rcDevices     = [0, 1]
  , rcTp           = 1
  , rcMaxSeqLen   = 4096
  , rcMaxTokens   = 256
  , rcPrompt      = "Hello"
  }
