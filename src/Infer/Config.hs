-- | Runtime configuration for the CLI (user-specified, not model knowledge).
--
-- Model architecture lives in "Infer.Descriptor"; this module only carries the
-- choices the user makes on the command line.
module Infer.Config
  ( RuntimeConfig(..)
  , defaultRuntimeConfig
  , SamplingConfig(..)
  , defaultSamplingConfig
  , parseSampling
  , seedIsUnused
  , samplingSummary
  , streamSeed
  , SpecConfig(..)
  , RollbackMode(..)
  , defaultSpecConfig
  , maxProposals
  , parseProposals
  ) where

import Data.Char (isDigit)
import Data.Word (Word64)

-- | Runtime configuration (user-specified).
data RuntimeConfig = RuntimeConfig
  { rcModelDir     :: FilePath        -- ^ Path to model weights directory
  , rcDescriptor   :: Maybe FilePath  -- ^ Optional descriptor JSON override
  , rcDevices      :: [Int]           -- ^ CUDA device ordinals to use
  , rcTp           :: Int             -- ^ Tensor-parallel ranks (1 = pipelined layer split)
  , rcEp           :: Int             -- ^ Expert-parallel ranks (1 = whole experts per rank)
  , rcMaxSeqLen    :: Int             -- ^ Maximum context length
  , rcMaxTokens    :: Int             -- ^ Maximum new tokens to generate
  , rcPrompt       :: String          -- ^ Input prompt
  , rcSampling     :: SamplingConfig  -- ^ How this request draws its tokens
  } deriving (Eq, Show)

defaultRuntimeConfig :: RuntimeConfig
defaultRuntimeConfig = RuntimeConfig
  { rcModelDir    = "weights/Qwen3.8-27B"
  , rcDescriptor  = Nothing
  , rcDevices     = [0, 1]
  , rcTp          = 1
  , rcEp          = 1
  , rcMaxSeqLen   = 4096
  , rcMaxTokens   = 256
  , rcPrompt      = "Hello"
  , rcSampling    = defaultSamplingConfig
  }

-- | How one request draws its tokens (plan "Temperature-sampling migration"). The type is
-- shared by the CLI and the generation loop on purpose, so the defaults and the validation
-- rules exist once rather than being duplicated per call site.
--
-- Temperature @0@ is greedy; a positive temperature samples from @softmax(logits/T)@. The
-- seed fixes the request's random stream and is unused at temperature @0@, where the
-- selection consumes no draw.
data SamplingConfig = SamplingConfig
  { scTemperature :: Double
  , scSeed        :: Maybe Word64
  } deriving (Eq, Show)

-- | The plan's target default: categorical sampling at temperature 1 with no truncation,
-- and a seed resolved and reported per request.
defaultSamplingConfig :: SamplingConfig
defaultSamplingConfig = SamplingConfig { scTemperature = 1.0, scSeed = Nothing }

-- | How a rejected round is rolled back. A pure full-attention model can just move the
-- append-only cache's length back (S1), which costs nothing; a model with a recurrent layer
-- cannot rewind that state, so it saves a round checkpoint and restores it (S2). The caller
-- picks from the descriptor, because that is where the mixer kinds are known.
data RollbackMode = RollbackTruncate | RollbackCheckpoint
  deriving (Eq, Show)

-- | The speculative window (plan S). S0 is greedy-only and takes a *fixed* proposal count -
-- the plan says to start at 2-4 and to select a fixed k from measurements before considering
-- anything adaptive - and the context capacity both engines share, which bounds how many
-- tokens a round may consume.
data SpecConfig = SpecConfig
  { spProposals :: Int
  , spMaxSeqLen :: Int
  , spRollback  :: RollbackMode
  } deriving (Eq, Show)

-- | The plan's starting window, with the rollback the attention-only case can use.
defaultSpecConfig :: SpecConfig
defaultSpecConfig = SpecConfig { spProposals = 3, spMaxSeqLen = 4096
                               , spRollback = RollbackTruncate }

-- | The largest window this prototype admits. A window is a bounded batch of proposals; a
-- large one is a different protocol (adaptive windows are explicitly later work), so the
-- ceiling is a refusal rather than a silently accepted number.
maxProposals :: Int
maxProposals = 16

-- | Parse @--speculative-k@. Zero is refused rather than turned into an ordinary step: a
-- window of no proposals is not speculative decoding, and pretending otherwise would hide a
-- typo behind a slower-but-correct path.
parseProposals :: String -> Either String Int
parseProposals raw = case reads raw of
  [(n, "")] | n >= 1 && n <= maxProposals -> Right n
            | otherwise ->
                Left ("--speculative-k must be between 1 and " ++ show maxProposals
                      ++ ", got " ++ raw)
  _ -> Left ("--speculative-k must be an integer, got " ++ raw)

-- | A seed supplied with temperature 0 is accepted and reported as unused rather than
-- silently dropped (the plan is explicit about both halves of that).
seedIsUnused :: SamplingConfig -> Bool
seedIsUnused cfg = scTemperature cfg == 0 && scSeed cfg /= Nothing

-- | One line for @stderr@. The plan keeps request metadata out of the generated stdout
-- text, so this is what the CLI prints before generation.
samplingSummary :: SamplingConfig -> String
samplingSummary cfg =
  "sampling: temperature=" ++ show (scTemperature cfg)
  ++ ", seed=" ++ maybe "none" show (scSeed cfg)
  ++ (if seedIsUnused cfg then " (unused at temperature 0)" else "")

-- | The seed the request's generator starts from. A configuration whose seed was omitted
-- must have it resolved by the caller - the CLI draws one from the OS and reports it - and
-- this treats an unresolved seed as 0 so the loop stays total and reproducible instead of
-- falling back to nondeterminism behind the caller's back.
streamSeed :: SamplingConfig -> Word64
streamSeed cfg = maybe 0 id (scSeed cfg)

-- | Parse the CLI's two sampling options into one validated configuration, or the reason
-- the pair is invalid. The caller does this before any model or tokenizer allocation, so a
-- bad configuration cannot get as far as an engine.
parseSampling :: Maybe String -> Maybe String -> Either String SamplingConfig
parseSampling rawTemperature rawSeed = do
  temperature <- maybe (Right (scTemperature defaultSamplingConfig)) parseTemperature rawTemperature
  seed <- maybe (Right Nothing) (fmap Just . parseSeed) rawSeed
  Right SamplingConfig { scTemperature = temperature, scSeed = seed }

-- | A finite decimal temperature, with the plan's rules read literally:
--
--   * a negative nonzero literal is invalid even when its magnitude would round to zero,
--     so the sign cannot be hidden by underflow;
--   * a nonzero literal that underflows to zero is invalid rather than silently greedy;
--   * a literal that overflows to infinity is invalid;
--   * literal negative zero is canonicalised to greedy zero.
parseTemperature :: String -> Either String Double
parseTemperature raw = case reads raw of
  [(v, "")] ->
    let negative = case dropWhile (== ' ') raw of
                     ('-':_) -> True
                     _       -> False
        literalNonzero = any (`elem` ("123456789" :: String)) raw
    in if negative && literalNonzero
         then Left ("sampling: a temperature must not be negative, got " ++ show raw)
         else if isNaN v || isInfinite v
           then Left ("sampling: a temperature must be finite, got " ++ show raw)
           else if v == 0 && literalNonzero
             then Left ("sampling: " ++ show raw ++ " underflows to zero; a nonzero "
                        ++ "temperature must be representable as a Double")
             else Right (if v < 0 then 0 else v)  -- -0.0 and 0 both mean greedy
  _ -> Left ("sampling: --temperature expects a decimal number, got " ++ show raw)

-- | An unsigned decimal 'Word64'. Signs, whitespace and every other character are refused,
-- and the range is the plan's @[0, 2^64-1]@.
parseSeed :: String -> Either String Word64
parseSeed raw
  | null raw = Left "sampling: --seed expects an unsigned decimal integer"
  | not (all isDigit raw) =
      Left ("sampling: --seed expects an unsigned decimal integer, got " ++ show raw)
  | otherwise = case reads raw :: [(Integer, String)] of
      [(n, "")] | n <= 18446744073709551615 -> Right (fromIntegral n)
      _ -> Left ("sampling: --seed is out of range for an unsigned 64-bit integer: "
                 ++ show raw)
