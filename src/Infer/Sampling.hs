-- | Temperature sampling: the pure binary64 softmax/CDF selector and the request's RNG.
--
-- The migration this implements is specified in docs/plan-numeric-contract.md under
-- "Temperature-sampling migration" (milestones T1 and T2). Two things are worth stating
-- about the shape of this module, because both are choices rather than accidents.
--
-- **The arithmetic is a policy, not an implementation detail.** The plan fixes it: each
-- input 'Float' logit becomes a 'Double' /before/ the subtraction and the division; the
-- exponentials, the normalisation and the CDF accumulation happen on the CPU in binary64
-- in increasing token-id order. That is deliberately not the CUDA FP32 logit path and not
-- the trainer's FP32 loss reduction, so that the sampler's own numbers are reproducible
-- against a stated rule instead of against whatever the device did. The inverse CDF is
-- specified too: the first /positive-weight/ token whose ordered cumulative weight is
-- strictly greater than @u * Z@, so a zero-mass leading token cannot be selected at
-- @u = 0@, and a target that rounds up to @Z@ selects the last positive-weight token as
-- the specified endpoint correction rather than an arbitrary final vocabulary entry.
--
-- **The RNG is implemented here rather than taken as a dependency.** The plan says to
-- start from a "version-pinned @splitmix@ dependency" and to freeze known seed-to-word
-- vectors. What that protects is the *algorithm*'s behaviour across builds, so this module
-- pins the algorithm itself - splitmix64's published gamma and mix constants - and freezes
-- the same vectors in @tests/SamplingSpec.hs@, which is strictly more stable than a
-- version bound and keeps the build free of a new Hackage dependency. The API is
-- Word64-in/Word64-out as the plan asks, it is for sampling and not cryptography, and the
-- mapping to @[0,1)@ is the plan's @(x >> 11) * 2^-53@.
module Infer.Sampling
  ( -- * The request RNG
    Rng
  , newRng
  , rngWord
  , rngUniform
  , uniformFromWord
    -- * The prepared distribution and one selection
  , Prepared
  , Drawn(..)
  , prepare
  , choose
  , select
  , stepToken
    -- * Greedy, and the tie rule the two modes share
  , argmax
  , validateLogits
    -- * Identity
  , samplerVersion
  ) where

import Data.Bits (shiftR, xor)
import Data.Int (Int64)
import Data.List (foldl')
import Data.Word (Word64)

import Infer.Config (SamplingConfig(..))

-- ---------------------------------------------------------------------------
-- Identity
-- ---------------------------------------------------------------------------

-- | The arithmetic and endpoint rules this selector implements, recorded next to the
-- selection it makes (plan: "Version the arithmetic and endpoint rules").
samplerVersion :: String
samplerVersion = "host-binary64-cdf-1"

-- ---------------------------------------------------------------------------
-- The request RNG (splitmix64)
-- ---------------------------------------------------------------------------

-- | A request-owned generator. There is no process-global state and no reset per token:
-- one state is threaded through a whole request, which is what makes the same seed replay
-- the same tokens without depending on how many draws another part of the code made.
newtype Rng = Rng Word64
  deriving (Eq, Show)

-- splitmix64's published constants. The state advances by the gamma *before* the mix, so
-- the first output for seed 0 is the algorithm's published 0xe220a8397b1dcdaf.
splitmixGamma :: Word64
splitmixGamma = 0x9E3779B97F4A7C15

splitmixMix :: Word64 -> Word64
splitmixMix z0 =
  let z1 = (z0 `xor` (z0 `shiftR` 30)) * 0xBF58476D1CE4E5B9
      z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
  in z2 `xor` (z2 `shiftR` 31)

newRng :: Word64 -> Rng
newRng seed = Rng seed

-- | Advance once and return the output word.
rngWord :: Rng -> (Word64, Rng)
rngWord (Rng s) =
  let s' = s + splitmixGamma
  in (splitmixMix s', Rng s')

-- | The plan's @u = Double(x >> 11) * 2^-53@: 53 bits of mantissa, so the grid is the
-- binary64 unit roundoff and @u@ is always in @[0,1)@.
uniformFromWord :: Word64 -> Double
uniformFromWord x = fromIntegral (x `shiftR` 11) * (2 ** (-53))

rngUniform :: Rng -> (Double, Rng)
rngUniform rng = let (word, rng') = rngWord rng in (uniformFromWord word, rng')

-- ---------------------------------------------------------------------------
-- Validation
-- ---------------------------------------------------------------------------

-- | Reject an empty or non-finite logit row. @-infinity@ is rejected too: this initial
-- sampler is unmasked by construction (plan: "including @-infinity@ in this initial
-- unmasked sampler"), so an infinite logit is corruption rather than a mask.
validateLogits :: [Float] -> Either String ()
validateLogits [] = Left "sampling: the logit row is empty"
validateLogits logits
  | all isFiniteFloat logits = Right ()
  | otherwise = Left "sampling: the logit row contains a non-finite value"
  where
    isFiniteFloat v = not (isNaN v) && not (isInfinite v)

-- | The index of the maximum, lowest token id on an exact tie. This is the tie rule both
-- modes share, so the greedy path and the sampled path cannot drift apart.
argmax :: [Float] -> Int64
argmax [] = 0
argmax xs = fromIntegral (snd (foldl' step (head zipped) (tail zipped)))
  where
    zipped = zip xs ([0 ..] :: [Int])
    step (best, bestIndex) (v, i)
      | v > best = (v, i)
      | otherwise = (best, bestIndex)

-- ---------------------------------------------------------------------------
-- The prepared distribution
-- ---------------------------------------------------------------------------

-- | One logit row prepared for a temperature: the plan's @m@, @a_i@, @w_i@, @Z@ and, when
-- the caller asked for it, the raw-model normalisation. The lists are in increasing token
-- id and are built with strict folds, so a decode step does not carry a lazy chain of
-- exponentials behind it.
data Prepared = Prepared
  { prZs       :: [Double]        -- ^ the logits widened. The selected token's a_i and its
                                  --   raw-model offset are derived from this rather than kept
                                  --   as a second vocabulary-sized table
  , prM        :: Double          -- ^ m = max_i z_i
  , prTau      :: Double          -- ^ the temperature the weights were prepared at
  , prWeights  :: [Double]        -- ^ w_i = exp((z_i - m)/tau); a zero here is zero numerical mass
  , prZ        :: Double          -- ^ Z = left fold of w_i in token order
  , prLogZ     :: Double          -- ^ log Z, the temperature normalisation
  , prLogZModel :: Maybe Double   -- ^ log(sum exp(z_i - m)), the raw-model normalisation
  , prLastPositive :: Maybe Int   -- ^ the last positive-weight token, the endpoint correction
  }
  deriving (Eq, Show)

-- | Prepare one row for a temperature. The caller passes @wantModel@ rather than always
-- computing the raw-model normalisation: the plan asks for it only when the selection
-- record is actually requested ("Only compute the raw-model normalization when its record
-- is requested"), and the ordinary CLI does not collect a trajectory.
--
-- A very small positive temperature can drive @a_i@ to negative infinity and underflow
-- @w_i@ to zero. That is zero numerical mass, as the plan specifies, not a reason to
-- reroute to 'argmax'.
prepare :: Bool -> Double -> [Float] -> Either String Prepared
prepare wantModel tau logits = do
  validateLogits logits
  if isNaN tau || isInfinite tau || tau <= 0
    then Left ("sampling: a temperature must be finite and positive, got " ++ show tau)
    else Right ()
  let zs = map realToFrac logits :: [Double]   -- widen before any arithmetic
      m = maximum zs
      as = map (\z -> (z - m) / tau) zs
      ws = map exp as
      zsum = foldl' (+) 0 ws
      logZ = log zsum
      logZModel = if wantModel
                    then Just (log (foldl' (+) 0 (map (\z -> exp (z - m)) zs)))
                    else Nothing
      lastPositive = foldl' (\acc (i, w) -> if w > 0 then Just i else acc) Nothing
                            (zip [0 ..] ws)
  if isNaN zsum || isInfinite zsum || zsum < 1
    then Left "sampling: the normaliser is not finite (Z must be finite and >= 1)"
    else Right Prepared
      { prZs = zs
      , prM = m
      , prTau = tau
      , prWeights = ws
      , prZ = zsum
      , prLogZ = logZ
      , prLogZModel = logZModel
      , prLastPositive = lastPositive
      }

-- ---------------------------------------------------------------------------
-- One selection
-- ---------------------------------------------------------------------------

-- | One selection: the token, and the two log-probabilities the plan distinguishes.
data Drawn = Drawn
  { drawnToken       :: Int64
  , drawnLogpSampler :: Double         -- ^ ell_sampler = a_i - log Z
  , drawnLogpModel   :: Maybe Double   -- ^ ell_model, present only when requested
  } deriving (Eq, Show)

-- | Choose with a supplied uniform @u@. Splitting this from 'prepare' is what lets the
-- CDF, the boundary and the endpoint tests be deterministic instead of PRNG-dependent.
--
-- The inverse CDF walks the tokens in increasing id and takes the first positive-weight
-- token whose ordered cumulative weight is *strictly* greater than @u * Z@. Because the
-- prefix sums are folded in the same order as @Z@, the final prefix equals @Z@ bitwise, so
-- a target strictly below @Z@ always finds a token and a target exactly at @Z@ is the
-- specified endpoint correction. A target above @Z@ cannot come from @u < 1@ and is an
-- error rather than a silent clamp.
choose :: Prepared -> Double -> Either String Drawn
choose p u
  | isNaN u || isInfinite u || u < 0 || u >= 1 =
      Left ("sampling: u must be finite and in [0,1), got " ++ show u)
  | otherwise =
      let r = u * prZ p
      in if isNaN r || isInfinite r
           then Left "sampling: the CDF target is not finite"
           else if r > prZ p * (1 + 1e-12)
             then Left "sampling: the CDF target exceeds Z; the row is inconsistent"
             else case firstHit (prWeights p) r of
                    Just i  -> Right (drawnAt p i)
                    Nothing -> case prLastPositive p of
                      Just i  -> Right (drawnAt p i)   -- the endpoint correction
                      Nothing -> Left "sampling: every token has zero numerical mass"
  where
    -- The first positive-weight token whose ordered cumulative weight passes r.
    firstHit ws0 r0 = go 0 0.0 ws0
      where
        go _ _ [] = Nothing
        go i acc (w:rest)
          | w > 0 && acc + w > r0 = Just i
          | otherwise = go (i + 1) (acc + w) rest

    -- The two log-probabilities are computed from the *original* logit in log space: the
    -- sampler one under the temperature, and the raw-model one at the model's own
    -- normalisation. They are not the same number scaled, so the raw-model one uses
    -- (z_i - m) and not a_i.
    drawnAt p0 i =
      let zi = prZs p0 !! i
          ai = (zi - prM p0) / prTau p0
      in Drawn { drawnToken = fromIntegral i
               , drawnLogpSampler = ai - prLogZ p0
               , drawnLogpModel = fmap (\lz -> (zi - prM p0) - lz) (prLogZModel p0)
               }

-- | Prepare and choose in one call, for the tests and for a caller that already holds a
-- @u@ (a frozen vector, say).
select :: Bool -> Double -> [Float] -> Double -> Either String Drawn
select wantModel tau logits u = prepare wantModel tau logits >>= \p -> choose p u

-- ---------------------------------------------------------------------------
-- The generation loop's contract
-- ---------------------------------------------------------------------------

-- | The next-token contract both generation entry points use.
--
-- At temperature 0 the plan's rules are: no division by zero, no softmax and no random
-- word consumed. The raw-model log-probability is still available on request, and it is
-- the model's softmax probability of the argmax token - not a sampler probability of one,
-- which is what temperature 0 would make of it.
--
-- Above temperature 0 exactly one word is consumed per selected token, including a
-- terminal EOS, because the draw happens before the token is known to end the completion.
stepToken :: Bool -> SamplingConfig -> Rng -> [Float] -> Either String (Drawn, Rng)
stepToken wantModel cfg rng logits
  | scTemperature cfg == 0 = do
      validateLogits logits
      let i = fromIntegral (argmax logits)
      modelLogp <- if wantModel
                     then case prepare True 1.0 logits of
                            Left err -> Left err
                            Right p -> case prLogZModel p of
                              Nothing -> Left "sampling: the model normalisation was not produced"
                              Just lz -> Right (Just ((prZs p !! fromIntegral i) - prM p - lz))
                     else Right Nothing
      Right (Drawn { drawnToken = i
                   , drawnLogpSampler = 0.0   -- the selected sampler probability is one
                   , drawnLogpModel = modelLogp
                   }, rng)
  | otherwise = do
      p <- prepare wantModel (scTemperature cfg) logits
      let (u, rng') = rngUniform rng
      drawn <- choose p u
      Right (drawn, rng')
