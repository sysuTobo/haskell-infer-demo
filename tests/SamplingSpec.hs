{-# LANGUAGE OverloadedStrings #-}

-- | The sampling gate (plan T1 and T2 of the temperature-sampling migration).
--
-- Two groups of checks, matching the plan's "Sampling gates and commands" items 1 and 3:
--
--   * the **pure math and boundaries**: the temperature distribution, the inverse-CDF
--     scan, the endpoint rules, exact CDF hits, zero-mass bins, shift invariance, and
--     every input validation the plan names. The expected distributions are computed here
--     independently in 'Double' rather than read back out of the selector, so the selector
--     cannot certify itself.
--   * the **replay and lifecycle** half this suite can carry: the frozen splitmix64
--     vectors, the draw-count contract (greedy consumes none, a positive temperature
--     exactly one per selected token) and the configuration parsing rules.
--
-- One honest remainder is recorded rather than papered over. The plan's endpoint
-- correction - "if floating multiplication rounds r up to Z, select the last
-- positive-weight token" - cannot be *reached* by any @u@ in @[0,1)@ here: @Z@ is at most
-- the vocabulary size, and @u <= 1-2^-53@ gives @u*Z <= Z*(1-2^-53) < Z@ with the prefix
-- sums folded in the same order, so the scan always finds a bin. The reachable half of the
-- rule - that the last draw never selects an arbitrary final vocabulary entry - is asserted
-- below; the branch itself is defensive.
module SamplingSpec (spec) where

import Data.Word (Word64)
import Test.Hspec

import Infer.Config
import Infer.Sampling

-- ---------------------------------------------------------------------------
-- Independent expectations (deliberately not the selector's own arithmetic)
-- ---------------------------------------------------------------------------

-- | The temperature distribution of a logit row, computed here in 'Double'.
independent :: Double -> [Float] -> ([Double], [Double], Double)
independent tau logits =
  let zs = map realToFrac logits :: [Double]
      m = maximum zs
      as = map (\z -> (z - m) / tau) zs
      ws = map exp as
      zsum = sum ws
  in (as, map (/ zsum) ws, log zsum)

-- | A failure predicate, shared by every group below.
isLeft' :: Either a b -> Bool
isLeft' (Left _) = True
isLeft' (Right _) = False

-- | The token the strictly-greater inverse CDF selects, computed independently.
independentChoice :: [Double] -> Double -> Int
independentChoice probs u =
  let r = u * sum probs
      go _ [] = length probs - 1
      go acc (p:rest)
        | p > 0 && acc + p > r = 0
        | otherwise = 1 + go (acc + p) rest
  in go 0 probs

-- ---------------------------------------------------------------------------
-- Config parsing (T0)
-- ---------------------------------------------------------------------------

spec_parse :: Spec
spec_parse = describe "sampling configuration" $ do
  it "defaults to temperature 1 with no seed" $
    parseSampling Nothing Nothing `shouldBe` Right (SamplingConfig 1.0 Nothing)

  it "reads an explicit temperature and seed" $
    parseSampling (Just "0.7") (Just "42") `shouldBe` Right (SamplingConfig 0.7 (Just 42))

  it "canonicalises literal zero and negative zero to greedy" $ do
    parseSampling (Just "0") Nothing `shouldBe` Right (SamplingConfig 0 Nothing)
    parseSampling (Just "-0") Nothing `shouldBe` Right (SamplingConfig 0 Nothing)
    parseSampling (Just "-0.0") Nothing `shouldBe` Right (SamplingConfig 0 Nothing)
    parseSampling (Just "0.0") (Just "7") `shouldBe` Right (SamplingConfig 0 (Just 7))

  it "refuses a negative nonzero temperature even when it would round to zero" $
    parseSampling (Just "-1e-400") Nothing `shouldSatisfy` isLeft'

  it "refuses a nonzero temperature that underflows to zero" $
    parseSampling (Just "1e-400") Nothing `shouldSatisfy` isLeft'
  it "refuses a temperature that overflows to infinity" $
    parseSampling (Just "1e400") Nothing `shouldSatisfy` isLeft'
  it "refuses a non-numeric temperature" $
    parseSampling (Just "warm") Nothing `shouldSatisfy` isLeft'
  it "accepts the whole unsigned 64-bit seed range" $ do
    parseSampling Nothing (Just "0") `shouldBe` Right (SamplingConfig 1.0 (Just 0))
    parseSampling Nothing (Just "18446744073709551615")
      `shouldBe` Right (SamplingConfig 1.0 (Just 18446744073709551615))
  it "refuses a seed outside the unsigned 64-bit range or malformed" $ do
    parseSampling Nothing (Just "18446744073709551616") `shouldSatisfy` isLeft'
    parseSampling Nothing (Just "-1") `shouldSatisfy` isLeft'
    parseSampling Nothing (Just "0x10") `shouldSatisfy` isLeft'
    parseSampling Nothing (Just "") `shouldSatisfy` isLeft'
  it "reports a seed given at temperature 0 as unused" $ do
    seedIsUnused (SamplingConfig 0 (Just 3)) `shouldBe` True
    seedIsUnused (SamplingConfig 1 (Just 3)) `shouldBe` False
    seedIsUnused (SamplingConfig 0 Nothing) `shouldBe` False

-- ---------------------------------------------------------------------------
-- The request RNG (T2)
-- ---------------------------------------------------------------------------

-- | The frozen seed-to-word vectors. seed 0's first word is splitmix64's published value,
-- and the rest were derived by an independent implementation of the same constants.
spec_rng :: Spec
spec_rng = describe "the request RNG" $ do
  it "reproduces the frozen seed-to-word vectors" $ do
    let wordsOf seed n = take n (go (newRng seed))
          where go r = let (w, r') = rngWord r in w : go r'
    wordsOf 0 4 `shouldBe`
      [0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4, 0x06c45d188009454f, 0xf88bb8a8724c81ec]
    wordsOf 1 4 `shouldBe`
      [0x910a2dec89025cc1, 0xbeeb8da1658eec67, 0xf893a2eefb32555e, 0x71c18690ee42c90b]
    wordsOf 42 4 `shouldBe`
      [0xbdd732262feb6e95, 0x28efe333b266f103, 0x47526757130f9f52, 0x581ce1ff0e4ae394]
    wordsOf 0xDEADBEEFCAFEBABE 4 `shouldBe`
      [0x0d7d93560d1929d2, 0x491dfb740e50d43f, 0x42722bf4473e5e7d, 0xd6ca8a0790fffc45]

  it "maps a word to [0,1) with 53 bits of mantissa" $ do
    uniformFromWord 0 `shouldBe` 0.0
    uniformFromWord 0xFFFFFFFFFFFFFFFF `shouldBe` (1 - 2 ** (-53))
    uniformFromWord 0x0008000000000000 `shouldBe` 2 ** (-13)
    uniformFromWord 0xe220a8397b1dcdaf `shouldBe` 0.88331080821364261

  it "advances the state by the gamma before mixing" $ do
    let (w1, r1) = rngWord (newRng 0)
        (w2, _) = rngWord r1
    w1 `shouldBe` 0xe220a8397b1dcdaf
    w2 `shouldBe` 0x6e789e6aa1b965f4

  it "produces the same stream for the same seed and different streams otherwise" $ do
    let take3 s = take 3 (iterate (snd . rngWord) (newRng s)) :: [Rng]
        words3 s = take 3 [w | (w, _) <- iterate (\(_, r) -> rngWord r) (rngWord (newRng s))]
    words3 (7 :: Word64) `shouldBe` words3 7
    (words3 7 == words3 (8 :: Word64)) `shouldBe` False
    length (take3 7) `shouldBe` 3

-- ---------------------------------------------------------------------------
-- The distribution and its boundaries (T1)
-- ---------------------------------------------------------------------------

spec_distribution :: Spec
spec_distribution = describe "the distribution and the inverse CDF" $ do
  it "splits two tokens as 1/4 and 3/4 at temperature 1" $ do
    let logits = [0, realToFrac (log 3 :: Double) :: Float]
        (_, probs, _) = independent 1.0 logits
    probs `shouldSatisfy` \[p0, p1] -> abs (p0 - 0.25) < 1e-6 && abs (p1 - 0.75) < 1e-6
    -- and the selection follows the same split
    tokenAt 1.0 logits 0.2 `shouldBe` Right 0
    tokenAt 1.0 logits 0.3 `shouldBe` Right 1

  it "concentrates the same logits at temperature 0.5" $ do
    let logits = [0, realToFrac (log 3 :: Double) :: Float]
        (_, probs, _) = independent 0.5 logits
    probs `shouldSatisfy` \[p0, p1] -> abs (p0 - 0.1) < 1e-6 && abs (p1 - 0.9) < 1e-6

  it "is invariant to a representable constant shift" $ do
    let base = [0, 0.5, 0.25] :: [Float]
        shifted = [4, 4.5, 4.25] :: [Float]
    tokenAt 0.7 base 0.4 `shouldBe` tokenAt 0.7 shifted 0.4
    logpAt 0.7 base 0.4 `shouldBe` logpAt 0.7 shifted 0.4

  it "keeps equal logits uniform" $ do
    let logits = [0, 0, 0, 0] :: [Float]
    [tokenAt 1.0 logits u | u <- [0.0, 0.25, 0.5, 0.75, 0.99]]
      `shouldBe` [Right 0, Right 1, Right 2, Right 3, Right 3]

  it "agrees with an independently written scan over a grid of u" $ do
    let logits = [0.3, -0.7, 1.1, 0.0] :: [Float]
        grid = [fromIntegral k / 1000 | k <- [0 .. 999]]
        (_, probs, _) = independent 1.0 logits
    [tokenAt 1.0 logits u | u <- grid]
      `shouldBe` [Right (fromIntegral (independentChoice probs u)) | u <- grid]

  it "handles a singleton vocabulary" $ do
    tokenAt 1.0 [3.5] 0.999 `shouldBe` Right 0
    tokenAt 1.0 [3.5] 0.0 `shouldBe` Right 0

  it "handles all-negative logits" $ do
    let logits = [-5, -3, -4] :: [Float]
        (_, probs, _) = independent 1.0 logits
    probs `shouldSatisfy` \[a, b, c] -> b > c && c > a

  it "sharpens at a low positive temperature and flattens at a high one" $ do
    let logits = [0, 1] :: [Float]
        maxProb tau = maximum (let (_, p, _) = independent tau logits in p)
    maxProb 0.01 `shouldSatisfy` (> maxProb 1.0)
    maxProb 100 `shouldSatisfy` (< maxProb 1.0)

  it "gives an exact CDF hit to the following positive bin" $ do
    -- Equal weights make Z = 4 and the prefix sums 1,2,3,4, so u = 0.25 lands exactly on
    -- the first boundary and must select the second token.
    tokenAt 1.0 [0, 0, 0, 0] 0.25 `shouldBe` Right 1
    tokenAt 1.0 [0, 0, 0, 0] 0.5 `shouldBe` Right 2

  it "never selects a zero-mass leading token at u = 0" $ do
    -- A tiny temperature underflows everything but the maximum to zero mass.
    tokenAt 0.001 [0, -1000, -1000] 0.0 `shouldBe` Right 0

  it "never selects an arbitrary final entry at the largest admitted u" $ do
    -- The trailing token has zero mass, so the last draw must be the last *positive* one.
    tokenAt 1.0 [0, 0, -1000] (1 - 2 ** (-53)) `shouldBe` Right 1

  it "does not reroute a zero-mass tail to the argmax at a tiny temperature" $ do
    let logits = [0, -50, -200] :: [Float]
    tokenAt 1e-3 logits 0.999999 `shouldBe` Right 0

  it "agrees with an independent log-sum-exp on both log-probabilities" $ do
    -- u = 0.1 deliberately selects a token that is *not* the row's maximum, because at the
    -- maximum z_i = m and the two log-probabilities differ only by the normalisations - the
    -- one fixture where a tau-scaled ell_model would still look right.
    let logits = [0.5, -0.25, 1.25, -2.0] :: [Float]
        tau = 0.8
        (as, _, logZ) = independent tau logits
        zs = map realToFrac logits :: [Double]
        m = maximum zs
        logZModel = log (sum (map (\z -> exp (z - m)) zs))
    case select True tau logits 0.1 of
      Left err -> expectationFailure ("select failed: " ++ err)
      Right d -> do
        let i = fromIntegral (drawnToken d)
        i `shouldSatisfy` (/= 2)   -- token 2 is the max; the fixture must not take it
        drawnLogpSampler d `shouldSatisfy` \v -> abs (v - (as !! i - logZ)) < 1e-12
        -- ell_model is the raw-model normalisation of the same row, computed from (z_i - m)
        -- and not from a_i: at tau /= 1 those are different numbers.
        drawnLogpModel d `shouldSatisfy` \ms -> case ms of
          Just v -> abs (v - (((zs !! i) - m) - logZModel)) < 1e-12
          Nothing -> False
        -- and they are genuinely different log-probabilities here, which is why the record
        -- keeps both rather than one scaled into the other
        drawnLogpModel d `shouldSatisfy` \ms -> case ms of
          Just v -> abs (v - drawnLogpSampler d) > 1e-6
          Nothing -> False

  it "omits the raw-model log-probability when it is not requested" $ do
    case select False 1.0 [0, 1, 2] 0.5 of
      Left err -> expectationFailure err
      Right d -> drawnLogpModel d `shouldBe` Nothing

  it "validates the logits before consuming anything" $ do
    prepare False 1.0 ([] :: [Float]) `shouldSatisfy` isLeft'
    prepare False 1.0 [0, 1 / 0] `shouldSatisfy` isLeft'
    prepare False 1.0 [0, -1 / 0] `shouldSatisfy` isLeft'
    prepare False 1.0 [0, 0 / 0] `shouldSatisfy` isLeft'

  it "refuses an invalid temperature or an invalid u" $ do
    prepare False 0 [0, 1] `shouldSatisfy` isLeft'
    prepare False (-1) [0, 1] `shouldSatisfy` isLeft'
    prepare False (0 / 0) [0, 1] `shouldSatisfy` isLeft'
    checkU (0 / 0) `shouldBe` True
    checkU (1 / 0) `shouldBe` True
    checkU (-0.5) `shouldBe` True
    checkU 1.0 `shouldBe` True
    checkU 1.0001 `shouldBe` True
    checkU 0.0 `shouldBe` False
    checkU 0.999999 `shouldBe` False
    checkU (1 - 2 ** (-53)) `shouldBe` False
  where
    tokenAt tau logits u = drawnToken <$> select False tau logits u
    logpAt tau logits u = drawnLogpSampler <$> select False tau logits u
    checkU u = case prepare False 1.0 [0, 1] of
      Left _ -> False
      Right p -> case choose p u of
        Left _ -> True
        Right _ -> False

-- ---------------------------------------------------------------------------
-- The generation loop's contract (T2)
-- ---------------------------------------------------------------------------

spec_step :: Spec
spec_step = describe "the next-token contract" $ do
  it "does no draw at temperature 0 and picks the lowest-id maximum" $ do
    let rng = newRng 12345
        logits = [1, 5, 5, 2] :: [Float]
    case stepToken False (SamplingConfig 0 Nothing) rng logits of
      Left err -> expectationFailure err
      Right (d, rng') -> do
        drawnToken d `shouldBe` 1          -- lowest id among the tied maxima
        drawnLogpSampler d `shouldBe` 0.0  -- the selected sampler probability is one
        rng' `shouldBe` rng                -- greedy consumes no word

  it "consumes exactly one word per draw above temperature 0" $ do
    let rng = newRng 9
        (_, expectedNext) = rngWord rng
    case stepToken False (SamplingConfig 1.0 (Just 9)) rng [0, 1, 2] of
      Left err -> expectationFailure err
      Right (_, rng') -> rng' `shouldBe` expectedNext

  it "reproduces the same token for the same seed and state" $ do
    let cfg = SamplingConfig 1.0 (Just 3)
        logits = [0.1, 0.2, 0.3, 0.4] :: [Float]
        once = drawnToken . fst <$> stepToken False cfg (newRng 3) logits
        twice = drawnToken . fst <$> stepToken False cfg (newRng 3) logits
    once `shouldBe` twice

  it "resolves an omitted seed to a total, reproducible stream" $ do
    streamSeed (SamplingConfig 1.0 Nothing) `shouldBe` 0
    streamSeed (SamplingConfig 1.0 (Just 7)) `shouldBe` 7
    drawnToken <$> select False 1.0 [0, 1] 0.9 `shouldSatisfy` (== Right 1)

  it "shares one tie rule between the greedy and the sampled paths" $ do
    -- argmax must agree with the plan's "lowest token id wins exact ties".
    argmax [1, 3, 3, 0] `shouldBe` 1
    argmax [-1, -1] `shouldBe` 0
    -- argmax itself does not validate - a NaN is simply not greater than anything - which
    -- is why the selector validates the row *before* it calls argmax. That ordering is a
    -- requirement, so it is asserted here rather than left to the reader.
    argmax [0 / 0, 5] `shouldBe` 0
    case stepToken False (SamplingConfig 0 Nothing) (newRng 1) [0 / 0, 5] of
      Left _ -> pure ()
      Right _ -> expectationFailure "a NaN logit was selected from"

-- ---------------------------------------------------------------------------

spec :: Spec
spec = describe "Sampling" $ do
  spec_parse
  spec_rng
  spec_distribution
  spec_step
