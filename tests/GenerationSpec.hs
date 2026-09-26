{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE OverloadedStrings #-}

-- | CPU regression for the generation loop: token budget, EOS stopping and
-- error propagation, for both the streaming and the non-streaming entry point.
--
-- No GPU is involved: the engine and the tokenizer come from
-- @tests/generation_engine_stub.c@, a scriptable stand-in for the C ABIs. The
-- code under test is the real "Infer.Generation" / "Infer.Runtime" /
-- "Infer.Tokenizer" stack.
module Main (main) where

import Control.Exception (SomeException, bracket, finally, throwIO, try)
import Data.Int (Int64)
import Data.List (isSuffixOf)
import Foreign.C.Types (CFloat (..), CInt (..))
import Foreign.Marshal.Array (allocaArray, peekArray, withArray)
import Foreign.Ptr (Ptr, nullPtr)
import System.Exit (ExitCode (..))
import Test.Hspec
import qualified SamplingSpec

import Infer.Config
import Infer.Descriptor
import Infer.FFI.Engine
import Infer.Generation
import Infer.Runtime
import Infer.Tokenizer

-- | The greedy baseline the pre-existing generation fixtures assert. The plan requires the
-- greedy regression fixtures to select temperature 0 explicitly rather than inherit a
-- stochastic default, so every call below says so.
greedy :: SamplingConfig
greedy = SamplingConfig { scTemperature = 0, scSeed = Nothing }

-- ---------------------------------------------------------------------------
-- Stub control surface (tests/generation_engine_stub.c)
-- ---------------------------------------------------------------------------

foreign import ccall unsafe "stub_reset"
  stubReset :: IO ()

foreign import ccall unsafe "stub_set_vocab"
  stubSetVocab :: CInt -> IO ()

foreign import ccall unsafe "stub_set_row_values"
  stubSetRowValues :: CFloat -> CFloat -> IO ()

foreign import ccall unsafe "stub_set_script"
  stubSetScript :: Ptr Int64 -> CInt -> IO ()

foreign import ccall unsafe "stub_set_prefill_status"
  stubSetPrefillStatus :: CInt -> IO ()

foreign import ccall unsafe "stub_set_decode_status"
  stubSetDecodeStatus :: CInt -> IO ()

foreign import ccall unsafe "stub_set_create_fails"
  stubSetCreateFails :: CInt -> IO ()

foreign import ccall unsafe "stub_prefill_calls"
  stubPrefillCalls :: IO CInt

foreign import ccall unsafe "stub_decode_calls"
  stubDecodeCalls :: IO CInt

foreign import ccall unsafe "stub_reset_calls"
  stubResetCalls :: IO CInt

foreign import ccall unsafe "stub_tokenizer_loads"
  stubTokenizerLoads :: IO CInt

foreign import ccall unsafe "stub_tokenizer_frees"
  stubTokenizerFrees :: IO CInt

foreign import ccall unsafe "stub_destroy_calls"
  stubDestroyCalls :: IO CInt

foreign import ccall unsafe "stub_stream_free_calls"
  stubStreamFreeCalls :: IO CInt

-- The two-handle surface (plan S0 owns two runtimes) and the consumed-history log.
foreign import ccall unsafe "stub_set_script_h"
  stubSetScriptH :: Ptr EngineHandle -> Ptr Int64 -> CInt -> IO ()

foreign import ccall unsafe "stub_set_script_mode_h"
  stubSetScriptModeH :: Ptr EngineHandle -> CInt -> IO ()

foreign import ccall unsafe "stub_clear_consumed_h"
  stubClearConsumedH :: Ptr EngineHandle -> IO ()

foreign import ccall unsafe "stub_consumed_len_h"
  stubConsumedLenH :: Ptr EngineHandle -> IO CInt

foreign import ccall unsafe "stub_consumed_copy_h"
  stubConsumedCopyH :: Ptr EngineHandle -> Ptr Int64 -> CInt -> IO CInt

foreign import ccall unsafe "stub_decode_calls_h"
  stubDecodeCallsH :: Ptr EngineHandle -> IO CInt

foreign import ccall unsafe "stub_verify_calls_h"
  stubVerifyCallsH :: Ptr EngineHandle -> IO CInt

foreign import ccall unsafe "stub_truncate_calls_h"
  stubTruncateCallsH :: Ptr EngineHandle -> IO CInt

-- | The stub ignores the handle, so a placeholder is enough.
stubEngine :: Ptr EngineHandle
stubEngine = nullPtr

vocab :: Int
vocab = 64

-- | Reset the stub and script the sequence of greedy tokens it will produce.
-- The first entry comes back from the prefill logits, the rest from decode.
script :: [Int64] -> IO ()
script tokens = do
  stubReset
  withArray tokens $ \p -> stubSetScript p (fromIntegral (length tokens))

shouldReport :: String -> IO [Int64] -> Expectation
shouldReport needle action = do
  result <- try action :: IO (Either SomeException [Int64])
  case result of
    Left err -> show err `shouldContain` needle
    Right tokens -> expectationFailure ("expected a failure, produced " ++ show tokens)

-- ---------------------------------------------------------------------------
-- Speculative fixtures (plan S0)
-- ---------------------------------------------------------------------------

-- | Two independent handles, in prefix-indexed mode (the only mode in which a reset and replay
-- is faithful: the argmax depends on what was consumed, not on how many calls happened), each
-- with its own script, and both histories cleared so what they show afterwards is this run's.
-- S0 admits a draft with a different architecture, so only the vocabulary is shared.
withPair :: [Int64] -> [Int64] -> (Ptr EngineHandle -> Ptr EngineHandle -> IO a) -> IO a
withPair targetScript draftScript action = do
  stubReset
  Just draft <- engineCreate "draft-model" "{}" [0] [0]
  Just target <- engineCreate "target-model" "{}" [0] [0]
  let setScript engine tokens =
        withArray tokens $ \p -> stubSetScriptH engine p (fromIntegral (length tokens))
  setScript target targetScript
  setScript draft draftScript
  stubSetScriptModeH target 1
  stubSetScriptModeH draft 1
  stubClearConsumedH target
  stubClearConsumedH draft
  action draft target `finally` (engineDestroy draft >> engineDestroy target)

-- | Every token a handle has consumed, in order. This is what the round protocol's "retain
-- exactly P + [x] + y[1:r] in both engines" is asserted against: each consume is sequential,
-- so a handle's *final* consumed sequence is its state.
consumedOf :: Ptr EngineHandle -> IO [Int64]
consumedOf engine = do
  n <- fromIntegral <$> stubConsumedLenH engine
  allocaArray n $ \buf -> do
    _ <- stubConsumedCopyH engine buf (fromIntegral n)
    peekArray n buf

-- | The plan's initial protocol: a fixed proposal count, and the context both engines share.
specWindow :: Int -> SpecConfig
specWindow k = defaultSpecConfig { spProposals = k, spMaxSeqLen = 64 }

main :: IO ()
main = hspec $ do
  SamplingSpec.spec
  describe "generate without streaming" $ do
    it "generates nothing and never touches the engine on a zero budget" $ do
      script [10, 11, 12]
      tokens <- generate stubEngine vocab [] [1, 2, 3] 0 greedy
      tokens `shouldBe` []
      stubResetCalls `shouldReturn` 0
      stubPrefillCalls `shouldReturn` 0
      stubDecodeCalls `shouldReturn` 0

    it "treats a negative budget as no work" $ do
      script [10, 11, 12]
      tokens <- generate stubEngine vocab [] [1, 2, 3] (-3) greedy
      tokens `shouldBe` []
      stubPrefillCalls `shouldReturn` 0
      stubDecodeCalls `shouldReturn` 0

    it "stops before decoding when the first token is EOS" $ do
      script [7, 8, 9]
      tokens <- generate stubEngine vocab [7] [1] 5 greedy
      tokens `shouldBe` [7]
      stubPrefillCalls `shouldReturn` 1
      stubDecodeCalls `shouldReturn` 0

    it "stops at a later EOS token, including it in the result" $ do
      script [1, 2, 9, 3]
      tokens <- generate stubEngine vocab [9] [1] 10 greedy
      tokens `shouldBe` [1, 2, 9]
      stubDecodeCalls `shouldReturn` 2

    it "produces exactly the budget when no EOS arrives" $ do
      script [1, 2, 3, 4]
      tokens <- generate stubEngine vocab [] [1] 3 greedy
      tokens `shouldBe` [1, 2, 3]
      stubPrefillCalls `shouldReturn` 1
      stubDecodeCalls `shouldReturn` 2

    it "reports a prefill failure instead of an empty result" $ do
      script [1, 2]
      stubSetPrefillStatus (-3)
      shouldReport "prefill failed" (generate stubEngine vocab [] [1] 2 greedy)

    it "reports a decode failure instead of a partial result" $ do
      script [1, 2]
      stubSetDecodeStatus (-3)
      result <- try (generate stubEngine vocab [] [1] 3 greedy) :: IO (Either SomeException [Int64])
      case result of
        Left err -> show err `shouldContain` "decode failed"
        Right tokens -> expectationFailure ("expected a failure, produced " ++ show tokens)

  describe "generateStreaming" $ do
    it "does not touch the engine on a zero budget and still frees the stream" $ do
      script [1, 2, 3]
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      tokens <- bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming stubEngine vocab [] stream [1] 0 greedy
      tokens `shouldBe` []
      stubPrefillCalls `shouldReturn` 0
      stubDecodeCalls `shouldReturn` 0
      stubStreamFreeCalls `shouldReturn` 1

    it "stops at a first-token EOS like the non-streaming loop" $ do
      script [5, 6, 7]
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      tokens <- bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming stubEngine vocab [5] stream [1] 4 greedy
      tokens `shouldBe` [5]
      stubDecodeCalls `shouldReturn` 0

    it "streams the budgeted tokens and stops at a later EOS" $ do
      script [1, 2, 8, 3]
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      tokens <- bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming stubEngine vocab [8] stream [1] 10 greedy
      tokens `shouldBe` [1, 2, 8]
      stubDecodeCalls `shouldReturn` 2

    it "uses the same negative-budget rule as the non-streaming loop" $ do
      script [1, 2, 3]
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      tokens <- bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming stubEngine vocab [] stream [1] (-2) greedy
      tokens `shouldBe` []
      stubPrefillCalls `shouldReturn` 0
      stubDecodeCalls `shouldReturn` 0

    it "reports a prefill failure instead of a partial result" $ do
      script [1, 2]
      stubSetPrefillStatus (-4)
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      result <- try $
        bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
          generateStreaming stubEngine vocab [] stream [1] 3 greedy
      case result :: Either SomeException [Int64] of
        Left err -> show err `shouldContain` "prefill failed"
        Right tokens -> expectationFailure ("expected a failure, produced " ++ show tokens)

    it "reports a decode failure instead of a partial result" $ do
      script [1, 2]
      stubSetDecodeStatus (-3)
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      result <- try $
        bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
          generateStreaming stubEngine vocab [] stream [1] 3 greedy
      case result :: Either SomeException [Int64] of
        Left err -> show err `shouldContain` "decode failed"
        Right tokens -> expectationFailure ("expected a failure, produced " ++ show tokens)

  describe "sampling through the loop" $ do
    it "keeps temperature 0 on the scripted greedy tokens" $ do
      script [10, 11, 12, 13]
      greedyRun <- generate stubEngine vocab [] [1] 4 greedy
      greedyRun `shouldBe` [10, 11, 12, 13]

    it "stops at a sampled EOS through the same stop path as greedy" $ do
      -- A decided row (the scripted winner dominates), so the first draw lands on the EOS
      -- token with probability 1 - O(1e-25) and the loop must stop there without decoding.
      script [7]
      stubSetRowValues 30 (-30)
      sampled <- generate stubEngine vocab [7] [1] 5 (SamplingConfig 1.0 (Just 42))
      sampled `shouldBe` [7]
      stubDecodeCalls `shouldReturn` 0

    it "gives the same tokens for the same seed with and without streaming" $ do
      script [10, 11, 12, 13]
      plain <- generate stubEngine vocab [] [1] 4 (SamplingConfig 1.0 (Just 7))
      Just tok <- loadTokenizer "/nonexistent/tokenizer.json"
      withStream <- bracket (newDecodeStream tok) freeDecodeStream $ \stream ->
        generateStreaming stubEngine vocab [] stream [1] 4 (SamplingConfig 1.0 (Just 7))
      withStream `shouldBe` plain

    it "makes a shorter request a prefix of a longer one under the same seed" $ do
      script [10, 11, 12, 13, 14]
      short <- generate stubEngine vocab [] [1] 2 (SamplingConfig 1.0 (Just 5))
      long <- generate stubEngine vocab [] [1] 4 (SamplingConfig 1.0 (Just 5))
      short `shouldBe` take 2 long
      length short `shouldBe` 2

  -- The plan's S0 round protocol is arithmetic over the draft's proposals and the target's
  -- rows, so most of its gates are checked here without an engine: full acceptance, rejection
  -- at every position, k = 1, ties, the window's bounds and the EOS rule. The scripted-engine
  -- half (two handles, consumed histories) is exercised by the speculative section below.
  describe "speculative round arithmetic (plan S0)" $ do
    it "accepts the whole window and takes the bonus row" $ do
      decideRound [10, 11, 12] [10, 11, 12, 13] `shouldBe` (3, 13)

    it "takes the correction row when the first proposal is rejected" $ do
      decideRound [10, 11, 12] [99, 11, 12, 13] `shouldBe` (0, 99)

    it "stops at the first rejection" $ do
      -- Row 1 mismatched, so rows 2 and 3 describe continuations of a rejected token.
      decideRound [10, 11, 12] [10, 99, 12, 13] `shouldBe` (1, 99)

    it "rejects at every position in turn" $ do
      decideRound [10, 11, 12] [10, 11, 12, 13] `shouldBe` (3, 13)
      decideRound [10, 11, 12] [10, 11, 99, 13] `shouldBe` (2, 99)
      decideRound [10, 11, 12] [10, 99, 12, 13] `shouldBe` (1, 99)
      decideRound [10, 11, 12] [99, 11, 12, 13] `shouldBe` (0, 99)

    it "handles k = 1" $ do
      decideRound [7] [7, 8] `shouldBe` (1, 8)
      decideRound [7] [9, 8] `shouldBe` (0, 9)

    it "repeats a rejection identically, because the decision is only the two rows" $ do
      decideRound [1, 2] [5, 6, 7] `shouldBe` (0, 5)
      decideRound [1, 2] [5, 6, 7] `shouldBe` (0, 5)

    it "breaks a tie towards the lowest id" $ do
      argmax [1, 1, 1] `shouldBe` 0
      argmax [-1, -1] `shouldBe` 0
      argmax [0, 2, 2] `shouldBe` 1

    it "bounds the window by the budget and the context" $ do
      -- k + 1 tokens are committed per round, so the budget must leave room for one more.
      proposalWindow 3 10 0 4096 `shouldBe` 3
      proposalWindow 3 4 0 4096 `shouldBe` 3
      proposalWindow 3 2 0 4096 `shouldBe` 1
      proposalWindow 3 1 0 4096 `shouldBe` 0
      proposalWindow 3 0 0 4096 `shouldBe` 0
      -- The context is what the engines still have, not the descriptor's maximum.
      proposalWindow 3 10 4090 4096 `shouldBe` 3
      proposalWindow 3 10 4092 4096 `shouldBe` 3
      proposalWindow 3 10 4093 4096 `shouldBe` 2
      proposalWindow 3 10 4095 4096 `shouldBe` 0

    it "emits up to and including the first confirmed EOS and no further" $ do
      takeConfirmed (== 9) [1, 2, 3] `shouldBe` ([1, 2, 3], False)
      takeConfirmed (== 9) [1, 9, 3] `shouldBe` ([1, 9], True)
      takeConfirmed (== 9) [9, 1] `shouldBe` ([9], True)
      takeConfirmed (== 9) [] `shouldBe` ([], False)

  -- Plan S0: the sequential correctness prototype. Explicitly not a speedup - the draft is
  -- recovered by reset and sequential replay - but the acceptance, pending-token and retention
  -- logic is what the later stages reuse, so each of the plan's gate cases is pinned here. Every
  -- case also runs the same request on the target alone, because the plan asks for that
  -- comparison independently of the round arithmetic.
  describe "speculative generation over the scripted engine (plan S0)" $ do
    it "confirms a fully agreeing draft and leaves both handles on the retained prefix" $ do
      let scriptT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      withPair scriptT scriptT $ \draft target -> do
        out <- generateSpeculative (specWindow 2) draft 64 target 64 [] [1] 4
        out `shouldBe` [1, 2, 3, 4]
        -- P + [x] + y[1:r] in both runtimes, which is observable as the sequence length: the
        -- prompt plus every emitted token except the still-pending last one.
        engineSeqLen target `shouldReturn` 4
        engineSeqLen draft `shouldReturn` 4
        targetHistory <- consumedOf target
        targetHistory `shouldSatisfy` isSuffixOf [1, 1, 2, 3]
        draftHistory <- consumedOf draft
        draftHistory `shouldSatisfy` isSuffixOf [1, 1, 2, 3]
        -- Nothing had to be rolled back, and the draft had to *advance* instead: on full
        -- acceptance its last proposal was never consumed, so the round makes it consume that
        -- token rather than replaying the prefix (S1's rollback, not S0's replay).
        draftTruncated <- stubTruncateCallsH draft
        draftTruncated `shouldBe` 0
        draftDecoded <- stubDecodeCallsH draft
        draftDecoded `shouldSatisfy` (> 0)
        -- The same request on the target alone, from a clean history.
        stubClearConsumedH target
        single <- generate target 64 [] [1] 4 greedy
        single `shouldBe` out
        consumedOf target `shouldReturn` [1, 1, 2, 3]

    it "corrects a draft that diverges at the first position" $ do
      let scriptT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          scriptD = [0, 1, 9, 9, 9, 9, 9, 9, 9, 9]
      withPair scriptT scriptD $ \draft target -> do
        out <- generateSpeculative (specWindow 2) draft 64 target 64 [] [1] 3
        out `shouldBe` [1, 2, 3]
        -- The draft really proposed and the target really verified - with S1's bounded batch,
        -- not with sequential decodes, and the rollback truncated the rejected suffix away.
        draftCalls <- stubDecodeCallsH draft
        draftCalls `shouldSatisfy` (> 0)
        verifyCalls <- stubVerifyCallsH target
        verifyCalls `shouldSatisfy` (> 0)
        truncated <- stubTruncateCallsH target
        truncated `shouldSatisfy` (> 0)
        -- The rejected proposal was truncated away, so the *length* is the retained prefix even
        -- though the consumed log still records what was fed and then rolled back.
        engineSeqLen target `shouldReturn` 3
        engineSeqLen draft `shouldReturn` 3

    it "handles k = 1" $ do
      let scriptT = [0, 1, 2, 3, 4, 5]
          scriptD = [0, 1, 2, 9, 9, 9]
      withPair scriptT scriptD $ \draft target -> do
        out <- generateSpeculative (specWindow 1) draft 64 target 64 [] [1] 4
        single <- generate target 64 [] [1] 4 greedy
        out `shouldBe` single
        out `shouldBe` [1, 2, 3, 4]

    it "repeats rejections across rounds and still matches the target alone" $ do
      let scriptT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
          scriptD = [0, 1, 9, 9, 9, 9, 9, 9, 9, 9]
      withPair scriptT scriptD $ \draft target -> do
        out <- generateSpeculative (specWindow 2) draft 64 target 64 [] [1] 5
        single <- generate target 64 [] [1] 5 greedy
        out `shouldBe` single
        out `shouldBe` [1, 2, 3, 4, 5]

    it "takes an ordinary target step when only one output slot is left" $ do
      let scriptT = [0, 1, 2, 3]
      withPair scriptT scriptT $ \draft target -> do
        out <- generateSpeculative (specWindow 2) draft 64 target 64 [] [1] 1
        out `shouldBe` [1]
        stubDecodeCallsH draft `shouldReturn` 0

    it "falls back to target-only when the window has no room in the context" $ do
      let scriptT = [0, 1, 2, 3, 4, 5]
          tight = (specWindow 2) { spMaxSeqLen = 2 }
      withPair scriptT scriptT $ \draft target -> do
        out <- generateSpeculative tight draft 64 target 64 [] [1] 4
        single <- generate target 64 [] [1] 4 greedy
        out `shouldBe` single
        -- The context bound refused every window, so no proposal was ever verified; the two
        -- runtimes still step together, which is why the draft is decoded but never proposed
        -- from.
        stubVerifyCallsH draft `shouldReturn` 0
        stubVerifyCallsH target `shouldReturn` 0

    it "stops at a confirmed EOS and emits nothing after it" $ do
      let scriptT = [0, 1, 2, 9, 4, 5]
      withPair scriptT scriptT $ \draft target -> do
        out <- generateSpeculative (specWindow 2) draft 64 target 64 [9] [1] 4
        single <- generate target 64 [9] [1] 4 greedy
        single `shouldBe` out
        out `shouldBe` [1, 2, 9]

    it "refuses a draft whose vocabulary differs from the target's" $ do
      withPair [0, 1, 2] [0, 1, 2] $ \draft target ->
        shouldReport "vocabularies differ"
          (generateSpeculative (specWindow 2) draft 32 target 64 [] [1] 3)

  describe "initRuntime" $ do
    it "releases the tokenizer when engine_create fails" $ do
      stubReset
      stubSetCreateFails 1
      let cfg = defaultRuntimeConfig
            { rcDescriptor = Just "descriptors/qwen38-27b.json"
            , rcDevices = [0]
            }
      result <- try (initRuntime cfg) :: IO (Either ExitCode Runtime)
      isExitFailure result `shouldBe` True
      stubTokenizerLoads `shouldReturn` 1
      stubTokenizerFrees `shouldReturn` 1

    it "releases the engine and tokenizer when initialization throws after the engine is live" $ do
      -- A throw between "engine created" and "Runtime returned" is exactly the
      -- window the caller's bracket cannot cover: its acquire action never
      -- completed, so the resources must be released here.
      stubReset
      Just tok <- loadTokenizer (rcModelDir defaultRuntimeConfig ++ "/tokenizer.json")
      Just engine <- engineCreate (rcModelDir defaultRuntimeConfig) "{}" [0] [0]
      result <- try (withRuntimeResources engine tok (throwIO (userError "init failed")))
        :: IO (Either SomeException ())
      isLeftError result `shouldBe` True
      stubDestroyCalls `shouldReturn` 1
      stubTokenizerFrees `shouldReturn` 1

isExitFailure :: Either ExitCode a -> Bool
isExitFailure (Left (ExitFailure _)) = True
isExitFailure _ = False

isLeftError :: Either SomeException a -> Bool
isLeftError (Left _) = True
isLeftError (Right _) = False
