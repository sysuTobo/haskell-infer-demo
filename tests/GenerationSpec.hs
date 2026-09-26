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

import Control.Exception (SomeException, bracket, throwIO, try)
import Data.Int (Int64)
import Foreign.C.Types (CFloat (..), CInt (..))
import Foreign.Marshal.Array (withArray)
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
