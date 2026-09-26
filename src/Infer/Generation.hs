-- | The generation loop: prefill → decode, EOS detection, and token-by-token output.
--
-- Token selection goes through "Infer.Sampling"'s one selector, so the greedy path and the
-- sampled path differ only in the 'Infer.Config.SamplingConfig' they are handed: at
-- temperature 0 the selector is the lowest-id 'argmax' and consumes no random word, and
-- above temperature 0 it draws once per selected token from the request's own generator.
-- Both entry points below thread the same generator state through the same call, which is
-- what makes a seed replay the same tokens with or without @--stream@.
--
-- Failure policy: a zero or negative token budget generates nothing and never
-- touches the engine; any engine error is thrown to the caller instead of being
-- turned into a short (or empty) result, so the entry point can report failure.
module Infer.Generation
  ( generate
  , generateStreaming
  , argmax
  ) where

import Control.Exception (throwIO)
import Data.Int (Int64)
import Foreign.Ptr (Ptr)
import System.IO (hFlush, stdout)

import Infer.Config
import Infer.FFI.Engine
import Infer.Sampling (Drawn(..), argmax, newRng, stepToken)
import Infer.Tokenizer

-- | Generate tokens (non-streaming, returns all tokens at once).
-- The vocabulary size comes from 'Infer.FFI.Engine.engineVocabSize'.
--
-- At most @maxNew@ tokens are produced, counting the first token decoded from
-- the prefill logits; the budget stop and the EOS stop apply to both the first
-- and the later tokens.
--
-- The record the selector produces (the token's sampler and raw-model
-- log-probabilities) is available to a caller that wants a trajectory; this entry
-- point does not collect one, because the plan's ordinary CLI output need not, and
-- a caller that needs it for a learner update must request it before the update
-- rather than reconstruct it later.
generate :: Ptr EngineHandle -> Int -> [Int] -> [Int64] -> Int -> SamplingConfig -> IO [Int64]
generate engine vocab eosTokens prompt maxNew sampling
  | maxNew <= 0 = return []
  | otherwise = do
      engineReset engine
      logits <- prefillLogits engine prompt vocab
      (firstToken, rng1) <- draw logits (newRng (streamSeed sampling))
      if isEos firstToken
        then return [firstToken]
        else go rng1 [firstToken] firstToken (maxNew - 1)
  where
    isEos token = fromIntegral token `elem` eosTokens

    draw logits rng = case stepToken False sampling rng logits of
      Left err -> throwIO (userError ("sampling failed: " ++ err))
      Right (drawn, rng') -> return (drawnToken drawn, rng')

    go _ acc _ remaining | remaining <= 0 = return (reverse acc)
    go rng acc lastTok remaining = do
      logits <- decodeLogits engine lastTok vocab
      (nextTok, rng') <- draw logits rng
      if isEos nextTok
        then return (reverse (nextTok : acc))
        else go rng' (nextTok : acc) nextTok (remaining - 1)

-- | Generate tokens with streaming output (prints each token as it's decoded).
--
-- The text of a token is produced by the incremental decoder, so a character
-- whose UTF-8 bytes span several tokens is only printed once complete. The
-- caller owns the stream handle (create it with 'newDecodeStream').
--
-- A sampled EOS takes the same stop path as a greedy one, and the two paths
-- consume the same generator words for the same seed: the selector call is
-- identical and its position in the loop is identical.
generateStreaming :: Ptr EngineHandle -> Int -> [Int] -> DecodeStream -> [Int64] -> Int
                  -> SamplingConfig -> IO [Int64]
generateStreaming engine vocab eosTokens stream prompt maxNew sampling
  | maxNew <= 0 = return []
  | otherwise = do
      engineReset engine
      logits <- prefillLogits engine prompt vocab
      (firstToken, rng1) <- draw logits (newRng (streamSeed sampling))
      emitToken stream firstToken
      if isEos firstToken
        then finish [firstToken]
        else go rng1 [firstToken] firstToken (maxNew - 1)
  where
    isEos token = fromIntegral token `elem` eosTokens

    draw logits rng = case stepToken False sampling rng logits of
      Left err -> throwIO (userError ("sampling failed: " ++ err))
      Right (drawn, rng') -> return (drawnToken drawn, rng')

    finish acc = do
      -- Flush a trailing partial character before closing the line, so the
      -- last token is never silently dropped.
      tailText <- finishStream stream
      putStr tailText
      putStrLn ""  -- newline after streaming
      return (reverse acc)

    go _ acc _ remaining | remaining <= 0 = finish acc
    go rng acc lastTok remaining = do
      logits <- decodeLogits engine lastTok vocab
      (nextTok, rng') <- draw logits rng
      if isEos nextTok
        then finish (nextTok : acc)
        else do
          emitToken stream nextTok
          go rng' (nextTok : acc) nextTok (remaining - 1)

-- | Prefill, propagating the engine's error instead of returning no logits.
prefillLogits :: Ptr EngineHandle -> [Int64] -> Int -> IO [Float]
prefillLogits engine prompt vocab = do
  result <- enginePrefill engine prompt vocab
  case result of
    Left err -> throwIO (userError ("prefill failed: " ++ err))
    Right logits -> return logits

-- | Decode one token, propagating the engine's error.
decodeLogits :: Ptr EngineHandle -> Int64 -> Int -> IO [Float]
decodeLogits engine token vocab = do
  result <- engineDecode engine token vocab
  case result of
    Left err -> throwIO (userError ("decode failed: " ++ err))
    Right logits -> return logits

-- | Feed one token to the incremental decoder and print whatever text became
-- available (nothing, while a multi-byte character is still incomplete).
emitToken :: DecodeStream -> Int64 -> IO ()
emitToken stream tid = do
  feedToken stream tid
  text <- drainStream stream
  putStr text
  hFlush stdout
