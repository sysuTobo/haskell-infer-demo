-- | Greedy decoding loop with streaming output.
--
-- Manages the prefill → decode cycle, EOS detection, and token-by-token
-- text output via the tokenizer.
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
import Data.List (foldl')
import Data.Ord (comparing)
import Foreign.Ptr (Ptr)
import System.IO (hFlush, stdout)

import Infer.FFI.Engine
import Infer.Tokenizer

-- | Find the index of the maximum value in a list.
argmax :: [Float] -> Int64
argmax [] = 0
argmax xs = fromIntegral (fst (maximumBy' (comparing snd) (zip [0..] xs)))
  where
    maximumBy' _ [] = error "argmax: empty list"
    maximumBy' cmp (x:xs') = foldl' (\acc y -> if cmp acc y == LT then y else acc) x xs'

-- | Generate tokens greedily (non-streaming, returns all tokens at once).
-- The vocabulary size comes from 'Infer.FFI.Engine.engineVocabSize'.
--
-- At most @maxNew@ tokens are produced, counting the first token decoded from
-- the prefill logits; the budget stop and the EOS stop apply to both the first
-- and the later tokens.
generate :: Ptr EngineHandle -> Int -> [Int] -> [Int64] -> Int -> IO [Int64]
generate engine vocab eosTokens prompt maxNew
  | maxNew <= 0 = return []
  | otherwise = do
      engineReset engine
      logits <- prefillLogits engine prompt vocab
      let firstToken = argmax logits
      if isEos firstToken
        then return [firstToken]
        else go [firstToken] firstToken (maxNew - 1)
  where
    isEos token = fromIntegral token `elem` eosTokens
    go acc _ remaining | remaining <= 0 = return (reverse acc)
    go acc lastTok remaining = do
      logits <- decodeLogits engine lastTok vocab
      let nextTok = argmax logits
      if isEos nextTok
        then return (reverse (nextTok : acc))
        else go (nextTok : acc) nextTok (remaining - 1)

-- | Generate tokens with streaming output (prints each token as it's decoded).
--
-- The text of a token is produced by the incremental decoder, so a character
-- whose UTF-8 bytes span several tokens is only printed once complete. The
-- caller owns the stream handle (create it with 'newDecodeStream').
generateStreaming :: Ptr EngineHandle -> Int -> [Int] -> DecodeStream -> [Int64] -> Int -> IO [Int64]
generateStreaming engine vocab eosTokens stream prompt maxNew
  | maxNew <= 0 = return []
  | otherwise = do
      engineReset engine
      logits <- prefillLogits engine prompt vocab
      let firstToken = argmax logits
      emitToken stream firstToken
      if isEos firstToken
        then finish [firstToken]
        else go [firstToken] firstToken (maxNew - 1)
  where
    isEos token = fromIntegral token `elem` eosTokens
    finish acc = do
      -- Flush a trailing partial character before closing the line, so the
      -- last token is never silently dropped.
      tailText <- finishStream stream
      putStr tailText
      putStrLn ""  -- newline after streaming
      return (reverse acc)
    go acc _ remaining | remaining <= 0 = finish acc
    go acc lastTok remaining = do
      logits <- decodeLogits engine lastTok vocab
      let nextTok = argmax logits
      if isEos nextTok
        then finish (nextTok : acc)
        else do
          emitToken stream nextTok
          go (nextTok : acc) nextTok (remaining - 1)

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
