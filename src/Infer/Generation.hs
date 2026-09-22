-- | Greedy decoding loop with streaming output.
--
-- Manages the prefill → decode cycle, EOS detection, and token-by-token
-- text output via the tokenizer.
module Infer.Generation
  ( generate
  , generateStreaming
  , argmax
  ) where

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
generate :: Ptr EngineHandle -> Int -> [Int] -> [Int64] -> Int -> IO [Int64]
generate engine vocab eosTokens prompt maxNew = do
  engineReset engine
  -- Prefill
  result <- enginePrefill engine prompt vocab
  case result of
    Left err -> do
      putStrLn $ "Prefill error: " ++ err
      return []
    Right logits -> do
      let firstToken = argmax logits
      go [firstToken] firstToken (maxNew - 1)
  where
    go acc _ 0 = return (reverse acc)
    go acc lastTok n = do
      result <- engineDecode engine lastTok vocab
      case result of
        Left err -> do
          putStrLn $ "Decode error: " ++ err
          return (reverse acc)
        Right logits -> do
          let nextTok = argmax logits
          if fromIntegral nextTok `elem` eosTokens
            then return (reverse (nextTok : acc))
            else go (nextTok : acc) nextTok (n - 1)

-- | Generate tokens with streaming output (prints each token as it's decoded).
generateStreaming :: Ptr EngineHandle -> Int -> [Int] -> Tokenizer -> [Int64] -> Int -> IO [Int64]
generateStreaming engine vocab eosTokens tok prompt maxNew = do
  engineReset engine
  -- Prefill
  result <- enginePrefill engine prompt vocab
  case result of
    Left err -> do
      putStrLn $ "Prefill error: " ++ err
      return []
    Right logits -> do
      let firstToken = argmax logits
      emitToken tok firstToken
      go [firstToken] firstToken (maxNew - 1)
  where
    go acc _ 0 = do
      putStrLn ""  -- newline after streaming
      return (reverse acc)
    go acc lastTok n = do
      result <- engineDecode engine lastTok vocab
      case result of
        Left err -> do
          putStrLn $ "\nDecode error: " ++ err
          return (reverse acc)
        Right logits -> do
          let nextTok = argmax logits
          if fromIntegral nextTok `elem` eosTokens
            then do
              putStrLn ""
              return (reverse (nextTok : acc))
            else do
              emitToken tok nextTok
              go (nextTok : acc) nextTok (n - 1)

-- | Decode and print a single token.
emitToken :: Tokenizer -> Int64 -> IO ()
emitToken tok tid = do
  text <- decodeSingle tok tid
  putStr text
  hFlush stdout
