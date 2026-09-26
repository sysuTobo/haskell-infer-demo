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
  , generateSpeculative
  , decideRound
  , proposalWindow
  , takeConfirmed
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

-- ---------------------------------------------------------------------------
-- Speculative decoding (plan S0)
-- ---------------------------------------------------------------------------

-- | The longest prefix of @proposals@ the target confirmed, and the token the target's next
-- row selects. @choices@ is the target's argmax for each of the @length proposals + 1@ rows:
-- row @i@ predicts @proposals !! i@, and the last row is the bonus row. Checking stops at the
-- first mismatch, because later rows condition on a token that was rejected and are therefore
-- unusable as continued generation.
--
-- Pure on purpose: the whole decision is arithmetic over the two lists, so the CPU suite can
-- exercise full acceptance, rejection at every position, ties and k = 1 without an engine.
decideRound :: [Int64] -> [Int64] -> (Int, Int64)
decideRound proposals choices =
  (accepted, choices !! accepted)
  where
    accepted = length (takeWhile id (zipWith (==) proposals choices))

-- | How many proposals a round may carry: the configured window, bounded by the remaining
-- output budget (a round commits at most @k + 1@ tokens, so @k + 1@ must fit) and by the
-- context both engines still have. Zero means "no legal window", and the caller then takes one
-- ordinary target step rather than overproducing a batch the context cannot hold.
proposalWindow :: Int -> Int -> Int -> Int -> Int
proposalWindow k remaining consumedLen maxSeqLen =
  max 0 (minimum [k, remaining - 1, maxSeqLen - consumedLen - 1])

-- | Emit up to and including the first confirmed EOS: no later speculative token may enter the
-- text decoder, and EOS terminates the round without a bonus or continuation.
takeConfirmed :: (Int64 -> Bool) -> [Int64] -> ([Int64], Bool)
takeConfirmed isEos = go []
  where
    go acc [] = (reverse acc, False)
    go acc (t : ts)
      | isEos t = (reverse (t : acc), True)
      | otherwise = go (t : acc) ts

-- | Speculative generation, the plan's S0: **two independent runtimes** that prefill the same
-- prefix, greedy only, a fixed proposal count, and the target verifying candidates with
-- ordinary sequential 'Infer.FFI.Engine.engineDecode' calls in the original order.
--
-- The invariant at every round boundary is the plan's: both engines have consumed @P@, and the
-- already emitted, target-confirmed token @x@ is still pending consumption. A round has the
-- draft consume @x@ and propose @y1 .. yk@ (consuming only through @y(k-1)@, so @yk@ stays
-- unconsumed), the target then consume @[x, y1 .. yk]@ and return its own argmax per row, and
-- the longest matching prefix @y[1:r]@ plus the target's correction (or bonus at @r = k@)
-- becomes the round's output. Only target-confirmed tokens are returned, in order.
--
-- Recovery is deliberately the slow, obviously-correct one: S0 owns no checkpoints (that is
-- S2), so both engines are reset and the retained prefix @P + [x] + y[1:r]@ is replayed
-- **sequentially** - one prefill of the first token and one decode per remaining token - so the
-- state matches the serial path rather than a chunked prefill, which Stage 2 measured to differ.
-- This is why S0 is a correctness prototype and not a speedup.
--
-- The draft and target must share a vocabulary: S0 admits a draft with a different architecture
-- but not a different token space. The runtime is what compares tokenizers and prompt templates.
generateSpeculative
  :: SpecConfig
  -> Ptr EngineHandle   -- ^ draft
  -> Int                -- ^ draft vocabulary size
  -> Ptr EngineHandle   -- ^ target
  -> Int                -- ^ target vocabulary size
  -> [Int]              -- ^ EOS ids, as in 'generate'
  -> [Int64]            -- ^ prompt
  -> Int                -- ^ maximum new tokens
  -> IO [Int64]
generateSpeculative spec draft draftVocab target targetVocab eosTokens prompt maxNew
  | draftVocab /= targetVocab =
      throwIO (userError ("speculative: the draft and target vocabularies differ ("
                          ++ show draftVocab ++ " and " ++ show targetVocab
                          ++ "); S0 requires one token space"))
  | maxNew <= 0 = return []
  | otherwise = do
      -- Both runtimes consume the identical prefix, sequentially, so their states are the
      -- serial path's states and a later replay can reproduce them exactly.
      _ <- consumeSequentially draft vocab prompt
      firstLogits <- consumeSequentially target vocab prompt
      let first = argmax firstLogits
      if isEos first
        then return [first]
        else go [first] prompt first (maxNew - 1)
  where
    isEos token = fromIntegral token `elem` eosTokens
    vocab = targetVocab
    k = spProposals spec

    -- Consume a nonempty sequence on one engine and return the last logits. A single-token
    -- prefill followed by decodes is the same path 'generate' takes after its own prefill.
    consumeSequentially engine v (t : ts) = do
      engineReset engine
      logits <- prefillLogits engine [t] v
      foldlMDecode engine v logits ts
    consumeSequentially _ _ [] =
      throwIO (userError "speculative: an empty sequence is not a consumption")

    foldlMDecode _engine _v logits [] = return logits
    foldlMDecode engine v _ (t : ts) = do
      next <- decodeLogits engine t v
      foldlMDecode engine v next ts

    -- The draft consumes x and proposes k' tokens, consuming only through y(k'-1).
    draftPropose k' x = do
      logits <- decodeLogits draft x vocab
      step k' [] logits
      where
        step 0 acc _ = return (reverse acc)
        step n acc logits = do
          let y = argmax logits
          if n == 1
            then return (reverse (y : acc))
            else do
              logits' <- decodeLogits draft y vocab
              step (n - 1) (y : acc) logits'

    -- emitted-so-far, the consumed prefix, the pending confirmed token, the budget left
    go emitted consumed pending remaining
      | remaining <= 0 = return (reverse emitted)
      | window <= 0 = do
          -- No legal window: one ordinary target step, so the run still finishes. The draft
          -- consumes the same token - its logits are discarded - because the two runtimes have
          -- to hold the same prefix for the next round to propose from the right position.
          logits <- decodeLogits target pending vocab
          _ <- decodeLogits draft pending vocab
          let next = argmax logits
          if isEos next
            then return (reverse (next : emitted))
            else go (next : emitted) (consumed ++ [pending]) next (remaining - 1)
      | otherwise = do
          proposals <- draftPropose window pending
          -- S1's bounded all-position verification: one batch over [x, y1 .. yk] whose row i
          -- conditions on inputs[0..i], so the target pays one forward instead of k+1 decodes.
          rows <- verifyRows target (pending : proposals)
          let (accepted, correction) = decideRound proposals (map argmax rows)
              confirmed = take accepted proposals
              (thisRound, stopped) = takeConfirmed isEos (confirmed ++ [correction])
              emitted' = reverse thisRound ++ emitted
          if stopped
            then return (reverse emitted')
            else do
              -- Retain exactly P + [x] + y[1:r]. The target's batch consumed [x] ++ proposals,
              -- so it is truncated back; the draft consumed only through y(k-1), so it is either
              -- truncated too or - on full acceptance - made to consume its missing yk. The
              -- correction becomes the new pending token and is not consumed here, precisely so
              -- it is not decoded twice.
              let retained = consumed ++ [pending] ++ confirmed
              reconcile target retained
              reconcile draft retained
              go emitted' retained correction (remaining - length thisRound)
      where
        window = proposalWindow k remaining (length consumed) (spMaxSeqLen spec)

    -- Bring an engine's consumed sequence to exactly the retained prefix: back to it when the
    -- round went past (the target always does unless every proposal was accepted), and forward
    -- to it otherwise (the draft's unconsumed yk on full acceptance).
    reconcile engine retained = do
      length_ <- engineSeqLen engine
      if length_ > length retained
        then do
          result <- engineTruncate engine (length retained)
          either (throwIO . userError . ("speculative: truncate failed: " ++)) return result
        else catchUp engine (drop length_ retained)

    catchUp _ [] = return ()
    catchUp engine (t : ts) = do
      _ <- decodeLogits engine t vocab
      catchUp engine ts

    verifyRows engine tokens = do
      result <- engineVerifyRows engine tokens vocab
      either (throwIO . userError . ("speculative: verification failed: " ++)) return result

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
