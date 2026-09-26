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
  , generateSpeculativeWithStats
  , SpecStats(..)
  , RoundStats(..)
  , decideRound
  , proposalWindow
  , takeConfirmed
  , argmax
  ) where

import Control.Exception (finally, throwIO)
import Control.Monad (when)
import Data.Int (Int64)
import Foreign.Ptr (Ptr)
import GHC.Clock (getMonotonicTime)
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

-- | What one round did, for the plan's S performance gate: what it proposed, what the target
-- confirmed, what it committed, and how long the round took. A round with no legal window
-- commits one token with no proposals, which is why the fields are counted rather than inferred.
data RoundStats = RoundStats
  { rsProposals :: Int
  , rsAccepted  :: Int
  , rsCommitted :: Int
  , rsSeconds   :: Double
  } deriving (Eq, Show)

data SpecStats = SpecStats
  { ssRounds :: [RoundStats]
  } deriving (Eq, Show)

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
-- Recovery follows the plan's two admissions, chosen by the caller from the descriptor:
-- S1's **truncate** mode moves an append-only cache's length back for a pure full-attention
-- model, which costs nothing; S2's **checkpoint** mode saves the round-start state (the buffers
-- @engine_reset@ would clear, the sequence length and the parameter identity) and restores it
-- on rejection, which is what admits a model with a recurrent layer - a GDN state cannot be
-- rewound, so it has to be restored. Either way the round then replays exactly the inputs it
-- kept, @[x] + y[1:r]@, and the checkpoints are released when the run ends.
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
generateSpeculative spec draft draftVocab target targetVocab eosTokens prompt maxNew =
  fst <$> generateSpeculativeWithStats spec draft draftVocab target targetVocab eosTokens
                                      prompt maxNew

-- | The same run, with the per-round statistics the plan's S performance gate reports.
generateSpeculativeWithStats
  :: SpecConfig
  -> Ptr EngineHandle
  -> Int
  -> Ptr EngineHandle
  -> Int
  -> [Int]
  -> [Int64]
  -> Int
  -> IO ([Int64], SpecStats)
generateSpeculativeWithStats spec draft draftVocab target targetVocab eosTokens prompt maxNew
  | draftVocab /= targetVocab =
      throwIO (userError ("speculative: the draft and target vocabularies differ ("
                          ++ show draftVocab ++ " and " ++ show targetVocab
                          ++ "); S0 requires one token space"))
  | maxNew <= 0 = return ([], SpecStats { ssRounds = [] })
  | otherwise = do
      (tokens, rounds) <- run `finally` releaseCheckpoints
      return (tokens, SpecStats { ssRounds = rounds })
  where
    isEos token = fromIntegral token `elem` eosTokens
    vocab = targetVocab
    k = spProposals spec
    checkpoints = spRollback spec == RollbackCheckpoint

    run = do
      -- Both runtimes consume the identical prefix, sequentially, so their states are the
      -- serial path's states and a later replay can reproduce them exactly.
      _ <- consumeSequentially draft vocab prompt
      firstLogits <- consumeSequentially target vocab prompt
      let first = argmax firstLogits
      if isEos first
        then return ([first], [])
        else go [] [first] prompt first (maxNew - 1)

    -- A run's checkpoints do not outlive it: a live one would refuse the next run's save.
    releaseCheckpoints = do
      _ <- engineCheckpointRelease draft
      _ <- engineCheckpointRelease target
      return ()

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

    -- rounds-so-far, emitted-so-far, the consumed prefix, the pending confirmed token, the
    -- budget left
    go rounds emitted consumed pending remaining
      | remaining <= 0 = return (reverse emitted, reverse rounds)
      | window <= 0 = do
          -- No legal window: one ordinary target step, so the run still finishes. The draft
          -- consumes the same token - its logits are discarded - because the two runtimes have
          -- to hold the same prefix for the next round to propose from the right position.
          started <- getMonotonicTime
          logits <- decodeLogits target pending vocab
          _ <- decodeLogits draft pending vocab
          ended <- getMonotonicTime
          let plain = RoundStats { rsProposals = 0, rsAccepted = 0, rsCommitted = 1
                                 , rsSeconds = ended - started }
              next = argmax logits
          if isEos next
            then return (reverse (next : emitted), reverse (plain : rounds))
            else go (plain : rounds) (next : emitted) (consumed ++ [pending]) next (remaining - 1)
      | otherwise = do
          started <- getMonotonicTime
          -- S2's checkpoint is taken before anything consumes, so a rejection can be undone;
          -- S1's mode needs nothing here because its cache can be moved back.
          when checkpoints saveBothCheckpoints
          proposals <- draftPropose window pending
          -- S1's bounded all-position verification: one batch over [x, y1 .. yk] whose row i
          -- conditions on inputs[0..i], so the target pays one forward instead of k+1 decodes.
          rows <- verifyRows target (pending : proposals)
          let (accepted, correction) = decideRound proposals (map argmax rows)
              confirmed = take accepted proposals
              (thisRound, stopped) = takeConfirmed isEos (confirmed ++ [correction])
              emitted' = reverse thisRound ++ emitted
          ended <- getMonotonicTime
          let entry = RoundStats { rsProposals = length proposals, rsAccepted = accepted
                                 , rsCommitted = length thisRound
                                 , rsSeconds = ended - started }
          if stopped
            then return (reverse emitted', reverse (entry : rounds))
            else do
              -- Retain exactly P + [x] + y[1:r], by whichever rollback the mode admits; the
              -- correction becomes the new pending token and is not consumed here, precisely so
              -- it is not decoded twice.
              let retained = consumed ++ [pending] ++ confirmed
              rollback (length consumed) (accepted < window) retained
              go (entry : rounds) emitted' retained correction (remaining - length thisRound)
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

    -- S2's rollback: a *rejected* round puts the round-start state back on both runtimes -
    -- which is what a model with a recurrent layer needs, since its state cannot be moved back
    -- like a cache's length - and then replays exactly the retained suffix, the plan's "restore
    -- the round-start state, then replay exactly the retained inputs". A fully accepted round
    -- has nothing to undo: the target is already at the retained prefix, and only the draft's
    -- never-consumed last proposal is missing, which is the same catch-up S1 does.
    rollback startLen rejected retained
      | checkpoints && rejected = do
          restoreCheckpoint draft
          restoreCheckpoint target
          catchUp draft (drop startLen retained)
          catchUp target (drop startLen retained)
      | otherwise = do
          reconcile target retained
          reconcile draft retained

    -- A live checkpoint from the previous round is finished with; the engine refuses a second.
    saveCheckpoint engine = do
      _ <- engineCheckpointRelease engine
      result <- engineCheckpointSave engine
      either (throwIO . userError . ("speculative: checkpoint save failed: " ++)) return result

    saveBothCheckpoints = saveCheckpoint draft >> saveCheckpoint target

    restoreCheckpoint engine = do
      result <- engineCheckpointRestore engine
      either (throwIO . userError . ("speculative: checkpoint restore failed: " ++)) return result

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
