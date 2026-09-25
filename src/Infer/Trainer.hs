{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | The trainable runtime's Haskell side (docs/plan-numeric-contract.md, Stage 3).
--
-- The plan's split is explicit: "Haskell owns the fixed model traversal, training
-- schedule and typed opaque handles; C/CUDA owns allocations, streams and kernel
-- execution. No raw CUDA pointers become user-facing Haskell values."
--
-- So the *schedule* lives here ("Infer.Trainer.Plan", cross-checked against the C
-- implementation by the test suite) and the *buffers* stay in C behind opaque
-- handles: a caller passes a 'TrainerStore' or a 'TrainerStep', never a pointer.
--
-- Two call classes are distinguished deliberately, because the plan requires it: the
-- calls that can run for a while -- attaching a store, a teacher-forced forward, a
-- publication, a step -- are imported @safe@ so another Haskell thread can run while
-- they do, and the small pure ones are @unsafe@, where a wrapper would only cost a
-- context switch. The engine is same-OS-thread today, so these are called from its
-- owner thread; a future execution worker is what makes the distinction load-bearing.
module Infer.Trainer
  ( -- * Re-exported plain types and the schedule
    module Infer.Trainer.Types
  , module Infer.Trainer.Plan
    -- * Typed handles
  , TrainerStore
  , TrainerStep
    -- * Attaching
  , attachStore
    -- * Updates
  , beginUpdate
  , writeMaster
  , publishUpdate
  , endUpdate
    -- * Steps
  , planStep
  , beginStep
  , endStep
    -- * Teacher-forced forward
  , forwardTeacherForced
    -- * Errors
  , engineError
  ) where

import Data.Int (Int64)
import Data.Word (Word8)
import Foreign.C.String (CString, peekCString)
import Foreign.C.Types (CInt(..))
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray, peekArray, withArray)
import Foreign.Ptr (Ptr, castPtr, nullPtr, plusPtr)
import Foreign.Storable (peek, poke)

import Infer.FFI.Engine (EngineHandle)
import Infer.Trainer.Plan
import Infer.Trainer.Types

-- | An opaque parameter store owned by an engine. The raw pointer stays inside this
-- module, so a caller cannot mistake it for something it may dereference or free.
newtype TrainerStore = TrainerStore (Ptr ())

-- | An opaque training step.
newtype TrainerStep = TrainerStep (Ptr ())

instance Show TrainerStore where
  show _ = "<trainer store>"

instance Show TrainerStep where
  show _ = "<trainer step>"

-- -----------------------------------------------------------------------
-- Raw imports
-- -----------------------------------------------------------------------

foreign import ccall safe "engine.h engine_train_attach"
  c_engine_train_attach :: Ptr EngineHandle -> Ptr AttachOptions -> IO (Ptr ())

foreign import ccall safe "engine.h engine_train_begin_update"
  c_engine_train_begin_update :: Ptr () -> IO CInt

foreign import ccall safe "engine.h engine_train_write_master"
  c_engine_train_write_master :: Ptr () -> CInt -> Ptr Float -> IO CInt

foreign import ccall safe "engine.h engine_train_publish"
  c_engine_train_publish :: Ptr () -> IO CInt

foreign import ccall safe "engine.h engine_train_end_update"
  c_engine_train_end_update :: Ptr () -> IO CInt

foreign import ccall unsafe "engine.h engine_train_step_plan"
  c_engine_train_step_plan :: Ptr () -> CInt -> CInt -> Ptr CInt -> Ptr Int64 -> IO CInt

foreign import ccall safe "engine.h engine_train_step_begin"
  c_engine_train_step_begin :: Ptr () -> CInt -> CInt -> Ptr (Ptr ()) -> IO CInt

foreign import ccall safe "engine.h engine_train_step_end"
  c_engine_train_step_end :: Ptr () -> IO CInt

foreign import ccall safe "engine.h engine_train_forward"
  c_engine_train_forward
    :: Ptr () -> Ptr CInt -> CInt -> Ptr Int64 -> Ptr CInt -> Ptr Word8 -> CInt -> Ptr ()
    -> IO CInt

foreign import ccall unsafe "engine.h engine_last_error"
  c_engine_last_error :: IO CString

-- | The last engine error, as text.
engineError :: IO String
engineError = c_engine_last_error >>= peekCString

-- -----------------------------------------------------------------------
-- Attaching and updates
-- -----------------------------------------------------------------------

-- | Attach a parameter store over this engine's loaded weights, or 'Nothing' with
-- 'engineError' set. Calling it twice returns the same store.
attachStore :: Ptr EngineHandle -> AttachOptions -> IO (Maybe TrainerStore)
attachStore engine options =
  withAttachOptions options $ \cOptions -> do
    store <- c_engine_train_attach engine cOptions
    if store == nullPtr then pure Nothing else pure (Just (TrainerStore store))

-- The roles array has to outlive the call, so the marshalling lives here rather than
-- in the Storable instance (which the test suite checks on its own).
withAttachOptions :: AttachOptions -> (Ptr AttachOptions -> IO a) -> IO a
withAttachOptions options f =
  -- The element type is annotated: without it the marshalled array's type is
  -- ambiguous, and an ambiguous 'Storable' would be resolved by luck rather than by
  -- the C layout.
  withArray [fromIntegral role :: CInt | role <- frozenRoles options] $ \roles ->
    alloca $ \ptr -> do
      -- The instance writes the flag, a null array pointer and the *real* count; the
      -- array itself is installed here, where its lifetime is the call.
      poke ptr options
      poke (castPtr (ptr `plusPtr` sizeOfPtr)) roles
      f ptr

-- | Open the exclusive update window. Fails while a context or a step borrows the
-- current version: a reader must never see a half-updated parameter set.
beginUpdate :: TrainerStore -> IO (Either String ())
beginUpdate (TrainerStore store) = resultOf (c_engine_train_begin_update store)

-- | Upload FP32 master values for one logical parameter. Requires an open window and a
-- parameter whose training state was allocated (a frozen one has none).
writeMaster :: TrainerStore -> Int -> [Float] -> IO (Either String ())
writeMaster (TrainerStore store) logical values =
  withArray values $ \ptr ->
    resultOf (c_engine_train_write_master store (fromIntegral logical) ptr)

-- | Publish: cast every master into the compute weight of every reader, bump the
-- version, refresh every derived copy, and close the window. Refuses to close while a
-- derived copy is stale, so "refresh gdn_norm_f32, not just its BF16 source" cannot be
-- forgotten.
publishUpdate :: TrainerStore -> IO (Either String ())
publishUpdate (TrainerStore store) = resultOf (c_engine_train_publish store)

-- | Close the window without publishing.
endUpdate :: TrainerStore -> IO (Either String ())
endUpdate (TrainerStore store) = resultOf (c_engine_train_end_update store)

resultOf :: IO CInt -> IO (Either String ())
resultOf action = do
  code <- action
  if code == 0 then pure (Right ()) else Left <$> engineError

-- -----------------------------------------------------------------------
-- Steps
-- -----------------------------------------------------------------------

-- | What a step of this size would retain, without creating it.
planStep :: Ptr EngineHandle -> Int -> Int -> IO (Maybe StepPlan)
planStep engine tokens chunks =
  alloca $ \countPtr ->
    alloca $ \elementsPtr -> do
      code <- c_engine_train_step_plan engine (fromIntegral tokens) (fromIntegral chunks)
                countPtr elementsPtr
      if code /= 0
        then pure Nothing
        else do
          count <- peek countPtr
          elements <- peek elementsPtr
          pure (Just (StepPlan (fromIntegral count) (fromIntegral elements)))

-- | Begin a step: retain the activations and GDN chunk-boundary states the backward
-- will consume. The step is a reader, so an update is refused until it ends.
beginStep :: Ptr EngineHandle -> Int -> Int -> IO (Maybe TrainerStep)
beginStep engine tokens chunks =
  alloca $ \ptrPtr -> do
    code <- c_engine_train_step_begin engine (fromIntegral tokens) (fromIntegral chunks) ptrPtr
    if code /= 0 then pure Nothing else Just . TrainerStep <$> peek ptrPtr

endStep :: TrainerStep -> IO (Either String ())
endStep (TrainerStep step) = resultOf (c_engine_train_step_end step)

-- -----------------------------------------------------------------------
-- Teacher-forced forward
-- -----------------------------------------------------------------------

-- | Run one teacher-forced forward over a single sequence from position 0 (reset
-- between steps) and return the selected positions with their natural-log
-- log-probabilities. The engine evaluates the LM head row by row on the device, so no
-- [tokens, vocab] tensor is ever materialised, and independent sequences are never
-- flattened into one causal sequence.
forwardTeacherForced
  :: Ptr EngineHandle
  -> [Int]            -- ^ token ids
  -> Int              -- ^ shift, normally 1
  -> Maybe [Bool]     -- ^ optional mask; False marks a prompt or padding position
  -> IO (Either String TeacherForcedForward)
forwardTeacherForced engine tokens shift mask =
  withArray [fromIntegral token :: CInt | token <- tokens] $ \cTokens ->
    withMaybeByteArray mask $ \cMask ->
      allocaArray (length tokens) $ \(logProbs :: Ptr Float) ->
        allocaArray (length tokens) $ \(selected :: Ptr CInt) ->
          alloca $ \outPtr -> do
            poke outPtr ForwardOutput { foAllLogits = nullPtr
                                      , foLogProbs = castPtr logProbs
                                      , foSelected = castPtr selected
                                      , foSelectedN = 0 }
            code <- c_engine_train_forward engine cTokens (fromIntegral (length tokens))
                      nullPtr nullPtr cMask (fromIntegral shift) (castPtr outPtr)
            if code /= 0
              then Left <$> engineError
              else do
                finished <- peek outPtr
                let count = fromIntegral (foSelectedN finished)
                picks <- peekArray count selected
                probs <- peekArray count logProbs
                pure (Right TeacherForcedForward { forwardSelected = map fromIntegral picks
                                                 , forwardLogProbs = probs })
  where
    withMaybeByteArray :: Maybe [Bool] -> (Ptr Word8 -> IO a) -> IO a
    withMaybeByteArray Nothing f = f (nullPtr :: Ptr Word8)
    withMaybeByteArray (Just xs) f = withArray [fromIntegral (fromEnum b) :: Word8 | b <- xs] f
