-- | The trainable runtime's plain types and their C layouts (plan Stage 3).
--
-- Split out from "Infer.Trainer" so the schedule and the layouts can be tested
-- without linking a GPU engine: this module has no FFI at all, only the wire shapes
-- the C side expects and the values a caller sets.
module Infer.Trainer.Types
  ( -- * Attaching
    AttachOptions(..)
  , defaultAttachOptions
    -- * Teacher forcing
  , ForcedPosition(..)
  , TeacherForcedForward(..)
  , ForwardOutput(..)
    -- * Steps
  , StepPlan(..)
    -- * Layout constants the C structs' offsets are expressed in
  , sizeOfPtr
  , sizeOfCInt
  ) where

import Foreign.C.Types (CInt(..))
import Foreign.Ptr (Ptr, castPtr, nullPtr, plusPtr)
import Foreign.Storable (Storable(..))

-- | Bytes of one pointer: the unit the C structs' byte offsets are expressed in.
sizeOfPtr :: Int
sizeOfPtr = sizeOf (nullPtr :: Ptr ())

sizeOfCInt :: Int
sizeOfCInt = sizeOf (0 :: CInt)

-- | What a caller wants attached. Training state is optional: a rollout-only caller
-- needs the parameter identities without paying for FP32 masters.
data AttachOptions = AttachOptions
  { allocateTrainingState :: Bool
  , frozenRoles           :: [Int]
  } deriving (Eq, Show)

defaultAttachOptions :: AttachOptions
defaultAttachOptions = AttachOptions { allocateTrainingState = False, frozenRoles = [] }

-- The C struct is @{int allocate_training_state; const int *frozen_roles; int
-- frozen_role_count;}@. The roles array is supplied at the call, so 'poke' writes a
-- null pointer member and 'peek' deliberately does not read it back: handing a caller
-- a raw pointer is what this layer exists to avoid.
instance Storable AttachOptions where
  sizeOf _ = 3 * sizeOf (0 :: CInt)
  alignment _ = alignment (0 :: CInt)
  poke ptr options = do
    poke (castPtr ptr) (fromIntegral (fromEnum (allocateTrainingState options)) :: CInt)
    poke (castPtr (ptr `plusPtr` sizeOfPtr)) (nullPtr :: Ptr CInt)
    poke (castPtr (ptr `plusPtr` (2 * sizeOfPtr)))
         (fromIntegral (length (frozenRoles options)) :: CInt)
  peek ptr = do
    flag <- peek (castPtr ptr) :: IO CInt
    count <- peek (castPtr (ptr `plusPtr` (2 * sizeOfPtr))) :: IO CInt
    pure AttachOptions { allocateTrainingState = flag /= 0
                       , frozenRoles = replicate (fromIntegral count) 0 }

-- | One selected position: the query whose logits are used, the label predicted
-- there, and the position those logits were computed at. Layout matches
-- @struct TrainForcedPosition@.
data ForcedPosition = ForcedPosition
  { forcedQuery    :: Int
  , forcedLabel    :: Int
  , forcedPosition :: Int
  } deriving (Eq, Show)

instance Storable ForcedPosition where
  sizeOf _ = 3 * sizeOf (0 :: CInt)
  alignment _ = alignment (0 :: CInt)
  poke ptr (ForcedPosition q l p) = do
    poke (castPtr ptr) (fromIntegral q :: CInt)
    poke (castPtr (ptr `plusPtr` sizeOfCInt)) (fromIntegral l :: CInt)
    poke (castPtr (ptr `plusPtr` (2 * sizeOfCInt))) (fromIntegral p :: CInt)
  peek ptr = do
    q <- peek (castPtr ptr) :: IO CInt
    l <- peek (castPtr (ptr `plusPtr` sizeOfCInt)) :: IO CInt
    p <- peek (castPtr (ptr `plusPtr` (2 * sizeOfCInt))) :: IO CInt
    pure (ForcedPosition (fromIntegral q) (fromIntegral l) (fromIntegral p))

-- | What a forward produced: the selected positions and their natural-log
-- log-probabilities. Per-position logits are not part of the contract: a loss is not
-- allowed to need a [tokens, vocab] tensor.
data TeacherForcedForward = TeacherForcedForward
  { forwardSelected :: [Int]
  , forwardLogProbs :: [Float]
  } deriving (Eq, Show)

-- | The C @struct TrainForwardOutput@: three pointers and a count. A caller fills the
-- buffer pointers; the trainer keeps them opaque.
data ForwardOutput = ForwardOutput
  { foAllLogits :: Ptr ()
  , foLogProbs  :: Ptr ()
  , foSelected  :: Ptr ()
  , foSelectedN :: CInt
  }

instance Storable ForwardOutput where
  sizeOf _ = 4 * sizeOf (nullPtr :: Ptr ())
  alignment _ = alignment (nullPtr :: Ptr ())
  poke ptr out = do
    poke (castPtr ptr) (foAllLogits out)
    poke (castPtr (ptr `plusPtr` sizeOfPtr)) (foLogProbs out)
    poke (castPtr (ptr `plusPtr` (2 * sizeOfPtr))) (foSelected out)
    poke (castPtr (ptr `plusPtr` (3 * sizeOfPtr))) (foSelectedN out)
  peek ptr = do
    allLogits <- peek (castPtr ptr)
    logProbs <- peek (castPtr (ptr `plusPtr` sizeOfPtr))
    selected <- peek (castPtr (ptr `plusPtr` (2 * sizeOfPtr)))
    count <- peek (castPtr (ptr `plusPtr` (3 * sizeOfPtr)))
    pure (ForwardOutput allLogits logProbs selected count)

-- | What a step of a given size retains, and how much of it is GDN chunk-boundary
-- state: the numbers the plan's full-vs-truncated BPTT choice is made from.
data StepPlan = StepPlan
  { stepSavedValues      :: Int
  , stepGdnStateElements :: Int
  } deriving (Eq, Show)
