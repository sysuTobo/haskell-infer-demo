-- | The teacher-forcing schedule (plan Stage 3).
--
-- The plan gives the schedule to Haskell and the buffers to C, so the schedule is
-- implemented here in pure Haskell *and* the C implementation of the same plan is
-- imported for the test suite to compare against. One schedule with two
-- implementations, checked against each other, is the only way that split stays
-- honest: the Haskell side owns the traversal, and the C side is not trusted to mean
-- the same thing by construction.
module Infer.Trainer.Plan
  ( teacherForcingPlan
  , teacherForcingPlanC
  , lastError
  ) where

import Data.Int (Int64)
import Data.Word (Word8)
import Foreign.C.String (CString, peekCString)
import Foreign.C.Types (CInt(..))
import Foreign.Marshal.Array (allocaArray, peekArray, withArray)
import Foreign.Ptr (Ptr, castPtr, nullPtr)

import Infer.Trainer.Types (ForcedPosition(..))

-- | The teacher-forcing plan: the label of query @t@ is token @t + shift@; a masked
-- target is skipped rather than zero-weighted; the last @shift@ positions predict
-- nothing inside this sequence; an explicit @labels@ array overrides the
-- self-supervised target. One sequence at a time: independent sequences are never
-- flattened into one causal sequence.
teacherForcingPlan
  :: Int                 -- ^ shift, normally 1
  -> [Int]               -- ^ token ids
  -> Maybe [Int]         -- ^ optional forced labels, parallel to the token ids
  -> Maybe [Bool]        -- ^ optional mask; False marks a prompt or padding position
  -> Maybe [Int64]       -- ^ optional positions
  -> [ForcedPosition]
teacherForcingPlan shift tokens labels mask positions =
  [ ForcedPosition query label position
  | query <- [0 .. length tokens - shift - 1]
  , let target = query + shift
  , maybe True (!! target) mask
  , let label = maybe (tokens !! target) (!! target) labels
        position = maybe (fromIntegral query) (fromIntegral . (!! query)) positions
  ]

-- The argument order is the C header's, and the leading token count is easy to drop:
-- the test suite caught exactly that, because a pointer passed where an int belongs
-- makes the C plan return an error rather than a wrong answer.
foreign import ccall unsafe "train.h train_plan_teacher_forcing"
  c_train_plan_teacher_forcing
    :: CInt -> Ptr CInt -> Ptr CInt -> Ptr Word8 -> Ptr Int64 -> CInt -> Ptr () -> CInt
    -> IO CInt

foreign import ccall unsafe "train.h train_last_error"
  c_train_last_error :: IO CString

-- | The trainer's last error, as text.
lastError :: IO String
lastError = c_train_last_error >>= peekCString

-- | The same plan computed by C, for the test suite's cross-check.
--
-- The optional arrays are marshalled as their *defaults* rather than as null
-- pointers: a mask of all True, labels equal to the token ids and positions 0..n-1 are
-- exactly what the C plan does when those arguments are absent, so the two paths agree
-- without a single polymorphic null-pointer helper.
teacherForcingPlanC
  :: Int -> [Int] -> Maybe [Int] -> Maybe [Bool] -> Maybe [Int64] -> IO [ForcedPosition]
teacherForcingPlanC shift tokens labels mask positions =
  withArray (map fromIntegral tokens) $ \cTokens ->
    withArray (map fromIntegral labels') $ \cLabels ->
      withArray (map (fromIntegral . fromEnum) mask') $ \cMask ->
        withArray positions' $ \cPositions -> do
          -- A first call with no buffer asks for the number of selected positions.
          requiredRaw <- c_train_plan_teacher_forcing (fromIntegral (length tokens)) cTokens
                           cLabels cMask cPositions (fromIntegral shift) nullPtr 0
          let required = fromIntegral requiredRaw :: Int
          if required <= 0
            then pure []
            else allocaArray required $ \out -> do
              writtenRaw <- c_train_plan_teacher_forcing (fromIntegral (length tokens)) cTokens
                              cLabels cMask cPositions (fromIntegral shift) (castPtr out)
                              (fromIntegral required)
              let written = fromIntegral writtenRaw :: Int
              if written <= 0 then pure [] else peekArray written out
  where
    count = length tokens
    labels' = maybe tokens id labels
    mask' = maybe (replicate count True) id mask
    positions' = maybe (map fromIntegral [0 .. count - 1]) id positions
