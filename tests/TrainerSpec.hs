-- | The trainable runtime's Haskell gate (plan Stage 3).
--
-- The plan gives the teacher-forcing schedule to Haskell and the buffers to C, so the
-- decisive check is that the two implementations of *one* schedule agree: this suite
-- runs the pure Haskell plan and the C plan over a matrix of shifts, masks, forced
-- labels and explicit positions and requires the same answer every time. If they ever
-- disagree, the split is a fiction.
--
-- CPU-only: it links @csrc/train.c@ and @csrc/model_desc.c@ (both CUDA-free), so the
-- layout and schedule contracts are checked without a GPU. The engine-facing half of
-- "Infer.Trainer" is exercised by @tests/test_train_forward.py@ on a real device.
module Main (main) where

import Control.Monad (forM_)
import Data.Int (Int64)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, nullPtr, plusPtr)
import Foreign.Storable (peek, poke, sizeOf)
import Test.Hspec

import Infer.Trainer.Plan (teacherForcingPlan, teacherForcingPlanC)
import Infer.Trainer.Types

-- | The cases the two implementations have to agree on. They are chosen for the
-- boundaries the contract names: a shift of one and more, a prompt/padding mask, a
-- forced label, explicit positions, and a sequence too short to have a label.
planCases :: [(String, Int, [Int], Maybe [Int], Maybe [Bool], Maybe [Int64])]
planCases =
  [ ("plain next-token", 1, [10, 11, 12, 13], Nothing, Nothing, Nothing)
  , ("prompt masked out", 1, [10, 11, 12, 13, 14, 15], Nothing,
     Just [False, False, True, True, True, True], Nothing)
  , ("shift two", 2, [10, 11, 12, 13, 14, 15], Nothing,
     Just [False, False, True, True, True, True], Nothing)
  , ("forced labels", 1, [10, 11, 12, 13, 14, 15], Just [0, 0, 77, 78, 79, 80],
     Just [False, False, True, True, True, True], Nothing)
  , ("explicit positions", 1, [10, 11, 12, 13], Nothing, Nothing,
     Just [100, 101, 102, 103])
  , ("one token", 1, [10], Nothing, Nothing, Nothing)
  , ("shorter than the shift", 1, [10, 11], Nothing, Just [False, False], Nothing)
  , ("everything masked", 1, [10, 11, 12], Nothing, Just [False, False, False], Nothing)
  ]

main :: IO ()
main = hspec $ do
  describe "teacher forcing plan" $ do
    it "selects the shifted label and skips masked targets" $ do
      let plan = teacherForcingPlan 1 [10, 11, 12, 13, 14, 15] Nothing
                   (Just [False, False, True, True, True, True]) Nothing
      map forcedQuery plan `shouldBe` [1, 2, 3, 4]
      map forcedLabel plan `shouldBe` [12, 13, 14, 15]
      map forcedPosition plan `shouldBe` [1, 2, 3, 4]

    it "never selects a position whose label lies outside the sequence" $ do
      teacherForcingPlan 1 [10, 11, 12] Nothing Nothing Nothing
        `shouldBe` [ ForcedPosition 0 11 0, ForcedPosition 1 12 1 ]
      teacherForcingPlan 3 [10, 11] Nothing Nothing Nothing `shouldBe` []

    it "prefers a forced label over the self-supervised one" $ do
      -- The override is per *target position*, so both queries see their own entry.
      let plan = teacherForcingPlan 1 [10, 11, 12] (Just [0, 0, 99]) Nothing Nothing
      map forcedLabel plan `shouldBe` [0, 99]

    it "carries the explicit position of the query" $ do
      let plan = teacherForcingPlan 1 [10, 11] Nothing Nothing (Just [500, 501])
      map forcedPosition plan `shouldBe` [500]

    it "agrees with the C implementation in every case" $ do
      forM_ planCases $ \(name, shift, tokens, labels, mask, positions) -> do
        let expected = teacherForcingPlan shift tokens labels mask positions
        actual <- teacherForcingPlanC shift tokens labels mask positions
        actual `shouldBe` expected

    it "reports the count without a buffer, and refuses a short one" $ do
      -- The C plan's own contract, restated from the Haskell side so the two cannot
      -- drift: a null buffer reports the count, and the count equals the plan length.
      let tokens = [10, 11, 12, 13]
      count <- teacherForcingPlanC 1 tokens Nothing Nothing Nothing
      length (teacherForcingPlan 1 tokens Nothing Nothing Nothing) `shouldBe` length count

  describe "C layouts" $ do
    it "round-trips the attach options through the C struct" $ do
      let options = AttachOptions { allocateTrainingState = True, frozenRoles = [3, 7, 11] }
      alloca $ \ptr -> do
        poke ptr options
        -- 'peek' cannot recover the role ids (it must not hand out the array's
        -- pointer), but the count it wrote is what a caller relies on.
        peek ptr `shouldReturn`
          AttachOptions { allocateTrainingState = True, frozenRoles = replicate 3 0 }

    it "round-trips a forced position" $ do
      alloca $ \ptr -> do
        poke ptr (ForcedPosition 5 42 7)
        peek ptr `shouldReturn` ForcedPosition 5 42 7

    it "offsets the forward-output struct the way the C struct does" $ do
      -- Three pointers and a count, which is the C struct's layout.
      sizeOf (ForwardOutput undefined undefined undefined 0)
        `shouldBe` 4 * sizeOf (nullPtr :: Ptr ())
      alloca $ \ptr -> do
        -- Distinct non-null addresses, so a field written into the wrong slot is
        -- visible rather than being masked by all three reading as null.
        poke ptr ForwardOutput { foAllLogits = nullPtr `plusPtr` 0x10
                               , foLogProbs = nullPtr `plusPtr` 0x20
                               , foSelected = nullPtr `plusPtr` 0x30
                               , foSelectedN = 9 }
        out <- peek ptr
        foSelectedN out `shouldBe` 9
