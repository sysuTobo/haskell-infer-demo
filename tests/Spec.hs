-- | Haskell test suite for the inference framework.
-- Tests pure logic (config, model definition, partitioning) without GPU.
module Main (main) where

import Test.Hspec

import Infer.Config
import Infer.Model

main :: IO ()
main = hspec $ do
  describe "Config" $ do
    it "has correct Qwen3.8-27B dimensions" $ do
      let cfg = qwen38_27bConfig
      mcNumLayers cfg `shouldBe` 64
      mcHiddenSize cfg `shouldBe` 5120
      mcVocabSize cfg `shouldBe` 248320
      mcNumHeads cfg `shouldBe` 24
      mcNumKvHeads cfg `shouldBe` 4
      mcHeadDim cfg `shouldBe` 256
      mcRotaryDim cfg `shouldBe` 64
      mcFullAttnInterval cfg `shouldBe` 4

    it "computes balanced 2-GPU partition" $ do
      let p = computePartition 64 [0, 1]
      length (gpLayerDevices p) `shouldBe` 64
      length (filter (== 0) (gpLayerDevices p)) `shouldBe` 32
      length (filter (== 1) (gpLayerDevices p)) `shouldBe` 32

    it "computes balanced 4-GPU partition" $ do
      let p = computePartition 64 [0, 1, 2, 3]
      length (gpLayersPerDev p) `shouldBe` 4
      all (\ls -> length ls == 16) (gpLayersPerDev p) `shouldBe` True

    it "handles uneven partition (3 GPUs)" $ do
      let p = computePartition 64 [0, 1, 2]
      -- 64 / 3 = 21 remainder 1 → first device gets 22
      length (gpLayersPerDev p !! 0) `shouldBe` 22
      length (gpLayersPerDev p !! 1) `shouldBe` 21
      length (gpLayersPerDev p !! 2) `shouldBe` 21

  describe "Model" $ do
    it "identifies attention layers correctly" $ do
      let cfg = qwen38_27bConfig
      -- Layers 3, 7, 11, ..., 63 are attention (0-indexed, every 4th)
      isAttentionIndex cfg 3 `shouldBe` True
      isAttentionIndex cfg 7 `shouldBe` True
      isAttentionIndex cfg 63 `shouldBe` True
      isAttentionIndex cfg 0 `shouldBe` False
      isAttentionIndex cfg 1 `shouldBe` False
      isAttentionIndex cfg 2 `shouldBe` False

    it "has 16 attention and 48 GDN layers" $ do
      let md = qwen38_27bModel [0, 1]
      length (attentionLayerIndices md) `shouldBe` 16
      length (gdnLayerIndices md) `shouldBe` 48

    it "assigns layers to devices in order" $ do
      let md = qwen38_27bModel [0, 1]
      all (\l -> layerDevice l == 0) (take 32 (mdLayers md)) `shouldBe` True
      all (\l -> layerDevice l == 1) (drop 32 (mdLayers md)) `shouldBe` True
