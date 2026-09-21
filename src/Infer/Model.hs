-- | Model architecture definition: layer types and the 64-layer structure.
--
-- Qwen3.8-27B is a hybrid architecture: every 4th layer is full attention,
-- the rest are GatedDeltaNet (GDN). This module defines the layer ADT and
-- constructs the full model description.
module Infer.Model
  ( LayerType(..)
  , Layer(..)
  , ModelDef(..)
  , qwen38_27bModel
  , isAttentionLayer
  , isGdnLayer
  , attentionLayerIndices
  , gdnLayerIndices
  ) where

import Infer.Config

-- | The two layer types in the hybrid architecture.
data LayerType
  = FullAttention  -- ^ Standard multi-head attention with GQA, RoPE, output gate
  | GatedDeltaNet  -- ^ GatedDeltaNet: causal conv1d + delta rule + gated norm
  deriving (Eq, Show, Enum, Bounded)

-- | A single transformer layer with its type and global index.
data Layer = Layer
  { layerIndex :: !Int       -- ^ Global layer index (0..63)
  , layerType  :: !LayerType
  , layerDevice :: !Int      -- ^ Assigned GPU device ordinal
  } deriving (Eq, Show)

-- | Complete model definition.
data ModelDef = ModelDef
  { mdConfig  :: ModelConfig
  , mdLayers  :: [Layer]     -- ^ All 64 layers in order
  , mdPartition :: GpuPartition
  }

-- | Construct the Qwen3.8-27B model definition with a given GPU partition.
qwen38_27bModel :: [Int] -> ModelDef
qwen38_27bModel devices = ModelDef
  { mdConfig    = cfg
  , mdLayers    = layers
  , mdPartition = partition
  }
  where
    cfg = qwen38_27bConfig
    partition = computePartition (mcNumLayers cfg) devices
    layerDevs = gpLayerDevices partition
    layers = zipWith mkLayer [0 .. mcNumLayers cfg - 1] layerDevs
    mkLayer i dev = Layer
      { layerIndex  = i
      , layerType   = if isAttentionIndex cfg i then FullAttention else GatedDeltaNet
      , layerDevice = dev
      }

-- | A layer is full attention if (index + 1) is divisible by the interval.
-- Layer indices are 0-based: layers 3, 7, 11, ..., 63 are attention.
isAttentionIndex :: ModelConfig -> Int -> Bool
isAttentionIndex cfg i = (i + 1) `mod` mcFullAttnInterval cfg == 0

isAttentionLayer :: Layer -> Bool
isAttentionLayer l = layerType l == FullAttention

isGdnLayer :: Layer -> Bool
isGdnLayer l = layerType l == GatedDeltaNet

attentionLayerIndices :: ModelDef -> [Int]
attentionLayerIndices md = [layerIndex l | l <- mdLayers md, isAttentionLayer l]

gdnLayerIndices :: ModelDef -> [Int]
gdnLayerIndices md = [layerIndex l | l <- mdLayers md, isGdnLayer l]
