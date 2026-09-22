-- | Model definition: the layer plan derived from a descriptor and a placement.
--
-- A layer is a (mixer, feed-forward) pair plus the device that runs it. Nothing
-- here is architecture-specific: the mixer/FFN kinds come from the descriptor.
module Infer.Model
  ( Layer(..)
  , ModelDef(..)
  , modelDef
  , isAttentionLayer
  , isGdnLayer
  , attentionLayerIndices
  , gdnLayerIndices
  ) where

import Infer.Descriptor
import Infer.Placement

-- | A single transformer layer.
data Layer = Layer
  { layerIndex :: !Int
  , layerMixer :: !MixerKind   -- ^ token-mixing sublayer
  , layerFfn :: !FfnKind       -- ^ feed-forward sublayer
  , layerDevice :: !Int        -- ^ Device ordinal that owns the weights and state
  } deriving (Eq, Show)

-- | Complete model definition: descriptor + placement + per-layer plan.
data ModelDef = ModelDef
  { mdDescriptor :: !Descriptor
  , mdPlacement :: !Placement
  , mdLayers :: [Layer]
  } deriving (Eq, Show)

-- | Build the layer plan for a descriptor and policy on the given devices.
modelDef :: Descriptor -> Policy -> [Int] -> Either String ModelDef
modelDef desc policy devices = do
  place <- placement desc policy devices
  let indices = zip3 [0 ..] (dLayerMixers desc) (dLayerFfns desc)
      layers =
        [ Layer i mixer ffn (plLayerDevices place !! i)
        | (i, mixer, ffn) <- indices
        ]
  pure ModelDef { mdDescriptor = desc, mdPlacement = place, mdLayers = layers }

isAttentionLayer :: Layer -> Bool
isAttentionLayer l = layerMixer l == MFullAttention

isGdnLayer :: Layer -> Bool
isGdnLayer l = layerMixer l == MGatedDeltaNet

attentionLayerIndices :: ModelDef -> [Int]
attentionLayerIndices md = [layerIndex l | l <- mdLayers md, isAttentionLayer l]

gdnLayerIndices :: ModelDef -> [Int]
gdnLayerIndices md = [layerIndex l | l <- mdLayers md, isGdnLayer l]
