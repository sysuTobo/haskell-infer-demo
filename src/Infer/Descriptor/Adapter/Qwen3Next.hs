-- | Family adapter for Qwen3-Next: the hybrid GatedDeltaNet + full-attention
-- stack with a sparse FFN and a shared expert.
--
-- Differences from Qwen3.5's adapter, all verified against the published
-- checkpoint and this repository's synthetic one:
--
--   * the GDN projections arrive fused (`in_proj_qkvz` = q+k+v+z rows and
--     `in_proj_ba` = b then a), so the descriptor declares the fused roles;
--   * the FFN is sparse with a single shared expert whose output is scaled by a
--     per-token sigmoid gate (`mlp.shared_expert_gate`);
--   * attention carries the same fused output gate as Qwen3.5 (q_proj is
--     2 * heads * head_dim wide, interleaved per head).
module Infer.Descriptor.Adapter.Qwen3Next
  ( qwen3NextDescriptorFromDir
  , qwen3NextDescriptorFromConfig
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeFileStrict)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Control.Monad ((>=>))
import Data.List (nub)
import Data.Maybe (catMaybes, fromMaybe, listToMaybe)
import qualified Data.Text as T
import Text.Read (readMaybe)

import Infer.Descriptor
import Infer.Descriptor.Adapter.Json

-- | Build a descriptor from a model directory (@config.json@ + optional
-- @tokenizer_config.json@).
qwen3NextDescriptorFromDir :: FilePath -> IO (Either String Descriptor)
qwen3NextDescriptorFromDir dir = do
  config <- eitherDecodeFileStrict (dir ++ "/config.json")
  case config of
    Left err -> pure (Left ("cannot parse " ++ dir ++ "/config.json: " ++ err))
    Right value -> do
      tokenizer <- readJsonValue (dir ++ "/tokenizer_config.json")
      pure $ do
        cfg <- textConfigOf value
        qwen3NextDescriptorFromConfig cfg tokenizer

-- | Build a descriptor from the (flat) config object.
qwen3NextDescriptorFromConfig :: Object -> Maybe Value -> Either String Descriptor
qwen3NextDescriptorFromConfig cfg tokenizer = do
  numLayers <- needInt cfg "num_hidden_layers"
  hidden <- needInt cfg "hidden_size"
  vocab <- needInt cfg "vocab_size"
  heads <- needInt cfg "num_attention_heads"
  kvHeads <- needInt cfg "num_key_value_heads"
  headDim <- needInt cfg "head_dim"
  kHeads <- needInt cfg "linear_num_key_heads"
  vHeads <- needInt cfg "linear_num_value_heads"
  gdnHeadDim <- needInt cfg "linear_key_head_dim"
  let partial = fromMaybe 1.0 (lookupDouble cfg "partial_rotary_factor")
      rotaryDim = round (partial * fromIntegral headDim)
      valueDim = vHeads * gdnHeadDim
      convDim = 2 * kHeads * gdnHeadDim + valueDim
      interval = fromMaybe 4 (lookupInt cfg "full_attention_interval")
      sparseStep = max 1 (fromMaybe 1 (lookupInt cfg "decoder_sparse_step"))
      denseLayers = fromMaybe [] (lookupIntList cfg "mlp_only_layers")
      numExperts = fromMaybe 0 (lookupInt cfg "num_experts")
      sharedIntermediate = fromMaybe 0 (lookupInt cfg "shared_expert_intermediate_size")
  mixers <- case lookupValue cfg "layer_types" of
    Just (Array values) -> traverse (asText >=> mixerOf) (foldr (:) [] values)
    _ -> pure [ if (i + 1) `mod` interval == 0 then MFullAttention else MGatedDeltaNet
              | i <- [0 .. numLayers - 1] ]
  pure Descriptor
    { dVersion = descVersion
    , dFamily = "qwen3_next"
    , dModelType = fromMaybe "qwen3_next" (lookupText cfg "model_type")
    , dNumLayers = numLayers
    , dHiddenSize = hidden
    , dIntermediateSize = fromMaybe (4 * hidden) (lookupInt cfg "intermediate_size")
    , dVocabSize = vocab
    , dRmsEps = fromMaybe 1e-6 (lookupDouble cfg "rms_norm_eps")
    , dMaxPositionEmbeddings = fromMaybe 4096 (lookupInt cfg "max_position_embeddings")
    , dMaxSeqLen = min 4096 (fromMaybe 4096 (lookupInt cfg "max_position_embeddings"))
    , dNumHeads = heads
    , dNumKvHeads = kvHeads
    , dHeadDim = headDim
    , dRotaryDim = rotaryDim
    , dRotaryTheta = fromMaybe 1e7 (lookupDouble cfg "rope_theta")
    , dNormStyle = "plain"
    , dAttnQkNorm = True
    , dAttnOutputGate = True
    , dQGateInterleave = True
    , dGdnConvDim = convDim
    , dGdnValueDim = valueDim
    , dGdnNumVHeads = vHeads
    , dGdnNumKHeads = kHeads
    , dGdnHeadDim = gdnHeadDim
    , dGdnConvKernel = fromMaybe 4 (lookupInt cfg "linear_conv_kernel_dim")
    , dFlaChunkSize = 64
    , dMaxChunk = 128
    , dMoeNumExperts = numExperts
    , dMoeTopK = fromMaybe 1 (lookupInt cfg "num_experts_per_tok")
    , dMoeIntermediateSize = fromMaybe 0 (lookupInt cfg "moe_intermediate_size")
    , dMoeRouterScoring = "softmax"
    , dMoeNormTopkProb = fromMaybe False (lookupBool cfg "norm_topk_prob")
    , dMoeNumSharedExperts = if sharedIntermediate > 0 then 1 else 0
    , dMoeSharedIntermediateSize = sharedIntermediate
    , dMoeRoutedScalingFactor = 1.0
    , dMoeSharedGateScalar = sharedIntermediate > 0
    , dEosTokens = eosTokens cfg tokenizer
    , dLayerMixers = mixers
    , dLayerFfns =
        [ if numExperts > 0 && i `notElem` denseLayers && (i + 1) `mod` sparseStep == 0
            then FMoe else FDense
        | i <- [0 .. numLayers - 1] ]
    , dRoleTemplates = weightRoles
    }
  where
    mixerOf :: String -> Either String MixerKind
    mixerOf "full_attention" = Right MFullAttention
    mixerOf "linear_attention" = Right MGatedDeltaNet
    mixerOf other = Left ("unsupported layer type in config: " ++ show other)

    asText :: Value -> Either String String
    asText (String t) = Right (T.unpack t)
    asText _ = Left "layer_types entries must be strings"

textConfigOf :: Value -> Either String Object
textConfigOf value = do
  top <- maybe (Left "config.json must be an object") Right (valueObject value)
  pure $ fromMaybe top (lookupValue top "text_config" >>= valueObject)

eosTokens :: Object -> Maybe Value -> [Int]
eosTokens cfg tokenizer = nub (catMaybes [fromTokenizer, lookupInt cfg "eos_token_id"])
  where
    fromTokenizer = do
      tok <- tokenizer
      tokObj <- valueObject tok
      eosName <- lookupText tokObj "eos_token"
      decoder <- lookupValue tokObj "added_tokens_decoder" >>= valueObject
      listToMaybe [ i
                  | (key, entry) <- KM.toList decoder
                  , Just i <- [readMaybe (Key.toString key)]
                  , lookupText' entry "content" == Just eosName
                  ]

-- | Qwen3-Next weight names: Qwen3.5's attention and norms, the fused GDN
-- projections, and a sparse FFN with a shared expert.
weightRoles :: [(Role, String)]
weightRoles =
  [ (REmbed, "model.embed_tokens.weight")
  , (RLmHead, "lm_head.weight")
  , (RFinalNorm, "model.norm.weight")
  , (RInputNorm, layer "input_layernorm.weight")
  , (RPostNorm, layer "post_attention_layernorm.weight")
  , (RAttnQ, layer "self_attn.q_proj.weight")
  , (RAttnK, layer "self_attn.k_proj.weight")
  , (RAttnV, layer "self_attn.v_proj.weight")
  , (RAttnO, layer "self_attn.o_proj.weight")
  , (RAttnQNorm, layer "self_attn.q_norm.weight")
  , (RAttnKNorm, layer "self_attn.k_norm.weight")
  , (RGdnQkvz, layer "linear_attn.in_proj_qkvz.weight")
  , (RGdnBa, layer "linear_attn.in_proj_ba.weight")
  , (RGdnConv1d, layer "linear_attn.conv1d.weight")
  , (RGdnDtBias, layer "linear_attn.dt_bias")
  , (RGdnALog, layer "linear_attn.A_log")
  , (RGdnOut, layer "linear_attn.out_proj.weight")
  , (RGdnNorm, layer "linear_attn.norm.weight")
  , (RMoeRouter, layer "mlp.gate.weight")
  , (RMoeExpertGate, layer "mlp.experts.%e.gate_proj.weight")
  , (RMoeExpertUp, layer "mlp.experts.%e.up_proj.weight")
  , (RMoeExpertDown, layer "mlp.experts.%e.down_proj.weight")
  , (RMoeSharedGate, layer "mlp.shared_expert.gate_proj.weight")
  , (RMoeSharedUp, layer "mlp.shared_expert.up_proj.weight")
  , (RMoeSharedDown, layer "mlp.shared_expert.down_proj.weight")
  , (RMoeSharedGateScalar, layer "mlp.shared_expert_gate.weight")
  ]
  where
    layer suffix = "model.layers.%d." ++ suffix
