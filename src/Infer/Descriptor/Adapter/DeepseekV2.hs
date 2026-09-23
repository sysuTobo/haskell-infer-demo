-- | Family adapter for DeepSeek-V2 (MLA + fine-grained MoE): `deepseek_v2`.
--
-- Layout facts taken from the checkpoint's config and index, not from the
-- modeling code (the 4c lesson):
--   * attention is MLA: one `q_proj` [H*(nope+rope), H] when @q_lora_rank@ is
--     null, `kv_a_proj_with_mqa` [kv_lora_rank+rope, H] plus `kv_a_layernorm`
--     [kv_lora_rank] and `kv_b_proj` [H*(nope+v), kv_lora_rank];
--   * the first @first_k_dense_replace@ layers have a dense MLP, the rest route
--     through `mlp.gate.weight` into `mlp.experts.%e.*` with
--     `mlp.shared_experts.*` always on;
--   * @norm_topk_prob@ and @routed_scaling_factor@ come from the config.
module Infer.Descriptor.Adapter.DeepseekV2
  ( deepseekV2DescriptorFromDir
  , deepseekV2DescriptorFromConfig
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeFileStrict)
import Data.Maybe (fromMaybe)

import Infer.Descriptor
import Infer.Descriptor.Adapter.Json

-- | Build a descriptor from a model directory (@config.json@ +
-- @tokenizer_config.json@).
deepseekV2DescriptorFromDir :: FilePath -> IO (Either String Descriptor)
deepseekV2DescriptorFromDir dir = do
  config <- eitherDecodeFileStrict (dir ++ "/config.json")
  case config of
    Left err -> pure (Left ("cannot parse " ++ dir ++ "/config.json: " ++ err))
    Right configValue -> do
      tokenizer <- readJsonValue (dir ++ "/tokenizer_config.json")
      pure (deepseekV2DescriptorFromConfig configValue tokenizer)

-- | Build a descriptor from the config object (a VL config nests the text tower).
deepseekV2DescriptorFromConfig :: Value -> Maybe Value -> Either String Descriptor
deepseekV2DescriptorFromConfig value tokenizer = do
  cfg <- textConfig value
  numLayers <- needInt cfg "num_hidden_layers"
  hidden <- needInt cfg "hidden_size"
  vocab <- needInt cfg "vocab_size"
  heads <- needInt cfg "num_attention_heads"
  let modelType = fromMaybe "deepseek_v2" (lookupText cfg "model_type")
      intermediate = fromMaybe (4 * hidden) (lookupInt cfg "intermediate_size")
      expertIntermediate = fromMaybe intermediate (lookupInt cfg "moe_intermediate_size")
      numExperts = fromMaybe 0 (lookupInt cfg "n_routed_experts")
      sharedExperts = fromMaybe 0 (lookupInt cfg "n_shared_experts")
      topK = fromMaybe 1 (lookupInt cfg "num_experts_per_tok")
      denseReplace = fromMaybe 0 (lookupInt cfg "first_k_dense_replace")
      denseStep = max 1 (fromMaybe 1 (lookupInt cfg "moe_layer_freq"))
      kvLoraRank = fromMaybe 512 (lookupInt cfg "kv_lora_rank")
      qkNope = fromMaybe 128 (lookupInt cfg "qk_nope_head_dim")
      qkRope = fromMaybe 64 (lookupInt cfg "qk_rope_head_dim")
      vHead = fromMaybe 128 (lookupInt cfg "v_head_dim")
      eps = fromMaybe 1e-6 (lookupDouble cfg "rms_norm_eps")
      maxPos = fromMaybe 4096 (lookupInt cfg "max_position_embeddings")
      theta = fromMaybe 1e4 (lookupDouble cfg "rope_theta")
      hasMoe = numExperts > 0 && numLayers > denseReplace
      -- DeepSeek fuses its shared experts into one MLP whose width is
      -- n_shared_experts * moe_intermediate_size (a single tensor pair in the
      -- checkpoint), so the descriptor models it as one shared expert of that
      -- width rather than S experts of moe_intermediate_size.
      sharedWidth = expertIntermediate * max 1 sharedExperts
      ffnFor i
        | not hasMoe = FDense
        | i < denseReplace = FDense
        | i `mod` denseStep /= 0 = FDense
        | otherwise = FMoe
      roles = weightRoles hasMoe
  pure Descriptor
    { dVersion = descVersion
    , dFamily = "deepseek_v2"
    , dModelType = modelType
    , dNumLayers = numLayers
    , dHiddenSize = hidden
    , dIntermediateSize = intermediate
    , dVocabSize = vocab
    , dRmsEps = eps
    , dMaxPositionEmbeddings = maxPos
    , dMaxSeqLen = min 4096 maxPos
    , dNumHeads = heads
    , dNumKvHeads = fromMaybe heads (lookupInt cfg "num_key_value_heads")
    , dHeadDim = qkNope + qkRope
    , dRotaryDim = qkRope
    , dRotaryTheta = theta
    , dNormStyle = "plain"
    , dAttnQkNorm = False
    , dAttnOutputGate = False
    , dQGateInterleave = False
    , dGdnConvDim = 0
    , dGdnValueDim = 0
    , dGdnNumVHeads = 0
    , dGdnNumKHeads = 0
    , dGdnHeadDim = 0
    , dGdnConvKernel = 0
    , dFlaChunkSize = 64
    , dMaxChunk = 128
    , dTpSize = 1, dTpRank = 0
    , dEpSize = 1, dEpRank = 0
    , dMlaKvLoraRank = kvLoraRank
    , dMlaQkNopeHeadDim = qkNope
    , dMlaQkRopeHeadDim = qkRope
    , dMlaVHeadDim = vHead
    , dMoeNumExperts = numExperts
    , dMoeTopK = topK
    , dMoeIntermediateSize = expertIntermediate
    , dMoeRouterScoring = fromMaybe "softmax" (lookupText cfg "scoring_func")
    , dMoeNormTopkProb = fromMaybe False (lookupBool cfg "norm_topk_prob")
    , dMoeNumSharedExperts = if sharedExperts > 0 then 1 else 0
    , dMoeSharedIntermediateSize = sharedWidth
    , dMoeRoutedScalingFactor = fromMaybe 1.0 (lookupDouble cfg "routed_scaling_factor")
    , dMoeSharedGateScalar = False
    , dEosTokens = eosTokens cfg tokenizer
    , dLayerMixers = replicate numLayers MMlaAttention
    , dLayerFfns = map ffnFor [0 .. numLayers - 1]
    , dRoleTemplates = roles
    , dRoleShards = defaultShards roles
    }

-- | Weight templates. Every layer has MLA; the dense layers (the first
-- @first_k_dense_replace@) own the MLP tensors and the sparse ones the router,
-- the routed experts and the shared experts.
weightRoles :: Bool -> [(Role, String)]
weightRoles hasMoe =
  [ (REmbed, "model.embed_tokens.weight")
  , (RLmHead, "lm_head.weight")
  , (RFinalNorm, "model.norm.weight")
  , (RInputNorm, layer "input_layernorm.weight")
  , (RPostNorm, layer "post_attention_layernorm.weight")
  , (RMlaQ, layer "self_attn.q_proj.weight")
  , (RMlaKvA, layer "self_attn.kv_a_proj_with_mqa.weight")
  , (RMlaKvANorm, layer "self_attn.kv_a_layernorm.weight")
  , (RMlaKvB, layer "self_attn.kv_b_proj.weight")
  , (RMlaO, layer "self_attn.o_proj.weight")
  ]
    ++ mlpRoles
    ++ [t | hasMoe, t <- moeRoles]
  where
    layer suffix = "model.layers.%d." ++ suffix
    mlpRoles =
      [ (RMlpGate, layer "mlp.gate_proj.weight")
      , (RMlpUp, layer "mlp.up_proj.weight")
      , (RMlpDown, layer "mlp.down_proj.weight")
      ]
    moeRoles =
      [ (RMoeRouter, layer "mlp.gate.weight")
      , (RMoeExpertGate, layer "mlp.experts.%e.gate_proj.weight")
      , (RMoeExpertUp, layer "mlp.experts.%e.up_proj.weight")
      , (RMoeExpertDown, layer "mlp.experts.%e.down_proj.weight")
      , (RMoeSharedGate, layer "mlp.shared_experts.gate_proj.weight")
      , (RMoeSharedUp, layer "mlp.shared_experts.up_proj.weight")
      , (RMoeSharedDown, layer "mlp.shared_experts.down_proj.weight")
      ]

-- | The text tower of a (possibly nested) config.
textConfig :: Value -> Either String Object
textConfig (Object top) =
  case lookupValue top "text_config" of
    Just inner -> maybe (Left "text_config must be an object") Right (valueObject inner)
    Nothing -> Right top
textConfig _ = Left "config.json must be an object"

-- | EOS ids: the config's @eos_token_id@ (single value or list), falling back to
-- the tokenizer config's @eos_token@.
eosTokens :: Object -> Maybe Value -> [Int]
eosTokens cfg tokenizer = case lookupValue cfg "eos_token_id" of
  Just (Number _) -> maybe [] pure (lookupInt cfg "eos_token_id")
  Just (Array _) -> fromMaybe [] (lookupIntList cfg "eos_token_id")
  _ -> case tokenizer >>= valueObject of
    Just tok -> maybe [] pure (lookupInt tok "eos_token")
    Nothing -> []
