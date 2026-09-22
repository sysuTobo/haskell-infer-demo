-- | Family adapter for the dense-attention line: Qwen3 (`qwen3`), Mixtral
-- (`mixtral`) and Qwen3-MoE (`qwen3_moe`). They share the GQA attention block and
-- differ in whether the feed-forward is dense or a sparse mixture of experts, in
-- field names and in weight templates -- which is exactly what this module holds.
-- Qwen3-MoE detects its sparse layers from @num_experts@; a config without it
-- gets ordinary MLP layers.
--
-- Note on @norm_topk_prob@: Mixtral's checkpoints predate the flag. The value is
-- taken from the config when present and otherwise left false here; the MoE
-- kernels verify it against PyTorch hooks per checkpoint before trusting it.
module Infer.Descriptor.Adapter.Mixtral
  ( moeDenseDescriptorFromDir
  , moeDenseDescriptorFromConfig
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeFileStrict)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.List (nub)
import Data.Maybe (catMaybes, fromMaybe, listToMaybe)
import Text.Read (readMaybe)

import Infer.Descriptor
import Infer.Descriptor.Adapter.Json

-- | Build a descriptor from a model directory (@config.json@ +
-- @tokenizer_config.json@).
moeDenseDescriptorFromDir :: FilePath -> IO (Either String Descriptor)
moeDenseDescriptorFromDir dir = do
  config <- eitherDecodeFileStrict (dir ++ "/config.json")
  case config of
    Left err -> pure (Left ("cannot parse " ++ dir ++ "/config.json: " ++ err))
    Right configValue -> do
      tokenizer <- readJsonValue (dir ++ "/tokenizer_config.json")
      pure $ do
        cfg <- topLevel configValue
        moeDenseDescriptorFromConfig cfg tokenizer

-- | Build a descriptor from the (flat) config object.
moeDenseDescriptorFromConfig :: Object -> Maybe Value -> Either String Descriptor
moeDenseDescriptorFromConfig cfg tokenizer = do
  numLayers <- needInt cfg "num_hidden_layers"
  hidden <- needInt cfg "hidden_size"
  vocab <- needInt cfg "vocab_size"
  heads <- needInt cfg "num_attention_heads"
  kvHeads <- needInt cfg "num_key_value_heads"
  let family = fromMaybe "mixtral" (lookupText cfg "model_type")
      headDim = fromMaybe (hidden `div` heads) (lookupInt cfg "head_dim")
      intermediate = fromMaybe (4 * hidden) (lookupInt cfg "intermediate_size")
      -- Mixtral names the per-expert width `intermediate_size`; Qwen3-MoE uses
      -- `moe_intermediate_size` and keeps `intermediate_size` for dense layers.
      expertIntermediate = fromMaybe intermediate (lookupInt cfg "moe_intermediate_size")
      numExperts = fromMaybe 0 (lookupInt cfg "num_local_experts" `orElse` lookupInt cfg "num_experts")
      tied = fromMaybe False (lookupBool cfg "tie_word_embeddings")
      hasMoe = numExperts > 0
      topK = fromMaybe 2 (lookupInt cfg "num_experts_per_tok")
      sparseStep = max 1 (fromMaybe 1 (lookupInt cfg "decoder_sparse_step"))
      denseLayers = fromMaybe [] (lookupIntList cfg "mlp_only_layers")
      eps = fromMaybe 1e-5 (lookupDouble cfg "rms_norm_eps")
      maxPos = fromMaybe 4096 (lookupInt cfg "max_position_embeddings")
      theta = fromMaybe 1e6 (lookupDouble cfg "rope_theta")
      ffnFor i
        | not hasMoe = FDense
        | i `elem` denseLayers || (i + 1) `mod` sparseStep /= 0 = FDense
        | otherwise = FMoe
      roles = weightRoles family hasMoe tied
  pure Descriptor
    { dVersion = descVersion
    , dFamily = family
    , dModelType = family
    , dNumLayers = numLayers
    , dHiddenSize = hidden
    , dIntermediateSize = intermediate
    , dVocabSize = vocab
    , dRmsEps = eps
    , dMaxPositionEmbeddings = maxPos
    , dMaxSeqLen = min 4096 maxPos
    , dNumHeads = heads
    , dNumKvHeads = kvHeads
    , dHeadDim = headDim
    , dRotaryDim = headDim            -- these families rotate every head dim
    , dRotaryTheta = theta
    , dNormStyle = "plain"    -- Qwen3-MoE and Mixtral use plain RMSNorm
    , dAttnQkNorm = family /= "mixtral"   -- Qwen3-MoE normalizes q/k; Mixtral does not
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
    , dMoeNumExperts = numExperts
    , dMoeTopK = topK
    , dMoeIntermediateSize = expertIntermediate
    , dMoeRouterScoring = "softmax"
    , dMoeNormTopkProb = fromMaybe False (lookupBool cfg "norm_topk_prob")
    , dMoeNumSharedExperts = 0
    , dMoeSharedIntermediateSize = 0
    , dMoeRoutedScalingFactor = 1.0
    , dMoeSharedGateScalar = False
    , dEosTokens = eosTokens cfg tokenizer
    , dLayerMixers = replicate numLayers MFullAttention
    , dLayerFfns = map ffnFor [0 .. numLayers - 1]
    , dRoleTemplates = roles
    , dRoleShards = allReplicated roles
    }
  where
    orElse (Just x) _ = Just x
    orElse Nothing y = y

-- | Shared-expert and GDN roles are absent for these families; the feed-forward
-- roles depend on whether the layers are sparse, and a tied checkpoint points
-- lm_head at the embedding tensor (the descriptor carries the name, so the engine
-- needs no tie logic).
weightRoles :: String -> Bool -> Bool -> [(Role, String)]
weightRoles family hasMoe tied = common ++ attn ++ ffn
  where
    common =
      [ (REmbed, "model.embed_tokens.weight")
      , (RLmHead, if tied then "model.embed_tokens.weight" else "lm_head.weight")
      , (RFinalNorm, "model.norm.weight")
      , (RInputNorm, layer "input_layernorm.weight")
      , (RPostNorm, layer "post_attention_layernorm.weight")
      ]
    attn =
      [ (RAttnQ, layer "self_attn.q_proj.weight")
      , (RAttnK, layer "self_attn.k_proj.weight")
      , (RAttnV, layer "self_attn.v_proj.weight")
      , (RAttnO, layer "self_attn.o_proj.weight")
      ] ++ qkNorm
    qkNorm
      | family == "mixtral" = []
      | otherwise =
          [ (RAttnQNorm, layer "self_attn.q_norm.weight")
          , (RAttnKNorm, layer "self_attn.k_norm.weight")
          ]
    -- Mixtral keeps the block under `block_sparse_moe` and names the three
    -- projections w1/w3 (gate/up) and w2 (down); Qwen3-MoE uses `mlp.gate`
    -- with per-expert `gate_proj`/`up_proj`/`down_proj`.
    ffn
      | not hasMoe =
          [ (RMlpGate, layer "mlp.gate_proj.weight")
          , (RMlpUp, layer "mlp.up_proj.weight")
          , (RMlpDown, layer "mlp.down_proj.weight")
          ]
      | otherwise = router ++ experts
    (router, experts)
      | family == "mixtral" =
          ( [(RMoeRouter, layer "block_sparse_moe.gate.weight")]
          , [ (RMoeExpertGate, layer "block_sparse_moe.experts.%e.w1.weight")
            , (RMoeExpertUp, layer "block_sparse_moe.experts.%e.w3.weight")
            , (RMoeExpertDown, layer "block_sparse_moe.experts.%e.w2.weight")
            ] )
      | otherwise =
          ( [(RMoeRouter, layer "mlp.gate.weight")]
          , [ (RMoeExpertGate, layer "mlp.experts.%e.gate_proj.weight")
            , (RMoeExpertUp, layer "mlp.experts.%e.up_proj.weight")
            , (RMoeExpertDown, layer "mlp.experts.%e.down_proj.weight")
            ] )
    layer suffix = "model.layers.%d." ++ suffix

-- | EOS: the tokenizer's @eos_token@ resolved through @added_tokens_decoder@,
-- plus the config's @eos_token_id@.
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

topLevel :: Value -> Either String Object
topLevel value = do
  top <- maybe (Left "config.json must be an object") Right (valueObject value)
  pure $ fromMaybe top (lookupValue top "text_config" >>= valueObject)
