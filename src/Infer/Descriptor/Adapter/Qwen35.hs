-- | Family adapter for Qwen3.5-style hybrid models (full attention + GatedDeltaNet).
--
-- Reads the HuggingFace @config.json@ (plus @tokenizer_config.json@ for EOS ids)
-- and produces the canonical 'Descriptor'. Everything family-specific lives here:
-- the weight-name templates, the head/layout conventions, and the EOS set.
module Infer.Descriptor.Adapter.Qwen35
  ( qwen35DescriptorFromDir
  , qwen35DescriptorFromConfig
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeFileStrict)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.List (nub)
import Data.Maybe (catMaybes, fromMaybe, listToMaybe)
import Data.Scientific (toRealFloat)
import qualified Data.Text as T
import qualified Data.Vector as V
import Text.Read (readMaybe)

import Infer.Descriptor
import Infer.Descriptor.Adapter.Json

-- | Build a descriptor from a model directory (@config.json@ +
-- @tokenizer_config.json@). The tokenizer config is optional: EOS ids then fall
-- back to the text config alone.
qwen35DescriptorFromDir :: FilePath -> IO (Either String Descriptor)
qwen35DescriptorFromDir dir = do
  config <- eitherDecodeFileStrict (dir ++ "/config.json")
  case config of
    Left err -> pure (Left ("cannot parse " ++ dir ++ "/config.json: " ++ err))
    Right configValue -> do
      tokenizer <- readJsonValue (dir ++ "/tokenizer_config.json")
      pure $ do
        textConfig <- textConfigOf configValue
        qwen35DescriptorFromConfig textConfig tokenizer

-- | Build a descriptor from the text-tower config object.
qwen35DescriptorFromConfig :: Object -> Maybe Value -> Either String Descriptor
qwen35DescriptorFromConfig tc tokenizer = do
  numLayers <- needInt tc "num_hidden_layers"
  hidden <- needInt tc "hidden_size"
  vocab <- needInt tc "vocab_size"
  heads <- needInt tc "num_attention_heads"
  kvHeads <- needInt tc "num_key_value_heads"
  headDim <- needInt tc "head_dim"
  let partial = fromMaybe 0.25 (lookupDouble tc "partial_rotary_factor"
                                  `orElse` (lookupValue tc "rope_parameters"
                                              >>= lookupDouble' "partial_rotary_factor"))
      rotaryDim = round (partial * fromIntegral headDim)
      theta = fromMaybe 1e7 (lookupValue tc "rope_parameters" >>= lookupDouble' "rope_theta")
      intermediate = fromMaybe (4 * hidden) (lookupInt tc "intermediate_size")
      eps = fromMaybe 1e-6 (lookupDouble tc "rms_norm_eps")
      maxPos = fromMaybe 4096 (lookupInt tc "max_position_embeddings")
      outGate = fromMaybe False (lookupBool tc "attn_output_gate")
      kHeads = fromMaybe 0 (lookupInt tc "linear_num_key_heads")
      vHeads = fromMaybe 0 (lookupInt tc "linear_num_value_heads")
      headDimGdn = fromMaybe 128 (lookupInt tc "linear_key_head_dim")
      convKernel = fromMaybe 4 (lookupInt tc "linear_conv_kernel_dim")
      valueDim = vHeads * headDimGdn
      convDim = 2 * kHeads * headDimGdn + valueDim
  mixers <- layerMixers tc numLayers
  pure Descriptor
    { dVersion = descVersion
    , dFamily = "qwen3_5"
    , dModelType = fromMaybe "qwen3_5_text" (lookupText tc "model_type")
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
    , dRotaryDim = rotaryDim
    , dRotaryTheta = theta
    , dNormStyle = "gemma"
    , dAttnQkNorm = True
    , dAttnOutputGate = outGate
    , dQGateInterleave = outGate
    , dGdnConvDim = convDim
    , dGdnValueDim = valueDim
    , dGdnNumVHeads = vHeads
    , dGdnNumKHeads = kHeads
    , dGdnHeadDim = headDimGdn
    , dGdnConvKernel = convKernel
    , dFlaChunkSize = 64
    , dMaxChunk = 128
    , dTpSize = 1, dTpRank = 0
    , dMoeNumExperts = 0, dMoeTopK = 0, dMoeIntermediateSize = 0
    , dMoeRouterScoring = "softmax", dMoeNormTopkProb = False
    , dMoeNumSharedExperts = 0, dMoeSharedIntermediateSize = 0
    , dMoeRoutedScalingFactor = 1.0
    , dMoeSharedGateScalar = False
    , dEosTokens = eosTokens tc tokenizer
    , dLayerMixers = mixers
    , dLayerFfns = replicate numLayers FDense
    , dRoleTemplates = weightRoles
    , dRoleShards = allReplicated weightRoles
    }
  where
    orElse (Just x) _ = Just x
    orElse Nothing y = y

-- | Per-layer mixer kinds: prefer the explicit @layer_types@ list, fall back to
-- the @full_attention_interval@ arithmetic.
layerMixers :: Object -> Int -> Either String [MixerKind]
layerMixers tc numLayers = case lookupValue tc "layer_types" of
  Just value -> case asTextArray value of
    Nothing -> Left "layer_types must be an array of strings"
    Just names -> traverse fromName names
  Nothing -> case lookupInt tc "full_attention_interval" of
    Just interval | interval > 0 ->
      Right [ if (i + 1) `mod` interval == 0 then MFullAttention else MGatedDeltaNet
            | i <- [0 .. numLayers - 1] ]
    _ -> Left "config has neither layer_types nor full_attention_interval"
  where
    fromName "full_attention" = Right MFullAttention
    fromName "linear_attention" = Right MGatedDeltaNet
    fromName other = Left ("unsupported layer type: " ++ show other)

-- | EOS set: the tokenizer's @eos_token@ resolved through @added_tokens_decoder@,
-- plus the text config's @eos_token_id@. Both are stop tokens for Qwen3.5
-- (@<|im_end|>@ and @<|endoftext|>@).
eosTokens :: Object -> Maybe Value -> [Int]
eosTokens tc tokenizer = nub (catMaybes [fromTokenizer, lookupInt tc "eos_token_id"])
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

-- | Weight-name templates for the Qwen3.5 family. @%d@ is the layer index;
-- expert/other expansions come with later families.
weightRoles :: [(Role, String)]
weightRoles =
  [ (REmbed, prefix ++ "embed_tokens.weight")
  , (RLmHead, "lm_head.weight")
  , (RFinalNorm, prefix ++ "norm.weight")
  , (RInputNorm, layer "input_layernorm.weight")
  , (RPostNorm, layer "post_attention_layernorm.weight")
  , (RMlpGate, layer "mlp.gate_proj.weight")
  , (RMlpUp, layer "mlp.up_proj.weight")
  , (RMlpDown, layer "mlp.down_proj.weight")
  , (RAttnQ, layer "self_attn.q_proj.weight")
  , (RAttnK, layer "self_attn.k_proj.weight")
  , (RAttnV, layer "self_attn.v_proj.weight")
  , (RAttnO, layer "self_attn.o_proj.weight")
  , (RAttnQNorm, layer "self_attn.q_norm.weight")
  , (RAttnKNorm, layer "self_attn.k_norm.weight")
  , (RGdnQkv, layer "linear_attn.in_proj_qkv.weight")
  , (RGdnZ, layer "linear_attn.in_proj_z.weight")
  , (RGdnA, layer "linear_attn.in_proj_a.weight")
  , (RGdnB, layer "linear_attn.in_proj_b.weight")
  , (RGdnConv1d, layer "linear_attn.conv1d.weight")
  , (RGdnDtBias, layer "linear_attn.dt_bias")
  , (RGdnALog, layer "linear_attn.A_log")
  , (RGdnOut, layer "linear_attn.out_proj.weight")
  , (RGdnNorm, layer "linear_attn.norm.weight")
  ]
  where
    prefix = "model.language_model."
    layer suffix = prefix ++ "layers.%d." ++ suffix

-- ---------------------------------------------------------------------------
-- config.json helpers
-- ---------------------------------------------------------------------------

-- | The text tower config: nested under @text_config@ for VL checkpoints, or the
-- top-level object for text-only ones.
textConfigOf :: Value -> Either String Object
textConfigOf value = do
  top <- maybe (Left "config.json must be an object") Right (valueObject value)
  pure $ fromMaybe top (lookupValue top "text_config" >>= valueObject)

-- | Look up a double inside a nested object value (@rope_parameters@).
lookupDouble' :: String -> Value -> Maybe Double
lookupDouble' key value = case valueObject value of
  Just obj -> lookupDouble obj key
  Nothing -> Nothing

asDouble :: Value -> Maybe Double
asDouble (Number n) = Just (toRealFloat n)
asDouble _ = Nothing

asTextArray :: Value -> Maybe [String]
asTextArray (Array values) = traverse asText (V.toList values)
asTextArray _ = Nothing

asText :: Value -> Maybe String
asText (String t) = Just (T.unpack t)
asText _ = Nothing

lookupText'' :: Object -> String -> Maybe String
lookupText'' obj key = case lookupValue obj key of
  Just (String t) -> Just (T.unpack t)
  _ -> Nothing
