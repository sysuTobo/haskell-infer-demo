{-# LANGUAGE OverloadedStrings #-}

-- | Canonical model descriptor: the single source of truth for everything the
-- engine needs to know about a model architecture.
--
-- A family adapter (see "Infer.Descriptor.Adapter") turns a HuggingFace
-- @config.json@ into a descriptor. It is serialized to a *flat* JSON object with
-- no nesting and parsed strictly on the C side (@csrc/model_desc.c@): unknown
-- keys and missing keys are both hard errors, so a typo fails loudly instead of
-- silently taking a default.
--
-- Haskell stays the only place where model knowledge lives; C is family-agnostic.
module Infer.Descriptor
  ( Descriptor(..)
  , MixerKind(..)
  , FfnKind(..)
  , Role(..)
  , ShardKind(..)
  , descVersion
  , encodeDescriptor
  , decodeDescriptor
  , validateDescriptor
  , withMaxSeqLen
  , mixerName
  , ffnName
  , roleName
  , shardName
  , allReplicated
  , defaultShards
  , kvBytesPerToken
  , attentionLayers
  , gdnLayers
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeStrict', encode, object, (.=))
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (toLower)
import Data.List (nub)
import qualified Data.Text as T
import qualified Data.Vector as V

-- | Mixer kind: the token-mixing sublayer of a transformer block.
data MixerKind = MFullAttention | MGatedDeltaNet | MMlaAttention
  deriving (Eq, Show, Read)

-- | Feed-forward kind: dense MLP or sparse MoE.
data FfnKind = FDense | FMoe
  deriving (Eq, Show, Read)

-- | Weight roles. The role vocabulary is compiled into both sides (here and
-- @csrc/model_desc.c@); which tensors exist for a family is data, carried in
-- @dRoleTemplates@.
data Role
  = REmbed | RLmHead | RFinalNorm
  | RInputNorm | RPostNorm
  | RMlpGate | RMlpUp | RMlpDown
  | RAttnQ | RAttnK | RAttnV | RAttnO | RAttnQNorm | RAttnKNorm
  | RGdnQkv | RGdnZ | RGdnA | RGdnB | RGdnConv1d | RGdnDtBias | RGdnALog
  | RGdnOut | RGdnNorm
  | RGdnQkvz | RGdnBa           -- fused projections (Qwen3-Next style)
  | RMoeRouter | RMoeRouterBias
  | RMoeExpertGate | RMoeExpertUp | RMoeExpertDown
  | RMoeSharedGate | RMoeSharedUp | RMoeSharedDown | RMoeSharedGateScalar
  deriving (Eq, Ord, Show, Read, Enum, Bounded)

mixerName :: MixerKind -> String
mixerName MFullAttention = "full_attn"
mixerName MGatedDeltaNet = "gdn"
mixerName MMlaAttention = "mla"

ffnName :: FfnKind -> String
ffnName FDense = "dense"
ffnName FMoe = "moe"

-- | Tensor-parallel sharding rule for one weight role, i.e. which dimension of
-- the tensor the engine splits across @tp_size@ ranks. The vocabulary is kept
-- deliberately small: the engine loads shards without any family knowledge.
--
--   * 'ShardNone'     -- replicated: every rank loads the whole tensor.
--   * 'ShardOutHeads' -- output dimension split by head: the tensor's rows are
--                        @heads * head_dim@ and each rank takes a contiguous
--                        block of heads (attention / per-head projections).
--   * 'ShardOutDim'   -- output dimension (rows of an @[out, in]@ weight) split
--                        into @tp_size@ contiguous blocks.
--   * 'ShardInDim'    -- input dimension (columns) split into @tp_size@
--                        contiguous blocks.
--
-- With @tp_size == 1@ every rule is a no-op; the default for a freshly built
-- descriptor is 'ShardNone' for every role (see 'allReplicated').
data ShardKind = ShardNone | ShardOutHeads | ShardOutDim | ShardInDim
  deriving (Eq, Ord, Show, Read, Enum, Bounded)

-- | Wire name of a 'ShardKind'. @csrc/model_desc.c@ carries the same names.
shardName :: ShardKind -> String
shardName ShardNone = "none"
shardName ShardOutHeads = "out_heads"
shardName ShardOutDim = "out_dim"
shardName ShardInDim = "in_dim"

-- | The default shard rules: one 'ShardNone' per role, parallel to the role
-- table. Building a descriptor with these plus @dTpSize = 1@ keeps the model
-- replicated, i.e. the historical single-rank behaviour.
allReplicated :: [(Role, String)] -> [ShardKind]
allReplicated = map (const ShardNone)

-- | The shard rules a role table gets by default: the projections whose output
-- is split by head (@q/k/v@), the row-parallel halves of a dense MLP
-- (@gate\/up@), their column-parallel counterparts (@o_proj@, MLP @down@), and
-- replication for everything else (norms, embeddings, GDN, MoE -- expert
-- sharding is a different rule and lands with expert parallelism).
--
-- With @tp_size == 1@ every rule is a no-op, so a replicated descriptor keeps
-- the historical numerics; the rules only describe how a tensor would be split.
defaultShards :: [(Role, String)] -> [ShardKind]
defaultShards = map (defaultShard . fst)

-- | See 'defaultShards'. Keeping this per role (rather than per family) means
-- every adapter describes the same tensor semantics.
defaultShard :: Role -> ShardKind
defaultShard role = case role of
  RAttnQ -> ShardOutHeads
  RAttnK -> ShardOutHeads
  RAttnV -> ShardOutHeads
  RAttnO -> ShardInDim
  RMlpGate -> ShardOutDim
  RMlpUp -> ShardOutDim
  RMlpDown -> ShardInDim
  _ -> ShardNone

-- | Which shard rules a role's tensor layout can carry. The engine enforces the
-- same restriction when it loads a shard.
shardFitsRole :: Role -> ShardKind -> Bool
shardFitsRole _ ShardNone = True
shardFitsRole role ShardOutHeads = role `elem` [RAttnQ, RAttnK, RAttnV]
shardFitsRole role ShardOutDim = role `elem` [RMlpGate, RMlpUp]
shardFitsRole role ShardInDim = role `elem` [RAttnO, RMlpDown]

-- | The role table paired with its shard rules (truncated to the shorter side
-- when a descriptor is malformed; the parallel-length check reports that).
shardRules :: Descriptor -> [(Role, ShardKind)]
shardRules d = zip (map fst (dRoleTemplates d)) (dRoleShards d)

-- | Wire name of a role: the constructor with its leading @R@ dropped and the
-- first letter lowercased (@REmbed@ -> @embed@, @RAttnQNorm@ -> @attnQNorm@).
-- @csrc/model_desc.c@ carries the same names in the same order.
roleName :: Role -> String
roleName = lowerFirst . drop 1 . show
  where
    lowerFirst [] = []
    lowerFirst (c:cs) = toLower c : cs

-- | Descriptor version. Bumping this is how C refuses a wire format it cannot read.
descVersion :: Int
descVersion = 1

data Descriptor = Descriptor
  { dVersion :: !Int
  , dFamily :: !String            -- ^ e.g. "qwen3_5"
  , dModelType :: !String         -- ^ HF @model_type@ of the text tower
  , dNumLayers :: !Int
  , dHiddenSize :: !Int
  , dIntermediateSize :: !Int     -- ^ dense MLP intermediate size
  , dVocabSize :: !Int
  , dRmsEps :: !Double
  , dMaxPositionEmbeddings :: !Int
  , dMaxSeqLen :: !Int            -- ^ runtime context length (overridable)
  , dNumHeads :: !Int
  , dNumKvHeads :: !Int
  , dHeadDim :: !Int
  , dRotaryDim :: !Int
  , dRotaryTheta :: !Double
  , dNormStyle :: !String         -- ^ "gemma" (weight + 1) or "plain"
  , dAttnQkNorm :: !Bool          -- ^ attention applies per-head q/k RMSNorm
  , dAttnOutputGate :: !Bool      -- ^ q_proj carries a fused output gate
  , dQGateInterleave :: !Bool     -- ^ fused Q+gate rows are interleaved per head
  , dGdnConvDim :: !Int
  , dGdnValueDim :: !Int
  , dGdnNumVHeads :: !Int
  , dGdnNumKHeads :: !Int
  , dGdnHeadDim :: !Int
  , dGdnConvKernel :: !Int
  , dFlaChunkSize :: !Int         -- ^ must match the AOT-compiled FLA chunk cubin
  , dMaxChunk :: !Int             -- ^ prefill batch size (<= the kernels' limit)
  , dTpSize :: !Int               -- ^ tensor-parallel ranks (1 = no sharding)
  , dTpRank :: !Int               -- ^ this rank, in [0, dTpSize)
  -- Mixture-of-experts feed-forward (used when a layer's ffn kind is moe)
  , dMoeNumExperts :: !Int        -- ^ routed experts per layer
  , dMoeTopK :: !Int              -- ^ experts selected per token
  , dMoeIntermediateSize :: !Int  -- ^ per-expert FFN hidden size
  , dMoeRouterScoring :: !String  -- ^ "softmax" or "sigmoid"
  , dMoeNormTopkProb :: !Bool     -- ^ renormalize the selected weights to sum 1
  , dMoeNumSharedExperts :: !Int  -- ^ always-on experts (0 for Mixtral/Qwen3-MoE)
  , dMoeSharedIntermediateSize :: !Int
  , dMoeRoutedScalingFactor :: !Double
  , dMoeSharedGateScalar :: !Bool   -- ^ scale the shared output by sigmoid(x @ w)
  , dEosTokens :: [Int]
  , dLayerMixers :: [MixerKind]
  , dLayerFfns :: [FfnKind]
  , dRoleTemplates :: [(Role, String)]
  , dRoleShards :: [ShardKind]    -- ^ parallel to 'dRoleTemplates'
  } deriving (Eq, Show)

withMaxSeqLen :: Int -> Descriptor -> Descriptor
withMaxSeqLen n d = d { dMaxSeqLen = n }

attentionLayers :: Descriptor -> [Int]
attentionLayers d = [i | (i, MFullAttention) <- zip [0 ..] (dLayerMixers d)]

gdnLayers :: Descriptor -> [Int]
gdnLayers d = [i | (i, MGatedDeltaNet) <- zip [0 ..] (dLayerMixers d)]

-- | Bytes of KV cache per token per attention layer (K and V, bf16).
kvBytesPerToken :: Descriptor -> Int
kvBytesPerToken d = 2 * dNumKvHeads d * dHeadDim d * 2

-- ---------------------------------------------------------------------------
-- Wire encoding (flat JSON, fixed key set)
-- ---------------------------------------------------------------------------

descriptorKeys :: [String]
descriptorKeys =
  [ "desc_version", "family", "model_type", "num_layers", "hidden_size"
  , "intermediate_size", "vocab_size", "rms_eps", "max_position_embeddings"
  , "max_seq_len", "num_heads", "num_kv_heads", "head_dim", "rotary_dim"
  , "rotary_theta", "norm_style", "attn_qk_norm", "attn_output_gate"
  , "q_gate_interleave", "gdn_conv_dim"
  , "gdn_value_dim", "gdn_num_v_heads", "gdn_num_k_heads", "gdn_head_dim"
  , "gdn_conv_kernel", "fla_chunk_size", "max_chunk", "tp_size", "tp_rank"
  , "moe_num_experts", "moe_top_k"
  , "moe_intermediate_size", "moe_router_scoring", "moe_norm_topk_prob"
  , "moe_num_shared_experts", "moe_shared_intermediate_size", "moe_routed_scaling_factor"
  , "moe_shared_gate_scalar"
  , "eos_tokens", "layer_mixers"
  , "layer_ffns", "role_names", "role_templates", "role_shards"
  ]

-- | Encode to the flat wire format consumed by @engine_create@.
encodeDescriptor :: Descriptor -> BS.ByteString
encodeDescriptor d = BL.toStrict . encode $ object
  [ "desc_version" .= dVersion d
  , "family" .= dFamily d
  , "model_type" .= dModelType d
  , "num_layers" .= dNumLayers d
  , "hidden_size" .= dHiddenSize d
  , "intermediate_size" .= dIntermediateSize d
  , "vocab_size" .= dVocabSize d
  , "rms_eps" .= dRmsEps d
  , "max_position_embeddings" .= dMaxPositionEmbeddings d
  , "max_seq_len" .= dMaxSeqLen d
  , "num_heads" .= dNumHeads d
  , "num_kv_heads" .= dNumKvHeads d
  , "head_dim" .= dHeadDim d
  , "rotary_dim" .= dRotaryDim d
  , "rotary_theta" .= dRotaryTheta d
  , "norm_style" .= dNormStyle d
  , "attn_qk_norm" .= dAttnQkNorm d
  , "attn_output_gate" .= dAttnOutputGate d
  , "q_gate_interleave" .= dQGateInterleave d
  , "gdn_conv_dim" .= dGdnConvDim d
  , "gdn_value_dim" .= dGdnValueDim d
  , "gdn_num_v_heads" .= dGdnNumVHeads d
  , "gdn_num_k_heads" .= dGdnNumKHeads d
  , "gdn_head_dim" .= dGdnHeadDim d
  , "gdn_conv_kernel" .= dGdnConvKernel d
  , "fla_chunk_size" .= dFlaChunkSize d
  , "max_chunk" .= dMaxChunk d
  , "tp_size" .= dTpSize d
  , "tp_rank" .= dTpRank d
  , "moe_num_experts" .= dMoeNumExperts d
  , "moe_top_k" .= dMoeTopK d
  , "moe_intermediate_size" .= dMoeIntermediateSize d
  , "moe_router_scoring" .= dMoeRouterScoring d
  , "moe_norm_topk_prob" .= dMoeNormTopkProb d
  , "moe_num_shared_experts" .= dMoeNumSharedExperts d
  , "moe_shared_intermediate_size" .= dMoeSharedIntermediateSize d
  , "moe_routed_scaling_factor" .= dMoeRoutedScalingFactor d
  , "moe_shared_gate_scalar" .= dMoeSharedGateScalar d
  , "eos_tokens" .= dEosTokens d
  , "layer_mixers" .= map mixerName (dLayerMixers d)
  , "layer_ffns" .= map ffnName (dLayerFfns d)
  , "role_names" .= map (roleName . fst) (dRoleTemplates d)
  , "role_templates" .= map snd (dRoleTemplates d)
  , "role_shards" .= map shardName (dRoleShards d)
  ]

-- | Decode the flat wire format. Unknown or missing keys are errors.
decodeDescriptor :: BS.ByteString -> Either String Descriptor
decodeDescriptor bytes = do
  value <- eitherDecodeStrict' bytes
  obj <- case value of
    Object o -> Right o
    _ -> Left "descriptor must be a JSON object"
  case [k | k <- map Key.toString (KM.keys obj), k `notElem` descriptorKeys] of
    [] -> Right ()
    unknown -> Left ("unknown descriptor keys: " ++ show unknown)
  version <- reqInt obj "desc_version"
  if version /= descVersion
    then Left ("descriptor version " ++ show version ++ " != supported " ++ show descVersion)
    else Right ()
  families <- reqText obj "family"
  modelType <- reqText obj "model_type"
  numLayers <- reqInt obj "num_layers"
  hidden <- reqInt obj "hidden_size"
  intermediate <- reqInt obj "intermediate_size"
  vocab <- reqInt obj "vocab_size"
  eps <- reqDouble obj "rms_eps"
  maxPos <- reqInt obj "max_position_embeddings"
  maxSeq <- reqInt obj "max_seq_len"
  heads <- reqInt obj "num_heads"
  kvHeads <- reqInt obj "num_kv_heads"
  headDim <- reqInt obj "head_dim"
  rotaryDim <- reqInt obj "rotary_dim"
  theta <- reqDouble obj "rotary_theta"
  normStyle <- reqText obj "norm_style"
  qkNorm <- reqBool obj "attn_qk_norm"
  outGate <- reqBool obj "attn_output_gate"
  qgInterleave <- reqBool obj "q_gate_interleave"
  convDim <- reqInt obj "gdn_conv_dim"
  valueDim <- reqInt obj "gdn_value_dim"
  vHeads <- reqInt obj "gdn_num_v_heads"
  kHeads <- reqInt obj "gdn_num_k_heads"
  gdnHeadDim <- reqInt obj "gdn_head_dim"
  convKernel <- reqInt obj "gdn_conv_kernel"
  chunkSize <- reqInt obj "fla_chunk_size"
  maxChunk <- reqInt obj "max_chunk"
  -- Tensor-parallel keys are optional: absent means "replicated single rank",
  -- which is what every descriptor written before they existed means.
  tpSize <- optInt obj "tp_size" 1
  tpRank <- optInt obj "tp_rank" 0
  moeExperts <- reqInt obj "moe_num_experts"
  moeTopK <- reqInt obj "moe_top_k"
  moeIntermediate <- reqInt obj "moe_intermediate_size"
  moeScoring <- reqText obj "moe_router_scoring"
  moeNormTopk <- reqBool obj "moe_norm_topk_prob"
  moeSharedExperts <- reqInt obj "moe_num_shared_experts"
  moeSharedIntermediate <- reqInt obj "moe_shared_intermediate_size"
  moeScaling <- reqDouble obj "moe_routed_scaling_factor"
  moeSharedGate <- reqBool obj "moe_shared_gate_scalar"
  eos <- reqIntList obj "eos_tokens"
  mixers <- traverse parseMixer =<< reqTextList obj "layer_mixers"
  ffns <- traverse parseFfn =<< reqTextList obj "layer_ffns"
  roleNames <- reqTextList obj "role_names"
  templates <- reqTextList obj "role_templates"
  roles <- traverse parseRole roleNames
  -- Absent role_shards means "everything replicated": the default mirrors the
  -- role table so the array is always parallel to role_names/role_templates.
  shardNames <- optTextList obj "role_shards" (replicate (length roles) "none")
  shards <- traverse parseShard shardNames
  if length shards /= length roles
    then Left ("role_shards (" ++ show (length shards)
               ++ ") must be parallel to role_names (" ++ show (length roles) ++ ")")
    else Right ()
  pure Descriptor
    { dVersion = version, dFamily = families, dModelType = modelType
    , dNumLayers = numLayers, dHiddenSize = hidden, dIntermediateSize = intermediate
    , dVocabSize = vocab, dRmsEps = eps, dMaxPositionEmbeddings = maxPos
    , dMaxSeqLen = maxSeq, dNumHeads = heads, dNumKvHeads = kvHeads, dHeadDim = headDim
    , dRotaryDim = rotaryDim, dRotaryTheta = theta, dNormStyle = normStyle
    , dAttnQkNorm = qkNorm
    , dAttnOutputGate = outGate
    , dQGateInterleave = qgInterleave, dGdnConvDim = convDim, dGdnValueDim = valueDim
    , dGdnNumVHeads = vHeads, dGdnNumKHeads = kHeads, dGdnHeadDim = gdnHeadDim
    , dGdnConvKernel = convKernel, dFlaChunkSize = chunkSize, dMaxChunk = maxChunk
    , dTpSize = tpSize, dTpRank = tpRank
    , dMoeNumExperts = moeExperts, dMoeTopK = moeTopK
    , dMoeIntermediateSize = moeIntermediate, dMoeRouterScoring = moeScoring
    , dMoeNormTopkProb = moeNormTopk, dMoeNumSharedExperts = moeSharedExperts
    , dMoeSharedIntermediateSize = moeSharedIntermediate, dMoeRoutedScalingFactor = moeScaling
    , dMoeSharedGateScalar = moeSharedGate
    , dEosTokens = eos
    , dLayerMixers = mixers, dLayerFfns = ffns
    , dRoleTemplates = zip roles templates
    , dRoleShards = shards
    }

parseMixer :: String -> Either String MixerKind
parseMixer s = case s of
  "full_attn" -> Right MFullAttention
  "gdn" -> Right MGatedDeltaNet
  "mla" -> Right MMlaAttention
  _ -> Left ("unknown layer_mixers entry: " ++ show s)

parseFfn :: String -> Either String FfnKind
parseFfn s = case s of
  "dense" -> Right FDense
  "moe" -> Right FMoe
  _ -> Left ("unknown layer_ffns entry: " ++ show s)

parseRole :: String -> Either String Role
parseRole s = case [r | r <- [minBound .. maxBound], roleName r == s] of
  [r] -> Right r
  _ -> Left ("unknown role name: " ++ show s)

parseShard :: String -> Either String ShardKind
parseShard s = case [k | k <- [minBound .. maxBound], shardName k == s] of
  [k] -> Right k
  _ -> Left ("unknown role_shards entry: " ++ show s)

reqInt :: Object -> String -> Either String Int
reqInt obj key = case KM.lookup (Key.fromString key) obj of
  Just (Number n) -> Right (round n)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a number")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")

-- | Like 'reqInt' but the key is optional (used for the tensor-parallel keys,
-- whose absence must keep older descriptors valid).
optInt :: Object -> String -> Int -> Either String Int
optInt obj key fallback = case KM.lookup (Key.fromString key) obj of
  Just (Number n) -> Right (round n)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a number")
  Nothing -> Right fallback

reqDouble :: Object -> String -> Either String Double
reqDouble obj key = case KM.lookup (Key.fromString key) obj of
  Just (Number n) -> Right (realToFrac n)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a number")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")

reqText :: Object -> String -> Either String String
reqText obj key = case KM.lookup (Key.fromString key) obj of
  Just (String t) -> Right (T.unpack t)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a string")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")

reqBool :: Object -> String -> Either String Bool
reqBool obj key = case KM.lookup (Key.fromString key) obj of
  Just (Bool b) -> Right b
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a boolean")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")

reqIntList :: Object -> String -> Either String [Int]
reqIntList obj key = case KM.lookup (Key.fromString key) obj of
  Just (Array values) -> traverse asInt (V.toList values)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be an array")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")
  where
    asInt (Number n) = Right (round n)
    asInt _ = Left ("descriptor key " ++ show key ++ " must contain only numbers")

-- | Like 'reqTextList' below but the key is optional.
optTextList :: Object -> String -> [String] -> Either String [String]
optTextList obj key fallback = case KM.lookup (Key.fromString key) obj of
  Just (Array values) -> traverse asText (V.toList values)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be an array")
  Nothing -> Right fallback
  where
    asText (String t) = Right (T.unpack t)
    asText _ = Left ("descriptor key " ++ show key ++ " must contain only strings")

reqTextList :: Object -> String -> Either String [String]
reqTextList obj key = case KM.lookup (Key.fromString key) obj of
  Just (Array values) -> traverse asText (V.toList values)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be an array")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")
  where
    asText (String t) = Right (T.unpack t)
    asText _ = Left ("descriptor key " ++ show key ++ " must contain only strings")

-- ---------------------------------------------------------------------------
-- Validation
-- ---------------------------------------------------------------------------

-- | Structural validation. Cheap checks only; no GPU or filesystem access.
validateDescriptor :: Descriptor -> Either String ()
validateDescriptor d = case [msg | Just msg <- checks d] of
  [] -> Right ()
  msg : _ -> Left msg

checks :: Descriptor -> [Maybe String]
checks d =
  [ err (dNumLayers d <= 0) "num_layers must be positive"
  , err (length (dLayerMixers d) /= dNumLayers d)
        "layer_mixers length must equal num_layers"
  , err (length (dLayerFfns d) /= dNumLayers d)
        "layer_ffns length must equal num_layers"
  , err (dHiddenSize d <= 0 || dVocabSize d <= 0 || dIntermediateSize d <= 0)
        "hidden/intermediate/vocab sizes must be positive"
  , err (dMaxSeqLen d <= 0) "max_seq_len must be positive"
  , err (dMaxSeqLen d > dMaxPositionEmbeddings d)
        "max_seq_len exceeds max_position_embeddings"
  , err (dNumHeads d <= 0 || dNumKvHeads d <= 0) "head counts must be positive"
  , err (dNumHeads d `mod` dNumKvHeads d /= 0)
        "num_heads must be a multiple of num_kv_heads"
  , err (dHeadDim d <= 0) "head_dim must be positive"
  , err (dRotaryDim d < 0 || dRotaryDim d > dHeadDim d || odd (dRotaryDim d))
        "rotary_dim must be even and <= head_dim"
  , err (dRotaryTheta d <= 0) "rotary_theta must be positive"
  , err (dNormStyle d `notElem` ["gemma", "plain"])
        "norm_style must be gemma or plain"
  , err (dRmsEps d <= 0) "rms_eps must be positive"
  , err (dMaxChunk d < 1 || dMaxChunk d > 128)
        "max_chunk must be in [1,128] (the kernels' batch limit)"
  , err (dTpSize d < 1)
        "tp_size must be at least 1"
  , err (dTpRank d < 0 || dTpRank d >= dTpSize d)
        "tp_rank must be in [0, tp_size)"
  , err (length (dRoleShards d) /= length (dRoleTemplates d))
        "role_shards must be parallel to role_names/role_templates"
  , err (not (all (uncurry shardFitsRole) (shardRules d)))
        "role_shards uses a rule the role cannot carry (out_heads: attn q/k/v; out_dim: mlp gate/up; in_dim: attn o / mlp down)"
  , err (null (dEosTokens d)) "eos_tokens must not be empty"
  , err (any (\t -> t < 0 || t >= dVocabSize d) (dEosTokens d))
        "eos_tokens must be within the vocabulary"
  , gdnCheck d
  , moeCheck d
  , rolesCheck d
  ]

err :: Bool -> String -> Maybe String
err True msg = Just msg
err False _ = Nothing

gdnCheck :: Descriptor -> Maybe String
gdnCheck d
  | null (gdnLayers d) = Nothing
  | dGdnHeadDim d <= 0 = Just "gdn_head_dim must be positive"
  | dGdnNumVHeads d <= 0 || dGdnNumKHeads d <= 0 = Just "gdn head counts must be positive"
  | dGdnNumVHeads d `mod` dGdnNumKHeads d /= 0 =
      Just "gdn_num_v_heads must be a multiple of gdn_num_k_heads"
  | dGdnValueDim d /= dGdnNumVHeads d * dGdnHeadDim d =
      Just "gdn_value_dim must equal gdn_num_v_heads * gdn_head_dim"
  | dGdnConvDim d /= 2 * dGdnNumKHeads d * dGdnHeadDim d + dGdnValueDim d =
      Just "gdn_conv_dim must equal 2 * gdn_num_k_heads * gdn_head_dim + gdn_value_dim"
  | dGdnConvKernel d <= 0 = Just "gdn_conv_kernel must be positive"
  | dFlaChunkSize d `notElem` [1 .. 128] = Just "fla_chunk_size must be in [1,128]"
  | otherwise = Nothing

moeCheck :: Descriptor -> Maybe String
moeCheck d
  | not (FMoe `elem` dLayerFfns d) = Nothing
  | dMoeNumExperts d <= 0 = Just "moe_num_experts must be positive"
  | dMoeTopK d < 1 || dMoeTopK d > dMoeNumExperts d =
      Just "moe_top_k must be in [1, moe_num_experts]"
  | dMoeIntermediateSize d <= 0 = Just "moe_intermediate_size must be positive"
  | dMoeRouterScoring d `notElem` ["softmax", "sigmoid"] =
      Just "moe_router_scoring must be softmax or sigmoid"
  | dMoeNumSharedExperts d < 0 || dMoeSharedIntermediateSize d < 0 =
      Just "shared-expert sizes must not be negative"
  | dMoeNumSharedExperts d > 0 && dMoeSharedIntermediateSize d <= 0 =
      Just "moe_shared_intermediate_size is required when shared experts are used"
  | dMoeRoutedScalingFactor d <= 0 = Just "moe_routed_scaling_factor must be positive"
  | dMoeSharedGateScalar d && dMoeNumSharedExperts d == 0 =
      Just "moe_shared_gate_scalar needs at least one shared expert"
  | otherwise = Nothing

rolesCheck :: Descriptor -> Maybe String
rolesCheck d
  | length roles /= length (nub roles) = Just "duplicate role names"
  | any null templates = Just "empty weight template"
  | otherwise = case [r | r <- required, r `notElem` roles] of
      [] -> Nothing
      missing -> Just ("missing weight roles: " ++ show (map roleName missing))
  where
    roles = map fst (dRoleTemplates d)
    templates = map snd (dRoleTemplates d)
    hasFull = MFullAttention `elem` dLayerMixers d
    hasGdn = MGatedDeltaNet `elem` dLayerMixers d
    hasMoe = FMoe `elem` dLayerFfns d
    global = [REmbed, RLmHead, RFinalNorm, RInputNorm, RPostNorm]
            ++ if hasMoe then [RMoeRouter, RMoeExpertGate, RMoeExpertUp, RMoeExpertDown]
               else [RMlpGate, RMlpUp, RMlpDown]
    attn
      | not hasFull = []
      | dAttnQkNorm d = [RAttnQ, RAttnK, RAttnV, RAttnO, RAttnQNorm, RAttnKNorm]
      | otherwise = [RAttnQ, RAttnK, RAttnV, RAttnO]
    -- GDN layers need either the separate projections (Qwen3.5) or the fused
    -- qkvz/ba pair (Qwen3-Next).
    gdnCommon = [RGdnConv1d, RGdnDtBias, RGdnALog, RGdnOut, RGdnNorm]
    gdnSeparate = [RGdnQkv, RGdnZ, RGdnA, RGdnB]
    gdnFused = [RGdnQkvz, RGdnBa]
    gdn
      | not hasGdn = []
      | all (`elem` roles) gdnSeparate = gdnCommon ++ gdnSeparate
      | otherwise = gdnCommon ++ gdnFused
    required = global ++ attn ++ gdn
