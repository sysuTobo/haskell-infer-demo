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
  , descVersion
  , encodeDescriptor
  , decodeDescriptor
  , validateDescriptor
  , withMaxSeqLen
  , mixerName
  , ffnName
  , roleName
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
  , dEosTokens :: [Int]
  , dLayerMixers :: [MixerKind]
  , dLayerFfns :: [FfnKind]
  , dRoleTemplates :: [(Role, String)]
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
  , "rotary_theta", "attn_output_gate", "q_gate_interleave", "gdn_conv_dim"
  , "gdn_value_dim", "gdn_num_v_heads", "gdn_num_k_heads", "gdn_head_dim"
  , "gdn_conv_kernel", "fla_chunk_size", "max_chunk", "eos_tokens", "layer_mixers"
  , "layer_ffns", "role_names", "role_templates"
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
  , "eos_tokens" .= dEosTokens d
  , "layer_mixers" .= map mixerName (dLayerMixers d)
  , "layer_ffns" .= map ffnName (dLayerFfns d)
  , "role_names" .= map (roleName . fst) (dRoleTemplates d)
  , "role_templates" .= map snd (dRoleTemplates d)
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
  eos <- reqIntList obj "eos_tokens"
  mixers <- traverse parseMixer =<< reqTextList obj "layer_mixers"
  ffns <- traverse parseFfn =<< reqTextList obj "layer_ffns"
  roleNames <- reqTextList obj "role_names"
  templates <- reqTextList obj "role_templates"
  roles <- traverse parseRole roleNames
  pure Descriptor
    { dVersion = version, dFamily = families, dModelType = modelType
    , dNumLayers = numLayers, dHiddenSize = hidden, dIntermediateSize = intermediate
    , dVocabSize = vocab, dRmsEps = eps, dMaxPositionEmbeddings = maxPos
    , dMaxSeqLen = maxSeq, dNumHeads = heads, dNumKvHeads = kvHeads, dHeadDim = headDim
    , dRotaryDim = rotaryDim, dRotaryTheta = theta, dAttnOutputGate = outGate
    , dQGateInterleave = qgInterleave, dGdnConvDim = convDim, dGdnValueDim = valueDim
    , dGdnNumVHeads = vHeads, dGdnNumKHeads = kHeads, dGdnHeadDim = gdnHeadDim
    , dGdnConvKernel = convKernel, dFlaChunkSize = chunkSize, dMaxChunk = maxChunk
    , dEosTokens = eos
    , dLayerMixers = mixers, dLayerFfns = ffns
    , dRoleTemplates = zip roles templates
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

reqInt :: Object -> String -> Either String Int
reqInt obj key = case KM.lookup (Key.fromString key) obj of
  Just (Number n) -> Right (round n)
  Just _ -> Left ("descriptor key " ++ show key ++ " must be a number")
  Nothing -> Left ("descriptor key " ++ show key ++ " is missing")

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
  , err (dRmsEps d <= 0) "rms_eps must be positive"
  , err (dMaxChunk d < 1 || dMaxChunk d > 128)
        "max_chunk must be in [1,128] (the kernels' batch limit)"
  , err (null (dEosTokens d)) "eos_tokens must not be empty"
  , err (any (\t -> t < 0 || t >= dVocabSize d) (dEosTokens d))
        "eos_tokens must be within the vocabulary"
  , gdnCheck d
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
    global = [REmbed, RLmHead, RFinalNorm, RInputNorm, RPostNorm, RMlpGate, RMlpUp, RMlpDown]
    attn = if hasFull then [RAttnQ, RAttnK, RAttnV, RAttnO] else []
    gdn = if hasGdn
      then [RGdnQkv, RGdnZ, RGdnA, RGdnB, RGdnConv1d, RGdnDtBias, RGdnALog, RGdnOut, RGdnNorm]
      else []
    required = global ++ attn ++ gdn
