{-# LANGUAGE OverloadedStrings #-}

-- | The versioned execution manifest: parsing, field-level diffing and the
-- comparison modes.
--
-- The engine produces the canonical manifest text (@engine_manifest@); this
-- module is the Haskell consumer. It answers the two questions the numerical
-- contract separates:
--
--   * which execution did a capture observe (the content identities plus the
--     build\/runtime provenance), and
--   * may a later run be compared with it bitwise?
--
-- The comparison modes are deliberately not interchangeable. 'StrictAdmission'
-- refuses a capture unless the identities and the provenance agree, and refuses
-- an unestablished provenance value outright: an unknown numerical setting is not
-- a default. 'Diagnostic' compares a deliberate manifest difference and reports
-- it, but never as a strict contract pass. A document without a manifest version
-- is 'LegacyUnverified': its numeric arrays stay comparable, but nothing about it
-- is blessed.
--
-- Field classification, which is the whole point of the exercise:
--
--   * @semantic.*@, @numerical_policy.*@ and the parameter identity are strict
--     identities. A difference is a rejection even under a declared scope.
--   * @deployment.*@ and @descriptor.*@ are scoped: the descriptor's canonical
--     text carries the placement fields (tp\/ep sizes, the shard plan), so a
--     descriptor reference that differs while both identities agree is a
--     placement-shaped difference. It is tolerated only when the caller declares
--     a scoped invariance claim covering it, and even then it is reported as
--     such rather than as an identity-level pass.
--   * @provenance.*@ is neither identity nor placement: it must be established
--     (no "unknown") and equal.
module Infer.Manifest
  ( -- * The manifest
    Manifest(..)
  , DescriptorRef(..)
  , IdentityBlock(..)
  , WeightsRef(..)
  , RegionBinding(..)
  , parseManifest
  , manifestVersionOf
  , manifestSummary
    -- * Comparison
  , CompareMode(..)
  , parseCompareMode
  , compareModeName
  , Verdict(..)
  , verdictKind
  , verdictExitCode
  , renderVerdict
  , FieldDiff(..)
  , compareDocuments
  , compareManifests
  , flattenValue
  , unestablishedPaths
  ) where

import Data.Aeson (Value(..), eitherDecodeStrict')
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import Data.List (intercalate, sort)
import qualified Data.Map.Strict as M
import Data.Scientific (floatingOrInteger)
import qualified Data.Text as T
import qualified Data.Vector as V

-- -----------------------------------------------------------------------
-- Types
-- -----------------------------------------------------------------------

-- | The descriptor this execution was built from: a reference, not the content.
data DescriptorRef = DescriptorRef
  { drVersion :: Int
  , drSha256  :: T.Text
  , drBytes   :: Integer
  } deriving (Eq, Show)

-- | The three content identities and the canonical field blocks they hash.
data IdentityBlock = IdentityBlock
  { ibSemanticId        :: T.Text
  , ibNumericalPolicyId :: T.Text
  , ibDeploymentId      :: T.Text
  , ibSemanticFields    :: M.Map T.Text Value
  , ibNumericalFields   :: M.Map T.Text Value
  , ibDeploymentFields  :: M.Map T.Text Value
  } deriving (Eq, Show)

-- | The parameter identity. A null @content_sha256@ means the raw bytes were not
-- hashed (it costs minutes of I/O on a 50 GiB checkpoint), which is a different
-- statement from "the content differs".
data WeightsRef = WeightsRef
  { wrParameterManifestSha256 :: T.Text
  , wrTensorCount             :: Maybe Integer
  , wrContentSha256           :: Maybe T.Text
  } deriving (Eq, Show)

-- | One region: which implementation ran, for which cases, and how repeatable
-- and RNG-dependent it is.
data RegionBinding = RegionBinding
  { rbRegion         :: T.Text
  , rbCases          :: T.Text
  , rbImplementation :: T.Text
  , rbDeterminism    :: T.Text
  , rbMechanism      :: T.Text
  , rbRngDependency  :: Bool
  } deriving (Eq, Show)

data Manifest = Manifest
  { mVersion    :: Int
  , mDescriptor :: DescriptorRef
  , mIdentities :: IdentityBlock
  , mWeights    :: WeightsRef
  , mRegions    :: [RegionBinding]
  , mProvenance :: Value
  , mSampling   :: Value
  , mRaw        :: Value
  } deriving (Eq, Show)

-- -----------------------------------------------------------------------
-- Parsing
-- -----------------------------------------------------------------------

type Obj = KM.KeyMap Value

lookupField :: T.Text -> Obj -> Either String Value
lookupField key obj =
  maybe (Left ("the manifest is missing '" ++ T.unpack key ++ "'")) Right
        (KM.lookup (Key.fromText key) obj)

asObj :: T.Text -> Value -> Either String Obj
asObj _ (Object o) = Right o
asObj what _ = Left ("'" ++ T.unpack what ++ "' is not a JSON object")

asArr :: T.Text -> Value -> Either String [Value]
asArr _ (Array a) = Right (V.toList a)
asArr what _ = Left ("'" ++ T.unpack what ++ "' is not a JSON array")

asText :: T.Text -> Value -> Either String T.Text
asText _ (String s) = Right s
asText what _ = Left ("'" ++ T.unpack what ++ "' is not a string")

asInteger :: T.Text -> Value -> Either String Integer
asInteger what (Number n) =
  either (const (Left ("'" ++ T.unpack what ++ "' is not an integer"))) Right
         (floatingOrInteger n :: Either Double Integer)
asInteger what _ = Left ("'" ++ T.unpack what ++ "' is not a number")

asInt :: T.Text -> Value -> Either String Int
asInt what value = fromIntegral <$> asInteger what value

asBool :: T.Text -> Value -> Either String Bool
asBool _ (Bool b) = Right b
asBool what _ = Left ("'" ++ T.unpack what ++ "' is not a boolean")

-- | The manifest version, or 'Nothing' for a document that has none (a legacy
-- capture, whose identity must not be invented).
manifestVersionOf :: BS.ByteString -> Either String (Maybe Int)
manifestVersionOf bytes = do
  raw <- eitherDecodeStrict' bytes
  case raw of
    Object obj -> case KM.lookup "manifest_version" obj of
      Nothing -> Right Nothing
      Just value -> Just <$> asInt "manifest_version" value
    _ -> Left "the document is not a JSON object"

parseManifest :: BS.ByteString -> Either String Manifest
parseManifest bytes = do
  raw <- eitherDecodeStrict' bytes
  top <- asObj "manifest" raw
  version <- lookupField "manifest_version" top >>= asInt "manifest_version"
  descriptor <- lookupField "descriptor" top >>= asObj "descriptor"
  semantic <- lookupField "semantic" top >>= asObj "semantic"
  numerical <- lookupField "numerical_policy" top >>= asObj "numerical_policy"
  deployment <- lookupField "deployment" top >>= asObj "deployment"
  weights <- lookupField "weights" top >>= asObj "weights"
  regions <- lookupField "regions" top >>= asArr "regions"
  sampling <- lookupField "sampling" top

  descRef <- DescriptorRef
    <$> (lookupField "desc_version" descriptor >>= asInt "descriptor.desc_version")
    <*> (lookupField "sha256" descriptor >>= asText "descriptor.sha256")
    <*> (lookupField "bytes" descriptor >>= asInteger "descriptor.bytes")
  identities <- IdentityBlock
    <$> blockId "semantic" "semantic_id" semantic
    <*> blockId "numerical_policy" "numerical_policy_id" numerical
    <*> blockId "deployment" "deployment_id" deployment
    <*> blockFields "semantic" semantic
    <*> blockFields "numerical_policy" numerical
    <*> blockFields "deployment" deployment
  weightsRef <- WeightsRef
    <$> (lookupField "parameter_manifest_sha256" weights
          >>= asText "weights.parameter_manifest_sha256")
    <*> optionalInteger "tensor_count" weights
    <*> optionalText "content_sha256" weights
  bindings <- mapM (parseRegion "regions") regions
  pure Manifest
    { mVersion = version
    , mDescriptor = descRef
    , mIdentities = identities
    , mWeights = weightsRef
    , mRegions = bindings
    , mProvenance = M.findWithDefault (Object KM.empty) "provenance" (objectMap top)
    , mSampling = sampling
    , mRaw = raw
    }
  where
    objectMap = M.fromList . map (\(k, v) -> (Key.toText k, v)) . KM.toList

blockId :: T.Text -> T.Text -> Obj -> Either String T.Text
blockId block key obj = lookupField key obj >>= asText (block <> "." <> key)

blockFields :: T.Text -> Obj -> Either String (M.Map T.Text Value)
blockFields block obj = do
  value <- lookupField "fields" obj
  mapValue <$> asObj (block <> ".fields") value

mapValue :: Obj -> M.Map T.Text Value
mapValue = M.fromList . map (\(k, v) -> (Key.toText k, v)) . KM.toList

optionalInteger :: T.Text -> Obj -> Either String (Maybe Integer)
optionalInteger key obj = case KM.lookup (Key.fromText key) obj of
  Nothing -> Right Nothing
  Just Null -> Right Nothing
  Just value -> Just <$> asInteger key value

optionalText :: T.Text -> Obj -> Either String (Maybe T.Text)
optionalText key obj = case KM.lookup (Key.fromText key) obj of
  Nothing -> Right Nothing
  Just Null -> Right Nothing
  Just value -> Just <$> asText key value

parseRegion :: T.Text -> Value -> Either String RegionBinding
parseRegion what value = do
  obj <- asObj what value
  let get key = lookupField key obj
  RegionBinding
    <$> (get "region" >>= asText "regions[].region")
    <*> (get "cases" >>= asText "regions[].cases")
    <*> (get "implementation" >>= asText "regions[].implementation")
    <*> (get "determinism" >>= asText "regions[].determinism")
    <*> (get "mechanism" >>= asText "regions[].mechanism")
    <*> (get "rng_dependency" >>= asBool "regions[].rng_dependency")

-- -----------------------------------------------------------------------
-- Summary
-- -----------------------------------------------------------------------

summaryField :: T.Text -> Value -> String
summaryField key (Object obj) = case KM.lookup (Key.fromText key) obj of
  Just (String s) -> T.unpack s
  Just Null -> "null"
  Just (Number n) -> show n
  Just (Bool b) -> show b
  Just _ -> "<structured>"
  Nothing -> "<absent>"
summaryField _ _ = "<absent>"

-- | A few lines a human reads before comparing anything.
manifestSummary :: Manifest -> [String]
manifestSummary manifest =
  [ "descriptor:    v" ++ show (drVersion (mDescriptor manifest))
      ++ ", " ++ show (drBytes (mDescriptor manifest)) ++ " bytes, sha256 "
      ++ T.unpack (drSha256 (mDescriptor manifest))
  , "semantic_id:   " ++ T.unpack (ibSemanticId ids)
  , "numerical_id:  " ++ T.unpack (ibNumericalPolicyId ids)
  , "deployment_id: " ++ T.unpack (ibDeploymentId ids)
  , "weights:       " ++ T.unpack (wrParameterManifestSha256 (mWeights manifest))
      ++ " over " ++ maybe "?" show (wrTensorCount (mWeights manifest)) ++ " tensors"
      ++ maybe ", content NOT hashed" (const ", content hashed")
             (wrContentSha256 (mWeights manifest))
  , "regions:       " ++ show (length (mRegions manifest)) ++ " recorded, "
      ++ show (length [ r | r <- mRegions manifest, rbDeterminism r == "unverified" ])
      ++ " unverified"
  , "sampling:      " ++ summaryField "mode" (mSampling manifest)
      ++ ", rng=" ++ summaryField "rng" (mSampling manifest)
  , "build:         " ++ summaryField "engine_build" build
      ++ " @ " ++ summaryField "git_commit" build
      ++ ", cuda " ++ summaryField "cuda_toolkit" build
      ++ ", triton " ++ summaryField "triton_version" build
      ++ ", fla " ++ summaryField "fla_version" build
  , "runtime:       " ++ summaryField "cuda_runtime_version" runtime
      ++ ", driver " ++ summaryField "cuda_driver_version" runtime
      ++ ", cublas " ++ summaryField "cublas_version" runtime
  ]
  where
    ids = mIdentities manifest
    build = M.findWithDefault Null "build" provenanceMap
    runtime = M.findWithDefault Null "runtime" provenanceMap
    provenanceMap = case mProvenance manifest of
      Object obj -> mapValue obj
      _ -> M.empty

-- -----------------------------------------------------------------------
-- Field-level diff
-- -----------------------------------------------------------------------

-- | One leaf that differs, as a dotted path.
data FieldDiff = FieldDiff
  { fdPath  :: T.Text
  , fdLeft  :: Maybe T.Text
  , fdRight :: Maybe T.Text
  } deriving (Eq, Show)

-- | Flatten a JSON value into leaf paths, so a difference reads as a field name
-- instead of two whole documents.
flattenValue :: Value -> M.Map T.Text T.Text
flattenValue = go []
  where
    go path (Object obj)
      | null (KM.toList obj) = M.singleton (joinPath path) "{}"
      | otherwise = M.unions
          [ go (path ++ [Key.toText k]) v | (k, v) <- KM.toList obj ]
    go path (Array items)
      | null (V.toList items) = M.singleton (joinPath path) "[]"
      | otherwise = M.unions
          [ go (path ++ [T.pack (show i)]) v
          | (i, v) <- zip [0 :: Int ..] (V.toList items) ]
    go path value = M.singleton (joinPath path) (renderLeaf value)
    joinPath [] = "<root>"
    joinPath parts = T.intercalate "." parts
    renderLeaf (String s) = s
    renderLeaf (Number n) = T.pack (either (const (show n)) show
                                           (floatingOrInteger n :: Either Double Integer))
    renderLeaf (Bool b) = if b then "true" else "false"
    renderLeaf Null = "null"
    renderLeaf (Object _) = "{}"
    renderLeaf (Array _) = "[]"

diffValues :: Value -> Value -> [FieldDiff]
diffValues left right =
  [ FieldDiff key (M.lookup key flatLeft) (M.lookup key flatRight)
  | key <- sort (M.keys (M.union flatLeft flatRight))
  , M.lookup key flatLeft /= M.lookup key flatRight
  ]
  where
    flatLeft = flattenValue left
    flatRight = flattenValue right

-- | Paths whose value is one of the spellings that mean "not established". A
-- strict comparison refuses these even when both sides agree on them: agreeing
-- on an unknown is not establishing a fact.
unestablishedPaths :: Manifest -> [T.Text]
unestablishedPaths manifest =
  [ path
  | (path, value) <- M.toList (flattenValue (mRaw manifest))
  , "provenance." `T.isPrefixOf` path || "sampling." `T.isPrefixOf` path
  , value `elem` ["unknown", "unavailable", "unsupported", "unspecified"]
  ]

-- -----------------------------------------------------------------------
-- Comparison
-- -----------------------------------------------------------------------

data CompareMode = StrictAdmission | Diagnostic
  deriving (Eq, Show)

parseCompareMode :: String -> Either String CompareMode
parseCompareMode "strict" = Right StrictAdmission
parseCompareMode "diagnostic" = Right Diagnostic
parseCompareMode other = Left ("unknown comparison mode: " ++ other
                               ++ " (expected 'strict' or 'diagnostic')")

compareModeName :: CompareMode -> String
compareModeName StrictAdmission = "strict"
compareModeName Diagnostic = "diagnostic"

-- | The outcome of a comparison, in the vocabulary the plan fixes.
data Verdict
  = Admitted
  | ScopedDeploymentAdmitted [FieldDiff]
  | Rejected [FieldDiff]
  | DiagnosticOnly [FieldDiff]
  | LegacyUnverified String
  deriving (Eq, Show)

verdictKind :: Verdict -> String
verdictKind Admitted = "admitted"
verdictKind (ScopedDeploymentAdmitted _) = "admitted-with-declared-deployment-scope"
verdictKind (Rejected _) = "rejected"
verdictKind (DiagnosticOnly _) = "diagnostic-only"
verdictKind (LegacyUnverified _) = "legacy-unverified"

-- | 0 admitted, 1 rejected, 2 legacy\/unverified, 3 diagnostic-only. A legacy or
-- diagnostic result is deliberately not 0: neither is a strict contract pass.
verdictExitCode :: Verdict -> Int
verdictExitCode Admitted = 0
verdictExitCode (ScopedDeploymentAdmitted _) = 0
verdictExitCode (Rejected _) = 1
verdictExitCode (LegacyUnverified _) = 2
verdictExitCode (DiagnosticOnly _) = 3

renderVerdict :: Verdict -> String
renderVerdict verdict = case verdict of
  Admitted ->
    "ADMITTED (strict): the identities, the parameter identity and the provenance agree"
  ScopedDeploymentAdmitted diffs ->
    "ADMITTED with a declared deployment scope: the identities agree and only the\n"
    ++ "placement differs, which the caller declared covered by a scoped invariance\n"
    ++ "claim. This is not an identity-level strict pass.\n" ++ renderDiffs diffs
  Rejected diffs ->
    "REJECTED: strict admission failed, so a bitwise comparison of these two\n"
    ++ "captures is not covered by the contract.\n" ++ renderDiffs diffs
  DiagnosticOnly diffs ->
    "DIAGNOSTIC ONLY: the manifest difference is reported for attribution.\n"
    ++ "This is NOT a strict contract pass.\n" ++ renderDiffs diffs
  LegacyUnverified reason ->
    "LEGACY/UNVERIFIED: " ++ reason ++ "\n"
    ++ "The numeric arrays remain comparable, but no identity or provenance is\n"
    ++ "established for this document and none is invented here."
  where
    renderDiffs [] = "  (no field differences)"
    renderDiffs diffs = "  " ++ intercalate "\n  " (map renderDiff diffs)
    renderDiff diff = T.unpack (fdPath diff) ++ ": " ++ show (fdLeft diff)
                      ++ " -> " ++ show (fdRight diff)

-- | Read both documents and compare them. A document without a manifest version
-- is legacy: no identity is invented for it and no bitwise pass is claimed.
compareDocuments :: CompareMode -> Bool -> BS.ByteString -> BS.ByteString
                 -> Either String (Verdict, [FieldDiff])
compareDocuments mode deploymentScoped leftBytes rightBytes = do
  leftVersion <- manifestVersionOf leftBytes
  rightVersion <- manifestVersionOf rightBytes
  case (leftVersion, rightVersion) of
    (Just 1, Just 1) -> do
      left <- parseManifest leftBytes
      right <- parseManifest rightBytes
      pure (compareManifests mode deploymentScoped left right)
    _ -> pure (LegacyUnverified (legacyReason leftVersion rightVersion), [])
  where
    legacyReason leftVersion rightVersion = case (leftVersion, rightVersion) of
      (Nothing, Nothing) -> "neither document has a manifest version"
      (Nothing, Just other) -> rightReason other
      (Just other, Nothing) -> leftReason other
      (Just other, Just newer) -> "the left document is manifest version " ++ show other
        ++ " and the right is " ++ show newer ++ "; this build understands version 1"
    leftReason other = "the left document is manifest version " ++ show other
                       ++ ", this build understands version 1"
    rightReason other = "the right document is manifest version " ++ show other
                        ++ ", this build understands version 1"

-- | Strict admission requires the semantic, numerical-policy and parameter
-- identities to agree; the placement fields to agree unless the caller declares a
-- scoped invariance claim covering the variation; and the provenance to be
-- established and equal. Anything less is reported, never rounded up.
compareManifests :: CompareMode -> Bool -> Manifest -> Manifest -> (Verdict, [FieldDiff])
compareManifests mode deploymentScoped left right
  | mode == Diagnostic = (DiagnosticOnly diffs, diffs)
  | not (null identityDiffs) = (Rejected identityDiffs, diffs)
  | not (null scopedDiffs) && not deploymentScoped = (Rejected scopedDiffs, diffs)
  | not (null unestablished) = (Rejected (unestablishedDiffs unestablished), diffs)
  | not (null provenanceDiffs) = (Rejected provenanceDiffs, diffs)
  | not (null scopedDiffs) = (ScopedDeploymentAdmitted scopedDiffs, diffs)
  | otherwise = (Admitted, diffs)
  where
    diffs = diffValues (mRaw left) (mRaw right)
    -- A strict identity. The descriptor reference is *not* here: its canonical
    -- text carries the placement fields, so it is classified as scoped below.
    identityPaths =
      [ "semantic.semantic_id"
      , "numerical_policy.numerical_policy_id"
      , "weights.parameter_manifest_sha256"
      ]
    identityDiffs = [ diff | diff <- diffs, fdPath diff `elem` identityPaths ]
    scopedDiffs =
      [ diff | diff <- diffs
             , "deployment." `T.isPrefixOf` fdPath diff
               || "descriptor." `T.isPrefixOf` fdPath diff ]
    provenanceDiffs = [ diff | diff <- diffs, "provenance." `T.isPrefixOf` fdPath diff ]
    unestablished = unestablishedPaths left ++ unestablishedPaths right
    unestablishedDiffs paths =
      [ FieldDiff path Nothing (Just "not established") | path <- paths ]
