{-# LANGUAGE OverloadedStrings #-}

-- | Manifest comparison gates (pure; no GPU, no weights).
--
-- These are the Stage 0 admission rules: a semantic/numerical/weight change
-- rejects, a placement-only change needs a declared scope and is still reported
-- as such, differing provenance rejects, an unestablished provenance rejects even
-- when both sides agree on it, and a document without a manifest version is
-- legacy/unverified rather than blessed.
module ManifestSpec (manifestSpec) where

import Data.Aeson (Value(..), encode)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.List (isInfixOf)
import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V
import Test.Hspec

import Infer.Manifest

-- -----------------------------------------------------------------------
-- Fixtures
-- -----------------------------------------------------------------------

-- | A structurally complete manifest. It is not a *canonical* document (the
-- emitter's encoding is verified in C and Python); what matters here is that the
-- comparison reads the same fields the engine writes.
fixture :: Value
fixture = Object $ KM.fromList
  [ ("manifest_version", Number 1)
  , ("descriptor", Object $ KM.fromList
      [ ("bytes", Number 10), ("desc_version", Number 1), ("sha256", String "desc-sha") ])
  , ("semantic", Object $ KM.fromList
      [ ("fields", Object $ KM.fromList [("hidden_size", Number 256)])
      , ("semantic_id", String "semantic-1") ])
  , ("numerical_policy", Object $ KM.fromList
      [ ("fields", Object $ KM.fromList [("max_chunk", Number 64)])
      , ("numerical_policy_id", String "numerical-1") ])
  , ("deployment", Object $ KM.fromList
      [ ("fields", Object $ KM.fromList [("placement", String "layer_split")])
      , ("deployment_id", String "deployment-1") ])
  , ("provenance", Object $ KM.fromList
      [ ("build", Object $ KM.fromList
          [ ("cuda_toolkit", String "12.9"), ("git_commit", String "abc123") ])
      , ("runtime", Object $ KM.fromList
          [ ("cublas_version", String "12.9.0")
          , ("cuda_runtime_version", String "12.9") ]) ])
  , ("sampling", Object $ KM.fromList
      [ ("mode", String "greedy"), ("rng", String "none") ])
  , ("weights", Object $ KM.fromList
      [ ("parameter_manifest_sha256", String "weights-1")
      , ("tensor_count", Number 42), ("content_sha256", Null) ])
  , ("regions", Array (V.fromList
      [ Object $ KM.fromList
          [ ("region", String "gemm_bf16"), ("cases", String "prefill,decode")
          , ("implementation", String "cublas bf16 in, fp32 accumulate")
          , ("determinism", String "unverified")
          , ("mechanism", String "cuBLAS algorithm selection is not pinned")
          , ("rng_dependency", Bool False) ] ]))
  ]

-- | Override a nested field, so a variant differs from the fixture in exactly
-- one place.
override :: [T.Text] -> Value -> Value -> Value
override path replacement root = go path root
  where
    go [] _ = replacement
    go (key : rest) (Object obj) =
      let child = maybe Null id (KM.lookup (Key.fromText key) obj)
      in Object (KM.insert (Key.fromText key) (go rest child) obj)
    go _ other = other

withField :: [T.Text] -> Value -> Value
withField path replacement = override path replacement fixture

document :: Value -> BS.ByteString
document = BL.toStrict . encode

-- | A canonical document as the C emitter writes one (keys sorted, no
-- whitespace, non-integer constants as strings). Parsing it is the structural
-- half of the cross-language check; the digests themselves are verified in
-- Python, where a second implementation re-derives them.
canonicalDocument :: BS.ByteString
canonicalDocument = TE.encodeUtf8 $ T.concat
  [ "{\"deployment\":{\"deployment_id\":\"dep\",\"fields\":{\"devices\":[0,1]}},"
  , "\"descriptor\":{\"bytes\":123,\"desc_version\":1,\"sha256\":\"desc\"},"
  , "\"manifest_version\":1,"
  , "\"numerical_policy\":{\"fields\":{\"max_chunk\":64},\"numerical_policy_id\":\"num\"},"
  , "\"provenance\":{\"build\":{\"cuda_toolkit\":\"12.9\"},"
  , "\"runtime\":{\"cuda_runtime_version\":\"12.9\"}},"
  , "\"regions\":[{\"cases\":\"prefill\",\"determinism\":\"deterministic\","
  , "\"implementation\":\"embedding row gather\",\"mechanism\":\"no reduction\","
  , "\"region\":\"embedding\",\"rng_dependency\":false}],"
  , "\"sampling\":{\"arithmetic\":\"host_fp32_logits_exact_compare\",\"mode\":\"greedy\","
  , "\"rng\":\"none\",\"transform\":\"argmax_lowest_token_id\",\"transform_version\":1},"
  , "\"semantic\":{\"fields\":{\"hidden_size\":256},\"semantic_id\":\"sem\"},"
  , "\"weights\":{\"content_sha256\":null,\"parameter_manifest_sha256\":\"w\","
  , "\"shards_sha256\":\"s\",\"tensor_count\":7}}"
  ]

-- -----------------------------------------------------------------------
-- Specs
-- -----------------------------------------------------------------------

compareFixture :: CompareMode -> Bool -> Value -> Value -> Verdict
compareFixture mode scoped left right =
  case compareDocuments mode scoped (document left) (document right) of
    Left err -> error ("comparison failed to parse its own fixture: " ++ err)
    Right (verdict, _) -> verdict

legacyDocument :: BS.ByteString
legacyDocument = document $ Object $ KM.fromList
  [ ("meta", String "a pre-manifest capture") ]

manifestSpec :: Spec
manifestSpec = describe "Manifest" $ do
  describe "parsing" $ do
    it "reads the identities, the parameter identity and the regions" $ do
      case parseManifest canonicalDocument of
        Left err -> expectationFailure err
        Right manifest -> do
          mVersion manifest `shouldBe` 1
          ibSemanticId (mIdentities manifest) `shouldBe` "sem"
          ibNumericalPolicyId (mIdentities manifest) `shouldBe` "num"
          ibDeploymentId (mIdentities manifest) `shouldBe` "dep"
          wrParameterManifestSha256 (mWeights manifest) `shouldBe` "w"
          wrTensorCount (mWeights manifest) `shouldBe` Just 7
          wrContentSha256 (mWeights manifest) `shouldBe` Nothing
          map rbRegion (mRegions manifest) `shouldBe` ["embedding"]
          rbRngDependency (head (mRegions manifest)) `shouldBe` False

    it "reads the sampling policy and the provenance blocks" $ do
      case parseManifest canonicalDocument of
        Left err -> expectationFailure err
        Right manifest -> do
          fieldIs ["sampling", "mode"] "greedy" (mRaw manifest)
          fieldIs ["provenance", "build", "cuda_toolkit"] "12.9" (mRaw manifest)
          fieldIs ["weights", "content_sha256"] "null" (mRaw manifest)
          unestablishedPaths manifest `shouldBe` []

    it "says whether the weight content was hashed" $ do
      case parseManifest canonicalDocument of
        Left err -> expectationFailure err
        Right manifest -> do
          let line = unwords [ l | l <- manifestSummary manifest, "weights:" `isInfixOf` l ]
          line `shouldContain` "content NOT hashed"
          line `shouldContain` "over 7 tensors"

    it "reports a missing manifest version as legacy, not as an error" $ do
      manifestVersionOf legacyDocument `shouldBe` Right Nothing
      parseManifest legacyDocument `shouldBe` Left "the manifest is missing 'manifest_version'"

  describe "strict admission" $ do
    it "admits two identical manifests" $ do
      compareFixture StrictAdmission False fixture fixture `shouldBe` Admitted
      verdictExitCode Admitted `shouldBe` 0

    it "rejects a semantic change and names the field" $ do
      let changed = withField ["semantic", "semantic_id"] (String "semantic-2")
      case compareFixture StrictAdmission False changed fixture of
        Rejected diffs -> map fdPath diffs `shouldContain` ["semantic.semantic_id"]
        other -> expectationFailure ("expected Rejected, got " ++ show other)

    it "rejects a numerical-policy change" $ do
      let changed = withField ["numerical_policy", "numerical_policy_id"] (String "numerical-2")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "numerical_policy.numerical_policy_id"

    it "rejects a parameter-identity change" $ do
      let changed = withField ["weights", "parameter_manifest_sha256"] (String "weights-2")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "weights.parameter_manifest_sha256"

    it "rejects a provenance change" $ do
      let changed = withField ["provenance", "build", "cuda_toolkit"] (String "13.0")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "provenance.build.cuda_toolkit"

    it "rejects provenance that is not established even when both sides agree" $ do
      let changed = withField ["provenance", "build", "cuda_toolkit"] (String "unknown")
      compareFixture StrictAdmission False changed changed
        `shouldSatisfy` isRejectedWithPath "provenance.build.cuda_toolkit"

    it "rejects a sampling policy that was not recorded" $ do
      let changed = withField ["sampling", "mode"] (String "unspecified")
      compareFixture StrictAdmission False changed changed
        `shouldSatisfy` isRejectedWithPath "sampling.mode"

    it "rejects a sampler-policy change, which is numerical policy and not request data" $ do
      let changed = withField ["sampling", "transform"] (String "categorical_softmax_cdf")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "sampling.transform"

    it "rejects two content hashes that disagree but not one that is missing" $ do
      let hashed = withField ["weights", "content_sha256"] (String "content-1")
          rehashed = withField ["weights", "content_sha256"] (String "content-2")
      compareFixture StrictAdmission False hashed rehashed
        `shouldSatisfy` isRejectedWithPath "weights.content_sha256"
      -- The fixture has no content hash: "not hashed" must not read as "different".
      compareFixture StrictAdmission False hashed fixture `shouldBe` Admitted

  describe "deployment scope" $ do
    it "rejects a placement-only change when no scope is declared" $ do
      let changed = withField ["deployment", "deployment_id"] (String "deployment-2")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "deployment.deployment_id"

    it "admits a placement-only change only with a declared scope" $ do
      let changed = withField ["deployment", "deployment_id"] (String "deployment-2")
      case compareFixture StrictAdmission True changed fixture of
        ScopedDeploymentAdmitted diffs -> do
          map fdPath diffs `shouldContain` ["deployment.deployment_id"]
          verdictKind (ScopedDeploymentAdmitted diffs)
            `shouldBe` "admitted-with-declared-deployment-scope"
        other -> expectationFailure ("expected a scoped admission, got " ++ show other)

    it "treats the descriptor reference as scoped, because it carries placement" $ do
      let changed = withField ["descriptor", "sha256"] (String "desc-sha-2")
      compareFixture StrictAdmission False changed fixture
        `shouldSatisfy` isRejectedWithPath "descriptor.sha256"
      compareFixture StrictAdmission True changed fixture
        `shouldSatisfy` isScopedAdmission

    it "does not let the deployment scope excuse an identity change" $ do
      let changed = withField ["semantic", "semantic_id"] (String "semantic-2")
      compareFixture StrictAdmission True changed fixture `shouldSatisfy` isRejected

  describe "diagnostic mode" $ do
    it "reports a difference without calling it a pass" $ do
      let changed = withField ["numerical_policy", "numerical_policy_id"] (String "numerical-2")
      case compareFixture Diagnostic False changed fixture of
        DiagnosticOnly diffs -> map fdPath diffs `shouldContain` ["numerical_policy.numerical_policy_id"]
        other -> expectationFailure ("expected DiagnosticOnly, got " ++ show other)
      verdictExitCode (compareFixture Diagnostic False changed fixture) `shouldBe` 3

  describe "legacy documents" $ do
    it "reports legacy/unverified instead of a verdict" $ do
      case compareDocuments StrictAdmission False legacyDocument legacyDocument of
        Left err -> expectationFailure err
        Right (verdict, diffs) -> do
          verdict `shouldSatisfy` isLegacy
          diffs `shouldBe` []
          verdictExitCode verdict `shouldBe` 2

  describe "field flattening" $ do
    it "renders a nested difference as a dotted path" $ do
      let flat = flattenValue fixture
      M.lookup "provenance.build.git_commit" flat `shouldBe` Just "abc123"
      M.lookup "regions.0.determinism" flat `shouldBe` Just "unverified"
      M.lookup "weights.content_sha256" flat `shouldBe` Just "null"

  where
    isRejected (Rejected _) = True
    isRejected _ = False
    isRejectedWithPath path (Rejected diffs) = path `elem` map fdPath diffs
    isRejectedWithPath _ _ = False
    isScopedAdmission (ScopedDeploymentAdmitted _) = True
    isScopedAdmission _ = False
    isLegacy (LegacyUnverified _) = True
    isLegacy _ = False
    fieldIs path expected value =
      M.lookup (T.intercalate "." path) (flattenValue value) `shouldBe` Just expected
