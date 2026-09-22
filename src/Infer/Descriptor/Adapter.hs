-- | Family dispatch: pick the adapter for a model directory from its
-- @config.json@ @model_type@.
--
-- This is the one place that maps a checkpoint to a family; everything
-- downstream is descriptor data.
module Infer.Descriptor.Adapter
  ( descriptorFromModelDir
  , descriptorFromConfigFile
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeFileStrict)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Maybe (fromMaybe)
import qualified Data.Text as T

import Infer.Descriptor
import Infer.Descriptor.Adapter.Json
import Infer.Descriptor.Adapter.Mixtral
import Infer.Descriptor.Adapter.Qwen35

-- | Load a descriptor for a model directory by dispatching on @model_type@.
descriptorFromModelDir :: FilePath -> IO (Either String Descriptor)
descriptorFromModelDir dir = do
  config <- eitherDecodeFileStrict (dir ++ "/config.json")
  case config of
    Left err -> pure (Left ("cannot parse " ++ dir ++ "/config.json: " ++ err))
    Right value -> do
      tokenizer <- readJsonValue (dir ++ "/tokenizer_config.json")
      pure $ do
        modelType <- modelTypeOf value
        cfg <- textConfig value
        let tok = tokenizer
        if modelType `elem` ["mixtral", "qwen3_moe", "qwen3"]
          then moeDenseDescriptorFromConfig cfg tok
          else if "qwen3_5" `prefixOf` modelType || "qwen3_5_text" == modelType
            then qwen35DescriptorFromConfig cfg tok
            else Left ("no adapter for model_type " ++ show modelType)
  where
    prefixOf prefix text = take (length prefix) text == prefix

descriptorFromConfigFile :: FilePath -> IO (Either String Descriptor)
descriptorFromConfigFile path = do
  config <- eitherDecodeFileStrict path
  case config of
    Left err -> pure (Left ("cannot parse " ++ path ++ ": " ++ err))
    Right value -> pure $ do
      modelType <- modelTypeOf value
      cfg <- textConfig value
      if modelType `elem` ["mixtral", "qwen3_moe", "qwen3"]
        then moeDenseDescriptorFromConfig cfg Nothing
        else qwen35DescriptorFromConfig cfg Nothing

modelTypeOf :: Value -> Either String String
modelTypeOf (Object top) =
  case textModelType top `orElse` topLevelText top of
    Just text -> Right text
    Nothing -> Left "config.json has no model_type"
modelTypeOf _ = Left "config.json must be an object"

-- | @text_config.model_type@ wins for VL checkpoints that nest the text tower.
textModelType :: Object -> Maybe String
textModelType top = do
  Object inner <- KM.lookup (Key.fromString "text_config") top
  String text <- KM.lookup (Key.fromString "model_type") inner
  pure (T.unpack text)

topLevelText :: Object -> Maybe String
topLevelText top = case KM.lookup (Key.fromString "model_type") top of
  Just (String text) -> Just (T.unpack text)
  _ -> Nothing

orElse :: Maybe a -> Maybe a -> Maybe a
orElse (Just x) _ = Just x
orElse Nothing y = y

textConfig :: Value -> Either String Object
textConfig (Object top) = Right $ fromMaybe top $ do
  Object inner <- KM.lookup (Key.fromString "text_config") top
  pure inner
textConfig _ = Left "config.json must be an object"
