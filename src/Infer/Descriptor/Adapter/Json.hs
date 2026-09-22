-- | Small JSON helpers shared by the family adapters.
--
-- HuggingFace configs are loosely typed: fields move between the top level and
-- @text_config@, and optional files (like @tokenizer_config.json@) may be
-- absent. These helpers keep the adapters declarative.
module Infer.Descriptor.Adapter.Json
  ( readJsonValue
  , valueObject
  , lookupValue
  , lookupInt
  , lookupIntList
  , lookupDouble
  , lookupText
  , lookupBool
  , lookupText'
  , needInt
  ) where

import Data.Aeson (Object, Value(..), eitherDecodeStrict')
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString as BS
import Data.Scientific (toBoundedInteger)
import qualified Data.Text as T
import System.Directory (doesFileExist)

-- | Read a JSON file, returning 'Nothing' when it is missing or unparsable.
-- Optional companion files must not abort descriptor derivation.
readJsonValue :: FilePath -> IO (Maybe Value)
readJsonValue path = do
  exists <- doesFileExist path
  if not exists
    then pure Nothing
    else do
      bytes <- BS.readFile path
      pure (either (const Nothing) Just (eitherDecodeStrict' bytes))

valueObject :: Value -> Maybe Object
valueObject (Object o) = Just o
valueObject _ = Nothing

lookupValue :: Object -> String -> Maybe Value
lookupValue obj key = KM.lookup (Key.fromString key) obj

lookupInt :: Object -> String -> Maybe Int
lookupInt obj key = lookupValue obj key >>= asInt

lookupIntList :: Object -> String -> Maybe [Int]
lookupIntList obj key = case lookupValue obj key of
  Just (Array values) -> traverse asInt (foldr (:) [] values)
  _ -> Nothing

lookupDouble :: Object -> String -> Maybe Double
lookupDouble obj key = case lookupValue obj key of
  Just (Number n) -> Just (realToFrac n)
  _ -> Nothing

lookupText :: Object -> String -> Maybe String
lookupText obj key = case lookupValue obj key of
  Just (String t) -> Just (T.unpack t)
  _ -> Nothing

lookupBool :: Object -> String -> Maybe Bool
lookupBool obj key = case lookupValue obj key of
  Just (Bool b) -> Just b
  _ -> Nothing

-- | Look up a key inside a nested object value.
lookupText' :: Value -> String -> Maybe String
lookupText' value key = case valueObject value of
  Just obj -> lookupText obj key
  Nothing -> Nothing

needInt :: Object -> String -> Either String Int
needInt obj key = maybe (Left ("config.json missing " ++ key)) Right (lookupInt obj key)

asInt :: Value -> Maybe Int
asInt (Number n) = toBoundedInteger n
asInt _ = Nothing
