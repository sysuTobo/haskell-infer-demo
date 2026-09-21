{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

-- | Safetensors weight file parser.
--
-- Format: 8-byte LE header length, JSON header, then raw tensor data.
-- We parse the header to locate tensors, then mmap the file for zero-copy
-- access to weight data.
module Infer.Safetensors
  ( TensorMeta(..)
  , TensorData(..)
  , SafetensorsFile(..)
  , loadSafetensorsHeader
  , listSafetensorsFiles
  , getTensorPtr
  , tensorByteSize
  ) where

import Data.Aeson
import qualified Data.ByteString as BS
import Data.Word (Word64)
import Data.List (sort)
import Foreign.Ptr
import Foreign.ForeignPtr
import System.Directory (listDirectory, doesFileExist)
import System.FilePath ((</>), takeExtension)
import GHC.Generics (Generic)
import Data.Bits ((.|.), shiftL)
import qualified Data.Map.Strict as Map

-- | Tensor metadata from the safetensors header.
data TensorMeta = TensorMeta
  { tmDtype       :: !String      -- ^ "BF16", "F32", "F16", etc.
  , tmShape       :: [Int]        -- ^ Tensor dimensions
  , tmDataOffsets :: (Int, Int)   -- ^ (start, end) byte offsets in data section
  } deriving (Eq, Show, Generic)

instance FromJSON TensorMeta where
  parseJSON = withObject "TensorMeta" $ \v -> do
    dtype <- v .: "dtype"
    shape <- v .: "shape"
    offsets <- v .: "data_offsets"
    case offsets of
      [s, e] -> return TensorMeta { tmDtype = dtype, tmShape = shape, tmDataOffsets = (s, e) }
      _ -> fail "data_offsets must have exactly 2 elements"

-- | A loaded safetensors file with its header and data pointer.
data SafetensorsFile = SafetensorsFile
  { sfPath    :: !FilePath
  , sfHeader  :: !(Map.Map String TensorMeta)
  , sfDataPtr :: !(Ptr ())      -- ^ Pointer to the start of tensor data (mmap'd)
  , sfDataFp  :: ForeignPtr ()  -- ^ Keeps the mmap alive
  }

-- | Tensor data: a pointer plus metadata.
data TensorData = TensorData
  { tdPtr  :: !(Ptr ())
  , tdMeta :: !TensorMeta
  }

-- | Read the header size from the first 8 bytes (little-endian u64).
readHeaderSize :: BS.ByteString -> Maybe Word64
readHeaderSize bs
  | BS.length bs < 8 = Nothing
  | otherwise = Just (foldr (\b acc -> acc `shiftL` 8 .|. fromIntegral b) 0 (BS.unpack (BS.take 8 bs)))

-- | Load and parse the header of a safetensors file.
-- Returns the tensor metadata map and the byte offset where data begins.
loadSafetensorsHeader :: FilePath -> IO (Maybe (Map.Map String TensorMeta, Int))
loadSafetensorsHeader path = do
  exists <- doesFileExist path
  if not exists then return Nothing else do
    -- Read just the first 8 bytes for header size
    h8 <- BS.readFile path
    case readHeaderSize h8 of
      Nothing -> return Nothing
      Just hdrSize -> do
        let hdrStart = 8
            hdrEnd = hdrStart + fromIntegral hdrSize
        -- Re-read with the header
        contents <- BS.readFile path
        let headerJson = BS.take (fromIntegral hdrSize) (BS.drop hdrStart contents)
        case eitherDecodeStrict' headerJson of
          Left _ -> return Nothing
          Right meta -> return (Just (meta, hdrEnd))

-- | List all .safetensors files in a directory, sorted by name.
listSafetensorsFiles :: FilePath -> IO [FilePath]
listSafetensorsFiles dir = do
  entries <- listDirectory dir
  let stFiles = sort [dir </> f | f <- entries, takeExtension f == ".safetensors"]
  filterM doesFileExist stFiles
  where
    filterM _ [] = return []
    filterM p (x:xs) = do
      b <- p x
      if b then (x:) <$> filterM p xs else filterM p xs

-- | Get a pointer to a specific tensor's data within a loaded file.
getTensorPtr :: SafetensorsFile -> String -> Maybe TensorData
getTensorPtr sf name =
  case Map.lookup name (sfHeader sf) of
    Nothing -> Nothing
    Just meta ->
      let (startOff, _) = tmDataOffsets meta
          ptr = plusPtr (sfDataPtr sf) startOff
      in Just (TensorData ptr meta)

-- | Compute the byte size of a tensor from its metadata.
tensorByteSize :: TensorMeta -> Int
tensorByteSize meta =
  let (s, e) = tmDataOffsets meta
  in e - s
