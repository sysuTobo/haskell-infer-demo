-- | Placement: which device runs which layer, and (later) how layers are sharded
-- across devices.
--
-- Replaces the old hard-coded balanced split. Policies are validated in Haskell
-- so a bad device/config combination fails before any GPU memory is allocated.
module Infer.Placement
  ( Policy(..)
  , Placement(..)
  , placement
  , validatePlacement
  , layerDevice
  ) where

import Data.List (nub)

import Infer.Descriptor

-- | Activation discipline. 'Pipelined' is the layer-wise split: each layer lives
-- on one device and the residual hops device-to-device once per layer.
-- (Tensor/expert parallel policies join here once implemented.)
data Policy = Pipelined
  deriving (Eq, Show)

data Placement = Placement
  { plPolicy :: !Policy
  , plDevices :: [Int]            -- ^ CUDA device ordinals
  , plLayerDevices :: [Int]       -- ^ per-layer device ordinal (length = num_layers)
  , plLayersPerDevice :: [[Int]]  -- ^ layer indices per device, in device order
  } deriving (Eq, Show)

-- | Build a placement for the given policy.
placement :: Descriptor -> Policy -> [Int] -> Either String Placement
placement desc policy devices = do
  case devices of
    [] -> Left "no devices configured"
    _ -> Right ()
  case [d | d <- devices, d < 0] of
    [] -> Right ()
    bad -> Left ("negative device ordinals: " ++ show bad)
  case length devices /= length (nub devices) of
    False -> Right ()
    True -> Left "duplicate device ordinals"
  case dNumLayers desc < length devices of
    True -> Left ("fewer layers (" ++ show (dNumLayers desc) ++ ") than devices ("
                  ++ show (length devices) ++ ")")
    False -> Right ()
  let assigned = contiguousSplit (dNumLayers desc) devices
      layerDevices = concat (zipWith (\dev ls -> replicate (length ls) dev) devices assigned)
      result = Placement policy devices layerDevices assigned
  validatePlacement desc result
  pure result

-- | Layers are assigned contiguously: device 0 gets the first @ceil(n/d)@ layers,
-- and so on, with the remainder spread over the first devices.
contiguousSplit :: Int -> [Int] -> [[Int]]
contiguousSplit numLayers devices = go 0 sizes
  where
    n = length devices
    base = numLayers `div` n
    extra = numLayers `mod` n
    sizes = replicate extra (base + 1) ++ replicate (n - extra) base
    go _ [] = []
    go start (sz : rest) = [start .. start + sz - 1] : go (start + sz) rest

validatePlacement :: Descriptor -> Placement -> Either String ()
validatePlacement desc p
  | length (plLayerDevices p) /= dNumLayers desc =
      Left "placement layer count does not match num_layers"
  | length (plLayersPerDevice p) /= length (plDevices p) =
      Left "placement group count does not match the device list"
  | any null (plLayersPerDevice p) =
      Left "placement assigns no layers to some device"
  | not (all (`elem` plDevices p) (plLayerDevices p)) =
      Left "placement references a device that is not in the device list"
  | otherwise = Right ()

-- | Device running a layer (convenience for tests and CLI output).
layerDevice :: Placement -> Int -> Int
layerDevice p i = plLayerDevices p !! i
