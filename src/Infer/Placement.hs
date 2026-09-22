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
import Data.Maybe (fromMaybe)

import Infer.Descriptor

-- | Activation discipline. 'Pipelined' is the layer-wise split: each layer lives
-- on one device and the residual hops device-to-device once per layer.
-- 'Replicated' keeps the whole model on every device and splits the weights of
-- the tensor-parallel roles (see 'defaultShards'); activations are replicated
-- and reduced after each sharded sublayer. @ep@ is expert parallelism, which is
-- validated here but lands in a later increment, so it must be 1 for now.
data Policy = Pipelined | Replicated { rpTp :: Int, rpEp :: Int }
  deriving (Eq, Show)

data Placement = Placement
  { plPolicy :: !Policy
  , plDevices :: [Int]            -- ^ CUDA device ordinals
  , plLayerDevices :: [Int]       -- ^ per-layer device ordinal: the owner under
                                  -- 'Pipelined'; the first device under
                                  -- 'Replicated', where every device runs every
                                  -- layer (the engine ignores this array then)
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
  case policy of
    Pipelined -> pipelined desc devices
    Replicated tp ep -> replicated desc tp ep devices

-- | Layer-wise (pipeline) split: every layer is owned by exactly one device.
pipelined :: Descriptor -> [Int] -> Either String Placement
pipelined desc devices = do
  case dNumLayers desc < length devices of
    True -> Left ("fewer layers (" ++ show (dNumLayers desc) ++ ") than devices ("
                  ++ show (length devices) ++ ")")
    False -> Right ()
  let assigned = contiguousSplit (dNumLayers desc) devices
      layerDevices = concat (zipWith (\dev ls -> replicate (length ls) dev) devices assigned)
      result = Placement Pipelined devices layerDevices assigned
  validatePlacement desc result
  pure result

-- | Replicated placement: @tp@ ranks holding shards of the same layers on
-- @tp * ep@ devices, all running the full forward pass. The device list order is
-- the rank order (rank r = devices !! r).
replicated :: Descriptor -> Int -> Int -> [Int] -> Either String Placement
replicated desc tp ep devices = do
  check (tp < 1) "tp must be at least 1"
  check (ep < 1) "ep must be at least 1"
  check (ep /= 1) "expert parallel (ep > 1) is not implemented"
  check (length devices /= tp * ep)
    ("replicated placement needs tp * ep = " ++ show (tp * ep)
     ++ " devices, got " ++ show (length devices))
  case [msg | msg <- shardSizeProblems desc tp] of
    [] -> Right ()
    msg : _ -> Left msg
  case [i | (i, FMoe) <- zip [0 :: Int ..] (dLayerFfns desc)] of
    [] -> Right ()
    i : _ -> Left ("layer " ++ show i ++ " is MoE; tensor parallelism with MoE layers"
                   ++ " is not implemented yet (expert parallel lands separately)")
  let layers = [0 .. dNumLayers desc - 1]
      result = Placement (Replicated tp ep) devices
                         (replicate (dNumLayers desc) (head devices))
                         (replicate (length devices) layers)
  validatePlacement desc result
  pure result
  where
    check True message = Left message
    check False _ = Right ()

-- | The sharded roles need their split dimension to divide evenly by @tp@.
shardSizeProblems :: Descriptor -> Int -> [String]
shardSizeProblems desc tp =
  concat
    [ problem "num_heads" (dNumHeads desc) [RAttnQ]
    , problem "num_kv_heads" (dNumKvHeads desc) [RAttnK, RAttnV]
    , problem "num_heads * head_dim" (dNumHeads desc * dHeadDim desc) [RAttnO]
    , problem "intermediate_size" (dIntermediateSize desc) [RMlpGate, RMlpUp, RMlpDown]
    ]
  where
    rule role = fromMaybe ShardNone (lookup role (shardRules desc))
    problem name size roles
      | any ((/= ShardNone) . rule) roles && size `mod` tp /= 0 =
          ["tp " ++ show tp ++ " does not divide " ++ name ++ " (" ++ show size ++ ")"]
      | otherwise = []

-- | The role table paired with its shard rules.
shardRules :: Descriptor -> [(Role, ShardKind)]
shardRules d = zip (map fst (dRoleTemplates d)) (dRoleShards d)

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
