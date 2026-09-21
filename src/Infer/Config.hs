-- | Model and runtime configuration for Qwen3.8-27B.
--
-- Dimensions are sourced from the model's config.json and the kern project's
-- manifest constants. The GPU partition is computed from the number of
-- available devices.
module Infer.Config
  ( ModelConfig(..)
  , qwen38_27bConfig
  , GpuPartition(..)
  , computePartition
  , RuntimeConfig(..)
  , defaultRuntimeConfig
  ) where


-- | Static model architecture parameters.
data ModelConfig = ModelConfig
  { mcNumLayers            :: !Int  -- ^ 64
  , mcHiddenSize           :: !Int  -- ^ 5120
  , mcIntermediateSize     :: !Int  -- ^ 17408
  , mcVocabSize            :: !Int  -- ^ 248320
  , mcNumHeads             :: !Int  -- ^ 24 query heads
  , mcNumKvHeads           :: !Int  -- ^ 4 KV heads (GQA)
  , mcHeadDim              :: !Int  -- ^ 256
  , mcRotaryDim            :: !Int  -- ^ 64 (partial rotary, 0.25 * 256)
  , mcRotaryTheta          :: !Double -- ^ 1e7
  , mcRmsNormEps           :: !Double -- ^ 1e-6
  , mcFullAttnInterval     :: !Int  -- ^ 4 (every 4th layer is full attention)
  , mcMaxPositionEmbeddings :: !Int -- ^ 262144
  -- GDN-specific
  , mcGdnQkvzDim           :: !Int  -- ^ 16384
  , mcGdnBaDim             :: !Int  -- ^ 96
  , mcGdnConvDim           :: !Int  -- ^ 10240
  , mcGdnQkDim             :: !Int  -- ^ 2048
  , mcGdnVDim              :: !Int  -- ^ 6144
  , mcGdnNumVHeads         :: !Int  -- ^ 48
  , mcGdnNumKHeads         :: !Int  -- ^ 16
  , mcGdnHeadDim           :: !Int  -- ^ 128
  , mcGdnConvKernelSize    :: !Int  -- ^ 4
  , mcFlaChunkSize         :: !Int  -- ^ 64
  -- Derived
  , mcQkvDim               :: !Int  -- ^ 14336 (q=6144 + k=1024 + v=1024 + gate=6144)
  , mcGateUpDim            :: !Int  -- ^ 34816 (2 * intermediate)
  , mcKvBytesPerToken      :: !Int  -- ^ 65536 (all attn layers)
  , mcGdnBytesPerSeq       :: !Int  -- ^ 154140672 (all GDN layers)
  , mcEosTokens            :: [Int] -- ^ [248046, 248044]
  }

-- | Qwen3.8-27B configuration.
qwen38_27bConfig :: ModelConfig
qwen38_27bConfig = ModelConfig
  { mcNumLayers            = 64
  , mcHiddenSize           = 5120
  , mcIntermediateSize     = 17408
  , mcVocabSize            = 248320
  , mcNumHeads             = 24
  , mcNumKvHeads           = 4
  , mcHeadDim              = 256
  , mcRotaryDim            = 64
  , mcRotaryTheta          = 1e7
  , mcRmsNormEps           = 1e-6
  , mcFullAttnInterval     = 4
  , mcMaxPositionEmbeddings = 262144
  , mcGdnQkvzDim           = 16384
  , mcGdnBaDim             = 96
  , mcGdnConvDim           = 10240
  , mcGdnQkDim             = 2048
  , mcGdnVDim              = 6144
  , mcGdnNumVHeads         = 48
  , mcGdnNumKHeads         = 16
  , mcGdnHeadDim           = 128
  , mcGdnConvKernelSize    = 4
  , mcFlaChunkSize         = 64
  , mcQkvDim               = 14336
  , mcGateUpDim            = 34816
  , mcKvBytesPerToken      = 65536
  , mcGdnBytesPerSeq       = 154140672
  , mcEosTokens            = [248046, 248044]
  }

-- | GPU partition: which layers go on which device.
data GpuPartition = GpuPartition
  { gpDevices      :: [Int]     -- ^ Device ordinals
  , gpLayerDevices :: [Int]     -- ^ Per-layer device assignment (length = numLayers)
  , gpLayersPerDev :: [[Int]]   -- ^ Layer indices grouped by device
  }

-- | Compute a balanced layer partition across devices.
--
-- Layers are assigned contiguously: device 0 gets the first N/P layers, etc.
-- Embedding goes on the first device, lm_head on the last.
computePartition :: Int -> [Int] -> GpuPartition
computePartition numLayers devices = GpuPartition
  { gpDevices      = devices
  , gpLayerDevices = layerDevices
  , gpLayersPerDev = layersPerDev
  }
  where
    n = length devices
    base = numLayers `div` n
    extra = numLayers `mod` n
    -- First `extra` devices get one more layer
    sizes = replicate extra (base + 1) ++ replicate (n - extra) base
    layerDevices = concat (zipWith (\dev sz -> replicate sz dev) devices sizes)
    layersPerDev = go 0 sizes
      where
        go _ [] = []
        go start (sz:rest) = [start .. start + sz - 1] : go (start + sz) rest

-- | Runtime configuration (user-specified).
data RuntimeConfig = RuntimeConfig
  { rcModelDir   :: FilePath  -- ^ Path to model weights directory
  , rcDevices    :: [Int]     -- ^ CUDA device ordinals to use
  , rcMaxSeqLen  :: Int       -- ^ Maximum context length
  , rcMaxTokens  :: Int       -- ^ Maximum new tokens to generate
  , rcPrompt     :: String    -- ^ Input prompt
  }

defaultRuntimeConfig :: RuntimeConfig
defaultRuntimeConfig = RuntimeConfig
  { rcModelDir   = "weights/Qwen3.8-27B"
  , rcDevices    = [0, 1]
  , rcMaxSeqLen  = 4096
  , rcMaxTokens  = 256
  , rcPrompt     = "Hello"
  }
