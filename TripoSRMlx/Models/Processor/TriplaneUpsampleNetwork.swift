//
//  PostProcessor.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/20.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
// MARK: - Post Processor

/// Post-processor that converts tokens to triplane representations
nonisolated public final class TriplaneUpsampleNetwork: Module {
    private let config: PostProcessorConfig

    @ModuleInfo(key: "upsample")
    private var upsample: ConvTransposed2d


    public init(config: PostProcessorConfig) {
        self.config = config
        // Mirror PyTorch's ConvTranspose2d parameter shapes so exported weights align
        self._upsample.wrappedValue = ConvTransposed2d(
            inputChannels: 1024,
            outputChannels: config.outputChannels,
            kernelSize: [2, 2],
            stride: [2, 2]
        )
        super.init()
    }

    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let batchSize = tokens.dim(0)
        let numPlanes = tokens.dim(1)
        let numChannels = tokens.dim(2)
        let planeSize = tokens.dim(3)
        
        let upsampled = upsample(tokens.reshaped([batchSize*numPlanes,numChannels,planeSize,planeSize]).transposed(axes:[0,2,3,1])).transposed(axes: [0,3,1,2])
        let upsampledPlaneSize = upsampled.dim(2)
        return upsampled.reshaped([batchSize,numPlanes,config.outputChannels,upsampledPlaneSize,upsampledPlaneSize])
    }
}

