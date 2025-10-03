//
//  Triplane1DTokenizer.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/20.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
nonisolated let Triplane1DTokenizerNumChannels:Int=1024
// MARK: - Triplane Tokenizer
/// Tokenizer for generating triplane tokens
nonisolated public final class Triplane1DTokenizer: Module {
    private let config: TokenizerConfig

    @ModuleInfo(key: "embeddings")
    private var embeddings: MLXArray

    public init(config: TokenizerConfig) {
        self.config = config

        // Create triplane embeddings to match PyTorch Triplane1DTokenizer structure
        // Shape: [3, num_channels, plane_size, plane_size] = [3, 1024, 32, 32]
        let planeSize = 32
        let numChannels = Triplane1DTokenizerNumChannels
        let scale = 1.0 / sqrt(Float(numChannels))

        self._embeddings.wrappedValue = MLXRandom.normal([3, numChannels, planeSize, planeSize]) * scale
    }

    /// Generate triplane tokens by repeating embeddings for batch
    public func forward(batchSize: Int) -> MLXArray {
        // Create batch by concatenating embeddings along new axis
        // [3, 1024, 32, 32] -> [B, 3, 1024, 32, 32]
        var batchedEmbeddings: [MLXArray] = []
        for _ in 0..<batchSize {
            batchedEmbeddings.append(embeddings)
        }
        let repeated = MLX.stacked(batchedEmbeddings, axis: 0)

        // Rearrange to match PyTorch: "B Np Ct Hp Wp -> B Ct (Np Hp Wp)"
        let numPlanes = 3
        let numChannels = 1024
        let planeSize = 32

        return repeated.transposed(axes:[0, 1 ,3,4, 2]).reshaped([batchSize, numPlanes * planeSize * planeSize,numChannels])
    }

    /// Convert tokens back to triplane representation
    public func detokenize(_ tokens: MLXArray) -> MLXArray {
        // Rearrange from "B Ct (Np Hp Wp)" to "B Np Ct Hp Wp"
        let batchSize = tokens.dim(0)
        let numChannels = tokens.dim(2)  // Should be 1024
        let totalTokens = tokens.dim(1)  // Should be 3 * 32 * 32 = 3072

        let numPlanes = 3
        let planeSize = 32
        assert(totalTokens == numPlanes * planeSize * planeSize, "Token count mismatch:\(tokens.shape)")
        assert(numChannels == 1024, "Channel count mismatch")

        return tokens.reshaped([batchSize, numPlanes, planeSize, planeSize,numChannels]).transposed(axes: [0,1,4,2,3])
    }
}
