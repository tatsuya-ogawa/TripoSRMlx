//
//  ImageTokenizer.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/20.
//

// MARK: - Image Tokenizer
import Foundation
import MLX
import MLXNN
import MLXRandom
/// DINO ViT-B/16 based image tokenizer matching PyTorch implementation
nonisolated public final class DINOSingleImageTokenizer: Module {
    private let config: ImageTokenizerConfig

    // ViT Model components to match PyTorch structure
    @ModuleInfo(key: "model")
    private var model: ViTModel

    public init(config: ImageTokenizerConfig) {
        self.config = config

        // Initialize ViT model with DINO configuration
        self._model.wrappedValue = ViTModel(
            hiddenSize: 768,
            numLayers: 12,
            numAttentionHeads: 12,
            intermediateSize: 3072,
            imageSize: 224,
            patchSize: 16,
            poolerActivation: config.poolerActivation ?? .tanh
        )
    }

    public func callAsFunction(_ images: MLXArray) -> ViTModelOutput {
        // Normalize images like DINO preprocessing
        let normalizedImages = normalizeImages(images)

        // Forward through ViT model
        let output = model(normalizedImages,interpolatePosEncoding: true)

        // Return local features (last hidden state)
        return output
    }

    private func normalizeImages(_ images: MLXArray) -> MLXArray {
        // DINO normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        let mean = MLXArray([0.485, 0.456, 0.406]  as [Float] ).reshaped([1, 1, 1, 3])
        let std = MLXArray([0.229, 0.224, 0.225]  as [Float]).reshaped([1, 1, 1, 3])
        return (images - mean) / std
    }
}
