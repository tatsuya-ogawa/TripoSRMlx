//
//  ViTComponents.swift
//  TripoSRMlx
//
//  Vision Transformer (ViT) components for TripoSR system.
//  Consolidated ViT implementation with all related classes.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Fake Dropout for inference mode (does nothing, preserves input)
/// In training mode, this would randomly zero out elements
private func fakeDropout(_ x: MLXArray, rate: Float = 0.1) -> MLXArray {
    // During inference, dropout is disabled, so we just return the input
    return x
}

// MARK: - Vision Transformer Model

/// Vision Transformer Model matching PyTorch ViT structure
nonisolated public final class ViTModel: Module {
    @ModuleInfo(key: "embeddings")
    private var embeddings: ViTEmbeddings

    @ModuleInfo(key: "encoder")
    private var encoder: ViTEncoder

    @ModuleInfo(key: "layernorm")
    private var layernorm: LayerNorm

    @ModuleInfo(key: "pooler")
    private var pooler: ViTPooler

    public init(hiddenSize: Int = 768, numLayers: Int = 12, numAttentionHeads: Int = 12,
                intermediateSize: Int = 3072, imageSize: Int, patchSize: Int = 16,
                poolerActivation: ViTPooler.ActivationType = .tanh) {
        self._embeddings.wrappedValue = ViTEmbeddings(
            hiddenSize: hiddenSize,
            imageSize: imageSize,
            patchSize: patchSize
        )

        self._encoder.wrappedValue = ViTEncoder(
            hiddenSize: hiddenSize,
            numLayers: numLayers,
            numAttentionHeads: numAttentionHeads,
            intermediateSize: intermediateSize
        )

        self._layernorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._pooler.wrappedValue = ViTPooler(hiddenSize: hiddenSize, activation: poolerActivation)
    }

    public func callAsFunction(_ pixelValues: MLXArray,interpolatePosEncoding:Bool) -> ViTModelOutput {
        let embeddingOutput = embeddings(pixelValues,interpolatePosEncoding:interpolatePosEncoding)
        var encoderOutput = encoder(embeddingOutput)

        encoderOutput = layernorm(encoderOutput)
        let pooledOutput = pooler(encoderOutput)

        return ViTModelOutput(
            lastHiddenState: encoderOutput,
            poolerOutput: pooledOutput,
            embeddingOutput: embeddingOutput
        )
    }
}

// MARK: - ViT Embeddings

/// ViT Embeddings (patch + position + cls token)
/// Construct the CLS token, position and patch embeddings.
nonisolated public final class ViTEmbeddings: Module {
    @ModuleInfo(key: "cls_token")
    private var clsToken: MLXArray

    @ModuleInfo(key: "position_embeddings")
    private var positionEmbeddings: MLXArray

    @ModuleInfo(key: "patch_embeddings")
    private var patchEmbeddings: ViTPatchEmbeddings

    private let hiddenSize: Int
    private let imageSize: Int
    private let patchSize: Int
    private let hiddenDropoutProb: Float

    public init(hiddenSize: Int = 768, imageSize: Int, patchSize: Int = 16, hiddenDropoutProb: Float = 0.0) {
        self.hiddenSize = hiddenSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.hiddenDropoutProb = hiddenDropoutProb

        let numPatches = (imageSize / patchSize) * (imageSize / patchSize)

        self._clsToken.wrappedValue = MLX.zeros([1, 1, hiddenSize])
        self._positionEmbeddings.wrappedValue = MLX.zeros([1, numPatches + 1, hiddenSize])

        self._patchEmbeddings.wrappedValue = ViTPatchEmbeddings(
            imageSize: imageSize,
            patchSize: patchSize,
            numChannels: 3,
            embedDim: hiddenSize
        )
    }

    /// This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    /// resolution images.
    ///
    /// Source:
    /// https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    private func interpolatePosEncoding(_ embeddings: MLXArray, height: Int, width: Int) -> MLXArray {
        let batchSize = embeddings.dim(0)
        let npatch = embeddings.dim(1) - 1
        let N = positionEmbeddings.dim(1) - 1

        if npatch == N && height == width {
            return positionEmbeddings
        }

        let classPosEmbed = positionEmbeddings[0..., 0..<1]
        let patchPosEmbed = positionEmbeddings[0..., 1...]
        let dim = embeddings.dim(-1)

        let h0 = Float(height / patchSize)
        let w0 = Float(width / patchSize)

        // we add a small number to avoid floating point error in the interpolation
        // see discussion at https://github.com/facebookresearch/dino/issues/8
        let h0Adjusted = h0 + 0.1
        let w0Adjusted = w0 + 0.1

        let sqrtN = Int(sqrt(Float(N)))
        let reshaped = patchPosEmbed.reshaped([-1, sqrtN, sqrtN, dim])//.transposed(axes: [0, 3, 1, 2])

        let scaleFactorH = h0Adjusted / sqrt(Float(N))
        let scaleFactorW = w0Adjusted / sqrt(Float(N))

        // Note: MLX doesn't have bicubic interpolation, using linear as approximation
        // In a full implementation, you might need to implement bicubic interpolation
        let newHeight = Int(h0)
        let newWidth = Int(w0)

        // Simple resize operation (placeholder for bicubic interpolation)
        let interpolated = resizeArray(reshaped, newHeight: newHeight, newWidth: newWidth)
        let finalPatchPosEmbed = interpolated.reshaped([batchSize, -1, dim])

        return MLX.concatenated([classPosEmbed, finalPatchPosEmbed], axis: 1)
    }

    // Helper function for array resizing (simplified version)
    private func resizeArray(_ array: MLXArray, newHeight: Int, newWidth: Int) -> MLXArray {
        // This is a simplified implementation. In practice, you'd implement proper bicubic interpolation
        let oldHeight = array.dim(1)
        let oldWidth = array.dim(2)
        let heightScale = Float(newHeight) / Float(oldHeight)
        let widthScale = Float(newWidth) / Float(oldWidth)

        let upsample = MLXNN.Upsample(
            scaleFactor: [heightScale, widthScale],
            mode: .cubic(alignCorners: false)
        )
        return upsample(array)
    }

    public func callAsFunction(_ pixelValues: MLXArray, interpolatePosEncoding: Bool = false) -> MLXArray {
        let batchSize = pixelValues.dim(0)
        let numChannels = pixelValues.dim(3)
        let height = pixelValues.dim(1)
        let width = pixelValues.dim(2)

        let embeddings = patchEmbeddings(pixelValues)

        // add the [CLS] token to the embedded patch tokens
        let clsTokens = MLX.broadcast(clsToken, to: [batchSize, 1, clsToken.dim(-1)])
        let embeddingsWithCls = MLX.concatenated([clsTokens, embeddings], axis: 1)

        // add positional encoding to each token
        let withPositions: MLXArray
        if interpolatePosEncoding {
            withPositions = embeddingsWithCls + self.interpolatePosEncoding(embeddingsWithCls, height: height, width: width)
        } else {
            withPositions = embeddingsWithCls + positionEmbeddings
        }

        return fakeDropout(withPositions, rate: hiddenDropoutProb)
    }
}

/// ViT Patch Embeddings (Convolutional projection)
/// Image to Patch Embedding.
///
/// Based on timm implementation, which can be found here:
/// https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
nonisolated public final class ViTPatchEmbeddings: Module {
    @ModuleInfo(key: "projection")
    private var projection: Conv2d

    private let imageSize: (Int, Int)
    private let patchSize: (Int, Int)
    private let numPatches: Int

    public init(imageSize: Int, patchSize: Int = 16, numChannels: Int = 3, embedDim: Int = 768) {
        self.imageSize = (imageSize, imageSize)
        self.patchSize = (patchSize, patchSize)
        self.numPatches = (imageSize / patchSize) * (imageSize / patchSize)

        self._projection.wrappedValue = Conv2d(
            inputChannels: numChannels,
            outputChannels: embedDim,
            kernelSize: [patchSize, patchSize],
            stride: [patchSize, patchSize]
        )
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let x = projection(pixelValues).transposed(axes: [0, 3, 1, 2]).flattened(start: 2).transposed(axes:[0, 2, 1])
        return x
    }
}

// MARK: - ViT Encoder

/// ViT Encoder (stack of transformer layers)
nonisolated public final class ViTEncoder: Module {
    @ModuleInfo(key: "layer")
    private var layers: [ViTLayer]

    public init(hiddenSize: Int = 768, numLayers: Int = 12, numAttentionHeads: Int = 12, intermediateSize: Int = 3072) {
        var layerArray: [ViTLayer] = []
        for _ in 0..<numLayers {
            layerArray.append(ViTLayer(
                hiddenSize: hiddenSize,
                numAttentionHeads: numAttentionHeads,
                intermediateSize: intermediateSize
            ))
        }
        self._layers.wrappedValue = layerArray
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var x = hiddenStates
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

/// Single ViT Transformer Layer
nonisolated public final class ViTLayer: Module {
    @ModuleInfo(key: "attention")
    private var attention: ViTAttention

    @ModuleInfo(key: "intermediate")
    private var intermediate: ViTIntermediate

    @ModuleInfo(key: "output")
    private var output: ViTLayerOutput

    @ModuleInfo(key: "layernorm_before")
    private var layerNormBefore: LayerNorm

    @ModuleInfo(key: "layernorm_after")
    private var layerNormAfter: LayerNorm

    public init(hiddenSize: Int = 768, numAttentionHeads: Int = 12, intermediateSize: Int = 3072) {
        self._attention.wrappedValue = ViTAttention(hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads)
        self._intermediate.wrappedValue = ViTIntermediate(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._output.wrappedValue = ViTLayerOutput(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._layerNormBefore.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._layerNormAfter.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let normalizedHidden = layerNormBefore(hiddenStates)
        let attentionOutput = attention(normalizedHidden)
        let hidden1 = hiddenStates + attentionOutput

        let normalizedHidden2 = layerNormAfter(hidden1)
        let intermediateOutput = intermediate(normalizedHidden2)
        let finalOutput = output(intermediateOutput, hidden1)

        return finalOutput
    }
}

// MARK: - ViT Attention

/// ViT Attention mechanism
nonisolated public final class ViTAttention: Module {
    @ModuleInfo(key: "attention")
    private var selfAttention: ViTSelfAttention

    @ModuleInfo(key: "output")
    private var outputProjection: ViTSelfOutput

    public init(hiddenSize: Int = 768, numAttentionHeads: Int = 12) {
        self._selfAttention.wrappedValue = ViTSelfAttention(hiddenSize: hiddenSize, numAttentionHeads: numAttentionHeads)
        self._outputProjection.wrappedValue = ViTSelfOutput(hiddenSize: hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let selfOutput = selfAttention(hiddenStates)
        let attentionOutput = outputProjection(selfOutput, hiddenStates)
        return attentionOutput
    }
}

/// ViT Self-Attention
nonisolated public final class ViTSelfAttention: Module {
    @ModuleInfo(key: "query")
    private var query: Linear

    @ModuleInfo(key: "key")
    private var key: Linear

    @ModuleInfo(key: "value")
    private var value: Linear

    private let numAttentionHeads: Int
    private let attentionHeadSize: Int

    public init(hiddenSize: Int = 768, numAttentionHeads: Int = 12) {
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadSize = hiddenSize / numAttentionHeads

        self._query.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._key.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._value.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLength = hiddenStates.dim(1)

        let queryLayer = transposeForScores(query(hiddenStates), batchSize: batchSize, seqLength: seqLength)
        let keyLayer = transposeForScores(key(hiddenStates), batchSize: batchSize, seqLength: seqLength)
        let valueLayer = transposeForScores(value(hiddenStates), batchSize: batchSize, seqLength: seqLength)

        var attentionScores = MLX.matmul(queryLayer, keyLayer.transposed(axes: [0, 1, 3, 2]))
        attentionScores = attentionScores / sqrt(Float(attentionHeadSize))
        let attentionProbs = MLX.softmax(attentionScores, axis: -1)

        let droppedAttentionProbs = fakeDropout(attentionProbs, rate: 0.0)

        var contextLayer = MLX.matmul(droppedAttentionProbs, valueLayer)
        contextLayer = contextLayer.transposed(axes: [0, 2, 1, 3])
        let newContextShape = [batchSize, seqLength, numAttentionHeads * attentionHeadSize]

        return contextLayer.reshaped(newContextShape)
    }

    private func transposeForScores(_ x: MLXArray, batchSize: Int, seqLength: Int) -> MLXArray {
        let newShape = [batchSize, seqLength, numAttentionHeads, attentionHeadSize]
        return x.reshaped(newShape).transposed(axes: [0, 2, 1, 3])
    }
}

/// ViT Self-Output (projection + residual connection)
nonisolated public final class ViTSelfOutput: Module {
    @ModuleInfo(key: "dense")
    private var dense: Linear

    public init(hiddenSize: Int = 768) {
        self._dense.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray, _ inputTensor: MLXArray) -> MLXArray {
        let processedStates = dense(hiddenStates)
        let droppedStates = fakeDropout(processedStates, rate: 0.0)
        return droppedStates
    }
}

// MARK: - ViT Feed Forward

/// ViT Intermediate (feed-forward network)
nonisolated public final class ViTIntermediate: Module {
    @ModuleInfo(key: "dense")
    private var dense: Linear

    public init(hiddenSize: Int = 768, intermediateSize: Int = 3072) {
        self._dense.wrappedValue = Linear(hiddenSize, intermediateSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        return gelu(dense(hiddenStates))
    }
}

/// ViT Output (final projection + residual)
nonisolated public final class ViTLayerOutput: Module {
    @ModuleInfo(key: "dense")
    private var dense: Linear

    public init(hiddenSize: Int = 768, intermediateSize: Int = 3072) {
        self._dense.wrappedValue = Linear(intermediateSize, hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray, _ inputTensor: MLXArray) -> MLXArray {
        let processedStates = dense(hiddenStates)
        let droppedStates = fakeDropout(processedStates, rate: 0.0)
        return droppedStates + inputTensor
    }
}

// MARK: - ViT Pooler

/// ViT Pooler layer matching PyTorch implementation
nonisolated public final class ViTPooler: Module {
    @ModuleInfo(key: "dense")
    private var dense: Linear

    private let activationType: ActivationType

    public enum ActivationType {
        case tanh
        case none
        case gelu
        case relu
        case silu
        case exp
    }

    public init(hiddenSize: Int = 768, activation: ActivationType = .tanh) {
        self._dense.wrappedValue = Linear(hiddenSize, hiddenSize)
        self.activationType = activation
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let firstTokenTensor = hiddenStates[0..., 0]
        let pooledOutput = dense(firstTokenTensor)

        switch activationType {
        case .tanh:
            return MLX.tanh(pooledOutput)
        case .none:
            return pooledOutput
        case .gelu:
            return gelu(pooledOutput)
        case .relu:
            return relu(pooledOutput)
        case .silu:
            return silu(pooledOutput)
        case .exp:
            return MLX.exp(pooledOutput)
        }
    }
}

// MARK: - Output Structures

/// Output structure for ViT model
public struct ViTModelOutput {
    public let lastHiddenState: MLXArray
    public let poolerOutput: MLXArray
    public let embeddingOutput: MLXArray
}
