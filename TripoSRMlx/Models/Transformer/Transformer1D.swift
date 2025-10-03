//
//  Transformer1D.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/20.
//

import Foundation
import MLX
import MLXNN
import MLXRandom
// MARK: - Transformer Backbone

/// Transformer backbone for processing query tokens with cross-attention to image features
nonisolated public final class Transformer1D: Module {
    @ModuleInfo(key: "norm")
    private var norm: GroupNorm
    
    @ModuleInfo(key: "proj_in")
    private var projIn: Linear
    
    @ModuleInfo(key: "proj_out")
    private var projOut: Linear
    
    @ModuleInfo(key: "transformer_blocks")
    private var transformerBlocks: [BasicTransformerBlock]
    
    public init(config: BackboneConfig) {
        let innerDim = config.numHeads * config.attentionHeadDim

        self._norm.wrappedValue = GroupNorm(groupCount: config.normNumGroups, dimensions: config.inChannels,affine: true,pytorchCompatible:true)
        self._projIn.wrappedValue = Linear(config.inChannels, innerDim)
        self._projOut.wrappedValue = Linear(innerDim, config.inChannels)

        var blocks: [BasicTransformerBlock] = []
        for _ in 0..<config.numLayers {
            blocks.append(BasicTransformerBlock(
                dim: innerDim,
                numAttentionHeads: config.numHeads,
                attentionHeadDim: config.attentionHeadDim,
                crossAttentionDim: config.crossAttentionDim,
                activationFn: .geglu,
                attentionBias: false,
                onlyCrossAttention: false,
                doubleSelfAttention: false,
                normElementwiseAffine: true
            ))
        }
        self._transformerBlocks.wrappedValue = blocks
    }
    
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        // 1. Input processing (following PyTorch implementation)
        let batch = hiddenStates.dim(0)
        let channels = hiddenStates.dim(2)
        let seqLen = hiddenStates.dim(1)

        let residual = hiddenStates

        // Apply GroupNorm
        var normalized = norm(hiddenStates)

        // Reshape from [batch, channels, seq_len] to [batch, seq_len, channels] for linear layer
//        normalized = normalized.transposed(axes: [0, 2, 1]).reshaped([batch, seqLen, channels])

        // Project input
        var processed = projIn(normalized)

        // 2. Apply transformer blocks
        for block in transformerBlocks {
            processed = block(
                processed,
                attentionMask: attentionMask, encoderHiddenStates: encoderHiddenStates,
                encoderAttentionMask: encoderAttentionMask
            )
        }

        // 3. Output projection
        processed = projOut(processed)

        // Reshape back to [batch, channels, seq_len]
        processed = processed.reshaped([batch, seqLen,channels])//.transposed(axes: [0, 2, 1])

        // Add residual connection
        let output = processed + residual

        return output
    }
}

/// Multi-head attention module compatible with PyTorch's Attention class
nonisolated public final class Attention: Module {
    private let queryDim: Int
    private let crossAttentionDim: Int
    private let heads: Int
    private let dimHead: Int
    private let dropout: Float
    private let bias: Bool
    private let onlyCrossAttention: Bool
    private let outBias: Bool
    private let scaleQK: Bool
    private let scale: Float
    private let innerDim: Int
    private let outDim: Int

    @ModuleInfo(key: "to_q")
    private var toQ: Linear

    @ModuleInfo(key: "to_k")
    private var toK: Linear?

    @ModuleInfo(key: "to_v")
    private var toV: Linear?

    @ModuleInfo(key: "to_out")
    private var toOut: [any UnaryLayer]

    private let dropoutLayer: Dropout?

    public init(
        queryDim: Int,
        crossAttentionDim: Int? = nil,
        heads: Int = 8,
        dimHead: Int = 64,
        dropout: Float = 0.0,
        bias: Bool = false,
        onlyCrossAttention: Bool = false,
        outBias: Bool = true,
        scaleQK: Bool = true,
        outDim: Int? = nil
    ) {
        self.queryDim = queryDim
        self.crossAttentionDim = crossAttentionDim ?? queryDim
        self.dimHead = dimHead
        self.dropout = dropout
        self.bias = bias
        self.onlyCrossAttention = onlyCrossAttention
        self.outBias = outBias
        self.scaleQK = scaleQK

        self.innerDim = outDim ?? dimHead * heads
        self.outDim = outDim ?? queryDim
        self.heads = outDim != nil ? outDim! / dimHead : heads

        self.scale = scaleQK ? Float(pow(Float(dimHead), -0.5)) : 1.0

        // Initialize projections
        self._toQ.wrappedValue = Linear(queryDim, self.innerDim, bias: bias)

        if !onlyCrossAttention {
            self._toK.wrappedValue = Linear(self.crossAttentionDim, self.innerDim, bias: bias)
            self._toV.wrappedValue = Linear(self.crossAttentionDim, self.innerDim, bias: bias)
        } else {
            self._toK.wrappedValue = nil
            self._toV.wrappedValue = nil
        }

        // Output projection and dropout
        var outLayers: [any UnaryLayer] = []
        outLayers.append(Linear(self.innerDim, self.outDim, bias: outBias))
        if dropout > 0 {
            let dropoutModule = Dropout(p: dropout)
            outLayers.append(dropoutModule)
            self.dropoutLayer = dropoutModule
        } else {
            self.dropoutLayer = nil
        }
        self._toOut.wrappedValue = outLayers
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        // Prepare context for cross-attention
        let context = onlyCrossAttention ? encoderHiddenStates : (encoderHiddenStates ?? hiddenStates)
        guard let actualContext = context else {
            fatalError("Context is required for cross-attention")
        }

        // Compute query, key, value projections
        let query = headToBatchDim(toQ(hiddenStates))

        guard let toK = toK, let toV = toV else {
            fatalError("Key and Value projections must be available for attention")
        }

        let key = headToBatchDim(toK(actualContext))
        let value = headToBatchDim(toV(actualContext))

        // Compute attention scores
        let attentionScores = getAttentionScores(query: query, key: key, attentionMask: attentionMask)

        // Apply attention to values
        var hiddenStatesOutput = MLX.matmul(attentionScores, value)
        hiddenStatesOutput = batchToHeadDim(hiddenStatesOutput)

        // Apply output projection
        for layer in toOut {
            if let linear = layer as? Linear {
                hiddenStatesOutput = linear(hiddenStatesOutput)
            } else if let dropout = layer as? Dropout {
                hiddenStatesOutput = dropout(hiddenStatesOutput)
            }
        }

        return hiddenStatesOutput
    }

    private func headToBatchDim(_ tensor: MLXArray, outDim: Int = 3) -> MLXArray {
        let batchSize = tensor.dim(0)
        let seqLen = tensor.dim(1)
        let dim = tensor.dim(2)

        let reshaped = tensor.reshaped([batchSize, seqLen, heads, dim / heads])
        let permuted = reshaped.transposed(axes: [0, 2, 1, 3])

        if outDim == 3 {
            return permuted.reshaped([batchSize * heads, seqLen, dim / heads])
        }
        return permuted
    }

    private func batchToHeadDim(_ tensor: MLXArray) -> MLXArray {
        let batchHeads = tensor.dim(0)
        let seqLen = tensor.dim(1)
        let dim = tensor.dim(2)

        let batchSize = batchHeads / heads
        let reshaped = tensor.reshaped([batchSize, heads, seqLen, dim])
        let permuted = reshaped.transposed(axes: [0, 2, 1, 3])

        return permuted.reshaped([batchSize, seqLen, dim * heads])
    }

    private func getAttentionScores(query: MLXArray, key: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        // Compute attention scores: query @ key.T * scale
        var attentionScores = MLX.matmul(query, key.transposed(axes: [0, 2, 1]))
        attentionScores = attentionScores * scale

        // Apply attention mask if provided
        if let mask = attentionMask {
            // Prepare attention mask to match dimensions
            let preparedMask = prepareAttentionMask(mask, targetLength: attentionScores.dim(2), batchSize: query.dim(0) / heads)
            attentionScores = attentionScores + preparedMask
        }

        // Apply softmax
        let attentionProbs = MLX.softmax(attentionScores, axis: -1)

        return attentionProbs
    }

    private func prepareAttentionMask(_ attentionMask: MLXArray, targetLength: Int, batchSize: Int, outDim: Int = 3) -> MLXArray {
        var mask = attentionMask
        let currentLength = mask.dim(-1)

        // Pad mask if needed
        if currentLength != targetLength {
            let padWidth = targetLength - currentLength
            // MLX padding equivalent to F.pad(attention_mask, (0, target_length), value=0.0)
            let padShape = Array(mask.shape.dropLast()) + [padWidth]
            let padding = MLX.zeros(padShape, dtype: mask.dtype)
            mask = MLX.concatenated([mask, padding], axis: -1)
        }

        // Expand mask for multi-head attention
        if outDim == 3 {
            if mask.dim(0) < batchSize * heads {
                // Repeat for each head
                mask = MLX.broadcast(mask, to: [batchSize * heads] + Array(mask.shape.dropFirst()))
            }
        }

        return mask
    }
}

// Type alias for backward compatibility
typealias SelfAttention = Attention

