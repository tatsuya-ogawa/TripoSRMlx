//
//  BasicTransformerBlock.swift
//  TripoSRMlx
//
//  Swift/MLX port of the TripoSR transformer building blocks.
//

import Foundation
import MLX
import MLXNN
nonisolated final class Identity: Module,UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x
    }
}
/// Wrapper that emulates PyTorch's feed-forward ModuleList structure (indices 0 and 2)
nonisolated final class TransformerFeedForward: Module {
    @ModuleInfo(key: "net")
    private var layers: [any UnaryLayer]

    public init(hiddenSize: Int, mult: Int,
                dropout: Float = 0.0,activationFn:FeedForwardActivationType = .geglu,finalDropOut:Bool=false) {
        let innerDim =  hiddenSize * mult
        var activation:any UnaryLayer = switch activationFn {
        case .gelu:
            GELUProjection(dimIn: hiddenSize, dimOut: innerDim, approximation: .none)
        case .geluApproximate:
            GELUProjection(dimIn: hiddenSize, dimOut: innerDim, approximation: .tanh)
        case .geglu:
            GEGLUProjection(dimIn: hiddenSize, dimOut: innerDim)
        case .gegluApproximate:
            ApproximateGELUProjection(dimIn: hiddenSize, dimOut: innerDim)
        }
        self._layers.wrappedValue = [
            activation,
            Dropout(p: dropout),
            Linear(innerDim, hiddenSize),
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x)
        }
        return x
    }
}

/// First feed-forward projection layer exposing `net.0.proj.*` keys
public enum FeedForwardActivationType: String {
    case gelu = "gelu"
    case geluApproximate = "gelu-approximate"
    case geglu = "geglu"
    case gegluApproximate = "geglu-approximate"

    static func from(_ name: String) -> FeedForwardActivationType {
        FeedForwardActivationType(rawValue: name) ?? .geglu
    }
}

protocol FeedForwardActivation: Module, UnaryLayer {
    func apply(on x: MLXArray) -> MLXArray
}

extension FeedForwardActivation {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        apply(on: x)
    }
}

nonisolated final class GELUProjection: Module, FeedForwardActivation {
    enum Approximation {
        case none
        case tanh
    }

    @ModuleInfo(key: "proj")
    private var projection: Linear

    private let approximation: Approximation

    init(dimIn: Int, dimOut: Int, approximation: Approximation) {
        self._projection.wrappedValue = Linear(dimIn, dimOut)
        self.approximation = approximation
    }

    func apply(on x: MLXArray) -> MLXArray {
        var hidden = projection(x)
        switch approximation {
        case .none:
            hidden = gelu(hidden)
        case .tanh:
            hidden = geluApproximate(hidden)
        }
        return hidden
    }
}

nonisolated final class GEGLUProjection: Module, FeedForwardActivation {
    @ModuleInfo(key: "proj")
    private var projection: Linear

    private let outputDim: Int

    init(dimIn: Int, dimOut: Int) {
        self._projection.wrappedValue = Linear(dimIn, dimOut * 2)
        self.outputDim = dimOut
    }

    func apply(on x: MLXArray) -> MLXArray {
        let projected = projection(x)
        let parts = split(projected, indices: [outputDim], axis: -1)
        let hidden = parts[0]
        let gate = parts[1]
        return hidden * gelu(gate)
    }
}

nonisolated final class ApproximateGELUProjection: Module, FeedForwardActivation {
    @ModuleInfo(key: "proj")
    private var projection: Linear

    init(dimIn: Int, dimOut: Int) {
        self._projection.wrappedValue = Linear(dimIn, dimOut)
    }

    func apply(on x: MLXArray) -> MLXArray {
        let hidden = projection(x)
        return hidden * sigmoid(1.702 * hidden)
    }
}

nonisolated public final class FeedForward: Module {
    @ModuleInfo(key: "activation")
    private var activation: FeedForwardActivation

    private let dropoutLayer: Dropout?

    @ModuleInfo(key: "proj_out")
    private var projectOut: Linear

    private let finalDropoutLayer: Dropout?

    public init(
        dim: Int,
        dimOut: Int? = nil,
        mult: Float = 4.0,
        dropout: Float = 0.0,
        activationFn: FeedForwardActivationType = .geglu,
        finalDropout: Bool = false
    ) {
        let innerDim = Int(Float(dim) * mult)
        switch activationFn {
        case .gelu:
            self._activation.wrappedValue = GELUProjection(dimIn: dim, dimOut: innerDim, approximation: .none)
        case .geluApproximate:
            self._activation.wrappedValue = GELUProjection(dimIn: dim, dimOut: innerDim, approximation: .tanh)
        case .geglu:
            self._activation.wrappedValue = GEGLUProjection(dimIn: dim, dimOut: innerDim)
        case .gegluApproximate:
            self._activation.wrappedValue = ApproximateGELUProjection(dimIn: dim, dimOut: innerDim)
        }

        if dropout > 0 {
            self.dropoutLayer = Dropout(p: dropout)
        } else {
            self.dropoutLayer = nil
        }

        let outputDim = dimOut ?? dim
        self._projectOut.wrappedValue = Linear(innerDim, outputDim)

        if finalDropout && dropout > 0 {
            self.finalDropoutLayer = Dropout(p: dropout)
        } else {
            self.finalDropoutLayer = nil
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hidden = activation.apply(on: hiddenStates)
        if let dropoutLayer {
            hidden = dropoutLayer(hidden)
        }
        hidden = projectOut(hidden)
        if let finalDropoutLayer {
            hidden = finalDropoutLayer(hidden)
        }
        return hidden
    }
}

nonisolated public final class BasicTransformerBlock: Module {
    public let onlyCrossAttention: Bool

    private let doubleSelfAttention: Bool

    private let norm1: LayerNorm
    private let norm2: LayerNorm?
    private let norm3: LayerNorm

    @ModuleInfo(key: "attn1")
    private var attn1: SelfAttention

    @ModuleInfo(key: "attn2")
    private var attn2: SelfAttention?

    @ModuleInfo(key: "ff")
    private var feedForward: TransformerFeedForward

    private var chunkFeedForwardSize: Int?
    private var chunkFeedForwardDim: Int = 0

    public init(
        dim: Int,
        numAttentionHeads: Int,
        attentionHeadDim: Int,
        dropout: Float = 0.0,
        crossAttentionDim: Int? = nil,
        activationFn: FeedForwardActivationType = .geglu,
        attentionBias: Bool = false,
        onlyCrossAttention: Bool = false,
        doubleSelfAttention: Bool = false,
        normElementwiseAffine: Bool = true,
        finalDropout: Bool = false
    ) {
        self.onlyCrossAttention = onlyCrossAttention
        self.doubleSelfAttention = doubleSelfAttention

        self.norm1 = LayerNorm(
            dimensions: dim,
            eps: 1e-5,
            affine: normElementwiseAffine
        )

        self._attn1.wrappedValue = SelfAttention(
            queryDim: dim,
            crossAttentionDim: onlyCrossAttention ? crossAttentionDim : nil,
            heads: numAttentionHeads,
            dimHead: attentionHeadDim,
            dropout: dropout,
            bias: attentionBias,
            onlyCrossAttention: onlyCrossAttention
        )

        if crossAttentionDim != nil || doubleSelfAttention {
            self.norm2 = LayerNorm(
                dimensions: dim,
                eps: 1e-5,
                affine: normElementwiseAffine
            )
            self._attn2.wrappedValue = SelfAttention(
                queryDim: dim,
                crossAttentionDim: doubleSelfAttention ? nil : crossAttentionDim,
                heads: numAttentionHeads,
                dimHead: attentionHeadDim,
                dropout: dropout,
                bias: attentionBias
            )
        } else {
            self.norm2 = nil
            self._attn2.wrappedValue = nil
        }

        self.norm3 = LayerNorm(
            dimensions: dim,
            eps: 1e-5,
            affine: normElementwiseAffine
        )

        self._feedForward.wrappedValue = TransformerFeedForward(
            hiddenSize: dim,
            mult: 4,
            dropout: dropout,
            activationFn: activationFn,
            finalDropOut: finalDropout
        )
    }

    public func setChunkFeedForward(chunkSize: Int?, dim: Int) {
        self.chunkFeedForwardSize = chunkSize
        self.chunkFeedForwardDim = dim
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        encoderHiddenStates: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        var hiddenStates = hiddenStates

        // 1. Self-attention
        var normHiddenStates = norm1(hiddenStates)
        let selfAttentionEncoder = onlyCrossAttention ? encoderHiddenStates : nil
        var attnOutput = attn1(
            normHiddenStates,
            encoderHiddenStates: selfAttentionEncoder, attentionMask: attentionMask
        )
        hiddenStates = hiddenStates + attnOutput

        // 2. Cross- or self-attention
        if let attn2, let norm2 {
            normHiddenStates = norm2(hiddenStates)
            attnOutput = attn2(
                normHiddenStates,
                encoderHiddenStates: encoderHiddenStates, attentionMask: encoderAttentionMask
            )
            hiddenStates = hiddenStates + attnOutput
        }

        // 3. Feed-forward
        normHiddenStates = norm3(hiddenStates)

        if let chunkSize = chunkFeedForwardSize, chunkSize > 0 {
            // Chunking is an optional optimisation in the PyTorch implementation.
            // MLX does not provide a direct equivalent yet, so we simply fall back
            // to processing the entire tensor when chunking is requested.
        }

        let ffOutput = feedForward(normHiddenStates)
        hiddenStates = hiddenStates + ffOutput

        return hiddenStates
    }
}
