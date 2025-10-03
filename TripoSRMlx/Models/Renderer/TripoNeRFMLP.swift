//
//  SimpleNeRFDecoder.swift
//  TripoSRMlx
//
//  TripoSR-compliant MLP decoder for NeRF rendering.
//  Matches the official TripoSR architecture: 9 hidden layers of 64 neurons with SiLU activation.
//

import Foundation
import MLX
import MLXNN

/// Apply activation function based on type
func applyActivation(_ x: MLXArray, type: NeRFActivationType) -> MLXArray {
    switch type {
    case .relu:
        return relu(x)
    case .silu:
        return silu(x)
    case .gelu:
        return gelu(x)
    case .tanh:
        return MLX.tanh(x)
    case .none:
        return x
    }
}

/// TripoSR-compliant MLP decoder for NeRF density and color prediction
/// Architecture matches PyTorch exactly: layers.{0,2,4,6,8,10,12,14,16,18,20}
/// PyTorch mapping: decoder.layers.0-16 → hidden layers, layers.18 → density, layers.20 → color
nonisolated public final class TripoNeRFMLP: Module, NeRFDecoder {
    @ModuleInfo(key: "layers")
    private var layers:  [UnaryLayer]

    private let activation: NeRFActivationType

    func callLayers(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers {
            x = layer(x)
        }
        return x
    }

    public init(config: NeRFDecoderConfig) {
        precondition(config.numLayers == 9, "TripoNeRFDecoder currently supports exactly 9 hidden layers")
        self.activation = config.activation
        let inputDim = config.inputDim
        let hiddenDim = config.hiddenDim
        let numHiddenLayers = config.numLayers
        let outputColorDim = config.outputColorDim
        precondition(numHiddenLayers == 9, "DecoderLayerStack expects 9 hidden layers")

        // Create layers array exactly like PyTorch NeRFMLP
        var layerArray: [any UnaryLayer] = []

        // First layer: input -> hidden with activation
        layerArray.append(Linear(inputDim, hiddenDim, bias: config.bias))
        layerArray.append(createActivationModule(activation))

        // Hidden layers: hidden -> hidden with activation
        for _ in 0..<(numHiddenLayers - 1) {
            layerArray.append(Linear(hiddenDim, hiddenDim, bias: config.bias))
            layerArray.append(createActivationModule(activation))
        }

        // Final output layer: hidden -> 4 (density + features)
        layerArray.append(Linear(hiddenDim, 4, bias: config.bias))

        self._layers.wrappedValue = layerArray
    }

    // Backward compatibility constructor
    public convenience init(
        inputDim: Int = 120,  // Official TripoSR: 3 planes × 40 channels
        hiddenDim: Int = 64,  // Official TripoSR: 64 neurons
        numLayers: Int = 9,   // Official TripoSR: 9 hidden layers
        outputColorDim: Int = 3
    ) {
        let config = NeRFDecoderConfig(
            inputDim: inputDim,
            hiddenDim: hiddenDim,
            numLayers: numLayers,
            outputColorDim: outputColorDim,
            activation: .silu
        )
        self.init(config: config)
    }

    public func callAsFunction(_ features: MLXArray) -> NeRFDecoderOutput {
        // Match PyTorch NeRFMLP.forward() behavior exactly
        let inputShape = Array(features.shape.dropLast())
        let flatFeatures = features.reshaped([-1, features.shape.last!])

        let output = callLayers(flatFeatures)

        // Reshape back to original spatial dimensions
        let reshapedOutput = output.reshaped(inputShape + [-1])

        let density = reshapedOutput[.ellipsis, 0..<1]
        let colorFeatures = reshapedOutput[.ellipsis, 1..<4]

        return NeRFDecoderOutput(density: density, features: colorFeatures)
    }
}
private func createActivationModule(_ activation: NeRFActivationType) -> any UnaryLayer {
    switch activation {
    case .relu:
        return ReLU()
    case .silu:
        return SiLU()
    case .gelu:
        return GELU()
    case .tanh:
        return Tanh()
    case .none:
        return Identity()
    }
}

/// Configuration for creating TripoSR-compliant NeRF decoders
/// Activation function types for NeRF decoder
public enum NeRFActivationType {
    case relu
    case silu
    case gelu
    case tanh
    case none
}

/// Weight initialization types following PyTorch NeRFMLP
public enum WeightInitType {
    case kaimingUniform
    case kaimingNormal
    case xavier
}

/// Bias initialization types following PyTorch NeRFMLP
public enum BiasInitType {
    case zero
    case constant(Float)
}

nonisolated public struct NeRFDecoderConfig {
    public let inputDim: Int
    public let hiddenDim: Int
    public let numLayers: Int
    public let outputColorDim: Int
    public let activation: NeRFActivationType
    public let bias: Bool
    public let weightInit: WeightInitType?
    public let biasInit: BiasInitType?

    public init(
        inputDim: Int = 120,  // TripoSR standard: 3 × 40 triplane channels
        hiddenDim: Int = 64,  // TripoSR standard: 64 neurons per layer
        numLayers: Int = 9,   // TripoSR standard: 9 hidden layers
        outputColorDim: Int = 3,
        activation: NeRFActivationType = .relu,  // Python NeRFMLP default: ReLU
        bias: Bool = true,
        weightInit: WeightInitType? = .kaimingUniform,
        biasInit: BiasInitType? = nil
    ) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.outputColorDim = outputColorDim
        self.activation = activation
        self.bias = bias
        self.weightInit = weightInit
        self.biasInit = biasInit
    }

    /// Create TripoSR official configuration
    nonisolated public static var tripoSRConfig: NeRFDecoderConfig {
        return NeRFDecoderConfig(
            inputDim: 120,   // 3 planes × 40 channels
            hiddenDim: 64,   // Narrow but deep architecture
            numLayers: 9,    // Deep network for complex spatial relationships
            outputColorDim: 3,
            activation: .silu
        )
    }
}

/// Factory for creating NeRF decoders
nonisolated public enum NeRFDecoderFactory {
    nonisolated  public static func createDecoder(config: NeRFDecoderConfig) -> TripoNeRFMLP {
        return TripoNeRFMLP(
            inputDim: config.inputDim,
            hiddenDim: config.hiddenDim,
            numLayers: config.numLayers,
            outputColorDim: config.outputColorDim
        )
    }
}
