//
//  TriplaneNeRFRenderer.swift
//  TripoSRMlx
//
//  Official TripoSR-compliant TriplaneNeRFRenderer implementation.
//  Matches the exact TripoSR triplane sampling, ray intersection, and volume rendering.
//

import Foundation
import MLX
import MLXNN

/// Configuration for the TriplaneNeRFRenderer
nonisolated public struct TriplaneNeRFRendererConfig {
    public let radius: Float
    public let featureReduction: String
    public let densityActivation: String
    public let densityBias: Float
    public let colorActivation: String
    public let numSamplesPerRay: Int
    public let randomized: Bool

    public init(
        radius: Float,
        featureReduction: String = "concat",   // TripoSR default
        densityActivation: String = "trunc_exp", // TripoSR default
        densityBias: Float = -1.0,              // TripoSR default
        colorActivation: String = "sigmoid",    // TripoSR default
        numSamplesPerRay: Int = 128,            // TripoSR default
        randomized: Bool = false                // TripoSR default
    ) {
        assert(featureReduction == "concat" || featureReduction == "mean",
               "featureReduction must be 'concat' or 'mean'")

        self.radius = radius
        self.featureReduction = featureReduction
        self.densityActivation = densityActivation
        self.densityBias = densityBias
        self.colorActivation = colorActivation
        self.numSamplesPerRay = numSamplesPerRay
        self.randomized = randomized
    }
}

nonisolated public struct NeRFDecoderOutput{
    let density: MLXArray
    let features: MLXArray
}
/// Protocol for decoder modules used with the NeRF renderer
nonisolated public protocol NeRFDecoder: Module {
    func callAsFunction(_ features: MLXArray) -> NeRFDecoderOutput
}

/// NeRF renderer output
nonisolated public struct NeRFRendererOutput {
    public let density: MLXArray
    public let densityAct: MLXArray
    public let features: MLXArray
    public let color: MLXArray
}

/// Triplane-based NeRF renderer implementation
nonisolated public final class TriplaneNeRFRenderer: Module {

    public let config: TriplaneNeRFRendererConfig
    private var chunkSize: Int = 0
    private var isRandomized: Bool = false

    public init(config: TriplaneNeRFRendererConfig) {
        self.config = config
        self.isRandomized = config.randomized
    }

    /// Set the chunk size for memory-efficient processing
    public func setChunkSize(_ chunkSize: Int) {
        assert(chunkSize >= 0, "chunk_size must be a non-negative integer (0 for no chunking).")
        self.chunkSize = chunkSize
    }

    /// Query triplane representation at given 3D positions
    public func queryTriplane(
        decoder: NeRFDecoder,
        positions: MLXArray,
        triplane: MLXArray
    ) -> NeRFRendererOutput {
        let inputShape = Array(positions.shape.dropLast())
        let positionsFlat = positions.reshaped([-1, 3])

        // Scale positions from (-radius, radius) to (-1, 1) for grid sampling
        let normalizedPositions = scaleTensor(
            positionsFlat,
            from: (-config.radius, config.radius),
            to: (-1.0, 1.0)
        )

        func queryChunk(_ x: MLXArray) -> NeRFDecoderOutput {
            // Following Python implementation: nerf_renderer.py:56-74
            // Create 2D indices for triplane sampling: torch.stack((x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]), dim=-3)
            let indices2D = stacked([
                x[.ellipsis, MLXArray([0, 1]) as MLXArrayIndex],  // XY plane: [0, 1]
                x[.ellipsis, MLXArray([0, 2]) as MLXArrayIndex],  // XZ plane: [0, 2]
                x[.ellipsis, MLXArray([1, 2]) as MLXArrayIndex]   // YZ plane: [1, 2]
            ], axis: -3)

            // Rearrange triplane: "Np Cp Hp Wp -> Np Cp Hp Wp" (already in correct format [3, C, H, W])
            // Rearrange indices2D: "Np N Nd -> Np () N Nd"
            let reshapedIndices = indices2D.expandedDimensions(axis: 1)  // Add singleton dimension at axis 1

            // Grid sample with bilinear interpolation and align_corners=False
            let out = gridSample(
                triplane,               // [Np=3, Cp, Hp, Wp]
                grid: reshapedIndices,  // [Np=3, 1, N, Nd=2]
                alignCorners: false,
                mode: "bilinear"
            )

            // Apply feature reduction following Python implementation
            let reducedOut: MLXArray
            if config.featureReduction == "concat" {
                // rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
                // From [Np=3, Cp, 1, N] to [N, Np*Cp]
                reducedOut = out.squeezed(axis: 2).transposed(axes: [2, 0, 1]).reshaped([-1, out.dim(0) * out.dim(1)])
            } else if config.featureReduction == "mean" {
                // reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")
                // From [Np=3, Cp, 1, N] to [N, Cp] by averaging over Np dimension
                reducedOut = out.squeezed(axis: 2).mean(axis: 0).transposed()
            } else {
                fatalError("Feature reduction method not implemented: \(config.featureReduction)")
            }

            return decoder(reducedOut)
        }

        let netOut: NeRFDecoderOutput
        if chunkSize > 0 {
            // Process in chunks
            let chunks = chunkBatch(queryChunk, chunkSize: chunkSize, normalizedPositions)
            netOut = combineChunkedResults(chunks)
        } else {
            netOut = queryChunk(normalizedPositions)
        }

        // Apply activation functions (TripoSR style)
        // Note: Density bias is already applied in the decoder for TripoSR compliance
        let densityAct = getActivation(config.densityActivation)(netOut.density+config.densityBias)
        let color = getActivation(config.colorActivation)(netOut.features)

        // Reshape back to original shape
        let finalShape = inputShape + [netOut.density.dim(-1)]
        let density = netOut.density.reshaped(finalShape)
        let densityActReshaped = densityAct.reshaped(finalShape)
        let features = netOut.features.reshaped(inputShape + [netOut.features.dim(-1)])
        let colorReshaped = color.reshaped(inputShape + [color.dim(-1)])

        return NeRFRendererOutput(
            density: density,
            densityAct: densityActReshaped,
            features: features,
            color: colorReshaped
        )
    }
    func conditionToIndices(condition: MLXArray) -> MLXArray {
        let arange = MLX.where(condition, MLXArray(0..<condition.shape[0]), MLXArray(Int32.max))
        let sorted = MLX.sorted(arange)
        if sorted.shape[0] == 0 {
            return MLXArray([])
        }
        let index = MLX.argMax(sorted)
        return sorted[0..<index.item(Int.self)]
    }
    /// Internal forward pass for volume rendering
    private func forward(
        decoder: NeRFDecoder,
        triplane: MLXArray,
        raysO: MLXArray,
        raysD: MLXArray
    ) -> MLXArray {
        let raysShape = Array(raysO.shape.dropLast())
        let raysOFlat = raysO.reshaped([-1, 3])
        let raysDFlat = raysD.reshaped([-1, 3])
        let nRays = raysOFlat.dim(0)

        // Compute ray-bounding box intersection
        let (tNear, tFar, raysValidCondition) = raysIntersectBbox(raysOFlat, raysDFlat, config.radius)
        // Get valid rays only - proper masking approach
        let raysValid = conditionToIndices(condition: raysValidCondition)
        let tNearValid = tNear[raysValid]
        let tFarValid = tFar[raysValid]
        
        // Sample points along rays using linear interpolation between 0 and 1
        let tVals = linspace(0.0 as Float, 1.0 as Float, count: config.numSamplesPerRay + 1)

        let tMid = (tVals[..<(tVals.shape[0]-1)]  + tVals[1...])/2.0
        // Properly expand dimensions for broadcasting: [N_valid] and [N_samples] -> [N_valid, N_samples]
        let tNearExpanded = tNearValid.expandedDimensions(axis: -1)  // [N_valid, 1]
        let tFarExpanded = tFarValid.expandedDimensions(axis: -1)    // [N_valid, 1]
        let tMidExpanded = tMid.expandedDimensions(axis: 0)          // [1, N_samples]
        let zVals = tNearExpanded * (1 - tMidExpanded) + tFarExpanded * tMidExpanded
        // Get valid rays for position calculation
        let validRaysO = raysOFlat[raysValid]  // [N_valid, 3]
        let validRaysD = raysDFlat[raysValid]  // [N_valid, 3]
        let xyz = validRaysO.expandedDimensions(axis: 1) +
                 zVals.expandedDimensions(axis: -1) * validRaysD.expandedDimensions(axis: 1)

        // Query triplane
        let mlpOut = queryTriplane(decoder: decoder, positions: xyz, triplane: triplane)

        // TripoSR-compliant volume rendering integration
        let eps: Float = 1e-10

        // Use t_vals spacing for deltas (not z_vals differences) - TripoSR official approach
        let deltas = tVals[1..<(config.numSamplesPerRay + 1)] - tVals[0..<config.numSamplesPerRay]

        // Compute alpha values from density
        let alpha = 1.0 - exp(-deltas * mlpOut.densityAct[.ellipsis, 0])

        // TripoSR alpha composition with proper accumulation
        let onesLike = ones(like: alpha[0..., 0..<1])
        let alphaShifted = alpha[0..., 0..<(config.numSamplesPerRay - 1)]
        let cumprodInput = 1.0 - alphaShifted + eps  // Add epsilon for numerical stability
        let cumprod = cumprod(cumprodInput, axis: -1)
        let accumProd = concatenated([onesLike, cumprod], axis: -1)

        // Final weights and rendering
        let weights = alpha * accumProd
        let compRgbValid = (weights.expandedDimensions(axis: -1) * mlpOut.color).sum(axis: -2)
        let opacityValid = weights.sum(axis: -1)

        // Initialize full-size output tensors
        var compRgb = zeros([nRays, 3], dtype: compRgbValid.dtype)
        var opacity = zeros([nRays], dtype: opacityValid.dtype)

        // Fill in values for valid rays - simplified approach
        // In full implementation, would use proper scatter operations
        compRgb[raysValid] = compRgbValid
        opacity[raysValid] = opacityValid
        return MLX.concatenated([compRgb.reshaped(raysShape + [3]), opacity.reshaped(raysShape + [1])],axis: -1)
        // TripoSR white background addition: comp_rgb += 1 - opacity[..., None]
//        compRgb = compRgb + (1.0 - opacity.expandedDimensions(axis: -1))
//
//        return compRgb.reshaped(raysShape + [3])
    }

    /// Main forward pass supporting batched inputs
    public func callAsFunction(
        decoder: NeRFDecoder,
        triplane: MLXArray,
        raysO: MLXArray,
        raysD: MLXArray
    ) -> MLXArray {
        if triplane.ndim == 4 {
            // Single triplane
            return forward(decoder: decoder, triplane: triplane, raysO: raysO, raysD: raysD)
        } else {
            // Batched triplanes
            var results: [MLXArray] = []
            for i in 0..<triplane.dim(0) {
                let result = forward(
                    decoder: decoder,
                    triplane: triplane[i],
                    raysO: raysO[i],
                    raysD: raysD[i]
                )
                results.append(result)
            }
            return stacked(results, axis: 0)
        }
    }

    /// Set training mode
    public func setTrainingMode(_ training: Bool) {
        isRandomized = training && config.randomized
    }

}


/// TripoSR official configuration extension
public extension TriplaneNeRFRendererConfig {
    /// Create TripoSR official renderer configuration
    nonisolated static func tripoSRConfig(radius: Float) -> TriplaneNeRFRendererConfig {
        return TriplaneNeRFRendererConfig(
            radius: radius,
            featureReduction: "concat",
            densityActivation: "trunc_exp",
            densityBias: -1.0,
            colorActivation: "sigmoid",
            numSamplesPerRay: 128,
            randomized: false
        )
    }
}
