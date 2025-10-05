//
//  RenderingUtils.swift
//  TripoSRMlx
//
//  Utility functions for NeRF rendering operations.
//

import Foundation
import MLX
import MLXNN

/// Normalize vector along last axis
nonisolated public func normalized(_ vector: MLXArray) -> MLXArray {
    let norm = sqrt((vector * vector).sum(axes: [-1], keepDims: true))
    return vector / (norm + 1e-8)
}

/// Scale tensor values from one range to another
nonisolated public func scaleTensor(_ tensor: MLXArray, from fromRange: (Float, Float), to toRange: (Float, Float)) -> MLXArray {
    let (fromMin, fromMax) = fromRange
    let (toMin, toMax) = toRange

    let normalized = (tensor - fromMin) / (fromMax - fromMin)
    return normalized * (toMax - toMin) + toMin
}

/// TripoSR-compliant ray-sphere intersection with proper numerical stability
nonisolated public func raysIntersectBbox(
    _ raysO: MLXArray,
    _ raysD: MLXArray,
    _ radius: Float,
    near: Float = 0.0,
    validThresh: Float = 0.01
) -> (tNear: MLXArray, tFar: MLXArray, raysValid: MLXArray) {

    // TripoSR numerical stability: handle degenerate ray directions
    let eps: Float = 1e-6
    let raysDValid = MLX.where(  // Keep MLX.where for complex conditional
        abs(raysD) .< eps,
        MLX.where(raysD .>= 0, MLXArray(eps), MLXArray(-eps)), 
        raysD
    )

    // TripoSR radius tightening for numerical stability
    let tightenedRadius = (1.0 - 1.0e-3) * radius

    // Create radius bounds for each axis
    let radiusMin = MLXArray([-tightenedRadius, -tightenedRadius, -tightenedRadius]  as [Float] )
    let radiusMax = MLXArray([tightenedRadius, tightenedRadius, tightenedRadius]  as [Float])

    // Compute intersection points
    let interx0 = (radiusMax - raysO) / raysDValid
    let interx1 = (radiusMin - raysO) / raysDValid

    let tNearAxis = MLX.minimum(interx0, interx1)
    let tFarAxis = MLX.maximum(interx0, interx1)

    // Get the furthest near and nearest far across all axes
    let tNear = clip(tNearAxis.max(axis: -1), min: near)
    let tFar = tFarAxis.min(axis: -1)

    // TripoSR validity check with threshold
    let raysValid = (tFar - tNear) .> MLXArray(validThresh)

    return (tNear, tFar, raysValid)
}

/// Process data in chunks to manage memory usage
nonisolated public func chunkBatch<T>(_ processFn: (MLXArray) -> T, chunkSize: Int, _ input: MLXArray) -> [T] {
    guard chunkSize > 0 else {
        return [processFn(input)]
    }

    let totalSize = input.dim(0)
    var results: [T] = []

    for i in stride(from: 0, to: totalSize, by: chunkSize) {
        autoreleasepool{
            let endIdx = min(i + chunkSize, totalSize)
            let chunk = input[i..<endIdx]
            results.append(processFn(chunk))
        }
    }

    return results
}

/// Combine MLXArrays from chunked processing
nonisolated public func combineChunkedResults(_ chunks: [NeRFDecoderOutput]) -> NeRFDecoderOutput {
    guard !chunks.isEmpty else {
        fatalError("No chunks to combine.")
    }
    var density: [MLXArray] = []
    var features: [MLXArray] = []

    for chunk in chunks {
        density.append(chunk.density)
        features.append(chunk.features)
    }
    return NeRFDecoderOutput(density:concatenated(density,axis: 0),features: concatenated(features,axis: 0))
}

/// Get activation function by name
nonisolated public func getActivation(_ name: String) -> (MLXArray) -> MLXArray {
    switch name {
    case "sigmoid":
        return { sigmoid($0) }
    case "relu":
        return { relu($0) }
    case "trunc_exp":
        return { x in
            // TripoSR official trunc_exp: direct exp (gradient clamping in backward pass)
            return exp(x)
        }
    case "softplus":
        return { softplus($0) }
    default:
        return { $0 } // Identity
    }
}

/// Grid sample operation similar to PyTorch's F.grid_sample
/// Implements vectorized bilinear interpolation for triplane sampling
nonisolated public func gridSample(_ input: MLXArray, grid: MLXArray, alignCorners: Bool = false, mode: String = "bilinear") -> MLXArray {
    // For TripoSR triplane sampling:
    // input: [Np=3, C, H, W] - triplane features
    // grid: [Np=3, 1, N, 2] - sampling coordinates in [-1, 1] range
    // Returns: [Np=3, C, 1, N] - sampled features
    let Np = input.dim(0)  // Number of planes (3 for triplane)
    let C = input.dim(1)   // Feature channels
    let H = input.dim(2)   // Height
    let W = input.dim(3)   // Width
    let N = grid.dim(2)    // Number of sample points

    // Extract x, y coordinates from grid [Np, 1, N, 2]
    let gridX = grid[0..., 0, 0..., 0]  // [Np, N]
    let gridY = grid[0..., 0, 0..., 1]  // [Np, N]

    // Convert normalized coordinates [-1, 1] to pixel coordinates
    let pixelX: MLXArray
    let pixelY: MLXArray

    if alignCorners {
        pixelX = (gridX + 1.0) * Float(W - 1) / 2.0
        pixelY = (gridY + 1.0) * Float(H - 1) / 2.0
    } else {
        pixelX = (gridX + 1.0) * Float(W) / 2.0 - 0.5
        pixelY = (gridY + 1.0) * Float(H) / 2.0 - 0.5
    }

    // Get integer coordinates for bilinear interpolation
    let x0 = floor(pixelX).asType(.int32)
    let y0 = floor(pixelY).asType(.int32)
    let x1 = x0 + 1
    let y1 = y0 + 1

    // Get fractional parts for interpolation weights
    let wx = pixelX - floor(pixelX)  // [Np, N]
    let wy = pixelY - floor(pixelY)  // [Np, N]

    // Clamp coordinates to valid range
    let x0_clamped = clip(x0, min: 0, max: W - 1)  // [Np, N]
    let y0_clamped = clip(y0, min: 0, max: H - 1)  // [Np, N]
    let x1_clamped = clip(x1, min: 0, max: W - 1)  // [Np, N]
    let y1_clamped = clip(y1, min: 0, max: H - 1)  // [Np, N]

    // Convert 2D coordinates to flat indices: y * W + x
    let idx00 = y0_clamped * W + x0_clamped  // [Np, N]
    let idx01 = y0_clamped * W + x1_clamped  // [Np, N]
    let idx10 = y1_clamped * W + x0_clamped  // [Np, N]
    let idx11 = y1_clamped * W + x1_clamped  // [Np, N]

    // Vectorized sampling using take operations
    // Process each plane separately but vectorize across channels and points
    var q00_values = zeros([Np, C, N])
    var q01_values = zeros([Np, C, N])
    var q10_values = zeros([Np, C, N])
    var q11_values = zeros([Np, C, N])

    for p in 0..<Np {
        // Get all indices for this plane: [N]
        let indices00_p = idx00[p]
        let indices01_p = idx01[p]
        let indices10_p = idx10[p]
        let indices11_p = idx11[p]

        // Get plane feature map: [C, H, W] -> [C, H*W]
        let planeFeatures = input[p].reshaped([C, H * W])  // [C, H*W]

        // Vectorized sampling across all channels using broadcasting
        // take(planeFeatures, indices, axis=1) samples from spatial dimension for all channels
        q00_values[p] = take(planeFeatures, indices00_p, axis: 1)  // [C, N]
        q01_values[p] = take(planeFeatures, indices01_p, axis: 1)  // [C, N]
        q10_values[p] = take(planeFeatures, indices10_p, axis: 1)  // [C, N]
        q11_values[p] = take(planeFeatures, indices11_p, axis: 1)  // [C, N]
    }

    // Broadcast interpolation weights to [Np, C, N]
    let wx_broad = MLX.broadcast(wx.expandedDimensions(axis: 1), to: [Np, C, N])
    let wy_broad = MLX.broadcast(wy.expandedDimensions(axis: 1), to: [Np, C, N])

    // Vectorized bilinear interpolation
    let interpolated = q00_values * (1.0 - wx_broad) * (1.0 - wy_broad) +
                      q01_values * wx_broad * (1.0 - wy_broad) +
                      q10_values * (1.0 - wx_broad) * wy_broad +
                      q11_values * wx_broad * wy_broad

    // Reshape to [Np, C, 1, N] to match expected output format
    return interpolated.expandedDimensions(axis: 2)
}
