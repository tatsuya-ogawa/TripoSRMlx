//
//  CameraUtils.swift
//  TripoSRMlx
//
//  Complete port of PyTorch get_spherical_cameras function for MLX Swift
//  Reference: TripoSR/tsr/utils.py get_spherical_cameras()
//

import Foundation
import MLX
import MLXNN

/// Utility functions for camera and ray generation, ported from PyTorch TripoSR
public struct CameraUtils {

    /// Get ray directions for all pixels in camera coordinate
    /// Port of get_ray_directions from PyTorch TripoSR
    ///
    /// - Parameters:
    ///   - height: Image height
    ///   - width: Image width
    ///   - focal: Focal length (single value or tuple)
    ///   - principal: Principal point (optional, defaults to center)
    ///   - usePixelCenters: Whether to use pixel centers (default: true)
    ///   - normalize: Whether to normalize directions (default: true)
    /// - Returns: Ray directions tensor of shape [H, W, 3]
    public static func getRayDirections(
        height: Int,
        width: Int,
        focal: Float,
        principal: (Float, Float)? = nil,
        usePixelCenters: Bool = true,
        normalize: Bool = true
    ) -> MLXArray {
        let pixelCenter: Float = usePixelCenters ? 0.5 : 0.0

        // Use focal as both fx and fy
        let fx = focal
        let fy = focal
        let cx = principal?.0 ?? Float(width) / 2.0
        let cy = principal?.1 ?? Float(height) / 2.0

        // Create meshgrid equivalent: i for x coordinates, j for y coordinates
        let iRange = MLXArray(0..<width).asType(.float32) + pixelCenter
        let jRange = MLXArray(0..<height).asType(.float32) + pixelCenter

        // Create meshgrid - broadcast to create coordinate matrices
        let i = iRange[.newAxis, 0...]  // [1, W]
        let j = jRange[0..., .newAxis]  // [H, 1]

        let iBroadcast = MLX.broadcast(i, to: [height, width])  // [H, W]
        let jBroadcast = MLX.broadcast(j, to: [height, width])  // [H, W]

        // Calculate direction components
        // directions = [(i - cx) / fx, -(j - cy) / fy, -ones_like(i)]
        let dirX = (iBroadcast - cx) / fx
        let dirY = -((jBroadcast - cy) / fy)
        let dirZ = -MLX.ones(like: iBroadcast)

        // Stack to create [H, W, 3]
        let directions = stacked([dirX, dirY, dirZ], axis: -1)

        if normalize {
            return normalized(directions, axis: -1)
        }

        return directions
    }

    /// Get rays (origins and directions) from camera-to-world transformation
    /// Port of get_rays from PyTorch TripoSR
    ///
    /// - Parameters:
    ///   - directions: Ray directions in camera coordinates [H, W, 3] or [B, H, W, 3]
    ///   - c2w: Camera-to-world transformation matrices [4, 4] or [B, 4, 4]
    ///   - keepDim: Whether to keep spatial dimensions (default: false)
    ///   - normalize: Whether to normalize ray directions (default: false)
    /// - Returns: Tuple of (rays_o, rays_d) tensors
    public static func getRays(
        directions: MLXArray,
        c2w: MLXArray,
        keepDim: Bool = false,
        normalize: Bool = false
    ) -> (MLXArray, MLXArray) {

        var raysD: MLXArray
        var raysO: MLXArray

        if directions.ndim == 3 {  // [H, W, 3]
            if c2w.ndim == 2 {  // [4, 4]
                // directions[:, :, None, :] * c2w[None, None, :3, :3]
                let directionsExpanded = directions.expandedDimensions(axis: 2)  // [H, W, 1, 3]
                let c2wRotation = c2w[0..<3, 0..<3]  // [3, 3]
                let c2wExpanded = c2wRotation[.newAxis, .newAxis, 0..., 0...]  // [1, 1, 3, 3]

                // Matrix multiplication and sum over last dimension
                let rotated = (directionsExpanded * c2wExpanded).sum(axis: -1)  // [H, W, 3]
                raysD = rotated

                let c2wTranslation = c2w[0..<3, 3]  // [3]
                let translationExpanded = c2wTranslation[.newAxis, .newAxis, 0...]  // [1, 1, 3]
                raysO = MLX.broadcast(translationExpanded, to: raysD.shape)

            } else if c2w.ndim == 3 {  // [B, 4, 4]
                _ = c2w.dim(0)  // batchSize not used in this implementation
                let directionsExpanded = directions[.newAxis, 0..., 0..., .newAxis, 0...]  // [1, H, W, 1, 3]
                let c2wRotation = c2w[0..., 0..<3, 0..<3]  // [B, 3, 3]
                let c2wExpanded = c2wRotation[0..., .newAxis, .newAxis, 0..., 0...]  // [B, 1, 1, 3, 3]

                let rotated = (directionsExpanded * c2wExpanded).sum(axis: -1)  // [B, H, W, 3]
                raysD = rotated

                let c2wTranslation = c2w[0..., 0..<3, 3]  // [B, 3]
                let translationExpanded = c2wTranslation[0..., .newAxis, .newAxis, 0...]  // [B, 1, 1, 3]
                raysO = MLX.broadcast(translationExpanded, to: raysD.shape)
            } else {
                fatalError("Invalid c2w dimensions: \(c2w.ndim)")
            }

        } else if directions.ndim == 4 {  // [B, H, W, 3]
            assert(c2w.ndim == 3, "c2w must be [B, 4, 4] for batched directions")

            let directionsExpanded = directions.expandedDimensions(axis: 3)  // [B, H, W, 1, 3]
            let c2wRotation = c2w[0..., 0..<3, 0..<3]  // [B, 3, 3]
            let c2wExpanded = c2wRotation[0..., .newAxis, .newAxis, 0..., 0...]  // [B, 1, 1, 3, 3]

            let rotated = (directionsExpanded * c2wExpanded).sum(axis: -1)  // [B, H, W, 3]
            raysD = rotated

            let c2wTranslation = c2w[0..., 0..<3, 3]  // [B, 3]
            let translationExpanded = c2wTranslation[0..., .newAxis, .newAxis, 0...]  // [B, 1, 1, 3]
            raysO = MLX.broadcast(translationExpanded, to: raysD.shape)

        } else {
            fatalError("Unsupported directions dimensions: \(directions.ndim)")
        }

        if normalize {
            raysD = normalized(raysD, axis: -1)
        }

        if !keepDim {
            raysO = raysO.reshaped([-1, 3])
            raysD = raysD.reshaped([-1, 3])
        }

        return (raysO, raysD)
    }

    /// Generate spherical camera parameters for multi-view rendering
    /// Complete port of get_spherical_cameras from PyTorch TripoSR
    ///
    /// - Parameters:
    ///   - nViews: Number of views to generate
    ///   - elevationDeg: Elevation angle in degrees
    ///   - cameraDistance: Distance from origin to camera
    ///   - fovyDeg: Vertical field of view in degrees
    ///   - height: Image height
    ///   - width: Image width
    /// - Returns: Tuple of (rays_o, rays_d) tensors, each with shape [n_views, height, width, 3]
    public static func getSphericalCameras(
        nViews: Int,
        elevationDeg: Float,
        cameraDistance: Float,
        fovyDeg: Float,
        height: Int,
        width: Int
    ) -> (MLXArray, MLXArray) {

        // Create azimuth angles: linspace(0, 360, n_views + 1)[:n_views]
        let azimuthDeg = linspace(0.0, 360.0, count: nViews + 1)[0..<nViews]
        let elevationDegArray = MLX.full([nViews], values: MLXArray(elevationDeg), dtype: .float32)
        let cameraDistances = MLX.full([nViews], values: MLXArray(cameraDistance), dtype: .float32)

        // Convert degrees to radians
        let elevation = elevationDegArray * (Float.pi / 180.0)
        let azimuth = azimuthDeg * (Float.pi / 180.0)

        // Convert spherical coordinates to cartesian coordinates
        // Right hand coordinate system, x back, y right, z up
        // elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        let x = cameraDistances * MLX.cos(elevation) * MLX.cos(azimuth)  // x
        let y = cameraDistances * MLX.cos(elevation) * MLX.sin(azimuth)  // y
        let z = cameraDistances * MLX.sin(elevation)                      // z
        let cameraPositions = stacked([x, y, z], axis: -1)  // [n_views, 3]

        // Default scene center at origin
        let center = zeros(like: cameraPositions)  // [n_views, 3]

        // Default camera up direction as +z
        let upVector = MLXArray([0.0, 0.0, 1.0] as [Float])
        let up = MLX.broadcast(upVector[.newAxis, 0...], to: [nViews, 3])  // [n_views, 3]

        // Convert fovy to radians
        let fovy = MLX.full([nViews], values: MLXArray(fovyDeg), dtype: .float32) * (Float.pi / 180.0)

        // Compute camera coordinate system
        let lookat = normalized(center - cameraPositions, axis: -1)  // [n_views, 3]
        let right = normalized(crossProduct(lookat, up), axis: -1)    // [n_views, 3]
        let upNormalized = normalized(crossProduct(right, lookat), axis: -1)  // [n_views, 3]

        // Build camera-to-world transformation matrix
        let rotationPart = stacked([right, upNormalized, -lookat], axis: -1)  // [n_views, 3, 3]
        let translationPart = cameraPositions.expandedDimensions(axis: -1)     // [n_views, 3, 1]
        let c2w3x4 = concatenated([rotationPart, translationPart], axis: -1)   // [n_views, 3, 4]

        // Add homogeneous row [0, 0, 0, 1]
        let homogeneousRow = zeros([nViews, 1, 4])
        var c2w = concatenated([c2w3x4, homogeneousRow], axis: 1)  // [n_views, 4, 4]
        c2w[0..., 3, 3] = MLXArray(1.0)

        // Get directions by dividing directions_unit_focal by focal length
        let focalLength = 0.5 * Float(height) / MLX.tan(0.5 * fovy)  // [n_views]

        // Get ray directions with unit focal length
        let directionsUnitFocal = getRayDirections(
            height: height,
            width: width,
            focal: 1.0,
            normalize: false
        )  // [H, W, 3]

        // Repeat for all views
        let directionsUnitFocalExpanded = directionsUnitFocal[.newAxis, 0..., 0..., 0...]  // [1, H, W, 3]
        var directions = MLX.broadcast(directionsUnitFocalExpanded, to: [nViews, height, width, 3])

        // Scale by focal length: directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]
        let focalLengthExpanded = focalLength.expandedDimensions(axes: [1, 2, 3])  // [n_views, 1, 1, 1]
        let xyComponents = directions[0..., 0..., 0..., 0..<2] / focalLengthExpanded
        directions[0..., 0..., 0..., 0..<2] = xyComponents

        // Get rays using camera-to-world transformation
        // Must use normalize=True to normalize directions
        let (raysO, raysD) = getRays(
            directions: directions,
            c2w: c2w,
            keepDim: true,
            normalize: true
        )

        return (raysO, raysD)
    }
}

// MARK: - Helper Functions

/// Cross product for 3D vectors
private func crossProduct(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    // a × b = [a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x]
    let ax = a[0..., 0]
    let ay = a[0..., 1]
    let az = a[0..., 2]

    let bx = b[0..., 0]
    let by = b[0..., 1]
    let bz = b[0..., 2]

    let crossX = ay * bz - az * by
    let crossY = az * bx - ax * bz
    let crossZ = ax * by - ay * bx

    return stacked([crossX, crossY, crossZ], axis: -1)
}

/// Normalized vector along specified axis
private func normalized(_ array: MLXArray, axis: Int) -> MLXArray {
    let norm = sqrt((array * array).sum(axis: axis, keepDims: true))
    return array / norm
}

/// Linear space function equivalent to torch.linspace
private func linspace(_ start: Float, _ end: Float, count: Int) -> MLXArray {
    if count == 1 {
        return MLXArray([start])
    }

    let step = (end - start) / Float(count - 1)
    var values: [Float] = []
    for i in 0..<count {
        values.append(start + Float(i) * step)
    }
    return MLXArray(values)
}
