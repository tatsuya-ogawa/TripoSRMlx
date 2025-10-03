//
//  MetalMarchingCubes.swift
//  TripoSRMlx
//
//  Metal-accelerated marching cubes implementation for mesh extraction
//

import Foundation
import Metal
import MLX
import simd

/// Metal-accelerated marching cubes implementation
public class MetalMarchingCubes {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipelineState: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalMarchingCubesError.deviceNotFound
        }

        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalMarchingCubesError.commandQueueCreationFailed
        }

        self.device = device
        self.commandQueue = commandQueue

        // Load and compile Metal shader
        guard let defaultLibrary = device.makeDefaultLibrary() else {
            throw MetalMarchingCubesError.libraryNotFound
        }

        guard let kernelFunction = defaultLibrary.makeFunction(name: "marchingCubesKernel") else {
            throw MetalMarchingCubesError.kernelFunctionNotFound
        }

        do {
            self.computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            throw MetalMarchingCubesError.pipelineStateCreationFailed(error)
        }
    }

    /// Extract mesh using Metal compute shader
    public func extractMesh(from densityField: MLXArray, resolution: Int, threshold: Float, bounds: (min: (Float, Float, Float), max: (Float, Float, Float))) -> (vertices: MLXArray, faces: MLXArray) {

        // Convert MLX array to Metal buffer
        let densityBuffer = densityField.asMTLBuffer(device: device)

        // Estimate maximum number of vertices and triangles
        let maxVertices = resolution * resolution * resolution * 5
        let maxTriangles = maxVertices / 3

        // Create output buffers
        guard let vertexBuffer = device.makeBuffer(length: maxVertices * MemoryLayout<MetalVertex>.stride, options: .storageModeShared),
              let triangleBuffer = device.makeBuffer(length: maxTriangles * 3 * MemoryLayout<Int32>.stride, options: .storageModeShared),
              let vertexCounterBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared),
              let triangleCounterBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared) else {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        // Initialize counters
        vertexCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee = 0
        triangleCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee = 0

        // Create parameter buffer
        var params = MetalMarchingCubesParams(
            resolution: Int32(resolution),
            threshold: threshold,
            boundsMin: simd_float3(bounds.min.0, bounds.min.1, bounds.min.2),
            boundsMax: simd_float3(bounds.max.0, bounds.max.1, bounds.max.2)
        )

        guard let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<MetalMarchingCubesParams>.stride, options: .storageModeShared) else {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        // Create compute command
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(triangleBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(vertexCounterBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(triangleCounterBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(paramBuffer, offset: 0, index: 5)

        // Calculate grid size
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroupsPerGrid = MTLSize(
            width: (resolution + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (resolution + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: (resolution + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth
        )

        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        // Execute computation
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read back results
        let vertexCount = vertexCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee
        let triangleCount = triangleCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee

        print("Metal marching cubes: vertexCount=\(vertexCount), triangleCount=\(triangleCount)")

        if vertexCount <= 0 || triangleCount <= 0 {
            print("No vertices or triangles generated")
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        // Validate buffer size
        let expectedVertexBufferSize = Int(vertexCount) * MemoryLayout<MetalVertex>.stride
        if vertexBuffer.length < expectedVertexBufferSize {
            print("Vertex buffer too small: \(vertexBuffer.length) < \(expectedVertexBufferSize)")
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        // Extract vertex positions safely
        let vertexData = vertexBuffer.contents().assumingMemoryBound(to: MetalVertex.self)
        var vertexPositions: [Float] = []
        vertexPositions.reserveCapacity(Int(vertexCount) * 3)

        for i in 0..<Int(vertexCount) {
            let vertex = vertexData[i]
            vertexPositions.append(vertex.position.x)
            vertexPositions.append(vertex.position.y)
            vertexPositions.append(vertex.position.z)
        }

        // Extract triangle indices safely
        let expectedTriangleBufferSize = Int(triangleCount) * 3 * MemoryLayout<Int32>.stride
        if triangleBuffer.length < expectedTriangleBufferSize {
            print("Triangle buffer too small: \(triangleBuffer.length) < \(expectedTriangleBufferSize)")
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        let triangleData = triangleBuffer.contents().assumingMemoryBound(to: Int32.self)
        var triangleIndices: [Int32] = []
        triangleIndices.reserveCapacity(Int(triangleCount) * 3)

        for i in 0..<Int(triangleCount * 3) {
            triangleIndices.append(triangleData[i])
        }

        let vertices = MLXArray(vertexPositions).reshaped([Int(vertexCount), 3])
        let faces = MLXArray(triangleIndices).reshaped([Int(triangleCount), 3])

        return (vertices, faces)
    }
}

/// Metal vertex structure - must match Metal shader layout
private struct MetalVertex {
    let position: simd_float3
    let normal: simd_float3
    let density: Float
}

/// Metal marching cubes parameters
private struct MetalMarchingCubesParams {
    let resolution: Int32
    let threshold: Float
    let boundsMin: simd_float3
    let boundsMax: simd_float3
}

/// Errors for Metal marching cubes
public enum MetalMarchingCubesError: Error, LocalizedError {
    case deviceNotFound
    case commandQueueCreationFailed
    case libraryNotFound
    case kernelFunctionNotFound
    case pipelineStateCreationFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "Metal device not found"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .libraryNotFound:
            return "Metal library not found"
        case .kernelFunctionNotFound:
            return "Marching cubes kernel function not found"
        case .pipelineStateCreationFailed(let error):
            return "Failed to create compute pipeline state: \(error.localizedDescription)"
        }
    }
}
