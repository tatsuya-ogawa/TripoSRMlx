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
        let densityBufferOptional: MTLBuffer? = densityField.asMTLBuffer(device: device)
        guard let densityBuffer = densityBufferOptional else {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        let cellResolution = resolution - 1
        if cellResolution <= 0 {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        let chunkStep = max(1, min(32, cellResolution))
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 8)

        var aggregatedVertices: [Float] = []
        var aggregatedTriangles: [Int32] = []
        var totalVertexCount = 0

        let boundsMin = simd_float3(bounds.min.0, bounds.min.1, bounds.min.2)
        let boundsMax = simd_float3(bounds.max.0, bounds.max.1, bounds.max.2)

        // Process the volume in manageable sub-grids to cap temporary buffer usage.
        for xStart in stride(from: 0, to: cellResolution, by: chunkStep) {
            let cellsX = min(chunkStep, cellResolution - xStart)
            for yStart in stride(from: 0, to: cellResolution, by: chunkStep) {
                autoreleasepool{
                let cellsY = min(chunkStep, cellResolution - yStart)
                    for zStart in stride(from: 0, to: cellResolution, by: chunkStep) {
                        let cellsZ = min(chunkStep, cellResolution - zStart)
                        
                        let cubeCount = cellsX * cellsY * cellsZ
                        if cubeCount <= 0 {
                            continue
                        }
                        
                        let maxTriangles = cubeCount * 5
                        let maxVertices = maxTriangles * 3
                        
                        guard let vertexBuffer = device.makeBuffer(length: maxVertices * MemoryLayout<MetalVertex>.stride, options: .storageModeShared),
                              let triangleBuffer = device.makeBuffer(length: maxTriangles * 3 * MemoryLayout<Int32>.stride, options: .storageModeShared),
                              let vertexCounterBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared),
                              let triangleCounterBuffer = device.makeBuffer(length: MemoryLayout<Int32>.stride, options: .storageModeShared) else {
                            continue
                        }
                        
                        vertexCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee = 0
                        triangleCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee = 0
                        
                        var params = MetalMarchingCubesParams(
                            resolution: Int32(resolution),
                            threshold: threshold,
                            boundsMin: boundsMin,
                            boundsMax: boundsMax,
                            chunkOrigin: simd_int3(Int32(xStart), Int32(yStart), Int32(zStart)),
                            chunkSize: simd_int3(Int32(cellsX), Int32(cellsY), Int32(cellsZ))
                        )
                        
                        guard let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<MetalMarchingCubesParams>.stride, options: .storageModeShared),
                              let commandBuffer = commandQueue.makeCommandBuffer(),
                              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                            continue
                        }
                        
                        computeEncoder.setComputePipelineState(computePipelineState)
                        computeEncoder.setBuffer(densityBuffer, offset: 0, index: 0)
                        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(triangleBuffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(vertexCounterBuffer, offset: 0, index: 3)
                        computeEncoder.setBuffer(triangleCounterBuffer, offset: 0, index: 4)
                        computeEncoder.setBuffer(paramBuffer, offset: 0, index: 5)
                        
                        let threadgroupsPerGrid = MTLSize(
                            width: (cellsX + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                            height: (cellsY + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                            depth: (cellsZ + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth
                        )
                        
                        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                        computeEncoder.endEncoding()
                        
                        commandBuffer.commit()
                        commandBuffer.waitUntilCompleted()
                        
                        let chunkVertexCount = Int(vertexCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee)
                        let chunkTriangleCount = Int(triangleCounterBuffer.contents().assumingMemoryBound(to: Int32.self).pointee)
                        
                        guard chunkVertexCount > 0, chunkTriangleCount > 0 else {
                            continue
                        }
                        
                        let expectedVertexBufferSize = chunkVertexCount * MemoryLayout<MetalVertex>.stride
                        if vertexBuffer.length < expectedVertexBufferSize {
                            continue
                        }
                        
                        let vertexData = vertexBuffer.contents().assumingMemoryBound(to: MetalVertex.self)
                        var chunkVertices: [Float] = []
                        chunkVertices.reserveCapacity(chunkVertexCount * 3)
                        for i in 0..<chunkVertexCount {
                            let vertex = vertexData[i]
                            chunkVertices.append(vertex.position.x)
                            chunkVertices.append(vertex.position.y)
                            chunkVertices.append(vertex.position.z)
                        }
                        
                        let expectedTriangleBufferSize = chunkTriangleCount * 3 * MemoryLayout<Int32>.stride
                        if triangleBuffer.length < expectedTriangleBufferSize {
                            continue
                        }
                        
                        let triangleData = triangleBuffer.contents().assumingMemoryBound(to: Int32.self)
                        var chunkTriangles: [Int32] = []
                        chunkTriangles.reserveCapacity(chunkTriangleCount * 3)
                        for i in 0..<(chunkTriangleCount * 3) {
                            chunkTriangles.append(triangleData[i])
                        }
                        
                        let vertexOffset = Int32(totalVertexCount)
                        aggregatedVertices.reserveCapacity(aggregatedVertices.count + chunkVertices.count)
                        aggregatedVertices.append(contentsOf: chunkVertices)
                        aggregatedTriangles.reserveCapacity(aggregatedTriangles.count + chunkTriangles.count)
                        for index in chunkTriangles {
                            aggregatedTriangles.append(index + vertexOffset)
                        }
                        
                        totalVertexCount += chunkVertexCount
                    }
                }
            }
        }

        if aggregatedVertices.isEmpty || aggregatedTriangles.isEmpty {
            return (MLX.zeros([0, 3]), MLX.zeros([0, 3], dtype: .int32))
        }

        let vertices = MLXArray(aggregatedVertices).reshaped([totalVertexCount, 3])
        let faces = MLXArray(aggregatedTriangles).reshaped([aggregatedTriangles.count / 3, 3])

        print("Metal marching cubes (chunked): vertexCount=\(totalVertexCount), triangleCount=\(aggregatedTriangles.count / 3)")

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
    let chunkOrigin: simd_int3
    let chunkSize: simd_int3
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
