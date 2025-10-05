//
//  MetalGridSample.swift
//  TripoSRMlx
//
//  Memory-efficient grid sampling using Metal compute shaders (inference-only)
//

import Foundation
import Metal
import simd

/// Metal-accelerated grid sampling for inference
/// This implementation uses compute shaders for memory efficiency
public class MetalGridSample {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let computePipelineState: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalGridSampleError.deviceNotFound
        }

        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalGridSampleError.commandQueueCreationFailed
        }

        self.device = device
        self.commandQueue = commandQueue

        // Load and compile Metal shader
        guard let defaultLibrary = device.makeDefaultLibrary() else {
            throw MetalGridSampleError.libraryNotFound
        }

        guard let kernelFunction = defaultLibrary.makeFunction(name: "gridSampleKernel") else {
            throw MetalGridSampleError.kernelFunctionNotFound
        }

        do {
            self.computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        } catch {
            throw MetalGridSampleError.pipelineStateCreationFailed(error)
        }
    }

    /// Perform grid sampling using Metal compute shader
    /// - Parameters:
    ///   - input: MTLBuffer containing input data [N, C, H, W] in float32
    ///   - inputShape: Shape of input [batch, channels, height, width]
    ///   - grid: MTLBuffer containing grid coordinates [N, H_out, W_out, 2] in float32
    ///   - gridShape: Shape of grid [batch, outHeight, outWidth, 2]
    ///   - alignCorners: Whether to align corners (default: false)
    ///   - mode: Sampling mode "bilinear" or "nearest" (default: "bilinear")
    /// - Returns: MTLBuffer containing sampled output [N, C, H_out, W_out]
    public func gridSample(
        input: MTLBuffer,
        inputShape: [Int],
        grid: MTLBuffer,
        gridShape: [Int],
        alignCorners: Bool = false,
        mode: String = "bilinear"
    ) throws -> MTLBuffer {
        // Validate input shapes
        guard inputShape.count == 4 else {
            throw MetalGridSampleError.invalidInputShape("Input must be 4D: [N, C, H, W]")
        }

        guard gridShape.count == 4 && gridShape[3] == 2 else {
            throw MetalGridSampleError.invalidGridShape("Grid must be 4D: [N, H_out, W_out, 2]")
        }

        guard inputShape[0] == gridShape[0] else {
            throw MetalGridSampleError.batchMismatch("Batch dimensions must match")
        }

        let batch = inputShape[0]
        let channels = inputShape[1]
        let inputHeight = inputShape[2]
        let inputWidth = inputShape[3]
        let outHeight = gridShape[1]
        let outWidth = gridShape[2]

        // Create output buffer
        let outputSize = batch * channels * outHeight * outWidth
        let outputBufferLength = outputSize * MemoryLayout<Float>.stride

        guard let outputBuffer = device.makeBuffer(
            length: outputBufferLength,
            options: .storageModeShared
        ) else {
            throw MetalGridSampleError.bufferCreationFailed("Failed to create output buffer")
        }

        // Create parameter buffer
        let modeValue: Int32 = (mode == "bilinear") ? 0 : 1
        var params = GridSampleParams(
            inputBatch: Int32(batch),
            inputChannels: Int32(channels),
            inputHeight: Int32(inputHeight),
            inputWidth: Int32(inputWidth),
            gridBatch: Int32(batch),
            gridHeight: Int32(outHeight),
            gridWidth: Int32(outWidth),
            alignCorners: alignCorners ? 1 : 0,
            mode: modeValue
        )

        guard let paramBuffer = device.makeBuffer(
            bytes: &params,
            length: MemoryLayout<GridSampleParams>.stride,
            options: .storageModeShared
        ) else {
            throw MetalGridSampleError.bufferCreationFailed("Failed to create parameter buffer")
        }

        // Create compute command
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalGridSampleError.commandEncoderCreationFailed
        }

        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(input, offset: 0, index: 0)
        computeEncoder.setBuffer(grid, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(paramBuffer, offset: 0, index: 3)

        // Calculate grid size
        // We dispatch one thread per output pixel per batch
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (outWidth + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (outHeight + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: batch
        )

        computeEncoder.dispatchThreadgroups(
            threadgroupsPerGrid,
            threadsPerThreadgroup: threadsPerThreadgroup
        )
        computeEncoder.endEncoding()

        // Execute computation
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Check for errors
        if let error = commandBuffer.error {
            throw MetalGridSampleError.executionFailed(error)
        }

        return outputBuffer
    }

    /// Convenience method that returns output shape
    public func getOutputShape(inputShape: [Int], gridShape: [Int]) -> [Int] {
        let batch = gridShape[0]
        let channels = inputShape[1]
        let outHeight = gridShape[1]
        let outWidth = gridShape[2]
        return [batch, channels, outHeight, outWidth]
    }
}

/// Metal grid sample parameters - must match Metal shader layout
private struct GridSampleParams {
    let inputBatch: Int32
    let inputChannels: Int32
    let inputHeight: Int32
    let inputWidth: Int32
    let gridBatch: Int32
    let gridHeight: Int32
    let gridWidth: Int32
    let alignCorners: Int32
    let mode: Int32
}

/// Errors for Metal grid sampling
public enum MetalGridSampleError: Error, LocalizedError {
    case deviceNotFound
    case commandQueueCreationFailed
    case libraryNotFound
    case kernelFunctionNotFound
    case pipelineStateCreationFailed(Error)
    case invalidInputShape(String)
    case invalidGridShape(String)
    case batchMismatch(String)
    case bufferCreationFailed(String)
    case commandEncoderCreationFailed
    case executionFailed(Error)

    public var errorDescription: String? {
        switch self {
        case .deviceNotFound:
            return "Metal device not found"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .libraryNotFound:
            return "Metal library not found"
        case .kernelFunctionNotFound:
            return "Grid sample kernel function not found"
        case .pipelineStateCreationFailed(let error):
            return "Failed to create compute pipeline state: \(error.localizedDescription)"
        case .invalidInputShape(let message):
            return "Invalid input shape: \(message)"
        case .invalidGridShape(let message):
            return "Invalid grid shape: \(message)"
        case .batchMismatch(let message):
            return "Batch mismatch: \(message)"
        case .bufferCreationFailed(let message):
            return "Buffer creation failed: \(message)"
        case .commandEncoderCreationFailed:
            return "Failed to create command encoder"
        case .executionFailed(let error):
            return "Execution failed: \(error.localizedDescription)"
        }
    }
}
