//
//  ImagePreprocessor.swift
//  TripoSRMlx
//
//  Image preprocessing utilities for TripoSR input pipeline.
//

import Foundation
import MLX
import SwiftUI
import MLXRandom
import CoreVideo
import Accelerate

/// Image preprocessing for TripoSR input pipeline
nonisolated public class ImagePreprocessor {
    public let targetSize: Int
    
    public init(targetSize: Int = 512) {
        self.targetSize = targetSize
    }
    
    /// Process input image to the required format for TripoSR
    //    public func process(_ image: MLXArray) -> MLXArray {
    //        // Resize image to target size while maintaining aspect ratio
    //        let resizedImage = resizeImage(image, targetSize: targetSize)
    //
    //        // Normalize to [0, 1] range
    //        let normalizedImage = normalizeImage(resizedImage)
    //
    //        // Convert to TripoSR format: [C, H, W]
    //        return normalizedImage
    //    }
    //
    //    /// Process multiple images in batch
    //    public func processBatch(_ images: [MLXArray]) -> MLXArray {
    //        let processedImages = images.map { process($0) }
    //        return stacked(processedImages, axis: 0)
    //    }
    //
    //    // MARK: - Private Helper Methods
    //
    //    private func resizeImage(_ image: MLXArray, targetSize: Int) -> MLXArray {
    //        // Placeholder implementation - would need proper image resizing
    //        // For now, assume image is already the right size
    //        return image
    //    }
    //    private func normalizeImage(_ image: MLXArray) -> MLXArray {
    //        // Normalize from [0, 255] to [0, 1]
    //        return image / 255.0
    //    }
    static func preprocessImageToMLX(_ image: PlatformImage,
                                     removeBackground: Bool = true) throws -> MLXArray {
        guard let cgImage = image.uiImage.cgImage else {
            fatalError("Invalid Image")
        }
        
        var cgImageFormat = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: Unmanaged.passUnretained(CGColorSpaceCreateDeviceRGB()),
            bitmapInfo: CGBitmapInfo(
                rawValue: CGImageAlphaInfo.last.rawValue
                | CGBitmapInfo.byteOrder32Big.rawValue
            ),
            version: 0,
            decode: nil,
            renderingIntent: .defaultIntent
        )
        
        let pixelBuffer = try vImage.PixelBuffer<vImage.Interleaved8x4>(
            cgImage: cgImage,
            cgImageFormat: &cgImageFormat
        )
        
        let width = pixelBuffer.width
        let height = pixelBuffer.height
        
        let mlxArray = pixelBuffer.withUnsafeBufferPointer { ptr in
            MLXArray(ptr, [height, width, 4])
        }
        let normalized = mlxArray.asType(.float32)/255.0
        if removeBackground{
            return normalized[.ellipsis, 0..<3] * normalized[.ellipsis, 3...] + 0.5 * (1-normalized[.ellipsis, 3...])
        }else{
            return normalized[.ellipsis, 0..<3]
        }
    }
}
