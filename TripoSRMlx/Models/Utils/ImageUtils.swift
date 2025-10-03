//
//  ImageUtils.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/21.
//
import Foundation
import MLX
import MLXNN
import SwiftUI
import MLXRandom
import SwiftUI
import Foundation
//private func processRenderOutput(_ image: MLXArray, returnType: RenderOutputType) -> Any {
//    switch returnType {
//    case .mlxArray:
//        return image
//    case .numpy:
//        // Convert MLXArray to numpy-equivalent data
//        return image.asArray(Float.self)
//    case .uiImage:
//        // Convert to UIImage
//        return convertToUIImage(image)
//    }
//}
extension MLXArray {
    nonisolated func toUIImage() -> UIImage? {
        var rgb = self
        let height = Int(rgb.shape[rgb.shape.count - 3])
        let width = Int(rgb.shape[rgb.shape.count - 2])
        if rgb.shape[rgb.shape.count - 1] == 3 {
            let alpha = MLXArray.ones(like: rgb[.ellipsis, 0..<1])
            rgb = MLX.concatenated([rgb, alpha], axis: -1)
        }
        var uint8Pixels = (rgb * 256).asMLXArray(dtype: .uint8).asArray(UInt8.self)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &uint8Pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        guard let cgImage = context?.makeImage() else { return nil }
        return UIImage(cgImage: cgImage)
    }
}
