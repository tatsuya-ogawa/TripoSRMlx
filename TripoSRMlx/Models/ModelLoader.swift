//
//  ModelLoader.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/20.
//
import MLX
import Foundation
import MLXNN

/// Model loading error types
public enum ModelLoadingError: Error {
    case bundleFileNotFound(String)
    case loadingFailed(String)
    case applyingFailed(String)
    case downloadFailed(String)
    case cacheDirectoryCreationFailed
    case imageLoadingFailed(String)
}

class ModelLoader{
    private func sanitize(weights: [String:MLXArray]) -> [String:MLXArray]{
        var sanitizedWeights: [String: MLXArray] = [:]

        for (key, array) in weights {
            sanitizedWeights[key] = applyWeightTransformation(key: key, array: array)
        }

        return sanitizedWeights
    }

    private func applyWeightTransformation(key: String, array: MLXArray) -> MLXArray {
        guard array.ndim == 4 else { return array }

        // Conv2d weights: PyTorch[out, in, h, w] -> MLX[out, h, w, in]
        if isConv2dWeight(key: key) {
            return array.transposed(axes: [0, 2, 3, 1])
        }

        // ConvTransposed2d weights: PyTorch[in, out, h, w] -> MLX[out, h, w, in]
        if isConvTransposed2dWeight(key: key) {
            return array.transposed(axes: [1, 2, 3, 0])
        }

        return array
    }

    private func isConv2dWeight(key: String) -> Bool {
        return key.contains("patch_embeddings.projection.weight")
    }

    private func isConvTransposed2dWeight(key: String) -> Bool {
        return key.contains("upsample.weight")
    }
    public func loadWeight(from weightsURL: URL) throws -> [String:MLXArray]{
        let weights = try MLX.loadArrays(url: weightsURL)
        return sanitize(weights: weights)
    }

    /// Load and apply weights to a TSRSystem model from HuggingFace cache or download
    public func loadAndApplyWeights(to model: Module) throws {
        let weights = try loadWeightsFromCacheOrDownload()
        let mappedWeights = mapDecoderWeights(weights)
        try applyWeights(mappedWeights, to: model)
        print("✅ Successfully loaded and applied weights from HuggingFace cache")
    }

    /// Load weights from cache or download from HuggingFace
    private func loadWeightsFromCacheOrDownload() throws -> [String: MLXArray] {
        let cacheURL = try getCacheURL()

        // Check if cached file exists
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            print("📁 Loading weights from cache: \(cacheURL.path)")
            return try loadWeight(from: cacheURL)
        }

        // Download from HuggingFace
        print("⬇️ Downloading weights from HuggingFace...")
        try downloadModelFromHuggingFace(to: cacheURL)
        print("✅ Downloaded and cached weights")

        return try loadWeight(from: cacheURL)
    }

    /// Get cache directory URL for model.safetensors
    private func getCacheURL() throws -> URL {
        let fileManager = FileManager.default

        // Get application support directory
        guard let appSupportDir = fileManager.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first else {
            throw ModelLoadingError.cacheDirectoryCreationFailed
        }

        // Create TripoSR cache directory
        let cacheDir = appSupportDir.appendingPathComponent("TripoSR/models")

        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: cacheDir.path) {
            try fileManager.createDirectory(at: cacheDir,
                                          withIntermediateDirectories: true,
                                          attributes: nil)
        }

        return cacheDir.appendingPathComponent("model.safetensors")
    }

    /// Download model from HuggingFace repository
    private func downloadModelFromHuggingFace(to cacheURL: URL) throws {
        let huggingFaceURL = "https://huggingface.co/togawa83/TripoSRSafetensors/resolve/main/model.safetensors"

        guard let url = URL(string: huggingFaceURL) else {
            throw ModelLoadingError.downloadFailed("Invalid HuggingFace URL")
        }

        let semaphore = DispatchSemaphore(value: 0)
        var downloadError: Error?

        let task = URLSession.shared.downloadTask(with: url) { tempURL, response, error in
            defer { semaphore.signal() }

            if let error = error {
                downloadError = ModelLoadingError.downloadFailed("Download failed: \(error.localizedDescription)")
                return
            }

            guard let tempURL = tempURL else {
                downloadError = ModelLoadingError.downloadFailed("No temporary file URL")
                return
            }

            do {
                // Move downloaded file to cache location
                try FileManager.default.moveItem(at: tempURL, to: cacheURL)
            } catch {
                downloadError = ModelLoadingError.downloadFailed("Failed to move file to cache: \(error.localizedDescription)")
            }
        }

        task.resume()
        semaphore.wait()

        if let error = downloadError {
            throw error
        }
    }

    /// Load example image from cache or download from HuggingFace
    public func loadExampleImage(filename: String) throws -> URL {
        let cacheURL = try getImageCacheURL(filename: filename)

        // Check if cached file exists
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            print("📁 Loading image from cache: \(cacheURL.path)")
            return cacheURL
        }

        // Download from HuggingFace
        print("⬇️ Downloading image from HuggingFace...")
        try downloadImageFromHuggingFace(filename: filename, to: cacheURL)
        print("✅ Downloaded and cached image: \(filename)")

        return cacheURL
    }

    /// Get cache directory URL for example images
    private func getImageCacheURL(filename: String) throws -> URL {
        let fileManager = FileManager.default

        // Get application support directory
        guard let appSupportDir = fileManager.urls(for: .applicationSupportDirectory,
                                                  in: .userDomainMask).first else {
            throw ModelLoadingError.cacheDirectoryCreationFailed
        }

        // Create TripoSR examples cache directory
        let cacheDir = appSupportDir.appendingPathComponent("TripoSR/examples")

        // Create directory if it doesn't exist
        if !fileManager.fileExists(atPath: cacheDir.path) {
            try fileManager.createDirectory(at: cacheDir,
                                          withIntermediateDirectories: true,
                                          attributes: nil)
        }

        return cacheDir.appendingPathComponent(filename)
    }

    /// Download example image from HuggingFace repository
    private func downloadImageFromHuggingFace(filename: String, to cacheURL: URL) throws {
        let huggingFaceURL = "https://huggingface.co/togawa83/TripoSRSafetensors/resolve/main/examples/\(filename)"

        guard let url = URL(string: huggingFaceURL) else {
            throw ModelLoadingError.downloadFailed("Invalid HuggingFace image URL")
        }

        let semaphore = DispatchSemaphore(value: 0)
        var downloadError: Error?

        let task = URLSession.shared.downloadTask(with: url) { tempURL, response, error in
            defer { semaphore.signal() }

            if let error = error {
                downloadError = ModelLoadingError.downloadFailed("Image download failed: \(error.localizedDescription)")
                return
            }

            guard let tempURL = tempURL else {
                downloadError = ModelLoadingError.downloadFailed("No temporary file URL for image")
                return
            }

            do {
                // Move downloaded file to cache location
                try FileManager.default.moveItem(at: tempURL, to: cacheURL)
            } catch {
                downloadError = ModelLoadingError.downloadFailed("Failed to move image file to cache: \(error.localizedDescription)")
            }
        }

        task.resume()
        semaphore.wait()

        if let error = downloadError {
            throw error
        }
    }

    /// Decoder weights are already in the correct format (0,2,4,6,8,10,12,14,16,18) - no mapping needed
    private func mapDecoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        // PyTorch weights already use the correct sparse indices, no remapping needed
        print("✅ Decoder weights already in correct format - no mapping required")
        return weights
    }

    /// Apply loaded weights to the model
    private func applyWeights(_ weights: [String: MLXArray], to model: Module) throws {
        let parameters = ModuleParameters.unflattened(weights)
        // Apply weights to model parameters
        model.update(parameters: parameters)
    }
}
