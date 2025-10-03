//
//  TSRTestViewModel.swift
//  TripoSRMlx
//
//  ViewModel for testing TSRSystem with HuggingFace model and image loading
//

import SwiftUI
import MLX
import MLXNN
import Foundation
import Combine

@MainActor
class TSRTestViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var currentStatus = "Ready"
    @Published var errorMessage: String?
    @Published var processedImage: UIImage?
    @Published var processedMLXImage: UIImage?
    @Published var renderedViews: [UIImage] = []
    @Published var extractedMeshes: [TriMesh] = []
    @Published var meshExtractionTime: TimeInterval = 0
    @Published var progress: Double = 0.0

    private var tsrSystem: TSRSystem?
    private var _lastSceneCodes: MLXArray?
    private let modelLoader = ModelLoader()

    // Read-only access to lastSceneCodes for UI
    var lastSceneCodes: MLXArray? {
        return _lastSceneCodes
    }

    init() {
        setupTSRSystem()
    }

    private func setupTSRSystem() {
        currentStatus = "Initializing TSRSystem..."
        tsrSystem = TSRSystem.createTripoSRModel()
        currentStatus = "TSRSystem initialized"
    }

    /// Load and process robot.png to generate scene codes
    func processRobotImage() async {
        isLoading = true
        errorMessage = nil
        progress = 0.0
        renderedViews = []

        Task.detached{
            do {
                await MainActor.run{
                    // Step 1: Load robot.png from HuggingFace cache or download
                    self.currentStatus = "Loading robot.png from HuggingFace..."
                    self.progress = 0.1
                }
                
                let robotImageURL = try self.modelLoader.loadExampleImage(filename: "robot.png")
                guard let robotImage = UIImage(contentsOfFile: robotImageURL.path) else {
                    throw TSRTestError.imageNotFound("Failed to load robot.png from cache")
                }
                
                // Resize image to INPUT_IMAGE_SIZExINPUT_IMAGE_SIZE using CoreGraphics for ViT model compatibility
                let resizedImage = self.resizeImageWithCoreGraphics(robotImage, targetSize: CGSize(width: INPUT_IMAGE_SIZE, height: INPUT_IMAGE_SIZE))
                await MainActor.run{
                    self.processedImage = resizedImage
                    self.progress = 0.2
                }

                await MainActor.run{
                    // Step 3: Convert UIImage to MLXArray
                    self.currentStatus = "Converting image to MLXArray..."
                    self.progress = 0.3
                }
                let mlxImage = try ImagePreprocessor.preprocessImageToMLX(PlatformImage(uiImage: resizedImage), removeBackground: true)
                
                await MainActor.run{
                    // Step 4: Load model weights from HuggingFace cache or download
                    self.currentStatus = "Loading model weights from HuggingFace..."
                    self.progress = 0.4
                }

                guard let system = await self.tsrSystem else {
                    throw TSRTestError.systemNotInitialized("TSRSystem not initialized")
                }
                
                try self.modelLoader.loadAndApplyWeights(to: system)
                
                await MainActor.run{
                    // Step 3.5: Convert MLXArray back to UIImage for debugging
                    self.currentStatus = "Converting MLXArray back to UIImage for debugging..."
                    self.progress = 0.6
                }
                let image = mlxImage.toUIImage()
                await MainActor.run{
                    self.processedMLXImage = image
                    
                    // Step 4: Process through TSRSystem
                    self.currentStatus = "Processing through TSRSystem..."
                    self.progress = 0.7
                }
                let result = system.forward([mlxImage])
                
                // Store scene codes for further processing
                await MainActor.run{
                    self._lastSceneCodes = result.sceneCodes
                    self.progress = 0.8
                    self.currentStatus = "Scene codes generated successfully!"
                }
            } catch {
                await MainActor.run{
                    self.errorMessage = error.localizedDescription
                    self.currentStatus = "Error: \(error.localizedDescription)"
                }
                print("❌ TSR processing error: \(error)")
            }
        }
        isLoading = false
    }

    /// Render scene codes to multiple viewpoints
    func renderViews() async {
        guard let sceneCodes = _lastSceneCodes,
              let system = tsrSystem else {
            errorMessage = "No scene codes available. Please process an image first."
            return
        }

        isLoading = true
        currentStatus = "Rendering multiple viewpoints..."
        progress = 0.0

        Task.detached {
            do {
                await MainActor.run {
                    self.progress = 0.2
                }

                let renderResults = system.render(
                    sceneCodes: sceneCodes,
                    nViews: 3,
                    elevationDeg: 0.0,
                    cameraDistance: 1.9,
                    fovyDeg: 40.0,
                    height: 256,
                    width: 256
                )

                await MainActor.run {
                    // Extract UIImages from render results
                    if let firstBatch = renderResults.first {
                        self.renderedViews = firstBatch.compactMap { MLX.stopGradient($0).toUIImage()}
                    }

                    self.progress = 1.0
                    self.currentStatus = "Rendering completed successfully!"
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Rendering failed: \(error.localizedDescription)"
                    self.currentStatus = "Error: \(error.localizedDescription)"
                    self.isLoading = false
                }
                print("❌ Rendering error: \(error)")
            }
        }
    }

    /// Resize UIImage using CoreGraphics (efficient and high quality)
    nonisolated private func resizeImageWithCoreGraphics(_ image: UIImage, targetSize: CGSize) -> UIImage {
        // Force scale to 1.0 to avoid Retina scaling issues
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)

        let resizedImage = renderer.image { context in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        return resizedImage
    }

    /// Extract mesh from processed scene codes
    func extractMesh(resolution: Int = 256, hasVertexColor: Bool = true, threshold: Float = 25.0) async {
        guard let sceneCodes = _lastSceneCodes,
              let system = tsrSystem else {
            errorMessage = "No scene codes available. Please process an image first."
            return
        }

        isLoading = true
        currentStatus = "Extracting mesh..."
        progress = 0.9

        Task.detached {
            do {
                let startTime = Date()

                // Extract mesh using TSRSystem - matching PyTorch parameters
                let meshes = system.extractMesh(
                    sceneCodes: sceneCodes,
                    hasVertexColor: hasVertexColor,
                    resolution: resolution,
                    threshold: threshold
                )

                let endTime = Date()
                let extractionTime = endTime.timeIntervalSince(startTime)

                await MainActor.run {
                    self.extractedMeshes = meshes
                    self.meshExtractionTime = extractionTime
                    self.progress = 1.0
                    self.currentStatus = "Mesh extraction completed successfully!"
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Mesh extraction failed: \(error.localizedDescription)"
                    self.currentStatus = "Error: \(error.localizedDescription)"
                    self.isLoading = false
                }
                print("❌ Mesh extraction error: \(error)")
            }
        }
    }

    /// Export mesh to file
    func exportMesh(at index: Int = 0, format: String = "obj") -> URL? {
        guard index < extractedMeshes.count else {
            errorMessage = "Invalid mesh index"
            return nil
        }

        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "extracted_mesh_\(timestamp).\(format)"
        let fileURL = documentsPath.appendingPathComponent(filename)

        do {
            try extractedMeshes[index].export(to: fileURL, format: format)
            return fileURL
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Reset the test state
    func reset() {
        isLoading = false
        currentStatus = "Ready"
        errorMessage = nil
        processedImage = nil
        processedMLXImage = nil
        renderedViews = []
        extractedMeshes = []
        meshExtractionTime = 0
        _lastSceneCodes = nil
        progress = 0.0
    }
}

/// Error types for TSR testing
enum TSRTestError: Error, LocalizedError {
    case imageNotFound(String)
    case imageConversionFailed(String)
    case systemNotInitialized(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .imageNotFound(let message):
            return "Image not found: \(message)"
        case .imageConversionFailed(let message):
            return "Image conversion failed: \(message)"
        case .systemNotInitialized(let message):
            return "System not initialized: \(message)"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}
