//
//  DINOOutputExporterViewCrossPlatform.swift
//  TripoSRMlx
//
//  Export MLX DINO outputs for Python comparison (Cross-platform: macOS + Catalyst)
//

import SwiftUI
import MLX
import MLXNN
import UniformTypeIdentifiers
import Foundation

#if canImport(UIKit)
import UIKit
#endif

#if canImport(AppKit)
import AppKit
#endif

// MARK: - Cross-platform Image Wrapper
struct PlatformImage {
    #if os(macOS)
    let nsImage: NSImage
    var size: CGSize { nsImage.size }

    init?(contentsOf url: URL) {
        guard let image = NSImage(contentsOf: url) else { return nil }
        self.nsImage = image
    }

    init?(data: Data) {
        guard let image = NSImage(data: data) else { return nil }
        self.nsImage = image
    }
    #else
    let uiImage: UIImage
    var size: CGSize { uiImage.size }

    init?(contentsOf url: URL) {
        guard let data = try? Data(contentsOf: url),
              let image = UIImage(data: data) else { return nil }
        self.uiImage = image
    }

    init?(data: Data) {
        guard let image = UIImage(data: data) else { return nil }
        self.uiImage = image
    }
    #endif
    init(uiImage: UIImage) {
        self.uiImage = uiImage
    }
}

// MARK: - Cross-platform Pasteboard
struct PlatformPasteboard {
    static func setString(_ string: String) {
        #if os(macOS)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(string, forType: .string)
        #else
        UIPasteboard.general.string = string
        #endif
    }
}

struct DINOOutputExporterView: View {
    @State private var selectedImage: PlatformImage?
    @State private var mlxOutput: MLXArray?
    @State private var embeddingOutput: MLXArray?
    @State private var sceneCode: MLXArray?
    @State private var tokens: MLXArray?
    @State private var originalTokens: MLXArray?
    @State private var raysO: [MLXArray]?
    @State private var raysD: [MLXArray]?
    @State private var isProcessing = false
    @State private var showingFilePicker = false
    @State private var showingDocumentPicker = false
    @State private var processingTime: Double = 0.0
    @State private var errorMessage: String?
    @State private var isLoadingDemo = false
    @State private var safetensorURL: URL?
    @State private var savedFiles: [String] = []
    @State private var exportDirectory: URL?
    @State private var statusMessage: String = ""

    private let modelLoader = ModelLoader()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
            // Header
            headerView

            Divider()

            // Image Selection
            imageSelectionSection

            Divider()

            // Processing Section
            if isProcessing {
                processingView
            } else if mlxOutput != nil {
                resultsView
            } else {
                readyToProcessView
            }

            }
            .padding()
        }
        .navigationTitle("DINO Output Exporter")
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [.image],
            onCompletion: handleImageImport
        )
        .alert("Error", isPresented: .constant(errorMessage != nil)) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .onAppear {
            // Auto-load demo image on first appearance
            if selectedImage == nil {
                loadDemoImage()
            }
        }
    }

    private var headerView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title)
                    .foregroundColor(.blue)

                VStack(alignment: .leading) {
                    Text("MLX DINO Output Exporter")
                        .font(.title2)
                        .fontWeight(.bold)

                    Text("Process images through MLX DINO and export outputs for Python comparison")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Text("🤖 Demo: Uses robot.png from HuggingFace")
                        .font(.caption)
                        .foregroundColor(.blue)
                        .opacity(0.8)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    private var imageSelectionSection: some View {
        VStack(spacing: 15) {
            Text("Step 1: Select Image")
                .font(.headline)

            HStack(spacing: 20) {
                // Image preview
                Group {
                    if let image = selectedImage {
                        #if os(macOS)
                        Image(nsImage: image.nsImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 200, height: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                        #else
                        Image(uiImage: image.uiImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 200, height: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                        #endif
                    } else {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(Color(.systemGray5))
                            .frame(width: 200, height: 200)
                            .overlay(
                                VStack {
                                    Image(systemName: "photo.badge.plus")
                                        .font(.system(size: 40))
                                        .foregroundColor(.secondary)
                                    Text("Select Image")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            )
                    }
                }

                // Image info
                VStack(alignment: .leading, spacing: 8) {
                    if let image = selectedImage {
                        Text("Image Info:")
                            .font(.headline)

                        Text("Size: \(Int(image.size.width)) × \(Int(image.size.height))")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("Will be resized to: INPUT_IMAGE_SIZE × INPUT_IMAGE_SIZE")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("Processing: RGB normalization")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    VStack(spacing: 8) {
                        Button("Select New Image") {
                            showingFilePicker = true
                        }
                        .buttonStyle(.borderedProminent)

                        Button("Load Demo (robot.png)") {
                            loadDemoImage()
                        }
                        .buttonStyle(.bordered)
                        .disabled(isLoadingDemo)


                        if isLoadingDemo {
                            HStack {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Loading demo image...")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var processingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)

            Text("Processing image through MLX DINO...")
                .font(.headline)

            Text("This may take a few seconds")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .frame(maxHeight: .infinity)
    }

    private var readyToProcessView: some View {
        VStack(spacing: 20) {
            Image(systemName: "play.circle")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("Ready to Process")
                .font(.title2)
                .fontWeight(.medium)

            Text("Click below to process the selected image through MLX DINO")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Button("Process Image") {
                processImage()
            }
            .buttonStyle(.borderedProminent)
            .disabled(selectedImage == nil)
        }
        .frame(maxHeight: .infinity)
    }

    private var resultsView: some View {
        VStack(spacing: 20) {
            // Processing info
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.title2)

                VStack(alignment: .leading) {
                    Text("Processing Complete")
                        .font(.headline)
                        .foregroundColor(.green)

                    Text("Processing time: \(String(format: "%.3f", processingTime))s")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if !statusMessage.isEmpty {
                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                }

                Spacer()
            }

            // Output info
            if let output = mlxOutput {
                outputInfoCard(output)
            }

            Divider()

            // Safetensor export section
            safetensorExportSection
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private func outputInfoCard(_ output: MLXArray) -> some View {
        GroupBox(label: Label("MLX DINO Output", systemImage: "brain")) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Shape:")
                    Spacer()
                    Text(formatShape(Array(output.shape)))
                        .font(.system(.body, design: .monospaced))
                }

                HStack {
                    Text("Data Type:")
                    Spacer()
                    Text(String(describing: output.dtype))
                        .font(.system(.body, design: .monospaced))
                }

                HStack {
                    Text("Total Elements:")
                    Spacer()
                    Text("\(output.size)")
                        .font(.system(.body, design: .monospaced))
                }

                HStack {
                    Text("Memory Size:")
                    Spacer()
                    Text(formatMemorySize(output.size * 4)) // Assuming float32
                        .font(.system(.body, design: .monospaced))
                }

                Divider()

                // Statistics
                VStack(alignment: .leading, spacing: 4) {
                    Text("Statistics:")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    HStack {
                        Text("Mean:")
                        Spacer()
                        Text(String(format: "%.6f", Float(output.mean().item(Float.self))))
                            .font(.system(.caption, design: .monospaced))
                    }

                    HStack {
                        Text("Std:")
                        Spacer()
                        Text(String(format: "%.6f", Float(output.variance().sqrt().item(Float.self))))
                            .font(.system(.caption, design: .monospaced))
                    }

                    HStack {
                        Text("Min:")
                        Spacer()
                        Text(String(format: "%.6f", Float(output.min().item(Float.self))))
                            .font(.system(.caption, design: .monospaced))
                    }

                    HStack {
                        Text("Max:")
                        Spacer()
                        Text(String(format: "%.6f", Float(output.max().item(Float.self))))
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
        }
    }

    private var safetensorExportSection: some View {
        VStack(spacing: 15) {
            HStack {
                Text("Step 2: Export for Python Comparison")
                    .font(.headline)
                Spacer()
            }

            Button("Save Safetensor") {
                saveSafetensorToTemp()
            }
            .buttonStyle(.borderedProminent)
            .disabled(mlxOutput == nil)

            // Display saved files
            if let exportDir = exportDirectory, !savedFiles.isEmpty {
                GroupBox("Saved Safetensor Files") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Directory:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                        }

                        HStack {
                            Text(exportDir.path)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .foregroundColor(.primary)
                                .lineLimit(nil)

                            Spacer()

                            Button("Copy Directory") {
                                PlatformPasteboard.setString(exportDir.path)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(4)

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Files:")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            ForEach(savedFiles, id: \.self) { filename in
                                HStack {
                                    Text("• \(filename)")
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundColor(.primary)

                                    Spacer()

                                    Button("Copy Path") {
                                        let fullPath = exportDir.appendingPathComponent(filename).path
                                        PlatformPasteboard.setString(fullPath)
                                    }
                                    .buttonStyle(.bordered)
                                    .controlSize(.mini)
                                }
                            }
                        }

                        Divider()

                        Text("💡 Python Usage:")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("python mlx_dino_output_copier.py image.jpg \"\(exportDir.path)\"")
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .foregroundColor(.blue)
                            .padding(6)
                            .background(Color(.systemGray6))
                            .cornerRadius(4)
                    }
                }
            }
        }
    }

    // MARK: - Helper Methods

    private func formatShape(_ shape: [Int]) -> String {
        return "[\(shape.map(String.init).joined(separator: ", "))]"
    }

    private func formatMemorySize(_ bytes: Int) -> String {
        let mb = Double(bytes) / (1024 * 1024)
        return String(format: "%.2f MB", mb)
    }

    private func handleImageImport(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            if let image = PlatformImage(contentsOf: url) {
                selectedImage = image
                // Reset previous results
                mlxOutput = nil
            } else {
                errorMessage = "Failed to load image from selected file"
            }
        case .failure(let error):
            errorMessage = "Failed to import image: \(error.localizedDescription)"
        }
    }

    private func processImage() {
        guard let image = selectedImage else {
            errorMessage = "No image selected"
            return
        }

        isProcessing = true
        let startTime = Date()

            do {
                // Preprocess image
                let mlxArray = try ImagePreprocessor.preprocessImageToMLX(image)

                // Load TSR system to get pre-trained image tokenizer
                let tsr = TSRSystem.createTripoSRModel()
                try self.modelLoader.loadAndApplyWeights(to: tsr)

                // Run full TSR pipeline to get all 5 outputs
                let output = tsr.forward([mlxArray])

                // Generate camera parameters for comparison using CameraUtils (same parameters as PyTorch)
                let (raysOArray, raysDArray) = CameraUtils.getSphericalCameras(
                    nViews: 30,
                    elevationDeg: 0.0,
                    cameraDistance: 1.9,
                    fovyDeg: 40.0,
                    height: 256,
                    width: 256
                )

                let endTime = Date()

                DispatchQueue.main.async {
                    self.mlxOutput = output.imageTokens
                    self.embeddingOutput = output.embeddingOutput
                    self.sceneCode = output.sceneCodes
                    self.tokens = output.tokens
                    self.originalTokens = output.originalTokens

                    // Convert from [n_views, height, width, 3] to array of [height, width, 3]
                    var raysOList: [MLXArray] = []
                    var raysDList: [MLXArray] = []
                    for i in 0..<30 {  // n_views = 30
                        raysOList.append(raysOArray[i])
                        raysDList.append(raysDArray[i])
                    }
                    self.raysO = raysOList
                    self.raysD = raysDList

                    self.processingTime = endTime.timeIntervalSince(startTime)
                    self.isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Processing failed: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
    }

    

    private func formatFileSize(_ url: URL) -> String {
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            let fileSize = attributes[.size] as? Int64 ?? 0

            if fileSize < 1024 {
                return "\(fileSize) B"
            } else if fileSize < 1024 * 1024 {
                return String(format: "%.1f KB", Double(fileSize) / 1024.0)
            } else {
                return String(format: "%.1f MB", Double(fileSize) / (1024.0 * 1024.0))
            }
        } catch {
            return "Unknown"
        }
    }

    private func getPlatformInfo() -> [String: String] {
        #if os(macOS)
        return ["os": "macOS"]
        #else
        return ["os": "iOS/Catalyst"]
        #endif
    }

    private func loadDemoImage() {
        isLoadingDemo = true

        Task {
            do {
                // Load robot.png from HuggingFace cache or download
                let robotImageURL = try modelLoader.loadExampleImage(filename: "robot.png")

                await MainActor.run {
                    // Create PlatformImage from URL
                    if let image = PlatformImage(contentsOf: robotImageURL) {
                        self.selectedImage = image
                        // Reset previous results
                        self.mlxOutput = nil
                        self.safetensorURL = nil
                        self.isLoadingDemo = false
                    } else {
                        self.errorMessage = "Failed to load demo image (robot.png)"
                        self.isLoadingDemo = false
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Failed to load demo image: \(error.localizedDescription)"
                    self.isLoadingDemo = false
                }
            }
        }
    }

    private func saveSafetensorToTemp() {
        guard let output = mlxOutput else {
            errorMessage = "No MLX output to save"
            return
        }

        do {
            // Create temporary directory for this export
            let tempDir = FileManager.default.temporaryDirectory
            let timestamp = Int(Date().timeIntervalSince1970)

            // Base metadata for all files
            let baseMetadata = [
                "framework": "MLX",
                "model": "DINOSingleImageTokenizer",
                "processing_time": String(processingTime),
                "image_size": selectedImage.map { "\(Int($0.size.width))x\(Int($0.size.height))" } ?? "unknown",
                "export_timestamp": ISO8601DateFormatter().string(from: Date()),
                "platform": getPlatformInfo()["os"] ?? "unknown"
            ]

            var fileList: [String] = []

            // 1. Save image_tokens (final DINO features)
            let imageTokensFilename = "mlx_image_tokens_\(timestamp).safetensors"
            let imageTokensURL = tempDir.appendingPathComponent(imageTokensFilename)
            var imageTokensMetadata = baseMetadata
            imageTokensMetadata["data_type"] = "image_tokens"
            imageTokensMetadata["description"] = "MLX DINO final features for comparison"
            imageTokensMetadata["shape"] = output.shape.map(String.init).joined(separator: ",")
            imageTokensMetadata["dtype"] = String(describing: output.dtype)

            try MLX.save(arrays: ["image_tokens": output], metadata: imageTokensMetadata, url: imageTokensURL)
            fileList.append(imageTokensFilename)

            // 2. Save embedding_output if available
            if let embedding = embeddingOutput {
                let embeddingFilename = "mlx_embedding_output_\(timestamp).safetensors"
                let embeddingURL = tempDir.appendingPathComponent(embeddingFilename)
                var embeddingMetadata = baseMetadata
                embeddingMetadata["data_type"] = "embedding_output"
                embeddingMetadata["description"] = "MLX DINO early-stage embeddings for comparison"
                embeddingMetadata["shape"] = embedding.shape.map(String.init).joined(separator: ",")
                embeddingMetadata["dtype"] = String(describing: embedding.dtype)

                try MLX.save(arrays: ["embedding_output": embedding], metadata: embeddingMetadata, url: embeddingURL)
                fileList.append(embeddingFilename)
            }

            // 3. Save scene_codes if available
            if let scene = sceneCode {
                let sceneFilename = "mlx_scene_codes_\(timestamp).safetensors"
                let sceneURL = tempDir.appendingPathComponent(sceneFilename)
                var sceneMetadata = baseMetadata
                sceneMetadata["data_type"] = "scene_codes"
                sceneMetadata["description"] = "MLX TripoSR scene codes for 3D generation"
                sceneMetadata["shape"] = scene.shape.map(String.init).joined(separator: ",")
                sceneMetadata["dtype"] = String(describing: scene.dtype)

                try MLX.save(arrays: ["scene_codes": scene], metadata: sceneMetadata, url: sceneURL)
                fileList.append(sceneFilename)
            }

            // 4. Save tokens if available
            if let token = tokens {
                let tokensFilename = "mlx_tokens_\(timestamp).safetensors"
                let tokensURL = tempDir.appendingPathComponent(tokensFilename)
                var tokensMetadata = baseMetadata
                tokensMetadata["data_type"] = "tokens"
                tokensMetadata["description"] = "MLX TripoSR tokens from main tokenizer"
                tokensMetadata["shape"] = token.shape.map(String.init).joined(separator: ",")
                tokensMetadata["dtype"] = String(describing: token.dtype)

                try MLX.save(arrays: ["tokens": token], metadata: tokensMetadata, url: tokensURL)
                fileList.append(tokensFilename)
            }

            // 5. Save original_tokens if available
            if let originalToken = originalTokens {
                let originalTokensFilename = "mlx_original_tokens_\(timestamp).safetensors"
                let originalTokensURL = tempDir.appendingPathComponent(originalTokensFilename)
                var originalTokensMetadata = baseMetadata
                originalTokensMetadata["data_type"] = "original_tokens"
                originalTokensMetadata["description"] = "MLX TripoSR original tokens before backbone processing"
                originalTokensMetadata["shape"] = originalToken.shape.map(String.init).joined(separator: ",")
                originalTokensMetadata["dtype"] = String(describing: originalToken.dtype)

                try MLX.save(arrays: ["original_tokens": originalToken], metadata: originalTokensMetadata, url: originalTokensURL)
                fileList.append(originalTokensFilename)
            }

            // 6. Save camera rays_o if available (all views combined)
            if let raysOArrays = raysO {
                let raysOFilename = "mlx_rays_o_\(timestamp).safetensors"
                let raysOURL = tempDir.appendingPathComponent(raysOFilename)
                let combinedRaysO = stacked(raysOArrays, axis: 0)  // Stack all views: [n_views, height, width, 3]
                var raysOMetadata = baseMetadata
                raysOMetadata["data_type"] = "rays_o"
                raysOMetadata["description"] = "MLX spherical camera ray origins (all views)"
                raysOMetadata["shape"] = combinedRaysO.shape.map(String.init).joined(separator: ",")
                raysOMetadata["dtype"] = String(describing: combinedRaysO.dtype)
                raysOMetadata["n_views"] = "4"
                raysOMetadata["elevation_deg"] = "0.0"
                raysOMetadata["camera_distance"] = "1.9"
                raysOMetadata["fovy_deg"] = "40.0"
                raysOMetadata["height"] = "256"
                raysOMetadata["width"] = "256"

                try MLX.save(arrays: ["rays_o": combinedRaysO], metadata: raysOMetadata, url: raysOURL)
                fileList.append(raysOFilename)
            }

            // 7. Save camera rays_d if available (all views combined)
            if let raysDArrays = raysD {
                let raysDFilename = "mlx_rays_d_\(timestamp).safetensors"
                let raysDURL = tempDir.appendingPathComponent(raysDFilename)
                let combinedRaysD = stacked(raysDArrays, axis: 0)  // Stack all views: [n_views, height, width, 3]
                var raysDMetadata = baseMetadata
                raysDMetadata["data_type"] = "rays_d"
                raysDMetadata["description"] = "MLX spherical camera ray directions (all views)"
                raysDMetadata["shape"] = combinedRaysD.shape.map(String.init).joined(separator: ",")
                raysDMetadata["dtype"] = String(describing: combinedRaysD.dtype)
                raysDMetadata["n_views"] = "4"
                raysDMetadata["elevation_deg"] = "0.0"
                raysDMetadata["camera_distance"] = "1.9"
                raysDMetadata["fovy_deg"] = "40.0"
                raysDMetadata["height"] = "256"
                raysDMetadata["width"] = "256"

                try MLX.save(arrays: ["rays_d": combinedRaysD], metadata: raysDMetadata, url: raysDURL)
                fileList.append(raysDFilename)
            }

            // Update UI state
            safetensorURL = imageTokensURL
            self.savedFiles = fileList
            self.exportDirectory = tempDir
            statusMessage = "💾 Saved \(fileList.count) files: \(fileList.joined(separator: ", "))"

            // Copy the directory path to clipboard for convenience
            PlatformPasteboard.setString(tempDir.path)

        } catch {
            errorMessage = "Failed to save safetensor: \(error.localizedDescription)"
        }
    }

}

#Preview {
    DINOOutputExporterView()
}
