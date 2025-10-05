//
//  TSRDemoView.swift
//  TripoSRMlx
//
//  Demo view for TripoSR 3D generation from images using robot.png
//

import SwiftUI
import MLX

struct TSRTestView: View {
    @StateObject private var viewModel = TSRTestViewModel()
    @State private var showingSaveSuccess = false
    @State private var savedFileName = ""

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                headerSection
                
                HStack{
                    // Robot Image Display
                    if let processedImage = viewModel.processedImage {
                        imageDisplaySection(image: processedImage)
                    }
                    
                    // MLX Processed Image Display (for debugging)
                    if let processedMLXImage = viewModel.processedMLXImage {
                        mlxImageDisplaySection(image: processedMLXImage)
                    }
                }

                // Control Panel
                controlPanelSection

                // Progress and Status
                progressSection

                // Error Display
                if let errorMessage = viewModel.errorMessage {
                    errorSection(message: errorMessage)
                }

                // Rendered Views Grid
                if !viewModel.renderedViews.isEmpty {
                    renderedViewsSection
                }

                // Mesh Extraction Section
                if viewModel.lastSceneCodes != nil {
                    meshExtractionSection
                }

                // Extracted Meshes Display
                if !viewModel.extractedMeshes.isEmpty {
                    extractedMeshesSection
                }

                Spacer(minLength: 50)
            }
            .padding()
        }
        .navigationTitle("TSR Robot Test")
        .navigationBarTitleDisplayMode(.inline)
        .overlay(
            Group {
                if showingSaveSuccess {
                    VStack {
                        Spacer()
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text("Saved: \(savedFileName)")
                                .foregroundColor(.white)
                        }
                        .padding()
                        .background(Color.black.opacity(0.8))
                        .cornerRadius(10)
                        .padding(.bottom, 50)
                    }
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        )
    }

    // MARK: - View Components

    private var headerSection: some View {
        VStack(spacing: 10) {
            Image(systemName: "cube.transparent")
                .font(.system(size: 60))
                .foregroundColor(.blue)

            Text("TSRSystem Robot Test")
                .font(.title2)
                .fontWeight(.bold)

            Text("Test TSRSystem with bundled robot.png")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    private func imageDisplaySection(image: UIImage) -> some View {
        VStack(spacing: 12) {
            Text("Input Image")
                .font(.headline)

            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 200, maxHeight: 200)
                .cornerRadius(12)
                .shadow(radius: 4)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private func mlxImageDisplaySection(image: UIImage) -> some View {
        VStack(spacing: 12) {
            Text("Preprocessed Image")
                .font(.headline)
                .foregroundColor(.orange)

            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 200, maxHeight: 200)
                .cornerRadius(12)
                .shadow(radius: 4)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.orange, lineWidth: 2)
                )
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(16)
    }

    private var controlPanelSection: some View {
        VStack(spacing: 16) {
            Button(action: {
                Task {
                    await viewModel.processRobotImage()
                }
            }) {
                HStack {
                    if viewModel.isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "brain")
                    }

                    Text(viewModel.isLoading ? "Processing..." : "Process Image")
                        .fontWeight(.semibold)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isLoading ? Color.gray : Color.blue)
                .cornerRadius(12)
            }
            .disabled(viewModel.isLoading)

            // Render button (only show if scene codes are available)
            if viewModel.lastSceneCodes != nil {
                Button(action: {
                    Task {
                        await viewModel.renderViews()
                    }
                }) {
                    HStack {
                        if viewModel.isLoading {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "camera.fill")
                        }

                        Text(viewModel.isLoading ? "Rendering..." : "Render Views")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isLoading ? Color.gray : Color.green)
                    .cornerRadius(12)
                }
                .disabled(viewModel.isLoading)
            }

            Button(action: {
                viewModel.reset()
            }) {
                HStack {
                    Image(systemName: "arrow.clockwise")
                    Text("Reset")
                }
                .foregroundColor(.orange)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
            }
        }
    }

    private var progressSection: some View {
        VStack(spacing: 12) {
            // Status Text
            Text(viewModel.currentStatus)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            // Progress Bar
            if viewModel.isLoading {
                VStack(spacing: 8) {
                    ProgressView(value: viewModel.progress)
                        .progressViewStyle(LinearProgressViewStyle())

                    Text("\(Int(viewModel.progress * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func errorSection(message: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                Text("Error")
                    .font(.headline)
                    .foregroundColor(.red)
            }

            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(12)
    }

    private var renderedViewsSection: some View {
        VStack(spacing: 16) {
            Text("Rendered Views")
                .font(.headline)

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 12) {
                ForEach(Array(viewModel.renderedViews.enumerated()), id: \.offset) { index, image in
                    VStack(spacing: 4) {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 80, height: 80)
                            .cornerRadius(8)
                            .shadow(radius: 2)

                        Text("View \(index + 1)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private var meshExtractionSection: some View {
        VStack(spacing: 16) {
            Text("Mesh Extraction")
                .font(.headline)

            VStack(spacing: 12) {
                HStack {
                    Text("Resolution:")
                        .font(.subheadline)
                    Spacer()
                    Text("256")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                HStack {
                    Text("Threshold:")
                        .font(.subheadline)
                    Spacer()
                    Text("25.0")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Toggle("Vertex Colors", isOn: .constant(true))
                    .font(.subheadline)
            }
            .padding(.horizontal)

            Button(action: {
                Task {
                    await viewModel.extractMesh()
                }
            }) {
                HStack {
                    if viewModel.isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "cube.fill")
                    }

                    Text(viewModel.isLoading ? "Extracting..." : "Extract Mesh")
                        .fontWeight(.semibold)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(viewModel.isLoading ? Color.gray : Color.purple)
                .cornerRadius(12)
            }
            .disabled(viewModel.isLoading)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private var extractedMeshesSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Extracted Meshes")
                    .font(.headline)

                Spacer()

                if viewModel.meshExtractionTime > 0 {
                    Text("\(String(format: "%.2f", viewModel.meshExtractionTime))s")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                ForEach(Array(viewModel.extractedMeshes.enumerated()), id: \.offset) { index, mesh in
                    VStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.purple.opacity(0.2))
                            .frame(height: 80)
                            .overlay(
                                VStack {
                                    Image(systemName: "cube.transparent")
                                        .font(.title2)
                                        .foregroundColor(.purple)
                                    Text("Mesh \(index + 1)")
                                        .font(.caption2)
                                        .foregroundColor(.purple)
                                }
                            )

                        VStack(spacing: 8) {
                            Text("Vertices: \(mesh.vertices.dim(0))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Faces: \(mesh.faces.dim(0))")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            VStack(spacing: 6) {
                                Button(action: {
                                    do {
                                        let savedFile = try viewModel.saveMeshToLibrary(at: index)
                                        print("Mesh saved to library: \(savedFile.name)")

                                        // Show success feedback
                                        savedFileName = savedFile.name
                                        withAnimation {
                                            showingSaveSuccess = true
                                        }

                                        // Hide after 2 seconds
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                            withAnimation {
                                                showingSaveSuccess = false
                                            }
                                        }
                                    } catch {
                                        print("Failed to save mesh: \(error)")
                                    }
                                }) {
                                    HStack {
                                        Image(systemName: "square.and.arrow.down")
                                            .font(.caption)
                                        Text("Save to Library")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                    }
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                                    .background(Color.green)
                                    .cornerRadius(8)
                                }

                                Button(action: {
                                    if let url = viewModel.exportMesh(at: index) {
                                        // Show success or share sheet
                                        print("Mesh exported to: \(url)")
                                    }
                                }) {
                                    HStack {
                                        Image(systemName: "square.and.arrow.up")
                                            .font(.caption)
                                        Text("Export")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                    }
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                                    .background(Color.blue)
                                    .cornerRadius(8)
                                }
                            }
                        }
                        .padding(.horizontal, 4)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }
}

#Preview {
    TSRTestView()
}
