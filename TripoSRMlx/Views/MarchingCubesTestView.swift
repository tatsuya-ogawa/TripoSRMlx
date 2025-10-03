//
//  MarchingCubesTestView.swift
//  TripoSRMlx
//
//  Test view for marching cubes implementation with synthetic data
//

import SwiftUI
import MLX

struct MarchingCubesTestView: View {
    @StateObject private var viewModel = MarchingCubesTestViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                headerSection

                // Controls
                controlsSection

                // Progress and Status
                progressSection

                // Error Display
                if let errorMessage = viewModel.errorMessage {
                    errorSection(message: errorMessage)
                }

                // Results
                if !viewModel.extractedMeshes.isEmpty {
                    resultsSection
                }

                Spacer(minLength: 50)
            }
            .padding()
        }
        .navigationTitle("Marching Cubes Test")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - View Components

    private var headerSection: some View {
        VStack(spacing: 10) {
            Image(systemName: "cube.transparent.fill")
                .font(.system(size: 60))
                .foregroundColor(.purple)

            Text("Marching Cubes Test")
                .font(.title2)
                .fontWeight(.bold)

            Text("Test marching cubes with synthetic sphere data")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
    }

    private var controlsSection: some View {
        VStack(spacing: 16) {
            // Resolution Control
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Resolution:")
                        .font(.subheadline)
                    Spacer()
                    Text("\(viewModel.resolution)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(
                    value: Binding(
                        get: { Double(viewModel.resolution) },
                        set: { viewModel.resolution = Int($0) }
                    ),
                    in: 32...128,
                    step: 16
                )
            }

            // Sphere Radius Control
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Sphere Radius:")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.sphereRadius))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(value: $viewModel.sphereRadius, in: 0.2...0.8, step: 0.05)
            }

            // Threshold Control
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Threshold:")
                        .font(.subheadline)
                    Spacer()
                    Text(String(format: "%.1f", viewModel.threshold))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Slider(value: $viewModel.threshold, in: 0.0...50.0, step: 1.0)
            }

            // Test Buttons
            HStack(spacing: 16) {
                Button(action: {
                    Task {
                        await viewModel.testMetalVersion()
                    }
                }) {
                    HStack {
                        if viewModel.isLoading {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text("Test Metal")
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
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(16)
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

    private var resultsSection: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Test Results")
                    .font(.headline)

                Spacer()

                if viewModel.processingTime > 0 {
                    Text("\(String(format: "%.3f", viewModel.processingTime))s")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 1), spacing: 12) {
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
                                    Text("Generated Mesh")
                                        .font(.caption2)
                                        .foregroundColor(.purple)
                                }
                            )

                        VStack(spacing: 4) {
                            Text("Vertices: \(mesh.vertices.dim(0))")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text("Faces: \(mesh.faces.dim(0))")
                                .font(.caption2)
                                .foregroundColor(.secondary)

                            Button(action: {
                                if let url = viewModel.exportMesh(at: index) {
                                    print("Mesh exported to: \(url)")
                                }
                            }) {
                                Text("Export OBJ")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.1))
                                    .cornerRadius(6)
                            }
                        }
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
    NavigationView {
        MarchingCubesTestView()
    }
}
