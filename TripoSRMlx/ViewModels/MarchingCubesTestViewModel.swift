//
//  MarchingCubesTestViewModel.swift
//  TripoSRMlx
//
//  ViewModel for testing marching cubes with synthetic sphere data
//

import SwiftUI
import MLX
import Foundation
import Combine

@MainActor
class MarchingCubesTestViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var currentStatus = "Ready to test marching cubes"
    @Published var errorMessage: String?
    @Published var extractedMeshes: [TriMesh] = []
    @Published var processingTime: TimeInterval = 0
    @Published var progress: Double = 0.0

    // Test Parameters
    @Published var resolution: Int = 64
    @Published var sphereRadius: Float = 0.4
    @Published var threshold: Float = 25.0

    init() {}

    /// Generate a sphere-shaped density field
    func generateSphereDensityField() -> MLXArray {
        currentStatus = "Generating sphere density field..."
        progress = 0.1

        // Create grid coordinates
        let coords = linspace(-1.0 as Float, 1.0 as Float, count: resolution)

        // Create 3D coordinate grids using expand and broadcast
        let x = coords.expandedDimensions(axes: [1, 2])  // [resolution, 1, 1]
        let y = coords.expandedDimensions(axes: [0, 2])  // [1, resolution, 1]
        let z = coords.expandedDimensions(axes: [0, 1])  // [1, 1, resolution]

        // Broadcast to full 3D grid [resolution, resolution, resolution]
        let xGrid = broadcast(x, to: [resolution, resolution, resolution])
        let yGrid = broadcast(y, to: [resolution, resolution, resolution])
        let zGrid = broadcast(z, to: [resolution, resolution, resolution])

        progress = 0.3

        // Calculate distance from center: sqrt(x² + y² + z²)
        let distanceSquared = xGrid * xGrid + yGrid * yGrid + zGrid * zGrid
        let distance = sqrt(distanceSquared)

        progress = 0.5

        // Create sphere density field
        // Higher values inside sphere, lower values outside
        let sphereRadiusValue = MLXArray(sphereRadius)
        let density = (sphereRadiusValue - distance) * 100.0  // Scale for better visualization

        progress = 0.7
        return density
    }

    /// Generate more complex shapes for testing
    func generateComplexDensityField() -> MLXArray {
        currentStatus = "Generating complex density field..."

        // Create multiple spheres or other shapes
        let coords = linspace(-1.0, 1.0, count: resolution)

        let x = coords.expandedDimensions(axes: [1, 2])
        let y = coords.expandedDimensions(axes: [0, 2])
        let z = coords.expandedDimensions(axes: [0, 1])

        let xGrid = broadcast(x, to: [resolution, resolution, resolution])
        let yGrid = broadcast(y, to: [resolution, resolution, resolution])
        let zGrid = broadcast(z, to: [resolution, resolution, resolution])

        // Create two spheres
        let center1 = MLXArray([-0.3, 0.0, 0.0] as [Float])
        let center2 = MLXArray([0.3, 0.0, 0.0] as [Float])

        let dist1 = sqrt((xGrid - center1[0]) * (xGrid - center1[0]) +
                        (yGrid - center1[1]) * (yGrid - center1[1]) +
                        (zGrid - center1[2]) * (zGrid - center1[2]))

        let dist2 = sqrt((xGrid - center2[0]) * (xGrid - center2[0]) +
                        (yGrid - center2[1]) * (yGrid - center2[1]) +
                        (zGrid - center2[2]) * (zGrid - center2[2]))

        let radius = MLXArray(sphereRadius)
        let density1 = (radius - dist1) * 100.0
        let density2 = (radius - dist2) * 100.0

        // Combine densities (max operation simulates union)
        return maximum(density1, density2)
    }

    /// Test Metal version of marching cubes
    func testMetalVersion() async {
        isLoading = true
        errorMessage = nil
        extractedMeshes = []
        progress = 0.0

        Task.detached {
            do {
                let startTime = Date()

                // Generate test data
                await MainActor.run {
                    self.currentStatus = "Generating test data..."
                    self.progress = 0.1
                }

                let densityField = await self.generateSphereDensityField()

                await MainActor.run {
                    self.currentStatus = "Creating marching cubes helper..."
                    self.progress = 0.8
                }

                // Create marching cubes helper and extract mesh
                let helper = MarchingCubeHelper(resolution: await self.resolution)

                await MainActor.run {
                    self.currentStatus = "Extracting mesh (Metal)..."
                    self.progress = 0.9
                }

                let (vertices, faces) = helper.extractIsosurface(densityField)

                let endTime = Date()
                let processingTime = endTime.timeIntervalSince(startTime)

                await MainActor.run {
                    if vertices.size > 0 {
                        let mesh = TriMesh(vertices: vertices, faces: faces, vertexColors: nil)
                        self.extractedMeshes = [mesh]
                        self.currentStatus = "Metal extraction completed successfully!"
                    } else {
                        self.currentStatus = "Metal extraction completed (empty mesh)"
                    }

                    self.processingTime = processingTime
                    self.progress = 1.0
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Metal test failed: \(error.localizedDescription)"
                    self.currentStatus = "Error: \(error.localizedDescription)"
                    self.isLoading = false
                }
                print("❌ Metal marching cubes test error: \(error)")
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
        let filename = "test_mesh_\(timestamp).\(format)"
        let fileURL = documentsPath.appendingPathComponent(filename)

        do {
            try extractedMeshes[index].export(to: fileURL, format: format)
            return fileURL
        } catch {
            errorMessage = "Export failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Reset test state
    func reset() {
        isLoading = false
        currentStatus = "Ready to test marching cubes"
        errorMessage = nil
        extractedMeshes = []
        processingTime = 0
        progress = 0.0
    }
}
