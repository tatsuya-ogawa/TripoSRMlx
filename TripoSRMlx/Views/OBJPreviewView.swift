//
//  OBJPreviewView.swift
//  TripoSRMlx
//
//  Preview view for saved OBJ files
//

import SwiftUI
import SceneKit

struct OBJPreviewView: View {
    let file: SavedOBJFile
    let viewModel: SavedModelsViewModel

    @Environment(\.dismiss) private var dismiss
    @State private var objContent: String = ""
    @State private var showingShareSheet = false
    @State private var showingARView = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // 3D Preview Section
                SceneKitPreview(objContent: objContent)
                    .frame(maxHeight: .infinity)
                    .background(Color.black.opacity(0.05))

                Divider()

                // Info Section
                VStack(spacing: 16) {
                    VStack(spacing: 8) {
                        Text(file.name)
                            .font(.title2)
                            .fontWeight(.bold)

                        Text("Created: \(formatDate(file.createdAt))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    if let error = errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }

                    // Action Buttons
                    VStack(spacing: 12) {
                        Button(action: {
                            showingARView = true
                        }) {
                            Label("Place in AR", systemImage: "arkit")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.purple)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        Button(action: {
                            showingShareSheet = true
                        }) {
                            Label("Share", systemImage: "square.and.arrow.up")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 20)
            }
            .navigationTitle("Model Preview")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                ShareSheet(items: [viewModel.getFileURL(file)])
            }
            .fullScreenCover(isPresented: $showingARView) {
                ARModelPlacementView(file: file, objContent: objContent)
            }
            .onAppear {
                loadOBJContent()
            }
        }
    }

    // MARK: - Private Methods

    private func loadOBJContent() {
        do {
            objContent = try viewModel.getOBJContent(file)
        } catch {
            errorMessage = "Failed to load OBJ file: \(error.localizedDescription)"
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - SceneKit Preview Component

struct SceneKitPreview: View {
    let objContent: String

    var body: some View {
        if objContent.isEmpty {
            VStack {
                ProgressView()
                    .scaleEffect(1.5)
                Text("Loading 3D model...")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.top, 8)
            }
        } else {
            SceneView(
                scene: createScene(),
                pointOfView: nil,
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
        }
    }

    private func createScene() -> SCNScene {
        let scene = SCNScene()

        // Try to load OBJ file
        if let objData = objContent.data(using: .utf8) {
            // Save to temporary file for SceneKit to load
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("preview.obj")
            try? objData.write(to: tempURL)

            if let objNode = loadOBJNode(from: tempURL) {
                scene.rootNode.addChildNode(objNode)

                // Center and scale the model
                centerAndScaleNode(objNode)
            }
        }

        // Add camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 3)
        scene.rootNode.addChildNode(cameraNode)

        return scene
    }

    private func loadOBJNode(from url: URL) -> SCNNode? {
        // Parse OBJ with vertex colors
        if let (geometry, hasColors) = parseOBJWithColors(url: url) {
            let node = SCNNode(geometry: geometry)

            // Apply material
            let material = SCNMaterial()
            material.lightingModel = .blinn

            if hasColors {
                material.diffuse.contents = UIColor.white
            } else {
                material.diffuse.contents = UIColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
            }
            material.isDoubleSided = true

            geometry.materials = [material]
            return node
        }

        // Fallback to default SceneKit loading
        guard let scene = try? SCNScene(url: url, options: nil) else {
            return nil
        }

        let node = SCNNode()
        for child in scene.rootNode.childNodes {
            node.addChildNode(child)
        }
        return node
    }

    /// Parse OBJ file with vertex colors
    private func parseOBJWithColors(url: URL) -> (SCNGeometry, Bool)? {
        guard let objContent = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }

        var vertices: [SCNVector3] = []
        var colors: [SCNVector3] = []
        var indices: [Int32] = []
        var hasColors = false

        let lines = objContent.components(separatedBy: .newlines)

        for line in lines {
            let components = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard let first = components.first else { continue }

            if first == "v" {
                // Vertex line: v x y z [r g b]
                if components.count >= 4 {
                    let x = Float(components[1]) ?? 0
                    let y = Float(components[2]) ?? 0
                    let z = Float(components[3]) ?? 0
                    vertices.append(SCNVector3(x, y, z))

                    if components.count >= 7 {
                        // Has color
                        let r = Float(components[4]) ?? 0.8
                        let g = Float(components[5]) ?? 0.8
                        let b = Float(components[6]) ?? 0.8
                        colors.append(SCNVector3(r, g, b))
                        hasColors = true
                    } else {
                        colors.append(SCNVector3(0.8, 0.8, 0.8))
                    }
                }
            } else if first == "f" {
                // Face line: f v1 v2 v3
                if components.count >= 4 {
                    for i in 1...3 {
                        let indexStr = components[i].components(separatedBy: "/")[0]
                        if let index = Int32(indexStr) {
                            indices.append(index - 1) // OBJ uses 1-based indexing
                        }
                    }
                }
            }
        }

        guard !vertices.isEmpty, !indices.isEmpty else {
            return nil
        }

        // Create geometry sources
        let vertexData = Data(bytes: vertices, count: vertices.count * MemoryLayout<SCNVector3>.size)
        let vertexSource = SCNGeometrySource(data: vertexData,
                                             semantic: .vertex,
                                             vectorCount: vertices.count,
                                             usesFloatComponents: true,
                                             componentsPerVector: 3,
                                             bytesPerComponent: MemoryLayout<Float>.size,
                                             dataOffset: 0,
                                             dataStride: MemoryLayout<SCNVector3>.size)

        var sources = [vertexSource]

        // Add color source if we have colors
        if hasColors {
            let colorData = Data(bytes: colors, count: colors.count * MemoryLayout<SCNVector3>.size)
            let colorSource = SCNGeometrySource(data: colorData,
                                                semantic: .color,
                                                vectorCount: colors.count,
                                                usesFloatComponents: true,
                                                componentsPerVector: 3,
                                                bytesPerComponent: MemoryLayout<Float>.size,
                                                dataOffset: 0,
                                                dataStride: MemoryLayout<SCNVector3>.size)
            sources.append(colorSource)
        }

        // Create geometry element
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(data: indexData,
                                        primitiveType: .triangles,
                                        primitiveCount: indices.count / 3,
                                        bytesPerIndex: MemoryLayout<Int32>.size)

        // Create geometry
        let geometry = SCNGeometry(sources: sources, elements: [element])

        return (geometry, hasColors)
    }

    private func centerAndScaleNode(_ node: SCNNode) {
        let (min, max) = node.boundingBox
        let center = SCNVector3(
            x: (min.x + max.x) / 2,
            y: (min.y + max.y) / 2,
            z: (min.z + max.z) / 2
        )

        // Center the model
        node.position = SCNVector3(-center.x, -center.y, -center.z)

        // Scale to fit
        let size = SCNVector3(
            x: max.x - min.x,
            y: max.y - min.y,
            z: max.z - min.z
        )
        let maxDimension = Swift.max(size.x, size.y, size.z)
        let scale = 2.0 / maxDimension
        node.scale = SCNVector3(scale, scale, scale)
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

#Preview {
    OBJPreviewView(
        file: SavedOBJFile(
            name: "Sample Model",
            fileName: "sample.obj"
        ),
        viewModel: SavedModelsViewModel()
    )
}
