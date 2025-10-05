//
//  ARModelPlacementView.swift
//  TripoSRMlx
//
//  AR view for placing OBJ models in the real world
//

import SwiftUI
import ARKit
import SceneKit

struct ARModelPlacementView: View {
    let file: SavedOBJFile
    let objContent: String

    @Environment(\.dismiss) private var dismiss
    @State private var arSupported = false
    @State private var showingPlacementInstructions = true

    var body: some View {
        ZStack {
            if arSupported {
                ARViewContainer(objContent: objContent, file: file)
                    .edgesIgnoringSafeArea(.all)

                // Overlay UI
                VStack {
                    // Top bar
                    HStack {
                        Button(action: {
                            dismiss()
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title)
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }

                        Spacer()

                        Text(file.name)
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(20)

                        Spacer()

                        // Placeholder for symmetry
                        Color.clear
                            .frame(width: 60, height: 60)
                    }
                    .padding()

                    Spacer()

                    // Instructions
                    if showingPlacementInstructions {
                        VStack(spacing: 12) {
                            Text("Tap on a surface to place the model")
                                .font(.subheadline)
                                .foregroundColor(.white)

                            Button(action: {
                                showingPlacementInstructions = false
                            }) {
                                Text("Got it")
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(Color.blue)
                                    .cornerRadius(20)
                            }
                        }
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(16)
                        .padding(.bottom, 50)
                    }
                }
            } else {
                // AR not supported
                VStack(spacing: 20) {
                    Image(systemName: "arkit")
                        .font(.system(size: 80))
                        .foregroundColor(.gray)

                    Text("AR Not Supported")
                        .font(.title2)
                        .fontWeight(.semibold)

                    Text("This device does not support ARKit")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Button(action: {
                        dismiss()
                    }) {
                        Text("Close")
                            .padding(.horizontal, 32)
                            .padding(.vertical, 12)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
            }
        }
        .onAppear {
            checkARSupport()
        }
    }

    private func checkARSupport() {
        arSupported = ARWorldTrackingConfiguration.isSupported
    }
}

// MARK: - ARView Container

struct ARViewContainer: UIViewRepresentable {
    let objContent: String
    let file: SavedOBJFile

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        arView.delegate = context.coordinator

        // Configure AR session
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }

        arView.session.run(configuration)

        // Enable default lighting
        arView.autoenablesDefaultLighting = true
        arView.automaticallyUpdatesLighting = true

        // Add tap gesture
        let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        arView.addGestureRecognizer(tapGesture)

        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(objContent: objContent, file: file)
    }

    class Coordinator: NSObject, ARSCNViewDelegate {
        let objContent: String
        let file: SavedOBJFile
        var placedNode: SCNNode?

        init(objContent: String, file: SavedOBJFile) {
            self.objContent = objContent
            self.file = file
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let arView = gesture.view as? ARSCNView else { return }
            let location = gesture.location(in: arView)

            // Hit test for planes
            let hitTestResults = arView.hitTest(location, types: [.existingPlaneUsingExtent, .estimatedHorizontalPlane])

            if let hitResult = hitTestResults.first {
                placeModel(at: hitResult, in: arView)
            }
        }

        func placeModel(at hitResult: ARHitTestResult, in arView: ARSCNView) {
            // Remove previous model if exists
            placedNode?.removeFromParentNode()

            // Load OBJ model
            guard let modelNode = loadOBJModel() else {
                print("Failed to load OBJ model")
                return
            }

            // Position the model at the hit location
            let transform = hitResult.worldTransform
            let position = SCNVector3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            modelNode.position = position

            // Add to scene
            arView.scene.rootNode.addChildNode(modelNode)
            placedNode = modelNode

            // Add animation
            modelNode.opacity = 0
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 0.5
            modelNode.opacity = 1
            SCNTransaction.commit()
        }

        func loadOBJModel() -> SCNNode? {
            // Save OBJ content to temporary file
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("ar_model_\(file.id.uuidString).obj")

            do {
                try objContent.write(to: tempURL, atomically: true, encoding: .utf8)
            } catch {
                print("Failed to write OBJ file: \(error)")
                return nil
            }

            // Parse OBJ with vertex colors
            let containerNode: SCNNode
            if let (geometry, hasColors) = parseOBJWithColors(url: tempURL) {
                containerNode = SCNNode(geometry: geometry)

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
            } else {
                // Fallback to default SceneKit loading
                guard let scene = try? SCNScene(url: tempURL, options: nil) else {
                    print("Failed to load SCNScene from OBJ")
                    return nil
                }

                containerNode = SCNNode()
                for child in scene.rootNode.childNodes {
                    containerNode.addChildNode(child)
                }
            }

            // Center and scale the model
            let (min, max) = containerNode.boundingBox
            let center = SCNVector3(
                x: (min.x + max.x) / 2,
                y: (min.y + max.y) / 2,
                z: (min.z + max.z) / 2
            )

            // Center the model
            containerNode.position = SCNVector3(-center.x, -center.y, -center.z)

            // Scale to reasonable size (about 20cm max dimension)
            let size = SCNVector3(
                x: max.x - min.x,
                y: max.y - min.y,
                z: max.z - min.z
            )
            let maxDimension = Swift.max(size.x, size.y, size.z)
            let desiredSize: Float = 0.2 // 20cm
            let scale = desiredSize / maxDimension

            // Wrapper node for proper positioning
            let wrapperNode = SCNNode()
            wrapperNode.addChildNode(containerNode)
            wrapperNode.scale = SCNVector3(scale, scale, scale)

            // Add physics body with collider
            if let geometry = containerNode.childNodes.first?.geometry {
                addPhysicsBody(to: wrapperNode, geometry: geometry)
            }

            return wrapperNode
        }

        func addPhysicsBody(to node: SCNNode, geometry: SCNGeometry) {
            // Options for complex shapes (concave polygons)
            let options: [SCNPhysicsShape.Option: Any] = [
                .type: SCNPhysicsShape.ShapeType.concavePolyhedron,
                .keepAsCompound: true
            ]

            let shape = SCNPhysicsShape(geometry: geometry, options: options)
            let body = SCNPhysicsBody(type: .static, shape: shape)

            // Optional: configure physics properties
            body.restitution = 0.5 // Bounciness
            body.friction = 0.8

            node.physicsBody = body
        }

        /// Parse OBJ file with vertex colors
        func parseOBJWithColors(url: URL) -> (SCNGeometry, Bool)? {
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

        // MARK: - ARSCNViewDelegate

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            // Handle plane detection if needed
        }
    }
}

#Preview {
    ARModelPlacementView(
        file: SavedOBJFile(
            name: "Sample Model",
            fileName: "sample.obj"
        ),
        objContent: ""
    )
}
