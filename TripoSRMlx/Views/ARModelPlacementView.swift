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
    @StateObject private var viewModel = SavedModelsViewModel()
    @State private var selectedFile: SavedOBJFile
    @State private var selectedOBJContent: String
    @State private var arSupported = false
    @State private var showingPlacementInstructions = true
    @State private var showingModelList = false
    @State private var maxObjects = 50
    @State private var currentObjectCount = 0
    @State private var showMesh = false
    @State private var horizontalOffset: Float = 0.0 // Left/Right: -1.0 to 1.0
    @State private var depthOffset: Float = 1.5 // Forward/Back: 0.5 to 3.0
    @State private var modelScale: Float = 1.0 // Scale: 0.5 to 3.0

    init(file: SavedOBJFile, objContent: String) {
        self.file = file
        self.objContent = objContent
        _selectedFile = State(initialValue: file)
        _selectedOBJContent = State(initialValue: objContent)
    }

    var body: some View {
        ZStack {
            if arSupported {
                ARCraneViewContainer(
                    objContent: selectedOBJContent,
                    file: selectedFile,
                    maxObjects: maxObjects,
                    showMesh: showMesh,
                    horizontalOffset: horizontalOffset,
                    depthOffset: depthOffset,
                    modelScale: modelScale,
                    currentObjectCount: $currentObjectCount
                )
                .edgesIgnoringSafeArea(.all)
                .id(selectedFile.id) // Force recreate when file changes

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

                        VStack(spacing: 4) {
                            Text(selectedFile.name)
                                .font(.headline)
                                .foregroundColor(.white)

                            // Ammo counter
                            Text("\(currentObjectCount) / \(maxObjects)")
                                .font(.caption)
                                .foregroundColor(currentObjectCount >= maxObjects ? .red : .white)
                                .fontWeight(.bold)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(20)

                        Spacer()

                        VStack(spacing: 8) {
                            // Model list toggle button
                            Button(action: {
                                withAnimation {
                                    showingModelList.toggle()
                                }
                            }) {
                                Image(systemName: showingModelList ? "list.bullet.circle.fill" : "list.bullet.circle")
                                    .font(.title)
                                    .foregroundColor(.white)
                                    .padding()
                                    .background(Color.black.opacity(0.5))
                                    .clipShape(Circle())
                            }

                            // Mesh toggle button
                            Button(action: {
                                showMesh.toggle()
                            }) {
                                Image(systemName: showMesh ? "cube.fill" : "cube")
                                    .font(.title)
                                    .foregroundColor(showMesh ? .green : .white)
                                    .padding()
                                    .background(Color.black.opacity(0.5))
                                    .clipShape(Circle())
                            }
                        }
                    }
                    .padding()

                    Spacer()

                    VStack(spacing: 16) {
                        // Instructions at top
                        if showingPlacementInstructions {
                            VStack(spacing: 12) {
                                Text("Use controls to position, then drop")
                                    .font(.subheadline)
                                    .foregroundColor(.white)

                                Text("Max \(maxObjects) objects")
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.8))

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
                        }

                        Spacer()

                        HStack(alignment: .bottom, spacing: 16) {
                            // Control buttons on bottom left
                            VStack(spacing: 12) {
                                // Depth controls (forward/back)
                                HStack(spacing: 12) {
                                    Button(action: {
                                        depthOffset = min(3.0, depthOffset + 0.2)
                                    }) {
                                        Image(systemName: "arrow.up")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }

                                    Button(action: {
                                        depthOffset = max(0.5, depthOffset - 0.2)
                                    }) {
                                        Image(systemName: "arrow.down")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }
                                }

                                // Horizontal controls (left/right)
                                HStack(spacing: 12) {
                                    Button(action: {
                                        horizontalOffset = max(-1.0, horizontalOffset - 0.1)
                                    }) {
                                        Image(systemName: "arrow.left")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }

                                    Button(action: {
                                        horizontalOffset = max(-1.0, min(1.0, 0))
                                        depthOffset = 1.5
                                    }) {
                                        Image(systemName: "arrow.counterclockwise")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }

                                    Button(action: {
                                        horizontalOffset = min(1.0, horizontalOffset + 0.1)
                                    }) {
                                        Image(systemName: "arrow.right")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }
                                }

                                // Scale controls
                                HStack(spacing: 12) {
                                    Button(action: {
                                        modelScale = max(0.5, modelScale - 0.2)
                                    }) {
                                        Image(systemName: "minus.magnifyingglass")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }

                                    Text(String(format: "%.1fx", modelScale))
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .frame(width: 62)

                                    Button(action: {
                                        modelScale = min(3.0, modelScale + 0.2)
                                    }) {
                                        Image(systemName: "plus.magnifyingglass")
                                            .font(.title2)
                                            .foregroundColor(.white)
                                            .frame(width: 50, height: 50)
                                            .background(Color.black.opacity(0.6))
                                            .cornerRadius(10)
                                    }
                                }

                                // Drop button
                                Button(action: {
                                    // Trigger drop via notification
                                    NotificationCenter.default.post(name: NSNotification.Name("DropObject"), object: nil)
                                }) {
                                    Text("DROP")
                                        .font(.headline)
                                        .foregroundColor(.white)
                                        .frame(width: 162, height: 50)
                                        .background(currentObjectCount >= maxObjects ? Color.red : Color.green)
                                        .cornerRadius(10)
                                }
                                .disabled(currentObjectCount >= maxObjects)
                            }
                            .padding()

                            Spacer()

                            // Model list on bottom right
                            if showingModelList {
                                ModelListView(
                                    models: viewModel.savedFiles,
                                    selectedFile: $selectedFile,
                                    selectedOBJContent: $selectedOBJContent,
                                    onClose: {
                                        withAnimation {
                                            showingModelList = false
                                        }
                                    }
                                )
                                .transition(.move(edge: .trailing).combined(with: .opacity))
                            }
                        }
                        .padding()
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
            viewModel.loadFiles()
        }
    }

    private func checkARSupport() {
        arSupported = ARWorldTrackingConfiguration.isSupported
    }
}

// MARK: - Model List View

struct ModelListView: View {
    let models: [SavedOBJFile]
    @Binding var selectedFile: SavedOBJFile
    @Binding var selectedOBJContent: String
    let onClose: () -> Void

    private let fileManager = OBJFileManager()

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Models")
                    .font(.headline)
                    .foregroundColor(.white)

                Spacer()

                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            .padding()
            .background(Color.black.opacity(0.8))

            // Model list
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(models) { model in
                        ModelListItemView(
                            model: model,
                            isSelected: model.id == selectedFile.id,
                            onTap: {
                                selectModel(model)
                            }
                        )
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: 300)
            .background(Color.black.opacity(0.7))
        }
        .frame(width: 200)
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.5), radius: 10, x: 0, y: 5)
    }

    private func selectModel(_ model: SavedOBJFile) {
        // Load OBJ content
        let fileURL = fileManager.getFileURL(model)
        if let content = try? String(contentsOf: fileURL, encoding: .utf8) {
            selectedFile = model
            selectedOBJContent = content
            onClose()
        }
    }
}

struct ModelListItemView: View {
    let model: SavedOBJFile
    let isSelected: Bool
    let onTap: () -> Void

    @State private var thumbnail: UIImage?

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 8) {
                // Thumbnail
                if let thumbnail = thumbnail {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 40, height: 40)
                        .cornerRadius(6)
                } else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 40, height: 40)
                        .cornerRadius(6)
                        .overlay(
                            ProgressView()
                                .scaleEffect(0.7)
                        )
                }

                // Name
                Text(model.name)
                    .font(.caption)
                    .foregroundColor(.white)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)

                // Selection indicator
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.blue)
                        .font(.caption)
                }
            }
            .padding(8)
            .background(isSelected ? Color.blue.opacity(0.3) : Color.clear)
            .cornerRadius(8)
        }
        .buttonStyle(PlainButtonStyle())
        .task {
            await loadThumbnail()
        }
    }

    private func loadThumbnail() async {
        let fileManager = OBJFileManager()
        let fileURL = fileManager.getFileURL(model)
        thumbnail = await ThumbnailGenerator.shared.generateThumbnail(for: fileURL, file: model)
    }
}

// MARK: - AR Crane View Container

struct ARCraneViewContainer: UIViewRepresentable {
    let objContent: String
    let file: SavedOBJFile
    let maxObjects: Int
    let showMesh: Bool
    let horizontalOffset: Float
    let depthOffset: Float
    let modelScale: Float
    @Binding var currentObjectCount: Int

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

        // Enable default lighting and physics
        arView.autoenablesDefaultLighting = true
        arView.automaticallyUpdatesLighting = true
        arView.scene.physicsWorld.gravity = SCNVector3(0, -9.8, 0)

        context.coordinator.arView = arView
        context.coordinator.attachPreviewIfNeeded()

        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.updateMeshVisibility(showMesh: showMesh)
        context.coordinator.updatePreviewPosition(horizontal: horizontalOffset, depth: depthOffset)
        context.coordinator.updateModelScale(scale: modelScale)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(
            objContent: objContent,
            file: file,
            maxObjects: maxObjects,
            currentObjectCount: $currentObjectCount
        )
    }

    class Coordinator: NSObject, ARSCNViewDelegate {
        let objContent: String
        let file: SavedOBJFile
        let maxObjects: Int
        var droppedObjects: [SCNNode] = []
        weak var arView: ARSCNView?
        let maxDistance: Float = 5.0
        @Binding var currentObjectCount: Int
        var meshAnchors: [ARMeshAnchor: SCNNode] = [:]
        var previewNode: SCNNode?
        var dropObserver: NSObjectProtocol?
        var horizontalOffset: Float = 0.0
        var depthOffset: Float = 1.5
        var modelScale: Float = 1.0
        var cachedModelTemplate: SCNNode?
        var cachedObjHash: Int?
        var baseModelScale: Float = 1.0 // The scale needed to normalize model to 20cm

        init(objContent: String, file: SavedOBJFile, maxObjects: Int, currentObjectCount: Binding<Int>) {
            self.objContent = objContent
            self.file = file
            self.maxObjects = maxObjects
            self._currentObjectCount = currentObjectCount
            super.init()

            // Listen for drop notification
            dropObserver = NotificationCenter.default.addObserver(
                forName: NSNotification.Name("DropObject"),
                object: nil,
                queue: .main
            ) { [weak self] _ in
                self?.dropObject()
            }

            // Create initial preview
            createPreview()
        }

        deinit {
            if let observer = dropObserver {
                NotificationCenter.default.removeObserver(observer)
            }
        }

        func updateMeshVisibility(showMesh: Bool) {
            for (_, node) in meshAnchors {
                node.isHidden = !showMesh
            }
        }

        func updatePreviewPosition(horizontal: Float, depth: Float) {
            // Store offset values for continuous camera tracking
            self.horizontalOffset = horizontal
            self.depthOffset = depth
        }

        func updateModelScale(scale: Float) {
            self.modelScale = scale
            if let preview = previewNode {
                // Apply combined scale: base normalization scale * user scale
                let finalScale = baseModelScale * scale
                preview.scale = SCNVector3(finalScale, finalScale, finalScale)

                // Update physics body to match the new scale
                updatePhysicsBodyForScale(node: preview, scale: CGFloat(finalScale))
            }
        }

        func updatePhysicsBodyForScale(node: SCNNode, scale: CGFloat) {
            // Recreate physics body with the new scale
            let options: [SCNPhysicsShape.Option: Any] = [
                .type: SCNPhysicsShape.ShapeType.convexHull,
                .scale: SCNVector3(scale, scale, scale)
            ]

            var shapeOptions = options
            shapeOptions[.collisionMargin] = 0.001
            let shape = SCNPhysicsShape(node: node, options: shapeOptions)

            // Preserve physics body properties
            let isPreview = node.physicsBody?.type == .kinematic
            let body = SCNPhysicsBody(type: isPreview ? .kinematic : .dynamic, shape: shape)

            // Configure physics properties
            body.mass = 0.5 * scale * scale * scale // Mass scales with volume
            body.restitution = 0.3
            body.friction = 0.8
            body.damping = 0.1
            body.angularDamping = 0.2

            body.categoryBitMask = 1
            body.collisionBitMask = isPreview ? 0 : (1 | 2)
            body.contactTestBitMask = isPreview ? 0 : 2
            body.isAffectedByGravity = !isPreview

            node.physicsBody = body
        }

        func attachPreviewIfNeeded() {
            guard let arView = arView,
                  let preview = previewNode,
                  preview.parent == nil else { return }

            arView.scene.rootNode.addChildNode(preview)
        }

        func updatePreviewToFollowCamera() {
            guard let arView = arView,
                  let currentFrame = arView.session.currentFrame else { return }

            attachPreviewIfNeeded()
            guard let preview = previewNode else { return }

            let camera = currentFrame.camera
            let transform = camera.transform

            // Get camera vectors
            // In ARKit: columns.2 is the -Z axis (view direction)
            let forward = -simd_float3(
                transform.columns.2.x,
                transform.columns.2.y,
                transform.columns.2.z
            )

            let right = simd_float3(
                transform.columns.0.x,
                transform.columns.0.y,
                transform.columns.0.z
            )

            let cameraPosition = simd_float3(
                transform.columns.3.x,
                transform.columns.3.y,
                transform.columns.3.z
            )

            // Always position in front of camera with offsets
            let newPosition = cameraPosition + forward * depthOffset + right * horizontalOffset

            preview.position = SCNVector3(newPosition.x, newPosition.y, newPosition.z)
        }

        func createPreview() {
            guard let modelNode = loadOBJModel() else { return }

            // Apply initial scale (base normalization + user scale)
            let finalScale = baseModelScale * modelScale
            modelNode.scale = SCNVector3(finalScale, finalScale, finalScale)

            // Disable collisions and gravity while previewing
            applyPhysicsMode(.preview, to: modelNode)

            // Update physics for the scaled model
            updatePhysicsBodyForScale(node: modelNode, scale: CGFloat(finalScale))

            // Add semi-transparent material to indicate it's a preview
            applyPreviewOpacity(to: modelNode)

            previewNode = modelNode
            attachPreviewIfNeeded()
            updatePreviewToFollowCamera()
        }

        private func applyPreviewOpacity(to node: SCNNode) {
            if let geometry = node.geometry {
                for material in geometry.materials {
                    material.transparency = 0.7
                }
            }

            node.enumerateChildNodes { child, _ in
                if let geometry = child.geometry {
                    for material in geometry.materials {
                        material.transparency = 0.7
                    }
                }
            }
        }

        func dropObject() {
            guard let preview = previewNode else { return }

            // Check if at limit
            if droppedObjects.count >= maxObjects {
                let oldest = droppedObjects.removeFirst()
                oldest.removeFromParentNode()
            }

            // Enable physics (turn gravity on)
            applyPhysicsMode(.dropped, to: preview)

            // Restore full opacity
            restoreOpacity(for: preview)

            // Preserve the current scale on the dropped object
            // (preview already has the scale applied from updateModelScale)

            // Add to dropped list
            droppedObjects.append(preview)
            updateObjectCount()

            // Start distance checking
            scheduleDistanceCheck(for: preview)

            // Create new preview
            previewNode = nil
            createPreview()
        }

        func scheduleDistanceCheck(for node: SCNNode) {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self, weak node] in
                guard let self = self, let node = node else { return }

                // Check if node still exists in scene
                guard node.parent != nil else { return }

                // Get current camera position
                guard let arView = self.arView,
                      let currentFrame = arView.session.currentFrame else { return }

                let currentCameraPosition = SCNVector3(
                    currentFrame.camera.transform.columns.3.x,
                    currentFrame.camera.transform.columns.3.y,
                    currentFrame.camera.transform.columns.3.z
                )

                // Calculate distance
                let distance = self.distance(from: node.position, to: currentCameraPosition)

                if distance > self.maxDistance {
                    // Remove node if too far
                    node.removeFromParentNode()
                    if let index = self.droppedObjects.firstIndex(of: node) {
                        self.droppedObjects.remove(at: index)
                        self.updateObjectCount()
                    }
                } else {
                    // Check again later
                    self.scheduleDistanceCheck(for: node)
                }
            }
        }

        private func restoreOpacity(for node: SCNNode) {
            if let geometry = node.geometry {
                for material in geometry.materials {
                    material.transparency = 1.0
                }
            }

            node.enumerateChildNodes { child, _ in
                if let geometry = child.geometry {
                    for material in geometry.materials {
                        material.transparency = 1.0
                    }
                }
            }
        }

        private enum PhysicsMode {
            case preview
            case dropped
        }

        private func applyPhysicsMode(_ mode: PhysicsMode, to node: SCNNode) {
            if let body = node.physicsBody {
                switch mode {
                case .preview:
                    body.type = .kinematic
                    body.isAffectedByGravity = false
                    body.clearAllForces()
                    body.velocity = SCNVector3Zero
                    body.angularVelocity = SCNVector4(0, 0, 0, 0)
                    body.collisionBitMask = 0
                    body.contactTestBitMask = 0
                case .dropped:
                    body.type = .dynamic
                    body.isAffectedByGravity = true
                    body.clearAllForces()
                    body.velocity = SCNVector3Zero
                    body.angularVelocity = SCNVector4(0, 0, 0, 0)
                    body.collisionBitMask = 1 | 2
                    body.contactTestBitMask = 2
                }

                if body.categoryBitMask == 0 {
                    body.categoryBitMask = 1
                }
            }

            node.enumerateChildNodes { child, _ in
                self.applyPhysicsMode(mode, to: child)
            }
        }

        func distance(from a: SCNVector3, to b: SCNVector3) -> Float {
            let dx = a.x - b.x
            let dy = a.y - b.y
            let dz = a.z - b.z
            return sqrt(dx*dx + dy*dy + dz*dz)
        }

        func updateObjectCount() {
            DispatchQueue.main.async {
                self.currentObjectCount = self.droppedObjects.count
            }
        }

        // MARK: - ARSCNViewDelegate

        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            // Update preview to follow camera every frame
            updatePreviewToFollowCamera()
        }

        func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }

            // Create mesh geometry
            let geometry = createGeometry(from: meshAnchor)

            // Create material for mesh visualization
            let material = SCNMaterial()
            material.fillMode = .fill
            material.diffuse.contents = UIColor.systemGreen.withAlphaComponent(0.18)
            material.emission.contents = UIColor.systemGreen.withAlphaComponent(0.05)
            material.isDoubleSided = true
            material.blendMode = .alpha

            geometry.materials = [material]

            // Create mesh node
            let meshNode = SCNNode(geometry: geometry)
            node.addChildNode(meshNode)

            // Add physics body for collision
            let shape = SCNPhysicsShape(geometry: geometry, options: [
                .type: SCNPhysicsShape.ShapeType.concavePolyhedron,
                .collisionMargin: 0.001
            ])
            let physicsBody = SCNPhysicsBody(type: .static, shape: shape)
            physicsBody.categoryBitMask = 2 // Mesh category
            node.physicsBody = physicsBody

            // Store reference
            meshAnchors[meshAnchor] = meshNode

            // Initially hidden
            meshNode.isHidden = true
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }

            // Update mesh geometry
            if let meshNode = meshAnchors[meshAnchor] {
                let geometry = createGeometry(from: meshAnchor)
                let material = SCNMaterial()
                material.fillMode = .fill
                material.diffuse.contents = UIColor.systemGreen.withAlphaComponent(0.18)
                material.emission.contents = UIColor.systemGreen.withAlphaComponent(0.05)
                material.isDoubleSided = true
                material.blendMode = .alpha
                geometry.materials = [material]
                meshNode.geometry = geometry

                // Update physics body
                let shape = SCNPhysicsShape(geometry: geometry, options: [
                    .type: SCNPhysicsShape.ShapeType.concavePolyhedron,
                    .collisionMargin: 0.001
                ])
                let physicsBody = SCNPhysicsBody(type: .static, shape: shape)
                physicsBody.categoryBitMask = 2
                node.physicsBody = physicsBody
            }
        }

        func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }
            meshAnchors.removeValue(forKey: meshAnchor)
        }

        func createGeometry(from meshAnchor: ARMeshAnchor) -> SCNGeometry {
            let meshGeometry = meshAnchor.geometry

            // Get vertices
            let vertices = meshGeometry.vertices
            let vertexSource = SCNGeometrySource(
                buffer: vertices.buffer,
                vertexFormat: vertices.format,
                semantic: .vertex,
                vertexCount: vertices.count,
                dataOffset: vertices.offset,
                dataStride: vertices.stride
            )

            // Get faces
            let faces = meshGeometry.faces
            let faceData = Data(
                bytesNoCopy: faces.buffer.contents(),
                count: faces.buffer.length,
                deallocator: .none
            )

            let geometryElement = SCNGeometryElement(
                data: faceData,
                primitiveType: .triangles,
                primitiveCount: faces.count,
                bytesPerIndex: faces.bytesPerIndex
            )

            return SCNGeometry(sources: [vertexSource], elements: [geometryElement])
        }

        func loadOBJModel() -> SCNNode? {
            let currentHash = objContent.hashValue

            if let template = cachedModelTemplate,
               cachedObjHash == currentHash {
                let instance = template.clone()
                makeMaterialsUnique(for: instance)
                addPhysicsBody(to: instance)
                return instance
            }

            guard let template = buildModelTemplate() else { return nil }
            cachedModelTemplate = template
            cachedObjHash = currentHash

            let instance = template.clone()
            makeMaterialsUnique(for: instance)
            addPhysicsBody(to: instance)
            return instance
        }

        func buildModelTemplate() -> SCNNode? {
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

            // Calculate scale needed to normalize to 20cm, but don't apply it to geometry
            let size = SCNVector3(
                x: max.x - min.x,
                y: max.y - min.y,
                z: max.z - min.z
            )
            let maxDimension = Swift.max(size.x, size.y, size.z)
            let desiredSize: Float = 0.2 // 20cm
            baseModelScale = desiredSize / maxDimension

            // Wrapper keeps preview transforms independent of geometry adjustments
            let wrapperNode = SCNNode()
            wrapperNode.addChildNode(containerNode)

            return wrapperNode
        }

        func makeMaterialsUnique(for node: SCNNode) {
            if let originalGeometry = node.geometry,
               let geometryCopy = originalGeometry.copy() as? SCNGeometry {
                geometryCopy.materials = originalGeometry.materials.map { material in
                    material.copy() as? SCNMaterial ?? material
                }
                node.geometry = geometryCopy
            }

            for child in node.childNodes {
                makeMaterialsUnique(for: child)
            }
        }

        func addPhysicsBody(to node: SCNNode) {
            node.physicsBody = nil

            // Use convex hull for better performance with dynamic physics
            let options: [SCNPhysicsShape.Option: Any] = [
                .type: SCNPhysicsShape.ShapeType.convexHull
            ]

            var shapeOptions = options
            shapeOptions[.collisionMargin] = 0.001
            let shape = SCNPhysicsShape(node: node, options: shapeOptions)
            // Start as kinematic (gravity off) for preview
            let body = SCNPhysicsBody(type: .kinematic, shape: shape)

            // Configure physics properties for when dropped
            body.mass = 0.5 // 500g
            body.restitution = 0.3 // Some bounciness
            body.friction = 0.8
            body.damping = 0.1 // Air resistance
            body.angularDamping = 0.2 // Rotational air resistance

            body.categoryBitMask = 1
            body.collisionBitMask = 2
            body.contactTestBitMask = 2

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
