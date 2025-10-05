//
//  ThumbnailGenerator.swift
//  TripoSRMlx
//
//  Generate thumbnail images from OBJ files
//

import Foundation
import SceneKit
import UIKit

class ThumbnailGenerator {
    static let shared = ThumbnailGenerator()

    private let thumbnailSize = CGSize(width: 300, height: 300)
    private let cacheDirectory: URL

    private init() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        cacheDirectory = documentsPath.appendingPathComponent("Thumbnails", isDirectory: true)

        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
    }

    /// Get thumbnail URL for a file
    func getThumbnailURL(for file: SavedOBJFile) -> URL {
        return cacheDirectory.appendingPathComponent("\(file.id.uuidString).png")
    }

    /// Generate thumbnail from OBJ file
    func generateThumbnail(for objURL: URL, file: SavedOBJFile) async -> UIImage? {
        // Check if thumbnail already exists
        let thumbnailURL = getThumbnailURL(for: file)
        if FileManager.default.fileExists(atPath: thumbnailURL.path),
           let existingImage = UIImage(contentsOfFile: thumbnailURL.path) {
            return existingImage
        }

        // Generate new thumbnail
        guard let thumbnail = await renderThumbnail(from: objURL) else {
            return nil
        }

        // Save to cache
        if let pngData = thumbnail.pngData() {
            try? pngData.write(to: thumbnailURL)
        }

        return thumbnail
    }

    /// Render thumbnail using SceneKit
    private func renderThumbnail(from objURL: URL) async -> UIImage? {
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                // Parse OBJ file with vertex colors
                guard let (geometry, hasColors) = self.parseOBJWithColors(url: objURL) else {
                    continuation.resume(returning: nil)
                    return
                }

                // Create container node
                let containerNode = SCNNode(geometry: geometry)

                // Apply material based on whether colors exist
                let material = SCNMaterial()
                material.lightingModel = .blinn  // Use simpler lighting for better color visibility

                if hasColors {
                    // Use vertex colors - the geometry already has color data
                    // Setting diffuse to white allows vertex colors to show through
                    material.diffuse.contents = UIColor.white
                    material.isDoubleSided = true
                } else {
                    // Default gray material
                    material.diffuse.contents = UIColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
                    material.isDoubleSided = true
                }

                geometry.materials = [material]

                // Center and scale the model
                let (min, max) = containerNode.boundingBox
                let center = SCNVector3(
                    x: (min.x + max.x) / 2,
                    y: (min.y + max.y) / 2,
                    z: (min.z + max.z) / 2
                )
                containerNode.position = SCNVector3(-center.x, -center.y, -center.z)

                let size = SCNVector3(
                    x: max.x - min.x,
                    y: max.y - min.y,
                    z: max.z - min.z
                )
                let maxDimension = Swift.max(size.x, size.y, size.z)
                let scale = 2.0 / maxDimension

                let wrapperNode = SCNNode()
                wrapperNode.addChildNode(containerNode)
                wrapperNode.scale = SCNVector3(scale, scale, scale)

                // Create render scene
                let renderScene = SCNScene()
                renderScene.rootNode.addChildNode(wrapperNode)

                // Add lighting
                let lightNode = SCNNode()
                lightNode.light = SCNLight()
                lightNode.light?.type = .omni
                lightNode.position = SCNVector3(x: 2, y: 2, z: 2)
                renderScene.rootNode.addChildNode(lightNode)

                let ambientLight = SCNNode()
                ambientLight.light = SCNLight()
                ambientLight.light?.type = .ambient
                ambientLight.light?.intensity = 300
                renderScene.rootNode.addChildNode(ambientLight)

                // Add camera
                let cameraNode = SCNNode()
                cameraNode.camera = SCNCamera()
                cameraNode.position = SCNVector3(x: 1.5, y: 1.5, z: 2.5)
                cameraNode.look(at: SCNVector3(0, 0, 0))
                renderScene.rootNode.addChildNode(cameraNode)

                // Render to image
                let renderer = SCNRenderer(device: nil, options: nil)
                renderer.scene = renderScene

                let image = renderer.snapshot(atTime: 0, with: self.thumbnailSize, antialiasingMode: .multisampling4X)

                continuation.resume(returning: image)
            }
        }
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

    /// Process materials to ensure vertex colors are displayed
    private func processMaterials(node: SCNNode) {
        // Process this node's geometry
        if let geometry = node.geometry {
            // Check if geometry has vertex colors
            if let colorSource = geometry.sources(for: .color).first {
                // Create or update material to use vertex colors
                let material = SCNMaterial()
                material.lightingModel = .physicallyBased

                // Enable vertex color usage
                material.diffuse.contents = UIColor.white
                material.multiply.contents = colorSource
                material.multiply.mappingChannel = 0

                // Better lighting response
                material.metalness.contents = 0.0
                material.roughness.contents = 0.6

                geometry.materials = [material]
            } else {
                // No vertex colors, use a default material
                let material = SCNMaterial()
                material.lightingModel = .physicallyBased
                material.diffuse.contents = UIColor(red: 0.8, green: 0.8, blue: 0.8, alpha: 1.0)
                material.metalness.contents = 0.1
                material.roughness.contents = 0.5

                if geometry.materials.isEmpty {
                    geometry.materials = [material]
                }
            }
        }

        // Recursively process children
        for child in node.childNodes {
            processMaterials(node: child)
        }
    }

    /// Load cached thumbnail if exists
    func loadCachedThumbnail(for file: SavedOBJFile) -> UIImage? {
        let thumbnailURL = getThumbnailURL(for: file)
        if FileManager.default.fileExists(atPath: thumbnailURL.path) {
            return UIImage(contentsOfFile: thumbnailURL.path)
        }
        return nil
    }
}
