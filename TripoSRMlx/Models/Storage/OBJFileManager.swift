//
//  OBJFileManager.swift
//  TripoSRMlx
//
//  Storage manager for OBJ files in the app
//

import Foundation
import SwiftUI

/// Represents a saved OBJ file with metadata
public struct SavedOBJFile: Identifiable, Codable, Equatable {
    public let id: UUID
    public let name: String
    public let createdAt: Date
    public let fileName: String  // Store filename instead of full URL
    public let thumbnailURL: URL?

    public init(id: UUID = UUID(), name: String, createdAt: Date = Date(), fileName: String, thumbnailURL: URL? = nil) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.fileName = fileName
        self.thumbnailURL = thumbnailURL
    }

    // Legacy support for old fileURL-based storage
    enum CodingKeys: String, CodingKey {
        case id, name, createdAt, fileName, fileURL, thumbnailURL
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        thumbnailURL = try container.decodeIfPresent(URL.self, forKey: .thumbnailURL)

        // Try new fileName first, fallback to extracting from fileURL
        if let fileName = try? container.decode(String.self, forKey: .fileName) {
            self.fileName = fileName
        } else if let fileURL = try? container.decode(URL.self, forKey: .fileURL) {
            self.fileName = fileURL.lastPathComponent
        } else {
            throw DecodingError.dataCorruptedError(forKey: .fileName, in: container, debugDescription: "No fileName or fileURL found")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(fileName, forKey: .fileName)
        try container.encodeIfPresent(thumbnailURL, forKey: .thumbnailURL)
    }
}

/// Manager for saving and loading OBJ files in the app (non-observable storage class)
public class OBJFileManager {
    private let storageDirectory: URL
    private let metadataURL: URL

    public init() {
        // Create storage directory in app's documents
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        self.storageDirectory = documentsPath.appendingPathComponent("SavedModels", isDirectory: true)
        self.metadataURL = storageDirectory.appendingPathComponent("metadata.json")

        // Create storage directory if it doesn't exist
        try? FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)
    }

    /// Save an OBJ file to the app's storage
    public func saveOBJFile(mesh: TriMesh, name: String) throws -> SavedOBJFile {
        // Create unique filename
        let timestamp = Int(Date().timeIntervalSince1970)
        let filename = "\(name)_\(timestamp).obj"
        let fileURL = storageDirectory.appendingPathComponent(filename)

        // Export mesh to file
        try mesh.export(to: fileURL, format: "obj")
        print("✅ Saved OBJ file: \(fileURL.path)")

        // Create saved file entry (store filename only, not full path)
        let savedFile = SavedOBJFile(
            name: name,
            fileName: filename,
            thumbnailURL: nil
        )

        // Load existing files and append
        var savedFiles = loadSavedFiles()
        savedFiles.append(savedFile)

        // Save metadata
        try saveMetadata(savedFiles)
        print("✅ Updated metadata with \(savedFiles.count) files")

        return savedFile
    }

    /// Get the full file URL for a saved file
    public func getFileURL(_ file: SavedOBJFile) -> URL {
        return storageDirectory.appendingPathComponent(file.fileName)
    }

    /// Delete a saved OBJ file
    public func deleteFile(_ file: SavedOBJFile) throws {
        // Remove file from disk
        let fileURL = getFileURL(file)
        try FileManager.default.removeItem(at: fileURL)

        // Remove thumbnail if exists
        if let thumbnailURL = file.thumbnailURL {
            try? FileManager.default.removeItem(at: thumbnailURL)
        }

        // Remove from saved files list
        var savedFiles = loadSavedFiles()
        savedFiles.removeAll { $0.id == file.id }

        // Save metadata
        try saveMetadata(savedFiles)
    }

    /// Load all saved files
    public func loadSavedFiles() -> [SavedOBJFile] {
        print("🔍 Loading saved files from: \(metadataURL.path)")

        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            print("⚠️ Metadata file does not exist")
            return []
        }

        do {
            let data = try Data(contentsOf: metadataURL)
            let savedFiles = try JSONDecoder().decode([SavedOBJFile].self, from: data)
            print("📋 Loaded \(savedFiles.count) files from metadata")

            // Filter out files that no longer exist on disk
            let existingFiles = savedFiles.filter {
                let fileURL = storageDirectory.appendingPathComponent($0.fileName)
                let exists = FileManager.default.fileExists(atPath: fileURL.path)
                if !exists {
                    print("⚠️ File missing: \(fileURL.path)")
                }
                return exists
            }
            print("✅ Found \(existingFiles.count) existing files on disk")

            return existingFiles
        } catch {
            print("❌ Failed to load metadata: \(error)")
            return []
        }
    }

    /// Load an OBJ file's content
    public func loadOBJContent(_ file: SavedOBJFile) throws -> String {
        let fileURL = getFileURL(file)
        return try String(contentsOf: fileURL, encoding: .utf8)
    }

    // MARK: - Private Methods

    private func saveMetadata(_ savedFiles: [SavedOBJFile]) throws {
        let data = try JSONEncoder().encode(savedFiles)
        try data.write(to: metadataURL)
    }
}
