//
//  SavedModelsViewModel.swift
//  TripoSRMlx
//
//  ViewModel for managing saved OBJ models
//

import Foundation
import SwiftUI
import Combine

@MainActor
class SavedModelsViewModel: ObservableObject {    
    @Published var savedFiles: [SavedOBJFile] = []

    private let fileManager = OBJFileManager()

    init() {
        loadFiles()
    }

    func loadFiles() {
        savedFiles = fileManager.loadSavedFiles()
    }

    func deleteFile(_ file: SavedOBJFile) {
        do {
            try fileManager.deleteFile(file)
            loadFiles() // Reload after deletion
        } catch {
            print("Failed to delete file: \(error)")
        }
    }

    func getOBJContent(_ file: SavedOBJFile) throws -> String {
        return try fileManager.loadOBJContent(file)
    }

    func getFileURL(_ file: SavedOBJFile) -> URL {
        return fileManager.getFileURL(file)
    }
}
