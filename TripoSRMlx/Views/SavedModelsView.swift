//
//  SavedModelsView.swift
//  TripoSRMlx
//
//  View for displaying saved OBJ models
//

import SwiftUI

struct SavedModelsView: View {
    @StateObject private var viewModel = SavedModelsViewModel()
    @State private var selectedFile: SavedOBJFile?
    @State private var showingDeleteConfirmation = false
    @State private var fileToDelete: SavedOBJFile?
    @State private var showingARView = false
    @State private var arFile: SavedOBJFile?
    @State private var useGridLayout = true

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.savedFiles.isEmpty {
                emptyStateView
            } else {
                if useGridLayout {
                    modelsGrid
                } else {
                    modelsList
                }
            }
        }
        .navigationTitle("Saved Models")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: {
                    useGridLayout.toggle()
                }) {
                    Image(systemName: useGridLayout ? "list.bullet" : "square.grid.2x2")
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    viewModel.loadFiles()
                }) {
                    Image(systemName: "arrow.clockwise")
                }
            }
        }
        .refreshable {
            viewModel.loadFiles()
        }
        .sheet(item: $selectedFile) { file in
            OBJPreviewView(file: file, viewModel: viewModel)
        }
        .fullScreenCover(isPresented: $showingARView) {
            if let file = arFile {
                ARModelPlacementView(
                    file: file,
                    objContent: (try? viewModel.getOBJContent(file)) ?? ""
                )
            }
        }
        .alert("Delete Model", isPresented: $showingDeleteConfirmation, presenting: fileToDelete) { file in
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteFile(file)
            }
        } message: { file in
            Text("Are you sure you want to delete '\(file.name)'? This action cannot be undone.")
        }
        .onAppear {
            viewModel.loadFiles()
        }
    }

    // MARK: - View Components

    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Spacer()

            Image(systemName: "cube.transparent")
                .font(.system(size: 80))
                .foregroundColor(.gray)

            Text("No Saved Models")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.primary)

            Text("Generate and save models from the TripoSR Demo")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()
        }
    }

    private var modelsGrid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 180))], spacing: 20) {
                ForEach(viewModel.savedFiles) { file in
                    ModelCard(file: file, isCompact: false, onDelete: {
                        fileToDelete = file
                        showingDeleteConfirmation = true
                    })
                    .onTapGesture {
                        selectedFile = file
                    }
                    .contextMenu {
                        Button(action: {
                            selectedFile = file
                        }) {
                            Label("Preview", systemImage: "eye")
                        }

                        Button(action: {
                            arFile = file
                            showingARView = true
                        }) {
                            Label("Place in AR", systemImage: "arkit")
                        }

                        Button(role: .destructive, action: {
                            fileToDelete = file
                            showingDeleteConfirmation = true
                        }) {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
            }
            .padding()
        }
    }

    private var modelsList: some View {
        List {
            ForEach(viewModel.savedFiles) { file in
                ModelCard(file: file, isCompact: true)
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowBackground(Color.clear)
                    .onTapGesture {
                        selectedFile = file
                    }
                    .contextMenu {
                        Button(action: {
                            selectedFile = file
                        }) {
                            Label("Preview", systemImage: "eye")
                        }

                        Button(action: {
                            arFile = file
                            showingARView = true
                        }) {
                            Label("Place in AR", systemImage: "arkit")
                        }

                        Button(role: .destructive, action: {
                            fileToDelete = file
                            showingDeleteConfirmation = true
                        }) {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) {
                            fileToDelete = file
                            showingDeleteConfirmation = true
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            arFile = file
                            showingARView = true
                        } label: {
                            Label("AR", systemImage: "arkit")
                        }
                        .tint(.purple)
                    }
            }
        }
        .listStyle(.plain)
    }

    // MARK: - Actions

    private func deleteFile(_ file: SavedOBJFile) {
        viewModel.deleteFile(file)
    }
}

// MARK: - Model Card Component

struct ModelCard: View {
    let file: SavedOBJFile
    var isCompact: Bool = false
    var onDelete: (() -> Void)? = nil
    var onAR: (() -> Void)? = nil

    @State private var thumbnail: UIImage?
    @State private var isLoadingThumbnail = false

    var body: some View {
        if isCompact {
            compactView
        } else {
            gridView
        }
    }

    private var compactView: some View {
        HStack(spacing: 12) {
            // Thumbnail
            RoundedRectangle(cornerRadius: 8)
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [Color.blue.opacity(0.6), Color.purple.opacity(0.6)]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 60, height: 60)
                .overlay(
                    Group {
                        if let thumbnail = thumbnail {
                            Image(uiImage: thumbnail)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 60, height: 60)
                                .clipped()
                        } else if isLoadingThumbnail {
                            ProgressView()
                                .scaleEffect(0.6)
                        } else {
                            Image(systemName: "cube.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white.opacity(0.9))
                        }
                    }
                )
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .onAppear {
                    loadThumbnail()
                }

            VStack(alignment: .leading, spacing: 4) {
                Text(file.name)
                    .font(.headline)
                    .lineLimit(1)
                    .foregroundColor(.primary)

                Text(formatDate(file.createdAt))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(8)
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private var gridView: some View {
        VStack(spacing: 12) {
            ZStack(alignment: .topTrailing) {
                // Thumbnail
                RoundedRectangle(cornerRadius: 12)
                    .fill(
                        LinearGradient(
                            gradient: Gradient(colors: [Color.blue.opacity(0.6), Color.purple.opacity(0.6)]),
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .aspectRatio(1.0, contentMode: .fit)
                    .overlay(
                        Group {
                            if let thumbnail = thumbnail {
                                Image(uiImage: thumbnail)
                                    .resizable()
                                    .scaledToFit()
                            } else if isLoadingThumbnail {
                                ProgressView()
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "cube.fill")
                                    .font(.system(size: 40))
                                    .foregroundColor(.white.opacity(0.9))
                            }
                        }
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .onAppear {
                        loadThumbnail()
                    }

                // Delete button overlay
                if let onDelete = onDelete {
                    Button(action: onDelete) {
                        Image(systemName: "trash.fill")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.red.opacity(0.9))
                            .clipShape(Circle())
                    }
                    .padding(8)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(file.name)
                    .font(.headline)
                    .lineLimit(2)
                    .foregroundColor(.primary)

                Text(formatDate(file.createdAt))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private func loadThumbnail() {
        // First, try to load cached thumbnail
        if let cached = ThumbnailGenerator.shared.loadCachedThumbnail(for: file) {
            thumbnail = cached
            return
        }

        // Generate thumbnail if not cached
        isLoadingThumbnail = true
        Task {
            let objURL = OBJFileManager().getFileURL(file)
            if let generated = await ThumbnailGenerator.shared.generateThumbnail(for: objURL, file: file) {
                await MainActor.run {
                    thumbnail = generated
                    isLoadingThumbnail = false
                }
            } else {
                await MainActor.run {
                    isLoadingThumbnail = false
                }
            }
        }
    }
}

#Preview {
    NavigationView {
        SavedModelsView()
    }
}
