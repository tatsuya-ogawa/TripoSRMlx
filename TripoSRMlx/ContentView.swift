//
//  ContentView.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/17.
//

import SwiftUI

struct ContentView: View {
    @State private var showingDevMenu = false
    @State private var showingARView = false

    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header with hamburger menu
                HStack {
                    Spacer()

                    NavigationLink(destination: DevMenuView()) {
                        Image(systemName: "line.3.horizontal")
                            .font(.title2)
                            .foregroundColor(.gray)
                            .padding()
                    }
                }

                VStack(spacing: 10) {
                    Image(systemName: "cube.transparent")
                        .font(.system(size: 80))
                        .foregroundColor(.blue)

                    Text("TripoSR MLX")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("3D Generation from Single Images")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Main Navigation buttons
                VStack(spacing: 20) {
                    NavigationLink(destination: TSRTestView()) {
                        MenuButton(
                            title: "Generate 3D Model",
                            subtitle: "Create 3D models from images",
                            icon: "cube.fill",
                            color: .green
                        )
                    }

                    NavigationLink(destination: SavedModelsView()) {
                        MenuButton(
                            title: "Saved Models",
                            subtitle: "View and manage your models",
                            icon: "folder.fill",
                            color: .orange
                        )
                    }

                    Button(action: {
                        showingARView = true
                    }) {
                        MenuButton(
                            title: "AR Mode",
                            subtitle: "Place models in augmented reality",
                            icon: "arkit",
                            color: .purple
                        )
                    }
                }

                Spacer()

                // Footer
                VStack(spacing: 4) {
                    Text("Swift/MLX Implementation")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("Based on TripoSR by Stability AI")
                        .font(.caption2)
                        .foregroundColor(.blue)
                }
            }
            .padding(30)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationBarHidden(true)
        }
        .navigationViewStyle(StackNavigationViewStyle())
        .fullScreenCover(isPresented: $showingARView) {
            ARModelPlacementView()
        }
    }
}

struct MenuButton: View {
    let title: String
    let subtitle: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.white)
                .frame(width: 50, height: 50)
                .background(color)
                .cornerRadius(12)

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text(subtitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.leading)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(16)
        .background(Color(.systemGray6))
        .cornerRadius(12)
        .frame(maxWidth: .infinity)
    }
}

#Preview {
    ContentView()
}
