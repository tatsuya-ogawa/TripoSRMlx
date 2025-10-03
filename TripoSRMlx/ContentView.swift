//
//  ContentView.swift
//  TripoSRMlx
//
//  Created by Tatsuya Ogawa on 2025/09/17.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
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

                // Navigation buttons
                VStack(spacing: 16) {
                    NavigationLink(destination: ModelInspectionView()) {
                        MenuButton(
                            title: "Model Inspector",
                            subtitle: "Inspect TSRSystem components and state_dict",
                            icon: "doc.text.magnifyingglass",
                            color: .blue
                        )
                    }

                    NavigationLink(destination: TSRTestView()) {
                        MenuButton(
                            title: "TripoSR Demo",
                            subtitle: "Generate 3D models from images",
                            icon: "cube.fill",
                            color: .green
                        )
                    }

                    NavigationLink(destination: ModelMappingView()) {
                        MenuButton(
                            title: "Model Mapping Checker",
                            subtitle: "Compare PyTorch vs MLX parameters",
                            icon: "checkmark.shield.fill",
                            color: .red
                        )
                    }

                    NavigationLink(destination: DINOOutputExporterView()) {
                        MenuButton(
                            title: "DINO Output Exporter",
                            subtitle: "Export MLX DINO outputs for Python comparison",
                            icon: "brain.head.profile",
                            color: .indigo
                        )
                    }

                    NavigationLink(destination: MarchingCubesTestView()) {
                        MenuButton(
                            title: "Marching Cubes Test",
                            subtitle: "Test marching cubes with synthetic sphere data",
                            icon: "cube.transparent.fill",
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
