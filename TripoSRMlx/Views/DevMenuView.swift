//
//  DevMenuView.swift
//  TripoSRMlx
//
//  Developer Mode Menu
//

import SwiftUI

struct DevMenuView: View {
    var body: some View {
        VStack(spacing: 30) {
            // Header
            VStack(spacing: 10) {
                Image(systemName: "wrench.and.screwdriver")
                    .font(.system(size: 60))
                    .foregroundColor(.orange)

                Text("Developer Mode")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("Debug & Inspection Tools")
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
        }
        .padding(30)
        .navigationTitle("Developer Mode")
    }
}

#Preview {
    NavigationView {
        DevMenuView()
    }
}
