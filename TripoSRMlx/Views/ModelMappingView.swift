//
//  ModelMappingView.swift
//  TripoSRMlx
//
//  モデルマッピングチェック結果を表示するビュー
//

import SwiftUI
import MLX
import MLXNN

struct ModelMappingView: View {
    @State private var checkResult: MappingComparisonResult?
    @State private var isChecking = false
    @State private var isGeneratingMLXStructure = false
    @State private var mlxDumpExists = false
    @State private var pytorchAnalysisExists = false
    @State private var pytorchAnalysisSource = "Not Found"
    @State private var exportedMLXStructure: String?
    @State private var showingMLXStructureText = false
    @State private var showingMLXMappingText = false

    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                fileStatusSection
                actionButtonsSection
                resultSection
                Spacer()
            }
            .padding()
            .navigationTitle("Model Mapping Checker")
            .onAppear {
                checkFileExistence()
            }

            // Detail view (right side)
            if showingMLXStructureText {
                TextDetailView(title: "MLX Structure", content: exportedMLXStructure ?? "")
            } else if showingMLXMappingText {
                TextDetailView(title: "MLX Mapping", content: exportedMLXStructure ?? "")
            } else {
                VStack {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text("Click 'View MLX Structure' to see details")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemGray6))
            }
        }
        .navigationViewStyle(DoubleColumnNavigationViewStyle())
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("モデルマッピング検証")
                .font(.title2)
                .fontWeight(.bold)

            Text("PyTorchモデルとMLXモデルのパラメータ構造を比較します")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var fileStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("📄 ファイル状況")
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: pytorchAnalysisExists ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(pytorchAnalysisExists ? .green : .red)
                    Text("pytorch_analysis.json")
                        .font(.system(.body, design: .monospaced))
                    Spacer()
                }
                if pytorchAnalysisExists {
                    Text("📍 \(pytorchAnalysisSource)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.leading, 20)
                }
            }

            HStack {
                Image(systemName: mlxDumpExists ? "checkmark.circle.fill" : "questionmark.circle.fill")
                    .foregroundColor(mlxDumpExists ? .green : .orange)
                Text("mlx_mapping_dump.json")
                    .font(.system(.body, design: .monospaced))
                Spacer()
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }

    private var actionButtonsSection: some View {
        VStack(spacing: 12) {
            Button(action: exportMLXStructure) {
                HStack {
                    if isGeneratingMLXStructure {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "doc.text")
                    }
                    Text(isGeneratingMLXStructure ? "MLX構造を生成中..." : "MLX構造を生成・表示")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.orange)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .disabled(isGeneratingMLXStructure)

            Button(action: runMappingCheck) {
                HStack {
                    if isChecking {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "arrow.triangle.2.circlepath")
                    }
                    Text(isChecking ? "チェック中..." : "マッピングチェック実行")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .disabled(isChecking || !pytorchAnalysisExists)
        }
    }

    private var resultSection: some View {
        Group {
            if let result = checkResult {
                VStack(alignment: .leading, spacing: 12) {
                    Text("📊 チェック結果")
                        .font(.headline)

                    ScrollView {
                        Text(result.summary)
                            .font(.system(.caption, design: .monospaced))
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                    }
                    .frame(maxHeight: 300)
                }
            }
        }
    }

    // MARK: - Actions

    private func checkFileExistence() {
        let fileManager = FileManager.default

        // PyTorch分析ファイルの存在確認
        // 1. まずbundleを確認
        if Bundle.main.path(forResource: "pytorch_analysis", ofType: "json") != nil {
            pytorchAnalysisExists = true
            pytorchAnalysisSource = "App Bundle"
        } else {
            // 2. 外部パスを確認
            let pytorchPaths = [
                ("converter/pytorch_analysis.json", "Converter Directory"),
                ("../converter/pytorch_analysis.json", "Parent/Converter Directory"),
                ("../../converter/pytorch_analysis.json", "Grandparent/Converter Directory"),
                ("pytorch_analysis.json", "Current Directory")
            ]

            var found = false
            for (path, source) in pytorchPaths {
                if fileManager.fileExists(atPath: path) {
                    pytorchAnalysisExists = true
                    pytorchAnalysisSource = source
                    found = true
                    break
                }
            }

            if !found {
                pytorchAnalysisExists = false
                pytorchAnalysisSource = "Not Found"
            }
        }

        // MLXダンプファイルの存在確認
        mlxDumpExists = fileManager.fileExists(atPath: "mlx_mapping_dump.json")
    }

    private func runMappingCheck() {
        isChecking = true

        DispatchQueue.global(qos: .userInitiated).async {
            // 実際のモデルを使用してチェック実行
            let checker = ModelMappingChecker.shared

            // PyTorch分析ファイルを読み込み
            var pytorchAnalysis: PyTorchAnalysis?

            // 1. まずbundleから読み込み
            if let analysis = checker.loadPyTorchAnalysisFromBundle() {
                pytorchAnalysis = analysis
            } else {
                // 2. 外部パスを確認
                let possiblePaths = [
                    "converter/pytorch_analysis.json",
                    "../converter/pytorch_analysis.json",
                    "../../converter/pytorch_analysis.json",
                    "pytorch_analysis.json"
                ]

                for path in possiblePaths {
                    if let analysis = checker.loadPyTorchAnalysis(from: path) {
                        pytorchAnalysis = analysis
                        break
                    }
                }
            }

            guard let pytorch = pytorchAnalysis else {
                DispatchQueue.main.async {
                    isChecking = false
                }
                return
            }

            // 実際のTripoSRモジュールを作成（プレースホルダー）
            let config = TSRSystemConfig.tripoSRConfig
            let tsrSystem = TSRSystem(config: config)
            
            let mlxParams: [String: MLXParameterInfo] = checker.extractMLXParameters(from: tsrSystem, prefix: nil)
            // 比較実行
            let result = checker.compareMapping(pytorch: pytorch, mlx: mlxParams)

            DispatchQueue.main.async {
                checkResult = result
                isChecking = false
                checkFileExistence() // ファイル状況を再確認
            }
        }
    }

    private func exportMLXStructure() {
        isGeneratingMLXStructure = true
        showingMLXMappingText = false // Reset other views

        DispatchQueue.global(qos: .userInitiated).async {
            let checker = ModelMappingChecker.shared
            let config = TSRSystemConfig.tripoSRConfig
            let tsrSystem = TSRSystem(config: config)
            let mlxParams: [String: MLXParameterInfo] = checker.extractMLXParameters(from: tsrSystem, prefix: nil)
            let allStructures = checker.generateMLXStructureData(mlxParams, moduleName: nil)

            let exportData: [String: Any] = [
                "export_timestamp": ISO8601DateFormatter().string(from: Date()),
                "modules": allStructures
            ]

            do {
                let jsonData = try JSONSerialization.data(withJSONObject: exportData, options: [.prettyPrinted])
                let jsonString = String(data: jsonData, encoding: .utf8) ?? "Failed to encode JSON"

                DispatchQueue.main.async {
                    exportedMLXStructure = jsonString
                    isGeneratingMLXStructure = false
                    showingMLXStructureText = true // Automatically show the results
                }
            } catch {
                DispatchQueue.main.async {
                    isGeneratingMLXStructure = false
                    print("❌ Failed to generate MLX structure JSON: \(error)")
                }
            }
        }
    }

    // MARK: - Helper

    private func createTripoSRSystem() -> Module {
        // 実際のTSRSystemを作成
        let config = TSRSystemConfig.tripoSRConfig
        return TSRSystem(config: config)
    }
}

// MARK: - Text Detail View

struct TextDetailView: View {
    let title: String
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.title2)
                    .fontWeight(.bold)

                HStack {
                    Label("\(content.split(separator: "\n").count) lines", systemImage: "doc.text")
                    Spacer()
                    Button("Copy") {
                        copyToClipboard()
                    }
                    .buttonStyle(.bordered)
                    .font(.caption)
                }
                .font(.caption)
                .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.systemGray6))

            Divider()

            // Content
            ScrollView {
                Text(content)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .textSelection(.enabled)
            }
            .background(Color(.systemBackground))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func copyToClipboard() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(content, forType: .string)
        #elseif os(iOS)
        UIPasteboard.general.string = content
        #endif
    }
}

struct ModelMappingView_Previews: PreviewProvider {
    static var previews: some View {
        ModelMappingView()
    }
}

// MARK: - Text View Sheet

import UIKit

struct TextViewSheet: View {
    let title: String
    let content: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack {
                ScrollView {
                    Text(content)
                        .font(.system(.caption, design: .monospaced))
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .padding()

                HStack {
                    Button("📋 コピー") {
                        UIPasteboard.general.string = content
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button("閉じる") {
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
