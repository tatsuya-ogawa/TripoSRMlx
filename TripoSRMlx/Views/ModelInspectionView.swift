//
//  ModelInspectionView.swift
//  TripoSRMlx
//
//  View for inspecting TSRSystem model parameters and state_dict structure
//

import SwiftUI
import MLX
import MLXNN

struct ModelInspectionView: View {
    @State private var tsrSystem: TSRSystem?
    @State private var isLoading = false
    @State private var modelInfo: [ModelComponentInfo] = []
    @State private var selectedComponent: ModelComponentInfo?
    @State private var searchText = ""

    var body: some View {
        NavigationView {
            VStack {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text("TSRSystem Model Inspector")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("Inspect model components and their state_dict structure")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)

                Divider()

                if isLoading {
                    VStack {
                        ProgressView("Loading TSRSystem...")
                        Text("Initializing models and extracting state_dict information")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if modelInfo.isEmpty {
                    VStack(spacing: 20) {
                        Button("Initialize TSRSystem") {
                            initializeTSRSystem()
                        }
                        .buttonStyle(.borderedProminent)
                        .font(.headline)

                        Button("Test Weight Loading") {
                            testWeightLoading()
                        }
                        .buttonStyle(.bordered)
                        .font(.headline)

                        Text("Click to create TSRSystem and inspect model structure, or test weight loading from bundle")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    // Model components list
                    VStack {
                        // Search bar
                        SearchBar(text: $searchText)
                            .padding(.horizontal)

                        List {
                            ForEach(filteredModelInfo, id: \.name) { component in
                                ModelComponentRow(
                                    component: component,
                                    isSelected: selectedComponent?.name == component.name
                                )
                                .onTapGesture {
                                    selectedComponent = component
                                }
                            }
                        }
                        .listStyle(PlainListStyle())
                    }
                }
            }
            .navigationBarHidden(true)

            // Detail view
            if let selectedComponent = selectedComponent {
                ModelDetailView(component: selectedComponent)
            } else {
                VStack {
                    Image(systemName: "doc.text.magnifyingglass")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text("Select a model component to view details")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemGray6))
            }
        }
        .navigationViewStyle(DoubleColumnNavigationViewStyle())
    }

    private var filteredModelInfo: [ModelComponentInfo] {
        if searchText.isEmpty {
            return modelInfo
        } else {
            return modelInfo.filter { component in
                component.name.localizedCaseInsensitiveContains(searchText) ||
                component.parameters.contains { $0.name.localizedCaseInsensitiveContains(searchText) }
            }
        }
    }

    private func initializeTSRSystem() {
        isLoading = true

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Initialize TSRSystem
                let system = TSRSystem.createTripoSRModel()

                // Extract model information
                let info = extractModelInfo(from: system)

                DispatchQueue.main.async {
                    self.tsrSystem = system
                    self.modelInfo = info
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    print("Error initializing TSRSystem: \(error)")
                    self.isLoading = false
                }
            }
        }
    }

    private func testWeightLoading() {
        isLoading = true

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                print("🔄 Starting weight loading test...")

                // Initialize TSRSystem
                let system = TSRSystem.createTripoSRModel()
                print("✅ TSRSystem initialized")

                // Create ModelLoader and test loading
                let modelLoader = ModelLoader()
                try modelLoader.loadAndApplyWeights(to: system)
                print("✅ Weight loading test completed successfully!")

                // Extract model information after weight loading
                let info = extractModelInfo(from: system)

                DispatchQueue.main.async {
                    self.tsrSystem = system
                    self.modelInfo = info
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    print("❌ Weight loading test failed: \(error)")
                    self.isLoading = false
                }
            }
        }
    }
}

// MARK: - Model Information Structures

struct ModelComponentInfo {
    let name: String
    let type: String
    let parameters: [ParameterInfo]
    let totalParameters: Int
    let memoryUsage: String
    let description: String
}

struct ParameterInfo {
    let name: String
    let shape: [Int]
    let dtype: String
    let size: Int
    let path: String
}

// MARK: - Supporting Views

struct SearchBar: View {
    @Binding var text: String

    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)

            TextField("Search models or parameters...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
        }
    }
}

struct ModelComponentRow: View {
    let component: ModelComponentInfo
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(component.name)
                        .font(.headline)
                        .foregroundColor(isSelected ? .white : .primary)

                    Text(component.type)
                        .font(.caption)
                        .foregroundColor(isSelected ? .white.opacity(0.8) : .secondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(component.totalParameters) params")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(isSelected ? .white : .primary)

                    Text(component.memoryUsage)
                        .font(.caption2)
                        .foregroundColor(isSelected ? .white.opacity(0.8) : .secondary)
                }
            }

            if !component.description.isEmpty {
                Text(component.description)
                    .font(.caption)
                    .foregroundColor(isSelected ? .white.opacity(0.9) : .secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
        .background(isSelected ? Color.blue : Color.clear)
        .cornerRadius(8)
    }
}

struct ModelDetailView: View {
    let component: ModelComponentInfo
    @State private var expandedSections: Set<String> = []

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 8) {
                Text(component.name)
                    .font(.title2)
                    .fontWeight(.bold)

                Text(component.type)
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                HStack {
                    Label("\(component.totalParameters) parameters", systemImage: "number.circle")
                    Spacer()
                    Label(component.memoryUsage, systemImage: "memorychip")
                }
                .font(.caption)
                .foregroundColor(.secondary)

                if !component.description.isEmpty {
                    Text(component.description)
                        .font(.caption)
                        .padding(.top, 4)
                }
            }
            .padding()
            .background(Color(.systemGray6))

            Divider()

            // Parameters list
            List {
                Section("State Dict Parameters") {
                    ForEach(component.parameters, id: \.name) { parameter in
                        ParameterRow(parameter: parameter)
                    }
                }

                Section("Parameter Summary") {
                    ParameterSummaryView(parameters: component.parameters)
                }
            }
            .listStyle(PlainListStyle())
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

struct ParameterRow: View {
    let parameter: ParameterInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(parameter.name)
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.medium)

                Spacer()

                Text("\(parameter.size) elements")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack {
                Text("Shape: \(shapeString)")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Text("Type: \(parameter.dtype)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if !parameter.path.isEmpty {
                Text("Path: \(parameter.path)")
                    .font(.caption2)
                    .foregroundColor(.blue)
            }
        }
        .padding(.vertical, 2)
    }

    private var shapeString: String {
        return "[\(parameter.shape.map(String.init).joined(separator: ", "))]"
    }
}

struct ParameterSummaryView: View {
    let parameters: [ParameterInfo]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Total Parameters:")
                Spacer()
                Text("\(totalParameters)")
                    .fontWeight(.medium)
            }

            HStack {
                Text("Largest Parameter:")
                Spacer()
                Text(largestParameter?.name ?? "N/A")
                    .font(.system(.caption, design: .monospaced))
            }

            HStack {
                Text("Parameter Types:")
                Spacer()
                Text(uniqueTypes.joined(separator: ", "))
                    .font(.caption)
            }
        }
        .font(.caption)
    }

    private var totalParameters: Int {
        parameters.reduce(0) { $0 + $1.size }
    }

    private var largestParameter: ParameterInfo? {
        parameters.max { $0.size < $1.size }
    }

    private var uniqueTypes: [String] {
        Array(Set(parameters.map(\.dtype))).sorted()
    }
}

// MARK: - Model Information Extraction

func extractModelInfo(from system: TSRSystem) -> [ModelComponentInfo] {
    var components: [ModelComponentInfo] = []

    // Image Tokenizer
    let imageTokenizerInfo = extractComponentInfo(
        name: "ImageTokenizer",
        module: system.imageTokenizer,
        description: "Converts input images to token embeddings using DINO-based vision transformer"
    )
    components.append(imageTokenizerInfo)

    // Triplane Tokenizer
    let tokenizerInfo = extractComponentInfo(
        name: "TriplaneTokenizer",
        module: system.tokenizer,
        description: "Generates learnable query tokens for triplane representation"
    )
    components.append(tokenizerInfo)

    // Transformer Backbone
    let backboneInfo = extractComponentInfo(
        name: "TransformerBackbone",
        module: system.backbone,
        description: "Multi-layer transformer with cross-attention to image tokens"
    )
    components.append(backboneInfo)

    // Post Processor
    let postProcessorInfo = extractComponentInfo(
        name: "PostProcessor",
        module: system.postProcessor,
        description: "Converts transformer tokens to triplane representations"
    )
    components.append(postProcessorInfo)

    // NeRF Decoder
    let decoderInfo = extractComponentInfo(
        name: "NeRFDecoder",
        module: system.decoder,
        description: "TripoSR-compliant MLP decoder for density and color prediction"
    )
    components.append(decoderInfo)

    // NeRF Renderer
    let rendererInfo = extractComponentInfo(
        name: "TriplaneNeRFRenderer",
        module: system.renderer,
        description: "Volume renderer with triplane sampling and ray marching"
    )
    components.append(rendererInfo)

    return components
}

func extractComponentInfo(name: String, module: Module, description: String) -> ModelComponentInfo {
    // Get module parameters using MLX reflection
    let moduleDict = module.parameters()
    var parameters: [ParameterInfo] = []
    var totalParams = 0
    
    // Extract parameter information
    for (key, nestedItem) in moduleDict {
        switch nestedItem {
        case .value(let array):
            let shape = Array(array.shape)
            let size = shape.reduce(1, *)
            let dtype = String(describing: array.dtype)

            let paramInfo = ParameterInfo(
                name: key,
                shape: shape,
                dtype: dtype,
                size: size,
                path: key
            )

            parameters.append(paramInfo)
            totalParams += size
        case .array(let arrays):
            // Handle array of MLXArrays
            for (index, nestedItem) in arrays.enumerated() {
                let arrayKey = "\(key)[\(index)]"

                // Extract parameters from each nested item
                let flattenedArrays = nestedItem.flattened()
                for (subKey, array) in flattenedArrays {
                    let fullKey = subKey.isEmpty ? arrayKey : "\(arrayKey).\(subKey)"
                    let shape = Array(array.shape)
                    let size = shape.reduce(1, *)
                    let dtype = String(describing: array.dtype)

                    let paramInfo = ParameterInfo(
                        name: fullKey,
                        shape: shape,
                        dtype: dtype,
                        size: size,
                        path: fullKey
                    )

                    parameters.append(paramInfo)
                    totalParams += size
                }
            }
        case .dictionary(let dictionary):
            // Handle nested dictionary recursively
            let nestedParams = extractParametersFromDictionary(dictionary, parentPath: key)
            parameters.append(contentsOf: nestedParams)
            totalParams += nestedParams.reduce(0) { $0 + $1.size }
        case .none:
            continue
        }
    }

    // Calculate memory usage (rough estimate)
    let memoryMB = Float(totalParams * 4) / (1024 * 1024) // Assuming float32
    let memoryUsage = String(format: "%.1f MB", memoryMB)

    return ModelComponentInfo(
        name: name,
        type: String(describing: type(of: module)),
        parameters: parameters.sorted { $0.name < $1.name },
        totalParameters: totalParams,
        memoryUsage: memoryUsage,
        description: description
    )
}

func extractParametersFromDictionary(_ dictionary: [String: NestedItem<String,MLXArray>], parentPath: String) -> [ParameterInfo] {
    var parameters: [ParameterInfo] = []

    for (key, nestedItem) in dictionary {
        let currentPath = parentPath.isEmpty ? key : "\(parentPath).\(key)"
        // Use flattened() method to extract all MLXArrays
        let flattenedArrays = nestedItem.flattened()

        for (subKey, array) in flattenedArrays {
            let fullKey = subKey.isEmpty ? key : "\(key).\(subKey)"
            let shape = Array(array.shape)
            let size = shape.reduce(1, *)
            let dtype = String(describing: array.dtype)

            let paramInfo = ParameterInfo(
                name: fullKey,
                shape: shape,
                dtype: dtype,
                size: size,
                path: subKey.isEmpty ? currentPath : "\(currentPath).\(subKey)"
            )

            parameters.append(paramInfo)
        }
    }

    return parameters
}

#Preview {
    ModelInspectionView()
}
