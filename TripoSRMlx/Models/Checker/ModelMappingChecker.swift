//
//  ModelMappingChecker.swift
//  TripoSRMlx
//
//  Utility to check parameter mapping between MLX and PyTorch models
//

import Foundation
import MLX
import MLXNN

/// Structure representing the PyTorch analysis results
struct PyTorchAnalysis: Codable {
    let modelName: String
    let analysisTimestamp: String
    let pytorchParameters: [String: ParameterInfo]
    let structure: ModelStructure
    let summary: AnalysisSummary

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case analysisTimestamp = "analysis_timestamp"
        case pytorchParameters = "pytorch_parameters"
        case structure, summary
    }

    struct ParameterInfo: Codable {
        let shape: [Int]
        let dtype: String
        let size: Int
        let ndim: Int
    }

    struct ModelStructure: Codable {
        let totalParameters: Int
        let parameterGroups: [String: [String]]
        let modelComponents: [String]

        enum CodingKeys: String, CodingKey {
            case totalParameters = "total_parameters"
            case parameterGroups = "parameter_groups"
            case modelComponents = "model_components"
        }
    }

    struct AnalysisSummary: Codable {
        let totalParameters: Int
        let totalSizeMb: Double
        let parameterGroups: [String: GroupInfo]

        enum CodingKeys: String, CodingKey {
            case totalParameters = "total_parameters"
            case totalSizeMb = "total_size_mb"
            case parameterGroups = "parameter_groups"
        }

        struct GroupInfo: Codable {
            let parameters: [String: GroupParameterInfo]
            let count: Int
            let totalParams: Int

            enum CodingKeys: String, CodingKey {
                case parameters, count
                case totalParams = "total_params"
            }

            struct GroupParameterInfo: Codable {
                let shape: [Int]
                let size: Int
            }
        }
    }
}

/// MLX parameter information
struct MLXParameterInfo {
    let key: String
    let shape: [Int]
    let dtype: String
    let size: Int
    let ndim: Int
}

/// Name mapping result
struct NameMappingResult {
    let mlxToTripoSR: [String: String]  // MLX name -> TripoSR (PyTorch) name
    let tripoSRToMLX: [String: String]  // TripoSR (PyTorch) name -> MLX name
    let unmatchedMLX: [String]          // Unmatched MLX names
    let unmatchedTripoSR: [String]      // Unmatched TripoSR names
}

/// Mapping comparison result
struct MappingComparisonResult {
    let isValid: Bool
    let totalPyTorchParams: Int
    let totalMLXParams: Int
    let matchingKeys: [String]
    let missingInMLX: [String]
    let extraInMLX: [String]
    let shapeMismatches: [(key: String, pytorch: [Int], mlx: [Int])]
    let dtypeMismatches: [(key: String, pytorch: String, mlx: String)]

    var summary: String {
        var result = """
        📊 Model Mapping Comparison Results
        =====================================
        ✅ Valid: \(isValid)
        📈 PyTorch Parameters: \(totalPyTorchParams)
        📊 MLX Parameters: \(totalMLXParams)
        🎯 Matching Keys: \(matchingKeys.count)

        """

        if !missingInMLX.isEmpty {
            result += "❌ Missing in MLX (\(missingInMLX.count)):\n"
            for key in missingInMLX.prefix(5) {
                result += "  - \(key)\n"
            }
            if missingInMLX.count > 5 {
                result += "  ... and \(missingInMLX.count - 5) more\n"
            }
            result += "\n"
        }

        if !extraInMLX.isEmpty {
            result += "⚠️ Extra in MLX (\(extraInMLX.count)):\n"
            for key in extraInMLX.prefix(5) {
                result += "  - \(key)\n"
            }
            if extraInMLX.count > 5 {
                result += "  ... and \(extraInMLX.count - 5) more\n"
            }
            result += "\n"
        }

        if !shapeMismatches.isEmpty {
            result += "🔺 Shape Mismatches (\(shapeMismatches.count)):\n"
            for mismatch in shapeMismatches.prefix(3) {
                result += "  - \(mismatch.key): PyTorch\(mismatch.pytorch) vs MLX\(mismatch.mlx)\n"
            }
            if shapeMismatches.count > 3 {
                result += "  ... and \(shapeMismatches.count - 3) more\n"
            }
            result += "\n"
        }

        if !dtypeMismatches.isEmpty {
            result += "🔺 Dtype Mismatches (\(dtypeMismatches.count)):\n"
            for mismatch in dtypeMismatches.prefix(3) {
                result += "  - \(mismatch.key): PyTorch[\(mismatch.pytorch)] vs MLX[\(mismatch.mlx)]\n"
            }
            if dtypeMismatches.count > 3 {
                result += "  ... and \(dtypeMismatches.count - 3) more\n"
            }
        }

        return result
    }
}

/// MLX model mapping checker
nonisolated class ModelMappingChecker {

    static let shared = ModelMappingChecker()

    private init() {}

    /// Load a PyTorch analysis file from a given path
    func loadPyTorchAnalysis(from path: String) -> PyTorchAnalysis? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            print("❌ Failed to load PyTorch analysis file: \(path)")
            return nil
        }

        do {
            let analysis = try JSONDecoder().decode(PyTorchAnalysis.self, from: data)
            print("✅ Loaded PyTorch analysis: \(analysis.pytorchParameters.count) parameters")
            return analysis
        } catch {
            print("❌ Failed to decode PyTorch analysis: \(error)")
            return nil
        }
    }

    /// Load `pytorch_analysis.json` from the app bundle
    func loadPyTorchAnalysisFromBundle() -> PyTorchAnalysis? {
        guard let bundlePath = Bundle.main.path(forResource: "pytorch_analysis", ofType: "json") else {
            print("❌ pytorch_analysis.json not found in bundle")
            return nil
        }

        print("📦 Found pytorch_analysis.json in bundle: \(bundlePath)")
        return loadPyTorchAnalysis(from: bundlePath)
    }

    /// Extract parameter info from an MLX module
    func extractMLXParameters(from module: Module, prefix: String? = nil) -> [String: MLXParameterInfo] {
        let moduleDict = module.parameters()
        var mlxParams: [String: MLXParameterInfo] = [:]

        for (key, nestedItem) in moduleDict {
            let flattenedArrays = nestedItem.flattened()

            for (subKey, array) in flattenedArrays {
                let baseKey = subKey.isEmpty ? key : "\(key).\(subKey)"
                let fullKey: String
                if let prefix, !prefix.isEmpty {
                    fullKey = "\(prefix).\(baseKey)"
                } else {
                    fullKey = baseKey
                }
                let shape = Array(array.shape)
                let size = shape.reduce(1, *)
                let dtype = String(describing: array.dtype)

                mlxParams[fullKey] = MLXParameterInfo(
                    key: fullKey,
                    shape: shape,
                    dtype: dtype,
                    size: size,
                    ndim: shape.count
                )
            }
        }

        return mlxParams
    }

    /// Create name mapping between PyTorch and MLX keys
    func createNameMapping(pytorch: PyTorchAnalysis, mlx: [String: MLXParameterInfo]) -> NameMappingResult {
        let pytorchKeys = Set(pytorch.pytorchParameters.keys)
        let mlxKeys = Set(mlx.keys)

        var mlxToTripoSR: [String: String] = [:]
        var tripoSRToMLX: [String: String] = [:]

    // Current mapping rules (extendable)
        for mlxKey in mlxKeys {
            let mappedName = mapMLXToTripoSR(mlxKey: mlxKey, availableTripoSRKeys: pytorchKeys)
            if let mapped = mappedName {
                mlxToTripoSR[mlxKey] = mapped
                tripoSRToMLX[mapped] = mlxKey
            }
        }

        let unmatchedMLX = mlxKeys.subtracting(mlxToTripoSR.keys).sorted()
        let unmatchedTripoSR = pytorchKeys.subtracting(tripoSRToMLX.keys).sorted()

        return NameMappingResult(
            mlxToTripoSR: mlxToTripoSR,
            tripoSRToMLX: tripoSRToMLX,
            unmatchedMLX: unmatchedMLX,
            unmatchedTripoSR: unmatchedTripoSR
        )
    }

    /// Map an MLX name to a TripoSR (PyTorch) name
    private func mapMLXToTripoSR(mlxKey: String, availableTripoSRKeys: Set<String>) -> String? {
    // Try the original key as fallback
        if availableTripoSRKeys.contains(mlxKey) {
            return mlxKey
        }

        return nil
    }

    /// Check whether MLX and PyTorch shapes are compatible (handles conv weight layout differences)
    private func areShapesCompatible(pytorchShape: [Int], mlxShape: [Int], paramKey: String) -> Bool {
        // Exact match
        if pytorchShape == mlxShape {
            return true
        }

        // 4D Conv2d weights: PyTorch[out, in, h, w] vs MLX[out, h, w, in]
        if paramKey.contains("patch_embeddings.projection.weight") && pytorchShape.count == 4 && mlxShape.count == 4 {
            let expectedMLXShape = [pytorchShape[0], pytorchShape[2], pytorchShape[3], pytorchShape[1]]
            return mlxShape == expectedMLXShape
        }

        // 4D ConvTransposed2d weights: PyTorch[in, out, h, w] vs MLX[out, h, w, in]
        if paramKey.contains("upsample.weight") && pytorchShape.count == 4 && mlxShape.count == 4 {
            let expectedMLXShape = [pytorchShape[1], pytorchShape[2], pytorchShape[3], pytorchShape[0]]
            return mlxShape == expectedMLXShape
        }

        // Other cases: strict check
        return false
    }

    /// Compare parameter mappings
    func compareMapping(pytorch: PyTorchAnalysis, mlx: [String: MLXParameterInfo]) -> MappingComparisonResult {
        let pytorchKeys = Set(pytorch.pytorchParameters.keys)

    // Normalize MLX keys and create mapping
    let nameMapping = createNameMapping(pytorch: pytorch, mlx: mlx)

    // Create the set of normalized MLX keys
    let normalizedMLXKeys = Set(nameMapping.mlxToTripoSR.values)

        let matchingKeys = Array(pytorchKeys.intersection(normalizedMLXKeys)).sorted()
        let missingInMLX = Array(pytorchKeys.subtracting(normalizedMLXKeys)).sorted()
        let extraInMLX = Array(nameMapping.unmatchedMLX).sorted()

        var shapeMismatches: [(key: String, pytorch: [Int], mlx: [Int])] = []
        var dtypeMismatches: [(key: String, pytorch: String, mlx: String)] = []

    // Detailed comparison for matching keys
        for pytorchKey in matchingKeys {
            guard let pytorchParam = pytorch.pytorchParameters[pytorchKey] else { continue }

            // Find the MLX key corresponding to the PyTorch key
            let mlxKey = nameMapping.tripoSRToMLX[pytorchKey]
            guard let actualMLXKey = mlxKey, let mlxParam = mlx[actualMLXKey] else { continue }

            // Shape comparison (handles MLX-PyTorch conv weight layout differences)
            if !areShapesCompatible(pytorchShape: pytorchParam.shape, mlxShape: mlxParam.shape, paramKey: pytorchKey) {
                shapeMismatches.append((
                    key: pytorchKey,
                    pytorch: pytorchParam.shape,
                    mlx: mlxParam.shape
                ))
            }

            // Dtype comparison (simple form)
            let normalizedPytorchDtype = pytorchParam.dtype.replacingOccurrences(of: "torch.", with: "")
            if normalizedPytorchDtype != mlxParam.dtype {
                dtypeMismatches.append((
                    key: pytorchKey,
                    pytorch: pytorchParam.dtype,
                    mlx: mlxParam.dtype
                ))
            }
        }

        let totalMLXParams = mlx.values.reduce(0) { $0 + $1.size }
        let isValid = missingInMLX.isEmpty && shapeMismatches.isEmpty

        return MappingComparisonResult(
            isValid: isValid,
            totalPyTorchParams: pytorch.summary.totalParameters,
            totalMLXParams: totalMLXParams,
            matchingKeys: matchingKeys,
            missingInMLX: missingInMLX,
            extraInMLX: extraInMLX,
            shapeMismatches: shapeMismatches,
            dtypeMismatches: dtypeMismatches
        )
    }

    /// Generate MLX structure data (in-memory)
    func generateMLXStructureData(_ mlxParams: [String: MLXParameterInfo], moduleName: String? = nil) -> [String: Any] {
        // Group parameters
        var parameterGroups: [String: [[String: Any]]] = [:]
        var totalParams = 0

        for (key, param) in mlxParams {
            let normalizedKey: String
            if let moduleName,key.hasPrefix("\(moduleName).") {
                normalizedKey = String(key.dropFirst(moduleName.count + 1))
            } else {
                normalizedKey = key
            }

            let component = normalizedKey.components(separatedBy: ".").first ?? "root"

            if parameterGroups[component] == nil {
                parameterGroups[component] = []
            }

            let paramInfo: [String: Any] = [
                "key": key,
                "shape": param.shape,
                "dtype": param.dtype,
                "size": param.size,
                "ndim": param.ndim,
                "shape_str": "[\(param.shape.map(String.init).joined(separator: ", "))]"
            ]

            parameterGroups[component]?.append(paramInfo)
            totalParams += param.size
        }

        return [
            "module_name": moduleName,
            "export_timestamp": ISO8601DateFormatter().string(from: Date()),
            "total_parameters": totalParams,
            "total_keys": mlxParams.count,
            "parameter_groups": parameterGroups,
            "summary": [
                "groups_count": parameterGroups.count,
                "groups": parameterGroups.keys.sorted(),
                "largest_group": parameterGroups.max { $0.value.count < $1.value.count }?.key ?? "none",
                "parameters_by_group": parameterGroups.mapValues { $0.count }
            ],
            "all_keys_sorted": mlxParams.keys.sorted()
        ]
    }

    /// Generate MLX mapping data (in-memory)
    func generateMLXMappingData(_ mlxParams: [String: MLXParameterInfo]) -> [String: Any] {
        return [
            "model_name": "TripoSR_MLX",
            "export_timestamp": ISO8601DateFormatter().string(from: Date()),
            "total_parameters": mlxParams.values.reduce(0) { $0 + $1.size },
            "total_keys": mlxParams.count,
            "parameters": mlxParams.mapValues { param in
                return [
                    "shape": param.shape,
                    "dtype": param.dtype,
                    "size": param.size,
                    "ndim": param.ndim
                ]
            }
        ]
    }

    /// Export MLX structure in detail (file, for debugging)
    func exportMLXStructure(_ mlxParams: [String: MLXParameterInfo], moduleName: String, to path: String) {
        let structureData = generateMLXStructureData(mlxParams, moduleName: moduleName)

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: structureData, options: [.prettyPrinted])
            try jsonData.write(to: URL(fileURLWithPath: path))
            print("📋 Exported MLX structure [\(moduleName)] to: \(path)")
            if let totalParams = structureData["total_parameters"] as? Int,
               let groups = structureData["parameter_groups"] as? [String: Any] {
                print("   - Total params: \(totalParams)")
                print("   - Groups: \(groups.keys.sorted().joined(separator: ", "))")
            }
        } catch {
            print("❌ Failed to export MLX structure: \(error)")
        }
    }

    /// Run automatic model mapping check (called at startup)
    func performAutoCheck() {
        print("🔍 Starting automatic model mapping check...")

    // First, try loading from the app bundle
        var pytorchAnalysis: PyTorchAnalysis?

    // 1. Load from bundle
        if let analysis = loadPyTorchAnalysisFromBundle() {
            pytorchAnalysis = analysis
            print("✅ Loaded PyTorch analysis from bundle")
        } else {
            fatalError("analysis.json not found in bundle. Aborting.")
        }

        guard let pytorch = pytorchAnalysis else {
            print("⚠️ PyTorch analysis file not found in bundle or external paths.")
            print("💡 Add pytorch_analysis.json to your Xcode project bundle")
            print("💡 Or run the converter with --analyze_only to generate pytorch_analysis.json")
            return
        }

    // Test using a sample module (replace with the real module)
        let sampleModule = createSampleModule()
        let mlxParams = extractMLXParameters(from: sampleModule)

    // Run the comparison
        let result = compareMapping(pytorch: pytorch, mlx: mlxParams)

    // Output the results
        print(result.summary)

        if result.isValid {
            print("🎉 Model mapping validation PASSED!")
        } else {
            print("⚠️ Model mapping validation FAILED!")
            print("💡 Check the differences above and update your MLX model implementation.")
        }
    }

    /// Create a sample module (for testing)
    private func createSampleModule() -> Module {
        // Use the real TripoSR module by default
        let config = TSRSystemConfig.tripoSRConfig
        return TSRSystem(config: config)
    }
}

/// Extension for running automatic checks at app startup
extension ModelMappingChecker {

    /// Automatically run at app startup
    static func runStartupCheck() {
        DispatchQueue.global(qos: .background).async {
            shared.performAutoCheck()
        }
    }
}
