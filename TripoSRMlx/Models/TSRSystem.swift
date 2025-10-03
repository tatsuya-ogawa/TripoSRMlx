//
//  TSRSystem.swift
//  TripoSRMlx
//
//  Main TripoSR system implementation - Swift/MLX port of system.py
//  Handles the complete pipeline from image input to 3D scene generation.
//

import Foundation
import MLX
import MLXNN
import SwiftUI
import MLXRandom
nonisolated let INPUT_IMAGE_SIZE:Int=512
nonisolated public let TRIPO_RADIUS:Float=0.87
/// Configuration for the complete TSR system
nonisolated public struct TSRSystemConfig {
    public let condImageSize: Int

    // Component configurations
    public let imageTokenizerConfig: ImageTokenizerConfig
    public let tokenizerConfig: TokenizerConfig
    public let backboneConfig: BackboneConfig
    public let postProcessorConfig: PostProcessorConfig
    public let decoderConfig: NeRFDecoderConfig
    public let rendererConfig: TriplaneNeRFRendererConfig

    public init(
        condImageSize: Int = 512,
        imageTokenizerConfig: ImageTokenizerConfig = ImageTokenizerConfig(),
        tokenizerConfig: TokenizerConfig = TokenizerConfig(),
        backboneConfig: BackboneConfig = BackboneConfig(),
        postProcessorConfig: PostProcessorConfig = PostProcessorConfig(),
        decoderConfig: NeRFDecoderConfig = NeRFDecoderConfig.tripoSRConfig,
        rendererConfig: TriplaneNeRFRendererConfig = TriplaneNeRFRendererConfig.tripoSRConfig(radius: TRIPO_RADIUS)
    ) {
        self.condImageSize = condImageSize
        self.imageTokenizerConfig = imageTokenizerConfig
        self.tokenizerConfig = tokenizerConfig
        self.backboneConfig = backboneConfig
        self.postProcessorConfig = postProcessorConfig
        self.decoderConfig = decoderConfig
        self.rendererConfig = rendererConfig
    }

    /// Create TripoSR official system configuration
    public static var tripoSRConfig: TSRSystemConfig {
        return TSRSystemConfig(
            condImageSize: 512,
            imageTokenizerConfig: ImageTokenizerConfig.tripoSROfficialConfig,
            tokenizerConfig: TokenizerConfig.tripoSROfficialConfig,
            backboneConfig: BackboneConfig.tripoSRConfig,
            postProcessorConfig: PostProcessorConfig.tripoSROfficialConfig,
            decoderConfig: NeRFDecoderConfig.tripoSRConfig,
            rendererConfig: TriplaneNeRFRendererConfig.tripoSRConfig(radius: TRIPO_RADIUS)
        )
    }
}

/// Placeholder configurations for system components (to be implemented)
nonisolated public struct ImageTokenizerConfig {
    public let outputDim: Int
    public let patchSize: Int
    public let poolerActivation: ViTPooler.ActivationType?

    public init(outputDim: Int = 1024, patchSize: Int = 14, poolerActivation: ViTPooler.ActivationType? = nil) {
        self.outputDim = outputDim
        self.patchSize = patchSize
        self.poolerActivation = poolerActivation
    }

    public static var tripoSROfficialConfig: ImageTokenizerConfig {
        return ImageTokenizerConfig(outputDim: 1024, patchSize: 14)
    }
}

nonisolated public struct TokenizerConfig {
    public let vocabSize: Int
    public let numTokens: Int

    public init(vocabSize: Int = 8192, numTokens: Int = 1024) {
        self.vocabSize = vocabSize
        self.numTokens = numTokens
    }

    public static var tripoSROfficialConfig: TokenizerConfig {
        return TokenizerConfig(vocabSize: 8192, numTokens: 1024)
    }
}

nonisolated public struct BackboneConfig {
    public let numLayers: Int
    public let attentionHeadDim: Int
    public let numHeads: Int
    public let crossAttentionDim: Int
    public let normNumGroups: Int = 32
    public let inChannels: Int = Triplane1DTokenizerNumChannels //${tokenizer.num_channels}

    public init(numLayers: Int = 16, attentionHeadDim: Int = 64, numHeads: Int = 16,crossAttentionDim:Int = 768) {
        self.numLayers = numLayers
        self.attentionHeadDim = attentionHeadDim
        self.numHeads = numHeads
        self.crossAttentionDim = crossAttentionDim
    }

    public static var tripoSRConfig: BackboneConfig {
        return BackboneConfig(numLayers: 16, attentionHeadDim: 64, numHeads: 16,crossAttentionDim: 768)
    }
}

nonisolated public struct PostProcessorConfig {
    public let outputChannels: Int
    public let outputSize: (Int, Int)

    public init(outputChannels: Int = 40, outputSize: (Int, Int) = (64, 64)) {
        self.outputChannels = outputChannels
        self.outputSize = outputSize
    }

    public static var tripoSROfficialConfig: PostProcessorConfig {
        return PostProcessorConfig(outputChannels: 40, outputSize: (64, 64))
    }
}

/// Rendering output type enumeration
//public enum RenderOutputType {
//    case mlxArray
//    case numpy
//    case uiImage
//}

/// Camera parameters for spherical rendering
public struct CameraParameters {
    public let elevationDeg: Float
    public let azimuthDeg: Float
    public let distance: Float
    public let fovyDeg: Float

    public init(elevationDeg: Float, azimuthDeg: Float, distance: Float, fovyDeg: Float) {
        self.elevationDeg = elevationDeg
        self.azimuthDeg = azimuthDeg
        self.distance = distance
        self.fovyDeg = fovyDeg
    }
}

/// TSR forward pass output
public struct TSRForwardOutput {
    public let sceneCodes: MLXArray
    public let imageTokens: MLXArray
    public let tokens: MLXArray
    public let embeddingOutput: MLXArray
    public let originalTokens: MLXArray

    public init(sceneCodes: MLXArray, imageTokens: MLXArray, tokens: MLXArray, embeddingOutput: MLXArray, originalTokens: MLXArray) {
        self.sceneCodes = sceneCodes
        self.imageTokens = imageTokens
        self.tokens = tokens
        self.embeddingOutput = embeddingOutput
        self.originalTokens = originalTokens
    }
}

/// Main TSR system class - Swift/MLX implementation
nonisolated public final class TSRSystem: Module {

    public let config: TSRSystemConfig

    // System components
    @ModuleInfo(key: "image_tokenizer")
    public var imageTokenizer: DINOSingleImageTokenizer
    @ModuleInfo(key: "tokenizer")
    public var tokenizer: Triplane1DTokenizer
    @ModuleInfo(key: "backbone")
    public var backbone: Transformer1D
    @ModuleInfo(key: "post_processor")
    public var postProcessor: TriplaneUpsampleNetwork
    @ModuleInfo(key: "decoder")
    public var decoder: TripoNeRFMLP
    public var renderer: TriplaneNeRFRenderer
    public var imagePreProcessor: ImagePreprocessor

    // Isosurface extraction helper (optional)
    private var isosurfaceHelper: MarchingCubeHelper?

    public init(config: TSRSystemConfig) {
        self.config = config

        // Initialize system components
        _imageTokenizer.wrappedValue = DINOSingleImageTokenizer(config: config.imageTokenizerConfig)
        _tokenizer.wrappedValue = Triplane1DTokenizer(config: config.tokenizerConfig)
        _backbone.wrappedValue = Transformer1D(config: config.backboneConfig)
        _postProcessor.wrappedValue = TriplaneUpsampleNetwork(config: config.postProcessorConfig)
        _decoder.wrappedValue = NeRFDecoderFactory.createDecoder(config: config.decoderConfig)
        self.renderer = TriplaneNeRFRenderer(config: config.rendererConfig)
        self.imagePreProcessor = ImagePreprocessor(targetSize: config.condImageSize)
    }

    /// Forward pass: Image → All outputs (scene codes, tokens, etc.)
    public func forward(_ images: [MLXArray]) -> TSRForwardOutput {
        // Process input images
//        let processedImages = skipImageProcessing ? images : images
//            .map { image in
//            imagePreProcessor.process(image)
//        }
        let batchedImages = stacked(images, axis: 0)

        // Image tokenization
        let imageTokens = imageTokenizer(batchedImages)

        // Generate initial tokens
        let batchSize = batchedImages.dim(0)
        let originalTokens = tokenizer.forward(batchSize: batchSize)

        // Clone original tokens before backbone processing
        let tokensForBackbone = originalTokens

        // Transformer backbone processing
        let processedTokens = backbone(tokensForBackbone, encoderHiddenStates: imageTokens.lastHiddenState)

        // Post-processing to scene codes (triplane representations)
        let sceneCodes = postProcessor(tokenizer.detokenize(processedTokens))

        return TSRForwardOutput(
            sceneCodes: sceneCodes,
            imageTokens: imageTokens.lastHiddenState,
            tokens: processedTokens,
            embeddingOutput: imageTokens.embeddingOutput,
            originalTokens: originalTokens
        )
    }

    /// Render scene codes to images from multiple viewpoints
    public func render(
        sceneCodes: MLXArray,
        nViews: Int,
        elevationDeg: Float = 0.0,
        cameraDistance: Float = 1.9,
        fovyDeg: Float = 40.0,
        height: Int = 256,
        width: Int = 256,
//        returnType: RenderOutputType = .uiImage
    ) -> [[MLXArray]] {

        // Generate spherical camera positions
        let (batchRaysO, batchRaysD) = CameraUtils.getSphericalCameras(
            nViews: nViews,
            elevationDeg: elevationDeg,
            cameraDistance: cameraDistance,
            fovyDeg: fovyDeg,
            height: height,
            width: width
        )

        var results: [[MLXArray]] = []

        // Render each scene code
        for batchIdx in 0..<sceneCodes.dim(0) {
            let sceneCode = sceneCodes[batchIdx]
            var viewResults: [MLXArray] = []

            // Render from each camera viewpoint
            for viewIdx in 0..<nViews {
                let raysO = batchRaysO[viewIdx]
                let raysD = batchRaysD[viewIdx]

                // Render the view
                let renderedImage = renderer(
                    decoder: decoder,
                    triplane: sceneCode,
                    raysO: raysO,
                    raysD: raysD
                )

                // Process output according to return type
                viewResults.append(renderedImage)
            }

            results.append(viewResults)
        }

        return results
    }

    /// Extract 3D mesh from scene codes using marching cubes
    public func extractMesh(
        sceneCodes: MLXArray,
        hasVertexColor: Bool = true,
        resolution: Int = 256,
        threshold: Float = 25.0
    ) -> [TriMesh] {

        setMarchingCubesResolution(resolution)

        var meshes: [TriMesh] = []

        for batchIdx in 0..<sceneCodes.dim(0) {
            let sceneCode = sceneCodes[batchIdx]

            // Query density at grid points
            let gridVertices = isosurfaceHelper!.gridVertices
            let scaledVertices = scaleTensor(
                gridVertices,
                from: isosurfaceHelper!.pointsRange,
                to: (-config.rendererConfig.radius, config.rendererConfig.radius)
            )

            let queryResult = renderer.queryTriplane(
                decoder: decoder,
                positions: scaledVertices,
                triplane: sceneCode
            )

            let density = queryResult.densityAct

            // Extract isosurface using marching cubes
            let (vertices, faces) = isosurfaceHelper!.extractIsosurface(-(density - threshold))

            // Scale vertices back to world coordinates
            let worldVertices = scaleTensor(
                vertices,
                from: isosurfaceHelper!.pointsRange,
                to: (-config.rendererConfig.radius, config.rendererConfig.radius)
            )

            // Extract vertex colors if requested
            var vertexColors: MLXArray? = nil
            if hasVertexColor {
                let colorResult = renderer.queryTriplane(
                    decoder: decoder,
                    positions: worldVertices,
                    triplane: sceneCode
                )
                vertexColors = colorResult.color
            }

            // Create mesh
            let mesh = TriMesh(
                vertices: worldVertices,
                faces: faces,
                vertexColors: vertexColors
            )

            meshes.append(mesh)
        }

        return meshes
    }

    /// Set marching cubes resolution for mesh extraction
    public func setMarchingCubesResolution(_ resolution: Int) {
        if let helper = isosurfaceHelper, helper.resolution == resolution {
            return
        }
        isosurfaceHelper = MarchingCubeHelper(resolution: resolution)
    }

    
}

/// Create TripoSR system with official configuration
public extension TSRSystem {
    static func createTripoSRModel() -> TSRSystem {
        return TSRSystem(config: TSRSystemConfig.tripoSRConfig)
    }
}
