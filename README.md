# TripoSRMlx

A Swift/MLX implementation of TripoSR for fast 3D object reconstruction from single images on Apple Silicon.

## Overview

This project is a complete Swift/MLX port of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), a fast feed-forward 3D generation model. The implementation leverages Apple's MLX framework for optimized performance on M1/M2/M3 chips.

## Features

- **Complete TripoSR Pipeline**: Full implementation from image input to 3D mesh generation
- **Vision Transformer (ViT)**: DINO-based image tokenization with DINOv2 support
- **Triplane NeRF Renderer**: Triplane-based neural radiance fields with volume rendering
- **Marching Cubes**: Metal-accelerated mesh extraction from implicit surfaces
- **Multi-View Rendering**: Spherical camera system for generating views from multiple angles
- **SwiftUI Interface**: Interactive UI with model inspection and testing tools
- **Native MLX Integration**: Optimized for Apple Silicon using MLX framework

## Architecture

### System Components

The complete TripoSR pipeline consists of:

1. **Image Tokenizer** (`DINOSingleImageTokenizer`)
   - Vision Transformer (ViT) based feature extraction
   - DINOv2 architecture for robust image understanding
   - Outputs 1024-dimensional image tokens

2. **Triplane Tokenizer** (`Triplane1DTokenizer`)
   - Learnable token embeddings (8192 vocab, 1024 tokens)
   - Generates initial triplane representation tokens

3. **Transformer Backbone** (`Transformer1D`)
   - 16-layer transformer with cross-attention
   - Processes triplane tokens conditioned on image features
   - 768-dimensional hidden states with 16 attention heads

4. **Post-Processor** (`TriplaneUpsampleNetwork`)
   - Upsamples tokens to 64x64 triplane resolution
   - 40-channel output (XY, YZ, ZX planes)
   - Conv2D layers with proper deconvolution

5. **NeRF Decoder** (`TripoNeRFMLP`)
   - MLP-based density and color prediction
   - Queries triplane features at 3D positions
   - Outputs density (1 channel) and RGB color (3 channels)

6. **Volume Renderer** (`TriplaneNeRFRenderer`)
   - Ray marching with triplane feature sampling
   - Alpha compositing for final image generation
   - Configurable sampling and rendering parameters

7. **Isosurface Extractor** (`MetalMarchingCubes`)
   - Metal-accelerated marching cubes algorithm
   - Extracts triangle mesh from implicit density field
   - Supports vertex color extraction

### Usage Example

```swift
// Create TripoSR system
let tsr = TSRSystem.createTripoSRModel()

// Load and process image
let image = loadImage("input.jpg")
let output = tsr.forward([image])

// Render multi-view images
let renderedViews = tsr.render(
    sceneCodes: output.sceneCodes,
    nViews: 8,
    elevationDeg: 20.0,
    height: 256,
    width: 256
)

// Extract 3D mesh
let meshes = tsr.extractMesh(
    sceneCodes: output.sceneCodes,
    resolution: 256,
    threshold: 25.0
)
```

## Requirements

- **Platform**: macOS 13.0+ (Apple Silicon required)
- **Swift**: 5.9+
- **Dependencies**:
  - [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's machine learning framework
  - Metal for GPU acceleration (Marching Cubes)
- **Hardware**: M1/M2/M3 chip for optimal performance

## Getting Started

### 1. Installation

Clone the repository and open the Xcode project:

```bash
git clone https://github.com/yourusername/TripoSRMlx.git
cd TripoSRMlx/TripoSRMlx
open TripoSRMlx.xcodeproj
```

### 2. Model Weights

Download the pre-trained TripoSR weights and convert to MLX format:

```bash
# Download TripoSR checkpoint
huggingface-cli download stabilityai/TripoSR model.ckpt

# Convert to MLX (conversion tools coming soon)
# Currently manual weight conversion is required
```

### 3. Run the App

Build and run the SwiftUI app in Xcode. The app includes several tools:

- **TripoSR Demo**: Generate 3D models from images
- **Model Inspector**: Inspect model components and weights
- **Model Mapping Checker**: Verify PyTorch ↔ MLX parameter mapping
- **DINO Output Exporter**: Export intermediate outputs for debugging
- **Marching Cubes Test**: Test mesh extraction with synthetic data

## Testing & Validation

The project includes Python utilities for comparing MLX and PyTorch implementations:

```bash
# Compare DINO outputs between MLX and PyTorch
python model_utils/mlx_dino_output_copier.py image.jpg /tmp/mlx_dino_output.safetensors

# Compare any two safetensor files
python model_utils/safetensor_comparator.py file1.safetensors file2.safetensors
```

See [QUICK_START.md](../QUICK_START.md) for detailed testing workflows.

## Project Status

### ✅ Implemented Components

- **Complete Pipeline**: Image → ViT → Transformer → Triplane → NeRF → Mesh
- **All Major Modules**:
  - ViT/DINO image encoder with proper architecture
  - Transformer1D backbone with cross-attention
  - Triplane upsampling network
  - NeRF decoder and volume renderer
  - Metal-based marching cubes
- **SwiftUI Tools**: Interactive debugging and validation interface
- **Python Integration**: Safetensor-based comparison utilities

### ⚠️ Current Limitations

1. **Model Weights**
   - No automated PyTorch → MLX conversion pipeline yet
   - Manual weight transfer required
   - Weight compatibility needs validation

2. **Grid Sampling**
   - Simplified triplane sampling (placeholder bilinear interpolation)
   - May affect rendering quality vs. PyTorch reference

3. **Performance**
   - Not fully optimized for production use
   - Some operations use workarounds due to MLX API constraints
   - Memory usage not optimized for large batches

4. **Testing**
   - Limited numerical validation against PyTorch
   - More extensive testing needed for edge cases

### 🚧 Roadmap

- [ ] Automated weight conversion from PyTorch checkpoints
- [ ] Proper grid sampling with exact bilinear interpolation
- [ ] Comprehensive numerical validation suite
- [ ] Performance optimization and profiling
- [ ] Complete API documentation
- [ ] Example notebooks and tutorials

### 📝 Use Cases

This implementation is suitable for:

- **Research**: Exploring NeRF and 3D generation on Apple Silicon
- **Education**: Learning MLX framework and 3D vision pipelines
- **Development**: Foundation for Swift-based 3D applications
- **Experimentation**: Testing modifications to TripoSR architecture

For production use, additional validation and optimization are recommended.

## Project Structure

```
TripoSRMlx/
├── TripoSRMlx/
│   ├── Models/
│   │   ├── ViT/              # Vision Transformer components
│   │   ├── Tokenizers/       # Image & triplane tokenizers
│   │   ├── Transformer/      # Transformer backbone
│   │   ├── Processor/        # Triplane upsampling
│   │   ├── Renderer/         # NeRF rendering pipeline
│   │   ├── IsoSurface/       # Marching cubes mesh extraction
│   │   ├── Utils/            # Camera, image utilities
│   │   ├── Checker/          # Model validation tools
│   │   └── TSRSystem.swift   # Main system orchestration
│   ├── Views/                # SwiftUI interface
│   ├── ViewModels/           # View logic
│   └── ContentView.swift     # Main app entry
├── model_utils/              # Python comparison tools
│   ├── mlx_dino_output_copier.py
│   └── safetensor_comparator.py
└── QUICK_START.md           # Testing workflows (日本語)
```

## Contributing

Contributions are welcome! Areas where help is needed:

- Weight conversion automation from PyTorch
- Numerical validation against reference implementation
- Performance optimization
- Documentation and examples
- Bug fixes and testing

Please feel free to submit issues and pull requests.

## License

This project follows the same MIT License as the original TripoSR repository.

## Acknowledgments

- Original [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by Stability AI and Tripo AI
- Apple's [MLX](https://github.com/ml-explore/mlx) framework team
- Neural Radiance Fields research community
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI Research