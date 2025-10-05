//
//  GridSample.metal
//  TripoSRMlx
//
//  Metal compute shader for grid sampling (inference-optimized, memory-efficient)
//

#include <metal_stdlib>
using namespace metal;

/// Parameters for grid sampling
struct GridSampleParams {
    int inputBatch;
    int inputChannels;
    int inputHeight;
    int inputWidth;
    int gridBatch;
    int gridHeight;
    int gridWidth;
    bool alignCorners;
    int mode;  // 0: bilinear, 1: nearest
};

/// Bilinear interpolation for a single channel
inline float bilinearInterpolate(device const float* input,
                                  int channels,
                                  int height,
                                  int width,
                                  int n,
                                  int c,
                                  float y,
                                  float x,
                                  bool alignCorners) {
    // Denormalize coordinates from [-1, 1] to pixel coordinates
    float ix, iy;
    if (alignCorners) {
        ix = ((x + 1.0f) / 2.0f) * (width - 1);
        iy = ((y + 1.0f) / 2.0f) * (height - 1);
    } else {
        ix = ((x + 1.0f) * width - 1.0f) / 2.0f;
        iy = ((y + 1.0f) * height - 1.0f) / 2.0f;
    }

    // Get integer coordinates
    int ix0 = floor(ix);
    int iy0 = floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    // Get interpolation weights
    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    // Boundary conditions: out of bounds pixels are zero
    bool valid00 = (ix0 >= 0 && ix0 < width && iy0 >= 0 && iy0 < height);
    bool valid01 = (ix0 >= 0 && ix0 < width && iy1 >= 0 && iy1 < height);
    bool valid10 = (ix1 >= 0 && ix1 < width && iy0 >= 0 && iy0 < height);
    bool valid11 = (ix1 >= 0 && ix1 < width && iy1 >= 0 && iy1 < height);

    // Calculate base index for this batch and channel
    int baseIdx = n * channels * height * width + c * height * width;

    // Sample values (zero if out of bounds)
    float v00 = valid00 ? input[baseIdx + iy0 * width + ix0] : 0.0f;
    float v01 = valid01 ? input[baseIdx + iy1 * width + ix0] : 0.0f;
    float v10 = valid10 ? input[baseIdx + iy0 * width + ix1] : 0.0f;
    float v11 = valid11 ? input[baseIdx + iy1 * width + ix1] : 0.0f;

    // Bilinear interpolation
    return wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11);
}

/// Nearest neighbor interpolation for a single channel
inline float nearestInterpolate(device const float* input,
                                 int channels,
                                 int height,
                                 int width,
                                 int n,
                                 int c,
                                 float y,
                                 float x,
                                 bool alignCorners) {
    // Denormalize coordinates from [-1, 1] to pixel coordinates
    float ix, iy;
    if (alignCorners) {
        ix = ((x + 1.0f) / 2.0f) * (width - 1);
        iy = ((y + 1.0f) / 2.0f) * (height - 1);
    } else {
        ix = ((x + 1.0f) * width - 1.0f) / 2.0f;
        iy = ((y + 1.0f) * height - 1.0f) / 2.0f;
    }

    // Round to nearest integer
    int ix_nearest = round(ix);
    int iy_nearest = round(iy);

    // Boundary check
    if (ix_nearest < 0 || ix_nearest >= width || iy_nearest < 0 || iy_nearest >= height) {
        return 0.0f;
    }

    // Calculate index
    int baseIdx = n * channels * height * width + c * height * width;
    return input[baseIdx + iy_nearest * width + ix_nearest];
}

/// Grid sampling compute kernel
/// Input: [N, C, H, W]
/// Grid: [N, H_out, W_out, 2]  (last dimension is (x, y) in [-1, 1])
/// Output: [N, C, H_out, W_out]
kernel void gridSampleKernel(device const float* input [[buffer(0)]],
                              device const float* grid [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant GridSampleParams& params [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {

    int outWidth = params.gridWidth;
    int outHeight = params.gridHeight;
    int channels = params.inputChannels;
    int batch = params.inputBatch;

    // gid.x = width index, gid.y = height index, gid.z = batch index
    if (gid.x >= outWidth || gid.y >= outHeight || gid.z >= batch) {
        return;
    }

    int n = gid.z;
    int h = gid.y;
    int w = gid.x;

    // Get grid coordinates for this position
    // Grid layout: [N, H_out, W_out, 2]
    int gridIdx = n * outHeight * outWidth * 2 + h * outWidth * 2 + w * 2;
    float gridX = grid[gridIdx];
    float gridY = grid[gridIdx + 1];

    // Sample all channels at this grid position
    for (int c = 0; c < channels; c++) {
        float value;

        if (params.mode == 0) {
            // Bilinear interpolation
            value = bilinearInterpolate(input,
                                        channels,
                                        params.inputHeight,
                                        params.inputWidth,
                                        n,
                                        c,
                                        gridY,
                                        gridX,
                                        params.alignCorners);
        } else {
            // Nearest neighbor
            value = nearestInterpolate(input,
                                       channels,
                                       params.inputHeight,
                                       params.inputWidth,
                                       n,
                                       c,
                                       gridY,
                                       gridX,
                                       params.alignCorners);
        }

        // Write to output: [N, C, H_out, W_out]
        int outIdx = n * channels * outHeight * outWidth + c * outHeight * outWidth + h * outWidth + w;
        output[outIdx] = value;
    }
}
