// Fully-connected / Gemm layer dimension descriptor.
//
// FcLayerDimensions holds the logical matrix sizes after tiling is applied.
// The actual Gemm computation is Y = A * B (+ optional bias in last rows of
// B).  A is [A_rows × A_cols], B is [A_cols × B_cols], so Y is [A_rows ×
// B_cols].

#pragma once

#include <cstdint>

struct FcLayerDimensions {
  uint16_t A_rows;  // Number of input vectors (batch or spatial positions)
  uint16_t A_cols;  // Input feature width = output feature width of prev layer
  uint16_t B_cols;  // Number of output neurons (columns of the weight matrix)
};
