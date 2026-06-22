// Dimension bundle for a convolution layer, populated by alloc_conv.
//
// Separating dimensions from the full ConvTaskParams keeps convTask() and
// handle_conv_inner_loop() readable — callers pass a pointer to this struct
// instead of threading a dozen separate parameters.

#pragma once

#include <cstdint>

struct ConvLayerDimensions {
  uint16_t H;  // Input feature-map height
  uint16_t W;  // Input feature-map width
  // OUTPUT_H and OUTPUT_W to handle stride != 1
  uint16_t OUTPUT_H;  // Output feature-map height after striding/padding
  uint16_t OUTPUT_W;  // Output feature-map width after striding/padding
  uint16_t kH;        // Filter kernel height
  uint16_t kW;        // Filter kernel width
  uint16_t STRIDE_H;
  uint16_t STRIDE_W;
  uint16_t CHANNEL;  // Cannot use C as a variable name here as C is a macro on
                     // MSP430 :(
  uint16_t OUTPUT_CHANNEL;  // Number of output channels (= N_FILTERS for
                            // standard conv)
  uint16_t N_FILTERS;       // Total number of filters in the weight tensor
};
