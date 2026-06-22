// Dropout operator — passthrough stub for inference.
//
// Dropout is only active during training to randomly zero out activations.
// During inference the ratio must be 0, so the output equals the input.
// We reuse the input slot (INPLACE_UPDATE_OPS_MAP must include Dropout) and
// assert that the ratio is zero rather than implementing the full op.

#include <cstdint>

#include "cnn_common.h"
#include "layer-defs.h"
#include "my_debug.h"

void handle_dropout(Model* model, const ParameterInfo* input[],
                    ParameterInfo* output, const Node* node,
                    CurNodeFlags* node_flags,
                    const NodeFlags* orig_node_flags) {
  const ParameterInfo* ratio = input[1];

  int16_t ratio_val = get_q15_param(model, ratio, /*offset_in_word=*/0);

  MY_ASSERT(ratio_val == 0, "Only no-op dropout is implemented.");
}
