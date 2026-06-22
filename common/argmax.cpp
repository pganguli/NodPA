// ArgMax operator: index of the maximum element along a given axis.
//
// alloc_arg_max sets up output dims and marks the output as INTEGER (the
// values are indices, not Q15 activations).  The axis dimension is either
// collapsed to size 1 (keepdims) or removed entirely.
//
// handle_arg_max has two paths:
//   General case — iterates over all (before_axis, after_axis) combinations,
//     scanning each slice along the axis with a scalar loop.
//   Fast path  — when num_indices_after_axis == 1, the data along the axis is
//     contiguous in memory, so a single my_memcpy_from_param + my_max_q15 call
//     is sufficient; no per-element NVM access needed.
//
// Note: ArgMax does not support intermittent recovery.  The output is small
// (a single index per example) and recomputing it is cheap enough.

#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"

void alloc_arg_max(Model* model, const ParameterInfo* input[],
                   ParameterInfo* output, const Node* node,
                   CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  const ParameterInfo* data = input[0];

  const ArgMaxNodeFlags& flags = node_flags->arg_max;
  if (node_flags->arg_max.keepdims) {
    output->dims[flags.axis] = 1;
  } else {
    uint8_t dim_idx = flags.axis;
    const uint8_t max_dims = sizeof(output->dims) / sizeof(output->dims[0]);
    // Remove the target axis and move all axes after it
    while (dim_idx + 1 < max_dims) {
      output->dims[dim_idx] = output->dims[dim_idx + 1];
      if (!output->dims[dim_idx]) {
        break;
      }
      dim_idx++;
    }
    output->dims[max_dims - 1] = 0;
  }

  output->params_len /= data->dims[flags.axis];

  output->param_flags |= INTEGER;
}

// Inspired by ChatGPT
void handle_arg_max(Model* model, const ParameterInfo* input[],
                    ParameterInfo* output, const Node* node,
                    CurNodeFlags* node_flags,
                    const NodeFlags* orig_node_flags) {
  const ParameterInfo* data = input[0];

  const uint8_t axis = node_flags->arg_max.axis;

  uint16_t num_indices_before_axis = 1, num_indices_at_axis = data->dims[axis],
           num_indices_after_axis = 1;

  for (uint8_t dim_idx = 0; dim_idx < axis; dim_idx++) {
    num_indices_before_axis *= data->dims[dim_idx];
  }
  for (uint8_t dim_idx = axis + 1; dim_idx < MAX_NUM_DIMS; dim_idx++) {
    if (!data->dims[dim_idx]) {
      break;
    }
    num_indices_after_axis *= data->dims[dim_idx];
  }

  if (num_indices_after_axis != 1) {
    /* general case */
    for (uint16_t idx_before_axis = 0;
         idx_before_axis < num_indices_before_axis; idx_before_axis++) {
      for (uint16_t idx_after_axis = 0; idx_after_axis < num_indices_after_axis;
           idx_after_axis++) {
        int16_t cur_max_val = INT16_MIN;
        uint16_t cur_max_idx = 0;

        uint32_t base_input_value_offset =
            idx_before_axis * num_indices_at_axis * num_indices_after_axis +
            idx_after_axis;
        uint32_t output_value_offset =
            idx_before_axis * num_indices_after_axis + idx_after_axis;

        for (uint16_t idx_at_axis = 0; idx_at_axis < num_indices_at_axis;
             idx_at_axis++) {
          uint32_t input_value_offset =
              base_input_value_offset + idx_at_axis * num_indices_after_axis;
          int16_t cur_val = get_q15_param(model, data, input_value_offset);

          if (cur_val > cur_max_val) {
            cur_max_val = cur_val;
            cur_max_idx = idx_at_axis;
          }

          my_printf_debug(
              "idx_before_axis=%d, idx_at_axis=%d, idx_after_axis=%d, "
              "cur_val=%d, cur_max_idx=%d" NEWLINE,
              idx_before_axis, idx_at_axis, idx_after_axis, cur_val,
              cur_max_idx);
        }

        my_printf_debug("output_value_offset=%d, cur_max_idx=%d" NEWLINE,
                        output_value_offset, cur_max_idx);

        put_q15_param(output, output_value_offset, cur_max_idx,
                      /*is_linear=*/false);
      }
    }
  } else {
    /* a special case where input data are continuous */
    int16_t* cur_val_buffer = lea_buffer;

    for (uint16_t idx_before_axis = 0;
         idx_before_axis < num_indices_before_axis; idx_before_axis++) {
      int16_t cur_max_val = INT16_MIN;
      uint16_t cur_max_idx = 0;

      uint32_t base_input_value_offset = idx_before_axis * num_indices_at_axis;
      uint32_t output_value_offset = idx_before_axis;

      my_memcpy_from_param(model, cur_val_buffer, data, base_input_value_offset,
                           num_indices_at_axis * sizeof(int16_t));

      my_max_q15(cur_val_buffer, num_indices_at_axis, &cur_max_val,
                 &cur_max_idx);

      my_printf_debug("output_value_offset=%d, cur_max_idx=%d" NEWLINE,
                      output_value_offset, cur_max_idx);

      put_q15_param(output, output_value_offset, cur_max_idx,
                    /*is_linear=*/false);
    }
  }

  dump_params_debug(model, output, node->output_name, "ArgMax");
}
