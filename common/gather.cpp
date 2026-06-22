// Gather operator: index-based tensor lookup along a specified axis.
//
// Semantics (ONNX Gather-13):
//   Let q = len(indices.shape).
//   output[..., i_0..i_{q-1}, ...] = data[..., indices[i_0..i_{q-1}], ...]
//   where the replaced dimension is `axis` in data.
//
// Iteration structure:
//   iterations_before_indices — product of data.shape[:axis]
//   iterations_in_indices     — product of indices.shape
//   iterations_after_indices  — product of data.shape[axis+1:]
//
// The implementation does not support intermittent recovery; Gather output
// tensors are typically small (embedding lookups, index selection).

#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "data_structures.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "op_utils.h"
#include "platform.h"

struct {
  uint16_t iterations_before_indices;
  uint16_t iterations_in_indices;
  uint16_t iterations_after_indices;
} gather_params;

void alloc_gather(Model* model, const ParameterInfo* input[],
                  ParameterInfo* output, const Node* node,
                  CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  const ParameterInfo *data = input[0], *indices = input[1];

  uint8_t axis = node_flags->gather.axis;
  uint8_t accumulated_axis = 0;

  gather_params.iterations_before_indices = 1;
  gather_params.iterations_in_indices = 1;
  gather_params.iterations_after_indices = 1;

  for (uint8_t dim_idx = 0; dim_idx < axis; dim_idx++) {
    uint16_t dim = data->dims[dim_idx];
    output->dims[dim_idx] = dim;
    gather_params.iterations_before_indices *= dim;
  }

  accumulated_axis += axis;

  for (uint8_t dim_idx = 0; dim_idx < MAX_NUM_DIMS && indices->dims[dim_idx];
       dim_idx++) {
    uint16_t dim = indices->dims[dim_idx];
    output->dims[accumulated_axis + dim_idx] = dim;
    gather_params.iterations_in_indices *= dim;
    accumulated_axis++;
  }

  for (uint8_t dim_idx = axis + 1;
       dim_idx < MAX_NUM_DIMS && data->dims[dim_idx]; dim_idx++) {
    uint16_t dim = data->dims[dim_idx];
    output->dims[accumulated_axis + dim_idx - 1] = dim;
    gather_params.iterations_after_indices *= dim;
  }

  recalculate_params_len(output);
}

// https://onnx.ai/onnx/operators/onnx__Gather.html#gather-13
void handle_gather(Model* model, const ParameterInfo* input[],
                   ParameterInfo* output, const Node* node,
                   CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  // let k = indices[i_{0}, ..., i_{q-1}]
  // If axis = 0, then output[       i_{0}, ..., i_{q-1}, j_{0}, j_{1}, ...,
  // j_{r-2}] = input[k , j_{0},    j_{1}, ..., j_{r-2}] If axis = 1, then
  // output[j_{0}, i_{0}, ..., i_{q-1},        j_{1}, ..., j_{r-2}] = input[
  // j_{0}, k, j_{1}, ..., j_{r-2}]

  const ParameterInfo *data = input[0], *indices = input[1];

  MY_ASSERT(indices->param_flags & INTEGER);

  uint8_t axis = node_flags->gather.axis;
  uint16_t input_iterations_in_indices = data->dims[axis];

  uint16_t iterations_before_indices = gather_params.iterations_before_indices,
           iterations_in_indices = gather_params.iterations_in_indices,
           iterations_after_indices = gather_params.iterations_after_indices;

  int16_t* gather_buffer = lea_buffer;

  for (uint16_t offset_before_indices = 0;
       offset_before_indices < iterations_before_indices;
       offset_before_indices++) {
    for (uint16_t offset_in_indices = 0;
         offset_in_indices < iterations_in_indices; offset_in_indices++) {
      int16_t k = get_q15_param(model, indices, offset_in_indices);

      uint32_t input_value_offset = offset_before_indices *
                                        input_iterations_in_indices *
                                        iterations_after_indices +
                                    k * iterations_after_indices;

      uint32_t output_value_offset =
          offset_before_indices * iterations_in_indices *
              iterations_after_indices +
          offset_in_indices * iterations_after_indices;

      my_memcpy_from_param(model, gather_buffer, data, input_value_offset,
                           iterations_after_indices * sizeof(int16_t));

      my_printf_debug(
          "offset_before_indices=%d, offset_in_indices=%d, k=%d, "
          "input_value_offset=%d, output_value_offset=%d" NEWLINE,
          offset_before_indices, offset_in_indices, k, input_value_offset,
          output_value_offset);

      my_printf_debug("gather_buffer" NEWLINE);
      dump_matrix_debug(gather_buffer, iterations_after_indices,
                        ValueInfo(output));

      my_memcpy_to_param(output, output_value_offset, gather_buffer,
                         iterations_after_indices * sizeof(int16_t),
                         /*timer_delay=*/0, /*is_linear=*/false);
    }
  }

  dump_params_debug(model, output, node->output_name, "Gather");
}
