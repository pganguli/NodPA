// Slice operator: extract a contiguous sub-tensor along one axis.
//
// Only single-axis slicing is implemented (axes[0] selects which dimension).
// The slice range is [start, end) along that dimension; other dimensions
// are copied in full.
//
// handle_slice iterates over the 3-D input (idx0, idx1, idx2) and copies
// matching elements element-by-element to the output.  This is intentionally
// simple — Slice is only used for lightweight operations (e.g., extracting a
// time window) where the cost of scalar NVM reads is acceptable.
//
// Intermittent recovery is supported; the output offset is restored from the
// footprint and the three indices are back-computed from it.

#include <cstddef>
#include <cstdint>

#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "op_utils.h"
#include "platform.h"

void alloc_slice(struct Model* model, const struct ParameterInfo* input[],
                 struct ParameterInfo* output, const struct Node* node,
                 CurNodeFlags*, const NodeFlags*) {
  const ParameterInfo *X = input[0], *start = input[1], *end = input[2],
                      *axes = input[3];

  uint16_t input_start = get_int64_param(start, 0);
  uint16_t input_end = get_int64_param(end, 0);
  uint16_t input_axes = get_int64_param(axes, 0);

  for (uint8_t dim_idx = 0; dim_idx < 3; dim_idx++) {
    if (dim_idx == input_axes) {
      output->dims[dim_idx] = input_end - input_start;
    } else {
      output->dims[dim_idx] = X->dims[dim_idx];
    }
  }

  recalculate_params_len(output);
}

void handle_slice(struct Model* model, const struct ParameterInfo* input[],
                  struct ParameterInfo* output, const struct Node* node,
                  CurNodeFlags*, const NodeFlags*) {
  my_printf_debug("Slice!" NEWLINE);

  const ParameterInfo *X = input[0], *start = input[1], *end = input[2];
  uint16_t input_start = get_int64_param(start, 0);
  uint16_t input_end = get_int64_param(end, 0);

  uint32_t output_offset = 0;

  uint16_t idx0 = input_start, idx1 = 0, idx2 = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_job_idx = run_recovery(model, output);
  output_offset =
      batch_start(job_index_to_offset(output, first_unfinished_job_idx));
  stop_cpu_counter();
#endif
  uint32_t tmp = output_offset;
  idx2 = tmp % X->dims[2];
  tmp /= X->dims[2];
  idx1 = tmp % X->dims[1];
  tmp /= X->dims[1];
  idx0 += tmp;
  my_printf_debug("output_offset=%d, idx0=%d, idx1=%d, idx2=%d" NEWLINE,
                  output_offset, idx0, idx1, idx2);

  for (; idx0 < input_end; idx0++) {
    for (; idx1 < X->dims[1]; idx1++) {
      for (; idx2 < X->dims[2]; idx2++) {
        uint32_t input_offset =
            idx0 * X->dims[1] * X->dims[2] + idx1 * X->dims[2] + idx2;
        int16_t input_val = get_q15_param(model, X, input_offset);
        put_q15_param(output, output_offset, input_val, /*is_linear=*/false);
        output_offset++;
#if HAWAII
        my_printf_debug("output_offset=%d, idx0=%d, idx1=%d, idx2=%d" NEWLINE,
                        output_offset, idx0, idx1, idx2);
        write_hawaii_layer_footprint(model->layer_idx, /*n_jobs=*/1);
#endif
      }
      idx2 = 0;
    }
    idx1 = 0;
  }
}
