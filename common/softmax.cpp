// Softmax operator — two-stage fixed-point implementation.
//
// Standard softmax is numerically unstable for large inputs because exp() can
// overflow.  Stage 1 subtracts the row maximum before computing exponentials:
//   exp(x_i - max(x)) is always in (0, 1].
//
// TWO-STAGE SPLIT
//   Stage 1 (handle_softmax):
//     For each row of softmax_length elements:
//       1. Find the maximum value with my_max_q15 and shift the row down by
//          subtracting it via my_offset_q15.
//       2. Compute exp() in floating point element-by-element and convert back
//          to Q15 using the output scale.
//       3. Write exp values to the output NVM slot.
//     The scale must be pre-determined by transform.py so all exp values fit
//     in Q15 without overflow.
//
//   Stage 2 (handle_softmax_stage2):
//     Reads back the exp values row by row, sums them (softmax_sum), then
//     multiplies each by softmax_sum_reciprocal = (1 << 30) / softmax_sum.
//     The result is normalized to scale 1.0 (SCALE_ONE), which is set in
//     alloc_softmax_stage2.  The 1<<30 shift avoids losing precision when
//     converting the reciprocal to Q15.
//
// Intermittent recovery is supported in both stages.  Stage 1 recovers to
// the row start (cannot resume mid-row because the max is needed for all
// elements).  Stage 2 recovers to the row start for the same reason.

#include <cmath>
#include <cstdint>

#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

void alloc_softmax(Model* model, const ParameterInfo* input[],
                   ParameterInfo* output, const Node* node,
                   CurNodeFlags* node_flags, const NodeFlags*) {}

void handle_softmax(Model* model, const ParameterInfo* input[],
                    ParameterInfo* output, const Node* node,
                    CurNodeFlags* node_flags, const NodeFlags*) {
  int axis = node_flags->softmax.axis;

  const ParameterInfo* X = input[0];
  const uint16_t softmax_length = X->dims[axis];

  uint32_t softmax_vector_idx = 0;
  uint32_t softmax_num_vectors =
      output->params_len / sizeof(int16_t) / softmax_length;
  uint32_t data_offset = 0;
  uint16_t softmax_idx = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_job_idx = run_recovery(model, output);
  data_offset =
      batch_start(job_index_to_offset(output, first_unfinished_job_idx));
  stop_cpu_counter();

  softmax_vector_idx = data_offset / softmax_length;
  softmax_idx = data_offset % softmax_length;
  // needs to start from the row head
  data_offset -= softmax_idx;

#endif

  for (; softmax_vector_idx < softmax_num_vectors;
       softmax_vector_idx++, data_offset += softmax_length) {
    my_memcpy_from_param(model, lea_buffer, X, data_offset,
                         softmax_length * sizeof(int16_t));

    // avoid exponential overflow and underflow
    // https://www.cnblogs.com/guoyaohua/p/8900683.html
    int16_t max_val = 0;
    uint16_t max_val_idx = 0;
    my_max_q15(lea_buffer, softmax_length, &max_val, &max_val_idx);
    my_offset_q15(lea_buffer, -(max_val), lea_buffer, softmax_length);

    for (; softmax_idx < softmax_length; softmax_idx++) {
      // exponentials
      float val = q15_to_float(lea_buffer[softmax_idx], ValueInfo(output));
      val = std::exp(val);
      int16_t exp_val =
          static_cast<int16_t>(val / output->scale.toFloat() * (1 << 15));

      put_q15_param(output, data_offset + softmax_idx, exp_val,
                    /*is_linear=*/false);
#if HAWAII
      write_hawaii_layer_footprint(model->layer_idx, 1);
#endif
    }
    softmax_idx = 0;
  }

  dump_params_debug(model, output, node->output_name, "Softmax");
}

void alloc_softmax_stage2(Model* model, const ParameterInfo* input[],
                          ParameterInfo* output, const Node* node,
                          CurNodeFlags* node_flags, const NodeFlags*) {
  // the output scale is always 1 (16384 * 2**1 / 32768, see Scale::toFloat()
  // function), as scales are cancelled out after normalization
  output->scale.fract = 16384;
  output->scale.shift = 1;
}

void handle_softmax_stage2(Model* model, const ParameterInfo* input[],
                           ParameterInfo* output, const Node* node,
                           CurNodeFlags* node_flags, const NodeFlags*) {
  int axis = node_flags->softmax.axis;

  const ParameterInfo* X = input[0];
  const uint16_t softmax_length = X->dims[axis];

  uint32_t softmax_vector_idx = 0;
  uint32_t softmax_num_vectors =
      output->params_len / sizeof(int16_t) / softmax_length;
  uint32_t data_offset = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_job_idx = run_recovery(model, output);
  data_offset =
      batch_start(job_index_to_offset(output, first_unfinished_job_idx));
  stop_cpu_counter();

  softmax_vector_idx = data_offset / softmax_length;
  // needs to start from the row head
  data_offset = softmax_vector_idx * softmax_length;
#endif

  for (; softmax_vector_idx < softmax_num_vectors;
       softmax_vector_idx++, data_offset += softmax_length) {
    my_memcpy_from_param(model, lea_buffer, X, data_offset,
                         softmax_length * sizeof(int16_t));

    int32_t softmax_sum = 0;  // the denominator in softmax equation
    for (uint16_t softmax_idx = 0; softmax_idx < softmax_length;
         softmax_idx++) {
      softmax_sum += lea_buffer[softmax_idx];
    }

    // softmax_sum * (2 ** -15) = 1 / (softmax_sum_reciprocal * (2 ** -15))
    int32_t softmax_sum_reciprocal = (1LL << 30) / softmax_sum;

    // normalize the row
    my_scale_q15(lea_buffer, softmax_sum_reciprocal, /*shift=*/0, lea_buffer,
                 softmax_length);

    my_memcpy_to_param(output, data_offset, lea_buffer,
                       softmax_length * sizeof(int16_t), /*timer_delay=*/0,
                       /*is_linear=*/false);
#if HAWAII
    write_hawaii_layer_footprint(model->layer_idx, softmax_length);
#endif
  }

  dump_params_debug(model, output, node->output_name, "SoftmaxStage2");
}
