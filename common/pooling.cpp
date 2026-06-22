// MaxPool and GlobalAveragePool operator implementations.
//
// MaxPool
//   Supports NHWC layout (need_nhwc2nchw == 0) and NCHW output (== 1).
//   The NHWC path tiles by [output_h, output_w, channel_block], calling
//   maxpool_patch() for each spatial position to scan the kernel window.
//   The NCHW path processes one channel at a time.
//
//   ceil_mode: when 1, output dimensions are computed with ceiling division
//   so that every input element is covered by at least one output cell.
//
// GlobalAveragePool
//   Implemented as a two-stage operation for input-channel tiling:
//   Stage 1 (handle_global_average_pool) accumulates partial row sums across
//   all W columns into an output slot of size H × CHANNEL.
//   Stage 2 (handle_global_average_pool_stage2) reads those H partial sums
//   and divides by H to produce the final CHANNEL-length vector.
//   The two-stage split is because the H × W × CHANNEL intermediate does not
//   fit in lea_buffer for large feature maps.
//
//   Integer overflow: accumulation is done in int32 (op_buffer cast to int32*).
//   MY_ASSERT catches overflow: |sum| < W * 32768 must hold.
//
// INTERMITTENT RECOVERY
//   Both MaxPool and GlobalAveragePool support intermittent recovery by calling
//   run_recovery() at the start of handle_* to skip already-written output
//   positions.

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "cnn_common.h"
#include "config.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

struct MaxPoolParams {
  uint16_t output_h;
  uint16_t output_w;
  uint16_t start_channel;
  uint16_t n_channels;
  uint8_t need_nhwc2nchw;
  uint8_t ceil_mode;
  uint8_t stride_h;
  uint8_t stride_w;
  uint16_t H;
  uint16_t W;
  uint16_t new_H;
  uint16_t new_W;
  const MaxPoolFlags* flags;
  const ParameterInfo* data;
  const ParameterInfo* output;
  Model* model;
};
static MaxPoolParams maxpool_params_obj;

enum {
  KERNEL_SHAPE_H = 0,
  KERNEL_SHAPE_W = 1,
  STRIDE_H = 0,
  STRIDE_W = 1,
};

void alloc_max_pool(Model* model, const ParameterInfo* input[],
                    ParameterInfo* output, const Node* node,
                    CurNodeFlags* node_flags, const NodeFlags*) {
  const ParameterInfo* data = input[0];

  uint16_t CHANNEL = data->dims[1];

  MaxPoolParams* maxpool_params = &maxpool_params_obj;
  maxpool_params->flags = &(node_flags->max_pool);

  maxpool_params->H = data->dims[2];
  maxpool_params->W = data->dims[3];
  maxpool_params->stride_h = maxpool_params->flags->strides[STRIDE_H];
  maxpool_params->stride_w = maxpool_params->flags->strides[STRIDE_W];
  maxpool_params->ceil_mode = node_flags->max_pool.ceil;
  if (!maxpool_params->ceil_mode) {
    maxpool_params->new_H = maxpool_params->H / maxpool_params->stride_h;
    maxpool_params->new_W = maxpool_params->W / maxpool_params->stride_w;
  } else {
    maxpool_params->new_H = (maxpool_params->H + maxpool_params->stride_h - 1) /
                            maxpool_params->stride_h;
    maxpool_params->new_W = (maxpool_params->W + maxpool_params->stride_w - 1) /
                            maxpool_params->stride_w;
  }
  maxpool_params->need_nhwc2nchw = node_flags->max_pool.nhwc2nchw;

  output->params_len =
      maxpool_params->new_H * maxpool_params->new_W * CHANNEL * sizeof(int16_t);
  output->dims[0] = 1;
  output->dims[1] = CHANNEL;
  output->dims[2] = maxpool_params->new_H;
  output->dims[3] = maxpool_params->new_W;
  if (maxpool_params->need_nhwc2nchw) {
    output->param_flags &= (~CHANNEL_LAST);
  }
}

static uint16_t maxpool_patch(MaxPoolParams* maxpool_params) {
  const uint16_t CHANNEL = maxpool_params->data->dims[1],
                 W = maxpool_params->data->dims[3];

  int16_t offset_h, offset_w;
  offset_h = W * CHANNEL;
  offset_w = CHANNEL;

  my_printf_debug("output_h=% 3d ", maxpool_params->output_h);
  my_printf_debug("output_w=% 3d ", maxpool_params->output_w);
  my_printf_debug("c=[% 3d, % 3d) ", maxpool_params->start_channel,
                  maxpool_params->start_channel + maxpool_params->n_channels);

  int16_t* const input_buffer = lea_buffer + maxpool_params->n_channels;
  int16_t* const output_buffer = lea_buffer;
  my_fill_q15(INT16_MIN, output_buffer, maxpool_params->n_channels);

  // explicitly initialize this as -Wmaybe-uninitialized may be triggered with
  // -O3 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60165
  uint16_t output_channel_offset = 0;

  for (uint16_t sH = 0;
       sH < maxpool_params->flags->kernel_shape[KERNEL_SHAPE_H]; sH++) {
    for (uint16_t sW = 0;
         sW < maxpool_params->flags->kernel_shape[KERNEL_SHAPE_W]; sW++) {
      uint16_t input_h =
                   maxpool_params->output_h * maxpool_params->stride_h + sH,
               input_w =
                   maxpool_params->output_w * maxpool_params->stride_w + sW;
      if (input_h >= maxpool_params->H || input_w >= maxpool_params->W) {
        continue;
      }
      uint16_t val_offset = input_h * offset_h + input_w * offset_w +
                            maxpool_params->start_channel;
      my_memcpy_from_param(maxpool_params->model, input_buffer,
                           maxpool_params->data, val_offset,
                           maxpool_params->n_channels * sizeof(int16_t));
      output_channel_offset = 0;
      for (uint16_t input_channel_offset = 0;
           input_channel_offset < maxpool_params->n_channels;
           input_channel_offset++) {
        int16_t val = input_buffer[input_channel_offset];
        my_printf_debug("% 6d ", val);
        if (val > output_buffer[output_channel_offset]) {
          output_buffer[output_channel_offset] = val;
        }
        output_channel_offset++;
      }
      my_printf_debug("; ");
    }
  }
  return output_channel_offset;
}

void handle_max_pool(Model* model, const ParameterInfo* input[],
                     ParameterInfo* output, const Node* node, CurNodeFlags*,
                     const NodeFlags*) {
  my_printf_debug("MaxPool!" NEWLINE);

  /* XXX: add flags; assume no padding for now */
  const ParameterInfo* data = input[0];

  MaxPoolParams* maxpool_params = &maxpool_params_obj;
  maxpool_params->data = data;
  maxpool_params->output = output;
  maxpool_params->model = model;

  const uint16_t CHANNEL = data->dims[1], OUTPUT_CHANNEL = output->dims[1];

  uint16_t output_h = 0, output_w = 0, c = 0;
  uint16_t output_offset = 0;

#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_value_offset =
      batch_start(job_index_to_offset(output, run_recovery(model, output)));
  if (first_unfinished_value_offset * sizeof(int16_t) == output->params_len) {
    // give up early, or initial_real_tile_c may be zero and results in SIGFPE
    stop_cpu_counter();
    goto finished;
  }

  uint16_t initial_c, initial_h, initial_w;

  output_offset = first_unfinished_value_offset;
  if (!maxpool_params->need_nhwc2nchw) {
    initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
    first_unfinished_value_offset /= OUTPUT_CHANNEL;
    initial_w = first_unfinished_value_offset % maxpool_params->new_W;
    first_unfinished_value_offset /= maxpool_params->new_W;
    initial_h = first_unfinished_value_offset % maxpool_params->new_H;
  } else {
    initial_w = first_unfinished_value_offset % maxpool_params->new_W;
    first_unfinished_value_offset /= maxpool_params->new_W;
    initial_h = first_unfinished_value_offset % maxpool_params->new_H;
    first_unfinished_value_offset /= maxpool_params->new_H;
    initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
  }
  output_h = initial_h;
  output_w = initial_w;
  c = initial_c;
  my_printf_debug("initial_h = %d" NEWLINE, initial_h);
  my_printf_debug("initial_w = %d" NEWLINE, initial_w);
  my_printf_debug("initial_c = %d" NEWLINE, initial_c);
  stop_cpu_counter();
#endif

  {
    if (!maxpool_params->need_nhwc2nchw) {
      // NHWC
      for (; output_h < maxpool_params->new_H; output_h++) {
        maxpool_params->output_h = output_h;
        for (; output_w < maxpool_params->new_W; output_w++) {
          uint16_t len = OUTPUT_CHANNEL - c;
          maxpool_params->output_w = output_w;
          maxpool_params->n_channels = len;
          maxpool_params->start_channel = c;
          len = maxpool_patch(maxpool_params);
          my_printf_debug("output_offset=[% 5d, % 5d) ", output_offset,
                          output_offset + len);
#if MY_DEBUG >= MY_DEBUG_VERBOSE
          my_printf_debug(" max=");
          for (uint8_t idx = 0; idx < len; idx++) {
            my_printf_debug("% 6d ", lea_buffer[idx]);
          }
          my_printf_debug(NEWLINE);
#endif
          my_memcpy_to_param(output, output_offset, lea_buffer,
                             len * sizeof(int16_t), 0, false);
#if HAWAII
          hawaii_record_footprints(model, len);
#endif
          output_offset += len;
          c = 0;
        }
        output_w = 0;
      }
      output_h = 0;
    } else {
      // NCHW
      uint8_t channel_stride = 1;
      for (; c < CHANNEL; c += channel_stride) {
        for (; output_h < maxpool_params->new_H; output_h++) {
          maxpool_params->output_h = output_h;
          maxpool_params->output_w = output_w;
          for (; output_w < maxpool_params->new_W; output_w++) {
            maxpool_params->start_channel = c;
            maxpool_params->n_channels = 1;
            uint16_t len = maxpool_patch(maxpool_params);
            if (!len) {
              my_printf_debug(NEWLINE);
              continue;
            }
            my_printf_debug("output_offset=% 5d ", output_offset);
            my_printf_debug("max=% 6d " NEWLINE, lea_buffer[0]);
            put_q15_param(output, output_offset, lea_buffer[0], false);
#if HAWAII
            if (output_offset % BATCH_SIZE ==
                BATCH_SIZE - 1) {  // last job in a batch
              write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
            }
#endif
            output_offset++;
            maxpool_params->output_w++;
          }
          output_w = 0;
        }
        output_h = 0;
      }
      c = 0;
    }
  }

  MY_ASSERT(output_offset == output->params_len / sizeof(int16_t),
            "Expect output offset %d, got %d" NEWLINE,
            output->params_len / sizeof(int16_t), output_offset);

#if INTERMITTENT
finished:
#endif

  my_printf_debug("handle_maxpool output" NEWLINE);
  if (!maxpool_params->need_nhwc2nchw) {
    dump_params_nhwc_debug(model, output, node->output_name, "MaxPool");
  } else {
    dump_params_debug(model, output, node->output_name, "MaxPool");
  }
}

void alloc_global_average_pool(Model* model, const ParameterInfo* input[],
                               ParameterInfo* output, const Node*,
                               CurNodeFlags*, const NodeFlags*) {
  const ParameterInfo* data = input[0];

  MY_ASSERT(data->dims[0] == 1);
  uint16_t output_len = data->dims[1];

  uint16_t H = data->dims[2];

  output->dims[0] = output->dims[2] = output->dims[3] = 1;
  output->dims[1] = output_len;
  output->params_len = H * output_len * sizeof(int16_t);
}

void handle_global_average_pool(Model* model, const ParameterInfo* input[],
                                ParameterInfo* output, const Node* node,
                                CurNodeFlags*, const NodeFlags*) {
  my_printf_debug("GlobalAveragePool!" NEWLINE);

  const ParameterInfo* data = input[0];

  uint32_t first_unfinished_value_offset = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  first_unfinished_value_offset =
      batch_start(job_index_to_offset(output, run_recovery(model, output)));
  stop_cpu_counter();
#endif

  uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
  uint16_t len = H * W;

  int32_t* accumulation_buffer = reinterpret_cast<int32_t*>(op_buffer);
  int16_t* data_buffer = lea_buffer;

  uint32_t vector_idx = first_unfinished_value_offset / CHANNEL,
           vector_offset = vector_idx * W;

  for (; vector_offset < len; vector_offset += W, vector_idx++) {
    memset(accumulation_buffer, 0, CHANNEL * sizeof(int32_t));

    for (uint16_t vector_idx_inner = 0; vector_idx_inner < W;
         vector_idx_inner++) {
      my_memcpy_from_param(model, data_buffer, data,
                           (vector_offset + vector_idx_inner) * CHANNEL,
                           CHANNEL * sizeof(int16_t));

      my_printf_debug("Input vector %d" NEWLINE,
                      vector_offset + vector_idx_inner);
      dump_matrix_debug(data_buffer, CHANNEL, ValueInfo(data));

      my_printf_debug("Accumulated vector" NEWLINE);
      for (uint16_t idx = 0; idx < CHANNEL; idx++) {
        accumulation_buffer[idx] += data_buffer[idx];
        my_printf_debug("% 6" PRId32 " ", accumulation_buffer[idx]);
      }
      my_printf_debug(NEWLINE NEWLINE);
    }

    for (uint16_t idx = 0; idx < CHANNEL; idx++) {
      MY_ASSERT(abs(accumulation_buffer[idx]) < W * 32768);
      data_buffer[idx] = accumulation_buffer[idx] / W;
    }

    my_printf_debug("Output vector %d, offset=%d" NEWLINE, vector_idx,
                    vector_idx * CHANNEL);
    dump_matrix_debug(data_buffer, CHANNEL, ValueInfo(data));

    my_memcpy_to_param(output, vector_idx * CHANNEL, data_buffer,
                       CHANNEL * sizeof(int16_t), /*timer_delay=*/0,
                       /*is_linear=*/0);

#if HAWAII
    hawaii_record_footprints(model, CHANNEL);
#endif
  }

  dump_params_debug(model, output, node->output_name, "GlobalAveragePool");
}

void alloc_global_average_pool_stage2(Model* model,
                                      const ParameterInfo* input[],
                                      ParameterInfo* output, const Node*,
                                      CurNodeFlags*, const NodeFlags*) {
  const ParameterInfo* data = input[0];

  uint16_t output_len = data->dims[1];

  output->dims[0] = output->dims[2] = output->dims[3] = 1;
  output->dims[1] = output_len;
  output->params_len = output_len * sizeof(int16_t);
}

void handle_global_average_pool_stage2(Model* model,
                                       const ParameterInfo* input[],
                                       ParameterInfo* output, const Node* node,
                                       CurNodeFlags*, const NodeFlags*) {
  my_printf_debug("GlobalAveragePoolMerge!" NEWLINE);

  const ParameterInfo* data = input[0];

  uint32_t first_unfinished_value_offset = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  first_unfinished_value_offset =
      batch_start(job_index_to_offset(output, run_recovery(model, output)));
  stop_cpu_counter();
#endif

  uint16_t CHANNEL = data->dims[1],
           H = data->params_len / sizeof(int16_t) / CHANNEL;

  int32_t* accumulation_buffer = reinterpret_cast<int32_t*>(op_buffer);
  int16_t* data_buffer = lea_buffer;

  if (first_unfinished_value_offset == CHANNEL) {
    goto finished;
  }

  memset(accumulation_buffer, 0, CHANNEL * sizeof(int32_t));

  for (uint32_t vector_idx = 0, vector_offset = 0; vector_idx < H;
       vector_idx++, vector_offset += CHANNEL) {
    my_memcpy_from_param(model, data_buffer, data, vector_offset,
                         CHANNEL * sizeof(int16_t));

    my_printf_debug("Input vector %d" NEWLINE, vector_idx);
    dump_matrix_debug(data_buffer, CHANNEL, ValueInfo(data));

    my_printf_debug("Accumulated vector" NEWLINE);
    for (uint16_t idx = 0; idx < CHANNEL; idx++) {
      accumulation_buffer[idx] += data_buffer[idx];
      my_printf_debug("% 6" PRId32 " ", accumulation_buffer[idx]);
    }
    my_printf_debug(NEWLINE NEWLINE);
  }

  for (uint16_t idx = 0; idx < CHANNEL; idx++) {
    data_buffer[idx] = accumulation_buffer[idx] / H;
  }

  my_printf_debug("Output vector" NEWLINE);
  dump_matrix_debug(data_buffer, CHANNEL, ValueInfo(data));

  my_memcpy_to_param(output, 0, data_buffer, CHANNEL * sizeof(int16_t),
                     /*timer_delay=*/0, /*is_linear=*/0);

#if HAWAII
  hawaii_record_footprints(model, CHANNEL);
#endif

finished:
  dump_params_debug(model, output, node->output_name, "GlobalAveragePool");
}
