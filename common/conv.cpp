// Convolution and ConvMerge operator implementations.
//
// TILING STRATEGY
// ---------------
// A 2-D convolution with N input channels may not fit into the LEA scratch
// buffer (lea_buffer) in one shot.  The computation is therefore tiled along
// two dimensions:
//
//   Input-channel tiling (input_tile_c):
//     The input channels are split into tiles of input_tile_c channels each.
//     Each tile pass produces partial sums for all output spatial positions and
//     all output channels, stored in an intermediate NVM slot in NWHC order
//     (transposed from the usual NHWC to make channel-wise access contiguous).
//     ConvMerge (handle_conv_stage2) sums these partial results and re-orders
//     back to NHWC in a second NVM slot.
//
//   Output-channel tiling (output_tile_c):
//     Within each input-channel tile, output channels are processed in groups
//     of output_tile_c.  Multiple filters are interleaved into the filter
//     buffer and a single matrix-multiply produces output_tile_c values at
//     once.
//
// FILTER/INPUT LAYOUT IN lea_buffer
// ----------------------------------
//   [0, inputs_len)                  — input feature-map tile (im2col rows)
//   [inputs_len, ...-filter_offset*n_filters)  — before-transpose area
//   [..., LEA_BUFFER_SIZE - OUTPUT_LEN - pState_len)  — interleaved filters
//   [LEA_BUFFER_SIZE - OUTPUT_LEN - pState_len, ...)  — matrix-mpy output
//   [LEA_BUFFER_SIZE - pState_len, LEA_BUFFER_SIZE)   — ARM CMSIS pState
//
// The bias trick: for grouped convolution (group == 1 in the standard path)
// the bias is absorbed into the filter by appending a −bias sentinel at the
// end of each filter row, while the last column of each im2col input row
// contains −0x8000 (Q15 –1.0).  The matrix multiply then automatically adds
// the bias without a separate pass.
//
// DYNAMIC CHANNEL PRUNING
// -----------------------
// When a pruning mask tensor is present (DYNAMIC_DNN_APPROACH != 0), each
// input or output channel tile is checked against the mask before computation.
// Below-threshold tiles are skipped entirely and their NVM footprint slots are
// updated to account for the skipped jobs, so that recovery works correctly.

#include "conv.h"

#include <cinttypes>  // for PRId32
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "cnn_common.h"
#include "config.h"
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "layers.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

#define ON_DEMAND_TILE_LOADING 1

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) { return a < b ? a : b; }

static inline int16_t int16_max(int16_t a, int16_t b) { return a > b ? a : b; }

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
  Model* model;
  const ParameterInfo* conv_input;
  const ParameterInfo* conv_filter;
  const ParameterInfo* conv_bias;
  const ParameterInfo* conv_channel_pruning_mask;
  ParameterInfo* output;
  CurNodeFlags* flags;
  const NodeFlags* orig_flags;

  /* aux vars remaining constant for a conv layer */
  ConvLayerDimensions layer_dims;
  uint16_t pState_len;
  uint8_t group;
  uint16_t input_tile_c;
  uint16_t input_tile_c_offset;
  uint16_t input_tile_c_index;
  uint16_t cur_input_tile_c;
  uint16_t cur_filter_tile_c;
  uint16_t n_tiles_c;
  uint16_t dest_offset;
  uint16_t filter_offset;
  uint8_t truncated;

  uint16_t filter_idx;
  uint16_t filter_tile_index;
  // (h, w) for left-top corner of each input window
  int16_t input_h;
  int16_t input_w;
  int16_t input_h_first, input_h_last;
  int16_t input_w_first, input_w_last;
  int16_t* filter_buffer_addr;
  int16_t cached_filter_idx;
  uint16_t cached_input_tile_c_offset;
} ConvTaskParams;

static ConvTaskParams conv_params_obj;

// for group convolution
int16_t* const biases = op_buffer;

static void convTask(int16_t cur_input_h, const ConvLayerDimensions* layer_dims,
                     ConvTaskParams* conv_params) {
  int16_t output_tile_c = conv_params->flags->conv.output_tile_c;
  int16_t cur_output_tile_c;
  if (conv_params->filter_idx >=
      layer_dims->N_FILTERS - layer_dims->N_FILTERS % output_tile_c) {
    cur_output_tile_c = layer_dims->N_FILTERS - conv_params->filter_idx;
  } else {
    cur_output_tile_c = output_tile_c - conv_params->filter_idx % output_tile_c;
  }
  my_printf_debug("cur_output_tile_c = %d" NEWLINE, cur_output_tile_c);
  MY_ASSERT(cur_output_tile_c > 0);

  int16_t n_filters = cur_output_tile_c;
  int16_t values_to_preserve = n_filters;
  int16_t channel_offset_c = conv_params->filter_idx;
  uint16_t output_h = (cur_input_h - conv_params->input_h_first) /
                      layer_dims->STRIDE_H,
           output_w = (conv_params->input_w - conv_params->input_w_first) /
                      layer_dims->STRIDE_W;
  // use NWHC so that output is written continuously on the address space
  uint32_t cur_output_data_offset =
      static_cast<uint32_t>(layer_dims->OUTPUT_W) * layer_dims->OUTPUT_H *
          (conv_params->input_tile_c_index * layer_dims->OUTPUT_CHANNEL) +  // n
      static_cast<uint32_t>(output_w) * layer_dims->OUTPUT_H *
          layer_dims->OUTPUT_CHANNEL +                                // w
      static_cast<uint32_t>(output_h) * layer_dims->OUTPUT_CHANNEL +  // h
      channel_offset_c;                                               // c

  int16_t* const matrix_mpy_results =
      lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->pState_len;

  /* copy filter data */
  if (conv_params->cached_filter_idx != conv_params->filter_idx ||
      conv_params->cached_input_tile_c_offset !=
          conv_params->input_tile_c_offset) {
    int16_t* filter_tmp =
        matrix_mpy_results -
        (conv_params->filter_offset + 1) / 2 * 2;  // before transpose
    conv_params->filter_buffer_addr =
        filter_tmp - conv_params->filter_offset * n_filters;
    my_fill_q15(0, conv_params->filter_buffer_addr,
                conv_params->filter_offset * n_filters);

    uint16_t fill_length = conv_params->filter_offset;
    my_fill_q15(0, filter_tmp, fill_length);
    uint16_t filter_len = layer_dims->kH * layer_dims->kW * layer_dims->CHANNEL;
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
      uint16_t filter_src_offset = (conv_params->filter_idx + idx) * filter_len;
      my_printf_debug("Copying filter %d" NEWLINE,
                      conv_params->filter_idx + idx);
      if (conv_params->cur_filter_tile_c == layer_dims->CHANNEL &&
          layer_dims->kW * layer_dims->CHANNEL == conv_params->dest_offset) {
        uint16_t cur_filter_src_offset =
            filter_src_offset + conv_params->input_tile_c_offset;
        uint16_t buffer_size =
            conv_params->cur_filter_tile_c * layer_dims->kH * layer_dims->kW;
        my_memcpy_from_param(conv_params->model, filter_tmp,
                             conv_params->conv_filter, cur_filter_src_offset,
                             sizeof(int16_t) * buffer_size);
        my_printf_debug(
            "[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
            cur_filter_src_offset, cur_filter_src_offset + buffer_size,
            filter_tmp - lea_buffer, filter_tmp + buffer_size - lea_buffer);
      } else {
        for (uint16_t h = 0; h < layer_dims->kH; h++) {
          int16_t* filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
          uint16_t cur_filter_src_offset =
              filter_src_offset + h * layer_dims->kW * layer_dims->CHANNEL +
              conv_params->input_tile_c_offset;
          if (conv_params->cur_filter_tile_c == layer_dims->CHANNEL) {
            uint16_t buffer_size =
                conv_params->cur_filter_tile_c * layer_dims->kW;
            my_printf_debug("[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
                            cur_filter_src_offset,
                            cur_filter_src_offset + buffer_size,
                            filter_dest_ptr - lea_buffer,
                            filter_dest_ptr + buffer_size - lea_buffer);
            my_memcpy_from_param(
                conv_params->model, filter_dest_ptr, conv_params->conv_filter,
                cur_filter_src_offset, sizeof(int16_t) * buffer_size);
          } else {
            for (uint16_t w = 0; w < layer_dims->kW; w++) {
              my_printf_debug(
                  "[%d, %d) => lea_buffer + [%ld, %ld)" NEWLINE,
                  cur_filter_src_offset,
                  cur_filter_src_offset + conv_params->cur_filter_tile_c,
                  filter_dest_ptr - lea_buffer,
                  filter_dest_ptr + conv_params->cur_filter_tile_c -
                      lea_buffer);
              my_memcpy_from_param(
                  conv_params->model, filter_dest_ptr, conv_params->conv_filter,
                  cur_filter_src_offset,
                  sizeof(int16_t) * conv_params->cur_filter_tile_c);
              filter_dest_ptr += conv_params->cur_filter_tile_c;
              cur_filter_src_offset += layer_dims->CHANNEL;
            }
          }
        }
      }
      int16_t last_elem = 0;
      if (conv_params->group == 1) {
        // XXX: why is this needed? Should already be zero with my_fill_q15
        // above
        last_elem = 0;
      }
      int16_t bias_val = 0;
      if (conv_params->input_tile_c_index == 0) {
        if (conv_params->conv_bias) {
          // convert int16_t to int32_t first as on MSP430, registers are 20 bit
          // while there are only 16 bits when int16_t is converted to uint16_t
          // If the dividend is negative, the quotient is wrong
          int32_t bias_val_i32 = static_cast<int32_t>(
              get_q15_param(conv_params->model, conv_params->conv_bias,
                            conv_params->filter_idx + idx));
          if (!(conv_params->flags->general_flags & INPUT_1_SCALE)) {
            bias_val = bias_val_i32 / conv_params->conv_input->scale.toFloat();
          } else {
            bias_val = bias_val_i32;
          }
        }
      }
      last_elem += bias_val;
      if (conv_params->group == 1) {
        filter_tmp[conv_params->filter_offset - 1] = -last_elem;
      } else {
        biases[idx] = last_elem;
      }

      uint16_t channel = idx;
      my_interleave_q15(filter_tmp, channel, n_filters,
                        conv_params->filter_buffer_addr,
                        conv_params->filter_offset);
    }

    conv_params->cached_filter_idx = conv_params->filter_idx;
    conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
  }

  int16_t* filter_buffer_addr = conv_params->filter_buffer_addr;

  int16_t* input_buffer_addr =
      lea_buffer + (cur_input_h - conv_params->input_h) *
                       conv_params->dest_offset * conv_params->group;

  uint16_t A_rows, A_cols, B_rows, B_cols;
  A_rows = 1;
  A_cols = conv_params->filter_offset * conv_params->group;
  B_rows = conv_params->filter_offset;
  B_cols = n_filters;
  MY_ASSERT(input_buffer_addr + A_rows * A_cols <= filter_buffer_addr);
  if (conv_params->group == 1) {
    MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr,
                      filter_buffer_addr, matrix_mpy_results,
                      conv_params->output, cur_output_data_offset,
                      values_to_preserve, conv_params->pState_len);
  } else {
    MY_ASSERT(B_rows * B_cols <= OUTPUT_LEN);
    if (n_filters == conv_params->group) {
      int16_t* cur_matrix_mpy_results = matrix_mpy_results + n_filters;
      my_vector_mult_q15(input_buffer_addr, filter_buffer_addr,
                         matrix_mpy_results, B_rows * B_cols);
      for (uint16_t row = 1; row < B_rows; row++) {
        my_add_q15(matrix_mpy_results, cur_matrix_mpy_results,
                   matrix_mpy_results, conv_params->group);
        cur_matrix_mpy_results += n_filters;
      }
    } else {
      int16_t *cur_input_buffer_addr =
                  input_buffer_addr + (conv_params->group - n_filters),
              *cur_filter_buffer_addr = filter_buffer_addr,
              *cur_matrix_mpy_results = matrix_mpy_results;
      for (uint16_t row = 0; row < B_rows; row++) {
        my_vector_mult_q15(cur_input_buffer_addr, cur_filter_buffer_addr,
                           cur_matrix_mpy_results, n_filters);
        if (row) {
          my_add_q15(matrix_mpy_results, cur_matrix_mpy_results,
                     matrix_mpy_results, n_filters);
        }
        cur_input_buffer_addr += conv_params->group;
        cur_filter_buffer_addr += n_filters;
        cur_matrix_mpy_results += n_filters;
      }
    }
    my_add_q15(matrix_mpy_results, biases, matrix_mpy_results, n_filters);
    my_memcpy_to_param(conv_params->output, cur_output_data_offset,
                       matrix_mpy_results, n_filters * sizeof(int16_t), 0,
                       true);
  }

  /* START dump data */
  my_printf_debug("input_h=%d" NEWLINE, cur_input_h);
  my_printf_debug("filter_idx=");
#if MY_DEBUG >= MY_DEBUG_VERBOSE
  for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
    my_printf_debug("%d ", conv_params->filter_idx + idx);
    MY_ASSERT(conv_params->filter_idx + idx < layer_dims->N_FILTERS);
  }
#endif
  my_printf_debug("output_h=%d output_w=%d" NEWLINE, output_h, output_w);

  my_printf_debug("input" NEWLINE);
  dump_matrix_debug(input_buffer_addr, A_rows, A_cols,
                    ValueInfo(conv_params->conv_input, nullptr));
  my_printf_debug(
      "filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN - "
      "pState_len - %d" NEWLINE,
      static_cast<int>(lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN -
                       conv_params->pState_len - filter_buffer_addr));
  my_printf_debug("filter" NEWLINE);
  dump_matrix_debug(filter_buffer_addr, B_rows, B_cols,
                    ValueInfo(conv_params->conv_filter, nullptr));
  if (conv_params->group != 1) {
    my_printf_debug("biases" NEWLINE);
    dump_matrix_debug(biases, 1, n_filters,
                      ValueInfo(conv_params->conv_filter, nullptr));
  }

  my_printf_debug("matrix_mpy_results" NEWLINE);
  dump_matrix_debug(matrix_mpy_results, A_rows, B_cols,
                    ValueInfo(conv_params->output));
  my_printf_debug(NEWLINE);

  compare_vm_nvm(matrix_mpy_results, conv_params->model, conv_params->output,
                 cur_output_data_offset, values_to_preserve);
  /* END dump data */

  my_printf_debug("output_data offset = %" PRIu32 NEWLINE,
                  cur_output_data_offset);

  MY_ASSERT(cur_output_data_offset + n_filters <
            INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);

#if HAWAII

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
    DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
  uint8_t num_completed_units = 0;
  if (conv_params->conv_channel_pruning_mask) {
    my_printf_debug(
        "input_w=%d STRIDE_W=%d input_w_last=%d cur_input_h=%d STRIDE_H=%d "
        "input_h_last=%d" NEWLINE,
        conv_params->input_w, conv_params->layer_dims.STRIDE_W,
        conv_params->input_w_last, cur_input_h,
        conv_params->layer_dims.STRIDE_H, conv_params->input_h_last);
    if (conv_params->flags->conv.pruning_target == PRUNING_OUTPUT_CHANNELS) {
      if ((conv_params->input_w + conv_params->layer_dims.STRIDE_W >
           conv_params->input_w_last) &&
          (cur_input_h + conv_params->layer_dims.STRIDE_H >
           conv_params->input_h_last)) {
        num_completed_units = 2;
      }
    } else {
      my_printf_debug(
          "filter_tile_index+1=%d output_tile_c=%d N_FILTERS=%d" NEWLINE,
          conv_params->filter_tile_index + 1,
          conv_params->flags->conv.output_tile_c, layer_dims->N_FILTERS);
      if ((conv_params->input_w + conv_params->layer_dims.STRIDE_W >
           conv_params->input_w_last) &&
          (cur_input_h + conv_params->layer_dims.STRIDE_H >
           conv_params->input_h_last) &&
          ((conv_params->filter_tile_index + 1) *
               conv_params->flags->conv.output_tile_c >=
           layer_dims->N_FILTERS)) {
        num_completed_units = 1;
      }
    }
  }
  if (num_completed_units) {
    write_hawaii_layer_two_footprints(
        conv_params->model->layer_idx, FootprintOffset::COMPUTATION_UNIT_INDEX,
        num_completed_units, FootprintOffset::NUM_COMPLETED_JOBS,
        values_to_preserve);
  } else
#endif
  {
    write_hawaii_layer_footprint(conv_params->model->layer_idx,
                                 values_to_preserve);
  }
#endif
}

static inline uint16_t load_input_vector(uint32_t src_addr, int16_t* dest_addr,
                                         uint16_t len,
                                         const ConvTaskParams* conv_params) {
  my_printf_debug("Load %d IFM values [%d, %d) => lea_buffer + [%ld, %ld) ",
                  len, src_addr, static_cast<int>(src_addr + len),
                  dest_addr - lea_buffer, dest_addr + len - lea_buffer);
  int16_t* memcpy_dest_addr = nullptr;
  uint16_t loaded_len = 0;

  MY_ASSERT(len != 0);

  {
    memcpy_dest_addr = dest_addr;
    loaded_len = len;
  }
  my_memcpy_from_param(conv_params->model, memcpy_dest_addr,
                       conv_params->conv_input, src_addr,
                       len * sizeof(int16_t));

#if MY_DEBUG >= MY_DEBUG_VERBOSE
  for (uint16_t idx = 0; idx < loaded_len; idx++) {
    my_printf_debug("%d ", dest_addr[idx]);
  }
  my_printf_debug(NEWLINE);
#endif
  return loaded_len;
}

static void load_ifm_tile_row(Model* model,
                              const ConvLayerDimensions* layer_dims,
                              const ConvTaskParams* conv_params,
                              int32_t h_start, int32_t h_end) {
  /* copy input data, row by row */

  /* int32_t instead of int16_t as TI's compiler cannot handle negative
   * offsets correctly. The expression `ptr + (int16_t)(-2)` is
   * compiled as:
   * 1. -2 is represented as 0x00FFFE (general registers are 24-bit long).
   *    Assume this value is stored in R11.
   * 2. RLAM.A #1,R11  # multiply by 2 to transform the offset for int16_t
   *    to the difference of addresses.
   * In step 2, R11 becomes 0x01FFFC, while it should be -4, or 0x00FFFC,
   * and thus the resultant address is offset by 0x10000.
   */
  int32_t w_start = int16_max(0, conv_params->input_w),
          w_end = int16_min(conv_params->input_w + layer_dims->kW - 1,
                            layer_dims->W - 1);
  int16_t* dest;

  dest = lea_buffer;

  dest += (h_start - conv_params->input_h) * conv_params->dest_offset *
          conv_params->group;

  my_printf_debug("h_start=%" PRId32 " ", h_start);

  uint16_t cur_input_tile_c = conv_params->cur_input_tile_c;
  uint8_t im2col_channel_offset = cur_input_tile_c;
  my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                  static_cast<int>(dest - lea_buffer));
  uint16_t cur_input_channel = layer_dims->CHANNEL;
  int16_t input_src_offset;
  uint8_t transposed = conv_params->conv_input->param_flags & TRANSPOSED;
  if (transposed) {
    input_src_offset = (w_start * layer_dims->H + h_start) * cur_input_channel *
                       conv_params->group;
  } else {
    input_src_offset = (h_start * layer_dims->W + w_start) * cur_input_channel *
                       conv_params->group;
  }
  input_src_offset += conv_params->input_tile_c_offset;
  int16_t input_src_vertical_offset;
  if (transposed) {
    input_src_vertical_offset = cur_input_channel * conv_params->group;
  } else {
    input_src_vertical_offset =
        layer_dims->W * cur_input_channel * conv_params->group;
  }
  uint16_t input_row_len =
      (w_end - w_start + 1) * cur_input_tile_c * conv_params->group;

  if (input_src_vertical_offset ==
      conv_params->dest_offset * conv_params->group) {
    // XXX: make them compatible
    MY_ASSERT(!ON_DEMAND_TILE_LOADING);
    int16_t* dest_addr = dest + (w_start - conv_params->input_w) *
                                    im2col_channel_offset * conv_params->group;
    uint16_t input_tile_len = (h_end - h_start + 1) * input_row_len;
    load_input_vector(input_src_offset, dest_addr, input_tile_len, conv_params);
  } else {
    for (int32_t h = h_start; h <= h_end; h++) {
      int16_t* dest_addr = dest + (w_start - conv_params->input_w) *
                                      im2col_channel_offset *
                                      conv_params->group;
      uint32_t src_addr = input_src_offset;
      if ((cur_input_tile_c == cur_input_channel) && !transposed) {
        load_input_vector(src_addr, dest_addr, input_row_len, conv_params);
      } else {
        for (int32_t w = w_start; w <= w_end; w++) {
          load_input_vector(src_addr, dest_addr,
                            cur_input_tile_c * conv_params->group, conv_params);
          dest_addr += im2col_channel_offset * conv_params->group;
          if (transposed) {
            src_addr += layer_dims->H * cur_input_channel * conv_params->group;
          } else {
            src_addr += cur_input_channel * conv_params->group;
          }
        }
      }

      dest += conv_params->dest_offset * conv_params->group;
      input_src_offset += input_src_vertical_offset;
    }
  }
}

static uint16_t handle_conv_inner_loop(Model* model,
                                       const ConvLayerDimensions* layer_dims,
                                       ConvTaskParams* conv_params) {
  int8_t field_size = (layer_dims->kH - 1) / 2;
  int16_t max_n_filters = conv_params->flags->conv.output_tile_c;
  // 1 additional filters for values before transpose
  uint16_t inputs_buffer_end = LEA_BUFFER_SIZE - OUTPUT_LEN -
                               conv_params->pState_len -
                               (max_n_filters + 1) * conv_params->filter_offset;
  // Align tile_h with the stride size to make sure the next tile starts from
  // the correct place
  uint16_t tile_h =
      MIN_VAL(
          inputs_buffer_end / (conv_params->group * conv_params->dest_offset) -
              2 * field_size,
          layer_dims->H) /
      conv_params->layer_dims.STRIDE_H * conv_params->layer_dims.STRIDE_H;
  uint16_t inputs_len = (tile_h + 2 * field_size) *
                        (conv_params->group * conv_params->dest_offset);
  MY_ASSERT(inputs_len <
            LEA_BUFFER_SIZE - OUTPUT_LEN -
                conv_params->pState_len);  // make sure no overflow occurs in
                                           // the previous line

  my_printf_debug("Reinitialize input buffer" NEWLINE
                  "tile_h = %d, inputs_len = %d" NEWLINE,
                  tile_h, inputs_len);

  my_fill_q15(0, lea_buffer, inputs_len);
  // XXX: write multipliers on demand?
  if (conv_params->group == 1) {
    uint16_t bias_multipler_offset = conv_params->dest_offset - 1;
    while (bias_multipler_offset < inputs_len) {
      lea_buffer[bias_multipler_offset] = -0x8000;  // _Q15(-1.0)
      bias_multipler_offset += conv_params->dest_offset;
    }
  }

  int16_t max_input_h =
      MIN_VAL(conv_params->input_h + tile_h - 1, conv_params->input_h_last);
  int32_t max_h_end = int16_min(conv_params->input_h + tile_h +
                                    (layer_dims->kH - layer_dims->STRIDE_H),
                                layer_dims->H) -
                      1;
  int32_t h_start = int16_max(conv_params->input_h, 0);
#if ON_DEMAND_TILE_LOADING
  int32_t h_end = int16_min(h_start + layer_dims->kH, layer_dims->H) - 1;
#else
  int32_t h_end = max_h_end;
  load_ifm_tile_row(model, layer_dims, conv_params, h_start, h_end);
#endif
  for (int16_t cur_input_h = conv_params->input_h; cur_input_h <= max_input_h;
       cur_input_h += layer_dims->STRIDE_H) {
#if ON_DEMAND_TILE_LOADING
    load_ifm_tile_row(model, layer_dims, conv_params, h_start, h_end);
    h_start = h_end + 1;
    h_end = int16_min(h_end + layer_dims->STRIDE_H, max_h_end);
#endif

    my_printf_debug("Loaded inputs" NEWLINE);
    dump_matrix_debug(lea_buffer, inputs_len,
                      ValueInfo(conv_params->conv_input, nullptr));

    // filter_idx is set to initial_c in handle_conv
    convTask(cur_input_h, layer_dims, conv_params);
    // reset here for further processing
    conv_params->filter_idx =
        conv_params->filter_tile_index * conv_params->flags->conv.output_tile_c;
  }
  return tile_h;
}

void alloc_conv(Model* model, const ParameterInfo* input[],
                ParameterInfo* output, const Node* node,
                CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  const ParameterInfo *conv_input = input[0], *conv_filter = input[1];

  /* input: N x C x H x W, filter: M x C x kH x kW */
  const uint16_t CHANNEL = conv_filter->dims[1], H = conv_input->dims[2],
                 W = conv_input->dims[3];
  uint16_t OUTPUT_CHANNEL = conv_filter->dims[0];

  ConvTaskParams* conv_params = &conv_params_obj;
  ConvLayerDimensions* layer_dims = &(conv_params->layer_dims);

  conv_params->model = model;
  conv_params->flags = node_flags;
  conv_params->orig_flags = orig_node_flags;

  layer_dims->kH = conv_filter->dims[2];
  layer_dims->kW = conv_filter->dims[3];

  layer_dims->STRIDE_H = conv_params->flags->conv.strides[0];
  layer_dims->STRIDE_W = conv_params->flags->conv.strides[1];

  conv_params->pState_len = orig_node_flags->conv.pState_len;
  conv_params->group = conv_params->flags->conv.group;

  const uint8_t* pads = conv_params->flags->conv.pads;
  enum { PAD_H_BEGIN = 0, PAD_W_BEGIN = 1, PAD_H_END = 2, PAD_W_END = 3 };
  conv_params->input_h_first = -pads[PAD_H_BEGIN];
  conv_params->input_w_first = -pads[PAD_W_BEGIN];
  conv_params->input_h_last = H + pads[PAD_H_END] - layer_dims->kH;
  conv_params->input_w_last = W + pads[PAD_W_END] - layer_dims->kW;

  layer_dims->OUTPUT_H =
      (conv_params->input_h_last - conv_params->input_h_first) /
          layer_dims->STRIDE_H +
      1;
  layer_dims->OUTPUT_W =
      (conv_params->input_w_last - conv_params->input_w_first) /
          layer_dims->STRIDE_W +
      1;

  MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1] * conv_params->group);
  MY_ASSERT(conv_params->group == 1 ||
            conv_params->group == conv_input->dims[1]);

  conv_params->input_tile_c = conv_params->flags->conv.input_tile_c;
  conv_params->n_tiles_c = upper_gauss(CHANNEL, conv_params->input_tile_c);

  my_printf_debug("input_tile_c=%d, output_tile_c=%d" NEWLINE,
                  conv_params->input_tile_c,
                  conv_params->flags->conv.output_tile_c);

  /* XXX: extend flags; assume dilation=(1, 1) for now */
  output->params_len = static_cast<uint32_t>(conv_params->n_tiles_c) *
                       layer_dims->OUTPUT_H * layer_dims->OUTPUT_W *
                       OUTPUT_CHANNEL * sizeof(int16_t);
  output->dims[0] = 1;
  output->dims[1] = OUTPUT_CHANNEL;
  output->dims[2] = layer_dims->OUTPUT_H;
  output->dims[3] = layer_dims->OUTPUT_W;
  if (conv_params->flags->general_flags & INPUT_1_SCALE) {
    output->scale = conv_filter->scale;
  } else {
    output->scale = conv_input->scale * conv_filter->scale;
  }
  output->param_flags |= TRANSPOSED;

#if ENABLE_DEMO_COUNTERS
  if (need_reset() && !model->run_counter) {
    node_flags->cumulative_jobs = get_counter(offsetof(Counters, total_jobs));
    commit_node_flags(node_flags);
    my_printf("cumulative_jobs=%d" NEWLINE, node_flags->cumulative_jobs);
  }
#endif
}

static uint32_t conv_layer_jobs(const ConvTaskParams* conv_params,
                                uint32_t slice_size_input_channel_tiling,
                                uint32_t num_jobs_in_unit) {
  // Not using input_tile_c_index, as that is for calculated output offsets and
  // is incremented only for active units
  return conv_params->input_tile_c_offset / conv_params->input_tile_c *
             slice_size_input_channel_tiling +
         conv_params->filter_tile_index * num_jobs_in_unit;
}

static void report_conv_progress(const ConvTaskParams* conv_params,
                                 uint32_t slice_size_input_channel_tiling,
                                 uint32_t num_jobs_in_unit) {
  report_progress(conv_params->flags->cumulative_jobs +
                  conv_layer_jobs(conv_params, slice_size_input_channel_tiling,
                                  num_jobs_in_unit));
}

void handle_conv(Model* model, const ParameterInfo* input[],
                 ParameterInfo* output, const Node* node,
                 CurNodeFlags* node_flags, const NodeFlags*) {
  const ParameterInfo *conv_input = input[0], *conv_filter = input[1],
                      *conv_bias = (node->inputs_len >= 3) ? input[2] : nullptr;

  const ParameterInfo* conv_channel_pruning_mask =
      (node->inputs_len >= 4) ? input[3] : nullptr;

  my_printf_debug("Conv!" NEWLINE);

  /* input: N x C x H x W, filter: M x C x kH x kW */
  const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                 CHANNEL = conv_filter->dims[1];

  ConvTaskParams* conv_params = &conv_params_obj;
  ConvLayerDimensions* layer_dims = &(conv_params->layer_dims);

  my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

  conv_params->conv_input = conv_input;
  conv_params->conv_filter = conv_filter;
  conv_params->conv_bias = conv_bias;
  conv_params->conv_channel_pruning_mask = conv_channel_pruning_mask;
  conv_params->output = output;
  conv_params->filter_buffer_addr = NULL;
  conv_params->cached_filter_idx = -1;
  layer_dims->H = H;
  layer_dims->W = W;

  layer_dims->CHANNEL = CHANNEL;
  layer_dims->OUTPUT_CHANNEL = output->dims[1];
  layer_dims->N_FILTERS = conv_filter->dims[0];

  conv_params->input_tile_c_offset = 0;
  conv_params->input_tile_c_index = 0;
  conv_params->input_h = conv_params->input_h_first;
  conv_params->input_w = conv_params->input_w_first;
  conv_params->filter_tile_index = 0;
  conv_params->filter_idx = 0;

  uint16_t slice_size_input_channel_tiling =
      layer_dims->OUTPUT_W * layer_dims->OUTPUT_H * layer_dims->OUTPUT_CHANNEL;

#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
#if HAWAII
  dump_footprints_debug(model->layer_idx);

  const Footprint* footprint = get_versioned_data<Footprint>(model->layer_idx);
  unshuffle_footprint_values(footprint);
  uint32_t first_unfinished_job_idx =
      unshuffled_footprint.values[FootprintOffset::NUM_COMPLETED_JOBS];
#else
  uint32_t first_unfinished_job_idx = run_recovery(model, output);
#endif

#if HAWAII &&                                                         \
    (DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
     DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS)
  uint32_t dynamic_dnn_skipped_jobs =
      unshuffled_footprint.values[FootprintOffset::NUM_SKIPPED_JOBS];

  if (conv_channel_pruning_mask &&
      conv_params->flags->conv.pruning_target == PRUNING_OUTPUT_CHANNELS) {
    first_unfinished_job_idx += dynamic_dnn_skipped_jobs;
  }
#endif

  uint32_t first_unfinished_value_offset =
      batch_start(job_index_to_offset(output, first_unfinished_job_idx));

  fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

  uint32_t first_unfinished_value_offset_with_skipped_jobs =
      first_unfinished_value_offset;
#if HAWAII &&                                                         \
    (DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
     DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS)
  if (conv_channel_pruning_mask &&
      conv_params->flags->conv.pruning_target == PRUNING_INPUT_CHANNELS) {
    // For dynamic channel pruning, the number of skipped jobs equals to the
    // offset for those jobs
    first_unfinished_value_offset_with_skipped_jobs += dynamic_dnn_skipped_jobs;
  }
#endif

  uint16_t cur_output_tile_c = conv_params->flags->conv.output_tile_c;

  conv_params->input_tile_c_index =
      first_unfinished_value_offset / slice_size_input_channel_tiling;
  {
    conv_params->input_tile_c_offset =
        first_unfinished_value_offset_with_skipped_jobs /
        slice_size_input_channel_tiling * conv_params->input_tile_c;
#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_ONE_INDICATOR
    uint16_t original_input_tile_c_offset = conv_params->input_tile_c_offset,
             non_skipped_channels = 0;
    if (conv_channel_pruning_mask &&
        conv_params->flags->conv.pruning_target == PRUNING_INPUT_CHANNELS) {
      // Go through the decision map to derive the actual job index from the
      // preserved footprint
      my_memcpy_from_param(model, lea_buffer, conv_channel_pruning_mask,
                           /*offset_in_word=*/0, CHANNEL * sizeof(int16_t));
      for (uint16_t idx = 0; idx < CHANNEL; idx++) {
        my_printf_debug("input channel=%d channel_mask=%d... ", idx,
                        lea_buffer[idx]);
        if (lea_buffer[idx] == 0) {
          conv_params->input_tile_c_offset++;
        } else {
          non_skipped_channels++;
        }
        if (non_skipped_channels >= original_input_tile_c_offset) {
          break;
        }
      }
    }
#endif
  }
  first_unfinished_value_offset %= slice_size_input_channel_tiling;

  conv_params->filter_tile_index =
      (first_unfinished_value_offset % layer_dims->OUTPUT_CHANNEL) /
      cur_output_tile_c;
  conv_params->filter_idx =
      first_unfinished_value_offset % layer_dims->OUTPUT_CHANNEL;

  first_unfinished_value_offset /= layer_dims->OUTPUT_CHANNEL;

  conv_params->input_w += first_unfinished_value_offset / layer_dims->OUTPUT_H *
                          conv_params->layer_dims.STRIDE_W;
  first_unfinished_value_offset %= layer_dims->OUTPUT_H;

  conv_params->input_h +=
      first_unfinished_value_offset * conv_params->layer_dims.STRIDE_H;

  my_printf_debug("initial input_tile_c_index = %d" NEWLINE,
                  conv_params->input_tile_c_index);
  my_printf_debug("initial input_tile_c_offset = %d" NEWLINE,
                  conv_params->input_tile_c_offset);
  my_printf_debug("initial output H = %d" NEWLINE,
                  (conv_params->input_h - conv_params->input_h_first) /
                      conv_params->layer_dims.STRIDE_H);
  my_printf_debug("initial output W = %d" NEWLINE,
                  (conv_params->input_w - conv_params->input_w_first) /
                      conv_params->layer_dims.STRIDE_W);
  my_printf_debug("initial output C = %d" NEWLINE, conv_params->filter_idx);
  // = happens when all values are finished
#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED
  MY_ASSERT(conv_params->input_tile_c_index <= conv_params->n_tiles_c);
#endif
  stop_cpu_counter();
#endif

  // recalculate n_tiles_c, as tile sizes may be changed during dynamic
  // reconfiguration
  conv_params->n_tiles_c = upper_gauss(CHANNEL, conv_params->input_tile_c);
  output->params_len = static_cast<uint32_t>(conv_params->n_tiles_c) *
                       layer_dims->OUTPUT_H * layer_dims->OUTPUT_W *
                       layer_dims->OUTPUT_CHANNEL * sizeof(int16_t);

  int16_t input_channels = conv_filter->dims[1];

  uint32_t num_jobs_in_unit = conv_params->flags->conv.output_tile_c *
                              conv_params->layer_dims.OUTPUT_H *
                              conv_params->layer_dims.OUTPUT_W;

#if HAWAII && DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS && \
    ENABLE_COUNTERS
  conv_params->cur_input_tile_c =
      MIN_VAL(conv_params->input_tile_c,
              input_channels - conv_params->input_tile_c_offset);
  add_demo_counter(offsetof(Counters, re_execution_macs),
                   conv_params->cur_input_tile_c);
#endif

  for (; conv_params->input_tile_c_offset < input_channels;
       conv_params->input_tile_c_offset += conv_params->input_tile_c) {
    conv_params->cur_input_tile_c =
        MIN_VAL(conv_params->input_tile_c,
                input_channels - conv_params->input_tile_c_offset);

#if !FORCE_STATIC_NETWORKS
    if (conv_channel_pruning_mask &&
        conv_params->flags->conv.pruning_target == PRUNING_INPUT_CHANNELS) {
      int16_t channel_mask;
      while (conv_params->input_tile_c_offset < input_channels) {
        int16_t pruning_threshold = conv_params->flags->conv.pruning_threshold;
#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
    DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
        uint16_t computation_unit_index =
            unshuffled_footprint.values[COMPUTATION_UNIT_INDEX];
#else
        uint16_t computation_unit_index = conv_params->input_tile_c_offset;
#endif
        channel_mask = get_q15_param(model, conv_channel_pruning_mask,
                                     computation_unit_index);

        my_printf_debug(
            "input_tile_c_offset=%d computation_unit_index=%d "
            "channel_mask=%d... ",
            conv_params->input_tile_c_offset, computation_unit_index,
            channel_mask);
        MY_ASSERT(conv_params->input_tile_c_offset == computation_unit_index);

        Scale inverse_scale = SCALE_ONE / conv_channel_pruning_mask->scale;
        my_scale_q15(&pruning_threshold, inverse_scale.fract,
                     inverse_scale.shift, &pruning_threshold, 1);

        if (abs(channel_mask) >= pruning_threshold) {
          my_printf_debug("running" NEWLINE);
          add_counter(offsetof(Counters, num_processed_units), 1);
          break;
        }
        my_printf_debug("skipping" NEWLINE);

        report_conv_progress(conv_params, slice_size_input_channel_tiling,
                             num_jobs_in_unit);

        conv_params->input_tile_c_offset += conv_params->input_tile_c;

        add_counter(offsetof(Counters, num_skipped_units), 1);
        add_counter(offsetof(Counters, num_skipped_jobs),
                    slice_size_input_channel_tiling);
#if HAWAII && DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_FINE_GRAINED
        // for FD, skipped MACs will be re-executed
        add_demo_counter(
            offsetof(Counters, re_execution_macs),
            slice_size_input_channel_tiling * conv_params->cur_input_tile_c);
#endif

#if ENABLE_COUNTERS
        num_skipped_jobs_since_boot += slice_size_input_channel_tiling;
#endif

#if HAWAII &&                                                         \
    (DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
     DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS)
        write_hawaii_layer_two_footprints(
            conv_params->model->layer_idx, FootprintOffset::NUM_SKIPPED_JOBS,
            slice_size_input_channel_tiling,
            FootprintOffset::COMPUTATION_UNIT_INDEX, 1);
#endif
      }

      if (conv_params->input_tile_c_offset >= input_channels) {
        break;
      }
    }
#endif

#if DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED
    MY_ASSERT(conv_params->input_tile_c_index < conv_params->n_tiles_c);
#endif

    conv_params->cur_filter_tile_c = conv_params->cur_input_tile_c;
    my_printf_debug("cur_input_tile_c = %d" NEWLINE,
                    conv_params->cur_input_tile_c);
    conv_params->dest_offset = layer_dims->kW * conv_params->cur_input_tile_c;
    if (conv_params->group == 1) {
      // +1 for bias
      conv_params->dest_offset++;
    }
    /* MSP430 LEA requires length to be even */
    conv_params->truncated =
        (conv_params->dest_offset / 2 * 2 != conv_params->dest_offset);
    if (conv_params->truncated && conv_params->group == 1) {
      // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
      // also odd, so dummy values are needed between slices to make
      // addresses even.
      // a dummy value for each slice (kW * CHANNEL q15 values)
      conv_params->dest_offset++;
    }
    conv_params->filter_offset = layer_dims->kH * conv_params->dest_offset;

    while (true) {
      bool skip_current_output_channel = false;
#if !FORCE_STATIC_NETWORKS
      if (conv_params->conv_channel_pruning_mask &&
          conv_params->flags->conv.pruning_target == PRUNING_OUTPUT_CHANNELS) {
        int16_t channel_masks[2];
        int16_t pruning_threshold = conv_params->flags->conv.pruning_threshold;

#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
    DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
        uint16_t computation_unit_index =
            unshuffled_footprint.values[COMPUTATION_UNIT_INDEX];
#else
        uint16_t computation_unit_index = conv_params->filter_idx;
#endif
        my_memcpy_from_param(model, channel_masks,
                             conv_params->conv_channel_pruning_mask,
                             computation_unit_index, 2 * sizeof(int16_t));

        my_printf_debug(
            "filter_idx=%d computation_unit_index=%d channel_masks=[%d, "
            "%d]... ",
            conv_params->filter_idx, computation_unit_index, channel_masks[0],
            channel_masks[1]);
        MY_ASSERT(conv_params->filter_idx == computation_unit_index);

        Scale inverse_scale = SCALE_ONE / conv_channel_pruning_mask->scale;
        my_scale_q15(&pruning_threshold, inverse_scale.fract,
                     inverse_scale.shift, &pruning_threshold, 1);

        if (abs(channel_masks[0]) < pruning_threshold &&
            abs(channel_masks[1]) < pruning_threshold) {
          my_printf_debug("skipping" NEWLINE);
          skip_current_output_channel = true;
        } else {
          my_printf_debug("running" NEWLINE);
        }
      }
#endif

      if (!skip_current_output_channel) {
        for (; conv_params->input_w <= conv_params->input_w_last;
             conv_params->input_w += conv_params->layer_dims.STRIDE_W) {
          for (; conv_params->input_h <= conv_params->input_h_last;) {
            conv_params->input_h +=
                handle_conv_inner_loop(model, layer_dims, conv_params);
          }
          conv_params->input_h = conv_params->input_h_first;
        }

        report_conv_progress(conv_params, slice_size_input_channel_tiling,
                             num_jobs_in_unit);

        if (conv_params->flags->conv.pruning_target ==
            PRUNING_OUTPUT_CHANNELS) {
          add_counter(offsetof(Counters, num_processed_units), 2);
        }
        add_counter(offsetof(Counters, num_processed_jobs), num_jobs_in_unit);
      } else {
        report_conv_progress(conv_params, slice_size_input_channel_tiling,
                             num_jobs_in_unit);

        add_counter(offsetof(Counters, num_skipped_units), 2);
        add_counter(offsetof(Counters, num_skipped_jobs), num_jobs_in_unit);
#if HAWAII && DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_FINE_GRAINED
        add_demo_counter(offsetof(Counters, re_execution_macs),
                         num_jobs_in_unit * conv_params->cur_input_tile_c);
#endif

#if ENABLE_COUNTERS
        num_skipped_jobs_since_boot += num_jobs_in_unit;
#endif

#if HAWAII &&                                                         \
    (DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || \
     DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS)
        write_hawaii_layer_two_footprints(
            conv_params->model->layer_idx, FootprintOffset::NUM_SKIPPED_JOBS,
            num_jobs_in_unit, FootprintOffset::COMPUTATION_UNIT_INDEX, 2);
#endif
      }

      conv_params->input_w = conv_params->input_w_first;
      conv_params->filter_tile_index++;
      if (conv_params->filter_tile_index *
              conv_params->flags->conv.output_tile_c >=
          layer_dims->N_FILTERS) {
        break;
      }
      conv_params->filter_idx = conv_params->filter_tile_index *
                                conv_params->flags->conv.output_tile_c;
    }
    conv_params->filter_idx = conv_params->filter_tile_index = 0;

    conv_params->input_tile_c_index++;
  }

  output->params_len = static_cast<uint32_t>(conv_params->input_tile_c_index) *
                       layer_dims->OUTPUT_H * layer_dims->OUTPUT_W *
                       layer_dims->OUTPUT_CHANNEL * sizeof(int16_t);

#if ENABLE_DEMO_COUNTERS
  if (need_reset() && !model->run_counter) {
    add_demo_counter(
        offsetof(Counters, total_jobs),
        conv_layer_jobs(conv_params, slice_size_input_channel_tiling,
                        num_jobs_in_unit));
  }
#endif

#if MY_DEBUG >= MY_DEBUG_LAYERS
  my_printf_debug("handle_conv output" NEWLINE);
  dump_params_nhwc_debug(model, output, node->output_name, "Conv");
  if (conv_params->flags->conv.pruning_target == PRUNING_OUTPUT_CHANNELS) {
    char output_name[NODE_NAME_LEN];
    strcpy(output_name, node->output_name);
    char* replaced = strstr(output_name, ":stage1");
    strcpy(replaced, ":mask");
    dump_params_debug(model, conv_channel_pruning_mask, output_name,
                      "ConvChannelMask");

    strcpy(replaced, ":thres");  // pruning threshold, use abbreviation here to
                                 // avoid buffer overflow
    dump_matrix(&conv_params->flags->conv.pruning_threshold, /*len=*/1,
                ValueInfo(1.0f), output_name, "ConvChannelPruningThreshold");
  }
#endif
}

void alloc_conv_stage2(Model* model, const ParameterInfo* input[],
                       ParameterInfo* output, const Node*, CurNodeFlags*,
                       const NodeFlags*) {
  const ParameterInfo* data = input[0];

  uint16_t OUTPUT_CHANNEL = data->dims[1], OUTPUT_H = data->dims[2],
           OUTPUT_W = data->dims[3];

  output->params_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(int16_t);
  output->param_flags &= (~TRANSPOSED);
}

void handle_conv_stage2(Model* model, const ParameterInfo* input[],
                        ParameterInfo* output, const Node* node, CurNodeFlags*,
                        const NodeFlags*) {
  // Do not use conv_params here as its intialization in alloc_conv and
  // handle_conv might be skipped if the Conv node has finished.
  const ParameterInfo *data = input[0], *bias = input[2];
  uint16_t OUTPUT_CHANNEL = data->dims[1], OUTPUT_H = data->dims[2],
           OUTPUT_W = data->dims[3];

  my_printf_debug("ConvMerge!" NEWLINE);

  uint8_t n_tiles_c = data->params_len / sizeof(int16_t) /
                      (OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W);

  uint32_t tiling_results_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;

  uint16_t chunk_len = OUTPUT_CHANNEL;
  if (n_tiles_c) {
    chunk_len = MIN_VAL(chunk_len, LEA_BUFFER_SIZE / n_tiles_c) / 2 * 2;
  }
  uint16_t output_h = 0, output_w = 0, chunk_offset = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_job_idx = run_recovery(model, output);
  uint32_t first_unfinished_value_offset =
      batch_start(job_index_to_offset(output, first_unfinished_job_idx));

  MY_ASSERT(chunk_len * n_tiles_c < LEA_BUFFER_SIZE);

  // value offset = output_h * OUTPUT_W * chunk_len + output_w * chunk_len +
  // chunk_offset;
  chunk_offset = first_unfinished_value_offset % chunk_len;
  first_unfinished_value_offset /= chunk_len;
  output_w = first_unfinished_value_offset % OUTPUT_W;
  first_unfinished_value_offset /= OUTPUT_W;
  output_h = first_unfinished_value_offset;
  stop_cpu_counter();
#endif

  // Here IFM and OFM have different data layouts as I do the conversion in this
  // handler
  uint32_t input_offset = output_w * OUTPUT_H * OUTPUT_CHANNEL +
                          output_h * OUTPUT_CHANNEL + chunk_offset;  // NWHC
  uint32_t output_offset = output_h * OUTPUT_W * OUTPUT_CHANNEL +
                           output_w * OUTPUT_CHANNEL + chunk_offset;  // NHWC

  // A special case - when all channels are pruned
  if (!n_tiles_c) {
    my_printf_debug("All channels are skipped - appending biases" NEWLINE);
    if (bias) {
      my_memcpy_from_param(model, lea_buffer, bias, /*offset_in_word=*/0,
                           chunk_len * sizeof(int16_t));
      output->scale = bias->scale;
    } else {
      my_fill_q15(0, lea_buffer, chunk_len);
      output->scale = SCALE_ONE;
    }
  }

  for (; output_h < OUTPUT_H;) {
    for (; output_w < OUTPUT_W; output_w++) {
      uint16_t real_chunk_len = chunk_len - chunk_offset;
      my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);

      if (n_tiles_c) {
        for (uint16_t input_tile_c_index = 0; input_tile_c_index < n_tiles_c;
             input_tile_c_index++) {
          int16_t* to_add = lea_buffer + input_tile_c_index * chunk_len;
          uint32_t cur_input_offset =
              input_tile_c_index * tiling_results_len + input_offset;
          my_memcpy_from_param(model, to_add, data, cur_input_offset,
                               real_chunk_len * sizeof(int16_t));
          my_printf_debug(
              NEWLINE
              "Input offset %d, input tile %d, output offset %d" NEWLINE,
              cur_input_offset, input_tile_c_index, output_offset);
          my_printf_debug("Added chunk" NEWLINE);
          dump_matrix_debug(to_add, real_chunk_len, ValueInfo(data));
          if (input_tile_c_index != 0) {
            my_add_q15(lea_buffer, to_add, lea_buffer, real_chunk_len);
          }
        }
      }

      my_memcpy_to_param(output, output_offset, lea_buffer,
                         real_chunk_len * sizeof(int16_t), 0, true);
#if HAWAII
      hawaii_record_footprints(model, real_chunk_len);
#endif
      output_offset += real_chunk_len;
      input_offset += OUTPUT_H * OUTPUT_CHANNEL - chunk_offset;  // NWHC
      chunk_offset = 0;
    }
    output_w = 0;
    output_h++;
    input_offset =
        output_h *
        OUTPUT_CHANNEL;  // NWHC, where only output_h is nonzero at this point
  }

  my_printf_debug("After merging tiling results" NEWLINE);

  dump_params_nhwc_debug(model, output, node->output_name, "ConvStage2");
}
