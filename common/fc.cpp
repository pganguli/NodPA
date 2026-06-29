// Fully-connected (Gemm) and MatMul operator implementations.
//
// ONNX Gemm computes:  Y = alpha * A * B + beta * C
// Here alpha == beta == 1 and the bias C is optional.
// MatMul is identical but without a bias term.
//
// TILING STRATEGY
// ---------------
// For large weight matrices that do not fit in lea_buffer at once, the
// computation is tiled along three dimensions:
//
//   tile_channel  — inner (K) dimension of A*B; partial sums across K tiles
//                   are accumulated in extra output slots (each tile writes to
//                   its own "part" of the output tensor in NVM).
//   tile_a_rows   — rows of A processed together.
//   tile_b_cols   — columns of B (== output columns) processed together.
//
// GemmMerge (handle_gemm_stage2) sums the K-tile partial results into the
// final output tensor, mirroring the ConvMerge pattern used by convolution.
//
// BIAS TRICK
// ----------
// The bias vector C is encoded directly into the B (weight) matrix: the last
// two rows of the padded B tile hold the negated bias for the first K-tile
// only (tile_channel_idx == 0).  The corresponding two extra elements in each
// A row are set to −0x8000 (Q15 –1.0), making the matrix multiply add the
// bias automatically.  This avoids a separate bias-addition pass.
//
// WEIGHT CACHE
// ------------
// If consecutive A-row tiles share the same B tile (same part_offset,
// tile_b_col_offset, tile_channel_offset), the B tile is not reloaded from
// NVM.  This is common when tile_a_rows > 1.

#include "fc.h"

#include <cstddef>
#include <cstdint>

#include "cnn_common.h"
#include "config.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "layer-defs.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

/**
 * For fully-connected layers, which are implemented via Gemm in ONNX.
 */

void alloc_gemm_impl(Model* model, const ParameterInfo* input[],
                     ParameterInfo* output, const Node* node,
                     CurNodeFlags* node_flags, const NodeFlags*) {
  const ParameterInfo *A = input[0], *B = input[1];

  uint8_t input_dims = node_flags->gemm.input_dims,
          weight_dims = node_flags->gemm.weight_dims;

  uint8_t output_dims = MAX_VAL(input_dims, weight_dims);
  for (uint8_t dim_idx = 0; dim_idx < output_dims - 2; dim_idx++) {
    if (dim_idx >= output_dims - input_dims &&
        dim_idx >= output_dims - weight_dims) {
      MY_ASSERT(A->dims[dim_idx - (output_dims - input_dims)] ==
                B->dims[dim_idx - (output_dims - weight_dims)]);
      output->dims[dim_idx] = A->dims[dim_idx - (output_dims - input_dims)];
    } else if (dim_idx >= output_dims - input_dims) {
      output->dims[dim_idx] = A->dims[output_dims - input_dims];
    } else if (dim_idx >= output_dims - weight_dims) {
      output->dims[dim_idx] = B->dims[output_dims - weight_dims];
    } else {
      MY_ASSERT(false);
    }
  }

  output->dims[output_dims - 2] = A->dims[input_dims - 2];
  output->dims[output_dims - 1] = B->dims[weight_dims - 1];
  output->scale = A->scale * B->scale;

  uint16_t output_len = 1;
  for (uint8_t dim_idx = 0; dim_idx < input_dims; dim_idx++) {
    output_len *= output->dims[dim_idx];
  }

  output->params_len =
      output_len *
      upper_gauss(B->dims[weight_dims - 2], node_flags->gemm.tile_channel) *
      sizeof(int16_t);
}

int16_t* const weights_tmp = op_buffer;

#if INTERMITTENT
static void gemm_recovery(
    Model* model, const ParameterInfo* input[], ParameterInfo* output,
    const Node* node, CurNodeFlags* node_flags,
    const NodeFlags* orig_node_flags,
    // layer dimensions
    uint16_t output_rows, uint16_t output_cols,
    // loop indices
    uint16_t* tile_channel_offset, uint16_t* tile_channel_idx,
    uint16_t* tile_a_row_offset, uint16_t* tile_b_col_offset,
    uint16_t* extended_tile_b_col_offset, uint16_t* part_idx) {
  const ParameterInfo* B = input[1];

  start_cpu_counter(offsetof(Counters, progress_seeking));
  uint32_t first_unfinished_value_offset =
      job_index_to_offset(output, run_recovery(model, output));

  first_unfinished_value_offset = batch_start(first_unfinished_value_offset);

  fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

  uint32_t output_len = 1;
  for (uint8_t dim_idx = 0; dim_idx < 4; dim_idx++) {
    if (!output->dims[dim_idx]) {
      break;
    }
    output_len *= output->dims[dim_idx];
  }

  // TODO: get part_idx

  *tile_channel_idx = first_unfinished_value_offset / output_len;
  *tile_channel_offset = (*tile_channel_idx) * node_flags->gemm.tile_channel;
  first_unfinished_value_offset %= output_len;

  *part_idx = first_unfinished_value_offset / (output_rows * output_cols);
  first_unfinished_value_offset %= (output_rows * output_cols);

  *tile_a_row_offset = first_unfinished_value_offset / output_cols;
  *extended_tile_b_col_offset = first_unfinished_value_offset % output_cols;

  *tile_b_col_offset = *extended_tile_b_col_offset;

  stop_cpu_counter();

  my_printf_debug("tile_a_rows=%d, tile_channel=%d, tile_b_cols=%d" NEWLINE,
                  node_flags->gemm.tile_a_rows, node_flags->gemm.tile_channel,
                  node_flags->gemm.tile_b_cols);

  uint8_t weight_dims = node_flags->gemm.weight_dims;
  output->params_len =
      output_len *
      upper_gauss(B->dims[weight_dims - 2], node_flags->gemm.tile_channel) *
      sizeof(int16_t);
  MY_ASSERT(node_flags->gemm.tile_b_cols / BATCH_SIZE * BATCH_SIZE ==
            node_flags->gemm.tile_b_cols);
}
#endif

void handle_gemm_impl(Model* model, const ParameterInfo* input[],
                      ParameterInfo* output, const Node* node,
                      CurNodeFlags* node_flags,
                      const NodeFlags* orig_node_flags) {
  const ParameterInfo *A = input[0], *B = input[1], *matC = nullptr;

#ifdef OpGemm
  // OpMatMul does not have a bias
  if (node->op_type == OpGemm) {
    matC = input[2];
  }
#endif

  my_printf_debug("Gemm!" NEWLINE);

  // Use original tile sizes here, as tile sizes might be dynamically
  // reconfigured, while the reconfigured size is never larger than the original
  // size
  int16_t buffer_a_size = orig_node_flags->gemm.tile_a_rows *
                          (orig_node_flags->gemm.tile_channel + 2);

  int16_t *buffer_a = lea_buffer,
          *buffer_temp = buffer_a + (buffer_a_size + 1) / 2 *
                                        2;  // guarantee even addresses, making
                                            // check_buffer_address happy
  int16_t* buffer_b =
      buffer_temp + node_flags->gemm.tile_a_rows * node_flags->gemm.tile_b_cols;
  make_buffer_aligned(&buffer_b);

  uint16_t tile_channel_offset = 0, tile_channel_idx = 0, tile_a_row_offset = 0,
           tile_b_col_offset = 0, extended_tile_b_col_offset = 0, part_idx = 0;

  uint8_t input_dims = node_flags->gemm.input_dims,
          weight_dims = node_flags->gemm.weight_dims;
  uint8_t output_dims = MAX_VAL(input_dims, weight_dims);

  uint16_t A_rows = A->dims[input_dims - 2], A_cols = A->dims[input_dims - 1],
           B_rows = B->dims[weight_dims - 2], B_cols = B->dims[weight_dims - 1],
           output_rows = output->dims[output_dims - 2],
           output_cols = output->dims[output_dims - 1];

#if DEBUG
  {
    my_printf("GEMMA slot=%d poff=%lu plen=%lu dims=%d,%d scale=%f" NEWLINE,
              (int)A->slot, (unsigned long)A->params_offset,
              (unsigned long)A->params_len, (int)A->dims[0], (int)A->dims[1],
              A->scale.toFloat());
    int16_t tmp[8];
    uint16_t n = MIN_VAL((uint16_t)8, A_cols);
    my_memcpy_from_param(model, tmp, A, 0, n * sizeof(int16_t));
    my_printf("GEMMA[0..%d]:", (int)(n - 1));
    for (uint16_t i = 0; i < n; i++) my_printf(" %d", (int)tmp[i]);
    my_printf(NEWLINE);
    my_printf("GEMMB scale=%f dims=%d,%d" NEWLINE,
              B->scale.toFloat(), (int)B->dims[0], (int)B->dims[1]);
  }
#endif

#if INTERMITTENT
  gemm_recovery(model, input, output, node, node_flags, orig_node_flags,
                output_rows, output_cols, &tile_channel_offset,
                &tile_channel_idx, &tile_a_row_offset, &tile_b_col_offset,
                &extended_tile_b_col_offset, &part_idx);
#endif

  bool weights_broadcasted = weight_dims < output_dims;

  uint16_t parts = 1;
  for (uint8_t dim_idx = 0; dim_idx < input_dims - 2; dim_idx++) {
    parts *= A->dims[dim_idx];
  }

  uint32_t output_len = 1;
  for (uint8_t dim_idx = 0; dim_idx < output_dims; dim_idx++) {
    output_len *= output->dims[dim_idx];
  }

  struct {
    uint16_t part_offset = UINT16_MAX;
    uint16_t tile_b_col_offset = UINT16_MAX;
    uint16_t tile_channel_offset = UINT16_MAX;
    uint16_t extended_tile_channels = UINT16_MAX;
    uint16_t full_tile_b_cols = UINT16_MAX;
    bool cache_valid = false;
  } weights_cache;

  for (; tile_channel_offset < B_rows;
       tile_channel_offset += node_flags->gemm.tile_channel,
       tile_channel_idx++) {
    const uint16_t tile_channels =
        MIN_VAL(node_flags->gemm.tile_channel, B_rows - tile_channel_offset);
    const uint16_t extended_tile_channels = tile_channels + 2;

    for (; part_idx < parts; part_idx++) {
      int16_t cur_tile_a_rows;  // Will be initialized later. This is declared
                                // outside the loop, so that tile_a_row_offset
                                // can be correctly incremented for arbitrary
                                // cur_tile_a_rows.
      for (; tile_a_row_offset < A_rows; tile_a_row_offset += cur_tile_a_rows) {
        uint16_t output_offset = tile_channel_idx * output_len +
                                 part_idx * output_rows * output_cols +
                                 tile_a_row_offset * output_cols +
                                 extended_tile_b_col_offset;
        cur_tile_a_rows =
            MIN_VAL(node_flags->gemm.tile_a_rows, A_rows - tile_a_row_offset);

        for (uint16_t tile_a_row_offset_inner = 0;
             tile_a_row_offset_inner < cur_tile_a_rows;
             tile_a_row_offset_inner++) {
          my_memcpy_from_param(
              model,
              buffer_a + tile_a_row_offset_inner * extended_tile_channels, A,
              (part_idx * A_rows + tile_a_row_offset +
               tile_a_row_offset_inner) *
                      A_cols +
                  tile_channel_offset,
              tile_channels * sizeof(uint16_t));
        }

        for (uint16_t tile_a_row_offset_inner = 0;
             tile_a_row_offset_inner < cur_tile_a_rows;
             tile_a_row_offset_inner++) {
          int16_t* bias_multiplier_ptr =
              buffer_a + tile_a_row_offset_inner * extended_tile_channels +
              tile_channels;
          *bias_multiplier_ptr = -0x8000;
          *(bias_multiplier_ptr + 1) = 0;
        }

        my_printf_debug("Tile for A" NEWLINE);
        dump_matrix_debug(buffer_a, cur_tile_a_rows, extended_tile_channels,
                          ValueInfo(A, model));

        for (; tile_b_col_offset < B_cols;
             tile_b_col_offset += node_flags->gemm.tile_b_cols) {
          int16_t tile_b_cols =
              MIN_VAL(node_flags->gemm.tile_b_cols, B_cols - tile_b_col_offset);
          int16_t values_to_preserve = tile_b_cols, full_tile_b_cols;
          full_tile_b_cols = (tile_b_cols + 1) / 2 * 2;
          uint32_t part_offset = 0;
          if (!weights_broadcasted) {
            part_offset += part_idx * B_rows * B_cols;
          }

          my_printf_debug(
              "Checking whether to load weights or not... "
              "part_offset=%d, tile_b_col_offset=%d, tile_channel_offset=%d, "
              "extended_tile_channels=%d, full_tile_b_cols=%d" NEWLINE,
              part_offset, tile_b_col_offset, tile_channel_offset,
              extended_tile_channels, full_tile_b_cols);
          if (weights_cache.cache_valid &&
              weights_cache.part_offset == part_offset &&
              weights_cache.tile_b_col_offset == tile_b_col_offset &&
              weights_cache.tile_channel_offset == tile_channel_offset &&
              weights_cache.extended_tile_channels == extended_tile_channels &&
              weights_cache.full_tile_b_cols == full_tile_b_cols) {
            my_printf_debug("Cached!" NEWLINE);
          } else {
            my_printf_debug("Not cached!" NEWLINE);

            int16_t* filter_ptr = buffer_b;
            my_fill_q15(0, filter_ptr,
                        extended_tile_channels * full_tile_b_cols);
            for (uint16_t row = 0; row < tile_b_cols; row++) {
              MY_ASSERT(tile_channels <= OP_BUFFER_LEN);

              if (B->param_flags & TRANSPOSED) {
                my_memcpy_from_param(model, weights_tmp, B,
                                     part_offset +
                                         (tile_b_col_offset + row) * B_rows +
                                         tile_channel_offset,
                                     tile_channels * sizeof(uint16_t));
              } else {
                // XXX: copy and interleave might be merged
                for (uint16_t tile_channel_copy_idx = 0;
                     tile_channel_copy_idx < tile_channels;
                     tile_channel_copy_idx++) {
                  weights_tmp[tile_channel_copy_idx] = get_q15_param(
                      model, B,
                      part_offset +
                          (tile_channel_offset + tile_channel_copy_idx) *
                              B_cols +
                          (tile_b_col_offset + row));
                }
              }

              my_interleave_q15(weights_tmp, row, full_tile_b_cols, filter_ptr,
                                tile_channels);
            }
            filter_ptr += tile_channels * full_tile_b_cols;
            if (tile_channel_idx == 0) {
              for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                if (matC) {
                  filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(
                                        model, matC, idx + tile_b_col_offset)) /
                                    A->scale.toFloat();
                } else {
                  filter_ptr[idx] = 0;
                }
              }
            }

            weights_cache.cache_valid = true;
            weights_cache.part_offset = part_offset;
            weights_cache.tile_b_col_offset = tile_b_col_offset;
            weights_cache.tile_channel_offset = tile_channel_offset;
            weights_cache.extended_tile_channels = extended_tile_channels;
            weights_cache.full_tile_b_cols = full_tile_b_cols;
          }

          my_printf_debug("Tile for B" NEWLINE);
          dump_matrix_debug(buffer_b, extended_tile_channels, full_tile_b_cols,
                            ValueInfo(B, model));
          my_matrix_mpy_q15(cur_tile_a_rows, extended_tile_channels,
                            extended_tile_channels, full_tile_b_cols, buffer_a,
                            buffer_b, buffer_temp, output, output_offset,
                            values_to_preserve,
                            orig_node_flags->gemm.pState_len);
          my_printf_debug("matrix_mpy_results" NEWLINE);
          dump_matrix_debug(buffer_temp, cur_tile_a_rows, full_tile_b_cols,
                            ValueInfo(output, model));
          my_printf_debug(NEWLINE);

          compare_vm_nvm(buffer_temp, model, output, output_offset,
                         values_to_preserve);

          my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if HAWAII
          hawaii_record_footprints(model, values_to_preserve);
#endif
          output_offset += values_to_preserve;
        }
        tile_b_col_offset = extended_tile_b_col_offset = 0;
      }
      tile_a_row_offset = 0;
    }
    part_idx = 0;
  }
}

void alloc_gemm_stage2_impl(Model* model, const ParameterInfo* input[],
                            ParameterInfo* output, const Node*, CurNodeFlags*,
                            const NodeFlags*) {
  int16_t output_len = 1;
  for (uint8_t dim_idx = 0; dim_idx < 4; dim_idx++) {
    if (!output->dims[dim_idx]) {
      break;
    }
    output_len *= output->dims[dim_idx];
  }
  output->params_len = output_len * sizeof(int16_t);
}

void handle_gemm_stage2_impl(Model* model, const ParameterInfo* input[],
                             ParameterInfo* output, const Node* node,
                             CurNodeFlags* node_flags, const NodeFlags*) {
  const ParameterInfo* X = input[0];

  my_printf_debug("GemmMerge!" NEWLINE);

  uint8_t output_dims = node_flags->gemm_stage2.input_dims;
  int16_t output_len = 1;
  for (uint8_t dim_idx = 0; dim_idx < output_dims; dim_idx++) {
    output_len *= X->dims[dim_idx];
  }

  int16_t output_tile_size = node_flags->gemm_stage2.tile_length;
  if (!output_tile_size) {
    // buffer_temp and buffer_gemm have the same size, and they occupy LEA
    // buffer, so divide by 2
    output_tile_size = MIN_VAL(LIMIT_DMA_SIZE(output_len), LEA_BUFFER_SIZE / 2);
  }

  uint16_t merge_offset = 0;
#if INTERMITTENT
  start_cpu_counter(offsetof(Counters, progress_seeking));
  merge_offset =
      batch_start(job_index_to_offset(output, run_recovery(model, output)));
  stop_cpu_counter();
#endif

  int16_t *buffer_temp = lea_buffer,
          *buffer_gemm = buffer_temp + output_tile_size;
  make_buffer_aligned(&buffer_gemm);

  int16_t n_tiles = X->params_len / output_len / sizeof(int16_t);
  my_printf_debug("n_tiles=%d" NEWLINE, n_tiles);
  MY_ASSERT(n_tiles > 0);

  for (; merge_offset < output_len; merge_offset += output_tile_size) {
    int16_t cur_tile_size =
        MIN_VAL(output_tile_size, output_len - merge_offset);
    my_fill_q15(0, buffer_gemm, cur_tile_size);

    for (uint16_t tile = 0; tile < n_tiles; tile++) {
      my_memcpy_from_param(model, buffer_temp, input[0],
                           tile * output_len + merge_offset,
                           cur_tile_size * sizeof(int16_t));
      my_add_q15(buffer_gemm, buffer_temp, buffer_gemm, cur_tile_size);
      my_printf_debug("accumulated buffer_gemm" NEWLINE);
      dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));
    }

    my_printf_debug("buffer_gemm after accumulation; merge_offset=%d" NEWLINE,
                    merge_offset);
    dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));

    my_memcpy_to_param(output, merge_offset, buffer_gemm,
                       cur_tile_size * sizeof(int16_t), 0, true);
#if HAWAII
    hawaii_record_footprints(model, cur_tile_size);
#endif
  }
}

/* Wrappers for Gemm */

void alloc_gemm(Model* model, const ParameterInfo* input[],
                ParameterInfo* output, const Node* node,
                CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  alloc_gemm_impl(model, input, output, node, node_flags, orig_node_flags);
}

void handle_gemm(Model* model, const ParameterInfo* input[],
                 ParameterInfo* output, const Node* node,
                 CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  handle_gemm_impl(model, input, output, node, node_flags, orig_node_flags);

  my_printf_debug("handle_gemm output" NEWLINE);
  dump_params_debug(model, output, node->output_name, "Gemm");
}

void alloc_gemm_stage2(Model* model, const ParameterInfo* input[],
                       ParameterInfo* output, const Node* node,
                       CurNodeFlags* node_flags,
                       const NodeFlags* orig_node_flags) {
  alloc_gemm_stage2_impl(model, input, output, node, node_flags,
                         orig_node_flags);
}

void handle_gemm_stage2(Model* model, const ParameterInfo* input[],
                        ParameterInfo* output, const Node* node,
                        CurNodeFlags* node_flags,
                        const NodeFlags* orig_node_flags) {
  handle_gemm_stage2_impl(model, input, output, node, node_flags,
                          orig_node_flags);

  my_printf_debug("handle_gemm_stage2 output" NEWLINE);
  dump_params_debug(model, output, node->output_name, "GemmStage2");
}

/* Wrappers for MatMul */

void alloc_mat_mul(Model* model, const ParameterInfo* input[],
                   ParameterInfo* output, const Node* node,
                   CurNodeFlags* node_flags, const NodeFlags* orig_node_flags) {
  alloc_gemm_impl(model, input, output, node, node_flags, orig_node_flags);
}

void handle_mat_mul(Model* model, const ParameterInfo* input[],
                    ParameterInfo* output, const Node* node,
                    CurNodeFlags* node_flags,
                    const NodeFlags* orig_node_flags) {
  handle_gemm_impl(model, input, output, node, node_flags, orig_node_flags);

  my_printf_debug("handle_mat_mul output" NEWLINE);
  dump_params_debug(model, output, node->output_name, "MatMul");
}

void alloc_mat_mul_stage2(Model* model, const ParameterInfo* input[],
                          ParameterInfo* output, const Node* node,
                          CurNodeFlags* node_flags,
                          const NodeFlags* orig_node_flags) {
  alloc_gemm_stage2_impl(model, input, output, node, node_flags,
                         orig_node_flags);
}

void handle_mat_mul_stage2(Model* model, const ParameterInfo* input[],
                           ParameterInfo* output, const Node* node,
                           CurNodeFlags* node_flags,
                           const NodeFlags* orig_node_flags) {
  handle_gemm_stage2_impl(model, input, output, node, node_flags,
                          orig_node_flags);

  my_printf_debug("handle_mat_mul_stage2 output" NEWLINE);
  dump_params_debug(model, output, node->output_name, "MatMulStage2");
}
