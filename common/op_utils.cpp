// Shared utility functions used by multiple operator implementations.
//
// KEY GLOBALS
//   lea_buffer  — the main scratch buffer for all DSP operations.  On MSP430
//                 it lives in LEARAM (section ".leaRAM") so the LEA can address
//                 it directly.  All my_dsplib wrappers require pointers into
//                 this buffer.
//   op_buffer   — secondary scratch space (OP_BUFFER_LEN q15 values) used by
//                 individual operator handlers for small temporaries (e.g.,
//                 bias values in group convolution).
//
// FOOTPRINT BATCHING (hawaii_record_footprints)
//   Writing to NVM is expensive.  Rather than updating the footprint after
//   every single output value, jobs are accumulated in a static counter and
//   the footprint is only written to NVM when a full BATCH_SIZE has been
//   completed.  The remainder carries over to the next call.  This dramatically
//   reduces NVM writes per layer.
//
// SCALE CONVERSION (float_to_scale_params)
//   Converts a floating-point scale factor into the (fract, shift) pair used
//   by my_scale_q15:  output = input * fract * 2^shift / 32768.
//   The while loop normalises fract into [0.5, 1.0) (i.e., fits in Q15).
//
// ALIGNMENT HELPERS
//   make_buffer_aligned  — advances a pointer by one element if it sits at an
//     odd index within lea_buffer (LEA requires even offsets).
//   fix_first_unfinished_value_offset — rounds the recovery offset down to an
//     even index when BATCH_SIZE == 1, because most DSPLib functions require
//     even operand counts.

#include "op_utils.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>  // for std::enable_if_t

#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "platform.h"

// Not using DSPLIB_DATA here as it does not work under C++ (?)
#ifdef __MSP430__
#pragma DATA_SECTION(".leaRAM")
#endif
NOINIT int16_t lea_buffer[LEA_BUFFER_SIZE];

NOINIT int16_t op_buffer[OP_BUFFER_LEN];
static_assert(OUTPUT_LEN <= OP_BUFFER_LEN, "invalid OP buffer size");

#if HAWAII
static uint32_t non_recorded_jobs = 0;
void hawaii_record_footprints(Model* model, uint32_t vector_len) {
  non_recorded_jobs += vector_len;
  // Batch the footprint write: record only whole batches, carry the remainder
  // to the next call. Writing per-element would be too expensive on NVM.
  write_hawaii_layer_footprint(model->layer_idx,
                               non_recorded_jobs / BATCH_SIZE * BATCH_SIZE);
  non_recorded_jobs %= BATCH_SIZE;
}
#endif

int16_t upper_gauss(int16_t a, int16_t b) { return (a + b - 1) / b; }

void float_to_scale_params(int16_t* scaleFract, uint8_t* shift,
                           const Scale& scale) {
  float_to_scale_params(scaleFract, shift, scale.toFloat());
}

void float_to_scale_params(int16_t* scaleFract, uint8_t* shift, float scale) {
  MY_ASSERT(scale > 0);
  *shift = 0;
  while (scale >= 1) {
    scale /= 2;
    (*shift)++;
  }
  *scaleFract = scale * 32768;
}

uint8_t count_dims(const ParameterInfo* data) {
  uint8_t dim_idx = 0;
  while (dim_idx < MAX_NUM_DIMS && data->dims[dim_idx]) {
    dim_idx++;
  }
  return dim_idx;
}

void recalculate_params_len(ParameterInfo* output) {
  output->params_len = sizeof(int16_t);
  for (uint8_t dim_idx = 0; (dim_idx < MAX_NUM_DIMS) && output->dims[dim_idx];
       dim_idx++) {
    output->params_len *= output->dims[dim_idx];
  }
}

void iterate_chunks(Model* model, const ParameterInfo* param,
                    uint16_t start_offset, uint16_t len,
                    const ChunkHandler& chunk_handler, void* params) {
  uint16_t params_len;
  if (!len) {
    params_len = param->params_len / sizeof(int16_t);
  } else {
    params_len = start_offset + len;
  }
  uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);

  uint16_t cur_chunk_len;
  for (uint32_t offset = start_offset; offset < params_len;
       offset += cur_chunk_len) {
    cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
    MY_ASSERT(cur_chunk_len != 0);
    chunk_handler(offset, cur_chunk_len, params);
  }
}

void fix_first_unfinished_value_offset(
    const Model* model, uint32_t* p_first_unfinished_value_offset) {
  if (BATCH_SIZE >= 2) {
    return;
  }
  // Force recovery from an even OFM index as most DSPLib function does not like
  // odd dimensions
  if (*p_first_unfinished_value_offset % 2) {
    (*p_first_unfinished_value_offset)--;
#if HAWAII
    write_hawaii_layer_footprint(model->layer_idx, -1);  // discard last job
#endif
  }
}

void make_buffer_aligned(int16_t** p_buffer) {
  if ((*p_buffer - lea_buffer) % 2) {
    (*p_buffer)++;
  }
}

float q15_to_float(int16_t val, const ValueInfo& val_info,
                   uint8_t* p_use_prefix) {
  return val_info.scale * static_cast<int32_t>(val) / 32768.0;
}

void my_offset_q15_batched(const int16_t* pSrc, int16_t offset, int16_t* pDst,
                           uint32_t blockSize) {
  MY_ASSERT(pSrc == pDst);
  if (BATCH_SIZE == 1) {
    my_offset_q15(pSrc, offset, pDst, blockSize);
  } else {
    for (uint32_t val_idx = BATCH_SIZE - 1; val_idx < blockSize;
         val_idx += BATCH_SIZE) {
      pDst[val_idx] += offset;
    }
  }
}
