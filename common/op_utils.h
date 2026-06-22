// Shared scratch buffers and utility functions used by all op handlers.
//
// lea_buffer (LEA_BUFFER_SIZE int16_t words, placed in LEARAM):
//   Primary scratch for LEA/CMSIS matrix operations.  Must be 4-byte aligned
//   and have an even number of elements for LEA constraints.  All op handlers
//   that use LEA load their input tiles here.
//
// op_buffer (OP_BUFFER_LEN int16_t words):
//   Secondary scratch for temporaries that don't need LEA alignment — e.g.,
//   GlobalAveragePool partial-sum accumulators, Concat rescaling buffers.
//
// upper_gauss(a, b): smallest multiple of b that is ≥ a.
//
// float_to_scale_params: converts a floating-point scale value to the
//   (fract, shift) pair used by Scale::toFloat().  fract is normalised into
//   [0.5, 1.0) so the Q15 representation retains maximum precision.
//
// iterate_chunks: calls callback in chunks of up to LEA_BUFFER_SIZE elements,
//   advancing through param's NVM storage.  Used by operators that tile over
//   large tensors.
//
// hawaii_record_footprints: batches HAWAII footprint writes; flushes to NVM
//   only when a full BATCH_SIZE batch of jobs has completed.
//
// fix_first_unfinished_value_offset: rounds down to the nearest BATCH_SIZE
//   boundary — needed for BATCH_SIZE == 1 (extended footprint) because the
//   footprint records three values per write, not one.
//
// make_buffer_aligned: advances a pointer by one element if it falls at an
//   odd LEA word offset.  Call before passing a pointer to LEA routines.
//
// ChunkHandler: callback type for iterate_chunks; receives (output_offset,
//   chunk_len, user_params).

#pragma once

#include <cstdint>

#include "data.h"
#include "platform.h"

#define OP_BUFFER_LEN 512

struct Model;
struct ParameterInfo;
struct SlotInfo;
struct ValueInfo;
struct Scale;

typedef void (*ChunkHandler)(uint32_t output_offset, uint16_t output_chunk_len,
                             void* params);

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
int16_t upper_gauss(int16_t a, int16_t b);
void float_to_scale_params(int16_t* scaleFract, uint8_t* shift, float scale);
void float_to_scale_params(int16_t* scaleFract, uint8_t* shift,
                           const Scale& scale);
uint8_t count_dims(const ParameterInfo* data);
void recalculate_params_len(ParameterInfo* output);
void iterate_chunks(Model* model, const ParameterInfo* param,
                    uint16_t start_offset, uint16_t len,
                    const ChunkHandler& callback, void* params);
void determine_tile_c(ParameterInfo* param, const ParameterInfo* input,
                      const ParameterInfo* filter = nullptr);

#if HAWAII
void hawaii_record_footprints(Model* model, uint32_t vector_len);
#endif

void fix_first_unfinished_value_offset(
    const Model* model, uint32_t* p_first_unfinished_value_offset);
void make_buffer_aligned(int16_t** p_buffer);
float q15_to_float(int16_t val, const ValueInfo& val_info,
                   uint8_t* p_use_prefix = nullptr);
void my_offset_q15_batched(const int16_t* pSrc, int16_t offset, int16_t* pDst,
                           uint32_t blockSize);

extern int16_t op_buffer[OP_BUFFER_LEN];
