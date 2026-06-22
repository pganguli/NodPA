// DSP operation wrappers that abstract over TI DSPLib (MSP430 LEA) and
// ARM CMSIS-DSP (MSP432 / PC simulation).
//
// All `blockSize` parameters are in ELEMENTS (int16_t units), not bytes.
//
// IMPORTANT CONSTRAINTS:
//   • All source and destination pointers must point into lea_buffer, and
//     must be 2-byte (even index) aligned within that buffer.  Use
//     make_buffer_aligned() and check_buffer_address() to verify.
//   • LEA requires even element counts for most operations.  The wrappers
//     handle odd remainders with scalar fallback code.
//   • my_matrix_mpy_q15 processes one row of A at a time internally;
//     B_cols must be even (LEA requirement).  A_cols (== B_rows) may be odd —
//     empirical testing on MSP430FR5994 shows LEA handles this correctly
//     despite the documentation requiring even sizes.
//   • my_interleave_q15 / my_deinterleave_q15 do NOT use LEA because the
//     source/destination is often not LEARAM-aligned.
//
// my_matrix_mpy_q15 also writes output directly to NVM via my_memcpy_to_param
// to avoid double-buffering in SRAM; `values_to_preserve` controls how many
// of the B_cols output values are actually committed (the rest are padding).

#pragma once

#include <cstdint>
#include <cstdlib>
struct ParameterInfo;

void my_add_q15(const int16_t* pSrcA, const int16_t* pSrcB, int16_t* pDst,
                uint32_t blockSize);
void my_fill_q15(int16_t value, int16_t* pDst, uint32_t blockSize);
void my_offset_q15(const int16_t* pSrc, int16_t offset, int16_t* pDst,
                   uint32_t blockSize);
void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows,
                       uint16_t B_cols, int16_t* pSrcA, int16_t* pSrcB,
                       int16_t* pDst, ParameterInfo* param,
                       uint32_t offset_in_word, size_t values_to_preserve,
                       uint16_t pState_len);
void my_vector_mult_q15(const int16_t* pSrcA, const int16_t* pSrcB,
                        int16_t* pDst, uint32_t blockSize);
void my_max_q15(const int16_t* pSrc, uint32_t blockSize, int16_t* pResult,
                uint16_t* pIndex);
void my_min_q15(const int16_t* pSrc, uint32_t blockSize, int16_t* pResult,
                uint16_t* pIndex);
void my_scale_q15(const int16_t* pSrc, int16_t scaleFract, uint8_t shift,
                  int16_t* pDst, uint32_t blockSize);
void my_interleave_q15(const int16_t* pSrc, uint16_t channel,
                       uint16_t numChannels, int16_t* pDst, uint32_t blockSize);
void my_deinterleave_q15(const int16_t* pSrc, uint16_t channel,
                         uint16_t numChannels, int16_t* pDst,
                         uint32_t blockSize);
int16_t padding_for_lea(int16_t val);
void check_buffer_address(const int16_t* addr, uint32_t blockSize);
