// DSP wrappers: concrete implementations for TI DSPLib (LEA) and ARM CMSIS-DSP.
//
// ALIGNMENT ENFORCEMENT
//   check_buffer_address_with_base() verifies at debug time that every pointer
//   passed to a LEA function:
//     1. Falls within lea_buffer.
//     2. Is at an even offset from the buffer base (LEA word-alignment).
//   This catches alignment bugs during development; it compiles away in
//   release.
//
// ODD-SIZE WORKAROUNDS
//   Most LEA functions require even blockSize.  Each wrapper rounds down to an
//   even count for the LEA call and then handles the trailing odd element with
//   plain scalar arithmetic.  See my_add_q15, my_offset_q15, my_scale_q15.
//
// my_max_q15 / my_min_q15 UNALIGNMENT
//   If pSrc is at an odd offset, the first element is compared separately
//   before the LEA/CMSIS call.  The found index is adjusted by +1 afterward.
//
// my_matrix_mpy_q15 NVM WRITE-BACK
//   The standard matrix multiply would fill pDst in SRAM.  The modified
//   versions call back via data_preservation_func (my_memcpy_to_param) row by
//   row so each output row goes directly to NVM, keeping SRAM usage bounded
//   regardless of output matrix size.
//
// my_interleave_q15 / my_deinterleave_q15
//   These scatter/gather operations do not use LEA because the source or
//   destination is typically not at a LEA-aligned address (the weight tensor
//   in NVM is row-major and filtering by channel would start at arbitrary
//   offsets).

#include <cstddef>
#include <cstdint>

#include "config.h"
#include "data.h"

#if !USE_ARM_CMSIS
#include <DSPLib.h>
#else
#include <arm_math.h>
#endif

#include "cnn_common.h"
#include "counters.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

#if !USE_ARM_CMSIS
#if DEBUG
#define my_checkStatus(expr)                                             \
  do {                                                                   \
    msp_status status = (expr);                                          \
    MY_ASSERT(status == MSP_SUCCESS, "Error from TI-DSPLib: %d" NEWLINE, \
              status);                                                   \
  } while (0);
#else
#define my_checkStatus(expr) (expr)
#endif
#endif

#if DEBUG
static uint8_t check_buffer_address_with_base(const int16_t* base_addr,
                                              uint32_t whole_buffer_size,
                                              const int16_t* addr,
                                              uint32_t blockSize) {
  uint8_t ret = 1;
  ret &= (addr >= base_addr && addr < base_addr + whole_buffer_size);
  ret &= (addr + blockSize - 1 >= base_addr &&
          addr + blockSize - 1 < base_addr + whole_buffer_size);
  ret &= ((addr - base_addr) % 2 == 0);
  return ret;
}
#endif

void check_buffer_address(const int16_t* addr, uint32_t blockSize) {
  MY_ASSERT(check_buffer_address_with_base(lea_buffer, LEA_BUFFER_SIZE, addr,
                                           blockSize));
}

void my_add_q15(const int16_t* pSrcA, const int16_t* pSrcB, int16_t* pDst,
                uint32_t blockSize) {
  check_buffer_address(pSrcA, blockSize);
  check_buffer_address(pSrcB, blockSize);
  check_buffer_address(pDst, blockSize);
#if !USE_ARM_CMSIS
  uint32_t blockSizeForLEA = blockSize / 2 * 2;
  if (blockSizeForLEA) {
    msp_add_q15_params add_params;
    add_params.length = blockSizeForLEA;
    my_checkStatus(msp_add_q15(&add_params, pSrcA, pSrcB, pDst));
  }
  if (blockSize % 2) {
    pDst[blockSize - 1] = pSrcA[blockSize - 1] + pSrcB[blockSize - 1];
  }
#else
  arm_add_q15(pSrcA, pSrcB, pDst, blockSize);
#endif
}

void my_fill_q15(int16_t value, int16_t* pDst, uint32_t blockSize) {
  // XXX: not using LEA/SIMD here as pDst may not be aligned
  int16_t* pEnd = pDst + blockSize;
  while (pDst < pEnd) {
    *pDst = value;
    pDst++;
  }
}

void my_offset_q15(const int16_t* pSrc, int16_t offset, int16_t* pDst,
                   uint32_t blockSize) {
#if !USE_ARM_CMSIS
  // XXX: the alignment adjustment code in this function only supports pSrc ==
  // pDst
  MY_ASSERT(pSrc == pDst);
  // if pSrc is not 4-byte aligned...
  if (reinterpret_cast<uint64_t>(pSrc) & 3) {
    *pDst = *pSrc + offset;
    pSrc++;
    pDst++;
    MY_ASSERT(blockSize);  // avoid overflow in the next line
    blockSize--;
  }
  check_buffer_address(pSrc, blockSize);
  check_buffer_address(pDst, blockSize);
  // LEA does not like zero-sized blocks
  uint16_t block_size_for_lea = blockSize / 2 * 2;
  if (block_size_for_lea) {
    msp_offset_q15_params offset_params;
    offset_params.length = block_size_for_lea;
    offset_params.offset = offset;
    my_checkStatus(msp_offset_q15(&offset_params, pSrc, pDst));
  }
  if (blockSize % 2) {
    pDst[blockSize - 1] = pSrc[blockSize - 1] + offset;
  }
#else
  arm_offset_q15(pSrc, offset, pDst, blockSize);
#endif
}

void my_max_q15(const int16_t* pSrc, uint32_t blockSize, int16_t* pResult,
                uint16_t* pIndex) {
  uint8_t unaligned = 0;
  if ((pSrc - lea_buffer) % 2) {
    unaligned = 1;
    pSrc++;
    MY_ASSERT(blockSize > 0);
    blockSize--;
  }
#if !USE_ARM_CMSIS
  uint32_t blockSizeForLEA = blockSize / 2 * 2;
  if (blockSizeForLEA) {
    msp_max_q15_params max_params;
    max_params.length = blockSizeForLEA;
    my_checkStatus(msp_max_q15(&max_params, pSrc, pResult, pIndex));
  }
  if (blockSize % 2) {
    if (*pResult < pSrc[blockSize - 1]) {
      *pResult = pSrc[blockSize - 1];
      *pIndex = blockSize - 1;
    };
  }
#else
  uint32_t pIndex_u32;
  arm_max_q15(pSrc, blockSize, pResult, &pIndex_u32);
  *pIndex = pIndex_u32;
#endif
  if (unaligned) {
    int16_t candidate = *(pSrc - 1);  // -1 as pSrc was +1
    if (*pResult > candidate) {
      (*pIndex)++;
    } else {
      *pIndex = 0;
      *pResult = candidate;
    }
  }
}

void my_min_q15(const int16_t* pSrc, uint32_t blockSize, int16_t* pResult,
                uint16_t* pIndex) {
  uint8_t unaligned = 0;
  if ((pSrc - lea_buffer) % 2) {
    unaligned = 1;
    pSrc++;
    MY_ASSERT(blockSize > 0);
    blockSize--;
  }
#if !USE_ARM_CMSIS
  uint32_t blockSizeForLEA = blockSize / 2 * 2;
  if (blockSizeForLEA) {
    msp_min_q15_params min_params;
    min_params.length = blockSizeForLEA;
    my_checkStatus(msp_min_q15(&min_params, pSrc, pResult, pIndex));
  }
  if (blockSize % 2) {
    if (*pResult > pSrc[blockSize - 1]) {
      *pResult = pSrc[blockSize - 1];
      *pIndex = blockSize - 1;
    };
  }
#else
  uint32_t pIndex_u32;
  arm_min_q15(pSrc, blockSize, pResult, &pIndex_u32);
  *pIndex = pIndex_u32;
#endif
  if (unaligned) {
    int16_t candidate = *(pSrc - 1);  // -1 as pSrc was +1
    if (*pResult < candidate) {
      (*pIndex)++;
    } else {
      *pIndex = 0;
      *pResult = candidate;
    }
  }
}

void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows,
                       uint16_t B_cols, int16_t* pSrcA, int16_t* pSrcB,
                       int16_t* pDst, ParameterInfo* param,
                       uint32_t offset_in_word, size_t values_to_preserve,
                       uint16_t pState_len) {
  // LEA documentation requires all matrix dimensions to be even. In practice,
  // LEA gives correct results when the inner dimension (A_cols == B_rows) is
  // odd. The output column count (B_cols) MUST be even.
  // Reference: http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353
  //
  // The assertion reads: (A_cols is odd) OR (B_cols is even).
  // Due to C++ precedence, `(B_cols & 1) == 0` binds tighter than `||`, so
  // this correctly asserts that at least one of the two tolerated conditions
  // holds. The case A_cols-even + B_cols-odd would violate LEA requirements
  // and is the only combination that makes this assertion fail.
  MY_ASSERT((A_cols & 1) || (B_cols & 1) == 0);
#if USE_ARM_CMSIS
  MY_ASSERT(B_rows * B_cols <= pState_len);
#endif
  MY_ASSERT(A_cols == B_rows);
  check_buffer_address(pSrcA, A_rows * A_cols);
  check_buffer_address(pSrcB, B_rows * B_cols);

#if !USE_ARM_CMSIS
  msp_matrix_mpy_q15_params matrix_mpy_params;
  for (uint16_t A_row_offset = 0; A_row_offset < A_rows; A_row_offset++) {
    matrix_mpy_params.srcARows = 1;
    matrix_mpy_params.srcACols = A_cols;
    matrix_mpy_params.srcBRows = B_rows;
    matrix_mpy_params.srcBCols = B_cols;
    my_checkStatus(msp_matrix_mpy_q15(&matrix_mpy_params, pSrcA, pSrcB, pDst,
                                      my_memcpy_to_param, param, offset_in_word,
                                      values_to_preserve));

    // Moving to the next row of matrix A
    pSrcA += A_cols;
    pDst += B_cols;
    offset_in_word += values_to_preserve;
  }
#else
  arm_matrix_instance_q15 A, B, C;
  int16_t* pState = lea_buffer + LEA_BUFFER_SIZE - pState_len;
  for (uint16_t A_row_offset = 0; A_row_offset < A_rows; A_row_offset++) {
    arm_mat_init_q15(&A, 1, A_cols, pSrcA);
    arm_mat_init_q15(&B, B_rows, B_cols, pSrcB);
    arm_mat_init_q15(&C, 1, B_cols, pDst);
#ifdef __MSP432__
    arm_status status =
        arm_mat_mult_fast_q15(&A, &B, &C, pState, my_memcpy_to_param, param,
                              offset_in_word, values_to_preserve);
    MY_ASSERT(status == ARM_MATH_SUCCESS);
#else
    arm_status status = arm_mat_mult_fast_q15(&A, &B, &C, pState,
                                              my_memcpy_to_param, NULL, 0, 0);
    MY_ASSERT(status == ARM_MATH_SUCCESS);
    if (param) {
      my_memcpy_to_param(param, offset_in_word, pDst,
                         values_to_preserve * sizeof(int16_t), 0, true);
    }
#endif  // __MSP432__
    (void)status;  // Suppress -Wunused-variable in release builds (DEBUG == 0)

    // Moving to the next row of matrix A
    pSrcA += A_cols;
    pDst += B_cols;
    offset_in_word += values_to_preserve;
  }
#endif  // !USE_ARM_CMSIS

  add_counter(offsetof(Counters, macs), A_rows * B_cols * A_cols);
}

void my_vector_mult_q15(const int16_t* pSrcA, const int16_t* pSrcB,
                        int16_t* pDst, uint32_t blockSize) {
#if !USE_ARM_CMSIS
  msp_mpy_q15_params params;
  params.length = blockSize;
  msp_mpy_q15(&params, pSrcA, pSrcB, pDst);
#else
  arm_mult_q15(pSrcA, pSrcB, pDst, blockSize);
#endif
}

void my_scale_q15(const int16_t* pSrc, int16_t scaleFract, uint8_t shift,
                  int16_t* pDst, uint32_t blockSize) {
#if !USE_ARM_CMSIS
  uint32_t blockSizeForLEA = blockSize / 2 * 2;
  if (blockSizeForLEA) {
    msp_scale_q15_params scale_params;
    scale_params.length = blockSizeForLEA;
    scale_params.scale = scaleFract;
    scale_params.shift = shift;
    my_checkStatus(msp_scale_q15(&scale_params, pSrc, pDst));
  }
  if (blockSize % 2) {
    pDst[blockSize - 1] = (pSrc[blockSize - 1] * scaleFract) >> (15 - shift);
  }
#else
  arm_scale_q15(pSrc, scaleFract, shift, pDst, blockSize);
#endif
}

void my_interleave_q15(const int16_t* pSrc, uint16_t channel,
                       uint16_t numChannels, int16_t* pDst,
                       uint32_t blockSize) {
  MY_ASSERT(channel < numChannels);
  // XXX: not using LEA here as pSrc and/or pDst is often unaligned
  // CMSIS does not have interleave (yet)
  for (uint32_t idx = 0; idx < blockSize; idx++) {
    *(pDst + channel) = *pSrc;
    pSrc++;
    pDst += numChannels;
  }
}

void my_deinterleave_q15(const int16_t* pSrc, uint16_t channel,
                         uint16_t numChannels, int16_t* pDst,
                         uint32_t blockSize) {
  // XXX: not using LEA here as I didn't allocate LEA memory for inputs with
  // footprints
  for (uint32_t idx = 0; idx < blockSize; idx++) {
    *pDst = *(pSrc + channel);
    pSrc += numChannels;
    pDst++;
  }
}

int16_t padding_for_lea(int16_t val) {
  // LEA requires parameters to be even in many places
  return (val + 1) / 2 * 2;
}
