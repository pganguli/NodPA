// C-linkage callback type used by the DSP library matrix-multiply wrappers.
//
// The TI DSPLib and ARM CMSIS-DSP matrix-multiply functions are modified to
// call back into the firmware so that each computed output row can be written
// directly to NVM (via my_memcpy_to_param) rather than buffered in SRAM first.
// This avoids holding the entire output matrix in SRAM simultaneously.
//
// The callback signature intentionally mirrors my_memcpy_to_param so it can
// be passed as a function pointer without a wrapper.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
struct ParameterInfo;
// Called once per computed batch to write `n` bytes from `src` to the NVM
// slot for `param` at word offset `offset_in_word`.  `timer_delay` defers the
// SPI write interrupt; `is_linear` classifies the access for counters.
typedef void (*data_preservation_func)(struct ParameterInfo* param,
                                       uint32_t offset_in_word, const void* src,
                                       size_t n, uint16_t timer_delay,
                                       bool is_linear);
#ifdef __cplusplus
}
#endif
